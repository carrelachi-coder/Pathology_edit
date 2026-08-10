"""Freeze the reviewed G2 image--instruction--mechanism ledger."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .g2_he_review import HE_REVIEW_SCHEMA_VERSION
from .g2_qualification import QUALIFICATION_SCHEMA_VERSION

G2_V2_MANIFEST_SCHEMA = "g2-v2-image-instruction-mechanism-manifest-v1"

CELL_EXTENT_PRIMITIVES = frozenset(
    {
        "cell-type-abundance-increase-v1",
        "cell-type-abundance-decrease-v1",
        "cellularity-increase-v1",
        "cellularity-decrease-v1",
        "neoplastic-cell-infiltration-increase-v1",
    }
)


def freeze_g2_v2_manifest(
    legacy_manifest_path: str | Path,
    qualification_jsonl: str | Path,
    he_decision_jsonl: str | Path,
    *,
    output_dir: str | Path,
    expected_cases: int = 600,
) -> dict[str, Any]:
    legacy_path = Path(legacy_manifest_path)
    qualification_path = Path(qualification_jsonl)
    decision_path = Path(he_decision_jsonl)
    legacy_payload = json.loads(legacy_path.read_text(encoding="utf-8"))
    legacy_rows = legacy_payload.get("cases") if isinstance(legacy_payload, dict) else None
    if not isinstance(legacy_rows, list):
        raise ValueError("legacy G2 manifest must contain cases[]")
    qualification = _read_jsonl(qualification_path)
    decisions = _read_jsonl(decision_path)
    if not (
        len(legacy_rows) == len(qualification) == len(decisions) == expected_cases
    ):
        raise ValueError(
            "legacy, qualification and H&E ledgers must have the same frozen case count"
        )
    qualification_digest = _sha256(qualification_path)
    decision_digest = _sha256(decision_path)
    frozen_cases = []
    for index, (legacy, qualified, decision) in enumerate(
        zip(legacy_rows, qualification, decisions, strict=True)
    ):
        case_id = str(legacy.get("case_id") or "")
        if not case_id or case_id != qualified.get("case_id") or case_id != decision.get("case_id"):
            raise ValueError(f"case order/identity mismatch at row {index}")
        if qualified.get("schema_version") != QUALIFICATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported qualification schema for {case_id}")
        if decision.get("schema_version") != HE_REVIEW_SCHEMA_VERSION:
            raise ValueError(f"unsupported H&E decision schema for {case_id}")
        for field in (
            "source_index",
            "organ",
            "dataset",
            "pathology_domain_id",
            "annotation_profile_id",
        ):
            if qualified.get(field) != decision.get(field):
                raise ValueError(f"H&E decision {field} drift for {case_id}")
        if int(qualified.get("source_index", index)) != index:
            raise ValueError(f"qualification source index drift for {case_id}")
        basis = decision.get("review_basis") or {}
        if basis.get("qualification_sha256") != qualification_digest:
            raise ValueError(f"H&E decision is detached from qualification for {case_id}")
        source_assets = qualified["source_assets"]
        for digest_field in (
            "image_sha256",
            "tissue_mask_sha256",
            "nuclei_mask_sha256",
        ):
            if basis.get(f"source_{digest_field}") != source_assets[digest_field]:
                raise ValueError(
                    f"H&E decision {digest_field} is detached from source for {case_id}"
                )
        if str(legacy.get("source_image")) != source_assets["image"]:
            raise ValueError(f"source image drift for {case_id}")
        if str(legacy.get("source_tissue_mask")) != source_assets["tissue_mask"]:
            raise ValueError(f"source tissue drift for {case_id}")
        if str(legacy.get("source_nuclei_mask")) != source_assets["nuclei_mask"]:
            raise ValueError(f"source nuclei drift for {case_id}")
        execution_allowed = bool(decision["execution_allowed"])
        if execution_allowed != (decision["decision_status"] != "abstain"):
            raise ValueError(f"decision execution flag is inconsistent for {case_id}")
        selected_primitive = decision["selected_joint_primitive"]
        budget_contract = _budget_contract(selected_primitive, execution_allowed)
        nuclei_instances_uri = legacy.get("source_nuclei_instances") or legacy.get(
            "source_nuclei_instances_uri"
        )
        auxiliary_uris = dict(
            sorted((legacy.get("auxiliary_structure_uris") or {}).items())
        )
        frozen_cases.append(
            {
                "source_index": index,
                "case_id": case_id,
                "sample_id": str(legacy.get("sample_id") or qualified.get("sample_id") or ""),
                "organ": qualified["organ"],
                "dataset": qualified["dataset"],
                "pathology_domain_id": qualified["pathology_domain_id"],
                "annotation_profile_id": qualified["annotation_profile_id"],
                "cell_observation_profile_id": qualified["cell_observation_profile_id"],
                "cell_population_profile_id": qualified["cell_population_profile_id"],
                "source_image_uri": source_assets["image"],
                "source_tissue_mask_uri": source_assets["tissue_mask"],
                "source_nuclei_mask_uri": source_assets["nuclei_mask"],
                "source_nuclei_instances_uri": nuclei_instances_uri,
                "auxiliary_structure_uris": auxiliary_uris,
                "source_digests": {
                    "image_sha256": source_assets["image_sha256"],
                    "tissue_mask_sha256": source_assets["tissue_mask_sha256"],
                    "nuclei_mask_sha256": source_assets["nuclei_mask_sha256"],
                    "nuclei_instances_sha256": _optional_asset_sha256(
                        nuclei_instances_uri, case_id=case_id
                    ),
                    "auxiliary_structure_sha256": {
                        key: _optional_asset_sha256(uri, case_id=case_id)
                        for key, uri in auxiliary_uris.items()
                    },
                },
                "original_instruction": qualified["instruction"],
                "legacy_primitive": qualified["legacy_primitive"],
                "decision_status": decision["decision_status"],
                "execution_allowed": execution_allowed,
                "instruction": decision["recommended_instruction"],
                "primitive_id": selected_primitive,
                "mechanism_id": decision["selected_mechanism_id"],
                "decision_reason_code": decision["reason_code"],
                "visual_observations": decision["visual_observations"],
                "review_basis": basis,
                "budget_contract": budget_contract,
                "joint_area_budget": budget_contract.get("joint_area_budget"),
                "seed": int(legacy.get("organic_seed", legacy.get("seed", 42))),
                "pixel_size_um": legacy.get("pixel_size_um"),
                "source_manifest_metadata": {
                    key: legacy.get(key)
                    for key in (
                        "profile",
                        "g2_primitive",
                        "patch_grade",
                        "provider",
                        "source_site",
                        "specimen_type",
                        "primary_or_metastatic",
                        "source_mask_annotation_provenance",
                    )
                    if legacy.get(key) is not None
                },
            }
        )

    payload = {
        "schema_version": G2_V2_MANIFEST_SCHEMA,
        "manifest_id": "G2-v2",
        "freeze_policy": "source-assets-read-only_he-qualified_fail-closed",
        "case_count": len(frozen_cases),
        "source_chain": {
            "legacy_manifest": str(legacy_path),
            "legacy_manifest_sha256": _sha256(legacy_path),
            "qualification_jsonl": str(qualification_path),
            "qualification_sha256": qualification_digest,
            "he_decision_jsonl": str(decision_path),
            "he_decision_sha256": decision_digest,
        },
        "decision_counts": dict(
            sorted(Counter(item["decision_status"] for item in frozen_cases).items())
        ),
        "execution_case_count": sum(item["execution_allowed"] for item in frozen_cases),
        "abstain_case_count": sum(not item["execution_allowed"] for item in frozen_cases),
        "cases": frozen_cases,
    }
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "g2_v2_image_instruction_mechanism_manifest.json"
    manifest.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = _sha256(manifest)
    sidecar = root / f"{manifest.name}.sha256"
    sidecar.write_text(f"{digest}  {manifest.name}\n", encoding="ascii")
    summary = {
        "schema_version": G2_V2_MANIFEST_SCHEMA,
        "manifest": str(manifest),
        "manifest_sha256": digest,
        "digest_sidecar": str(sidecar),
        "case_count": len(frozen_cases),
        "execution_case_count": payload["execution_case_count"],
        "abstain_case_count": payload["abstain_case_count"],
        "decision_counts": payload["decision_counts"],
        "target_mask_created": False,
        "source_asset_mutated": False,
    }
    summary_path = root / "g2_v2_freeze_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _budget_contract(
    primitive_id: str | None,
    execution_allowed: bool,
) -> dict[str, Any]:
    if not execution_allowed:
        return {"mode": "none", "policy_id": "abstain-no-execution-v1"}
    if primitive_id in CELL_EXTENT_PRIMITIVES:
        return {
            "mode": "cell_count_extent",
            "policy_id": "scene-calibrated-local-population-budget-v1",
            "compile_stage": "after_instance_authority_before_candidate_generation",
            "requires_exact_instance_capacity": True,
            "joint_area_budget": None,
        }
    return {
        "mode": "joint_area",
        "policy_id": "g2-joint-19pct-capacity-adaptive-v1",
        "joint_area_budget": {
            "target_fraction": 0.19,
            "min_fraction": 0.14,
            "max_fraction": 0.24,
            "tissue_min_fraction": 0.14,
            "relative_tolerance": 0.02,
            "fallback_policy": "max_feasible_below_target",
            "capacity_floor_policy": "lower_to_proven_max_safe",
            "minimum_effective_fraction": 0.14,
        },
    }


def _optional_asset_sha256(uri: Any, *, case_id: str) -> str | None:
    if uri in (None, ""):
        return None
    path = Path(str(uri))
    if not path.is_file():
        raise FileNotFoundError(f"optional source asset does not exist for {case_id}: {path}")
    return _sha256(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

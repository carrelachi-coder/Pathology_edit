"""Freeze the reviewed G2 image--instruction--mechanism ledger."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .g2_he_review import HE_REVIEW_SCHEMA_VERSION
from .g2_qualification import QUALIFICATION_SCHEMA_VERSION

G2_V2_MANIFEST_SCHEMA = "g2-v2-image-instruction-mechanism-manifest-v2"
EXECUTION_QUALIFICATION_SCHEMA = "g2-v2-read-only-execution-qualification-v1"
PRIMITIVE_ONTOLOGY_VERSION = "joint-primitive-v2"

DEPRECATED_PRIMITIVE_IDS = frozenset(
    {"neoplastic-cell-infiltration-increase-v1"}
)

CELL_EXTENT_PRIMITIVES = frozenset(
    {
        "cell-type-abundance-increase-v1",
        "cell-type-abundance-decrease-v1",
        "cellularity-increase-v1",
        "cellularity-decrease-v1",
        "neoplastic-microinfiltration-increase-v1",
    }
)


def freeze_g2_v2_manifest(
    legacy_manifest_path: str | Path,
    qualification_jsonl: str | Path,
    he_decision_jsonl: str | Path,
    *,
    output_dir: str | Path,
    expected_cases: int = 600,
    execution_qualification_jsonl: str | Path | None = None,
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
    execution_qualification_path = (
        Path(execution_qualification_jsonl)
        if execution_qualification_jsonl is not None
        else None
    )
    execution_qualification = (
        _read_jsonl(execution_qualification_path)
        if execution_qualification_path is not None
        else None
    )
    if execution_qualification is not None and len(execution_qualification) != expected_cases:
        raise ValueError(
            "execution qualification must cover the same frozen case count"
        )
    execution_qualification_digest = (
        _sha256(execution_qualification_path)
        if execution_qualification_path is not None
        else None
    )
    frozen_cases = []
    qualification_rows = (
        execution_qualification
        if execution_qualification is not None
        else [None] * expected_cases
    )
    for index, (legacy, qualified, decision, execution_preflight) in enumerate(
        zip(
            legacy_rows,
            qualification,
            decisions,
            qualification_rows,
            strict=True,
        )
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
        he_execution_allowed = bool(decision["execution_allowed"])
        if he_execution_allowed != (decision["decision_status"] != "abstain"):
            raise ValueError(f"decision execution flag is inconsistent for {case_id}")
        preflight_passed = he_execution_allowed
        preflight_status = None
        preflight_failures: list[str] = []
        if execution_preflight is not None:
            if (
                execution_preflight.get("schema_version")
                != EXECUTION_QUALIFICATION_SCHEMA
                or execution_preflight.get("case_id") != case_id
                or int(execution_preflight.get("source_index", -1)) != index
            ):
                raise ValueError(
                    f"execution qualification identity drift for {case_id}"
                )
            preflight_status = str(execution_preflight.get("status") or "")
            preflight_failures = [
                str(item)
                for item in execution_preflight.get("failure_reasons", ())
            ]
            preflight_passed = (
                preflight_status == "executable_preflight_passed"
            )
            if not he_execution_allowed and preflight_status != "upstream_abstain":
                raise ValueError(
                    f"upstream abstain has inconsistent execution qualification: {case_id}"
                )
        execution_allowed = he_execution_allowed and preflight_passed
        reviewed_primitive = decision["selected_joint_primitive"]
        reviewed_mechanism = decision["selected_mechanism_id"]
        reviewed_instruction = decision["recommended_instruction"]
        reviewed_semantic_intent = decision.get("prebound_semantic_intent")
        reviewed_semantic_digest = decision.get(
            "prebound_semantic_intent_sha256"
        )
        if he_execution_allowed and reviewed_primitive in DEPRECATED_PRIMITIVE_IDS:
            raise ValueError(
                f"H&E decision uses deprecated primitive-v1 ID for {case_id}: "
                f"{reviewed_primitive}"
            )
        if he_execution_allowed:
            semantic_intent = reviewed_semantic_intent
            semantic_digest = reviewed_semantic_digest
            if not isinstance(semantic_intent, dict) or not semantic_digest:
                raise ValueError(
                    f"executable case lacks Codex semantic binding: {case_id}"
                )
            actual_semantic_digest = hashlib.sha256(
                json.dumps(
                    semantic_intent,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if actual_semantic_digest != semantic_digest:
                raise ValueError(
                    f"Codex semantic binding digest drift for {case_id}"
                )
            if (
                semantic_intent.get("instruction")
                != reviewed_instruction
                or semantic_intent.get("primitive_id") != reviewed_primitive
            ):
                raise ValueError(
                    f"Codex semantic binding conflicts with H&E decision for {case_id}"
                )
        elif reviewed_semantic_intent is not None or reviewed_semantic_digest is not None:
            raise ValueError(
                f"abstained case must not carry executable semantics: {case_id}"
            )
        selected_primitive = reviewed_primitive if execution_allowed else None
        selected_mechanism = reviewed_mechanism if execution_allowed else None
        instruction = reviewed_instruction if execution_allowed else None
        semantic_intent = reviewed_semantic_intent if execution_allowed else None
        semantic_digest = reviewed_semantic_digest if execution_allowed else None
        final_decision_status = (
            decision["decision_status"] if execution_allowed else "abstain"
        )
        final_reason_code = (
            decision["reason_code"]
            if execution_allowed or not he_execution_allowed
            else "execution_preflight_failed:"
            + (preflight_failures[0] if preflight_failures else preflight_status or "unknown")
        )
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
                "decision_status": final_decision_status,
                "he_decision_status": decision["decision_status"],
                "execution_allowed": execution_allowed,
                "instruction": instruction,
                "primitive_id": selected_primitive,
                "mechanism_id": selected_mechanism,
                "prebound_semantic_intent": semantic_intent,
                "prebound_semantic_intent_sha256": semantic_digest,
                "decision_reason_code": final_reason_code,
                "reviewed_candidate_before_execution_preflight": {
                    "instruction": reviewed_instruction,
                    "primitive_id": reviewed_primitive,
                    "mechanism_id": reviewed_mechanism,
                    "semantic_intent_sha256": reviewed_semantic_digest,
                },
                "execution_qualification": (
                    {
                        "status": preflight_status,
                        "failure_reasons": preflight_failures,
                        "ledger_sha256": execution_qualification_digest,
                    }
                    if execution_preflight is not None
                    else None
                ),
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
        "primitive_ontology_version": PRIMITIVE_ONTOLOGY_VERSION,
        "deprecated_primitive_ids_forbidden": sorted(DEPRECATED_PRIMITIVE_IDS),
        "freeze_policy": "source-assets-read-only_he-qualified_fail-closed",
        "case_count": len(frozen_cases),
        "source_chain": {
            "legacy_manifest": str(legacy_path),
            "legacy_manifest_sha256": _sha256(legacy_path),
            "qualification_jsonl": str(qualification_path),
            "qualification_sha256": qualification_digest,
            "he_decision_jsonl": str(decision_path),
            "he_decision_sha256": decision_digest,
            "execution_qualification_jsonl": (
                str(execution_qualification_path)
                if execution_qualification_path is not None
                else None
            ),
            "execution_qualification_sha256": execution_qualification_digest,
            "execution_qualification_source_manifest_sha256": (
                execution_qualification[0].get("source_manifest_sha256")
                if execution_qualification
                else None
            ),
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

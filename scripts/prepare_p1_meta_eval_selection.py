#!/usr/bin/env python3
"""Freeze GLaS/PANDA five-case visual-evaluation assignments without running them.

This preparation is intentionally read-only with respect to source assets and
does not call a model, renderer, API, or visualizer.  The resulting manifest
stays non-executable until independent review, case-level auxiliary authority,
and every frozen runtime digest are present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from phase3_joint_edit_refine.semantic_parser import RuleBasedSemanticParser
from phase3_joint_edit_refine.skills.repository import JointSkillRepository

SCHEMA_VERSION = "p1-glas-panda-meta-eval-selection-v1"
SOURCE_POOL_SCHEMA_VERSION = "p1-glas-panda-source-case-pool-v1"
PROFILES = {
    "glas-gland-v1": "colorectal",
    "panda-gleason-v1": "prostate",
}
PROBNET_CHECKPOINT_SHA256 = (
    "8efc4c0100fb0f013e70c64a8a01718ce5d6a2b2646af72878adf5e7726ee2d8"
)
PROFILE_PRODUCIBLE_AUXILIARY_STRUCTURES = {
    "glas-gland-v1": {
        "gland_or_lumen_support",
        "external_cellular_stroma_map",
    },
    "panda-gleason-v1": {
        "native_pattern_and_lumen_map",
        "native_pattern_map",
        "gland_lumen_map",
        "gland_unit_wall_map",
        "external_cellular_stroma_map",
    },
}
RUNTIME_DIGEST_FIELDS = (
    "mature_probnet_checkpoint_sha256",
    "frozen_spatial_ranker_sha256",
    "instance_library_sha256",
    "generator_checkpoint_sha256",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_source_case_pool(source: dict[str, Any]) -> None:
    if source.get("schema_version") != SOURCE_POOL_SCHEMA_VERSION:
        raise ValueError("unsupported P1 source case-pool schema")
    cases = source.get("cases")
    if not isinstance(cases, list) or len(cases) != 10:
        raise ValueError("P1 source case pool must contain exactly ten cases")
    case_ids: set[str] = set()
    organs: dict[str, int] = {organ: 0 for organ in PROFILES.values()}
    for row in cases:
        if not isinstance(row, dict):
            raise TypeError("P1 source case record must be an object")
        declared = row.get("case_record_sha256")
        canonical = dict(row)
        canonical.pop("case_record_sha256", None)
        if declared != _canonical_sha256(canonical):
            raise ValueError(
                f"P1 source case record digest mismatch: {row.get('case_id')}"
            )
        case_id = row.get("case_id")
        organ = row.get("organ")
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise ValueError("P1 source case IDs must be present and unique")
        if organ not in organs:
            raise ValueError("P1 source case has an unsupported organ")
        case_ids.add(case_id)
        organs[str(organ)] += 1
        for uri_field in (
            "source_image",
            "source_tissue_mask",
            "source_nuclei_mask",
        ):
            if not isinstance(row.get(uri_field), str) or not row[uri_field]:
                raise ValueError(f"P1 source case lacks {uri_field}: {case_id}")
        if not _is_sha256(row.get("source_tissue_mask_sha256")):
            raise ValueError("P1 source case lacks tissue-mask content digest")
        for digest_field in (
            "source_image_sha256",
            "source_nuclei_mask_sha256",
        ):
            digest = row.get(digest_field)
            if digest is not None and not _is_sha256(digest):
                raise ValueError(
                    f"P1 source case has invalid {digest_field}: {case_id}"
                )
    if set(organs.values()) != {5}:
        raise ValueError("P1 source case pool must bind five cases per organ")


def build_selection(
    *,
    root: Path,
    source_manifest: Path,
    capability_matrix: Path,
) -> dict[str, Any]:
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    _validate_source_case_pool(source)
    matrix = json.loads(capability_matrix.read_text(encoding="utf-8"))
    repository = JointSkillRepository()
    rows_by_organ = {
        organ: sorted(
            (row for row in source["cases"] if row.get("organ") == organ),
            key=lambda row: str(row["case_id"]),
        )[:5]
        for organ in PROFILES.values()
    }
    if any(len(rows) != 5 for rows in rows_by_organ.values()):
        raise ValueError("source manifest does not contain five fixed cases per P1 organ")

    evaluations: list[dict[str, Any]] = []
    for profile in matrix["profiles"]:
        profile_id = profile["annotation_profile_id"]
        if profile_id not in PROFILES:
            continue
        organ = PROFILES[profile_id]
        for capability in profile["capabilities"]:
            if capability["status"] == "closed":
                continue
            mechanism_id = capability["mechanism_id"]
            primitive_id = capability["primitive_id"]
            mechanism = repository.mechanisms[mechanism_id]
            required_auxiliary = sorted(
                set(capability.get("required_auxiliary_structures", ()))
                | set(mechanism.representability.required_auxiliary_structures)
            )
            selected_cases = []
            producible = sorted(
                set(required_auxiliary)
                & PROFILE_PRODUCIBLE_AUXILIARY_STRUCTURES.get(profile_id, set())
            )
            for row in rows_by_organ[organ]:
                available = sorted((row.get("auxiliary_structure_uris") or {}).keys())
                missing = sorted(
                    set(required_auxiliary) - set(available) - set(producible)
                )
                missing_source_asset_digests = sorted(
                    field
                    for field in (
                        "source_image_sha256",
                        "source_tissue_mask_sha256",
                        "source_nuclei_mask_sha256",
                    )
                    if not _is_sha256(row.get(field))
                )
                selected_cases.append(
                    {
                        "case_id": row["case_id"],
                        "source_image": row["source_image"],
                        "source_tissue_mask": row["source_tissue_mask"],
                        "source_nuclei_mask": row["source_nuclei_mask"],
                        "source_image_sha256": row.get("source_image_sha256"),
                        "source_tissue_mask_sha256": row.get(
                            "source_tissue_mask_sha256"
                        ),
                        "source_nuclei_mask_sha256": row.get(
                            "source_nuclei_mask_sha256"
                        ),
                        "source_case_record_sha256": row["case_record_sha256"],
                        "source_nuclei_instances": row.get(
                            "source_nuclei_instances"
                        )
                        or row.get("source_nuclei_instances_uri"),
                        "seed": int(row.get("organic_seed", 42)),
                        "joint_mechanism_id": mechanism_id,
                        "joint_primitive_id": primitive_id,
                        "available_auxiliary_structures": available,
                        "profile_producible_auxiliary_structures": producible,
                        "missing_required_auxiliary_structures": missing,
                        "missing_source_asset_digests": missing_source_asset_digests,
                        "execution_allowed": False,
                        "fixed_case_no_replacement": True,
                    }
                )
            evaluations.append(
                {
                    "evaluation_id": f"{profile_id}::{mechanism_id}::{primitive_id}",
                    "pathology_domain_id": mechanism.pathology_domain_id,
                    "annotation_profile_id": profile_id,
                    "mechanism_id": mechanism_id,
                    "primitive_id": primitive_id,
                    "instruction": capability["simple_instructions"][0],
                    "required_auxiliary_structures": required_auxiliary,
                    "selected_cases": selected_cases,
                    "case_count": 5,
                    "execution_status": "blocked_pending_independent_review_and_authority_preflight",
                }
            )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "production_status": "shadow_only",
        "execution_status": "blocked_pending_independent_review_and_authority_preflight",
        "source_manifest": str(source_manifest.relative_to(root)),
        "source_manifest_schema_version": source["schema_version"],
        "source_manifest_sha256": _sha256(source_manifest),
        "capability_matrix": str(capability_matrix.relative_to(root)),
        "capability_matrix_sha256": _sha256(capability_matrix),
        "selection_policy": {
            "fixed_cases_per_mechanism_primitive": 5,
            "failed_rejected_abstained_case_replacement_allowed": False,
            "visualization_or_api_run_during_preparation": False,
        },
        "runtime_authority": {
            "selection_generator_sha256": _sha256(Path(__file__)),
            "mature_probnet_checkpoint_sha256": PROBNET_CHECKPOINT_SHA256,
            "frozen_spatial_ranker_sha256": None,
            "instance_library_sha256": None,
            "generator_checkpoint_sha256": None,
            "all_required_digests_bound": False,
        },
        "visual_acceptance_contract": {
            "required_panels": [
                "source_tissue",
                "target_tissue",
                "source_nuclei",
                "target_nuclei",
                "tissue_change_overlay",
                "nuclei_change_overlay",
                "complete_manifest",
            ],
            "changed_region_requirement": (
                "must meet the primitive-owned meaningful tissue floor or exact "
                "cell count/extent quota; visually negligible edits are failures"
            ),
            "mechanism_congruence_requirement": (
                "final tissue and nuclei effects must match the bound mask-level "
                "mechanism and executor postconditions"
            ),
            "claim_boundary": (
                "annotation-level synthetic counterfactual only; no diagnostic, "
                "prognostic, treatment-benefit, pCR, fibrosis, budding-grade, or "
                "H&E-derived execution claim"
            ),
        },
        "evaluations": evaluations,
        "evaluation_count": len(evaluations),
        "visualization_run": False,
        "api_used": False,
    }
    validate_selection(payload, repository=repository)
    return payload


def validate_selection(
    payload: dict[str, Any], *, repository: JointSkillRepository | None = None
) -> None:
    repository = repository or JointSkillRepository()
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported P1 meta-eval selection schema")
    if payload.get("visualization_run") is not False or payload.get("api_used") is not False:
        raise ValueError("pre-review selection must not report visualization or API use")
    runtime = payload.get("runtime_authority") or {}
    runtime_digest_state = all(
        _is_sha256(runtime.get(field)) for field in RUNTIME_DIGEST_FIELDS
    )
    if runtime.get("all_required_digests_bound") is not runtime_digest_state:
        raise ValueError("P1 runtime digest completeness declaration is stale")
    if not _is_sha256(runtime.get("selection_generator_sha256")):
        raise ValueError("P1 selection generator digest is absent or malformed")
    runtime_complete = runtime_digest_state
    parser = RuleBasedSemanticParser()
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list) or len(evaluations) != int(
        payload.get("evaluation_count", -1)
    ):
        raise ValueError("P1 meta-eval evaluation count is inconsistent")
    if len(evaluations) != 21:
        raise ValueError("P1 meta-eval must contain exactly 21 evaluations")
    expected_case_ids_by_profile: dict[str, tuple[str, ...]] = {}
    evaluation_ids: set[str] = set()
    for evaluation in evaluations:
        mechanism_id = evaluation.get("mechanism_id")
        primitive_id = evaluation.get("primitive_id")
        if not isinstance(mechanism_id, str) or not mechanism_id:
            raise ValueError("meta-eval row lacks mechanism binding")
        evaluation_id = evaluation.get("evaluation_id")
        expected_evaluation_id = (
            f"{evaluation.get('annotation_profile_id')}::{mechanism_id}::{primitive_id}"
        )
        if evaluation_id != expected_evaluation_id or evaluation_id in evaluation_ids:
            raise ValueError("meta-eval evaluation ID is duplicate or unbound")
        evaluation_ids.add(str(evaluation_id))
        mechanism = repository.mechanisms.get(mechanism_id)
        if mechanism is None or primitive_id not in mechanism.supported_primitives:
            raise ValueError("meta-eval mechanism/primitive binding is invalid")
        intent = parser.parse(str(evaluation.get("instruction") or ""))
        declared_primitives = {
            item.primitive_id for item in intent.primitive_hypotheses
        }
        if primitive_id not in declared_primitives:
            raise ValueError("meta-eval instruction lacks primitive binding")
        if mechanism_id == "prostate-operational-tumor-retreat" and (
            intent.treatment_context != "post_treatment"
            or intent.scenario
            not in {
                "treatment_response",
                "disease_regression",
                "residual_disease",
            }
        ):
            raise ValueError(
                "meta-eval treatment mechanism lacks compatible post-treatment binding"
            )
        cases = evaluation.get("selected_cases")
        if not isinstance(cases, list) or len(cases) != 5:
            raise ValueError("every P1 mechanism/primitive requires exactly five cases")
        case_ids = [str(row.get("case_id") or "") for row in cases]
        if "" in case_ids or len(set(case_ids)) != 5:
            raise ValueError("P1 fixed case IDs must be present and unique")
        profile_id = str(evaluation.get("annotation_profile_id") or "")
        fixed_ids = tuple(case_ids)
        if profile_id in expected_case_ids_by_profile:
            if expected_case_ids_by_profile[profile_id] != fixed_ids:
                raise ValueError("P1 fixed cases changed across profile evaluations")
        else:
            expected_case_ids_by_profile[profile_id] = fixed_ids
        required_auxiliary = set(
            evaluation.get("required_auxiliary_structures") or ()
        )
        for row in cases:
            if row.get("joint_mechanism_id") != mechanism_id:
                raise ValueError("meta-eval case lacks exact mechanism binding")
            if row.get("joint_primitive_id") != primitive_id:
                raise ValueError("meta-eval case lacks exact primitive binding")
            missing = set(row.get("missing_required_auxiliary_structures") or ())
            available = set(row.get("available_auxiliary_structures") or ())
            producible = set(
                row.get("profile_producible_auxiliary_structures") or ()
            )
            allowed_producible = (
                PROFILE_PRODUCIBLE_AUXILIARY_STRUCTURES.get(
                    evaluation.get("annotation_profile_id"), set()
                )
                & required_auxiliary
            )
            if producible != allowed_producible:
                raise ValueError("meta-eval profile-produced auxiliary accounting is stale")
            if missing != required_auxiliary - available - producible:
                raise ValueError("meta-eval required auxiliary accounting is stale")
            missing_asset_digests = {
                field
                for field in (
                    "source_image_sha256",
                    "source_tissue_mask_sha256",
                    "source_nuclei_mask_sha256",
                )
                if not _is_sha256(row.get(field))
            }
            if missing_asset_digests != set(
                row.get("missing_source_asset_digests") or ()
            ):
                raise ValueError("meta-eval source asset digest accounting is stale")
            if not _is_sha256(row.get("source_case_record_sha256")):
                raise ValueError("meta-eval source case record digest is malformed")
            if row.get("execution_allowed") and (
                missing or missing_asset_digests or not runtime_complete
            ):
                raise ValueError(
                    "meta-eval case cannot execute without required auxiliary, "
                    "source digests, and runtime authority"
                )
            if row.get("execution_allowed") is not False:
                raise ValueError("pre-review P1 fixed cases must remain non-executable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="explicit digest-bound P1 source case pool",
    )
    parser.add_argument("--capability-matrix", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = args.source_manifest or (
        root
        / "phase3_joint_edit_refine"
        / "resources"
        / "p1_glas_panda_source_case_pool_v1.json"
    )
    matrix = args.capability_matrix or (
        root
        / "phase3_joint_edit_refine"
        / "resources"
        / "non_breast_organ_annotation_capability_matrix_v1.json"
    )
    output = args.output or (
        root
        / "phase3_joint_edit_refine"
        / "resources"
        / "p1_glas_panda_meta_eval_selection_v1.json"
    )
    rendered = json.dumps(
        build_selection(root=root, source_manifest=source, capability_matrix=matrix),
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    ) + "\n"
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != rendered:
            raise SystemExit("P1 meta-eval selection drift detected")
        return 0
    output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

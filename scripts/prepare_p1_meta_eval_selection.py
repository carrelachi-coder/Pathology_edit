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

from phase3_joint_edit_refine.skills.repository import JointSkillRepository
from phase3_joint_edit_refine.semantic_parser import RuleBasedSemanticParser


SCHEMA_VERSION = "p1-glas-panda-meta-eval-selection-v1"
PROFILES = {
    "glas-gland-v1": "colorectal",
    "panda-gleason-v1": "prostate",
}
PROBNET_CHECKPOINT_SHA256 = (
    "8efc4c0100fb0f013e70c64a8a01718ce5d6a2b2646af72878adf5e7726ee2d8"
)
PROFILE_PRODUCIBLE_AUXILIARY_STRUCTURES = {
    "glas-gland-v1": {"gland_or_lumen_support"},
    "panda-gleason-v1": {
        "native_pattern_and_lumen_map",
        "native_pattern_map",
        "gland_lumen_map",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_selection(
    *,
    root: Path,
    source_manifest: Path,
    capability_matrix: Path,
) -> dict[str, Any]:
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
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
                selected_cases.append(
                    {
                        "case_id": row["case_id"],
                        "source_image": row["source_image"],
                        "source_tissue_mask": row["source_tissue_mask"],
                        "source_nuclei_mask": row["source_nuclei_mask"],
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
        "source_manifest_sha256": _sha256(source_manifest),
        "capability_matrix": str(capability_matrix.relative_to(root)),
        "capability_matrix_sha256": _sha256(capability_matrix),
        "selection_policy": {
            "fixed_cases_per_mechanism_primitive": 5,
            "failed_rejected_abstained_case_replacement_allowed": False,
            "visualization_or_api_run_during_preparation": False,
        },
        "runtime_authority": {
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
    runtime_complete = bool(runtime.get("all_required_digests_bound"))
    parser = RuleBasedSemanticParser()
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list) or len(evaluations) != int(
        payload.get("evaluation_count", -1)
    ):
        raise ValueError("P1 meta-eval evaluation count is inconsistent")
    for evaluation in evaluations:
        mechanism_id = evaluation.get("mechanism_id")
        primitive_id = evaluation.get("primitive_id")
        if not isinstance(mechanism_id, str) or not mechanism_id:
            raise ValueError("meta-eval row lacks mechanism binding")
        mechanism = repository.mechanisms.get(mechanism_id)
        if mechanism is None or primitive_id not in mechanism.supported_primitives:
            raise ValueError("meta-eval mechanism/primitive binding is invalid")
        intent = parser.parse(str(evaluation.get("instruction") or ""))
        declared_primitives = {
            item.primitive_id for item in intent.primitive_hypotheses
        }
        if primitive_id not in declared_primitives:
            raise ValueError("meta-eval instruction lacks primitive binding")
        if mechanism_id == "prostate-treatment-associated-fibrotic-replacement":
            if (
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
            if row.get("execution_allowed") and (missing or not runtime_complete):
                raise ValueError(
                    "meta-eval case cannot execute without required auxiliary and runtime digests"
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "artifacts"
        / "joint_edit_refine_g2_pilot_20260805"
        / "g2_600_frozen_product_manifest.json"
    )
    matrix = (
        root
        / "phase3_joint_edit_refine"
        / "resources"
        / "non_breast_organ_annotation_capability_matrix_v1.json"
    )
    output = (
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

#!/usr/bin/env python3
"""Build the mask-only capability/evidence audit for lung, oral and skin."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3_joint_edit_refine.skills.repository import JointSkillRepository


ORGAN_CONFIG = {
    "lung": {
        "pathology_domain_id": "lung-carcinoma-v1",
        "annotation_profile_id": "ignite-semantic-v1",
        "dataset": "IGNITE",
    },
    "oral": {
        "pathology_domain_id": "oral-squamous-cell-carcinoma-v1",
        "annotation_profile_id": "orca-semantic-v1",
        "dataset": "ORCA",
    },
    "skin": {
        "pathology_domain_id": "melanoma-v1",
        "annotation_profile_id": "puma-semantic-v1",
        "dataset": "PUMA",
    },
}


def build() -> dict:
    repository = JointSkillRepository()
    governance = json.loads(
        (repository.root / "evidence-governance-v2.json").read_text(
            encoding="utf-8"
        )
    )
    sources = governance["sources"]
    mechanism_sources = governance["mechanism_pathology_sources"]
    organs = []
    for organ, config in ORGAN_CONFIG.items():
        schema = repository.annotation_schema(config["annotation_profile_id"])
        pairs = []
        for mechanism in sorted(
            (
                item
                for item in repository.mechanisms.values()
                if item.pathology_domain_id == config["pathology_domain_id"]
            ),
            key=lambda item: item.mechanism_id,
        ):
            source_ids = list(mechanism_sources[mechanism.mechanism_id])
            source_records = [sources[source_id] for source_id in source_ids]
            for primitive_id in mechanism.supported_primitives:
                reason = repository.execution_selection_reason(
                    primitive_id=primitive_id,
                    mechanism_id=mechanism.mechanism_id,
                )
                closure_category = repository.execution_closure_category(
                    primitive_id=primitive_id,
                    mechanism_id=mechanism.mechanism_id,
                )
                pairs.append(
                    {
                        "primitive_id": primitive_id,
                        "primitive_scope": repository.primitives[primitive_id].scope,
                        "mechanism_id": mechanism.mechanism_id,
                        "status": (
                            "closed" if reason else "executable_mask_only"
                        ),
                        "closed_reason": reason,
                        "closure_category": closure_category,
                        "required_observation_sources": list(
                            mechanism.planner_policy.allowed_observation_sources
                        ),
                        "prohibited_observation_sources": list(
                            mechanism.planner_policy.prohibited_observation_sources
                        ),
                        "required_auxiliary_structures": list(
                            mechanism.representability.required_auxiliary_structures
                        ),
                        "pathology_source_ids": source_ids,
                        "pathology_sources": [
                            {
                                "title": record["title"],
                                "uri": record["uri"],
                                "locator": record["locator"],
                                "verification_status": record[
                                    "verification_status"
                                ],
                            }
                            for record in source_records
                        ],
                    }
                )
        open_pairs = [item for item in pairs if item["status"] != "closed"]
        open_primitives = sorted({item["primitive_id"] for item in open_pairs})
        closed_pairs = [item for item in pairs if item["status"] == "closed"]
        organs.append(
            {
                "organ": organ,
                **config,
                "annotation_labels": sorted(schema.readable_labels),
                "summary": {
                    "catalog_pair_count": len(pairs),
                    "executable_pair_count": len(open_pairs),
                    "closed_pair_count": len(closed_pairs),
                    "executable_unique_primitive_count": len(open_primitives),
                    "five_case_review_target_count": 5 * len(open_primitives),
                },
                "executable_unique_primitives": open_primitives,
                "pairs": pairs,
            }
        )
    payload = {
        "schema_version": "lung-oral-skin-mask-only-primitive-audit-v1",
        "planner_authority": {
            "allowed_rasters": [
                "source_tissue_semantic_panel",
                "source_tissue_component_panel",
                "source_interface_anchor_panel",
                "source_tissue_nuclei_panel",
                "candidate_mask_condition_board",
            ],
            "forbidden_rasters": ["source_he", "reference_he", "generated_he"],
            "llm_decision_scope": (
                "select only compiler-certified primitive/mechanism, interface, "
                "anchor, tissue-plan and cell-plan candidate IDs"
            ),
            "compiler_owned": [
                "pixels",
                "coordinates",
                "areas",
                "counts",
                "geometry",
                "layouts",
                "parameter_values",
            ],
        },
        "visual_review_policy": {
            "cases_per_executable_organ_primitive": 5,
            "boundary_or_retreat_target_fraction": 0.12,
            "boundary_or_retreat_min_fraction": 0.08,
            "compartment_or_cord_target_fraction": 0.12,
            "compartment_or_cord_min_fraction": 0.08,
            "absolute_tissue_fallback_floor_fraction": 0.05,
            "cell_only_minimum_delta_count_range": [4, 12],
            "cell_only_target_delta_count_range": [6, 16],
            "tiny_effect_action": "reject_case_not_relax_floor",
        },
        "organs": organs,
    }
    _validate(payload)
    return payload


def _validate(payload: dict) -> None:
    for organ in payload["organs"]:
        for item in organ["pairs"]:
            if "source_he_for_execution" not in item[
                "prohibited_observation_sources"
            ]:
                raise ValueError(
                    f"{item['mechanism_id']} does not prohibit source H&E"
                )
            if item["status"] == "closed":
                if not item["closed_reason"]:
                    raise ValueError("closed pair lacks reason")
                if item["closure_category"] not in {
                    "annotation_limited",
                    "dataset_case_limited",
                }:
                    raise ValueError(
                        "target-organ implementation defect must be repaired, "
                        f"not closed: {item}"
                    )
                continue
            if item["closure_category"] is not None:
                raise ValueError("open pair retains a closure category")
            if item["required_auxiliary_structures"]:
                raise ValueError(
                    f"open pair still requires unavailable auxiliary authority: {item}"
                )
            if not item["pathology_sources"] or any(
                source["verification_status"] != "verified"
                for source in item["pathology_sources"]
            ):
                raise ValueError(
                    f"open pair lacks verified pathology evidence: {item}"
                )


def main() -> int:
    output = (
        ROOT
        / "phase3_joint_edit_refine"
        / "resources"
        / "lung_oral_skin_primitive_audit_v1.json"
    )
    output.write_text(
        json.dumps(build(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from phase3_joint_edit_refine.clarification import (
    PlannerClarificationRequired,
    build_primitive_clarification_request,
    create_clarification_decision,
    resolve_clarification_decision,
)
from phase3_joint_edit_refine.models import JointCaseContext, JointContractError
from phase3_joint_edit_refine.planner import HeuristicJointPlanner


def _case(**updates) -> JointCaseContext:
    values = {
        "case_id": "case-clarify",
        "instruction": "增加肿瘤浸润",
        "source_image_uri": "/tmp/image.png",
        "source_tissue_mask_uri": "/tmp/tissue.png",
        "source_nuclei_mask_uri": "/tmp/nuclei.png",
        "pathology_domain_id": "lung-adenocarcinoma-v1",
        "annotation_profile_id": "ignite-semantic-v1",
        "cell_observation_profile_id": "cellvit-five-class-v1",
        "cell_population_profile_id": "lung-cell-population-v1",
        "primitive_id": "neoplastic-microinfiltration-increase-v1",
        "joint_area_budget": None,
        "seed": 7,
        "provenance": {
            "source_image_sha256": "image-a",
            "source_tissue_mask_sha256": "tissue-a",
            "source_nuclei_mask_sha256": "nuclei-a",
        },
    }
    values.update(updates)
    return JointCaseContext(**values)


def _option(primitive_id: str, mechanism_id: str, priority: int):
    return SimpleNamespace(
        primitive_id=primitive_id,
        semantic_priority=priority,
        mechanism=SimpleNamespace(mechanism_id=mechanism_id),
    )


def test_request_contains_only_preflight_executable_primitives():
    case = _case()
    options = (
        _option("neoplastic-microinfiltration-increase-v1", "lung-local", 0),
        _option("invasive-front-expansion-v1", "lung-front", 1),
    )
    request = build_primitive_clarification_request(
        case=case,
        prepared_options=options,
        why_required="两种变化尺度都可执行，但原 instruction 未区分",
        primitive_ids=(
            "neoplastic-microinfiltration-increase-v1",
            "invasive-front-expansion-v1",
        ),
    ).to_metadata()

    assert [item["primitive_id"] for item in request["options"]] == [
        "neoplastic-microinfiltration-increase-v1",
        "invasive-front-expansion-v1",
    ]
    assert all(item["compatible_mechanism_ids"] for item in request["options"])
    with pytest.raises(JointContractError, match="non-executable"):
        build_primitive_clarification_request(
            case=case,
            prepared_options=options,
            why_required="invalid",
            primitive_ids=(
                "invasive-front-expansion-v1",
                "structural-void-spread-v1",
            ),
        )


def test_decision_is_digest_bound_and_locks_only_the_primitive():
    case = _case()
    options = (
        _option("neoplastic-microinfiltration-increase-v1", "lung-local", 0),
        _option("invasive-front-expansion-v1", "lung-front-a", 1),
        _option("invasive-front-expansion-v1", "lung-front-b", 1),
    )
    request = build_primitive_clarification_request(
        case=case,
        prepared_options=options,
        why_required="H&E cannot recover the user's intended edit scale",
        primitive_ids=(
            "neoplastic-microinfiltration-increase-v1",
            "invasive-front-expansion-v1",
        ),
    ).to_metadata()
    decision = create_clarification_decision(
        request,
        selected_option_id="primitive:invasive-front-expansion-v1",
        responder="doctor-1",
        provider="interactive_user_choice",
    )
    resolved = resolve_clarification_decision(
        case=replace(case, clarification_decision=decision),
        prepared_options=options,
    )

    assert resolved is not None
    assert resolved[0] == "invasive-front-expansion-v1"
    assert "mechanism" not in resolved[1]

    stale_case = replace(
        case,
        clarification_decision=decision,
        provenance={**case.provenance, "source_image_sha256": "image-b"},
    )
    with pytest.raises(JointContractError, match="detached"):
        resolve_clarification_decision(
            case=stale_case,
            prepared_options=options,
        )


def test_offline_planner_can_emit_a_clarification_signal():
    case = _case(
        provenance={
            "source_image_sha256": "image-a",
            "source_tissue_mask_sha256": "tissue-a",
            "source_nuclei_mask_sha256": "nuclei-a",
            "joint_mechanism_id": "__clarify__",
        }
    )
    options = (
        _option("neoplastic-microinfiltration-increase-v1", "lung-local", 0),
        _option("invasive-front-expansion-v1", "lung-front", 1),
    )
    with pytest.raises(PlannerClarificationRequired) as error:
        HeuristicJointPlanner().select_interpretation(
            case=case,
            scene=None,
            options=options,
            image_paths=(),
        )
    assert error.value.primitive_ids == (
        "neoplastic-microinfiltration-increase-v1",
        "invasive-front-expansion-v1",
    )

import json
from pathlib import Path
from types import SimpleNamespace

from phase3_joint_edit_refine.models import CellCountExtentBudget
from phase3_joint_edit_refine.planner_inputs import MASK_PLANNER_ARTIFACT_KINDS
from phase3_joint_edit_refine.skills.repository import JointSkillRepository
from phase3_joint_edit_refine.workflow import _apply_profile_visible_cell_budget
from scripts.build_lung_oral_skin_primitive_audit import build


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    ROOT
    / "phase3_joint_edit_refine"
    / "resources"
    / "lung_oral_skin_primitive_audit_v1.json"
)


def test_frozen_audit_matches_catalog_and_classifies_every_pair():
    frozen = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    assert frozen == build()
    expected = {
        "lung": (24, 5, 19),
        "oral": (16, 6, 10),
        "skin": (19, 5, 14),
    }
    for organ in frozen["organs"]:
        summary = organ["summary"]
        assert (
            summary["catalog_pair_count"],
            summary["executable_pair_count"],
            summary["closed_pair_count"],
        ) == expected[organ["organ"]]
        assert all(
            item["status"] in {"closed", "executable_mask_only"}
            for item in organ["pairs"]
        )
        assert all(
            item["closed_reason"]
            for item in organ["pairs"]
            if item["status"] == "closed"
        )


def test_open_pairs_have_verified_pathology_sources_and_no_auxiliary_gap():
    audit = build()
    for organ in audit["organs"]:
        for item in organ["pairs"]:
            if item["status"] != "executable_mask_only":
                continue
            assert not item["required_auxiliary_structures"]
            assert item["pathology_sources"]
            assert {
                source["verification_status"]
                for source in item["pathology_sources"]
            } == {"verified"}


def test_all_target_organ_planners_are_mask_only():
    assert set(MASK_PLANNER_ARTIFACT_KINDS.values()) == {
        "source_tissue_semantic_panel",
        "source_tissue_component_panel",
        "source_interface_anchor_panel",
        "source_tissue_nuclei_panel",
        "candidate_mask_condition_board",
    }
    repository = JointSkillRepository()
    target_domains = {
        "lung-carcinoma-v1",
        "oral-squamous-cell-carcinoma-v1",
        "melanoma-v1",
    }
    for mechanism in repository.mechanisms.values():
        if mechanism.pathology_domain_id not in target_domains:
            continue
        assert "source_he_for_execution" in (
            mechanism.planner_policy.prohibited_observation_sources
        )
        assert not any(
            source in {"source_he", "reference_he", "generated_he"}
            for source in mechanism.planner_policy.allowed_observation_sources
        )


def test_runtime_scope_closes_skin_peritumoral_focus_programs():
    repository = JointSkillRepository()
    assert repository.execution_selection_reason(
        primitive_id="peritumoral-neoplastic-scatter-increase-v1",
        mechanism_id="melanoma-discohesive-junctional",
    )
    assert repository.execution_selection_reason(
        primitive_id="peritumoral-neoplastic-scatter-increase-v1",
        mechanism_id="melanoma-peritumoral-small-focus",
    )
    assert repository.execution_selection_reason(
        primitive_id="peritumoral-small-cluster-increase-v1",
        mechanism_id="melanoma-peritumoral-small-focus",
    )


def test_runtime_scope_supports_evidence_backed_pair_closures():
    repository = JointSkillRepository()
    assert isinstance(repository.execution_scope["closed_pairs"], dict)


def test_target_profiles_receive_visible_cell_effect_budget():
    original = CellCountExtentBudget(4, 2, 6, 64)
    for profile in (
        "ignite-semantic-v1",
        "orca-semantic-v1",
        "puma-semantic-v1",
    ):
        case = SimpleNamespace(annotation_profile_id=profile)
        budget, metadata = _apply_profile_visible_cell_budget(
            case,
            primitive_id="generic-inflammatory-cell-abundance-increase-v1",
            budget=original,
            metadata={},
        )
        assert budget.target_delta_count == 20
        assert budget.min_delta_count == 12
        assert budget.maximum_extent_px >= 384
        assert metadata["policy_id"] == "mask-review-visible-cell-effect-budget-v4"

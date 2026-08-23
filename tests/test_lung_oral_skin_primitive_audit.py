import json
from pathlib import Path
from types import SimpleNamespace

from phase3_joint_edit_refine.models import CellCountExtentBudget
from phase3_joint_edit_refine.planner_inputs import MASK_PLANNER_ARTIFACT_KINDS
from phase3_joint_edit_refine.skills.repository import JointSkillRepository
from phase3_joint_edit_refine.workflow import _apply_profile_visible_cell_budget
from scripts.build_lung_oral_skin_primitive_audit import build
from scripts.run_lung_oral_skin_primitive_mask_review import (
    CELL_BUDGETS,
    CORD_TISSUE_BUDGET,
    INTERFACE_TISSUE_BUDGET,
    _eligible_score,
)


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
        "lung": (23, 19, 4),
        "oral": (16, 8, 8),
        "skin": (18, 14, 4),
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
        assert {
            item["closure_category"]
            for item in organ["pairs"]
            if item["status"] == "closed"
        } <= {"annotation_limited", "dataset_case_limited"}
        if organ["organ"] in {"lung", "skin"}:
            assert (
                "tumor-burden-increase-v1"
                not in organ["executable_unique_primitives"]
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


def test_runtime_scope_distinguishes_annotation_closure_from_repaired_execution():
    repository = JointSkillRepository()
    assert repository.execution_selection_reason(
        primitive_id="peritumoral-neoplastic-scatter-increase-v1",
        mechanism_id="melanoma-discohesive-junctional",
    )
    assert not repository.execution_selection_reason(
        primitive_id="peritumoral-neoplastic-scatter-increase-v1",
        mechanism_id="melanoma-peritumoral-small-focus",
    )
    assert not repository.execution_selection_reason(
        primitive_id="peritumoral-small-cluster-increase-v1",
        mechanism_id="melanoma-peritumoral-small-focus",
    )
    assert repository.execution_closure_category(
        primitive_id="peritumoral-neoplastic-scatter-increase-v1",
        mechanism_id="melanoma-discohesive-junctional",
    ) == "annotation_limited"


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
            minimum_delta_count=12,
            budget=original,
            metadata={},
        )
        assert budget.target_delta_count == 16
        assert budget.min_delta_count == 12
        assert budget.maximum_extent_px >= 384
        assert metadata["policy_id"] == "mask-review-visible-cell-effect-budget-v4"


def test_lung_cord_uses_a_narrow_topology_appropriate_area_floor():
    assert CORD_TISSUE_BUDGET.target_fraction == 0.009
    assert CORD_TISSUE_BUDGET.min_fraction == 0.004
    assert CORD_TISSUE_BUDGET.tissue_min_fraction == 0.004
    assert CORD_TISSUE_BUDGET.min_fraction < INTERFACE_TISSUE_BUDGET.min_fraction


def test_small_cluster_review_budget_matches_two_compact_native_foci():
    budget = CELL_BUDGETS["peritumoral-small-cluster-increase-v1"]
    assert budget.min_delta_count == 4
    assert budget.minimum_effect_foci == 2
    assert budget.minimum_effect_span_px == 32


def test_oral_cell_decrease_has_visible_but_residual_preserving_contract():
    repository = JointSkillRepository()
    primitive = repository.primitives["cell-type-abundance-decrease-v1"]
    assert primitive.minimum_effect_delta_count_for(
        "oral-squamous-cell-carcinoma-v1"
    ) == 10
    mechanism = repository.mechanisms["oral-scc-local-population-modulation"]
    depletion = mechanism.cell_program.cellularity_depletion
    assert depletion is not None
    assert depletion.core_width_cell_diameters == 4
    assert depletion.transition_width_cell_diameters == 8
    assert depletion.core_target_removal_fraction == 0.68
    assert depletion.transition_start_removal_fraction == 0.55
    assert depletion.transition_end_removal_fraction == 0.12
    assert depletion.minimum_core_residual_fraction == 0.30
    assert depletion.minimum_transition_residual_fraction == 0.42


def test_peritumoral_review_ranking_uses_local_connected_mask_capacity():
    row = {
        "shape": [512, 512],
        "areas": {"1": 50000, "2": 150000},
        "free": {"1": 10000, "2": 100000},
        "counts": {"1": 12},
        "contacts": {"1:2": 512},
        "peritumoral_free_stroma": {
            "area": 4096,
            "largest_component_area": 2048,
            "largest_component_span_px": 64.0,
        },
    }
    eligible, score = _eligible_score(
        "lung", "peritumoral-small-cluster-increase-v1", row
    )
    assert eligible
    assert _eligible_score(
        "lung", "infiltrative-nest-cord-extension-v1", row
    )[0]
    disconnected = dict(row)
    disconnected["peritumoral_free_stroma"] = {
        "area": 4096,
        "largest_component_area": 256,
        "largest_component_span_px": 64.0,
    }
    assert not _eligible_score(
        "lung", "peritumoral-small-cluster-increase-v1", disconnected
    )[0]
    assert score > 0

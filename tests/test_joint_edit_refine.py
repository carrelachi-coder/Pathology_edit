"""Contracts and regression tests for the independent joint edit pipeline."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy import ndimage

from inpaint_cells.instance_authority import array_sha256
from phase3_joint_edit_refine.agents import (
    CELL_PLAN_SELECTION_SCHEMA,
    JOINT_PLAN_JSON_SCHEMA,
    OpenAIMultimodalJointCritic,
    OpenAIMultimodalJointPlanner,
    _reject_prohibited_geometry_payload,
)
from phase3_joint_edit_refine.agents import (
    _mask_planner_case_metadata as joint_planner_case_metadata,
)
from phase3_joint_edit_refine.auxiliary import materialize_profile_auxiliaries
from phase3_joint_edit_refine.budget import JointFeasibilitySolver
from phase3_joint_edit_refine.candidate_feasibility import (
    _structural_event_risk_count,
)
from phase3_joint_edit_refine.cell_layouts import (
    ReferenceNucleusShape,
    _calibrated_reference_variants,
    _centers_satisfy_minimum_span,
    _certified_witness_first_anchors,
    _effect_first_anchors,
    _multisite_population_anchor_order,
    _place_layout,
    _prioritize_local_references,
    _probnet_hard_core_anchor_order,
    _reference_sampling_order,
    _reference_shape_digest,
    _unique_reference_shapes,
    build_reference_shape_library,
    certificate_aligned_cluster_size_range,
    certificate_aligned_references,
    independent_focus_minimum_center_separation_px,
)
from phase3_joint_edit_refine.cell_programs import (
    CellToolProgramCompiler,
    _cap_density_field_quotas,
    _depletion_band_edges,
    _enforce_density_field_gradient_quotas,
    _select_density_field_removals_preserving_coverage,
    depletion_field_area_is_sufficient,
)
from phase3_joint_edit_refine.critic import DeterministicJointResearchCritic
from phase3_joint_edit_refine.feasibility import (
    _target_interface_population_density,
    augment_tissue_scene_with_nuclei_preflight,
)
from phase3_joint_edit_refine.g2_pilot import build_local_joint_records
from phase3_joint_edit_refine.g2_plan_overrides import apply_plan_overrides
from phase3_joint_edit_refine.gates import (
    MECHANISM_POSTCONDITION_IDS,
    JointGateRegistry,
    _added_instance_areas_by_class,
    _cell_quota,
    _cell_tissue_compatibility,
    _discrete_radial_profile_is_monotonic,
    _fine_pattern_preserved,
    _mechanism_realization,
    _nearest_reference_area_ratio,
    _recorded_instance_areas_by_class,
    _reference_shape_integrity,
    audit_added_class1_foci,
    audit_cell_effect_foci,
    audit_directional_extension_raster,
    mechanism_postcondition_checker_id,
)
from phase3_joint_edit_refine.generator_adapter import (
    build_agentic_joint_route,
    build_frozen_generator_inputs,
    route_joint_handoff,
)
from phase3_joint_edit_refine.instance_authority import (
    build_scene_instance_authority,
)
from phase3_joint_edit_refine.ledger import analyze_joint_change
from phase3_joint_edit_refine.mature_probnet_adapter import (
    MatureProbNetCellExecutor,
    MatureProbNetConfig,
    _architecture_placement_trace,
    _compile_packing_witness,
    _mature_nucleus_area_medians,
)
from phase3_joint_edit_refine.models import (
    CellCountExtentBudget,
    JointAreaBudget,
    JointCaseContext,
    JointContractError,
    JointCriticRanking,
    JointCriticResult,
    JointGateCheck,
    JointGateReport,
)
from phase3_joint_edit_refine.nuclei import iter_instances, load_native_instances
from phase3_joint_edit_refine.packing import certify_complete_footprint_packing
from phase3_joint_edit_refine.planner import (
    CellPlanSelectionHandle,
    CompilerOwnedDepletionAnchor,
    HeuristicJointPlanner,
    JointInterpretationOption,
    _issue_cell_plan_portfolio,
    _structural_units_for_components,
)
from phase3_joint_edit_refine.planner_inputs import (
    MASK_PLANNER_ARTIFACT_KINDS,
    MaskPlannerArtifactRegistry,
    validate_mask_planner_image_paths,
)
from phase3_joint_edit_refine.planner_policy import PLANNER_DECISIONS
from phase3_joint_edit_refine.post_generation import (
    audit_joint_generation_handoff,
)
from phase3_joint_edit_refine.profile_statistics import (
    build_annotation_profile_statistics,
)
from phase3_joint_edit_refine.reference_shapes import (
    load_reference_shape_authority,
)
from phase3_joint_edit_refine.scene import build_joint_scene_analysis
from phase3_joint_edit_refine.seam import (
    anchor_coverage_fraction,
    compile_continuity_center_quota,
    compile_executable_continuity_count,
)
from phase3_joint_edit_refine.semantic_parser import (
    RuleBasedSemanticParser,
    bind_semantic_intent,
)
from phase3_joint_edit_refine.skills.execution_aliases import (
    tissue_tool_primitive_id,
)
from phase3_joint_edit_refine.skills.repository import JointSkillRepository
from phase3_joint_edit_refine.spatial_contracts import (
    BREAST_SMALL_CLUSTER_MINIMUM_ANCHOR_SEPARATION_DIAMETERS,
    small_cluster_maximum_hotspot_span_px,
)
from phase3_joint_edit_refine.tissue_execution import (
    _bind_and_validate_tissue_candidate_traces,
)
from phase3_joint_edit_refine.tissue_planner import (
    JOINT_TISSUE_DECISION_SCHEMA,
    MultiInterfaceResearchTissuePlanner,
    OpenAIJointAwareTissuePlanner,
    _component_capped_allocation_capacities,
    _directional_sector_selection_limit,
    _effective_tissue_topology,
    _normalize_integer_allocations,
    _rank_interfaces_by_marginal_capacity,
    _select_executable_anchor_ids,
)
from phase3_joint_edit_refine.tissue_planner import (
    _mask_planner_case_metadata as tissue_planner_case_metadata,
)
from phase3_joint_edit_refine.tissue_tools import (
    compile_tissue_tool_program,
    validate_tissue_plan_tool_binding,
)
from phase3_joint_edit_refine.workflow import (
    INFILTRATION_BUDGET_PRIMITIVES,
    PANDA_CELL_CAPACITY_FALLBACK_PRIMITIVES,
    PANDA_CELL_EFFECT_COUNT_OVERRIDES,
    PANDA_CELL_EFFECT_EXTENT_OVERRIDES,
    JointPathologyEditWorkflow,
    JointWorkflowConfig,
    _as_tissue_case,
    _apply_panda_profile_cell_budget,
    _candidate_preserving_closure_pixels,
    _CertifiedCellExecutionChoice,
    _CertifiedCellExecutionPortfolio,
    _derive_infiltration_budget,
    _joint_area_feedback_candidate_ids,
    _localized_focus_capacity_metrics,
    _maximum_safe_below_target_joint_pixels,
    _minimum_safe_above_target_joint_pixels,
    _provisional_union_requires_rebalance,
    _retain_visible_regression_whole_instance_closure,
    _select_cell_execution_choice,
)
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.agents import HeuristicInterfacePlanner
from phase3_mask_edit_refine.gates import GateRegistry, _check_edited_label_topology
from phase3_mask_edit_refine.models import (
    AreaBudget,
    CandidateMask,
    GateReport,
    RefineContractError,
)
from phase3_mask_edit_refine.scene import build_scene_analysis
from scripts import refine_other_organ_primitives as other_organ_generator
from scripts.prepare_p1_meta_eval_selection import validate_selection


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class JointSkillTests(unittest.TestCase):
    def test_panda_profile_keeps_primitive_count_and_extent_distinct(self):
        case = JointCaseContext(
            case_id="panda-budget",
            instruction="decrease neoplastic cells",
            source_image_uri="image.png",
            source_tissue_mask_uri="tissue.png",
            source_nuclei_mask_uri="nuclei.png",
            pathology_domain_id="prostate-adenocarcinoma-v1",
            annotation_profile_id="panda-gleason-v1",
            cell_observation_profile_id="cellvit-five-class-v1",
            cell_population_profile_id="prostate-cell-population-v1",
            primitive_id="neoplastic-cell-abundance-decrease-v1",
            joint_area_budget=None,
            seed=7,
            provenance={},
        )
        source = CellCountExtentBudget(
            target_delta_count=9,
            min_delta_count=5,
            max_delta_count=13,
            maximum_extent_px=48,
            minimum_effect_span_px=12,
        )
        budget, metadata = _apply_panda_profile_cell_budget(
            case,
            primitive_id=case.primitive_id,
            budget=source,
            metadata={"skill_minimum_effect_delta_count": 2},
        )

        self.assertEqual(
            (budget.min_delta_count, budget.target_delta_count),
            (4, 6),
        )
        self.assertEqual(budget.max_delta_count, 8)
        self.assertEqual(budget.minimum_effect_span_px, 12)
        self.assertEqual(
            metadata["policy_id"],
            "panda-profile-cell-effect-budget-v2-amplitude",
        )
        self.assertEqual(
            PANDA_CELL_EFFECT_EXTENT_OVERRIDES[case.primitive_id], (3.0, 0)
        )
        self.assertEqual(
            PANDA_CELL_EFFECT_EXTENT_OVERRIDES["cellularity-increase-v1"],
            (4.0, 1),
        )
        self.assertEqual(
            PANDA_CELL_EFFECT_COUNT_OVERRIDES[
                "cell-type-abundance-decrease-v1"
            ],
            (6, 4, 8),
        )

    def test_panda_cell_type_decrease_uses_a_four_cell_floor(self):
        case = JointCaseContext(
            case_id="panda-connective-depletion-budget",
            instruction="decrease connective tissue cells",
            source_image_uri="image.png",
            source_tissue_mask_uri="tissue.png",
            source_nuclei_mask_uri="nuclei.png",
            pathology_domain_id="prostate-adenocarcinoma-v1",
            annotation_profile_id="panda-gleason-v1",
            cell_observation_profile_id="cellvit-five-class-v1",
            cell_population_profile_id="prostate-cell-population-v1",
            primitive_id="cell-type-abundance-decrease-v1",
            joint_area_budget=None,
            seed=7,
            provenance={},
        )
        source = CellCountExtentBudget(
            target_delta_count=2,
            min_delta_count=2,
            max_delta_count=3,
            maximum_extent_px=48,
            minimum_effect_span_px=24,
        )
        budget, _metadata = _apply_panda_profile_cell_budget(
            case,
            primitive_id=case.primitive_id,
            budget=source,
            metadata={
                "skill_minimum_effect_delta_count": 2,
                "selected_zone_executable_capacity": 10,
            },
        )

        self.assertEqual(
            (
                budget.min_delta_count,
                budget.target_delta_count,
                budget.max_delta_count,
            ),
            (4, 6, 8),
        )

    def test_infiltration_budget_primitives_cover_all_peripheral_additions(self):
        self.assertEqual(
            INFILTRATION_BUDGET_PRIMITIVES,
            {
                "neoplastic-microinfiltration-increase-v1",
                "peritumoral-neoplastic-scatter-increase-v1",
                "peritumoral-small-cluster-increase-v1",
            },
        )

    def test_infiltration_budget_respects_primitive_span_count_and_foci(self):
        scene = SimpleNamespace(
            source_nuclei=np.zeros((512, 512), dtype=np.uint8),
            cells=SimpleNamespace(
                instances=(
                    SimpleNamespace(
                        area_px=314,
                        completeness_status="complete",
                        class_id=1,
                    ),
                )
            ),
            tissue=SimpleNamespace(
                graph=SimpleNamespace(
                    interfaces=(
                        SimpleNamespace(
                            source_label="Tumor",
                            target_label="Stroma",
                            source_component_id="tumor-1",
                            target_component_id="stroma-1",
                            contact_pixels=500,
                        ),
                    )
                )
            ),
        )
        budget, metadata = _derive_infiltration_budget(
            scene,
            minimum_effect_delta_count=4,
            minimum_effect_span_cell_diameters=4.0,
            minimum_effect_foci=3,
        )

        self.assertGreaterEqual(budget.min_delta_count, 4)
        self.assertGreaterEqual(
            budget.maximum_extent_px, budget.minimum_effect_span_px
        )
        self.assertGreaterEqual(budget.minimum_effect_span_px, 79)
        self.assertEqual(budget.minimum_effect_foci, 3)
        self.assertEqual(metadata["skill_minimum_effect_delta_count"], 4)

    def test_depletion_field_area_preflight_matches_gate_tolerance(self):
        core = np.ones((8, 8), dtype=bool)
        transition = np.ones((8, 8), dtype=bool)
        outer = np.ones((8, 8), dtype=bool)

        passed, observed, effective_minimum = (
            depletion_field_area_is_sufficient(
                core_region=core,
                transition_region=transition,
                outer_reference_region=outer,
                nominal_nucleus_diameter_px=1.0,
                minimum_field_area_cell_diameter_squares=68.0,
            )
        )

        self.assertFalse(passed)
        self.assertEqual(observed, 64.0)
        self.assertEqual(effective_minimum, 64.6)

    def test_single_anchor_interface_is_not_a_directional_sector(self):
        self.assertEqual(
            _directional_sector_selection_limit(
                interface_anchor_ids=("anchor:only",),
                maximum_selected_anchor_fraction=0.8,
                minimum_unselected_anchor_count=1,
            ),
            0,
        )
        self.assertEqual(
            _directional_sector_selection_limit(
                interface_anchor_ids=("a", "b", "c", "d", "e"),
                allowed_anchor_ids=("a", "b"),
                maximum_selected_anchor_fraction=0.8,
                minimum_unselected_anchor_count=1,
            ),
            2,
        )

    def test_llm_planner_schema_excludes_compiler_owned_geometry(self):
        encoded_joint = json.dumps(JOINT_PLAN_JSON_SCHEMA, sort_keys=True)
        self.assertNotIn('"coordinates"', encoded_joint)
        self.assertNotIn('"area_budget"', encoded_joint)
        self.assertNotIn('"plan"', JOINT_TISSUE_DECISION_SCHEMA["properties"])
        self.assertIn(
            "selected_candidate_id",
            JOINT_TISSUE_DECISION_SCHEMA["properties"],
        )
        for key in (
            "coordinates",
            "nucleus_count",
            "density_multiplier",
            "area_budget",
            "shape_masks",
        ):
            with self.subTest(key=key), self.assertRaisesRegex(
                JointContractError, "compiler-owned numeric geometry"
            ):
                _reject_prohibited_geometry_payload(
                    {"cell_plan": {key: 1}}
                )

    def test_mask_planner_direct_caller_cannot_bypass_registry(self):
        class NeverCalledClient:
            def call(self, **kwargs):
                raise AssertionError("client must not receive an unauthorized raster")

        case = _breast_case_stub()
        mechanism = JointSkillRepository().mechanisms[
            "breast-annotation-anchored-boundary-growth"
        ]
        option = JointInterpretationOption(
            primitive_id="cohesive-boundary-expansion-v1",
            semantic_fit="direct",
            semantic_priority=0,
            semantic_rationale="fixture",
            mechanism=mechanism,
            feasibility={
                "aggregate_tissue_capacity_pixels": 100,
                "meaningful_tissue_floor_pixels": 10,
                "feasible_interface_count": 1,
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            disguised = Path(tmp) / "planner_01_tissue_mask.png"
            Image.new("RGB", (8, 8), "red").save(disguised)
            with self.assertRaisesRegex(
                JointContractError, "artifact registry"
            ):
                OpenAIMultimodalJointPlanner(
                    client=NeverCalledClient()
                ).select_interpretation(
                    case=case,
                    scene=SimpleNamespace(to_metadata=dict),
                    options=(option,),
                    image_paths=(disguised,),
                    artifact_registry=None,
                )

    def test_llm_interpretation_permissions_fail_closed(self):
        class FixtureClient:
            def __init__(self, response):
                self.response = response

            def call(self, **kwargs):
                return self.response, {"model": "fixture"}

        case = _breast_case_stub()
        mechanism = JointSkillRepository().mechanisms[
            "breast-annotation-anchored-boundary-growth"
        ]
        option = JointInterpretationOption(
            primitive_id="cohesive-boundary-expansion-v1",
            semantic_fit="direct",
            semantic_priority=0,
            semantic_rationale="fixture",
            mechanism=mechanism,
            feasibility={
                "aggregate_tissue_capacity_pixels": 100,
                "meaningful_tissue_floor_pixels": 10,
                "feasible_interface_count": 1,
            },
        )
        base = {
            "abstain": False,
            "abstain_reason": None,
            "clarification_required": False,
            "clarification_reason": None,
            "clarification_primitive_ids": [],
            "primitive_id": option.primitive_id,
            "mechanism_id": mechanism.mechanism_id,
            "decision_id": "select_primitive_mechanism_pair",
            "interpretation_explanation": "certified_semantic_option_selected",
            "supporting_observations": ["certified_capability_metrics"],
            "supporting_capability_metric_ids": ["semantic_priority"],
            "observed_contraindications": [],
            "confidence": 0.8,
        }
        adversaries = {
            "unknown capability metric": {
                **base,
                "supporting_capability_metric_ids": ["metric:forged"],
            },
            "illegal decision": {**base, "decision_id": "draw_polygon"},
            "unannotated claim": {
                **base,
                "supporting_observations": ["fibrosis is visible"],
            },
        }
        for label, response in adversaries.items():
            with self.subTest(label=label), self.assertRaises(JointContractError):
                OpenAIMultimodalJointPlanner(
                    client=FixtureClient(response),
                    max_contract_attempts=1,
                ).select_interpretation(
                    case=case,
                    scene=SimpleNamespace(to_metadata=dict),
                    options=(option,),
                    image_paths=(),
                    artifact_registry=None,
                )

    def test_remaining_population_may_use_unused_seam_capacity(self):
        source = np.zeros((32, 32), dtype=np.uint8)
        reference = ReferenceNucleusShape(
            instance_id="complete-local-1",
            class_id=1,
            mask=np.ones((3, 3), dtype=bool),
            source="native_instance",
            area_px=9,
        )
        legal = np.zeros_like(source, dtype=bool)
        legal[4:28, 4:28] = True
        certificate = certify_complete_footprint_packing(
            source_nuclei=source,
            erased_footprint=np.zeros_like(legal),
            center_region=legal,
            valid_footprint_region=legal,
            references_by_class={1: (reference,)},
            requested_count=6,
            class_request_weights={1: 1.0},
            continuity_region=legal,
            required_seam_count=2,
            minimum_seam_count=2,
            required_seam_class=1,
        )
        self.assertTrue(certificate.passed, certificate.failure_reasons)
        self.assertEqual(certificate.placed_seam_count, 2)
        self.assertEqual(certificate.placed_count, 6)

    def test_open_breast_planner_policies_are_mask_only_and_executable(self):
        repository = JointSkillRepository()
        for mechanism in repository.mechanisms.values():
            if mechanism.pathology_domain_id != "breast-invasive-carcinoma-v1":
                continue
            if mechanism.mechanism_id in repository.execution_scope["closed_mechanisms"]:
                continue
            with self.subTest(mechanism=mechanism.mechanism_id):
                policy = mechanism.planner_policy
                self.assertIn(
                    "source_he_for_execution",
                    policy.prohibited_observation_sources,
                )
                self.assertIn(
                    "unannotated_histology_inference",
                    policy.prohibited_observation_sources,
                )
                self.assertNotIn("source_he_for_execution", policy.allowed_observation_sources)
                self.assertTrue(policy.allowed_decisions)
                self.assertTrue(policy.hard_constraint_checker_ids)
                self.assertTrue(policy.selection_preferences)

    def test_all_non_breast_planner_policies_are_mask_only_and_certified(self):
        repository = JointSkillRepository()
        required_decisions = {
            "select_certified_tissue_plan_candidate",
            "select_certified_cell_plan_candidate",
            "request_clarification",
            "abstain",
        }
        prohibited_entry_tokens = (
            "h&e",
            "visible morphology",
            "inspect h&e",
            "true stroma",
        )
        for mechanism in repository.mechanisms.values():
            if mechanism.pathology_domain_id == "breast-invasive-carcinoma-v1":
                continue
            with self.subTest(mechanism=mechanism.mechanism_id):
                policy = mechanism.planner_policy
                self.assertIn(
                    "source_he_for_execution",
                    policy.prohibited_observation_sources,
                )
                self.assertIn(
                    "unannotated_histology_inference",
                    policy.prohibited_observation_sources,
                )
                self.assertNotIn(
                    "source_he_for_execution",
                    policy.allowed_observation_sources,
                )
                self.assertTrue(
                    required_decisions.issubset(policy.allowed_decisions)
                )
                self.assertTrue(policy.hard_constraint_checker_ids)
                self.assertTrue(policy.selection_preferences)
                recognition = " ".join(
                    mechanism.recognition.required_observations
                ).casefold()
                self.assertFalse(
                    any(token in recognition for token in prohibited_entry_tokens),
                    recognition,
                )

    def test_breast_colorectal_prostate_pathology_sources_are_verified(self):
        repository = JointSkillRepository()
        audited_domains = {
            "breast-invasive-carcinoma-v1",
            "colorectal-adenocarcinoma-v1",
            "prostate-adenocarcinoma-v1",
        }
        for mechanism in repository.mechanisms.values():
            if mechanism.pathology_domain_id not in audited_domains:
                continue
            with self.subTest(mechanism=mechanism.mechanism_id):
                evidence = repository.skill_evidence_status[
                    f"joint-mechanism:{mechanism.mechanism_id}"
                ]
                self.assertEqual(
                    evidence.category_status["pathology_fact"],
                    "verified_source_binding",
                )
                policy = mechanism.planner_policy
                self.assertIn(
                    "source_he_for_execution",
                    policy.prohibited_observation_sources,
                )
                self.assertIn(
                    "unannotated_histology_inference",
                    policy.prohibited_observation_sources,
                )
                self.assertNotIn(
                    "source_he_for_execution",
                    policy.allowed_observation_sources,
                )

    def test_non_breast_capability_matrix_matches_catalog_source_of_truth(self):
        repository = JointSkillRepository()
        matrix_path = (
            Path(__file__).parents[1]
            / "phase3_joint_edit_refine"
            / "resources"
            / "non_breast_organ_annotation_capability_matrix_v1.json"
        )
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        self.assertEqual(matrix["production_status"], "shadow_only")
        by_domain = {
            item["pathology_domain_id"]: item
            for item in matrix["profiles"]
        }
        expected_domains = {
            item.pathology_domain_id
            for item in repository.mechanisms.values()
            if item.pathology_domain_id != "breast-invasive-carcinoma-v1"
        }
        self.assertEqual(set(by_domain), expected_domains)
        for domain_id, profile in by_domain.items():
            with self.subTest(domain=domain_id):
                self.assertEqual(profile["production_status"], "shadow_only")
                pairs = {
                    (item["primitive_id"], item["mechanism_id"])
                    for item in profile["capabilities"]
                }
                expected = {
                    (primitive_id, mechanism.mechanism_id)
                    for mechanism in repository.mechanisms.values()
                    if mechanism.pathology_domain_id == domain_id
                    for primitive_id in mechanism.supported_primitives
                }
                self.assertEqual(pairs, expected)
                self.assertEqual(len(pairs), len(profile["capabilities"]))
                for capability in profile["capabilities"]:
                    self.assertEqual(
                        capability["production_status"], "shadow_only"
                    )
                    self.assertTrue(capability["simple_instructions"])
                    for instruction in capability["simple_instructions"]:
                        parsed = RuleBasedSemanticParser().parse(instruction)
                        self.assertIn(
                            capability["primitive_id"],
                            {
                                item.primitive_id
                                for item in parsed.primitive_hypotheses
                            },
                            (domain_id, instruction),
                        )
                        if capability["mechanism_id"] in {
                            "prostate-operational-tumor-retreat",
                            "lung-operational-tumor-retreat",
                            "melanoma-operational-tumor-retreat",
                            "oral-scc-operational-tumor-retreat",
                        }:
                            self.assertEqual(
                                parsed.treatment_context, "post_treatment"
                            )
                selection = profile["planner_selection_contract"]
                self.assertEqual(
                    selection["tissue"]["decision_id"],
                    "select_certified_tissue_plan_candidate",
                )
                self.assertEqual(
                    selection["cell"]["decision_id"],
                    "select_certified_cell_plan_candidate",
                )

    def test_p1_meta_eval_selection_is_fixed_and_fail_closed(self):
        path = (
            Path(__file__).parents[1]
            / "phase3_joint_edit_refine"
            / "resources"
            / "p1_glas_panda_meta_eval_selection_v1.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_selection(payload)
        self.assertFalse(payload["visualization_run"])
        self.assertFalse(payload["api_used"])
        self.assertTrue(
            all(
                len(evaluation["selected_cases"]) == 5
                for evaluation in payload["evaluations"]
            )
        )

        missing_mechanism = json.loads(json.dumps(payload))
        missing_mechanism["evaluations"][0]["mechanism_id"] = ""
        with self.assertRaisesRegex(ValueError, "lacks mechanism binding"):
            validate_selection(missing_mechanism)

        missing_authority = json.loads(json.dumps(payload))
        evaluation = next(
            item
            for item in missing_authority["evaluations"]
            if item["primitive_id"] == "local-invasive-clearance-v1"
        )
        evaluation["selected_cases"][0]["execution_allowed"] = True
        with self.assertRaisesRegex(ValueError, "required auxiliary"):
            validate_selection(missing_authority)

        invalid_treatment = json.loads(json.dumps(payload))
        treatment = next(
            item
            for item in invalid_treatment["evaluations"]
            if item["mechanism_id"]
            == "prostate-operational-tumor-retreat"
        )
        treatment["instruction"] = "Reduce tumor burden."
        with self.assertRaisesRegex(ValueError, "post-treatment binding"):
            validate_selection(invalid_treatment)

        panda_produced = next(
            item
            for item in payload["evaluations"]
            if item["mechanism_id"]
            == "prostate-pattern-5-peripheral-scatter"
        )
        self.assertTrue(
            all(
                "native_pattern_and_lumen_map"
                in row["profile_producible_auxiliary_structures"]
                and "native_pattern_and_lumen_map"
                not in row["missing_required_auxiliary_structures"]
                for row in panda_produced["selected_cases"]
            )
        )

    def test_planner_decision_vocabulary_matches_schema_and_breast_skills(self):
        tissue_decision = next(
            value
            for value in JOINT_TISSUE_DECISION_SCHEMA["properties"][
                "decision_id"
            ]["enum"]
            if value is not None
        )
        cell_decision = next(
            value
            for value in CELL_PLAN_SELECTION_SCHEMA["properties"][
                "decision_id"
            ]["enum"]
            if value is not None
        )
        self.assertIn(tissue_decision, PLANNER_DECISIONS)
        self.assertIn(cell_decision, PLANNER_DECISIONS)
        repository = JointSkillRepository()
        for mechanism in repository.mechanisms.values():
            if (
                mechanism.pathology_domain_id
                != "breast-invasive-carcinoma-v1"
                or mechanism.mechanism_id
                in repository.execution_scope["closed_mechanisms"]
            ):
                continue
            with self.subTest(mechanism=mechanism.mechanism_id):
                self.assertIn(
                    tissue_decision,
                    mechanism.planner_policy.allowed_decisions,
                )
                self.assertIn(
                    cell_decision,
                    mechanism.planner_policy.allowed_decisions,
                )

    def test_semantic_options_expose_only_measured_capability_metrics(self):
        mechanism = JointSkillRepository().mechanisms[
            "breast-annotation-anchored-boundary-growth"
        ]
        metadata = JointInterpretationOption(
            primitive_id="cohesive-boundary-expansion-v1",
            semantic_fit="direct",
            semantic_priority=0,
            semantic_rationale="fixture",
            mechanism=mechanism,
            feasibility={
                "aggregate_tissue_capacity_pixels": 100,
                "meaningful_tissue_floor_pixels": 10,
                "feasible_interface_count": 2,
            },
        ).to_metadata()
        metrics = metadata["deterministic_candidate_metrics"]
        self.assertEqual(metrics["feasible_interface_count"], 2)
        for placeholder in (
            "projection_merge_count",
            "protected_distance_px",
            "median_tumor_distance_px",
            "bridge_risk_count",
            "structural_risk_count",
        ):
            self.assertNotIn(placeholder, metrics)

    def test_mask_planner_metadata_excludes_histology_locator(self):
        case = _breast_case_stub()
        self.assertNotIn("source_image_uri", joint_planner_case_metadata(case))
        tissue_case = _as_tissue_case(
            case,
            allocation=SimpleNamespace(
                tissue_target_pixels=36_701,
                tissue_execution_floor_pixels=36_701,
            ),
            shape=(512, 512),
        )
        self.assertNotIn(
            "source_image_uri", tissue_planner_case_metadata(tissue_case)
        )
        self.assertIn("source_tissue_mask_uri", joint_planner_case_metadata(case))
        self.assertIn("source_nuclei_mask_uri", joint_planner_case_metadata(case))

    def test_mask_graph_llm_rejects_raw_histology_and_reader_boards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = _as_breast_growth_case(_write_synthetic_case(root))
            source_tissue = np.load(case.source_tissue_mask_uri)
            source_nuclei = np.asarray(Image.open(case.source_nuclei_mask_uri))
            case_root = root / case.case_id
            registry = MaskPlannerArtifactRegistry.issue(
                case=case,
                pipeline_owned_root=case_root,
                source_tissue=source_tissue,
                source_nuclei=source_nuclei,
                schema=MaskProfileSchema.from_reference_profile("BCSS"),
                pixel_size_um=case.pixel_size_um,
            )
            paths = [Path(value) for value in registry.source_image_paths]
            accepted = validate_mask_planner_image_paths(
                paths,
                case=case,
                artifact_registry=registry,
            )
            self.assertEqual(len(accepted), 4)

            forged_root = root / "forged-registry"
            forged_root.mkdir()
            disguised_registered = forged_root / "planner_01_tissue_mask.png"
            Image.open(case.source_image_uri).convert("RGB").save(disguised_registered)
            with self.assertRaisesRegex(
                JointContractError, "deterministic panel factory"
            ):
                MaskPlannerArtifactRegistry(
                    case=case,
                    pipeline_owned_root=forged_root,
                    source_tissue=source_tissue,
                    source_nuclei=source_nuclei,
                    schema=MaskProfileSchema.from_reference_profile("BCSS"),
                    pixel_size_um=case.pixel_size_um,
                )
            for name, kind in MASK_PLANNER_ARTIFACT_KINDS.items():
                forged = forged_root / name
                Image.open(case.source_image_uri).convert("RGB").save(forged)
                with self.subTest(name=name), self.assertRaisesRegex(
                    JointContractError, "generic.*registration"
                ):
                    registry.register(
                        forged,
                        artifact_kind=kind,
                        producer_id=(
                            "sealed-deterministic-mask-panel-writer-v3"
                        ),
                        producer_version="v3",
                    )

            disguised = root / "planner_01_tissue_mask.png"
            Image.new("RGB", (4, 4), "red").save(disguised)
            with self.assertRaises(JointContractError, msg="renamed H&E"):
                validate_mask_planner_image_paths(
                    (disguised,), case=case, artifact_registry=registry
                )
            reader_board = case_root / "joint_execution_review.png"
            Image.new("RGB", (4, 4), "white").save(reader_board)
            with self.assertRaises(JointContractError, msg="unregistered review board"):
                validate_mask_planner_image_paths(
                    (reader_board,), case=case, artifact_registry=registry
                )
            traversing = (
                case_root
                / "planner_panels"
                / ".."
                / "planner_02_component_map.png"
            )
            with self.assertRaises(JointContractError, msg="path traversal"):
                validate_mask_planner_image_paths(
                    (traversing,), case=case, artifact_registry=registry
                )
            symlink = case_root / "symlink.png"
            symlink.symlink_to(paths[0])
            with self.assertRaises(JointContractError, msg="symlink"):
                validate_mask_planner_image_paths(
                    (symlink,), case=case, artifact_registry=registry
                )
            paths[0].write_bytes(b"mutated")
            with self.assertRaises(JointContractError, msg="digest mismatch"):
                validate_mask_planner_image_paths(
                    (paths[0],), case=case, artifact_registry=registry
                )
            other_case = replace(case, case_id="other-case")
            with self.assertRaises(JointContractError, msg="cross case"):
                validate_mask_planner_image_paths(
                    (paths[1],), case=other_case, artifact_registry=registry
                )

    def test_mask_panel_writer_rejects_symlinks_before_mutating_sentinel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = _as_breast_growth_case(_write_synthetic_case(root))
            tissue = np.load(case.source_tissue_mask_uri)
            nuclei = np.asarray(Image.open(case.source_nuclei_mask_uri))
            sentinel = root / "sentinel.png"
            sentinel.write_bytes(b"sentinel-source-bytes")
            before = sentinel.read_bytes()
            case_root = root / "source-attack"
            panel_root = case_root / "planner_panels"
            panel_root.mkdir(parents=True)
            target = panel_root / "planner_01_tissue_mask.png"
            target.symlink_to(sentinel)
            with self.assertRaisesRegex(JointContractError, "target symlink"):
                MaskPlannerArtifactRegistry.issue(
                    case=case,
                    pipeline_owned_root=case_root,
                    source_tissue=tissue,
                    source_nuclei=nuclei,
                    schema=MaskProfileSchema.from_reference_profile("BCSS"),
                    pixel_size_um=case.pixel_size_um,
                )
            self.assertEqual(sentinel.read_bytes(), before)

            clean_root = root / "candidate-attack"
            registry = MaskPlannerArtifactRegistry.issue(
                case=case,
                pipeline_owned_root=clean_root,
                source_tissue=tissue,
                source_nuclei=nuclei,
                schema=MaskProfileSchema.from_reference_profile("BCSS"),
                pixel_size_um=case.pixel_size_um,
            )
            board_sentinel = root / "board-sentinel.png"
            board_sentinel.write_bytes(b"sentinel-board-bytes")
            board_before = board_sentinel.read_bytes()
            board_target = clean_root / "joint_condition_mask_review.png"
            board_target.symlink_to(board_sentinel)
            candidate = SimpleNamespace(
                candidate_id="joint:fixture",
                target_tissue_mask=tissue.copy(),
                target_nuclei_mask=nuclei.copy(),
                tissue_change=np.zeros_like(tissue, dtype=bool),
                cell_change=np.zeros_like(tissue, dtype=bool),
                joint_change=np.zeros_like(tissue, dtype=bool),
            )
            with self.assertRaisesRegex(JointContractError, "target symlink"):
                registry.write_candidate_board(candidates=(candidate,))
            self.assertEqual(board_sentinel.read_bytes(), board_before)

            hardlink_root = root / "hardlink-attack"
            hardlink_sentinel = root / "hardlink-sentinel.png"
            hardlink_sentinel.write_bytes(b"sentinel-hardlink-bytes")
            hardlink_before = hardlink_sentinel.read_bytes()
            hardlink_panel_root = hardlink_root / "planner_panels"
            hardlink_panel_root.mkdir(parents=True)
            hardlink_target = (
                hardlink_panel_root / "planner_01_tissue_mask.png"
            )
            hardlink_target.hardlink_to(hardlink_sentinel)
            with self.assertRaisesRegex(
                JointContractError, "multiply-linked target"
            ):
                MaskPlannerArtifactRegistry.issue(
                    case=case,
                    pipeline_owned_root=hardlink_root,
                    source_tissue=tissue,
                    source_nuclei=nuclei,
                    schema=MaskProfileSchema.from_reference_profile("BCSS"),
                    pixel_size_um=case.pixel_size_um,
                )
            self.assertEqual(hardlink_sentinel.read_bytes(), hardlink_before)

            parent_root = root / "parent-symlink-attack"
            parent_root.mkdir()
            external_parent = root / "external-planner-panels"
            external_parent.mkdir()
            (parent_root / "planner_panels").symlink_to(
                external_parent, target_is_directory=True
            )
            with self.assertRaisesRegex(
                JointContractError, "parent chain contains a symlink"
            ):
                MaskPlannerArtifactRegistry.issue(
                    case=case,
                    pipeline_owned_root=parent_root,
                    source_tissue=tissue,
                    source_nuclei=nuclei,
                    schema=MaskProfileSchema.from_reference_profile("BCSS"),
                    pixel_size_um=case.pixel_size_um,
                )
            self.assertEqual(list(external_parent.iterdir()), [])

    def test_narrow_cord_compiler_rejects_organic_v2(self):
        program = compile_tissue_tool_program(
            primitive_id="infiltrative-nest-cord-extension-v1",
            mechanism_id="breast-infiltrative-nest-cord-extension",
            mechanism_allowed_families=(
                "interface_band_sdf",
                "topology_safe_morphology",
            ),
            primitive_allowed_executors=(
                "interface_sdf",
                "connected_morphology",
                "organic_v2",
            ),
        )
        self.assertNotIn("organic_v2", program.allowed_concrete_executors)
        plan = SimpleNamespace(
            tool_program=SimpleNamespace(
                allowed_tools=("organic_v2",),
                parameter_ranges={
                    "joint_tissue_tool_program": program.to_metadata()
                },
            )
        )
        with self.assertRaises(JointContractError):
            validate_tissue_plan_tool_binding(plan, compiled=program)

    def test_tissue_tool_mapping_and_trace_binding_fail_closed(self):
        with self.assertRaisesRegex(JointContractError, "unmapped"):
            compile_tissue_tool_program(
                primitive_id="infiltrative-nest-cord-extension-v1",
                mechanism_id="breast-infiltrative-nest-cord-extension",
                mechanism_allowed_families=("unknown-family",),
                primitive_allowed_executors=("directional_tapered_projection",),
            )
        program = compile_tissue_tool_program(
            primitive_id="infiltrative-nest-cord-extension-v1",
            mechanism_id="breast-infiltrative-nest-cord-extension",
            mechanism_allowed_families=("directional_tapered_projection",),
            primitive_allowed_executors=("directional_tapered_projection",),
        )
        candidate = CandidateMask(
            candidate_id="forged",
            interface_id="interface-1",
            tool_name="directional_tapered_projection",
            target_mask=np.ones((8, 8), dtype=np.uint8),
            change_region=np.ones((8, 8), dtype=bool),
            tool_trace={
                "joint_tissue_tool_program": {"program_sha256": "forged"},
                "concrete_tissue_executor": "organic_v2",
            },
        )
        with self.assertRaisesRegex(JointContractError, "detached"):
            _bind_and_validate_tissue_candidate_traces((candidate,), program)

    def test_directional_extension_raster_adversaries_fail_closed(self):
        shape = (80, 80)
        parent = np.zeros(shape, dtype=bool)
        parent[20:61, 5:25] = True
        anchor = np.zeros(shape, dtype=bool)
        anchor[32:49, 24] = True

        def valid_projection():
            result = np.zeros(shape, dtype=bool)
            for col in range(25, 61):
                half_width = max(1, round(8 * (60 - col) / 35))
                result[40 - half_width : 41 + half_width, col] = True
            return result

        valid = valid_projection()
        self.assertTrue(
            audit_directional_extension_raster(
                change=valid,
                parent=parent,
                other_tumor=np.zeros(shape, dtype=bool),
                selected_anchor=anchor,
                nominal_nucleus_diameter_px=5,
            )["passed"]
        )

        double = valid_projection()
        double[10:13, 45:50] = True
        rows, cols = np.ogrid[: shape[0], : shape[1]]
        broad = ((rows - 40) / 14) ** 2 + ((cols - 43) / 20) ** 2 <= 1
        branched = np.zeros(shape, dtype=bool)
        branched[25:42, 25:30] = True
        for col in range(30, 61):
            upper = 40 - (col - 30) // 3
            lower = 40 + (col - 30) // 3
            branched[upper - 1 : upper + 2, col] = True
            branched[lower - 1 : lower + 2, col] = True
        merged = valid_projection()
        merged[35:46, 60:63] = True
        second_parent = np.zeros(shape, dtype=bool)
        second_parent[31:50, 62:69] = True
        cases = {
            "two separated projections": (double, np.zeros(shape, bool), anchor),
            "broad ellipse": (broad, np.zeros(shape, bool), anchor),
            "Y branch": (branched, np.zeros(shape, bool), anchor),
            "side merge": (merged, second_parent, anchor),
            "wrong anchor": (valid, np.zeros(shape, bool), np.roll(anchor, 25, axis=0)),
        }
        for label, (change, other_tumor, selected_anchor) in cases.items():
            with self.subTest(label=label):
                audit = audit_directional_extension_raster(
                    change=change,
                    parent=parent,
                    other_tumor=other_tumor,
                    selected_anchor=selected_anchor,
                    nominal_nucleus_diameter_px=5,
                )
                self.assertFalse(audit["passed"])

    def test_final_added_focus_audit_ignores_forged_cluster_trace(self):
        shape = (80, 80)
        source = np.zeros(shape, dtype=bool)
        source[35:38, 10:13] = True
        valid_extent = np.zeros(shape, dtype=bool)
        valid_extent[5:75, 15:70] = True

        def target_and_ledger(centers):
            target = source.copy()
            ledger = []
            for row, col in centers:
                target[row - 1 : row + 2, col - 1 : col + 2] = True
                ledger.append(
                    {
                        "row": row,
                        "col": col,
                        "class_id": 1,
                        "cluster_size": 1,
                    }
                )
            return target, ledger

        target, ledger = target_and_ledger(((20, 25), (20, 31), (50, 55)))
        valid = audit_added_class1_foci(
            source_class1=source,
            target_class1=target,
            accepted_center_ledger=ledger,
            valid_footprint_region=valid_extent,
            nominal_nucleus_diameter_px=4,
        )
        self.assertEqual(valid["focus_sizes"], [2, 1])
        self.assertTrue(valid["ledger_matches_instances"])

        adversaries = {
            "single focus pretending multiple": ((20, 25), (20, 31), (20, 37)),
            "footprint crosses extent": ((6, 15), (30, 40)),
            "bridge to source": ((35, 14), (55, 55)),
        }
        for label, centers in adversaries.items():
            with self.subTest(label=label):
                target, ledger = target_and_ledger(centers)
                audit = audit_added_class1_foci(
                    source_class1=source,
                    target_class1=target,
                    accepted_center_ledger=ledger,
                    valid_footprint_region=valid_extent,
                    nominal_nucleus_diameter_px=4,
                )
                if label == "single focus pretending multiple":
                    self.assertEqual(audit["focus_count"], 1)
                elif label == "footprint crosses extent":
                    self.assertFalse(audit["all_footprints_in_extent"])
                else:
                    self.assertGreater(audit["source_bridge_pixels"], 0)

        target, ledger = target_and_ledger(((20, 25), (50, 55)))
        ledger = ledger[:1]
        forged = audit_added_class1_foci(
            source_class1=source,
            target_class1=target,
            accepted_center_ledger=ledger,
            valid_footprint_region=valid_extent,
            nominal_nucleus_diameter_px=4,
        )
        self.assertFalse(forged["ledger_matches_instances"])

    def test_final_focus_audit_accepts_two_seeded_touching_cluster_members(self):
        shape = (48, 48)
        source = np.zeros(shape, dtype=bool)
        target = source.copy()
        # Two complete 3x3 instances touch diagonally and therefore form one
        # 8-connected raster component, while retaining distinct executor
        # centers and a legal two-cell focus.
        target[18:21, 18:21] = True
        target[21:24, 21:24] = True
        audit = audit_added_class1_foci(
            source_class1=source,
            target_class1=target,
            accepted_center_ledger=(
                {"row": 19, "col": 19, "class_id": 1},
                {"row": 22, "col": 22, "class_id": 1},
            ),
            valid_footprint_region=np.ones(shape, dtype=bool),
            nominal_nucleus_diameter_px=4,
        )

        self.assertTrue(audit["ledger_matches_instances"])
        self.assertFalse(audit["one_center_per_raster_instance"])
        self.assertEqual(audit["focus_sizes"], [2])

    def test_final_focus_audit_measures_localized_tight_clusters(self):
        shape = (64, 64)
        source = np.zeros(shape, dtype=bool)
        centers = (
            (20, 20),
            (20, 24),
            (24, 20),
            (20, 36),
            (20, 40),
            (24, 36),
            (40, 28),
            (40, 32),
        )
        target = source.copy()
        ledger = []
        for row, col in centers:
            target[row, col] = True
            ledger.append({"row": row, "col": col, "class_id": 1})

        audit = audit_added_class1_foci(
            source_class1=source,
            target_class1=target,
            accepted_center_ledger=ledger,
            valid_footprint_region=np.ones(shape, dtype=bool),
            nominal_nucleus_diameter_px=4,
            focus_link_distance_diameters=1.35,
        )

        self.assertEqual(sorted(audit["focus_sizes"]), [2, 3, 3])
        self.assertLessEqual(audit["maximum_focus_diameter_px"], 9.0)
        self.assertGreater(
            audit["minimum_inter_focus_center_distance_px"], 9.0
        )
        self.assertLessEqual(audit["effect_center_span_px"], 30.0)

    def test_new_breast_primitives_bind_parser_mechanism_and_executor_contract(self):
        repository = JointSkillRepository()
        expected = {
            "cohesive-boundary-expansion-v1": (
                "breast-annotation-anchored-boundary-growth",
                "tissue_and_cell",
            ),
            "infiltrative-nest-cord-extension-v1": (
                "breast-infiltrative-nest-cord-extension",
                "tissue_and_cell",
            ),
            "peritumoral-neoplastic-scatter-increase-v1": (
                "breast-peritumoral-neoplastic-scatter",
                "cell_only",
            ),
            "peritumoral-small-cluster-increase-v1": (
                "breast-peritumoral-small-cluster",
                "cell_only",
            ),
        }
        checker_ids = set(JointGateRegistry().available_checker_ids)
        for primitive_id, (mechanism_id, scope) in expected.items():
            with self.subTest(primitive=primitive_id):
                case = replace(
                    _breast_case_stub(),
                    primitive_id=primitive_id,
                    joint_area_budget=(
                        JointAreaBudget() if scope == "tissue_and_cell" else None
                    ),
                    cell_count_extent_budget=(
                        None
                        if scope == "tissue_and_cell"
                        else CellCountExtentBudget(
                            8,
                            6,
                            12,
                            64,
                            0,
                            64,
                            minimum_effect_span_px=32,
                            minimum_effect_foci=3,
                        )
                    ),
                )
                bundle = repository.compose(
                    case=case,
                    mechanism_id=mechanism_id,
                    available_checker_ids=checker_ids,
                    production=False,
                )
                self.assertEqual(bundle.primitive.scope, scope)
                self.assertEqual(
                    tissue_tool_primitive_id(primitive_id),
                    (
                        "tumor-burden-increase-v1"
                        if scope == "tissue_and_cell"
                        else primitive_id
                    ),
                )
                self.assertTrue(
                    set(bundle.required_checker_ids).issubset(checker_ids)
                )
                if primitive_id == "peritumoral-tumor-nest-formation-v1":
                    self.assertGreaterEqual(
                        bundle.mechanism.tissue_program.front.maximum_band_px,
                        144,
                    )

    def test_regression_and_fragmentation_bind_visible_effect_contracts(self):
        repository = JointSkillRepository()
        checker_ids = set(JointGateRegistry().available_checker_ids)
        expected = {
            "invasive-tumor-footprint-decrease-v1": (
                "breast-post-treatment-invasive-regression",
                0.08,
                1,
            ),
            "residual-tumor-fragmentation-v1": (
                "breast-residual-disease-fragmentation",
                0.12,
                1,
            ),
        }
        for primitive_id, (mechanism_id, minimum_changed, maximum_sources) in (
            expected.items()
        ):
            with self.subTest(primitive=primitive_id):
                case = replace(
                    _breast_case_stub(),
                    primitive_id=primitive_id,
                    joint_area_budget=JointAreaBudget(),
                    semantic_intent={
                        "treatment_context": "post_treatment",
                        "scenario": (
                            "residual_disease"
                            if primitive_id
                            == "residual-tumor-fragmentation-v1"
                            else "treatment_response"
                        ),
                    },
                )
                bundle = repository.compose(
                    case=case,
                    mechanism_id=mechanism_id,
                    available_checker_ids=checker_ids,
                    production=False,
                )
                primitive = bundle.primitive
                self.assertEqual(
                    primitive.minimum_source_component_changed_fraction,
                    minimum_changed,
                )
                self.assertEqual(
                    primitive.maximum_selected_source_components,
                    maximum_sources,
                )
                if primitive_id == "residual-tumor-fragmentation-v1":
                    self.assertEqual(primitive.minimum_residual_components, 2)
                    self.assertEqual(primitive.maximum_residual_components, 6)
                    self.assertEqual(
                        primitive.maximum_residual_area_fraction, 0.88
                    )
                    self.assertEqual(
                        primitive.minimum_residual_component_fraction, 0.025
                    )
                    self.assertEqual(
                        primitive.maximum_dominant_residual_component_fraction,
                        0.75,
                    )
                    self.assertEqual(primitive.minimum_residual_spacing_px, 24)
                    self.assertEqual(
                        bundle.mechanism.cell_program.seam.density_ratio_range,
                        (0.2, 2.5),
                    )
                    self.assertTrue(
                        any(
                            "thin-to-thick-to-thin" in finding
                            for finding in bundle.mechanism.render.veto_findings
                        )
                    )

    def test_breast_bcss_capability_matrix_matches_executable_skills(self):
        repository = JointSkillRepository()
        matrix_path = (
            Path(__file__).parents[1]
            / "phase3_joint_edit_refine"
            / "resources"
            / "breast_bcss_capability_matrix_v1.json"
        )
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        entries = matrix["capabilities"]
        pairs = {
            (item["primitive_id"], item["mechanism_id"])
            for item in entries
        }
        expected = {
            (primitive_id, mechanism.mechanism_id)
            for mechanism in repository.mechanisms.values()
            if mechanism.pathology_domain_id
            == "breast-invasive-carcinoma-v1"
            for primitive_id in mechanism.supported_primitives
            if repository.execution_selection_reason(
                primitive_id=primitive_id,
                mechanism_id=mechanism.mechanism_id,
            )
            is None
        }
        self.assertEqual(pairs, expected)
        self.assertEqual(len(entries), len(pairs))
        self.assertEqual(matrix["production_status"], "shadow_only")
        selection_contract = matrix["profile_policy"][
            "planner_selection_contract"
        ]
        tissue_decision = next(
            value
            for value in JOINT_TISSUE_DECISION_SCHEMA["properties"][
                "decision_id"
            ]["enum"]
            if value is not None
        )
        cell_decision = next(
            value
            for value in CELL_PLAN_SELECTION_SCHEMA["properties"][
                "decision_id"
            ]["enum"]
            if value is not None
        )
        self.assertEqual(
            selection_contract["tissue"],
            {
                "decision_id": tissue_decision,
                "selectable_fields": [
                    "selected_candidate_id",
                    "selected_tool_family",
                    "supporting_preference_rule_ids",
                ],
            },
        )
        self.assertEqual(
            selection_contract["cell"],
            {
                "decision_id": cell_decision,
                "selectable_fields": [
                    "selected_candidate_id",
                    "selected_tool_program_id",
                    "supporting_preference_rule_ids",
                ],
            },
        )
        for stage, schema in (
            ("tissue", JOINT_TISSUE_DECISION_SCHEMA),
            ("cell", CELL_PLAN_SELECTION_SCHEMA),
        ):
            with self.subTest(stage=stage):
                self.assertIn(
                    selection_contract[stage]["decision_id"],
                    PLANNER_DECISIONS,
                )
                self.assertTrue(
                    set(selection_contract[stage]["selectable_fields"])
                    .issubset(schema["properties"])
                )
        for item in entries:
            self.assertTrue(item["simple_instructions"])
            self.assertTrue(item["tissue_action"])
            self.assertTrue(item["nuclei_action"])
            self.assertTrue(item["required_conditions"])
            self.assertEqual(item["production_status"], "shadow_only")

    def test_breast_matrix_planner_roles_cannot_expand_llm_spatial_authority(self):
        matrix_path = (
            Path(__file__).parents[1]
            / "phase3_joint_edit_refine"
            / "resources"
            / "breast_bcss_capability_matrix_v1.json"
        )
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        compiler_owned = set(
            matrix["profile_policy"]["planner_selection_contract"][
                "compiler_owned_fields"
            ]
        )
        self.assertEqual(
            compiler_owned,
            {
                "interfaces",
                "components",
                "anchors",
                "zones",
                "annuli",
                "layouts",
                "parameter_values",
                "parameter_ranges",
                "area",
                "depth",
                "geometry",
                "cell_counts",
                "coordinates",
                "pixels",
            },
        )
        tissue_role = (
            "The LLM ranks compiler-certified immutable tissue candidates and "
            "selects only a candidate ID, an allowed tool family, and supporting "
            "skill preference IDs. The selected certificate already binds every "
            "interface, component, anchor, parameter value and range, area, depth, "
            "and geometry; the LLM cannot alter the certificate."
        )
        cell_role = (
            "The LLM ranks compiler-certified immutable cell candidates and selects "
            "only a candidate ID, an allowed cell tool program ID, and supporting "
            "skill preference IDs. The selected certificate already binds every "
            "interface, component, zone, annulus, layout, cell count, coordinate, "
            "pixel footprint, parameter value and range; the LLM cannot alter the "
            "certificate."
        )
        forbidden_selection = re.compile(
            r"\b(?:choose|select|specify|modify|set|construct)\w*\s+"
            r"(?:(?:a|an|the|any|permitted|allowed)\s+)?"
            r"(?:interface|component|anchor|zone|annulus|layout|parameter|range|"
            r"area|depth|geometry|count|coordinate|pixel)s?\b",
            flags=re.IGNORECASE,
        )
        for item in matrix["capabilities"]:
            role = item["planner_role"]
            expected = (
                cell_role
                if item["execution_scope"] == "cell_only"
                else tissue_role
            )
            with self.subTest(primitive=item["primitive_id"]):
                self.assertEqual(role, expected)
                self.assertIsNone(forbidden_selection.search(role))

    def test_breast_capability_matrix_instructions_parse_to_declared_primitive(self):
        matrix_path = (
            Path(__file__).parents[1]
            / "phase3_joint_edit_refine"
            / "resources"
            / "breast_bcss_capability_matrix_v1.json"
        )
        entries = json.loads(matrix_path.read_text(encoding="utf-8"))[
            "capabilities"
        ]
        parser = RuleBasedSemanticParser()
        for entry in entries:
            expected = entry["primitive_id"]
            for instruction in entry["simple_instructions"]:
                with self.subTest(primitive=expected, instruction=instruction):
                    intent = parser.parse(instruction)
                    declared = {
                        intent.primitive_id,
                        *(
                            item.primitive_id
                            for item in intent.primitive_hypotheses
                        ),
                    }
                    self.assertIn(expected, declared)

    def test_mask_condition_critic_does_not_receive_histology_veto_authority(self):
        client = _RecordingJointCriticClient()
        repository = JointSkillRepository()
        case = _breast_case_stub()
        bundle = repository.compose(
            case=case,
            mechanism_id="breast-annotation-anchored-boundary-growth",
            available_checker_ids=JointGateRegistry().available_checker_ids,
            production=False,
        )

        result = OpenAIMultimodalJointCritic(client).review(
            case=case,
            bundle=bundle,
            candidates=(),
            gate_reports=(),
            image_paths=(),
        )

        self.assertTrue(result.abstain)
        payload = json.loads(client.calls[0]["user_prompt"])
        self.assertNotIn("annotation_visual_veto_requirements", payload)
        self.assertTrue(payload["requirements"]["source_H&E_is_prohibited"])
        self.assertIn("Raw H&E is prohibited", client.calls[0]["system_prompt"])

    def test_mask_condition_critic_direct_raster_bypass_fails_closed(self):
        case = _breast_case_stub()
        repository = JointSkillRepository()
        bundle = repository.compose(
            case=case,
            mechanism_id="breast-annotation-anchored-boundary-growth",
            available_checker_ids=JointGateRegistry().available_checker_ids,
            production=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            disguised = Path(tmp) / "joint_condition_mask_review.png"
            Image.new("RGB", (8, 8), "red").save(disguised)
            with self.assertRaisesRegex(
                JointContractError, "artifact registry"
            ):
                OpenAIMultimodalJointCritic(
                    _RecordingJointCriticClient()
                ).review(
                    case=case,
                    bundle=bundle,
                    candidates=(),
                    gate_reports=(),
                    image_paths=(disguised,),
                    artifact_registry=None,
                )

    def test_planner_and_critic_free_text_cannot_assert_unannotated_pathology(self):
        tissue_client = _CertifiedTissueSelectionClient(
            lambda value: {
                **value,
                "selection_explanation": "best fibrotic tumor bed margin",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = _as_breast_growth_case(_write_synthetic_case(root))
            case = replace(
                case,
                case_id="claim-filter-tissue",
                instruction="cohesive-boundary-expansion-v1",
                primitive_id="cohesive-boundary-expansion-v1",
                provenance={
                    **case.provenance,
                    "joint_mechanism_id": (
                        "breast-annotation-anchored-boundary-growth"
                    ),
                    "joint_primitive_id": "cohesive-boundary-expansion-v1",
                },
            )
            result = JointPathologyEditWorkflow(
                tissue_planner=OpenAIJointAwareTissuePlanner(
                    client=tissue_client,
                    max_contract_attempts=1,
                ),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
            ).run(case, output_root=root / "result")
            self.assertEqual(result.status, "abstained")
            feedback_path = next(
                (root / "result" / case.case_id).glob(
                    "execution_feedback_pass_*.json"
                )
            )
            feedback = json.loads(feedback_path.read_text())
            self.assertIn("neutral audit token", feedback["errors"][0])

        case = _breast_case_stub()
        bundle = JointSkillRepository().compose(
            case=case,
            mechanism_id="breast-annotation-anchored-boundary-growth",
            available_checker_ids=JointGateRegistry().available_checker_ids,
            production=False,
        )

        class ClaimingCriticClient:
            def __init__(self, value):
                self.value = value

            def call(self, **_kwargs):
                return {
                    "rankings": [],
                    "abstain": True,
                    "summary": self.value,
                }, {"model": "fixture"}

        variants = (
            "desmoplastic reaction is present",
            "the invasive front is histologically evident",
            "fibrotic stroma marks a treatment bed",
            "this is luminal A disease",
            "lymphovascular invasion is present",
            "residual cancer burden is low",
        )
        for value in variants:
            with self.subTest(value=value), self.assertRaisesRegex(
                JointContractError, "free text"
            ):
                OpenAIMultimodalJointCritic(
                    ClaimingCriticClient(value)
                ).review(
                    case=case,
                    bundle=bundle,
                    candidates=(),
                    gate_reports=(),
                    image_paths=(),
                )

        class ClaimingVetoClient:
            def __init__(self, value):
                self.value = value

            def call(self, **_kwargs):
                return {
                    "rankings": [
                        {
                            "candidate_id": "joint:fixture",
                            "score": 0.0,
                            "confidence": 0.8,
                            "supporting_rule_ids": list(
                                bundle.active_rule_ids
                            ),
                            "veto_reasons": [self.value],
                        }
                    ],
                    "abstain": True,
                    "summary": "certified_mask_condition_abstained",
                }, {"model": "fixture"}

        for value in variants:
            with self.subTest(veto=value), self.assertRaisesRegex(
                JointContractError, "free text"
            ):
                OpenAIMultimodalJointCritic(
                    ClaimingVetoClient(value)
                ).review(
                    case=case,
                    bundle=bundle,
                    candidates=(),
                    gate_reports=(
                        JointGateReport("joint:fixture", True, ()),
                    ),
                    image_paths=(),
                )

    def test_semantic_and_cell_selection_text_use_closed_audit_tokens(self):
        case = _breast_case_stub()
        mechanism = JointSkillRepository().mechanisms[
            "breast-annotation-anchored-boundary-growth"
        ]
        option = JointInterpretationOption(
            primitive_id="cohesive-boundary-expansion-v1",
            semantic_fit="direct",
            semantic_priority=0,
            semantic_rationale="fixture",
            mechanism=mechanism,
            feasibility={
                "aggregate_tissue_capacity_pixels": 100,
                "meaningful_tissue_floor_pixels": 10,
                "feasible_interface_count": 1,
            },
        )
        variants = (
            "desmoplastic reaction is present",
            "the invasive front is histologically evident",
            "fibrotic stroma marks a treatment bed",
            "this is luminal A disease",
            "lymphovascular invasion is present",
            "residual cancer burden is low",
        )

        class SemanticClient:
            def __init__(self, value):
                self.value = value

            def call(self, **_kwargs):
                return {
                    "abstain": False,
                    "abstain_reason": None,
                    "clarification_required": False,
                    "clarification_reason": None,
                    "clarification_primitive_ids": [],
                    "primitive_id": option.primitive_id,
                    "mechanism_id": mechanism.mechanism_id,
                    "decision_id": "select_primitive_mechanism_pair",
                    "interpretation_explanation": self.value,
                    "supporting_observations": [
                        "certified_capability_metrics"
                    ],
                    "supporting_capability_metric_ids": [
                        "semantic_priority"
                    ],
                    "observed_contraindications": [],
                    "confidence": 0.8,
                }, {}

        for value in variants:
            for field in (
                "interpretation_explanation",
                "supporting_observations",
            ):
                client = SemanticClient(value)
                if field == "supporting_observations":
                    original_call = client.call

                    def replace_observation(
                        *,
                        _original_call=original_call,
                        _value=value,
                        **kwargs,
                    ):
                        response, usage = _original_call(**kwargs)
                        response["interpretation_explanation"] = (
                            "certified_semantic_option_selected"
                        )
                        response["supporting_observations"] = [_value]
                        return response, usage

                    client.call = replace_observation
                with self.subTest(field=field, value=value), self.assertRaisesRegex(
                    JointContractError, "neutral audit token"
                ):
                    OpenAIMultimodalJointPlanner(
                        client=client, max_contract_attempts=1
                    ).select_interpretation(
                        case=case,
                        scene=SimpleNamespace(to_metadata=dict),
                        options=(option,),
                        image_paths=(),
                    )

    def test_multimodal_plan_schema_requires_unique_structural_units(self):
        structural_units = JOINT_PLAN_JSON_SCHEMA["properties"][
            "structural_unit_ids"
        ]
        self.assertTrue(structural_units["uniqueItems"])

    def test_structural_unit_selection_deduplicates_auxiliary_observations(self):
        scene = SimpleNamespace(
            structural_hierarchy={
                "structure_units": [
                    {
                        "unit_id": "fine:8:unit:0002",
                        "parent_tissue_component_id": "cmp:tumor:0001",
                    },
                    {
                        "unit_id": "fine:8:unit:0001",
                        "parent_tissue_component_id": "cmp:tumor:0001",
                    },
                    {
                        "unit_id": "fine:8:unit:0002",
                        "parent_tissue_component_id": "cmp:tumor:0001",
                    },
                    {
                        "unit_id": "fine:9:unit:0001",
                        "parent_tissue_component_id": "cmp:tumor:0002",
                    },
                ]
            }
        )

        selected = _structural_units_for_components(
            scene, ("cmp:tumor:0001",)
        )

        self.assertEqual(
            selected,
            ("fine:8:unit:0001", "fine:8:unit:0002"),
        )

    def test_scene_merges_duplicate_structural_unit_producer_bindings(self):
        tissue = np.full((32, 32), 2, dtype=np.uint8)
        tissue[8:24, 8:24] = 8
        nuclei = np.zeros_like(tissue, dtype=np.uint8)
        unit_mask = tissue == 8
        digest = hashlib.sha256(
            np.packbits(unit_mask.astype(np.uint8), axis=None).tobytes()
        ).hexdigest()
        unit = {
            "unit_id": "fine:8:unit:0001",
            "fine_id": 8,
            "component_sha256": digest,
            "enclosed_space_ids": [],
        }
        scene = build_joint_scene_analysis(
            tissue,
            nuclei,
            schema=MaskProfileSchema.from_reference_profile("PANDA"),
            pixel_size_um=None,
            auxiliary_structure_provenance={
                "native_pattern_map": {"structure_units": [unit]},
                "native_pattern_and_lumen_map": {
                    "structure_units": [unit]
                },
            },
        )

        records = scene.structural_hierarchy["structure_units"]
        self.assertEqual(len(records), 1)
        self.assertEqual(
            records[0]["auxiliary_structure_ids"],
            ["native_pattern_and_lumen_map", "native_pattern_map"],
        )

    def test_codex_visual_plan_overrides_are_digest_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "shadow.json"
            overrides = root / "overrides.json"
            output = root / "reviewed.json"
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "case_id": "case-1",
                            "primitive_id": "cellularity-decrease-v1",
                            "provenance": {},
                        }
                    ]
                ),
                encoding="utf-8",
            )
            overrides.write_text(
                json.dumps(
                    {
                        "schema_version": (
                            "g2-v2-codex-visual-plan-overrides-v1"
                        ),
                        "reviewer": "current_codex_session",
                        "cases": {
                            "case-1": {
                                "cellularity_depletion_anchor": {
                                    "type": "interface",
                                    "interface_ids": ["if-1"],
                                    "anchor_ids": ["anchor-1"],
                                    "observation": "visible density transition",
                                    "confidence": 0.9,
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = apply_plan_overrides(
                manifest, overrides, output_path=output
            )
            reviewed = json.loads(output.read_text(encoding="utf-8"))[0]
            audit = reviewed["provenance"]["codex_visual_plan_override"]
            self.assertEqual(audit["reviewer"], "current_codex_session")
            self.assertEqual(
                result["output_manifest_sha256"], _sha(output)
            )
            self.assertTrue(output.with_suffix(".json.sha256").is_file())

    def test_cellularity_and_decrease_skills_require_a_macroscopic_density_effect(self):
        repository = JointSkillRepository()
        for primitive_id in (
            "cell-type-abundance-decrease-v1",
            "cellularity-decrease-v1",
            "cellularity-increase-v1",
        ):
            with self.subTest(primitive_id=primitive_id):
                primitive = repository.primitives[primitive_id]
                self.assertGreaterEqual(primitive.minimum_effect_delta_count, 12)
                self.assertGreaterEqual(
                    primitive.minimum_effect_span_cell_diameters, 6.0
                )
                if primitive_id == "cellularity-increase-v1":
                    self.assertGreaterEqual(primitive.minimum_effect_foci, 4)
        abundance_decrease = repository.primitives[
            "cell-type-abundance-decrease-v1"
        ]
        self.assertEqual(
            abundance_decrease.minimum_effect_delta_count_for(
                "breast-invasive-carcinoma-v1"
            ),
            12,
        )
        self.assertEqual(
            abundance_decrease.minimum_effect_delta_count_for(
                "prostate-adenocarcinoma-v1"
            ),
            2,
        )

    def test_density_gradient_quota_can_fill_transition_shortfall(self):
        repaired = _enforce_density_field_gradient_quotas(
            quotas=[7, 2, 1, 1, 0],
            source_counts=[13, 5, 2, 4, 1],
            maximum_removals=[7, 2, 1, 2, 0],
            target_fractions=[0.55, 0.42, 0.38, 0.34, 0.20],
            minimum_count=12,
            maximum_count=18,
            minimum_core=2,
            minimum_transition=2,
        )

        self.assertEqual(repaired, [7, 2, 1, 2, 0])

    def test_cluster_members_each_require_a_legal_center(self):
        shape = np.ones((3, 3), dtype=bool)
        legal = np.zeros((40, 40), dtype=bool)
        legal[20, 20] = True
        valid = np.ones_like(legal)
        target, placed, trace = _place_layout(
            base=np.zeros_like(legal, dtype=np.uint8),
            references=(
                ReferenceNucleusShape("ref", 2, shape, "test", 9),
            ),
            class_id=2,
            legal_zone=legal,
            valid_footprint_region=valid,
            halo=np.zeros_like(legal),
            score=legal.astype(float),
            requested_count=3,
            layout_program="small_cluster",
            cluster_size_range=(3, 3),
            nominal_nucleus_diameter_px=8.0,
            orientation_mask=np.zeros_like(legal),
            continuity_region=np.zeros_like(legal),
            continuity_anchor_mask=np.zeros_like(legal),
            continuity_maximum_empty_run_px=0,
            continuity_minimum_anchor_coverage_fraction=0.0,
            continuity_preferred_count=0,
            minimum_effect_span_px=0,
            minimum_effect_foci=0,
            seed=1,
        )
        self.assertEqual(placed, 0)
        self.assertEqual(len(trace), 0)
        self.assertEqual(int(np.count_nonzero(target)), 0)

    def test_meaningful_cell_effect_is_distributed_across_foci(self):
        shape = np.ones((3, 3), dtype=bool)
        legal = np.zeros((80, 80), dtype=bool)
        legal[4:-4, 4:-4] = True
        score = np.zeros_like(legal, dtype=float)
        score[40, 40] = 10.0
        _target, placed, trace = _place_layout(
            base=np.zeros_like(legal, dtype=np.uint8),
            references=(
                ReferenceNucleusShape("ref", 2, shape, "test", 9),
            ),
            class_id=2,
            legal_zone=legal,
            valid_footprint_region=legal,
            halo=legal,
            score=score,
            requested_count=12,
            layout_program="small_cluster",
            cluster_size_range=(1, 4),
            nominal_nucleus_diameter_px=8.0,
            orientation_mask=np.zeros_like(legal),
            continuity_region=np.zeros_like(legal),
            continuity_anchor_mask=np.zeros_like(legal),
            continuity_maximum_empty_run_px=0,
            continuity_minimum_anchor_coverage_fraction=0.0,
            continuity_preferred_count=0,
            minimum_effect_span_px=40,
            minimum_effect_foci=4,
            enforce_small_cluster_group_separation=False,
            seed=1,
        )
        self.assertEqual(placed, 12)
        self.assertGreaterEqual(len({item["cluster_id"] for item in trace}), 4)
        centers = np.asarray([item["center_xy"] for item in trace], dtype=float)
        distances = centers[:, None, :] - centers[None, :, :]
        self.assertGreaterEqual(
            float(np.sqrt(np.max(np.sum(distances**2, axis=2)))), 40.0
        )
        effect_geometry = audit_cell_effect_foci(
            centers_xy=(item["center_xy"] for item in trace),
            nominal_nucleus_diameter_px=8.0,
        )
        self.assertGreaterEqual(
            effect_geometry["raw_spatial_component_count"], 4
        )
        self.assertTrue(effect_geometry["spatial_focus_contract_passed"])

    def test_small_cluster_count_uses_enough_bounded_groups(self):
        shape = np.ones((1, 1), dtype=bool)
        legal = np.zeros((96, 96), dtype=bool)
        legal[4:-4, 4:-4] = True
        score = np.zeros_like(legal, dtype=float)

        _target, placed, trace = _place_layout(
            base=np.zeros_like(legal, dtype=np.uint8),
            references=(
                ReferenceNucleusShape("ref", 1, shape, "test", 1),
            ),
            class_id=1,
            legal_zone=legal,
            valid_footprint_region=legal,
            halo=legal,
            score=score,
            requested_count=12,
            layout_program="small_cluster",
            cluster_size_range=(1, 4),
            nominal_nucleus_diameter_px=2.0,
            orientation_mask=np.zeros_like(legal),
            continuity_region=np.zeros_like(legal),
            continuity_anchor_mask=np.zeros_like(legal),
            continuity_maximum_empty_run_px=0,
            continuity_minimum_anchor_coverage_fraction=0.0,
            continuity_preferred_count=0,
            minimum_effect_span_px=18,
            minimum_effect_foci=2,
            enforce_small_cluster_group_separation=False,
            seed=1,
        )

        self.assertEqual(placed, 12)
        group_sizes = Counter(item["cluster_id"] for item in trace)
        self.assertGreaterEqual(len(group_sizes), 3)
        self.assertLessEqual(max(group_sizes.values()), 4)

    def test_effect_span_uses_exact_legal_center_diameter(self):
        # From this ranked first point, two farthest-point sweeps reach only
        # 46.62 px while the true legal-center diameter is 52.70 px.
        anchors = np.asarray(
            [
                [8, 39],
                [12, 26],
                [18, 4],
                [33, 3],
                [34, 46],
                [35, 11],
                [46, 12],
                [47, 48],
            ],
            dtype=int,
        )

        ordered = _effect_first_anchors(
            anchors,
            minimum_effect_span_px=50,
            minimum_effect_foci=2,
        )

        self.assertGreaterEqual(
            float(np.linalg.norm(ordered[0] - ordered[1])),
            50.0,
        )

    def test_exact_packing_witnesses_precede_probnet_anchor_order(self):
        anchors = np.asarray(
            [[4, 4], [8, 8], [12, 12], [16, 16]],
            dtype=int,
        )

        ordered = _certified_witness_first_anchors(
            anchors,
            certified_witness_centers=((12, 12), (4, 4)),
        )

        self.assertEqual(
            ordered.tolist(),
            [[12, 12], [4, 4], [8, 8], [16, 16]],
        )
        self.assertTrue(
            _centers_satisfy_minimum_span(
                ((44, 375), (200, 310), (90, 429)),
                minimum_span_px=126,
            )
        )

        self.assertFalse(
            _centers_satisfy_minimum_span(
                ((44, 375), (90, 429)),
                minimum_span_px=126,
            )
        )
        # Witness execution is a property of a passing independent-focus
        # certificate, not only of the optional smallest-shape fallback. A
        # nominal mixed-shape certificate owns equally exact legal centers.
        nominal_certificate = {
            "passed": True,
            "capacity_optimized_shape_fallback_used": False,
            "placements": [
                {"row": 44, "col": 375, "class_id": 1},
                {"row": 200, "col": 310, "class_id": 1},
            ],
        }
        nominal_centers = tuple(
            (item["row"], item["col"])
            for item in nominal_certificate["placements"]
            if nominal_certificate["passed"]
        )
        self.assertTrue(
            _centers_satisfy_minimum_span(
                nominal_centers,
                minimum_span_px=126,
            )
        )

    def test_colorectal_cellularity_increase_uses_clustered_foci(self):
        mechanism = JointSkillRepository().mechanisms[
            "colorectal-local-population-modulation"
        ]
        self.assertEqual(
            mechanism.cell_program.layout_for("cellularity-increase-v1"),
            "small_cluster",
        )

    def test_raster_native_instance_authority_exactly_partitions_semantics(self):
        semantic = np.zeros((12, 12), dtype=np.uint8)
        semantic[2:5, 2:5] = 1
        semantic[7:10, 7:11] = 2
        labels = np.zeros_like(semantic, dtype=np.int32)
        labels[2:5, 2:5] = 1
        labels[7:10, 7:11] = 2
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            label_map = root / "instances.npy"
            np.save(label_map, labels, allow_pickle=False)
            manifest = root / "instances.json"
            manifest.write_text(
                json.dumps(
                    {
                        "raster_instance_authority": {
                            "label_map_uri": label_map.name,
                            "label_map_sha256": hashlib.sha256(
                                label_map.read_bytes()
                            ).hexdigest(),
                            "instances": [
                                {
                                    "label_id": 1,
                                    "type": 1,
                                    "seed_source": "cellvit",
                                },
                                {
                                    "label_id": 2,
                                    "type": 2,
                                    "seed_source": "semantic_fallback",
                                },
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )

            instances = load_native_instances(
                manifest,
                shape=semantic.shape,
                semantic_mask=semantic,
            )

        self.assertEqual(len(instances), 2)
        self.assertEqual(
            {item[0] for item in instances},
            {
                "native-raster-cellvit-00001",
                "native-raster-semantic-fallback-00002",
            },
        )
        reconstructed = np.zeros_like(semantic)
        for _instance_id, class_id, component in instances:
            reconstructed[component] = class_id
        np.testing.assert_array_equal(reconstructed, semantic)

    def test_semantic_fallback_is_not_a_reference_shape_authority(self):
        semantic = np.zeros((24, 24), dtype=np.uint8)
        semantic[4:9, 4:9] = 1
        semantic[16, 16] = 1
        labels = np.zeros_like(semantic, dtype=np.int32)
        labels[4:9, 4:9] = 1
        labels[16, 16] = 2
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            label_map = root / "instances.npy"
            np.save(label_map, labels, allow_pickle=False)
            manifest = root / "instances.json"
            manifest.write_text(
                json.dumps(
                    {
                        "raster_instance_authority": {
                            "label_map_uri": label_map.name,
                            "label_map_sha256": hashlib.sha256(
                                label_map.read_bytes()
                            ).hexdigest(),
                            "instances": [
                                {
                                    "label_id": 1,
                                    "type": 1,
                                    "seed_source": "cellvit",
                                },
                                {
                                    "label_id": 2,
                                    "type": 1,
                                    "seed_source": "semantic_fallback",
                                },
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            scene = build_joint_scene_analysis(
                np.full_like(semantic, 2),
                semantic,
                schema=MaskProfileSchema.from_reference_profile("GLaS"),
                pixel_size_um=None,
                nuclei_instances_path=str(manifest),
            )
            references, rejected = build_reference_shape_library(
                scene,
                class_id=1,
            )

        self.assertEqual(
            [item.instance_id for item in references],
            ["native-raster-cellvit-00001"],
        )
        self.assertEqual(
            rejected["native-raster-semantic-fallback-00002"],
            "semantic_fallback_not_morphology_authority",
        )
        selected = CellToolProgramCompiler._select_removal_instances(
            scene=scene,
            center_region=np.ones_like(semantic, dtype=bool),
            cell_classes=(1,),
            protected_instance_ids=(),
            target_count=2,
            minimum_count=1,
            preserve_class_composition=False,
        )
        self.assertEqual(selected, ("native-raster-cellvit-00001",))

    def test_native_partition_distinguishes_unseeded_from_seeded_residual(self):
        from scripts.run_glas_primitive_mask_review import (
            _semantic_partition_from_native_seeds,
        )

        semantic = np.zeros((24, 24), dtype=np.uint8)
        semantic[3:9, 3:9] = 1
        semantic[14:20, 14:20] = 1
        seed = np.zeros_like(semantic, dtype=bool)
        seed[4:8, 4:8] = True

        labels, records, native_count = _semantic_partition_from_native_seeds(
            semantic,
            [(1, seed)],
        )

        self.assertEqual(native_count, 1)
        self.assertEqual(
            {item["seed_source"] for item in records},
            {
                "cellvit",
                "semantic_seeded_residual",
                "semantic_unseeded",
            },
        )
        np.testing.assert_array_equal(labels > 0, semantic > 0)

    def test_glas_immune_decrease_screen_defers_native_count_to_compiler(self):
        from scripts.run_glas_primitive_mask_review import _eligible_and_score

        eligible, _score = _eligible_and_score(
            "cell-type-abundance-decrease-v1",
            {
                "complete_instance_counts": {"2": 1},
                "malignant_gland_instance_counts": {},
                "class_spans_px": {"2": 24.0},
                "class_local_count_radius_6d_max": {"2": 1},
            },
        )

        self.assertTrue(eligible)

    def test_glas_review_parser_separates_rank_and_seed_offsets(self):
        from scripts.run_glas_primitive_mask_review import build_parser

        parser = build_parser()
        seed_action = next(
            action for action in parser._actions if action.dest == "seed_offset"
        )

        self.assertEqual(seed_action.default, 0)
        self.assertNotEqual(seed_action.dest, "attempt_offset")

        portfolio_action = next(
            action
            for action in parser._actions
            if action.dest == "portfolio_index"
        )
        self.assertEqual(portfolio_action.default, 0)

        removal_action = next(
            action
            for action in parser._actions
            if action.dest == "removal_variant"
        )
        self.assertEqual(removal_action.default, 0)

    def test_depletion_removal_variant_changes_complete_instance_choice(self):
        items = [
            SimpleNamespace(
                instance_id=f"cell-{index}",
                centroid_xy=(float(col), float(row)),
            )
            for index, (row, col) in enumerate(
                ((2, 2), (2, 18), (10, 10), (18, 2), (18, 18))
            )
        ]
        default = _select_density_field_removals_preserving_coverage(
            [items],
            removal_quotas=[2],
            fixed_retained=[],
            selection_variant=0,
        )
        variant = _select_density_field_removals_preserving_coverage(
            [items],
            removal_quotas=[2],
            fixed_retained=[],
            selection_variant=1,
        )

        self.assertEqual(len(default), 2)
        self.assertEqual(len(variant), 2)
        self.assertNotEqual(
            {item.instance_id for item in default},
            {item.instance_id for item in variant},
        )

    def test_native_seed_scale_ignores_tiny_semantic_residual_records(self):
        semantic = np.zeros((64, 64), dtype=np.uint8)
        labels = np.zeros_like(semantic, dtype=np.int32)
        records = []
        label_id = 0
        for row, col in ((8, 8), (8, 32), (32, 8), (32, 32)):
            label_id += 1
            semantic[row : row + 5, col : col + 5] = 2
            labels[row : row + 5, col : col + 5] = label_id
            records.append(
                {"label_id": label_id, "type": 2, "seed_source": "cellvit"}
            )
        for index in range(12):
            row = 52 + index // 6
            col = 4 + 8 * (index % 6)
            label_id += 1
            semantic[row, col] = 2
            labels[row, col] = label_id
            records.append(
                {
                    "label_id": label_id,
                    "type": 2,
                    "seed_source": "semantic_seeded_residual",
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            label_map = root / "instances.npy"
            np.save(label_map, labels, allow_pickle=False)
            manifest = root / "instances.json"
            manifest.write_text(
                json.dumps(
                    {
                        "raster_instance_authority": {
                            "label_map_uri": label_map.name,
                            "label_map_sha256": hashlib.sha256(
                                label_map.read_bytes()
                            ).hexdigest(),
                            "instances": records,
                        }
                    }
                ),
                encoding="utf-8",
            )
            scene = build_joint_scene_analysis(
                np.full_like(semantic, 2),
                semantic,
                schema=MaskProfileSchema.from_reference_profile("GLaS"),
                pixel_size_um=None,
                nuclei_instances_path=str(manifest),
            )
            references, _rejected = build_reference_shape_library(
                scene,
                class_id=2,
            )

        self.assertEqual(len(references), 1)
        self.assertTrue(
            all("merged_suspect" not in item.quality_flags for item in scene.cells.instances[:4])
        )
        self.assertAlmostEqual(
            scene.population.nominal_nucleus_diameter_px,
            2.0 * np.sqrt(25.0 / np.pi),
        )

    def test_layout_tries_another_eligible_shape_at_same_center(self):
        legal = np.zeros((24, 24), dtype=bool)
        legal[12, 12] = True
        valid = np.zeros_like(legal)
        valid[11:14, 11:14] = True
        references = (
            ReferenceNucleusShape(
                "small", 1, np.ones((1, 1), dtype=bool), "test", 1
            ),
            ReferenceNucleusShape(
                "large", 1, np.ones((7, 7), dtype=bool), "test", 49
            ),
        )

        _target, placed, trace = _place_layout(
            base=np.zeros_like(legal, dtype=np.uint8),
            references=references,
            class_id=1,
            legal_zone=legal,
            valid_footprint_region=valid,
            halo=legal,
            score=legal.astype(float),
            requested_count=1,
            layout_program="single",
            cluster_size_range=(1, 1),
            nominal_nucleus_diameter_px=3.0,
            orientation_mask=np.zeros_like(legal),
            continuity_region=np.zeros_like(legal),
            continuity_anchor_mask=np.zeros_like(legal),
            continuity_maximum_empty_run_px=0,
            continuity_minimum_anchor_coverage_fraction=0.0,
            continuity_preferred_count=0,
            minimum_effect_span_px=0,
            minimum_effect_foci=0,
            enforce_single_scatter_separation=False,
            seed=1,
        )

        self.assertEqual(placed, 1)
        self.assertEqual(trace[0]["reference_instance_id"], "small")

    def test_reference_sampling_is_source_first_unique_and_without_replacement(self):
        duplicate = np.asarray([[1, 1], [1, 0]], dtype=bool)
        distinct = np.asarray([[1, 0], [1, 1]], dtype=bool)
        references = (
            ReferenceNucleusShape("library-a", 1, duplicate, "calibrated_dataset_instance_library", 3),
            ReferenceNucleusShape("source-a", 1, distinct, "semantic_distance_watershed", 3),
            ReferenceNucleusShape("library-duplicate", 1, duplicate, "calibrated_dataset_instance_library", 3),
        )

        unique = _unique_reference_shapes(references)
        ordered = _reference_sampling_order(
            unique,
            rng=np.random.default_rng(7),
        )

        self.assertEqual(len(unique), 2)
        self.assertEqual(ordered[0].instance_id, "source-a")

    def test_missing_calibrated_library_uses_bounded_unique_source_variants(self):
        masks = (
            np.asarray(
                [[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 0], [0, 1, 0, 0]],
                dtype=bool,
            ),
            np.asarray(
                [[0, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]],
                dtype=bool,
            ),
            np.asarray(
                [[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 0]],
                dtype=bool,
            ),
            np.asarray(
                [[0, 1, 1, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]],
                dtype=bool,
            ),
        )
        references = tuple(
            ReferenceNucleusShape(
                instance_id=f"source-{index}",
                class_id=1,
                mask=mask,
                source="semantic_distance_watershed",
                area_px=int(mask.sum()),
            )
            for index, mask in enumerate(masks)
        )

        expanded = _calibrated_reference_variants(
            references,
            minimum_count=8,
        )
        ordered = _reference_sampling_order(
            expanded,
            rng=np.random.default_rng(9),
        )

        self.assertEqual(len(expanded), 8)
        self.assertEqual(
            len({_reference_shape_digest(item) for item in expanded}),
            8,
        )
        self.assertTrue(
            all(
                item.source == "same_patch_complete_instance_resized"
                and item.parent_instance_id in {ref.instance_id for ref in references}
                and 0.82 <= item.scale_factor <= 1.18
                for item in expanded[4:]
            )
        )
        self.assertTrue(
            all(
                item.source == "semantic_distance_watershed"
                for item in ordered[:4]
            )
        )

    def test_layout_exhausts_unique_shapes_before_capacity_reuse(self):
        legal = np.zeros((48, 48), dtype=bool)
        legal[5:-5, 5:-5] = True
        references = tuple(
            ReferenceNucleusShape(
                instance_id=f"shape-{index}",
                class_id=1,
                mask=mask,
                source="semantic_distance_watershed",
                area_px=int(mask.sum()),
            )
            for index, mask in enumerate(
                (
                    np.ones((1, 1), dtype=bool),
                    np.asarray([[1, 1], [1, 0]], dtype=bool),
                    np.asarray([[0, 1], [1, 1]], dtype=bool),
                )
            )
        )

        _target, placed, trace = _place_layout(
            base=np.zeros_like(legal, dtype=np.uint8),
            references=references,
            class_id=1,
            legal_zone=legal,
            valid_footprint_region=legal,
            halo=legal,
            score=legal.astype(float),
            requested_count=3,
            layout_program="single",
            cluster_size_range=(1, 1),
            nominal_nucleus_diameter_px=2.0,
            orientation_mask=np.zeros_like(legal),
            continuity_region=np.zeros_like(legal),
            continuity_anchor_mask=np.zeros_like(legal),
            continuity_maximum_empty_run_px=0,
            continuity_minimum_anchor_coverage_fraction=0.0,
            continuity_preferred_count=0,
            minimum_effect_span_px=0,
            minimum_effect_foci=0,
            enforce_single_scatter_separation=False,
            seed=3,
        )

        self.assertEqual(placed, 3)
        self.assertEqual(
            len({item["reference_shape_sha256"] for item in trace}),
            3,
        )
        self.assertFalse(any(item["reference_reused"] for item in trace))

    def test_multisite_population_prefix_balances_four_local_hotspots(self):
        site_centers = np.asarray(
            ((20, 20), (20, 340), (340, 20), (340, 340)), dtype=int
        )
        points = []
        for center_y, center_x in site_centers:
            for offset_y in range(-42, 43, 14):
                for offset_x in range(-42, 43, 14):
                    points.append((center_y + offset_y, center_x + offset_x))
        # Put one site's candidates first to mimic a strongly dominant ProbNet
        # mode. The multisite prefix must still balance all four local sites.
        points = np.asarray(points, dtype=int)

        ordered, site_by_center, prefix_count = (
            _multisite_population_anchor_order(
                points,
                requested_count=24,
                required_site_count=4,
                minimum_effect_span_px=256,
                nominal_nucleus_diameter_px=12.0,
            )
        )

        first = ordered[:24]
        site_ids = [
            site_by_center[(int(row), int(col))] for row, col in first
        ]
        counts = Counter(site_ids)
        site_means = np.asarray(
            [
                np.mean(
                    first[np.asarray(site_ids) == site_id], axis=0
                )
                for site_id in sorted(counts)
            ]
        )
        distances = np.linalg.norm(
            site_means[:, None, :] - site_means[None, :, :], axis=2
        )

        self.assertGreaterEqual(prefix_count, 24)
        self.assertEqual(sorted(counts.values()), [6, 6, 6, 6])
        self.assertGreaterEqual(float(np.max(distances)), 256.0)

    def test_scatter_uses_probnet_weighted_hard_core_not_regular_grid(self):
        rows, cols = np.mgrid[0:80, 0:80]
        anchors = np.column_stack((rows.ravel(), cols.ravel()))
        score = np.exp(-((rows - 30) ** 2 + (cols - 24) ** 2) / 800.0)

        ordered = _probnet_hard_core_anchor_order(
            anchors,
            values=score.ravel(),
            minimum_center_separation_px=12.0,
            minimum_effect_span_px=36,
            requested_count=6,
            rng=np.random.default_rng(11),
        )

        selected = ordered[:6].astype(float)
        distances = np.linalg.norm(
            selected[:, None, :] - selected[None, :, :], axis=2
        )
        positive = distances[distances > 0]
        self.assertGreater(float(positive.min()), 12.0)
        self.assertGreaterEqual(float(positive.max()), 36.0)
        nearest = np.min(np.where(distances > 0, distances, np.inf), axis=1)
        self.assertGreater(float(np.ptp(nearest)), 1.0)

    def test_local_abundance_does_not_inherit_peritumoral_group_spacing(self):
        canvas = np.zeros((32, 32), dtype=np.uint8)
        legal = np.zeros_like(canvas, dtype=bool)
        legal[3:-3, 3:-3] = True
        reference = ReferenceNucleusShape(
            "ref", 1, np.ones((1, 1), dtype=bool), "test", 1
        )
        common = {
            "base": canvas,
            "references": (reference,),
            "class_id": 1,
            "legal_zone": legal,
            "valid_footprint_region": legal,
            "halo": legal,
            "score": legal.astype(float),
            "requested_count": 12,
            "layout_program": "small_cluster",
            "cluster_size_range": (1, 1),
            "nominal_nucleus_diameter_px": 8.0,
            "orientation_mask": np.zeros_like(legal),
            "continuity_region": np.zeros_like(legal),
            "continuity_anchor_mask": np.zeros_like(legal),
            "continuity_maximum_empty_run_px": 0,
            "continuity_minimum_anchor_coverage_fraction": 0.0,
            "continuity_preferred_count": 0,
            "minimum_effect_span_px": 0,
            "minimum_effect_foci": 0,
            "seed": 1,
        }

        _target, local_placed, _trace = _place_layout(
            **common,
            enforce_small_cluster_group_separation=False,
        )
        _target, peritumoral_placed, _trace = _place_layout(
            **common,
            enforce_small_cluster_group_separation=True,
        )

        self.assertEqual(local_placed, 12)
        self.assertLess(peritumoral_placed, local_placed)

    def test_local_abundance_balances_cells_across_multiple_broad_sites(self):
        shape = (120, 120)
        legal = np.zeros(shape, dtype=bool)
        legal[5:-5, 5:-5] = True
        rows, cols = np.indices(shape)
        score = -((rows - 60) ** 2 + (cols - 60) ** 2).astype(float)

        _target, placed, trace = _place_layout(
            base=np.zeros(shape, dtype=np.uint8),
            references=(
                ReferenceNucleusShape(
                    "ref", 1, np.ones((3, 3), dtype=bool), "test", 9
                ),
            ),
            class_id=1,
            legal_zone=legal,
            valid_footprint_region=legal,
            halo=legal,
            score=score,
            requested_count=18,
            layout_program="single",
            cluster_size_range=(1, 1),
            nominal_nucleus_diameter_px=8.0,
            orientation_mask=np.zeros(shape, dtype=bool),
            continuity_region=np.zeros(shape, dtype=bool),
            continuity_anchor_mask=np.zeros(shape, dtype=bool),
            continuity_maximum_empty_run_px=0,
            continuity_minimum_anchor_coverage_fraction=0.0,
            continuity_preferred_count=0,
            minimum_effect_span_px=50,
            minimum_effect_foci=4,
            seed=7,
            enforce_small_cluster_group_separation=False,
            enforce_multisite_population=True,
        )

        self.assertEqual(placed, 18)
        site_ids = [item["population_site_id"] for item in trace]
        self.assertNotIn(None, site_ids)
        groups = Counter(site_ids)
        self.assertEqual(len(groups), 4)
        self.assertEqual(
            sorted(groups.values()),
            [4, 4, 5, 5],
        )
        self.assertEqual(
            {item["anchor_sampling_policy"] for item in trace},
            {"probnet_ranked_balanced_multisite_hotspots"},
        )
        centers = np.asarray(
            [item["center_xy"] for item in trace], dtype=float
        )
        distances = np.linalg.norm(
            centers[:, None, :] - centers[None, :, :], axis=2
        )
        self.assertGreaterEqual(float(np.max(distances)), 50.0)

    def test_every_mechanism_has_a_unique_registered_postcondition_gate(self):
        repository = JointSkillRepository()
        registry = JointGateRegistry()
        expected = {
            mechanism_postcondition_checker_id(mechanism_id)
            for mechanism_id in repository.mechanisms
        }
        self.assertEqual(set(MECHANISM_POSTCONDITION_IDS), set(repository.mechanisms))
        self.assertTrue(expected.issubset(set(registry.available_checker_ids)))

    def test_simple_doctor_instructions_parse_without_mechanism_invention(self):
        parser = RuleBasedSemanticParser()
        examples = {
            "increase tumor burden": "tumor-burden-increase-v1",
            "reduce tumor burden": "invasive-tumor-footprint-decrease-v1",
            "increase necrosis": "necrosis-appearance-v1",
            "increase intratumoral necrosis": "necrosis-appearance-v1",
            "reduce intratumoral necrosis": "necrosis-resolution-v1",
            "increase tumor-associated stroma": "stroma-increase-v1",
            "减少坏死": "necrosis-resolution-v1",
            "increase tumor budding": "peritumoral-small-cluster-increase-v1",
            "increase immune cells": "cell-type-abundance-increase-v1",
            "increase immune infiltrate": (
                "generic-immune-infiltrate-increase-v1"
            ),
            "decrease immune infiltrate": (
                "generic-immune-infiltrate-decrease-v1"
            ),
            "降低细胞密度": "cellularity-decrease-v1",
        }
        for instruction, expected in examples.items():
            with self.subTest(instruction=instruction):
                intent = parser.parse(instruction)
                self.assertEqual(intent.primitive_id, expected)
        self.assertNotIn("mechanism", intent.to_metadata())

    def test_breast_generic_immune_turnover_is_direction_specific(self):
        repository = JointSkillRepository()
        mechanism = repository.mechanisms[
            "breast-generic-immune-compartment-turnover"
        ]
        self.assertEqual(
            mechanism.tissue_program.primitive_label_contracts[
                "generic-immune-infiltrate-increase-v1"
            ],
            {
                "source_labels": ("Stroma",),
                "target_labels": ("Immune infiltrate",),
            },
        )
        self.assertEqual(
            mechanism.tissue_program.primitive_label_contracts[
                "generic-immune-infiltrate-decrease-v1"
            ],
            {
                "source_labels": ("Immune infiltrate",),
                "target_labels": ("Stroma",),
            },
        )
        self.assertEqual(
            repository.primitives[
                "generic-immune-infiltrate-increase-v1"
            ].target_cell_classes,
            (2,),
        )
        self.assertEqual(
            repository.primitives[
                "generic-immune-infiltrate-decrease-v1"
            ].target_cell_classes,
            (3,),
        )
        profile = repository.annotation_profiles["bcss-semantic-v1"]
        self.assertEqual(
            profile.mechanism_editable_source_fine_ids[
                "breast-generic-immune-compartment-turnover::generic-immune-infiltrate-increase-v1"
            ],
            (2,),
        )
        self.assertEqual(
            profile.mechanism_editable_target_fine_ids[
                "breast-generic-immune-compartment-turnover::generic-immune-infiltrate-decrease-v1"
            ],
            (2,),
        )

    def test_semantic_intent_is_compiler_owned_after_binding(self):
        raw = {
            **_case_stub().to_metadata(),
            "instruction": "increase tumor burden",
            "primitive_id": "tumor-burden-increase-v1",
        }
        case, intent = bind_semantic_intent(raw, RuleBasedSemanticParser())

        self.assertEqual(intent.subject, "tumor-burden")
        self.assertEqual(case.compiled_normalized_intent(), "increase; tumor-burden")
        self.assertEqual(case.semantic_intent["parser"], intent.parser)

    def test_semantic_cell_class_is_resolved_by_observation_profile(self):
        raw = {
            **_case_stub().to_metadata(),
            "instruction": "increase immune cells",
            "primitive_id": "cell-type-abundance-increase-v1",
            "joint_area_budget": None,
        }
        case, intent = bind_semantic_intent(raw, RuleBasedSemanticParser())

        self.assertEqual(intent.explicit_cell_class, "immune")
        self.assertEqual(case.provenance["target_cell_class_ids"], [2])
        self.assertEqual(case.semantic_intent["resolved_cell_class_ids"], [2])
        self.assertEqual(
            case.provenance["target_cell_class_resolution"]["authority"],
            "versioned_observation_profile",
        )

    def test_generic_tumor_increase_exposes_burden_and_budding_hypotheses(self):
        intent = RuleBasedSemanticParser().parse("increase tumor")

        self.assertEqual(intent.subject, "tumor")
        self.assertEqual(
            [item.primitive_id for item in intent.primitive_hypotheses],
            [
                "tumor-burden-increase-v1",
                "cohesive-boundary-expansion-v1",
                "peritumoral-neoplastic-scatter-increase-v1",
            ],
        )
        self.assertEqual(
            [item.semantic_fit for item in intent.primitive_hypotheses],
            ["direct", "contextual", "contextual"],
        )

    def test_explicit_tumor_scope_does_not_expand_to_another_primitive(self):
        parser = RuleBasedSemanticParser()
        burden = parser.parse("increase tumor burden")
        budding = parser.parse("increase tumor budding")

        self.assertEqual(len(burden.primitive_hypotheses), 1)
        self.assertEqual(len(budding.primitive_hypotheses), 1)
        self.assertEqual(
            budding.primitive_hypotheses[0].primitive_id,
            "peritumoral-small-cluster-increase-v1",
        )

    def test_manifest_may_hint_a_contextual_generic_tumor_interpretation(self):
        raw = {
            **_case_stub().to_metadata(),
            "instruction": "increase tumor",
            "primitive_id": "peritumoral-neoplastic-scatter-increase-v1",
        }
        case, _intent = bind_semantic_intent(
            raw, RuleBasedSemanticParser()
        )

        self.assertEqual(
            case.primitive_id,
            "peritumoral-neoplastic-scatter-increase-v1",
        )
        self.assertIn(
            case.primitive_id,
            {
                item["primitive_id"]
                for item in case.semantic_intent["primitive_hypotheses"]
            },
        )

    def test_semantic_parser_rejects_manifest_primitive_conflict(self):
        raw = {
            **_case_stub().to_metadata(),
            "instruction": "reduce tumor burden",
            "primitive_id": "tumor-burden-increase-v1",
        }
        with self.assertRaisesRegex(JointContractError, "contradicts"):
            bind_semantic_intent(raw, RuleBasedSemanticParser())

    def test_preflight_density_matches_shared_scene_instance_authority(self):
        tissue = np.full((32, 32), 11, dtype=np.uint8)
        nuclei = np.zeros_like(tissue)
        nuclei[4:7, 4:7] = 1
        nuclei[6:9, 6:9] = 1  # one 8-connected semantic component
        nuclei[20:23, 20:23] = 2
        scene = build_joint_scene_analysis(
            tissue,
            nuclei,
            schema=MaskProfileSchema.from_reference_profile("GLaS"),
            pixel_size_um=None,
        )

        density, by_class = _target_interface_population_density(
            scene,
            source_tissue=tissue,
            target_classes=(1,),
            target_label="Tumor",
            schema=MaskProfileSchema.from_reference_profile("GLaS"),
            reference_area_p95=9.0,
        )

        expected_class_1 = sum(
            item.class_id == 1 for item in scene.cells.instances
        )
        expected_total = len(scene.cells.instances)
        authority = build_scene_instance_authority(scene, nuclei)
        self.assertEqual(len(authority["instances"]), expected_total)
        self.assertTrue(authority["authority_sha256"])
        self.assertAlmostEqual(density, expected_total / tissue.size)
        self.assertAlmostEqual(by_class[1], expected_class_1 / tissue.size)

    def test_instance_authority_digest_is_loader_dtype_invariant(self):
        raw = np.asarray([[0, 101], [102, 0]], dtype=np.uint8)

        self.assertEqual(array_sha256(raw), array_sha256(raw.astype(np.int64)))

    def test_packing_total_is_at_least_required_seam_quota(self):
        shape = (24, 24)
        region = np.zeros(shape, dtype=bool)
        region[3:21, 3:21] = True
        reference = ReferenceNucleusShape(
            instance_id="complete-ref-1",
            class_id=1,
            mask=np.ones((1, 1), dtype=bool),
            source="semantic_complete_instance",
            area_px=1,
        )

        certificate = certify_complete_footprint_packing(
            source_nuclei=np.zeros(shape, dtype=np.uint8),
            erased_footprint=np.zeros(shape, dtype=bool),
            center_region=region,
            valid_footprint_region=region,
            references_by_class={1: (reference,)},
            requested_count=1,
            continuity_region=region,
            required_seam_count=3,
            required_seam_class=1,
        )

        self.assertTrue(certificate.passed)
        self.assertEqual(certificate.requested_count, 3)
        self.assertEqual(certificate.placed_count, 3)

    def test_packing_distributes_seam_witness_over_uncovered_anchor(self):
        shape = (24, 64)
        centers = np.zeros(shape, dtype=bool)
        centers[12, [8, 16, 40, 48]] = True
        anchor = np.zeros(shape, dtype=bool)
        anchor[12, 4:53] = True
        reference = ReferenceNucleusShape(
            instance_id="complete-ref-1",
            class_id=1,
            mask=np.ones((1, 1), dtype=bool),
            source="semantic_complete_instance",
            area_px=1,
        )

        certificate = certify_complete_footprint_packing(
            source_nuclei=np.zeros(shape, dtype=np.uint8),
            erased_footprint=np.zeros(shape, dtype=bool),
            center_region=centers,
            valid_footprint_region=np.ones(shape, dtype=bool),
            references_by_class={1: (reference,)},
            requested_count=2,
            continuity_region=centers,
            continuity_anchor_mask=anchor,
            preexisting_continuity_centers=np.zeros(shape, dtype=bool),
            continuity_maximum_empty_run_px=8,
            required_seam_count=2,
            required_seam_class=1,
        )

        witness = np.zeros(shape, dtype=bool)
        for placement in certificate.placements:
            witness[placement.row, placement.col] = True
        coverage = anchor_coverage_fraction(
            anchor,
            witness,
            maximum_empty_run_px=8,
        )
        self.assertTrue(certificate.passed, certificate.failure_reasons)
        self.assertGreaterEqual(coverage, 0.60)

    def test_scatter_packing_certificate_enforces_focus_separation(self):
        shape = (20, 20)
        centers = np.zeros(shape, dtype=bool)
        centers[8, 4] = True
        centers[8, 8] = True
        centers[8, 12] = True
        reference = ReferenceNucleusShape(
            instance_id="unit-shape",
            class_id=1,
            mask=np.ones((1, 1), dtype=bool),
            source="test",
            area_px=1,
        )

        certificate = certify_complete_footprint_packing(
            source_nuclei=np.zeros(shape, dtype=np.uint8),
            erased_footprint=np.zeros(shape, dtype=bool),
            center_region=centers,
            valid_footprint_region=np.ones(shape, dtype=bool),
            references_by_class={1: (reference,)},
            requested_count=3,
            minimum_acceptable_count=3,
            minimum_center_separation_px=5.0,
        )

        self.assertFalse(certificate.passed)
        self.assertEqual(certificate.placed_count, 2)
        self.assertEqual(certificate.minimum_center_separation_px, 5.0)

    def test_small_cluster_preflight_and_executor_share_focus_separation(self):
        diameter = 8.0
        separation = independent_focus_minimum_center_separation_px(
            "peritumoral-small-cluster-increase-v1",
            diameter,
        )
        self.assertEqual(separation, 12.8)
        self.assertEqual(
            independent_focus_minimum_center_separation_px(
                "peritumoral-neoplastic-scatter-increase-v1",
                diameter,
            ),
            18.0,
        )
        # cellularity-increase-v1 has a strict spatial_focus_contract in
        # its _joint_area gate, so preflight certifies 3-diameter inter-focus
        # separation to match audit_cell_effect_foci.
        self.assertEqual(
            independent_focus_minimum_center_separation_px(
                "cellularity-increase-v1",
                diameter,
            ),
            24.0,
        )
        # Other increase primitives (cell-type-abundance, neoplastic-cell-abundance)
        # use raw_spatial_component_count in their gate, not the strict focus
        # contract, so they do not need a preflight focus separation witness.
        self.assertEqual(
            independent_focus_minimum_center_separation_px(
                "cell-type-abundance-increase-v1",
                diameter,
            ),
            0.0,
        )
        self.assertEqual(
            independent_focus_minimum_center_separation_px(
                "neoplastic-cell-abundance-increase-v1",
                diameter,
            ),
            0.0,
        )
        # Decrease primitives remain 0.0 — they remove whole instances and
        # do not need a focus packing witness.
        self.assertEqual(
            independent_focus_minimum_center_separation_px(
                "neoplastic-cell-abundance-decrease-v1",
                diameter,
            ),
            0.0,
        )
        self.assertEqual(
            certificate_aligned_cluster_size_range(
                primitive_id="peritumoral-small-cluster-increase-v1",
                mechanism_id="breast-peritumoral-small-cluster",
                configured_range=(1, 4),
                packing_certificate={
                    "passed": True,
                    "requested_count": 12,
                    "minimum_center_separation_px": separation,
                },
                nominal_nucleus_diameter_px=diameter,
            ),
            (3, 4),
        )
        self.assertEqual(
            certificate_aligned_cluster_size_range(
                primitive_id="peritumoral-small-cluster-increase-v1",
                configured_range=(1, 4),
                packing_certificate={
                    "passed": False,
                    "requested_count": 8,
                    "minimum_center_separation_px": separation,
                },
                nominal_nucleus_diameter_px=diameter,
            ),
            (1, 4),
        )

    def test_small_cluster_forms_one_localized_three_focus_hotspot(self):
        shape = (100, 100)
        legal = np.zeros(shape, dtype=bool)
        legal[5:-5, 5:-5] = True
        rows, cols = np.indices(shape)
        score = -((rows - 50) ** 2 + (cols - 50) ** 2).astype(float)
        _target, placed, trace = _place_layout(
            base=np.zeros(shape, dtype=np.uint8),
            references=(
                ReferenceNucleusShape(
                    "ref",
                    1,
                    np.ones((3, 3), dtype=bool),
                    "test",
                    9,
                ),
            ),
            class_id=1,
            legal_zone=legal,
            valid_footprint_region=legal,
            halo=legal,
            score=score,
            requested_count=12,
            layout_program="small_cluster",
            cluster_size_range=(3, 4),
            nominal_nucleus_diameter_px=8.0,
            orientation_mask=np.zeros(shape, dtype=bool),
            continuity_region=np.zeros(shape, dtype=bool),
            continuity_anchor_mask=np.zeros(shape, dtype=bool),
            continuity_maximum_empty_run_px=0,
            continuity_minimum_anchor_coverage_fraction=0.0,
            continuity_preferred_count=0,
            minimum_effect_span_px=20,
            minimum_effect_foci=3,
            seed=1,
            strict_breast_small_cluster=True,
        )

        self.assertEqual(placed, 12)
        group_ids = tuple(dict.fromkeys(item["cluster_id"] for item in trace))
        self.assertEqual(
            sorted(
                sum(item["cluster_id"] == group_id for item in trace)
                for group_id in group_ids
            ),
            [4, 4, 4],
        )
        centers = np.asarray(
            [item["center_xy"][::-1] for item in trace], dtype=float
        )
        distances = np.linalg.norm(
            centers[:, None, :] - centers[None, :, :], axis=2
        )
        self.assertGreaterEqual(float(np.max(distances)), 20.0)
        self.assertLessEqual(
            float(np.max(distances)),
            small_cluster_maximum_hotspot_span_px(
                8.0,
                20,
                compact_breast=True,
            ),
        )
        self.assertEqual(
            {item["anchor_sampling_policy"] for item in trace},
            {"probnet_ranked_localized_front_segment"},
        )

    def test_small_cluster_planner_scores_localized_witness_capacity(self):
        capacity, span_margin = _localized_focus_capacity_metrics(
            center_rows=(20, 20, 20, 20),
            center_cols=(10, 30, 60, 110),
            nominal_nucleus_diameter_px=8.0,
            minimum_effect_span_px=20,
            required_focus_count=3,
            minimum_anchor_separation_diameters=(
                BREAST_SMALL_CLUSTER_MINIMUM_ANCHOR_SEPARATION_DIAMETERS
            ),
        )

        self.assertEqual(capacity, 2)
        self.assertEqual(span_margin, 0.0)

        breast_capacity, breast_span_margin = (
            _localized_focus_capacity_metrics(
                center_rows=(20, 20, 20),
                center_cols=(10, 35, 60),
                nominal_nucleus_diameter_px=8.0,
                minimum_effect_span_px=20,
                required_focus_count=3,
                minimum_anchor_separation_diameters=(
                    BREAST_SMALL_CLUSTER_MINIMUM_ANCHOR_SEPARATION_DIAMETERS
                ),
                strict_breast_small_cluster=True,
            )
        )

        self.assertEqual(breast_capacity, 3)
        self.assertEqual(breast_span_margin, 2.0)

    def test_capacity_optimized_certificate_preserves_diverse_execution_family(self):
        references = tuple(
            ReferenceNucleusShape(
                instance_id=name,
                class_id=1,
                mask=np.ones(shape, dtype=bool),
                source="test",
                area_px=int(np.prod(shape)),
            )
            for name, shape in (("small", (2, 2)), ("large", (5, 5)))
        )

        aligned = certificate_aligned_references(
            references,
            {
                "passed": True,
                "capacity_optimized_shape_fallback_used": True,
                "placements": [
                    {"reference_instance_id": "small"},
                    {"reference_instance_id": "small"},
                ],
            },
        )

        self.assertEqual(
            tuple(item.instance_id for item in aligned),
            ("small", "large"),
        )
        self.assertEqual(
            certificate_aligned_references(
                references,
                {
                    "passed": True,
                    "capacity_optimized_shape_fallback_used": False,
                    "placements": [{"reference_instance_id": "small"}],
                },
            ),
            references,
        )

    def test_typed_seam_does_not_exclude_other_compatible_classes(self):
        shape = (32, 32)
        region = np.zeros(shape, dtype=bool)
        region[3:29, 3:29] = True
        references = {
            class_id: (
                ReferenceNucleusShape(
                    instance_id=f"complete-ref-{class_id}",
                    class_id=class_id,
                    mask=np.ones((1, 1), dtype=bool),
                    source="semantic_complete_instance",
                    area_px=1,
                ),
            )
            for class_id in (2, 3)
        }

        certificate = certify_complete_footprint_packing(
            source_nuclei=np.zeros(shape, dtype=np.uint8),
            erased_footprint=np.zeros(shape, dtype=bool),
            center_region=region,
            valid_footprint_region=region,
            references_by_class=references,
            requested_count=10,
            class_request_weights={2: 0.8, 3: 0.2},
            # The complete placement domain is also the seam.  A spatial seam
            # exclusion would leave no remainder capacity, while the typed
            # contract correctly permits class 2 beside the two class-3 cells.
            continuity_region=region,
            required_seam_count=2,
            required_seam_class=3,
        )

        self.assertTrue(certificate.passed, certificate.failure_reasons)
        self.assertEqual(certificate.class_requested_counts, {2: 8, 3: 2})
        self.assertEqual(certificate.class_placed_counts, {2: 8, 3: 2})
        self.assertEqual(certificate.placed_seam_count, 2)

    def test_packing_can_fall_back_within_the_certified_seam_interval(self):
        shape = (12, 12)
        centers = np.zeros(shape, dtype=bool)
        centers[3, 3] = True
        centers[3, 7] = True
        centers[8, 8] = True
        seam = np.zeros(shape, dtype=bool)
        seam[3, 3] = True
        seam[3, 7] = True
        valid = np.ones(shape, dtype=bool)
        reference = ReferenceNucleusShape(
            instance_id="complete-ref-1",
            class_id=1,
            mask=np.ones((1, 1), dtype=bool),
            source="semantic_complete_instance",
            area_px=1,
        )

        certificate = certify_complete_footprint_packing(
            source_nuclei=np.zeros(shape, dtype=np.uint8),
            erased_footprint=np.zeros(shape, dtype=bool),
            center_region=centers,
            valid_footprint_region=valid,
            references_by_class={1: (reference,)},
            requested_count=3,
            continuity_region=seam,
            required_seam_count=3,
            minimum_seam_count=2,
            required_seam_class=1,
        )

        self.assertTrue(certificate.passed, certificate.failure_reasons)
        self.assertEqual(certificate.placed_count, 3)
        self.assertEqual(certificate.required_seam_count, 2)
        self.assertEqual(certificate.placed_seam_count, 2)
        self.assertEqual(certificate.nominal_required_seam_count, 3)
        self.assertEqual(certificate.minimum_safe_seam_count, 2)
        self.assertTrue(certificate.seam_count_fallback_used)

    def test_packing_witness_excludes_locally_unsupported_shape_sizes(self):
        shape = (48, 48)
        region = np.zeros(shape, dtype=bool)
        region[2:46, 2:46] = True
        references = tuple(
            ReferenceNucleusShape(
                instance_id=f"shape-{area}",
                class_id=1,
                mask=np.ones((1, area), dtype=bool),
                source="semantic_complete_instance",
                area_px=area,
            )
            for area in (1, 10, 100)
        )

        certificate = certify_complete_footprint_packing(
            source_nuclei=np.zeros(shape, dtype=np.uint8),
            erased_footprint=np.zeros(shape, dtype=bool),
            center_region=region,
            valid_footprint_region=region,
            references_by_class={1: references},
            requested_count=3,
        )

        self.assertTrue(certificate.passed, certificate.failure_reasons)
        self.assertEqual(certificate.class_reference_median_area_px, {1: 10.0})
        self.assertEqual(
            {item.area_px for item in certificate.placements},
            {10},
        )

    def test_packing_uses_eight_connected_instance_separation(self):
        shape = (12, 12)
        centers = np.zeros(shape, dtype=bool)
        centers[5, 5] = True
        centers[6, 6] = True  # diagonal contact under 8-connectivity
        reference = ReferenceNucleusShape(
            instance_id="unit-shape",
            class_id=1,
            mask=np.ones((1, 1), dtype=bool),
            source="semantic_complete_instance",
            area_px=1,
        )

        certificate = certify_complete_footprint_packing(
            source_nuclei=np.zeros(shape, dtype=np.uint8),
            erased_footprint=np.zeros(shape, dtype=bool),
            center_region=centers,
            valid_footprint_region=np.ones(shape, dtype=bool),
            references_by_class={1: (reference,)},
            requested_count=2,
            allow_finite_count_fallback=False,
        )

        self.assertFalse(certificate.passed)
        self.assertEqual(certificate.placed_count, 1)

    def test_packing_uses_bounded_max_safe_finite_count(self):
        shape = (20, 20)
        centers = np.zeros(shape, dtype=bool)
        for row in (3, 8, 13):
            for col in (3, 8, 13):
                centers[row, col] = True
        reference = ReferenceNucleusShape(
            instance_id="unit-shape",
            class_id=1,
            mask=np.ones((1, 1), dtype=bool),
            source="semantic_complete_instance",
            area_px=1,
        )

        certificate = certify_complete_footprint_packing(
            source_nuclei=np.zeros(shape, dtype=np.uint8),
            erased_footprint=np.zeros(shape, dtype=bool),
            center_region=centers,
            valid_footprint_region=np.ones(shape, dtype=bool),
            references_by_class={1: (reference,)},
            requested_count=10,
        )

        self.assertTrue(certificate.passed, certificate.failure_reasons)
        self.assertTrue(certificate.finite_count_fallback_used)
        self.assertEqual(certificate.nominal_requested_count, 10)
        self.assertEqual(certificate.minimum_safe_count, 8)
        self.assertEqual(certificate.requested_count, 9)
        self.assertEqual(certificate.placed_count, 9)

        strict_recheck = certify_complete_footprint_packing(
            source_nuclei=np.zeros(shape, dtype=np.uint8),
            erased_footprint=np.zeros(shape, dtype=bool),
            center_region=centers,
            valid_footprint_region=np.ones(shape, dtype=bool),
            references_by_class={1: (reference,)},
            requested_count=10,
            allow_finite_count_fallback=False,
        )
        self.assertFalse(strict_recheck.passed)
        self.assertFalse(strict_recheck.finite_count_fallback_used)
        self.assertEqual(strict_recheck.requested_count, 10)
        self.assertEqual(strict_recheck.placed_count, 9)

    def test_packing_fallback_honors_manifest_minimum_count(self):
        shape = (24, 24)
        centers = np.zeros(shape, dtype=bool)
        for index in range(12):
            row = 3 + 5 * (index // 4)
            col = 3 + 5 * (index % 4)
            centers[row, col] = True
        reference = ReferenceNucleusShape(
            instance_id="unit-shape",
            class_id=1,
            mask=np.ones((1, 1), dtype=bool),
            source="test",
            area_px=1,
        )

        certificate = certify_complete_footprint_packing(
            source_nuclei=np.zeros(shape, dtype=np.uint8),
            erased_footprint=np.zeros(shape, dtype=bool),
            center_region=centers,
            valid_footprint_region=np.ones(shape, dtype=bool),
            references_by_class={1: (reference,)},
            requested_count=18,
            minimum_acceptable_count=12,
        )

        self.assertTrue(certificate.passed, certificate.failure_reasons)
        self.assertTrue(certificate.finite_count_fallback_used)
        self.assertEqual(certificate.nominal_requested_count, 18)
        self.assertEqual(certificate.minimum_safe_count, 12)
        self.assertEqual(certificate.requested_count, 12)
        self.assertEqual(certificate.placed_count, 12)

    def test_finite_count_bound_uses_sampling_scale_not_fixed_ten_percent(self):
        shape = (36, 36)
        centers = np.zeros(shape, dtype=bool)
        for index in range(17):
            row = 3 + 5 * (index // 6)
            col = 3 + 5 * (index % 6)
            centers[row, col] = True
        reference = ReferenceNucleusShape(
            instance_id="unit-shape",
            class_id=1,
            mask=np.ones((1, 1), dtype=bool),
            source="semantic_complete_instance",
            area_px=1,
        )

        certificate = certify_complete_footprint_packing(
            source_nuclei=np.zeros(shape, dtype=np.uint8),
            erased_footprint=np.zeros(shape, dtype=bool),
            center_region=centers,
            valid_footprint_region=np.ones(shape, dtype=bool),
            references_by_class={1: (reference,)},
            requested_count=20,
        )

        self.assertTrue(certificate.passed, certificate.failure_reasons)
        self.assertTrue(certificate.finite_count_fallback_used)
        self.assertEqual(certificate.minimum_safe_count, 16)
        self.assertEqual(certificate.requested_count, 17)
        self.assertEqual(certificate.placed_count, 17)

    def test_packing_can_use_small_complete_local_shapes_for_capacity(self):
        shape = (20, 40)
        region = np.zeros(shape, dtype=bool)
        region[3:6, 3:12] = True
        references = (
            ReferenceNucleusShape(
                instance_id="small-complete",
                class_id=1,
                mask=np.ones((2, 2), dtype=bool),
                source="semantic_complete_instance",
                area_px=4,
            ),
            ReferenceNucleusShape(
                instance_id="large-complete",
                class_id=1,
                mask=np.ones((3, 3), dtype=bool),
                source="semantic_complete_instance",
                area_px=9,
            ),
        )

        certificate = certify_complete_footprint_packing(
            source_nuclei=np.zeros(shape, dtype=np.uint8),
            erased_footprint=np.zeros(shape, dtype=bool),
            center_region=region,
            valid_footprint_region=region,
            references_by_class={1: references},
            requested_count=3,
            allow_finite_count_fallback=False,
        )

        self.assertTrue(certificate.passed, certificate.failure_reasons)
        self.assertTrue(certificate.capacity_optimized_shape_fallback_used)
        self.assertEqual(certificate.class_reference_median_area_px, {1: 6.5})
        self.assertEqual(
            {item.reference_instance_id for item in certificate.placements},
            {"small-complete"},
        )

    def test_packing_preserves_bounded_directional_rotation_family(self):
        shape = (21, 21)
        center = np.zeros(shape, dtype=bool)
        center[10, 10] = True
        valid = np.zeros(shape, dtype=bool)
        valid[7:14, 10] = True
        references = []
        for parent_index in range(3):
            parent_id = f"parent-{parent_index}"
            references.append(
                ReferenceNucleusShape(
                    instance_id=parent_id,
                    class_id=1,
                    mask=np.ones((1, 5), dtype=bool),
                    source="semantic_complete_instance",
                    area_px=5,
                )
            )
            for degrees in (45, 90, 135):
                is_only_fitting_orientation = (
                    parent_index == 2 and degrees == 135
                )
                references.append(
                    ReferenceNucleusShape(
                        instance_id=(
                            "zz-fitting:rotate-135"
                            if is_only_fitting_orientation
                            else f"{parent_id}:rotate-{degrees}"
                        ),
                        class_id=1,
                        mask=np.ones(
                            (5, 1) if is_only_fitting_orientation else (1, 5),
                            dtype=bool,
                        ),
                        source=(
                            "semantic_complete_instance_rotated_135"
                            if is_only_fitting_orientation
                            else f"semantic_complete_instance_rotated_{degrees}"
                        ),
                        area_px=5,
                        parent_instance_id=parent_id,
                        scale_factor=1.0,
                    )
                )

        certificate = certify_complete_footprint_packing(
            source_nuclei=np.zeros(shape, dtype=np.uint8),
            erased_footprint=np.zeros(shape, dtype=bool),
            center_region=center,
            valid_footprint_region=valid,
            references_by_class={1: tuple(references)},
            requested_count=1,
            allow_finite_count_fallback=False,
        )

        self.assertTrue(certificate.passed, certificate.failure_reasons)
        self.assertEqual(
            certificate.placements[0].reference_instance_id,
            "zz-fitting:rotate-135",
        )

    def test_added_shape_measurement_excludes_retained_same_class_neighbour(self):
        source = np.zeros((20, 20), dtype=np.uint8)
        source[2:6, 2:6] = 1
        target = source.copy()
        target[6:9, 2:6] = 1

        areas = _added_instance_areas_by_class(source, target)

        self.assertEqual(areas, {1: [12.0]})

    def test_mature_shape_measurement_uses_realized_instance_ledger(self):
        trace = {
            "accepted_instance_area_ledger": [
                {"class_id": 1, "area_px": 900},
                {"class_id": 1, "area_px": 1050},
                {"class_id": 2, "area_px": 420},
            ]
        }

        self.assertEqual(
            _recorded_instance_areas_by_class(trace),
            {1: [900.0, 1050.0], 2: [420.0]},
        )

    def test_deterministic_shape_measurement_uses_exact_placement_ledger(self):
        trace = {
            "placements": [
                {"cell_class": 1, "area_px": 193},
                {"cell_class": 1, "area_px": 527},
            ]
        }

        self.assertEqual(
            _recorded_instance_areas_by_class(trace),
            {1: [193.0, 527.0]},
        )

    def test_shape_support_uses_nearest_complete_reference_not_mixture_median(self):
        references = [193.0, 231.0, 264.0, 377.0, 527.0]

        self.assertEqual(
            _nearest_reference_area_ratio(527.0, references),
            1.0,
        )
        self.assertLess(
            _nearest_reference_area_ratio(569.0, references),
            1.10,
        )
        self.assertGreater(
            _nearest_reference_area_ratio(1000.0, references),
            1.67,
        )

    def test_max_safe_area_fallback_excludes_larger_unsafe_candidate(self):
        candidates = (
            SimpleNamespace(
                candidate_id="safe-1",
                ledger=SimpleNamespace(joint_pixels=48_686),
            ),
            SimpleNamespace(
                candidate_id="safe-2",
                ledger=SimpleNamespace(joint_pixels=48_700),
            ),
            SimpleNamespace(
                candidate_id="unsafe-larger",
                ledger=SimpleNamespace(joint_pixels=49_000),
            ),
        )

        def report(candidate_id, failures):
            checks = tuple(
                JointGateCheck(
                    check_id=check_id,
                    passed=False,
                    severity="hard",
                    detail="test",
                )
                for check_id in failures
            )
            return JointGateReport(
                candidate_id=candidate_id,
                passed=False,
                checks=checks,
            )

        reports = (
            report("safe-1", {"joint_area"}),
            report("safe-2", {"joint_area"}),
            report("unsafe-larger", {"joint_area", "local_shape_distribution"}),
        )

        self.assertEqual(
            _maximum_safe_below_target_joint_pixels(
                candidates,
                reports,
                hard_min_pixels=36_701,
                desired_min_pixels=48_810,
            ),
            48_700,
        )

    def test_depletion_band_edges_reserve_outer_band_under_extent_cap(self):
        core_end, transition_end, outer_end = _depletion_band_edges(
            diameter_px=28.0,
            core_width_cell_diameters=1.25,
            transition_width_cell_diameters=6.0,
            outer_width_cell_diameters=2.0,
            maximum_extent_px=192,
            maximum_observed_distance_px=240.0,
        )
        self.assertLess(core_end, transition_end)
        self.assertLess(transition_end, outer_end)
        self.assertAlmostEqual(outer_end, 192.0)
        self.assertAlmostEqual(outer_end - transition_end, 28.0)

    def test_glas_depletion_can_use_remaining_extent_as_reference(self):
        core_end, transition_end, outer_end = _depletion_band_edges(
            diameter_px=20.0,
            core_width_cell_diameters=1.25,
            transition_width_cell_diameters=3.0,
            outer_width_cell_diameters=1.5,
            maximum_extent_px=192,
            maximum_observed_distance_px=160.0,
            maximize_outer_reference=True,
        )
        self.assertAlmostEqual(core_end, 25.0)
        self.assertAlmostEqual(transition_end, 85.0)
        self.assertAlmostEqual(outer_end, 160.0)

    def test_density_field_count_cap_preserves_radial_targets(self):
        quotas = _cap_density_field_quotas(
            quotas=[3, 6, 4, 3, 2, 0, 0],
            source_counts=[5, 14, 12, 12, 9, 2, 0],
            target_fractions=[0.55, 0.42, 0.348, 0.276, 0.204, 0.132, 0.06],
            maximum_count=15,
            minimum_core=1,
            minimum_transition=1,
        )
        self.assertEqual(quotas, [3, 5, 4, 2, 1, 0, 0])

    def test_radial_gradient_allows_only_one_instance_quantization_step(self):
        self.assertTrue(
            _discrete_radial_profile_is_monotonic(
                [(8, 16), (4, 15), (1, 9), (2, 9), (0, 3)]
            )
        )
        self.assertFalse(
            _discrete_radial_profile_is_monotonic(
                [(8, 16), (1, 15), (3, 9)]
            )
        )

    def test_min_safe_over_target_requires_tissue_floor_and_other_gates(self):
        candidates = (
            SimpleNamespace(
                candidate_id="safe-nearest",
                ledger=SimpleNamespace(joint_pixels=51_252, tissue_pixels=36_701),
            ),
            SimpleNamespace(
                candidate_id="safe-larger",
                ledger=SimpleNamespace(joint_pixels=51_364, tissue_pixels=36_701),
            ),
            SimpleNamespace(
                candidate_id="not-at-floor",
                ledger=SimpleNamespace(joint_pixels=50_900, tissue_pixels=37_000),
            ),
            SimpleNamespace(
                candidate_id="unsafe-smaller",
                ledger=SimpleNamespace(joint_pixels=51_000, tissue_pixels=36_701),
            ),
        )

        def report(candidate_id, failures):
            return JointGateReport(
                candidate_id=candidate_id,
                passed=False,
                checks=tuple(
                    JointGateCheck(
                        check_id=check_id,
                        passed=False,
                        severity="hard",
                        detail="test",
                    )
                    for check_id in failures
                ),
            )

        reports = (
            report("safe-nearest", {"joint_area"}),
            report("safe-larger", {"joint_area"}),
            report("not-at-floor", {"joint_area"}),
            report("unsafe-smaller", {"joint_area", "whole_instance_changes"}),
        )
        self.assertEqual(
            _minimum_safe_above_target_joint_pixels(
                candidates,
                reports,
                desired_max_pixels=50_804,
                hard_max_pixels=62_914,
                tissue_floor_pixels=36_701,
            ),
            51_252,
        )
        self.assertEqual(
            _minimum_safe_above_target_joint_pixels(
                candidates,
                reports,
                desired_max_pixels=50_804,
                hard_max_pixels=62_914,
                tissue_floor_pixels=36_701,
                require_tissue_floor=False,
            ),
            50_900,
        )

    def test_area_feedback_uses_safe_sibling_despite_bad_shape_variant(self):
        def report(candidate_id, failures):
            return JointGateReport(
                candidate_id=candidate_id,
                passed=False,
                checks=tuple(
                    JointGateCheck(
                        check_id=check_id,
                        passed=False,
                        severity="hard",
                        detail="test",
                    )
                    for check_id in failures
                ),
            )

        reports = (
            report("shape-bad", {"joint_area", "local_shape_distribution"}),
            report("safe-for-feedback", {"joint_area"}),
        )

        self.assertEqual(
            _joint_area_feedback_candidate_ids(reports),
            {"safe-for-feedback"},
        )

    def test_inventory_has_six_domains_and_four_independent_axes(self):
        repository = JointSkillRepository()
        self.assertEqual(len(repository.mechanisms), 51)
        self.assertEqual(len(repository.primitives), 29)
        self.assertEqual(len(repository.annotation_profiles), 6)
        self.assertEqual(len(repository.cell_observation_profiles), 1)
        self.assertEqual(len(repository.cell_population_profiles), 6)
        self.assertEqual(
            {item.pathology_domain_id for item in repository.mechanisms.values()},
            {
                "breast-invasive-carcinoma-v1",
                "colorectal-adenocarcinoma-v1",
                "prostate-adenocarcinoma-v1",
                "lung-carcinoma-v1",
                "melanoma-v1",
                "oral-squamous-cell-carcinoma-v1",
            },
        )

    def test_scene_component_records_only_observed_panda_fine_ids(self):
        mask = np.full((64, 64), 2, dtype=np.uint8)
        mask[16:48, 16:48] = 10
        scene = build_scene_analysis(
            mask,
            schema=MaskProfileSchema.from_reference_profile("PANDA"),
        )
        tumor_components = [
            item for item in scene.graph.components if item.label == "Tumor"
        ]
        self.assertEqual(len(tumor_components), 1)
        self.assertEqual(tumor_components[0].fine_ids, (10,))

    def test_growth_mechanisms_are_increase_only_and_do_not_own_replacement(self):
        repository = JointSkillRepository()
        growth_mechanism_ids = (
            mechanism_id
            for mechanism_id in repository.mechanisms
            if mechanism_id.endswith("-growth")
            or mechanism_id in {
                "breast-cohesive-nst-front",
                "colorectal-gland-forming-front",
                "melanoma-cohesive-nest-sheet",
                "oral-scc-cohesive-nest-cord",
            }
        )
        for mechanism_id in growth_mechanism_ids:
            mechanism = repository.mechanisms[mechanism_id]
            if mechanism.pathology_domain_id == "prostate-adenocarcinoma-v1":
                self.assertNotIn(
                    "tumor-burden-increase-v1", mechanism.supported_primitives
                )
                self.assertIn(
                    "cohesive-boundary-expansion-v1",
                    mechanism.supported_primitives,
                )
            else:
                self.assertIn(
                    "tumor-burden-increase-v1", mechanism.supported_primitives
                )
            self.assertNotIn(
                "tumor-burden-decrease-v1", mechanism.supported_primitives
            )
            self.assertNotIn("stroma-increase-v1", mechanism.supported_primitives)
            self.assertNotIn(
                "tumor-burden-decrease-v1",
                mechanism.tissue_program.primitive_label_contracts,
            )
            self.assertNotIn(
                "stroma-increase-v1",
                mechanism.tissue_program.primitive_label_contracts,
            )
        self.assertEqual(
            [
                item.mechanism_id
                for item in repository.mechanisms_for(
                    pathology_domain_id="lung-carcinoma-v1",
                    primitive_id="stroma-increase-v1",
                )
            ],
            ["lung-operational-tumor-retreat"],
        )
        self.assertEqual(
            [
                item.mechanism_id
                for item in repository.mechanisms_for(
                    pathology_domain_id="prostate-adenocarcinoma-v1",
                    primitive_id="stroma-increase-v1",
                )
            ],
            [],
        )

    def test_evidence_authorities_are_explicit_and_non_interchangeable(self):
        repository = JointSkillRepository()
        mechanism = repository.skill_evidence_status[
            "joint-mechanism:lung-stas-airspace-spread"
        ]
        self.assertEqual(
            mechanism.field_categories["recognition_contract"],
            "pathology_fact",
        )
        self.assertEqual(
            mechanism.field_categories["tissue_program"],
            "engineering_proxy",
        )
        self.assertEqual(
            mechanism.field_categories["render_contract"],
            "model_representability",
        )
        profile = repository.skill_evidence_status[
            "annotation-profile:ignite-semantic-v1"
        ]
        self.assertEqual(
            profile.field_categories["prohibited_fine_ids"],
            "dataset_fact",
        )
        self.assertFalse(profile.production_allowed)
        self.assertTrue(
            any("dataset_fact pending" in gap for gap in profile.gaps)
        )

    def test_breast_seam_contract_is_anchor_conditioned_and_skill_owned(self):
        repository = JointSkillRepository()
        cohesive = repository.mechanisms[
            "breast-annotation-anchored-boundary-growth"
        ]
        seam = cohesive.cell_program.seam
        self.assertEqual(seam.mode, "adaptive_population_continuity")
        self.assertEqual(seam.reference_area_quantiles, (0.25, 0.75))
        self.assertGreater(seam.maximum_empty_run_cell_diameters, 0)
        self.assertTrue(seam.requires_new_target_cells)

    def test_melanoma_growth_seam_is_increase_only(self):
        mechanism = JointSkillRepository().mechanisms[
            "melanoma-cohesive-nest-sheet"
        ]
        increase = mechanism.cell_program.seam_for(
            "tumor-burden-increase-v1"
        )
        self.assertEqual(increase.minimum_anchor_coverage_fraction, 0.5)
        self.assertNotIn(
            "tumor-burden-decrease-v1",
            mechanism.cell_program.layout_program_by_primitive,
        )

    def test_joint_primitive_execution_scope_is_explicit(self):
        repository = JointSkillRepository()
        self.assertEqual(
            set(repository.executable_primitive_ids),
            {
                "cell-type-abundance-decrease-v1",
                "cell-type-abundance-increase-v1",
                "cellularity-decrease-v1",
                "cellularity-increase-v1",
                "generic-immune-infiltrate-decrease-v1",
                "generic-immune-infiltrate-increase-v1",
                "generic-inflammatory-cell-abundance-decrease-v1",
                "generic-inflammatory-cell-abundance-increase-v1",
                "cohesive-boundary-expansion-v1",
                "invasive-cord-formation-v1",
                "infiltrative-nest-cord-extension-v1",
                "invasive-front-expansion-v1",
                "invasive-tumor-footprint-decrease-v1",
                "local-invasive-clearance-v1",
                "necrosis-appearance-v1",
                "necrosis-resolution-v1",
                "neoplastic-cell-abundance-decrease-v1",
                "neoplastic-cell-abundance-increase-v1",
                "neoplastic-microinfiltration-increase-v1",
                "peritumoral-neoplastic-scatter-increase-v1",
                "peritumoral-small-cluster-increase-v1",
                "peritumoral-tumor-nest-formation-v1",
                "residual-tumor-fragmentation-v1",
                "stroma-increase-v1",
                "tumor-burden-increase-v1",
            },
        )
        self.assertEqual(
            set(repository.execution_scope["closed_primitives"]),
            {
                "architecture-progression-v1",
                "neoplastic-cell-infiltration-increase-v1",
                "structural-void-spread-v1",
                "tumor-burden-decrease-v1",
            },
        )
        self.assertEqual(
            set(repository.execution_scope["closed_mechanisms"]),
            {
                "breast-cohesive-nst-front",
                "breast-discohesive-single-file",
                "colorectal-gland-forming-front",
                "prostate-pattern-3-growth",
            },
        )
        colorectal_growth = replace(
            _case_stub(primitive="tumor-burden-increase-v1"),
            pathology_domain_id="colorectal-adenocarcinoma-v1",
            annotation_profile_id="glas-gland-v1",
            cell_population_profile_id=(
                "colorectal-cellvit-source-first-v1"
            ),
        )
        eligible, rejected = repository.eligible_mechanisms_for_case(
            case=colorectal_growth,
            available_checker_ids=JointGateRegistry().available_checker_ids,
            production=False,
        )
        self.assertFalse(eligible)
        self.assertIn("colorectal-gland-forming-front", rejected)
        self.assertIn("explicitly closed", rejected["colorectal-gland-forming-front"])
        self.assertIn(
            "lacks explicit stromal authority",
            repository.execution_selection_reason(
                primitive_id="tumor-burden-increase-v1",
                mechanism_id="colorectal-gland-forming-front",
            ),
        )
        self.assertIn(
            "no longer asks a general visual LLM",
            repository.execution_selection_reason(
                primitive_id="tumor-burden-increase-v1",
                mechanism_id="breast-cohesive-nst-front",
            ),
        )
        self.assertIsNone(
            repository.execution_selection_reason(
                primitive_id="tumor-burden-increase-v1",
                mechanism_id="breast-annotation-anchored-boundary-growth",
            )
        )
        local = _case_stub(
            primitive="cellularity-increase-v1",
            cell_budget=CellCountExtentBudget(12, 12, 15, 48, 0, 32),
        )
        eligible, rejected = repository.eligible_mechanisms_for_case(
            case=local,
            available_checker_ids=JointGateRegistry().available_checker_ids,
            production=False,
        )
        self.assertFalse(rejected)
        self.assertEqual(
            [item.mechanism_id for item in eligible],
            ["colorectal-local-population-modulation"],
        )
        breast_necrosis = replace(
            _case_stub(primitive="necrosis-appearance-v1"),
            pathology_domain_id="breast-invasive-carcinoma-v1",
            annotation_profile_id="bcss-semantic-v1",
            cell_population_profile_id="breast-cellvit-source-first-v1",
        )
        eligible, rejected = repository.eligible_mechanisms_for_case(
            case=breast_necrosis,
            available_checker_ids=JointGateRegistry().available_checker_ids,
            production=False,
        )
        self.assertFalse(rejected)
        self.assertEqual(
            [item.mechanism_id for item in eligible],
            ["breast-intratumoral-necrosis-turnover"],
        )
        colorectal_necrosis = _case_stub(primitive="necrosis-appearance-v1")
        eligible, _ = repository.eligible_mechanisms_for_case(
            case=colorectal_necrosis,
            available_checker_ids=JointGateRegistry().available_checker_ids,
            production=False,
        )
        self.assertEqual(eligible, ())

    def test_lung_cord_uses_joint_semantics_with_audited_tissue_adapter(self):
        repository = JointSkillRepository()
        case = replace(
            _case_stub(primitive="infiltrative-nest-cord-extension-v1"),
            pathology_domain_id="lung-carcinoma-v1",
            annotation_profile_id="ignite-semantic-v1",
            cell_population_profile_id="lung-cellvit-source-first-v1",
        )
        bundle = repository.compose(
            case=case,
            mechanism_id="lung-stromal-invasive-front",
            available_checker_ids=JointGateRegistry().available_checker_ids,
            production=False,
        )
        self.assertEqual(bundle.primitive.scope, "tissue_and_cell")
        self.assertEqual(
            bundle.mechanism.tissue_program.primitive_label_contracts[
                "infiltrative-nest-cord-extension-v1"
            ],
            {"source_labels": ("Stroma",), "target_labels": ("Tumor",)},
        )

    def test_cross_domain_cell_population_is_rejected(self):
        repository = JointSkillRepository()
        case = replace(
            _breast_case_stub(),
            cell_population_profile_id="colorectal-cellvit-source-first-v1",
        )
        with self.assertRaisesRegex(ValueError, "domain mismatch"):
            repository.compose(
                case=case,
                mechanism_id="breast-cohesive-nst-front",
                available_checker_ids=JointGateRegistry().available_checker_ids,
                production=False,
            )

    def test_necrosis_capability_follows_annotation_semantics_not_organ(self):
        rows = [
            item
            for item in JointSkillRepository().capability_matrix()
            if item["mechanism_id"] == "breast-intratumoral-necrosis-turnover"
        ]
        by_profile = {item["annotation_profile_id"]: item["status"] for item in rows}
        for profile in (
            "bcss-semantic-v1",
            "ignite-semantic-v1",
            "puma-semantic-v1",
        ):
            self.assertEqual(by_profile[profile], "conditionally_supported")
        for profile in (
            "glas-gland-v1",
            "orca-semantic-v1",
            "panda-gleason-v1",
        ):
            self.assertEqual(by_profile[profile], "unsupported")

    def test_production_rejects_draft_joint_skills(self):
        repository = JointSkillRepository()
        with self.assertRaisesRegex(ValueError, "internally reviewed"):
            repository.compose(
                case=_breast_case_stub(),
                mechanism_id="breast-cohesive-nst-front",
                available_checker_ids=JointGateRegistry().available_checker_ids,
                production=True,
            )

    def test_stroma_increase_mechanisms_reject_normal_epithelium_source(self):
        repository = JointSkillRepository()
        for mechanism in repository.mechanisms.values():
            contract = mechanism.tissue_program.primitive_label_contracts.get(
                "stroma-increase-v1"
            )
            if contract is not None:
                self.assertEqual(contract["source_labels"], ("Tumor",))
                self.assertEqual(contract["target_labels"], ("Stroma",))

    def test_stromal_replacement_cannot_invent_treatment_history_from_he(self):
        repository = JointSkillRepository()
        case = replace(
            _case_stub(primitive="stroma-increase-v1"),
            pathology_domain_id="lung-carcinoma-v1",
            annotation_profile_id="ignite-semantic-v1",
            cell_population_profile_id="lung-cellvit-source-first-v1",
            semantic_intent={"treatment_context": "none"},
        )
        with self.assertRaisesRegex(ValueError, "cannot invent treatment history"):
            repository.compose(
                case=case,
                mechanism_id="lung-operational-tumor-retreat",
                available_checker_ids=JointGateRegistry().available_checker_ids,
                production=False,
            )
        bundle = repository.compose(
            case=replace(
                case,
                semantic_intent={
                    "treatment_context": "post_treatment",
                    "scenario": "treatment_response",
                },
            ),
            mechanism_id="lung-operational-tumor-retreat",
            available_checker_ids=JointGateRegistry().available_checker_ids,
            production=False,
        )
        self.assertEqual(
            bundle.mechanism.mechanism_id,
            "lung-operational-tumor-retreat",
        )

    def test_budget_broker_reserves_whole_instance_union_without_lowering_floor(self):
        repository = JointSkillRepository()
        case = _breast_case_stub()
        bundle = repository.compose(
            case=case,
            mechanism_id="breast-cohesive-nst-front",
            available_checker_ids=JointGateRegistry().available_checker_ids,
            production=False,
        )
        solver = JointFeasibilitySolver()
        initial = solver.allocate(
            shape=(512, 512), budget=case.joint_area_budget, bundle=bundle
        )
        revised = solver.reserve_complete_instances(initial, reserve_pixels=4000)
        self.assertEqual(revised.reserved_complete_instance_pixels, 4000)
        self.assertGreaterEqual(
            revised.tissue_target_pixels, revised.tissue_floor_pixels
        )
        self.assertEqual(
            revised.tissue_target_pixels + revised.reserved_cell_only_pixels,
            revised.joint_target_pixels,
        )
        less_conservative = solver.reserve_complete_instances(
            revised,
            reserve_pixels=2000,
        )
        self.assertGreater(
            less_conservative.tissue_target_pixels,
            revised.tissue_target_pixels,
        )
        self.assertEqual(
            less_conservative.reserved_complete_instance_pixels,
            2000,
        )

    def test_budget_broker_keeps_seam_footprint_reserve_distinct_from_halo(self):
        repository = JointSkillRepository()
        case = replace(
            _case_stub(),
            pathology_domain_id="breast-invasive-carcinoma-v1",
            annotation_profile_id="bcss-semantic-v1",
            cell_population_profile_id="breast-cellvit-source-first-v1",
            primitive_id="tumor-burden-increase-v1",
        )
        bundle = repository.compose(
            case=case,
            mechanism_id="breast-cohesive-nst-front",
            available_checker_ids=JointGateRegistry().available_checker_ids,
            production=False,
        )
        allocation = JointFeasibilitySolver().allocate(
            shape=(512, 512), budget=case.joint_area_budget, bundle=bundle
        )
        self.assertEqual(allocation.reserved_layout_halo_pixels, 0)
        self.assertEqual(
            allocation.reserved_cell_footprint_spill_pixels,
            round(512 * 512 * 0.0055),
        )
        self.assertEqual(
            allocation.tissue_target_pixels
            + allocation.reserved_cell_only_pixels,
            allocation.joint_target_pixels,
        )

    def test_fixed_point_uses_candidate_preserving_closure(self):
        self.assertEqual(
            _candidate_preserving_closure_pixels([5200, 1800, 3400]),
            1800,
        )
        self.assertEqual(_candidate_preserving_closure_pixels([]), 0)

    def test_provisional_union_cannot_trigger_underfill_rebalance(self):
        self.assertFalse(
            _provisional_union_requires_rebalance(
                [36_701, 41_000], maximum_pixels=50_804
            )
        )
        self.assertTrue(
            _provisional_union_requires_rebalance(
                [60_114, 60_812], maximum_pixels=50_804
            )
        )
        self.assertFalse(
            _provisional_union_requires_rebalance(
                [50_000, 63_000], maximum_pixels=50_804
            )
        )

    def test_panda_growth_retains_hard_range_complete_instance_closure(self):
        self.assertTrue(
            _retain_visible_regression_whole_instance_closure(
                annotation_profile_id="panda-gleason-v1",
                primitive_id="tumor-burden-increase-v1",
                fallback_policy="max_feasible_below_target",
                predicted_pixels=[22_000],
                desired_max_pixels=20_972,
                hard_max_pixels=31_457,
            )
        )
        self.assertFalse(
            _retain_visible_regression_whole_instance_closure(
                annotation_profile_id="glas-gland-v1",
                primitive_id="tumor-burden-increase-v1",
                fallback_policy="max_feasible_below_target",
                predicted_pixels=[22_000],
                desired_max_pixels=20_972,
                hard_max_pixels=31_457,
            )
        )

    def test_dense_sheet_accepts_certified_mature_center_ledger(self):
        context = SimpleNamespace(
            bundle=SimpleNamespace(
                mechanism=SimpleNamespace(
                    cell_program=SimpleNamespace(
                        layout_programs=("dense_sheet",),
                        cluster_size_range=(1, 1),
                    )
                )
            ),
            candidate=SimpleNamespace(
                ledger=SimpleNamespace(removed_instance_ids=()),
                tool_trace={
                    "accepted_center_ledger": [
                        {"row": 10, "col": 12, "class_id": 1},
                        {"row": 20, "col": 22, "class_id": 1},
                    ],
                    "mature_probnet_contract": True,
                    "placed_count": 2,
                    "placements": [],
                    "reference_shape_integrity_certified": True,
                    "whole_instance_changes": True,
                },
            ),
            plan=SimpleNamespace(
                cell_plan=SimpleNamespace(
                    actions=("add",),
                    mechanism_program_id="dense_sheet",
                )
            ),
        )
        result = _mechanism_realization(context)
        self.assertTrue(result.passed)
        self.assertTrue(result.metrics["mature_center_ledger_certified"])
        context.candidate.tool_trace["placed_count"] = 1
        self.assertFalse(_mechanism_realization(context).passed)

    def test_density_decrease_accepts_final_whole_instance_removal_ledger(self):
        context = SimpleNamespace(
            bundle=SimpleNamespace(
                mechanism=SimpleNamespace(
                    cell_program=SimpleNamespace(
                        layout_programs=("localized_density_gradient",),
                        cluster_size_range=(1, 4),
                    )
                )
            ),
            candidate=SimpleNamespace(
                ledger=SimpleNamespace(
                    removed_instance_ids=("native-7", "native-10")
                ),
                tool_trace={
                    "placements": [],
                    "removed_source_instance_ids": ["native-7", "native-10"],
                    "whole_instance_changes": True,
                    "partial_source_instance_edits": 0,
                },
            ),
            plan=SimpleNamespace(
                cell_plan=SimpleNamespace(
                    actions=("remove_whole",),
                    mechanism_program_id="localized_density_gradient",
                )
            ),
        )

        result = _mechanism_realization(context)

        self.assertTrue(result.passed)
        self.assertTrue(result.metrics["removal_ledger_certified"])

    def test_budget_broker_rebalances_from_exact_executed_cell_spill(self):
        repository = JointSkillRepository()
        case = _breast_case_stub()
        bundle = repository.compose(
            case=case,
            mechanism_id="breast-cohesive-nst-front",
            available_checker_ids=JointGateRegistry().available_checker_ids,
            production=False,
        )
        solver = JointFeasibilitySolver()
        initial = solver.allocate(
            shape=(512, 512), budget=case.joint_area_budget, bundle=bundle
        )
        revised = solver.reserve_observed_cell_spill(
            initial,
            complete_instance_pixels=4_457,
            footprint_spill_pixels=3_220,
        )
        self.assertEqual(revised.reserved_complete_instance_pixels, 4_457)
        self.assertEqual(revised.reserved_cell_footprint_spill_pixels, 3_220)
        self.assertEqual(
            revised.tissue_target_pixels,
            max(
                revised.tissue_floor_pixels,
                revised.joint_target_pixels - 4_457 - 3_220,
            ),
        )

    def test_exact_spill_replaces_instead_of_double_counting_layout_reserve(self):
        repository = JointSkillRepository()
        case = replace(
            _case_stub(),
            primitive_id="cellularity-increase-v1",
            cell_count_extent_budget=CellCountExtentBudget(
                12, 12, 15, 48, 0, 32
            ),
        )
        bundle = repository.compose(
            case=case,
            mechanism_id="colorectal-local-population-modulation",
            available_checker_ids=JointGateRegistry().available_checker_ids,
            production=False,
        )
        solver = JointFeasibilitySolver()
        initial = solver.allocate(
            shape=(512, 512), budget=case.joint_area_budget, bundle=bundle
        )
        self.assertGreater(initial.reserved_layout_halo_pixels, 0)
        revised = solver.reserve_observed_cell_spill(
            initial,
            complete_instance_pixels=2_000,
            footprint_spill_pixels=3_000,
        )
        self.assertEqual(
            revised.tissue_target_pixels,
            max(
                revised.tissue_floor_pixels,
                revised.joint_target_pixels - 5_000,
            ),
        )

    def test_capacity_adaptive_budget_can_compile_below_standard_floor(self):
        repository = JointSkillRepository()
        case = _breast_case_stub(
            budget=JointAreaBudget(capacity_floor_policy="lower_to_proven_max_safe")
        )
        bundle = repository.compose(
            case=case,
            mechanism_id="breast-cohesive-nst-front",
            available_checker_ids=JointGateRegistry().available_checker_ids,
            production=False,
        )
        solver = JointFeasibilitySolver()
        initial = solver.allocate(
            shape=(512, 512), budget=case.joint_area_budget, bundle=bundle
        )
        revised = solver.reserve_complete_instances(
            initial,
            reserve_pixels=20000,
            allow_capacity_floor_fallback=True,
        )
        self.assertLess(revised.tissue_target_pixels, revised.tissue_floor_pixels)
        self.assertEqual(revised.tissue_execution_floor_pixels, 0)

    def test_panda_cell_capacity_fallback_rebrokers_to_existing_hard_floor(self):
        repository = JointSkillRepository()
        case = replace(
            _case_stub(),
            pathology_domain_id="prostate-adenocarcinoma-v1",
            annotation_profile_id="panda-gleason-v1",
            cell_population_profile_id="prostate-cellvit-source-first-v1",
            primitive_id="cohesive-boundary-expansion-v1",
            joint_area_budget=JointAreaBudget(
                target_fraction=0.08,
                min_fraction=0.04,
                max_fraction=0.12,
                tissue_min_fraction=0.04,
                fallback_policy="max_feasible_below_target",
            ),
        )
        bundle = repository.compose(
            case=case,
            mechanism_id="prostate-pattern-4-growth",
            available_checker_ids=JointGateRegistry().available_checker_ids,
            production=False,
        )
        solver = JointFeasibilitySolver()
        initial = solver.allocate(
            shape=(512, 512), budget=case.joint_area_budget, bundle=bundle
        )
        revised = solver.fallback_tissue_target_to_execution_floor(initial)
        self.assertIn(
            case.primitive_id, PANDA_CELL_CAPACITY_FALLBACK_PRIMITIVES
        )
        self.assertTrue(
            {
                "local-invasive-clearance-v1",
                "stroma-increase-v1",
            }.issubset(PANDA_CELL_CAPACITY_FALLBACK_PRIMITIVES)
        )
        self.assertEqual(
            revised.tissue_target_pixels,
            revised.tissue_execution_floor_pixels,
        )
        self.assertEqual(revised.joint_hard_min_pixels, initial.joint_hard_min_pixels)
        self.assertEqual(revised.joint_hard_max_pixels, initial.joint_hard_max_pixels)

    def test_capacity_adaptive_budget_enforces_meaningful_edit_floor(self):
        repository = JointSkillRepository()
        case = _breast_case_stub(
            budget=JointAreaBudget(
                capacity_floor_policy="lower_to_proven_max_safe",
                minimum_effective_fraction=0.05,
            )
        )
        bundle = repository.compose(
            case=case,
            mechanism_id="breast-cohesive-nst-front",
            available_checker_ids=JointGateRegistry().available_checker_ids,
            production=False,
        )
        initial = JointFeasibilitySolver().allocate(
            shape=(512, 512), budget=case.joint_area_budget, bundle=bundle
        )
        self.assertEqual(initial.tissue_execution_floor_pixels, 13108)
        self.assertLess(
            initial.tissue_execution_floor_pixels,
            initial.tissue_floor_pixels,
        )

    def test_underfill_replan_keeps_prior_front_and_adds_unique_capacity(self):
        interfaces = [
            SimpleNamespace(
                interface_id="if-a",
                source_component_id="source-1",
                contact_pixels=90,
            ),
            SimpleNamespace(
                interface_id="if-b",
                source_component_id="source-1",
                contact_pixels=80,
            ),
            SimpleNamespace(
                interface_id="if-c",
                source_component_id="source-2",
                contact_pixels=40,
            ),
        ]
        capacities = {"if-a": 80, "if-b": 70, "if-c": 40}
        component_limits = {"source-1": 100, "source-2": 40}
        ranked, marginal = _rank_interfaces_by_marginal_capacity(
            interfaces,
            capacity_by_id=capacities,
            component_capacity_limits=component_limits,
            locked_interface_ids=("if-a",),
            previous_actual_by_interface={"if-a": 30},
            previous_actual_by_source={"source-1": 30},
        )
        self.assertEqual(
            [item.interface_id for item in ranked],
            ["if-a", "if-b", "if-c"],
        )
        self.assertEqual(marginal, {"if-a": 30, "if-b": 70, "if-c": 40})
        allocation_capacities = _component_capped_allocation_capacities(
            interfaces,
            capacity_by_id=capacities,
            component_capacity_limits=component_limits,
        )
        self.assertEqual(sum(allocation_capacities[:2]), 100)
        self.assertEqual(allocation_capacities[2], 40)
        self.assertEqual(sum(allocation_capacities), 140)

    def test_fixed_breast_fallback_compiles_against_manifest_three_percent(self):
        case = _breast_case_stub(
            budget=JointAreaBudget(
                target_fraction=0.08,
                min_fraction=0.04,
                max_fraction=0.08,
                tissue_min_fraction=0.04,
                capacity_floor_policy="lower_to_proven_max_safe",
                minimum_effective_fraction=0.03,
            )
        )
        bundle = JointSkillRepository().compose(
            case=case,
            mechanism_id="breast-cohesive-nst-front",
            available_checker_ids=JointGateRegistry().available_checker_ids,
            production=False,
        )
        allocation = JointFeasibilitySolver().allocate(
            shape=(512, 512), budget=case.joint_area_budget, bundle=bundle
        )
        tissue_case = _as_tissue_case(
            case, allocation=allocation, shape=(512, 512)
        )
        self.assertEqual(allocation.tissue_floor_pixels, 10_486)
        self.assertEqual(allocation.tissue_execution_floor_pixels, 7_865)
        self.assertEqual(
            tissue_case.area_budget.hard_pixel_interval(
                np.zeros((512, 512), dtype=np.uint8),
                np.ones((512, 512), dtype=bool),
            )[0],
            7_865,
        )

    def test_tissue_tool_rejects_allocation_below_binding_execution_floor(self):
        repository = JointSkillRepository()
        case = _breast_case_stub(
            budget=JointAreaBudget(
                capacity_floor_policy="lower_to_proven_max_safe",
                minimum_effective_fraction=0.05,
            )
        )
        bundle = repository.compose(
            case=case,
            mechanism_id="breast-cohesive-nst-front",
            available_checker_ids=JointGateRegistry().available_checker_ids,
            production=False,
        )
        allocation = JointFeasibilitySolver().allocate(
            shape=(512, 512), budget=case.joint_area_budget, bundle=bundle
        )
        allocation = replace(
            allocation,
            tissue_target_pixels=12_000,
            tissue_execution_floor_pixels=13_108,
        )

        with self.assertRaisesRegex(
            JointContractError, "below the binding meaningful floor"
        ):
            _as_tissue_case(case, allocation=allocation, shape=(512, 512))

    def test_joint_router_recognizes_nuclei_only_change_as_non_noop(self):
        manifest = {
            "ledger": {
                "tissue_fraction": 0.0,
                "cell_fraction": 0.04,
                "joint_fraction": 0.04,
                "generation_support_fraction": 0.07,
            }
        }
        route = route_joint_handoff(manifest)
        self.assertEqual(route.mode, "inpaint")
        self.assertGreater(route.joint_fraction, 0)

        change = np.zeros((20, 20), dtype=bool)
        change[2:4, 2:4] = True
        change[14:16, 14:16] = True
        support = change.copy()
        support.flat[:20] = True
        agentic = build_agentic_joint_route(
            manifest,
            joint_change_mask=change,
            generation_support_mask=support,
            reference_tissue_mask=np.ones((20, 20), dtype=np.uint8),
        )
        self.assertEqual(agentic.primary_mode, "inpaint")
        self.assertEqual(agentic.candidate_modes, ("inpaint", "cross"))
        self.assertEqual(agentic.features.transition_count, 0)
        self.assertEqual(agentic.features.component_count, 3)

    def test_joint_router_forces_cross_for_large_generation_support(self):
        manifest = {
            "ledger": {
                "tissue_fraction": 0.0,
                "cell_fraction": 0.04,
                "joint_fraction": 0.04,
                "generation_support_fraction": 0.65,
            }
        }
        route = route_joint_handoff(manifest)
        self.assertEqual(route.mode, "cross")
        self.assertTrue(route.force_cross)

        change = np.zeros((20, 20), dtype=bool)
        change[2:4, 2:4] = True
        change[14:16, 14:16] = True
        support = np.zeros((20, 20), dtype=bool)
        support.flat[:260] = True
        support |= change
        manifest["ledger"]["generation_support_fraction"] = float(
            np.mean(support)
        )
        agentic = build_agentic_joint_route(
            manifest,
            joint_change_mask=change,
            generation_support_mask=support,
            reference_tissue_mask=np.ones((20, 20), dtype=np.uint8),
        )
        self.assertEqual(agentic.primary_mode, "cross")
        self.assertEqual(agentic.candidate_modes, ("cross",))
        self.assertAlmostEqual(
            agentic.features.change_ratio_image,
            float(np.mean(support)),
        )

    def test_joint_router_rejects_generation_support_ledger_drift(self):
        manifest = {
            "ledger": {
                "joint_fraction": 0.01,
                "generation_support_fraction": 0.10,
            }
        }
        change = np.zeros((20, 20), dtype=bool)
        change[0:2, 0:2] = True
        support = np.zeros((20, 20), dtype=bool)
        support[0:10, :] = True
        with self.assertRaisesRegex(
            JointContractError,
            "does not match the approved handoff ledger",
        ):
            build_agentic_joint_route(
                manifest,
                joint_change_mask=change,
                generation_support_mask=support,
                reference_tissue_mask=np.ones((20, 20), dtype=np.uint8),
            )

    def test_melanoma_scatter_is_cell_only_and_does_not_borrow_tissue_floor(self):
        repository = JointSkillRepository()
        case = replace(
            _case_stub(
                primitive="peritumoral-neoplastic-scatter-increase-v1",
                cell_budget=CellCountExtentBudget(8, 6, 10, 48, 4, 32),
            ),
            pathology_domain_id="melanoma-v1",
            annotation_profile_id="puma-semantic-v1",
            cell_population_profile_id="melanoma-cellvit-source-first-v1",
        )
        bundle = repository.compose(
            case=case,
            mechanism_id="melanoma-discohesive-junctional",
            available_checker_ids=JointGateRegistry().available_checker_ids,
            production=False,
        )
        self.assertEqual(bundle.primitive.scope, "cell_only")
        self.assertFalse(bundle.mechanism.coupling.tissue_floor_applies)
        self.assertEqual(
            bundle.mechanism.tissue_program.allowed_tools, ("preserve_tissue",)
        )

    def test_mature_probnet_adapter_binds_the_frozen_online_contract(self):
        executor = MatureProbNetCellExecutor(
            MatureProbNetConfig(
                dataset_name="GlaS",
                checkpoint="/models/probnet.pt",
                instance_library="/models/nuclei-library",
                device="cuda:0",
            )
        )
        command = executor.build_command(
            seed=17,
            target_tissue_path=Path("target.png"),
            source_tissue_path=Path("source.png"),
            source_nuclei_path=Path("nuclei.png"),
            reference_nuclei_shapes_path=Path("reference-shapes.png"),
            source_instance_authority_path=Path("instance-authority.json"),
            generation_region_path=Path("G.png"),
            population_region_path=Path("T-pop.png"),
            placement_region_path=Path("P.png"),
            erasure_region_path=Path("E.png"),
            required_placement_region_path=Path("seam.png"),
            packing_witness_path=Path("packing-witness.json"),
            minimum_required_placements=1,
            maximum_required_placements=1,
            required_nucleus_class=1,
            output_path=Path("out.png"),
            prohibited_tissue_ids=(0, 9),
            allowed_new_cell_classes=(1, 3),
        )
        self.assertIn("inpaint_cells.generate", command)
        self.assertIn("--no-widen-edit-region", command)
        self.assertIn("--require-sampling-audit", command)
        self.assertIn("--require-exact-target-count", command)
        self.assertIn("--reference-nuclei-shapes", command)
        self.assertIn("--source-instance-authority", command)
        self.assertIn("--placement-region", command)
        self.assertIn("--population-region", command)
        self.assertIn("--required-placement-region", command)
        self.assertIn("--minimum-required-placements", command)
        self.assertIn("--maximum-required-placements", command)
        self.assertIn("--required-nucleus-type", command)
        self.assertIn("--packing-witness", command)
        self.assertIn("--trust-complete-deletion-region", command)
        self.assertIn("--allowed-nucleus-types", command)
        self.assertIn("101", command)
        self.assertIn("103", command)
        self.assertEqual(command[command.index("--device") + 1], "cuda")
        shape_ratio_index = command.index("--reference-shape-max-area-ratio")
        self.assertEqual(command[shape_ratio_index + 1], "0.0")
        shape_min_index = command.index("--reference-shape-min-area")
        self.assertEqual(command[shape_min_index + 1], "8")
        self.assertEqual(
            command[command.index("--maximum-required-placements") + 1],
            "1",
        )
        self.assertEqual(
            command[command.index("--required-nucleus-type") + 1],
            "101",
        )

    def test_packing_witness_maps_cellvit_classes_to_mature_schema(self):
        self.assertEqual(
            _mature_nucleus_area_medians(
                {"class_reference_median_area_px": {"1": 626.5, "3": 260}}
            ),
            {"101": 626.5, "103": 260.0},
        )

    def test_mature_packing_reconstructs_bound_rotation_variant(self):
        component = np.zeros((12, 12), dtype=bool)
        component[2:7, 4:7] = np.asarray(
            [
                [1, 0, 0],
                [1, 1, 0],
                [1, 1, 1],
                [0, 1, 1],
                [0, 0, 1],
            ],
            dtype=bool,
        )
        source_shape = component[2:7, 4:7]
        rotated = ndimage.rotate(
            source_shape.astype(np.uint8),
            angle=45,
            reshape=True,
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        ).astype(bool)
        contract = SimpleNamespace(
            contract_id="contract-1",
            packing_certificate={
                "version": "complete-footprint-packing-v21",
                "requested_count": 1,
                "required_seam_count": 1,
                "placements": [
                    {
                        "row": 8,
                        "col": 8,
                        "class_id": 1,
                        "area_px": int(np.count_nonzero(rotated)),
                        "reference_instance_id": "native-9:rotate-45",
                        "required_seam": True,
                    }
                ],
            },
        )
        scene = SimpleNamespace(
            cells=SimpleNamespace(
                instances=(
                    SimpleNamespace(
                        instance_id="native-9",
                        class_id=1,
                        bbox_xyxy=(4, 2, 7, 7),
                    ),
                )
            ),
            instance_masks={"native-9": component},
        )
        witness = _compile_packing_witness(contract=contract, scene=scene)
        placement = witness["placements"][0]
        self.assertEqual(placement["reference_instance_id"], "native-9:rotate-45")
        self.assertEqual(placement["source_reference_instance_id"], "native-9")
        self.assertEqual(placement["rotation_degrees"], 45)
        self.assertEqual(len(placement["offsets_yx"]), np.count_nonzero(rotated))

    def test_mature_architecture_replay_preserves_group_cardinality_trace(self):
        ledger = [
            {
                "row": 18,
                "col": 24,
                "class_id": 1,
                "area_px": 350,
                "shape_source": "compiled_reference_witness",
                "reference_instance_id": "nuc-c1-0029",
            },
            {
                "row": 22,
                "col": 29,
                "class_id": 1,
                "area_px": 307,
                "shape_source": "library",
                "reference_instance_id": None,
            },
        ]
        contract = SimpleNamespace(
            primitive_id="peritumoral-tumor-nest-formation-v1",
            packing_certificate={
                "placements": [
                    {
                        "row": item["row"] + 2,
                        "col": item["col"] - 1,
                        "class_id": item["class_id"],
                    }
                    for item in ledger
                ]
            },
        )

        placements = _architecture_placement_trace(
            contract=contract,
            accepted_instance_area_ledger=ledger,
        )

        self.assertEqual(len(placements), 2)
        self.assertEqual({item["cluster_size"] for item in placements}, {2})
        self.assertEqual(
            {
                item["reference_instance_id"]
                for item in placements
                if item["reference_instance_id"]
            },
            {"nuc-c1-0029"},
        )
        self.assertEqual(
            {item["reference_source"] for item in placements},
            {"compiled_reference_witness", "calibrated_instance_library"},
        )
        self.assertTrue(
            all(item["packing_witness_replayed"] for item in placements)
        )

    def test_reference_gate_distinguishes_calibrated_library_provenance(self):
        trace = {
            "reference_shape_ids": ["nuc-c1-0029"],
            "reference_shape_rejections": {},
            "reference_shape_integrity_certified": True,
            "mature_probnet_contract": True,
            "shape_sampling": {
                "policy": (
                    "component_local_same_class_reference_then_"
                    "component_calibrated_library"
                ),
                "library_size_calibration": {
                    "uncalibrated_no_reference_by_type": {}
                },
            },
            "placements": [
                {
                    "reference_instance_id": "nuc-c1-0029",
                    "reference_source": "compiled_reference_witness",
                },
                {
                    "reference_instance_id": None,
                    "reference_source": "calibrated_instance_library",
                },
            ],
        }
        context = SimpleNamespace(
            plan=SimpleNamespace(
                cell_plan=SimpleNamespace(actions=("add",), core_zone="global")
            ),
            candidate=SimpleNamespace(tool_trace=trace),
            scene=SimpleNamespace(cells=SimpleNamespace(instances=())),
        )

        passed = _reference_shape_integrity(context)
        self.assertTrue(passed.passed)
        self.assertEqual(
            passed.metrics["calibrated_library_placement_count"], 1
        )

        trace["placements"][1]["reference_source"] = "unknown"
        failed = _reference_shape_integrity(context)
        self.assertFalse(failed.passed)
        self.assertEqual(failed.metrics["missing_reference_bindings"], ["placement:1"])

    def test_continuity_density_interval_compiles_to_an_exact_tool_target(self):
        nuclei = np.zeros((24, 24), dtype=np.uint8)
        nuclei[8, 5] = 1
        nuclei[12, 5] = 1
        target_tissue = np.ones_like(nuclei, dtype=np.uint8)
        change = np.zeros_like(nuclei, dtype=bool)
        change[6:16, 8:18] = True
        anchor = np.zeros_like(change)
        anchor[6:16, 8] = True
        continuity = change.copy()

        quota = compile_continuity_center_quota(
            nuclei_mask=nuclei,
            target_tissue_mask=target_tissue,
            tissue_change=change,
            continuity_region=continuity,
            continuity_anchor_mask=anchor,
            continuity_width_px=6,
            density_ratio_range=(0.5, 2.0),
            requires_new_target_cells=True,
            target_class=1,
            target_fine_ids=(1,),
        )

        self.assertEqual(quota.outer_count, 2)
        self.assertIsNotNone(quota.maximum_count)
        self.assertGreaterEqual(quota.target_count, quota.minimum_count)
        self.assertLessEqual(quota.target_count, quota.maximum_count)

        executable_count = compile_executable_continuity_count(
            quota,
            anchor_pixels=1000,
            maximum_empty_run_px=50,
            minimum_anchor_coverage_fraction=0.5,
        )
        self.assertGreaterEqual(executable_count, quota.minimum_count)
        self.assertLessEqual(executable_count, quota.maximum_count)

    def test_continuity_quota_can_use_executor_center_ledger(self):
        nuclei = np.zeros((24, 24), dtype=np.uint8)
        target_tissue = np.ones_like(nuclei, dtype=np.uint8)
        change = np.zeros_like(nuclei, dtype=bool)
        change[6:16, 8:18] = True
        anchor = np.zeros_like(change)
        anchor[6:16, 8] = True
        centers = np.zeros_like(change)
        centers[8, 5] = True
        centers[12, 5] = True

        quota = compile_continuity_center_quota(
            nuclei_mask=nuclei,
            target_tissue_mask=target_tissue,
            tissue_change=change,
            continuity_region=change,
            continuity_anchor_mask=anchor,
            continuity_width_px=6,
            density_ratio_range=(0.5, 2.0),
            requires_new_target_cells=True,
            target_class=1,
            target_fine_ids=(1,),
            target_center_mask=centers,
        )

        self.assertEqual(quota.outer_count, 2)

    def test_api_planner_schema_exposes_cell_execution_linkage(self):
        cell_schema = JOINT_PLAN_JSON_SCHEMA["properties"]["cell_plan"]
        required = set(cell_schema["required"])
        self.assertTrue(
            {
                "baseline_mode",
                "mechanism_program_id",
                "mechanism_quota_role",
                "anchor_ids",
            }.issubset(required)
        )
        self.assertNotIn("protected_instance_ids", required)

    def test_cell_tissue_compatibility_is_versioned_observation_knowledge(self):
        profile = JointSkillRepository().cell_observation_profiles[
            "cellvit-five-class-v1"
        ]
        self.assertEqual(profile.tissue_compatible_classes["Stroma"], (2, 3))
        self.assertNotIn(1, profile.tissue_compatible_classes["Stroma"])

    def test_cell_tissue_compatibility_audits_only_the_added_footprint(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        tumor_id = schema.resolve_fine_ids("Tumor")[0]
        stroma_id = schema.resolve_fine_ids("Stroma")[0]
        source = np.zeros((7, 7), dtype=np.uint8)
        source[3, 3] = 1
        target = source.copy()
        target[3, 4] = 1  # same-class addition touches an existing instance
        tissue = np.full(source.shape, tumor_id, dtype=np.uint8)
        tissue[3, 4] = stroma_id
        authorized = np.zeros(source.shape, dtype=bool)
        authorized[3, 4] = True
        context = SimpleNamespace(
            source_nuclei=source,
            schema=schema,
            candidate=SimpleNamespace(
                target_nuclei_mask=target,
                target_tissue_mask=tissue,
                cell_change=target != source,
                tissue_change=np.zeros(source.shape, dtype=bool),
            ),
            plan=SimpleNamespace(
                cell_plan=SimpleNamespace(
                    interface_ids=(),
                    core_zone="pop:component:test",
                ),
                coupling_plan=SimpleNamespace(
                    allow_neoplastic_in_non_tumor_tissue=True,
                ),
            ),
            scene=SimpleNamespace(
                population_zone_masks={"pop:component:test": authorized},
            ),
            bundle=SimpleNamespace(
                primitive=SimpleNamespace(scope="cell_only"),
                cell_observation_profile=SimpleNamespace(
                    tissue_compatible_classes={
                        "Tumor": (1,),
                        "Stroma": (2, 3),
                    }
                ),
            ),
            executable_contract=SimpleNamespace(
                allowed_new_cell_classes=(1,),
                cell_program=SimpleNamespace(
                    depletion_profile_id=None,
                    support_context_region=authorized,
                ),
            ),
        )

        check = _cell_tissue_compatibility(context)

        self.assertTrue(check.passed, check.metrics)
        self.assertEqual(check.metrics["incompatible_host_pixels"], 0)

    def test_cell_quota_rejects_fallback_below_packing_safe_minimum(self):
        context = SimpleNamespace(
            candidate=SimpleNamespace(
                tool_trace={
                    "biological_desired_count": 8,
                    "desired_count": 8,
                    "resolved_count": 5,
                    "requested_count": 5,
                    "placed_count": 5,
                    "batch_max_attainable_count": 5,
                    "capacity_max_count": 5,
                    "cell_capacity_certified": True,
                    "cell_capacity_fallback_used": True,
                }
            ),
            executable_contract=SimpleNamespace(
                packing_certificate={
                    "passed": True,
                    "requested_count": 8,
                    "minimum_safe_count": 6,
                }
            ),
        )

        check = _cell_quota(context)

        self.assertFalse(check.passed)
        self.assertEqual(
            check.metrics["packing_certificate_minimum_safe_count"], 6
        )
        self.assertFalse(
            check.metrics["packing_certificate_minimum_satisfied"]
        )

    def test_g2_defaults_never_assign_cell_only_or_unrepresentable_mechanism(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("image.png", "tissue.png", "nuclei.png"):
                Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(root / name)
            base = {
                "case_id": "g2-fixture",
                "instruction": "increase tumor",
                "g2_primitive": "tumor_increase",
                "source_image": "image.png",
                "source_tissue_mask": "tissue.png",
                "source_nuclei_mask": "nuclei.png",
            }
            oral = build_local_joint_records(
                [{**base, "organ": "oral"}], asset_root=root
            )[0]
            self.assertEqual(
                oral["joint_area_budget"]["capacity_floor_policy"],
                "lower_to_proven_max_safe",
            )
            self.assertEqual(
                oral["joint_area_budget"]["minimum_effective_fraction"],
                0.14,
            )
            self.assertTrue(
                oral["provenance"]["require_mature_probnet_regeneration"]
            )
            self.assertEqual(
                oral["provenance"]["joint_mechanism_id"], "__abstain__"
            )
            self.assertIn(
                "visual mechanism selection",
                oral["provenance"]["joint_mechanism_assignment_reason"],
            )
            colorectal = build_local_joint_records(
                [{**base, "organ": "colorectal"}], asset_root=root
            )[0]
            self.assertEqual(
                colorectal["provenance"]["joint_mechanism_id"],
                "__abstain__",
            )
            self.assertIn(
                "gland_or_lumen_support",
                colorectal["provenance"]["joint_mechanism_assignment_reason"],
            )


class JointLedgerTests(unittest.TestCase):
    def test_joint_union_does_not_double_count_tissue_and_cells(self):
        tissue = np.zeros((32, 32), dtype=np.uint8)
        target_tissue = tissue.copy()
        target_tissue[8:24, 8:24] = 1
        source_cells = np.zeros_like(tissue)
        source_cells[10:13, 10:13] = 1
        target_cells = np.zeros_like(tissue)
        target_cells[18:21, 18:21] = 1
        result = analyze_joint_change(
            source_tissue=tissue,
            target_tissue=target_tissue,
            source_nuclei=source_cells,
            target_nuclei=target_cells,
            generation_halo_px=2,
        )
        self.assertEqual(result.ledger.tissue_pixels, 256)
        self.assertEqual(result.ledger.joint_pixels, 256)
        self.assertGreater(result.ledger.cell_pixels, 0)
        self.assertTrue(result.whole_instance_changes)

    def test_partial_source_nucleus_is_detected(self):
        tissue = np.ones((16, 16), dtype=np.uint8)
        source = np.zeros_like(tissue)
        source[4:8, 4:8] = 1
        target = source.copy()
        target[4:6, 4:8] = 0
        result = analyze_joint_change(
            source_tissue=tissue,
            target_tissue=tissue,
            source_nuclei=source,
            target_nuclei=target,
            generation_halo_px=0,
        )
        self.assertFalse(result.whole_instance_changes)
        self.assertTrue(result.partial_source_instance_ids)

    def test_added_pixels_do_not_expand_across_retained_semantic_component(self):
        tissue = np.ones((24, 24), dtype=np.uint8)
        source = np.zeros_like(tissue)
        source[2:22, 2:8] = 2
        target = source.copy()
        target[10:14, 8:11] = 2
        support = np.zeros_like(tissue, dtype=bool)
        support[10:14, 8:11] = True

        result = analyze_joint_change(
            source_tissue=tissue,
            target_tissue=tissue,
            source_nuclei=source,
            target_nuclei=target,
            generation_halo_px=0,
            generation_support_contract=support,
        )

        self.assertTrue(np.array_equal(result.cell_change, support))
        self.assertEqual(result.ledger.added_nucleus_pixels, 12)
        self.assertEqual(len(result.ledger.added_instance_ids), 1)

    def test_native_class_disagreement_does_not_mark_unchanged_instance_removed(self):
        tissue = np.ones((24, 24), dtype=np.uint8)
        source = np.zeros_like(tissue)
        source[2:8, 2:8] = 3
        target = source.copy()
        target[14:17, 14:17] = 2
        support = np.zeros_like(tissue, dtype=bool)
        support[14:17, 14:17] = True
        native = np.zeros_like(tissue, dtype=bool)
        native[2:8, 2:8] = True

        result = analyze_joint_change(
            source_tissue=tissue,
            target_tissue=tissue,
            source_nuclei=source,
            target_nuclei=target,
            generation_halo_px=0,
            generation_support_contract=support,
            source_instance_masks={"native-1": native},
            source_instance_classes={"native-1": 2},
            erased_source_instance_ids=(),
        )

        self.assertTrue(np.array_equal(result.cell_change, support))
        self.assertEqual(result.ledger.removed_nucleus_pixels, 0)
        self.assertEqual(result.ledger.retained_instance_ids, ("native-1",))

    def test_local_population_reference_priority_excludes_distant_native_shapes(self):
        local = ReferenceNucleusShape(
            instance_id="local-1",
            class_id=2,
            mask=np.ones((3, 3), dtype=bool),
            source="native_instance_json",
            area_px=9,
        )
        distant = ReferenceNucleusShape(
            instance_id="distant-1",
            class_id=2,
            mask=np.ones((2, 2), dtype=bool),
            source="native_instance_json",
            area_px=4,
        )
        calibrated = ReferenceNucleusShape(
            instance_id="calibrated-1",
            class_id=2,
            mask=np.ones((4, 4), dtype=bool),
            source="calibrated_dataset_instance_library",
            area_px=16,
        )
        scene = SimpleNamespace(
            cells=SimpleNamespace(
                instances=(
                    SimpleNamespace(
                        instance_id="local-1",
                        tissue_component_id="component-7",
                        nearest_interface_id=None,
                    ),
                    SimpleNamespace(
                        instance_id="distant-1",
                        tissue_component_id="component-9",
                        nearest_interface_id=None,
                    ),
                )
            )
        )

        selected = _prioritize_local_references(
            (distant, local, calibrated),
            scene=scene,
            interface_ids=(),
            core_zone="pop:component:component-7",
        )

        self.assertEqual(
            tuple(item.instance_id for item in selected),
            ("local-1", "calibrated-1"),
        )

    def test_reference_library_rejects_all_patch_border_shapes(self):
        tissue = np.full((32, 32), 2, dtype=np.uint8)
        nuclei = np.zeros_like(tissue)
        nuclei[10:14, 10:14] = 1
        nuclei[0:3, 3:6] = 1
        nuclei[29:32, 10:13] = 1
        nuclei[16:19, 0:3] = 1
        nuclei[22:25, 29:32] = 1
        scene = build_joint_scene_analysis(
            tissue,
            nuclei,
            schema=MaskProfileSchema.from_reference_profile("GLaS"),
            pixel_size_um=None,
        )
        references, rejected = build_reference_shape_library(scene, class_id=1)
        self.assertEqual(len(references), 1)
        self.assertEqual(references[0].area_px, 16)
        self.assertEqual(
            list(rejected.values()).count("patch_boundary_censored_shape"), 4
        )

    def test_semantic_scene_uses_digest_bound_calibrated_shape_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bucket = root / "nuclei_instances" / "tissue_01_Tumor"
            bucket.mkdir(parents=True)
            (root / "statistics.json").write_text(
                json.dumps(
                    {
                        "dataset": "BCSS",
                        "statistics": {
                            "1": {
                                "name": "Tumor",
                                "nuclei_types": {
                                    "101": {
                                        "stored_count": 3,
                                        "mean_area": 9.0,
                                    }
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            for index, size in enumerate((2, 3, 4), start=1):
                mask = np.ones((size, size), dtype=bool)
                np.savez(
                    bucket / f"{index:06d}.npz",
                    mask=mask,
                    type=np.asarray(101),
                    area=np.asarray(int(mask.sum())),
                )
            authority = load_reference_shape_authority(
                root,
                dataset_name="BCSS",
                class_ids=(1,),
            )
            tissue = np.full((32, 32), 2, dtype=np.uint8)
            nuclei = np.zeros_like(tissue)
            nuclei[8:13, 8:13] = 1
            scene = build_joint_scene_analysis(
                tissue,
                nuclei,
                schema=MaskProfileSchema.from_reference_profile("GLaS"),
                pixel_size_um=None,
                reference_shape_authority=authority,
            )

            references, rejected = build_reference_shape_library(
                scene,
                class_id=1,
                allow_calibrated_fallback=True,
            )

            self.assertEqual(
                scene.cells.observation_quality,
                "semantic_distance_watershed",
            )
            self.assertTrue(references)
            self.assertTrue(
                references[0].instance_id.startswith("nuc-c1-")
            )
            self.assertTrue(
                any(
                    item.source == "calibrated_dataset_instance_library"
                    for item in references
                )
            )
            self.assertFalse(
                any(
                    value == "semantic_shape_superseded_by_calibrated_library_authority"
                    for value in rejected.values()
                )
            )
            self.assertEqual(len(authority.authority_sha256), 64)

    def test_calibrated_shape_must_remain_one_semantic_instance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bucket = root / "nuclei_instances" / "tissue_01_Tumor"
            bucket.mkdir(parents=True)
            (root / "statistics.json").write_text(
                json.dumps(
                    {
                        "dataset": "BCSS",
                        "statistics": {
                            "1": {
                                "name": "Tumor",
                                "nuclei_types": {
                                    "101": {
                                        "stored_count": 2,
                                        "mean_area": 25.0,
                                    }
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            single = np.ones((5, 5), dtype=bool)
            two_lobes = np.zeros((9, 17), dtype=bool)
            yy, xx = np.ogrid[:9, :17]
            two_lobes |= (yy - 4) ** 2 + (xx - 4) ** 2 <= 4**2
            two_lobes |= (yy - 4) ** 2 + (xx - 12) ** 2 <= 4**2
            two_lobes[4, 4:13] = True
            for name, mask in (
                ("000001.npz", single),
                ("000002.npz", two_lobes),
            ):
                np.savez(
                    bucket / name,
                    mask=mask,
                    type=np.asarray(101),
                    area=np.asarray(int(mask.sum())),
                )

            authority = load_reference_shape_authority(
                root,
                dataset_name="BCSS",
                class_ids=(1,),
            )

            selected = authority.shapes_by_class[1]
            self.assertEqual(len(selected), 1)
            self.assertIn("000001.npz", selected[0].instance_id)
            self.assertNotIn(
                "000002.npz",
                " ".join(item.instance_id for item in selected),
            )

    def test_calibrated_authority_deduplicates_identical_library_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bucket = root / "nuclei_instances" / "tissue_01_Tumor"
            bucket.mkdir(parents=True)
            (root / "statistics.json").write_text(
                json.dumps(
                    {
                        "dataset": "BCSS",
                        "statistics": {
                            "1": {
                                "name": "Tumor",
                                "nuclei_types": {
                                    "101": {"stored_count": 3, "mean_area": 9.0}
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            masks = (
                np.ones((3, 3), dtype=bool),
                np.ones((3, 3), dtype=bool),
                np.pad(np.ones((2, 2), dtype=bool), ((0, 1), (0, 1))),
            )
            for index, mask in enumerate(masks, start=1):
                np.savez(
                    bucket / f"{index:06d}.npz",
                    mask=mask,
                    type=np.asarray(101),
                    area=np.asarray(int(mask.sum())),
                )

            authority = load_reference_shape_authority(
                root,
                dataset_name="BCSS",
                class_ids=(1,),
                shapes_per_class=3,
            )

            self.assertEqual(len(authority.shapes_by_class[1]), 2)

    def test_semantic_fallback_splits_touching_same_class_nuclei(self):
        rows, cols = np.ogrid[:40, :40]
        nuclei = np.zeros((40, 40), dtype=np.uint8)
        touching = ((rows - 20) ** 2 + (cols - 15) ** 2 <= 6**2) | (
            (rows - 20) ** 2 + (cols - 25) ** 2 <= 6**2
        )
        nuclei[touching] = 1
        instances = tuple(iter_instances(nuclei))
        self.assertEqual(len(instances), 2)
        self.assertEqual(
            sum(int(mask.sum()) for _, _, mask in instances), int(touching.sum())
        )

    def test_semantic_fallback_keeps_unseeded_disconnected_small_nuclei(self):
        nuclei = np.zeros((48, 48), dtype=np.uint8)
        for row, col in ((3, 3), (3, 12), (12, 3), (12, 12)):
            nuclei[row:row + 2, col:col + 2] = 1
        nuclei[25:30, 25:30] = 1

        instances = tuple(iter_instances(nuclei))

        self.assertEqual(len(instances), 5)
        self.assertEqual(
            sum(int(mask.sum()) for _, _, mask in instances),
            int(np.count_nonzero(nuclei)),
        )


class JointProfileStatisticsTests(unittest.TestCase):
    def test_statistics_bind_patient_wsi_digest_and_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[4:28, 4:28] = 12
            path = root / "mask.npy"
            np.save(path, mask, allow_pickle=False)
            result = build_annotation_profile_statistics(
                [
                    {
                        "case_id": "case-1",
                        "patient_id": "patient-1",
                        "wsi_id": "wsi-1",
                        "split": "train",
                        "mask_uri": str(path),
                        "mask_sha256": _sha(path),
                    }
                ],
                annotation_profile_id="glas-gland-v1",
                data_revision="fixture-v1",
                evidence_manifest_sha256="fixture-manifest-digest",
            )
            self.assertEqual(result["sample_counts"]["patients"], 1)
            self.assertEqual(result["split_leakage_audit"]["wsi_leakage_count"], 0)
            self.assertIn("background_fragmentation", result)
            self.assertEqual(result["review_status"], "draft")


class _ApprovingJointCritic(DeterministicJointResearchCritic):
    supports_pathology_vision = True

    def review(self, *, case, bundle, candidates, gate_reports, image_paths, artifact_registry=None):
        del artifact_registry
        del case, gate_reports, image_paths
        candidate = candidates[0]
        return JointCriticResult(
            rankings=(
                JointCriticRanking(
                    candidate_id=candidate.candidate_id,
                    score=0.95,
                    confidence=0.95,
                    supporting_rule_ids=bundle.active_rule_ids,
                ),
            ),
            abstain=False,
            summary="fixture joint visual approval",
            usage={"provider": "fixture"},
        )


class _RecordingJointCriticClient:
    def __init__(self):
        self.calls = []

    def call(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "rankings": [],
            "abstain": True,
            "summary": "certified_mask_condition_abstained",
        }, {"model": "fixture"}


class _CertifiedTissueSelectionClient:
    def __init__(self, mutation=None, *, candidate_index=0):
        self.mutation = mutation
        self.candidate_index = candidate_index
        self.calls = []

    def call(self, **kwargs):
        self.calls.append(kwargs)
        payload = json.loads(kwargs["user_prompt"])
        candidate = payload["certified_tissue_plan_candidates"][
            self.candidate_index
        ]
        preferences = payload["joint_mechanism_contract"]["planner_policy"][
            "selection_preferences"
        ]
        response = {
            "abstain": False,
            "abstain_reason": None,
            "decision_id": "select_certified_tissue_plan_candidate",
            "selected_candidate_id": candidate["candidate_id"],
            "selected_tool_family": candidate["allowed_tool_families"][0],
            "supporting_preference_rule_ids": [preferences[0]],
            "selection_explanation": "certified_tissue_candidate_selected",
            "confidence": 0.83,
        }
        if self.mutation is not None:
            response = self.mutation(response)
        return response, {"model": "fixture-mask-planner"}


class _CertifiedCellSelectionClient:
    def __init__(self, mutation=None, *, candidate_index=0, usage=None):
        self.mutation = mutation
        self.candidate_index = candidate_index
        self.usage = (
            dict(usage)
            if usage is not None
            else {"model": "fixture-cell-planner"}
        )
        self.calls = []

    def call(self, **kwargs):
        self.calls.append(kwargs)
        payload = json.loads(kwargs["user_prompt"])
        candidate = payload["certified_cell_plan_candidates"][
            self.candidate_index
        ]
        preferences = payload["planner_policy"]["selection_preferences"]
        response = {
            "abstain": False,
            "abstain_reason": None,
            "decision_id": "select_certified_cell_plan_candidate",
            "selected_candidate_id": candidate["candidate_id"],
            "selected_tool_program_id": candidate[
                "allowed_tool_program_ids"
            ][0],
            "supporting_preference_rule_ids": [preferences[0]],
            "selection_explanation": "certified_cell_candidate_selected",
            "confidence": 0.8,
        }
        if self.mutation is not None:
            response = self.mutation(response)
        return response, dict(self.usage)


class _PassingTissueGateFixture(GateRegistry):
    def run(self, context):
        report = super().run(context)
        return GateReport(report.candidate_id, True, report.checks)


class _PassingJointGateFixture(JointGateRegistry):
    """Isolate workflow orchestration tests from mechanism gate semantics."""

    def run(self, context):
        report = super().run(context)
        return JointGateReport(
            report.candidate_id,
            True,
            tuple(
                replace(
                    check,
                    passed=True,
                    detail="fixture isolates workflow orchestration",
                )
                for check in report.checks
            ),
        )


class _RetryThenPassingTissueGate(GateRegistry):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def run(self, context):
        self.calls += 1
        report = super().run(context)
        runtime_bound = bool(
            context.candidate.tool_trace.get("joint_tissue_tool_program")
        )
        if runtime_bound:
            runtime_calls = getattr(self, "runtime_calls", 0) + 1
            self.runtime_calls = runtime_calls
            # One early replay plus the 12-variant fallback make the first
            # runtime execution batch. Pre-LLM compiler executions lack this
            # runtime trace binding and remain honestly certified.
            passed = runtime_calls > 13
        else:
            passed = report.passed
        return GateReport(
            report.candidate_id,
            passed,
            report.checks,
        )


class _AlwaysFailingCandidateCellExecutor:
    def __init__(self):
        self.calls = 0

    @staticmethod
    def supports(contract):
        del contract
        return True

    def execute(self, **kwargs):
        del kwargs
        self.calls += 1
        raise JointContractError("fixture candidate-local executor failure")


class JointWorkflowTests(unittest.TestCase):
    def test_joint_llm_plan_rejects_unknown_ids_missing_rules_and_numeric_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = _as_breast_growth_case(_write_synthetic_case(root))
            case = replace(
                case,
                case_id="joint-plan-adversarial",
                instruction="cohesive-boundary-expansion-v1",
                primitive_id="cohesive-boundary-expansion-v1",
                joint_area_budget=JointAreaBudget(
                    target_fraction=0.12,
                    min_fraction=0.10,
                    max_fraction=0.14,
                    tissue_min_fraction=0.10,
                ),
                provenance={
                    **case.provenance,
                    "joint_mechanism_id": (
                        "breast-annotation-anchored-boundary-growth"
                    ),
                    "joint_primitive_id": "cohesive-boundary-expansion-v1",
                },
            )
            workflow = JointPathologyEditWorkflow(
                tissue_planner=MultiInterfaceResearchTissuePlanner(),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
            )
            result = workflow.run(case, output_root=root / "baseline")
            self.assertEqual(result.status, "selected_research")
            heuristic = result.joint_plan
            schema = workflow.mask_skills.annotation_schema(
                case.annotation_profile_id
            )
            scene = build_joint_scene_analysis(
                np.load(case.source_tissue_mask_uri),
                np.asarray(Image.open(case.source_nuclei_mask_uri)),
                schema=schema,
                pixel_size_um=case.pixel_size_um,
                nuclei_instances_path=case.source_nuclei_instances_uri,
            )
            bundle = workflow.joint_skills.compose(
                case=case,
                mechanism_id=heuristic.selected_mechanism_id,
                available_checker_ids=workflow.joint_gates.available_checker_ids,
                production=False,
            )
            cell = heuristic.cell_plan
            raw = {
                "abstain": False,
                "abstain_reason": None,
                "selected_mechanism_id": heuristic.selected_mechanism_id,
                "decision_ids": [
                    "select_certified_interface_anchor_ids",
                    "select_allowed_tool_program",
                ],
                "supporting_observations": ["certified_mask_graph_inputs"],
                "supporting_rule_ids": list(bundle.active_rule_ids),
                "supporting_preference_rule_ids": list(
                    bundle.mechanism.planner_policy.selection_preferences
                ),
                "representability_confidence": 0.8,
                "tissue_plan_accepted": True,
                "bound_interface_ids": list(cell.interface_ids),
                "structural_unit_ids": list(heuristic.structural_unit_ids),
                "cell_plan": {
                    "core_zone": cell.core_zone,
                    "halo_zone": cell.halo_zone,
                    "actions": list(cell.actions),
                    "allowed_cell_classes": list(cell.allowed_cell_classes),
                    "layout_program_id": cell.layout_program_id,
                    "anchor_ids": list(cell.anchor_ids),
                    "spatial_anchor_type": cell.spatial_anchor_type,
                    "spatial_anchor_observation": cell.spatial_anchor_observation,
                    "baseline_mode": cell.baseline_mode,
                    "mechanism_program_id": cell.mechanism_program_id,
                    "mechanism_quota_role": cell.mechanism_quota_role,
                    "supporting_rule_ids": list(cell.supporting_rule_ids),
                    "expected_morphology": "compiler_owned_render_expectations",
                },
                "coupling_plan": {
                    "compatibility_rule_ids": list(
                        heuristic.coupling_plan.compatibility_rule_ids
                    )
                },
                "uncertainties": [],
                "escalation_reason": None,
            }
            parser = OpenAIMultimodalJointPlanner(client=SimpleNamespace())
            parsed = parser._parse_plan(
                raw=raw,
                case=case,
                scene=scene,
                bundle=bundle,
                tissue_plan=heuristic.tissue_plan,
            )
            self.assertEqual(
                parsed.supporting_preference_rule_ids,
                tuple(raw["supporting_preference_rule_ids"]),
            )
            free_text_variants = (
                "desmoplastic reaction is present",
                "the invasive front is histologically evident",
                "fibrotic stroma marks a treatment bed",
                "this is luminal A disease",
                "lymphovascular invasion is present",
                "residual cancer burden is low",
            )
            for value in free_text_variants:
                for field, payload in (
                    ("supporting_observations", [value]),
                    ("uncertainties", [value]),
                    ("escalation_reason", value),
                ):
                    with self.subTest(field=field, value=value), self.assertRaises(
                        JointContractError
                    ):
                        parser._parse_plan(
                            raw={**raw, field: payload},
                            case=case,
                            scene=scene,
                            bundle=bundle,
                            tissue_plan=heuristic.tissue_plan,
                        )
                with self.subTest(
                    field="expected_morphology", value=value
                ), self.assertRaises(JointContractError):
                    parser._parse_plan(
                        raw={
                            **raw,
                            "cell_plan": {
                                **raw["cell_plan"],
                                "expected_morphology": value,
                            },
                        },
                        case=case,
                        scene=scene,
                        bundle=bundle,
                        tissue_plan=heuristic.tissue_plan,
                    )
            adversaries = {
                "vetoed interface": {
                    **raw,
                    "bound_interface_ids": ["if:forged"],
                },
                "unknown anchor": {
                    **raw,
                    "cell_plan": {**raw["cell_plan"], "anchor_ids": ["anchor:forged"]},
                },
                "missing required rule": {
                    **raw,
                    "supporting_rule_ids": [
                        rule_id
                        for rule_id in raw["supporting_rule_ids"]
                        if rule_id
                        not in heuristic.coupling_plan.compatibility_rule_ids
                    ],
                },
                "illegal decision": {
                    **raw,
                    "decision_ids": ["abstain"],
                },
                "numeric geometry": {
                    **raw,
                    "cell_plan": {**raw["cell_plan"], "nucleus_count": 9},
                },
            }
            for label, payload in adversaries.items():
                with self.subTest(label=label), self.assertRaises(
                    JointContractError
                ):
                    parser._parse_plan(
                        raw=payload,
                        case=case,
                        scene=scene,
                        bundle=bundle,
                        tissue_plan=heuristic.tissue_plan,
                    )

    def test_online_tissue_planner_only_selects_compiler_certified_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = _as_breast_growth_case(_write_synthetic_case(root))
            case = replace(
                case,
                case_id="online-certified-selection",
                instruction="cohesive-boundary-expansion-v1",
                primitive_id="cohesive-boundary-expansion-v1",
                joint_area_budget=JointAreaBudget(
                    target_fraction=0.12,
                    min_fraction=0.10,
                    max_fraction=0.14,
                    tissue_min_fraction=0.10,
                ),
                provenance={
                    **case.provenance,
                    "joint_mechanism_id": (
                        "breast-annotation-anchored-boundary-growth"
                    ),
                    "joint_primitive_id": "cohesive-boundary-expansion-v1",
                },
            )
            client = _CertifiedTissueSelectionClient()
            result = JointPathologyEditWorkflow(
                tissue_planner=OpenAIJointAwareTissuePlanner(client=client),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
            ).run(case, output_root=root / "result")
            self.assertEqual(
                result.status, "selected_research", result.abstain_reasons
            )
            certificate = result.joint_plan.tissue_plan.tool_program.parameter_ranges[
                "planner_selection_certificate"
            ]
            self.assertEqual(
                certificate["decision_id"],
                "select_certified_tissue_plan_candidate",
            )
            self.assertTrue(certificate["selected_candidate_id"])
            schema = client.calls[0]["json_schema"]
            self.assertNotIn("plan", schema["properties"])

    def test_online_tissue_planner_receives_real_multi_candidate_portfolio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = _as_breast_growth_case(_write_synthetic_case(root))
            tissue_path = Path(case.source_tissue_mask_uri)
            nuclei_path = Path(case.source_nuclei_mask_uri)
            instances_path = Path(case.source_nuclei_instances_uri)
            rows, cols = np.ogrid[:128, :128]
            tissue = np.full((128, 128), 2, dtype=np.uint8)
            # Two independently executable invasive-tumor components provide
            # two real interface/anchor choices. Every advertised survivor is
            # later selected and executed in the regression below.
            tumor = (
                ((rows - 38) ** 2 + (cols - 38) ** 2 <= 25**2)
                | ((rows - 90) ** 2 + (cols - 90) ** 2 <= 25**2)
            )
            tissue[tumor] = 1
            np.save(tissue_path, tissue, allow_pickle=False)
            nuclei = np.asarray(Image.open(nuclei_path)).copy()
            payload = json.loads(instances_path.read_text(encoding="utf-8"))
            for item in payload["nuc"].values():
                row, col = map(int, item["centroid"])
                class_id = 1 if tissue[row, col] == 1 else 3
                contour = np.asarray(item["contour"], dtype=int)
                x0, y0 = contour.min(axis=0)
                x1, y1 = contour.max(axis=0)
                nuclei[y0 : y1 + 1, x0 : x1 + 1] = class_id
                item["type"] = class_id
            Image.fromarray(nuclei).save(nuclei_path)
            instances_path.write_text(json.dumps(payload), encoding="utf-8")
            case = replace(
                case,
                case_id="online-multi-candidate",
                primitive_id="cohesive-boundary-expansion-v1",
                instruction="cohesive-boundary-expansion-v1",
                joint_area_budget=JointAreaBudget(
                    target_fraction=0.08,
                    min_fraction=0.06,
                    max_fraction=0.10,
                    tissue_min_fraction=0.06,
                ),
                provenance={
                    **case.provenance,
                    "source_tissue_mask_sha256": _sha(tissue_path),
                    "source_nuclei_mask_sha256": _sha(nuclei_path),
                    "source_nuclei_instances_sha256": _sha(instances_path),
                    "original_label_map_digest": _sha(tissue_path),
                    "original_instance_mask_digest": _sha(instances_path),
                    "joint_mechanism_id": (
                        "breast-annotation-anchored-boundary-growth"
                    ),
                    "joint_primitive_id": "cohesive-boundary-expansion-v1",
                },
            )
            tissue_auxiliary_path = root / "tissue-authority-auxiliary.png"
            Image.fromarray(np.zeros((128, 128), dtype=np.uint8)).save(
                tissue_auxiliary_path
            )
            tissue_auxiliary_digest = _sha(tissue_auxiliary_path)
            case = replace(
                case,
                auxiliary_structure_uris={
                    "authority_probe": str(tissue_auxiliary_path)
                },
                provenance={
                    **case.provenance,
                    "auxiliary_structure_sha256": {
                        "authority_probe": tissue_auxiliary_digest
                    },
                    "auxiliary_structure_provenance": {
                        "authority_probe": {
                            "producer_id": "synthetic-authority-probe",
                            "producer_version": "synthetic-authority-probe-v1",
                            "source_tissue_mask_sha256": case.provenance[
                                "source_tissue_mask_sha256"
                            ],
                            "output_sha256": tissue_auxiliary_digest,
                        }
                    },
                },
            )
            client = _CertifiedTissueSelectionClient()
            result = JointPathologyEditWorkflow(
                tissue_planner=OpenAIJointAwareTissuePlanner(client=client),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
            ).run(case, output_root=root / "multi")
            self.assertEqual(
                result.status, "selected_research", result.abstain_reasons
            )
            payload = json.loads(client.calls[0]["user_prompt"])
            candidates = payload["certified_tissue_plan_candidates"]
            portfolio_path = (
                root
                / "multi"
                / case.case_id
                / "candidate_feasibility_portfolio.json"
            )
            prepared = json.loads(portfolio_path.read_text(encoding="utf-8"))
            prepared_survivors = prepared["surviving_candidates"]
            self.assertGreaterEqual(len(prepared_survivors), 2)
            self.assertGreaterEqual(len(candidates), 2)
            for candidate in candidates:
                self.assertTrue(candidate["selected_interface_ids"])
                self.assertTrue(candidate["selected_anchor_ids"])
                self.assertTrue(candidate["compiler_certificate_sha256"])
                self.assertTrue(candidate["tool_program_sha256"])
                self.assertFalse(candidate["veto_reasons"])
                self.assertTrue(candidate["deterministic_candidate_metrics"])

            authority_workflow = JointPathologyEditWorkflow(
                tissue_planner=MultiInterfaceResearchTissuePlanner(),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
            )
            authority_scene = build_joint_scene_analysis(
                tissue,
                nuclei,
                schema=MaskProfileSchema.from_reference_profile("BCSS"),
                pixel_size_um=case.pixel_size_um,
                nuclei_instances_path=case.source_nuclei_instances_uri,
                auxiliary_structure_paths=case.auxiliary_structure_uris,
                auxiliary_structure_provenance=case.provenance[
                    "auxiliary_structure_provenance"
                ],
            )
            prepared_objects, _ = authority_workflow._prepare_interpretations(
                case=case,
                source_tissue=tissue,
                schema=MaskProfileSchema.from_reference_profile("BCSS"),
                scene=authority_scene,
            )
            prepared_object = next(iter(prepared_objects.values()))
            object_portfolio = prepared_object.tissue_feasibility_portfolio
            forged = replace(
                object_portfolio.survivors[0],
                deterministic_candidate_metrics={
                    **object_portfolio.survivors[
                        0
                    ].deterministic_candidate_metrics,
                    "protected_distance_px": 999999.0,
                },
            )
            forged_portfolio = replace(
                object_portfolio,
                survivors=(forged,),
            )
            tissue_case = _as_tissue_case(
                prepared_object.case,
                allocation=prepared_object.allocation,
                shape=tissue.shape,
            )
            tissue_scene = augment_tissue_scene_with_nuclei_preflight(
                authority_scene.tissue,
                prepared_object.nuclei_preflight,
                auxiliary_structure_masks=(
                    authority_scene.auxiliary_structure_masks
                ),
                required_auxiliary_structure_ids=(
                    prepared_object.bundle.mechanism.representability.protected_auxiliary_structures
                ),
            )
            with self.assertRaisesRegex(
                (JointContractError, RefineContractError),
                "compiler-issued capability",
            ):
                OpenAIJointAwareTissuePlanner(
                    client=_CertifiedTissueSelectionClient(),
                    max_contract_attempts=1,
                ).create_joint_tissue_plan(
                    case=tissue_case,
                    scene=tissue_scene,
                    bundle=prepared_object.tissue_bundle,
                    joint_bundle=prepared_object.bundle,
                    image_paths=(),
                    nuclei_preflight=prepared_object.nuclei_preflight,
                    joint_case=prepared_object.case,
                    allocation=prepared_object.allocation,
                    candidate_portfolio=forged_portfolio,
                )

            # A valid portfolio is bound to the exact source raster and
            # nuclei-preflight witness. Reusing it after source mutation or
            # with a different tissue budget must fail before the LLM call.
            original_source = tissue_path.read_bytes()
            changed_source = tissue.copy()
            changed_source[0, 0] = 1
            np.save(tissue_path, changed_source, allow_pickle=False)
            with self.assertRaisesRegex(
                (JointContractError, RefineContractError), "detached"
            ):
                OpenAIJointAwareTissuePlanner(
                    client=_CertifiedTissueSelectionClient(),
                    max_contract_attempts=1,
                ).create_joint_tissue_plan(
                    case=tissue_case,
                    scene=tissue_scene,
                    bundle=prepared_object.tissue_bundle,
                    joint_bundle=prepared_object.bundle,
                    image_paths=(),
                    nuclei_preflight=prepared_object.nuclei_preflight,
                    joint_case=prepared_object.case,
                    allocation=prepared_object.allocation,
                    candidate_portfolio=object_portfolio,
                )
            tissue_path.write_bytes(original_source)

            # Direct tissue-Planner callers must re-read the current nuclei
            # raster and native-instance authority before any LLM call.  A
            # stale preflight or unchanged provenance cannot authorize live
            # bytes that changed after portfolio compilation.
            original_nuclei = nuclei_path.read_bytes()
            changed_nuclei = np.asarray(Image.open(nuclei_path)).copy()
            changed_nuclei[0, 0] = 1
            Image.fromarray(changed_nuclei).save(nuclei_path)
            mutated_nuclei_client = _CertifiedTissueSelectionClient()
            with self.assertRaisesRegex(
                (JointContractError, RefineContractError), "detached"
            ):
                OpenAIJointAwareTissuePlanner(
                    client=mutated_nuclei_client,
                    max_contract_attempts=1,
                ).create_joint_tissue_plan(
                    case=tissue_case,
                    scene=tissue_scene,
                    bundle=prepared_object.tissue_bundle,
                    joint_bundle=prepared_object.bundle,
                    image_paths=(),
                    nuclei_preflight=prepared_object.nuclei_preflight,
                    joint_case=prepared_object.case,
                    allocation=prepared_object.allocation,
                    candidate_portfolio=object_portfolio,
                )
            self.assertFalse(mutated_nuclei_client.calls)
            nuclei_path.write_bytes(original_nuclei)

            original_instances = instances_path.read_bytes()
            instances_path.write_bytes(original_instances + b"\n")
            mutated_instance_client = _CertifiedTissueSelectionClient()
            with self.assertRaisesRegex(
                (JointContractError, RefineContractError), "detached"
            ):
                OpenAIJointAwareTissuePlanner(
                    client=mutated_instance_client,
                    max_contract_attempts=1,
                ).create_joint_tissue_plan(
                    case=tissue_case,
                    scene=tissue_scene,
                    bundle=prepared_object.tissue_bundle,
                    joint_bundle=prepared_object.bundle,
                    image_paths=(),
                    nuclei_preflight=prepared_object.nuclei_preflight,
                    joint_case=prepared_object.case,
                    allocation=prepared_object.allocation,
                    candidate_portfolio=object_portfolio,
                )
            self.assertFalse(mutated_instance_client.calls)
            instances_path.write_bytes(original_instances)

            original_auxiliary = tissue_auxiliary_path.read_bytes()
            Image.fromarray(np.ones((128, 128), dtype=np.uint8)).save(
                tissue_auxiliary_path
            )
            mutated_auxiliary_client = _CertifiedTissueSelectionClient()
            with self.assertRaisesRegex(
                (JointContractError, RefineContractError), "detached"
            ):
                OpenAIJointAwareTissuePlanner(
                    client=mutated_auxiliary_client,
                    max_contract_attempts=1,
                ).create_joint_tissue_plan(
                    case=tissue_case,
                    scene=tissue_scene,
                    bundle=prepared_object.tissue_bundle,
                    joint_bundle=prepared_object.bundle,
                    image_paths=(),
                    nuclei_preflight=prepared_object.nuclei_preflight,
                    joint_case=prepared_object.case,
                    allocation=prepared_object.allocation,
                    candidate_portfolio=object_portfolio,
                )
            self.assertFalse(mutated_auxiliary_client.calls)
            tissue_auxiliary_path.write_bytes(original_auxiliary)
            changed_tissue_case = replace(
                tissue_case,
                area_budget=AreaBudget(
                    target_fraction=tissue_case.area_budget.target_fraction,
                    min_fraction=max(
                        0.0, tissue_case.area_budget.min_fraction - 0.001
                    ),
                    max_fraction=tissue_case.area_budget.max_fraction,
                    basis=tissue_case.area_budget.basis,
                    relative_tolerance=(
                        tissue_case.area_budget.relative_tolerance
                    ),
                    fallback_policy=tissue_case.area_budget.fallback_policy,
                ),
            )
            with self.assertRaisesRegex(
                (JointContractError, RefineContractError), "detached"
            ):
                OpenAIJointAwareTissuePlanner(
                    client=_CertifiedTissueSelectionClient(),
                    max_contract_attempts=1,
                ).create_joint_tissue_plan(
                    case=changed_tissue_case,
                    scene=tissue_scene,
                    bundle=prepared_object.tissue_bundle,
                    joint_bundle=prepared_object.bundle,
                    image_paths=(),
                    nuclei_preflight=prepared_object.nuclei_preflight,
                    joint_case=prepared_object.case,
                    allocation=prepared_object.allocation,
                    candidate_portfolio=object_portfolio,
                )

            stale_tissue_authorities = (
                (
                    replace(
                        prepared_object.bundle,
                        mechanism=replace(
                            prepared_object.bundle.mechanism,
                            version=(
                                prepared_object.bundle.mechanism.version
                                + "-mutated"
                            ),
                        ),
                    ),
                    prepared_object.tissue_bundle,
                    prepared_object.case,
                ),
                (
                    replace(
                        prepared_object.bundle,
                        active_rule_ids=(
                            prepared_object.bundle.active_rule_ids
                            + ("rule:mutated",)
                        ),
                    ),
                    prepared_object.tissue_bundle,
                    prepared_object.case,
                ),
                (
                    prepared_object.bundle,
                    replace(
                        prepared_object.tissue_bundle,
                        warnings=(
                            prepared_object.tissue_bundle.warnings
                            + ("mutated",)
                        ),
                    ),
                    prepared_object.case,
                ),
                (
                    prepared_object.bundle,
                    prepared_object.tissue_bundle,
                    replace(
                        prepared_object.case,
                        provenance={
                            key: value
                            for key, value in prepared_object.case.provenance.items()
                            if key != "original_label_map_digest"
                        },
                    ),
                ),
            )
            for stale_bundle, stale_tissue_bundle, stale_joint_case in (
                stale_tissue_authorities
            ):
                stale_client = _CertifiedTissueSelectionClient()
                with self.assertRaisesRegex(
                    (JointContractError, RefineContractError), "detached"
                ):
                    OpenAIJointAwareTissuePlanner(
                        client=stale_client,
                        max_contract_attempts=1,
                    ).create_joint_tissue_plan(
                        case=tissue_case,
                        scene=tissue_scene,
                        bundle=stale_tissue_bundle,
                        joint_bundle=stale_bundle,
                        image_paths=(),
                        nuclei_preflight=prepared_object.nuclei_preflight,
                        joint_case=stale_joint_case,
                        allocation=prepared_object.allocation,
                        candidate_portfolio=object_portfolio,
                    )
                self.assertFalse(stale_client.calls)

            # Every exposed candidate/tool pair must remain executable when it
            # is the LLM selection, including the second survivor that exposed
            # the prior false-certificate bug.
            for index, candidate in enumerate(candidates):
                for family in candidate["allowed_tool_families"]:
                    with self.subTest(index=index, family=family):
                        replay_client = _CertifiedTissueSelectionClient(
                            candidate_index=index,
                        )
                        replay = JointPathologyEditWorkflow(
                            tissue_planner=OpenAIJointAwareTissuePlanner(
                                client=replay_client
                            ),
                            joint_planner=HeuristicJointPlanner(),
                            critic=_ApprovingJointCritic(),
                        ).run(
                            case,
                            output_root=(
                                root / f"survivor-{index}-{family}"
                            ),
                        )
                        self.assertEqual(
                            replay.status,
                            "selected_research",
                            replay.abstain_reasons,
                        )

    def test_cell_candidate_certificate_rejects_detached_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _as_breast_growth_case(_write_synthetic_case(root))
            case = replace(
                source,
                case_id="cell-certificate-adversarial",
                instruction="increase tumor cell abundance",
                primitive_id="neoplastic-cell-abundance-increase-v1",
                joint_area_budget=None,
                cell_count_extent_budget=CellCountExtentBudget(
                    12, 12, 15, 96, 0, 128,
                    minimum_effect_span_px=72,
                    minimum_effect_foci=2,
                ),
                provenance={
                    **source.provenance,
                    "joint_mechanism_id": "breast-local-population-modulation",
                    "joint_primitive_id": (
                        "neoplastic-cell-abundance-increase-v1"
                    ),
                    "target_cell_class_ids": [1],
                },
            )
            auxiliary_path = root / "cell-authority-auxiliary.png"
            Image.fromarray(np.zeros((128, 128), dtype=np.uint8)).save(
                auxiliary_path
            )
            auxiliary_digest = _sha(auxiliary_path)
            case = replace(
                case,
                auxiliary_structure_uris={
                    "authority_probe": str(auxiliary_path)
                },
                provenance={
                    **case.provenance,
                    "auxiliary_structure_sha256": {
                        "authority_probe": auxiliary_digest
                    },
                    "auxiliary_structure_provenance": {
                        "authority_probe": {
                            "producer_id": "synthetic-authority-probe",
                            "producer_version": "synthetic-authority-probe-v1",
                            "source_tissue_mask_sha256": case.provenance[
                                "source_tissue_mask_sha256"
                            ],
                            "output_sha256": auxiliary_digest,
                        }
                    },
                },
            )
            workflow = JointPathologyEditWorkflow(
                tissue_planner=MultiInterfaceResearchTissuePlanner(),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
            )
            source_tissue = np.load(case.source_tissue_mask_uri)
            source_nuclei = np.asarray(Image.open(case.source_nuclei_mask_uri))
            schema = workflow.mask_skills.annotation_schema(
                case.annotation_profile_id
            )
            scene = build_joint_scene_analysis(
                source_tissue,
                source_nuclei,
                schema=schema,
                pixel_size_um=case.pixel_size_um,
                nuclei_instances_path=case.source_nuclei_instances_uri,
                auxiliary_structure_paths=case.auxiliary_structure_uris,
                auxiliary_structure_provenance=case.provenance[
                    "auxiliary_structure_provenance"
                ],
            )
            bundle = workflow.joint_skills.compose(
                case=case,
                mechanism_id="breast-local-population-modulation",
                available_checker_ids=workflow.joint_gates.available_checker_ids,
                production=False,
            )
            choices = workflow._compile_cell_only_candidate_portfolio(
                case=case,
                source_tissue=source_tissue,
                source_nuclei=source_nuclei,
                schema=schema,
                scene=scene,
                bundle=bundle,
            )
            self.assertTrue(choices.choices)
            self.assertEqual(
                choices.certificates.survivors,
                tuple(item.certificate for item in choices.choices),
            )
            candidate = choices.choices[0].certificate
            client = _CertifiedCellSelectionClient()
            selected, usage = OpenAIMultimodalJointPlanner(
                client=client,
                max_contract_attempts=1,
            ).create_plan(
                case=case,
                scene=scene,
                bundle=bundle,
                tissue_plan=None,
                image_paths=(),
                candidate_portfolio=choices.certificates,
            )
            self.assertEqual(selected, candidate.plan)
            self.assertEqual(
                usage["selected_candidate_id"], candidate.candidate_id
            )
            adversaries = {
                "unknown": _CertifiedCellSelectionClient(
                    lambda value: {
                        **value,
                        "selected_candidate_id": "cell-plan:forged",
                    }
                ),
                "tool": _CertifiedCellSelectionClient(
                    lambda value: {
                        **value,
                        "selected_tool_program_id": "forged-program",
                    }
                ),
            }
            for label, bad_client in adversaries.items():
                with self.subTest(label=label), self.assertRaises(
                    JointContractError
                ):
                    OpenAIMultimodalJointPlanner(
                        client=bad_client,
                        max_contract_attempts=1,
                    ).create_plan(
                        case=case,
                        scene=scene,
                        bundle=bundle,
                        tissue_plan=None,
                        image_paths=(),
                        candidate_portfolio=choices.certificates,
                    )
            for value in (
                "desmoplastic reaction is present",
                "the invasive front is histologically evident",
                "fibrotic stroma marks a treatment bed",
                "this is luminal A disease",
                "lymphovascular invasion is present",
                "residual cancer burden is low",
            ):
                claiming_client = _CertifiedCellSelectionClient(
                    lambda response, value=value: {
                        **response,
                        "selection_explanation": value,
                    }
                )
                with self.subTest(value=value), self.assertRaisesRegex(
                    JointContractError, "neutral audit token"
                ):
                    OpenAIMultimodalJointPlanner(
                        client=claiming_client,
                        max_contract_attempts=1,
                    ).create_plan(
                        case=case,
                        scene=scene,
                        bundle=bundle,
                        tissue_plan=None,
                        image_paths=(),
                        candidate_portfolio=choices.certificates,
                    )
            detached = replace(
                candidate,
                compiler_certificate_sha256="0" * 64,
            )
            detached_portfolio = replace(
                choices.certificates,
                survivors=(detached,),
            )
            with self.assertRaisesRegex(
                JointContractError, "compiler-issued capability"
            ):
                OpenAIMultimodalJointPlanner(
                    client=client,
                    max_contract_attempts=1,
                ).create_plan(
                    case=case,
                    scene=scene,
                    bundle=bundle,
                    tissue_plan=None,
                    image_paths=(),
                    candidate_portfolio=detached_portfolio,
                )

            # Public callers cannot self-sign a new certificate or portfolio,
            # even if they reproduce the ordinary hash algorithm after
            # modifying a metric.
            forged_metric_candidate = replace(
                candidate,
                deterministic_candidate_metrics={
                    **candidate.deterministic_candidate_metrics,
                    "protected_distance_px": 999999.0,
                },
            )
            forged_portfolio = replace(
                choices.certificates,
                survivors=(forged_metric_candidate,),
            )
            with self.assertRaisesRegex(
                JointContractError, "compiler-issued capability"
            ):
                OpenAIMultimodalJointPlanner(
                    client=client,
                    max_contract_attempts=1,
                ).create_plan(
                    case=case,
                    scene=scene,
                    bundle=bundle,
                    tissue_plan=None,
                    image_paths=(),
                    candidate_portfolio=forged_portfolio,
                )

            changed_budget = replace(
                case,
                cell_count_extent_budget=CellCountExtentBudget(
                    10,
                    10,
                    12,
                    96,
                    0,
                    128,
                    minimum_effect_span_px=72,
                    minimum_effect_foci=2,
                ),
            )
            with self.assertRaisesRegex(JointContractError, "detached"):
                OpenAIMultimodalJointPlanner(
                    client=client,
                    max_contract_attempts=1,
                ).create_plan(
                    case=changed_budget,
                    scene=scene,
                    bundle=bundle,
                    tissue_plan=None,
                    image_paths=(),
                    candidate_portfolio=choices.certificates,
                )

            missing_source_digest = replace(
                case,
                provenance={
                    key: value
                    for key, value in case.provenance.items()
                    if key != "original_instance_mask_digest"
                },
            )
            missing_digest_client = _CertifiedCellSelectionClient()
            with self.assertRaisesRegex(JointContractError, "detached"):
                OpenAIMultimodalJointPlanner(
                    client=missing_digest_client,
                    max_contract_attempts=1,
                ).create_plan(
                    case=missing_source_digest,
                    scene=scene,
                    bundle=bundle,
                    tissue_plan=None,
                    image_paths=(),
                    candidate_portfolio=choices.certificates,
                )
            self.assertFalse(missing_digest_client.calls)

            stale_bundles = (
                replace(
                    bundle,
                    mechanism=replace(
                        bundle.mechanism,
                        version=bundle.mechanism.version + "-mutated",
                    ),
                ),
                replace(
                    bundle,
                    active_rule_ids=bundle.active_rule_ids + ("rule:mutated",),
                ),
            )
            for stale_bundle in stale_bundles:
                stale_client = _CertifiedCellSelectionClient()
                with self.assertRaisesRegex(JointContractError, "detached"):
                    OpenAIMultimodalJointPlanner(
                        client=stale_client,
                        max_contract_attempts=1,
                    ).create_plan(
                        case=case,
                        scene=scene,
                        bundle=stale_bundle,
                        tissue_plan=None,
                        image_paths=(),
                        candidate_portfolio=choices.certificates,
                    )
                self.assertFalse(stale_client.calls)

            # The cell direct-Planner boundary independently re-hashes every
            # live authority that can affect the scene or capacity witness.
            live_mutations = (
                (
                    "nuclei",
                    Path(case.source_nuclei_mask_uri),
                    lambda path: Image.fromarray(
                        np.where(
                            np.indices((128, 128))[0] == 0,
                            1,
                            np.asarray(Image.open(path)),
                        ).astype(np.uint8)
                    ).save(path),
                ),
                (
                    "instances",
                    Path(case.source_nuclei_instances_uri),
                    lambda path: path.write_bytes(path.read_bytes() + b"\n"),
                ),
                (
                    "auxiliary",
                    auxiliary_path,
                    lambda path: Image.fromarray(
                        np.ones((128, 128), dtype=np.uint8)
                    ).save(path),
                ),
            )
            for label, path, mutate in live_mutations:
                original = path.read_bytes()
                mutate(path)
                mutated_client = _CertifiedCellSelectionClient()
                with self.subTest(live_asset=label), self.assertRaisesRegex(
                    JointContractError, "detached"
                ):
                    OpenAIMultimodalJointPlanner(
                        client=mutated_client,
                        max_contract_attempts=1,
                    ).create_plan(
                        case=case,
                        scene=scene,
                        bundle=bundle,
                        tissue_plan=None,
                        image_paths=(),
                        candidate_portfolio=choices.certificates,
                    )
                self.assertFalse(mutated_client.calls)
                path.write_bytes(original)

            # Provider telemetry is untrusted.  Even with two value-equal
            # plans and two valid compiler certificates, a provider cannot
            # overwrite the raw-selected second handle with the first one.
            first_choice = choices.choices[0]
            second_contract = first_choice.executable_contract.bind_packing_certificate(
                {
                    **dict(
                        first_choice.preflight.exact_packing_certificate
                        or {}
                    ),
                    "passed": True,
                    "authority_test_variant": "second",
                }
            )
            equal_plan_certificates = _issue_cell_plan_portfolio(
                candidates=(
                    {
                        "plan": first_choice.certificate.plan,
                        "deterministic_candidate_metrics": (
                            first_choice.certificate.deterministic_candidate_metrics
                        ),
                        "allowed_tool_program_ids": (
                            first_choice.executable_contract.execution_program_id,
                        ),
                        "executable_contract_id": (
                            first_choice.executable_contract.contract_id
                        ),
                    },
                    {
                        "plan": first_choice.certificate.plan,
                        "deterministic_candidate_metrics": (
                            first_choice.certificate.deterministic_candidate_metrics
                        ),
                        "allowed_tool_program_ids": (
                            second_contract.execution_program_id,
                        ),
                        "executable_contract_id": second_contract.contract_id,
                    },
                ),
                vetoed=(),
                authority_binding=choices.certificates.authority_binding,
            )
            equal_plan_execution = _CertifiedCellExecutionPortfolio(
                choices=(
                    _CertifiedCellExecutionChoice(
                        equal_plan_certificates.survivors[0],
                        first_choice.executable_contract,
                        first_choice.preflight,
                    ),
                    _CertifiedCellExecutionChoice(
                        equal_plan_certificates.survivors[1],
                        second_contract,
                        first_choice.preflight,
                    ),
                ),
                certificates=equal_plan_certificates,
            )
            injected_handle = CellPlanSelectionHandle.from_candidate(
                equal_plan_certificates.survivors[0],
                selected_tool_program_id=(
                    first_choice.executable_contract.execution_program_id
                ),
            )
            collision_client = _CertifiedCellSelectionClient(
                candidate_index=1,
                usage={
                    "model": "malicious-fixture",
                    "selection_handle": injected_handle.to_metadata(),
                    "selected_candidate_id": (
                        equal_plan_certificates.survivors[0].candidate_id
                    ),
                },
            )
            online_cell_planner = OpenAIMultimodalJointPlanner(
                client=collision_client,
                max_contract_attempts=1,
            )

            class _SplitStagePlanner:
                def select_interpretation(self, **kwargs):
                    return HeuristicJointPlanner().select_interpretation(
                        **kwargs
                    )

                def create_plan(self, **kwargs):
                    return online_cell_planner.create_plan(**kwargs)

            class _InjectedCellPortfolioWorkflow(JointPathologyEditWorkflow):
                def _compile_cell_only_candidate_portfolio(self, **_kwargs):
                    return equal_plan_execution

            collision_result = _InjectedCellPortfolioWorkflow(
                tissue_planner=MultiInterfaceResearchTissuePlanner(),
                joint_planner=_SplitStagePlanner(),
                critic=_ApprovingJointCritic(),
            ).run(case, output_root=root / "provider-usage-collision")
            self.assertEqual(collision_result.status, "abstained")
            self.assertTrue(
                any(
                    "reserved authority fields" in reason
                    for reason in collision_result.abstain_reasons
                ),
                collision_result.abstain_reasons,
            )
            self.assertEqual(len(collision_client.calls), 1)

    def test_cell_selection_handle_distinguishes_value_equal_plans(self):
        plan = SimpleNamespace(to_metadata=lambda: {"plan": "same"})
        first_certificate = SimpleNamespace(
            candidate_id="cell-plan:first",
            compiler_certificate_sha256="a" * 64,
            allowed_tool_program_ids=("program:first",),
            executable_contract_id="contract:first",
            authority_binding_sha256="1" * 64,
            plan=plan,
        )
        second_certificate = SimpleNamespace(
            candidate_id="cell-plan:second",
            compiler_certificate_sha256="b" * 64,
            allowed_tool_program_ids=("program:second",),
            executable_contract_id="contract:second",
            authority_binding_sha256="2" * 64,
            plan=plan,
        )
        first_contract = SimpleNamespace(
            execution_program_id="program:first", contract_id="contract:first"
        )
        second_contract = SimpleNamespace(
            execution_program_id="program:second", contract_id="contract:second"
        )
        portfolio = _CertifiedCellExecutionPortfolio(
            choices=(
                _CertifiedCellExecutionChoice(
                    first_certificate, first_contract, SimpleNamespace()
                ),
                _CertifiedCellExecutionChoice(
                    second_certificate, second_contract, SimpleNamespace()
                ),
            ),
            certificates=SimpleNamespace(),
        )
        handle = CellPlanSelectionHandle.from_candidate(
            second_certificate,
            selected_tool_program_id="program:second",
        )
        selected = _select_cell_execution_choice(
            portfolio=portfolio,
            plan=plan,
            planner_usage={"selection_handle": handle.to_metadata()},
        )
        self.assertIs(selected.certificate, second_certificate)

    def test_structural_risk_counts_events_not_false_policy_flags(self):
        no_event = {
            "passed": True,
            "source_components_before": 1,
            "source_components_after": 1,
            "target_components_before": 1,
            "target_components_after": 1,
            "source_holes_before": 0,
            "source_holes_after": 0,
            "target_holes_before": 0,
            "target_holes_after": 0,
            "allow_source_component_split": False,
            "allow_target_hole_resolution": False,
            "target_merge": False,
        }
        one_event = {**no_event, "source_components_after": 2}
        two_events = {**one_event, "target_holes_after": 1}
        self.assertEqual(_structural_event_risk_count(no_event), 0.0)
        self.assertLess(
            _structural_event_risk_count(no_event),
            _structural_event_risk_count(one_event),
        )
        self.assertLess(
            _structural_event_risk_count(one_event),
            _structural_event_risk_count(two_events),
        )

    def test_online_tissue_planner_rejects_forged_candidate_tool_and_preference(self):
        mutations = {
            "unknown candidate": lambda value: {
                **value,
                "selected_candidate_id": "tissue-plan:forged",
            },
            "forbidden tool family": lambda value: {
                **value,
                "selected_tool_family": "organic_v2",
            },
            "unknown preference": lambda value: {
                **value,
                "supporting_preference_rule_ids": ["pref:forged"],
            },
        }
        for label, mutation in mutations.items():
            with (
                self.subTest(label=label),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                case = _as_breast_growth_case(_write_synthetic_case(root))
                case = replace(
                    case,
                    case_id="online-forged-" + label.replace(" ", "-"),
                    instruction="infiltrative-nest-cord-extension-v1",
                    primitive_id="infiltrative-nest-cord-extension-v1",
                    joint_area_budget=JointAreaBudget(
                        target_fraction=0.025,
                        min_fraction=0.01,
                        max_fraction=0.04,
                        tissue_min_fraction=0.01,
                    ),
                    provenance={
                        **case.provenance,
                        "joint_mechanism_id": (
                            "breast-infiltrative-nest-cord-extension"
                        ),
                        "joint_primitive_id": (
                            "infiltrative-nest-cord-extension-v1"
                        ),
                    },
                )
                result = JointPathologyEditWorkflow(
                    tissue_planner=OpenAIJointAwareTissuePlanner(
                        client=_CertifiedTissueSelectionClient(mutation),
                        max_contract_attempts=1,
                    ),
                    joint_planner=HeuristicJointPlanner(),
                    critic=_ApprovingJointCritic(),
                ).run(case, output_root=root / "result")
                self.assertEqual(result.status, "abstained")

    def test_annotation_anchored_breast_primitives_execute_real_gates(self):
        fixtures = (
            (
                "cohesive-boundary-expansion-v1",
                "breast-annotation-anchored-boundary-growth",
                JointAreaBudget(
                    target_fraction=0.12,
                    min_fraction=0.10,
                    max_fraction=0.14,
                    tissue_min_fraction=0.10,
                    relative_tolerance=0.05,
                ),
                None,
            ),
            (
                "infiltrative-nest-cord-extension-v1",
                "breast-infiltrative-nest-cord-extension",
                JointAreaBudget(
                    target_fraction=0.025,
                    min_fraction=0.01,
                    max_fraction=0.04,
                    tissue_min_fraction=0.01,
                    relative_tolerance=0.05,
                ),
                None,
            ),
            (
                "peritumoral-neoplastic-scatter-increase-v1",
                "breast-peritumoral-neoplastic-scatter",
                None,
                CellCountExtentBudget(
                    target_delta_count=8,
                    min_delta_count=6,
                    max_delta_count=10,
                    maximum_extent_px=48,
                    interface_min_px=4,
                    interface_max_px=48,
                    minimum_effect_span_px=20,
                    minimum_effect_foci=3,
                ),
            ),
            (
                "peritumoral-small-cluster-increase-v1",
                "breast-peritumoral-small-cluster",
                None,
                CellCountExtentBudget(
                    target_delta_count=12,
                    min_delta_count=12,
                    max_delta_count=16,
                    maximum_extent_px=64,
                    interface_min_px=4,
                    interface_max_px=64,
                    minimum_effect_span_px=20,
                    minimum_effect_foci=3,
                ),
            ),
        )
        for index, (primitive, mechanism, joint_budget, cell_budget) in enumerate(
            fixtures
        ):
            with (
                self.subTest(primitive=primitive),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                synthetic = (
                    _write_synthetic_case(root, size=256, tumor_radius=70)
                    if primitive == "peritumoral-small-cluster-increase-v1"
                    else _write_synthetic_case(root)
                )
                case = _as_breast_growth_case(synthetic)
                case = replace(
                    case,
                    case_id=f"breast-mask-graph-{index:02d}",
                    instruction=primitive,
                    primitive_id=primitive,
                    joint_area_budget=joint_budget,
                    cell_count_extent_budget=cell_budget,
                    provenance={
                        **case.provenance,
                        "joint_mechanism_id": mechanism,
                        "joint_primitive_id": primitive,
                    },
                )
                result = JointPathologyEditWorkflow(
                    tissue_planner=MultiInterfaceResearchTissuePlanner(),
                    joint_planner=HeuristicJointPlanner(),
                    critic=_ApprovingJointCritic(),
                ).run(case, output_root=root / "result")

                self.assertEqual(
                    result.status,
                    "selected_research",
                    result.abstain_reasons,
                )
                reports = json.loads(
                    Path(result.artifact_paths["joint_gate_reports.json"]).read_text(
                        encoding="utf-8"
                    )
                )
                selected = next(
                    item
                    for item in reports
                    if item["candidate_id"] == result.selected_candidate_id
                )
                self.assertTrue(selected["passed"])
                self.assertTrue(
                    all(
                        item["passed"]
                        for item in selected["checks"]
                        if item["severity"] == "hard"
                    )
                )
                if primitive == "peritumoral-tumor-nest-formation-v1":
                    self.assertTrue(
                        next(
                            check
                            for check in selected["checks"]
                            if check["check_id"]
                            == "annotation_anchored_extension_geometry"
                        )["passed"]
                    )
                    self.assertTrue(
                        next(
                            check
                            for check in selected["checks"]
                            if check["check_id"]
                            == "interface_seam_continuity"
                        )["passed"]
                    )
                    realization = next(
                        check
                        for check in selected["checks"]
                        if check["check_id"] == "mechanism_realization"
                    )
                    self.assertTrue(realization["passed"])
                    self.assertGreaterEqual(
                        realization["metrics"]["placement_count"],
                        2,
                    )
                    self.assertTrue(
                        all(
                            size >= 2
                            for size in realization["metrics"][
                                "declared_cluster_sizes"
                            ]
                        )
                    )

    def test_breast_generic_immune_compartment_turnover_executes_both_directions(self):
        for primitive, source_label, target_label, source_cell, target_cell in (
            ("generic-immune-infiltrate-increase-v1", 2, 4, 3, 2),
            ("generic-immune-infiltrate-decrease-v1", 4, 2, 2, 3),
        ):
            with (
                self.subTest(primitive=primitive),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                case = _write_breast_immune_case(root, primitive=primitive)
                result = JointPathologyEditWorkflow(
                    tissue_planner=MultiInterfaceResearchTissuePlanner(),
                    joint_planner=HeuristicJointPlanner(),
                    critic=_ApprovingJointCritic(),
                ).run(case, output_root=root / "immune-turnover")

                self.assertEqual(
                    result.status, "selected_research", result.abstain_reasons
                )
                change = result.condition.tissue_change
                source_tissue = np.load(case.source_tissue_mask_uri)
                target_tissue = result.condition.target_tissue_mask
                self.assertEqual(set(np.unique(source_tissue[change])), {source_label})
                self.assertEqual(set(np.unique(target_tissue[change])), {target_label})
                source_nuclei = np.asarray(Image.open(case.source_nuclei_mask_uri))
                target_nuclei = result.condition.target_nuclei_mask
                self.assertGreater(
                    np.count_nonzero(target_nuclei == target_cell),
                    np.count_nonzero(source_nuclei == target_cell),
                )
                self.assertLess(
                    np.count_nonzero(target_nuclei == source_cell),
                    np.count_nonzero(source_nuclei == source_cell),
                )
                reports = json.loads(
                    Path(result.artifact_paths["joint_gate_reports.json"]).read_text(
                        encoding="utf-8"
                    )
                )
                mechanism_checks = [
                    check
                    for report in reports
                    for check in report["checks"]
                    if check["check_id"]
                    == "mechanism_postcondition:breast-generic-immune-compartment-turnover"
                ]
                self.assertTrue(mechanism_checks)
                self.assertTrue(any(item["passed"] for item in mechanism_checks))

    def test_integer_interface_allocations_are_normalized_after_rounding(self):
        # Independent integer rounding used to produce 1.00002008 for breast
        # case 073 and incorrectly fail a valid multi-interface plan.
        weights = _normalize_integer_allocations((16_603, 16_603, 16_602))
        self.assertTrue(np.isclose(sum(weights), 1.0, rtol=0.0, atol=1e-12))

    def test_tissue_gate_failure_is_replanned_and_retooled_before_abstain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = _as_breast_growth_case(_write_synthetic_case(root))
            gates = _RetryThenPassingTissueGate()
            result = JointPathologyEditWorkflow(
                tissue_planner=HeuristicInterfacePlanner(),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
                tissue_gates=gates,
                joint_gates=_PassingJointGateFixture(),
            ).run(case, output_root=root / "retry")
            self.assertEqual(
                result.status, "selected_research", result.abstain_reasons
            )
            case_dir = root / "retry" / case.case_id
            feedback = json.loads(
                (case_dir / "execution_feedback_pass_1.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(feedback["stage"], "tissue_gate")
            self.assertTrue(
                (case_dir / "tissue_execution_contract_pass_2.json").is_file()
            )

    def test_candidate_local_cell_failure_is_audited_and_replanned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = _as_breast_growth_case(_write_synthetic_case(root))
            executor = _AlwaysFailingCandidateCellExecutor()
            result = JointPathologyEditWorkflow(
                tissue_planner=HeuristicInterfacePlanner(),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
                tissue_gates=_PassingTissueGateFixture(),
                cell_executor=executor,
                config=JointWorkflowConfig(
                    maximum_tissue_candidates=1,
                    maximum_tissue_planning_attempts=6,
                ),
            ).run(case, output_root=root / "cell-execution-retry")

            self.assertEqual(result.status, "abstained")
            self.assertGreaterEqual(executor.calls, 2)
            case_dir = root / "cell-execution-retry" / case.case_id
            failures = json.loads(
                (case_dir / "cell_execution_failures.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(failures)
            self.assertIn(
                "fixture candidate-local executor failure",
                failures[0]["error"],
            )
            feedback = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in case_dir.glob("execution_feedback_pass_*.json")
            ]
            self.assertTrue(
                any(item.get("stage") == "cell_execution" for item in feedback)
            )

    def test_explicit_mature_regeneration_requirement_rejects_ranker_only_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _as_breast_growth_case(_write_synthetic_case(root))
            case = replace(
                source,
                provenance={
                    **source.provenance,
                    "require_mature_probnet_regeneration": True,
                },
            )
            result = JointPathologyEditWorkflow(
                tissue_planner=HeuristicInterfacePlanner(),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
                tissue_gates=_PassingTissueGateFixture(),
            ).run(case, output_root=root / "mature-required")
            self.assertEqual(result.status, "abstained")
            self.assertTrue(
                any(
                    "mature ProbNet regeneration" in reason
                    for reason in result.abstain_reasons
                ),
                result.abstain_reasons,
            )

    def test_offline_joint_workflow_emits_review_artifacts_and_never_calls_api(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = _as_breast_growth_case(_write_synthetic_case(root))
            workflow = JointPathologyEditWorkflow(
                tissue_planner=MultiInterfaceResearchTissuePlanner(),
                joint_planner=HeuristicJointPlanner(),
                critic=DeterministicJointResearchCritic(),
            )
            result = workflow.run(case, output_root=root / "review")
            self.assertIn(result.status, {"review_required", "abstained"})
            self.assertTrue(Path(result.artifact_paths["result.json"]).is_file())
            self.assertTrue(
                (
                    root / "review" / case.case_id / "joint_nuclei_preflight.json"
                ).is_file()
            )
            if result.status == "review_required":
                self.assertTrue(
                    Path(result.artifact_paths["joint_condition_review"]).is_file()
                )
                self.assertIsNone(result.condition)

    def test_approved_joint_candidate_emits_frozen_generator_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = _as_breast_growth_case(_write_synthetic_case(root))
            workflow = JointPathologyEditWorkflow(
                tissue_planner=HeuristicInterfacePlanner(),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
                tissue_gates=_PassingTissueGateFixture(),
                joint_gates=_PassingJointGateFixture(),
            )
            result = workflow.run(case, output_root=root / "approved")
            self.assertEqual(result.status, "selected_research", result.abstain_reasons)
            self.assertIsNotNone(result.condition)
            canonical_reports = json.loads(
                Path(result.artifact_paths["joint_gate_reports.json"]).read_text(
                    encoding="utf-8"
                )
            )
            selected_report = next(
                report
                for report in canonical_reports
                if report["candidate_id"] == result.selected_candidate_id
            )
            self.assertTrue(selected_report["passed"])
            self.assertFalse(
                any(
                    not check["passed"] and check["severity"] == "hard"
                    for check in selected_report["checks"]
                )
            )
            manifest = Path(result.artifact_paths["handoff_manifest"])
            self.assertTrue(manifest.is_file())
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "joint-generation-handoff-v3")
            self.assertEqual(
                payload["result_binding"]["schema_version"],
                "joint-result-binding-v2",
            )
            self.assertIn("joint_change", payload["paths"])
            self.assertIn("generation_support", payload["paths"])
            self.assertIn("contract_T_population", payload["paths"])
            self.assertIn("contract_E_erasure", payload["paths"])
            self.assertIn("contract_S_support_context", payload["paths"])
            self.assertIn("contract_C_continuity_region", payload["paths"])
            self.assertIn("contract_A_selected_anchor", payload["paths"])
            self.assertEqual(
                payload["executable_contract_id"],
                result.condition.executable_contract_id,
            )
            contract = payload["execution_contract"]["executable_contract"]
            self.assertEqual(
                contract["contract_id"], result.condition.executable_contract_id
            )
            packing = contract["packing_certificate"]
            self.assertTrue(packing["passed"])
            self.assertEqual(
                len(packing["placements"]), packing["requested_count"]
            )
            program = contract["cell_program"]
            self.assertEqual(
                program["compiler_version"], "joint-cell-tool-compiler-v17"
            )
            self.assertEqual(
                program["policies"]["P"],
                "contract-legal-centers-exact-footprint-certified-against-V",
            )
            self.assertEqual(
                program["population_target_region_pixels"],
                result.condition.ledger.tissue_pixels,
            )
            self.assertGreaterEqual(
                program["population_target_region_pixels"],
                program["placement_center_region_pixels"],
            )
            self.assertEqual(
                program["continuity_mode"],
                "adaptive_population_continuity",
            )
            self.assertGreater(program["continuity_anchor_mask_pixels"], 0)
            self.assertGreater(program["continuity_region_pixels"], 0)
            self.assertNotIn("seam_required_count", program)
            self.assertIn(
                "executable_contract_binding", contract["required_checker_ids"]
            )
            actual_support = (
                np.asarray(Image.open(payload["paths"]["contract_S_support_context"]))
                > 0
            )
            np.testing.assert_array_equal(
                actual_support, result.condition.generation_support
            )
            support_digest = hashlib.sha256()
            support_digest.update(str(actual_support.shape).encode("ascii"))
            support_digest.update(np.ascontiguousarray(actual_support).tobytes())
            self.assertEqual(
                contract["cell_program"]["support_context_region_sha256"],
                support_digest.hexdigest(),
            )
            source_tissue = np.load(case.source_tissue_mask_uri)
            source_nuclei = np.asarray(Image.open(case.source_nuclei_mask_uri))
            scene = build_joint_scene_analysis(
                source_tissue,
                source_nuclei,
                schema=MaskProfileSchema.from_reference_profile("BCSS"),
                pixel_size_um=case.pixel_size_um,
                nuclei_instances_path=case.source_nuclei_instances_uri,
                auxiliary_structure_paths=case.auxiliary_structure_uris,
            )
            expected_erasure = np.zeros_like(source_tissue, dtype=bool)
            for instance_id in contract["cell_instance_contract"]["erase_instance_ids"]:
                expected_erasure |= scene.instance_masks[instance_id]
            actual_erasure = (
                np.asarray(Image.open(payload["paths"]["contract_E_erasure"])) > 0
            )
            np.testing.assert_array_equal(actual_erasure, expected_erasure)
            actual_population = (
                np.asarray(
                    Image.open(payload["paths"]["contract_T_population"])
                )
                > 0
            )
            np.testing.assert_array_equal(
                actual_population,
                result.condition.tissue_change,
            )
            execution = json.loads(
                (
                    root
                    / "approved"
                    / case.case_id
                    / "tissue_execution_contract_pass_1.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(execution["certified_candidate_ids"])
            self.assertEqual(
                execution["joint_preflight_pass_count"],
                len(execution["certified_candidate_ids"]),
            )
            inputs, route, verified = build_frozen_generator_inputs(
                manifest,
                output_dir=root / "generator-inputs",
            )
            self.assertEqual(
                inputs.target_nuclei_mask,
                payload["paths"]["target_nuclei_mask"],
            )
            self.assertGreater(route.joint_fraction, 0)
            self.assertEqual(
                verified["result_binding"]["candidate_id"],
                payload["candidate_id"],
            )
            post = audit_joint_generation_handoff(
                manifest_path=manifest,
                generated_image=root / "not-yet-rendered.png",
                output_path=root / "post-generation-audit.json",
                tissue_evaluator=None,
                cell_evaluator=None,
                visual_critic=None,
            )
            self.assertFalse(post.passed)
            self.assertEqual(post.capability_status, "render_unsupported")

    def test_stroma_contract_forbids_neoplastic_cell_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_synthetic_case(root)
            case = replace(
                source,
                case_id="synthetic-stroma",
                instruction="increase stroma burden",
                primitive_id="stroma-increase-v1",
                joint_area_budget=JointAreaBudget(
                    target_fraction=0.08,
                    min_fraction=0.04,
                    max_fraction=0.12,
                    tissue_min_fraction=0.04,
                ),
            )
            result = JointPathologyEditWorkflow(
                tissue_planner=HeuristicInterfacePlanner(),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
                tissue_gates=_PassingTissueGateFixture(),
            ).run(case, output_root=root / "stroma")
            self.assertEqual(result.status, "abstained")
            self.assertIn(
                "no joint mechanism",
                result.abstain_reasons[0],
            )

    def test_glas_periglandular_scatter_materializes_native_gland_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_synthetic_case(root)
            provenance = dict(source.provenance)
            provenance["joint_mechanism_id"] = "colorectal-tumor-budding-front"
            case = replace(
                source,
                case_id="synthetic-budding",
                instruction="add scattered tumor cells near the gland boundary",
                primitive_id="peritumoral-neoplastic-scatter-increase-v1",
                joint_area_budget=None,
                cell_count_extent_budget=CellCountExtentBudget(8, 6, 10, 48, 4, 32),
                provenance=provenance,
            )
            result = JointPathologyEditWorkflow(
                tissue_planner=HeuristicInterfacePlanner(),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
            ).run(case, output_root=root / "cell-only")
            self.assertEqual(
                result.status, "selected_research", result.abstain_reasons
            )
            producer_report = json.loads(
                Path(
                    result.artifact_paths["auxiliary_producer_report.json"]
                ).read_text(encoding="utf-8")
            )
            instance_record = next(
                item
                for item in producer_report
                if item["structure_id"] == "native_gland_instance_map"
            )
            self.assertEqual(instance_record["provenance"]["connectivity"], 8)
            self.assertGreater(
                instance_record["provenance"]["instance_count"], 0
            )

    def test_glas_periglandular_primitives_execute_from_native_instance_authority(self):
        contract = json.loads(
            (
                Path(__file__).parents[1]
                / "phase3_joint_edit_refine"
                / "skills"
                / "catalog"
                / "joint-mechanism"
                / "colorectal-tumor-budding-front"
                / "references"
                / "joint_contract.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            set(
                contract["representability_contract"][
                    "required_auxiliary_structures"
                ]
            ),
            {
                "native_gland_instance_map",
                "gland_or_lumen_support",
                "external_cellular_stroma_map",
            },
        )
        self.assertIn(
            "native_gland_instance_annulus_binding",
            contract["cell_program"]["required_checker_ids"],
        )
        for primitive, minimum_foci in (
            ("peritumoral-neoplastic-scatter-increase-v1", 3),
            ("peritumoral-small-cluster-increase-v1", 2),
        ):
            with (
                self.subTest(primitive=primitive),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                source = _with_native_gland_instance_authority(
                    _write_synthetic_case(root)
                )
                case = replace(
                    source,
                    case_id="glas-native-" + primitive,
                    instruction=primitive,
                    primitive_id=primitive,
                    joint_area_budget=None,
                    cell_count_extent_budget=CellCountExtentBudget(
                        8,
                        6,
                        10,
                        48,
                        4,
                        48,
                        minimum_effect_span_px=20,
                        minimum_effect_foci=minimum_foci,
                    ),
                    provenance={
                        **source.provenance,
                        "joint_mechanism_id": "colorectal-tumor-budding-front",
                        "joint_primitive_id": primitive,
                    },
                )
                result = JointPathologyEditWorkflow(
                    tissue_planner=HeuristicInterfacePlanner(),
                    joint_planner=HeuristicJointPlanner(),
                    critic=_ApprovingJointCritic(),
                ).run(case, output_root=root / "glas-periglandular")
                self.assertEqual(
                    result.status, "selected_research", result.abstain_reasons
                )
                self.assertEqual(result.condition.ledger.tissue_pixels, 0)
                self.assertGreater(
                    result.condition.ledger.added_nucleus_pixels, 0
                )
                candidate_manifest = json.loads(
                    Path(result.artifact_paths["candidates.json"]).read_text(
                        encoding="utf-8"
                    )
                )
                selected_candidate = next(
                    item
                    for item in candidate_manifest
                    if item["candidate_id"] == result.selected_candidate_id
                )
                trace = selected_candidate["tool_trace"]
                placements = trace.get("placements", [])
                placed_count = int(trace.get("placed_count", len(placements)))
                self.assertEqual(placed_count, len(placements))
                self.assertGreaterEqual(
                    placed_count,
                    case.cell_count_extent_budget.min_delta_count,
                )
                realized_area = sum(
                    int(item["area_px"]) for item in placements
                )
                self.assertEqual(
                    result.condition.ledger.added_nucleus_pixels,
                    realized_area,
                )
                centers = np.asarray(
                    [item["center_xy"] for item in placements],
                    dtype=float,
                )
                if len(centers) >= 2:
                    pairwise = centers[:, None, :] - centers[None, :, :]
                    observed_span = float(
                        np.sqrt(np.max(np.sum(pairwise**2, axis=2)))
                    )
                else:
                    observed_span = 0.0
                self.assertGreaterEqual(
                    observed_span,
                    case.cell_count_extent_budget.minimum_effect_span_px,
                )
                reports = json.loads(
                    Path(result.artifact_paths["joint_gate_reports.json"]).read_text(
                        encoding="utf-8"
                    )
                )
                selected = next(
                    item
                    for item in reports
                    if item["candidate_id"] == result.selected_candidate_id
                )
                native_binding = next(
                    check
                    for check in selected["checks"]
                    if check["check_id"] == "native_gland_instance_annulus_binding"
                )
                self.assertTrue(native_binding["passed"])
                self.assertEqual(
                    native_binding["metrics"]["available_native_raster_ids"],
                    [1, 2],
                )
                self.assertGreaterEqual(
                    len(
                        native_binding["metrics"][
                            "selected_native_instance_component_ids"
                        ]
                    ),
                    2,
                )

    def test_panda_pattern4_pattern5_and_cord_execute_real_workflows(self):
        fixtures = (
            (
                9,
                "cohesive-boundary-expansion-v1",
                "prostate-pattern-4-growth",
                JointAreaBudget(0.08, 0.04, 0.12, 0.04),
            ),
            (
                10,
                "cohesive-boundary-expansion-v1",
                "prostate-pattern-5-growth",
                JointAreaBudget(0.08, 0.04, 0.12, 0.04),
            ),
            (
                10,
                "infiltrative-nest-cord-extension-v1",
                "prostate-pattern-5-infiltrative-front",
                JointAreaBudget(0.025, 0.01, 0.04, 0.01),
            ),
        )
        for fine_id, primitive, mechanism, budget in fixtures:
            with (
                self.subTest(primitive=primitive, mechanism=mechanism),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                case = _as_panda_case(
                    _write_synthetic_case(root),
                    fine_id=fine_id,
                    mechanism_id=mechanism,
                    primitive_id=primitive,
                )
                case = replace(
                    case,
                    case_id=f"panda-{fine_id}-{primitive}",
                    joint_area_budget=budget,
                )
                result = JointPathologyEditWorkflow(
                    tissue_planner=MultiInterfaceResearchTissuePlanner(),
                    joint_planner=HeuristicJointPlanner(),
                    critic=_ApprovingJointCritic(),
                ).run(case, output_root=root / "panda-growth")
                self.assertEqual(
                    result.status, "selected_research", result.abstain_reasons
                )
                self.assertGreater(result.condition.ledger.tissue_pixels, 0)
                if primitive == "infiltrative-nest-cord-extension-v1":
                    reports = json.loads(
                        Path(
                            result.artifact_paths["joint_gate_reports.json"]
                        ).read_text(encoding="utf-8")
                    )
                    selected = next(
                        item
                        for item in reports
                        if item["candidate_id"] == result.selected_candidate_id
                    )
                    geometry = next(
                        check
                        for check in selected["checks"]
                        if check["check_id"]
                        == "annotation_anchored_extension_geometry"
                    )
                    self.assertTrue(geometry["passed"])

    def test_panda_cord_four_percent_floor_conflict_fails_without_replan_stall(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = _as_panda_case(
                _write_synthetic_case(root),
                fine_id=10,
                mechanism_id="prostate-pattern-5-infiltrative-front",
                primitive_id="infiltrative-nest-cord-extension-v1",
            )
            case = replace(
                case,
                case_id="panda-cord-four-percent-floor",
                joint_area_budget=JointAreaBudget(0.05, 0.04, 0.06, 0.04),
            )
            result = JointPathologyEditWorkflow(
                tissue_planner=MultiInterfaceResearchTissuePlanner(),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
            ).run(case, output_root=root / "panda-cord-floor")
            self.assertEqual(result.status, "abstained")
            reason = " ".join(result.abstain_reasons)
            self.assertNotIn("deterministic_replan_stalled", reason)
            self.assertTrue(
                "meaningful" in reason or "capacity" in reason or "safe" in reason,
                reason,
            )

    def test_panda_pattern5_scatter_executes_in_certified_stromal_annulus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primitive = "peritumoral-neoplastic-scatter-increase-v1"
            case = _as_panda_case(
                _write_synthetic_case(root),
                fine_id=10,
                mechanism_id="prostate-pattern-5-peripheral-scatter",
                primitive_id=primitive,
            )
            case = replace(
                case,
                case_id="panda-pattern5-scatter",
                joint_area_budget=None,
                cell_count_extent_budget=CellCountExtentBudget(
                    8,
                    6,
                    10,
                    48,
                    4,
                    48,
                    minimum_effect_span_px=20,
                    minimum_effect_foci=3,
                ),
            )
            result = JointPathologyEditWorkflow(
                tissue_planner=HeuristicInterfacePlanner(),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
                config=JointWorkflowConfig(
                    require_evaluation_input_bindings=True
                ),
            ).run(case, output_root=root / "panda-scatter")
            self.assertEqual(
                result.status, "selected_research", result.abstain_reasons
            )
            self.assertEqual(result.condition.ledger.tissue_pixels, 0)
            self.assertGreaterEqual(result.condition.ledger.added_nucleus_pixels, 54)
            reports = json.loads(
                Path(result.artifact_paths["joint_gate_reports.json"]).read_text(
                    encoding="utf-8"
                )
            )
            selected = next(
                item
                for item in reports
                if item["candidate_id"] == result.selected_candidate_id
            )
            binding = next(
                check
                for check in selected["checks"]
                if check["check_id"] == "panda_pattern5_scatter_binding"
            )
            self.assertTrue(binding["passed"])

    def test_panda_pattern5_scatter_rejects_fine9_only_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primitive = "peritumoral-neoplastic-scatter-increase-v1"
            case = _as_panda_case(
                _write_synthetic_case(root),
                fine_id=9,
                mechanism_id="prostate-pattern-5-peripheral-scatter",
                primitive_id=primitive,
            )
            case = replace(
                case,
                case_id="panda-pattern5-scatter-fine9-veto",
                joint_area_budget=None,
                cell_count_extent_budget=CellCountExtentBudget(
                    8,
                    6,
                    10,
                    48,
                    4,
                    48,
                    minimum_effect_span_px=20,
                    minimum_effect_foci=3,
                ),
            )
            result = JointPathologyEditWorkflow(
                tissue_planner=HeuristicInterfacePlanner(),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
            ).run(case, output_root=root / "panda-scatter-fine9")
            self.assertEqual(result.status, "abstained")
            self.assertNotIn(
                "prostate-local-population-modulation",
                " ".join(result.abstain_reasons),
            )

    def test_p2_boundary_cord_scatter_and_small_focus_primitives_execute(self):
        fixtures = (
            (
                "ignite-semantic-v1",
                "lung-carcinoma-v1",
                "lung-cellvit-source-first-v1",
                2,
                {"source_site": "lung", "specimen_type": "resection"},
                "lung-solid-squamous-growth",
                "cohesive-boundary-expansion-v1",
                "growth",
            ),
            (
                "ignite-semantic-v1",
                "lung-carcinoma-v1",
                "lung-cellvit-source-first-v1",
                2,
                {"source_site": "lung", "specimen_type": "resection"},
                "lung-stromal-invasive-front",
                "infiltrative-nest-cord-extension-v1",
                "cord",
            ),
            (
                "ignite-semantic-v1",
                "lung-carcinoma-v1",
                "lung-cellvit-source-first-v1",
                2,
                {"source_site": "lung", "specimen_type": "resection"},
                "lung-local-population-modulation",
                "peritumoral-neoplastic-scatter-increase-v1",
                "scatter",
            ),
            (
                "ignite-semantic-v1",
                "lung-carcinoma-v1",
                "lung-cellvit-source-first-v1",
                2,
                {"source_site": "lung", "specimen_type": "resection"},
                "lung-local-population-modulation",
                "peritumoral-small-cluster-increase-v1",
                "cluster",
            ),
            (
                "puma-semantic-v1",
                "melanoma-v1",
                "melanoma-cellvit-source-first-v1",
                2,
                {"source_site": "skin", "primary_or_metastatic": "primary"},
                "melanoma-cohesive-nest-sheet",
                "cohesive-boundary-expansion-v1",
                "growth",
            ),
            (
                "puma-semantic-v1",
                "melanoma-v1",
                "melanoma-cellvit-source-first-v1",
                2,
                {"source_site": "skin", "primary_or_metastatic": "primary"},
                "melanoma-peritumoral-small-focus",
                "peritumoral-small-cluster-increase-v1",
                "cluster",
            ),
            (
                "orca-semantic-v1",
                "oral-squamous-cell-carcinoma-v1",
                "oral-scc-cellvit-source-first-v1",
                7,
                {},
                "oral-scc-cohesive-nest-cord",
                "cohesive-boundary-expansion-v1",
                "growth",
            ),
            (
                "orca-semantic-v1",
                "oral-squamous-cell-carcinoma-v1",
                "oral-scc-cellvit-source-first-v1",
                7,
                {},
                "oral-scc-annotation-anchored-cord-extension",
                "infiltrative-nest-cord-extension-v1",
                "cord",
            ),
            (
                "orca-semantic-v1",
                "oral-squamous-cell-carcinoma-v1",
                "oral-scc-cellvit-source-first-v1",
                7,
                {},
                "oral-scc-dispersed-invasive-front",
                "peritumoral-neoplastic-scatter-increase-v1",
                "scatter",
            ),
            (
                "orca-semantic-v1",
                "oral-squamous-cell-carcinoma-v1",
                "oral-scc-cellvit-source-first-v1",
                7,
                {},
                "oral-scc-dispersed-invasive-front",
                "peritumoral-small-cluster-increase-v1",
                "cluster",
            ),
        )
        for (
            profile_id,
            domain_id,
            population_id,
            host_fine_id,
            required_provenance,
            mechanism_id,
            primitive_id,
            geometry_kind,
        ) in fixtures:
            with (
                self.subTest(
                    profile=profile_id,
                    mechanism=mechanism_id,
                    primitive=primitive_id,
                ),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                case = _as_organ_profile_case(
                    _write_synthetic_case(root),
                    pathology_domain_id=domain_id,
                    annotation_profile_id=profile_id,
                    cell_population_profile_id=population_id,
                    tumor_fine_id=1,
                    host_fine_id=host_fine_id,
                    mechanism_id=mechanism_id,
                    primitive_id=primitive_id,
                    required_provenance={
                        "preprocessing_revision": "synthetic-p2-v1",
                        **required_provenance,
                    },
                )
                if profile_id == "orca-semantic-v1":
                    tissue_path = Path(case.source_tissue_mask_uri)
                    tissue = np.load(tissue_path, allow_pickle=False)
                    tissue[:4, :] = 0
                    np.save(tissue_path, tissue, allow_pickle=False)
                    tissue_digest = _sha(tissue_path)
                    case = replace(
                        case,
                        provenance={
                            **case.provenance,
                            "source_tissue_mask_sha256": tissue_digest,
                            "original_label_map_digest": tissue_digest,
                        },
                    )
                if geometry_kind == "growth":
                    joint_budget = JointAreaBudget(0.08, 0.04, 0.12, 0.04)
                    cell_budget = None
                elif geometry_kind == "cord":
                    joint_budget = JointAreaBudget(0.025, 0.01, 0.04, 0.01)
                    cell_budget = None
                else:
                    joint_budget = None
                    cell_budget = CellCountExtentBudget(
                        8,
                        6,
                        10,
                        48,
                        4,
                        48,
                        minimum_effect_span_px=20,
                        minimum_effect_foci=(
                            3 if geometry_kind == "scatter" else 2
                        ),
                    )
                case = replace(
                    case,
                    case_id=f"p2-{profile_id}-{primitive_id}",
                    joint_area_budget=joint_budget,
                    cell_count_extent_budget=cell_budget,
                )
                result = JointPathologyEditWorkflow(
                    tissue_planner=(
                        MultiInterfaceResearchTissuePlanner()
                        if joint_budget is not None
                        else HeuristicInterfacePlanner()
                    ),
                    joint_planner=HeuristicJointPlanner(),
                    critic=_ApprovingJointCritic(),
                ).run(case, output_root=root / "p2-geometry")
                self.assertEqual(
                    result.status, "selected_research", result.abstain_reasons
                )
                if joint_budget is None:
                    self.assertEqual(result.condition.ledger.tissue_pixels, 0)
                    self.assertGreaterEqual(
                        result.condition.ledger.added_nucleus_pixels, 54
                    )
                else:
                    self.assertGreater(result.condition.ledger.tissue_pixels, 0)
                if profile_id == "orca-semantic-v1":
                    source_tissue = np.load(
                        case.source_tissue_mask_uri, allow_pickle=False
                    )
                    np.testing.assert_array_equal(
                        result.condition.target_tissue_mask[source_tissue == 0],
                        source_tissue[source_tissue == 0],
                    )

    def test_puma_scatter_binds_final_footprints_to_explicit_epidermis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primitive_id = "peritumoral-neoplastic-scatter-increase-v1"
            case = _as_organ_profile_case(
                _write_synthetic_case(root),
                pathology_domain_id="melanoma-v1",
                annotation_profile_id="puma-semantic-v1",
                cell_population_profile_id="melanoma-cellvit-source-first-v1",
                tumor_fine_id=1,
                host_fine_id=2,
                mechanism_id="melanoma-discohesive-junctional",
                primitive_id=primitive_id,
                required_provenance={
                    "preprocessing_revision": "synthetic-puma-v1",
                    "source_site": "skin",
                    "primary_or_metastatic": "primary",
                },
            )
            tissue_path = Path(case.source_tissue_mask_uri)
            tissue = np.load(tissue_path, allow_pickle=False)
            tissue[28:34, 48:80] = 5
            np.save(tissue_path, tissue, allow_pickle=False)
            tissue_digest = _sha(tissue_path)
            case = replace(
                case,
                case_id="puma-explicit-junction-scatter",
                joint_area_budget=None,
                cell_count_extent_budget=CellCountExtentBudget(
                    8,
                    6,
                    10,
                    48,
                    4,
                    48,
                    minimum_effect_span_px=20,
                    minimum_effect_foci=3,
                ),
                provenance={
                    **case.provenance,
                    "source_tissue_mask_sha256": tissue_digest,
                    "original_label_map_digest": tissue_digest,
                },
            )
            result = JointPathologyEditWorkflow(
                tissue_planner=HeuristicInterfacePlanner(),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
            ).run(case, output_root=root / "puma-junction")
            self.assertEqual(
                result.status, "selected_research", result.abstain_reasons
            )
            reports = json.loads(
                Path(result.artifact_paths["joint_gate_reports.json"]).read_text(
                    encoding="utf-8"
                )
            )
            selected = next(
                item
                for item in reports
                if item["candidate_id"] == result.selected_candidate_id
            )
            binding = next(
                check
                for check in selected["checks"]
                if check["check_id"] == "puma_epidermal_junction_binding"
            )
            self.assertTrue(binding["passed"])
            self.assertEqual(binding["metrics"]["violation_pixels"], 0)

    def test_p2_lung_and_puma_necrosis_turnover_executes_both_directions(self):
        fixtures = (
            (
                "lung-carcinoma-v1",
                "ignite-semantic-v1",
                "lung-cellvit-source-first-v1",
                "lung-intratumoral-necrosis-turnover",
                {"source_site": "lung", "specimen_type": "resection"},
            ),
            (
                "melanoma-v1",
                "puma-semantic-v1",
                "melanoma-cellvit-source-first-v1",
                "melanoma-intratumoral-necrosis-turnover",
                {"source_site": "skin", "primary_or_metastatic": "primary"},
            ),
        )
        for (
            domain_id,
            profile_id,
            population_id,
            mechanism_id,
            required_provenance,
        ) in fixtures:
            for primitive_id in (
                "necrosis-appearance-v1",
                "necrosis-resolution-v1",
            ):
                with (
                    self.subTest(profile=profile_id, primitive=primitive_id),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    root = Path(directory)
                    source = _write_necrosis_case(root, primitive=primitive_id)
                    case = replace(
                        source,
                        case_id=f"p2-{profile_id}-{primitive_id}",
                        pathology_domain_id=domain_id,
                        annotation_profile_id=profile_id,
                        cell_population_profile_id=population_id,
                        provenance={
                            **source.provenance,
                            **required_provenance,
                            "preprocessing_revision": "synthetic-p2-necrosis-v1",
                            "joint_mechanism_id": mechanism_id,
                            "joint_primitive_id": primitive_id,
                        },
                    )
                    result = JointPathologyEditWorkflow(
                        tissue_planner=MultiInterfaceResearchTissuePlanner(),
                        joint_planner=HeuristicJointPlanner(),
                        critic=_ApprovingJointCritic(),
                    ).run(case, output_root=root / "p2-necrosis")
                    self.assertEqual(
                        result.status,
                        "selected_research",
                        result.abstain_reasons,
                    )
                    self.assertGreater(result.condition.ledger.tissue_pixels, 0)
                    reports = json.loads(
                        Path(
                            result.artifact_paths["joint_gate_reports.json"]
                        ).read_text(encoding="utf-8")
                    )
                    selected = next(
                        item
                        for item in reports
                        if item["candidate_id"] == result.selected_candidate_id
                    )
                    turnover = next(
                        check
                        for check in selected["checks"]
                        if check["check_id"] == "necrosis_cell_turnover"
                    )
                    self.assertTrue(turnover["passed"])

    def test_p2_post_treatment_operational_retreat_executes_without_pathology_claims(self):
        fixtures = (
            (
                "lung-carcinoma-v1",
                "ignite-semantic-v1",
                "lung-cellvit-source-first-v1",
                2,
                "lung-operational-tumor-retreat",
                {"source_site": "lung", "specimen_type": "resection"},
            ),
            (
                "melanoma-v1",
                "puma-semantic-v1",
                "melanoma-cellvit-source-first-v1",
                2,
                "melanoma-operational-tumor-retreat",
                {"source_site": "skin", "primary_or_metastatic": "primary"},
            ),
            (
                "oral-squamous-cell-carcinoma-v1",
                "orca-semantic-v1",
                "oral-scc-cellvit-source-first-v1",
                7,
                "oral-scc-operational-tumor-retreat",
                {},
            ),
        )
        for (
            domain_id,
            profile_id,
            population_id,
            host_fine_id,
            mechanism_id,
            required_provenance,
        ) in fixtures:
            with (
                self.subTest(profile=profile_id),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                primitive_id = "invasive-tumor-footprint-decrease-v1"
                source = _as_organ_profile_case(
                    _write_synthetic_case(root),
                    pathology_domain_id=domain_id,
                    annotation_profile_id=profile_id,
                    cell_population_profile_id=population_id,
                    tumor_fine_id=1,
                    host_fine_id=host_fine_id,
                    mechanism_id=mechanism_id,
                    primitive_id=primitive_id,
                    required_provenance={
                        "preprocessing_revision": "synthetic-p2-retreat-v1",
                        **required_provenance,
                    },
                )
                raw = source.to_metadata()
                raw.update(
                    case_id=f"p2-{profile_id}-retreat",
                    instruction=(
                        "Simulate a post-treatment response by decreasing tumor area."
                    ),
                    primitive_id=primitive_id,
                    joint_area_budget={
                        "target_fraction": 0.08,
                        "min_fraction": 0.04,
                        "max_fraction": 0.12,
                        "tissue_min_fraction": 0.04,
                        "relative_tolerance": 0.02,
                        "fallback_policy": "max_feasible_below_target",
                        "capacity_floor_policy": "strict",
                        "minimum_effective_fraction": 0.04,
                    },
                    provenance={
                        **raw["provenance"],
                        "joint_mechanism_id": mechanism_id,
                        "joint_primitive_id": primitive_id,
                    },
                )
                case, intent = bind_semantic_intent(
                    raw, RuleBasedSemanticParser()
                )
                self.assertEqual(intent.treatment_context, "post_treatment")
                result = JointPathologyEditWorkflow(
                    tissue_planner=MultiInterfaceResearchTissuePlanner(),
                    joint_planner=HeuristicJointPlanner(),
                    critic=_ApprovingJointCritic(),
                ).run(case, output_root=root / "p2-retreat")
                self.assertEqual(
                    result.status, "selected_research", result.abstain_reasons
                )
                change = result.condition.tissue_change
                source_tissue = np.load(
                    case.source_tissue_mask_uri, allow_pickle=False
                )
                self.assertEqual(set(np.unique(source_tissue[change])), {1})
                self.assertEqual(
                    set(np.unique(result.condition.target_tissue_mask[change])),
                    {host_fine_id},
                )
                manifest = json.loads(
                    Path(result.artifact_paths["handoff_manifest"]).read_text(
                        encoding="utf-8"
                    )
                )
                claim_text = " ".join(
                    [
                        *manifest["render_expectations"],
                        *manifest["render_vetoes"],
                    ]
                ).casefold()
                self.assertNotIn("major pathologic response", claim_text)
                self.assertNotIn("complete response achieved", claim_text)

    def test_lung_generic_immune_compartment_turnover_executes_both_directions(self):
        for primitive_id in (
            "generic-immune-infiltrate-increase-v1",
            "generic-immune-infiltrate-decrease-v1",
        ):
            with (
                self.subTest(primitive=primitive_id),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                source = _write_breast_immune_case(root, primitive=primitive_id)
                case = replace(
                    source,
                    case_id="lung-" + primitive_id,
                    pathology_domain_id="lung-carcinoma-v1",
                    annotation_profile_id="ignite-semantic-v1",
                    cell_population_profile_id="lung-cellvit-source-first-v1",
                    provenance={
                        **source.provenance,
                        "preprocessing_revision": "synthetic-ignite-immune-v1",
                        "source_site": "lung",
                        "specimen_type": "resection",
                        "joint_mechanism_id": (
                            "lung-generic-immune-compartment-turnover"
                        ),
                        "joint_primitive_id": primitive_id,
                    },
                )
                result = JointPathologyEditWorkflow(
                    tissue_planner=MultiInterfaceResearchTissuePlanner(),
                    joint_planner=HeuristicJointPlanner(),
                    critic=_ApprovingJointCritic(),
                ).run(case, output_root=root / "lung-immune")
                self.assertEqual(
                    result.status, "selected_research", result.abstain_reasons
                )
                reports = json.loads(
                    Path(result.artifact_paths["joint_gate_reports.json"]).read_text(
                        encoding="utf-8"
                    )
                )
                selected = next(
                    item
                    for item in reports
                    if item["candidate_id"] == result.selected_candidate_id
                )
                mechanism = next(
                    check
                    for check in selected["checks"]
                    if check["check_id"]
                    == "mechanism_postcondition:lung-generic-immune-compartment-turnover"
                )
                self.assertTrue(mechanism["passed"])
                self.assertTrue(
                    mechanism["metrics"]["subcheck_results"][
                        "stroma_immune_transition_only"
                    ]
                )

    def test_p2_neoplastic_and_generic_inflammatory_abundance_additions_execute(self):
        fixtures = (
            (
                "lung-carcinoma-v1",
                "ignite-semantic-v1",
                "lung-cellvit-source-first-v1",
                2,
                "lung-local-population-modulation",
                {"source_site": "lung", "specimen_type": "resection"},
            ),
            (
                "melanoma-v1",
                "puma-semantic-v1",
                "melanoma-cellvit-source-first-v1",
                2,
                "melanoma-local-population-modulation",
                {"source_site": "skin", "primary_or_metastatic": "primary"},
            ),
            (
                "oral-squamous-cell-carcinoma-v1",
                "orca-semantic-v1",
                "oral-scc-cellvit-source-first-v1",
                7,
                "oral-scc-local-population-modulation",
                {},
            ),
        )
        for (
            domain_id,
            profile_id,
            population_id,
            host_fine_id,
            mechanism_id,
            required_provenance,
        ) in fixtures:
            for primitive_id, target_class in (
                ("neoplastic-cell-abundance-increase-v1", 1),
                ("generic-inflammatory-cell-abundance-increase-v1", 2),
            ):
                with (
                    self.subTest(profile=profile_id, primitive=primitive_id),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    root = Path(directory)
                    source = _write_synthetic_case(
                        root, size=256, tumor_radius=70
                    )
                    if target_class == 2:
                        source = _make_stroma_multiclass(source)
                    case = _as_organ_profile_case(
                        source,
                        pathology_domain_id=domain_id,
                        annotation_profile_id=profile_id,
                        cell_population_profile_id=population_id,
                        tumor_fine_id=1,
                        host_fine_id=host_fine_id,
                        mechanism_id=mechanism_id,
                        primitive_id=primitive_id,
                        required_provenance={
                            "preprocessing_revision": "synthetic-p2-population-v1",
                            **required_provenance,
                        },
                    )
                    case = replace(
                        case,
                        case_id=f"p2-{profile_id}-{primitive_id}",
                        joint_area_budget=None,
                        cell_count_extent_budget=CellCountExtentBudget(
                            12,
                            12,
                            15,
                            96,
                            0,
                            128,
                            minimum_effect_span_px=66,
                            minimum_effect_foci=2,
                        ),
                        provenance={
                            **case.provenance,
                            "target_cell_class_ids": [target_class],
                        },
                    )
                    source_tissue = np.load(
                        case.source_tissue_mask_uri, allow_pickle=False
                    )
                    source_nuclei = np.asarray(
                        Image.open(case.source_nuclei_mask_uri)
                    )
                    result = JointPathologyEditWorkflow(
                        tissue_planner=HeuristicInterfacePlanner(),
                        joint_planner=HeuristicJointPlanner(),
                        critic=_ApprovingJointCritic(),
                    ).run(case, output_root=root / "p2-population")
                    self.assertEqual(
                        result.status,
                        "selected_research",
                        result.abstain_reasons,
                    )
                    np.testing.assert_array_equal(
                        result.condition.target_tissue_mask, source_tissue
                    )
                    source_count = len(tuple(iter_instances(source_nuclei)))
                    target_count = len(
                        tuple(iter_instances(result.condition.target_nuclei_mask))
                    )
                    self.assertGreaterEqual(target_count - source_count, 12)
                    self.assertLessEqual(target_count - source_count, 15)

    def test_p2_local_clearance_is_digest_bound_and_roi_contained(self):
        fixtures = (
            (
                "lung-carcinoma-v1",
                "ignite-semantic-v1",
                "lung-cellvit-source-first-v1",
                2,
                "lung-local-tumor-clearance",
                {"source_site": "lung", "specimen_type": "resection"},
            ),
            (
                "melanoma-v1",
                "puma-semantic-v1",
                "melanoma-cellvit-source-first-v1",
                2,
                "melanoma-local-tumor-clearance",
                {"source_site": "skin", "primary_or_metastatic": "primary"},
            ),
            (
                "oral-squamous-cell-carcinoma-v1",
                "orca-semantic-v1",
                "oral-scc-cellvit-source-first-v1",
                7,
                "oral-scc-local-carcinoma-clearance",
                {},
            ),
        )
        for (
            domain_id,
            profile_id,
            population_id,
            host_fine_id,
            mechanism_id,
            required_provenance,
        ) in fixtures:
            with (
                self.subTest(profile=profile_id),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                primitive_id = "local-invasive-clearance-v1"
                case = _as_organ_profile_case(
                    _write_synthetic_case(root),
                    pathology_domain_id=domain_id,
                    annotation_profile_id=profile_id,
                    cell_population_profile_id=population_id,
                    tumor_fine_id=1,
                    host_fine_id=host_fine_id,
                    mechanism_id=mechanism_id,
                    primitive_id=primitive_id,
                    required_provenance={
                        "preprocessing_revision": "synthetic-p2-clearance-v1",
                        **required_provenance,
                    },
                )
                roi = np.zeros((128, 128), dtype=np.uint8)
                roi[30:99, 64:105] = 255
                roi_path = root / "local_clearance_roi.png"
                Image.fromarray(roi).save(roi_path)
                roi_digest = _sha(roi_path)
                tissue_digest = _sha(Path(case.source_tissue_mask_uri))
                case = replace(
                    case,
                    case_id=f"p2-{profile_id}-local-clearance",
                    instruction="Clear tumor in this local ROI.",
                    joint_area_budget=JointAreaBudget(
                        0.04, 0.02, 0.07, 0.02
                    ),
                    auxiliary_structure_uris={
                        "local_clearance_roi": str(roi_path)
                    },
                    provenance={
                        **case.provenance,
                        "available_auxiliary_structures": [
                            "local_clearance_roi"
                        ],
                        "auxiliary_structure_sha256": {
                            "local_clearance_roi": roi_digest
                        },
                        "auxiliary_structure_provenance": {
                            "local_clearance_roi": {
                                "producer_id": "synthetic-user-roi",
                                "producer_version": "v1",
                                "observation_scope": "user_roi",
                                "source_tissue_mask_sha256": tissue_digest,
                                "output_sha256": roi_digest,
                            }
                        },
                    },
                )
                result = JointPathologyEditWorkflow(
                    tissue_planner=MultiInterfaceResearchTissuePlanner(),
                    joint_planner=HeuristicJointPlanner(),
                    critic=_ApprovingJointCritic(),
                ).run(case, output_root=root / "p2-clearance")
                self.assertEqual(
                    result.status, "selected_research", result.abstain_reasons
                )
                changed = result.condition.tissue_change
                self.assertTrue(np.any(changed))
                self.assertFalse(np.any(changed & ~(roi > 0)))

    def test_cross_organ_cord_without_receiving_interface_abstains_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primitive_id = "infiltrative-nest-cord-extension-v1"
            case = _as_organ_profile_case(
                _write_synthetic_case(root),
                pathology_domain_id="lung-carcinoma-v1",
                annotation_profile_id="ignite-semantic-v1",
                cell_population_profile_id="lung-cellvit-source-first-v1",
                tumor_fine_id=1,
                host_fine_id=2,
                mechanism_id="lung-stromal-invasive-front",
                primitive_id=primitive_id,
                required_provenance={
                    "preprocessing_revision": "synthetic-cord-failure-v1",
                    "source_site": "lung",
                    "specimen_type": "resection",
                },
            )
            tissue_path = Path(case.source_tissue_mask_uri)
            tissue = np.load(tissue_path, allow_pickle=False)
            tissue[tissue == 2] = 5
            np.save(tissue_path, tissue, allow_pickle=False)
            tissue_digest = _sha(tissue_path)
            case = replace(
                case,
                case_id="lung-cord-no-receiving-interface",
                joint_area_budget=JointAreaBudget(0.025, 0.01, 0.04, 0.01),
                provenance={
                    **case.provenance,
                    "source_tissue_mask_sha256": tissue_digest,
                    "original_label_map_digest": tissue_digest,
                },
            )
            result = JointPathologyEditWorkflow(
                tissue_planner=MultiInterfaceResearchTissuePlanner(),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
            ).run(case, output_root=root / "cord-no-candidate")
            self.assertEqual(result.status, "abstained")
            reason = " ".join(result.abstain_reasons)
            self.assertIn("interface", reason.casefold())
            self.assertNotIn("deterministic_replan_stalled", reason)

    def test_cross_organ_scatter_and_cluster_without_legal_placement_fail_packing(self):
        for primitive_id in (
            "peritumoral-neoplastic-scatter-increase-v1",
            "peritumoral-small-cluster-increase-v1",
        ):
            with (
                self.subTest(primitive=primitive_id),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                case = _as_organ_profile_case(
                    _write_synthetic_case(root),
                    pathology_domain_id="oral-squamous-cell-carcinoma-v1",
                    annotation_profile_id="orca-semantic-v1",
                    cell_population_profile_id=(
                        "oral-scc-cellvit-source-first-v1"
                    ),
                    tumor_fine_id=1,
                    host_fine_id=7,
                    mechanism_id="oral-scc-dispersed-invasive-front",
                    primitive_id=primitive_id,
                    required_provenance={
                        "preprocessing_revision": "synthetic-packing-failure-v1"
                    },
                )
                tissue = np.load(case.source_tissue_mask_uri, allow_pickle=False)
                nuclei_path = Path(case.source_nuclei_mask_uri)
                nuclei = np.asarray(Image.open(nuclei_path)).copy()
                nuclei[tissue == 7] = 3
                Image.fromarray(nuclei).save(nuclei_path)
                nuclei_digest = _sha(nuclei_path)
                provenance = {
                    **case.provenance,
                    "source_nuclei_mask_sha256": nuclei_digest,
                    "original_instance_mask_digest": nuclei_digest,
                }
                provenance.pop("source_nuclei_instances_sha256", None)
                case = replace(
                    case,
                    case_id="orca-no-placement-" + primitive_id,
                    source_nuclei_instances_uri=None,
                    joint_area_budget=None,
                    cell_count_extent_budget=CellCountExtentBudget(
                        8,
                        6,
                        10,
                        48,
                        4,
                        48,
                        minimum_effect_span_px=20,
                        minimum_effect_foci=(
                            3 if "scatter" in primitive_id else 2
                        ),
                    ),
                    provenance=provenance,
                )
                result = JointPathologyEditWorkflow(
                    tissue_planner=HeuristicInterfacePlanner(),
                    joint_planner=HeuristicJointPlanner(),
                    critic=_ApprovingJointCritic(),
                ).run(case, output_root=root / "no-placement")
                self.assertEqual(result.status, "abstained")
                reason = " ".join(result.abstain_reasons)
                self.assertIn(
                    "exact_complete_footprint_packing_capacity_shortfall",
                    reason,
                )
                self.assertNotIn("deterministic_replan_stalled", reason)

    def test_cross_organ_abundance_decrease_unreachable_quota_abstains(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primitive_id = "neoplastic-cell-abundance-decrease-v1"
            case = _as_panda_case(
                _write_synthetic_case(root, size=256, tumor_radius=70),
                fine_id=10,
                mechanism_id="prostate-local-population-modulation",
                primitive_id=primitive_id,
            )
            source_tissue = np.load(
                case.source_tissue_mask_uri, allow_pickle=False
            )
            source_nuclei = np.asarray(Image.open(case.source_nuclei_mask_uri))
            scene = build_joint_scene_analysis(
                source_tissue,
                source_nuclei,
                schema=MaskProfileSchema.from_reference_profile("PANDA"),
                pixel_size_um=case.pixel_size_um,
                nuclei_instances_path=case.source_nuclei_instances_uri,
            )
            interface = next(
                item
                for item in scene.tissue.graph.interfaces
                if {item.source_label, item.target_label} == {"Tumor", "Stroma"}
            )
            labels = {
                item.component_id: item.label
                for item in scene.tissue.graph.components
            }
            tumor_component_id = next(
                component_id
                for component_id in (
                    interface.source_component_id,
                    interface.target_component_id,
                )
                if labels[component_id] == "Tumor"
            )
            zone = next(
                item
                for item in scene.population.zones
                if item.zone_kind == "component"
                and item.tissue_component_id == tumor_component_id
            )
            case = replace(
                case,
                case_id="panda-unreachable-depletion-quota",
                joint_area_budget=None,
                cell_count_extent_budget=CellCountExtentBudget(
                    200,
                    200,
                    210,
                    96,
                    0,
                    128,
                    minimum_effect_span_px=66,
                ),
                provenance={
                    **case.provenance,
                    "target_cell_class_ids": [1],
                    "joint_population_zone_id": zone.zone_id,
                    "cellularity_depletion_anchor": {
                        "type": "interface",
                        "interface_ids": [interface.interface_id],
                        "anchor_ids": [interface.anchor_segment_ids[0]],
                        "observation": "deterministic mask interface",
                        "confidence": 1.0,
                    },
                },
            )
            result = JointPathologyEditWorkflow(
                tissue_planner=HeuristicInterfacePlanner(),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
            ).run(case, output_root=root / "unreachable-quota")
            self.assertEqual(result.status, "abstained")
            reason = " ".join(result.abstain_reasons)
            self.assertTrue(
                "quota" in reason
                or "capacity" in reason
                or "does not contain 200" in reason,
                reason,
            )
            self.assertNotIn("deterministic_replan_stalled", reason)

    def test_cross_organ_necrosis_without_executor_survivor_abstains(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primitive_id = "necrosis-appearance-v1"
            source = _write_necrosis_case(root, primitive=primitive_id)
            nuclei_path = Path(source.source_nuclei_mask_uri)
            nuclei = np.asarray(Image.open(nuclei_path)).copy()
            nuclei[nuclei > 0] = 1
            Image.fromarray(nuclei).save(nuclei_path)
            instances_path = Path(source.source_nuclei_instances_uri)
            payload = json.loads(instances_path.read_text(encoding="utf-8"))
            for item in payload["nuc"].values():
                item["type"] = 1
            instances_path.write_text(json.dumps(payload), encoding="utf-8")
            case = replace(
                source,
                case_id="lung-necrosis-no-executor-survivor",
                pathology_domain_id="lung-carcinoma-v1",
                annotation_profile_id="ignite-semantic-v1",
                cell_population_profile_id="lung-cellvit-source-first-v1",
                provenance={
                    **source.provenance,
                    "source_site": "lung",
                    "specimen_type": "resection",
                    "preprocessing_revision": "synthetic-no-executor-v1",
                    "joint_mechanism_id": (
                        "lung-intratumoral-necrosis-turnover"
                    ),
                    "joint_primitive_id": primitive_id,
                    "source_nuclei_mask_sha256": _sha(nuclei_path),
                    "source_nuclei_instances_sha256": _sha(instances_path),
                    "original_instance_mask_digest": _sha(instances_path),
                },
            )
            result = JointPathologyEditWorkflow(
                tissue_planner=MultiInterfaceResearchTissuePlanner(),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
            ).run(case, output_root=root / "no-necrosis-executor")
            self.assertEqual(result.status, "abstained")
            reason = " ".join(result.abstain_reasons)
            self.assertIn("no nuclei-safe executable interface", reason)
            self.assertNotIn("deterministic_replan_stalled", reason)

    def test_cross_organ_fragmentation_underfill_has_no_replan_stall(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primitive_id = "residual-tumor-fragmentation-v1"
            mechanism_id = "prostate-operational-tumor-retreat"
            source = _as_panda_case(
                _write_synthetic_case(root),
                fine_id=10,
                mechanism_id=mechanism_id,
                primitive_id=primitive_id,
            )
            raw = source.to_metadata()
            raw.update(
                case_id="panda-residual-packing-underfill",
                instruction="Fragment the residual tumor after treatment.",
                primitive_id=primitive_id,
                joint_area_budget={
                    "target_fraction": 0.18,
                    "min_fraction": 0.12,
                    "max_fraction": 0.25,
                    "tissue_min_fraction": 0.12,
                    "relative_tolerance": 0.02,
                    "fallback_policy": "max_feasible_below_target",
                    "capacity_floor_policy": "strict",
                    "minimum_effective_fraction": 0.12,
                },
                provenance={
                    **raw["provenance"],
                    "joint_mechanism_id": mechanism_id,
                    "joint_primitive_id": primitive_id,
                },
            )
            case, _intent = bind_semantic_intent(
                raw, RuleBasedSemanticParser()
            )
            result = JointPathologyEditWorkflow(
                tissue_planner=MultiInterfaceResearchTissuePlanner(),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
            ).run(case, output_root=root / "fragmentation-underfill")
            self.assertEqual(result.status, "abstained")
            reason = " ".join(result.abstain_reasons)
            self.assertIn("TopologySafeAreaUnderfillError", reason)
            self.assertIn("minimum=1967, realized=1551", reason)
            self.assertNotIn("deterministic_replan_stalled", reason)

    def test_meta_eval_rejects_missing_mechanism_binding_and_required_roi(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            growth_root = root / "growth-input"
            growth_root.mkdir()
            growth = _as_organ_profile_case(
                _write_synthetic_case(growth_root),
                pathology_domain_id="lung-carcinoma-v1",
                annotation_profile_id="ignite-semantic-v1",
                cell_population_profile_id="lung-cellvit-source-first-v1",
                tumor_fine_id=1,
                host_fine_id=2,
                mechanism_id="lung-solid-squamous-growth",
                primitive_id="cohesive-boundary-expansion-v1",
                required_provenance={
                    "preprocessing_revision": "synthetic-eval-v1",
                    "source_site": "lung",
                    "specimen_type": "resection",
                },
            )
            unbound_provenance = dict(growth.provenance)
            unbound_provenance.pop("joint_mechanism_id", None)
            growth = replace(
                growth,
                case_id="meta-eval-missing-mechanism",
                joint_area_budget=JointAreaBudget(0.08, 0.04, 0.12, 0.04),
                provenance=unbound_provenance,
            )
            workflow = JointPathologyEditWorkflow(
                tissue_planner=MultiInterfaceResearchTissuePlanner(),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
                config=JointWorkflowConfig(
                    require_evaluation_input_bindings=True
                ),
            )
            result = workflow.run(growth, output_root=root / "missing-mechanism")
            self.assertEqual(result.status, "abstained")
            self.assertIn(
                "explicit joint_mechanism_id binding",
                " ".join(result.abstain_reasons),
            )

            clearance_root = root / "clearance-input"
            clearance_root.mkdir()
            clearance = _as_organ_profile_case(
                _write_synthetic_case(clearance_root),
                pathology_domain_id="lung-carcinoma-v1",
                annotation_profile_id="ignite-semantic-v1",
                cell_population_profile_id="lung-cellvit-source-first-v1",
                tumor_fine_id=1,
                host_fine_id=2,
                mechanism_id="lung-local-tumor-clearance",
                primitive_id="local-invasive-clearance-v1",
                required_provenance={
                    "preprocessing_revision": "synthetic-eval-v1",
                    "source_site": "lung",
                    "specimen_type": "resection",
                },
            )
            clearance = replace(
                clearance,
                case_id="meta-eval-missing-roi",
                instruction="Clear tumor in this local ROI.",
                joint_area_budget=JointAreaBudget(0.04, 0.02, 0.07, 0.02),
            )
            result = workflow.run(
                clearance, output_root=root / "missing-roi"
            )
            self.assertEqual(result.status, "abstained")
            self.assertIn(
                "lacks required auxiliary authority: local_clearance_roi",
                " ".join(result.abstain_reasons),
            )

    def test_generic_glas_tumor_increase_does_not_fall_back_to_budding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_synthetic_case(root)
            raw = source.to_metadata()
            raw["instruction"] = "increase tumor"
            raw["primitive_id"] = "tumor-burden-increase-v1"
            # Make tissue-level burden infeasible while leaving a real
            # tumor/stroma interface for the contextual budding hypothesis.
            raw["joint_area_budget"] = {
                "target_fraction": 0.95,
                "min_fraction": 0.90,
                "max_fraction": 0.99,
                "tissue_min_fraction": 0.90,
                "relative_tolerance": 0.02,
                "fallback_policy": "max_feasible_below_target",
            }
            raw["provenance"] = {
                **raw["provenance"],
                "joint_mechanism_id": "colorectal-tumor-budding-front",
                "joint_primitive_id": (
                    "peritumoral-neoplastic-scatter-increase-v1"
                ),
            }
            case, _intent = bind_semantic_intent(
                raw, RuleBasedSemanticParser()
            )
            output_root = root / "generic-tumor-budding"
            result = JointPathologyEditWorkflow(
                tissue_planner=HeuristicInterfacePlanner(),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
            ).run(case, output_root=output_root)

            self.assertEqual(result.status, "abstained")
            self.assertIn(
                "no joint mechanism supports this primitive",
                result.abstain_reasons[0],
            )

    def test_local_population_primitives_use_component_contract(self):
        for primitive, expected_sign, explicit_class in (
            ("cell-type-abundance-increase-v1", 1, True),
            ("cell-type-abundance-decrease-v1", -1, True),
            ("cellularity-increase-v1", 1, False),
            ("cellularity-decrease-v1", -1, False),
        ):
            with (
                self.subTest(primitive=primitive),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                source = _write_synthetic_case(root)
                if primitive.startswith("cellularity-"):
                    source = _make_stroma_multiclass(source)
                population_zone = "pop:component:cmp:stroma:0001"
                depletion_anchor = None
                if primitive in {
                    "cellularity-decrease-v1",
                    "cell-type-abundance-decrease-v1",
                }:
                    population_zone = "pop:component:cmp:tumor:0001"
                    interface_id = "if:tumor:0001->stroma:0001:seg:0001"
                    depletion_anchor = {
                        "type": "interface",
                        "interface_ids": [interface_id],
                        "anchor_ids": [f"{interface_id}:anchor:0001"],
                        "observation": (
                            "visible tumor-stroma boundary supports a localized "
                            "phenomenologic cellularity transition"
                        ),
                        "confidence": 0.90,
                    }
                provenance = {
                    **source.provenance,
                    "joint_mechanism_id": "colorectal-local-population-modulation",
                    "joint_population_zone_id": population_zone,
                }
                if depletion_anchor is not None:
                    provenance["cellularity_depletion_anchor"] = depletion_anchor
                if explicit_class:
                    provenance["target_cell_class_ids"] = [
                        1
                        if primitive == "cell-type-abundance-decrease-v1"
                        else 3
                    ]
                case = replace(
                    source,
                    case_id="synthetic-" + primitive,
                    instruction=primitive,
                    primitive_id=primitive,
                    joint_area_budget=None,
                    cell_count_extent_budget=(
                        CellCountExtentBudget(
                            12,
                            12,
                            15,
                            96,
                            0,
                            96,
                            minimum_effect_span_px=66,
                        )
                        if primitive in {
                            "cellularity-decrease-v1",
                            "cellularity-increase-v1",
                            "cell-type-abundance-decrease-v1",
                            "cell-type-abundance-increase-v1",
                        }
                        else CellCountExtentBudget(3, 3, 3, 48, 0, 48)
                    ),
                    provenance=provenance,
                )
                result = JointPathologyEditWorkflow(
                    tissue_planner=HeuristicInterfacePlanner(),
                    joint_planner=HeuristicJointPlanner(),
                    critic=_ApprovingJointCritic(),
                ).run(case, output_root=root / "local-population")
                if primitive in {
                    "cellularity-decrease-v1",
                    "cell-type-abundance-decrease-v1",
                }:
                    self.assertEqual(result.status, "abstained")
                    self.assertTrue(
                        any(
                            "whole-instance stronger-core gradient" in reason
                            for reason in result.abstain_reasons
                        )
                    )
                    continue
                self.assertEqual(
                    result.status, "selected_research", result.abstain_reasons
                )
                source_count = len(
                    tuple(
                        iter_instances(
                            np.asarray(Image.open(case.source_nuclei_mask_uri))
                        )
                    )
                )
                target_count = len(
                    tuple(iter_instances(result.condition.target_nuclei_mask))
                )
                if primitive in {
                    "cell-type-abundance-increase-v1",
                    "cellularity-increase-v1",
                }:
                    self.assertGreaterEqual(target_count - source_count, 12)
                    self.assertLessEqual(target_count - source_count, 15)
                else:
                    self.assertEqual(target_count - source_count, expected_sign * 3)
                self.assertEqual(result.condition.ledger.tissue_pixels, 0)
                candidate_manifest = json.loads(
                    Path(result.artifact_paths["candidates.json"]).read_text(
                        encoding="utf-8"
                    )
                )
                trace = next(
                    item["tool_trace"]
                    for item in candidate_manifest
                    if item["candidate_id"] == result.selected_candidate_id
                )
                if primitive == "cellularity-increase-v1":
                    requested = {
                        int(key): value
                        for key, value in trace["class_requested_counts"].items()
                    }
                    self.assertEqual(set(requested), {2, 3})
                    self.assertEqual(sum(requested.values()), 12)
                if primitive in {
                    "cellularity-decrease-v1",
                    "cell-type-abundance-decrease-v1",
                }:
                    self.assertEqual(
                        trace["execution_engine"],
                        "deterministic_anchored_density_gradient_removal_v1",
                    )
                    self.assertFalse(trace["ranker_provenance"]["probnet_used"])
                    fractions = trace["depletion_removal_fractions_by_band"]
                    self.assertGreater(
                        fractions["core"], fractions["transition"]
                    )
                    self.assertGreater(fractions["transition"], 0)
                    self.assertEqual(fractions["outer_reference"], 0)
                    reports = json.loads(
                        (
                            root
                            / "local-population"
                            / case.case_id
                            / "joint_gate_reports.json"
                        ).read_text(encoding="utf-8")
                    )
                    gradient = next(
                        check
                        for report in reports
                        for check in report["checks"]
                        if check["check_id"]
                        == "cellularity_depletion_gradient"
                    )
                    self.assertTrue(gradient["passed"])
                    self.assertTrue(
                        gradient["metrics"]["outer_reference_unchanged"]
                    )
                if expected_sign > 0:
                    self.assertEqual(
                        trace["reference_shape_locality"],
                        "selected_tissue_component",
                    )

    def test_non_breast_cellularity_decrease_uses_compiler_owned_anchors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_synthetic_case(root)
            case = replace(
                source,
                case_id="synthetic-unanchored-cellularity-decrease",
                instruction="decrease local cellularity",
                primitive_id="cellularity-decrease-v1",
                joint_area_budget=None,
                cell_count_extent_budget=CellCountExtentBudget(
                    12,
                    12,
                    15,
                    96,
                    0,
                    96,
                    minimum_effect_span_px=66,
                ),
                provenance={
                    **source.provenance,
                    "joint_mechanism_id": (
                        "colorectal-local-population-modulation"
                    ),
                    "joint_population_zone_id": (
                        "pop:component:cmp:tumor:0001"
                    ),
                },
            )
            result = JointPathologyEditWorkflow(
                tissue_planner=HeuristicInterfacePlanner(),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
            ).run(case, output_root=root / "unanchored")
            self.assertEqual(result.status, "abstained")
            reason = " ".join(result.abstain_reasons)
            self.assertNotIn("explicit mask-graph depletion anchor", reason)
            self.assertIn(
                "cell-only pre-LLM portfolio has no exact-capacity survivor",
                reason,
            )

    def test_cell_only_budget_cannot_undercut_skill_minimum_effect(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_synthetic_case(root)
            interface_id = "if:tumor:0001->stroma:0001:seg:0001"
            case = replace(
                source,
                case_id="synthetic-below-skill-minimum",
                instruction="decrease tumor cellularity",
                primitive_id="cellularity-decrease-v1",
                joint_area_budget=None,
                cell_count_extent_budget=CellCountExtentBudget(
                    3, 3, 3, 48, 0, 48
                ),
                provenance={
                    **source.provenance,
                    "joint_mechanism_id": (
                        "colorectal-local-population-modulation"
                    ),
                    "joint_population_zone_id": (
                        "pop:component:cmp:tumor:0001"
                    ),
                    "cellularity_depletion_anchor": {
                        "type": "interface",
                        "interface_ids": [interface_id],
                        "anchor_ids": [f"{interface_id}:anchor:0001"],
                        "observation": "visible anchored cellularity transition",
                        "confidence": 0.90,
                    },
                },
            )
            result = JointPathologyEditWorkflow(
                tissue_planner=HeuristicInterfacePlanner(),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
            ).run(case, output_root=root / "below-minimum")
            self.assertEqual(result.status, "abstained")
            self.assertTrue(
                any(
                    "skill-owned minimum effect count" in reason
                    for reason in result.abstain_reasons
                )
            )

    def test_cellularity_increase_budget_cannot_undercut_effect_floor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _make_stroma_multiclass(_write_synthetic_case(root))
            case = replace(
                source,
                case_id="synthetic-cellularity-increase-below-floor",
                instruction="increase local cellularity",
                primitive_id="cellularity-increase-v1",
                joint_area_budget=None,
                cell_count_extent_budget=CellCountExtentBudget(
                    3, 3, 3, 48, 0, 48
                ),
                provenance={
                    **source.provenance,
                    "joint_mechanism_id": (
                        "colorectal-local-population-modulation"
                    ),
                    "joint_population_zone_id": (
                        "pop:component:cmp:stroma:0001"
                    ),
                },
            )
            result = JointPathologyEditWorkflow(
                tissue_planner=HeuristicInterfacePlanner(),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
            ).run(case, output_root=root / "cellularity-increase-below-floor")
            self.assertEqual(result.status, "abstained")
            self.assertIn(
                "skill-owned minimum effect count",
                " ".join(result.abstain_reasons),
            )

    def test_necrosis_appearance_and_resolution_bind_dead_viable_turnover(self):
        for primitive in ("necrosis-appearance-v1", "necrosis-resolution-v1"):
            with (
                self.subTest(primitive=primitive),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                source = _write_necrosis_case(root, primitive=primitive)
                result = JointPathologyEditWorkflow(
                    tissue_planner=HeuristicInterfacePlanner(),
                    joint_planner=HeuristicJointPlanner(),
                    critic=_ApprovingJointCritic(),
                    tissue_gates=_PassingTissueGateFixture(),
                ).run(source, output_root=root / "necrosis")
                self.assertEqual(
                    result.status, "selected_research", result.abstain_reasons
                )
                self.assertGreater(result.condition.ledger.tissue_pixels, 0)
                manifest = json.loads(
                    Path(result.artifact_paths["handoff_manifest"]).read_text(
                        encoding="utf-8"
                    )
                )
                checks = json.loads(
                    (
                        root / "necrosis" / source.case_id / "joint_gate_reports.json"
                    ).read_text(encoding="utf-8")
                )
                turnover = [
                    check
                    for report in checks
                    for check in report["checks"]
                    if check["check_id"] == "necrosis_cell_turnover" and check["passed"]
                ]
                self.assertTrue(turnover)
                self.assertIn(
                    manifest["mechanism_id"],
                    {"breast-intratumoral-necrosis-turnover"},
                )
                if primitive == "necrosis-appearance-v1":
                    self.assertEqual(
                        manifest["execution_contract"]["cell_plan"][
                            "baseline_mode"
                        ],
                        "regenerate_target_population",
                    )
                    self.assertTrue(
                        any(
                            "sparse ProbNet" in finding
                            for finding in manifest["render_expectations"]
                        )
                    )
                    added_classes = set(
                        turnover[0]["metrics"]["added_classes"]
                    )
                    self.assertTrue(added_classes)
                    self.assertTrue(added_classes.issubset({2, 4}))
                    self.assertFalse(
                        any(
                            "restored viable" in finding
                            for finding in manifest["render_expectations"]
                        )
                    )
                else:
                    self.assertTrue(
                        any(
                            "viable tumor" in finding
                            for finding in manifest["render_expectations"]
                        )
                    )
                    self.assertFalse(
                        any(
                            "nuclear debris" in finding
                            for finding in manifest["render_expectations"]
                        )
                    )
                generator_inputs, _, _ = build_frozen_generator_inputs(
                    result.artifact_paths["handoff_manifest"],
                    output_dir=root / "generator",
                    dataset="BCSS",
                )
                self.assertIn(
                    "Within the provided editable generation support",
                    generator_inputs.prompt,
                )
                for finding in manifest["render_expectations"]:
                    self.assertIn(finding, generator_inputs.prompt)

    def test_necrosis_appearance_without_existing_interface_abstains(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_necrosis_case(root, primitive="necrosis-appearance-v1")
            source = _remove_native_necrosis_interface(source)
            result = JointPathologyEditWorkflow(
                tissue_planner=HeuristicInterfacePlanner(),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
            ).run(source, output_root=root / "no-necrosis-interface")
            self.assertEqual(result.status, "abstained")
            self.assertTrue(
                any("interface" in reason.lower() for reason in result.abstain_reasons),
                result.abstain_reasons,
            )

    def test_necrosis_appearance_without_dead_reference_uses_inflammatory_probnet_population(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_necrosis_case(
                root,
                primitive="necrosis-appearance-v1",
            )
            nuclei_path = Path(source.source_nuclei_mask_uri)
            nuclei = np.asarray(Image.open(nuclei_path)).copy()
            nuclei[nuclei == 4] = 1
            Image.fromarray(nuclei).save(nuclei_path)
            instances_path = Path(source.source_nuclei_instances_uri)
            payload = json.loads(instances_path.read_text(encoding="utf-8"))
            for item in payload["nuc"].values():
                if item["type"] == 4:
                    item["type"] = 1
            instances_path.write_text(json.dumps(payload), encoding="utf-8")
            source = replace(
                source,
                case_id="synthetic-necrosis-no-class4-reference",
                provenance={
                    **source.provenance,
                    "source_nuclei_mask_sha256": _sha(nuclei_path),
                    "source_nuclei_instances_sha256": _sha(instances_path),
                    "original_instance_mask_digest": _sha(instances_path),
                },
            )
            result = JointPathologyEditWorkflow(
                tissue_planner=HeuristicInterfacePlanner(),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
                tissue_gates=_PassingTissueGateFixture(),
            ).run(source, output_root=root / "no-dead-cell-reference")
            self.assertEqual(
                result.status,
                "selected_research",
                result.abstain_reasons,
            )
            self.assertEqual(
                result.selected_candidate.ledger.added_instance_ids,
                (),
            )
            preflight = json.loads(
                (
                    root
                    / "no-dead-cell-reference"
                    / source.case_id
                    / "joint_nuclei_preflight.json"
                ).read_text(encoding="utf-8")
            )
            self.assertGreater(preflight["target_density_per_pixel"], 0.0)
            self.assertTrue(preflight["eligible_reference_ids"])
            for interface in preflight["interfaces"]:
                self.assertGreater(interface["required_add_count"], 0)
                self.assertNotIn(
                    "no_complete_same_class_reference_shape",
                    interface["reasons"],
                )
            self.assertFalse(np.any(result.condition.target_nuclei_mask == 4))
            self.assertTrue(np.any(result.condition.target_nuclei_mask == 2))
            candidates = json.loads(
                (
                    root
                    / "no-dead-cell-reference"
                    / source.case_id
                    / "candidates.json"
                ).read_text(encoding="utf-8")
            )
            trace = next(
                item["tool_trace"]
                for item in candidates
                if item["candidate_id"] == result.selected_candidate_id
            )
            self.assertEqual(
                trace["execution_engine"],
                "deterministic_research_layout_v1",
            )
            self.assertEqual(trace["target_cell_class"], 2)
            self.assertEqual(
                set(trace["compiled_cell_tool_program"]["target_classes"]),
                {2, 4},
            )
            self.assertTrue(trace["placements"])


def _case_stub(
    *,
    population="colorectal-cellvit-source-first-v1",
    budget=None,
    primitive="tumor-burden-increase-v1",
    cell_budget=None,
):
    return JointCaseContext(
        case_id="stub",
        instruction="increase tumor burden",
        source_image_uri="image.png",
        source_tissue_mask_uri="tissue.npy",
        source_nuclei_mask_uri="nuclei.png",
        pathology_domain_id="colorectal-adenocarcinoma-v1",
        annotation_profile_id="glas-gland-v1",
        cell_observation_profile_id="cellvit-five-class-v1",
        cell_population_profile_id=population,
        primitive_id=primitive,
        joint_area_budget=budget or JointAreaBudget(),
        seed=17,
        provenance={
            "source_image_sha256": "x",
            "source_tissue_mask_sha256": "y",
            "source_nuclei_mask_sha256": "z",
        },
        cell_count_extent_budget=cell_budget,
    )


def _breast_case_stub(*, budget=None) -> JointCaseContext:
    return replace(
        _case_stub(budget=budget),
        pathology_domain_id="breast-invasive-carcinoma-v1",
        annotation_profile_id="bcss-semantic-v1",
        cell_population_profile_id="breast-cellvit-source-first-v1",
    )


def _write_synthetic_case(
    root: Path, *, size: int = 128, tumor_radius: int = 30
) -> JointCaseContext:
    center = size // 2
    rows, cols = np.ogrid[:size, :size]
    tissue = np.full((size, size), 2, dtype=np.uint8)
    tumor = (
        (rows - center) ** 2 + (cols - center) ** 2 <= tumor_radius**2
    )
    tissue[tumor] = 12
    nuclei = np.zeros_like(tissue)
    native_instances = {}
    native_index = 0
    for y in range(10, size - 10, 11):
        for x in range(10, size - 10, 11):
            class_id = 1 if tumor[y, x] else 3
            nuclei[y - 1 : y + 2, x - 1 : x + 2] = class_id
            native_instances[str(native_index)] = {
                "type": class_id,
                "centroid": [y, x],
                "contour": [
                    [x - 1, y - 1],
                    [x + 1, y - 1],
                    [x + 1, y + 1],
                    [x - 1, y + 1],
                ],
            }
            native_index += 1
    tissue_path = root / "tissue.npy"
    nuclei_path = root / "nuclei.png"
    image_path = root / "image.png"
    instances_path = root / "instances.json"
    auxiliary_path = root / "gland_or_lumen_support.png"
    np.save(tissue_path, tissue, allow_pickle=False)
    Image.fromarray(nuclei).save(nuclei_path)
    image = np.full((size, size, 3), (222, 183, 202), dtype=np.uint8)
    image[tumor] = (172, 88, 130)
    Image.fromarray(image).save(image_path)
    instances_path.write_text(json.dumps({"nuc": native_instances}), encoding="utf-8")
    auxiliary = (
        (rows - center) ** 2 + (cols - center) ** 2 <= 2**2
    ).astype(np.uint8)
    Image.fromarray(auxiliary).save(auxiliary_path)
    provenance = {
        "source_image_sha256": _sha(image_path),
        "source_tissue_mask_sha256": _sha(tissue_path),
        "source_nuclei_mask_sha256": _sha(nuclei_path),
        "source_nuclei_instances_sha256": _sha(instances_path),
        "original_label_map_digest": _sha(tissue_path),
        "auxiliary_structure_sha256": {"gland_or_lumen_support": _sha(auxiliary_path)},
        "auxiliary_structure_provenance": {
            "gland_or_lumen_support": {
                "producer_id": "synthetic-test-auxiliary",
                "producer_version": "synthetic-test-auxiliary-v1",
                "observation_scope": "synthetic_fixture",
                "source_tissue_mask_sha256": _sha(tissue_path),
                "output_sha256": _sha(auxiliary_path),
                "structure_units": [
                    {
                        "unit_id": "fine:12:unit:0001",
                        "unit_type": "synthetic_malignant_gland_unit",
                        "fine_id": 12,
                        "area_px": int(np.count_nonzero(tumor)),
                        "bbox_xyxy": [
                            center - tumor_radius,
                            center - tumor_radius,
                            center + tumor_radius + 1,
                            center + tumor_radius + 1,
                        ],
                        "component_sha256": hashlib.sha256(
                            np.packbits(
                                tumor.astype(np.uint8), axis=None
                            ).tobytes()
                        ).hexdigest(),
                        "enclosed_space_ids": [
                            "fine:12:unit:0001:space:001"
                        ],
                    }
                ],
                "hierarchy_relations": [
                    {
                        "source_id": "fine:12:unit:0001:space:001",
                        "relation": "enclosed_space_of",
                        "target_id": "fine:12:unit:0001",
                    }
                ],
            }
        },
        "preprocessing_revision": "synthetic-glas-v1",
        "original_instance_mask_digest": _sha(instances_path),
        "patch_grade": "moderately_differentiated",
        "joint_mechanism_id": "colorectal-gland-forming-front",
        "available_auxiliary_structures": ["gland_or_lumen_support"],
    }
    return JointCaseContext(
        case_id="synthetic-joint",
        instruction="increase tumor burden",
        source_image_uri=str(image_path),
        source_tissue_mask_uri=str(tissue_path),
        source_nuclei_mask_uri=str(nuclei_path),
        source_nuclei_instances_uri=str(instances_path),
        auxiliary_structure_uris={"gland_or_lumen_support": str(auxiliary_path)},
        pathology_domain_id="colorectal-adenocarcinoma-v1",
        annotation_profile_id="glas-gland-v1",
        cell_observation_profile_id="cellvit-five-class-v1",
        cell_population_profile_id="colorectal-cellvit-source-first-v1",
        primitive_id="tumor-burden-increase-v1",
        joint_area_budget=JointAreaBudget(),
        seed=17,
        provenance=provenance,
        pixel_size_um=0.465,
    )


def _with_native_gland_instance_authority(
    source: JointCaseContext,
) -> JointCaseContext:
    """Bind a synthetic native gland-instance raster for GLaS annulus tests."""

    tissue = np.load(source.source_tissue_mask_uri, allow_pickle=False)
    tumor = np.isin(tissue, (11, 12, 13))
    columns = np.indices(tissue.shape)[1]
    midpoint = int(np.median(columns[tumor]))
    native = np.zeros(tissue.shape, dtype=np.uint16)
    native[tumor & (columns <= midpoint)] = 1
    native[tumor & (columns > midpoint)] = 2
    path = Path(source.source_tissue_mask_uri).with_name(
        "native_gland_instance_map.png"
    )
    Image.fromarray(native).save(path)
    digest = _sha(path)
    tissue_digest = _sha(Path(source.source_tissue_mask_uri))
    provenance = dict(source.provenance)
    provenance["available_auxiliary_structures"] = sorted(
        {
            *provenance.get("available_auxiliary_structures", []),
            "native_gland_instance_map",
        }
    )
    provenance["auxiliary_structure_sha256"] = {
        **provenance.get("auxiliary_structure_sha256", {}),
        "native_gland_instance_map": digest,
    }
    provenance["auxiliary_structure_provenance"] = {
        **provenance.get("auxiliary_structure_provenance", {}),
        "native_gland_instance_map": {
            "producer_id": "synthetic-native-gland-instance-fixture",
            "producer_version": "v1",
            "observation_scope": "native_instance",
            "source_tissue_mask_sha256": tissue_digest,
            "output_sha256": digest,
        },
    }
    return replace(
        source,
        auxiliary_structure_uris={
            **source.auxiliary_structure_uris,
            "native_gland_instance_map": str(path),
        },
        provenance=provenance,
    )


def _as_organ_profile_case(
    source: JointCaseContext,
    *,
    pathology_domain_id: str,
    annotation_profile_id: str,
    cell_population_profile_id: str,
    tumor_fine_id: int,
    host_fine_id: int,
    mechanism_id: str,
    primitive_id: str,
    required_provenance: dict[str, str],
) -> JointCaseContext:
    """Rebind the generic circular fixture to one non-Breast mask profile."""

    tissue_path = Path(source.source_tissue_mask_uri)
    tissue = np.load(tissue_path, allow_pickle=False)
    tissue[tissue == 2] = host_fine_id
    tissue[np.isin(tissue, (11, 12, 13))] = tumor_fine_id
    np.save(tissue_path, tissue, allow_pickle=False)
    tissue_digest = _sha(tissue_path)
    provenance = {
        **source.provenance,
        **required_provenance,
        "source_tissue_mask_sha256": tissue_digest,
        "original_label_map_digest": tissue_digest,
        "joint_mechanism_id": mechanism_id,
        "joint_primitive_id": primitive_id,
        "available_auxiliary_structures": [],
    }
    provenance.pop("auxiliary_structure_sha256", None)
    provenance.pop("auxiliary_structure_provenance", None)
    return replace(
        source,
        instruction=primitive_id,
        primitive_id=primitive_id,
        pathology_domain_id=pathology_domain_id,
        annotation_profile_id=annotation_profile_id,
        cell_population_profile_id=cell_population_profile_id,
        auxiliary_structure_uris={},
        provenance=provenance,
    )


def _as_panda_case(
    source: JointCaseContext,
    *,
    fine_id: int,
    mechanism_id: str,
    primitive_id: str,
) -> JointCaseContext:
    return _as_organ_profile_case(
        source,
        pathology_domain_id="prostate-adenocarcinoma-v1",
        annotation_profile_id="panda-gleason-v1",
        cell_population_profile_id="prostate-cellvit-source-first-v1",
        tumor_fine_id=fine_id,
        host_fine_id=2,
        mechanism_id=mechanism_id,
        primitive_id=primitive_id,
        required_provenance={
            "preprocessing_revision": "synthetic-panda-v1",
            "provider": "synthetic-fixture",
        },
    )


def _as_breast_growth_case(source: JointCaseContext) -> JointCaseContext:
    """Use an executable broad-front mechanism for generic workflow tests."""

    tissue_path = Path(source.source_tissue_mask_uri)
    tissue = np.load(tissue_path, allow_pickle=False)
    tissue[np.isin(tissue, (11, 12, 13))] = 1
    np.save(tissue_path, tissue, allow_pickle=False)
    tissue_digest = _sha(tissue_path)
    provenance = {
        **source.provenance,
        "source_tissue_mask_sha256": tissue_digest,
        "original_label_map_digest": tissue_digest,
        "preprocessing_revision": "synthetic-bcss-v1",
        "joint_mechanism_id": "breast-annotation-anchored-boundary-growth",
        "joint_primitive_id": "tumor-burden-increase-v1",
        "available_auxiliary_structures": [],
    }
    provenance.pop("auxiliary_structure_sha256", None)
    provenance.pop("auxiliary_structure_provenance", None)
    return replace(
        source,
        pathology_domain_id="breast-invasive-carcinoma-v1",
        annotation_profile_id="bcss-semantic-v1",
        cell_population_profile_id="breast-cellvit-source-first-v1",
        auxiliary_structure_uris={},
        provenance=provenance,
    )


def _make_stroma_multiclass(source: JointCaseContext) -> JointCaseContext:
    """Turn alternating stroma instances into inflammatory cells in-place."""

    tissue = np.load(source.source_tissue_mask_uri, allow_pickle=False)
    nuclei_path = Path(source.source_nuclei_mask_uri)
    instances_path = Path(source.source_nuclei_instances_uri)
    nuclei = np.asarray(Image.open(nuclei_path)).copy()
    payload = json.loads(instances_path.read_text(encoding="utf-8"))
    changed = 0
    for item in payload["nuc"].values():
        contour = np.asarray(item["contour"], dtype=int)
        x = round(float(np.mean(contour[:, 0])))
        y = round(float(np.mean(contour[:, 1])))
        if tissue[y, x] != 2 or changed % 2:
            if tissue[y, x] == 2:
                changed += 1
            continue
        x0, y0 = contour.min(axis=0)
        x1, y1 = contour.max(axis=0)
        nuclei[y0 : y1 + 1, x0 : x1 + 1] = 2
        item["type"] = 2
        changed += 1
    if changed < 6:
        raise AssertionError("synthetic fixture lacks enough stroma instances")
    Image.fromarray(nuclei).save(nuclei_path)
    instances_path.write_text(json.dumps(payload), encoding="utf-8")
    return replace(
        source,
        provenance={
            **source.provenance,
            "source_nuclei_mask_sha256": _sha(nuclei_path),
            "source_nuclei_instances_sha256": _sha(instances_path),
            "original_instance_mask_digest": _sha(instances_path),
        },
    )


def _write_breast_immune_case(
    root: Path, *, primitive: str
) -> JointCaseContext:
    """BCSS fixture with an existing Stroma/Immune interface and two populations."""

    size = 160
    rows, cols = np.ogrid[:size, :size]
    tissue = np.full((size, size), 2, dtype=np.uint8)
    immune = (rows - 80) ** 2 + (cols - 80) ** 2 <= 34**2
    tissue[immune] = 4
    nuclei = np.zeros_like(tissue)
    native_instances = {}
    native_index = 0
    for y in range(8, size - 8, 9):
        for x in range(8, size - 8, 9):
            class_id = 2 if immune[y, x] else 3
            nuclei[y - 1 : y + 2, x - 1 : x + 2] = class_id
            native_instances[str(native_index)] = {
                "type": class_id,
                "centroid": [x, y],
                "contour": [
                    [x - 1, y - 1],
                    [x + 1, y - 1],
                    [x + 1, y + 1],
                    [x - 1, y + 1],
                ],
            }
            native_index += 1
    tissue_path = root / "immune-tissue.npy"
    nuclei_path = root / "immune-nuclei.png"
    image_path = root / "immune-image.png"
    instances_path = root / "immune-instances.json"
    np.save(tissue_path, tissue, allow_pickle=False)
    Image.fromarray(nuclei).save(nuclei_path)
    image = np.full((size, size, 3), (210, 182, 200), dtype=np.uint8)
    image[immune] = (174, 128, 170)
    Image.fromarray(image).save(image_path)
    instances_path.write_text(
        json.dumps({"nuc": native_instances}), encoding="utf-8"
    )
    provenance = {
        "source_image_sha256": _sha(image_path),
        "source_tissue_mask_sha256": _sha(tissue_path),
        "source_nuclei_mask_sha256": _sha(nuclei_path),
        "source_nuclei_instances_sha256": _sha(instances_path),
        "original_label_map_digest": _sha(tissue_path),
        "original_instance_mask_digest": _sha(instances_path),
        "preprocessing_revision": "synthetic-bcss-immune-v1",
        "joint_mechanism_id": "breast-generic-immune-compartment-turnover",
        "joint_primitive_id": primitive,
    }
    return JointCaseContext(
        case_id="synthetic-" + primitive,
        instruction=primitive,
        source_image_uri=str(image_path),
        source_tissue_mask_uri=str(tissue_path),
        source_nuclei_mask_uri=str(nuclei_path),
        source_nuclei_instances_uri=str(instances_path),
        pathology_domain_id="breast-invasive-carcinoma-v1",
        annotation_profile_id="bcss-semantic-v1",
        cell_observation_profile_id="cellvit-five-class-v1",
        cell_population_profile_id="breast-cellvit-source-first-v1",
        primitive_id=primitive,
        joint_area_budget=JointAreaBudget(
            target_fraction=0.03,
            min_fraction=0.01,
            max_fraction=0.06,
            tissue_min_fraction=0.01,
        ),
        seed=33,
        provenance=provenance,
        pixel_size_um=0.5,
    )


def _write_necrosis_case(root: Path, *, primitive: str) -> JointCaseContext:
    size = 128
    rows, cols = np.ogrid[:size, :size]
    tissue = np.full((size, size), 2, dtype=np.uint8)
    tumor = (rows - 64) ** 2 + (cols - 64) ** 2 <= 38**2
    necrosis = (rows - 64) ** 2 + (cols - 64) ** 2 <= 14**2
    tissue[tumor] = 1
    tissue[necrosis] = 3
    nuclei = np.zeros_like(tissue)
    native_instances = {}
    native_index = 0
    for y in range(10, size - 10, 9):
        for x in range(10, size - 10, 9):
            if necrosis[y, x]:
                # CellViT class 4 is rare in real BCSS necrosis.  Keep a
                # stable inflammatory (class 2) scaffold as well as sparse
                # observed-dead instances so either direction is executable.
                class_id = 4 if (x + y) % 18 == 0 else 2
            elif tumor[y, x]:
                class_id = 1
            else:
                class_id = 3
            nuclei[y - 1 : y + 2, x - 1 : x + 2] = class_id
            native_instances[str(native_index)] = {
                "type": class_id,
                "centroid": [x, y],
                "contour": [
                    [x - 1, y - 1],
                    [x + 1, y - 1],
                    [x + 1, y + 1],
                    [x - 1, y + 1],
                ],
            }
            native_index += 1
    tissue_path = root / "necrosis-tissue.npy"
    nuclei_path = root / "necrosis-nuclei.png"
    image_path = root / "necrosis-image.png"
    instances_path = root / "necrosis-instances.json"
    np.save(tissue_path, tissue, allow_pickle=False)
    Image.fromarray(nuclei).save(nuclei_path)
    image = np.full((size, size, 3), (220, 184, 201), dtype=np.uint8)
    image[tumor] = (170, 92, 132)
    image[necrosis] = (207, 155, 176)
    Image.fromarray(image).save(image_path)
    instances_path.write_text(json.dumps({"nuc": native_instances}), encoding="utf-8")
    provenance = {
        "source_image_sha256": _sha(image_path),
        "source_tissue_mask_sha256": _sha(tissue_path),
        "source_nuclei_mask_sha256": _sha(nuclei_path),
        "source_nuclei_instances_sha256": _sha(instances_path),
        "preprocessing_revision": "synthetic-bcss-v1",
        "original_label_map_digest": _sha(tissue_path),
        "joint_mechanism_id": "breast-intratumoral-necrosis-turnover",
    }
    return JointCaseContext(
        case_id="synthetic-" + primitive,
        instruction=primitive,
        source_image_uri=str(image_path),
        source_tissue_mask_uri=str(tissue_path),
        source_nuclei_mask_uri=str(nuclei_path),
        source_nuclei_instances_uri=str(instances_path),
        pathology_domain_id="breast-invasive-carcinoma-v1",
        annotation_profile_id="bcss-semantic-v1",
        cell_observation_profile_id="cellvit-five-class-v1",
        cell_population_profile_id="breast-cellvit-source-first-v1",
        primitive_id=primitive,
        joint_area_budget=JointAreaBudget(
            target_fraction=0.04,
            min_fraction=0.02,
            max_fraction=0.08,
            tissue_min_fraction=0.02,
            fallback_policy="max_feasible_below_target",
        ),
        seed=29,
        provenance=provenance,
        pixel_size_um=0.5,
    )


def _remove_native_necrosis_interface(source: JointCaseContext) -> JointCaseContext:
    """Make a tumor-only fixture while keeping all provenance internally valid."""

    tissue_path = Path(source.source_tissue_mask_uri)
    nuclei_path = Path(source.source_nuclei_mask_uri)
    instances_path = Path(source.source_nuclei_instances_uri)
    tissue = np.load(tissue_path, allow_pickle=False)
    tissue[tissue == 3] = 1
    np.save(tissue_path, tissue, allow_pickle=False)
    nuclei = np.asarray(Image.open(nuclei_path)).copy()
    nuclei[nuclei == 4] = 1
    Image.fromarray(nuclei).save(nuclei_path)
    payload = json.loads(instances_path.read_text(encoding="utf-8"))
    for item in payload["nuc"].values():
        if item["type"] == 4:
            item["type"] = 1
    instances_path.write_text(json.dumps(payload), encoding="utf-8")
    return replace(
        source,
        provenance={
            **source.provenance,
            "source_tissue_mask_sha256": _sha(tissue_path),
            "source_nuclei_mask_sha256": _sha(nuclei_path),
            "source_nuclei_instances_sha256": _sha(instances_path),
            "original_label_map_digest": _sha(tissue_path),
        },
    )


class StructuralHierarchyTests(unittest.TestCase):
    def test_colorectal_component_resolution_is_retry_only(self):
        repository = JointSkillRepository()
        bundle = SimpleNamespace(
            primitive=repository.primitives["tumor-burden-decrease-v1"],
            mechanism=repository.mechanisms[
                "colorectal-gland-forming-front"
            ],
        )

        initial = _effective_tissue_topology(
            bundle,
            primitive_id="tumor-burden-decrease-v1",
            retry_index=0,
            feedback_stage=None,
        )
        fallback = _effective_tissue_topology(
            bundle,
            primitive_id="tumor-burden-decrease-v1",
            retry_index=1,
            feedback_stage="planning_or_compilation",
        )

        self.assertFalse(initial["allow_source_component_resolution"])
        self.assertFalse(initial["fallback_activated"])
        self.assertFalse(fallback["allow_source_component_resolution"])
        self.assertFalse(fallback["allow_target_hole_resolution"])
        self.assertEqual(fallback["geometry_mode"], "interface_front")
        self.assertFalse(fallback["fallback_activated"])

    def test_prostate_mechanism_topology_and_shape_bounds_are_skill_owned(self):
        repository = JointSkillRepository()
        pattern3 = repository.mechanisms["prostate-pattern-3-growth"]
        pattern4 = repository.mechanisms["prostate-pattern-4-growth"]
        self.assertEqual(
            pattern3.tissue_program.target_component_merge_policy, "forbid"
        )
        self.assertEqual(
            pattern4.tissue_program.target_component_merge_policy,
            "selected_only",
        )
        self.assertGreater(
            pattern4.tissue_program.front.maximum_boundary_compactness,
            pattern3.tissue_program.front.maximum_boundary_compactness,
        )
        for mechanism in (pattern3, pattern4):
            self.assertEqual(
                mechanism.cell_program.seam_for(
                    "cohesive-boundary-expansion-v1"
                ).minimum_anchor_coverage_fraction,
                0.5,
            )
            self.assertNotIn(
                "tumor-burden-decrease-v1", mechanism.supported_primitives
            )

    def test_colorectal_gland_front_requires_a_directional_boundary_sector(self):
        mechanism = JointSkillRepository().mechanisms[
            "colorectal-gland-forming-front"
        ]
        front = mechanism.tissue_program.front
        self.assertTrue(front.directional_sector_required)
        self.assertEqual(front.maximum_selected_anchor_fraction, 0.8)
        self.assertEqual(front.minimum_unselected_anchor_count, 1)

        schema = MaskProfileSchema.from_reference_profile("GLaS")
        rows, cols = np.ogrid[:96, :96]
        source = np.full((96, 96), 2, dtype=np.uint8)
        source[(rows - 48) ** 2 + (cols - 48) ** 2 <= 25**2] = 11
        scene = build_scene_analysis(source, schema=schema)
        interface = next(
            item
            for item in scene.graph.interfaces
            if item.source_label == "Stroma" and item.target_label == "Tumor"
        )
        selected = _select_executable_anchor_ids(
            scene,
            interface=interface,
            required_pixels=2_000,
            maximum_depth_px=48,
            maximum_selected_anchor_fraction=(
                front.maximum_selected_anchor_fraction
            ),
            minimum_unselected_anchor_count=(
                front.minimum_unselected_anchor_count
            ),
        )
        self.assertGreater(len(selected), 0)
        self.assertLess(len(selected), len(interface.anchor_segment_ids))
        self.assertLessEqual(
            len(selected) / len(interface.anchor_segment_ids),
            front.maximum_selected_anchor_fraction,
        )

    def test_footprint_front_requires_one_long_shallow_anchor_sector(self):
        front = JointSkillRepository().mechanisms[
            "breast-post-treatment-invasive-regression"
        ].tissue_program.front

        self.assertEqual(front.minimum_selected_anchor_count, 3)
        self.assertLessEqual(front.maximum_depth_span_ratio, 0.65)

    def test_anchor_selection_follows_interface_path_not_opposing_cleft_bank(self):
        shape = (80, 48)
        interface_mask = np.zeros(shape, dtype=bool)
        interface_mask[8:70, 20] = True
        interface_mask[69, 20:25] = True
        interface_mask[8:70, 24] = True
        anchors = {}
        for anchor_id, row_slice, col in (
            ("left:1", slice(8, 20), 20),
            ("left:2", slice(20, 32), 20),
            ("left:3", slice(32, 44), 20),
            ("right:1", slice(8, 20), 24),
        ):
            mask = np.zeros(shape, dtype=bool)
            mask[row_slice, col] = True
            anchors[anchor_id] = mask
        interface = SimpleNamespace(
            interface_id="interface:u",
            source_component_id="source:1",
            anchor_segment_ids=tuple(anchors),
        )
        scene = SimpleNamespace(
            component_masks={"source:1": np.ones(shape, dtype=bool)},
            prohibited_region_masks={},
            interface_masks={"interface:u": interface_mask},
            anchor_masks=anchors,
            graph=SimpleNamespace(anchor_segments=()),
        )

        selected = _select_executable_anchor_ids(
            scene,
            interface=interface,
            required_pixels=1,
            maximum_depth_px=48,
            minimum_selected_anchor_count=3,
            preferred_anchor_ids=("left:1",),
        )

        self.assertEqual(selected, ("left:1", "left:2", "left:3"))

    def test_footprint_anchor_selection_prefers_shallow_external_front(self):
        shape = (80, 80)
        source = np.zeros(shape, dtype=bool)
        source[8:72, 8:72] = True
        # Form a deep stromal cleft while keeping the tumor connected around
        # its sides and base.  Its bottom is locally exposed just like the
        # outer edge, but is not a natural footprint-retreat front.
        source[8:56, 28:52] = False
        interface_mask = np.zeros(shape, dtype=bool)
        interface_mask[8, 8:37] = True
        interface_mask[56, 28:52] = True
        anchors = {}
        for anchor_id, row, columns in (
            ("external", 8, slice(8, 20)),
            ("internal", 56, slice(34, 46)),
        ):
            mask = np.zeros(shape, dtype=bool)
            mask[row, columns] = True
            anchors[anchor_id] = mask
        interface = SimpleNamespace(
            interface_id="interface:depth-choice",
            source_component_id="source:1",
            anchor_segment_ids=("external", "internal"),
        )
        scene = SimpleNamespace(
            component_masks={"source:1": source},
            prohibited_region_masks={},
            interface_masks={"interface:depth-choice": interface_mask},
            anchor_masks=anchors,
            graph=SimpleNamespace(anchor_segments=()),
        )

        selected = _select_executable_anchor_ids(
            scene,
            interface=interface,
            required_pixels=1,
            maximum_depth_px=48,
            prefer_shallow_front=True,
            preferred_anchor_ids=("internal",),
        )

        self.assertEqual(selected, ("external",))

    def test_pattern3_skill_forbids_selected_gland_component_merge(self):
        schema = MaskProfileSchema.from_reference_profile("PANDA")
        source = np.full((24, 32), 2, dtype=np.uint8)
        source[8:16, 3:10] = 8
        source[8:16, 22:29] = 8
        target = source.copy()
        target[11:13, 10:22] = 8
        scene = build_scene_analysis(source, schema=schema)
        interfaces = tuple(
            item
            for item in scene.graph.interfaces
            if item.source_label == "Stroma" and item.target_label == "Tumor"
        )
        self.assertEqual(len(interfaces), 2)
        plan = SimpleNamespace(
            source_labels=("Stroma",),
            target_label="Tumor",
            candidate_interfaces=interfaces,
            tool_program=SimpleNamespace(
                parameter_ranges={
                    "target_component_merge_policy": "forbid",
                    "allow_source_component_resolution": False,
                    "allow_target_hole_resolution": False,
                }
            ),
        )
        result = _check_edited_label_topology(
            SimpleNamespace(
                source_mask=source,
                candidate=SimpleNamespace(
                    target_mask=target,
                    change_region=source != target,
                ),
                schema=schema,
                scene=scene,
                plan=plan,
            )
        )
        self.assertFalse(result.passed)
        self.assertTrue(result.metrics["target_merge"])
        self.assertEqual(
            result.metrics["target_component_merge_policy"], "forbid"
        )

    def test_panda_pattern_gate_checks_tumor_side_of_burden_transition(self):
        schema = MaskProfileSchema.from_reference_profile("PANDA")
        required = SimpleNamespace(
            mechanism_required_fine_ids={"prostate-pattern-3-growth": (8,)}
        )

        def run(source, target):
            source = np.asarray(source, dtype=np.uint8)
            target = np.asarray(target, dtype=np.uint8)
            return _fine_pattern_preserved(
                SimpleNamespace(
                    source_tissue=source,
                    candidate=SimpleNamespace(
                        target_tissue_mask=target,
                        tissue_change=source != target,
                    ),
                    schema=schema,
                    bundle=SimpleNamespace(annotation_profile=required),
                    plan=SimpleNamespace(
                        selected_mechanism_id="prostate-pattern-3-growth"
                    ),
                )
            )

        increase = run([[8, 2]], [[8, 8]])
        decrease = run([[8, 8]], [[8, 2]])
        wrong_pattern = run([[8, 2]], [[8, 9]])
        self.assertTrue(increase.passed)
        self.assertTrue(decrease.passed)
        self.assertFalse(wrong_pattern.passed)
        self.assertEqual(
            increase.metrics["observed_changed_pattern_fine_ids"], [8]
        )

    def test_required_auxiliary_is_a_pre_candidate_tissue_exclusion(self):
        tissue = np.full((24, 24), 2, dtype=np.uint8)
        tissue[8:16, 8:16] = 8
        scene = build_scene_analysis(
            tissue,
            schema=MaskProfileSchema.from_reference_profile("PANDA"),
        )
        lumen = np.zeros_like(tissue, dtype=bool)
        lumen[10:14, 10:14] = True
        augmented = augment_tissue_scene_with_nuclei_preflight(
            scene,
            SimpleNamespace(
                protected_tissue_change_mask=np.zeros_like(tissue, dtype=bool)
            ),
            auxiliary_structure_masks={"native_pattern_and_lumen_map": lumen},
            required_auxiliary_structure_ids=(
                "native_pattern_and_lumen_map",
            ),
        )
        self.assertTrue(
            np.array_equal(
                augmented.prohibited_region_masks[
                    "joint:auxiliary:native_pattern_and_lumen_map"
                ],
                lumen,
            )
        )
        with self.assertRaises(JointContractError):
            augment_tissue_scene_with_nuclei_preflight(
                scene,
                SimpleNamespace(
                    protected_tissue_change_mask=np.zeros_like(
                        tissue, dtype=bool
                    )
                ),
                auxiliary_structure_masks={},
                required_auxiliary_structure_ids=(
                    "native_pattern_and_lumen_map",
                ),
            )

    def test_semantic_auxiliary_binds_gland_unit_lumen_and_component(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_synthetic_case(root)
            rows, cols = np.ogrid[:128, :128]
            radius = (rows - 64) ** 2 + (cols - 64) ** 2
            tissue = np.full((128, 128), 2, dtype=np.uint8)
            tissue[(radius >= 15**2) & (radius <= 31**2)] = 12
            np.save(source.source_tissue_mask_uri, tissue, allow_pickle=False)
            digest = _sha(Path(source.source_tissue_mask_uri))
            provenance = dict(source.provenance)
            provenance.pop("auxiliary_structure_sha256", None)
            provenance.pop("auxiliary_structure_provenance", None)
            provenance["source_tissue_mask_sha256"] = digest
            provenance["original_label_map_digest"] = digest
            case = replace(
                source,
                auxiliary_structure_uris={},
                provenance=provenance,
            )

            effective, produced = materialize_profile_auxiliaries(
                case,
                source_tissue=tissue,
                output_dir=root / "auxiliary",
            )
            self.assertEqual(
                [item.structure_id for item in produced],
                [
                    "gland_or_lumen_support",
                    "native_gland_instance_map",
                    "external_cellular_stroma_map",
                ],
            )
            with Image.open(
                effective.auxiliary_structure_uris[
                    "native_gland_instance_map"
                ]
            ) as instance_image:
                self.assertIn(instance_image.mode, {"I", "I;16"})
                instance_map = np.asarray(instance_image)
            self.assertEqual(set(np.unique(instance_map)), {0, 1})
            np.testing.assert_array_equal(instance_map > 0, tissue == 12)
            producer = effective.provenance[
                "auxiliary_structure_provenance"
            ]["gland_or_lumen_support"]
            self.assertEqual(producer["enclosed_space_count"], 1)
            scene = build_joint_scene_analysis(
                tissue,
                np.asarray(Image.open(source.source_nuclei_mask_uri)),
                schema=MaskProfileSchema.from_reference_profile("GlaS"),
                pixel_size_um=source.pixel_size_um,
                nuclei_instances_path=source.source_nuclei_instances_uri,
                auxiliary_structure_paths=effective.auxiliary_structure_uris,
                auxiliary_structure_provenance=effective.provenance[
                    "auxiliary_structure_provenance"
                ],
            )
            units = scene.structural_hierarchy["structure_units"]
            self.assertTrue(units)
            self.assertTrue(
                all(item["parent_tissue_component_id"] for item in units)
            )
            self.assertEqual(
                scene.structural_hierarchy["levels"],
                [
                    "structural_compartment",
                    "cellular_population",
                    "morphology",
                ],
            )
            self.assertTrue(
                scene.structural_hierarchy["cellular_populations"]
            )
            self.assertGreater(
                np.count_nonzero(
                    scene.auxiliary_structure_masks[
                        "gland_or_lumen_support"
                    ]
                ),
                0,
            )

    def test_puma_auxiliary_uses_only_explicit_epidermis_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_synthetic_case(root)
            tissue = np.full((128, 128), 2, dtype=np.uint8)
            tissue[8:15, :] = 5
            tissue[48:96, 40:88] = 1
            provenance = dict(source.provenance)
            provenance.pop("auxiliary_structure_sha256", None)
            provenance.pop("auxiliary_structure_provenance", None)
            case = replace(
                source,
                annotation_profile_id="puma-semantic-v1",
                auxiliary_structure_uris={},
                provenance=provenance,
            )

            effective, produced = materialize_profile_auxiliaries(
                case,
                source_tissue=tissue,
                output_dir=root / "puma-auxiliary",
            )

            self.assertEqual(
                [item.structure_id for item in produced],
                ["epidermis_or_junction_map"],
            )
            mask = np.asarray(
                Image.open(
                    effective.auxiliary_structure_uris[
                        "epidermis_or_junction_map"
                    ]
                )
            ) != 0
            np.testing.assert_array_equal(mask, tissue == 5)
            producer = effective.provenance[
                "auxiliary_structure_provenance"
            ]["epidermis_or_junction_map"]
            self.assertEqual(
                producer["protection_semantics"],
                "explicit_profile_structure",
            )
            self.assertFalse(producer["empty_map_is_valid_observation"])


class OtherOrganIndependentReviewBlockerTests(unittest.TestCase):
    def test_non_breast_generated_contracts_are_free_of_cross_organ_pollution(self):
        root = Path(__file__).parents[1]
        catalog = (
            root
            / "phase3_joint_edit_refine"
            / "skills"
            / "catalog"
            / "joint-mechanism"
        )
        forbidden = (
            "breast",
            "bcss",
            "dcis",
            "angioinvasion",
            "benign_duct",
            "breast.mask",
            "cap breast",
        )
        for path in sorted(catalog.glob("*/references/joint_contract.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("pathology_domain_id") == "breast-invasive-carcinoma-v1":
                continue
            rendered = json.dumps(payload, sort_keys=True).lower()
            with self.subTest(mechanism_id=payload.get("mechanism_id")):
                self.assertFalse(
                    {token for token in forbidden if token in rendered},
                    path,
                )

        cord_source = inspect.getsource(other_organ_generator._cord).lower()
        self.assertNotIn("breast", cord_source)
        self.assertNotIn("deepcopy", cord_source)

        for mechanism_id in (
            "prostate-operational-tumor-retreat",
            "lung-operational-tumor-retreat",
        ):
            contract = json.loads(
                (
                    catalog
                    / mechanism_id
                    / "references"
                    / "joint_contract.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                contract["tissue_program"]["mode"],
                "operational_tumor_retreat_to_profile_receiver",
            )
            positive_surface = json.dumps(
                {
                    "mechanism_id": contract["mechanism_id"],
                    "summary": contract["summary"],
                    "required_observations": contract["recognition_contract"][
                        "required_observations"
                    ],
                    "mode": contract["tissue_program"]["mode"],
                    "cell_program": contract["cell_program"],
                    "required_findings": contract["render_contract"][
                        "required_findings"
                    ],
                    "mask_guarantees": contract["render_contract"][
                        "mask_guarantees"
                    ],
                },
                sort_keys=True,
            ).lower()
            self.assertNotIn("fibrot", positive_surface)
            self.assertNotIn("tumor bed", positive_surface)
            self.assertNotIn("tumor-bed", positive_surface)
            self.assertEqual(
                contract["cell_program"]["halo_policy"],
                "operational_receiver_compatible_non_neoplastic_population",
            )

        old_ids = (
            "prostate-treatment-associated-fibrotic-replacement",
            "lung-treatment-associated-fibrotic-replacement",
        )
        generated_surfaces = (
            root / "phase3_joint_edit_refine" / "resources",
            root / "phase3_joint_edit_refine" / "skills",
        )
        for old_id in old_ids:
            self.assertTrue(
                all(
                    old_id not in path.read_text(encoding="utf-8", errors="ignore")
                    for surface in generated_surfaces
                    for path in surface.rglob("*")
                    if path.is_file()
                )
            )

    def test_orca_contracts_use_only_annotation_observable_other_tissue(self):
        root = (
            Path(__file__).parents[1]
            / "phase3_joint_edit_refine"
            / "skills"
            / "catalog"
            / "joint-mechanism"
        )
        dispersed = json.loads(
            (
                root
                / "oral-scc-dispersed-invasive-front"
                / "references"
                / "joint_contract.json"
            ).read_text(encoding="utf-8")
        )
        required = " ".join(
            dispersed["recognition_contract"]["required_observations"]
        ).lower()
        self.assertNotIn("strom", required)
        self.assertNotIn("verified dispersed front", required)
        self.assertNotIn("source evidence", required)
        self.assertEqual(
            set(dispersed["supported_primitives"]),
            {
                "peritumoral-neoplastic-scatter-increase-v1",
                "peritumoral-small-cluster-increase-v1",
            },
        )
        for label_contract in dispersed["tissue_program"][
            "primitive_label_contracts"
        ].values():
            self.assertEqual(label_contract["source_labels"], ["Other tissue"])
            self.assertEqual(label_contract["target_labels"], ["Other tissue"])

        local = json.loads(
            (
                root
                / "oral-scc-local-population-modulation"
                / "references"
                / "joint_contract.json"
            ).read_text(encoding="utf-8")
        )
        for label_contract in local["tissue_program"][
            "primitive_label_contracts"
        ].values():
            self.assertNotIn("Stroma", label_contract["source_labels"])
            self.assertNotIn("Stroma", label_contract["target_labels"])

    def test_p1_selection_binds_versioned_source_pool_and_all_authority_digests(self):
        resources = (
            Path(__file__).parents[1]
            / "phase3_joint_edit_refine"
            / "resources"
        )
        selection_path = resources / "p1_glas_panda_meta_eval_selection_v1.json"
        source_path = resources / "p1_glas_panda_source_case_pool_v1.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        source = json.loads(source_path.read_text(encoding="utf-8"))
        validate_selection(selection)
        self.assertEqual(selection["evaluation_count"], 21)
        self.assertEqual(
            sum(len(item["selected_cases"]) for item in selection["evaluations"]),
            105,
        )
        self.assertEqual(
            selection["source_manifest"],
            "phase3_joint_edit_refine/resources/"
            "p1_glas_panda_source_case_pool_v1.json",
        )
        self.assertEqual(selection["source_manifest_sha256"], _sha(source_path))
        self.assertEqual(
            selection["source_manifest_schema_version"],
            source["schema_version"],
        )
        self.assertEqual(len(source["cases"]), 10)
        self.assertTrue(
            all(
                not row["execution_allowed"]
                for evaluation in selection["evaluations"]
                for row in evaluation["selected_cases"]
            )
        )
        gland_scatter = next(
            item
            for item in selection["evaluations"]
            if item["annotation_profile_id"] == "glas-gland-v1"
            and item["primitive_id"]
            == "peritumoral-neoplastic-scatter-increase-v1"
        )
        self.assertIn(
            "native_gland_instance_map",
            gland_scatter["required_auxiliary_structures"],
        )
        self.assertTrue(
            all(
                "native_gland_instance_map"
                in row["missing_required_auxiliary_structures"]
                for row in gland_scatter["selected_cases"]
            )
        )
        clearance = next(
            item
            for item in selection["evaluations"]
            if item["primitive_id"] == "local-invasive-clearance-v1"
        )
        self.assertTrue(
            all(
                "local_clearance_roi"
                in row["missing_required_auxiliary_structures"]
                for row in clearance["selected_cases"]
            )
        )
        runtime = selection["runtime_authority"]
        self.assertFalse(runtime["all_required_digests_bound"])
        self.assertEqual(len(runtime["selection_generator_sha256"]), 64)
        tampered = json.loads(json.dumps(selection))
        tampered["runtime_authority"]["all_required_digests_bound"] = True
        with self.assertRaisesRegex(ValueError, "completeness declaration"):
            validate_selection(tampered)

    def test_cellularity_meaningful_effect_rebuilds_four_spatial_foci(self):
        nominal = 10.0
        adjacent_chain = [(index * 10.0, 10.0) for index in range(12)]
        adjacent = audit_cell_effect_foci(
            centers_xy=adjacent_chain,
            nominal_nucleus_diameter_px=nominal,
        )
        self.assertEqual(adjacent["focus_count"], 0)
        self.assertFalse(adjacent["no_chain_no_bridge_passed"])
        self.assertFalse(adjacent["trace_cluster_ids_used_as_evidence"])

        compact_single_focus = [
            (float(col), float(row))
            for row in (10, 15, 20)
            for col in (10, 15, 20, 22)
        ]
        compact = audit_cell_effect_foci(
            centers_xy=compact_single_focus,
            nominal_nucleus_diameter_px=nominal,
        )
        self.assertEqual(compact["focus_count"], 1)

        bridge = audit_cell_effect_foci(
            centers_xy=((0, 0), (10, 0), (20, 0), (30, 0)),
            nominal_nucleus_diameter_px=nominal,
        )
        self.assertEqual(bridge["focus_count"], 0)
        self.assertFalse(bridge["no_chain_no_bridge_passed"])

        three = audit_cell_effect_foci(
            centers_xy=((0, 0), (40, 0), (80, 0)),
            nominal_nucleus_diameter_px=nominal,
        )
        self.assertEqual(three["focus_count"], 3)
        self.assertLess(three["focus_count"], 4)

        four_groups = [
            (base + offset, row)
            for base, row in ((0, 0), (50, 40), (100, 0), (150, 40))
            for offset in (0, 5, 10)
        ]
        accepted = audit_cell_effect_foci(
            centers_xy=four_groups,
            nominal_nucleus_diameter_px=nominal,
        )
        self.assertTrue(accepted["spatial_focus_contract_passed"])
        self.assertEqual(accepted["reconstructed_complete_instance_count"], 12)
        self.assertEqual(accepted["focus_count"], 4)
        self.assertGreaterEqual(accepted["effect_center_span_px"], 6 * nominal)
        self.assertGreaterEqual(
            accepted["observed_minimum_inter_focus_distance_px"],
            accepted["minimum_inter_focus_separation_px"],
        )

        forged_incomplete_ledger = audit_cell_effect_foci(
            centers_xy=four_groups,
            nominal_nucleus_diameter_px=nominal,
            complete_instance_ledger_valid=False,
        )
        self.assertEqual(forged_incomplete_ledger["focus_count"], 0)

    def _panda_depletion_context(self, root: Path):
        primitive_id = "cellularity-decrease-v1"
        case = _as_panda_case(
            _write_synthetic_case(root, size=256, tumor_radius=70),
            fine_id=10,
            mechanism_id="prostate-local-population-modulation",
            primitive_id=primitive_id,
        )
        case = replace(
            case,
            case_id="panda-compiler-owned-depletion",
            joint_area_budget=None,
            cell_count_extent_budget=CellCountExtentBudget(
                12,
                12,
                15,
                96,
                0,
                128,
                minimum_effect_span_px=66,
                minimum_effect_foci=4,
            ),
        )
        workflow = JointPathologyEditWorkflow(
            tissue_planner=HeuristicInterfacePlanner(),
            joint_planner=HeuristicJointPlanner(),
            critic=_ApprovingJointCritic(),
        )
        source_tissue = np.load(case.source_tissue_mask_uri, allow_pickle=False)
        source_nuclei = np.asarray(Image.open(case.source_nuclei_mask_uri))
        schema = workflow.mask_skills.annotation_schema(case.annotation_profile_id)
        scene = build_joint_scene_analysis(
            source_tissue,
            source_nuclei,
            schema=schema,
            pixel_size_um=case.pixel_size_um,
            nuclei_instances_path=case.source_nuclei_instances_uri,
        )
        component_labels = {
            item.component_id: item.label
            for item in scene.tissue.graph.components
        }
        zone = next(
            item
            for item in scene.population.zones
            if item.zone_kind == "component"
            and component_labels.get(item.tissue_component_id) == "Tumor"
        )
        case = replace(
            case,
            provenance={
                **case.provenance,
                "joint_population_zone_id": zone.zone_id,
            },
        )
        bundle = workflow.joint_skills.compose(
            case=case,
            mechanism_id="prostate-local-population-modulation",
            available_checker_ids=workflow.joint_gates.available_checker_ids,
            production=False,
        )
        return workflow, case, source_tissue, source_nuclei, schema, scene, bundle

    def test_non_breast_depletion_anchor_is_compiler_owned_and_full_portfolio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                workflow,
                case,
                source_tissue,
                source_nuclei,
                schema,
                scene,
                bundle,
            ) = self._panda_depletion_context(root)
            malicious = replace(
                case,
                provenance={
                    **case.provenance,
                    "cellularity_depletion_anchor": {
                        "type": "interface",
                        "interface_ids": ["if:caller-single-candidate"],
                        "anchor_ids": ["anchor:caller-detached"],
                        "observation": "renamed H&E assertion",
                        "confidence": 1.0,
                    },
                },
            )

            with self.assertRaisesRegex(
                JointContractError, "compiler-owned depletion anchor"
            ):
                HeuristicJointPlanner().create_plan(
                    case=malicious,
                    scene=scene,
                    bundle=bundle,
                    tissue_plan=None,
                    image_paths=(),
                )

            interface = next(
                item
                for item in scene.tissue.graph.interfaces
                if {item.source_label, item.target_label} == {"Tumor", "Stroma"}
                and item.anchor_segment_ids
            )
            valid_capability = CompilerOwnedDepletionAnchor.issue(
                case=case,
                zone_id=case.provenance["joint_population_zone_id"],
                interface_ids=(interface.interface_id,),
                anchor_ids=(interface.anchor_segment_ids[0],),
            )
            detached = replace(valid_capability, binding_sha256="0" * 64)
            with self.assertRaisesRegex(JointContractError, "digest-detached"):
                HeuristicJointPlanner().create_plan(
                    case=case,
                    scene=scene,
                    bundle=bundle,
                    tissue_plan=None,
                    image_paths=(),
                    compiler_owned_depletion_anchor=detached,
                )

            unknown = CompilerOwnedDepletionAnchor.issue(
                case=case,
                zone_id=case.provenance["joint_population_zone_id"],
                interface_ids=("if:unknown",),
                anchor_ids=("anchor:unknown",),
            )
            with self.assertRaisesRegex(JointContractError, "unknown"):
                HeuristicJointPlanner().create_plan(
                    case=case,
                    scene=scene,
                    bundle=bundle,
                    tissue_plan=None,
                    image_paths=(),
                    compiler_owned_depletion_anchor=unknown,
                )

            def compile_portfolio(current_case):
                return workflow._compile_cell_only_candidate_portfolio(
                    case=current_case,
                    source_tissue=source_tissue,
                    source_nuclei=source_nuclei,
                    schema=schema,
                    scene=scene,
                    bundle=bundle,
                )

            baseline = compile_portfolio(case)
            adversarial = compile_portfolio(malicious)
            replay = compile_portfolio(case)
            self.assertGreater(len(baseline.choices), 1)
            baseline_bindings = tuple(
                (
                    item.certificate.candidate_id,
                    item.certificate.compiler_certificate_sha256,
                    item.certificate.authority_binding_sha256,
                )
                for item in baseline.choices
            )
            self.assertEqual(
                baseline_bindings,
                tuple(
                    (
                        item.certificate.candidate_id,
                        item.certificate.compiler_certificate_sha256,
                        item.certificate.authority_binding_sha256,
                    )
                    for item in adversarial.choices
                ),
            )
            self.assertEqual(
                baseline_bindings,
                tuple(
                    (
                        item.certificate.candidate_id,
                        item.certificate.compiler_certificate_sha256,
                        item.certificate.authority_binding_sha256,
                    )
                    for item in replay.choices
                ),
            )
            result = workflow.run(
                malicious,
                output_root=root / "positive-compiler-owned-depletion",
            )
            self.assertEqual(
                result.status,
                "selected_research",
                result.abstain_reasons,
            )

    def test_panda_residual_fragmentation_rebuilds_final_mask_topology(self):
        from tests.test_joint_breast_workflow import _write_breast_case

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primitive_id = "residual-tumor-fragmentation-v1"
            mechanism_id = "prostate-operational-tumor-retreat"
            source = _write_breast_case(
                root,
                instruction="Fragment residual tumor after treatment.",
                mechanism_id="breast-residual-disease-fragmentation",
            )
            tissue_path = Path(source.source_tissue_mask_uri)
            tissue = np.load(tissue_path, allow_pickle=False)
            tissue[tissue == 1] = 10
            np.save(tissue_path, tissue, allow_pickle=False)
            tissue_digest = _sha(tissue_path)

            nuclei = np.asarray(Image.open(source.source_nuclei_mask_uri))
            native_instances = {}
            for index, (instance_id, class_id, component) in enumerate(
                iter_instances(nuclei)
            ):
                rows, cols = np.nonzero(component)
                native_instances[str(index)] = {
                    "type": int(class_id),
                    "centroid": [float(rows.mean()), float(cols.mean())],
                    "contour": [
                        [int(cols.min()), int(rows.min())],
                        [int(cols.max()), int(rows.min())],
                        [int(cols.max()), int(rows.max())],
                        [int(cols.min()), int(rows.max())],
                    ],
                    "source_instance_id": instance_id,
                }
            instances_path = root / "panda_instances.json"
            instances_path.write_text(
                json.dumps({"nuc": native_instances}),
                encoding="utf-8",
            )
            instances_digest = _sha(instances_path)
            case = replace(
                source,
                case_id="panda-residual-fragmentation-positive",
                pathology_domain_id="prostate-adenocarcinoma-v1",
                annotation_profile_id="panda-gleason-v1",
                cell_population_profile_id="prostate-cellvit-source-first-v1",
                source_nuclei_instances_uri=str(instances_path),
                primitive_id=primitive_id,
                joint_area_budget=JointAreaBudget(
                    target_fraction=0.035,
                    min_fraction=0.03,
                    max_fraction=0.05,
                    tissue_min_fraction=0.03,
                ),
                provenance={
                    **source.provenance,
                    "source_tissue_mask_sha256": tissue_digest,
                    "original_label_map_digest": tissue_digest,
                    "source_nuclei_instances_sha256": instances_digest,
                    "original_instance_mask_digest": instances_digest,
                    "preprocessing_revision": "synthetic-panda-fragment-v1",
                    "provider": "synthetic-fixture",
                    "joint_mechanism_id": mechanism_id,
                    "joint_primitive_id": primitive_id,
                },
            )
            case, _intent = bind_semantic_intent(
                case.to_metadata(), RuleBasedSemanticParser()
            )
            result = JointPathologyEditWorkflow(
                tissue_planner=MultiInterfaceResearchTissuePlanner(),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
            ).run(case, output_root=root / "panda-fragmentation")
            self.assertEqual(
                result.status,
                "selected_research",
                result.abstain_reasons,
            )
            self.assertEqual(
                result.joint_plan.tissue_plan.tool_program.parameter_ranges[
                    "max_depth_span_ratio"
                ],
                4.0,
            )

            final_tumor = result.condition.target_tissue_mask == 10
            labels, component_count = ndimage.label(
                final_tumor,
                structure=np.ones((3, 3), dtype=bool),
            )
            sizes = [
                int(np.count_nonzero(labels == component_id))
                for component_id in range(1, component_count + 1)
            ]
            self.assertGreaterEqual(component_count, 2)
            self.assertLessEqual(component_count, 6)
            self.assertGreaterEqual(min(sizes), 96)
            self.assertEqual(
                ndimage.label(final_tumor, structure=np.ones((3, 3)))[1],
                component_count,
            )
            reports = json.loads(
                Path(result.artifact_paths["joint_gate_reports.json"]).read_text(
                    encoding="utf-8"
                )
            )
            selected = next(
                item
                for item in reports
                if item["candidate_id"] == result.selected_candidate_id
            )
            topology = next(
                check
                for check in selected["checks"]
                if check["check_id"] == "residual_fragmentation_topology"
            )
            self.assertTrue(topology["passed"])
            self.assertGreaterEqual(
                topology["metrics"]["observed_minimum_residual_spacing_px"],
                topology["metrics"]["minimum_residual_spacing_px"],
            )
            self.assertEqual(topology["metrics"]["holes_after"], 0)


if __name__ == "__main__":
    unittest.main()

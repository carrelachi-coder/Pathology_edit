"""Contracts and regression tests for the independent joint edit pipeline."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from inpaint_cells.instance_authority import array_sha256
from phase3_joint_edit_refine.agents import JOINT_PLAN_JSON_SCHEMA
from phase3_joint_edit_refine.auxiliary import materialize_profile_auxiliaries
from phase3_joint_edit_refine.budget import JointFeasibilitySolver
from phase3_joint_edit_refine.cell_layouts import (
    ReferenceNucleusShape,
    _place_layout,
    build_reference_shape_library,
)
from phase3_joint_edit_refine.cell_programs import (
    _cap_density_field_quotas,
    _depletion_band_edges,
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
    _discrete_radial_profile_is_monotonic,
    _fine_pattern_preserved,
    _recorded_instance_areas_by_class,
    mechanism_postcondition_checker_id,
)
from phase3_joint_edit_refine.generator_adapter import (
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
from phase3_joint_edit_refine.nuclei import iter_instances
from phase3_joint_edit_refine.packing import certify_complete_footprint_packing
from phase3_joint_edit_refine.planner import (
    HeuristicJointPlanner,
    _structural_units_for_components,
)
from phase3_joint_edit_refine.post_generation import (
    audit_joint_generation_handoff,
)
from phase3_joint_edit_refine.profile_statistics import (
    build_annotation_profile_statistics,
)
from phase3_joint_edit_refine.scene import build_joint_scene_analysis
from phase3_joint_edit_refine.seam import (
    compile_continuity_center_quota,
    compile_executable_continuity_count,
)
from phase3_joint_edit_refine.semantic_parser import (
    RuleBasedSemanticParser,
    bind_semantic_intent,
)
from phase3_joint_edit_refine.skills.repository import JointSkillRepository
from phase3_joint_edit_refine.tissue_planner import (
    MultiInterfaceResearchTissuePlanner,
    _effective_tissue_topology,
    _normalize_integer_allocations,
    _select_executable_anchor_ids,
)
from phase3_joint_edit_refine.workflow import (
    JointPathologyEditWorkflow,
    JointWorkflowConfig,
    _as_tissue_case,
    _candidate_preserving_closure_pixels,
    _joint_area_feedback_candidate_ids,
    _maximum_safe_below_target_joint_pixels,
    _minimum_safe_above_target_joint_pixels,
    _provisional_union_requires_rebalance,
)
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.agents import HeuristicInterfacePlanner
from phase3_mask_edit_refine.gates import GateRegistry, _check_edited_label_topology
from phase3_mask_edit_refine.models import GateReport
from phase3_mask_edit_refine.scene import build_scene_analysis


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class JointSkillTests(unittest.TestCase):
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

    def test_cell_decrease_skills_require_a_macroscopic_density_effect(self):
        repository = JointSkillRepository()
        for primitive_id in (
            "cell-type-abundance-decrease-v1",
            "cellularity-decrease-v1",
        ):
            with self.subTest(primitive_id=primitive_id):
                primitive = repository.primitives[primitive_id]
                self.assertGreaterEqual(primitive.minimum_effect_delta_count, 12)
                self.assertGreaterEqual(
                    primitive.minimum_effect_span_cell_diameters, 6.0
                )

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
            minimum_effect_span_px=0,
            minimum_effect_foci=0,
            seed=1,
        )
        self.assertEqual(placed, 1)
        self.assertEqual(len(trace), 1)
        col, row = trace[0]["center_xy"]
        self.assertTrue(legal[row, col])
        self.assertEqual(int(np.count_nonzero(target)), 9)

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
            minimum_effect_span_px=40,
            minimum_effect_foci=4,
            seed=1,
        )
        self.assertEqual(placed, 12)
        self.assertGreaterEqual(len({item["cluster_id"] for item in trace}), 4)
        centers = np.asarray([item["center_xy"] for item in trace], dtype=float)
        distances = centers[:, None, :] - centers[None, :, :]
        self.assertGreaterEqual(
            float(np.sqrt(np.max(np.sum(distances**2, axis=2)))), 40.0
        )

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
            "reduce tumor burden": "tumor-burden-decrease-v1",
            "increase necrosis": "necrosis-appearance-v1",
            "increase intratumoral necrosis": "necrosis-appearance-v1",
            "reduce intratumoral necrosis": "necrosis-resolution-v1",
            "increase tumor-associated stroma": "stroma-increase-v1",
            "减少坏死": "necrosis-resolution-v1",
            "increase tumor budding": (
                "neoplastic-microinfiltration-increase-v1"
            ),
            "increase immune cells": "cell-type-abundance-increase-v1",
            "降低细胞密度": "cellularity-decrease-v1",
        }
        for instruction, expected in examples.items():
            with self.subTest(instruction=instruction):
                intent = parser.parse(instruction)
                self.assertEqual(intent.primitive_id, expected)
                self.assertNotIn("mechanism", intent.to_metadata())

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
                "invasive-front-expansion-v1",
                "neoplastic-microinfiltration-increase-v1",
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
            "neoplastic-microinfiltration-increase-v1",
        )

    def test_manifest_may_hint_a_contextual_generic_tumor_interpretation(self):
        raw = {
            **_case_stub().to_metadata(),
            "instruction": "increase tumor",
            "primitive_id": "neoplastic-microinfiltration-increase-v1",
        }
        case, _intent = bind_semantic_intent(
            raw, RuleBasedSemanticParser()
        )

        self.assertEqual(
            case.primitive_id,
            "neoplastic-microinfiltration-increase-v1",
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
        self.assertEqual(len(repository.mechanisms), 29)
        self.assertEqual(len(repository.primitives), 14)
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
            self.assertIn("tumor-burden-increase-v1", mechanism.supported_primitives)
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
            ["lung-treatment-associated-fibrotic-replacement"],
        )
        self.assertEqual(
            [
                item.mechanism_id
                for item in repository.mechanisms_for(
                    pathology_domain_id="prostate-adenocarcinoma-v1",
                    primitive_id="stroma-increase-v1",
                )
            ],
            ["prostate-treatment-associated-fibrotic-replacement"],
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
        cohesive = repository.mechanisms["breast-cohesive-nst-front"]
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
                "invasive-front-expansion-v1",
                "necrosis-appearance-v1",
                "necrosis-resolution-v1",
                "neoplastic-microinfiltration-increase-v1",
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
            "unit-aware executor",
            repository.execution_selection_reason(
                primitive_id="tumor-burden-increase-v1",
                mechanism_id="colorectal-gland-forming-front",
            ),
        )
        self.assertIsNone(
            repository.execution_selection_reason(
                primitive_id="tumor-burden-increase-v1",
                mechanism_id="breast-cohesive-nst-front",
            )
        )
        local = _case_stub(
            primitive="cellularity-increase-v1",
            cell_budget=CellCountExtentBudget(3, 2, 4, 48, 0, 32),
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

    def test_invasive_front_uses_joint_semantics_with_audited_tissue_adapter(self):
        repository = JointSkillRepository()
        case = replace(
            _case_stub(primitive="invasive-front-expansion-v1"),
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
                "invasive-front-expansion-v1"
            ],
            {"source_labels": ("Stroma", "Other tissue"), "target_labels": ("Tumor",)},
        )

    def test_cross_domain_cell_population_is_rejected(self):
        repository = JointSkillRepository()
        case = _case_stub(population="breast-cellvit-source-first-v1")
        with self.assertRaisesRegex(ValueError, "domain mismatch"):
            repository.compose(
                case=case,
                mechanism_id="colorectal-gland-forming-front",
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
                case=_case_stub(),
                mechanism_id="colorectal-gland-forming-front",
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
                mechanism_id="lung-treatment-associated-fibrotic-replacement",
                available_checker_ids=JointGateRegistry().available_checker_ids,
                production=False,
            )
        bundle = repository.compose(
            case=replace(
                case,
                semantic_intent={"treatment_context": "post_treatment"},
            ),
            mechanism_id="lung-treatment-associated-fibrotic-replacement",
            available_checker_ids=JointGateRegistry().available_checker_ids,
            production=False,
        )
        self.assertEqual(
            bundle.mechanism.mechanism_id,
            "lung-treatment-associated-fibrotic-replacement",
        )

    def test_budget_broker_reserves_whole_instance_union_without_lowering_floor(self):
        repository = JointSkillRepository()
        case = _case_stub()
        bundle = repository.compose(
            case=case,
            mechanism_id="colorectal-gland-forming-front",
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

    def test_budget_broker_rebalances_from_exact_executed_cell_spill(self):
        repository = JointSkillRepository()
        case = _case_stub()
        bundle = repository.compose(
            case=case,
            mechanism_id="colorectal-gland-forming-front",
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
            cell_count_extent_budget=CellCountExtentBudget(3, 2, 4, 48, 0, 32),
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
        case = _case_stub(
            budget=JointAreaBudget(capacity_floor_policy="lower_to_proven_max_safe")
        )
        bundle = repository.compose(
            case=case,
            mechanism_id="colorectal-gland-forming-front",
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

    def test_capacity_adaptive_budget_enforces_meaningful_edit_floor(self):
        repository = JointSkillRepository()
        case = _case_stub(
            budget=JointAreaBudget(
                capacity_floor_policy="lower_to_proven_max_safe",
                minimum_effective_fraction=0.05,
            )
        )
        bundle = repository.compose(
            case=case,
            mechanism_id="colorectal-gland-forming-front",
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

    def test_tissue_tool_rejects_allocation_below_binding_public_floor(self):
        repository = JointSkillRepository()
        case = _case_stub(
            budget=JointAreaBudget(
                capacity_floor_policy="lower_to_proven_max_safe",
                minimum_effective_fraction=0.05,
            )
        )
        bundle = repository.compose(
            case=case,
            mechanism_id="colorectal-gland-forming-front",
            available_checker_ids=JointGateRegistry().available_checker_ids,
            production=False,
        )
        allocation = JointFeasibilitySolver().allocate(
            shape=(512, 512), budget=case.joint_area_budget, bundle=bundle
        )
        allocation = replace(
            allocation,
            tissue_target_pixels=20_000,
            tissue_execution_floor_pixels=13_108,
        )

        with self.assertRaisesRegex(
            JointContractError, "below the binding meaningful floor"
        ):
            _as_tissue_case(case, allocation=allocation, shape=(512, 512))

    def test_joint_router_recognizes_nuclei_only_change_as_non_noop(self):
        route = route_joint_handoff(
            {
                "ledger": {
                    "tissue_fraction": 0.0,
                    "cell_fraction": 0.04,
                    "joint_fraction": 0.04,
                    "generation_support_fraction": 0.07,
                }
            }
        )
        self.assertEqual(route.mode, "inpaint")
        self.assertGreater(route.joint_fraction, 0)

    def test_tumor_budding_is_cell_only_and_does_not_borrow_tissue_floor(self):
        repository = JointSkillRepository()
        case = _case_stub(
            primitive="neoplastic-microinfiltration-increase-v1",
            cell_budget=CellCountExtentBudget(3, 3, 4, 48, 4, 32),
        )
        bundle = repository.compose(
            case=case,
            mechanism_id="colorectal-tumor-budding-front",
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

    def review(self, *, case, bundle, candidates, gate_reports, image_paths):
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
        return GateReport(
            report.candidate_id,
            # One compiler witness plus the 12-variant fallback portfolio
            # constitute one planning pass.
            self.calls > 13,
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
                program["compiler_version"], "joint-cell-tool-compiler-v11"
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

    def test_cell_only_budding_preserves_tissue_and_uses_count_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_synthetic_case(root)
            provenance = dict(source.provenance)
            provenance["joint_mechanism_id"] = "colorectal-tumor-budding-front"
            case = replace(
                source,
                case_id="synthetic-budding",
                instruction="increase tumor budding at the invasive front",
                primitive_id="neoplastic-microinfiltration-increase-v1",
                joint_area_budget=None,
                cell_count_extent_budget=CellCountExtentBudget(3, 3, 3, 48, 4, 32),
                provenance=provenance,
            )
            result = JointPathologyEditWorkflow(
                tissue_planner=HeuristicInterfacePlanner(),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
            ).run(case, output_root=root / "cell-only")
            self.assertEqual(result.status, "selected_research", result.abstain_reasons)
            self.assertIsNotNone(result.condition)
            self.assertEqual(result.condition.ledger.tissue_pixels, 0)
            self.assertGreater(result.condition.ledger.cell_pixels, 0)
            np.testing.assert_array_equal(
                result.condition.target_tissue_mask,
                np.load(source.source_tissue_mask_uri),
            )

    def test_generic_tumor_increase_can_resolve_to_visible_budding(self):
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
                    "neoplastic-microinfiltration-increase-v1"
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

            self.assertEqual(
                result.status, "selected_research", result.abstain_reasons
            )
            resolution = json.loads(
                (
                    output_root
                    / case.case_id
                    / "semantic_resolution.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                resolution["selected_option_id"],
                "neoplastic-microinfiltration-increase-v1::colorectal-tumor-budding-front",
            )
            self.assertEqual(
                resolution["selection"]["semantic_fit"], "contextual"
            )
            self.assertIsNotNone(resolution["selected_cell_budget"])
            self.assertIn(
                "tumor-burden-increase-v1",
                resolution["rejected_interpretations"],
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
                if primitive == "cell-type-abundance-increase-v1":
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
                    self.assertEqual(sum(requested.values()), 3)
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

    def test_cellularity_decrease_without_visible_anchor_abstains(self):
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
            self.assertTrue(
                any("explicit visual depletion anchor" in reason for reason in result.abstain_reasons)
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


def _write_synthetic_case(root: Path) -> JointCaseContext:
    size = 128
    rows, cols = np.ogrid[:size, :size]
    tissue = np.full((size, size), 2, dtype=np.uint8)
    tumor = (rows - 64) ** 2 + (cols - 64) ** 2 <= 30**2
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
    auxiliary = ((rows - 64) ** 2 + (cols - 64) ** 2 <= 2**2).astype(np.uint8)
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
                        "bbox_xyxy": [34, 34, 95, 95],
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
        "joint_mechanism_id": "breast-cohesive-nst-front",
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
                    "tumor-burden-increase-v1"
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
                ["gland_or_lumen_support"],
            )
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


if __name__ == "__main__":
    unittest.main()

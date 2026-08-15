"""Regression tests for the independent Architecture-B mask editor."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image
from scipy import ndimage

from phase3_joint_edit_refine.candidate_feasibility import (
    _is_subcomponent_raster_tail,
    _tissue_portfolio_allows_anchor_diversification,
)
from phase3_joint_edit_refine.tissue_planner import (
    _tissue_geometry_candidate_count,
)
from phase3_joint_edit_refine.workflow import (
    _retain_visible_regression_whole_instance_closure,
)
from phase3_mask_edit_refine import cli as mask_cli
from phase3_mask_edit_refine.agents import (
    HeuristicInterfacePlanner,
    OpenAIMultimodalCritic,
    OpenAIMultimodalPlanner,
    _critic_prompt,
    _planner_prompt,
    critic_satisfies_hard_rules,
    validate_edit_plan,
    validate_non_breast_legacy_workflow_planner,
)
from phase3_mask_edit_refine.candidates import (
    compile_depth_profile_map,
    generate_candidates,
)
from phase3_mask_edit_refine.execution import (
    TopologySafeAreaUnderfillError,
    _low_frequency_organic_field,
    _minimum_component_spacing_px,
    _natural_external_retreat_priority,
    _normalized_organic_field,
    _prepare_compiler_work,
    _rebalance_fragmentation_residual_islands,
    _residual_fragmentation_priority,
    _simulate_topology_safe_execution,
    _whole_mask_topology_audit,
    compile_edit_plan,
)
from phase3_mask_edit_refine.gates import (
    GateContext,
    GateRegistry,
    _check_boundary_naturalness,
    _check_component_topology,
)
from phase3_mask_edit_refine.models import (
    AreaBudget,
    CandidateMask,
    CaseContext,
    CriticRanking,
    CriticResult,
    DepthProfile,
    EditPlan,
    InterfaceExecutionContract,
    PlannedInterface,
    RefineContractError,
    ResolvedAreaContract,
    ToolProgram,
)
from phase3_mask_edit_refine.scene import build_scene_analysis
from phase3_mask_edit_refine.skills import (
    SkillRepository,
    bind_active_bundle_to_case,
    validate_active_bundle_authority,
)
from phase3_mask_edit_refine.skills.schema import KnowledgeRule, ObservationAuthority
from phase3_mask_edit_refine.workflow import (
    EscalationBudget,
    MaskEditRefineWorkflow,
    WorkflowConfig,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _glas_circle_mask(size: int = 128) -> np.ndarray:
    rows, cols = np.ogrid[:size, :size]
    mask = np.full((size, size), 2, dtype=np.int64)
    tumor = (rows - size // 2) ** 2 + (cols - size // 2) ** 2 <= 30**2
    mask[tumor] = 12
    return mask


def _case(
    *,
    primitive: str,
    area: float,
    image_uri: str = "image.png",
    mask_uri: str = "mask.npy",
    provenance: dict | None = None,
) -> CaseContext:
    return CaseContext(
        case_id="synthetic",
        instruction="edit the requested tissue area",
        source_image_uri=image_uri,
        source_mask_uri=mask_uri,
        pathology_domain_id="colorectal-adenocarcinoma-v1",
        annotation_profile_id="glas-gland-v1",
        primitive_id=primitive,
        area_budget=AreaBudget(area, area, area),
        seed=17,
        provenance=provenance
        or {
            "source_image_sha256": "test",
            "source_mask_sha256": "test",
            "preprocessing_revision": "synthetic-glas-v1",
            "original_instance_mask_digest": "synthetic-instance-digest",
            "patch_grade": "moderately_differentiated",
        },
        pixel_size_um=0.465,
    )


class DualAxisSkillTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = SkillRepository()
        self.gates = GateRegistry()

    def test_breast_pathology_cannot_bypass_orca_mask_only_authority(self):
        with self.assertRaisesRegex(
            RefineContractError, "unbound observation axes|typed observation authority"
        ):
            self.repository.compose(
                pathology_domain_id="breast-invasive-carcinoma-v1",
                annotation_profile_id="orca-semantic-v1",
                primitive_id="tumor-burden-increase-v1",
                production=False,
                available_checker_ids=self.gates.available_checker_ids,
            )

    def test_orca_rejects_stroma_target(self):
        with self.assertRaisesRegex(
            RefineContractError, "cannot represent target label"
        ):
            self.repository.compose(
                pathology_domain_id="breast-invasive-carcinoma-v1",
                annotation_profile_id="orca-semantic-v1",
                primitive_id="stroma-increase-v1",
                production=False,
                available_checker_ids=self.gates.available_checker_ids,
            )

    def test_production_rejects_draft_skills(self):
        with self.assertRaisesRegex(RefineContractError, "internally_reviewed"):
            self.repository.compose(
                pathology_domain_id="colorectal-adenocarcinoma-v1",
                annotation_profile_id="glas-gland-v1",
                primitive_id="tumor-burden-increase-v1",
                production=True,
                available_checker_ids=self.gates.available_checker_ids,
            )

    def test_every_annotation_profile_has_fail_closed_background_contract(self):
        for skill_id in self.repository.list(kind="annotation_profile"):
            with self.subTest(skill_id=skill_id):
                package = self.repository.get(skill_id)
                policy = package.capabilities["background_policy"]
                self.assertIs(policy["editable"], False)
                self.assertIs(policy["may_seed_edit"], False)
                self.assertIs(policy["preserve_exactly"], True)
                self.assertGreaterEqual(
                    len(package.capabilities["required_provenance_fields"]), 2
                )
                self.assertTrue(
                    any(
                        rule.deterministic_check_id == "background_seed_protection"
                        for rule in package.rules
                    )
                )
                self.assertGreaterEqual(len(package.mask_constraints), 4)
                self.assertTrue(
                    all(item.mask_statement for item in package.mask_constraints)
                )

    def test_domain_and_profile_skills_have_detailed_evidence_rules(self):
        for kind in ("pathology_domain", "annotation_profile"):
            for skill_id in self.repository.list(kind=kind):
                with self.subTest(kind=kind, skill_id=skill_id):
                    package = self.repository.get(skill_id)
                    self.assertGreaterEqual(len(package.rules), 5)
                    self.assertTrue(
                        all(rule.evidence_citations for rule in package.rules)
                    )
                    morphology_rules = [
                        rule
                        for rule in package.rules
                        if rule.scope not in {"input_validation"}
                    ]
                    self.assertTrue(
                        any(rule.expected_morphology for rule in morphology_rules)
                    )
                    self.assertTrue(
                        any(rule.forbidden_morphology for rule in morphology_rules)
                    )
                    self.assertGreaterEqual(len(package.mask_constraints), 4)
                    self.assertTrue(
                        all(
                            item.enforcement
                            in {"deterministic", "conditional", "planner_veto"}
                            for item in package.mask_constraints
                        )
                    )
                    self.assertTrue(
                        all(
                            item.generation_handoff
                            or item.enforcement == "deterministic"
                            for item in package.mask_constraints
                        )
                    )


class FailClosedGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = SkillRepository()
        self.gates = GateRegistry()
        self.schema = self.repository.annotation_schema("glas-gland-v1")
        self.mask = _glas_circle_mask()
        self.scene = build_scene_analysis(
            self.mask, schema=self.schema, pixel_size_um=0.465
        )

    def _bundle(self, primitive: str):
        return self.repository.compose(
            pathology_domain_id="colorectal-adenocarcinoma-v1",
            annotation_profile_id="glas-gland-v1",
            primitive_id=primitive,
            production=False,
            available_checker_ids=self.gates.available_checker_ids,
        )

    def test_boundary_naturalness_honors_typed_mechanism_bound(self):
        bundle = self._bundle("tumor-burden-increase-v1")
        interface = max(
            self.scene.interfaces_for(source_labels=("Stroma",), target_label="Tumor"),
            key=lambda item: item.contact_pixels,
        )
        change = np.zeros_like(self.mask, dtype=bool)
        change[20:108, 20:40] = True
        for row in np.linspace(22, 104, 8, dtype=int):
            change[row : row + 2, 40:105] = True
        area = int(np.count_nonzero(change)) / change.size
        case = _case(primitive="tumor-burden-increase-v1", area=area)
        base_plan = _manual_plan(
            case=case,
            interface=interface,
            source_label="Stroma",
            target_label="Tumor",
            area=area,
            supporting_rules=_bundle_ids(bundle),
            band_max=128.0,
        )
        target = np.array(self.mask, copy=True)
        candidate = CandidateMask(
            "mechanism-shape-bound",
            interface.interface_id,
            "fixture",
            target,
            change,
            {},
        )

        def run_with(maximum: float, *, geometry_mode: str = "free_region"):
            ranges = dict(base_plan.tool_program.parameter_ranges)
            ranges["max_boundary_compactness"] = maximum
            ranges["tissue_geometry_mode"] = geometry_mode
            plan = replace(
                base_plan,
                tool_program=replace(base_plan.tool_program, parameter_ranges=ranges),
            )
            return _check_boundary_naturalness(
                GateContext(
                    case=case,
                    source_mask=self.mask,
                    schema=self.schema,
                    scene=self.scene,
                    bundle=bundle,
                    plan=plan,
                    candidate=candidate,
                )
            )

        strict = run_with(40.0)
        mechanism_specific = run_with(55.0)
        self.assertFalse(strict.passed)
        self.assertTrue(mechanism_specific.passed)
        self.assertEqual(
            mechanism_specific.metrics["maximum_allowed_component_compactness"],
            55.0,
        )

        interface_front = run_with(4.0, geometry_mode="interface_front")
        self.assertTrue(interface_front.passed, interface_front.metrics)
        self.assertFalse(interface_front.metrics["change_band_compactness_applicable"])
        self.assertTrue(interface_front.metrics["boundary_attached_geometry"])

    def test_case152_style_short_deep_notch_is_rejected(self):
        bundle = self.repository.compose(
            pathology_domain_id="colorectal-adenocarcinoma-v1",
            annotation_profile_id="glas-gland-v1",
            primitive_id="tumor-burden-decrease-v1",
            production=False,
            available_checker_ids=self.gates.available_checker_ids,
        )
        interface = max(
            self.scene.interfaces_for(source_labels=("Tumor",), target_label="Stroma"),
            key=lambda item: item.contact_pixels,
        )
        change = np.zeros_like(self.mask, dtype=bool)
        # Narrow radial retreat extending deep into the tumor component.
        change[60:68, 34:64] = self.mask[60:68, 34:64] == 12
        changed_fraction = int(change.sum()) / self.mask.size
        plan = _manual_plan(
            case=_case(primitive="tumor-burden-decrease-v1", area=changed_fraction),
            interface=interface,
            source_label="Tumor",
            target_label="Stroma",
            area=changed_fraction,
            supporting_rules=_bundle_ids(bundle),
            band_max=128.0,
        )
        target = self.mask.copy()
        target[change] = 2
        candidate = CandidateMask(
            "case152-like",
            interface.interface_id,
            "fixture",
            target,
            change,
            {
                "seed": 1,
                "target_fine_id": 2,
                "tool_adapter_version": "fixture",
                "source_component_id": interface.source_component_id,
                "target_component_id": interface.target_component_id,
            },
        )
        report = self.gates.run(
            GateContext(
                case=_case(primitive="tumor-burden-decrease-v1", area=changed_fraction),
                source_mask=self.mask,
                schema=self.schema,
                scene=self.scene,
                bundle=bundle,
                plan=plan,
                candidate=candidate,
            )
        )
        depth_check = next(
            item for item in report.checks if item.check_id == "depth_span_ratio"
        )
        self.assertFalse(depth_check.passed)
        self.assertFalse(report.passed)

    def test_case175_style_normal_epithelium_to_stroma_is_rejected(self):
        mask = self.mask.copy()
        mask[5:25, 5:25] = 5
        scene = build_scene_analysis(mask, schema=self.schema, pixel_size_um=0.465)
        bundle = self._bundle("stroma-increase-v1")
        interface = max(
            scene.interfaces_for(source_labels=("Tumor",), target_label="Stroma"),
            key=lambda item: item.contact_pixels,
        )
        change = np.zeros_like(mask, dtype=bool)
        change[8:18, 8:18] = True
        fraction = int(change.sum()) / mask.size
        plan = _manual_plan(
            case=_case(primitive="stroma-increase-v1", area=fraction),
            interface=interface,
            source_label="Tumor",
            target_label="Stroma",
            area=fraction,
            supporting_rules=_bundle_ids(bundle),
            band_max=128.0,
        )
        target = mask.copy()
        target[change] = 2
        candidate = CandidateMask(
            "case175-like",
            interface.interface_id,
            "fixture",
            target,
            change,
            {
                "seed": 2,
                "target_fine_id": 2,
                "tool_adapter_version": "fixture",
                "source_component_id": interface.source_component_id,
                "target_component_id": interface.target_component_id,
            },
        )
        report = self.gates.run(
            GateContext(
                case=_case(primitive="stroma-increase-v1", area=fraction),
                source_mask=mask,
                schema=self.schema,
                scene=scene,
                bundle=bundle,
                plan=plan,
                candidate=candidate,
            )
        )
        transition = next(
            item for item in report.checks if item.check_id == "label_transition"
        )
        self.assertFalse(transition.passed)
        self.assertFalse(report.passed)

    def test_orca_fragmented_background_cannot_seed_tumor_growth(self):
        schema = self.repository.annotation_schema("orca-semantic-v1")
        mask = np.full((96, 96), 7, dtype=np.int64)
        mask[24:72, 24:72] = 1
        mask[14, 14] = 0
        mask[18:20, 50:53] = 0
        mask[42:44, 15:17] = 0
        scene = build_scene_analysis(mask, schema=schema, pixel_size_um=0.25)
        bundle = self.repository.compose(
            pathology_domain_id="oral-squamous-cell-carcinoma-v1",
            annotation_profile_id="orca-semantic-v1",
            primitive_id="tumor-burden-increase-v1",
            production=False,
            available_checker_ids=self.gates.available_checker_ids,
        )
        interface = max(
            scene.interfaces_for(source_labels=("Other tissue",), target_label="Tumor"),
            key=lambda item: item.contact_pixels,
        )
        change = mask == 0
        fraction = float(np.mean(change))
        case = CaseContext(
            case_id="orca-fragmented-background",
            instruction="increase carcinoma area",
            source_image_uri="image.png",
            source_mask_uri="mask.npy",
            pathology_domain_id="oral-squamous-cell-carcinoma-v1",
            annotation_profile_id="orca-semantic-v1",
            primitive_id="tumor-burden-increase-v1",
            area_budget=AreaBudget(fraction, fraction, fraction),
            seed=19,
            provenance={
                "source_image_sha256": "test",
                "source_mask_sha256": "test",
                "preprocessing_revision": "synthetic-orca-v1",
                "original_label_map_digest": "synthetic-orca-map",
            },
            pixel_size_um=0.25,
        )
        plan = _manual_plan(
            case=case,
            interface=interface,
            source_label="Other tissue",
            target_label="Tumor",
            area=fraction,
            supporting_rules=_bundle_ids(bundle),
            band_max=128.0,
        )
        target = mask.copy()
        target[change] = 1
        candidate = CandidateMask(
            "orca-illegal-zero-seed",
            interface.interface_id,
            "fixture",
            target,
            change,
            {
                "seed": 19,
                "target_fine_id": 1,
                "tool_adapter_version": "fixture",
                "source_component_id": interface.source_component_id,
                "target_component_id": interface.target_component_id,
            },
        )
        report = self.gates.run(
            GateContext(
                case=case,
                source_mask=mask,
                schema=schema,
                scene=scene,
                bundle=bundle,
                plan=plan,
                candidate=candidate,
            )
        )
        background = next(
            item
            for item in report.checks
            if item.check_id == "background_seed_protection"
        )
        self.assertFalse(background.passed)
        self.assertEqual(
            background.metrics["background_changed_pixels"], int(change.sum())
        )
        self.assertFalse(report.passed)

    def test_planner_cannot_relax_source_component_retention(self):
        bundle = self.repository.compose(
            pathology_domain_id="colorectal-adenocarcinoma-v1",
            annotation_profile_id="glas-gland-v1",
            primitive_id="tumor-burden-decrease-v1",
            production=False,
            available_checker_ids=self.gates.available_checker_ids,
        )
        interface = max(
            self.scene.interfaces_for(source_labels=("Tumor",), target_label="Stroma"),
            key=lambda item: item.contact_pixels,
        )
        tumor = self.mask == 12
        coordinates = np.argwhere(tumor)
        changed_count = int(np.ceil(coordinates.shape[0] * 0.60))
        change = np.zeros_like(tumor)
        selected = coordinates[:changed_count]
        change[selected[:, 0], selected[:, 1]] = True
        area = float(np.mean(change))
        plan = _manual_plan(
            case=_case(primitive="tumor-burden-decrease-v1", area=area),
            interface=interface,
            source_label="Tumor",
            target_label="Stroma",
            area=area,
            supporting_rules=_bundle_ids(bundle),
            band_max=128.0,
        )
        plan = replace(
            plan,
            tool_program=replace(
                plan.tool_program,
                parameter_ranges={
                    **plan.tool_program.parameter_ranges,
                    # The Planner is deliberately trying to relax the gate.
                    "max_source_component_changed_fraction": 0.99,
                    "min_source_component_remaining_px": 1,
                },
            ),
        )
        target = self.mask.copy()
        target[change] = 2
        candidate = CandidateMask(
            "over-consume",
            interface.interface_id,
            "fixture",
            target,
            change,
            {
                "seed": 23,
                "target_fine_id": 2,
                "tool_adapter_version": "fixture",
                "source_component_id": interface.source_component_id,
                "target_component_id": interface.target_component_id,
            },
        )
        report = self.gates.run(
            GateContext(
                case=_case(primitive="tumor-burden-decrease-v1", area=area),
                source_mask=self.mask,
                schema=self.schema,
                scene=self.scene,
                bundle=bundle,
                plan=plan,
                candidate=candidate,
            )
        )
        retention = next(
            item
            for item in report.checks
            if item.check_id == "source_component_retention"
        )
        self.assertFalse(retention.passed)
        self.assertEqual(retention.metrics["max_changed_fraction"], 0.55)
        self.assertEqual(retention.metrics["min_remaining_px"], 64)


class CandidateAndSceneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = SkillRepository()
        self.gates = GateRegistry()
        self.schema = self.repository.annotation_schema("glas-gland-v1")

    def _bundle(self):
        return self.repository.compose(
            pathology_domain_id="colorectal-adenocarcinoma-v1",
            annotation_profile_id="glas-gland-v1",
            primitive_id="tumor-burden-increase-v1",
            production=False,
            available_checker_ids=self.gates.available_checker_ids,
        )

    def test_fragmentation_portfolio_skips_ignored_anchor_preferences(self):
        self.assertFalse(
            _tissue_portfolio_allows_anchor_diversification(
                "residual-tumor-fragmentation-v1"
            )
        )
        self.assertTrue(
            _tissue_portfolio_allows_anchor_diversification(
                "invasive-tumor-footprint-decrease-v1"
            )
        )
        self.assertEqual(
            _tissue_geometry_candidate_count(residual_fragmentation=True), 1
        )
        self.assertEqual(
            _tissue_geometry_candidate_count(residual_fragmentation=False), 4
        )

    def test_candidate_schedule_produces_distinct_masks(self):
        mask = _glas_circle_mask()
        scene = build_scene_analysis(mask, schema=self.schema, pixel_size_um=0.465)
        bundle = self._bundle()
        interface = max(
            scene.interfaces_for(source_labels=("Stroma",), target_label="Tumor"),
            key=lambda item: item.contact_pixels,
        )
        case = _case(primitive="tumor-burden-increase-v1", area=0.03)
        plan = _manual_plan(
            case=case,
            interface=interface,
            source_label="Stroma",
            target_label="Tumor",
            area=0.03,
            supporting_rules=_bundle_ids(bundle),
            band_max=48.0,
        )
        plan = replace(
            plan,
            tool_program=ToolProgram(
                allowed_tools=("interface_sdf", "connected_morphology", "organic_v2"),
                parameter_ranges=plan.tool_program.parameter_ranges,
                candidate_count=12,
            ),
        )
        candidates = generate_candidates(
            mask,
            schema=self.schema,
            scene=scene,
            plan=plan,
            bundle=bundle,
            seed=case.seed,
        )
        digests = {candidate.target_mask.tobytes() for candidate in candidates}
        self.assertGreaterEqual(len(candidates), 6)
        self.assertEqual(len(digests), len(candidates))
        self.assertGreaterEqual(len({item.tool_name for item in candidates}), 2)

    def test_curved_equal_width_annulus_is_rejected_as_parallel_artifact(self):
        mask = _glas_circle_mask()
        scene = build_scene_analysis(mask, schema=self.schema, pixel_size_um=0.465)
        bundle = self._bundle()
        interface = max(
            scene.interfaces_for(source_labels=("Stroma",), target_label="Tumor"),
            key=lambda item: item.contact_pixels,
        )
        rows, cols = np.ogrid[: mask.shape[0], : mask.shape[1]]
        radius_squared = (rows - mask.shape[0] // 2) ** 2 + (
            cols - mask.shape[1] // 2
        ) ** 2
        change = (radius_squared > 30**2) & (radius_squared <= 45**2)
        target = np.array(mask, copy=True)
        target[change] = 12
        area = float(np.mean(change))
        case = _case(primitive="tumor-burden-increase-v1", area=area)
        plan = _manual_plan(
            case=case,
            interface=interface,
            source_label="Stroma",
            target_label="Tumor",
            area=area,
            supporting_rules=_bundle_ids(bundle),
            band_max=24.0,
        )
        candidate = CandidateMask(
            "annulus",
            interface.interface_id,
            "fixture",
            target,
            change,
            {
                "seed": 17,
                "target_fine_id": 12,
                "target_fine_ids": [12],
                "tool_adapter_version": "fixture",
                "interface_ids": [interface.interface_id],
                "source_component_id": interface.source_component_id,
                "source_component_ids": [interface.source_component_id],
                "target_component_id": interface.target_component_id,
                "target_component_ids": [interface.target_component_id],
            },
        )
        report = self.gates.run(
            GateContext(
                case=case,
                source_mask=mask,
                schema=self.schema,
                scene=scene,
                bundle=bundle,
                plan=plan,
                candidate=candidate,
            )
        )
        parallel = next(
            item
            for item in report.checks
            if item.check_id == "parallel_boundary_artifact"
        )
        self.assertFalse(parallel.passed, parallel.metrics)
        interface_metrics = next(iter(parallel.metrics["interfaces"].values()))
        self.assertLess(interface_metrics["front_depth_cv"], 0.25)

    def test_left_and_right_anchor_contracts_produce_different_pixels(self):
        mask = _glas_circle_mask()
        scene = build_scene_analysis(mask, schema=self.schema, pixel_size_um=0.465)
        bundle = self._bundle()
        interface = max(
            scene.interfaces_for(source_labels=("Stroma",), target_label="Tumor"),
            key=lambda item: item.contact_pixels,
        )
        self.assertGreaterEqual(len(interface.anchor_segment_ids), 2)
        case = _case(primitive="tumor-burden-increase-v1", area=0.01)
        base = _manual_plan(
            case=case,
            interface=interface,
            source_label="Stroma",
            target_label="Tumor",
            area=0.01,
            supporting_rules=_bundle_ids(bundle),
            band_max=32.0,
        )
        left_execution = replace(
            base.candidate_interfaces[0].execution_contract,
            anchor_segment_ids=(interface.anchor_segment_ids[0],),
        )
        right_execution = replace(
            base.candidate_interfaces[0].execution_contract,
            anchor_segment_ids=(interface.anchor_segment_ids[-1],),
        )
        left_plan = replace(
            base,
            candidate_interfaces=(
                replace(
                    base.candidate_interfaces[0],
                    anchor_segment="first executable anchor",
                    execution_contract=left_execution,
                ),
            ),
        )
        right_plan = replace(
            base,
            candidate_interfaces=(
                replace(
                    base.candidate_interfaces[0],
                    anchor_segment="last executable anchor",
                    execution_contract=right_execution,
                ),
            ),
        )
        validate_edit_plan(left_plan, case=case, scene=scene, bundle=bundle)
        validate_edit_plan(right_plan, case=case, scene=scene, bundle=bundle)
        left = generate_candidates(
            mask,
            schema=self.schema,
            scene=scene,
            plan=left_plan,
            bundle=bundle,
            seed=case.seed,
        )[0]
        right = generate_candidates(
            mask,
            schema=self.schema,
            scene=scene,
            plan=right_plan,
            bundle=bundle,
            seed=case.seed,
        )[0]
        self.assertFalse(np.array_equal(left.target_mask, right.target_mask))
        overlap = int(np.count_nonzero(left.change_region & right.change_region))
        union = int(np.count_nonzero(left.change_region | right.change_region))
        self.assertLess(overlap / max(union, 1), 0.05)

        tampered = CandidateMask(
            candidate_id="wrong-anchor-pixels",
            interface_id=left.interface_id,
            tool_name=left.tool_name,
            target_mask=right.target_mask,
            change_region=right.change_region,
            tool_trace=left.tool_trace,
        )
        report = self.gates.run(
            GateContext(
                case=case,
                source_mask=mask,
                schema=self.schema,
                scene=scene,
                bundle=bundle,
                plan=left_plan,
                candidate=tampered,
            )
        )
        fidelity = next(
            item
            for item in report.checks
            if item.check_id == "execution_contract_fidelity"
        )
        self.assertFalse(fidelity.passed)

    def test_execution_compiler_preserves_anchor_intent_and_resolves_depth(self):
        mask = _glas_circle_mask()
        scene = build_scene_analysis(mask, schema=self.schema, pixel_size_um=0.465)
        bundle = self._bundle()
        interface = max(
            scene.interfaces_for(source_labels=("Stroma",), target_label="Tumor"),
            key=lambda item: item.contact_pixels,
        )
        case = _case(primitive="tumor-burden-increase-v1", area=0.03)
        raw = _manual_plan(
            case=case,
            interface=interface,
            source_label="Stroma",
            target_label="Tumor",
            area=0.03,
            supporting_rules=_bundle_ids(bundle),
            band_max=64.0,
        )
        compiled, audit = compile_edit_plan(
            raw, source_mask=mask, schema=self.schema, scene=scene
        )
        self.assertEqual(
            compiled.candidate_interfaces[0].execution_contract.anchor_segment_ids,
            raw.candidate_interfaces[0].execution_contract.anchor_segment_ids,
        )
        self.assertEqual(
            compiled.candidate_interfaces[
                0
            ].execution_contract.area_allocation_fraction,
            1.0,
        )
        self.assertLessEqual(
            compiled.candidate_interfaces[
                0
            ].execution_contract.depth_profile.peak_depth_px,
            64.0,
        )
        self.assertEqual(audit["target_pixels"], round(mask.size * 0.03))
        candidates = generate_candidates(
            mask,
            schema=self.schema,
            scene=scene,
            plan=compiled,
            bundle=bundle,
            seed=case.seed,
        )
        report = self.gates.run(
            GateContext(
                case=case,
                source_mask=mask,
                schema=self.schema,
                scene=scene,
                bundle=bundle,
                plan=compiled,
                candidate=candidates[0],
            )
        )
        fidelity = next(
            item
            for item in report.checks
            if item.check_id == "execution_contract_fidelity"
        )
        self.assertTrue(fidelity.passed, fidelity.metrics)
        self.assertTrue(
            all(
                "anchor_endpoint_tolerance_pixels" in metrics
                for metrics in fidelity.metrics["interfaces"].values()
            )
        )
        labels, count = ndimage.label(
            candidates[0].change_region,
            structure=np.ones((3, 3), dtype=bool),
        )
        component_sizes = [
            int(np.count_nonzero(labels == component_id))
            for component_id in range(1, count + 1)
        ]
        self.assertTrue(component_sizes)
        self.assertGreaterEqual(min(component_sizes), 16)

    def test_ranged_budget_uses_maximum_safe_area_below_desired(self):
        mask = _glas_circle_mask()
        scene = build_scene_analysis(mask, schema=self.schema, pixel_size_um=0.465)
        bundle = self.repository.compose(
            pathology_domain_id="colorectal-adenocarcinoma-v1",
            annotation_profile_id="glas-gland-v1",
            primitive_id="tumor-burden-decrease-v1",
            production=False,
            available_checker_ids=self.gates.available_checker_ids,
        )
        interface = max(
            scene.interfaces_for(source_labels=("Tumor",), target_label="Stroma"),
            key=lambda item: item.contact_pixels,
        )
        budget = AreaBudget(
            target_fraction=0.19,
            min_fraction=0.05,
            max_fraction=0.24,
            fallback_policy="max_feasible_below_target",
        )
        case = replace(
            _case(primitive="tumor-burden-decrease-v1", area=0.19),
            area_budget=budget,
        )
        raw = replace(
            _manual_plan(
                case=case,
                interface=interface,
                source_label="Tumor",
                target_label="Stroma",
                area=0.19,
                supporting_rules=_bundle_ids(bundle),
                band_max=64.0,
            ),
            area_budget=budget,
        )
        compiled, audit = compile_edit_plan(
            raw, source_mask=mask, schema=self.schema, scene=scene
        )
        self.assertTrue(compiled.resolved_area.used_fallback)
        self.assertLess(
            compiled.resolved_area.resolved_pixels,
            compiled.resolved_area.desired_pixels,
        )
        self.assertGreaterEqual(
            compiled.resolved_area.resolved_pixels,
            compiled.resolved_area.hard_min_pixels,
        )
        self.assertEqual(
            audit["resolved_pixels"], compiled.resolved_area.resolved_pixels
        )
        candidates = generate_candidates(
            mask,
            schema=self.schema,
            scene=scene,
            plan=compiled,
            bundle=bundle,
            seed=case.seed,
        )
        report = self.gates.run(
            GateContext(
                case=case,
                source_mask=mask,
                schema=self.schema,
                scene=scene,
                bundle=bundle,
                plan=compiled,
                candidate=candidates[0],
            )
        )
        area = next(item for item in report.checks if item.check_id == "changed_area")
        topology = next(
            item for item in report.checks if item.check_id == "edited_label_topology"
        )
        self.assertTrue(area.passed, area.metrics)
        self.assertTrue(topology.passed, topology.metrics)

    def test_ranged_budget_abstains_when_maximum_safe_area_is_below_minimum(self):
        mask = _glas_circle_mask()
        scene = build_scene_analysis(mask, schema=self.schema, pixel_size_um=0.465)
        bundle = self.repository.compose(
            pathology_domain_id="colorectal-adenocarcinoma-v1",
            annotation_profile_id="glas-gland-v1",
            primitive_id="tumor-burden-decrease-v1",
            production=False,
            available_checker_ids=self.gates.available_checker_ids,
        )
        interface = max(
            scene.interfaces_for(source_labels=("Tumor",), target_label="Stroma"),
            key=lambda item: item.contact_pixels,
        )
        budget = AreaBudget(
            target_fraction=0.19,
            min_fraction=0.12,
            max_fraction=0.24,
            fallback_policy="max_feasible_below_target",
        )
        case = replace(
            _case(primitive="tumor-burden-decrease-v1", area=0.19),
            area_budget=budget,
        )
        raw = replace(
            _manual_plan(
                case=case,
                interface=interface,
                source_label="Tumor",
                target_label="Stroma",
                area=0.19,
                supporting_rules=_bundle_ids(bundle),
                band_max=64.0,
            ),
            area_budget=budget,
        )
        with self.assertRaises(TopologySafeAreaUnderfillError) as raised:
            compile_edit_plan(raw, source_mask=mask, schema=self.schema, scene=scene)
        feedback = raised.exception.feedback
        self.assertEqual(feedback["stage"], "tissue_area_underfill")
        self.assertEqual(
            feedback["policy_floor_pixels"], int(np.ceil(mask.size * 0.12))
        )
        self.assertGreater(feedback["deficit_to_floor_pixels"], 0)
        self.assertTrue(feedback["interface_contributions"])
        self.assertIn(
            feedback["required_action"],
            {
                "expand_interface_set_and_redistribute",
                "redistribute_across_alternate_interfaces",
            },
        )

    def test_fragmentation_cleanup_fills_micro_island_at_constant_area(self):
        shape = (160, 240)
        source = np.zeros(shape, dtype=bool)
        source[5:155, 5:235] = True
        target = ~source
        change = np.zeros(shape, dtype=bool)
        change[5:155, 100:140] = True
        # An aggregate 81-pixel raster cap surrounded by the otherwise
        # complete corridor must not count as a biological third residual
        # focus. This exercises the production case where many sub-minimum
        # caps collectively exceed the former 64-pixel cleanup ceiling.
        change[50:59, 112:121] = False
        priority = np.zeros(shape, dtype=float)
        # More than 4096 high-priority pixels lie inside the corridor. They
        # must be excluded before the bounded audit loop rather than retried
        # as invalid one-pixel source islands.
        priority[5:155, 104:136] = 100.0
        legal_source = np.array(source, copy=True)
        work = SimpleNamespace(
            source_component=source,
            legal_source=legal_source,
            priority=priority,
            planned=SimpleNamespace(
                source_component_id="tumor:1",
                target_component_id="stroma:1",
            ),
        )

        cleaned, audit = _rebalance_fragmentation_residual_islands(
            (change,),
            works=(work,),
            source_region=source,
            target_region=target,
            minimum_residual_components=2,
            maximum_residual_components=6,
            minimum_residual_component_area_px=96,
            minimum_residual_spacing_px=4,
            residual_area_floor_fraction=0.3,
        )

        self.assertTrue(audit["applied"], audit)
        self.assertEqual(audit["pixels_added"], 81)
        self.assertEqual(audit["pixels_reclaimed"], 81)
        self.assertEqual(audit["tiny_pixels_added"], 81)
        self.assertEqual(audit["spacing_pixels_added"], 0)
        self.assertEqual(audit["balance_pixels_added"], 0)
        self.assertEqual(np.count_nonzero(cleaned[0]), np.count_nonzero(change))
        labels, count = ndimage.label(
            source & ~cleaned[0], structure=np.ones((3, 3), dtype=bool)
        )
        sizes = np.bincount(labels.ravel())[1:]
        self.assertEqual(count, 2)
        self.assertGreaterEqual(int(sizes.min()), 96)

    def test_fragmentation_cleanup_repairs_target_connected_junction_cap(self):
        shape = (160, 240)
        source = np.zeros(shape, dtype=bool)
        source[5:155, 5:235] = True
        # A third-class slit reaches the exterior, so it does not introduce a
        # pre-existing source hole, and ends beside the future raster cap.
        source[5:55, 111] = False
        target = ~source
        target[5:55, 111] = False
        change = np.zeros(shape, dtype=bool)
        change[5:155, 100:140] = True
        change &= source
        change[50:59, 112:121] = False
        # Model a third-class tissue pixel beside the cap.  The cap is not
        # fully surrounded by target, but it is target-connected and remains
        # a sub-minimum raster remnant rather than a valid residual focus.
        priority = np.zeros(shape, dtype=float)
        legal_source = np.array(source, copy=True)
        work = SimpleNamespace(
            source_component=source,
            legal_source=legal_source,
            priority=priority,
            planned=SimpleNamespace(
                source_component_id="tumor:1",
                target_component_id="stroma:1",
            ),
        )

        cleaned, audit = _rebalance_fragmentation_residual_islands(
            (change,),
            works=(work,),
            source_region=source,
            target_region=target,
            minimum_residual_components=2,
            maximum_residual_components=6,
            minimum_residual_component_area_px=96,
            minimum_residual_spacing_px=4,
            residual_area_floor_fraction=0.3,
        )

        self.assertTrue(audit["applied"], audit)
        self.assertEqual(audit["pixels_added"], 81)
        self.assertEqual(audit["pixels_reclaimed"], 81)
        self.assertEqual(audit["tiny_pixels_added"], 81)
        self.assertEqual(audit["spacing_pixels_added"], 0)
        self.assertEqual(audit["balance_pixels_added"], 0)
        self.assertEqual(np.count_nonzero(cleaned[0]), np.count_nonzero(change))

    def test_fragmentation_cleanup_merges_underweight_focus_by_residual_bridge(self):
        shape = (100, 200)
        source = np.zeros(shape, dtype=bool)
        source[5:95, 5:195] = True
        target = ~source
        change = np.zeros(shape, dtype=bool)
        change[5:95, 60:70] = True
        change[5:95, 73:83] = True
        # Offset the short residual bridge with a small raster cap so the
        # transactional cleanup can preserve the exact edited area.
        change[40:44, 63:67] = False
        work = SimpleNamespace(
            source_component=source,
            legal_source=np.array(source, copy=True),
            priority=np.zeros(shape, dtype=float),
            planned=SimpleNamespace(
                source_component_id="tumor:1",
                target_component_id="stroma:1",
            ),
        )

        cleaned, audit = _rebalance_fragmentation_residual_islands(
            (change,),
            works=(work,),
            source_region=source,
            target_region=target,
            minimum_residual_components=2,
            maximum_residual_components=4,
            minimum_residual_component_area_px=96,
            minimum_residual_spacing_px=4,
            residual_area_floor_fraction=0.3,
            minimum_residual_component_fraction=0.025,
            maximum_dominant_residual_component_fraction=0.75,
        )

        self.assertTrue(audit["applied"], audit)
        self.assertGreater(audit["balance_bridge_pixels_reclaimed"], 0)
        self.assertEqual(np.count_nonzero(cleaned[0]), np.count_nonzero(change))
        labels, count = ndimage.label(
            source & ~cleaned[0], structure=np.ones((3, 3), dtype=bool)
        )
        sizes = np.bincount(labels.ravel())[1:]
        fractions = sizes / sizes.sum()
        self.assertEqual(count, 2)
        self.assertGreaterEqual(float(fractions.min()), 0.025)
        self.assertLessEqual(float(fractions.max()), 0.75)

    def test_fragmentation_cleanup_preserves_nonlegal_residual_cap(self):
        shape = (160, 240)
        source = np.zeros(shape, dtype=bool)
        source[5:155, 5:235] = True
        target = ~source
        change = np.zeros(shape, dtype=bool)
        change[5:155, 100:140] = True
        change[50:59, 112:121] = False
        legal_source = np.array(source, copy=True)
        legal_source[50:59, 112:121] = False
        work = SimpleNamespace(
            source_component=source,
            legal_source=legal_source,
            priority=np.zeros(shape, dtype=float),
            planned=SimpleNamespace(
                source_component_id="tumor:1",
                target_component_id="stroma:1",
            ),
        )

        with mock.patch(
            "phase3_mask_edit_refine.execution._fragmentation_spacing_repair",
            side_effect=AssertionError(
                "nonlegal cleanup must be rejected before spacing repair"
            ),
        ):
            cleaned, audit = _rebalance_fragmentation_residual_islands(
                (change,),
                works=(work,),
                source_region=source,
                target_region=target,
                minimum_residual_components=2,
                maximum_residual_components=6,
                minimum_residual_component_area_px=96,
                minimum_residual_spacing_px=4,
                residual_area_floor_fraction=0.3,
            )

        self.assertFalse(audit["applied"], audit)
        self.assertTrue(np.array_equal(cleaned[0], change))

    def test_fragmentation_cleanup_preserves_compiler_pixel_ownership(self):
        shape = (160, 240)
        source = np.zeros(shape, dtype=bool)
        source[5:155, 5:235] = True
        target = ~source
        change = np.zeros(shape, dtype=bool)
        change[5:155, 100:140] = True
        change[50:59, 112:121] = False
        legal_left = source & (np.indices(shape)[1] < 112)
        legal_right = source & ~legal_left
        work_left = SimpleNamespace(
            source_component=source,
            legal_source=legal_left,
            anchor_mask=target,
            priority=np.zeros(shape, dtype=float),
            planned=SimpleNamespace(
                source_component_id="tumor:1",
                target_component_id="stroma:1",
            ),
        )
        work_right = SimpleNamespace(
            source_component=source,
            legal_source=legal_right,
            anchor_mask=target,
            priority=np.zeros(shape, dtype=float),
            planned=SimpleNamespace(
                source_component_id="tumor:1",
                target_component_id="stroma:1",
            ),
        )
        selected = (change & legal_left, change & legal_right)

        cleaned, audit = _rebalance_fragmentation_residual_islands(
            selected,
            works=(work_left, work_right),
            source_region=source,
            target_region=target,
            minimum_residual_components=2,
            maximum_residual_components=6,
            minimum_residual_component_area_px=96,
            minimum_residual_spacing_px=4,
            residual_area_floor_fraction=0.3,
        )

        self.assertTrue(audit["applied"], audit)
        self.assertFalse(np.any(cleaned[0] & ~legal_left))
        self.assertFalse(np.any(cleaned[1] & ~legal_right))
        self.assertTrue(np.any(cleaned[1][50:59, 112:121]))

    def test_subcomponent_raster_tail_does_not_force_global_replan(self):
        resolved = ResolvedAreaContract(
            desired_pixels=35127,
            hard_min_pixels=23593,
            hard_max_pixels=47186,
            resolved_pixels=35121,
            fallback_policy="max_feasible_below_target",
            used_fallback=True,
            binding_constraint="reachable_interface_capacity",
            solver_version="fixture",
        )
        plan = SimpleNamespace(
            resolved_area=resolved,
            tool_program=SimpleNamespace(
                parameter_ranges={"min_component_area_px": 16}
            ),
        )

        self.assertTrue(_is_subcomponent_raster_tail(plan))
        self.assertFalse(
            _is_subcomponent_raster_tail(
                SimpleNamespace(
                    resolved_area=replace(resolved, resolved_pixels=35111),
                    tool_program=plan.tool_program,
                )
            )
        )

    def test_tiny_change_cleanup_continues_only_the_connected_front(self):
        shape = (40, 40)
        source = np.zeros(shape, dtype=bool)
        source[3:37, 3:37] = True
        # A nearby stromal cleft makes a pixel two steps from the established
        # top front independently target-connected. A double-dilated cleanup
        # frontier used to recreate the same 1 px satellite on every pass.
        source[6, 10:30] = False
        target = ~source
        anchor = np.zeros(shape, dtype=bool)
        anchor[3, 20] = True
        anchor[36, 32] = True
        priority = np.full(shape, 1000.0)
        priority[3, 3:37] = 0.0
        priority[4, :] = 100.0
        priority[5, 10:30] = -500.0
        priority[36, 32] = -1000.0
        work = SimpleNamespace(
            planned=SimpleNamespace(
                source_component_id="source:1",
                target_component_id="target:1",
            ),
            legal_source=source,
            anchor_mask=anchor,
            priority=priority,
            item_capacity_px=int(np.count_nonzero(source)),
            source_deletion_limit_px=int(np.count_nonzero(source)) - 1,
            protected_source_necks=np.zeros(shape, dtype=bool),
        )
        scene = SimpleNamespace(
            component_masks={"source:1": source, "target:1": target}
        )

        selected, audits = _simulate_topology_safe_execution(
            (work,),
            allocations=(20,),
            desired_pixels=20,
            source_region=source,
            target_region=target,
            scene=scene,
            seed=0,
            minimum_changed_component_area_px=16,
        )

        labels, count = ndimage.label(
            selected[0], structure=np.ones((3, 3), dtype=bool)
        )
        sizes = [
            int(np.count_nonzero(labels == component_id))
            for component_id in range(1, count + 1)
        ]
        self.assertEqual(int(np.count_nonzero(selected[0])), 20)
        self.assertTrue(all(size >= 16 for size in sizes), sizes)
        self.assertGreater(audits[0]["tiny_component_pixels_reclaimed"], 0)

    def test_fragmentation_retains_hard_safe_whole_instance_closure(self):
        self.assertTrue(
            _retain_visible_regression_whole_instance_closure(
                primitive_id="residual-tumor-fragmentation-v1",
                fallback_policy="max_feasible_below_target",
                predicted_pixels=(40930,),
                desired_max_pixels=38168,
                hard_max_pixels=47186,
            )
        )
        self.assertFalse(
            _retain_visible_regression_whole_instance_closure(
                primitive_id="neoplastic-cell-abundance-decrease-v1",
                fallback_policy="max_feasible_below_target",
                predicted_pixels=(40930,),
                desired_max_pixels=38168,
                hard_max_pixels=47186,
            )
        )
        self.assertFalse(
            _retain_visible_regression_whole_instance_closure(
                primitive_id="residual-tumor-fragmentation-v1",
                fallback_policy="max_feasible_below_target",
                predicted_pixels=(48000,),
                desired_max_pixels=38168,
                hard_max_pixels=47186,
            )
        )
        self.assertTrue(
            _retain_visible_regression_whole_instance_closure(
                primitive_id="invasive-tumor-footprint-decrease-v1",
                fallback_policy="max_feasible_below_target",
                predicted_pixels=(30809,),
                desired_max_pixels=29989,
                hard_max_pixels=36700,
            )
        )

    def test_fragmentation_cleanup_widens_under_spaced_corridors(self):
        shape = (160, 300)
        source = np.zeros(shape, dtype=bool)
        source[5:155, 5:295] = True
        target = ~source
        change = np.zeros(shape, dtype=bool)
        # Preserve a broad external retreat front from which the solver can
        # reclaim area after widening both true fragmentation corridors.
        change[5:155, 5:25] = True
        change[5:155, 90:95] = True
        change[5:155, 200:205] = True
        priority = np.zeros(shape, dtype=float)
        priority[:, :30] = 100.0
        work = SimpleNamespace(
            source_component=source,
            legal_source=source,
            priority=priority,
            planned=SimpleNamespace(
                source_component_id="tumor:1",
                target_component_id="stroma:1",
            ),
        )

        cleaned, audit = _rebalance_fragmentation_residual_islands(
            (change,),
            works=(work,),
            source_region=source,
            target_region=target,
            minimum_residual_components=3,
            maximum_residual_components=6,
            minimum_residual_component_area_px=96,
            minimum_residual_spacing_px=8,
            residual_area_floor_fraction=0.3,
        )

        self.assertTrue(audit["applied"], audit)
        self.assertEqual(audit["tiny_pixels_added"], 0)
        self.assertGreater(audit["spacing_pixels_added"], 0)
        self.assertEqual(audit["balance_pixels_added"], 0)
        self.assertEqual(audit["pixels_added"], audit["pixels_reclaimed"])
        self.assertEqual(np.count_nonzero(cleaned[0]), np.count_nonzero(change))
        labels, count = ndimage.label(
            source & ~cleaned[0], structure=np.ones((3, 3), dtype=bool)
        )
        self.assertEqual(count, 3)
        self.assertGreaterEqual(_minimum_component_spacing_px(labels, count), 8)

    def test_fragmentation_cleanup_merges_subrelative_focus(self):
        shape = (160, 400)
        source = np.zeros(shape, dtype=bool)
        source[5:155, 5:395] = True
        target = ~source
        change = np.zeros(shape, dtype=bool)
        change[5:155, 5:30] = True
        change[5:155, 80:90] = True
        change[5:155, 100:110] = True
        change[5:155, 250:260] = True
        priority = np.zeros(shape, dtype=float)
        priority[:, :35] = 100.0
        work = SimpleNamespace(
            source_component=source,
            legal_source=source,
            priority=priority,
            planned=SimpleNamespace(
                source_component_id="tumor:1",
                target_component_id="stroma:1",
            ),
        )

        cleaned, audit = _rebalance_fragmentation_residual_islands(
            (change,),
            works=(work,),
            source_region=source,
            target_region=target,
            minimum_residual_components=3,
            maximum_residual_components=6,
            minimum_residual_component_area_px=96,
            minimum_residual_spacing_px=8,
            residual_area_floor_fraction=0.3,
            minimum_residual_component_fraction=0.08,
            maximum_dominant_residual_component_fraction=0.75,
        )

        self.assertTrue(audit["applied"], audit)
        self.assertEqual(audit["tiny_pixels_added"], 0)
        self.assertEqual(audit["balance_pixels_added"], 0)
        self.assertGreater(audit["balance_bridge_pixels_reclaimed"], 0)
        self.assertGreater(audit["balance_bridge_replacement_pixels_added"], 0)
        self.assertEqual(audit["pixels_added"], audit["pixels_reclaimed"])
        labels, count = ndimage.label(
            source & ~cleaned[0], structure=np.ones((3, 3), dtype=bool)
        )
        sizes = np.bincount(labels.ravel())[1:]
        self.assertEqual(count, 3)
        self.assertGreaterEqual(float(np.min(sizes / sizes.sum())), 0.08)

    def test_fragmentation_component_gate_allows_target_connected_cap_repair(self):
        source = np.ones((80, 120), dtype=np.int64)
        source[:, :10] = 2
        change = np.zeros_like(source, dtype=bool)
        change[10:70, 30:50] = True
        change[20, 10] = True
        context = SimpleNamespace(
            source_mask=source,
            schema=self.schema,
            plan=SimpleNamespace(
                target_label="Stroma",
                tool_program=SimpleNamespace(
                    parameter_ranges={
                        "max_changed_components": 6,
                        "min_component_area_px": 16,
                        "tissue_geometry_mode": "residual_fragmentation",
                    }
                ),
            ),
            candidate=SimpleNamespace(change_region=change),
        )

        fragmentation = _check_component_topology(context)
        self.assertTrue(fragmentation.passed, fragmentation.metrics)
        self.assertEqual(
            fragmentation.metrics["fragmentation_target_connected_repair_pixels"],
            1,
        )

        context.plan.tool_program.parameter_ranges[
            "tissue_geometry_mode"
        ] = "interface_front"
        ordinary = _check_component_topology(context)
        self.assertFalse(ordinary.passed)

    def test_fragmentation_priority_builds_three_shrunken_organic_foci(self):
        shape = (96, 120)
        source = np.zeros(shape, dtype=bool)
        source[12:84, 10:110] = True
        default = ndimage.distance_transform_edt(source)
        target_pixels = round(int(source.sum()) * 0.32)

        priority = _residual_fragmentation_priority(
            source_component=source,
            legal_envelope=source,
            default_priority=default,
            minimum_residual_components=3,
            maximum_residual_components=6,
            minimum_residual_component_area_px=96,
            minimum_residual_spacing_px=16,
            minimum_residual_component_fraction=0.08,
            maximum_dominant_residual_component_fraction=0.75,
            target_change_pixels=target_pixels,
        )

        cleavage_seed = source & (priority < 0.17)
        labels, count = ndimage.label(
            source & ~cleavage_seed, structure=np.ones((3, 3), dtype=bool)
        )
        sizes = np.bincount(labels.ravel())[1:]
        fractions = sizes / sizes.sum()
        self.assertEqual(count, 3)
        self.assertGreaterEqual(float(fractions.min()), 0.08)
        self.assertLessEqual(float(fractions.max()), 0.75)
        # The topological seed establishes the split, but most of the area
        # budget must reshape the residual objects.  This prevents the old
        # behavior where nearly every changed pixel formed one dilated line.
        self.assertLess(int(np.count_nonzero(cleavage_seed)), target_pixels // 2)
        self.assertEqual(
            ndimage.label(cleavage_seed, structure=np.ones((3, 3), dtype=bool))[1],
            1,
        )

        eligible_ids = np.flatnonzero(source)
        order = np.argsort(priority.ravel()[eligible_ids], kind="stable")
        changed = np.zeros_like(source)
        changed.ravel()[eligible_ids[order[:target_pixels]]] = True
        residual_labels, residual_count = ndimage.label(
            source & ~changed, structure=np.ones((3, 3), dtype=bool)
        )
        residual_sizes = np.bincount(residual_labels.ravel())[1:]
        self.assertEqual(residual_count, 3)
        self.assertLessEqual(
            float(np.max(residual_sizes) / np.sum(residual_sizes)), 0.75
        )
        self.assertGreater(
            int(np.count_nonzero(changed & ~cleavage_seed)),
            int(np.count_nonzero(cleavage_seed)),
        )
        self.assertGreaterEqual(
            _minimum_component_spacing_px(residual_labels, residual_count), 16.0
        )
        source_boundary = source & ~ndimage.binary_erosion(
            source, structure=np.ones((3, 3), dtype=bool)
        )
        changed_boundary_fraction = np.count_nonzero(
            changed & source_boundary
        ) / np.count_nonzero(source_boundary)
        # A few broad external retreat bays should accompany the cleavage;
        # eroding the entire perimeter would merely turn this into footprint
        # decrease, while touching none would leave the old slot-cut look.
        self.assertGreater(changed_boundary_fraction, 0.05)
        self.assertLess(changed_boundary_fraction, 0.60)

    def test_fragmentation_priority_does_not_isolate_immutable_annotation_cap(self):
        shape = (96, 120)
        source = np.zeros(shape, dtype=bool)
        source[12:84, 10:110] = True
        legal = np.array(source, copy=True)
        legal[47:50, 59:62] = False
        target_pixels = round(int(source.sum()) * 0.28)

        priority = _residual_fragmentation_priority(
            source_component=source,
            legal_envelope=legal,
            default_priority=ndimage.distance_transform_edt(source),
            minimum_residual_components=3,
            maximum_residual_components=6,
            minimum_residual_component_area_px=96,
            minimum_residual_spacing_px=16,
            minimum_residual_component_fraction=0.08,
            maximum_dominant_residual_component_fraction=0.75,
            target_change_pixels=target_pixels,
        )

        eligible_ids = np.flatnonzero(legal)
        order = np.argsort(priority.ravel()[eligible_ids], kind="stable")
        changed = np.zeros_like(source)
        changed.ravel()[eligible_ids[order[:target_pixels]]] = True
        residual_labels, residual_count = ndimage.label(
            source & ~changed, structure=np.ones((3, 3), dtype=bool)
        )
        residual_sizes = np.bincount(residual_labels.ravel())[1:]

        self.assertFalse(np.any(changed & ~legal))
        self.assertGreaterEqual(residual_count, 3)
        self.assertLessEqual(residual_count, 6)
        self.assertGreaterEqual(int(residual_sizes.min()), 96)

    def test_fragmentation_priority_exaggerates_local_breakup_width(self):
        shape = (192, 240)
        source = np.zeros(shape, dtype=bool)
        source[16:176, 12:228] = True
        # The production high-visibility fragmentation manifest reserves 36%
        # of the source component so both the cell-bearing floor and the
        # deliberately broad rupture bays are observable.
        target_pixels = round(int(source.sum()) * 0.36)

        priority = _residual_fragmentation_priority(
            source_component=source,
            legal_envelope=source,
            default_priority=ndimage.distance_transform_edt(source),
            minimum_residual_components=3,
            maximum_residual_components=6,
            minimum_residual_component_area_px=192,
            minimum_residual_spacing_px=24,
            minimum_residual_component_fraction=0.08,
            maximum_dominant_residual_component_fraction=0.75,
            target_change_pixels=target_pixels,
        )

        eligible_ids = np.flatnonzero(source)
        order = np.argsort(priority.ravel()[eligible_ids], kind="stable")
        changed = np.zeros_like(source)
        changed.ravel()[eligible_ids[order[:target_pixels]]] = True
        local_radius = ndimage.distance_transform_edt(changed)[changed]

        narrow_radius = float(np.percentile(local_radius, 25.0))
        rupture_radius = float(np.percentile(local_radius, 99.0))
        self.assertGreaterEqual(rupture_radius, 19.0)
        self.assertGreaterEqual(rupture_radius, 4.3 * narrow_radius)
        residual_labels, residual_count = ndimage.label(
            source & ~changed, structure=np.ones((3, 3), dtype=bool)
        )
        self.assertGreaterEqual(
            _minimum_component_spacing_px(residual_labels, residual_count),
            24.0,
        )

    def test_fragmentation_cleanup_rejects_holey_balance_bridge_and_erases_satellite(
        self,
    ):
        shape = (160, 400)
        source = np.zeros(shape, dtype=bool)
        source[5:155, 5:395] = True
        target = ~source
        change = np.zeros(shape, dtype=bool)
        change[5:155, 80:100] = True
        change[5:155, 250:270] = True
        # This target-enclosed 160-pixel island clears the absolute raster
        # floor but is far below the required relative residual-focus share.
        change[60:80, 86:94] = False
        priority = np.zeros(shape, dtype=float)
        work = SimpleNamespace(
            source_component=source,
            legal_source=source,
            priority=priority,
            planned=SimpleNamespace(
                source_component_id="tumor:1",
                target_component_id="stroma:1",
            ),
        )
        holey_bridge = np.zeros(shape, dtype=bool)
        holey_bridge[50, 252:262] = True
        holey_bridge[59, 252:262] = True
        holey_bridge[50:60, 252] = True
        holey_bridge[50:60, 261] = True

        with mock.patch(
            "phase3_mask_edit_refine.execution._fragmentation_balance_bridge_reclaim",
            return_value=holey_bridge,
        ):
            cleaned, audit = _rebalance_fragmentation_residual_islands(
                (change,),
                works=(work,),
                source_region=source,
                target_region=target,
                minimum_residual_components=3,
                maximum_residual_components=6,
                minimum_residual_component_area_px=96,
                minimum_residual_spacing_px=8,
                residual_area_floor_fraction=0.3,
                minimum_residual_component_fraction=0.025,
                maximum_dominant_residual_component_fraction=0.75,
            )

        self.assertTrue(audit["applied"], audit)
        self.assertEqual(audit["balance_bridge_pixels_reclaimed"], 0)
        self.assertEqual(audit["balance_pixels_added"], 160)
        self.assertEqual(np.count_nonzero(cleaned[0]), np.count_nonzero(change))
        labels, count = ndimage.label(
            source & ~cleaned[0], structure=np.ones((3, 3), dtype=bool)
        )
        sizes = np.bincount(labels.ravel())[1:]
        self.assertEqual(count, 3)
        self.assertGreaterEqual(float(np.min(sizes / sizes.sum())), 0.025)

    def test_fragmentation_organic_field_has_effective_dynamic_range(self):
        shape = (96, 120)
        source = np.zeros(shape, dtype=bool)
        source[12:84, 10:110] = True
        raw = np.linspace(-0.002, 0.003, num=np.prod(shape)).reshape(shape)

        normalized = _normalized_organic_field(raw, source)

        supported = np.abs(normalized[source])
        self.assertGreaterEqual(float(np.percentile(supported, 95.0)), 0.95)
        self.assertLessEqual(float(np.max(supported)), 1.0)

    def test_fragmentation_warp_field_cannot_fold_at_short_scale(self):
        source = np.ones((192, 256), dtype=bool)

        field = _low_frequency_organic_field(
            source.shape,
            support=source,
            seed=41,
            correlation_px=48.0,
        )

        maximum_step = max(
            float(np.max(np.abs(np.diff(field, axis=0)))),
            float(np.max(np.abs(np.diff(field, axis=1)))),
        )
        self.assertLess(maximum_step, 0.08)

    def test_footprint_priority_prefers_broad_shallow_external_retreat(self):
        shape = (160, 220)
        source = np.zeros(shape, dtype=bool)
        source[20:140, 20:200] = True
        interface = np.zeros(shape, dtype=bool)
        interface[20:140, 20] = True
        default = ndimage.distance_transform_edt(~interface)

        priority = _natural_external_retreat_priority(
            source_component=source,
            legal_envelope=source,
            interface_mask=interface,
            anchor_mask=interface,
            default_priority=default,
        )
        eligible_ids = np.flatnonzero(source)
        order = np.argsort(priority.ravel()[eligible_ids], kind="stable")
        changed = np.zeros_like(source)
        target_pixels = round(int(source.sum()) * 0.15)
        changed.ravel()[eligible_ids[order[:target_pixels]]] = True
        rows, cols = np.where(changed)

        self.assertEqual(
            ndimage.label(changed, structure=np.ones((3, 3), dtype=bool))[1],
            1,
        )
        self.assertGreaterEqual(int(rows.max() - rows.min() + 1), 110)
        self.assertLessEqual(float(np.percentile(cols - 20, 95)), 32.0)

    def test_footprint_priority_tapers_selected_anchor_sector_ends(self):
        shape = (160, 220)
        source = np.zeros(shape, dtype=bool)
        source[20:140, 20:200] = True
        interface = np.zeros(shape, dtype=bool)
        interface[20:140, 20] = True
        anchor = np.zeros(shape, dtype=bool)
        anchor[40:120, 20] = True
        _, nearest_interface = ndimage.distance_transform_edt(
            ~interface, return_indices=True
        )
        legal = source & anchor[nearest_interface[0], nearest_interface[1]]
        unit_profile = DepthProfile(
            mode="multi_lobe",
            peak_depth_px=1.0,
            edge_depth_px=0.55,
            taper_fraction=0.24,
            lobe_count=3,
            noise_amplitude_px=0.0,
            noise_correlation_px=24.0,
        )
        unit_depth = compile_depth_profile_map(
            (anchor,), profile=unit_profile, shape=shape
        )
        default = ndimage.distance_transform_edt(~anchor) / np.maximum(
            unit_depth, 1e-3
        )

        priority = _natural_external_retreat_priority(
            source_component=source,
            legal_envelope=legal,
            interface_mask=interface,
            anchor_mask=anchor,
            default_priority=default,
        )
        eligible_ids = np.flatnonzero(legal)
        order = np.argsort(priority.ravel()[eligible_ids], kind="stable")
        changed = np.zeros_like(source)
        target_pixels = round(int(source.sum()) * 0.15)
        changed.ravel()[eligible_ids[order[:target_pixels]]] = True
        row_widths = np.asarray(
            [int(np.count_nonzero(changed[row])) for row in range(40, 120)]
        )
        endpoint_mean = float(
            np.mean(np.concatenate((row_widths[:4], row_widths[-4:])))
        )

        self.assertLess(endpoint_mean, float(np.median(row_widths)) * 0.8)
        self.assertGreaterEqual(int(np.ptp(row_widths)), 12)

    def test_fragmentation_topology_rejects_two_or_imbalanced_foci(self):
        shape = (60, 90)
        source = np.zeros(shape, dtype=bool)
        source[5:55, 5:85] = True
        target = ~source
        change = np.zeros(shape, dtype=bool)
        change[5:55, 44:54] = True
        work = SimpleNamespace(
            source_component=source,
            planned=SimpleNamespace(target_component_id="stroma:1"),
        )

        audit = _whole_mask_topology_audit(
            source_region=source,
            target_region=target,
            selected_by_work=(change,),
            works=(work,),
            allow_source_component_split=True,
            minimum_residual_components=3,
            maximum_residual_components=6,
            minimum_residual_component_area_px=96,
            minimum_residual_spacing_px=8,
            residual_area_floor_fraction=0.3,
            maximum_residual_area_fraction=0.88,
            minimum_residual_component_fraction=0.08,
            maximum_dominant_residual_component_fraction=0.75,
        )

        self.assertFalse(audit["passed"])
        self.assertEqual(audit["selected_source_components_after"], 2)

    def test_adjacent_addressable_anchors_compile_as_one_continuous_arc(self):
        mask = _glas_circle_mask()
        scene = build_scene_analysis(mask, schema=self.schema, pixel_size_um=0.465)
        interface = max(
            scene.interfaces_for(source_labels=("Stroma",), target_label="Tumor"),
            key=lambda item: item.contact_pixels,
        )
        anchors = tuple(
            scene.anchor_masks[item] for item in interface.anchor_segment_ids
        )
        union = np.logical_or.reduce(anchors)
        profile = DepthProfile(
            mode="tapered_lobe",
            peak_depth_px=30.0,
            edge_depth_px=10.0,
            taper_fraction=0.2,
            lobe_count=1,
            noise_amplitude_px=3.0,
            noise_correlation_px=16.0,
        )
        split_map = compile_depth_profile_map(
            anchors, profile=profile, shape=mask.shape
        )
        union_map = compile_depth_profile_map(
            (union,), profile=profile, shape=mask.shape
        )
        np.testing.assert_allclose(split_map, union_map)

    def test_addressable_anchor_partition_is_bounded_connected_and_exhaustive(self):
        mask = _glas_circle_mask()
        scene = build_scene_analysis(mask, schema=self.schema, pixel_size_um=0.465)
        interface = max(
            scene.interfaces_for(source_labels=("Stroma",), target_label="Tumor"),
            key=lambda item: item.contact_pixels,
        )
        anchors = tuple(
            scene.anchor_masks[item] for item in interface.anchor_segment_ids
        )
        self.assertGreaterEqual(len(anchors), 2)
        self.assertLessEqual(len(anchors), 8)
        self.assertTrue(
            all(
                ndimage.label(anchor, structure=np.ones((3, 3), dtype=bool))[1] == 1
                for anchor in anchors
            )
        )
        np.testing.assert_array_equal(
            np.logical_or.reduce(anchors),
            scene.interface_masks[interface.interface_id],
        )

    def test_fragmentation_full_interface_closes_preflight_anchor_sampling_gaps(self):
        mask = _glas_circle_mask()
        scene = build_scene_analysis(mask, schema=self.schema, pixel_size_um=0.465)
        interface = max(
            scene.interfaces_for(source_labels=("Tumor",), target_label="Stroma"),
            key=lambda item: item.contact_pixels,
        )
        case = _case(primitive="residual-tumor-fragmentation-v1", area=0.10)
        plan = _manual_plan(
            case=case,
            interface=interface,
            source_label="Tumor",
            target_label="Stroma",
            area=0.10,
            supporting_rules=_bundle_ids(self._bundle()),
            band_max=48.0,
        )
        plan = replace(
            plan,
            tool_program=replace(
                plan.tool_program,
                parameter_ranges={
                    **plan.tool_program.parameter_ranges,
                    "tissue_geometry_mode": "residual_fragmentation",
                    "minimum_residual_components": 3,
                    "maximum_residual_components": 6,
                    "minimum_residual_component_area_px": 32,
                    "minimum_residual_spacing_px": 4,
                },
            ),
        )
        rows = np.indices(mask.shape)[0]
        sparse_anchor_masks = dict(scene.anchor_masks)
        for anchor_id in interface.anchor_segment_ids:
            sparse_anchor_masks[anchor_id] = scene.anchor_masks[anchor_id] & (
                rows % 4 == 0
            )
        sparse_scene = replace(scene, anchor_masks=sparse_anchor_masks)

        full_works = _prepare_compiler_work(
            plan,
            source_mask=mask,
            source_region=mask == 12,
            scene=sparse_scene,
        )
        full_legal = np.logical_or.reduce([item.legal_source for item in full_works])
        interface_mask = scene.interface_masks[interface.interface_id]
        expected = scene.component_masks[interface.source_component_id] & (
            ndimage.distance_transform_edt(~interface_mask) <= 48.0
        )
        self.assertFalse(np.any(expected & ~full_legal))

        kept_anchor = max(
            interface.anchor_segment_ids,
            key=lambda item: int(np.count_nonzero(sparse_anchor_masks[item])),
        )
        partial_plan = replace(
            plan,
            candidate_interfaces=(
                replace(
                    plan.candidate_interfaces[0],
                    execution_contract=replace(
                        plan.candidate_interfaces[0].execution_contract,
                        anchor_segment_ids=(kept_anchor,),
                    ),
                ),
            ),
        )
        partial_works = _prepare_compiler_work(
            partial_plan,
            source_mask=mask,
            source_region=mask == 12,
            scene=sparse_scene,
        )
        partial_legal = np.logical_or.reduce(
            [item.legal_source for item in partial_works]
        )
        # Fragmentation is a component-scale tissue operation.  Sparse
        # cell-feasibility anchors may govern where seam cells are sampled,
        # but they must not carve holes in the tissue ownership envelope.
        self.assertFalse(np.any(expected & ~partial_legal))

    def test_multi_interface_candidate_records_both_anchors(self):
        size = 160
        rows, cols = np.ogrid[:size, :size]
        mask = np.full((size, size), 2, dtype=np.int64)
        mask[(rows - 52) ** 2 + (cols - 48) ** 2 <= 22**2] = 12
        mask[(rows - 108) ** 2 + (cols - 112) ** 2 <= 24**2] = 12
        scene = build_scene_analysis(mask, schema=self.schema, pixel_size_um=0.465)
        interfaces = sorted(
            scene.interfaces_for(source_labels=("Stroma",), target_label="Tumor"),
            key=lambda item: -item.contact_pixels,
        )[:2]
        bundle = self._bundle()
        case = _case(primitive="tumor-burden-increase-v1", area=0.025)
        base = _manual_plan(
            case=case,
            interface=interfaces[0],
            source_label="Stroma",
            target_label="Tumor",
            area=0.025,
            supporting_rules=_bundle_ids(bundle),
            band_max=32.0,
        )
        planned = tuple(
            replace(
                base.candidate_interfaces[0],
                interface_id=item.interface_id,
                source_component_id=item.source_component_id,
                target_component_id=item.target_component_id,
                execution_contract=replace(
                    base.candidate_interfaces[0].execution_contract,
                    anchor_segment_ids=item.anchor_segment_ids,
                    area_allocation_fraction=0.5,
                ),
            )
            for item in interfaces
        )
        plan = replace(
            base,
            candidate_interfaces=planned,
            tool_program=replace(base.tool_program, candidate_count=4),
        )
        candidates = generate_candidates(
            mask,
            schema=self.schema,
            scene=scene,
            plan=plan,
            bundle=bundle,
            seed=case.seed,
        )
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertEqual(len(candidate.tool_trace["interface_ids"]), 2)
            self.assertEqual(
                int(candidate.change_region.sum()),
                case.area_budget.target_pixels(mask, mask == 2),
            )

    def test_adding_interface_cannot_discard_existing_executable_capacity(self):
        size = 160
        rows, cols = np.ogrid[:size, :size]
        mask = np.full((size, size), 2, dtype=np.int64)
        mask[(rows - 52) ** 2 + (cols - 48) ** 2 <= 22**2] = 12
        mask[(rows - 108) ** 2 + (cols - 112) ** 2 <= 24**2] = 12
        scene = build_scene_analysis(mask, schema=self.schema, pixel_size_um=0.465)
        interfaces = sorted(
            scene.interfaces_for(source_labels=("Stroma",), target_label="Tumor"),
            key=lambda item: -item.contact_pixels,
        )[:2]
        bundle = self._bundle()
        case = _case(primitive="tumor-burden-increase-v1", area=0.025)
        base = _manual_plan(
            case=case,
            interface=interfaces[0],
            source_label="Stroma",
            target_label="Tumor",
            area=0.025,
            supporting_rules=_bundle_ids(bundle),
            band_max=32.0,
        )
        planned = tuple(
            replace(
                base.candidate_interfaces[0],
                interface_id=item.interface_id,
                source_component_id=item.source_component_id,
                target_component_id=item.target_component_id,
                execution_contract=replace(
                    base.candidate_interfaces[0].execution_contract,
                    anchor_segment_ids=item.anchor_segment_ids,
                    area_allocation_fraction=0.5,
                ),
            )
            for item in interfaces
        )
        single_works = _prepare_compiler_work(
            base,
            source_mask=mask,
            source_region=mask == 2,
            scene=scene,
        )
        multi_works = _prepare_compiler_work(
            replace(base, candidate_interfaces=planned),
            source_mask=mask,
            source_region=mask == 2,
            scene=scene,
        )
        single_union = np.logical_or.reduce(
            [item.legal_source for item in single_works]
        )
        multi_union = np.logical_or.reduce([item.legal_source for item in multi_works])
        self.assertFalse(np.any(single_union & ~multi_union))
        self.assertGreaterEqual(
            int(np.count_nonzero(multi_union)),
            int(np.count_nonzero(single_union)),
        )

    def test_only_explicitly_selected_target_components_may_merge(self):
        mask = np.full((64, 64), 2, dtype=np.int64)
        mask[24:40, 8:20] = 12
        mask[24:40, 44:56] = 12
        scene = build_scene_analysis(mask, schema=self.schema, pixel_size_um=0.465)
        interfaces = sorted(
            scene.interfaces_for(source_labels=("Stroma",), target_label="Tumor"),
            key=lambda item: item.target_component_id,
        )
        self.assertEqual(len({item.target_component_id for item in interfaces}), 2)
        selected = []
        for target_component_id in sorted(
            {item.target_component_id for item in interfaces}
        ):
            selected.append(
                max(
                    [
                        item
                        for item in interfaces
                        if item.target_component_id == target_component_id
                    ],
                    key=lambda item: item.contact_pixels,
                )
            )
        bundle = self._bundle()
        case = _case(primitive="tumor-burden-increase-v1", area=0.05)
        base = _manual_plan(
            case=case,
            interface=selected[0],
            source_label="Stroma",
            target_label="Tumor",
            area=0.05,
            supporting_rules=_bundle_ids(bundle),
            band_max=40.0,
        )
        planned = tuple(
            replace(
                base.candidate_interfaces[0],
                interface_id=item.interface_id,
                source_component_id=item.source_component_id,
                target_component_id=item.target_component_id,
                execution_contract=replace(
                    base.candidate_interfaces[0].execution_contract,
                    anchor_segment_ids=item.anchor_segment_ids,
                    area_allocation_fraction=0.5,
                ),
            )
            for item in selected
        )
        change = np.zeros_like(mask, dtype=bool)
        change[28:36, 20:44] = True
        target = np.array(mask, copy=True)
        target[change] = 12
        candidate = CandidateMask(
            "merge-fixture",
            selected[0].interface_id,
            "fixture",
            target,
            change,
            {
                "interface_ids": [item.interface_id for item in selected],
                "source_component_ids": sorted(
                    {item.source_component_id for item in selected}
                ),
                "target_component_ids": [item.target_component_id for item in selected],
                "target_fine_ids": [12],
            },
        )
        selected_plan = replace(base, candidate_interfaces=planned)
        selected_report = self.gates.run(
            GateContext(
                case=case,
                source_mask=mask,
                schema=self.schema,
                scene=scene,
                bundle=bundle,
                plan=selected_plan,
                candidate=candidate,
            )
        )
        selected_topology = next(
            item
            for item in selected_report.checks
            if item.check_id == "edited_label_topology"
        )
        self.assertTrue(selected_topology.metrics["target_merge"])
        self.assertTrue(selected_topology.passed, selected_topology.metrics)

        unselected_plan = replace(base, candidate_interfaces=(planned[0],))
        unselected_report = self.gates.run(
            GateContext(
                case=case,
                source_mask=mask,
                schema=self.schema,
                scene=scene,
                bundle=bundle,
                plan=unselected_plan,
                candidate=candidate,
            )
        )
        unselected_topology = next(
            item
            for item in unselected_report.checks
            if item.check_id == "edited_label_topology"
        )
        self.assertFalse(unselected_topology.passed)
        self.assertTrue(unselected_topology.metrics["unallowed_target_merge"])

    def test_disconnected_component_pair_interfaces_are_separate_ids(self):
        mask = np.full((64, 64), 2, dtype=np.int64)
        mask[20:44, 20:44] = 12
        # A third label blocks the top and bottom contacts. Left/right stroma
        # remain one component via the outer border, but form two edit anchors.
        mask[19, 14:50] = 5
        mask[44, 14:50] = 5
        scene = build_scene_analysis(mask, schema=self.schema, pixel_size_um=0.465)
        interfaces = scene.interfaces_for(
            source_labels=("Stroma",), target_label="Tumor"
        )
        matching_pairs: dict[tuple[str, str], list[str]] = {}
        for interface in interfaces:
            matching_pairs.setdefault(
                (interface.source_component_id, interface.target_component_id), []
            ).append(interface.interface_id)
        segmented = [ids for ids in matching_pairs.values() if len(ids) >= 2]
        self.assertTrue(segmented)
        self.assertTrue(
            all(":seg:" in interface_id for ids in segmented for interface_id in ids)
        )


class _PassingCritic:
    name = "passing_test_critic"
    supports_pathology_vision = True

    def review(self, *, bundle, candidates, **kwargs):
        del kwargs
        candidate = candidates[0]
        hard_visual_rules = tuple(
            rule.rule_id
            for rule in bundle.active_rules
            if rule.severity == "hard" and rule.critic_requirement
        ) + tuple(
            item.constraint_id
            for item in bundle.active_mask_constraints
            if item.critic_requirement
        )
        return CriticResult(
            rankings=(
                CriticRanking(
                    candidate_id=candidate.candidate_id,
                    score=0.95,
                    confidence=0.95,
                    supporting_rule_ids=hard_visual_rules,
                ),
            ),
            abstain=False,
            summary="fixture visual review passed",
            usage={"provider": self.name},
        )


class WorkflowTests(unittest.TestCase):
    def test_research_workflow_runs_end_to_end_without_legacy_fallback(self):
        critic_calls = []

        class RecordingCritic(_PassingCritic):
            def review(self, **kwargs):
                critic_calls.append(kwargs)
                candidate = kwargs["candidates"][0]
                return CriticResult(
                    rankings=(
                        CriticRanking(
                            candidate_id=candidate.candidate_id,
                            score=1.0,
                            confidence=1.0,
                            supporting_rule_ids=(),
                            veto_reasons=(
                                "histologically evident keratin pearl",
                            ),
                        ),
                    ),
                    abstain=False,
                    summary="provider-injected histology veto",
                    usage={"provider": "adversarial_free_text_critic"},
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask_path = root / "mask.npy"
            image_path = root / "image.png"
            mask = _glas_circle_mask()
            np.save(mask_path, mask, allow_pickle=False)
            image = np.full((*mask.shape, 3), 220, dtype=np.uint8)
            image[mask == 12] = (175, 90, 130)
            Image.fromarray(image).save(image_path)
            case = _case(
                primitive="tumor-burden-increase-v1",
                area=0.04,
                image_uri=str(image_path),
                mask_uri=str(mask_path),
                provenance={
                    "source_image_sha256": _sha(image_path),
                    "source_mask_sha256": _sha(mask_path),
                    "preprocessing_revision": "synthetic-glas-v1",
                    "original_instance_mask_digest": "synthetic-instance-digest",
                    "patch_grade": "moderately_differentiated",
                },
            )
            workflow = MaskEditRefineWorkflow(
                planner=HeuristicInterfacePlanner(),
                critic=RecordingCritic(),
                config=WorkflowConfig(production=False, critic_min_score_margin=0.0),
            )
            result = workflow.run(case, output_root=root / "artifacts")
            self.assertEqual(result.status, "selected_research", result.abstain_reasons)
            self.assertIsNotNone(result.target_mask)
            self.assertTrue(
                all(
                    report.passed
                    for report in result.gate_reports
                    if report.candidate_id == result.selected_candidate_id
                )
            )
            self.assertTrue(Path(result.artifact_paths["selection"]).is_file())
            self.assertTrue(Path(result.artifact_paths["active_skills"]).is_file())
            self.assertFalse(critic_calls)
            self.assertEqual(
                result.usage["planner_calls"][0]["provider"],
                "heuristic_interface_planner",
            )
            self.assertEqual(
                result.critic_result.usage["provider"],
                "compiler_owned_gate_certificate_selector",
            )
            self.assertFalse(result.critic_result.rankings[0].veto_reasons)

    def test_non_breast_legacy_workflow_rejects_caller_planner_injection(self):
        class CallerPlanner:
            def create_plan(self, **_kwargs):
                raise AssertionError("caller Planner must not be invoked")

        case = _case(primitive="tumor-burden-increase-v1", area=0.04)
        with self.assertRaisesRegex(
            RefineContractError, "rejects caller-supplied Planner"
        ):
            validate_non_breast_legacy_workflow_planner(
                case,
                planner=CallerPlanner(),
                escalation_planner=None,
            )


class AgentContractTests(unittest.TestCase):
    def test_non_breast_repository_startup_rejects_rule_without_typed_authority(self):
        source_catalog = (
            Path(__file__).resolve().parents[1]
            / "phase3_mask_edit_refine"
            / "skills"
            / "catalog"
        )
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog"
            shutil.copytree(source_catalog, catalog)
            rules_path = (
                catalog
                / "annotation-profile"
                / "glas-gland-v1"
                / "references"
                / "rules.json"
            )
            payload = json.loads(rules_path.read_text(encoding="utf-8"))
            active = next(
                item
                for item in payload["rules"]
                if not item["scope"].startswith("reader_only_")
            )
            active["observation_authority"] = []
            rules_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RefineContractError, "lacks typed observation authority"
            ):
                SkillRepository(catalog)

    def test_compose_rejects_unbound_structure_even_with_generic_mask_checker(self):
        repository = SkillRepository()
        pathology = repository.get("colorectal-adenocarcinoma-v1")
        template = pathology.rules[0]
        injected = replace(
            template,
            rule_id="adversarial.identify_keratin_pearl",
            scope="planner_and_critic",
            applies_when={"structure": "keratin_pearl"},
            severity="hard",
            deterministic_check_id="edited_label_topology",
            critic_requirement=None,
            execution_role="deterministic_mask_invariant",
            observation_authority=(
                ObservationAuthority("tissue_mask", "source_mask_sha256"),
                ObservationAuthority("scene_graph", "compiler_scene_graph"),
                ObservationAuthority(
                    "deterministic_metric", "checker:edited_label_topology"
                ),
            ),
            preference_rule_id=None,
        )
        repository._skills[pathology.skill_id] = replace(
            pathology, rules=(*pathology.rules, injected)
        )
        with self.assertRaisesRegex(
            RefineContractError, "structure without a typed profile-auxiliary"
        ):
            repository.compose(
                pathology_domain_id="colorectal-adenocarcinoma-v1",
                annotation_profile_id="glas-gland-v1",
                primitive_id="tumor-burden-increase-v1",
                production=False,
                available_checker_ids=GateRegistry().available_checker_ids,
            )

    def test_compose_applies_when_filters_mismatched_rules(self):
        repository = SkillRepository()
        pathology = repository.get("colorectal-adenocarcinoma-v1")
        source = repository.get("glas-gland-v1").rules[1]
        injected = replace(
            source,
            rule_id="adversarial.only_tumor_decrease",
            applies_when={"primitive": "tumor-burden-decrease-v1"},
        )
        repository._skills[pathology.skill_id] = replace(
            pathology, rules=(*pathology.rules, injected)
        )
        bundle = repository.compose(
            pathology_domain_id="colorectal-adenocarcinoma-v1",
            annotation_profile_id="glas-gland-v1",
            primitive_id="tumor-burden-increase-v1",
            production=False,
            available_checker_ids=GateRegistry().available_checker_ids,
        )
        self.assertNotIn(
            injected.rule_id, {item.rule_id for item in bundle.active_rules}
        )

    def test_reader_only_rules_and_freeform_pathology_never_enter_execution_prompts(self):
        repository = SkillRepository()
        gates = GateRegistry()
        case = _case(primitive="tumor-burden-increase-v1", area=0.04)
        bundle = repository.compose(
            pathology_domain_id=case.pathology_domain_id,
            annotation_profile_id=case.annotation_profile_id,
            primitive_id=case.primitive_id,
            production=False,
            available_checker_ids=gates.available_checker_ids,
            case_provenance=case.provenance,
        )
        scene = build_scene_analysis(
            _glas_circle_mask(),
            schema=repository.annotation_schema(case.annotation_profile_id),
            pixel_size_um=case.pixel_size_um,
        )
        prompts = (
            json.loads(_planner_prompt(case=case, scene=scene, bundle=bundle)),
            json.loads(
                _critic_prompt(
                    case=case,
                    bundle=bundle,
                    gate_reports=(),
                    passed_ids=(),
                )
            ),
        )
        reader_ids = {
            rule.rule_id
            for package in (
                repository.get(case.pathology_domain_id),
                repository.get(case.annotation_profile_id),
            )
            for rule in package.rules
            if rule.scope.startswith("reader_only_")
        }
        for prompt in prompts:
            rendered = json.dumps(prompt, sort_keys=True)
            with self.subTest(prompt_keys=sorted(prompt)):
                self.assertTrue(all(rule_id not in rendered for rule_id in reader_ids))
                self.assertNotIn('"claim"', rendered)
                self.assertNotIn('"required_observation"', rendered)
                self.assertNotIn('"expected_morphology"', rendered)
                self.assertNotIn("source_image_uri", rendered)
                self.assertNotIn(case.source_image_uri, rendered)
                self.assertNotIn("source_paths", rendered)

    def test_panda_bundle_never_injects_pattern3_identity_into_p4_p5_execution(self):
        repository = SkillRepository()
        provenance = {
            "source_mask_sha256": "a" * 64,
            "provider": "radboud",
            "preprocessing_revision": "panda-fixture-v1",
            "original_label_map_digest": "b" * 64,
        }
        bundle = repository.compose(
            pathology_domain_id="prostate-adenocarcinoma-v1",
            annotation_profile_id="panda-gleason-v1",
            primitive_id="tumor-burden-increase-v1",
            production=False,
            available_checker_ids=GateRegistry().available_checker_ids,
            case_provenance=provenance,
        )
        rendered = json.dumps(bundle.to_metadata(), sort_keys=True)
        self.assertNotIn("pattern3_remains_separate", rendered)
        self.assertNotIn("well_formed_gland", rendered)
        self.assertNotIn("gleason_pattern_3", rendered)

    def test_panda_sparse_topology_rule_is_unconditional_and_not_caller_switched(self):
        repository = SkillRepository()
        base_provenance = {
            "source_mask_sha256": "a" * 64,
            "provider": "radboud",
            "preprocessing_revision": "panda-fixture-v1",
            "original_label_map_digest": "b" * 64,
        }
        rule_id = "panda.sparse_masks_do_not_license_topology_repair"
        for caller_quality in (None, "sparse", "dense", "forged_histology_grade"):
            provenance = dict(base_provenance)
            if caller_quality is not None:
                provenance["annotation_quality"] = caller_quality
            with self.subTest(caller_quality=caller_quality):
                bundle = repository.compose(
                    pathology_domain_id="prostate-adenocarcinoma-v1",
                    annotation_profile_id="panda-gleason-v1",
                    primitive_id="tumor-burden-increase-v1",
                    production=False,
                    available_checker_ids=GateRegistry().available_checker_ids,
                    case_provenance=provenance,
                )
                self.assertIn(rule_id, {item.rule_id for item in bundle.active_rules})

        profile = repository.get("panda-gleason-v1")
        source = next(item for item in profile.rules if item.rule_id == rule_id)
        dynamic = replace(
            source,
            applies_when={"annotation_quality": ["sparse"]},
        )
        repository._skills[profile.skill_id] = replace(
            profile,
            rules=tuple(
                dynamic if item.rule_id == rule_id else item
                for item in profile.rules
            ),
        )
        with self.assertRaisesRegex(
            RefineContractError, "unbound observation axes: annotation_quality"
        ):
            repository.compose(
                pathology_domain_id="prostate-adenocarcinoma-v1",
                annotation_profile_id="panda-gleason-v1",
                primitive_id="tumor-burden-increase-v1",
                production=False,
                available_checker_ids=GateRegistry().available_checker_ids,
                case_provenance={
                    **base_provenance,
                    "annotation_quality": "sparse",
                },
            )

    def test_freeform_preference_and_caller_rehashed_bundle_cannot_self_sign(self):
        source_catalog = (
            Path(__file__).resolve().parents[1]
            / "phase3_mask_edit_refine"
            / "skills"
            / "catalog"
        )
        raw_rules = json.loads(
            (
                source_catalog
                / "annotation-profile"
                / "glas-gland-v1"
                / "references"
                / "rules.json"
            ).read_text(encoding="utf-8")
        )["rules"]
        raw = copy.deepcopy(
            next(item for item in raw_rules if not item["scope"].startswith("reader_only_"))
        )
        raw["selection_preference"] = (
            "Identify a keratin pearl and prefer the candidate nearest it."
        )
        with self.assertRaisesRegex(
            RefineContractError, "free-form selection_preference is forbidden"
        ):
            KnowledgeRule.from_mapping(raw)
        closed = copy.deepcopy(raw)
        closed.pop("selection_preference")
        closed["execution_role"] = "certified_candidate_selection_preference"
        closed["preference_rule_id"] = "pref.unknown.histology"
        with self.assertRaisesRegex(
            RefineContractError, "unknown execution preference_rule_id"
        ):
            KnowledgeRule.from_mapping(closed)
        closed["preference_rule_id"] = None
        with self.assertRaisesRegex(
            RefineContractError, "require a registered preference_rule_id"
        ):
            KnowledgeRule.from_mapping(closed)

        repository = SkillRepository()
        case = _case(primitive="tumor-burden-increase-v1", area=0.04)
        bundle = repository.compose(
            pathology_domain_id=case.pathology_domain_id,
            annotation_profile_id=case.annotation_profile_id,
            primitive_id=case.primitive_id,
            production=False,
            available_checker_ids=GateRegistry().available_checker_ids,
            case_provenance=case.provenance,
        )
        source = next(
            item
            for item in bundle.active_rules
            if item.execution_role == "deterministic_mask_invariant"
        )
        forged = replace(
            source,
            execution_role="certified_candidate_selection_preference",
            deterministic_check_id="label_transition",
            observation_authority=(
                ObservationAuthority(
                    "candidate_certificate", "compiler_candidate_certificate"
                ),
                ObservationAuthority(
                    "deterministic_metric", "checker:label_transition"
                ),
            ),
            preference_rule_id="pref.capacity_margin.maximize",
        )
        repository._skills[bundle.pathology_domain.skill_id] = replace(
            bundle.pathology_domain,
            rules=(forged, *bundle.pathology_domain.rules[1:]),
        )
        with self.assertRaisesRegex(
            RefineContractError, "preference metric is detached from authority"
        ):
            repository.compose(
                pathology_domain_id=case.pathology_domain_id,
                annotation_profile_id=case.annotation_profile_id,
                primitive_id=case.primitive_id,
                production=False,
                available_checker_ids=GateRegistry().available_checker_ids,
                case_provenance=case.provenance,
            )
        forged_bundle = replace(
            bundle,
            active_rules=(forged, *bundle.active_rules[1:]),
            authority_binding_sha256=hashlib.sha256(
                b"caller-side rehash is not a repository signature"
            ).hexdigest(),
        )
        with self.assertRaisesRegex(
            RefineContractError, "repository-issued sealed capability"
        ):
            validate_active_bundle_authority(
                forged_bundle,
                case_provenance=case.provenance,
            )

    def test_live_bundle_binding_covers_source_bytes_scene_case_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask_path = root / "mask.npy"
            image_path = root / "image.png"
            mask = _glas_circle_mask()
            np.save(mask_path, mask, allow_pickle=False)
            Image.fromarray(np.zeros((*mask.shape, 3), dtype=np.uint8)).save(image_path)
            case = _case(
                primitive="tumor-burden-increase-v1",
                area=0.04,
                image_uri=str(image_path),
                mask_uri=str(mask_path),
                provenance={
                    "source_image_sha256": _sha(image_path),
                    "source_mask_sha256": _sha(mask_path),
                    "preprocessing_revision": "synthetic-glas-v1",
                    "original_instance_mask_digest": "instance-digest",
                    "patch_grade": "moderately_differentiated",
                },
            )
            repository = SkillRepository()
            bundle = repository.compose(
                pathology_domain_id=case.pathology_domain_id,
                annotation_profile_id=case.annotation_profile_id,
                primitive_id=case.primitive_id,
                production=False,
                available_checker_ids=GateRegistry().available_checker_ids,
                case_provenance=case.provenance,
            )
            scene = build_scene_analysis(
                mask,
                schema=repository.annotation_schema(case.annotation_profile_id),
                pixel_size_um=case.pixel_size_um,
            )
            bound = bind_active_bundle_to_case(bundle, case=case, scene=scene)
            validate_active_bundle_authority(
                bound,
                case_provenance=case.provenance,
                require_live_binding=True,
            )
            self.assertEqual(bound.live_authority["status"], "bound")
            self.assertEqual(
                bound.live_authority["source_mask_live_sha256"], _sha(mask_path)
            )
            tampered = replace(
                bound,
                live_authority={
                    **bound.live_authority,
                    "budget_sha256": "f" * 64,
                },
            )
            with self.assertRaisesRegex(
                RefineContractError, "repository-issued sealed capability"
            ):
                validate_active_bundle_authority(
                    tampered,
                    case_provenance=case.provenance,
                    require_live_binding=True,
                )
            changed = np.array(mask, copy=True)
            changed[0, 0] = 12
            np.save(mask_path, changed, allow_pickle=False)
            with self.assertRaisesRegex(
                RefineContractError, "source-mask bytes are detached"
            ):
                bind_active_bundle_to_case(bundle, case=case, scene=scene)

    def test_cli_closes_non_breast_openai_before_client_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_path = root / "case.json"
            case_path.write_text(
                json.dumps(
                    _case(
                        primitive="tumor-burden-increase-v1",
                        area=0.04,
                    ).to_metadata()
                ),
                encoding="utf-8",
            )
            with mock.patch.object(
                mask_cli, "OpenAIResponsesJSONClient"
            ) as client_constructor, mock.patch("builtins.print"):
                exit_code = mask_cli.main(
                    [
                        "run",
                        "--case",
                        str(case_path),
                        "--output-root",
                        str(root / "output"),
                        "--research",
                    ]
                )
            self.assertEqual(exit_code, 2)
            client_constructor.assert_not_called()

    def test_authority_failure_prevents_planner_and_critic_provider_calls(self):
        class NoCallClient:
            model = "fixture"

            def __init__(self):
                self.calls = 0

            def call(self, **_kwargs):
                self.calls += 1
                raise AssertionError("provider must not be called")

        repository = SkillRepository()
        case = _case(primitive="tumor-burden-increase-v1", area=0.04)
        bundle = repository.compose(
            pathology_domain_id=case.pathology_domain_id,
            annotation_profile_id=case.annotation_profile_id,
            primitive_id=case.primitive_id,
            production=False,
            available_checker_ids=GateRegistry().available_checker_ids,
            case_provenance=case.provenance,
        )
        detached_index = next(
            index
            for index, rule in enumerate(bundle.active_rules)
            if rule.execution_role == "deterministic_mask_invariant"
        )
        detached_source = bundle.active_rules[detached_index]
        detached = replace(
            detached_source,
            observation_authority=(
                ObservationAuthority("tissue_mask", "caller_named_panel"),
                ObservationAuthority("scene_graph", "compiler_scene_graph"),
                ObservationAuthority(
                    "deterministic_metric",
                    f"checker:{detached_source.deterministic_check_id}",
                ),
            ),
        )
        invalid_rules = list(bundle.active_rules)
        invalid_rules[detached_index] = detached
        invalid_bundle = replace(
            bundle, active_rules=tuple(invalid_rules)
        )
        with self.assertRaisesRegex(
            RefineContractError, "repository-issued sealed capability"
        ):
            validate_active_bundle_authority(
                invalid_bundle,
                case_provenance=case.provenance,
            )

        planner_client = NoCallClient()
        with self.assertRaisesRegex(
            RefineContractError, "legacy online Planner/Critic is disabled"
        ):
            OpenAIMultimodalPlanner(
                client=planner_client, max_schema_attempts=1
            ).create_plan(
                case=case,
                scene=SimpleNamespace(),
                bundle=invalid_bundle,
                image_paths=(),
            )
        self.assertEqual(planner_client.calls, 0)

        critic_client = NoCallClient()
        with self.assertRaisesRegex(
            RefineContractError, "legacy online Planner/Critic is disabled"
        ):
            OpenAIMultimodalCritic(client=critic_client).review(
                case=case,
                bundle=invalid_bundle,
                candidates=(),
                gate_reports=(),
                image_paths=(),
            )
        self.assertEqual(critic_client.calls, 0)

    def test_non_breast_closed_critic_tokens_reject_free_text_without_using_it(self):
        repository = SkillRepository()
        case = _case(primitive="tumor-burden-increase-v1", area=0.04)
        bundle = repository.compose(
            pathology_domain_id=case.pathology_domain_id,
            annotation_profile_id=case.annotation_profile_id,
            primitive_id=case.primitive_id,
            production=False,
            available_checker_ids=GateRegistry().available_checker_ids,
            case_provenance=case.provenance,
        )
        injected = CriticResult(
            rankings=(
                CriticRanking(
                    candidate_id="candidate-1",
                    score=1.0,
                    confidence=1.0,
                    supporting_rule_ids=(),
                    veto_reasons=("histologically evident keratin pearl",),
                ),
            ),
            abstain=False,
            summary="deterministic_gate_certificate_selection",
            usage={"provider": "compiler_owned_gate_certificate_selector"},
        )
        accepted, reasons = critic_satisfies_hard_rules(
            injected,
            bundle=bundle,
        )
        self.assertFalse(accepted)
        self.assertEqual(reasons, ("invalid_non_breast_compiler_critic_payload",))
        self.assertNotIn("keratin", " ".join(reasons))

    def test_all_non_breast_execution_agents_reject_every_caller_raster_before_api(self):
        class NoCallClient:
            model = "fixture"

            def __init__(self):
                self.calls = 0

            def call(self, **_kwargs):
                self.calls += 1
                raise AssertionError("execution client must not be called")

        profiles = (
            ("colorectal-adenocarcinoma-v1", "glas-gland-v1"),
            ("prostate-adenocarcinoma-v1", "panda-gleason-v1"),
            ("lung-carcinoma-v1", "ignite-semantic-v1"),
            ("melanoma-v1", "puma-semantic-v1"),
            ("oral-squamous-cell-carcinoma-v1", "orca-semantic-v1"),
            ("breast-invasive-carcinoma-v1", "orca-semantic-v1"),
        )
        disguised_paths = (
            None,
            "renamed-neutral-panel.png",
            "self-registered-component-map.png",
            "reader-board-without-he-name.png",
            "arbitrary-caller-raster.bin",
        )
        for domain_id, profile_id in profiles:
            case = replace(
                _case(primitive="tumor-burden-increase-v1", area=0.04),
                pathology_domain_id=domain_id,
                annotation_profile_id=profile_id,
            )
            for disguised_path in disguised_paths:
                with self.subTest(
                    domain_id=domain_id,
                    disguised_path=disguised_path,
                ):
                    planner_client = NoCallClient()
                    with self.assertRaisesRegex(
                        RefineContractError,
                        "legacy online Planner/Critic is disabled",
                    ):
                        OpenAIMultimodalPlanner(
                            client=planner_client,
                            max_schema_attempts=1,
                        ).create_plan(
                            case=case,
                            scene=SimpleNamespace(),
                            bundle=SimpleNamespace(),
                            image_paths=(
                                ()
                                if disguised_path is None
                                else (disguised_path,)
                            ),
                        )
                    self.assertEqual(planner_client.calls, 0)

                    critic_client = NoCallClient()
                    with self.assertRaisesRegex(
                        RefineContractError,
                        "legacy online Planner/Critic is disabled",
                    ):
                        OpenAIMultimodalCritic(client=critic_client).review(
                            case=case,
                            bundle=SimpleNamespace(),
                            candidates=(),
                            gate_reports=(),
                            image_paths=(
                                ()
                                if disguised_path is None
                                else (disguised_path,)
                            ),
                        )
                    self.assertEqual(critic_client.calls, 0)

    def test_non_breast_skill_catalog_uses_positive_typed_authority_only(self):
        repository = SkillRepository()
        non_breast_skill_ids = {
            "glas-gland-v1",
            "panda-gleason-v1",
            "ignite-semantic-v1",
            "puma-semantic-v1",
            "orca-semantic-v1",
            "colorectal-adenocarcinoma-v1",
            "prostate-adenocarcinoma-v1",
            "lung-carcinoma-v1",
            "melanoma-v1",
            "oral-squamous-cell-carcinoma-v1",
        }
        allowed_sources = {
            "instruction_semantic_intent",
            "tissue_mask",
            "nuclei_mask",
            "scene_graph",
            "profile_owned_auxiliary_map",
            "case_provenance",
            "candidate_certificate",
            "deterministic_metric",
        }
        for skill_id in sorted(non_breast_skill_ids):
            package = repository.get(skill_id)
            with self.subTest(skill_id=skill_id):
                self.assertTrue(
                    all(
                        (
                            not rule.deterministic_check_id
                            and not rule.critic_requirement
                            and not rule.execution_role
                            and not rule.observation_authority
                        )
                        if rule.scope.startswith("reader_only_")
                        else (
                            not rule.critic_requirement
                            and bool(rule.execution_role)
                            and bool(rule.observation_authority)
                            and all(
                                item.source in allowed_sources
                                for item in rule.observation_authority
                            )
                        )
                        for rule in package.rules
                    )
                )
                self.assertTrue(
                    all(
                        not item.critic_requirement
                        and "source_he" not in item.observability
                        for item in package.mask_constraints
                    )
                )

    def test_planner_cannot_omit_an_active_mask_constraint(self):
        repository = SkillRepository()
        gates = GateRegistry()
        schema = repository.annotation_schema("glas-gland-v1")
        mask = _glas_circle_mask()
        scene = build_scene_analysis(mask, schema=schema, pixel_size_um=0.465)
        bundle = repository.compose(
            pathology_domain_id="colorectal-adenocarcinoma-v1",
            annotation_profile_id="glas-gland-v1",
            primitive_id="tumor-burden-increase-v1",
            production=False,
            available_checker_ids=gates.available_checker_ids,
        )
        case = _case(primitive="tumor-burden-increase-v1", area=0.04)
        interface = max(
            scene.interfaces_for(source_labels=("Stroma",), target_label="Tumor"),
            key=lambda item: item.contact_pixels,
        )
        omitted = bundle.active_mask_constraints[0].constraint_id
        plan = _manual_plan(
            case=case,
            interface=interface,
            source_label="Stroma",
            target_label="Tumor",
            area=0.04,
            supporting_rules=tuple(
                item for item in _bundle_ids(bundle) if item != omitted
            ),
            band_max=64.0,
        )
        with self.assertRaisesRegex(
            RefineContractError, "omits active mask constraints"
        ):
            validate_edit_plan(plan, case=case, scene=scene, bundle=bundle)

    def test_non_breast_online_planner_never_calls_or_retries_provider(self):
        repository = SkillRepository()
        gates = GateRegistry()
        schema = repository.annotation_schema("glas-gland-v1")
        mask = _glas_circle_mask()
        scene = build_scene_analysis(mask, schema=schema, pixel_size_um=0.465)
        bundle = repository.compose(
            pathology_domain_id="colorectal-adenocarcinoma-v1",
            annotation_profile_id="glas-gland-v1",
            primitive_id="tumor-burden-increase-v1",
            production=False,
            available_checker_ids=gates.available_checker_ids,
        )
        case = _case(primitive="tumor-burden-increase-v1", area=0.04)
        interface = max(
            scene.interfaces_for(source_labels=("Stroma",), target_label="Tumor"),
            key=lambda item: item.contact_pixels,
        )
        valid = json.loads(
            json.dumps(
                _manual_plan(
                    case=case,
                    interface=interface,
                    source_label="Stroma",
                    target_label="Tumor",
                    area=0.04,
                    supporting_rules=_bundle_ids(bundle),
                    band_max=64.0,
                ).to_metadata()
            )
        )
        invalid = copy.deepcopy(valid)
        invalid["area_budget"]["target_fraction"] = 0.03
        invalid["area_budget"]["min_fraction"] = 0.03
        invalid["area_budget"]["max_fraction"] = 0.03
        client = _FixturePlannerClient([invalid, valid])
        planner = OpenAIMultimodalPlanner(client=client, max_schema_attempts=2)

        with self.assertRaisesRegex(
            RefineContractError, "legacy online Planner/Critic is disabled"
        ):
            planner.create_plan(
                case=case,
                scene=scene,
                bundle=bundle,
                image_paths=(),
            )
        self.assertEqual(client.prompts, [])

    def test_escalation_budget_allows_at_most_one_upgrade_per_case(self):
        budget = EscalationBudget(max_fraction=1.0)
        budget.register_case()
        self.assertTrue(budget.consume(case_id="152"))
        self.assertFalse(budget.consume(case_id="152"))
        self.assertEqual(budget.cases_escalated, 1)


class _FixturePlannerClient:
    model = "gpt-5.6-terra"

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def call(self, *, user_prompt, **kwargs):
        del kwargs
        self.prompts.append(user_prompt)
        index = len(self.prompts) - 1
        return self.responses[index], {
            "model": self.model,
            "reasoning_effort": "medium",
            "input_tokens": 10 * (index + 1),
            "output_tokens": 3 + index,
            "input_tokens_details": {"cached_tokens": index},
        }


def _manual_plan(
    *,
    case: CaseContext,
    interface,
    source_label: str,
    target_label: str,
    area: float,
    supporting_rules: tuple[str, ...],
    band_max: float,
) -> EditPlan:
    return EditPlan(
        schema_version="mask-edit-refine-plan-v2",
        case_id=case.case_id,
        normalized_intent=case.instruction,
        primitive_id=case.primitive_id,
        source_labels=(source_label,),
        target_label=target_label,
        area_budget=AreaBudget(area, area, area),
        candidate_interfaces=(
            PlannedInterface(
                interface_id=interface.interface_id,
                source_component_id=interface.source_component_id,
                target_component_id=interface.target_component_id,
                anchor_segment="fixture",
                allowed_edit_band_px=(0.0, band_max),
                execution_contract=InterfaceExecutionContract(
                    anchor_segment_ids=interface.anchor_segment_ids,
                    area_allocation_fraction=1.0,
                    depth_profile=DepthProfile(
                        mode="tapered_lobe",
                        peak_depth_px=band_max,
                        edge_depth_px=min(2.0, band_max),
                        taper_fraction=0.20,
                        lobe_count=1,
                        noise_amplitude_px=min(4.0, band_max),
                        noise_correlation_px=16.0,
                    ),
                    min_anchor_coverage_fraction=0.25,
                    max_off_anchor_contact_fraction=0.10,
                    allocation_tolerance_fraction=0.02,
                ),
                prohibited_region_ids=(),
                supporting_rule_ids=supporting_rules,
                expected_morphology="broad interface change",
                confidence=0.9,
            ),
        ),
        tool_program=ToolProgram(
            allowed_tools=("interface_sdf",),
            parameter_ranges={
                "max_changed_components": 2,
                "min_component_area_px": 4,
                "max_depth_span_ratio": 1.25,
                "max_bbox_fill_fraction": 0.999,
                "max_boundary_compactness": 100.0,
            },
            candidate_count=1,
        ),
        hard_invariants=("label_transition", "depth_span_ratio"),
        uncertainties=(),
        planner_confidence=0.9,
    )


def _bundle_ids(bundle) -> tuple[str, ...]:
    return tuple(rule.rule_id for rule in bundle.active_rules) + tuple(
        item.constraint_id for item in bundle.active_mask_constraints
    )


if __name__ == "__main__":
    unittest.main()

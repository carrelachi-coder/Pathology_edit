"""Regression tests for the independent Architecture-B mask editor."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy import ndimage

from phase3_mask_edit_refine.agents import (
    HeuristicInterfacePlanner,
    OpenAIMultimodalPlanner,
    validate_edit_plan,
)
from phase3_mask_edit_refine.candidates import (
    compile_depth_profile_map,
    generate_candidates,
)
from phase3_mask_edit_refine.execution import (
    TopologySafeAreaUnderfillError,
    _prepare_compiler_work,
    _rebalance_fragmentation_residual_islands,
    _residual_fragmentation_priority,
    _whole_mask_topology_audit,
    compile_edit_plan,
)
from phase3_mask_edit_refine.gates import (
    GateContext,
    GateRegistry,
    _check_boundary_naturalness,
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
    ToolProgram,
)
from phase3_mask_edit_refine.scene import build_scene_analysis
from phase3_mask_edit_refine.skills import SkillRepository
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

    def test_breast_pathology_can_compose_with_orca_annotation(self):
        bundle = self.repository.compose(
            pathology_domain_id="breast-invasive-carcinoma-v1",
            annotation_profile_id="orca-semantic-v1",
            primitive_id="tumor-burden-increase-v1",
            production=False,
            available_checker_ids=self.gates.available_checker_ids,
        )
        self.assertEqual(bundle.edit_contract.source_label_options, ("Other tissue",))
        self.assertEqual(bundle.edit_contract.target_label, "Tumor")
        self.assertIn("mixed non-carcinoma", " ".join(bundle.warnings).lower())

    def test_orca_rejects_stroma_target(self):
        with self.assertRaisesRegex(RefineContractError, "cannot represent target label"):
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
        self.scene = build_scene_analysis(self.mask, schema=self.schema, pixel_size_um=0.465)

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
            self.scene.interfaces_for(
                source_labels=("Stroma",), target_label="Tumor"
            ),
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
                tool_program=replace(
                    base_plan.tool_program, parameter_ranges=ranges
                ),
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
            mechanism_specific.metrics[
                "maximum_allowed_component_compactness"
            ],
            55.0,
        )

        interface_front = run_with(4.0, geometry_mode="interface_front")
        self.assertTrue(interface_front.passed, interface_front.metrics)
        self.assertFalse(
            interface_front.metrics["change_band_compactness_applicable"]
        )
        self.assertTrue(
            interface_front.metrics["boundary_attached_geometry"]
        )

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
        depth_check = next(item for item in report.checks if item.check_id == "depth_span_ratio")
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
        transition = next(item for item in report.checks if item.check_id == "label_transition")
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
            item for item in report.checks
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
            item for item in report.checks
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
            item for item in report.checks if item.check_id == "parallel_boundary_artifact"
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
            item for item in report.checks
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
            compiled.candidate_interfaces[0].execution_contract.area_allocation_fraction,
            1.0,
        )
        self.assertLessEqual(
            compiled.candidate_interfaces[0].execution_contract.depth_profile.peak_depth_px,
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
            item for item in report.checks
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
        self.assertEqual(audit["resolved_pixels"], compiled.resolved_area.resolved_pixels)
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
        # The cap sits one discrete pixel beyond the continuous band. It is
        # still safe to fill because the selected corridor fully encloses it.
        legal_source[50:59, 112:121] = False
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
        legal_source[50:59, 112:121] = False
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
        self.assertEqual(np.count_nonzero(cleaned[0]), np.count_nonzero(change))

    def test_fragmentation_priority_builds_three_balanced_traversing_foci(self):
        shape = (96, 120)
        source = np.zeros(shape, dtype=bool)
        source[12:84, 10:110] = True
        default = ndimage.distance_transform_edt(source)

        priority = _residual_fragmentation_priority(
            source_component=source,
            legal_envelope=source,
            default_priority=default,
            minimum_residual_components=3,
            maximum_residual_components=6,
            minimum_residual_component_area_px=96,
            minimum_residual_spacing_px=8,
            minimum_residual_component_fraction=0.08,
            maximum_dominant_residual_component_fraction=0.75,
        )

        corridor = source & (priority < 0.5)
        labels, count = ndimage.label(
            source & ~corridor, structure=np.ones((3, 3), dtype=bool)
        )
        sizes = np.bincount(labels.ravel())[1:]
        fractions = sizes / sizes.sum()
        self.assertEqual(count, 3)
        self.assertGreaterEqual(float(fractions.min()), 0.08)
        self.assertLessEqual(float(fractions.max()), 0.75)
        self.assertGreaterEqual(
            int(np.count_nonzero(corridor)), int(source.sum() * 0.12)
        )

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
        anchors = tuple(scene.anchor_masks[item] for item in interface.anchor_segment_ids)
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
        anchors = tuple(scene.anchor_masks[item] for item in interface.anchor_segment_ids)
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
        multi_union = np.logical_or.reduce(
            [item.legal_source for item in multi_works]
        )
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
        for target_component_id in sorted({item.target_component_id for item in interfaces}):
            selected.append(
                max(
                    [item for item in interfaces if item.target_component_id == target_component_id],
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
                "source_component_ids": sorted({item.source_component_id for item in selected}),
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
            item for item in selected_report.checks if item.check_id == "edited_label_topology"
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
            item for item in unselected_report.checks if item.check_id == "edited_label_topology"
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
                critic=_PassingCritic(),
                config=WorkflowConfig(production=False, critic_min_score_margin=0.0),
            )
            result = workflow.run(case, output_root=root / "artifacts")
            self.assertEqual(result.status, "selected_research", result.abstain_reasons)
            self.assertIsNotNone(result.target_mask)
            self.assertTrue(all(report.passed for report in result.gate_reports if report.candidate_id == result.selected_candidate_id))
            self.assertTrue(Path(result.artifact_paths["selection"]).is_file())
            self.assertTrue(Path(result.artifact_paths["active_skills"]).is_file())


class AgentContractTests(unittest.TestCase):
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

    def test_planner_retries_once_and_audits_rejected_contract_response(self):
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

        plan, usage = planner.create_plan(
            case=case,
            scene=scene,
            bundle=bundle,
            image_paths=(),
        )

        self.assertEqual(plan.area_budget, case.area_budget)
        self.assertEqual(usage["schema_attempt_count"], 2)
        self.assertEqual(usage["input_tokens"], 30)
        self.assertEqual(usage["output_tokens"], 7)
        self.assertEqual(
            usage["schema_attempts"][0]["status"], "rejected_by_contract"
        )
        self.assertIn("previous response was rejected", client.prompts[1])

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

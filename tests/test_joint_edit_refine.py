"""Contracts and regression tests for the independent joint edit pipeline."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from phase3_joint_edit_refine.critic import DeterministicJointResearchCritic
from phase3_joint_edit_refine.agents import JOINT_PLAN_JSON_SCHEMA
from phase3_joint_edit_refine.budget import JointFeasibilitySolver
from phase3_joint_edit_refine.cell_layouts import build_reference_shape_library
from phase3_joint_edit_refine.gates import JointGateRegistry
from phase3_joint_edit_refine.generator_adapter import route_joint_handoff
from phase3_joint_edit_refine.ledger import analyze_joint_change
from phase3_joint_edit_refine.models import (
    CellCountExtentBudget,
    JointAreaBudget,
    JointCaseContext,
    JointCriticRanking,
    JointCriticResult,
)
from phase3_joint_edit_refine.mature_probnet_adapter import (
    MatureProbNetCellExecutor,
    MatureProbNetConfig,
)
from phase3_joint_edit_refine.nuclei import iter_instances
from phase3_joint_edit_refine.planner import HeuristicJointPlanner
from phase3_joint_edit_refine.profile_statistics import (
    build_annotation_profile_statistics,
)
from phase3_joint_edit_refine.scene import build_joint_scene_analysis
from phase3_joint_edit_refine.skills.repository import JointSkillRepository
from phase3_joint_edit_refine.workflow import JointPathologyEditWorkflow
from phase3_mask_edit_refine.agents import HeuristicInterfacePlanner
from phase3_mask_edit_refine.gates import GateRegistry
from phase3_mask_edit_refine.models import GateReport
from phase3_mask_edit.core.labels import MaskProfileSchema


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class JointSkillTests(unittest.TestCase):
    def test_inventory_has_six_domains_and_four_independent_axes(self):
        repository = JointSkillRepository()
        self.assertEqual(len(repository.mechanisms), 14)
        self.assertEqual(len(repository.primitives), 8)
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
        self.assertGreaterEqual(revised.tissue_target_pixels, revised.tissue_floor_pixels)
        self.assertEqual(
            revised.tissue_target_pixels + revised.reserved_cell_only_pixels,
            revised.joint_target_pixels,
        )

    def test_capacity_adaptive_budget_can_compile_below_standard_floor(self):
        repository = JointSkillRepository()
        case = _case_stub(
            budget=JointAreaBudget(
                capacity_floor_policy="lower_to_proven_max_safe"
            )
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
        revised = solver.reserve_complete_instances(initial, reserve_pixels=20000)
        self.assertLess(revised.tissue_target_pixels, revised.tissue_floor_pixels)
        self.assertEqual(revised.tissue_execution_floor_pixels, 0)

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
            primitive="neoplastic-cell-infiltration-increase-v1",
            cell_budget=CellCountExtentBudget(3, 2, 4, 48, 4, 32),
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
            )
        )
        command = executor.build_command(
            seed=17,
            target_tissue_path=Path("target.png"),
            source_tissue_path=Path("source.png"),
            source_nuclei_path=Path("nuclei.png"),
            placement_region_path=Path("P.png"),
            erasure_region_path=Path("E.png"),
            output_path=Path("out.png"),
            prohibited_tissue_ids=(0, 9),
        )
        self.assertIn("inpaint_cells.generate", command)
        self.assertIn("--no-widen-edit-region", command)
        self.assertIn("--require-sampling-audit", command)
        self.assertIn("--require-exact-target-count", command)
        self.assertIn("--reference-nuclei-shapes", command)
        self.assertEqual(command[-3:], ["--skip-tissue-ids", "0", "9"])

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
        touching = (
            ((rows - 20) ** 2 + (cols - 15) ** 2 <= 6**2)
            | ((rows - 20) ** 2 + (cols - 25) ** 2 <= 6**2)
        )
        nuclei[touching] = 1
        instances = tuple(iter_instances(nuclei))
        self.assertEqual(len(instances), 2)
        self.assertEqual(sum(int(mask.sum()) for _, _, mask in instances), int(touching.sum()))


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


class JointWorkflowTests(unittest.TestCase):
    def test_offline_joint_workflow_emits_review_artifacts_and_never_calls_api(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = _write_synthetic_case(root)
            workflow = JointPathologyEditWorkflow(
                tissue_planner=HeuristicInterfacePlanner(),
                joint_planner=HeuristicJointPlanner(),
                critic=DeterministicJointResearchCritic(),
            )
            result = workflow.run(case, output_root=root / "review")
            self.assertIn(result.status, {"review_required", "abstained"})
            self.assertTrue(Path(result.artifact_paths["result.json"]).is_file())
            self.assertTrue(
                (root / "review" / case.case_id / "joint_nuclei_preflight.json").is_file()
            )
            if result.status == "review_required":
                self.assertTrue(Path(result.artifact_paths["joint_condition_review"]).is_file())
                self.assertIsNone(result.condition)

    def test_approved_joint_candidate_emits_frozen_generator_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = _write_synthetic_case(root)
            workflow = JointPathologyEditWorkflow(
                tissue_planner=HeuristicInterfacePlanner(),
                joint_planner=HeuristicJointPlanner(),
                critic=_ApprovingJointCritic(),
                tissue_gates=_PassingTissueGateFixture(),
            )
            result = workflow.run(case, output_root=root / "approved")
            self.assertEqual(result.status, "selected_research", result.abstain_reasons)
            self.assertIsNotNone(result.condition)
            manifest = Path(result.artifact_paths["handoff_manifest"])
            self.assertTrue(manifest.is_file())
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertIn("joint_change", payload["paths"])
            self.assertIn("generation_support", payload["paths"])
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
                primitive_id="neoplastic-cell-infiltration-increase-v1",
                joint_area_budget=None,
                cell_count_extent_budget=CellCountExtentBudget(3, 3, 3, 48, 4, 32),
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
            self.assertIsNotNone(result.condition)
            self.assertEqual(result.condition.ledger.tissue_pixels, 0)
            self.assertGreater(result.condition.ledger.cell_pixels, 0)
            np.testing.assert_array_equal(
                result.condition.target_tissue_mask,
                np.load(source.source_tissue_mask_uri),
            )


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
        provenance={"source_image_sha256": "x", "source_tissue_mask_sha256": "y", "source_nuclei_mask_sha256": "z"},
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
                "contour": [[x - 1, y - 1], [x + 1, y - 1], [x + 1, y + 1], [x - 1, y + 1]],
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
        "auxiliary_structure_sha256": {
            "gland_or_lumen_support": _sha(auxiliary_path)
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
        auxiliary_structure_uris={
            "gland_or_lumen_support": str(auxiliary_path)
        },
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


if __name__ == "__main__":
    unittest.main()

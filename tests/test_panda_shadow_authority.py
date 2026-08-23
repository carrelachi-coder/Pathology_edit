from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from phase3_joint_edit_refine.candidate_feasibility import (
    _panda_architecture_portfolio_mode,
)
from phase3_joint_edit_refine.g2_v2_shadow import _materialize_joint_context
from phase3_joint_edit_refine.mature_probnet_adapter import _tree_sha256
from phase3_joint_edit_refine.portfolio_authority import (
    canonical_metadata_sha256,
)
from phase3_joint_edit_refine.semantic_parser import RuleBasedSemanticParser
from phase3_joint_edit_refine.skills.repository import JointSkillRepository
from phase3_mask_edit_refine.evidence import sha256_file
from scripts.materialize_panda_shadow_authority import (
    RUNTIME_CODE_RELATIVE_PATHS,
    _directory_sha256 as _authority_directory_sha256,
    _fixed_roi,
    _joint_area_budget,
    _parse_evaluation_count_overrides,
    _semantic_intent,
    _validate_native_instances,
)
from scripts.prepare_panda_primitive_shadow_selection import (
    EVALUATIONS,
    _coarse_eligible,
    _diverse_top,
    _fill_distinct_overlap_minimized,
    _minimum_feasible_max_cases_per_slide,
    _parse_evaluation_count_overrides as _parse_selection_count_overrides,
)
from scripts.run_panda_primitive_shadow_replay import (
    _compiled_review_candidate,
    _directory_sha256 as _replay_directory_sha256,
    _frozen_ranker_binding_passed,
    _load_qualification_records,
    _qualification_record_passed,
    _select_diverse_passes,
)


class PandaShadowAuthorityTests(unittest.TestCase):
    def test_replay_loads_materializer_json_qualification_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qualification.json"
            path.write_text(
                json.dumps({"cases": [{"case_id": "case-1"}]}),
                encoding="utf-8",
            )
            self.assertEqual(
                _load_qualification_records(path),
                [{"case_id": "case-1"}],
            )

    def test_replay_accepts_current_and_legacy_qualification_records(self):
        self.assertTrue(
            _qualification_record_passed(
                {"execution_allowed": True, "decision_status": "eligible"}
            )
        )
        self.assertTrue(
            _qualification_record_passed(
                {"status": "executable_preflight_passed"}
            )
        )
        self.assertFalse(
            _qualification_record_passed(
                {"execution_allowed": False, "decision_status": "eligible"}
            )
        )

    def test_panda_component_architecture_uses_bounded_portfolios(self):
        for primitive_id in (
            "invasive-tumor-footprint-decrease-v1",
            "residual-tumor-fragmentation-v1",
        ):
            self.assertEqual(
                _panda_architecture_portfolio_mode(
                    "panda-gleason-v1", primitive_id
                ),
                "single_global",
            )
        for primitive_id in (
            "tumor-burden-increase-v1",
            "cohesive-boundary-expansion-v1",
        ):
            self.assertEqual(
                _panda_architecture_portfolio_mode(
                    "panda-gleason-v1", primitive_id
                ),
                "global_plus_ranked_front",
            )
        self.assertEqual(
            _panda_architecture_portfolio_mode(
                "glas-gland-v1", "tumor-burden-increase-v1"
            ),
            "standard",
        )

    def test_shadow_evaluations_cover_every_supported_panda_pair(self):
        matrix_path = (
            Path(__file__).resolve().parents[1]
            / "phase3_joint_edit_refine"
            / "resources"
            / "non_breast_organ_annotation_capability_matrix_v1.json"
        )
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        panda = next(
            item
            for item in matrix["profiles"]
            if item["annotation_profile_id"] == "panda-gleason-v1"
        )
        supported = {
            (item["mechanism_id"], item["primitive_id"])
            for item in panda["capabilities"]
            if item["status"] == "conditionally_supported"
        }
        evaluated = {
            (item.mechanism_id, item.primitive_id) for item in EVALUATIONS
        }
        self.assertEqual(evaluated, supported)

    def test_runtime_inventory_binds_joint_change_ledger(self):
        self.assertIn(
            "phase3_joint_edit_refine/ledger.py",
            RUNTIME_CODE_RELATIVE_PATHS,
        )

    def test_runtime_directory_digest_matches_mature_executor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.bin").write_bytes(b"a")
            nested = root / "nested"
            nested.mkdir()
            (nested / "b.bin").write_bytes(b"b")
            expected = _tree_sha256(root)
            self.assertEqual(_authority_directory_sha256(root), expected)
            self.assertEqual(_replay_directory_sha256(root), expected)

    def test_final_diversity_selection_backtracks_across_slides(self):
        records = []
        for rank, slide_id in enumerate(("a", "a", "b", "b", "c", "d"), 1):
            records.append(
                {
                    "case_id": f"case-{rank}",
                    "candidate_pool_rank": rank,
                    "source_slide_id": slide_id,
                    "source_sample_id": f"sample-{rank}",
                    "full_gate_passed": True,
                }
            )
        selected = _select_diverse_passes(
            records,
            policy={
                "final_case_count": 5,
                "minimum_distinct_source_slides": 4,
                "maximum_cases_per_source_slide": 2,
            },
        )
        self.assertEqual(
            [item["candidate_pool_rank"] for item in selected],
            [1, 2, 3, 5, 6],
        )

    def test_offline_visual_review_keeps_hard_gate_passing_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.json"
            gates = root / "gates.json"
            critic = root / "critic.json"
            candidates.write_text(
                json.dumps([{"candidate_id": "candidate-1"}]),
                encoding="utf-8",
            )
            gates.write_text(
                json.dumps(
                    [
                        {
                            "candidate_id": "candidate-1",
                            "passed": True,
                            "checks": [
                                {"severity": "hard", "passed": True}
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            critic.write_text(
                json.dumps(
                    {
                        "rankings": [
                            {"candidate_id": "candidate-1", "veto_reasons": []}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                _compiled_review_candidate(
                    artifact_paths={
                        "candidates.json": str(candidates),
                        "joint_gate_reports.json": str(gates),
                        "joint_critic.json": str(critic),
                    },
                    abstain_reasons=[
                        "independent_mask_condition_critic_approval_required"
                    ],
                ),
                "candidate-1",
            )

    def test_diverse_top_maximizes_nonoverlap_count_before_score(self):
        rows = [
            {
                "slide_id": "slide-a",
                "sample_id": "slide-a_py0_px256",
                "filename": "slide-a_py0_px256.png",
                "selection_score": 10.0,
            },
            {
                "slide_id": "slide-a",
                "sample_id": "slide-a_py0_px0",
                "filename": "slide-a_py0_px0.png",
                "selection_score": 9.0,
            },
            {
                "slide_id": "slide-a",
                "sample_id": "slide-a_py0_px512",
                "filename": "slide-a_py0_px512.png",
                "selection_score": 8.0,
            },
        ]
        selected = _diverse_top(rows, maximum=3)
        self.assertEqual(
            {item["sample_id"] for item in selected},
            {"slide-a_py0_px0", "slide-a_py0_px512"},
        )

    def test_targeted_overlap_fill_preserves_distinct_patch_identity(self):
        rows = [
            {
                "slide_id": "slide-a",
                "sample_id": f"slide-a_py0_px{x}",
                "filename": f"slide-a_py0_px{x}.png",
                "selection_score": score,
            }
            for x, score in ((0, 9.0), (256, 10.0), (512, 8.0))
        ]
        strict = _diverse_top(rows, maximum=3)
        selected = _fill_distinct_overlap_minimized(
            strict, rows, maximum=3
        )
        self.assertEqual(len(strict), 2)
        self.assertEqual(len(selected), 3)
        self.assertEqual(
            len({item["filename"] for item in selected}),
            len(selected),
        )

    def test_slide_case_cap_is_feasible_for_uneven_candidate_counts(self):
        candidates = [
            {"slide_id": slide_id}
            for slide_id in ("a", "a", "a", "a", "b", "c")
        ]
        self.assertEqual(
            _minimum_feasible_max_cases_per_slide(
                candidates, final_case_count=5
            ),
            3,
        )

    def test_prostate_stromal_depletion_can_anchor_to_tumor(self):
        mechanism = JointSkillRepository().mechanisms[
            "prostate-local-population-modulation"
        ]
        self.assertIn(
            "Tumor",
            mechanism.cell_program.cellularity_depletion.allowed_neighbor_labels,
        )

    def test_targeted_native_candidate_count_overrides_are_explicit(self):
        self.assertEqual(
            _parse_evaluation_count_overrides("1:20,15:15"),
            {1: 20, 15: 15},
        )
        with self.assertRaises(ValueError):
            _parse_evaluation_count_overrides("1:4")
        self.assertEqual(
            _parse_selection_count_overrides("15:48"), {15: 48}
        )

    def test_panda_fragmentation_screen_uses_profile_owned_three_percent_floor(self):
        metrics = {
            "fine_pixel_counts": {"10": 70000},
            "stroma_pixels": 60000,
            "p5_stroma_contact_pixels": 800,
            "tumor_largest_component_pixels": 180000,
        }
        self.assertTrue(_coarse_eligible("fragmentation", metrics))
        metrics["tumor_largest_component_pixels"] = 19000
        self.assertFalse(_coarse_eligible("fragmentation", metrics))

    def test_full_replay_requires_exact_frozen_ranker_assets(self):
        evidence = [
            {
                "ranker": "frozen_probnet_context_stabilized_spatial_sampler",
                "ranker_provenance": {
                    "checkpoint_sha256": "checkpoint",
                    "instance_library_sha256": "library",
                },
            }
        ]
        self.assertTrue(
            _frozen_ranker_binding_passed(
                evidence,
                checkpoint_sha256="checkpoint",
                instance_library_sha256="library",
            )
        )
        self.assertFalse(
            _frozen_ranker_binding_passed(
                evidence,
                checkpoint_sha256="drifted",
                instance_library_sha256="library",
            )
        )

    def test_source_template_anchor_ranker_binds_checkpoint_only(self):
        evidence = [
            {
                "ranker": "frozen_probnet_spatial_ranker",
                "ranker_provenance": {
                    "checkpoint_sha256": "checkpoint",
                    "role": "legal_template_anchor_ranking_only",
                },
            }
        ]
        self.assertTrue(
            _frozen_ranker_binding_passed(
                evidence,
                checkpoint_sha256="checkpoint",
                instance_library_sha256="unused-for-source-template",
            )
        )

    def test_frozen_joint_area_budget_is_primitive_specific(self):
        self.assertEqual(
            _joint_area_budget("infiltrative-nest-cord-extension-v1")
            ["min_fraction"],
            0.010,
        )
        self.assertEqual(
            _joint_area_budget("infiltrative-nest-cord-extension-v1")
            ["max_fraction"],
            0.028,
        )
        self.assertEqual(
            _joint_area_budget("infiltrative-nest-cord-extension-v1")
            ["target_fraction"],
            0.018,
        )
        cord_bundle = JointSkillRepository().mechanisms[
            "prostate-pattern-5-infiltrative-front"
        ]
        self.assertEqual(
            cord_bundle.tissue_program.front.maximum_band_px,
            144,
        )
        self.assertEqual(
            _joint_area_budget("local-invasive-clearance-v1")["max_fraction"],
            0.14,
        )
        self.assertEqual(
            _joint_area_budget("local-invasive-clearance-v1")["target_fraction"],
            0.06,
        )
        self.assertEqual(
            _joint_area_budget("residual-tumor-fragmentation-v1")
            ["target_fraction"],
            0.045,
        )
        self.assertEqual(
            _joint_area_budget("residual-tumor-fragmentation-v1")
            ["min_fraction"],
            0.025,
        )
        self.assertEqual(
            _joint_area_budget("cohesive-boundary-expansion-v1")
            ["target_fraction"],
            0.03,
        )

    def test_frozen_semantic_intent_binds_reviewed_primitive(self):
        metadata, digest = _semantic_intent(
            "Reduce tumor burden.",
            "invasive-tumor-footprint-decrease-v1",
        )
        self.assertEqual(
            metadata["selected_primitive_id"],
            "invasive-tumor-footprint-decrease-v1",
        )
        self.assertEqual(digest, canonical_metadata_sha256(metadata))

    def test_fixed_roi_does_not_depend_on_pathology_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _fixed_roi(
                shape=(512, 512),
                path=root / "first.png",
                source_tissue_sha256="a" * 64,
                user_authority_sha256="b" * 64,
            )
            second = _fixed_roi(
                shape=(512, 512),
                path=root / "second.png",
                source_tissue_sha256="c" * 64,
                user_authority_sha256="b" * 64,
            )
            first_array = np.asarray(Image.open(first[0]["local_clearance_roi"]))
            second_array = np.asarray(Image.open(second[0]["local_clearance_roi"]))
            np.testing.assert_array_equal(first_array, second_array)
            self.assertEqual(int(np.count_nonzero(first_array)), 384 * 384)
            provenance = first[2]["local_clearance_roi"]
            self.assertEqual(
                provenance["authority_type"], "digest_bound_user_local_roi"
            )
            self.assertFalse(provenance["he_or_llm_used"])

    def test_native_json_must_agree_with_frozen_semantic_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            semantic = np.zeros((64, 64), dtype=np.uint8)
            cells = []
            for index in range(16):
                row = 2 + (index // 4) * 14
                col = 2 + (index % 4) * 14
                semantic[row : row + 5, col : col + 5] = 101
                cells.append(
                    {
                        "type": 1,
                        "contour": [
                            [col, row],
                            [col + 4, row],
                            [col + 4, row + 4],
                            [col, row + 4],
                        ],
                    }
                )
            semantic_path = root / "semantic.png"
            cells_path = root / "cells.json"
            Image.fromarray(semantic, mode="L").save(semantic_path)
            cells_path.write_text(
                json.dumps({"cells": cells}), encoding="utf-8"
            )
            result = _validate_native_instances(
                cells_json=cells_path,
                semantic_path=semantic_path,
            )
            self.assertEqual(result["status"], "verified")
            self.assertEqual(result["metrics"]["native_instance_count"], 16)
            self.assertEqual(result["metrics"]["foreground_dice"], 1.0)
            self.assertFalse(result["llm_api_used"])

    def test_g2_context_preserves_digest_bound_external_roi(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = {}
            for name in ("image.png", "tissue.png", "nuclei.png", "roi.png"):
                path = root / name
                path.write_bytes(name.encode("ascii"))
                assets[name] = path
            semantic = RuleBasedSemanticParser().parse(
                "Clear tumor in this local ROI."
            ).to_metadata()
            row = {
                "case_id": "panda-roi-test",
                "source_index": 0,
                "dataset": "PANDA",
                "source_image_uri": str(assets["image.png"]),
                "source_tissue_mask_uri": str(assets["tissue.png"]),
                "source_nuclei_mask_uri": str(assets["nuclei.png"]),
                "source_digests": {
                    "image_sha256": sha256_file(assets["image.png"]),
                    "tissue_mask_sha256": sha256_file(assets["tissue.png"]),
                    "nuclei_mask_sha256": sha256_file(assets["nuclei.png"]),
                },
                "source_manifest_metadata": {"provider": "PANDA"},
                "pathology_domain_id": "prostate-adenocarcinoma-v1",
                "annotation_profile_id": "panda-gleason-v1",
                "cell_observation_profile_id": "cellvit-five-class-v1",
                "cell_population_profile_id": (
                    "prostate-cellvit-source-first-v1"
                ),
                "mechanism_id": "prostate-local-tumor-clearance",
                "primitive_id": "local-invasive-clearance-v1",
                "instruction": "Clear tumor in this local ROI.",
                "prebound_semantic_intent": semantic,
                "prebound_semantic_intent_sha256": (
                    canonical_metadata_sha256(semantic)
                ),
                "joint_area_budget": 0.19,
                "seed": 42,
                "pixel_size_um": 0.25,
                "decision_reason_code": "test",
                "decision_status": "eligible",
                "review_basis": {},
                "visual_observations": {},
                "auxiliary_structure_uris": {
                    "local_clearance_roi": str(assets["roi.png"])
                },
                "auxiliary_structure_sha256": {
                    "local_clearance_roi": sha256_file(assets["roi.png"])
                },
                "auxiliary_structure_provenance": {
                    "local_clearance_roi": {
                        "producer_id": "test-user-roi",
                        "producer_version": "v1",
                        "source_tissue_mask_sha256": sha256_file(
                            assets["tissue.png"]
                        ),
                        "output_sha256": sha256_file(assets["roi.png"]),
                    }
                },
            }
            context = _materialize_joint_context(
                row, manifest_sha256="d" * 64
            )
            self.assertEqual(
                context["auxiliary_structure_uris"]["local_clearance_roi"],
                str(assets["roi.png"]),
            )
            self.assertEqual(
                context["provenance"]["auxiliary_structure_sha256"]
                ["local_clearance_roi"],
                sha256_file(assets["roi.png"]),
            )


if __name__ == "__main__":
    unittest.main()

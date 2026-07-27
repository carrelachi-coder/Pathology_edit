from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import PngImagePlugin

import scripts.build_tcga_patch_selection as selection_builder
import scripts.eval_segmentator_human_gt as human_eval

from segmentator.patch_selection import (
    ORGAN_RULES,
    compute_image_quality,
    compute_mask_features,
    finalize_row_scores,
    organ_valid_confusion,
    organ_from_project,
    parse_tcga_patch_name,
    passes_organ_constraints,
    select_candidate_pool,
    select_case_disjoint_sets,
)


class OrganRuleTests(unittest.TestCase):
    def test_project_mapping_merges_tcga_projects_into_six_organs(self):
        expected = {
            "TCGA-BRCA": "breast",
            "TCGA-PRAD": "prostate",
            "TCGA-COAD": "colorectal",
            "TCGA-READ": "colorectal",
            "TCGA-LUAD": "lung",
            "TCGA-LUSC": "lung",
            "TCGA-SKCM": "skin",
            "TCGA-HNSC": "head_neck",
        }
        self.assertEqual({project: organ_from_project(project) for project in expected}, expected)

    def test_other_caps_and_valid_labels_are_organ_specific(self):
        self.assertEqual(ORGAN_RULES["breast"].other_cap, 0.05)
        self.assertEqual(ORGAN_RULES["lung"].other_cap, 0.10)
        self.assertEqual(ORGAN_RULES["head_neck"].other_cap, 0.15)
        self.assertNotIn(7, ORGAN_RULES["breast"].complexity_labels)
        self.assertNotIn(6, ORGAN_RULES["lung"].complexity_labels)
        self.assertTrue(ORGAN_RULES["head_neck"].head_neck_boundary_only)

    def test_tcga_patch_name_parser_extracts_case_slide_and_coordinates(self):
        parsed = parse_tcga_patch_name(
            "TCGA-BH-A18G-01Z-00-DX1.DB2B5819-CE83-4E07-BD03-2CD9CF2E246C_32096_31776.png"
        )
        self.assertEqual(parsed.case_id, "TCGA-BH-A18G")
        self.assertEqual(parsed.x, 32096)
        self.assertEqual(parsed.y, 31776)
        self.assertTrue(parsed.wsi.startswith("TCGA-BH-A18G-01Z-00-DX1"))


class MaskFeatureTests(unittest.TestCase):
    def test_other_never_increases_multitissue_complexity(self):
        base = np.ones((64, 64), dtype=np.uint8)
        base[:, 32:] = 2
        with_other = base.copy()
        with_other[24:40, 24:40] = 7

        base_features = compute_mask_features(base, "breast")
        other_features = compute_mask_features(with_other, "breast")

        self.assertLessEqual(other_features.positive_complexity, base_features.positive_complexity)
        self.assertGreater(other_features.other_fraction, base_features.other_fraction)

    def test_other_gap_does_not_bridge_two_effective_tissues_for_scoring(self):
        background_gap = np.ones((64, 64), dtype=np.uint8)
        background_gap[:, 24:40] = 0
        background_gap[:, 40:] = 2
        other_gap = background_gap.copy()
        other_gap[:, 24:40] = 7

        background_features = compute_mask_features(background_gap, "breast")
        other_features = compute_mask_features(other_gap, "breast")

        self.assertLessEqual(other_features.positive_complexity, background_features.positive_complexity)
        self.assertEqual(other_features.interface_density, 0.0)

    def test_head_neck_scores_tumor_boundary_but_penalizes_other_area(self):
        compact = np.ones((64, 64), dtype=np.uint8)
        compact[28:36, 16:48] = 7
        bulky = compact.copy()
        bulky[8:28, 8:56] = 7

        compact_features = compute_mask_features(compact, "head_neck")
        bulky_features = compute_mask_features(bulky, "head_neck")

        self.assertGreater(compact_features.interface_density, 0.0)
        self.assertGreater(bulky_features.other_fraction, compact_features.other_fraction)
        self.assertLess(bulky_features.pre_normalized_score, compact_features.pre_normalized_score)

    def test_tiny_components_do_not_count_as_valid_tissues(self):
        mask = np.ones((100, 100), dtype=np.uint8)
        mask[:, 50:] = 2
        mask[0:2, 0:2] = 3

        features = compute_mask_features(mask, "breast")

        self.assertEqual(features.valid_class_count, 2)

    def test_constraints_apply_other_cap_and_minimum_effective_classes(self):
        valid = np.ones((100, 100), dtype=np.uint8)
        valid[:, 50:] = 2
        too_much_other = valid.copy()
        too_much_other[:, :10] = 7

        self.assertTrue(passes_organ_constraints(compute_mask_features(valid, "prostate"), "prostate"))
        self.assertFalse(
            passes_organ_constraints(compute_mask_features(too_much_other, "prostate"), "prostate")
        )


class ImageQualityTests(unittest.TestCase):
    def test_selection_builder_raises_png_metadata_limits(self):
        self.assertGreaterEqual(PngImagePlugin.MAX_TEXT_CHUNK, 256 * 1024 * 1024)
        self.assertGreaterEqual(PngImagePlugin.MAX_TEXT_MEMORY, 1024 * 1024 * 1024)

    def test_sharp_texture_has_higher_focus_than_flat_image(self):
        flat = np.full((64, 64, 3), 128, dtype=np.uint8)
        checker = np.indices((64, 64)).sum(axis=0) % 2
        sharp = np.repeat((checker * 180 + 40)[..., None], 3, axis=2).astype(np.uint8)

        flat_metrics = compute_image_quality(flat)
        sharp_metrics = compute_image_quality(sharp)

        self.assertGreater(sharp_metrics.laplacian_variance, flat_metrics.laplacian_variance)
        self.assertGreater(sharp_metrics.tenengrad, flat_metrics.tenengrad)

    def test_predicted_tissue_mask_controls_tissue_and_white_fraction(self):
        image = np.full((10, 10, 3), 120, dtype=np.uint8)
        image[:2] = 255
        tissue = np.ones((10, 10), dtype=bool)

        metrics = compute_image_quality(image, tissue_mask=tissue)

        self.assertEqual(metrics.tissue_fraction, 1.0)
        self.assertAlmostEqual(metrics.near_white_tissue_fraction, 0.2)


class SelectionTests(unittest.TestCase):
    def test_exported_manifests_have_csv_and_json_representations(self):
        rows = [{"filename": "sample.png", "organ": "breast", "score": 0.75}]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            selection_builder._write_manifest(root / "manifest", rows)

            self.assertTrue((root / "manifest.csv").exists())
            self.assertEqual(json.loads((root / "manifest.json").read_text()), rows)

    def test_finalize_scores_applies_quality_and_other_constraints(self):
        rows = []
        for index in range(20):
            rows.append(
                {
                    "organ": "breast",
                    "tissue_fraction": 0.9,
                    "dynamic_range": 80.0,
                    "near_black_fraction": 0.0,
                    "near_white_tissue_fraction": 0.0,
                    "laplacian_variance": float(index + 1),
                    "tenengrad": float(index + 1),
                    "other_fraction": 0.01,
                    "valid_class_count": 3,
                    "interface_density": float(index),
                    "class_entropy": 0.5,
                    "shape_irregularity": 0.5,
                    "speckle_fraction": 0.0,
                }
            )
        rows[-1]["other_fraction"] = 0.20

        scored = finalize_row_scores(rows)

        self.assertFalse(scored[0]["quality_pass"])
        self.assertTrue(scored[10]["quality_pass"])
        self.assertFalse(scored[-1]["organ_constraints_pass"])
        self.assertLess(scored[5]["selection_score"], scored[10]["selection_score"])

    def test_head_neck_final_score_prefers_irregular_tumor_shape_over_plain_interface_length(self):
        common = {
            "organ": "head_neck",
            "tissue_fraction": 0.9,
            "dynamic_range": 80.0,
            "near_black_fraction": 0.0,
            "near_white_tissue_fraction": 0.0,
            "laplacian_variance": 10.0,
            "tenengrad": 10.0,
            "other_fraction": 0.05,
            "valid_class_count": 1,
            "class_entropy": 0.0,
            "speckle_fraction": 0.0,
        }
        irregular = dict(common, filename="irregular.png", interface_density=0.0, shape_irregularity=1.0)
        plain_interface = dict(common, filename="plain.png", interface_density=1.0, shape_irregularity=0.0)

        scored = {row["filename"]: row for row in finalize_row_scores([irregular, plain_interface])}

        self.assertGreater(scored["irregular.png"]["selection_score"], scored["plain.png"]["selection_score"])

    def test_complex_and_random_sets_are_case_disjoint_and_balanced(self):
        rows = []
        for organ in ORGAN_RULES:
            for case_idx in range(8):
                case_id = f"{organ}-{case_idx:02d}"
                for patch_idx in range(3):
                    rows.append(
                        {
                            "organ": organ,
                            "case_id": case_id,
                            "wsi": case_id,
                            "filename": f"{case_id}_{patch_idx * 2048}_0.png",
                            "x": patch_idx * 2048,
                            "y": 0,
                            "selection_score": float(100 - case_idx - patch_idx / 10),
                            "quality_pass": True,
                            "organ_constraints_pass": True,
                            "training_overlap": False,
                        }
                    )

        result = select_case_disjoint_sets(
            rows,
            complex_per_organ=4,
            random_per_organ=2,
            seed=42,
            case_caps={organ: 1 for organ in ORGAN_RULES},
            random_case_caps={organ: 1 for organ in ORGAN_RULES},
        )

        for organ in ORGAN_RULES:
            complex_rows = [r for r in result.complex_rows if r["organ"] == organ]
            random_rows = [r for r in result.random_rows if r["organ"] == organ]
            self.assertEqual(len(complex_rows), 4)
            self.assertEqual(len(random_rows), 2)
            self.assertTrue(
                {r["case_id"] for r in complex_rows}.isdisjoint({r["case_id"] for r in random_rows})
            )

        repeated = select_case_disjoint_sets(
            list(reversed(rows)),
            complex_per_organ=4,
            random_per_organ=2,
            seed=42,
            case_caps={organ: 1 for organ in ORGAN_RULES},
            random_case_caps={organ: 1 for organ in ORGAN_RULES},
        )
        self.assertEqual(
            [row["filename"] for row in result.random_rows],
            [row["filename"] for row in repeated.random_rows],
        )

    def test_candidate_pool_contains_required_rows_and_organ_floors(self):
        rows = []
        for organ_index, organ in enumerate(ORGAN_RULES):
            for index in range(10):
                rows.append(
                    {
                        "organ": organ,
                        "case_id": f"{organ}-{index}",
                        "filename": f"{organ}-{index}.png",
                        "selection_score": float(1000 - organ_index * 100 - index),
                        "quality_pass": True,
                        "organ_constraints_pass": True,
                        "training_overlap": False,
                    }
                )
        required = [row for row in rows if row["filename"] in {"skin-9.png", "head_neck-9.png"}]

        selected = select_candidate_pool(rows, target=30, organ_floor=3, case_cap=2, required_rows=required)

        self.assertEqual(len(selected), 30)
        self.assertTrue({row["filename"] for row in required}.issubset({row["filename"] for row in selected}))
        for organ in ORGAN_RULES:
            self.assertGreaterEqual(sum(row["organ"] == organ for row in selected), 3)

    def test_random_selection_expands_cases_when_initial_cases_have_low_capacity(self):
        rows = []
        for organ in ORGAN_RULES:
            for case_idx in range(8):
                rows.append(
                    {
                        "organ": organ,
                        "case_id": f"{organ}-{case_idx}",
                        "wsi": f"{organ}-{case_idx}",
                        "filename": f"{organ}-{case_idx}_0_0.png",
                        "x": 0,
                        "y": 0,
                        "selection_score": float(case_idx),
                        "quality_pass": True,
                        "organ_constraints_pass": True,
                        "training_overlap": False,
                    }
                )

        result = select_case_disjoint_sets(
            rows,
            complex_per_organ=2,
            random_per_organ=5,
            seed=42,
            case_caps={organ: 1 for organ in ORGAN_RULES},
            random_case_caps={organ: 3 for organ in ORGAN_RULES},
        )

        for organ in ORGAN_RULES:
            self.assertEqual(sum(row["organ"] == organ for row in result.random_rows), 5)
            self.assertEqual(result.deficits[organ]["random"], 0)

        repeated = select_case_disjoint_sets(
            list(reversed(rows)),
            complex_per_organ=2,
            random_per_organ=5,
            seed=42,
            case_caps={organ: 1 for organ in ORGAN_RULES},
            random_case_caps={organ: 3 for organ in ORGAN_RULES},
        )
        self.assertEqual(
            [row["filename"] for row in result.random_rows],
            [row["filename"] for row in repeated.random_rows],
        )


class EvaluationTests(unittest.TestCase):
    def test_evaluation_groups_include_stratum_by_organ_reports(self):
        records = [
            {"stratum": "complex", "organ": "breast"},
            {"stratum": "random", "organ": "breast"},
            {"stratum": "complex", "organ": "lung"},
        ]

        groups = human_eval._evaluation_groups(records)

        self.assertEqual(len(groups["stratum:complex/organ:breast"]), 1)
        self.assertEqual(len(groups["stratum:random/organ:breast"]), 1)
        self.assertEqual(len(groups["stratum:complex/organ:lung"]), 1)

    def test_case_bootstrap_reports_miou_and_mdice_confidence_intervals(self):
        matrix = np.zeros((8, 8), dtype=np.int64)
        matrix[1, 1] = 10
        records = [{"case_id": "case-1", "raw_confusion": matrix}]

        result = human_eval._bootstrap(records, "raw_confusion", iterations=5, seed=42)

        self.assertEqual(result["mIoU_mean"], 1.0)
        self.assertEqual(result["mDice_mean"], 1.0)
        self.assertEqual(result["mDice_ci_low"], 1.0)
        self.assertEqual(result["mDice_ci_high"], 1.0)

    def test_organ_valid_confusion_ignores_unsupported_gt_but_not_bad_predictions(self):
        gt = np.array([[1, 2, 3, 5]], dtype=np.uint8)
        pred = np.array([[1, 7, 1, 5]], dtype=np.uint8)

        matrix = organ_valid_confusion(pred, gt, "prostate")

        self.assertEqual(int(matrix.sum()), 3)
        self.assertEqual(int(matrix[1, 1]), 1)
        self.assertEqual(int(matrix[2, 7]), 1)
        self.assertEqual(int(matrix[5, 5]), 1)


if __name__ == "__main__":
    unittest.main()

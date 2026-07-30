import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from phase3_mask_edit.core.mask_io import save_id_mask
from phase3_mask_edit.parser.semantic_diff import DEFAULT_SEMANTIC_DIFF
from scripts.run_phase3_inpaint_pipeline import (
    _build_nucleus_aware_generation_region,
    _probnet_sampling_contract,
    _retain_complete_reference_cells,
    _select_generation_mode,
    main as run_phase3_inpaint_pipeline,
)


def _write_rgb(path: Path, value: tuple[int, int, int] = (180, 120, 160)) -> None:
    array = np.full((96, 96, 3), value, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _write_nuclei(path: Path) -> None:
    nuclei = np.zeros((96, 96), dtype=np.uint8)
    nuclei[20:24, 20:24] = 101
    nuclei[50:54, 50:54] = 102
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(nuclei).save(path)


def _source_mask() -> np.ndarray:
    mask = np.zeros((96, 96), dtype=np.int64)
    mask[16:72, 16:72] = 1
    mask[72:88, 16:72] = 2
    return mask


class Phase3InpaintPipelineTests(unittest.TestCase):
    def test_nucleus_boundary_context_uses_largest_deleted_diameter(self):
        change = np.zeros((20, 20), dtype=bool)
        change[8:12, 10:14] = True
        nuclei = np.zeros((20, 20), dtype=np.uint8)
        nuclei[7:11, 7:11] = 101
        nuclei[2:5, 2:5] = 102

        expanded, metadata = _build_nucleus_aware_generation_region(
            deletion_region=change,
            base_generation_region=change,
            reference_nuclei=nuclei,
            biological_support=np.ones_like(change),
        )

        expected_diameter = 2.0 * np.sqrt(16.0 / np.pi)
        self.assertAlmostEqual(
            metadata["largest_deleted_nucleus_diameter_px"],
            expected_diameter,
        )
        self.assertEqual(
            metadata["buffer_radius_px"],
            int(np.ceil(1.5 * expected_diameter)),
        )
        self.assertGreater(metadata["added_buffer_pixels"], 0)

    def test_buffer_only_nucleus_is_retained(self):
        deletion = np.zeros((32, 32), dtype=bool)
        deletion[14:18, 14:18] = True
        nuclei = np.zeros((32, 32), dtype=np.uint8)
        nuclei[14:18, 14:18] = 101
        nuclei[5:8, 14:17] = 102

        generation, _ = _build_nucleus_aware_generation_region(
            deletion_region=deletion,
            base_generation_region=deletion,
            reference_nuclei=nuclei,
            biological_support=np.ones_like(deletion),
        )
        self.assertTrue(np.any(generation[5:8, 14:17]))

        retained, stats = _retain_complete_reference_cells(
            nuclei,
            deletion,
            policy="delete",
        )
        self.assertFalse(np.any(retained[14:18, 14:18]))
        self.assertTrue(np.all(retained[5:8, 14:17] == 102))
        self.assertEqual(stats["deleted_components"], 1)
        self.assertEqual(stats["kept_components"], 1)

    def test_merged_component_does_not_define_generation_buffer_radius(self):
        deletion = np.zeros((96, 96), dtype=bool)
        nuclei = np.zeros((96, 96), dtype=np.uint8)
        for index, (row, col) in enumerate(
            [(4, 4), (4, 20), (20, 4), (20, 20), (36, 4)]
        ):
            nuclei[row:row + 4, col:col + 4] = 101
            if index == 0:
                deletion[row:row + 4, col:col + 4] = True
        nuclei[50:82, 50:82] = 101
        deletion[60:62, 60:62] = True

        _, metadata = _build_nucleus_aware_generation_region(
            deletion_region=deletion,
            base_generation_region=deletion,
            reference_nuclei=nuclei,
            biological_support=np.ones_like(deletion),
        )

        self.assertEqual(metadata["rejected_merged_diameter_components"], 1)
        self.assertEqual(metadata["largest_deleted_nucleus_area_px"], 16)

    def test_probnet_retention_deletes_crossing_instance_without_centroid_rule(self):
        source = np.zeros((20, 20), dtype=np.uint8)
        source[5:11, 4:13] = 101
        changed = np.zeros((20, 20), dtype=bool)
        changed[:, 11:] = True

        retained, stats = _retain_complete_reference_cells(
            source,
            changed,
            policy="delete",
        )

        self.assertEqual(stats["crossing_components"], 1)
        self.assertEqual(stats["deleted_components"], 1)
        self.assertFalse(np.any(retained))

    def test_probnet_contract_is_generic_and_auditable(self):
        contract = _probnet_sampling_contract(
            {
                "tissues": {
                    "2": {
                        "candidate_queue_policy": (
                            "probnet_log_odds_quality_diversity_prefix"
                        ),
                        "quota_coverage_spacing_scale": 0.75,
                        "quota_coverage_max_radius": 48.0,
                        "retry_tail_policy": "stable_descending_probnet_score",
                        "accepted_center_probability": {
                            "median": 0.81,
                        },
                    },
                    "7": {
                        "candidate_queue_policy": (
                            "probnet_log_odds_quality_diversity_prefix"
                        ),
                        "quota_coverage_spacing_scale": 0.75,
                        "quota_coverage_max_radius": 48.0,
                        "retry_tail_policy": "stable_descending_probnet_score",
                        "accepted_center_probability": {
                            "median": 0.74,
                        },
                    },
                },
                "nucleus_spacing_margin_px": 1,
                "type_quota_routing_policy": (
                    "changed_target_tissue_density_head_"
                    "unchanged_target_tissue_pre_edit_patch"
                ),
                "patch_adaptive_priors": {
                    "count_policy": (
                        "pre_edit_source_tissue_density_or_target_prior_"
                        "calibrated_by_pre_edit_source_times_post_edit_target_area"
                    ),
                    "generation_support": {
                        "source_nucleus_erasure_policy": (
                            "complete_component_on_any_deletion_region_intersection"
                        ),
                        "buffer_nucleus_policy": (
                            "retain_generation_buffer_only_nuclei_as_"
                            "placement_obstacles"
                        ),
                    },
                },
                "shape_sampling": {
                    "policy": (
                        "component_local_same_class_reference_then_"
                        "component_calibrated_library"
                    ),
                },
            }
        )

        self.assertEqual(
            contract["candidate_queue_policy"],
            "probnet_log_odds_quality_diversity_prefix",
        )
        self.assertFalse(contract["organ_specific_constraints"])
        self.assertEqual(contract["quota_coverage_spacing_scale"], 0.75)
        self.assertEqual(contract["quota_coverage_max_radius"], 48.0)
        self.assertEqual(
            contract["retry_tail_policy"],
            "stable_descending_probnet_score",
        )
        self.assertEqual(
            contract["accepted_center_probability_by_tissue"]["7"]["median"],
            0.74,
        )
        self.assertEqual(contract["nucleus_spacing_margin_px"], 1)
        self.assertEqual(
            contract["type_quota_routing_policy"],
            "changed_target_tissue_density_head_"
            "unchanged_target_tissue_pre_edit_patch",
        )

    def test_glas_boundary_change_rewrites_complete_gland_component(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            source = root / "source_mask.png"
            target = root / "target_mask.png"
            nuclei = root / "nuclei.png"
            output = root / "out"

            source_mask = np.full((96, 96), 2, dtype=np.uint8)
            source_mask[24:48, 24:48] = 11
            source_mask[64:80, 64:80] = 5
            target_mask = source_mask.copy()
            target_mask[20:52, 20:52] = 11
            source_nuclei = np.zeros((96, 96), dtype=np.uint8)
            source_nuclei[32:36, 32:36] = 101
            source_nuclei[68:72, 68:72] = 105

            _write_rgb(image)
            save_id_mask(source_mask, source)
            save_id_mask(target_mask, target)
            Image.fromarray(source_nuclei).save(nuclei)

            exit_code = run_phase3_inpaint_pipeline(
                [
                    "--mode",
                    "gen",
                    "--profile",
                    "GlaS",
                    "--reference-image",
                    str(image),
                    "--reference-tissue-mask",
                    str(source),
                    "--reference-nuclei-mask",
                    str(nuclei),
                    "--target-tissue-mask",
                    str(target),
                    "--output",
                    str(output),
                    "--generation-mode",
                    "dry-run",
                    "--cell-fill-mode",
                    "blank",
                ]
            )

            self.assertEqual(exit_code, 0)
            semantic = np.asarray(
                Image.open(output / "semantic_change_region.png").convert("L")
            ) > 0
            generation = np.asarray(
                Image.open(output / "change_region.png").convert("L")
            ) > 0
            retained = np.asarray(
                Image.open(output / "retained_nuclei_mask.png").convert("L")
            )
            self.assertLess(np.count_nonzero(semantic), np.count_nonzero(generation))
            self.assertTrue(np.all(generation[semantic]))
            self.assertTrue(np.all(generation[20:52, 20:52]))
            self.assertFalse(np.any(generation[64:80, 64:80]))
            self.assertFalse(np.any(retained[32:36, 32:36] == 101))
            self.assertTrue(np.any(retained[68:72, 68:72] == 105))

            summary = json.loads(
                (output / "pipeline_summary.json").read_text(encoding="utf-8")
            )
            self.assertTrue(summary["gland_structure_policy"]["applied"])
            self.assertFalse(
                summary["gland_structure_policy"]["context_bound"]["capped"]
            )
            self.assertEqual(
                summary["gland_structure_policy"][
                    "cell_deletion_region_policy"
                ],
                "whole_glas_connected_component",
            )

    def test_gen_dry_run_writes_generation_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            source = root / "source_mask.png"
            target = root / "target_mask.png"
            nuclei = root / "nuclei.png"
            output = root / "out"

            source_mask = _source_mask()
            target_mask = source_mask.copy()
            target_mask[34:46, 34:46] = 3
            _write_rgb(image)
            save_id_mask(source_mask, source)
            save_id_mask(target_mask, target)
            _write_nuclei(nuclei)

            exit_code = run_phase3_inpaint_pipeline(
                [
                    "--mode",
                    "gen",
                    "--profile",
                    "BCSS",
                    "--reference-image",
                    str(image),
                    "--reference-tissue-mask",
                    str(source),
                    "--reference-nuclei-mask",
                    str(nuclei),
                    "--target-tissue-mask",
                    str(target),
                    "--output",
                    str(output),
                    "--generation-mode",
                    "dry-run",
                    "--cell-fill-mode",
                    "blank",
                ]
            )

            self.assertEqual(exit_code, 0)
            for name in [
                "erased_image.png",
                "target_mask_rgb.png",
                "change_region.png",
                "target_nuclei_mask.png",
                "generated_image.png",
                "compare_panel.png",
                "generation_info.json",
                "cell_fill_log.json",
                "target_combined_mask.png",
                "pipeline_summary.json",
            ]:
                self.assertTrue((output / name).exists(), name)

            summary = json.loads((output / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["mode"], "gen")
            self.assertEqual(summary["generation_mode"], "dry-run")
            self.assertEqual(summary["cell_fill_mode"], "blank")
            self.assertGreater(summary["changed_pixels"], 0)

    def test_gen_blank_deletes_whole_source_cells_crossing_change_region(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            source = root / "source_mask.png"
            target = root / "target_mask.png"
            nuclei = root / "nuclei.png"
            output = root / "out"

            source_mask = _source_mask()
            target_mask = source_mask.copy()
            target_mask[30:40, 30:40] = 3
            source_nuclei = np.zeros((96, 96), dtype=np.uint8)
            source_nuclei[28:34, 32:36] = 101  # Crosses into the changed region.
            source_nuclei[60:64, 60:64] = 102  # Fully outside, should remain.

            _write_rgb(image)
            save_id_mask(source_mask, source)
            save_id_mask(target_mask, target)
            Image.fromarray(source_nuclei).save(nuclei)

            exit_code = run_phase3_inpaint_pipeline(
                [
                    "--mode",
                    "gen",
                    "--profile",
                    "BCSS",
                    "--reference-image",
                    str(image),
                    "--reference-tissue-mask",
                    str(source),
                    "--reference-nuclei-mask",
                    str(nuclei),
                    "--target-tissue-mask",
                    str(target),
                    "--output",
                    str(output),
                    "--generation-mode",
                    "dry-run",
                    "--cell-fill-mode",
                    "blank",
                ]
            )

            self.assertEqual(exit_code, 0)
            retained = np.asarray(Image.open(output / "retained_nuclei_mask.png").convert("L"))
            self.assertFalse(np.any(retained[28:34, 32:36]))
            self.assertTrue(np.any(retained[60:64, 60:64] == 102))

            log = json.loads((output / "cell_fill_log.json").read_text(encoding="utf-8"))
            self.assertEqual(
                log["gland_structure_policy"]["nuclei_boundary_context"][
                    "deleted_seed_components"
                ],
                1,
            )
            self.assertGreater(
                log["gland_structure_policy"]["nuclei_boundary_context"][
                    "added_buffer_pixels"
                ],
                0,
            )
            self.assertEqual(log["source_cell_integrity"]["crossing_components"], 1)
            self.assertGreaterEqual(log["source_cell_integrity"]["deleted_components"], 1)

    def test_auto_generation_route_uses_large_change_for_inpaint(self):
        self.assertEqual(_select_generation_mode("auto", 0.36, 0.35), "cross-v1")
        self.assertEqual(_select_generation_mode("auto", 0.35, 0.35), "cross-v1")
        self.assertEqual(_select_generation_mode("auto", 0.10, 0.35), "inpaint")
        self.assertEqual(_select_generation_mode("cross-v1", 0.90, 0.35), "cross-v1")

    def test_diff_mode_rejects_retired_deterministic_executor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            source = root / "source_mask.png"
            nuclei = root / "nuclei.png"
            diff_path = root / "semantic_diff.json"
            output = root / "out"

            source_mask = _source_mask()
            _write_rgb(image)
            save_id_mask(source_mask, source)
            _write_nuclei(nuclei)

            diff = json.loads(json.dumps(DEFAULT_SEMANTIC_DIFF))
            diff["necrosis_change"] = {"action": "add", "extent": "focal"}
            diff_path.write_text(json.dumps(diff), encoding="utf-8")

            self.assertRaisesRegex(
                RuntimeError,
                "retired non-LLM deterministic primitive executor",
                run_phase3_inpaint_pipeline,
                [
                    "--mode",
                    "diff",
                    "--profile",
                    "BCSS",
                    "--reference-image",
                    str(image),
                    "--reference-tissue-mask",
                    str(source),
                    "--reference-nuclei-mask",
                    str(nuclei),
                    "--semantic-diff",
                    str(diff_path),
                    "--output",
                    str(output),
                    "--generation-mode",
                    "dry-run",
                    "--cell-fill-mode",
                    "preserve",
                ]
            )

    def test_prompt_fixture_rejects_retired_deterministic_executor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            source = root / "source_mask.png"
            nuclei = root / "nuclei.png"
            diff_path = root / "semantic_diff.json"
            output = root / "out"

            _write_rgb(image)
            save_id_mask(_source_mask(), source)
            _write_nuclei(nuclei)

            diff = json.loads(json.dumps(DEFAULT_SEMANTIC_DIFF))
            diff["necrosis_change"] = {"action": "add", "extent": "focal"}
            diff_path.write_text(json.dumps(diff), encoding="utf-8")

            self.assertRaisesRegex(
                RuntimeError,
                "retired non-LLM deterministic primitive executor",
                run_phase3_inpaint_pipeline,
                [
                    "--mode",
                    "prompt",
                    "--profile",
                    "BCSS",
                    "--reference-image",
                    str(image),
                    "--reference-tissue-mask",
                    str(source),
                    "--reference-nuclei-mask",
                    str(nuclei),
                    "--old-prompt",
                    "High-grade carcinoma without necrosis.",
                    "--new-prompt",
                    "High-grade carcinoma with focal necrosis.",
                    "--parser",
                    "fixture",
                    "--semantic-diff",
                    str(diff_path),
                    "--output",
                    str(output),
                    "--generation-mode",
                    "dry-run",
                ]
            )

    def test_prompt_api_rejects_retired_deterministic_executor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            source = root / "source_mask.png"
            nuclei = root / "nuclei.png"
            output = root / "out"

            _write_rgb(image)
            save_id_mask(_source_mask(), source)
            _write_nuclei(nuclei)

            diff = json.loads(json.dumps(DEFAULT_SEMANTIC_DIFF))
            diff["necrosis_change"] = {"action": "add", "extent": "focal"}

            with patch("scripts.run_phase3_inpaint_pipeline.parse_prompts_with_api", return_value=diff):
                self.assertRaisesRegex(
                    RuntimeError,
                    "retired non-LLM deterministic primitive executor",
                    run_phase3_inpaint_pipeline,
                    [
                        "--mode",
                        "prompt",
                        "--profile",
                        "BCSS",
                        "--reference-image",
                        str(image),
                        "--reference-tissue-mask",
                        str(source),
                        "--reference-nuclei-mask",
                        str(nuclei),
                        "--old-prompt",
                        "High-grade carcinoma without necrosis.",
                        "--new-prompt",
                        "High-grade carcinoma with focal necrosis.",
                        "--parser",
                        "api",
                        "--api-model",
                        "mock-model",
                        "--output",
                        str(output),
                        "--generation-mode",
                        "dry-run",
                    ]
                )


if __name__ == "__main__":
    unittest.main()

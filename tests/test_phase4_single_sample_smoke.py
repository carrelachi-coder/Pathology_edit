import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.phase4_single_sample_smoke import (
    build_single_sample_probe_inputs,
    compare_nuclei_in_edit_region,
    expand_mask_to_full_nuclei_components,
    make_erasure_mask,
    write_gt_pred_comparison,
)

_TMP_ROOT = Path.cwd() / ".tmp_testdata"
_TMP_ROOT.mkdir(exist_ok=True)


def _write_mask(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8)).save(path)


class Phase4SingleSampleSmokeTests(unittest.TestCase):
    def test_expand_mask_to_full_nuclei_components_includes_whole_intersecting_cells(self):
        nuclei = np.zeros((10, 10), dtype=np.uint8)
        nuclei[2:5, 2:5] = 101
        nuclei[6:8, 6:8] = 102
        candidate = np.zeros((10, 10), dtype=bool)
        candidate[3:7, 3:7] = True

        expanded = expand_mask_to_full_nuclei_components(candidate, nuclei)

        self.assertTrue(np.all(expanded[2:5, 2:5]))
        self.assertTrue(np.all(expanded[6:8, 6:8]))
        self.assertFalse(expanded[0, 0])

    def test_make_erasure_mask_retries_until_min_erased_nuclei_fraction_is_met(self):
        tissue = np.ones((20, 20), dtype=np.uint8)
        nuclei = np.zeros((20, 20), dtype=np.uint8)
        nuclei[2:5, 2:5] = 101
        nuclei[8:12, 8:12] = 101
        nuclei[14:18, 14:18] = 102

        small = np.zeros((20, 20), dtype=bool)
        small[3, 3] = True
        large = np.zeros((20, 20), dtype=bool)
        large[9, 9] = True
        large[15, 15] = True
        calls = []

        def fake_erasure(tissue_map, cell_mask, skip_tissues, rng):
            calls.append(1)
            return (small if len(calls) == 1 else large), "local"

        with patch.dict("scripts.phase4_single_sample_smoke.ERASURE_FUNCTIONS", {"local": fake_erasure}):
            edit_mask, mode = make_erasure_mask(
                tissue_map=tissue,
                nuclei_map=nuclei,
                skip_tissues=set(),
                erasure_mode="local",
                seed=7,
                min_erased_nuclei_fraction=0.5,
            )

        self.assertEqual(mode, "local")
        self.assertEqual(len(calls), 2)
        self.assertFalse(np.any(edit_mask & (nuclei == 101) & small))
        self.assertTrue(np.all(edit_mask[nuclei == 102]))
        erased_fraction = np.count_nonzero(edit_mask & (nuclei > 0)) / np.count_nonzero(nuclei > 0)
        self.assertGreaterEqual(erased_fraction, 0.5)

    def test_build_probe_inputs_erases_cells_only_in_edit_region_and_writes_trace(self):
        tmpdir = _TMP_ROOT / f"phase4_single_{uuid.uuid4().hex}"
        try:
            tissue = np.ones((64, 64), dtype=np.uint8)
            tissue[:4, :] = 0
            tissue[:, :4] = 0

            nuclei = np.zeros((64, 64), dtype=np.uint8)
            nuclei[18:34, 18:34] = 101
            nuclei[38:48, 38:48] = 102

            tissue_path = tmpdir / "tissue.png"
            nuclei_path = tmpdir / "nuclei.png"
            output_dir = tmpdir / "probe"
            _write_mask(tissue_path, tissue)
            _write_mask(nuclei_path, nuclei)

            result = build_single_sample_probe_inputs(
                dataset="BCSS",
                tissue_path=tissue_path,
                nuclei_path=nuclei_path,
                output_dir=output_dir,
                erasure_mode="local",
                seed=7,
            )

            edit_mask = np.asarray(Image.open(result.edit_region_path)) > 0
            erased = np.asarray(Image.open(result.erased_nuclei_path))
            copied_gt = np.asarray(Image.open(result.gt_nuclei_path))
            metadata = json.loads(result.metadata_path.read_text(encoding="utf8"))

            self.assertGreater(int(np.count_nonzero(edit_mask)), 0)
            self.assertTrue(np.any(edit_mask & (nuclei > 0)))
            self.assertTrue(np.array_equal(copied_gt, nuclei))
            self.assertTrue(np.all(erased[edit_mask] == 0))
            self.assertTrue(np.array_equal(erased[~edit_mask], nuclei[~edit_mask]))
            for raw_id in (101, 102):
                component = nuclei == raw_id
                if np.any(component & edit_mask):
                    self.assertTrue(np.all(edit_mask[component]))
                    self.assertTrue(np.all(erased[component] == 0))
            self.assertEqual(metadata["dataset"], "BCSS")
            self.assertEqual(metadata["erasure_mode"], "local")
            self.assertGreater(metadata["gt_nuclei_pixels_in_edit"], 0)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_compare_nuclei_in_edit_region_reports_density_and_type_counts(self):
        edit_mask = np.zeros((6, 6), dtype=bool)
        edit_mask[1:5, 1:5] = True
        gt = np.zeros((6, 6), dtype=np.uint8)
        pred = np.zeros((6, 6), dtype=np.uint8)
        gt[1:3, 1:3] = 101
        gt[3:5, 3:5] = 102
        pred[1:4, 1:4] = 101

        metrics = compare_nuclei_in_edit_region(gt, pred, edit_mask)

        self.assertEqual(metrics["edit_region_pixels"], 16)
        self.assertEqual(metrics["gt_nuclei_pixels"], 8)
        self.assertEqual(metrics["pred_nuclei_pixels"], 9)
        self.assertAlmostEqual(metrics["pixel_density_ratio"], 9 / 8)
        self.assertEqual(metrics["gt_type_pixel_counts"]["101"], 4)
        self.assertEqual(metrics["gt_type_pixel_counts"]["102"], 4)
        self.assertEqual(metrics["pred_type_pixel_counts"]["101"], 9)

    def test_write_gt_pred_comparison_saves_gt_input_and_prediction_panels(self):
        tmpdir = _TMP_ROOT / f"phase4_vis_{uuid.uuid4().hex}"
        try:
            tissue = np.ones((8, 8), dtype=np.uint8)
            gt = np.zeros((8, 8), dtype=np.uint8)
            erased = np.zeros((8, 8), dtype=np.uint8)
            pred = np.zeros((8, 8), dtype=np.uint8)
            edit = np.zeros((8, 8), dtype=np.uint8)
            gt[2:4, 2:4] = 101
            pred[4:6, 4:6] = 102
            edit[1:7, 1:7] = 255

            tissue_path = tmpdir / "tissue.png"
            gt_path = tmpdir / "gt.png"
            erased_path = tmpdir / "erased.png"
            pred_path = tmpdir / "pred.png"
            edit_path = tmpdir / "edit.png"
            out_path = tmpdir / "gt_pred_comparison.png"
            for path, values in [
                (tissue_path, tissue),
                (gt_path, gt),
                (erased_path, erased),
                (pred_path, pred),
                (edit_path, edit),
            ]:
                _write_mask(path, values)

            result = write_gt_pred_comparison(
                tissue_path=tissue_path,
                gt_nuclei_path=gt_path,
                erased_nuclei_path=erased_path,
                pred_nuclei_path=pred_path,
                edit_region_path=edit_path,
                output_path=out_path,
            )

            self.assertEqual(result, out_path)
            self.assertTrue(out_path.exists())
            image = np.asarray(Image.open(out_path))
            self.assertEqual(image.shape, (42, 24, 3))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

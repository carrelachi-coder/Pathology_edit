import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_cross_v3_ref_mismatch.py"
_SPEC = importlib.util.spec_from_file_location("diagnose_cross_v3_ref_mismatch", _MODULE_PATH)
diag = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diag)


class CrossV3RefMismatchDiagnosticTests(unittest.TestCase):
    def test_glcm_stats_reports_smooth_and_textured_differently(self):
        smooth = np.full((16, 16), 0.5, dtype=np.float32)
        textured = np.indices((16, 16)).sum(axis=0).astype(np.float32) % 2

        smooth_stats = diag.glcm_stats(smooth, levels=8, distances=[1], angles=[0])
        textured_stats = diag.glcm_stats(textured, levels=8, distances=[1], angles=[0])

        self.assertAlmostEqual(smooth_stats["glcm_contrast"], 0.0)
        self.assertGreater(textured_stats["glcm_contrast"], smooth_stats["glcm_contrast"])
        self.assertLess(textured_stats["glcm_homogeneity"], smooth_stats["glcm_homogeneity"])

    def test_nuclei_morphology_counts_connected_components(self):
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[1:4, 1:4] = 101
        mask[7:10, 8:11] = 102

        stats = diag.nuclei_morphology_stats(mask)

        self.assertEqual(stats["component_count"], 2.0)
        self.assertGreater(stats["density"], 0.0)
        self.assertTrue(math.isfinite(stats["area_mean"]))

    def test_summary_flags_reference_texture_transfer_when_prediction_tracks_ref(self):
        rows = []
        for value in (0.1, 0.4, 0.8):
            row = {
                "full_l1": 0.2,
                "pred_ref_glcm_l2": 0.1,
                "pred_target_glcm_l2": 1.0,
                "pred_ref_color_l2": 0.2,
                "pred_target_color_l2": 0.3,
            }
            for key in diag._GLCM_FEATURE_KEYS:
                row[f"reference_{key}"] = value
                row[f"prediction_{key}"] = value + 0.01
                row[f"target_{key}"] = 1.0 - value
            rows.append(row)

        summary = diag.build_mismatch_summary(rows)

        self.assertEqual(summary["texture_transfer_hint"], "prediction_glcm_moves_with_reference")
        self.assertGreater(summary["reference_prediction_glcm_correlation"], summary["target_prediction_glcm_correlation"])

    def test_image_quant_stats_includes_color_and_glcm(self):
        image = Image.fromarray(np.tile(np.arange(16, dtype=np.uint8), (16, 1)) * 16, mode="L").convert("RGB")

        stats = diag.image_quant_stats(image, levels=8, distances=[1], angles=[0, 90])

        self.assertIn("rgb_mean_r", stats)
        self.assertIn("gray_glcm_contrast", stats)
        self.assertIn("hema_glcm_energy", stats)


if __name__ == "__main__":
    unittest.main()

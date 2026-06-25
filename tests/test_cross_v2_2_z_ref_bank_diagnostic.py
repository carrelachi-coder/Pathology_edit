import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np
import torch


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "diagnose_cross_v2_2_z_ref_bank.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "diagnose_cross_v2_2_z_ref_bank",
    _MODULE_PATH,
)
diagnostic = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diagnostic)


class CrossV22ZRefBankDiagnosticTests(unittest.TestCase):
    def test_compute_seam_metrics_flags_block_boundaries(self):
        latents = torch.zeros(1, 1, 4, 4)
        latents[:, :, :, 2:] = 10.0

        metrics = diagnostic.compute_seam_metrics(latents, block_size=2)

        self.assertGreater(metrics["boundary_mean_abs_adjacent_delta"], 0.0)
        self.assertEqual(metrics["non_boundary_mean_abs_adjacent_delta"], 0.0)
        self.assertGreater(metrics["seam_over_non_boundary_ratio"], 1e6)

    def test_compute_seam_metrics_has_ratio_one_for_smooth_ramp(self):
        ramp = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)

        metrics = diagnostic.compute_seam_metrics(ramp, block_size=2)

        self.assertTrue(math.isfinite(metrics["seam_over_non_boundary_ratio"]))
        self.assertGreater(metrics["boundary_mean_abs_adjacent_delta"], 0.0)
        self.assertGreater(metrics["non_boundary_mean_abs_adjacent_delta"], 0.0)

    def test_label_grid_report_reports_missing_latent_labels(self):
        original = np.array(
            [
                [0, 0, 0, 0],
                [0, 7, 7, 0],
                [0, 0, 0, 0],
                [0, 3, 3, 0],
            ],
            dtype=np.int64,
        )
        latent = np.array([[0, 0], [0, 3]], dtype=np.int64)

        report = diagnostic.label_grid_report(original, latent)

        self.assertEqual(report["retained_nonzero_labels"], [3])
        self.assertEqual(report["missing_nonzero_labels"], [7])
        self.assertAlmostEqual(report["nonzero_label_retention_fraction"], 0.5)

    def test_compute_seam_metrics_handles_block_size_one(self):
        latents = torch.randn(1, 2, 4, 4)

        metrics = diagnostic.compute_seam_metrics(latents, block_size=1)

        self.assertTrue(math.isfinite(metrics["boundary_mean_abs_adjacent_delta"]))
        self.assertTrue(math.isnan(metrics["non_boundary_mean_abs_adjacent_delta"]))
        self.assertTrue(math.isnan(metrics["seam_over_non_boundary_ratio"]))

    def test_build_zero_reference_order_report_zeros_reference_side(self):
        z_ref_bank = torch.randn(1, 3, 2, 2)

        report = diagnostic.build_zero_reference_order_report(z_ref_bank)

        self.assertTrue(report["build_happens_before_zero_ref_ablation"])
        self.assertEqual(report["with_ref_preserves_z_ref_bank_max_abs_delta"], 0.0)
        self.assertEqual(report["zero_ref_z_ref_bank_max_abs"], 0.0)
        self.assertEqual(report["zero_ref_ref_tissue_feat_max_abs"], 0.0)
        self.assertEqual(report["zero_ref_ref_nuclei_feat_max_abs"], 0.0)


if __name__ == "__main__":
    unittest.main()

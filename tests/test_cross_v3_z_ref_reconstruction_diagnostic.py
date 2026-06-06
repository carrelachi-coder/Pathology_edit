import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_cross_v3_z_ref_reconstruction.py"
_SPEC = importlib.util.spec_from_file_location("diagnose_cross_v3_z_ref_reconstruction", _MODULE_PATH)
diag = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = diag
_SPEC.loader.exec_module(diag)


class CrossV3ZRefReconstructionDiagnosticTests(unittest.TestCase):
    def test_parse_accepts_zero_z_ref_ablation_and_fixed_t(self):
        args = diag.parse_args(
            [
                "--pretrained-model-name-or-path",
                "flux",
                "--checkpoint",
                "ckpt",
                "--metadata",
                "metadata.json",
                "--output-dir",
                "out",
                "--run-zero-z-ref-ablation",
                "--fixed-t-eval-timesteps",
                "100,300.5",
            ]
        )

        self.assertTrue(args.run_zero_z_ref_ablation)
        self.assertEqual(diag.reference_variants(True), ["z_ref_only", "zero_tokens"])
        self.assertEqual(diag.parse_fixed_t_eval_timesteps(args.fixed_t_eval_timesteps), [100.0, 300.5])

    def test_select_reference_records_deduplicates_reference_ids(self):
        records = [
            {"sample_id": "a", "reference_sample_id": "r1", "reference_image": "r1.png"},
            {"sample_id": "b", "reference_sample_id": "r1", "reference_image": "r1_dup.png"},
            {"sample_id": "c", "reference_sample_id": "r2", "reference_image": "r2.png"},
        ]

        selected = diag.select_reference_records(
            records,
            reference_sample_ids=[],
            num_samples=10,
            seed=7,
        )

        self.assertEqual([row["reference_sample_id"] for row in selected], ["r1", "r2"])

    def test_attach_zero_token_delta_adds_prediction_distance(self):
        z_pred = np.zeros((3, 4, 4), dtype=np.float32)
        zero_pred = np.ones((3, 4, 4), dtype=np.float32) * 0.25
        variant_results = [
            {"variant": "z_ref_only", "row": {}, "pred_array": z_pred},
            {"variant": "zero_tokens", "row": {}, "pred_array": zero_pred},
        ]

        with TemporaryDirectory() as tmpdir:
            diag.attach_zero_token_delta(variant_results, sample_dir=Path(tmpdir))

            self.assertAlmostEqual(variant_results[0]["row"]["prediction_l1_vs_zero_tokens"], 0.25)
            self.assertTrue((Path(tmpdir) / "z_ref_vs_zero_tokens_diff.png").exists())

    def test_attach_zero_token_delta_reports_noise_prediction_change(self):
        z_pred = np.zeros((3, 4, 4), dtype=np.float32)
        zero_pred = np.ones((3, 4, 4), dtype=np.float32) * 0.25
        variant_results = [
            {
                "variant": "z_ref_only",
                "row": {"preview_timestep_key": "t500", "preview_velocity_mse": 0.4},
                "pred_array": z_pred,
                "timestep_results": {"t500": {"noise_pred_flat": torch.tensor([1.0, 0.0])}},
            },
            {
                "variant": "zero_tokens",
                "row": {"preview_timestep_key": "t500", "preview_velocity_mse": 0.5},
                "pred_array": zero_pred,
                "timestep_results": {"t500": {"noise_pred_flat": torch.tensor([0.0, 1.0])}},
            },
        ]

        with TemporaryDirectory() as tmpdir:
            diag.attach_zero_token_delta(variant_results, sample_dir=Path(tmpdir))

        self.assertAlmostEqual(
            variant_results[0]["row"]["preview_velocity_mse_delta_zero_minus_z_ref"],
            0.1,
        )
        self.assertGreater(variant_results[0]["row"]["preview_noise_pred_relative_l2_vs_zero"], 1.0)

    def test_summary_interprets_strong_and_poor_reconstruction(self):
        strong = diag.build_z_ref_reconstruction_summary(
            [
                {
                    "variant": "z_ref_only",
                    "full_l1": 0.04,
                    "full_mse": 0.002,
                    "full_psnr": 28.0,
                    "pred_ref_glcm_l2": 0.1,
                }
            ]
        )
        poor = diag.build_z_ref_reconstruction_summary(
            [
                {
                    "variant": "z_ref_only",
                    "full_l1": 0.22,
                    "full_mse": 0.08,
                    "full_psnr": 12.0,
                    "pred_ref_glcm_l2": 1.0,
                }
            ]
        )

        self.assertEqual(strong["reference_reconstruction_hint"], "strong_reference_reconstruction")
        self.assertEqual(poor["reference_reconstruction_hint"], "poor_reference_reconstruction")


if __name__ == "__main__":
    unittest.main()

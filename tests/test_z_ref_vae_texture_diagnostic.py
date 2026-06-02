import importlib.util
import unittest
from pathlib import Path

import numpy as np

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_z_ref_vae_texture.py"
_SPEC = importlib.util.spec_from_file_location("diagnose_z_ref_vae_texture", _MODULE_PATH)
diag = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diag)


class ZRefVaeTextureDiagnosticTests(unittest.TestCase):
    def test_build_reference_records_from_images(self):
        records = diag.build_reference_records(
            image_paths=["/tmp/a.png", "/tmp/b.png"],
            metadata_records=None,
            reference_sample_ids=[],
            num_samples=8,
            seed=42,
        )

        self.assertEqual([record["reference_sample_id"] for record in records], ["a", "b"])
        self.assertEqual(records[0]["reference_image"], "/tmp/a.png")

    def test_unique_reference_records_deduplicates_metadata_refs(self):
        records = [
            {"reference_sample_id": "r1", "reference_image": "r1.png"},
            {"reference_sample_id": "r1", "reference_image": "r1_again.png"},
            {"reference_sample_id": "r2", "reference_image": "r2.png"},
        ]

        selected = diag.unique_reference_records(records)

        self.assertEqual([record["reference_sample_id"] for record in selected], ["r1", "r2"])

    def test_compute_array_metrics_reports_l1_and_psnr(self):
        target = np.zeros((3, 2, 2), dtype=np.float32)
        prediction = np.ones_like(target) * 0.5

        metrics = diag.compute_array_metrics(prediction, target)

        self.assertAlmostEqual(metrics["full_l1"], 0.5)
        self.assertAlmostEqual(metrics["full_mse"], 0.25)
        self.assertAlmostEqual(metrics["full_psnr"], 4.0 * 0 + 6.020599913, places=5)

    def test_interpret_summary_flags_good_and_poor_capacity(self):
        good = {"full_l1_mean": 0.04, "full_psnr_mean": 25.0, "recon_ref_glcm_l2_mean": 0.1}
        poor = {"full_l1_mean": 0.2, "full_psnr_mean": 12.0, "recon_ref_glcm_l2_mean": 1.0}

        self.assertEqual(diag.interpret_summary(good), "z_ref_vae_preserves_reference_texture")
        self.assertEqual(diag.interpret_summary(poor), "z_ref_vae_reconstruction_is_poor")


if __name__ == "__main__":
    unittest.main()

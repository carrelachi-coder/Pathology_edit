import unittest
import importlib.util
from pathlib import Path

import numpy as np

_MODULE_PATH = Path(__file__).resolve().parents[1] / "controlnet_train" / "cli" / "eval_controlnet_flux_inpaint.py"
_SPEC = importlib.util.spec_from_file_location("eval_controlnet_flux_inpaint", _MODULE_PATH)
eval_inpaint = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(eval_inpaint)


class InpaintEvalMetricTests(unittest.TestCase):
    def test_compute_inpaint_metrics_reports_region_specific_errors(self):
        prediction = np.array(
            [
                [[0.0, 0.5], [1.0, 0.0]],
                [[0.0, 0.5], [1.0, 0.0]],
                [[0.0, 0.5], [1.0, 0.0]],
            ],
            dtype=np.float32,
        )
        target = np.zeros_like(prediction)
        change_mask = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)

        metrics = eval_inpaint.compute_inpaint_metrics(prediction, target, change_mask)

        self.assertAlmostEqual(metrics["full_l1"], 0.375)
        self.assertAlmostEqual(metrics["change_l1"], 0.25)
        self.assertAlmostEqual(metrics["keep_l1"], 0.5)
        self.assertAlmostEqual(metrics["change_mse"], 0.125)
        self.assertAlmostEqual(metrics["keep_mse"], 0.5)

    def test_select_eval_records_is_seeded_and_limited(self):
        records = [{"sample_id": f"sample_{idx}"} for idx in range(8)]

        selected_a = eval_inpaint.select_eval_records(records, num_samples=3, seed=7)
        selected_b = eval_inpaint.select_eval_records(records, num_samples=3, seed=7)

        self.assertEqual(selected_a, selected_b)
        self.assertEqual(len(selected_a), 3)
        self.assertTrue(all(record in records for record in selected_a))


if __name__ == "__main__":
    unittest.main()

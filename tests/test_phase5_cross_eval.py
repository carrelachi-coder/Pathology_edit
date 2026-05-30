import importlib.util
import unittest
from pathlib import Path

import numpy as np

_MODULE_PATH = Path(__file__).resolve().parents[1] / "controlnet_train" / "cli" / "eval_controlnet_flux_cross.py"
_SPEC = importlib.util.spec_from_file_location("eval_controlnet_flux_cross", _MODULE_PATH)
eval_cross = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(eval_cross)


class CrossEvalMetricTests(unittest.TestCase):
    def test_parse_cross_v1_eval_accepts_ip_scale(self):
        module_path = (
            Path(__file__).resolve().parents[1]
            / "controlnet_train"
            / "cli"
            / "eval_controlnet_flux_cross_v1.py"
        )
        spec = importlib.util.spec_from_file_location("eval_controlnet_flux_cross_v1", module_path)
        eval_cross_v1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eval_cross_v1)

        args = eval_cross_v1.parse_args(
            [
                "--pretrained-model-name-or-path",
                "flux",
                "--checkpoint",
                "ckpt",
                "--uni-checkpoint-path",
                "uni.bin",
                "--metadata",
                "metadata.json",
                "--output-dir",
                "out",
                "--ip-scale",
                "2.0",
            ]
        )

        self.assertEqual(args.ip_scale, 2.0)

    def test_compute_cross_metrics_reports_full_image_errors(self):
        prediction = np.array(
            [
                [[0.0, 0.5], [1.0, 0.0]],
                [[0.0, 0.5], [1.0, 0.0]],
                [[0.0, 0.5], [1.0, 0.0]],
            ],
            dtype=np.float32,
        )
        target = np.zeros_like(prediction)

        metrics = eval_cross.compute_cross_metrics(prediction, target)

        self.assertAlmostEqual(metrics["full_l1"], 0.375)
        self.assertAlmostEqual(metrics["full_mse"], 0.3125)

    def test_select_eval_records_is_seeded_and_limited(self):
        records = [{"sample_id": f"sample_{idx}"} for idx in range(8)]

        selected_a = eval_cross.select_eval_records(records, num_samples=3, seed=7)
        selected_b = eval_cross.select_eval_records(records, num_samples=3, seed=7)

        self.assertEqual(selected_a, selected_b)
        self.assertEqual(len(selected_a), 3)
        self.assertTrue(all(record in records for record in selected_a))

    def test_read_cross_metadata_accepts_pairs_object_and_raw_list(self):
        object_payload = {"pairs": [{"sample_id": "a"}]}
        list_payload = [{"sample_id": "b"}]

        self.assertEqual(eval_cross.normalize_cross_records(object_payload), [{"sample_id": "a"}])
        self.assertEqual(eval_cross.normalize_cross_records(list_payload), [{"sample_id": "b"}])


if __name__ == "__main__":
    unittest.main()

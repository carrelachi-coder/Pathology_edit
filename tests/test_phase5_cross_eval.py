import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

_MODULE_PATH = Path(__file__).resolve().parents[1] / "controlnet_train" / "cli" / "eval_controlnet_flux_cross.py"
_SPEC = importlib.util.spec_from_file_location("eval_controlnet_flux_cross", _MODULE_PATH)
eval_cross = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(eval_cross)

_V21_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "controlnet_train"
    / "cli"
    / "eval_controlnet_flux_cross_v2_1.py"
)
_V21_SPEC = importlib.util.spec_from_file_location("eval_controlnet_flux_cross_v2_1", _V21_MODULE_PATH)
eval_cross_v2_1 = importlib.util.module_from_spec(_V21_SPEC)
_V21_SPEC.loader.exec_module(eval_cross_v2_1)

_V3_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "controlnet_train"
    / "cli"
    / "eval_controlnet_flux_cross_v3.py"
)
_V3_SPEC = importlib.util.spec_from_file_location("eval_controlnet_flux_cross_v3", _V3_MODULE_PATH)
eval_cross_v3 = importlib.util.module_from_spec(_V3_SPEC)
_V3_SPEC.loader.exec_module(eval_cross_v3)

_PLOT_LOGS_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "plot_phase5_cross_training_logs.py"
)
_PLOT_LOGS_SPEC = importlib.util.spec_from_file_location(
    "plot_phase5_cross_training_logs",
    _PLOT_LOGS_MODULE_PATH,
)
plot_logs = importlib.util.module_from_spec(_PLOT_LOGS_SPEC)
sys.modules[_PLOT_LOGS_SPEC.name] = plot_logs
_PLOT_LOGS_SPEC.loader.exec_module(plot_logs)


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

    def test_parse_cross_v2_1_eval_accepts_zero_ref_ablation(self):
        args = eval_cross_v2_1.parse_args(
            [
                "--pretrained-model-name-or-path",
                "flux",
                "--checkpoint",
                "ckpt",
                "--metadata",
                "metadata.json",
                "--output-dir",
                "out",
                "--run-zero-ref-ablation",
            ]
        )

        self.assertTrue(args.run_zero_ref_ablation)
        self.assertEqual(eval_cross_v2_1._reference_variants(True), ["with_ref", "zero_ref"])

    def test_parse_cross_v2_1_eval_accepts_fixed_t_eval(self):
        args = eval_cross_v2_1.parse_args(
            [
                "--pretrained-model-name-or-path",
                "flux",
                "--checkpoint",
                "ckpt",
                "--metadata",
                "metadata.json",
                "--output-dir",
                "out",
                "--fixed-t-eval-timesteps",
                "100,300.5",
                "--fixed-t-eval-seed",
                "9",
            ]
        )

        self.assertEqual(eval_cross_v2_1._parse_fixed_t_eval_timesteps(args.fixed_t_eval_timesteps), [100.0, 300.5])
        self.assertEqual(args.fixed_t_eval_seed, 9)

    def test_cross_v2_1_ref_ablation_delta_groups_paired_rows(self):
        rows = [
            {"index": 0, "reference_condition_mode": "with_ref", "full_l1": 0.2, "full_mse": 0.04, "full_psnr": 14.0},
            {"index": 0, "reference_condition_mode": "zero_ref", "full_l1": 0.5, "full_mse": 0.25, "full_psnr": 8.0},
            {"index": 1, "reference_condition_mode": "with_ref", "full_l1": 0.3, "full_mse": 0.09, "full_psnr": 10.0},
            {"index": 1, "reference_condition_mode": "zero_ref", "full_l1": 0.4, "full_mse": 0.16, "full_psnr": 9.0},
        ]

        summary = eval_cross_v2_1._aggregate_ref_ablation_delta(rows)

        self.assertEqual(summary["num_pairs"], 2.0)
        self.assertAlmostEqual(summary["full_l1_delta_mean"], 0.2)
        self.assertAlmostEqual(summary["full_mse_delta_mean"], 0.14)
        self.assertAlmostEqual(summary["full_psnr_delta_mean"], -3.5)

    def test_cross_v2_1_fixed_t_eval_aggregates_modes_and_deltas(self):
        rows = [
            {"index": 0, "reference_condition_mode": "with_ref", "fixed_t_loss_t100": 0.4},
            {"index": 0, "reference_condition_mode": "zero_ref", "fixed_t_loss_t100": 0.5},
            {"index": 1, "reference_condition_mode": "with_ref", "fixed_t_loss_t100": 0.6},
            {"index": 1, "reference_condition_mode": "zero_ref", "fixed_t_loss_t100": 0.9},
        ]

        summary = eval_cross_v2_1._aggregate_fixed_t_losses(rows)

        self.assertAlmostEqual(summary["with_ref"]["fixed_t_loss_t100_mean"], 0.5)
        self.assertAlmostEqual(summary["zero_ref"]["fixed_t_loss_t100_mean"], 0.7)
        self.assertAlmostEqual(
            summary["delta_zero_ref_minus_with_ref"]["fixed_t_loss_t100_delta_mean"],
            0.2,
        )

    def test_parse_cross_v3_eval_uses_fixed_prompt_and_zero_ref_ablation(self):
        args = eval_cross_v3.parse_args(
            [
                "--pretrained-model-name-or-path",
                "flux",
                "--checkpoint",
                "ckpt",
                "--metadata",
                "metadata.json",
                "--output-dir",
                "out",
                "--run-zero-ref-ablation",
            ]
        )

        self.assertEqual(args.prompt_source, "fixed")
        self.assertTrue(args.run_zero_ref_ablation)
        self.assertEqual(eval_cross_v3._reference_variants(True), ["with_ref", "zero_ref"])

    def test_plot_training_logs_filters_conditional_loss_by_sample_count(self):
        rows = [
            plot_logs.ScalarRow(tag="cross_denoise_loss", step=1, value=0.0),
            plot_logs.ScalarRow(tag="cross_samples", step=1, value=0.0),
            plot_logs.ScalarRow(tag="cross_denoise_loss", step=2, value=0.7),
            plot_logs.ScalarRow(tag="cross_samples", step=2, value=1.0),
            plot_logs.ScalarRow(tag="cross_denoise_loss", step=3, value=0.0),
            plot_logs.ScalarRow(tag="cross_samples", step=3, value=0.0),
        ]

        points = plot_logs.build_filtered_loss_points(
            rows,
            ["cross_denoise_loss"],
            rolling_window=2,
        )

        self.assertEqual([point.step for point in points["cross_denoise_loss"]], [2])
        self.assertEqual(points["cross_denoise_loss"][0].valid_reason, "cross_samples>0")

    def test_plot_training_logs_filters_zero_placeholder_without_sample_count(self):
        rows = [
            plot_logs.ScalarRow(tag="self_reconstruction_denoise_loss", step=1, value=0.0),
            plot_logs.ScalarRow(tag="self_reconstruction_denoise_loss", step=2, value=0.4),
        ]

        points = plot_logs.build_filtered_loss_points(
            rows,
            ["self_reconstruction_denoise_loss"],
            rolling_window=2,
        )

        self.assertEqual([point.step for point in points["self_reconstruction_denoise_loss"]], [2])

    def test_plot_training_logs_parses_loss_ylim(self):
        self.assertEqual(plot_logs.parse_ylim("0.4,0.6"), (0.4, 0.6))
        with self.assertRaises(ValueError):
            plot_logs.parse_ylim("0.6,0.4")


if __name__ == "__main__":
    unittest.main()

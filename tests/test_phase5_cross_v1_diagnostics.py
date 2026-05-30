import importlib.util
import types
import unittest
from pathlib import Path


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "controlnet_train"
    / "cli"
    / "diagnose_cross_v1_ip_sensitivity.py"
)
_SPEC = importlib.util.spec_from_file_location("diagnose_cross_v1_ip_sensitivity", _MODULE_PATH)
diagnostics = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diagnostics)


class CrossV1DiagnosticsTests(unittest.TestCase):
    def test_parse_scale_values_accepts_comma_separated_floats(self):
        self.assertEqual(
            diagnostics.parse_scale_values("0, 0.5,1,2,4"),
            [0.0, 0.5, 1.0, 2.0, 4.0],
        )

    def test_parse_reference_variants_adds_normal_baseline(self):
        self.assertEqual(
            diagnostics.parse_reference_variants("zero,random,zero"),
            ["normal", "zero", "random"],
        )

    def test_aggregate_diagnostic_rows_groups_by_variant_and_scale(self):
        rows = [
            {
                "sample_id": "a",
                "variant": "normal",
                "scale": 0.0,
                "l1_to_target": 0.2,
                "mse_to_target": 0.04,
                "l1_vs_normal_same_scale": 0.0,
            },
            {
                "sample_id": "b",
                "variant": "normal",
                "scale": 0.0,
                "l1_to_target": 0.4,
                "mse_to_target": 0.16,
                "l1_vs_normal_same_scale": 0.0,
            },
            {
                "sample_id": "a",
                "variant": "zero",
                "scale": 1.0,
                "l1_to_target": 0.5,
                "mse_to_target": 0.25,
                "l1_vs_normal_same_scale": 0.1,
            },
        ]

        summary = diagnostics.aggregate_diagnostic_rows(rows)

        self.assertEqual(summary["num_outputs"], 3)
        self.assertEqual(summary["num_samples"], 2)
        self.assertAlmostEqual(
            summary["by_variant_scale"]["normal@0"]["l1_to_target_mean"],
            0.3,
        )
        self.assertAlmostEqual(
            summary["by_variant_scale"]["zero@1"]["l1_vs_normal_same_scale_mean"],
            0.1,
        )

    def test_aggregate_diagnostic_rows_separates_controlnet_scale_grid(self):
        rows = [
            {
                "sample_id": "a",
                "variant": "zero",
                "scale": 1.0,
                "controlnet_scale": 0.5,
                "l1_to_target": 0.2,
            },
            {
                "sample_id": "a",
                "variant": "zero",
                "scale": 1.0,
                "controlnet_scale": 1.0,
                "l1_to_target": 0.6,
            },
        ]

        summary = diagnostics.aggregate_diagnostic_rows(rows)

        self.assertAlmostEqual(
            summary["by_variant_scale"]["zero@ip1_cn0p5"]["l1_to_target_mean"],
            0.2,
        )
        self.assertAlmostEqual(
            summary["by_variant_scale"]["zero@ip1_cn1"]["l1_to_target_mean"],
            0.6,
        )

    def test_set_ip_adapter_scale_updates_double_and_single_processors(self):
        double_processor = types.SimpleNamespace(scale=[1.0])
        single_processor = types.SimpleNamespace(scale=[1.0])
        transformer = types.SimpleNamespace(
            transformer_blocks=[types.SimpleNamespace(attn=types.SimpleNamespace(processor=double_processor))],
            single_transformer_blocks=[
                types.SimpleNamespace(attn=types.SimpleNamespace(processor=single_processor))
            ],
        )

        diagnostics.set_ip_adapter_scale(transformer, 2.5)

        self.assertEqual(double_processor.scale, [2.5])
        self.assertEqual(single_processor.scale, [2.5])


if __name__ == "__main__":
    unittest.main()

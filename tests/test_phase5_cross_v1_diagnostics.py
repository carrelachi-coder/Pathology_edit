import importlib.util
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


if __name__ == "__main__":
    unittest.main()

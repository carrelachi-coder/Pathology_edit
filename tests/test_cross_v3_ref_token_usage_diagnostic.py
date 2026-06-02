import importlib.util
import sys
import unittest
from pathlib import Path

import torch

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_cross_v3_ref_token_usage.py"
_SPEC = importlib.util.spec_from_file_location("diagnose_cross_v3_ref_token_usage", _MODULE_PATH)
diag = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = diag
_SPEC.loader.exec_module(diag)


class CrossV3RefTokenUsageDiagnosticTests(unittest.TestCase):
    def test_context_gradient_stats_splits_text_and_reference_tokens(self):
        context = torch.ones(1, 5, 2)
        grad = torch.zeros_like(context)
        grad[:, :2, :] = 0.5
        grad[:, 2:, :] = 1.0

        stats = diag.context_gradient_stats(context=context, grad=grad, text_token_count=2)

        self.assertGreater(stats["grad_ref_token_mean"], stats["grad_text_token_mean"])
        self.assertGreater(stats["grad_ref_vs_text_token_mean_ratio"], 1.0)
        self.assertGreater(stats["grad_ref_token_sum_share"], 0.5)

    def test_usage_summary_interprets_productive_reference_tokens(self):
        rows = [
            {
                "zero_minus_correct_loss_t700": 0.03,
                "mismatch_minus_correct_loss_t700": 0.04,
                "token_mean_minus_correct_loss_t700": 0.02,
                "noise_pred_correct_vs_zero_relative_l2_t700": 0.05,
                "noise_pred_correct_vs_mismatch_relative_l2_t700": 0.04,
                "grad_ref_vs_text_token_mean_ratio_t700": 0.20,
                "grad_ref_token_sum_share_t700": 0.30,
                "grad_ref_entropy_norm_t700": 0.80,
            }
        ]

        summary = diag.build_usage_summary(rows)

        self.assertEqual(diag.interpret_usage_summary(summary), "reference_tokens_used_productively")

    def test_usage_summary_interprets_barely_used_reference_tokens(self):
        rows = [
            {
                "zero_minus_correct_loss_t700": 0.0,
                "mismatch_minus_correct_loss_t700": 0.0,
                "token_mean_minus_correct_loss_t700": 0.0,
                "noise_pred_correct_vs_zero_relative_l2_t700": 0.001,
                "noise_pred_correct_vs_mismatch_relative_l2_t700": 0.001,
                "grad_ref_vs_text_token_mean_ratio_t700": 0.01,
                "grad_ref_token_sum_share_t700": 0.01,
                "grad_ref_entropy_norm_t700": 0.80,
            }
        ]

        summary = diag.build_usage_summary(rows)

        self.assertEqual(diag.interpret_usage_summary(summary), "reference_tokens_barely_used_by_transformer")

    def test_select_records_can_pick_specific_reference_ids(self):
        records = [
            {"reference_sample_id": "a", "reference_image": "a.png"},
            {"reference_sample_id": "b", "reference_image": "b.png"},
        ]

        selected = diag.select_records(records, reference_sample_ids=["b"], num_samples=1, seed=42)

        self.assertEqual([record["reference_sample_id"] for record in selected], ["b"])


if __name__ == "__main__":
    unittest.main()

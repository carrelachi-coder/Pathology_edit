import importlib.util
import sys
import unittest
from pathlib import Path

import torch

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_cross_v3_ref_retrieval.py"
_SPEC = importlib.util.spec_from_file_location("diagnose_cross_v3_ref_retrieval", _MODULE_PATH)
diag = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = diag
_SPEC.loader.exec_module(diag)


class CrossV3RefRetrievalDiagnosticTests(unittest.TestCase):
    def test_select_anchor_and_mismatch_records(self):
        records = [
            {"reference_sample_id": "a", "reference_image": "a.png"},
            {"reference_sample_id": "b", "reference_image": "b.png"},
            {"reference_sample_id": "c", "reference_image": "c.png"},
        ]

        anchors = diag.select_anchor_records(
            records,
            anchor_reference_sample_ids=["b"],
            num_anchors=1,
            seed=42,
        )
        mismatches = diag.select_mismatch_records(
            records,
            anchor=anchors[0],
            num_mismatches=2,
            rng=__import__("random").Random(7),
        )

        self.assertEqual(diag.reference_record_id(anchors[0]), "b")
        self.assertNotIn("b", [diag.reference_record_id(record) for record in mismatches])
        self.assertEqual(len(mismatches), 2)

    def test_build_retrieval_row_computes_correct_preference(self):
        correct = {
            "t700": {
                "loss": 0.4,
                "noise_pred_flat": torch.tensor([1.0, 0.0]),
            }
        }
        mismatch = {
            "t700": {
                "loss": 0.6,
                "noise_pred_flat": torch.tensor([0.0, 1.0]),
            }
        }
        zero = {
            "t700": {
                "loss": 0.7,
                "noise_pred_flat": torch.tensor([0.5, 0.5]),
            }
        }

        row = diag.build_retrieval_row(
            anchor_index=0,
            mismatch_index=0,
            anchor={"reference_sample_id": "a", "reference_image": "a.png"},
            mismatch={"reference_sample_id": "b", "reference_image": "b.png"},
            correct_results=correct,
            mismatch_results=mismatch,
            zero_results=zero,
            preview_key="t700",
        )

        self.assertAlmostEqual(row["mismatch_minus_correct_loss_t700"], 0.2)
        self.assertAlmostEqual(row["zero_minus_correct_loss_t700"], 0.3)
        self.assertGreater(row["noise_pred_correct_vs_mismatch_relative_l2_t700"], 1.0)

    def test_summary_interprets_reference_retrieval(self):
        strong = [
            {
                "mismatch_minus_correct_loss_t700": 0.03,
                "zero_minus_correct_loss_t700": 0.04,
            }
        ]
        weak = [
            {
                "mismatch_minus_correct_loss_t700": 0.001,
                "zero_minus_correct_loss_t700": 0.001,
            }
        ]

        self.assertEqual(
            diag.interpret_retrieval_summary(diag.build_retrieval_summary(strong)),
            "cross_attention_prefers_correct_reference",
        )
        self.assertEqual(
            diag.interpret_retrieval_summary(diag.build_retrieval_summary(weak)),
            "no_clear_correct_reference_retrieval",
        )


if __name__ == "__main__":
    unittest.main()

import importlib.util
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "controlnet_train"
    / "cli"
    / "diagnose_ref_signal.py"
)
_SPEC = importlib.util.spec_from_file_location("diagnose_ref_signal", _MODULE_PATH)
diag = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diag)


class DiagnoseRefSignalTests(unittest.TestCase):
    def test_parse_args_allows_metadata_driven_pairwise_mode(self):
        args = diag.parse_args(
            [
                "--checkpoint",
                "ckpt",
                "--uni-checkpoint-path",
                "uni.bin",
                "--metadata",
                "metadata.jsonl",
            ]
        )

        self.assertEqual(args.metadata, "metadata.jsonl")
        self.assertIsNone(args.reference_image)
        self.assertIsNone(args.reference_image_b)

    def test_select_reference_records_deduplicates_reference_ids(self):
        records = [
            {"sample_id": "a", "reference_sample_id": "r1", "reference_image": "r1.png"},
            {"sample_id": "b", "reference_sample_id": "r1", "reference_image": "r1_dup.png"},
            {"sample_id": "c", "reference_sample_id": "r2", "reference_image": "r2.png"},
        ]

        selected = diag.select_reference_records(records, selection_mode="random", seed=7)

        self.assertEqual(len(selected), 2)
        self.assertEqual([row["reference_sample_id"] for row in selected], ["r2", "r1"])

    def test_reference_image_path_resolves_relative_to_metadata_dir(self):
        with TemporaryDirectory() as tmpdir:
            metadata_path = Path(tmpdir) / "metadata.jsonl"
            record = {"reference_image": "images/ref_a.png"}

            resolved = diag.reference_image_path(record, metadata_path=metadata_path)

            self.assertEqual(resolved, Path(tmpdir) / "images" / "ref_a.png")

    def test_encode_stages_casts_uni_tokens_to_proj_dtype(self):
        class FakeProj(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
                self.seen_dtype = None

            def forward(self, x):
                self.seen_dtype = x.dtype
                if x.dtype != torch.bfloat16:
                    raise AssertionError(f"expected bf16 input, got {x.dtype}")
                return x

        class FakeEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.skip_perceiver = True
                self.proj_mlp = FakeProj()
                self.latent_queries = torch.zeros(1, 1, 1)

            def extract_uni_features(self, image):
                return torch.ones(1, 2, 3, dtype=torch.float32)

            def forward(self, image):
                return torch.ones(1, 2, 4, dtype=torch.bfloat16)

        encoder = FakeEncoder()
        results = diag.encode_stages(encoder, None, torch.zeros(1, 3, 8, 8))

        self.assertEqual(results["1_uni"].dtype, torch.float32)
        self.assertEqual(encoder.proj_mlp.seen_dtype, torch.bfloat16)
        self.assertEqual(results["2_proj_mlp"].dtype, torch.bfloat16)

    def test_tensor_pair_metrics_reports_relative_distance(self):
        left = torch.tensor([[1.0, 0.0, 0.0]])
        right = torch.tensor([[0.0, 1.0, 0.0]])

        metrics = diag.tensor_pair_metrics(left, right)

        self.assertAlmostEqual(metrics["l2"], math.sqrt(2.0))
        self.assertAlmostEqual(metrics["relative_l2"], math.sqrt(2.0))
        self.assertAlmostEqual(metrics["cosine"], 0.0)

    def test_pairwise_verdict_calls_proj_collapsed_when_proj_is_small(self):
        results = {
            "2_proj_mlp": {"relative_l2": 0.001, "cosine": 0.9999},
            "3_perceiver_layer_1": {"relative_l2": 0.4, "cosine": 0.7},
        }

        verdict, stage_name = diag.pairwise_verdict(results, collapse_relative_l2_threshold=0.02)

        self.assertEqual(verdict, "proj_collapsed")
        self.assertIsNone(stage_name)

    def test_pairwise_verdict_calls_downstream_collapsed_when_proj_separates(self):
        results = {
            "2_proj_mlp": {"relative_l2": 0.5, "cosine": 0.1},
            "3_perceiver_layer_1": {"relative_l2": 0.01, "cosine": 0.9999},
        }

        verdict, stage_name = diag.pairwise_verdict(results, collapse_relative_l2_threshold=0.02)

        self.assertEqual(verdict, "downstream_collapsed")
        self.assertEqual(stage_name, "3_perceiver_layer_1")

    def test_pairwise_verdict_calls_proj_informative_when_all_stages_separate(self):
        results = {
            "2_proj_mlp": {"relative_l2": 0.5, "cosine": 0.1},
            "3_perceiver_layer_1": {"relative_l2": 0.3, "cosine": 0.2},
            "4_perceiver_norm": {"relative_l2": 0.25, "cosine": 0.3},
        }

        verdict, stage_name = diag.pairwise_verdict(results, collapse_relative_l2_threshold=0.02)

        self.assertEqual(verdict, "proj_informative")
        self.assertIsNone(stage_name)


if __name__ == "__main__":
    unittest.main()

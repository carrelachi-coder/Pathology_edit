import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_cross_v3_ref_token_texture_probe.py"
_SPEC = importlib.util.spec_from_file_location("diagnose_cross_v3_ref_token_texture_probe", _MODULE_PATH)
diag = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = diag
_SPEC.loader.exec_module(diag)


class CrossV3RefTokenTextureProbeTests(unittest.TestCase):
    def test_tensor_texture_signature_has_expected_shape(self):
        grid = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)

        signature = diag.tensor_texture_signature(grid)

        self.assertEqual(signature.shape, (16,))
        self.assertTrue(np.isfinite(signature).all())

    def test_retrieval_metrics_reward_matching_texture_distances(self):
        coords = np.asarray([[0.0], [1.0], [2.0], [8.0], [9.0]], dtype=np.float32)
        texture_distance = diag.pairwise_l2(diag.zscore_columns(coords))
        color_distance = diag.pairwise_l2(diag.zscore_columns(coords[::-1]))

        metrics = diag.build_retrieval_metrics(
            texture_distance,
            texture_distance=texture_distance,
            color_distance=color_distance,
            top_k=2,
        )

        self.assertAlmostEqual(metrics["pair_spearman_texture"], 1.0)
        self.assertAlmostEqual(metrics["topk_texture_overlap_mean"], 1.0)
        self.assertAlmostEqual(metrics["texture_nn_rank_in_feature_mean"], 1.0)

    def test_interpret_summary_flags_mlp_input_loss(self):
        summary = {
            "num_samples": 32.0,
            "random_topk_overlap": 5.0 / 31.0,
            "random_neighbor_rank_mean": 16.0,
            "z_ref_raw_pair_spearman_texture": 0.45,
            "z_ref_packed_pair_spearman_texture": 0.44,
            "mlp_hidden_pair_spearman_texture": 0.04,
            "mlp_hidden_pair_spearman_color": 0.03,
            "reference_tokens_pair_spearman_texture": 0.03,
            "reference_tokens_pair_spearman_color": 0.02,
        }

        self.assertEqual(
            diag.interpret_probe_summary(summary),
            "reference_mlp_input_projection_likely_loses_texture",
        )

    def test_interpret_summary_flags_tokens_preserve_texture(self):
        summary = {
            "num_samples": 32.0,
            "random_topk_overlap": 5.0 / 31.0,
            "random_neighbor_rank_mean": 16.0,
            "z_ref_raw_pair_spearman_texture": 0.45,
            "z_ref_packed_pair_spearman_texture": 0.44,
            "mlp_hidden_pair_spearman_texture": 0.38,
            "reference_tokens_pair_spearman_texture": 0.35,
            "reference_tokens_pair_spearman_color": 0.20,
        }

        self.assertEqual(
            diag.interpret_probe_summary(summary),
            "reference_tokens_preserve_texture_cross_attention_or_training_is_suspect",
        )

    def test_build_reference_records_deduplicates_metadata_refs(self):
        records = [
            {"reference_sample_id": "a", "reference_image": "a.png"},
            {"reference_sample_id": "a", "reference_image": "a_again.png"},
            {"reference_sample_id": "b", "reference_image": "b.png"},
        ]

        selected = diag.build_reference_records(
            image_paths=[],
            metadata_records=records,
            reference_sample_ids=[],
            num_samples=10,
            seed=42,
        )

        self.assertEqual([record["reference_sample_id"] for record in selected], ["a", "b"])


if __name__ == "__main__":
    unittest.main()

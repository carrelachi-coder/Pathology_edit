import unittest

import torch

from controlnet_train.modules.change_mask_encoder import ChangeMaskEncoder
from controlnet_train.modules.cross_v1_conditioning import (
    CROSS_V1_SPATIAL_REFERENCE_TARGET_DELTA,
    CrossV1ControlSpec,
    build_cross_v1_condition,
)
from controlnet_train.modules.fixed_tissue_encoder import FixedOneHotTissueEncoder
from controlnet_train.modules.hte_embedding import HierarchicalTissueEmbedding
from controlnet_train.modules.nuclei_condition_encoder import NucleiConditionEncoder
from controlnet_train.modules.tissue_condition_downsampler import TissueConditionDownsampler


class HierarchicalTissueEmbeddingTests(unittest.TestCase):
    def test_embedding_table_matches_parent_plus_delta(self):
        module = HierarchicalTissueEmbedding(embedding_dim=2)
        with torch.no_grad():
            module.parent_embeddings.weight.zero_()
            module.delta_embeddings.weight.zero_()
            module.parent_embeddings.weight[1] = torch.tensor([1.5, -2.0])
            module.delta_embeddings.weight[8] = torch.tensor([0.25, 0.75])

        table = module.embedding_table()

        self.assertTrue(torch.allclose(table[8], torch.tensor([1.75, -1.25])))


class TissueConditionDownsamplerTests(unittest.TestCase):
    def test_downsamples_to_flux_latent_resolution(self):
        module = TissueConditionDownsampler(in_channels=64, hidden_channels=64, num_blocks=3)
        x = torch.randn(2, 64, 64, 64)

        out = module(x)

        self.assertEqual(out.shape, (2, 64, 8, 8))


class FixedOneHotTissueEncoderTests(unittest.TestCase):
    def test_encodes_fixed_one_hot_layout_without_trainable_parameters(self):
        module = FixedOneHotTissueEncoder(num_classes=4, downsample_factor=2, scale=4.0)
        tissue = torch.tensor(
            [[
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [3, 3, 0, 0],
                [3, 3, 0, 0],
            ]]
        )

        out = module(tissue)

        self.assertEqual(out.shape, (1, 4, 2, 2))
        self.assertEqual(sum(param.numel() for param in module.parameters()), 0)
        self.assertTrue(torch.equal(out[0, :, 0, 0], torch.tensor([0.0, 4.0, 0.0, 0.0])))
        self.assertTrue(torch.equal(out[0, :, 1, 1], torch.tensor([4.0, 0.0, 0.0, 0.0])))


class ChangeMaskEncoderTests(unittest.TestCase):
    def test_encodes_binary_mask_to_four_channel_feature(self):
        module = ChangeMaskEncoder(out_channels=4)
        x = torch.randint(0, 2, (2, 1, 8, 8), dtype=torch.float32)

        out = module(x)

        self.assertEqual(out.shape, (2, 4, 8, 8))


class NucleiConditionEncoderTests(unittest.TestCase):
    def test_remaps_raw_nuclei_ids_and_downsamples(self):
        module = NucleiConditionEncoder(embedding_dim=8, out_channels=16, num_blocks=3)
        nuclei_map = torch.tensor(
            [[
                [0, 101, 102, 103, 104, 105, 0, 101],
                [105, 104, 103, 102, 101, 0, 105, 104],
                [0, 101, 102, 103, 104, 105, 0, 101],
                [105, 104, 103, 102, 101, 0, 105, 104],
                [0, 101, 102, 103, 104, 105, 0, 101],
                [105, 104, 103, 102, 101, 0, 105, 104],
                [0, 101, 102, 103, 104, 105, 0, 101],
                [105, 104, 103, 102, 101, 0, 105, 104],
            ]],
            dtype=torch.long,
        )

        remapped = module.remap_ids(nuclei_map)
        out = module(nuclei_map)

        self.assertEqual(remapped.min().item(), 0)
        self.assertEqual(remapped.max().item(), 5)
        self.assertEqual(out.shape, (1, 16, 1, 1))


class CrossV1ConditioningTests(unittest.TestCase):
    def test_reference_target_delta_appends_target_minus_reference_features(self):
        ref_tissue = torch.full((1, 2, 2, 2), 1.0)
        ref_nuclei = torch.full((1, 1, 2, 2), 2.0)
        target_tissue = torch.full((1, 2, 2, 2), 4.0)
        target_nuclei = torch.full((1, 1, 2, 2), 7.0)

        out = build_cross_v1_condition(
            reference_tissue_feat=ref_tissue,
            reference_nuclei_feat=ref_nuclei,
            target_tissue_feat=target_tissue,
            target_nuclei_feat=target_nuclei,
            spatial_mode=CROSS_V1_SPATIAL_REFERENCE_TARGET_DELTA,
        )

        self.assertEqual(out.shape, (1, 9, 2, 2))
        self.assertTrue(torch.equal(out[:, 0:2], ref_tissue))
        self.assertTrue(torch.equal(out[:, 2:3], ref_nuclei))
        self.assertTrue(torch.equal(out[:, 3:5], target_tissue))
        self.assertTrue(torch.equal(out[:, 5:6], target_nuclei))
        self.assertTrue(torch.equal(out[:, 6:8], target_tissue - ref_tissue))
        self.assertTrue(torch.equal(out[:, 8:9], target_nuclei - ref_nuclei))

    def test_reference_target_delta_spec_counts_three_spatial_groups(self):
        spec = CrossV1ControlSpec(
            tissue_channels=2,
            nuclei_channels=1,
            spatial_mode=CROSS_V1_SPATIAL_REFERENCE_TARGET_DELTA,
        )

        self.assertEqual(spec.raw_channels, 9)


if __name__ == "__main__":
    unittest.main()

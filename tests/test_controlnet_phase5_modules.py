import unittest
from types import SimpleNamespace

import torch

from controlnet_train.modules.change_mask_encoder import ChangeMaskEncoder
from controlnet_train.modules.conditioning import build_cross_v0_condition
from controlnet_train.modules.cross_v1_conditioning import (
    CROSS_V1_SPATIAL_REFERENCE_TARGET_DELTA,
    CrossV1ControlSpec,
    build_cross_v1_condition,
)
from controlnet_train.modules.cross_v2_1_conditioning import (
    CrossV21ControlSpec,
    build_cross_v2_1_condition,
    deterministic_latent_from_posterior,
)
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


class CrossV0ConditioningTests(unittest.TestCase):
    def test_build_cross_v0_condition_uses_plan_channel_order(self):
        ref_image = torch.randn(2, 16, 8, 8)
        ref_tissue = torch.randn(2, 64, 8, 8)
        ref_nuclei = torch.randn(2, 16, 8, 8)
        target_tissue = torch.randn(2, 64, 8, 8)
        target_nuclei = torch.randn(2, 16, 8, 8)

        out = build_cross_v0_condition(
            reference_image_latent=ref_image,
            reference_tissue_feat=ref_tissue,
            reference_nuclei_feat=ref_nuclei,
            target_tissue_feat=target_tissue,
            target_nuclei_feat=target_nuclei,
        )

        self.assertEqual(out.shape, (2, 176, 8, 8))
        self.assertTrue(torch.equal(out[:, :16], ref_image))
        self.assertTrue(torch.equal(out[:, 16:80], ref_tissue))
        self.assertTrue(torch.equal(out[:, 80:96], ref_nuclei))
        self.assertTrue(torch.equal(out[:, 96:160], target_tissue))
        self.assertTrue(torch.equal(out[:, 160:176], target_nuclei))


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


class CrossV21ConditioningTests(unittest.TestCase):
    def test_build_cross_v2_1_condition_uses_fixed_channel_order(self):
        z_ref = torch.full((2, 2, 3, 3), 1.0)
        ref_tissue = torch.full((2, 3, 3, 3), 2.0)
        ref_nuclei = torch.full((2, 1, 3, 3), 3.0)
        tar_tissue = torch.full((2, 3, 3, 3), 4.0)
        tar_nuclei = torch.full((2, 1, 3, 3), 5.0)

        out = build_cross_v2_1_condition(
            z_ref=z_ref,
            ref_tissue_feat=ref_tissue,
            ref_nuclei_feat=ref_nuclei,
            tar_tissue_feat=tar_tissue,
            tar_nuclei_feat=tar_nuclei,
        )

        self.assertEqual(out.shape, (2, 10, 3, 3))
        self.assertTrue(torch.equal(out[:, 0:2], z_ref))
        self.assertTrue(torch.equal(out[:, 2:5], ref_tissue))
        self.assertTrue(torch.equal(out[:, 5:6], ref_nuclei))
        self.assertTrue(torch.equal(out[:, 6:9], tar_tissue))
        self.assertTrue(torch.equal(out[:, 9:10], tar_nuclei))

    def test_cross_v2_1_spec_counts_reference_latent_and_two_mask_groups(self):
        spec = CrossV21ControlSpec(
            reference_latent_channels=2,
            tissue_channels=3,
            nuclei_channels=1,
        )

        self.assertEqual(spec.raw_channels, 10)
        self.assertEqual(spec.packed_channels, 40)
        self.assertEqual(spec.packed_reference_mask_start, 8)
        self.assertEqual(spec.packed_target_mask_start, 24)

    def test_cross_v2_1_reference_latent_helper_uses_posterior_mode(self):
        class Posterior:
            mean = torch.full((1, 1, 2, 2), 3.0)

            def sample(self):
                return torch.full((1, 1, 2, 2), 7.0)

            def mode(self):
                return torch.full((1, 1, 2, 2), 5.0)

        latents = deterministic_latent_from_posterior(Posterior())

        self.assertTrue(torch.equal(latents, torch.full((1, 1, 2, 2), 5.0)))

    def test_cross_v2_1_reference_latent_helper_falls_back_to_mean(self):
        posterior = SimpleNamespace(mean=torch.full((1, 1, 2, 2), 3.0))

        latents = deterministic_latent_from_posterior(posterior)

        self.assertTrue(torch.equal(latents, torch.full((1, 1, 2, 2), 3.0)))


if __name__ == "__main__":
    unittest.main()

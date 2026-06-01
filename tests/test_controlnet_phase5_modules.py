import unittest
import importlib.util
import sys
from pathlib import Path
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
    CROSS_V2_1_REFERENCE_ZERO_REF,
    CrossV21ControlSpec,
    apply_cross_v2_1_reference_mode,
    build_cross_v2_1_condition,
    deterministic_latent_from_posterior,
    normalize_cross_v2_1_reference_mode,
)
from controlnet_train.modules.cross_v3_conditioning import (
    CROSS_V3_PROMPT,
    CROSS_V3_REFERENCE_ZERO_REF,
    CrossV3ControlSpec,
    CrossV3ReferenceContextEncoder,
    CrossV3ReferenceSpec,
    append_cross_v3_reference_context,
    apply_cross_v3_reference_mode,
    apply_cross_v3_reference_token_mode,
    build_cross_v3_control_condition,
    normalize_cross_v3_reference_mode,
    pack_cross_v3_reference_grid,
)
from controlnet_train.modules.hte_embedding import HierarchicalTissueEmbedding
from controlnet_train.modules.nuclei_condition_encoder import NucleiConditionEncoder
from controlnet_train.modules.tissue_condition_downsampler import TissueConditionDownsampler

_Z_REF_DIAG_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_cross_v2_1_z_ref.py"
_Z_REF_DIAG_SPEC = importlib.util.spec_from_file_location("diagnose_cross_v2_1_z_ref", _Z_REF_DIAG_PATH)
z_ref_diag = importlib.util.module_from_spec(_Z_REF_DIAG_SPEC)
sys.modules[_Z_REF_DIAG_SPEC.name] = z_ref_diag
_Z_REF_DIAG_SPEC.loader.exec_module(z_ref_diag)


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

    def test_cross_v2_1_zero_ref_mode_zeros_reference_side_features(self):
        z_ref = torch.ones(1, 2, 2, 2)
        ref_tissue = torch.full((1, 3, 2, 2), 2.0)
        ref_nuclei = torch.full((1, 1, 2, 2), 3.0)

        z_out, tissue_out, nuclei_out = apply_cross_v2_1_reference_mode(
            z_ref=z_ref,
            ref_tissue_feat=ref_tissue,
            ref_nuclei_feat=ref_nuclei,
            mode=CROSS_V2_1_REFERENCE_ZERO_REF,
        )

        self.assertTrue(torch.equal(z_out, torch.zeros_like(z_ref)))
        self.assertTrue(torch.equal(tissue_out, torch.zeros_like(ref_tissue)))
        self.assertTrue(torch.equal(nuclei_out, torch.zeros_like(ref_nuclei)))
        self.assertEqual(normalize_cross_v2_1_reference_mode("zero-ref"), CROSS_V2_1_REFERENCE_ZERO_REF)

    def test_z_ref_diagnostic_slices_packed_projection_groups(self):
        spec = CrossV21ControlSpec(reference_latent_channels=2, tissue_channels=3, nuclei_channels=1)
        weight = torch.zeros(5, spec.packed_channels)
        weight[:, : spec.packed_reference_latent_channels] = 1.0
        weight[:, spec.packed_reference_mask_start : spec.packed_target_mask_start] = 2.0

        summary = z_ref_diag.summarize_x_embedder_projection(weight, spec)

        self.assertEqual(summary["groups"]["z_ref"]["input_span"], [0, 8])
        self.assertEqual(summary["groups"]["ref_masks"]["input_span"], [8, 24])
        self.assertEqual(summary["groups"]["target_masks"]["input_span"], [24, 40])
        self.assertGreater(summary["groups"]["ref_masks"]["fro_norm"], summary["groups"]["z_ref"]["fro_norm"])


class CrossV3ConditioningTests(unittest.TestCase):
    def test_cross_v3_control_condition_is_target_only(self):
        tar_tissue = torch.full((2, 3, 4, 4), 4.0)
        tar_nuclei = torch.full((2, 1, 4, 4), 5.0)

        out = build_cross_v3_control_condition(
            tar_tissue_feat=tar_tissue,
            tar_nuclei_feat=tar_nuclei,
        )

        self.assertEqual(out.shape, (2, 4, 4, 4))
        self.assertTrue(torch.equal(out[:, 0:3], tar_tissue))
        self.assertTrue(torch.equal(out[:, 3:4], tar_nuclei))

    def test_cross_v3_specs_separate_control_and_reference_channels(self):
        control_spec = CrossV3ControlSpec(tissue_channels=3, nuclei_channels=1)
        reference_spec = CrossV3ReferenceSpec(
            reference_latent_channels=2,
            tissue_channels=3,
            nuclei_channels=1,
            token_dim=8,
        )

        self.assertEqual(CROSS_V3_PROMPT, "histopathology image")
        self.assertEqual(control_spec.raw_channels, 4)
        self.assertEqual(control_spec.packed_channels, 16)
        self.assertEqual(reference_spec.raw_channels, 6)
        self.assertEqual(reference_spec.packed_channels, 24)
        self.assertEqual(reference_spec.token_dim, 8)

    def test_cross_v3_reference_grid_packs_latent_and_ref_masks(self):
        z_ref = torch.full((1, 2, 4, 4), 1.0)
        ref_tissue = torch.full((1, 3, 4, 4), 2.0)
        ref_nuclei = torch.full((1, 1, 4, 4), 3.0)

        packed = pack_cross_v3_reference_grid(
            z_ref=z_ref,
            ref_tissue_feat=ref_tissue,
            ref_nuclei_feat=ref_nuclei,
        )

        self.assertEqual(packed.shape, (1, 4, 24))
        self.assertTrue(torch.equal(packed[:, :, :8], torch.ones(1, 4, 8)))
        self.assertTrue(torch.equal(packed[:, :, 8:20], torch.full((1, 4, 12), 2.0)))
        self.assertTrue(torch.equal(packed[:, :, 20:24], torch.full((1, 4, 4), 3.0)))

    def test_cross_v3_reference_context_encoder_projects_tokens(self):
        encoder = CrossV3ReferenceContextEncoder(
            reference_latent_channels=2,
            tissue_channels=3,
            nuclei_channels=1,
            token_dim=7,
            hidden_dim=5,
        )

        tokens = encoder(
            z_ref=torch.randn(2, 2, 4, 4),
            ref_tissue_feat=torch.randn(2, 3, 4, 4),
            ref_nuclei_feat=torch.randn(2, 1, 4, 4),
        )

        self.assertEqual(tokens.shape, (2, 4, 7))

    def test_cross_v3_reference_context_encoder_uses_small_output_init(self):
        encoder = CrossV3ReferenceContextEncoder(
            reference_latent_channels=2,
            tissue_channels=3,
            nuclei_channels=1,
            token_dim=512,
            hidden_dim=256,
            output_init_std=0.02,
        )

        self.assertAlmostEqual(float(encoder.proj_out.weight.std().item()), 0.02, delta=0.004)
        self.assertTrue(torch.equal(encoder.proj_out.bias, torch.zeros_like(encoder.proj_out.bias)))

    def test_cross_v3_appends_reference_tokens_with_zero_text_ids(self):
        prompt_embeds = torch.randn(2, 5, 7)
        text_ids = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        ref_tokens = torch.randn(2, 4, 7)

        context, context_ids = append_cross_v3_reference_context(
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            reference_tokens=ref_tokens,
        )

        self.assertEqual(context.shape, (2, 9, 7))
        self.assertEqual(context_ids.shape, (9, 3))
        self.assertTrue(torch.equal(context[:, :5], prompt_embeds))
        self.assertTrue(torch.equal(context[:, 5:], ref_tokens))
        self.assertTrue(torch.equal(context_ids[:5], text_ids))
        self.assertTrue(torch.equal(context_ids[5:], torch.zeros(4, 3)))

    def test_cross_v3_zero_ref_modes_zero_reference_path_only(self):
        z_ref = torch.ones(1, 2, 2, 2)
        ref_tissue = torch.full((1, 3, 2, 2), 2.0)
        ref_nuclei = torch.full((1, 1, 2, 2), 3.0)
        tokens = torch.ones(1, 4, 8)

        z_out, tissue_out, nuclei_out = apply_cross_v3_reference_mode(
            z_ref=z_ref,
            ref_tissue_feat=ref_tissue,
            ref_nuclei_feat=ref_nuclei,
            mode=CROSS_V3_REFERENCE_ZERO_REF,
        )

        self.assertTrue(torch.equal(z_out, torch.zeros_like(z_ref)))
        self.assertTrue(torch.equal(tissue_out, torch.zeros_like(ref_tissue)))
        self.assertTrue(torch.equal(nuclei_out, torch.zeros_like(ref_nuclei)))
        self.assertTrue(torch.equal(apply_cross_v3_reference_token_mode(tokens, "zero-ref"), torch.zeros_like(tokens)))
        self.assertEqual(normalize_cross_v3_reference_mode("zero-ref"), CROSS_V3_REFERENCE_ZERO_REF)


if __name__ == "__main__":
    unittest.main()

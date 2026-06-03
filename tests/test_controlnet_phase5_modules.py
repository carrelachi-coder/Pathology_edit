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
    CROSS_V3_ROUTE_COARSE,
    CROSS_V3_ROUTE_FINE,
    CROSS_V3_ROUTE_NONE,
    CrossV3ControlSpec,
    CrossV3ReferenceContextEncoder,
    CrossV3ReferenceSpec,
    append_cross_v3_reference_context,
    apply_cross_v3_reference_mode,
    apply_cross_v3_reference_token_mode,
    build_cross_v3_control_condition,
    build_cross_v3_reference_route_ids,
    cross_v3_route_class_count,
    normalize_cross_v3_reference_mode,
    normalize_cross_v3_reference_route_mode,
    pack_cross_v3_reference_grid,
)
from controlnet_train.modules.cross_v4_conditioning import (
    NUM_CELL_WITH_BG,
    CrossV4CorrespondenceBiasConfig,
    CrossV4PriorTokenBank,
    CrossV4ReferenceContextEncoder,
    append_cross_v4_context,
    apply_cross_v4_reference_encoding_mode,
    build_cross_v4_correspondence_bias,
    build_cross_v4_token_metadata,
    remap_cross_v4_cell_ids,
)
from controlnet_train.training.cross_v4_attention import (
    FluxCrossV4AttnProcessor2_0,
    parse_cross_v4_block_indices,
)
from controlnet_train.modules.fixed_tissue_encoder import FixedOneHotTissueEncoder
from controlnet_train.modules.hte_embedding import HierarchicalTissueEmbedding
from controlnet_train.modules.nuclei_condition_encoder import NucleiConditionEncoder
from controlnet_train.modules.tissue_condition_downsampler import TissueConditionDownsampler

_Z_REF_DIAG_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_cross_v2_1_z_ref.py"
_Z_REF_DIAG_SPEC = importlib.util.spec_from_file_location("diagnose_cross_v2_1_z_ref", _Z_REF_DIAG_PATH)
z_ref_diag = importlib.util.module_from_spec(_Z_REF_DIAG_SPEC)
sys.modules[_Z_REF_DIAG_SPEC.name] = z_ref_diag
_Z_REF_DIAG_SPEC.loader.exec_module(z_ref_diag)

_CROSS_V3_TRAINING_PATH = (
    Path(__file__).resolve().parents[1] / "controlnet_train" / "training" / "flux_phase5_cross_v3.py"
)
_CROSS_V3_TRAINING_SPEC = importlib.util.spec_from_file_location("flux_phase5_cross_v3", _CROSS_V3_TRAINING_PATH)
cross_v3_training = importlib.util.module_from_spec(_CROSS_V3_TRAINING_SPEC)
try:
    sys.modules[_CROSS_V3_TRAINING_SPEC.name] = cross_v3_training
    _CROSS_V3_TRAINING_SPEC.loader.exec_module(cross_v3_training)
except (ModuleNotFoundError, RuntimeError):
    cross_v3_training = None


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


@unittest.skipIf(cross_v3_training is None, "diffusers/accelerate are required for cross-v3 training helpers")
class CrossV3TargetTissuePathTests(unittest.TestCase):
    def test_target_tissue_path_defaults_to_shared_hte(self):
        modules = {
            "hte": torch.nn.Identity(),
            "tissue_downsampler": torch.nn.Identity(),
        }

        self.assertIs(cross_v3_training._target_hte(modules), modules["hte"])
        self.assertIs(cross_v3_training._target_tissue_downsampler(modules), modules["tissue_downsampler"])

    def test_target_tissue_path_prefers_low_capacity_target_branch(self):
        modules = {
            "hte": torch.nn.Identity(),
            "tissue_downsampler": torch.nn.Identity(),
            "target_hte": torch.nn.Linear(2, 2),
            "target_tissue_downsampler": torch.nn.Conv2d(2, 2, 1),
        }

        self.assertIs(cross_v3_training._target_hte(modules), modules["target_hte"])
        self.assertIs(
            cross_v3_training._target_tissue_downsampler(modules),
            modules["target_tissue_downsampler"],
        )

    def test_target_tissue_path_prefers_fixed_one_hot_encoder(self):
        modules = {
            "hte": torch.nn.Identity(),
            "tissue_downsampler": torch.nn.Identity(),
            "target_tissue_encoder": FixedOneHotTissueEncoder(num_classes=4, downsample_factor=2),
        }

        self.assertIs(cross_v3_training._target_hte(modules), modules["target_tissue_encoder"])
        self.assertIsInstance(cross_v3_training._target_tissue_downsampler(modules), torch.nn.Identity)

    def test_feature_stats_split_target_tissue_and_nuclei_channels(self):
        tar_tissue = torch.ones(1, 3, 2, 2) * 2.0
        tar_nuclei = torch.ones(1, 2, 2, 2) * 0.5
        ref_tissue = torch.ones(1, 3, 2, 2) * 4.0
        ref_nuclei = torch.ones(1, 2, 2, 2) * 0.25
        reference_tokens = torch.ones(1, 4, 6) * 0.25

        stats = cross_v3_training._cross_v3_feature_stats(
            tar_tissue_feat=tar_tissue,
            tar_nuclei_feat=tar_nuclei,
            ref_tissue_feat=ref_tissue,
            ref_nuclei_feat=ref_nuclei,
            reference_tokens=reference_tokens,
        )

        self.assertAlmostEqual(stats["target_tissue_abs_mean"], 2.0)
        self.assertAlmostEqual(stats["reference_tissue_abs_mean"], 4.0)
        self.assertAlmostEqual(stats["target_to_reference_tissue_abs_mean_ratio"], 0.5)
        self.assertAlmostEqual(stats["target_nuclei_abs_mean"], 0.5)
        self.assertAlmostEqual(stats["reference_nuclei_abs_mean"], 0.25)
        self.assertAlmostEqual(stats["reference_token_abs_mean"], 0.25)
        self.assertEqual(stats["target_feature_height"], 2.0)


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
        self.assertEqual(reference_spec.route_class_count, 0)

    @unittest.skipIf(cross_v3_training is None, "diffusers/accelerate are required for cross-v3 training helpers")
    def test_cross_v3_ref_swap_token_variants_zero_or_shuffle_reference_tokens(self):
        tokens = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)

        zero_tokens = cross_v3_training._build_swapped_reference_tokens(tokens, "zero")
        random_tokens = cross_v3_training._build_swapped_reference_tokens(tokens, "random")
        single_random = cross_v3_training._build_swapped_reference_tokens(tokens[:1], "random")

        self.assertTrue(torch.equal(zero_tokens, torch.zeros_like(tokens)))
        self.assertTrue(torch.equal(random_tokens, tokens.index_select(0, torch.tensor([2, 0, 1]))))
        self.assertIsNone(single_random)
        self.assertEqual(cross_v3_training._parse_ref_swap_variants("zero,batch-shuffle"), ["zero", "random"])

    def test_cross_v3_reference_route_modes_keep_fine_tumor_ids_when_requested(self):
        self.assertEqual(normalize_cross_v3_reference_route_mode("fine-anchor"), CROSS_V3_ROUTE_FINE)
        self.assertEqual(normalize_cross_v3_reference_route_mode("coarse"), CROSS_V3_ROUTE_COARSE)
        self.assertEqual(normalize_cross_v3_reference_route_mode("off"), CROSS_V3_ROUTE_NONE)
        self.assertEqual(cross_v3_route_class_count("fine"), 16)
        self.assertEqual(cross_v3_route_class_count("coarse"), 8)
        self.assertEqual(cross_v3_route_class_count("none"), 0)
        self.assertEqual(CrossV3ReferenceSpec(route_anchor_mode="fine").route_class_count, 16)

        fine_tumor_mask = torch.tensor(
            [
                [8, 8, 9, 9],
                [8, 8, 9, 9],
                [10, 10, 1, 1],
                [10, 10, 1, 1],
            ],
            dtype=torch.long,
        )

        fine_ids, fine_confidence = build_cross_v3_reference_route_ids(
            ref_tissue_ids=fine_tumor_mask,
            token_height=2,
            token_width=2,
            route_anchor_mode="fine",
        )
        coarse_ids, coarse_confidence = build_cross_v3_reference_route_ids(
            ref_tissue_ids=fine_tumor_mask,
            token_height=2,
            token_width=2,
            route_anchor_mode="coarse",
        )

        self.assertTrue(torch.equal(fine_ids[0], torch.tensor([[8, 9], [10, 1]])))
        self.assertTrue(torch.equal(coarse_ids[0], torch.ones(2, 2, dtype=torch.long)))
        self.assertTrue(torch.equal(fine_confidence, torch.ones_like(fine_confidence)))
        self.assertTrue(torch.equal(coarse_confidence, torch.ones_like(coarse_confidence)))

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

    def test_cross_v3_reference_context_encoder_appends_fine_route_anchors(self):
        encoder = CrossV3ReferenceContextEncoder(
            reference_latent_channels=2,
            tissue_channels=3,
            nuclei_channels=1,
            token_dim=7,
            hidden_dim=5,
            route_anchor_mode="fine",
        )
        ref_tissue_ids = torch.tensor(
            [
                [8, 8, 9, 9],
                [8, 8, 9, 9],
                [10, 10, 1, 1],
                [10, 10, 1, 1],
            ],
            dtype=torch.long,
        ).unsqueeze(0)

        tokens = encoder(
            z_ref=torch.randn(1, 2, 4, 4),
            ref_tissue_feat=torch.randn(1, 3, 4, 4),
            ref_nuclei_feat=torch.randn(1, 1, 4, 4),
            ref_tissue_ids=ref_tissue_ids,
        )

        self.assertEqual(tokens.shape, (1, 16 + 4, 7))

    def test_cross_v3_reference_context_encoder_requires_route_ids_when_enabled(self):
        encoder = CrossV3ReferenceContextEncoder(
            reference_latent_channels=2,
            tissue_channels=3,
            nuclei_channels=1,
            token_dim=7,
            hidden_dim=5,
            route_anchor_mode="fine",
        )

        with self.assertRaisesRegex(ValueError, "ref_tissue_ids"):
            encoder(
                z_ref=torch.randn(1, 2, 4, 4),
                ref_tissue_feat=torch.randn(1, 3, 4, 4),
                ref_nuclei_feat=torch.randn(1, 1, 4, 4),
            )

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


class CrossV4ConditioningTests(unittest.TestCase):
    def test_cross_v4_token_metadata_downsamples_tissue_and_cell_masks(self):
        tissue = torch.tensor(
            [[
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [3, 3, 4, 4],
                [3, 3, 4, 4],
            ]],
            dtype=torch.long,
        )
        nuclei = torch.tensor(
            [[
                [101, 101, 0, 0],
                [101, 102, 0, 0],
                [104, 104, 102, 102],
                [104, 0, 102, 0],
            ]],
            dtype=torch.long,
        )

        meta = build_cross_v4_token_metadata(
            tissue_ids=tissue,
            nuclei_ids=nuclei,
            token_height=2,
            token_width=2,
        )

        self.assertTrue(torch.equal(meta.tissue_fine_id[0], torch.tensor([1, 2, 3, 4])))
        self.assertTrue(torch.equal(meta.tissue_coarse_id[0], torch.tensor([1, 2, 3, 4])))
        self.assertTrue(torch.equal(meta.tissue_confidence, torch.ones_like(meta.tissue_confidence)))
        self.assertEqual(meta.cell_hist.shape, (1, 4, NUM_CELL_WITH_BG))
        self.assertAlmostEqual(float(meta.cell_density[0, 0].item()), 1.0)
        self.assertAlmostEqual(float(meta.cell_hist[0, 1, 0].item()), 1.0)
        self.assertAlmostEqual(float(meta.cell_density[0, 2].item()), 0.75)
        self.assertTrue(torch.equal(remap_cross_v4_cell_ids(torch.tensor([0, 101, 105])), torch.tensor([0, 1, 5])))

    def test_cross_v4_reference_encoder_returns_local_tokens_route_anchors_and_metadata(self):
        encoder = CrossV4ReferenceContextEncoder(
            reference_latent_channels=2,
            tissue_channels=3,
            nuclei_channels=1,
            token_dim=7,
            hidden_dim=5,
            route_anchor_mode="coarse",
        )
        tissue = torch.tensor(
            [[
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [3, 3, 4, 4],
                [3, 3, 4, 4],
            ]],
            dtype=torch.long,
        )
        nuclei = torch.zeros(1, 4, 4, dtype=torch.long)

        encoding = encoder(
            z_ref=torch.randn(1, 2, 4, 4),
            ref_tissue_feat=torch.randn(1, 3, 4, 4),
            ref_nuclei_feat=torch.randn(1, 1, 4, 4),
            ref_tissue_ids=tissue,
            ref_nuclei_ids=nuclei,
        )

        self.assertEqual(encoding.local_tokens.shape, (1, 4, 7))
        self.assertEqual(encoding.route_anchor_tokens.shape, (1, 8, 7))
        self.assertEqual(encoding.tokens.shape, (1, 12, 7))
        self.assertTrue(torch.equal(encoding.metadata.tissue_fine_id[0], torch.tensor([1, 2, 3, 4])))

    def test_cross_v4_context_segments_and_prior_bias_fallback(self):
        prompt_embeds = torch.zeros(1, 2, 6)
        text_ids = torch.zeros(2, 3)
        ref_encoder = CrossV4ReferenceContextEncoder(
            reference_latent_channels=1,
            tissue_channels=1,
            nuclei_channels=1,
            token_dim=6,
            hidden_dim=4,
        )
        ref_tissue = torch.tensor(
            [[
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [2, 2, 2, 2],
                [2, 2, 2, 2],
            ]],
            dtype=torch.long,
        )
        ref_encoding = ref_encoder(
            z_ref=torch.randn(1, 1, 4, 4),
            ref_tissue_feat=torch.randn(1, 1, 4, 4),
            ref_nuclei_feat=torch.randn(1, 1, 4, 4),
            ref_tissue_ids=ref_tissue,
            ref_nuclei_ids=torch.zeros(1, 4, 4, dtype=torch.long),
        )
        prior_bank = CrossV4PriorTokenBank(
            token_dim=6,
            tissue_prior_tokens_per_class=1,
            cell_prior_tokens_per_class=1,
            global_style_tokens=1,
        )
        prior = prior_bank(ref_encoding.local_tokens)
        context = append_cross_v4_context(
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            reference_encoding=ref_encoding,
            prior_tokens=prior,
        )
        target_meta = build_cross_v4_token_metadata(
            tissue_ids=torch.tensor(
                [[
                    [1, 1, 3, 3],
                    [1, 1, 3, 3],
                    [2, 2, 3, 3],
                    [2, 2, 3, 3],
                ]],
                dtype=torch.long,
            ),
            nuclei_ids=torch.zeros(1, 4, 4, dtype=torch.long),
            token_height=2,
            token_width=2,
        )

        bias = build_cross_v4_correspondence_bias(
            target_metadata=target_meta,
            context=context,
            config=CrossV4CorrespondenceBiasConfig(
                cell_similarity=0.0,
                density_gap=0.0,
                cell_prior=0.0,
            ),
        )

        self.assertEqual(bias.shape, (1, 4, context.encoder_hidden_states.shape[1]))
        self.assertEqual(context.segments.text, (0, 2))
        self.assertEqual(context.segments.global_style, (2, 3))
        self.assertEqual(context.segments.tissue_prior, (3, 11))
        self.assertEqual(context.segments.cell_prior, (11, 17))
        tumor_ref_start, tumor_ref_end = context.segments.reference_local
        self.assertGreater(float(bias[0, 0, tumor_ref_start].item()), 0.0)
        missing_necrosis_prior_index = context.segments.tissue_prior[0] + 3
        covered_tumor_prior_index = context.segments.tissue_prior[0] + 1
        self.assertAlmostEqual(float(bias[0, 1, missing_necrosis_prior_index].item()), 3.0)
        self.assertAlmostEqual(float(bias[0, 0, covered_tumor_prior_index].item()), 0.5)
        self.assertLess(float(bias[0, 1, tumor_ref_end - 1].item()), 0.0)

    def test_cross_v4_zero_reference_mode_preserves_metadata(self):
        encoder = CrossV4ReferenceContextEncoder(
            reference_latent_channels=1,
            tissue_channels=1,
            nuclei_channels=1,
            token_dim=4,
            hidden_dim=4,
        )
        encoding = encoder(
            z_ref=torch.randn(1, 1, 4, 4),
            ref_tissue_feat=torch.randn(1, 1, 4, 4),
            ref_nuclei_feat=torch.randn(1, 1, 4, 4),
            ref_tissue_ids=torch.ones(1, 4, 4, dtype=torch.long),
            ref_nuclei_ids=torch.zeros(1, 4, 4, dtype=torch.long),
        )

        zero = apply_cross_v4_reference_encoding_mode(encoding, "zero-ref")

        self.assertTrue(torch.equal(zero.local_tokens, torch.zeros_like(encoding.local_tokens)))
        self.assertTrue(torch.equal(zero.metadata.tissue_fine_id, encoding.metadata.tissue_fine_id))

    def test_cross_v4_attention_block_index_parser(self):
        self.assertEqual(parse_cross_v4_block_indices("last"), (-1,))
        self.assertEqual(parse_cross_v4_block_indices("1,3"), (1, 3))
        self.assertEqual(parse_cross_v4_block_indices("off"), ())
        self.assertEqual(parse_cross_v4_block_indices("all", total_blocks=3), (0, 1, 2))

    def test_cross_v4_attention_processor_accepts_bias_for_double_block_path(self):
        class FakeAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = 1
                self.to_q = torch.nn.Linear(4, 4, bias=False)
                self.to_k = torch.nn.Linear(4, 4, bias=False)
                self.to_v = torch.nn.Linear(4, 4, bias=False)
                self.add_q_proj = torch.nn.Linear(4, 4, bias=False)
                self.add_k_proj = torch.nn.Linear(4, 4, bias=False)
                self.add_v_proj = torch.nn.Linear(4, 4, bias=False)
                self.to_add_out = torch.nn.Identity()
                self.to_out = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
                self.norm_q = None
                self.norm_k = None
                self.norm_added_q = None
                self.norm_added_k = None
                for module in (
                    self.to_q,
                    self.to_k,
                    self.to_v,
                    self.add_q_proj,
                    self.add_k_proj,
                    self.add_v_proj,
                ):
                    module.weight.data.copy_(torch.eye(4))

        processor = FluxCrossV4AttnProcessor2_0(apply_cross_v4_bias=True)
        attn = FakeAttention()
        hidden_states = torch.randn(1, 3, 4)
        encoder_hidden_states = torch.randn(1, 2, 4)
        bias = torch.zeros(1, 3, 2)
        bias[:, :, 0] = 5.0

        image_out, context_out = processor(
            attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            cross_v4_bias=bias,
        )

        self.assertEqual(image_out.shape, hidden_states.shape)
        self.assertEqual(context_out.shape, encoder_hidden_states.shape)

    @unittest.skipIf(cross_v3_training is None, "diffusers/accelerate are required for cross-v4 training helpers")
    def test_cross_v4_attention_diagnostics_summarize_same_ref_and_prior_buckets(self):
        prompt_embeds = torch.zeros(1, 2, 6)
        text_ids = torch.zeros(2, 3)
        ref_encoder = CrossV4ReferenceContextEncoder(
            reference_latent_channels=1,
            tissue_channels=1,
            nuclei_channels=1,
            token_dim=6,
            hidden_dim=4,
        )
        ref_encoding = ref_encoder(
            z_ref=torch.randn(1, 1, 4, 4),
            ref_tissue_feat=torch.randn(1, 1, 4, 4),
            ref_nuclei_feat=torch.randn(1, 1, 4, 4),
            ref_tissue_ids=torch.tensor(
                [[
                    [1, 1, 2, 2],
                    [1, 1, 2, 2],
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                ]],
                dtype=torch.long,
            ),
            ref_nuclei_ids=torch.zeros(1, 4, 4, dtype=torch.long),
        )
        prior = CrossV4PriorTokenBank(token_dim=6, tissue_prior_tokens_per_class=1)(ref_encoding.local_tokens)
        context = append_cross_v4_context(
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            reference_encoding=ref_encoding,
            prior_tokens=prior,
        )
        target_meta = build_cross_v4_token_metadata(
            tissue_ids=torch.tensor(
                [[
                    [1, 1, 3, 3],
                    [1, 1, 3, 3],
                    [2, 2, 3, 3],
                    [2, 2, 3, 3],
                ]],
                dtype=torch.long,
            ),
            nuclei_ids=torch.zeros(1, 4, 4, dtype=torch.long),
            token_height=2,
            token_width=2,
        )
        bias = build_cross_v4_correspondence_bias(
            target_metadata=target_meta,
            context=context,
            config=CrossV4CorrespondenceBiasConfig(cell_similarity=0.0, density_gap=0.0),
        )

        diagnostics = cross_v3_training._build_cross_v4_attention_diagnostics(
            context=context,
            target_metadata=target_meta,
            correspondence_bias=bias,
        )
        records = diagnostics["records"]
        records.append(
            {
                "cross_v4_attention_covered_ref_same_total": 0.4,
                "cross_v4_attention_covered_ref_all_local": 0.5,
                "cross_v4_attention_covered_ref_mismatch": 0.05,
                "cross_v4_attention_covered_tissue_prior_target": 0.1,
                "cross_v4_attention_missing_tissue_prior_target": 0.4,
                "cross_v4_attention_missing_ref_mismatch": 0.1,
                "cross_v4_attention_missing_tissue_prior_other": 0.01,
            }
        )
        summary = {}
        summary.update(diagnostics["static"])
        summary.update(cross_v3_training._summarize_cross_v4_attention_records(diagnostics))
        verdict, issues = cross_v3_training._cross_v4_diagnostic_verdict(summary)

        self.assertEqual(verdict, "pass")
        self.assertEqual(issues, [])
        self.assertGreater(summary["cross_v4_covered_target_tokens"], 0.0)
        self.assertGreater(summary["cross_v4_missing_target_tokens"], 0.0)


if __name__ == "__main__":
    unittest.main()

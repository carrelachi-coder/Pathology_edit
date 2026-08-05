import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from controlnet_train.pix2pix_transfer.adversarial import (
    RegionAwarePatchDiscriminator,
    discriminator_hinge_loss,
    generator_hinge_loss,
    patch_mask_from_region,
)
from controlnet_train.pix2pix_transfer.dataset import I0ReferenceTextureDataset
from controlnet_train.pix2pix_transfer.losses import (
    EMALossNormalizer,
    Pix2PixTransferLoss,
    boundary_band_mask,
    build_lowtrust_hf_weight,
    contextual_directional_loss,
    high_frequency_residual_loss,
    image_l1_loss,
    regional_contextual_loss,
    regional_gram_loss,
)
from controlnet_train.pix2pix_transfer.identity_adapter import FamilyWSIIdentityAdapter
from controlnet_train.pix2pix_transfer.identity_losses import (
    contrast_normalized_grayscale,
    family_image_descriptor,
    feature_moment_loss,
    grayscale_structure_losses,
    optical_density_moment_loss,
    selected_reference_ranking_loss,
)
from controlnet_train.pix2pix_transfer.ood_diagnose import (
    compute_identity_metrics,
    save_identity_metrics,
    save_ood_panel,
    save_ood_summary_grid,
)
from controlnet_train.pix2pix_transfer.orientation_supervision import (
    windowed_fine_texture_energy_floor_loss,
    windowed_i0_mean_orientation_loss,
)
from controlnet_train.pix2pix_transfer.inference_orientation import (
    build_fine_texture_steering_weights,
)
from controlnet_train.pix2pix_transfer.regional_cross_attention import (
    Pix2PixCrossAttnUNet,
    RegionalCrossAttention,
    build_region_attention_bias_and_strength,
    build_region_attention_mask_and_strength,
)
from controlnet_train.pix2pix_transfer.reference_augmentation import rotate_reference_bundle
from controlnet_train.pix2pix_transfer.trust_gate import (
    build_highres_nuclei_reference_trust_map,
    build_reference_trust_map,
)
from controlnet_train.pix2pix_transfer.training_modes import (
    DistributedWeightedSampler,
    build_cross_wsi_permutation,
    build_difficulty_sampling_weights,
)


class Pix2PixTextureLossTests(unittest.TestCase):
    class TinyFeatureExtractor(nn.Module):
        def forward(self, image):
            layer3 = image.float()
            layer8 = F.avg_pool2d(layer3, 2)
            layer15 = F.avg_pool2d(layer8, 2)
            layer22 = F.avg_pool2d(layer15, 2)
            return {3: layer3, 8: layer8, 15: layer15, 22: layer22}

    @staticmethod
    def stripe_image(
        angle_degrees: float,
        *,
        phase: float = 0.0,
        size: int = 96,
        period: float = 10.0,
    ) -> torch.Tensor:
        coordinates = torch.arange(size, dtype=torch.float32)
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        angle = torch.deg2rad(torch.tensor(float(angle_degrees)))
        normal_coordinate = xx * torch.cos(angle) + yy * torch.sin(angle)
        pattern = torch.sin(2.0 * torch.pi * (normal_coordinate / period + phase))
        return pattern.view(1, 1, size, size).repeat(1, 3, 1, 1)

    def test_windowed_i0_orientation_matches_mean_direction_not_line_phase(self):
        i0 = self.stripe_image(25.0)
        phase_shifted = self.stripe_image(25.0, phase=0.37).requires_grad_(True)
        orthogonal = self.stripe_image(115.0).requires_grad_(True)
        tissue = torch.ones(1, 1, 96, 96, dtype=torch.long)
        nuclei = torch.zeros_like(tissue)
        kwargs = {
            "target_tissue_mask": tissue,
            "target_nuclei_mask": nuclei,
            "window_sizes": (32, 64),
            "window_strides": (16, 32),
            "boundary_exclusion_radius": 0,
            "nuclei_exclusion_radius": 0,
            "min_trust": 0.0,
        }

        matching = windowed_i0_mean_orientation_loss(phase_shifted, i0, **kwargs)
        mismatching = windowed_i0_mean_orientation_loss(orthogonal, i0, **kwargs)

        self.assertLess(float(matching.direction), 0.05)
        self.assertLess(float(matching.mean_angle_degrees), 5.0)
        self.assertGreater(float(mismatching.direction), 0.80)
        self.assertGreater(float(mismatching.mean_angle_degrees), 70.0)
        matching.direction.backward()
        self.assertTrue(torch.isfinite(phase_shifted.grad).all())

    def test_windowed_i0_orientation_excludes_nuclei_neighborhood(self):
        i0 = self.stripe_image(20.0)
        prediction = self.stripe_image(20.0)
        prediction[:, :, 24:72, 24:72] = self.stripe_image(110.0, size=48)
        tissue = torch.ones(1, 1, 96, 96, dtype=torch.long)
        nuclei = torch.zeros_like(tissue)
        nuclei[:, :, 20:76, 20:76] = 1
        common = {
            "target_tissue_mask": tissue,
            "window_sizes": (32,),
            "window_strides": (16,),
            "boundary_exclusion_radius": 0,
            "min_trust": 0.0,
        }

        included = windowed_i0_mean_orientation_loss(
            prediction,
            i0,
            target_nuclei_mask=torch.zeros_like(nuclei),
            nuclei_exclusion_radius=0,
            **common,
        )
        excluded = windowed_i0_mean_orientation_loss(
            prediction,
            i0,
            target_nuclei_mask=nuclei,
            nuclei_exclusion_radius=4,
            **common,
        )

        self.assertGreater(float(included.direction), float(excluded.direction) + 0.05)

    def test_fine_texture_energy_floor_rejects_blur(self):
        baseline = self.stripe_image(22.0)
        matching = baseline.clone().requires_grad_(True)
        blurred = F.avg_pool2d(baseline, kernel_size=9, stride=1, padding=4).requires_grad_(
            True
        )
        tissue = torch.ones(1, 1, 96, 96, dtype=torch.long)
        nuclei = torch.zeros_like(tissue)
        kwargs = {
            "target_tissue_mask": tissue,
            "target_nuclei_mask": nuclei,
            "window_sizes": (32, 64),
            "window_strides": (16, 32),
            "boundary_exclusion_radius": 0,
            "nuclei_exclusion_radius": 0,
            "min_trust": 0.0,
            "energy_floor_ratio": 0.95,
        }

        sharp_result = windowed_fine_texture_energy_floor_loss(
            matching, baseline, **kwargs
        )
        blurred_result = windowed_fine_texture_energy_floor_loss(
            blurred, baseline, **kwargs
        )

        self.assertLess(float(sharp_result.loss), 1.0e-5)
        self.assertAlmostEqual(float(sharp_result.mean_energy_ratio), 1.0, places=4)
        self.assertGreater(float(blurred_result.loss), 0.20)
        self.assertLess(float(blurred_result.mean_energy_ratio), 0.75)
        blurred_result.loss.backward()
        self.assertTrue(torch.isfinite(blurred.grad).all())

    def test_fine_texture_steering_selects_matching_rotated_reference(self):
        reference = self.stripe_image(0.0, size=96)
        tissue = torch.ones(1, 1, 96, 96, dtype=torch.long)
        nuclei = torch.zeros_like(tissue)
        reference_cond = torch.cat([reference, tissue.float()], dim=1)
        rotated = rotate_reference_bundle(
            reference_cond,
            tissue,
            tissue,
            nuclei,
            angles_degrees=45.0,
        )
        result = build_fine_texture_steering_weights(
            rotated.reference_cond[:, :3],
            reference,
            target_tissue_mask=tissue,
            target_nuclei_mask=nuclei,
            reference_tissue_mask=tissue,
            reference_nuclei_mask=nuclei,
            candidate_angles_degrees=(0.0, 45.0, 90.0, 135.0),
            smoothing_sigma=5.0,
            boundary_exclusion_radius=0,
            nuclei_exclusion_radius=0,
            temperature=0.05,
        )

        fractions = result.candidate_fractions
        self.assertEqual(tuple(result.weights.shape), (1, 4, 96, 96))
        self.assertGreater(float(fractions[1]), float(fractions[0]))
        self.assertGreater(float(fractions[1]), 0.50)
        self.assertGreater(float(result.active_fraction), 0.50)

    def test_local_histogram_steering_preserves_opposing_reference_modes(self):
        reference = self.stripe_image(0.0, size=96)
        reference[:, :, :, 48:] = self.stripe_image(90.0, size=96)[:, :, :, 48:]
        target = self.stripe_image(45.0, size=96)
        tissue = torch.ones(1, 1, 96, 96, dtype=torch.long)
        nuclei = torch.zeros_like(tissue)
        kwargs = {
            "target_tissue_mask": tissue,
            "target_nuclei_mask": nuclei,
            "reference_tissue_mask": tissue,
            "reference_nuclei_mask": nuclei,
            "candidate_angles_degrees": (0.0, 45.0, 90.0, 135.0),
            "smoothing_sigma": 5.0,
            "boundary_exclusion_radius": 0,
            "nuclei_exclusion_radius": 0,
            "minimum_strength": 0.70,
            "minimum_support": 0.05,
            "temperature": 0.05,
        }

        global_result = build_fine_texture_steering_weights(
            target,
            reference,
            reference_direction_mode="global_mean",
            **kwargs,
        )
        local_result = build_fine_texture_steering_weights(
            target,
            reference,
            reference_direction_mode="local_histogram",
            local_histogram_bins=36,
            local_histogram_concentration=8.0,
            **kwargs,
        )

        global_nonzero = 1.0 - float(global_result.candidate_fractions[0])
        local_nonzero = 1.0 - float(local_result.candidate_fractions[0])
        self.assertGreater(local_nonzero, global_nonzero + 0.25)
        self.assertGreater(local_nonzero, 0.60)
        self.assertGreater(float(local_result.mean_confidence), 0.60)

    def test_multi_reference_attention_one_hot_matches_single_reference(self):
        torch.manual_seed(17)
        module = RegionalCrossAttention(dim=4, num_heads=1, use_region_mask=False)
        module.gamma.data.fill_(1.0)
        query = torch.randn(1, 4, 4, 4)
        first = torch.randn(1, 4, 4, 4)
        second = torch.randn(1, 4, 4, 4)
        first_weights = torch.zeros(1, 2, 4, 4)
        first_weights[:, 0] = 1.0
        second_weights = 1.0 - first_weights

        expected_first = module(query, first)
        expected_second = module(query, second)
        actual_first = module.forward_multi_reference(
            query,
            (first, second),
            reference_weights=first_weights,
        )
        actual_second = module.forward_multi_reference(
            query,
            (first, second),
            reference_weights=second_weights,
        )

        self.assertTrue(torch.allclose(actual_first, expected_first, atol=1.0e-6))
        self.assertTrue(torch.allclose(actual_second, expected_second, atol=1.0e-6))

    def test_regional_gram_is_invariant_to_spatial_permutation(self):
        prediction = torch.tensor(
            [[[[0.0, 1.0], [2.0, 3.0]], [[1.0, 0.0], [3.0, 2.0]]]],
            requires_grad=True,
        )
        reference = prediction.detach().flip(-1)
        region = torch.ones(1, 1, 2, 2, dtype=torch.long)

        loss, count = regional_gram_loss(
            prediction,
            reference,
            region,
            region,
            min_pixels=2,
        )

        self.assertEqual(count, 1)
        self.assertLess(loss.item(), 1e-6)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_low_frequency_l1_deemphasizes_checkerboard_misalignment(self):
        checkerboard = torch.tensor(
            [[[[1.0, -1.0] * 8, [-1.0, 1.0] * 8] * 8]],
            dtype=torch.float32,
        ).reshape(1, 1, 16, 16)
        shifted = torch.roll(checkerboard, shifts=1, dims=-1)

        pixel_loss = image_l1_loss(checkerboard, shifted, blur_sigma=0.0)
        low_frequency_loss = image_l1_loss(checkerboard, shifted, blur_sigma=2.0)

        self.assertLess(low_frequency_loss.item(), pixel_loss.item() * 0.1)

    def test_regional_gram_detects_different_texture_covariance(self):
        prediction = torch.tensor(
            [[[[0.0, 1.0], [2.0, 3.0]], [[0.0, 1.0], [2.0, 3.0]]]],
            requires_grad=True,
        )
        reference = torch.tensor(
            [[[[0.0, 1.0], [2.0, 3.0]], [[3.0, 2.0], [1.0, 0.0]]]]
        )
        region = torch.ones(1, 1, 2, 2, dtype=torch.long)

        loss, count = regional_gram_loss(
            prediction,
            reference,
            region,
            region,
            min_pixels=2,
        )

        self.assertEqual(count, 1)
        self.assertGreater(loss.item(), 0.1)

    def test_contextual_loss_prefers_permuted_matching_features(self):
        query = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            requires_grad=True,
        )
        matching = query.detach()[torch.tensor([2, 0, 1])]
        mismatching = torch.tensor(
            [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
        )

        matching_loss = contextual_directional_loss(query, matching)
        mismatching_loss = contextual_directional_loss(query, mismatching)

        self.assertLess(matching_loss.item(), mismatching_loss.item())
        matching_loss.backward()
        self.assertTrue(torch.isfinite(query.grad).all())

    def test_regional_contextual_matches_only_shared_non_background_regions(self):
        prediction = torch.randn(1, 4, 4, 4, requires_grad=True)
        reference = torch.randn(1, 4, 4, 4)
        target_region = torch.tensor(
            [[[[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]]]]
        )
        reference_region = target_region.clone()

        loss, count = regional_contextual_loss(
            prediction,
            reference,
            target_region,
            reference_region,
            min_pixels=4,
            max_samples=8,
        )

        self.assertEqual(count, 2)
        self.assertGreaterEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_ema_normalizer_equalizes_initial_loss_magnitudes_and_restores(self):
        normalizer = EMALossNormalizer(decay=0.9, calibration_steps=1)
        losses = {
            "l1": torch.tensor(0.2, requires_grad=True),
            "content": torch.tensor(2.0, requires_grad=True),
            "gram": torch.tensor(0.02, requires_grad=True),
            "contextual": torch.tensor(0.5, requires_grad=True),
        }

        normalized, _ = normalizer(losses)

        for value in normalized.values():
            self.assertTrue(torch.allclose(value.detach(), torch.tensor(1.0)))
        state = normalizer.state_dict()
        frozen_ema = normalizer.ema.clone()
        normalizer({name: value * 10.0 for name, value in losses.items()})
        self.assertTrue(torch.allclose(normalizer.ema, frozen_ema))
        restored = EMALossNormalizer(decay=0.9, calibration_steps=1)
        restored.load_state_dict(state)
        self.assertTrue(torch.equal(restored.initialized, torch.ones(4, dtype=torch.bool)))
        self.assertTrue(torch.allclose(restored.ema, normalizer.ema))

    def test_default_model_enables_quarter_scale_cross_attention(self):
        model = Pix2PixCrossAttnUNet(in_ch=25, base=8, num_heads=4)

        self.assertEqual(model.cross_attn_scales, ("1/4", "1/8", "1/16"))
        self.assertIsNotNone(model.cross_4)

    def test_model_uses_resize_conv_upsampling(self):
        model = Pix2PixCrossAttnUNet(in_ch=25, base=8, num_heads=4)

        self.assertEqual(model.upsample_mode, "bilinear")
        self.assertFalse(any(isinstance(module, nn.ConvTranspose2d) for module in model.modules()))

        nearest_model = Pix2PixCrossAttnUNet(
            in_ch=25,
            base=8,
            num_heads=4,
            upsample_mode="nearest",
        )
        target_cond = torch.randn(1, 25, 32, 32)
        reference_cond = torch.randn(1, 25, 32, 32)
        region = torch.ones(1, 1, 32, 32, dtype=torch.long)

        out = nearest_model(
            target_cond,
            reference_cond,
            target_region=region,
            reference_region=region,
        )

        self.assertEqual(nearest_model.upsample_mode, "nearest")
        self.assertEqual(tuple(out.shape), (1, 3, 32, 32))

    def test_full_pyramid_steering_routes_rotated_reference_at_every_scale(self):
        torch.manual_seed(23)
        model = Pix2PixCrossAttnUNet(
            in_ch=25,
            base=8,
            num_heads=4,
            full_pyramid_texture_steering=True,
            steering_highres_reference_size=4,
        )
        target_cond = torch.randn(1, 25, 32, 32)
        reference_cond = torch.randn(1, 25, 32, 32)
        region = torch.ones(1, 1, 32, 32, dtype=torch.long)
        tissue = torch.ones_like(region)
        nuclei = torch.zeros_like(region)
        weights = torch.zeros(1, 2, 32, 32)
        weights[:, 1] = 1.0

        output = model(
            target_cond,
            reference_cond,
            target_region=region,
            reference_region=region,
            target_tissue_mask=tissue,
            target_nuclei_mask=nuclei,
            reference_tissue_mask=tissue,
            reference_nuclei_mask=nuclei,
            cross4_rotation_weights=weights,
            cross4_rotation_angles=(0.0, 90.0),
            texture_steering_scales=("1/1", "1/2", "1/4", "1/8", "1/16"),
        )
        output.mean().backward()

        self.assertEqual(tuple(output.shape), (1, 3, 32, 32))
        self.assertIsNotNone(model.steering_cross_1)
        self.assertIsNotNone(model.steering_cross_2)
        self.assertGreater(float(model.steering_cross_1.proj.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.steering_cross_2.proj.weight.grad.abs().sum()), 0.0)

    def test_full_pyramid_uses_nuclei_guard_only_for_highres_steering(self):
        class RecordingAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.query_trust_map = None

            def forward_multi_reference(self, query_feat, reference_features, **kwargs):
                del reference_features
                self.query_trust_map = kwargs["query_trust_map"].detach().clone()
                return query_feat

        model = Pix2PixCrossAttnUNet(
            in_ch=25,
            base=8,
            num_heads=4,
            full_pyramid_texture_steering=True,
            steering_highres_reference_size=4,
        )
        cross1 = RecordingAttention()
        cross2 = RecordingAttention()
        model.steering_cross_1 = cross1
        model.steering_cross_2 = cross2
        target_cond = torch.randn(1, 25, 32, 32)
        reference_cond = torch.randn(1, 25, 32, 32)
        region = torch.ones(1, 1, 32, 32, dtype=torch.long)
        target_trust = torch.full((1, 1, 32, 32), 0.8)
        nuclei_guard = torch.full((1, 1, 32, 32), 0.4)
        weights = torch.ones(1, 1, 32, 32)

        output = model(
            target_cond,
            reference_cond,
            target_region=region,
            reference_region=region,
            target_trust_map=target_trust,
            highres_nuclei_trust_map=nuclei_guard,
            target_tissue_mask=region,
            target_nuclei_mask=torch.zeros_like(region),
            reference_tissue_mask=region,
            reference_nuclei_mask=torch.zeros_like(region),
            cross4_rotation_weights=weights,
            cross4_rotation_angles=(0.0,),
            texture_steering_scales=("1/1", "1/2"),
        )

        self.assertEqual(tuple(output.shape), (1, 3, 32, 32))
        self.assertTrue(torch.allclose(cross1.query_trust_map, nuclei_guard))
        self.assertTrue(torch.allclose(cross2.query_trust_map, nuclei_guard))

    def test_region_aware_patchgan_outputs_patch_logits(self):
        discriminator = RegionAwarePatchDiscriminator(
            condition_channels=22,
            base_channels=8,
            max_channels=32,
            num_layers=2,
            spectral_norm=False,
        )
        image = torch.randn(2, 3, 64, 64)
        region_condition = torch.randn(2, 22, 64, 64)

        logits = discriminator(image, region_condition)

        self.assertEqual(logits.shape[0], 2)
        self.assertEqual(logits.shape[1], 1)
        self.assertLess(logits.shape[-1], image.shape[-1])

    def test_masked_hinge_gan_loss_has_image_gradient(self):
        discriminator = RegionAwarePatchDiscriminator(
            condition_channels=22,
            base_channels=8,
            max_channels=32,
            num_layers=2,
            spectral_norm=False,
        )
        real = torch.randn(2, 3, 64, 64)
        fake = torch.randn(2, 3, 64, 64, requires_grad=True)
        region_condition = torch.randn(2, 22, 64, 64)
        target_region = torch.ones(2, 1, 64, 64, dtype=torch.long)
        target_region[:, :, :8, :8] = 0

        real_logits = discriminator(real, region_condition)
        fake_logits = discriminator(fake.detach(), region_condition)
        mask = patch_mask_from_region(target_region, real_logits)
        d_loss = discriminator_hinge_loss(real_logits, fake_logits, mask=mask)

        g_logits = discriminator(fake, region_condition)
        g_loss = generator_hinge_loss(g_logits, mask=mask)
        g_loss.backward()

        self.assertTrue(torch.isfinite(d_loss))
        self.assertTrue(torch.isfinite(g_loss))
        self.assertIsNotNone(fake.grad)
        self.assertTrue(torch.isfinite(fake.grad).all())

    def test_combined_loss_has_finite_end_to_end_gradient(self):
        criterion = Pix2PixTransferLoss(
            feature_extractor=self.TinyFeatureExtractor(),
            contextual_max_samples=32,
            texture_min_pixels=2,
        )
        prediction = torch.randn(2, 3, 32, 32, requires_grad=True).tanh()
        prediction.retain_grad()
        target = torch.randn(2, 3, 32, 32).clamp(-1.0, 1.0)
        reference = torch.randn(2, 3, 32, 32).clamp(-1.0, 1.0)
        target_region = torch.ones(2, 1, 32, 32, dtype=torch.long)
        reference_region = torch.ones(2, 1, 32, 32, dtype=torch.long)

        loss, logs = criterion(
            prediction,
            target,
            reference=reference,
            target_region=target_region,
            reference_region=reference_region,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(logs["l1"], 0.0)
        self.assertGreater(logs["perc"], 0.0)
        self.assertGreater(logs["gram"], 0.0)
        self.assertGreaterEqual(logs["contextual"], 0.0)

    def test_unmatched_region_attention_uses_weak_fallback_not_full_strength(self):
        query_labels = torch.tensor([[1, 257, 9, 300]])
        key_labels = torch.tensor([[1, 257, 2]])

        allow, strength = build_region_attention_mask_and_strength(
            query_labels,
            key_labels,
            fallback_scale=0.05,
        )

        self.assertTrue(torch.equal(allow[0, 0], torch.tensor([True, False, False])))
        self.assertTrue(torch.equal(allow[0, 1], torch.tensor([False, True, False])))
        self.assertTrue(torch.equal(allow[0, 2], torch.tensor([True, False, True])))
        self.assertTrue(torch.equal(allow[0, 3], torch.tensor([False, True, False])))
        self.assertAlmostEqual(float(strength[0, 0]), 1.0)
        self.assertAlmostEqual(float(strength[0, 1]), 1.0)
        self.assertAlmostEqual(float(strength[0, 2]), 0.05, places=5)
        self.assertAlmostEqual(float(strength[0, 3]), 0.05, places=5)

    def test_boundary_soft_bias_allows_weak_adjacent_microenvironment(self):
        query_grid = torch.tensor([[[1, 2]]])
        query_labels = query_grid.flatten(1)
        key_labels = torch.tensor([[1, 2, 3]])

        bias, strength = build_region_attention_bias_and_strength(
            query_labels,
            key_labels,
            query_label_grid=query_grid,
            fallback_scale=0.05,
            soft_context_scale=0.20,
            nuclei_context_scale=0.04,
            soft_context_radius=1,
        )

        self.assertEqual(tuple(bias.shape), (1, 2, 3))
        self.assertAlmostEqual(float(bias[0, 0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(bias[0, 0, 1]), torch.log(torch.tensor(0.20)).item(), places=5)
        self.assertLess(float(bias[0, 0, 2]), -1e6)
        self.assertAlmostEqual(float(strength[0, 0]), 1.0, places=5)

    def test_soft_context_keeps_tissue_and_nuclei_families_separate(self):
        query_grid = torch.tensor([[[257, 258, 1, 2]]])
        query_labels = query_grid.flatten(1)
        key_labels = torch.tensor([[257, 258, 1, 2]])

        bias, _ = build_region_attention_bias_and_strength(
            query_labels,
            key_labels,
            query_label_grid=query_grid,
            fallback_scale=0.05,
            soft_context_scale=0.20,
            nuclei_context_scale=0.0,
            soft_context_radius=1,
        )

        self.assertAlmostEqual(float(bias[0, 0, 0]), 0.0, places=5)
        self.assertLess(float(bias[0, 0, 1]), -1e6)
        self.assertLess(float(bias[0, 1, 2]), -1e6)
        self.assertLess(float(bias[0, 2, 1]), -1e6)
        self.assertAlmostEqual(float(bias[0, 2, 3]), torch.log(torch.tensor(0.20)).item(), places=5)

    def test_production_training_script_pins_final_full_pyramid_recipe(self):
        script = Path("scripts/train_pix2pix_postprocess.sh").read_text()

        self.assertIn("pix2pix_texture_transfer_lazy_ver4_wsi_identity_i0_local_full_pyramid_v3_ft", script)
        self.assertIn("pilot_step002000.pt", script)
        self.assertIn('EPOCHS="${EPOCHS:-27}"', script)
        self.assertIn('MAX_CONTINUATION_STEPS="${MAX_CONTINUATION_STEPS:-1000}"', script)
        self.assertIn('MASTER_PORT="${MASTER_PORT:-', script)
        self.assertIn('--master_port="${MASTER_PORT}"', script)
        self.assertIn('CROSS4_STEERING_REFERENCE_MODE="${CROSS4_STEERING_REFERENCE_MODE:-local_histogram}"', script)
        self.assertIn('CROSS4_STEERING_SCALES="${CROSS4_STEERING_SCALES:-1/1,1/2,1/4,1/8,1/16}"', script)
        self.assertIn("--wsi-identity-adapter", script)
        self.assertIn("--cross-wsi-style-prob 0.30", script)
        self.assertIn("--matched-tissue-trust-floor 0.65", script)
        self.assertIn("--matched-nuclei-trust-floor 0.70", script)
        self.assertIn("--lambda-lowtrust-hf 0.5", script)
        self.assertIn("--ref-nuclei-context-scale 0.00", script)

    def test_reference_trust_keeps_nuclei_specific_but_penalizes_density_mismatch(self):
        target_region = torch.ones(1, 1, 8, 8, dtype=torch.long)
        reference_region = torch.ones(1, 1, 8, 8, dtype=torch.long)
        target_region[:, :, :4, :] = 257
        reference_region[:, :, :1, :] = 257

        trust, stats = build_reference_trust_map(
            target_region,
            reference_region,
            fallback_scale=0.05,
            min_region_pixels=2,
        )

        nuclei_trust = trust[target_region == 257].mean()
        tissue_trust = trust[target_region == 1].mean()
        self.assertLess(float(nuclei_trust), float(tissue_trust))
        self.assertGreaterEqual(float(nuclei_trust), 0.05)
        self.assertGreater(stats["low_trust_fraction"], 0.0)
        self.assertEqual(stats["unmatched_regions"], 0.0)

    def test_reference_trust_applies_matched_family_floors(self):
        target_region = torch.ones(1, 1, 8, 8, dtype=torch.long)
        reference_region = torch.ones(1, 1, 8, 8, dtype=torch.long)
        target_region[:, :, :4, :] = 257
        reference_region[:, :, :1, :] = 257

        trust, _ = build_reference_trust_map(
            target_region,
            reference_region,
            fallback_scale=0.05,
            min_region_pixels=2,
            matched_tissue_floor=0.65,
            matched_nuclei_floor=0.70,
        )

        self.assertAlmostEqual(float(trust[target_region == 1].min()), 0.65, places=5)
        self.assertAlmostEqual(float(trust[target_region == 257].min()), 0.70, places=5)

    def test_highres_nuclei_trust_preserves_tissue_and_attenuates_sparse_classes(self):
        target_nuclei = torch.zeros(1, 1, 8, 8, dtype=torch.long)
        reference_nuclei = torch.zeros_like(target_nuclei)
        target_nuclei[:, :, :2, :] = 1
        target_nuclei[:, :, 6:, :] = 2
        reference_nuclei[:, :, 0, 0] = 1
        weights = torch.ones(1, 1, 8, 8)

        trust, stats = build_highres_nuclei_reference_trust_map(
            target_nuclei,
            reference_nuclei,
            reference_weights=weights,
            candidate_angles_degrees=(0.0,),
            reference_pool_size=2,
            unmatched_scale=0.20,
            matched_floor=0.60,
            sufficient_reference_tokens=2,
            min_reference_pixels=1,
        )

        self.assertTrue(torch.allclose(trust[target_nuclei == 0], torch.ones(32)))
        self.assertAlmostEqual(float(trust[target_nuclei == 1].mean()), 0.60, places=5)
        self.assertAlmostEqual(float(trust[target_nuclei == 2].mean()), 0.20, places=5)
        self.assertEqual(stats["class_count"], 2.0)
        self.assertEqual(stats["missing_effective_classes"], 1.0)

    def test_identity_adapter_zero_init_preserves_features(self):
        adapter = FamilyWSIIdentityAdapter(
            channels_by_scale={"1/4": 4},
            tissue_scales=("1/4",),
            nuclei_scales=("1/4",),
            gamma_max=0.30,
            gamma_init=0.10,
            min_tissue_pixels=4,
            min_nuclei_pixels=2,
        )
        target = torch.randn(1, 4, 4, 4)
        reference = torch.randn(1, 4, 4, 4)
        tissue = torch.ones(1, 1, 8, 8, dtype=torch.long)
        nuclei = torch.zeros_like(tissue)
        nuclei[:, :, :2, :2] = 1

        output, logs = adapter.forward_scale(
            "1/4",
            target,
            reference,
            target_tissue_mask=tissue,
            target_nuclei_mask=nuclei,
            reference_tissue_mask=tissue,
            reference_nuclei_mask=nuclei,
        )

        self.assertTrue(torch.equal(output, target))
        self.assertAlmostEqual(logs["tissue_support"], 1.0)
        self.assertAlmostEqual(logs["nuclei_support"], 1.0)

    def test_identity_adapter_keeps_tissue_and_nuclei_modulation_separate(self):
        adapter = FamilyWSIIdentityAdapter(
            channels_by_scale={"1/4": 2},
            tissue_scales=("1/4",),
            nuclei_scales=("1/4",),
            gamma_max=0.30,
            gamma_init=0.10,
            min_tissue_pixels=4,
            min_nuclei_pixels=2,
        )
        tissue_film = adapter.tissue_adapters["1/4"]
        with torch.no_grad():
            tissue_film.output.bias[tissue_film.channels :].fill_(1.0)
            tissue_film.identity_gamma.fill_(10.0)
        target = torch.zeros(1, 2, 4, 4)
        reference = torch.randn(1, 2, 4, 4)
        tissue = torch.ones(1, 1, 8, 8, dtype=torch.long)
        nuclei = torch.zeros_like(tissue)
        nuclei[:, :, :4, :4] = 1

        output, _ = adapter.forward_scale(
            "1/4",
            target,
            reference,
            target_tissue_mask=tissue,
            target_nuclei_mask=nuclei,
            reference_tissue_mask=tissue,
            reference_nuclei_mask=nuclei,
        )

        resized_nuclei = F.interpolate(nuclei.float(), size=(4, 4), mode="nearest").bool()
        self.assertTrue(torch.equal(output.masked_select(resized_nuclei.expand_as(output)), torch.zeros(8)))
        self.assertGreater(float(output.masked_select(~resized_nuclei.expand_as(output)).mean()), 0.0)
        self.assertAlmostEqual(abs(tissue_film.effective_gamma()), 0.30, places=5)

    def test_identity_adapter_uses_global_same_wsi_tissue_family(self):
        class RecordingFiLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.reference_masks = []

            def forward(
                self,
                target_feature,
                reference_feature,
                *,
                target_mask,
                reference_mask,
                min_pixels,
                target_gain_map=None,
            ):
                self.reference_masks.append(reference_mask.detach().clone())
                support = reference_mask.flatten(1).any(dim=1).to(target_feature.dtype)
                return target_feature, support[:, None]

        adapter = FamilyWSIIdentityAdapter(
            channels_by_scale={"1/4": 2},
            tissue_scales=("1/4",),
            nuclei_scales=(),
            min_tissue_pixels=1,
        )
        recorder = RecordingFiLM()
        adapter.tissue_adapters["1/4"] = recorder
        tissue = torch.ones(1, 1, 8, 8, dtype=torch.long)
        tissue[:, :, :, 4:] = 2
        nuclei = torch.zeros_like(tissue)

        _, logs = adapter.forward_scale(
            "1/4",
            torch.zeros(1, 2, 4, 4),
            torch.zeros(1, 2, 4, 4),
            target_tissue_mask=tissue,
            target_nuclei_mask=nuclei,
            reference_tissue_mask=tissue,
            reference_nuclei_mask=nuclei,
        )

        self.assertEqual(len(recorder.reference_masks), 1)
        self.assertEqual(int(recorder.reference_masks[0].sum().item()), 64)
        self.assertEqual(logs["tissue_support"], 1.0)

    def test_identity_adapter_continuous_gain_preserves_supported_region(self):
        adapter = FamilyWSIIdentityAdapter(
            channels_by_scale={"1/4": 2},
            tissue_scales=("1/4",),
            nuclei_scales=(),
            min_tissue_pixels=1,
        )
        film = adapter.tissue_adapters["1/4"]
        with torch.no_grad():
            film.output.bias[film.channels :].fill_(1.0)
            film.identity_gamma.fill_(0.2)
        target = torch.zeros(1, 2, 4, 4)
        reference = torch.ones_like(target)
        tissue = torch.ones(1, 1, 8, 8, dtype=torch.long)
        nuclei = torch.zeros_like(tissue)

        baseline, _ = adapter.forward_scale(
            "1/4",
            target,
            reference,
            target_tissue_mask=tissue,
            target_nuclei_mask=nuclei,
            reference_tissue_mask=tissue,
            reference_nuclei_mask=nuclei,
        )
        gain = torch.ones_like(tissue, dtype=torch.float32)
        gain[:, :, :, 4:] = 1.5
        adaptive, _ = adapter.forward_scale(
            "1/4",
            target,
            reference,
            target_tissue_mask=tissue,
            target_nuclei_mask=nuclei,
            reference_tissue_mask=tissue,
            reference_nuclei_mask=nuclei,
            tissue_gain_map=gain,
        )

        torch.testing.assert_close(adaptive[:, :, :, :2], baseline[:, :, :, :2])
        self.assertGreater(
            float(adaptive[:, :, :, 2:].mean()),
            float(baseline[:, :, :, 2:].mean()),
        )

    def test_identity_model_loads_old_weights_and_zero_init_matches_old_output(self):
        torch.manual_seed(7)
        old_model = Pix2PixCrossAttnUNet(in_ch=25, base=8, num_heads=4)
        identity_model = Pix2PixCrossAttnUNet(
            in_ch=25,
            base=8,
            num_heads=4,
            use_wsi_identity=True,
            identity_min_tissue_pixels=4,
            identity_min_nuclei_pixels=2,
        )
        incompatible = identity_model.load_state_dict(old_model.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(all(key.startswith("identity_adapter.") for key in incompatible.missing_keys))

        target_cond = torch.randn(1, 25, 32, 32)
        reference_cond = torch.randn(1, 25, 32, 32)
        region = torch.ones(1, 1, 32, 32, dtype=torch.long)
        tissue = torch.ones_like(region)
        nuclei = torch.zeros_like(region)
        nuclei[:, :, :8, :8] = 1
        old_output = old_model(
            target_cond,
            reference_cond,
            target_region=region,
            reference_region=region,
        )
        identity_output = identity_model(
            target_cond,
            reference_cond,
            target_region=region,
            reference_region=region,
            target_tissue_mask=tissue,
            target_nuclei_mask=nuclei,
            reference_tissue_mask=tissue,
            reference_nuclei_mask=nuclei,
        )

        self.assertTrue(torch.equal(identity_output, old_output))
        self.assertEqual(set(identity_model.identity_adapter.tissue_adapters), {"1/4", "1/8", "1/16"})
        self.assertEqual(set(identity_model.identity_adapter.nuclei_adapters), {"1/4"})

    def test_lowtrust_hf_weight_excludes_all_nuclei_pixels(self):
        trust = torch.full((1, 1, 4, 4), 0.2)
        nuclei = torch.zeros(1, 1, 4, 4, dtype=torch.long)
        nuclei[:, :, :2, :2] = 3

        weight = build_lowtrust_hf_weight(trust, target_nuclei_mask=nuclei)

        self.assertTrue(torch.equal(weight[:, :, :2, :2], torch.zeros(1, 1, 2, 2)))
        self.assertTrue(torch.allclose(weight[:, :, 2:, 2:], torch.full((1, 1, 2, 2), 0.8)))

    def test_transfer_loss_uses_tissue_only_boundary_and_exempts_nuclei_hf(self):
        criterion = Pix2PixTransferLoss(
            feature_extractor=self.TinyFeatureExtractor(),
            contextual_max_samples=16,
            texture_min_pixels=2,
            boundary_feather_radius=1,
            lambda_boundary_hf=2.0,
            lambda_lowtrust_hf=1.0,
        )
        i0 = torch.zeros(1, 3, 16, 16)
        prediction = i0.clone()
        prediction[:, :, :8, :8:2] = 1.0
        prediction[:, :, :8, 1:8:2] = -1.0
        prediction.requires_grad_(True)
        target = prediction.detach().clone()
        reference = prediction.detach().clone()
        target_region = torch.ones(1, 1, 16, 16, dtype=torch.long)
        target_region[:, :, :8, :8] = 257
        tissue_region = torch.ones_like(target_region)
        nuclei = torch.zeros_like(target_region)
        nuclei[:, :, :8, :8] = 1
        trust = torch.full((1, 1, 16, 16), 0.2)

        loss, logs = criterion(
            prediction,
            target,
            reference=reference,
            target_region=target_region,
            reference_region=target_region,
            boundary_region=tissue_region,
            target_nuclei_mask=nuclei,
            i0=i0,
            trust_map=trust,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertLess(logs["boundary_hf"], 1e-6)
        self.assertLess(logs["lowtrust_hf"], 1e-6)

    def test_optical_density_and_feature_moments_follow_reference_appearance(self):
        mask = torch.ones(1, 1, 8, 8)
        prediction = torch.zeros(1, 3, 8, 8, requires_grad=True)
        matching = torch.zeros_like(prediction)
        shifted = torch.full_like(prediction, 0.6)

        matching_od = optical_density_moment_loss(prediction, matching, mask, mask)
        shifted_od = optical_density_moment_loss(prediction, shifted, mask, mask)
        matching_feature = feature_moment_loss(prediction, matching, mask, mask)
        shifted_feature = feature_moment_loss(prediction, shifted, mask, mask)

        self.assertLess(float(matching_od), 1e-6)
        self.assertGreater(float(shifted_od), 0.01)
        self.assertLess(float(matching_feature), 1e-6)
        self.assertGreater(float(shifted_feature), 0.01)
        (shifted_od + shifted_feature).backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_family_descriptor_detects_high_frequency_texture_without_layout_copy(self):
        mask = torch.ones(1, 1, 16, 16)
        smooth = torch.zeros(1, 3, 16, 16)
        checker = smooth.clone()
        checker[:, :, :, ::2] = 1.0
        checker[:, :, :, 1::2] = -1.0

        smooth_descriptor = family_image_descriptor(smooth, mask)
        checker_descriptor = family_image_descriptor(checker, mask)

        self.assertEqual(smooth_descriptor.shape, checker_descriptor.shape)
        self.assertGreater(float((checker_descriptor - smooth_descriptor).abs().mean()), 0.01)

    def test_selected_reference_ranking_rejects_dataset_average_style(self):
        output = torch.tensor([[0.1, 0.2, 0.3]])
        selected = torch.tensor([[0.1, 0.2, 0.3]])
        negative = torch.tensor([[0.8, 0.7, 0.6]])

        good = selected_reference_ranking_loss(output, selected, negative, margin=0.10)
        bad = selected_reference_ranking_loss(negative, selected, output, margin=0.10)

        self.assertLess(float(good), 1e-6)
        self.assertGreater(float(bad), 0.1)

    def test_cross_wsi_structure_loss_is_contrast_and_stain_invariant(self):
        base = torch.linspace(-0.8, 0.8, steps=16).view(1, 1, 4, 4).repeat(1, 3, 1, 1)
        recolored = (base * 0.45 + 0.35).clamp(-1.0, 1.0)

        base_gray = contrast_normalized_grayscale(base)
        recolored_gray = contrast_normalized_grayscale(recolored)
        gray_loss, edge_loss = grayscale_structure_losses(base, recolored)

        self.assertTrue(torch.allclose(base_gray, recolored_gray, atol=1e-4))
        self.assertLess(float(gray_loss), 1e-4)
        self.assertLess(float(edge_loss), 1e-4)

    def test_cross_wsi_permutation_never_pairs_same_case(self):
        case_ids = ["a", "a", "b", "c"]

        permutation = build_cross_wsi_permutation(case_ids)

        self.assertIsNotNone(permutation)
        self.assertEqual(sorted(permutation), list(range(len(case_ids))))
        self.assertTrue(all(case_ids[index] != case_ids[source] for index, source in enumerate(permutation)))
        self.assertIsNone(build_cross_wsi_permutation(["same", "same", "same"]))

    def test_difficulty_weights_have_expected_full_to_hard_mass(self):
        records = [
            {"pair_difficulty": "full"},
            {"pair_difficulty": "full"},
            {"pair_difficulty": "full"},
            {"pair_difficulty": "partial"},
            {"pair_difficulty": "low"},
        ]

        weights = build_difficulty_sampling_weights(records, full_mass=0.40, hard_mass=0.30)
        full_total = float(weights[:3].sum())
        hard_total = float(weights[3:].sum())

        self.assertAlmostEqual(full_total / hard_total, 4.0 / 3.0, places=5)

    def test_distributed_weighted_sampler_is_deterministic_and_sharded(self):
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.double)
        rank0 = DistributedWeightedSampler(weights, num_replicas=2, rank=0, seed=17)
        rank1 = DistributedWeightedSampler(weights, num_replicas=2, rank=1, seed=17)

        first0 = list(iter(rank0))
        first1 = list(iter(rank1))
        rank0.set_epoch(1)
        second0 = list(iter(rank0))

        self.assertEqual(len(first0), 2)
        self.assertEqual(len(first1), 2)
        self.assertNotEqual(first0, second0)

    def test_dataset_returns_case_and_pair_difficulty_for_training_modes(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ("i0.png", "target.png", "reference.png"):
                Image.new("RGB", (8, 8), (180, 120, 160)).save(root / name)
            for name in ("target_tissue.png", "reference_tissue.png"):
                Image.new("L", (8, 8), 1).save(root / name)
            for name in ("target_nuclei.png", "reference_nuclei.png"):
                Image.new("L", (8, 8), 0).save(root / name)
            record = {
                "sample_id": "target",
                "reference_sample_id": "reference",
                "case_id": "WSI-123",
                "pair_difficulty": "partial",
                "i0_image": "i0.png",
                "target_image": "target.png",
                "reference_image": "reference.png",
                "target_tissue_mask": "target_tissue.png",
                "reference_tissue_mask": "reference_tissue.png",
                "target_nuclei_mask": "target_nuclei.png",
                "reference_nuclei_mask": "reference_nuclei.png",
            }
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({"pairs": [record]}), encoding="utf8")
            dataset = I0ReferenceTextureDataset(metadata, image_size=8)

            sample = dataset[0]

            self.assertEqual(sample["case_id"], "WSI-123")
            self.assertEqual(sample["pair_difficulty"], "partial")

    def test_cross_wsi_loss_disables_rgb_reconstruction_but_keeps_identity_and_structure(self):
        criterion = Pix2PixTransferLoss(
            feature_extractor=self.TinyFeatureExtractor(),
            contextual_max_samples=16,
            texture_min_pixels=2,
            normalize_losses=False,
            lambda_identity_od=0.20,
            lambda_identity_feature=0.40,
            lambda_identity_band=0.20,
            lambda_identity_rank=0.15,
            lambda_structure_gray=0.50,
            lambda_structure_edge=0.50,
        )
        prediction = torch.randn(2, 3, 16, 16, requires_grad=True).tanh()
        prediction.retain_grad()
        target = torch.randn(2, 3, 16, 16).clamp(-0.8, 0.8)
        reference = torch.roll(target, shifts=1, dims=0)
        negative_reference = target.clone()
        tissue = torch.ones(2, 1, 16, 16, dtype=torch.long)
        nuclei = torch.zeros_like(tissue)
        nuclei[:, :, :4, :4] = 1
        region = tissue.clone()
        region[nuclei != 0] = 257

        loss, logs = criterion(
            prediction,
            target,
            reference=reference,
            negative_reference=negative_reference,
            target_region=region,
            reference_region=region,
            target_tissue_mask=tissue,
            target_nuclei_mask=nuclei,
            reference_tissue_mask=tissue,
            reference_nuclei_mask=nuclei,
            negative_reference_tissue_mask=tissue,
            negative_reference_nuclei_mask=nuclei,
            boundary_region=tissue,
            training_mode="cross_wsi",
        )
        loss.backward()

        self.assertEqual(logs["rgb_supervision_active"], 0.0)
        self.assertGreater(logs["structure_gray"] + logs["structure_edge"], 0.0)
        self.assertGreater(logs["identity_od"] + logs["identity_feature"] + logs["identity_band"], 0.0)
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_same_wsi_loss_keeps_rgb_reconstruction_active(self):
        criterion = Pix2PixTransferLoss(
            feature_extractor=self.TinyFeatureExtractor(),
            contextual_max_samples=16,
            texture_min_pixels=2,
            normalize_losses=False,
        )
        image = torch.randn(1, 3, 16, 16).clamp(-1.0, 1.0)
        region = torch.ones(1, 1, 16, 16, dtype=torch.long)

        _, logs = criterion(
            image * 0.5,
            image,
            reference=image,
            target_region=region,
            reference_region=region,
            training_mode="same_wsi",
        )

        self.assertEqual(logs["rgb_supervision_active"], 1.0)

    def test_boundary_band_mask_marks_only_label_transitions(self):
        region = torch.ones(1, 1, 8, 8, dtype=torch.long)
        region[:, :, :, 4:] = 2

        boundary = boundary_band_mask(region, radius=1)

        self.assertEqual(tuple(boundary.shape), (1, 1, 8, 8))
        self.assertGreater(float(boundary[:, :, :, 3:5].mean()), 0.9)
        self.assertLess(float(boundary[:, :, :, :2].mean()), 0.1)
        self.assertLess(float(boundary[:, :, :, 6:].mean()), 0.1)

    def test_high_frequency_residual_loss_respects_boundary_and_trust_weights(self):
        i0 = torch.zeros(1, 3, 16, 16)
        pred = i0.clone()
        pred[:, :, :, ::2] = 1.0
        pred[:, :, :, 1::2] = -1.0
        left_weight = torch.zeros(1, 1, 16, 16)
        left_weight[:, :, :, :8] = 1.0
        right_weight = 1.0 - left_weight

        left_loss = high_frequency_residual_loss(pred, i0, left_weight, blur_sigma=1.0)
        right_loss = high_frequency_residual_loss(pred, i0, right_weight, blur_sigma=1.0)
        zero_loss = high_frequency_residual_loss(i0, i0, left_weight, blur_sigma=1.0)

        self.assertGreater(float(left_loss), 0.1)
        self.assertGreater(float(right_loss), 0.1)
        self.assertLess(float(zero_loss), 1e-6)

    def test_ood_panel_and_summary_grid_are_written(self):
        target_i0 = torch.zeros(3, 16, 16)
        target = torch.ones(3, 16, 16) * 0.25
        references = [torch.ones(3, 16, 16) * -0.25, torch.ones(3, 16, 16) * 0.5]
        outputs = [torch.zeros(3, 16, 16), torch.ones(3, 16, 16) * 0.75]

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            panel_a = root / "probe_00_panel.png"
            panel_b = root / "probe_01_panel.png"

            save_ood_panel(
                output_path=panel_a,
                target_i0=target_i0,
                target=target,
                references=references,
                outputs=outputs,
                title="probe 0",
            )
            save_ood_panel(
                output_path=panel_b,
                target_i0=target_i0,
                target=target,
                references=references,
                outputs=outputs,
                title="probe 1",
            )
            summary = save_ood_summary_grid([panel_a, panel_b], root / "summary_grid.png")

            self.assertTrue(panel_a.exists())
            self.assertTrue(panel_b.exists())
            self.assertTrue(summary.exists())
            self.assertGreater(panel_a.stat().st_size, 0)
            self.assertGreater(summary.stat().st_size, panel_a.stat().st_size)

    def test_identity_metrics_retrieve_the_selected_reference_and_write_json(self):
        target_i0 = torch.zeros(3, 16, 16)
        target = torch.zeros(3, 16, 16)
        reference_a = torch.full((3, 16, 16), -0.5)
        reference_b = torch.full((3, 16, 16), 0.5)
        tissue = torch.ones(1, 16, 16, dtype=torch.long)
        nuclei = torch.zeros_like(tissue)

        metrics = compute_identity_metrics(
            target_i0=target_i0,
            target=target,
            references=[reference_a, reference_b],
            outputs=[reference_a.clone(), reference_b.clone()],
            target_tissue_mask=tissue,
            target_nuclei_mask=nuclei,
            reference_tissue_masks=[tissue, tissue],
            reference_nuclei_masks=[nuclei, nuclei],
        )

        self.assertAlmostEqual(metrics["selected_ref_top1"], 1.0)
        self.assertLess(metrics["own_ref_descriptor_distance"], 1e-6)
        with TemporaryDirectory() as tmpdir:
            path = save_identity_metrics([metrics], Path(tmpdir) / "identity_metrics.json")
            payload = json.loads(path.read_text())
            self.assertAlmostEqual(payload["aggregate"]["selected_ref_top1"], 1.0)


if __name__ == "__main__":
    unittest.main()

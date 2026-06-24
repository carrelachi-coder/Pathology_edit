import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from controlnet_train.pix2pix_transfer.adversarial import (
    RegionAwarePatchDiscriminator,
    discriminator_hinge_loss,
    generator_hinge_loss,
    patch_mask_from_region,
)
from controlnet_train.pix2pix_transfer.losses import (
    EMALossNormalizer,
    Pix2PixTransferLoss,
    contextual_directional_loss,
    image_l1_loss,
    regional_contextual_loss,
    regional_gram_loss,
)
from controlnet_train.pix2pix_transfer.regional_cross_attention import Pix2PixCrossAttnUNet


class Pix2PixTextureLossTests(unittest.TestCase):
    class TinyFeatureExtractor(nn.Module):
        def forward(self, image):
            layer3 = image.float()
            layer8 = F.avg_pool2d(layer3, 2)
            layer15 = F.avg_pool2d(layer8, 2)
            layer22 = F.avg_pool2d(layer15, 2)
            return {3: layer3, 8: layer8, 15: layer15, 22: layer22}

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


if __name__ == "__main__":
    unittest.main()

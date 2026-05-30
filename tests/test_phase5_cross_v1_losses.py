import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from controlnet_train.training.cross_v1_losses import (
        RegionalStainStyleLossConfig,
        per_sample_mse,
        ref_swap_sensitivity_loss,
        regional_stain_style_loss,
        self_reconstruction_l1_loss,
        uni_token_cosine_perceptual_loss,
        unpack_flux_packed_latents,
    )
    try:
        from controlnet_train.training.flux_phase5_cross_v1 import (
            _insert_self_reconstruction_samples,
            _use_random_reference,
        )
    except ModuleNotFoundError:
        _insert_self_reconstruction_samples = None
        _use_random_reference = None


@unittest.skipIf(torch is None, "torch is required for Cross V1 loss tests")
class CrossV1AuxiliaryLossTests(unittest.TestCase):
    def test_regional_stain_style_loss_matches_shared_tissue_and_nuclei_regions(self):
        prediction = torch.zeros(1, 3, 4, 4)
        reference = torch.zeros(1, 3, 4, 4)
        prediction[:, :, :2, :] = 0.25
        reference[:, :, :2, :] = 0.75
        prediction[:, :, 2:, :] = 0.1
        reference[:, :, 2:, :] = 0.9
        tissue_target = torch.tensor(
            [[[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]]]
        )
        tissue_reference = tissue_target.clone()
        nuclei_target = torch.tensor(
            [[[0, 0, 3, 3], [0, 0, 3, 3], [0, 0, 4, 4], [0, 0, 4, 4]]]
        )
        nuclei_reference = nuclei_target.clone()

        result = regional_stain_style_loss(
            prediction=prediction,
            reference=reference,
            target_tissue_mask=tissue_target,
            reference_tissue_mask=tissue_reference,
            target_nuclei_mask=nuclei_target,
            reference_nuclei_mask=nuclei_reference,
            config=RegionalStainStyleLossConfig(
                mean_weight=1.0,
                std_weight=0.0,
                covariance_weight=0.0,
                min_pixels=2,
            ),
        )

        self.assertEqual(result["tissue_regions"], 2)
        self.assertEqual(result["nuclei_regions"], 2)
        self.assertGreater(result["total"].item(), 0.0)

    def test_ref_swap_sensitivity_loss_penalizes_swapped_loss_inside_margin(self):
        normal = torch.tensor([0.10, 0.20])
        zero = torch.tensor([0.11, 0.40])
        random = torch.tensor([0.08, 0.21])

        loss = ref_swap_sensitivity_loss(normal, [zero, random], margin=0.05)

        expected_zero = torch.relu(torch.tensor([0.05 + 0.10 - 0.11, 0.05 + 0.20 - 0.40])).mean()
        expected_random = torch.relu(torch.tensor([0.05 + 0.10 - 0.08, 0.05 + 0.20 - 0.21])).mean()
        self.assertTrue(torch.allclose(loss, torch.stack([expected_zero, expected_random]).mean()))

    def test_per_sample_mse_returns_batch_values(self):
        prediction = torch.tensor([[[1.0, 3.0]], [[2.0, 4.0]]])
        target = torch.tensor([[[0.0, 1.0]], [[2.0, 1.0]]])

        values = per_sample_mse(prediction, target)

        self.assertTrue(torch.allclose(values, torch.tensor([2.5, 4.5])))

    def test_self_reconstruction_l1_loss_can_mask_inserted_samples(self):
        prediction = torch.tensor(
            [
                [[[0.0, 0.0]]],
                [[[0.0, 0.0]]],
            ]
        ).repeat(1, 3, 1, 1)
        reference = torch.tensor(
            [
                [[[1.0, 1.0]]],
                [[[3.0, 3.0]]],
            ]
        ).repeat(1, 3, 1, 1)

        loss = self_reconstruction_l1_loss(
            prediction=prediction,
            reference=reference,
            sample_mask=torch.tensor([True, False]),
        )

        self.assertTrue(torch.allclose(loss, torch.tensor(1.0)))

    def test_uni_token_cosine_perceptual_loss_uses_cosine_distance(self):
        prediction = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        target = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])

        loss = uni_token_cosine_perceptual_loss(
            prediction_features=prediction,
            target_features=target,
        )

        self.assertTrue(torch.allclose(loss, torch.tensor(0.5)))

    def test_unpack_flux_packed_latents_inverts_two_by_two_packing_order(self):
        latents = torch.arange(1 * 1 * 4 * 4, dtype=torch.float32).reshape(1, 1, 4, 4)
        packed = latents.reshape(1, 1, 2, 2, 2, 2)
        packed = packed.permute(0, 2, 4, 1, 3, 5).reshape(1, 4, 4)

        unpacked = unpack_flux_packed_latents(packed, channels=1, height=4, width=4)

        self.assertTrue(torch.equal(unpacked, latents))

    def test_random_reference_swap_rejects_batch_size_one(self):
        if _use_random_reference is None:
            self.skipTest("flux_phase5_cross_v1 optional dependencies are not installed")
        batch = {
            "reference_image": torch.zeros(1, 3, 4, 4),
            "reference_tissue_mask": torch.zeros(1, 4, 4, dtype=torch.long),
            "reference_nuclei_mask": torch.zeros(1, 4, 4, dtype=torch.long),
        }

        with self.assertRaisesRegex(ValueError, "train-batch-size > 1"):
            _use_random_reference(batch)

    def test_random_reference_swap_accepts_dataset_random_batch_for_batch_size_one(self):
        if _use_random_reference is None:
            self.skipTest("flux_phase5_cross_v1 optional dependencies are not installed")
        batch = {
            "reference_image": torch.zeros(1, 3, 4, 4),
            "reference_tissue_mask": torch.zeros(1, 4, 4, dtype=torch.long),
            "reference_nuclei_mask": torch.zeros(1, 4, 4, dtype=torch.long),
        }
        random_batch = {
            "reference_image": torch.ones(1, 3, 4, 4),
            "reference_tissue_mask": torch.ones(1, 4, 4, dtype=torch.long),
            "reference_nuclei_mask": torch.full((1, 4, 4), 2, dtype=torch.long),
        }

        swapped = _use_random_reference(batch, random_batch=random_batch)

        self.assertTrue(torch.equal(swapped["reference_image"], random_batch["reference_image"]))
        self.assertTrue(torch.equal(swapped["reference_tissue_mask"], random_batch["reference_tissue_mask"]))
        self.assertTrue(torch.equal(swapped["reference_nuclei_mask"], random_batch["reference_nuclei_mask"]))

    def test_insert_self_reconstruction_samples_replaces_only_selected_references(self):
        if _insert_self_reconstruction_samples is None:
            self.skipTest("flux_phase5_cross_v1 optional dependencies are not installed")
        batch = {
            "target_image": torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2),
            "reference_image": torch.zeros(2, 3, 2, 2),
            "target_tissue_mask": torch.ones(2, 2, 2, dtype=torch.long),
            "reference_tissue_mask": torch.zeros(2, 2, 2, dtype=torch.long),
            "target_nuclei_mask": torch.full((2, 2, 2), 2, dtype=torch.long),
            "reference_nuclei_mask": torch.zeros(2, 2, 2, dtype=torch.long),
        }

        mixed = _insert_self_reconstruction_samples(batch, torch.tensor([True, False]))

        self.assertTrue(torch.equal(mixed["reference_image"][0], batch["target_image"][0]))
        self.assertTrue(torch.equal(mixed["reference_tissue_mask"][0], batch["target_tissue_mask"][0]))
        self.assertTrue(torch.equal(mixed["reference_nuclei_mask"][0], batch["target_nuclei_mask"][0]))
        self.assertTrue(torch.equal(mixed["reference_image"][1], batch["reference_image"][1]))
        self.assertTrue(torch.equal(batch["reference_image"], torch.zeros(2, 3, 2, 2)))


if __name__ == "__main__":
    unittest.main()

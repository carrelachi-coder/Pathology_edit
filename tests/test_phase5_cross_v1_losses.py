import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from controlnet_train.training.cross_v1_losses import (
        RegionalStainStyleLossConfig,
        RegionalFeatureLossConfig,
        per_sample_mse,
        ref_swap_sensitivity_loss,
        regional_feature_map_loss,
        regional_stain_style_loss,
        self_reconstruction_l1_loss,
        uni_token_cosine_perceptual_loss,
        uni_token_distribution_perceptual_loss,
        unpack_flux_packed_latents,
    )
    from controlnet_train.training.same_wsi_appearance import (
        SameWSIAppearanceConfig,
        SameWSIAppearanceEncoder,
        SameWSIPairClassifier,
        load_same_wsi_encoder,
        same_wsi_perceptual_loss,
        save_same_wsi_checkpoint,
    )
    from controlnet_train.training.vgg_perceptual import (
        RGB_TO_LUMA,
        VGGPerceptualLoss,
        normalize_vgg_loss_type,
        parse_vgg_layer_indices,
    )
    try:
        from controlnet_train.training.flux_phase5_cross_v1 import (
            collate_cross_batch,
            _configure_controlnet_trainable_params,
            _build_region_attention_mask,
            _insert_self_reconstruction_samples,
            _use_random_reference,
        )
    except ModuleNotFoundError:
        collate_cross_batch = None
        _configure_controlnet_trainable_params = None
        _build_region_attention_mask = None
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

    def test_regional_stain_style_loss_can_mask_batch_samples(self):
        prediction = torch.zeros(2, 3, 4, 4)
        reference = torch.zeros(2, 3, 4, 4)
        prediction[1] = 0.25
        reference[1] = 0.75
        tissue = torch.ones(2, 4, 4, dtype=torch.long)

        inactive = regional_stain_style_loss(
            prediction=prediction,
            reference=reference,
            target_tissue_mask=tissue,
            reference_tissue_mask=tissue,
            sample_mask=torch.tensor([True, False]),
            config=RegionalStainStyleLossConfig(
                mean_weight=1.0,
                std_weight=0.0,
                covariance_weight=0.0,
                min_pixels=2,
            ),
        )
        active = regional_stain_style_loss(
            prediction=prediction,
            reference=reference,
            target_tissue_mask=tissue,
            reference_tissue_mask=tissue,
            sample_mask=torch.tensor([False, True]),
            config=RegionalStainStyleLossConfig(
                mean_weight=1.0,
                std_weight=0.0,
                covariance_weight=0.0,
                min_pixels=2,
            ),
        )

        self.assertEqual(inactive["tissue_regions"], 1)
        self.assertTrue(torch.allclose(inactive["total"], torch.tensor(0.0)))
        self.assertEqual(active["tissue_regions"], 1)
        self.assertGreater(active["total"].item(), 0.0)

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

    def test_uni_token_distribution_perceptual_loss_does_not_require_spatial_alignment(self):
        prediction = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        reference = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])

        loss = uni_token_distribution_perceptual_loss(
            prediction_features=prediction,
            reference_features=reference,
            pooled_cosine_weight=0.0,
        )

        self.assertTrue(torch.allclose(loss, torch.tensor(0.0)))

    def test_regional_feature_map_loss_matches_same_label_feature_regions(self):
        prediction = torch.zeros(1, 4, 2)
        reference = torch.zeros(1, 4, 2)
        prediction[:, :2] = torch.tensor([1.0, 0.0])
        reference[:, :2] = torch.tensor([0.0, 1.0])
        tissue = torch.tensor([[[1, 1], [2, 2]]])

        result = regional_feature_map_loss(
            prediction_features=prediction,
            reference_features=reference,
            target_tissue_mask=tissue,
            reference_tissue_mask=tissue,
            config=RegionalFeatureLossConfig(
                mean_weight=1.0,
                std_weight=0.0,
                pooled_cosine_weight=0.0,
                min_tokens=1,
            ),
        )

        self.assertEqual(result["tissue_regions"], 2)
        self.assertGreater(result["total"].item(), 0.0)

    def test_region_attention_mask_blocks_cross_label_ip_tokens(self):
        if _build_region_attention_mask is None:
            self.skipTest("flux_phase5_cross_v1 optional dependencies are not installed")
        query_labels = torch.tensor([[1, 2, 3]])
        key_labels = torch.tensor([[1, 1, 2, 2]])

        mask = _build_region_attention_mask(
            query_region_labels=query_labels,
            key_region_labels=key_labels,
            batch_size=1,
            query_len=3,
            key_len=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
            strict=True,
        )

        self.assertIsNotNone(mask)
        self.assertEqual(tuple(mask.shape), (1, 1, 3, 4))
        self.assertEqual(float(mask[0, 0, 0, 0]), 0.0)
        self.assertLess(float(mask[0, 0, 0, 2]), -1e20)
        # Label 3 is absent in the reference token bank, so it falls back to all tokens.
        self.assertTrue(torch.all(mask[0, 0, 2] == 0))

    def test_same_wsi_perceptual_loss_uses_masked_reference_texture_features(self):
        encoder = SameWSIAppearanceEncoder(
            SameWSIAppearanceConfig(
                backbone_channels=(4, 8),
                embedding_dim=8,
                input_size=8,
                feature_layers=(1,),
            )
        )
        encoder.eval()
        encoder.requires_grad_(False)
        prediction = torch.zeros(1, 3, 8, 8, requires_grad=True)
        reference = torch.ones(1, 3, 8, 8)
        mask = torch.ones(1, 8, 8, dtype=torch.long)

        loss, terms = same_wsi_perceptual_loss(
            encoder=encoder,
            prediction=prediction,
            reference=reference,
            target_tissue_mask=mask,
            reference_tissue_mask=mask,
            min_pixels=1,
        )

        self.assertEqual(terms, 1)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(prediction.grad)

    def test_same_wsi_encoder_checkpoint_round_trips(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/best.pt"
            config = SameWSIAppearanceConfig(
                backbone_channels=(4, 8),
                embedding_dim=8,
                input_size=8,
                feature_layers=(0, 1),
            )
            model = SameWSIPairClassifier(SameWSIAppearanceEncoder(config))
            save_same_wsi_checkpoint(path, model)

            loaded = load_same_wsi_encoder(path)

        self.assertEqual(loaded.config.input_size, 8)
        self.assertEqual(loaded.config.feature_layers, (0, 1))
        self.assertFalse(any(param.requires_grad for param in loaded.parameters()))

    def test_vgg_perceptual_loss_uses_masked_gram_style_features(self):
        features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=1, bias=False),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(4, 4, kernel_size=1, bias=False),
            torch.nn.ReLU(inplace=False),
        )
        for param in features.parameters():
            torch.nn.init.constant_(param, 0.25)
        loss_fn = VGGPerceptualLoss(
            features,
            layer_indices=(1, 3),
            loss_type="gram",
            input_size=8,
            normalize_mean=(0.0, 0.0, 0.0),
            normalize_std=(1.0, 1.0, 1.0),
        )
        prediction = torch.zeros(1, 3, 8, 8, requires_grad=True)
        reference = torch.ones(1, 3, 8, 8)
        mask = torch.ones(1, 8, 8, dtype=torch.long)

        loss, terms = loss_fn(
            prediction,
            reference,
            target_tissue_mask=mask,
            reference_tissue_mask=mask,
            min_pixels=1,
        )

        self.assertEqual(terms, 2)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertFalse(any(param.requires_grad for param in loss_fn.parameters()))

    def test_vgg_region_gram_resizes_masks_per_feature_layer_and_filters_small_regions(self):
        features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 2, kernel_size=1, bias=False),
            torch.nn.ReLU(inplace=False),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(2, 2, kernel_size=1, bias=False),
            torch.nn.ReLU(inplace=False),
        )
        for param in features.parameters():
            torch.nn.init.constant_(param, 0.5)
        loss_fn = VGGPerceptualLoss(
            features,
            layer_indices=(1, 4),
            loss_type="gram",
            input_size=4,
            normalize_mean=(0.0, 0.0, 0.0),
            normalize_std=(1.0, 1.0, 1.0),
        )
        prediction = torch.zeros(1, 3, 4, 4, requires_grad=True)
        reference = torch.ones(1, 3, 4, 4)
        mask = torch.ones(1, 4, 4, dtype=torch.long)
        mask[:, 2:, 2:] = 2

        loss, terms = loss_fn(
            prediction,
            reference,
            target_tissue_mask=mask,
            reference_tissue_mask=mask,
            min_pixels=2,
        )

        self.assertEqual(terms, 2)
        self.assertGreater(loss.item(), 0.0)

    def test_vgg_grayscale_removes_color_only_gram_difference(self):
        features = torch.nn.Sequential(torch.nn.Identity())
        loss_fn = VGGPerceptualLoss(
            features,
            layer_indices=(0,),
            loss_type="gram",
            input_size=0,
            normalize_mean=(0.0, 0.0, 0.0),
            normalize_std=(1.0, 1.0, 1.0),
        )
        rgb_to_luma = torch.tensor(RGB_TO_LUMA)
        prediction_color = torch.tensor([1.0, 0.0, 0.0])
        reference_color = torch.tensor([0.0, rgb_to_luma[0].item() / rgb_to_luma[1].item(), 0.0])
        self.assertTrue(torch.allclose(prediction_color @ rgb_to_luma, reference_color @ rgb_to_luma))
        prediction = prediction_color.view(1, 3, 1, 1).expand(1, 3, 4, 4).clone().requires_grad_(True)
        reference = reference_color.view(1, 3, 1, 1).expand(1, 3, 4, 4).clone()

        loss, terms = loss_fn(prediction, reference)

        self.assertEqual(terms, 1)
        self.assertTrue(torch.allclose(loss, torch.tensor(0.0), atol=1e-6))
        loss.backward()
        self.assertIsNotNone(prediction.grad)

        rgb_loss_fn = VGGPerceptualLoss(
            features,
            layer_indices=(0,),
            loss_type="gram",
            grayscale=False,
            input_size=0,
            normalize_mean=(0.0, 0.0, 0.0),
            normalize_std=(1.0, 1.0, 1.0),
        )
        rgb_loss, _ = rgb_loss_fn(prediction.detach().requires_grad_(True), reference)
        self.assertGreater(rgb_loss.item(), 0.0)

    def test_parse_vgg_layer_indices_accepts_aliases_and_numbers(self):
        self.assertEqual(parse_vgg_layer_indices("relu1_2,8,conv3_3"), (3, 8, 14))

    def test_normalize_vgg_loss_type_accepts_style_alias(self):
        self.assertEqual(normalize_vgg_loss_type("style"), "gram")
        self.assertEqual(normalize_vgg_loss_type("feature"), "feature_l1")

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

    def test_collate_expands_paired_counterfactual_examples(self):
        if collate_cross_batch is None:
            self.skipTest("flux_phase5_cross_v1 optional dependencies are not installed")

        def make_item(value: float) -> dict:
            return {
                "sample_id": f"sample-{value}",
                "reference_sample_id": f"ref-{value}",
                "target_image": torch.full((3, 2, 2), value),
                "reference_image": torch.full((3, 2, 2), value + 10),
                "target_tissue_mask": torch.ones(2, 2, dtype=torch.long),
                "target_nuclei_mask": torch.ones(2, 2, dtype=torch.long),
                "reference_tissue_mask": torch.zeros(2, 2, dtype=torch.long),
                "reference_nuclei_mask": torch.zeros(2, 2, dtype=torch.long),
                "clean_image_for_noising": torch.full((3, 2, 2), value + 20),
                "uses_degraded_noising": True,
                "prompt": "prompt",
            }

        batch = collate_cross_batch(
            [{"paired_counterfactual": [make_item(1.0), make_item(2.0)]}]
        )

        self.assertEqual(batch["target_image"].shape[0], 2)
        self.assertEqual(batch["clean_image_for_noising"].shape[0], 2)
        self.assertTrue(torch.equal(batch["uses_degraded_noising"], torch.tensor([True, True])))
        self.assertEqual(batch["sample_ids"], ["sample-1.0", "sample-2.0"])
        self.assertTrue(torch.equal(batch["target_tissue_mask"][0], batch["target_tissue_mask"][1]))

    def test_controlnet_outputs_train_mode_only_unfreezes_residual_outputs_by_default(self):
        if _configure_controlnet_trainable_params is None:
            self.skipTest("flux_phase5_cross_v1 optional dependencies are not installed")

        class TinyControlNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.controlnet_x_embedder = torch.nn.Linear(2, 2)
                self.transformer_blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
                self.single_transformer_blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
                self.controlnet_blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
                self.controlnet_single_blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])

        model = TinyControlNet()
        trainable = _configure_controlnet_trainable_params(model, mode="outputs")

        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith(("controlnet_blocks", "controlnet_single_blocks")) for name in trainable))
        self.assertFalse(any(param.requires_grad for param in model.controlnet_x_embedder.parameters()))
        self.assertFalse(any(param.requires_grad for block in model.transformer_blocks for param in block.parameters()))

    def test_controlnet_outputs_train_mode_can_unfreeze_x_embedder_and_tail_blocks(self):
        if _configure_controlnet_trainable_params is None:
            self.skipTest("flux_phase5_cross_v1 optional dependencies are not installed")

        class TinyControlNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.controlnet_x_embedder = torch.nn.Linear(2, 2)
                self.transformer_blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
                self.single_transformer_blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
                self.controlnet_blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
                self.controlnet_single_blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])

        model = TinyControlNet()
        trainable = _configure_controlnet_trainable_params(
            model,
            mode="outputs",
            train_x_embedder=True,
            train_last_n_blocks=1,
            train_last_n_single_blocks=1,
        )

        self.assertIn("controlnet_x_embedder.weight", trainable)
        self.assertIn("transformer_blocks.1.weight", trainable)
        self.assertIn("single_transformer_blocks.0.weight", trainable)
        self.assertFalse(any(param.requires_grad for param in model.transformer_blocks[0].parameters()))


if __name__ == "__main__":
    unittest.main()

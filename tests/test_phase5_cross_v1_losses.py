import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from controlnet_train.training.cross_v1_losses import (
        RegionalFeatureLossConfig,
        RegionalRgbFftLossConfig,
        RegionalStainStyleLossConfig,
        per_sample_mse,
        ref_swap_sensitivity_loss,
        regional_feature_map_loss,
        regional_rgb_fft_loss,
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
    from controlnet_train.modules.reference_image_encoder import (
        ReferenceImageEncoder,
        build_region_ip_token_labels,
    )
    try:
        from controlnet_train.training.flux_phase5_cross_v1 import (
            collate_cross_batch,
            _configure_controlnet_trainable_params,
            _build_region_attention_mask,
            _build_region_attention_mask_and_query_gate,
            _reference_region_sigma_mask,
            _insert_self_reconstruction_samples,
            _use_random_reference,
        )
    except ModuleNotFoundError:
        collate_cross_batch = None
        _configure_controlnet_trainable_params = None
        _build_region_attention_mask = None
        _build_region_attention_mask_and_query_gate = None
        _reference_region_sigma_mask = None
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

    def test_regional_feature_map_loss_matches_tissue_nuclei_composite_regions(self):
        prediction = torch.zeros(1, 4, 2)
        reference = torch.zeros(1, 4, 2)
        prediction[:, 1] = torch.tensor([1.0, 0.0])
        reference[:, 1] = torch.tensor([0.0, 1.0])
        tissue = torch.tensor([[[1, 1], [1, 1]]])
        nuclei = torch.tensor([[[0, 3], [0, 3]]])

        result = regional_feature_map_loss(
            prediction_features=prediction,
            reference_features=reference,
            target_tissue_mask=tissue,
            reference_tissue_mask=tissue,
            target_nuclei_mask=nuclei,
            reference_nuclei_mask=nuclei,
            config=RegionalFeatureLossConfig(
                tissue_weight=0.0,
                nuclei_weight=0.0,
                composite_weight=1.0,
                mean_weight=1.0,
                std_weight=0.0,
                pooled_cosine_weight=0.0,
                min_tokens=1,
            ),
        )

        self.assertEqual(result["composite_regions"], 2)
        self.assertGreater(result["composite"].item(), 0.0)
        self.assertTrue(torch.allclose(result["total"], result["composite"]))

    def test_regional_rgb_fft_loss_matches_same_label_regions_with_gradient(self):
        prediction = torch.zeros(1, 3, 8, 8)
        reference = torch.ones(1, 3, 8, 8)
        prediction.requires_grad_()
        tissue = torch.ones(1, 8, 8, dtype=torch.long)

        result = regional_rgb_fft_loss(
            prediction=prediction,
            reference=reference,
            target_tissue_mask=tissue,
            reference_tissue_mask=tissue,
            config=RegionalRgbFftLossConfig(
                mean_weight=1.0,
                std_weight=0.0,
                fft_weight=0.25,
                fft_size=8,
                fft_bins=4,
                min_pixels=1,
            ),
        )

        self.assertEqual(result["tissue_regions"], 1)
        self.assertGreater(result["total"].item(), 0.0)
        result["total"].backward()
        self.assertIsNotNone(prediction.grad)
        self.assertGreater(prediction.grad.abs().sum().item(), 0.0)

    def test_build_region_ip_token_labels_combines_tissue_and_nuclei_labels(self):
        tissue = torch.tensor([[[1, 1], [2, 2]]])
        nuclei = torch.tensor([[[0, 3], [0, 4]]])

        labels = build_region_ip_token_labels(
            tissue_mask=tissue,
            nuclei_mask=nuclei,
            num_tokens=4,
            label_mode="tissue_nuclei",
        )

        expected = torch.tensor([[256, 259, 512, 516]])
        self.assertTrue(torch.equal(labels, expected))

    def test_build_region_ip_token_labels_routes_background_to_null_label(self):
        tissue = torch.tensor([[[0, 1], [2, 0]]])
        nuclei = torch.tensor([[[0, 3], [0, 4]]])

        tissue_labels = build_region_ip_token_labels(
            tissue_mask=tissue,
            num_tokens=4,
            label_mode="tissue",
        )
        composite_labels = build_region_ip_token_labels(
            tissue_mask=tissue,
            nuclei_mask=nuclei,
            num_tokens=4,
            label_mode="tissue_nuclei",
        )

        self.assertTrue(torch.equal(tissue_labels, torch.tensor([[-1, 1, 2, -1]])))
        self.assertTrue(torch.equal(composite_labels, torch.tensor([[-1, 259, 512, -1]])))

    def test_build_region_ip_token_labels_coarse_tissue_maps_fine_tumor_labels(self):
        tissue = torch.tensor([[[8, 9, 10], [14, 15, 2], [0, 1, 3]]])

        labels = build_region_ip_token_labels(
            tissue_mask=tissue,
            num_tokens=9,
            label_mode="coarse_tissue",
        )

        expected = torch.tensor([[1, 1, 1, 1, 1, 2, -1, 1, 3]])
        self.assertTrue(torch.equal(labels, expected))

    def test_stats_region_tokens_emit_separate_mean_and_std_tokens_per_label(self):
        encoder = object.__new__(ReferenceImageEncoder)
        projected = torch.tensor(
            [
                [
                    [1.0, 2.0],
                    [3.0, 6.0],
                    [10.0, 20.0],
                    [30.0, 60.0],
                    [0.0, 0.0],
                ]
            ]
        )
        labels = torch.tensor([[1, 1, 2, 2, -1]])

        tokens, token_labels = encoder._stats_by_region_labels(projected, labels)

        expected_label_1_mean = torch.tensor([2.0, 4.0])
        expected_label_1_std = torch.tensor([1.0, 2.0])
        expected_label_2_mean = torch.tensor([20.0, 40.0])
        expected_label_2_std = torch.tensor([10.0, 20.0])
        self.assertEqual(tuple(tokens.shape), (1, 4, 2))
        self.assertTrue(torch.equal(token_labels, torch.tensor([[1, 1, 2, 2]])))
        self.assertTrue(torch.allclose(tokens[0, 0], expected_label_1_mean))
        self.assertTrue(torch.allclose(tokens[0, 1], expected_label_1_std))
        self.assertTrue(torch.allclose(tokens[0, 2], expected_label_2_mean))
        self.assertTrue(torch.allclose(tokens[0, 3], expected_label_2_std))

    def test_soft_region_attention_bias_is_dense_finite_label_bias(self):
        if _build_region_attention_mask_and_query_gate is None:
            self.skipTest("flux_phase5_cross_v1 optional dependencies are not installed")
        query_labels = torch.tensor([[1, 2, -1]])
        key_labels = torch.tensor([[1, 2, 3, -1]])

        mask, query_gate, stats = _build_region_attention_mask_and_query_gate(
            query_region_labels=query_labels,
            key_region_labels=key_labels,
            batch_size=1,
            query_len=3,
            key_len=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
            strict=False,
            soft_bias=torch.tensor(1.5),
            use_soft_bias=True,
        )

        self.assertIsNotNone(mask)
        self.assertIsNone(query_gate)
        self.assertEqual(tuple(mask.shape), (1, 1, 3, 4))
        self.assertAlmostEqual(float(mask[0, 0, 0, 0]), 1.5)
        self.assertAlmostEqual(float(mask[0, 0, 0, 1]), -1.5)
        self.assertAlmostEqual(float(mask[0, 0, 1, 1]), 1.5)
        self.assertAlmostEqual(float(mask[0, 0, 1, 0]), -1.5)
        self.assertLess(float(mask[0, 0, 0, 3]), -1e20)
        self.assertTrue(torch.isfinite(mask[0, 0, :, :3]).all())
        self.assertTrue(bool(stats["soft_bias_enabled"]))
        self.assertAlmostEqual(float(stats["soft_bias"]), 1.5)
        self.assertEqual(float(stats["null_query_fraction"]), 0.0)

    def test_reference_region_sigma_mask_keeps_low_mid_band(self):
        if _reference_region_sigma_mask is None:
            self.skipTest("flux_phase5_cross_v1 optional dependencies are not installed")

        mask = _reference_region_sigma_mask(
            torch.tensor([0.0, 0.2, 0.6, 0.8]),
            min_sigma=0.1,
            max_sigma=0.6,
        )

        self.assertTrue(torch.equal(mask, torch.tensor([False, True, True, False])))

    def test_region_attention_mask_routes_missing_label_to_learned_null_token(self):
        if _build_region_attention_mask_and_query_gate is None:
            self.skipTest("flux_phase5_cross_v1 optional dependencies are not installed")
        query_labels = torch.tensor([[1, 2, 3]])
        key_labels = torch.tensor([[1, 1, 2, 2]])

        mask, query_gate, stats = _build_region_attention_mask_and_query_gate(
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
        self.assertEqual(tuple(mask.shape), (1, 1, 3, 5))
        self.assertEqual(float(mask[0, 0, 0, 0]), 0.0)
        self.assertLess(float(mask[0, 0, 0, 2]), -1e20)
        self.assertLess(float(mask[0, 0, 0, 4]), -1e20)
        # Label 3 is absent in the reference token bank. Strict regional routing
        # blocks all real reference keys and allows only the appended null token.
        self.assertIsNone(query_gate)
        self.assertLess(float(mask[0, 0, 2, 0]), -1e20)
        self.assertLess(float(mask[0, 0, 2, 3]), -1e20)
        self.assertEqual(float(mask[0, 0, 2, 4]), 0.0)
        self.assertEqual(float(stats["active_query_fraction"]), 2 / 3)
        self.assertEqual(float(stats["missing_query_fraction"]), 1 / 3)
        self.assertEqual(float(stats["fallback_query_fraction"]), 0.0)
        self.assertEqual(float(stats["null_query_fraction"]), 1 / 3)
        self.assertAlmostEqual(float(stats["allowed_tokens_per_query_mean"]), 5 / 3)
        self.assertEqual(int(stats["allowed_tokens_per_query_min"]), 1)
        self.assertEqual(int(stats["allowed_tokens_per_query_max"]), 2)

        legacy_mask = _build_region_attention_mask(
            query_region_labels=query_labels,
            key_region_labels=key_labels,
            batch_size=1,
            query_len=3,
            key_len=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
            strict=True,
        )
        self.assertIsNotNone(legacy_mask)
        self.assertEqual(tuple(legacy_mask.shape), (1, 1, 3, 5))

    def test_region_attention_mask_does_not_silently_fallback_to_same_tissue(self):
        if _build_region_attention_mask_and_query_gate is None:
            self.skipTest("flux_phase5_cross_v1 optional dependencies are not installed")
        query_labels = torch.tensor([[259, 516]])
        key_labels = torch.tensor([[256, 512, -1]])
        query_fallback = torch.tensor([[1, 2]])
        key_fallback = torch.tensor([[1, 2, -1]])

        mask, query_gate, stats = _build_region_attention_mask_and_query_gate(
            query_region_labels=query_labels,
            key_region_labels=key_labels,
            query_fallback_labels=query_fallback,
            key_fallback_labels=key_fallback,
            batch_size=1,
            query_len=2,
            key_len=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
            strict=True,
        )

        self.assertIsNotNone(mask)
        self.assertIsNone(query_gate)
        self.assertEqual(tuple(mask.shape), (1, 1, 2, 4))
        self.assertLess(float(mask[0, 0, 0, 0]), -1e20)
        self.assertLess(float(mask[0, 0, 0, 1]), -1e20)
        self.assertLess(float(mask[0, 0, 0, 2]), -1e20)
        self.assertEqual(float(mask[0, 0, 0, 3]), 0.0)
        self.assertLess(float(mask[0, 0, 1, 0]), -1e20)
        self.assertLess(float(mask[0, 0, 1, 1]), -1e20)
        self.assertLess(float(mask[0, 0, 1, 2]), -1e20)
        self.assertEqual(float(mask[0, 0, 1, 3]), 0.0)
        self.assertEqual(float(stats["missing_query_fraction"]), 1.0)
        self.assertEqual(float(stats["fallback_query_fraction"]), 0.0)
        self.assertEqual(float(stats["null_query_fraction"]), 1.0)

    def test_region_attention_mask_all_miss_batch_is_finite_with_null_token(self):
        if _build_region_attention_mask_and_query_gate is None:
            self.skipTest("flux_phase5_cross_v1 optional dependencies are not installed")
        torch.manual_seed(7)
        batch_size, heads, query_len, key_len, head_dim = 2, 2, 3, 4, 8
        query_labels = torch.tensor([[9, 10, -1], [11, -1, 12]])
        key_labels = torch.tensor([[1, 2, -1, -1], [3, 4, -1, -1]])

        mask, query_gate, stats = _build_region_attention_mask_and_query_gate(
            query_region_labels=query_labels,
            key_region_labels=key_labels,
            batch_size=batch_size,
            query_len=query_len,
            key_len=key_len,
            device=torch.device("cpu"),
            dtype=torch.float32,
            strict=True,
        )

        self.assertIsNotNone(mask)
        self.assertIsNone(query_gate)
        self.assertEqual(tuple(mask.shape), (batch_size, 1, query_len, key_len + 1))
        self.assertEqual(float(stats["active_query_fraction"]), 0.0)
        self.assertTrue(torch.all(mask[:, :, :, :key_len] < -1e20))
        self.assertTrue(torch.all(mask[:, :, :, key_len] == 0.0))

        query = torch.randn(batch_size, heads, query_len, head_dim)
        key = torch.randn(batch_size, heads, key_len + 1, head_dim)
        value = torch.randn(batch_size, heads, key_len + 1, head_dim)
        out = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
        )
        self.assertTrue(torch.isfinite(out).all())

    def test_region_attention_mask_disallowed_garbage_keys_and_values_do_not_change_output(self):
        if _build_region_attention_mask_and_query_gate is None:
            self.skipTest("flux_phase5_cross_v1 optional dependencies are not installed")
        torch.manual_seed(11)
        batch_size, heads, query_len, key_len, head_dim = 2, 2, 3, 4, 8
        query_labels = torch.tensor([[1, 2, 3], [4, -1, 5]])
        key_labels = torch.tensor([[1, 1, 2, -1], [4, 6, -1, -1]])
        mask, _, _ = _build_region_attention_mask_and_query_gate(
            query_region_labels=query_labels,
            key_region_labels=key_labels,
            batch_size=batch_size,
            query_len=query_len,
            key_len=key_len,
            device=torch.device("cpu"),
            dtype=torch.float32,
            strict=True,
        )
        self.assertIsNotNone(mask)

        query = torch.randn(batch_size, heads, query_len, head_dim)
        key = torch.randn(batch_size, heads, key_len + 1, head_dim)
        value = torch.randn(batch_size, heads, key_len + 1, head_dim)
        clean = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
        )

        for query_index in range(query_len):
            query_slice = query[:, :, query_index : query_index + 1, :]
            mask_slice = mask[:, :, query_index : query_index + 1, :]
            garbage_key = key.clone()
            garbage_value = value.clone()
            blocked = (mask_slice[:, :, 0, :] < -1e20).expand(-1, heads, -1)
            garbage_key[blocked] = torch.randn_like(garbage_key[blocked]) * 1e4
            garbage_value[blocked] = torch.randn_like(garbage_value[blocked]) * 1e4
            dirty = torch.nn.functional.scaled_dot_product_attention(
                query_slice,
                garbage_key,
                garbage_value,
                attn_mask=mask_slice,
                dropout_p=0.0,
                is_causal=False,
            )
            self.assertTrue(torch.allclose(dirty, clean[:, :, query_index : query_index + 1, :], atol=1e-5))

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

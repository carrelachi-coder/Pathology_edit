import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from controlnet_train.data.cross_v5_pairing import CrossV5PairingSampler
    from controlnet_train.modules.cross_v5_conditioning import (
        CrossV5AdaLNModulator,
        CrossV5GeometryControlSpec,
        CrossV5RefBankBuilder,
        CrossV5SpatialAdaLNModulator,
        CrossV5TissueBank,
        build_cross_v5_geometry_control_condition,
        build_cross_v5_hed_stat_prototypes,
        build_cross_v5_spatial_structure_tokens,
        build_cross_v5_token_class_probs,
        gather_cross_v5_class_values,
    )
    from controlnet_train.training.cross_v5_losses import (
        CrossV5AppearanceLossConfig,
        CrossV5GeometryConsistencyLossConfig,
        cross_v5_appearance_fidelity_loss,
        cross_v5_geometry_consistency_loss,
        masked_gram_matrix,
    )
    from controlnet_train.training.cross_v5_glue import (
        CrossV5AdaLNAdapterMixin,
        CrossV5AdaLNHookSpec,
        CrossV5LatentDecodeConfig,
        CrossV5LossIntervals,
        CrossV5LossWeights,
        CrossV5PairingPolicy,
        CrossV5StepContext,
        assemble_cross_v5_losses,
        assemble_cross_v5_step_losses,
        decode_cross_v5_prediction_rgb,
        freeze_predictor_for_v5_loss,
        install_cross_v5_adaln_hooks,
        reconstruct_cross_v5_x0_latents,
        should_run_cross_v5_branch,
        validate_cross_v5_predictor_grad_bridge,
    )
    from controlnet_train.training.cross_v5_flux_adapters import (
        CROSS_V5_BANK_KEY,
        CROSS_V5_IMAGE_TOKEN_START_KEY,
        CROSS_V5_TARGET_CLASS_IDS_KEY,
        CROSS_V5_TARGET_STRUCTURE_TOKENS_KEY,
        install_cross_v5_flux_adaln_adapters,
    )
    from controlnet_train.training.flux_phase5_cross_v3 import _should_train_controlnet_x_embedder


@unittest.skipIf(torch is None, "torch is required for Cross V5 tests")
class CrossV5MinInterfaceTests(unittest.TestCase):
    def test_cross_v5_forces_controlnet_x_embedder_trainable(self):
        class Args:
            controlnet_train_x_embedder = False

        self.assertTrue(_should_train_controlnet_x_embedder(Args(), is_cross_v5=True))
        self.assertFalse(_should_train_controlnet_x_embedder(Args(), is_cross_v5=False))
        Args.controlnet_train_x_embedder = True
        self.assertTrue(_should_train_controlnet_x_embedder(Args(), is_cross_v5=False))

    def test_geometry_control_condition_is_class_agnostic(self):
        tissue_a = torch.tensor(
            [
                [
                    [1, 1, 2, 2],
                    [1, 1, 2, 2],
                    [3, 3, 4, 4],
                    [3, 3, 4, 4],
                ]
            ]
        )
        tissue_b = torch.tensor(
            [
                [
                    [5, 5, 6, 6],
                    [5, 5, 6, 6],
                    [7, 7, 1, 1],
                    [7, 7, 1, 1],
                ]
            ]
        )
        nuclei = torch.tensor(
            [
                [
                    [0, 1, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 1],
                ]
            ]
        )

        control_a = build_cross_v5_geometry_control_condition(
            target_tissue_mask=tissue_a,
            target_nuclei_mask=nuclei,
            output_height=4,
            output_width=4,
        )
        control_b = build_cross_v5_geometry_control_condition(
            target_tissue_mask=tissue_b,
            target_nuclei_mask=nuclei,
            output_height=4,
            output_width=4,
        )

        self.assertEqual(CrossV5GeometryControlSpec().raw_channels, 4)
        self.assertEqual(CrossV5GeometryControlSpec().packed_channels, 16)
        self.assertEqual(tuple(control_a.shape), (1, 4, 4, 4))
        self.assertTrue(torch.allclose(control_a, control_b))
        self.assertGreater(float(control_a[:, 0].sum()), 0.0)
        self.assertGreater(float(control_a[:, 1].sum()), 0.0)

    def test_token_class_probs_downsample_masks_to_token_grid(self):
        class_ids = torch.tensor(
            [
                [
                    [1, 1, 2, 2],
                    [1, 1, 2, 2],
                    [3, 3, 0, 0],
                    [3, 3, 0, 0],
                ]
            ]
        )

        token_ids, confidence, probs = build_cross_v5_token_class_probs(
            class_ids=class_ids,
            num_classes=4,
            token_height=2,
            token_width=2,
        )

        self.assertTrue(torch.equal(token_ids, torch.tensor([[1, 2, 3, 0]])))
        self.assertTrue(torch.allclose(confidence, torch.ones(1, 4)))
        self.assertEqual(tuple(probs.shape), (1, 4, 4))

    def test_ref_bank_builder_masked_pools_prototypes_and_topk_tokens(self):
        tokens = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [3.0, 0.0],
                    [0.0, 2.0],
                    [0.0, 4.0],
                ]
            ]
        )
        class_ids = torch.tensor([[[1, 1], [2, 2]]])
        builder = CrossV5RefBankBuilder(num_classes=3, local_tokens_per_class=2, prototype_source="token_pool")

        bank = builder(reference_tokens=tokens, reference_class_ids=class_ids)

        self.assertEqual(tuple(bank.prototypes.shape), (1, 3, 2))
        self.assertTrue(torch.allclose(bank.prototypes[0, 1], torch.tensor([2.0, 0.0])))
        self.assertTrue(torch.allclose(bank.prototypes[0, 2], torch.tensor([0.0, 3.0])))
        self.assertFalse(bool(bank.class_present[0, 0].item()))
        self.assertTrue(bool(bank.class_present[0, 1].item()))
        self.assertEqual(tuple(bank.local_tokens.shape), (1, 3, 2, 2))

    def test_ref_bank_builder_uses_hed_stats_as_default_prototypes(self):
        tokens = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [3.0, 0.0],
                    [0.0, 2.0],
                    [0.0, 4.0],
                ]
            ]
        )
        image = torch.tensor(
            [
                [
                    [[0.8, 0.7], [0.2, 0.3]],
                    [[0.6, 0.5], [0.4, 0.4]],
                    [[0.7, 0.6], [0.5, 0.5]],
                ]
            ]
        )
        class_ids = torch.tensor([[[1, 1], [2, 2]]])
        builder = CrossV5RefBankBuilder(num_classes=3, local_tokens_per_class=2)

        bank = builder(reference_tokens=tokens, reference_image=image, reference_class_ids=class_ids)
        expected = build_cross_v5_hed_stat_prototypes(
            reference_image=image,
            reference_class_ids=class_ids,
            num_classes=3,
        )

        self.assertEqual(tuple(bank.prototypes.shape), (1, 3, 4))
        self.assertEqual(tuple(bank.local_tokens.shape), (1, 3, 2, 2))
        self.assertTrue(torch.allclose(bank.prototypes, expected.to(dtype=bank.prototypes.dtype)))
        self.assertFalse(torch.allclose(bank.prototypes[0, 1, :2], tokens[0, :2].mean(dim=0)))

    def test_gather_class_values_uses_batched_class_ids(self):
        values = torch.tensor(
            [
                [
                    [10.0, 11.0],
                    [20.0, 21.0],
                    [30.0, 31.0],
                ]
            ]
        )

        gathered = gather_cross_v5_class_values(values, torch.tensor([[2, 1, 2, 0]]))

        expected = torch.tensor([[[30.0, 31.0], [20.0, 21.0], [30.0, 31.0], [10.0, 11.0]]])
        self.assertTrue(torch.equal(gathered, expected))

    def test_adaln_modulator_uses_prior_for_missing_classes(self):
        hidden = torch.zeros(1, 3, 4)
        bank = CrossV5TissueBank(
            prototypes=torch.zeros(1, 3, 4),
            local_tokens=torch.zeros(1, 3, 1, 4),
            class_present=torch.tensor([[True, False, True]]),
            class_mass=torch.tensor([[1.0, 0.0, 1.0]]),
            token_class_ids=torch.zeros(1, 1, dtype=torch.long),
            token_class_confidence=torch.ones(1, 1),
        )
        fallback = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
            ]
        )
        modulator = CrossV5AdaLNModulator(hidden_dim=4, output_init_std=0.01)

        output = modulator(
            hidden_states=hidden,
            target_class_ids=torch.tensor([[0, 1, 2]]),
            bank=bank,
            fallback_prototypes=fallback,
        )

        self.assertEqual(tuple(output.hidden_states.shape), (1, 3, 4))
        self.assertTrue(torch.allclose(output.source_prototypes[0, 1], fallback[1]))
        self.assertFalse(torch.allclose(output.gamma, torch.zeros_like(output.gamma)))

    def test_spatial_adaln_modulator_uses_target_structure_tokens(self):
        hidden = torch.zeros(1, 2, 4)
        bank = CrossV5TissueBank(
            prototypes=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            local_tokens=torch.zeros(1, 1, 1, 2),
            class_present=torch.ones(1, 1, dtype=torch.bool),
            class_mass=torch.ones(1, 1),
            token_class_ids=torch.zeros(1, 1, dtype=torch.long),
            token_class_confidence=torch.ones(1, 1),
        )
        structure = torch.tensor([[[1.0, 0.0, -1.0], [1.0, 0.0, 1.0]]])
        modulator = CrossV5SpatialAdaLNModulator(
            hidden_dim=4,
            prototype_dim=4,
            structure_dim=3,
            output_init_std=0.01,
        )

        output = modulator(
            hidden_states=hidden,
            target_class_ids=torch.zeros(1, 2, dtype=torch.long),
            bank=bank,
            target_structure_tokens=structure,
        )

        self.assertEqual(tuple(output.hidden_states.shape), (1, 2, 4))
        self.assertFalse(torch.allclose(output.gamma[:, 0], output.gamma[:, 1]))

    def test_spatial_adaln_modulator_does_not_use_local_tokens_as_residual_attention(self):
        hidden = torch.zeros(1, 2, 4)
        structure = torch.tensor([[[1.0, 0.0, -1.0], [1.0, 0.0, 1.0]]])
        base_bank = CrossV5TissueBank(
            prototypes=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            local_tokens=torch.zeros(1, 1, 2, 3),
            class_present=torch.ones(1, 1, dtype=torch.bool),
            class_mass=torch.ones(1, 1),
            token_class_ids=torch.zeros(1, 1, dtype=torch.long),
            token_class_confidence=torch.ones(1, 1),
        )
        changed_local_bank = CrossV5TissueBank(
            prototypes=base_bank.prototypes.clone(),
            local_tokens=torch.randn(1, 1, 2, 3) * 100.0,
            class_present=base_bank.class_present,
            class_mass=base_bank.class_mass,
            token_class_ids=base_bank.token_class_ids,
            token_class_confidence=base_bank.token_class_confidence,
        )
        modulator = CrossV5SpatialAdaLNModulator(
            hidden_dim=4,
            prototype_dim=4,
            structure_dim=3,
            output_init_std=0.01,
        )

        output_a = modulator(
            hidden_states=hidden,
            target_class_ids=torch.zeros(1, 2, dtype=torch.long),
            bank=base_bank,
            target_structure_tokens=structure,
        )
        output_b = modulator(
            hidden_states=hidden,
            target_class_ids=torch.zeros(1, 2, dtype=torch.long),
            bank=changed_local_bank,
            target_structure_tokens=structure,
        )

        self.assertTrue(torch.allclose(output_a.hidden_states, output_b.hidden_states))
        self.assertTrue(torch.allclose(output_a.gamma, output_b.gamma))

    def test_structure_token_builder_defaults_to_class_probs_and_coordinates(self):
        class_ids = torch.tensor([[[0, 1], [0, 1]]])

        tokens = build_cross_v5_spatial_structure_tokens(
            class_ids=class_ids,
            num_classes=2,
            token_height=2,
            token_width=2,
        )

        self.assertEqual(tuple(tokens.shape), (1, 4, 4))
        self.assertFalse(torch.allclose(tokens[:, 0], tokens[:, -1]))

    def test_structure_token_builder_can_pool_geometry_when_explicitly_requested(self):
        class_ids = torch.tensor([[[1, 1], [1, 1]]])
        geometry = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])

        tokens = build_cross_v5_spatial_structure_tokens(
            class_ids=class_ids,
            num_classes=2,
            token_height=2,
            token_width=2,
            geometry_maps=geometry,
        )

        self.assertEqual(tuple(tokens.shape), (1, 4, 5))
        self.assertFalse(torch.allclose(tokens[:, 0], tokens[:, -1]))

    def test_masked_gram_matrix_uses_only_selected_region(self):
        features = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[2.0, 0.0], [0.0, 0.0]],
            ]
        )
        mask = torch.tensor([[1, 0], [0, 0]])

        gram = masked_gram_matrix(features, mask)

        expected = torch.tensor([[0.5, 1.0], [1.0, 2.0]])
        self.assertTrue(torch.allclose(gram, expected))

    def test_appearance_loss_combines_color_and_masked_texture(self):
        prediction = torch.zeros(1, 3, 4, 4)
        reference = torch.zeros(1, 3, 4, 4)
        prediction[:, :, :2, :] = 0.25
        reference[:, :, :2, :] = 0.75
        prediction[:, :, 2:, :] = 0.2
        reference[:, :, 2:, :] = 0.2
        target_mask = torch.tensor(
            [[[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]]]
        )
        reference_mask = target_mask.clone()
        pred_features = {"relu1_2": prediction[:, :1]}
        ref_features = {"relu1_2": reference[:, :1]}

        result = cross_v5_appearance_fidelity_loss(
            prediction=prediction,
            reference=reference,
            target_tissue_mask=target_mask,
            reference_tissue_mask=reference_mask,
            prediction_vgg_features=pred_features,
            reference_vgg_features=ref_features,
            config=CrossV5AppearanceLossConfig(
                min_pixels=2,
                color_space="rgb",
                color_weight=2.0,
                texture_weight=1.0,
            ),
        )

        self.assertEqual(result["regions"], 2)
        self.assertEqual(result["texture_regions"], 2)
        self.assertGreater(result["color"].item(), 0.0)
        self.assertGreater(result["texture"].item(), 0.0)
        self.assertGreater(result["total"].item(), 0.0)

    def test_geometry_consistency_loss_accepts_dense_logits_and_maps(self):
        tissue_logits = torch.zeros(1, 3, 4, 4)
        tissue_logits[:, 1, :2, :] = 4.0
        tissue_logits[:, 2, 2:, :] = 4.0
        target_tissue = torch.tensor(
            [[[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]]]
        )
        binary_logits = torch.zeros(1, 1, 4, 4)
        binary_logits[:, :, 1:3, 1:3] = 3.0
        target_binary = torch.zeros(1, 4, 4)
        target_binary[:, 1:3, 1:3] = 1.0
        dense_prediction = {"distance": torch.zeros(1, 1, 2, 2)}
        dense_target = {"distance": torch.ones(1, 1, 4, 4)}

        result = cross_v5_geometry_consistency_loss(
            tissue_logits=tissue_logits,
            target_tissue_mask=target_tissue,
            nuclei_binary_logits=binary_logits,
            target_nuclei_binary=target_binary,
            dense_predictions=dense_prediction,
            dense_targets=dense_target,
            config=CrossV5GeometryConsistencyLossConfig(
                nuclei_ce_weight=0.0,
                nuclei_dice_weight=0.0,
            ),
        )

        self.assertGreaterEqual(result["total"].item(), 0.0)
        self.assertGreaterEqual(result["tissue_ce"].item(), 0.0)
        self.assertGreaterEqual(result["nuclei_binary_bce"].item(), 0.0)
        self.assertEqual(result["dense_terms"], 1)
        self.assertTrue(torch.allclose(result["dense_l1"], torch.tensor(1.0)))

    def test_v5_glue_assembles_losses_and_keeps_prediction_grad(self):
        prediction = torch.zeros(1, 3, 4, 4, requires_grad=True)
        reference = torch.ones(1, 3, 4, 4) * 0.5
        mask = torch.ones(1, 4, 4, dtype=torch.long)
        context = CrossV5StepContext(
            prediction_rgb=prediction,
            reference_rgb=reference,
            target_tissue_mask=mask,
            reference_tissue_mask=mask,
        )

        bundle = assemble_cross_v5_losses(
            denoise_loss=torch.tensor(0.25, requires_grad=True),
            context=context,
            weights=CrossV5LossWeights(geometry=0.0),
            appearance_config=CrossV5AppearanceLossConfig(min_pixels=1, color_space="rgb"),
        )
        bundle.total.backward()

        self.assertGreater(bundle.total.item(), 0.0)
        self.assertGreater(prediction.grad.detach().abs().mean().item(), 0.0)

    def test_v5_step_wrapper_gates_geometry_by_interval_and_timestep(self):
        class CountingPredictor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, image):
                self.calls += 1
                return {"nuclei_binary_logits": image[:, :1] * 2.0}

        prediction = torch.zeros(1, 3, 4, 4, requires_grad=True)
        reference = torch.ones(1, 3, 4, 4)
        mask = torch.ones(1, 4, 4, dtype=torch.long)
        nuclei = torch.zeros(1, 4, 4)
        context = CrossV5StepContext(
            prediction_rgb=prediction,
            reference_rgb=reference,
            target_tissue_mask=mask,
            reference_tissue_mask=mask,
            target_nuclei_binary=nuclei,
        )
        predictor = freeze_predictor_for_v5_loss(CountingPredictor())
        intervals = CrossV5LossIntervals(
            appearance=0,
            geometry=2,
            geometry_timestep_max=300.0,
        )

        skipped = assemble_cross_v5_step_losses(
            denoise_loss=torch.tensor(0.1, requires_grad=True),
            context=context,
            weights=CrossV5LossWeights(appearance=0.0, geometry=1.0),
            global_step=2,
            timestep=torch.tensor([900.0]),
            intervals=intervals,
            geometry_predictor=predictor,
        )
        self.assertEqual(predictor.calls, 0)
        self.assertEqual(skipped.components["gate_geometry"], 0)

        ran = assemble_cross_v5_step_losses(
            denoise_loss=torch.tensor(0.1, requires_grad=True),
            context=context,
            weights=CrossV5LossWeights(appearance=0.0, geometry=1.0),
            global_step=4,
            timestep=torch.tensor([100.0]),
            intervals=intervals,
            geometry_predictor=predictor,
        )
        self.assertEqual(predictor.calls, 1)
        self.assertEqual(ran.components["gate_geometry"], 1)
        self.assertIn("geometry_total", ran.components)

    def test_latent_decode_bridge_reconstructs_x0_and_keeps_gradients(self):
        class ToyVAE(torch.nn.Module):
            class Config:
                scaling_factor = 2.0
                shift_factor = 0.25

            config = Config()

            def decode(self, latents, return_dict=False):
                self.last_latents = latents
                return (latents * 2.0,)

        noisy = torch.ones(1, 1, 2, 2, requires_grad=True)
        prediction = torch.full((1, 1, 2, 2), 0.5, requires_grad=True)
        x0 = reconstruct_cross_v5_x0_latents(
            noisy_latents=noisy,
            model_prediction=prediction,
            sigma=torch.tensor([0.2]),
            prediction_type="velocity",
        )
        self.assertTrue(torch.allclose(x0, torch.full_like(x0, 0.9)))

        rgb = decode_cross_v5_prediction_rgb(
            vae=ToyVAE(),
            noisy_latents=noisy,
            model_prediction=prediction,
            sigma=torch.tensor([0.2]),
            config=CrossV5LatentDecodeConfig(prediction_type="velocity"),
        )
        rgb.mean().backward()

        self.assertTrue(rgb.requires_grad)
        self.assertGreater(noisy.grad.detach().abs().mean().item(), 0.0)
        self.assertGreater(prediction.grad.detach().abs().mean().item(), 0.0)

    def test_predictor_bridge_freezes_params_but_keeps_rgb_grad(self):
        class ToyPredictor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 1, kernel_size=1, bias=False)

            def forward(self, x):
                return {"nuclei_binary_logits": self.conv(x)}

        predictor = freeze_predictor_for_v5_loss(ToyPredictor())
        image = torch.rand(1, 3, 4, 4, requires_grad=True)

        metrics = validate_cross_v5_predictor_grad_bridge(
            predictor=predictor,
            prediction_rgb=image,
        )

        self.assertFalse(any(param.requires_grad for param in predictor.parameters()))
        self.assertGreater(metrics["rgb_grad_abs_mean"], 0.0)

    def test_adaln_hook_installer_and_adapter_change_hidden_by_bank(self):
        class V5ReadyBlock(CrossV5AdaLNAdapterMixin, torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, hidden, *, target_class_ids, bank):
                return self._apply_cross_v5_adaln(hidden, target_class_ids=target_class_ids, bank=bank)

        class ToyTransformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = torch.nn.ModuleList([V5ReadyBlock(), V5ReadyBlock()])

        transformer = ToyTransformer()
        modulator = CrossV5AdaLNModulator(hidden_dim=4, output_init_std=0.01)
        bank_a = CrossV5TissueBank(
            prototypes=torch.zeros(1, 2, 4),
            local_tokens=torch.zeros(1, 2, 1, 4),
            class_present=torch.ones(1, 2, dtype=torch.bool),
            class_mass=torch.ones(1, 2),
            token_class_ids=torch.zeros(1, 1, dtype=torch.long),
            token_class_confidence=torch.ones(1, 1),
        )
        bank_b = CrossV5TissueBank(
            prototypes=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]),
            local_tokens=torch.zeros(1, 2, 1, 4),
            class_present=torch.ones(1, 2, dtype=torch.bool),
            class_mass=torch.ones(1, 2),
            token_class_ids=torch.zeros(1, 1, dtype=torch.long),
            token_class_confidence=torch.ones(1, 1),
        )

        summary = install_cross_v5_adaln_hooks(
            transformer=transformer,
            modulator=modulator,
            spec=CrossV5AdaLNHookSpec(block_indices=(-1,), hook_point="post_norm_hidden"),
        )

        self.assertEqual(summary.installed_block_indices, (1,))
        hidden = torch.zeros(1, 3, 4)
        class_ids = torch.tensor([[0, 1, 0]])
        out_a = transformer.transformer_blocks[1](hidden, target_class_ids=class_ids, bank=bank_a)
        out_b = transformer.transformer_blocks[1](hidden, target_class_ids=class_ids, bank=bank_b)
        self.assertFalse(torch.allclose(out_a, hidden))
        self.assertFalse(torch.allclose(out_a, out_b))

    def test_adaln_hook_installer_rejects_zero_initialized_gamma_path(self):
        class V5ReadyBlock(CrossV5AdaLNAdapterMixin, torch.nn.Module):
            pass

        transformer = torch.nn.Module()
        transformer.transformer_blocks = torch.nn.ModuleList([V5ReadyBlock()])
        modulator = CrossV5AdaLNModulator(hidden_dim=4, output_init_std=0.01)
        with torch.no_grad():
            final = modulator.mlp[-1]
            final.weight[:4].zero_()
            final.bias[:4].zero_()

        with self.assertRaisesRegex(ValueError, "zero-initialized"):
            install_cross_v5_adaln_hooks(
                transformer=transformer,
                modulator=modulator,
                spec=CrossV5AdaLNHookSpec(block_indices=(0,), require_nonzero_gamma=True),
            )

    def test_pairing_policy_normalizes_sampling_targets(self):
        policy = CrossV5PairingPolicy()

        self.assertAlmostEqual(sum(policy.normalized_pair_mode_weights().values()), 1.0)
        self.assertAlmostEqual(sum(policy.normalized_coverage_weights().values()), 1.0)

    def test_cross_v5_pairing_sampler_uses_gap_coverage_and_bank_dropout(self):
        records = [
            {
                "sample_id": "target",
                "reference_sample_id": "same_low_gap",
                "case_id": "wsi_a",
                "reference_case_id": "wsi_a",
                "pair_difficulty": "full",
                "appearance_gap": 0.1,
                "covered_target_tissue_ids": [1, 2],
            },
            {
                "sample_id": "target",
                "reference_sample_id": "cross_partial",
                "case_id": "wsi_a",
                "reference_case_id": "wsi_b",
                "pair_difficulty": "partial",
                "appearance_gap": 0.4,
                "covered_target_tissue_ids": [1],
            },
            {
                "sample_id": "target",
                "reference_sample_id": "cross_high_full",
                "case_id": "wsi_a",
                "reference_case_id": "wsi_c",
                "pair_difficulty": "full",
                "appearance_gap": 0.9,
                "covered_target_tissue_ids": [1, 2],
            },
        ]
        sampler = CrossV5PairingSampler(
            records,
            policy=CrossV5PairingPolicy(
                same_wsi_fraction=0.0,
                cross_wsi_fraction=0.0,
                high_appearance_gap_fraction=1.0,
                full_coverage_fraction=1.0,
                partial_coverage_fraction=0.0,
                low_coverage_fraction=0.0,
                class_bank_dropout_prob=1.0,
            ),
            seed=3,
        )

        sampled = sampler.sample()

        self.assertEqual(sampled["reference_sample_id"], "cross_high_full")
        self.assertEqual(sampled["v5_pair_mode"], "high_appearance_gap")
        self.assertEqual(sampled["v5_coverage_mode"], "full")
        self.assertEqual(len(sampled["v5_reference_bank_keep_tissue_ids"]), 1)
        self.assertEqual(len(sampled["v5_reference_bank_drop_tissue_ids"]), 1)

    def test_cross_v5_branch_gate_uses_step_and_timestep(self):
        self.assertTrue(should_run_cross_v5_branch(global_step=10, interval=5))
        self.assertFalse(should_run_cross_v5_branch(global_step=11, interval=5))
        self.assertFalse(should_run_cross_v5_branch(global_step=10, interval=0))
        self.assertTrue(
            should_run_cross_v5_branch(
                global_step=10,
                interval=5,
                timestep=torch.tensor([400.0, 600.0]),
                timestep_min=300.0,
                timestep_max=700.0,
            )
        )
        self.assertFalse(
            should_run_cross_v5_branch(
                global_step=10,
                interval=5,
                timestep=torch.tensor([900.0]),
                timestep_min=300.0,
                timestep_max=700.0,
            )
        )

    def test_flux_adapters_patch_double_and_single_post_norm_points(self):
        class ToyAdaNorm(torch.nn.Module):
            def forward(self, hidden, emb):
                batch, _, dim = hidden.shape
                gate = torch.ones(batch, dim, device=hidden.device, dtype=hidden.dtype)
                shift = torch.zeros_like(gate)
                scale = torch.zeros_like(gate)
                return hidden, gate, shift, scale, gate

        class ToySingleNorm(torch.nn.Module):
            def forward(self, hidden, emb):
                batch, _, dim = hidden.shape
                return hidden, torch.ones(batch, dim, device=hidden.device, dtype=hidden.dtype)

        class ToyAttention(torch.nn.Module):
            def __init__(self, double=False):
                super().__init__()
                self.double = double

            def forward(self, hidden_states, encoder_hidden_states=None, image_rotary_emb=None, **kwargs):
                self.seen_kwargs = dict(kwargs)
                self.assert_no_cross_keys = not any(key.startswith("cross_v5_") for key in kwargs)
                if self.double:
                    return hidden_states * 0.0, encoder_hidden_states * 0.0
                return hidden_states * 0.0

        class ToyDoubleBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = ToyAdaNorm()
                self.norm1_context = ToyAdaNorm()
                self.attn = ToyAttention(double=True)
                self.norm2 = torch.nn.Identity()
                self.ff = torch.nn.Identity()
                self.norm2_context = torch.nn.Identity()
                self.ff_context = torch.nn.Identity()

            def forward(
                self,
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb=None,
                joint_attention_kwargs=None,
            ):
                kwargs = dict(joint_attention_kwargs or {})
                attn_output, context_attn_output = self.attn(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                    **kwargs,
                )
                return encoder_hidden_states + context_attn_output, hidden_states + attn_output

        class ToySingleBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = ToySingleNorm()
                self.proj_mlp = torch.nn.Linear(4, 4)
                self.act_mlp = torch.nn.Identity()
                self.attn = ToyAttention(double=False)
                self.proj_out = torch.nn.Linear(8, 4)

        class ToyTransformer(torch.nn.Module):
            def __init__(self, double_blocks=1):
                super().__init__()
                self.transformer_blocks = torch.nn.ModuleList([ToyDoubleBlock() for _ in range(int(double_blocks))])
                self.single_transformer_blocks = torch.nn.ModuleList([ToySingleBlock()])

        transformer = ToyTransformer(double_blocks=2)
        modulator = CrossV5SpatialAdaLNModulator(
            hidden_dim=4,
            prototype_dim=4,
            structure_dim=3,
            output_init_std=0.01,
        )
        summary = install_cross_v5_flux_adaln_adapters(
            transformer=transformer,
            modulator=modulator,
            double_block_indices=(0,),
            single_block_indices=(0,),
        )
        self.assertEqual(summary.double_blocks, (0,))
        self.assertEqual(summary.double_strip_blocks, (1,))
        self.assertEqual(summary.single_blocks, (0,))

        bank = CrossV5TissueBank(
            prototypes=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]),
            local_tokens=torch.zeros(1, 2, 1, 4),
            class_present=torch.ones(1, 2, dtype=torch.bool),
            class_mass=torch.ones(1, 2),
            token_class_ids=torch.zeros(1, 1, dtype=torch.long),
            token_class_confidence=torch.ones(1, 1),
        )
        kwargs = {
            CROSS_V5_TARGET_CLASS_IDS_KEY: torch.tensor([[0, 1, 0]]),
            CROSS_V5_TARGET_STRUCTURE_TOKENS_KEY: torch.tensor([[[1.0, 0.0, -1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]]]),
            CROSS_V5_BANK_KEY: bank,
        }
        hidden = torch.zeros(1, 3, 4)
        encoder = torch.zeros(1, 2, 4)
        _, double_out = transformer.transformer_blocks[0](
            hidden_states=hidden,
            encoder_hidden_states=encoder,
            temb=torch.zeros(1, 4),
            joint_attention_kwargs=kwargs,
        )
        self.assertFalse(torch.allclose(double_out, hidden))
        self.assertTrue(transformer.transformer_blocks[0].attn.assert_no_cross_keys)
        _, stripped_double_out = transformer.transformer_blocks[1](
            hidden_states=hidden,
            encoder_hidden_states=encoder,
            temb=torch.zeros(1, 4),
            joint_attention_kwargs=kwargs,
        )
        self.assertEqual(tuple(stripped_double_out.shape), tuple(hidden.shape))
        self.assertTrue(transformer.transformer_blocks[1].attn.assert_no_cross_keys)
        self.assertIsNone(getattr(transformer.transformer_blocks[1], "cross_v5_adaln_modulator", None))
        with self.assertRaisesRegex(ValueError, "missing"):
            transformer.transformer_blocks[0](
                hidden_states=hidden,
                encoder_hidden_states=encoder,
                temb=torch.zeros(1, 4),
                joint_attention_kwargs={},
            )

        single_hidden = torch.zeros(1, 5, 4)
        single_kwargs = dict(kwargs)
        single_kwargs[CROSS_V5_IMAGE_TOKEN_START_KEY] = 2
        single_out = transformer.single_transformer_blocks[0](
            hidden_states=single_hidden,
            temb=torch.zeros(1, 4),
            joint_attention_kwargs=single_kwargs,
        )
        self.assertEqual(tuple(single_out.shape), (1, 5, 4))
        self.assertTrue(transformer.single_transformer_blocks[0].attn.assert_no_cross_keys)
        single_encoder_out, single_image_out = transformer.single_transformer_blocks[0](
            hidden_states=torch.zeros(1, 3, 4),
            encoder_hidden_states=torch.zeros(1, 2, 4),
            temb=torch.zeros(1, 4),
            joint_attention_kwargs=kwargs,
        )
        self.assertEqual(tuple(single_encoder_out.shape), (1, 2, 4))
        self.assertEqual(tuple(single_image_out.shape), (1, 3, 4))

        strip_transformer = ToyTransformer(double_blocks=2)
        strip_summary = install_cross_v5_flux_adaln_adapters(
            transformer=strip_transformer,
            modulator=modulator,
            double_block_indices=(0,),
            single_block_indices=(),
        )
        self.assertEqual(strip_summary.double_blocks, (0,))
        self.assertEqual(strip_summary.double_strip_blocks, (1,))
        self.assertEqual(strip_summary.single_blocks, ())
        self.assertEqual(strip_summary.single_strip_blocks, (0,))
        strip_single_kwargs = dict(kwargs)
        strip_single_kwargs[CROSS_V5_IMAGE_TOKEN_START_KEY] = 2
        strip_single_out = strip_transformer.single_transformer_blocks[0](
            hidden_states=torch.zeros(1, 5, 4),
            temb=torch.zeros(1, 4),
            joint_attention_kwargs=strip_single_kwargs,
        )
        self.assertEqual(tuple(strip_single_out.shape), (1, 5, 4))
        self.assertTrue(strip_transformer.single_transformer_blocks[0].attn.assert_no_cross_keys)
        self.assertIsNone(getattr(strip_transformer.single_transformer_blocks[0], "cross_v5_adaln_modulator", None))


if __name__ == "__main__":
    unittest.main()

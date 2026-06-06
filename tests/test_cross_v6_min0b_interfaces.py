import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from controlnet_train.modules.cross_v6_conditioning import (
        CrossV6Min0bComposerCondition,
        CrossV6Min0bControlCondition,
        CrossV6Min0bControlSpec,
        CrossV6Min0bLatentComposer,
        build_cross_v6_control_condition_tensor,
        build_cross_v6_geometry_maps,
        diagnose_cross_v6_vae_class_internal_variance,
        masked_pool_cross_v6_latent_by_probs,
        resize_cross_v6_class_mask_to_probs,
    )


@unittest.skipIf(torch is None, "torch is required for Cross V6 tests")
class CrossV6Min0bInterfaceTests(unittest.TestCase):
    def _condition(self, *, ref_latent=None, ref_tissue=None, target_tissue=None):
        if ref_latent is None:
            ref_latent = torch.tensor(
                [
                    [
                        [[1.0, 3.0], [10.0, 30.0]],
                        [[2.0, 4.0], [20.0, 40.0]],
                    ]
                ]
            )
        if ref_tissue is None:
            ref_tissue = torch.tensor([[[1, 1], [2, 2]]])
        if target_tissue is None:
            target_tissue = torch.tensor([[[1, 2], [1, 2]]])
        target_nuclei = torch.tensor([[[0, 1], [0, 1]]])
        nuclei_binary, nuclei_boundary, nuclei_distance, tissue_boundary = build_cross_v6_geometry_maps(
            target_tissue_mask=target_tissue,
            target_nuclei_mask=target_nuclei,
            output_height=2,
            output_width=2,
        )
        return CrossV6Min0bComposerCondition(
            ref_latent=ref_latent,
            ref_tissue_mask=ref_tissue,
            ref_nuclei_mask=torch.tensor([[[0, 1], [0, 1]]]),
            target_tissue_mask=target_tissue,
            target_nuclei_mask=target_nuclei,
            nuclei_binary=nuclei_binary,
            nuclei_boundary=nuclei_boundary,
            nuclei_distance_map=nuclei_distance,
            tissue_boundary=tissue_boundary,
        )

    def test_control_spec_exposes_reference_latent_plus_six_scalar_maps_only(self):
        spec = CrossV6Min0bControlSpec(latent_channels=16)

        self.assertEqual(spec.raw_channels, 22)
        self.assertEqual(spec.packed_channels, 88)
        self.assertEqual(
            spec.condition_order,
            (
                "z_ref_to_target",
                "nuclei_binary",
                "nuclei_boundary",
                "nuclei_distance_map",
                "tissue_boundary",
                "retrieval_confidence_map",
                "missing_class_map",
            ),
        )

    def test_masked_pooling_routes_reference_latent_by_target_class(self):
        latent = torch.tensor(
            [
                [
                    [[1.0, 3.0], [10.0, 30.0]],
                    [[2.0, 4.0], [20.0, 40.0]],
                ]
            ]
        )
        class_probs = resize_cross_v6_class_mask_to_probs(
            torch.tensor([[[1, 1], [2, 2]]]),
            num_classes=3,
            output_height=2,
            output_width=2,
        )

        proto, mass, present = masked_pool_cross_v6_latent_by_probs(latent, class_probs)

        self.assertTrue(torch.allclose(proto[0, 1], torch.tensor([2.0, 3.0])))
        self.assertTrue(torch.allclose(proto[0, 2], torch.tensor([20.0, 30.0])))
        self.assertTrue(torch.equal(mass[0], torch.tensor([0.0, 2.0, 2.0])))
        self.assertTrue(torch.equal(present[0], torch.tensor([False, True, True])))

    def test_composer_pooling_gamma_attention_and_nuclei_residual_shapes(self):
        condition = self._condition()
        composer = CrossV6Min0bLatentComposer(
            latent_channels=2,
            num_tissue_classes=3,
            attention_dim=4,
            max_ref_tokens_per_class=2,
            gamma_init=1e-2,
            alpha=0.25,
        )

        out = composer(condition)

        self.assertEqual(tuple(out.z_tissue_pool.shape), (1, 2, 2, 2))
        self.assertEqual(tuple(out.z_tissue_attn.shape), (1, 2, 2, 2))
        self.assertEqual(tuple(out.gamma.shape), (1, 2, 1, 1))
        self.assertEqual(tuple(out.z_ref_to_target.shape), (1, 2, 2, 2))
        self.assertEqual(tuple(out.retrieval_confidence_map.shape), (1, 1, 2, 2))
        self.assertTrue(torch.allclose(out.gamma.detach(), torch.full((1, 2, 1, 1), 1e-2)))
        self.assertGreater(float((out.z_tissue_attn - out.z_tissue_pool).abs().sum()), 0.0)

    def test_attention_uses_per_position_queries_not_one_query_per_class(self):
        condition = self._condition(
            ref_latent=torch.tensor(
                [
                    [
                        [[1.0, 4.0], [8.0, 10.0]],
                        [[0.0, 1.0], [2.0, 3.0]],
                    ]
                ]
            ),
            ref_tissue=torch.tensor([[[1, 1], [1, 1]]]),
            target_tissue=torch.tensor([[[1, 1], [1, 1]]]),
        )
        composer = CrossV6Min0bLatentComposer(
            latent_channels=2,
            num_tissue_classes=2,
            attention_dim=4,
            max_ref_tokens_per_class=4,
            gamma_init=1e-2,
            alpha=0.0,
        )

        out = composer(condition)
        top_left = out.z_tissue_attn[:, :, 0, 0]
        bottom_right = out.z_tissue_attn[:, :, 1, 1]

        self.assertFalse(torch.allclose(top_left, bottom_right))

    def test_empty_reference_bank_falls_back_to_pooling_and_marks_missing(self):
        condition = self._condition(
            ref_tissue=torch.tensor([[[1, 1], [1, 1]]]),
            target_tissue=torch.tensor([[[1, 2], [1, 2]]]),
        )
        composer = CrossV6Min0bLatentComposer(
            latent_channels=2,
            num_tissue_classes=3,
            attention_dim=4,
            max_ref_tokens_per_class=2,
            gamma_init=1e-2,
            alpha=0.0,
        )

        out = composer(condition)

        self.assertTrue(torch.equal(out.retrieval_confidence_map[0, 0], torch.tensor([[1.0, 0.0], [1.0, 0.0]])))
        self.assertTrue(torch.equal(out.missing_class_map[0, 0], torch.tensor([[0.0, 1.0], [0.0, 1.0]])))
        self.assertTrue(torch.allclose(out.z_tissue_attn[:, :, 0, 1], out.z_tissue_pool[:, :, 0, 1]))

    def test_final_condition_concatenates_without_target_onehot(self):
        z = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
        maps = [torch.ones(1, 1, 2, 2) * value for value in range(1, 7)]
        condition = CrossV6Min0bControlCondition(
            z_ref_to_target=z,
            nuclei_binary=maps[0],
            nuclei_boundary=maps[1],
            nuclei_distance_map=maps[2],
            tissue_boundary=maps[3],
            retrieval_confidence_map=maps[4],
            missing_class_map=maps[5],
        )

        final = build_cross_v6_control_condition_tensor(condition, normalize_z=False)

        self.assertEqual(tuple(final.shape), (1, 8, 2, 2))
        self.assertTrue(torch.equal(final[:, :2], z))
        self.assertTrue(torch.equal(final[:, 2], maps[0][:, 0]))
        self.assertTrue(torch.equal(final[:, -1], maps[-1][:, 0]))

    def test_vae_class_internal_variance_diagnostic_reports_texture_variation(self):
        ref_latent = torch.tensor(
            [
                [
                    [[1.0, 3.0], [5.0, 7.0]],
                    [[2.0, 4.0], [6.0, 8.0]],
                ]
            ]
        )
        diag = diagnose_cross_v6_vae_class_internal_variance(
            ref_latent=ref_latent,
            ref_tissue_mask=torch.tensor([[[1, 1], [1, 1]]]),
            num_classes=2,
        )

        self.assertEqual(tuple(diag.class_mass.shape), (1, 2))
        self.assertGreater(float(diag.token_variance[0, 1]), 0.0)
        self.assertGreater(float(diag.mean_pairwise_distance[0, 1]), 0.0)
        self.assertGreaterEqual(float(diag.pca_top1_energy[0, 1]), 0.0)
        self.assertLessEqual(float(diag.pca_top1_energy[0, 1]), 1.0)


if __name__ == "__main__":
    unittest.main()

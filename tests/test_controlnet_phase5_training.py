import unittest

import torch
import torch.nn as nn

from controlnet_train.modules.conditioning import build_inpaint_condition
from controlnet_train.training.conditioning import (
    CrossV0ControlSpec,
    InpaintControlSpec,
    patch_controlnet_x_embedder,
)
from controlnet_train.cli.train_controlnet_flux_cross import parse_args as parse_cross_args
from controlnet_train.cli.train_controlnet_flux_inpaint import parse_args as parse_inpaint_args


class InpaintConditioningTests(unittest.TestCase):
    def test_build_inpaint_condition_uses_plan_channel_order(self):
        source_image = torch.randn(2, 16, 8, 8)
        target_tissue = torch.randn(2, 64, 8, 8)
        target_nuclei = torch.randn(2, 16, 8, 8)
        change_mask = torch.randn(2, 4, 8, 8)

        out = build_inpaint_condition(
            source_image_latent=source_image,
            target_tissue_feat=target_tissue,
            target_nuclei_feat=target_nuclei,
            change_mask_feat=change_mask,
        )

        self.assertEqual(out.shape, (2, 100, 8, 8))
        self.assertTrue(torch.equal(out[:, :16], source_image))
        self.assertTrue(torch.equal(out[:, 16:80], target_tissue))
        self.assertTrue(torch.equal(out[:, 80:96], target_nuclei))
        self.assertTrue(torch.equal(out[:, 96:100], change_mask))


class ControlSpecTests(unittest.TestCase):
    def test_default_channel_specs_match_phase5_plan(self):
        self.assertEqual(InpaintControlSpec().raw_channels, 100)
        self.assertEqual(InpaintControlSpec().packed_channels, 400)
        self.assertEqual(CrossV0ControlSpec().raw_channels, 176)
        self.assertEqual(CrossV0ControlSpec().packed_channels, 704)

    def test_patch_controlnet_x_embedder_expands_to_packed_channels(self):
        controlnet = nn.Module()
        controlnet.controlnet_x_embedder = nn.Linear(64, 32)
        original_weight = controlnet.controlnet_x_embedder.weight.detach().clone()
        original_bias = controlnet.controlnet_x_embedder.bias.detach().clone()

        patched = patch_controlnet_x_embedder(controlnet, packed_control_channels=400)

        self.assertIs(patched, controlnet)
        self.assertEqual(controlnet.controlnet_x_embedder.in_features, 400)
        self.assertTrue(torch.equal(controlnet.controlnet_x_embedder.weight[:, :64], original_weight))
        self.assertTrue(torch.equal(controlnet.controlnet_x_embedder.bias, original_bias))
        self.assertTrue(torch.equal(controlnet.controlnet_x_embedder.weight[:, 64:], torch.zeros(32, 336)))


class TrainingCliTests(unittest.TestCase):
    def test_inpaint_cli_accepts_phase5_metadata_arguments(self):
        args = parse_inpaint_args(
            [
                "--pretrained_model_name_or_path",
                "flux-dev",
                "--train-metadata",
                "phase5_runs/inpaint_meta/metadata_inpaint_train.jsonl",
            ]
        )

        self.assertEqual(args.train_metadata, "phase5_runs/inpaint_meta/metadata_inpaint_train.jsonl")
        self.assertEqual(args.tissue_embedding_dim, 64)
        self.assertEqual(args.nuclei_embedding_dim, 16)
        self.assertEqual(args.nuclei_out_channels, 16)

    def test_cross_cli_accepts_phase5_metadata_arguments(self):
        args = parse_cross_args(
            [
                "--pretrained_model_name_or_path",
                "flux-dev",
                "--train-metadata",
                "phase5_runs/cross_meta/metadata_cross_train.json",
            ]
        )

        self.assertEqual(args.train_metadata, "phase5_runs/cross_meta/metadata_cross_train.json")
        self.assertEqual(args.cross_version, "v0")
        self.assertEqual(args.prompt_source, "dataset")
        self.assertIsNone(args.prompt)
        self.assertEqual(args.tissue_embedding_dim, 64)
        self.assertEqual(args.nuclei_out_channels, 16)

    def test_cross_cli_accepts_prompt_override(self):
        args = parse_cross_args(
            [
                "--pretrained_model_name_or_path",
                "flux-dev",
                "--train-metadata",
                "phase5_runs/cross_meta/metadata_cross_train.json",
                "--prompt-source",
                "metadata",
                "--prompt",
                "H&E stained pathology",
            ]
        )

        self.assertEqual(args.prompt_source, "metadata")
        self.assertEqual(args.prompt, "H&E stained pathology")


if __name__ == "__main__":
    unittest.main()

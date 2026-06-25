import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from controlnet_train.postprocess.masked_gatys import (
        GatysTransferConfig,
        MaskedGatysStyleTransfer,
        NUCLEI_STAIN_LABEL_OFFSET,
        masked_style_loss,
        parse_region_labels,
        precompute_reference_grams,
        precompute_pooled_reference_grams,
        overlay_nuclei_on_tissue_mask,
        region_gram,
        resize_label_mask,
    )

_CLI_PATH = Path(__file__).resolve().parents[1] / "controlnet_train" / "cli" / "run_masked_gatys_transfer.py"
_CLI_SPEC = importlib.util.spec_from_file_location("run_masked_gatys_transfer", _CLI_PATH)
gatys_cli = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(gatys_cli)


@unittest.skipIf(torch is None, "torch is required for masked Gatys tests")
class MaskedGatysPostprocessTests(unittest.TestCase):
    def test_region_gram_normalizes_by_channels_and_selected_pixels(self):
        features = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 0.0], [1.0, 3.0]],
            ]
        )
        mask = torch.tensor([[True, False], [False, True]])

        gram = region_gram(features, mask)

        selected = torch.tensor([[1.0, 4.0], [2.0, 3.0]])
        expected = (selected @ selected.T) / 4.0
        self.assertTrue(torch.allclose(gram, expected))

    def test_resize_label_mask_uses_nearest_labels(self):
        mask = torch.tensor([[[1, 2], [3, 4]]])

        resized = resize_label_mask(mask, (4, 4))

        self.assertEqual(resized.shape, (1, 4, 4))
        self.assertEqual(int(resized[0, 0, 0]), 1)
        self.assertEqual(int(resized[0, 0, 3]), 2)
        self.assertEqual(int(resized[0, 3, 0]), 3)
        self.assertEqual(int(resized[0, 3, 3]), 4)

    def test_masked_style_loss_matches_shared_label_regions(self):
        target_features = {"conv1_1": torch.zeros(1, 2, 2, 2)}
        reference_features = {"conv1_1": torch.ones(1, 2, 2, 2)}
        target_mask = torch.tensor([[[1, 1], [2, 2]]])
        reference_mask = target_mask.clone()
        grams = precompute_reference_grams(
            reference_features,
            reference_mask,
            regions=(1, 2),
            layers=("conv1_1",),
            min_region_pixels=1,
        )

        loss, terms = masked_style_loss(
            target_features,
            grams,
            target_mask,
            style_layers=("conv1_1",),
            regions=(1, 2),
            layer_weights={"conv1_1": 1.0},
            min_region_pixels=1,
        )

        self.assertEqual(terms, 2)
        self.assertGreater(loss.item(), 0.0)

    def test_masked_style_loss_ignores_content_layer_even_if_present(self):
        target_features = {
            "conv1_1": torch.zeros(1, 2, 2, 2),
            "conv4_2": torch.full((1, 2, 2, 2), 9.0),
        }
        reference_features = {
            "conv1_1": torch.ones(1, 2, 2, 2),
            "conv4_2": torch.full((1, 2, 2, 2), 3.0),
        }
        target_mask = torch.tensor([[[1, 1], [2, 2]]])
        reference_mask = target_mask.clone()
        grams = precompute_reference_grams(
            reference_features,
            reference_mask,
            regions=(1, 2),
            layers=("conv1_1",),
            min_region_pixels=1,
        )

        loss, terms = masked_style_loss(
            target_features,
            grams,
            target_mask,
            style_layers=("conv1_1",),
            regions=(1, 2),
            layer_weights={"conv1_1": 1.0},
            min_region_pixels=1,
        )

        self.assertEqual(terms, 2)
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(loss.item(), 100.0)

    def test_parse_region_labels_accepts_comma_separated_ids(self):
        self.assertEqual(parse_region_labels("1, 2,5"), (1, 2, 5))
        self.assertIsNone(parse_region_labels(None))

    def test_cli_parser_accepts_core_knobs(self):
        args = gatys_cli.parse_args(
            [
                "--initial-image",
                "i0.png",
                "--target-mask",
                "target.png",
                "--target-nuclei-mask",
                "target_nuclei.png",
                "--reference-image",
                "ref.png",
                "--reference-mask",
                "ref_mask.png",
                "--reference-nuclei-mask",
                "ref_nuclei.png",
                "--output-dir",
                "out",
                "--regions",
                "1,2",
                "--steps",
                "7",
            "--style-weight",
            "1000",
            "--no-content-loss",
            "--vgg-weights",
            "none",
            "--missing-region-fallback",
            "skip",
            "--pre-gatys-color-match",
            "macenko",
            "--color-match-scope",
            "region",
        ]
        )

        self.assertEqual(args.steps, 7)
        self.assertEqual(args.style_weight, 1000)
        self.assertTrue(args.no_content_loss)
        self.assertEqual(args.missing_region_fallback, "skip")
        self.assertEqual(args.target_nuclei_mask, "target_nuclei.png")
        self.assertEqual(args.reference_nuclei_mask, "ref_nuclei.png")
        self.assertEqual(args.pre_gatys_color_match, "macenko")
        self.assertEqual(args.color_match_scope, "region")
        self.assertEqual(parse_region_labels(args.regions), (1, 2))

    def test_overlay_nuclei_on_tissue_mask_offsets_nuclei_labels(self):
        tissue = torch.tensor([[1, 1, 2], [2, 3, 3]])
        nuclei = torch.tensor([[0, 101, 0], [102, 0, 103]])

        result = overlay_nuclei_on_tissue_mask(tissue, nuclei)

        expected = torch.tensor(
            [
                [1, 101 + NUCLEI_STAIN_LABEL_OFFSET, 2],
                [102 + NUCLEI_STAIN_LABEL_OFFSET, 3, 103 + NUCLEI_STAIN_LABEL_OFFSET],
            ]
        )
        self.assertTrue(torch.equal(result, expected))

    def test_tiny_runner_without_content_loss_does_not_need_content_layer(self):
        class StyleOnlyExtractor(torch.nn.Module):
            def forward(self, x):
                return {"conv1_1": x}

        image = torch.zeros(1, 3, 4, 4)
        reference = torch.ones(1, 3, 4, 4)
        mask = torch.ones(1, 4, 4, dtype=torch.long)
        runner = MaskedGatysStyleTransfer(
            StyleOnlyExtractor(),
            GatysTransferConfig(
                steps=1,
                optimizer="adam",
                adam_lr=0.01,
                content_weight=1.0,
                use_content_loss=False,
                style_layers=("conv1_1",),
                content_layer="conv4_2",
                layer_weights={"conv1_1": 1.0},
                device="cpu",
                log_every=1,
                min_region_pixels=1,
                preserve_background=False,
                vgg_weights="none",
            ),
        )

        result = runner.run(
            initial_image=image,
            reference_image=reference,
            target_mask=mask,
            reference_mask=mask,
        )

        self.assertEqual(result.active_regions, (1,))
        self.assertEqual(result.history[-1]["content"], 0.0)

    def test_tiny_runner_writes_image_and_metrics_with_custom_extractor(self):
        class TinyExtractor(torch.nn.Module):
            def forward(self, x):
                return {
                    "conv1_1": x,
                    "conv4_2": torch.nn.functional.avg_pool2d(x, kernel_size=2),
                }

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            image = torch.zeros(1, 3, 4, 4)
            reference = torch.ones(1, 3, 4, 4)
            mask = torch.ones(1, 4, 4, dtype=torch.long)
            runner = MaskedGatysStyleTransfer(
                TinyExtractor(),
                GatysTransferConfig(
                    steps=1,
                    optimizer="adam",
                    adam_lr=0.01,
                    style_layers=("conv1_1",),
                    content_layer="conv4_2",
                    layer_weights={"conv1_1": 1.0},
                    device="cpu",
                    log_every=1,
                    min_region_pixels=1,
                    preserve_background=False,
                    vgg_weights="none",
                ),
            )

            result = runner.run(
                initial_image=image,
                reference_image=reference,
                target_mask=mask,
                reference_mask=mask,
                output_dir=output_dir,
            )

            self.assertTrue(result.output_path.exists())
            self.assertTrue(result.metrics_path.exists())
            self.assertEqual(result.active_regions, (1,))
            self.assertIsInstance(Image.open(result.output_path), Image.Image)

    def test_missing_target_region_can_use_pooled_reference_style(self):
        target_features = {"conv1_1": torch.zeros(1, 2, 2, 2)}
        reference_features = {"conv1_1": torch.ones(1, 2, 2, 2)}
        target_mask = torch.tensor([[[2, 2], [2, 2]]])
        reference_mask = torch.tensor([[[1, 1], [1, 1]]])
        grams = precompute_reference_grams(
            reference_features,
            reference_mask,
            regions=(),
            layers=("conv1_1",),
            min_region_pixels=1,
        )
        pooled = precompute_pooled_reference_grams(
            reference_features,
            reference_mask,
            layers=("conv1_1",),
            background_label=0,
            min_region_pixels=1,
        )
        grams[("conv1_1", 2)] = pooled["conv1_1"]

        loss, terms = masked_style_loss(
            target_features,
            grams,
            target_mask,
            style_layers=("conv1_1",),
            regions=(2,),
            layer_weights={"conv1_1": 1.0},
            min_region_pixels=1,
        )

        self.assertEqual(terms, 1)
        self.assertGreater(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()

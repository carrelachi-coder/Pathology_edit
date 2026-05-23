import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn

from segmentator.config import SEGMENTATOR_CLASSES, SampleRecord
from segmentator.data import IMAGENET_MEAN, IMAGENET_STD, TissueSegmentationDataset, dataset_balanced_weights
from segmentator.losses import segmentation_loss
from segmentator.metrics import segmentation_metrics
from segmentator.model import SimpleFeaturePyramid, UPerLikeDecoder, Uni2hFeatureEncoder


class SegmentatorDataTests(unittest.TestCase):
    def test_segmentator_classes_follow_unified_coarse_label_ids(self):
        self.assertEqual(SEGMENTATOR_CLASSES[3], "necrosis")
        self.assertEqual(SEGMENTATOR_CLASSES[4], "immune_infiltrate")
        self.assertEqual(SEGMENTATOR_CLASSES[5], "normal_epithelium")

    def test_dataset_applies_imagenet_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "sample.png"
            mask_path = root / "sample_mask.png"
            Image.new("RGB", (4, 4), (255, 255, 255)).save(image_path)
            Image.new("L", (4, 4), 1).save(mask_path)

            dataset = TissueSegmentationDataset(
                [SampleRecord(image_path=image_path, mask_path=mask_path, sample_id="sample")],
                image_size=4,
            )

            item = dataset[0]
            image = item["image"]

            self.assertIsInstance(image, torch.Tensor)
            self.assertTrue(torch.allclose(image[:, 0, 0], torch.tensor([(1.0 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)])))
            self.assertFalse(math.isclose(float(image.max()), 1.0))

    def test_dataset_remaps_fine_labels_to_coarse_and_ignores_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "sample.png"
            mask_path = root / "sample_mask.png"
            Image.new("RGB", (3, 1), (255, 255, 255)).save(image_path)
            Image.fromarray(torch.tensor([[8, 14, 99]], dtype=torch.uint8).numpy(), mode="L").save(mask_path)

            dataset = TissueSegmentationDataset(
                [SampleRecord(image_path=image_path, mask_path=mask_path, sample_id="sample", dataset_id="panda")],
                image_size=3,
                ignore_index=255,
            )

            mask = dataset[0]["mask"]

            self.assertEqual(mask.tolist(), [[1, 1, 255]])

    def test_losses_and_metrics_ignore_partial_label_pixels(self):
        logits = torch.zeros(1, 2, 1, 2)
        target = torch.tensor([[[1, 255]]])

        losses = segmentation_loss(logits, target, num_classes=2, invalid_to=255)
        metrics = segmentation_metrics(torch.tensor([[[1, 0]]]), target, num_classes=2, ignore_index=255)

        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertEqual(metrics["per_class"]["class_1"]["support_pixels"], 1)

    def test_dataset_balanced_weights_equalize_dataset_sampling(self):
        root = Path("unused")
        records = [
            SampleRecord(root / "a.png", root / "a.png", "a", dataset_id="big"),
            SampleRecord(root / "b.png", root / "b.png", "b", dataset_id="big"),
            SampleRecord(root / "c.png", root / "c.png", "c", dataset_id="small"),
        ]

        weights = dataset_balanced_weights(records)

        self.assertEqual(weights.tolist(), [0.5, 0.5, 1.0])


class _FakeUniBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.requested_n = None

    def get_intermediate_layers(self, x, *, n, reshape):
        self.requested_n = n
        self.requested_reshape = reshape
        return [torch.zeros(x.shape[0], 1536, 2, 2) for _ in n]


class SegmentatorModelTests(unittest.TestCase):
    def test_uni2h_encoder_uses_spaced_intermediate_layers(self):
        fake_backbone = _FakeUniBackbone()
        with patch("segmentator.model._load_uni2h_model", return_value=fake_backbone):
            encoder = Uni2hFeatureEncoder(local_repo="unused")

        features = encoder(torch.zeros(1, 3, 28, 28))

        self.assertEqual(fake_backbone.requested_n, [5, 11, 17, 23])
        self.assertTrue(fake_backbone.requested_reshape)
        self.assertEqual(len(features), 4)

    def test_uni2h_encoder_can_request_single_intermediate_layer(self):
        fake_backbone = _FakeUniBackbone()
        with patch("segmentator.model._load_uni2h_model", return_value=fake_backbone):
            encoder = Uni2hFeatureEncoder(local_repo="unused", intermediate_layers=(23,))

        features = encoder(torch.zeros(1, 3, 28, 28))

        self.assertEqual(fake_backbone.requested_n, [23])
        self.assertEqual(len(features), 1)

    def test_uper_decoder_normalizes_channel_last_features(self):
        decoder = UPerLikeDecoder((4, 4, 4, 4), num_classes=3)
        decoder.eval()
        feats = [
            torch.randn(2, 4, 8, 8) * 0.1,
            torch.randn(2, 4, 4, 4) * 1.0,
            torch.randn(2, 4, 2, 2) * 10.0,
            torch.randn(2, 4, 1, 1) * 100.0,
        ]

        with torch.no_grad():
            logits = decoder(feats)

        self.assertEqual(tuple(logits.shape), (2, 3, 8, 8))

    def test_simple_feature_pyramid_builds_patch14_compatible_features(self):
        pyramid = SimpleFeaturePyramid(in_channels=32, out_channels=32)
        pyramid.eval()
        feats = [torch.randn(2, 32, 16, 16) for _ in range(4)]

        with torch.no_grad():
            outputs = pyramid(feats)

        self.assertEqual(pyramid.strides, (7, 14, 28, 56))
        self.assertEqual([tuple(x.shape[-2:]) for x in outputs], [(32, 32), (16, 16), (8, 8), (4, 4)])
        self.assertEqual([x.shape[1] for x in outputs], [32, 32, 32, 32])


if __name__ == "__main__":
    unittest.main()

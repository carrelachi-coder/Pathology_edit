import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn

from segmentator.config import SEGMENTATOR_CLASSES, SampleRecord
from segmentator.data import IMAGENET_MEAN, IMAGENET_STD, TissueSegmentationDataset
from segmentator.model import UPerLikeDecoder, Uni2hFeatureEncoder


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


class _FakeUniBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.requested_n = None

    def get_intermediate_layers(self, x, *, n, reshape):
        self.requested_n = n
        self.requested_reshape = reshape
        return [torch.zeros(x.shape[0], 1536, 2, 2) for _ in range(4)]


class SegmentatorModelTests(unittest.TestCase):
    def test_uni2h_encoder_uses_spaced_intermediate_layers(self):
        fake_backbone = _FakeUniBackbone()
        with patch("segmentator.model._load_uni2h_model", return_value=fake_backbone):
            encoder = Uni2hFeatureEncoder(local_repo="unused")

        features = encoder(torch.zeros(1, 3, 28, 28))

        self.assertEqual(fake_backbone.requested_n, [5, 11, 17, 23])
        self.assertTrue(fake_backbone.requested_reshape)
        self.assertEqual(len(features), 4)

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


if __name__ == "__main__":
    unittest.main()

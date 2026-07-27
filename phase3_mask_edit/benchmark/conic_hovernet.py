"""CoNIC HoVer-Net architecture and post-processing used by PathDiff.

The network definition follows the official CoNIC baseline ``net_desc.py``.
The post-processing follows TIAToolbox HoVer-Net 1.6.0, but is kept local so
the benchmark does not need to mutate any of the existing generator envs.
"""

from __future__ import annotations

from collections import OrderedDict
import math

import cv2
import numpy as np
from scipy import ndimage
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import Bottleneck as ResNetBottleneck
from torchvision.models.resnet import ResNet
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed


CONIC_CLASS_NAMES = {
    0: "background",
    1: "neutrophil",
    2: "epithelial",
    3: "lymphocyte",
    4: "plasma",
    5: "eosinophil",
    6: "connective",
}

PATHDIFF_COLOR_MAP = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
    4: (255, 255, 0),
    5: (255, 0, 255),
    6: (0, 255, 255),
    9: (255, 255, 255),
}


class UpSample2x(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("unpool_mat", torch.ones((2, 2), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = list(x.shape)
        values = torch.tensordot(x.unsqueeze(-1), self.unpool_mat.unsqueeze(0), dims=1)
        values = values.permute(0, 1, 2, 4, 3, 5)
        return values.reshape(
            -1, input_shape[1], input_shape[2] * 2, input_shape[3] * 2
        )


class ResNetExt(ResNet):
    def _forward_impl(self, x: torch.Tensor, freeze: bool):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        with torch.set_grad_enabled(self.training and not freeze):
            x1 = x = self.layer1(x)
            x2 = x = self.layer2(x)
            x3 = x = self.layer3(x)
            x4 = x = self.layer4(x)
        return x1, x2, x3, x4

    def forward(self, x: torch.Tensor, freeze: bool = False):
        return self._forward_impl(x, freeze)

    @staticmethod
    def resnet50(num_input_channels: int) -> "ResNetExt":
        model = ResNetExt(ResNetBottleneck, [3, 4, 6, 3])
        model.conv1 = nn.Conv2d(
            num_input_channels, 64, 7, stride=1, padding=3
        )
        return model


class DenseBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        unit_kernel_sizes: list[int],
        unit_channels: list[int],
        unit_count: int,
        split: int = 1,
    ) -> None:
        super().__init__()
        unit_in_channels = in_channels
        padding = [value // 2 for value in unit_kernel_sizes]
        self.units = nn.ModuleList()
        for _ in range(unit_count):
            self.units.append(
                nn.Sequential(
                    nn.BatchNorm2d(unit_in_channels, eps=1e-5),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        unit_in_channels,
                        unit_channels[0],
                        unit_kernel_sizes[0],
                        padding=padding[0],
                        bias=False,
                    ),
                    nn.BatchNorm2d(unit_channels[0], eps=1e-5),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        unit_channels[0],
                        unit_channels[1],
                        unit_kernel_sizes[1],
                        padding=padding[1],
                        bias=False,
                        groups=split,
                    ),
                )
            )
            unit_in_channels += unit_channels[1]
        self.blk_bna = nn.Sequential(
            nn.BatchNorm2d(unit_in_channels, eps=1e-5), nn.ReLU(inplace=True)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        for unit in self.units:
            features = torch.cat([features, unit(features)], dim=1)
        return self.blk_bna(features)


class HoVerNetConic(nn.Module):
    """Official padded HoVer-Net baseline architecture for seven labels."""

    def __init__(self, num_types: int = 7, freeze: bool = False) -> None:
        super().__init__()
        self.freeze = freeze
        self.backbone = ResNetExt.resnet50(3)
        self.conv_bot = nn.Conv2d(2048, 1024, 1, bias=False)
        self.decoder = nn.ModuleDict(
            OrderedDict(
                [
                    ("tp", self._decoder_branch(num_types)),
                    ("np", self._decoder_branch(2)),
                    ("hv", self._decoder_branch(2)),
                ]
            )
        )
        self.upsample2x = UpSample2x()

    @staticmethod
    def _decoder_branch(out_channels: int, kernel_size: int = 3) -> nn.Sequential:
        padding = kernel_size // 2
        u3 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size, padding=padding, bias=False),
            DenseBlock(256, [1, kernel_size], [128, 32], 8, split=4),
            nn.Conv2d(512, 512, 1, bias=False),
        )
        u2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size, padding=padding, bias=False),
            DenseBlock(128, [1, kernel_size], [128, 32], 4, split=4),
            nn.Conv2d(256, 256, 1, bias=False),
        )
        u1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size, padding=padding, bias=False)
        )
        u0 = nn.Sequential(
            nn.BatchNorm2d(64, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 1, bias=True),
        )
        return nn.Sequential(
            OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0)])
        )

    def forward(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        features = self.backbone(images / 255.0, self.freeze)
        features = (*features[:3], self.conv_bot(features[3]))
        outputs: OrderedDict[str, torch.Tensor] = OrderedDict()
        for branch_name, branch in self.decoder.items():
            u3 = branch[0](self.upsample2x(features[3]) + features[2])
            u2 = branch[1](self.upsample2x(u3) + features[1])
            u1 = branch[2](self.upsample2x(u2) + features[0])
            outputs[branch_name] = branch[3](u1)
        return outputs


def infer_raw_maps(
    model: HoVerNetConic, batch_rgb_u8: torch.Tensor, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch = batch_rgb_u8.to(device=device, dtype=torch.float32)
    batch = batch.permute(0, 3, 1, 2).contiguous()
    with torch.inference_mode():
        outputs = model(batch)
        np_map = F.softmax(outputs["np"].permute(0, 2, 3, 1), dim=-1)[..., 1:]
        hv_map = outputs["hv"].permute(0, 2, 3, 1)
        type_map = F.softmax(outputs["tp"].permute(0, 2, 3, 1), dim=-1)
        type_map = torch.argmax(type_map, dim=-1).to(torch.uint8)
    return (
        np_map.cpu().numpy(),
        hv_map.cpu().numpy(),
        type_map.cpu().numpy(),
    )


def segment_instances(np_map: np.ndarray, hv_map: np.ndarray) -> np.ndarray:
    nuclei = (np_map[..., 0] >= 0.5).astype(np.int32)
    nuclei = ndimage.label(nuclei)[0]
    nuclei = remove_small_objects(nuclei, min_size=10)
    nuclei[nuclei > 0] = 1

    horizontal = cv2.normalize(
        hv_map[..., 0], None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    vertical = cv2.normalize(
        hv_map[..., 1], None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    sobel_h = cv2.Sobel(horizontal, cv2.CV_64F, 1, 0, ksize=21)
    sobel_v = cv2.Sobel(vertical, cv2.CV_64F, 0, 1, ksize=21)
    sobel_h = 1 - cv2.normalize(
        sobel_h, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    sobel_v = 1 - cv2.normalize(
        sobel_v, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    overall = np.maximum(sobel_h, sobel_v) - (1 - nuclei)
    overall[overall < 0] = 0
    distance = -cv2.GaussianBlur((1.0 - overall) * nuclei, (3, 3), 0)
    boundary = (overall >= 0.4).astype(np.int32)
    markers = nuclei - boundary
    markers[markers < 0] = 0
    markers = ndimage.binary_fill_holes(markers).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    markers = cv2.morphologyEx(markers, cv2.MORPH_OPEN, kernel)
    markers = ndimage.label(markers)[0]
    markers = remove_small_objects(markers, min_size=math.ceil(10))
    return watershed(distance, markers=markers, mask=nuclei).astype(np.int32)


def _remap_instances(instance_map: np.ndarray) -> np.ndarray:
    output = np.zeros_like(instance_map, dtype=np.int32)
    for new_id, old_id in enumerate(np.unique(instance_map)[1:], start=1):
        output[instance_map == old_id] = new_id
    return output


def assign_instance_types(
    instance_map: np.ndarray, pixel_type_map: np.ndarray
) -> np.ndarray:
    output = np.zeros_like(instance_map, dtype=np.uint8)
    for instance_id in np.unique(instance_map)[1:]:
        values = pixel_type_map[instance_map == instance_id]
        labels, counts = np.unique(values, return_counts=True)
        ranked = sorted(zip(labels.tolist(), counts.tolist()), key=lambda x: x[1], reverse=True)
        instance_type = ranked[0][0]
        if instance_type == 0 and len(ranked) > 1:
            instance_type = ranked[1][0]
        output[instance_map == instance_id] = int(instance_type)
    return output


def postprocess_conic_maps(
    np_map: np.ndarray, hv_map: np.ndarray, pixel_type_map: np.ndarray
) -> np.ndarray:
    """Return an official CoNIC-style ``256 x 256 x 2`` map.

    Input images cover 128 microns at 0.5 microns per pixel. The HoVer-Net
    watershed is run at 0.25 microns per pixel, matching the official CoNIC
    baseline notebook, then reduced back to PathDiff's 256-pixel condition.
    """

    np_high = cv2.resize(np_map, (512, 512), interpolation=cv2.INTER_LINEAR)
    if np_high.ndim == 2:
        np_high = np_high[..., None]
    hv_high = cv2.resize(hv_map, (512, 512), interpolation=cv2.INTER_LINEAR)
    type_high = cv2.resize(
        pixel_type_map, (512, 512), interpolation=cv2.INTER_NEAREST
    )
    instances_high = segment_instances(np_high, hv_high)
    types_high = assign_instance_types(instances_high, type_high)
    instances = cv2.resize(
        instances_high, (256, 256), interpolation=cv2.INTER_NEAREST
    )
    instances = _remap_instances(instances)
    types = cv2.resize(types_high, (256, 256), interpolation=cv2.INTER_NEAREST)
    types[instances == 0] = 0
    if not set(np.unique(types)).issubset(set(CONIC_CLASS_NAMES)):
        raise ValueError(f"Unexpected CoNIC labels: {np.unique(types).tolist()}")
    return np.stack([instances, types.astype(np.int32)], axis=-1)


def pathdiff_edge_map(instance_map: np.ndarray) -> np.ndarray:
    edge = np.zeros_like(instance_map, dtype=bool)
    edge[:, 1:] |= instance_map[:, 1:] != instance_map[:, :-1]
    edge[:, :-1] |= instance_map[:, 1:] != instance_map[:, :-1]
    edge[1:, :] |= instance_map[1:, :] != instance_map[:-1, :]
    edge[:-1, :] |= instance_map[1:, :] != instance_map[:-1, :]
    return edge.astype(np.uint8)


def colorize_pathdiff_mask(mask: np.ndarray) -> np.ndarray:
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label, color in PATHDIFF_COLOR_MAP.items():
        output[mask == label] = color
    return output

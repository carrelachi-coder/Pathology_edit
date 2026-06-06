"""Frozen same-WSI appearance encoder and perceptual loss.

The encoder is pretrained on real patch pairs to predict whether two patches
come from the same WSI/case. During generator training it is frozen and only
its intermediate feature maps are used as a pathology-specific perceptual
space for H&E stain and texture.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SameWSIAppearanceConfig:
    """Architecture and preprocessing config saved with pretrained weights."""

    backbone_channels: tuple[int, ...] = (32, 64, 128, 256)
    embedding_dim: int = 256
    input_size: int = 256
    feature_layers: tuple[int, ...] = (1, 2, 3)
    normalize_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalize_std: tuple[float, float, float] = (0.5, 0.5, 0.5)


class SameWSIAppearanceEncoder(nn.Module):
    """Small CNN encoder for same-WSI appearance pretraining."""

    def __init__(self, config: SameWSIAppearanceConfig | None = None) -> None:
        super().__init__()
        self.config = config or SameWSIAppearanceConfig()
        channels = (3, *self.config.backbone_channels)
        blocks: list[nn.Module] = []
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
                    nn.SiLU(inplace=True),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.projection = nn.Linear(channels[-1], self.config.embedding_dim)

    def forward_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        param = next(self.parameters())
        x = self._normalize(images.to(device=param.device, dtype=param.dtype))
        features: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(images)
        pooled = features[-1].mean(dim=(2, 3))
        return F.normalize(self.projection(pooled), dim=-1, eps=1e-6)

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        mean = images.new_tensor(self.config.normalize_mean).view(1, 3, 1, 1)
        std = images.new_tensor(self.config.normalize_std).view(1, 3, 1, 1).clamp_min(1e-6)
        return (images - mean) / std


class SameWSIPairClassifier(nn.Module):
    """Siamese same/different WSI classifier used only for pretraining."""

    def __init__(self, encoder: SameWSIAppearanceEncoder | None = None) -> None:
        super().__init__()
        self.encoder = encoder or SameWSIAppearanceEncoder()
        dim = self.encoder.config.embedding_dim
        self.head = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(dim, 1),
        )

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor) -> torch.Tensor:
        emb_a = self.encoder(image_a)
        emb_b = self.encoder(image_b)
        pair = torch.cat([emb_a, emb_b, (emb_a - emb_b).abs(), emb_a * emb_b], dim=-1)
        return self.head(pair).flatten()


def same_wsi_perceptual_loss(
    *,
    encoder: SameWSIAppearanceEncoder,
    prediction: torch.Tensor,
    reference: torch.Tensor,
    target_tissue_mask: torch.Tensor | None = None,
    reference_tissue_mask: torch.Tensor | None = None,
    layers: Iterable[int] | None = None,
    min_pixels: int = 8,
) -> tuple[torch.Tensor, int]:
    """Feature distance between generated and reference appearance.

    If tissue masks are supplied, feature maps are compared only for labels
    present in both target and reference. Otherwise whole-image features are
    compared. The encoder parameters stay frozen, but gradients flow through
    ``prediction``.
    """

    if prediction.shape != reference.shape:
        raise ValueError(f"prediction/reference shapes differ: {tuple(prediction.shape)} vs {tuple(reference.shape)}")
    feature_layers = tuple(layers if layers is not None else encoder.config.feature_layers)
    pred_features = encoder.forward_features(prediction)
    with torch.no_grad():
        ref_features = encoder.forward_features(reference.detach())

    total = prediction.new_zeros(())
    terms = 0
    for layer_index in feature_layers:
        if layer_index < 0 or layer_index >= len(pred_features):
            raise ValueError(f"same-WSI feature layer {layer_index} is out of range for {len(pred_features)} blocks.")
        pred_feat = pred_features[layer_index]
        ref_feat = ref_features[layer_index]
        if target_tissue_mask is None or reference_tissue_mask is None:
            total = total + _feature_distance(pred_feat, ref_feat)
            terms += 1
            continue
        layer_loss, layer_regions = _masked_feature_distance(
            pred_feat=pred_feat,
            ref_feat=ref_feat,
            target_mask=target_tissue_mask,
            reference_mask=reference_tissue_mask,
            min_pixels=min_pixels,
        )
        if layer_regions > 0:
            total = total + layer_loss
            terms += 1
    if terms == 0:
        return prediction.new_zeros(()), 0
    return total / terms, terms


def save_same_wsi_checkpoint(
    path: str | Path,
    *,
    model: SameWSIPairClassifier,
    extra: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": _config_to_dict(model.encoder.config),
        "encoder": model.encoder.state_dict(),
        "head": model.head.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_same_wsi_encoder(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> SameWSIAppearanceEncoder:
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    config = SameWSIAppearanceConfig(**checkpoint.get("config", {}))
    encoder = SameWSIAppearanceEncoder(config)
    state = checkpoint.get("encoder", checkpoint)
    encoder.load_state_dict(state)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


def _masked_feature_distance(
    *,
    pred_feat: torch.Tensor,
    ref_feat: torch.Tensor,
    target_mask: torch.Tensor,
    reference_mask: torch.Tensor,
    min_pixels: int,
) -> tuple[torch.Tensor, int]:
    size = pred_feat.shape[-2:]
    target_small = F.interpolate(target_mask.unsqueeze(1).float(), size=size, mode="nearest").squeeze(1).long()
    reference_small = F.interpolate(reference_mask.unsqueeze(1).float(), size=size, mode="nearest").squeeze(1).long()
    losses = []
    for batch_index in range(pred_feat.shape[0]):
        labels = torch.unique(target_small[batch_index])
        for label in labels.tolist():
            if int(label) == 0:
                continue
            target_region = target_small[batch_index] == int(label)
            reference_region = reference_small[batch_index] == int(label)
            if int(target_region.sum().item()) < min_pixels or int(reference_region.sum().item()) < min_pixels:
                continue
            pred_vec = _region_mean(pred_feat[batch_index], target_region)
            ref_vec = _region_mean(ref_feat[batch_index], reference_region)
            losses.append(_feature_distance(pred_vec.unsqueeze(0), ref_vec.unsqueeze(0)))
    if not losses:
        return pred_feat.new_zeros(()), 0
    return torch.stack(losses).mean(), len(losses)


def _region_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=features.dtype).unsqueeze(0)
    return (features * weights).sum(dim=(1, 2)) / weights.sum().clamp_min(1.0)


def _feature_distance(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    pred_norm = F.normalize(pred.float().flatten(1), dim=-1, eps=1e-6)
    ref_norm = F.normalize(ref.float().flatten(1), dim=-1, eps=1e-6)
    cosine = 1.0 - (pred_norm * ref_norm.detach()).sum(dim=-1)
    l1 = F.l1_loss(pred.float(), ref.detach().float())
    return cosine.mean() + 0.25 * l1


def _config_to_dict(config: SameWSIAppearanceConfig) -> dict:
    return {
        "backbone_channels": tuple(config.backbone_channels),
        "embedding_dim": int(config.embedding_dim),
        "input_size": int(config.input_size),
        "feature_layers": tuple(config.feature_layers),
        "normalize_mean": tuple(config.normalize_mean),
        "normalize_std": tuple(config.normalize_std),
    }


__all__ = [
    "SameWSIAppearanceConfig",
    "SameWSIAppearanceEncoder",
    "SameWSIPairClassifier",
    "load_same_wsi_encoder",
    "same_wsi_perceptual_loss",
    "save_same_wsi_checkpoint",
]

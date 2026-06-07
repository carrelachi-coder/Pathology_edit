"""Frozen VGG perceptual/style loss for reference appearance matching."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


VGG16_LAYER_ALIASES = {
    "conv1_1": 0,
    "relu1_1": 1,
    "conv1_2": 2,
    "relu1_2": 3,
    "pool1": 4,
    "conv2_1": 5,
    "relu2_1": 6,
    "conv2_2": 7,
    "relu2_2": 8,
    "pool2": 9,
    "conv3_1": 10,
    "relu3_1": 11,
    "conv3_2": 12,
    "relu3_2": 13,
    "conv3_3": 14,
    "relu3_3": 15,
    "pool3": 16,
    "conv4_1": 17,
    "relu4_1": 18,
    "conv4_2": 19,
    "relu4_2": 20,
    "conv4_3": 21,
    "relu4_3": 22,
    "pool4": 23,
    "conv5_1": 24,
    "relu5_1": 25,
    "conv5_2": 26,
    "relu5_2": 27,
    "conv5_3": 28,
    "relu5_3": 29,
}

DEFAULT_VGG16_PERCEPTUAL_LAYERS = ("relu1_1", "relu1_2", "relu2_1", "relu2_2")
DEFAULT_VGG_LOSS_TYPE = "gram"
IMAGENET_RGB_MEAN = (0.485, 0.456, 0.406)
IMAGENET_RGB_STD = (0.229, 0.224, 0.225)
RGB_TO_LUMA = (0.299, 0.587, 0.114)


class VGGPerceptualLoss(nn.Module):
    """Feature-space loss over frozen VGG feature maps.

    Inputs are expected to be RGB tensors in ``[0, 1]`` and are converted to
    grayscale by default before VGG normalization. When tissue masks are
    supplied, feature maps are compared by matching class-wise regions so
    non-aligned target/reference patches can still share an appearance/style
    loss. ``loss_type="gram"`` compares Gram matrices and is intended for
    shallow stain/texture style matching without color shortcuts.
    """

    def __init__(
        self,
        features: nn.Module,
        *,
        layer_indices: Iterable[int | str] = DEFAULT_VGG16_PERCEPTUAL_LAYERS,
        layer_weights: Iterable[float] | None = None,
        loss_type: str = DEFAULT_VGG_LOSS_TYPE,
        grayscale: bool = True,
        input_size: int = 256,
        normalize_mean: Iterable[float] = IMAGENET_RGB_MEAN,
        normalize_std: Iterable[float] = IMAGENET_RGB_STD,
    ) -> None:
        super().__init__()
        self.features = features.eval()
        self.layer_indices = parse_vgg_layer_indices(layer_indices)
        if not self.layer_indices:
            raise ValueError("VGG perceptual loss requires at least one feature layer.")
        weights = tuple(float(value) for value in (layer_weights or (1.0,) * len(self.layer_indices)))
        if len(weights) != len(self.layer_indices):
            raise ValueError(
                "VGG layer weights must match layer count: "
                f"{len(weights)} vs {len(self.layer_indices)}"
        )
        self.loss_type = normalize_vgg_loss_type(loss_type)
        self.grayscale = bool(grayscale)
        self.layer_weights = weights
        self.input_size = int(input_size or 0)
        self.register_buffer(
            "rgb_to_luma",
            torch.tensor(RGB_TO_LUMA, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "normalize_mean",
            torch.tensor(tuple(normalize_mean), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "normalize_std",
            torch.tensor(tuple(normalize_std), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.requires_grad_(False)

    def forward_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        x = self._prepare_images(images)
        outputs: dict[int, torch.Tensor] = {}
        target_layers = set(self.layer_indices)
        max_layer = max(target_layers)
        for layer_index, layer in enumerate(self.features):
            x = layer(x)
            if layer_index in target_layers:
                outputs[layer_index] = x
            if layer_index >= max_layer:
                break
        missing = [index for index in self.layer_indices if index not in outputs]
        if missing:
            raise ValueError(
                "VGG feature module ended before requested layers: "
                f"requested={self.layer_indices} missing={tuple(missing)}"
            )
        return [outputs[index] for index in self.layer_indices]

    def forward(
        self,
        prediction: torch.Tensor,
        reference: torch.Tensor,
        *,
        target_tissue_mask: torch.Tensor | None = None,
        reference_tissue_mask: torch.Tensor | None = None,
        min_pixels: int = 8,
    ) -> tuple[torch.Tensor, int]:
        if prediction.shape != reference.shape:
            raise ValueError(
                "prediction/reference shapes differ: "
                f"{tuple(prediction.shape)} vs {tuple(reference.shape)}"
            )

        pred_features = self.forward_features(prediction)
        with torch.no_grad():
            ref_features = self.forward_features(reference.detach())

        total = prediction.new_zeros((), dtype=torch.float32)
        normalizer = 0.0
        terms = 0
        for pred_feat, ref_feat, weight in zip(pred_features, ref_features, self.layer_weights):
            if target_tissue_mask is None or reference_tissue_mask is None:
                layer_loss = _feature_loss(
                    pred_feat,
                    ref_feat,
                    loss_type=self.loss_type,
                )
                layer_terms = 1
            elif self.loss_type == "gram":
                layer_loss, layer_terms = _masked_feature_gram_loss(
                    pred_feat=pred_feat,
                    ref_feat=ref_feat,
                    target_mask=target_tissue_mask,
                    reference_mask=reference_tissue_mask,
                    min_pixels=min_pixels,
                )
            else:
                layer_loss, layer_terms = _masked_feature_l1(
                    pred_feat=pred_feat,
                    ref_feat=ref_feat,
                    target_mask=target_tissue_mask,
                    reference_mask=reference_tissue_mask,
                    min_pixels=min_pixels,
                )
            if layer_terms > 0 and weight > 0.0:
                total = total + float(weight) * layer_loss
                normalizer += float(weight)
                terms += 1

        if normalizer <= 0.0:
            return prediction.new_zeros(()), 0
        return total / normalizer, terms

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"VGG perceptual images must have shape (B, 3, H, W), got {tuple(images.shape)}")
        device, dtype = _module_device_dtype(self)
        x = images.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        if self.grayscale:
            gray = (x * self.rgb_to_luma.to(device=device, dtype=dtype)).sum(dim=1, keepdim=True)
            x = gray.repeat(1, 3, 1, 1)
        if self.input_size > 0 and tuple(x.shape[-2:]) != (self.input_size, self.input_size):
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
        mean = self.normalize_mean.to(device=device, dtype=dtype)
        std = self.normalize_std.to(device=device, dtype=dtype).clamp_min(1e-6)
        return (x - mean) / std


def build_vgg16_perceptual_loss(
    *,
    weights: str = "imagenet",
    weights_path: str | Path | None = None,
    layers: Iterable[int | str] | str = DEFAULT_VGG16_PERCEPTUAL_LAYERS,
    loss_type: str = DEFAULT_VGG_LOSS_TYPE,
    grayscale: bool = True,
    input_size: int = 256,
) -> VGGPerceptualLoss:
    """Build a frozen VGG16 perceptual module.

    ``torchvision`` is imported lazily so environments that only parse CLI args
    do not need it installed. A local ``weights_path`` can be used to avoid
    downloading ImageNet weights at training time.
    """
    try:
        from torchvision.models import VGG16_Weights, vgg16
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "VGG perceptual loss requires torchvision. Install the phase5 "
            "environment or set up torchvision before training with "
            "--reference-perceptual-backend=vgg."
        ) from exc

    weight_mode = str(weights or "imagenet").strip().lower()
    if weights_path:
        model = vgg16(weights=None)
        _load_vgg16_weights(model, weights_path)
    elif weight_mode in {"imagenet", "default", "pretrained"}:
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    elif weight_mode in {"none", "random", "untrained"}:
        model = vgg16(weights=None)
    else:
        raise ValueError("--reference-vgg-weights must be one of: imagenet, none.")

    return VGGPerceptualLoss(
        model.features,
        layer_indices=parse_vgg_layer_indices(layers),
        loss_type=loss_type,
        grayscale=grayscale,
        input_size=input_size,
    )


def vgg_perceptual_loss(
    *,
    encoder: VGGPerceptualLoss,
    prediction: torch.Tensor,
    reference: torch.Tensor,
    target_tissue_mask: torch.Tensor | None = None,
    reference_tissue_mask: torch.Tensor | None = None,
    min_pixels: int = 8,
) -> tuple[torch.Tensor, int]:
    return encoder(
        prediction,
        reference,
        target_tissue_mask=target_tissue_mask,
        reference_tissue_mask=reference_tissue_mask,
        min_pixels=min_pixels,
    )


def parse_vgg_layer_indices(layers: Iterable[int | str] | str) -> tuple[int, ...]:
    if isinstance(layers, str):
        raw_layers: Iterable[int | str] = [value.strip() for value in layers.split(",")]
    else:
        raw_layers = layers

    parsed = []
    for layer in raw_layers:
        if isinstance(layer, int):
            parsed.append(layer)
            continue
        value = str(layer).strip().lower()
        if not value:
            continue
        if value in VGG16_LAYER_ALIASES:
            parsed.append(VGG16_LAYER_ALIASES[value])
        else:
            parsed.append(int(value))
    return tuple(parsed)


def normalize_vgg_loss_type(loss_type: str) -> str:
    value = str(loss_type or DEFAULT_VGG_LOSS_TYPE).strip().lower().replace("-", "_")
    aliases = {
        "style": "gram",
        "gram_style": "gram",
        "l1": "feature_l1",
        "feature": "feature_l1",
    }
    value = aliases.get(value, value)
    if value not in {"gram", "feature_l1"}:
        raise ValueError("--reference-vgg-loss-type must be one of: gram, feature_l1.")
    return value


def _feature_loss(
    pred_feat: torch.Tensor,
    ref_feat: torch.Tensor,
    *,
    loss_type: str,
) -> torch.Tensor:
    if loss_type == "gram":
        return F.l1_loss(
            _gram_matrix(pred_feat.float()),
            _gram_matrix(ref_feat.detach().float()),
        )
    return F.l1_loss(pred_feat.float(), ref_feat.detach().float())


def _gram_matrix(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 4:
        raise ValueError(f"Gram features must have shape (B, C, H, W), got {tuple(features.shape)}")
    batch, _channels, height, width = features.shape
    pixels = height * width
    flattened = features.reshape(batch, features.shape[1], pixels)
    gram = torch.bmm(flattened, flattened.transpose(1, 2))
    return gram / float(max(1, pixels))


def _masked_feature_gram_loss(
    *,
    pred_feat: torch.Tensor,
    ref_feat: torch.Tensor,
    target_mask: torch.Tensor,
    reference_mask: torch.Tensor,
    min_pixels: int,
) -> tuple[torch.Tensor, int]:
    size = pred_feat.shape[-2:]
    target_small = _resize_mask_to_feature_map(target_mask, size)
    reference_small = _resize_mask_to_feature_map(reference_mask, size)

    losses = []
    for batch_index in range(pred_feat.shape[0]):
        labels = torch.unique(target_small[batch_index])
        for label in labels.tolist():
            label = int(label)
            if label == 0:
                continue
            target_region = target_small[batch_index] == label
            reference_region = reference_small[batch_index] == label
            if int(target_region.sum().item()) < min_pixels or int(reference_region.sum().item()) < min_pixels:
                continue
            pred_gram = _region_gram(pred_feat[batch_index], target_region)
            ref_gram = _region_gram(ref_feat[batch_index].detach(), reference_region)
            losses.append(F.l1_loss(pred_gram.float(), ref_gram.float()))

    if not losses:
        return pred_feat.new_zeros(()), 0
    return torch.stack(losses).mean(), len(losses)


def _masked_feature_l1(
    *,
    pred_feat: torch.Tensor,
    ref_feat: torch.Tensor,
    target_mask: torch.Tensor,
    reference_mask: torch.Tensor,
    min_pixels: int,
) -> tuple[torch.Tensor, int]:
    size = pred_feat.shape[-2:]
    target_small = _resize_mask_to_feature_map(target_mask, size)
    reference_small = _resize_mask_to_feature_map(reference_mask, size)

    losses = []
    for batch_index in range(pred_feat.shape[0]):
        labels = torch.unique(target_small[batch_index])
        for label in labels.tolist():
            label = int(label)
            if label == 0:
                continue
            target_region = target_small[batch_index] == label
            reference_region = reference_small[batch_index] == label
            if int(target_region.sum().item()) < min_pixels or int(reference_region.sum().item()) < min_pixels:
                continue
            pred_vec = _region_mean(pred_feat[batch_index], target_region)
            ref_vec = _region_mean(ref_feat[batch_index], reference_region)
            losses.append(F.l1_loss(pred_vec.float(), ref_vec.detach().float()))

    if not losses:
        return pred_feat.new_zeros(()), 0
    return torch.stack(losses).mean(), len(losses)


def _region_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=features.dtype).unsqueeze(0)
    return (features * weights).sum(dim=(1, 2)) / weights.sum().clamp_min(1.0)


def _region_gram(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = features[:, mask]
    if selected.numel() == 0:
        return features.new_zeros((features.shape[0], features.shape[0]))
    gram = selected @ selected.transpose(0, 1)
    return gram / float(max(1, selected.shape[1]))


def _resize_mask_to_feature_map(mask: torch.Tensor, feature_size: tuple[int, int]) -> torch.Tensor:
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 3:
        raise ValueError(f"mask must have shape (B,H,W) or (B,1,H,W), got {tuple(mask.shape)}")
    if tuple(int(value) for value in mask.shape[-2:]) == tuple(int(value) for value in feature_size):
        return mask.to(dtype=torch.long)
    resized = F.interpolate(
        mask.unsqueeze(1).float(),
        size=feature_size,
        mode="nearest",
    )
    return resized[:, 0].to(dtype=torch.long)


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    for parameter in module.parameters():
        return parameter.device, parameter.dtype
    for buffer in module.buffers():
        return buffer.device, buffer.dtype
    return torch.device("cpu"), torch.float32


def _load_vgg16_weights(model: nn.Module, weights_path: str | Path) -> None:
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        state = (
            checkpoint.get("state_dict")
            or checkpoint.get("model")
            or checkpoint.get("model_state_dict")
            or checkpoint
        )
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise ValueError(f"VGG weights must be a state dict, got {type(state).__name__}.")

    cleaned = {}
    for key, value in state.items():
        key = str(key)
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("features."):
            cleaned[key] = value
        elif key.startswith("vgg.features."):
            cleaned[key[len("vgg."):]] = value
    missing, unexpected = model.load_state_dict(cleaned or state, strict=False)
    feature_missing = [key for key in missing if key.startswith("features.")]
    if feature_missing:
        raise ValueError(f"VGG weights are missing feature tensors: {feature_missing[:4]}")
    feature_unexpected = [key for key in unexpected if key.startswith("features.")]
    if feature_unexpected:
        raise ValueError(f"Unexpected VGG feature tensors: {feature_unexpected[:4]}")


__all__ = [
    "DEFAULT_VGG_LOSS_TYPE",
    "DEFAULT_VGG16_PERCEPTUAL_LAYERS",
    "RGB_TO_LUMA",
    "VGG16_LAYER_ALIASES",
    "VGGPerceptualLoss",
    "build_vgg16_perceptual_loss",
    "normalize_vgg_loss_type",
    "parse_vgg_layer_indices",
    "vgg_perceptual_loss",
]

"""Losses for supervised pix2pix reference texture transfer.

Content supervision is computed against the paired target image. Texture
supervision is region-aware and computed against the (spatially unaligned)
reference image, so it does not force individual nuclei to occupy identical
pixel coordinates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_CONTENT_LAYERS = (3, 8, 15, 22)
DEFAULT_GRAM_LAYERS = (3, 8, 15)
DEFAULT_CONTEXTUAL_LAYERS = (8, 15)
LOSS_NAMES = ("l1", "content", "gram", "contextual")


def parse_layer_indices(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        raw = [part.strip() for part in value.split(",")]
        layers = tuple(int(part) for part in raw if part)
    else:
        layers = tuple(int(layer) for layer in value)
    if not layers or any(layer < 0 for layer in layers):
        raise ValueError("VGG layer indices must be a non-empty list of non-negative integers")
    return tuple(dict.fromkeys(layers))


class VGGFeatureExtractor(nn.Module):
    """Frozen VGG16 feature slices shared by content and texture losses."""

    def __init__(
        self,
        *,
        layers: Sequence[int],
        weights: str = "imagenet",
        resize: bool = False,
    ) -> None:
        super().__init__()
        import torchvision

        if str(weights).lower() == "imagenet":
            vgg_weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
        elif str(weights).lower() == "none":
            vgg_weights = None
        else:
            raise ValueError("--vgg-weights must be 'imagenet' or 'none'")

        self.layers = parse_layer_indices(layers)
        vgg = torchvision.models.vgg16(weights=vgg_weights)
        self.features = vgg.features[: max(self.layers) + 1].eval()
        for parameter in self.features.parameters():
            parameter.requires_grad = False
        self.resize = bool(resize)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def _prepare(self, image: torch.Tensor) -> torch.Tensor:
        image = ((image.float() + 1.0) * 0.5).clamp(0.0, 1.0)
        image = (image - self.mean) / self.std
        if self.resize:
            image = F.interpolate(image, size=224, mode="bilinear", align_corners=False)
        return image

    def forward(self, image: torch.Tensor) -> dict[int, torch.Tensor]:
        feature = self._prepare(image)
        outputs: dict[int, torch.Tensor] = {}
        for index, layer in enumerate(self.features):
            feature = layer(feature)
            if index in self.layers:
                outputs[index] = feature
        return outputs


class VGGPerceptualLoss(nn.Module):
    """Backward-compatible standalone VGG content loss."""

    def __init__(
        self,
        *,
        layers: tuple[int, ...] = DEFAULT_CONTENT_LAYERS,
        weights: str = "imagenet",
        resize: bool = False,
    ) -> None:
        super().__init__()
        self.extractor = VGGFeatureExtractor(layers=layers, weights=weights, resize=resize)
        self.layers = parse_layer_indices(layers)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_features = self.extractor(pred)
        with torch.no_grad():
            target_features = self.extractor(target)
        return feature_content_loss(pred_features, target_features, self.layers)


def feature_content_loss(
    prediction: Mapping[int, torch.Tensor],
    target: Mapping[int, torch.Tensor],
    layers: Sequence[int],
) -> torch.Tensor:
    losses = [F.l1_loss(prediction[layer], target[layer]) for layer in layers]
    if not losses:
        raise ValueError("At least one content layer is required")
    return torch.stack(losses).mean()


def gaussian_blur_image(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Differentiable channel-wise Gaussian blur without a torchvision dependency."""

    sigma = float(sigma)
    if sigma <= 0.0:
        return image.float()
    radius = max(1, int(round(3.0 * sigma)))
    coordinates = torch.arange(
        -radius,
        radius + 1,
        device=image.device,
        dtype=torch.float32,
    )
    kernel = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    channels = int(image.shape[1])
    horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    values = image.float()
    values = F.pad(values, (radius, radius, 0, 0), mode="reflect")
    values = F.conv2d(values, horizontal, groups=channels)
    values = F.pad(values, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(values, vertical, groups=channels)


def image_l1_loss(prediction: torch.Tensor, target: torch.Tensor, *, blur_sigma: float) -> torch.Tensor:
    prediction_values = gaussian_blur_image(prediction, blur_sigma)
    target_values = gaussian_blur_image(target, blur_sigma)
    return F.l1_loss(prediction_values, target_values)


def _resize_region_labels(region: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if region.ndim == 3:
        region = region.unsqueeze(1)
    if region.ndim != 4 or region.shape[1] != 1:
        raise ValueError(f"region labels must have shape [B,1,H,W] or [B,H,W], got {region.shape}")
    return F.interpolate(region.float(), size=size, mode="nearest")[:, 0].long()


def _shared_region_ids(
    target_labels: torch.Tensor,
    reference_labels: torch.Tensor,
    *,
    min_pixels: int,
    exclude_background: bool,
) -> list[int]:
    target_ids = set(int(value) for value in torch.unique(target_labels).tolist())
    reference_ids = set(int(value) for value in torch.unique(reference_labels).tolist())
    shared = sorted(target_ids.intersection(reference_ids))
    result: list[int] = []
    for region_id in shared:
        if exclude_background and region_id == 0:
            continue
        if int((target_labels == region_id).sum().item()) < min_pixels:
            continue
        if int((reference_labels == region_id).sum().item()) < min_pixels:
            continue
        result.append(region_id)
    return result


def _masked_vectors(feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return masked feature vectors with shape [N,C]."""

    return feature[:, mask].transpose(0, 1).float()


def gram_matrix(vectors: torch.Tensor, *, center: bool = True) -> torch.Tensor:
    """Channel Gram matrix for vectors shaped [N,C]."""

    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError(f"vectors must be non-empty [N,C], got {vectors.shape}")
    values = vectors.float()
    if center:
        values = values - values.mean(dim=0, keepdim=True)
    return values.transpose(0, 1) @ values / float(values.shape[0])


def regional_gram_loss(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    target_region: torch.Tensor,
    reference_region: torch.Tensor,
    *,
    min_pixels: int = 8,
    exclude_background: bool = True,
    center: bool = True,
) -> tuple[torch.Tensor, int]:
    """Compare region-wise feature covariance without spatial correspondence."""

    if prediction.shape != reference.shape:
        raise ValueError(
            f"prediction/reference features must match, got {prediction.shape} and {reference.shape}"
        )
    target_labels = _resize_region_labels(target_region, prediction.shape[-2:])
    reference_labels = _resize_region_labels(reference_region, reference.shape[-2:])
    losses: list[torch.Tensor] = []
    for batch_index in range(prediction.shape[0]):
        region_ids = _shared_region_ids(
            target_labels[batch_index],
            reference_labels[batch_index],
            min_pixels=min_pixels,
            exclude_background=exclude_background,
        )
        for region_id in region_ids:
            pred_vectors = _masked_vectors(
                prediction[batch_index], target_labels[batch_index] == region_id
            )
            ref_vectors = _masked_vectors(
                reference[batch_index], reference_labels[batch_index] == region_id
            )
            pred_gram = gram_matrix(pred_vectors, center=center)
            with torch.no_grad():
                ref_gram = gram_matrix(ref_vectors, center=center)
            losses.append(F.l1_loss(pred_gram, ref_gram))
    if not losses:
        return prediction.float().sum() * 0.0, 0
    return torch.stack(losses).mean(), len(losses)


def _subsample_vectors(vectors: torch.Tensor, max_samples: int) -> torch.Tensor:
    if max_samples <= 0 or vectors.shape[0] <= max_samples:
        return vectors
    # Even sampling is deterministic and covers the complete spatial ordering.
    indices = torch.linspace(
        0,
        vectors.shape[0] - 1,
        steps=max_samples,
        device=vectors.device,
    ).round().long()
    return vectors.index_select(0, indices)


def contextual_directional_loss(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    temperature: float = 0.1,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Contextual affinity from query vectors to spatially unordered key vectors."""

    if query.ndim != 2 or key.ndim != 2 or query.shape[1] != key.shape[1]:
        raise ValueError(f"query/key must be [N,C] with matching C, got {query.shape}, {key.shape}")
    if query.shape[0] < 2 or key.shape[0] < 2:
        return query.sum() * 0.0
    if temperature <= 0:
        raise ValueError("contextual temperature must be positive")

    reference_mean = key.mean(dim=0, keepdim=True)
    query_normalized = F.normalize(query - reference_mean, dim=1, eps=eps)
    key_normalized = F.normalize(key - reference_mean, dim=1, eps=eps)
    cosine_distance = (1.0 - query_normalized @ key_normalized.transpose(0, 1)).clamp_min(0.0)
    relative_distance = cosine_distance / (
        cosine_distance.min(dim=1, keepdim=True).values.detach() + eps
    )
    affinity = torch.softmax((1.0 - relative_distance) / temperature, dim=1)
    contextual_similarity = affinity.max(dim=1).values.mean()
    return -torch.log(contextual_similarity.clamp_min(eps))


def regional_contextual_loss(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    target_region: torch.Tensor,
    reference_region: torch.Tensor,
    *,
    min_pixels: int = 8,
    max_samples: int = 256,
    temperature: float = 0.1,
    exclude_background: bool = True,
    bidirectional: bool = True,
) -> tuple[torch.Tensor, int]:
    """Region-wise contextual matching that tolerates local spatial displacement."""

    if prediction.shape != reference.shape:
        raise ValueError(
            f"prediction/reference features must match, got {prediction.shape} and {reference.shape}"
        )
    target_labels = _resize_region_labels(target_region, prediction.shape[-2:])
    reference_labels = _resize_region_labels(reference_region, reference.shape[-2:])
    losses: list[torch.Tensor] = []
    for batch_index in range(prediction.shape[0]):
        region_ids = _shared_region_ids(
            target_labels[batch_index],
            reference_labels[batch_index],
            min_pixels=min_pixels,
            exclude_background=exclude_background,
        )
        for region_id in region_ids:
            pred_vectors = _subsample_vectors(
                _masked_vectors(prediction[batch_index], target_labels[batch_index] == region_id),
                max_samples,
            )
            ref_vectors = _subsample_vectors(
                _masked_vectors(reference[batch_index], reference_labels[batch_index] == region_id),
                max_samples,
            ).detach()
            forward_loss = contextual_directional_loss(
                pred_vectors,
                ref_vectors,
                temperature=temperature,
            )
            if bidirectional:
                reverse_loss = contextual_directional_loss(
                    ref_vectors,
                    pred_vectors,
                    temperature=temperature,
                )
                forward_loss = 0.5 * (forward_loss + reverse_loss)
            losses.append(forward_loss)
    if not losses:
        return prediction.float().sum() * 0.0, 0
    return torch.stack(losses).mean(), len(losses)


class EMALossNormalizer(nn.Module):
    """Normalize heterogeneous losses by a synchronized detached EMA scale."""

    def __init__(
        self,
        names: Sequence[str] = LOSS_NAMES,
        *,
        decay: float = 0.99,
        eps: float = 1e-6,
        enabled: bool = True,
        calibration_steps: int = 200,
    ) -> None:
        super().__init__()
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must satisfy 0 <= decay < 1")
        self.names = tuple(str(name) for name in names)
        self.decay = float(decay)
        self.eps = float(eps)
        self.enabled = bool(enabled)
        self.calibration_steps = int(calibration_steps)
        self.register_buffer("ema", torch.ones(len(self.names), dtype=torch.float32))
        self.register_buffer("initialized", torch.zeros(len(self.names), dtype=torch.bool))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def _global_values(self, values: torch.Tensor) -> torch.Tensor:
        result = values.detach().float()
        if dist.is_available() and dist.is_initialized():
            result = result.clone()
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
            result.div_(dist.get_world_size())
        return result

    def forward(
        self,
        losses: Mapping[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        missing = [name for name in self.names if name not in losses]
        if missing:
            raise KeyError(f"Missing losses for normalization: {missing}")
        if not self.enabled:
            return dict(losses), {name: 1.0 for name in self.names}

        values = torch.stack([losses[name].detach().float() for name in self.names])
        global_values = self._global_values(values)
        calibrating = self.calibration_steps <= 0 or int(self.updates.item()) < self.calibration_steps
        if self.training and calibrating:
            with torch.no_grad():
                valid = torch.isfinite(global_values) & (global_values.abs() > self.eps)
                first = valid & ~self.initialized
                update = valid & self.initialized
                self.ema[first] = global_values[first].abs()
                self.ema[update] = (
                    self.decay * self.ema[update]
                    + (1.0 - self.decay) * global_values[update].abs()
                )
                self.initialized[valid] = True
                self.updates.add_(1)

        scales = torch.where(self.initialized, self.ema, torch.ones_like(self.ema)).clamp_min(
            self.eps
        )
        normalized = {
            name: losses[name] / scales[index].detach().to(losses[name].device)
            for index, name in enumerate(self.names)
        }
        scale_logs = {name: float(scales[index].detach().cpu().item()) for index, name in enumerate(self.names)}
        return normalized, scale_logs


class Pix2PixTransferLoss(nn.Module):
    """Target content plus spatially tolerant regional reference texture loss."""

    def __init__(
        self,
        *,
        lambda_l1: float = 1.0,
        lambda_perc: float = 1.0,
        lambda_gram: float = 1.0,
        lambda_contextual: float = 1.0,
        vgg_weights: str = "imagenet",
        content_layers: Sequence[int] = DEFAULT_CONTENT_LAYERS,
        gram_layers: Sequence[int] = DEFAULT_GRAM_LAYERS,
        contextual_layers: Sequence[int] = DEFAULT_CONTEXTUAL_LAYERS,
        texture_min_pixels: int = 8,
        contextual_max_samples: int = 256,
        contextual_temperature: float = 0.1,
        normalize_losses: bool = True,
        normalization_decay: float = 0.99,
        normalization_steps: int = 200,
        l1_blur_sigma: float = 0.0,
        feature_extractor: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.weights = {
            "l1": float(lambda_l1),
            "content": float(lambda_perc),
            "gram": float(lambda_gram),
            "contextual": float(lambda_contextual),
        }
        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("Loss weights must be non-negative")
        if sum(self.weights.values()) <= 0:
            raise ValueError("At least one loss weight must be positive")

        self.content_layers = parse_layer_indices(content_layers)
        self.gram_layers = parse_layer_indices(gram_layers)
        self.contextual_layers = parse_layer_indices(contextual_layers)
        requested_layers = tuple(
            sorted(set(self.content_layers + self.gram_layers + self.contextual_layers))
        )
        self.feature_extractor = feature_extractor or VGGFeatureExtractor(
            layers=requested_layers,
            weights=vgg_weights,
        )
        self.texture_min_pixels = int(texture_min_pixels)
        self.contextual_max_samples = int(contextual_max_samples)
        self.contextual_temperature = float(contextual_temperature)
        self.l1_blur_sigma = float(l1_blur_sigma)
        if self.l1_blur_sigma < 0.0:
            raise ValueError("L1 blur sigma must be non-negative")
        self.normalizer = EMALossNormalizer(
            enabled=normalize_losses,
            decay=normalization_decay,
            calibration_steps=normalization_steps,
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        reference: torch.Tensor,
        target_region: torch.Tensor,
        reference_region: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l1 = image_l1_loss(pred, target, blur_sigma=self.l1_blur_sigma)
        pred_features = self.feature_extractor(pred)
        with torch.no_grad():
            supervision = torch.cat([target, reference], dim=0)
            supervision_features = self.feature_extractor(supervision)
            target_features = {
                layer: feature[: target.shape[0]] for layer, feature in supervision_features.items()
            }
            reference_features = {
                layer: feature[target.shape[0] :] for layer, feature in supervision_features.items()
            }

        content = feature_content_loss(pred_features, target_features, self.content_layers)

        gram_values: list[torch.Tensor] = []
        contextual_values: list[torch.Tensor] = []
        gram_regions = 0
        contextual_regions = 0
        for layer in self.gram_layers:
            value, count = regional_gram_loss(
                pred_features[layer],
                reference_features[layer],
                target_region,
                reference_region,
                min_pixels=self.texture_min_pixels,
            )
            gram_values.append(value)
            gram_regions += count
        for layer in self.contextual_layers:
            value, count = regional_contextual_loss(
                pred_features[layer],
                reference_features[layer],
                target_region,
                reference_region,
                min_pixels=self.texture_min_pixels,
                max_samples=self.contextual_max_samples,
                temperature=self.contextual_temperature,
            )
            contextual_values.append(value)
            contextual_regions += count

        gram = torch.stack(gram_values).mean()
        contextual = torch.stack(contextual_values).mean()
        raw_losses = {
            "l1": l1,
            "content": content,
            "gram": gram,
            "contextual": contextual,
        }
        normalized, scales = self.normalizer(raw_losses)
        active_weight = sum(weight for weight in self.weights.values() if weight > 0)
        total = sum(
            self.weights[name] * normalized[name]
            for name in LOSS_NAMES
            if self.weights[name] > 0
        ) / active_weight

        logs = {
            "total": float(total.detach().item()),
            "l1": float(l1.detach().item()),
            "perc": float(content.detach().item()),
            "gram": float(gram.detach().item()),
            "contextual": float(contextual.detach().item()),
            "norm_l1": float(normalized["l1"].detach().item()),
            "norm_perc": float(normalized["content"].detach().item()),
            "norm_gram": float(normalized["gram"].detach().item()),
            "norm_contextual": float(normalized["contextual"].detach().item()),
            "scale_l1": scales["l1"],
            "scale_perc": scales["content"],
            "scale_gram": scales["gram"],
            "scale_contextual": scales["contextual"],
            "gram_regions": float(gram_regions),
            "contextual_regions": float(contextual_regions),
        }
        return total, logs

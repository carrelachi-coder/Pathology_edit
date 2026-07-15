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

from .identity_losses import (
    contrast_normalized_grayscale,
    family_image_descriptor,
    feature_moment_loss,
    grayscale_structure_losses,
    laplacian_band_descriptor,
    optical_density_moment_loss,
    selected_reference_ranking_loss,
)


DEFAULT_CONTENT_LAYERS = (3, 8, 15, 22)
DEFAULT_GRAM_LAYERS = (3, 8, 15)
DEFAULT_CONTEXTUAL_LAYERS = (8, 15)
LOSS_NAMES = ("l1", "content", "gram", "contextual")
IDENTITY_LOSS_NAMES = ("identity_od", "identity_feature", "identity_band", "identity_rank", "structure_gray", "structure_edge")


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


def boundary_band_mask(region: torch.Tensor, *, radius: int) -> torch.Tensor:
    """Return a binary band around label transitions."""

    radius = int(radius)
    if region.ndim == 3:
        region = region.unsqueeze(1)
    if region.ndim != 4 or region.shape[1] != 1:
        raise ValueError(f"region must have shape [B,1,H,W] or [B,H,W], got {tuple(region.shape)}")
    if radius <= 0:
        return torch.zeros_like(region, dtype=torch.float32)
    labels = region.float()
    kernel = 2 * radius + 1
    max_values = F.max_pool2d(labels, kernel_size=kernel, stride=1, padding=radius)
    min_values = -F.max_pool2d(-labels, kernel_size=kernel, stride=1, padding=radius)
    return (max_values != min_values).float()


def high_frequency_residual_loss(
    prediction: torch.Tensor,
    i0: torch.Tensor,
    weight: torch.Tensor,
    *,
    blur_sigma: float = 1.0,
) -> torch.Tensor:
    """Penalize high-frequency residual energy under a spatial weight map."""

    residual = prediction.float() - i0.float()
    high = residual - gaussian_blur_image(residual, blur_sigma)
    if weight.ndim == 3:
        weight = weight.unsqueeze(1)
    if weight.ndim != 4 or weight.shape[1] != 1:
        raise ValueError(f"weight must have shape [B,1,H,W] or [B,H,W], got {tuple(weight.shape)}")
    if tuple(weight.shape[-2:]) != tuple(high.shape[-2:]):
        weight = F.interpolate(weight.float(), size=high.shape[-2:], mode="bilinear", align_corners=False)
    weight = weight.to(device=high.device, dtype=high.dtype).clamp(0.0, 1.0)
    denominator = weight.sum() * high.shape[1]
    if float(denominator.detach().cpu().item()) <= 0.0:
        return high.sum() * 0.0
    return (high.abs() * weight).sum() / denominator.clamp_min(1.0)


def _masked_relative_band_error(
    prediction_band: torch.Tensor,
    target_band: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    weight = weight.to(device=prediction_band.device, dtype=prediction_band.dtype)
    denominator = weight.sum() * prediction_band.shape[1]
    if float(denominator.detach().item()) <= 0.0:
        return prediction_band.sum() * 0.0
    numerator = ((prediction_band - target_band.detach()).abs() * weight).sum()
    target_scale = (target_band.detach().abs() * weight).sum() / denominator.clamp_min(1.0)
    return numerator / denominator.clamp_min(1.0) / target_scale.clamp_min(0.05)


def masked_multiband_detail_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    target_region: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compare fine/mid detail only where I0 detail was deliberately removed."""

    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    weight = mask.to(device=prediction.device, dtype=prediction.dtype).clamp(0.0, 1.0)
    if tuple(weight.shape[-2:]) != tuple(prediction.shape[-2:]):
        weight = F.interpolate(weight, size=prediction.shape[-2:], mode="bilinear", align_corners=False)
    # Outside values are detached from the graph so blurred inside pixels cannot
    # send correction gradients into unrelated image regions.
    local_prediction = prediction * weight + target.detach() * (1.0 - weight)
    pred_low1 = gaussian_blur_image(local_prediction, 1.0)
    target_low1 = gaussian_blur_image(target.detach(), 1.0)
    pred_low2 = gaussian_blur_image(local_prediction, 2.5)
    target_low2 = gaussian_blur_image(target.detach(), 2.5)
    pred_bands = (local_prediction - pred_low1, pred_low1 - pred_low2)
    target_bands = (target.detach() - target_low1, target_low1 - target_low2)

    if target_region is None:
        return tuple(
            _masked_relative_band_error(pred_band, target_band, weight)
            for pred_band, target_band in zip(pred_bands, target_bands, strict=True)
        )
    if target_region.ndim == 3:
        target_region = target_region.unsqueeze(1)
    if tuple(target_region.shape[-2:]) != tuple(prediction.shape[-2:]):
        target_region = F.interpolate(target_region.float(), size=prediction.shape[-2:], mode="nearest").long()
    per_band: list[list[torch.Tensor]] = [[], []]
    for label in torch.unique(target_region.detach()).tolist():
        if int(label) == 0:
            continue
        region_weight = weight * target_region.eq(int(label)).to(weight.dtype)
        if float(region_weight.sum().detach().item()) <= 0.0:
            continue
        for band_index, (pred_band, target_band) in enumerate(
            zip(pred_bands, target_bands, strict=True)
        ):
            per_band[band_index].append(
                _masked_relative_band_error(pred_band, target_band, region_weight)
            )
    zero = prediction.sum() * 0.0
    return tuple(
        torch.stack(values).mean() if values else zero
        for values in per_band
    )


def _orientation_field(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gray = contrast_normalized_grayscale(image)
    kernel_x = gray.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    )
    kernels = torch.stack([kernel_x, kernel_x.transpose(0, 1)], dim=0).unsqueeze(1)
    gradient = F.conv2d(gray, kernels, padding=1)
    gx, gy = gradient[:, :1], gradient[:, 1:]
    magnitude_squared = gx.square() + gy.square()
    vectors = torch.cat(
        [gx.square() - gy.square(), 2.0 * gx * gy],
        dim=1,
    ) / magnitude_squared.clamp_min(1.0e-6)
    return vectors, magnitude_squared.sqrt()


def masked_orientation_consistency_loss(
    prediction_a: torch.Tensor,
    prediction_b: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize output direction changes caused only by rotating the reference."""

    vectors_a, _ = _orientation_field(prediction_a)
    vectors_b, _ = _orientation_field(prediction_b)
    vectors_target, target_magnitude = _orientation_field(target.detach())
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    weight = mask.to(device=prediction_a.device, dtype=prediction_a.dtype).clamp(0.0, 1.0)
    if tuple(weight.shape[-2:]) != tuple(prediction_a.shape[-2:]):
        weight = F.interpolate(weight, size=prediction_a.shape[-2:], mode="bilinear", align_corners=False)
    weight = weight * (target_magnitude / target_magnitude.mean(dim=(2, 3), keepdim=True).clamp_min(1.0e-6)).clamp(0.0, 3.0)
    denominator = weight.sum() * vectors_a.shape[1]
    if float(denominator.detach().item()) <= 0.0:
        zero = prediction_a.sum() * 0.0
        return zero, zero
    consistency = ((vectors_a - vectors_b).abs() * weight).sum() / denominator.clamp_min(1.0)
    alignment = 0.5 * (
        ((vectors_a - vectors_target).abs() * weight).sum()
        + ((vectors_b - vectors_target).abs() * weight).sum()
    ) / denominator.clamp_min(1.0)
    return consistency, alignment


def regional_rotation_invariant_style_loss(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    target_region: torch.Tensor,
    reference_region: torch.Tensor,
    *,
    min_pixels: int = 8,
) -> torch.Tensor:
    """Match exact-label color/frequency descriptors without copying direction."""

    if target_region.ndim == 3:
        target_region = target_region.unsqueeze(1)
    if reference_region.ndim == 3:
        reference_region = reference_region.unsqueeze(1)
    shared = set(int(value) for value in torch.unique(target_region.detach()).tolist())
    shared &= set(int(value) for value in torch.unique(reference_region.detach()).tolist())
    values = []
    for label in sorted(shared):
        if label == 0:
            continue
        target_mask = target_region.eq(label)
        reference_mask = reference_region.eq(label)
        if (
            int(target_mask.sum().detach().item()) < int(min_pixels)
            or int(reference_mask.sum().detach().item()) < int(min_pixels)
        ):
            continue
        pred_descriptor = family_image_descriptor(prediction, target_mask)
        reference_descriptor = family_image_descriptor(reference, reference_mask).detach()
        values.append(F.l1_loss(pred_descriptor, reference_descriptor))
    if not values:
        return prediction.sum() * 0.0
    return torch.stack(values).mean()


def build_lowtrust_hf_weight(
    trust_map: torch.Tensor,
    *,
    target_nuclei_mask: torch.Tensor | None = None,
    nuclei_exclusion_radius: int = 0,
) -> torch.Tensor:
    """Weight low-trust residual suppression while exempting nuclei pixels."""

    weight = 1.0 - trust_map.float().clamp(0.0, 1.0)
    if target_nuclei_mask is None:
        return weight
    nuclei = target_nuclei_mask
    if nuclei.ndim == 3:
        nuclei = nuclei.unsqueeze(1)
    if nuclei.ndim != 4 or nuclei.shape[1] != 1:
        raise ValueError(
            "target_nuclei_mask must have shape [B,1,H,W] or [B,H,W], "
            f"got {tuple(nuclei.shape)}"
        )
    if tuple(nuclei.shape[-2:]) != tuple(weight.shape[-2:]):
        nuclei = F.interpolate(nuclei.float(), size=weight.shape[-2:], mode="nearest")
    nuclei = nuclei.to(device=weight.device).ne(0).float()
    radius = max(0, int(nuclei_exclusion_radius))
    if radius > 0:
        nuclei = F.max_pool2d(
            nuclei,
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
        )
    return weight * nuclei.eq(0).to(weight.dtype)


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


def _family_masks(
    tissue_mask: torch.Tensor,
    nuclei_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tissue_mask.ndim == 3:
        tissue_mask = tissue_mask.unsqueeze(1)
    if nuclei_mask.ndim == 3:
        nuclei_mask = nuclei_mask.unsqueeze(1)
    if tissue_mask.ndim != 4 or nuclei_mask.ndim != 4:
        raise ValueError("tissue and nuclei masks must be batched spatial masks")
    nuclei = nuclei_mask.ne(0)
    tissue = tissue_mask.ne(0) & ~nuclei
    foreground = tissue_mask.ne(0) | nuclei
    return foreground, tissue, nuclei


def _supported_pair(
    prediction_mask: torch.Tensor,
    reference_mask: torch.Tensor,
    *,
    min_pixels: int,
) -> torch.Tensor:
    pred_count = prediction_mask.flatten(1).ne(0).sum(dim=1)
    ref_count = reference_mask.flatten(1).ne(0).sum(dim=1)
    return pred_count.ge(int(min_pixels)) & ref_count.ge(int(min_pixels))


def _mean_supported_loss(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    mask_pairs: Sequence[tuple[torch.Tensor, torch.Tensor, int]],
    loss_fn,
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for prediction_mask, reference_mask, min_pixels in mask_pairs:
        valid = _supported_pair(prediction_mask, reference_mask, min_pixels=min_pixels)
        if bool(valid.any().item()):
            values.append(
                loss_fn(
                    prediction[valid],
                    reference[valid],
                    prediction_mask[valid],
                    reference_mask[valid],
                )
            )
    if not values:
        return prediction.float().sum() * 0.0
    return torch.stack(values).mean()


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
        boundary_feather_radius: int = 0,
        lambda_boundary_hf: float = 0.0,
        lambda_lowtrust_hf: float = 0.0,
        residual_hf_blur_sigma: float = 1.0,
        lambda_identity_od: float = 0.0,
        lambda_identity_feature: float = 0.0,
        lambda_identity_band: float = 0.0,
        lambda_identity_rank: float = 0.0,
        cross_lambda_identity_od: float | None = None,
        cross_lambda_identity_feature: float | None = None,
        cross_lambda_identity_band: float | None = None,
        cross_lambda_identity_rank: float | None = None,
        cross_lambda_gram: float = 0.5,
        cross_lambda_contextual: float = 0.5,
        lambda_structure_gray: float = 0.5,
        lambda_structure_edge: float = 0.5,
        identity_rank_margin: float = 0.10,
        identity_feature_layers: Sequence[int] = (3, 8),
        identity_min_tissue_pixels: int = 256,
        identity_min_nuclei_pixels: int = 64,
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
        self.identity_feature_layers = parse_layer_indices(identity_feature_layers)
        requested_layers = tuple(
            sorted(
                set(
                    self.content_layers
                    + self.gram_layers
                    + self.contextual_layers
                    + self.identity_feature_layers
                )
            )
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
        self.boundary_feather_radius = int(boundary_feather_radius)
        self.lambda_boundary_hf = float(lambda_boundary_hf)
        self.lambda_lowtrust_hf = float(lambda_lowtrust_hf)
        self.residual_hf_blur_sigma = float(residual_hf_blur_sigma)
        self.identity_weights = {
            "identity_od": float(lambda_identity_od),
            "identity_feature": float(lambda_identity_feature),
            "identity_band": float(lambda_identity_band),
            "identity_rank": float(lambda_identity_rank),
        }
        self.cross_identity_weights = {
            "identity_od": float(cross_lambda_identity_od if cross_lambda_identity_od is not None else lambda_identity_od),
            "identity_feature": float(
                cross_lambda_identity_feature
                if cross_lambda_identity_feature is not None
                else lambda_identity_feature
            ),
            "identity_band": float(
                cross_lambda_identity_band
                if cross_lambda_identity_band is not None
                else lambda_identity_band
            ),
            "identity_rank": float(
                cross_lambda_identity_rank
                if cross_lambda_identity_rank is not None
                else lambda_identity_rank
            ),
        }
        self.cross_texture_weights = {
            "gram": float(cross_lambda_gram),
            "contextual": float(cross_lambda_contextual),
        }
        self.structure_weights = {
            "structure_gray": float(lambda_structure_gray),
            "structure_edge": float(lambda_structure_edge),
        }
        all_new_weights = (
            list(self.identity_weights.values())
            + list(self.cross_identity_weights.values())
            + list(self.cross_texture_weights.values())
            + list(self.structure_weights.values())
        )
        if any(weight < 0.0 for weight in all_new_weights):
            raise ValueError("Identity, structure and cross-WSI loss weights must be non-negative")
        self.identity_rank_margin = float(identity_rank_margin)
        self.identity_min_tissue_pixels = int(identity_min_tissue_pixels)
        self.identity_min_nuclei_pixels = int(identity_min_nuclei_pixels)
        self.normalizer = EMALossNormalizer(
            enabled=normalize_losses,
            decay=normalization_decay,
            calibration_steps=normalization_steps,
        )
        self.identity_normalizer = EMALossNormalizer(
            names=IDENTITY_LOSS_NAMES,
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
        boundary_region: torch.Tensor | None = None,
        target_tissue_mask: torch.Tensor | None = None,
        target_nuclei_mask: torch.Tensor | None = None,
        reference_tissue_mask: torch.Tensor | None = None,
        reference_nuclei_mask: torch.Tensor | None = None,
        negative_reference: torch.Tensor | None = None,
        negative_reference_tissue_mask: torch.Tensor | None = None,
        negative_reference_nuclei_mask: torch.Tensor | None = None,
        i0: torch.Tensor | None = None,
        trust_map: torch.Tensor | None = None,
        corruption_mask: torch.Tensor | None = None,
        reference_texture_scale: float = 1.0,
        l1_scale: float = 1.0,
        content_scale: float = 1.0,
        gram_scale: float = 1.0,
        contextual_scale: float = 1.0,
        training_mode: str = "same_wsi",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        mode = str(training_mode).strip().lower()
        if mode not in {"same_wsi", "cross_wsi"}:
            raise ValueError("training_mode must be 'same_wsi' or 'cross_wsi'")
        l1 = image_l1_loss(pred, target, blur_sigma=self.l1_blur_sigma) * float(l1_scale)
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

        content = feature_content_loss(pred_features, target_features, self.content_layers) * float(
            content_scale
        )

        gram_values: list[torch.Tensor] = []
        contextual_values: list[torch.Tensor] = []
        gram_regions = 0
        contextual_regions = 0
        boundary_source = target_region if boundary_region is None else boundary_region
        boundary = boundary_band_mask(boundary_source, radius=self.boundary_feather_radius)
        texture_target_region = target_region
        if self.boundary_feather_radius > 0:
            texture_target_region = target_region.clone()
            texture_target_region[boundary.to(device=texture_target_region.device, dtype=torch.bool)] = 0
        for layer in self.gram_layers:
            value, count = regional_gram_loss(
                pred_features[layer],
                reference_features[layer],
                texture_target_region,
                reference_region,
                min_pixels=self.texture_min_pixels,
            )
            gram_values.append(value)
            gram_regions += count
        for layer in self.contextual_layers:
            value, count = regional_contextual_loss(
                pred_features[layer],
                reference_features[layer],
                texture_target_region,
                reference_region,
                min_pixels=self.texture_min_pixels,
                max_samples=self.contextual_max_samples,
                temperature=self.contextual_temperature,
            )
            contextual_values.append(value)
            contextual_regions += count

        texture_scale = float(reference_texture_scale)
        gram = torch.stack(gram_values).mean() * texture_scale * float(gram_scale)
        contextual = (
            torch.stack(contextual_values).mean() * texture_scale * float(contextual_scale)
        )
        raw_losses = {
            "l1": l1,
            "content": content,
            "gram": gram,
            "contextual": contextual,
        }
        normalized, scales = self.normalizer(raw_losses)
        base_weights = (
            {"l1": 0.0, "content": 0.0, **self.cross_texture_weights}
            if mode == "cross_wsi"
            else self.weights
        )
        active_weight = sum(weight for weight in base_weights.values() if weight > 0)
        total = sum(
            base_weights[name] * normalized[name]
            for name in LOSS_NAMES
            if base_weights[name] > 0
        ) / active_weight

        zero = pred.float().sum() * 0.0
        identity_raw = {
            "identity_od": zero,
            "identity_feature": zero,
            "identity_band": zero,
            "identity_rank": zero,
            "structure_gray": zero,
            "structure_edge": zero,
        }
        identity_weights = self.cross_identity_weights if mode == "cross_wsi" else self.identity_weights
        identity_requested = any(weight > 0.0 for weight in identity_weights.values())
        masks = (
            target_tissue_mask,
            target_nuclei_mask,
            reference_tissue_mask,
            reference_nuclei_mask,
        )
        if identity_requested:
            if not all(mask is not None for mask in masks):
                raise ValueError("Identity losses require target/reference tissue and nuclei masks")
            target_families = _family_masks(target_tissue_mask, target_nuclei_mask)
            reference_families = _family_masks(reference_tissue_mask, reference_nuclei_mask)
            mask_pairs = (
                (target_families[0], reference_families[0], self.identity_min_tissue_pixels),
                (target_families[1], reference_families[1], self.identity_min_tissue_pixels),
                (target_families[2], reference_families[2], self.identity_min_nuclei_pixels),
            )
            identity_raw["identity_od"] = _mean_supported_loss(
                pred,
                reference,
                mask_pairs,
                optical_density_moment_loss,
            )
            feature_values = []
            for layer in self.identity_feature_layers:
                feature_values.append(
                    _mean_supported_loss(
                        pred_features[layer],
                        reference_features[layer],
                        mask_pairs,
                        feature_moment_loss,
                    )
                )
            identity_raw["identity_feature"] = torch.stack(feature_values).mean()
            identity_raw["identity_band"] = _mean_supported_loss(
                pred,
                reference,
                mask_pairs,
                lambda pred_image, ref_image, pred_mask, ref_mask: F.l1_loss(
                    laplacian_band_descriptor(pred_image, pred_mask),
                    laplacian_band_descriptor(ref_image, ref_mask).detach(),
                ),
            )
            if (
                negative_reference is not None
                and negative_reference_tissue_mask is not None
                and negative_reference_nuclei_mask is not None
            ):
                negative_families = _family_masks(
                    negative_reference_tissue_mask,
                    negative_reference_nuclei_mask,
                )
                output_descriptor = torch.cat(
                    [family_image_descriptor(pred, family) for family in target_families],
                    dim=1,
                )
                selected_descriptor = torch.cat(
                    [family_image_descriptor(reference, family) for family in reference_families],
                    dim=1,
                )
                negative_descriptor = torch.cat(
                    [
                        family_image_descriptor(negative_reference, family)
                        for family in negative_families
                    ],
                    dim=1,
                )
                identity_raw["identity_rank"] = selected_reference_ranking_loss(
                    output_descriptor,
                    selected_descriptor,
                    negative_descriptor,
                    margin=self.identity_rank_margin,
                )
        if mode == "cross_wsi":
            gray_loss, edge_loss = grayscale_structure_losses(pred, target)
            identity_raw["structure_gray"] = gray_loss
            identity_raw["structure_edge"] = edge_loss
        identity_normalized, identity_scales = self.identity_normalizer(identity_raw)
        total = total + sum(
            weight * identity_normalized[name]
            for name, weight in identity_weights.items()
            if weight > 0.0
        )
        if mode == "cross_wsi":
            total = total + sum(
                weight * identity_normalized[name]
                for name, weight in self.structure_weights.items()
                if weight > 0.0
            )
        boundary_hf = pred.new_zeros(())
        lowtrust_hf = pred.new_zeros(())
        safe_hf_weight = None
        if corruption_mask is not None:
            if corruption_mask.ndim == 3:
                corruption_mask = corruption_mask.unsqueeze(1)
            safe_hf_weight = 1.0 - corruption_mask.to(
                device=pred.device,
                dtype=pred.dtype,
            ).clamp(0.0, 1.0)
        if i0 is not None and self.lambda_boundary_hf > 0.0 and self.boundary_feather_radius > 0:
            boundary_weight = boundary.to(device=pred.device)
            if safe_hf_weight is not None:
                boundary_weight = boundary_weight * safe_hf_weight
            boundary_hf = high_frequency_residual_loss(
                pred,
                i0,
                boundary_weight,
                blur_sigma=self.residual_hf_blur_sigma,
            )
            total = total + boundary_hf * self.lambda_boundary_hf
        if i0 is not None and trust_map is not None and self.lambda_lowtrust_hf > 0.0:
            lowtrust_weight = build_lowtrust_hf_weight(
                trust_map.to(device=pred.device, dtype=pred.dtype),
                target_nuclei_mask=target_nuclei_mask,
                nuclei_exclusion_radius=max(0, int(round(3.0 * self.residual_hf_blur_sigma))),
            )
            if safe_hf_weight is not None:
                lowtrust_weight = lowtrust_weight * safe_hf_weight
            lowtrust_hf = high_frequency_residual_loss(
                pred,
                i0,
                lowtrust_weight,
                blur_sigma=self.residual_hf_blur_sigma,
            )
            total = total + lowtrust_hf * self.lambda_lowtrust_hf

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
            "boundary_hf": float(boundary_hf.detach().item()),
            "lowtrust_hf": float(lowtrust_hf.detach().item()),
            "reference_texture_scale": float(texture_scale),
            "l1_scale": float(l1_scale),
            "content_scale": float(content_scale),
            "gram_scale": float(gram_scale),
            "contextual_scale": float(contextual_scale),
            "rgb_supervision_active": float(mode == "same_wsi"),
            "identity_od": float(identity_raw["identity_od"].detach().item()),
            "identity_feature": float(identity_raw["identity_feature"].detach().item()),
            "identity_band": float(identity_raw["identity_band"].detach().item()),
            "identity_rank": float(identity_raw["identity_rank"].detach().item()),
            "structure_gray": float(identity_raw["structure_gray"].detach().item()),
            "structure_edge": float(identity_raw["structure_edge"].detach().item()),
            "norm_identity_od": float(identity_normalized["identity_od"].detach().item()),
            "norm_identity_feature": float(identity_normalized["identity_feature"].detach().item()),
            "norm_identity_band": float(identity_normalized["identity_band"].detach().item()),
            "norm_identity_rank": float(identity_normalized["identity_rank"].detach().item()),
            "scale_identity_od": identity_scales["identity_od"],
            "scale_identity_feature": identity_scales["identity_feature"],
            "scale_identity_band": identity_scales["identity_band"],
            "scale_identity_rank": identity_scales["identity_rank"],
        }
        return total, logs

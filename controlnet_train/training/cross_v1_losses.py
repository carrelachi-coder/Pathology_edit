"""Auxiliary losses for Cross V1 reference-conditioned training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RegionalStainStyleLossConfig:
    """Weights for region-level stain/style matching."""

    tissue_weight: float = 1.0
    nuclei_weight: float = 1.0
    mean_weight: float = 1.0
    std_weight: float = 1.0
    covariance_weight: float = 0.25
    min_pixels: int = 32
    max_regions_per_sample: int | None = None
    exclude_labels: tuple[int, ...] = (0,)


def per_sample_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return one denoising MSE value per batch item."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    return F.mse_loss(prediction.float(), target.float(), reduction="none").flatten(1).mean(dim=1)


def self_reconstruction_l1_loss(
    *,
    prediction: torch.Tensor,
    reference: torch.Tensor,
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pixel L1 loss for same-patch reference reconstruction samples."""
    _validate_images(prediction, reference)
    per_sample = F.l1_loss(
        prediction.float(),
        reference.detach().float(),
        reduction="none",
    ).flatten(1).mean(dim=1)
    if sample_mask is None:
        return per_sample.mean()

    mask = sample_mask.to(device=per_sample.device, dtype=torch.bool).flatten()
    if mask.shape != per_sample.shape:
        raise ValueError(
            f"sample_mask shape {tuple(mask.shape)} does not match batch shape {tuple(per_sample.shape)}"
        )
    if not bool(mask.any().item()):
        return prediction.new_zeros(())
    return per_sample[mask].mean()


def uni_token_cosine_perceptual_loss(
    *,
    prediction_features: torch.Tensor,
    target_features: torch.Tensor,
) -> torch.Tensor:
    """Cosine distance over frozen UNI patch tokens."""
    if prediction_features.shape != target_features.shape:
        raise ValueError(
            "prediction_features and target_features shapes differ: "
            f"{tuple(prediction_features.shape)} vs {tuple(target_features.shape)}"
        )
    cosine = F.cosine_similarity(
        prediction_features.float(),
        target_features.detach().float(),
        dim=-1,
        eps=1e-6,
    )
    return (1.0 - cosine).mean()


def uni_token_distribution_perceptual_loss(
    *,
    prediction_features: torch.Tensor,
    reference_features: torch.Tensor,
    mean_weight: float = 1.0,
    std_weight: float = 1.0,
    pooled_cosine_weight: float = 0.25,
) -> torch.Tensor:
    """Distributional perceptual loss over frozen UNI patch tokens.

    Unlike token-wise cosine, this compares patch-token statistics and does not
    assume the reference and target images are spatially aligned.
    """
    if prediction_features.shape != reference_features.shape:
        raise ValueError(
            "prediction_features and reference_features shapes differ: "
            f"{tuple(prediction_features.shape)} vs {tuple(reference_features.shape)}"
        )
    if prediction_features.ndim != 3:
        raise ValueError(
            "UNI features must have shape (B, tokens, channels), "
            f"got {tuple(prediction_features.shape)}"
        )

    prediction = prediction_features.float()
    reference = reference_features.detach().float()
    pred_mean = prediction.mean(dim=1)
    ref_mean = reference.mean(dim=1)
    pred_std = torch.sqrt(prediction.var(dim=1, unbiased=False) + 1e-6)
    ref_std = torch.sqrt(reference.var(dim=1, unbiased=False) + 1e-6)

    total = prediction.new_zeros(())
    normalizer = 0.0
    if mean_weight > 0.0:
        total = total + float(mean_weight) * F.l1_loss(pred_mean, ref_mean)
        normalizer += float(mean_weight)
    if std_weight > 0.0:
        total = total + float(std_weight) * F.l1_loss(pred_std, ref_std)
        normalizer += float(std_weight)
    if pooled_cosine_weight > 0.0:
        cosine = F.cosine_similarity(pred_mean, ref_mean, dim=-1, eps=1e-6)
        total = total + float(pooled_cosine_weight) * (1.0 - cosine).mean()
        normalizer += float(pooled_cosine_weight)
    return total / normalizer if normalizer > 0.0 else total


def unpack_flux_packed_latents(
    packed_latents: torch.Tensor,
    *,
    channels: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Inverse of FluxControlNetPipeline._pack_latents for 2x2 latent patches."""
    bsz = packed_latents.shape[0]
    packed_height = int(height) // 2
    packed_width = int(width) // 2
    expected_tokens = packed_height * packed_width
    expected_channels = int(channels) * 4
    if packed_latents.shape[1] != expected_tokens or packed_latents.shape[2] != expected_channels:
        raise ValueError(
            "Packed FLUX latent shape does not match requested unpack shape: "
            f"packed={tuple(packed_latents.shape)}, expected=({bsz}, {expected_tokens}, {expected_channels})"
        )
    latents = packed_latents.reshape(bsz, packed_height, packed_width, channels, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(bsz, channels, height, width)


def ref_swap_sensitivity_loss(
    normal_loss: torch.Tensor,
    swapped_losses: Iterable[torch.Tensor],
    *,
    margin: float,
) -> torch.Tensor:
    """Rank normal-reference denoising below zero/random-reference denoising.

    The loss is active when a swapped reference is not worse than the normal reference
    by at least ``margin``.
    """
    normal = normal_loss.float()
    penalties = []
    for swapped in swapped_losses:
        swapped = swapped.float()
        if swapped.shape != normal.shape:
            raise ValueError(
                f"swapped loss shape {tuple(swapped.shape)} does not match normal {tuple(normal.shape)}"
            )
        penalties.append(F.relu(float(margin) + normal - swapped).mean())
    if not penalties:
        return normal.new_zeros(())
    return torch.stack(penalties).mean()


def regional_stain_style_loss(
    *,
    prediction: torch.Tensor,
    reference: torch.Tensor,
    target_tissue_mask: torch.Tensor,
    reference_tissue_mask: torch.Tensor,
    target_nuclei_mask: torch.Tensor | None = None,
    reference_nuclei_mask: torch.Tensor | None = None,
    sample_mask: torch.Tensor | None = None,
    config: RegionalStainStyleLossConfig | None = None,
) -> dict[str, torch.Tensor | int]:
    """Match color/style statistics inside shared tissue and nuclei classes.

    ``prediction`` is compared on target regions, while ``reference`` is compared on
    reference regions with the same mask label. This handles non-aligned target/ref
    patches: regions are matched by class ID rather than by pixel position.
    """
    cfg = config or RegionalStainStyleLossConfig()
    _validate_images(prediction, reference)
    zero = prediction.new_zeros(())

    tissue_loss, tissue_regions = _regional_mask_loss(
        prediction=prediction,
        reference=reference,
        target_mask=target_tissue_mask,
        reference_mask=reference_tissue_mask,
        sample_mask=sample_mask,
        config=cfg,
    )
    nuclei_loss = zero
    nuclei_regions = 0
    if target_nuclei_mask is not None and reference_nuclei_mask is not None:
        nuclei_loss, nuclei_regions = _regional_mask_loss(
            prediction=prediction,
            reference=reference,
            target_mask=target_nuclei_mask,
            reference_mask=reference_nuclei_mask,
            sample_mask=sample_mask,
            config=cfg,
        )

    weighted = zero
    active_weight = 0.0
    if tissue_regions > 0 and cfg.tissue_weight > 0.0:
        weighted = weighted + float(cfg.tissue_weight) * tissue_loss
        active_weight += float(cfg.tissue_weight)
    if nuclei_regions > 0 and cfg.nuclei_weight > 0.0:
        weighted = weighted + float(cfg.nuclei_weight) * nuclei_loss
        active_weight += float(cfg.nuclei_weight)

    total = weighted / active_weight if active_weight > 0.0 else zero
    return {
        "total": total,
        "tissue": tissue_loss,
        "nuclei": nuclei_loss,
        "tissue_regions": tissue_regions,
        "nuclei_regions": nuclei_regions,
    }


def _regional_mask_loss(
    *,
    prediction: torch.Tensor,
    reference: torch.Tensor,
    target_mask: torch.Tensor,
    reference_mask: torch.Tensor,
    sample_mask: torch.Tensor | None = None,
    config: RegionalStainStyleLossConfig,
) -> tuple[torch.Tensor, int]:
    image_size = tuple(int(v) for v in prediction.shape[-2:])
    target_mask = _resize_mask_to_image(target_mask, image_size)
    reference_mask = _resize_mask_to_image(reference_mask, image_size)
    if target_mask.shape != reference_mask.shape:
        raise ValueError(
            f"target/reference mask batch shapes differ: {tuple(target_mask.shape)} vs {tuple(reference_mask.shape)}"
        )
    if target_mask.shape[0] != prediction.shape[0]:
        raise ValueError(
            f"mask batch size {target_mask.shape[0]} does not match image batch size {prediction.shape[0]}"
        )
    if sample_mask is None:
        active_samples = torch.ones(prediction.shape[0], device=prediction.device, dtype=torch.bool)
    else:
        active_samples = sample_mask.to(device=prediction.device, dtype=torch.bool).flatten()
        if active_samples.shape[0] != prediction.shape[0]:
            raise ValueError(
                f"sample_mask shape {tuple(active_samples.shape)} does not match batch size {prediction.shape[0]}"
            )

    losses = []
    exclude = set(int(label) for label in config.exclude_labels)
    min_pixels = max(1, int(config.min_pixels))
    for batch_index in range(prediction.shape[0]):
        if not bool(active_samples[batch_index].item()):
            continue
        labels = _shared_labels(
            target_mask[batch_index],
            reference_mask[batch_index],
            exclude_labels=exclude,
        )
        if config.max_regions_per_sample is not None and len(labels) > config.max_regions_per_sample:
            labels = _largest_labels(
                labels,
                target_mask[batch_index],
                max_regions=int(config.max_regions_per_sample),
            )
        for label in labels:
            target_region = target_mask[batch_index] == label
            reference_region = reference_mask[batch_index] == label
            if int(target_region.sum().item()) < min_pixels:
                continue
            if int(reference_region.sum().item()) < min_pixels:
                continue
            losses.append(
                _region_stain_style_loss(
                    prediction[batch_index],
                    reference[batch_index],
                    target_region,
                    reference_region,
                    config=config,
                )
            )
    if not losses:
        return prediction.new_zeros(()), 0
    return torch.stack(losses).mean(), len(losses)


def _region_stain_style_loss(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    target_region: torch.Tensor,
    reference_region: torch.Tensor,
    *,
    config: RegionalStainStyleLossConfig,
) -> torch.Tensor:
    pred_stats = _region_color_stats(prediction, target_region)
    ref_stats = _region_color_stats(reference.detach(), reference_region)
    total = prediction.new_zeros(())
    normalizer = 0.0
    if config.mean_weight > 0.0:
        total = total + float(config.mean_weight) * F.l1_loss(pred_stats["mean"], ref_stats["mean"])
        normalizer += float(config.mean_weight)
    if config.std_weight > 0.0:
        total = total + float(config.std_weight) * F.l1_loss(pred_stats["std"], ref_stats["std"])
        normalizer += float(config.std_weight)
    if config.covariance_weight > 0.0:
        total = total + float(config.covariance_weight) * F.l1_loss(
            pred_stats["covariance"],
            ref_stats["covariance"],
        )
        normalizer += float(config.covariance_weight)
    return total / normalizer if normalizer > 0.0 else total


def _region_color_stats(image: torch.Tensor, region: torch.Tensor) -> dict[str, torch.Tensor]:
    values = image[:, region].float()
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("region must select at least one pixel")
    mean = values.mean(dim=1)
    centered = values - mean[:, None]
    variance = centered.square().mean(dim=1)
    std = torch.sqrt(variance + 1e-6)
    covariance = centered @ centered.t() / max(int(values.shape[1]), 1)
    return {"mean": mean, "std": std, "covariance": covariance}


def _resize_mask_to_image(mask: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 3:
        raise ValueError(f"mask must have shape (B,H,W) or (B,1,H,W), got {tuple(mask.shape)}")
    if tuple(int(v) for v in mask.shape[-2:]) == image_size:
        return mask.to(dtype=torch.long)
    resized = F.interpolate(
        mask.unsqueeze(1).float(),
        size=image_size,
        mode="nearest",
    )
    return resized[:, 0].to(dtype=torch.long)


def _shared_labels(
    target_mask: torch.Tensor,
    reference_mask: torch.Tensor,
    *,
    exclude_labels: set[int],
) -> list[int]:
    target_labels = {int(value) for value in torch.unique(target_mask.detach()).cpu().tolist()}
    reference_labels = {int(value) for value in torch.unique(reference_mask.detach()).cpu().tolist()}
    return sorted((target_labels & reference_labels) - exclude_labels)


def _largest_labels(labels: list[int], mask: torch.Tensor, *, max_regions: int) -> list[int]:
    ranked = sorted(
        labels,
        key=lambda label: int((mask == label).sum().item()),
        reverse=True,
    )
    return ranked[: max(0, max_regions)]


def _validate_images(prediction: torch.Tensor, reference: torch.Tensor) -> None:
    if prediction.ndim != 4 or reference.ndim != 4:
        raise ValueError(
            f"prediction/reference must have shape (B,C,H,W), got {tuple(prediction.shape)} and {tuple(reference.shape)}"
        )
    if prediction.shape != reference.shape:
        raise ValueError(
            f"prediction/reference shapes differ: {tuple(prediction.shape)} vs {tuple(reference.shape)}"
        )
    if prediction.shape[1] != 3:
        raise ValueError(f"expected RGB tensors with C=3, got C={prediction.shape[1]}")

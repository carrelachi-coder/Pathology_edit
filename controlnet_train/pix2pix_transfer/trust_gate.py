"""Reference trust utilities for pix2pix texture refinement."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

NUCLEI_LABEL_OFFSET = 256


def _as_label_map(region: torch.Tensor) -> torch.Tensor:
    if region.ndim == 4:
        if region.shape[1] != 1:
            raise ValueError(f"region must have one channel, got {tuple(region.shape)}")
        region = region[:, 0]
    if region.ndim != 3:
        raise ValueError(f"region must be BxHxW or Bx1xHxW, got {tuple(region.shape)}")
    return region.long()


def _resize_reference_labels(reference_region: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    reference = _as_label_map(reference_region)
    if tuple(reference.shape[-2:]) == tuple(size):
        return reference
    return F.interpolate(reference[:, None].float(), size=size, mode="nearest")[:, 0].long()


def build_reference_trust_map(
    target_region: torch.Tensor,
    reference_region: torch.Tensor,
    *,
    fallback_scale: float = 0.05,
    min_region_pixels: int = 8,
    low_trust_threshold: float = 0.5,
    matched_tissue_floor: float = 0.0,
    matched_nuclei_floor: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Estimate how much each target pixel should trust local reference texture.

    Trust is region-specific.  Matched regions receive trust based on area
    compatibility; unmatched target regions receive ``fallback_scale`` so I0
    remains the dominant source while still permitting weak global inspiration.
    Nuclei labels are kept separate from tissue labels by their offset.
    """

    target = _as_label_map(target_region)
    reference = _resize_reference_labels(reference_region, tuple(target.shape[-2:])).to(device=target.device)
    fallback = float(max(0.0, min(1.0, fallback_scale)))
    min_pixels = max(1, int(min_region_pixels))
    tissue_floor = float(max(fallback, min(1.0, matched_tissue_floor)))
    nuclei_floor = float(max(fallback, min(1.0, matched_nuclei_floor)))
    trust = torch.full((target.shape[0], 1, target.shape[1], target.shape[2]), fallback, device=target.device)
    unmatched_regions = 0
    region_count = 0

    for batch_index in range(target.shape[0]):
        target_labels = target[batch_index]
        reference_labels = reference[batch_index]
        foreground_pixels = int((target_labels != 0).sum().item())
        reference_foreground = int((reference_labels != 0).sum().item())
        target_denom = max(1, foreground_pixels)
        reference_denom = max(1, reference_foreground)
        for raw_label in torch.unique(target_labels).tolist():
            label = int(raw_label)
            if label == 0:
                continue
            region_count += 1
            target_mask = target_labels == label
            reference_mask = reference_labels == label
            target_count = int(target_mask.sum().item())
            reference_count = int(reference_mask.sum().item())
            if target_count < min_pixels or reference_count < min_pixels:
                unmatched_regions += 1
                value = fallback
            else:
                target_fraction = float(target_count) / float(target_denom)
                reference_fraction = float(reference_count) / float(reference_denom)
                larger = max(target_fraction, reference_fraction, 1.0e-6)
                smaller = min(target_fraction, reference_fraction)
                value = max(fallback, min(1.0, smaller / larger))
                value = max(value, nuclei_floor if label >= NUCLEI_LABEL_OFFSET else tissue_floor)
            trust[batch_index, 0][target_mask] = value

    low_trust = (trust < float(low_trust_threshold)).float()
    stats = {
        "mean_trust": float(trust.detach().float().mean().cpu().item()),
        "low_trust_fraction": float(low_trust.detach().mean().cpu().item()),
        "unmatched_regions": float(unmatched_regions),
        "region_count": float(region_count),
    }
    return trust.to(dtype=torch.float32), stats


def _rotate_label_map(label_map: torch.Tensor, angle_degrees: float) -> torch.Tensor:
    labels = _as_label_map(label_map)
    angle = float(angle_degrees)
    if abs(angle) <= 1.0e-6:
        return labels
    radians = angle * (math.pi / 180.0)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    theta = torch.tensor(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0]],
        device=labels.device,
        dtype=torch.float32,
    ).unsqueeze(0).expand(labels.shape[0], -1, -1)
    grid = F.affine_grid(
        theta,
        size=(labels.shape[0], 1, labels.shape[1], labels.shape[2]),
        align_corners=False,
    )
    return F.grid_sample(
        labels[:, None].float(),
        grid,
        mode="nearest",
        padding_mode="reflection",
        align_corners=False,
    )[:, 0].round().long()


def build_highres_nuclei_reference_trust_map(
    target_nuclei_mask: torch.Tensor,
    reference_nuclei_mask: torch.Tensor,
    *,
    reference_weights: torch.Tensor | None = None,
    candidate_angles_degrees: Sequence[float] = (0.0,),
    reference_pool_size: int = 8,
    unmatched_scale: float = 0.20,
    matched_floor: float = 0.60,
    sufficient_reference_tokens: int = 4,
    min_reference_pixels: int = 64,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Build a nuclei-only trust gate for pooled high-resolution attention.

    Tissue and background pixels always retain trust 1.0.  Each target nuclei
    class is attenuated according to both its full-resolution reference density
    and the number of same-class tokens that survive reference pooling.  When
    rotation weights are provided, support is combined per candidate angle so
    an unsupported rotated reference cannot dominate an otherwise valid one.
    """

    target = _as_label_map(target_nuclei_mask)
    reference = _as_label_map(reference_nuclei_mask).to(device=target.device)
    if reference.shape[0] != target.shape[0]:
        raise ValueError("target and reference nuclei masks must have the same batch size")
    angles = tuple(float(value) for value in candidate_angles_degrees)
    if not angles:
        raise ValueError("candidate_angles_degrees cannot be empty")
    pool_size = max(1, int(reference_pool_size))
    sufficient_tokens = max(1, int(sufficient_reference_tokens))
    minimum_pixels = max(1, int(min_reference_pixels))
    unmatched = float(max(0.0, min(1.0, unmatched_scale)))
    floor = float(max(unmatched, min(1.0, matched_floor)))

    if reference_weights is None:
        weights = torch.full(
            (target.shape[0], len(angles), target.shape[1], target.shape[2]),
            1.0 / float(len(angles)),
            device=target.device,
            dtype=torch.float32,
        )
    else:
        weights = reference_weights.float().to(device=target.device)
        if weights.ndim != 4 or weights.shape[:2] != (target.shape[0], len(angles)):
            raise ValueError(
                "reference_weights must have shape [B,K,H,W] matching candidate angles"
            )
        if tuple(weights.shape[-2:]) != tuple(target.shape[-2:]):
            weights = F.interpolate(
                weights,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        weights = weights.clamp_min(0.0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-8)

    pooled_references = []
    for angle in angles:
        rotated = _rotate_label_map(reference, angle)
        pooled_references.append(
            F.interpolate(
                rotated[:, None].float(),
                size=(pool_size, pool_size),
                mode="nearest",
            )[:, 0].round().long()
        )

    trust = torch.ones(
        (target.shape[0], 1, target.shape[1], target.shape[2]),
        device=target.device,
        dtype=torch.float32,
    )
    class_count = 0
    attenuated_classes = 0
    missing_effective_classes = 0
    for batch_index in range(target.shape[0]):
        target_pixels_total = max(1, target[batch_index].numel())
        reference_pixels_total = max(1, reference[batch_index].numel())
        for raw_label in torch.unique(target[batch_index]).tolist():
            label = int(raw_label)
            if label == 0:
                continue
            class_count += 1
            target_class = target[batch_index].eq(label)
            target_count = int(target_class.sum().item())
            reference_count = int(reference[batch_index].eq(label).sum().item())
            target_fraction = float(target_count) / float(target_pixels_total)
            reference_fraction = float(reference_count) / float(reference_pixels_total)
            density_support = min(
                1.0,
                reference_fraction / max(target_fraction, 1.0e-8),
            )
            candidate_support = []
            has_missing_effective_candidate = False
            for pooled in pooled_references:
                token_count = int(pooled[batch_index].eq(label).sum().item())
                if reference_count < minimum_pixels or token_count == 0:
                    value = unmatched
                    has_missing_effective_candidate = True
                else:
                    token_support = min(1.0, float(token_count) / float(sufficient_tokens))
                    value = max(floor, min(density_support, token_support))
                candidate_support.append(value)
            support_tensor = torch.tensor(
                candidate_support,
                device=target.device,
                dtype=torch.float32,
            )
            class_weights = weights[batch_index].permute(1, 2, 0)[target_class]
            pixel_trust = (class_weights * support_tensor).sum(dim=1)
            trust[batch_index, 0, target_class] = pixel_trust
            if bool(pixel_trust.lt(1.0 - 1.0e-6).any().item()):
                attenuated_classes += 1
            if has_missing_effective_candidate:
                missing_effective_classes += 1

    nuclei = target.ne(0)
    nuclei_values = trust[:, 0][nuclei]
    if nuclei_values.numel() == 0:
        mean_nuclei_trust = 1.0
        min_nuclei_trust = 1.0
        attenuated_nuclei_fraction = 0.0
    else:
        mean_nuclei_trust = float(nuclei_values.mean().item())
        min_nuclei_trust = float(nuclei_values.min().item())
        attenuated_nuclei_fraction = float(
            nuclei_values.lt(1.0 - 1.0e-6).float().mean().item()
        )
    stats = {
        "mean_nuclei_trust": mean_nuclei_trust,
        "min_nuclei_trust": min_nuclei_trust,
        "attenuated_nuclei_fraction": attenuated_nuclei_fraction,
        "class_count": float(class_count),
        "attenuated_classes": float(attenuated_classes),
        "missing_effective_classes": float(missing_effective_classes),
        "reference_pool_size": float(pool_size),
        "unmatched_scale": unmatched,
        "matched_floor": floor,
        "sufficient_reference_tokens": float(sufficient_tokens),
        "min_reference_pixels": float(minimum_pixels),
    }
    return trust, stats

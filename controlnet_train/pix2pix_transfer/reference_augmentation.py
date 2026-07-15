"""Synchronized train-time augmentation for active pix2pix references."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RotatedReferenceBundle:
    reference_cond: torch.Tensor
    reference_region: torch.Tensor
    reference_tissue_mask: torch.Tensor
    reference_nuclei_mask: torch.Tensor
    angles_degrees: torch.Tensor


def ramped_rotation_probability(
    target_probability: float,
    *,
    step: int,
    ramp_steps: int,
) -> float:
    """Linearly warm the rotation rate while preserving the requested ceiling."""

    target = min(max(float(target_probability), 0.0), 1.0)
    if ramp_steps <= 0:
        return target
    progress = min(max((int(step) + 1) / float(ramp_steps), 0.0), 1.0)
    return target * progress


def sample_continuous_rotation_angle(
    *,
    probability: float,
    min_degrees: float,
    max_degrees: float,
    generator: torch.Generator | None = None,
) -> float:
    probability = min(max(float(probability), 0.0), 1.0)
    minimum = max(0.0, float(min_degrees))
    maximum = max(minimum, float(max_degrees))
    if probability <= 0.0 or float(torch.rand((), generator=generator).item()) >= probability:
        return 0.0
    magnitude = minimum + float(torch.rand((), generator=generator).item()) * (maximum - minimum)
    direction = -1.0 if float(torch.rand((), generator=generator).item()) < 0.5 else 1.0
    return direction * magnitude


def _label_tensor(value: torch.Tensor, *, name: str, batch_size: int) -> torch.Tensor:
    if value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != 4 or value.shape[0] != batch_size or value.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B,1,H,W] or [B,H,W]")
    return value


def _rotate_labels(value: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    dtype = value.dtype
    rotated = F.grid_sample(
        value.float(),
        grid,
        mode="nearest",
        padding_mode="reflection",
        align_corners=False,
    )
    return rotated.round().to(dtype=dtype)


def rotate_reference_bundle(
    reference_cond: torch.Tensor,
    reference_region: torch.Tensor,
    reference_tissue_mask: torch.Tensor,
    reference_nuclei_mask: torch.Tensor,
    *,
    angles_degrees: torch.Tensor | float,
) -> RotatedReferenceBundle:
    """Rotate RGB and every reference-side mask with one shared continuous transform."""

    if reference_cond.ndim != 4 or reference_cond.shape[1] < 3:
        raise ValueError("reference_cond must have shape [B,C,H,W] with at least three RGB channels")
    batch_size, _, height, width = reference_cond.shape
    region = _label_tensor(reference_region, name="reference_region", batch_size=batch_size)
    tissue = _label_tensor(reference_tissue_mask, name="reference_tissue_mask", batch_size=batch_size)
    nuclei = _label_tensor(reference_nuclei_mask, name="reference_nuclei_mask", batch_size=batch_size)
    if any(tuple(value.shape[-2:]) != (height, width) for value in (region, tissue, nuclei)):
        raise ValueError("reference condition and masks must have identical spatial dimensions")

    angles = torch.as_tensor(angles_degrees, device=reference_cond.device, dtype=torch.float32)
    if angles.ndim == 0:
        angles = angles.expand(batch_size)
    if angles.ndim != 1 or angles.shape[0] != batch_size:
        raise ValueError(f"angles_degrees must be scalar or shape [{batch_size}]")
    if not bool(angles.ne(0.0).any().item()):
        return RotatedReferenceBundle(reference_cond, region, tissue, nuclei, angles)

    radians = angles * (math.pi / 180.0)
    cosine = torch.cos(radians)
    sine = torch.sin(radians)
    theta = torch.zeros(batch_size, 2, 3, device=reference_cond.device, dtype=torch.float32)
    theta[:, 0, 0] = cosine
    theta[:, 0, 1] = -sine
    theta[:, 1, 0] = sine
    theta[:, 1, 1] = cosine
    grid = F.affine_grid(theta, size=(batch_size, 1, height, width), align_corners=False)

    rgb = F.grid_sample(
        reference_cond[:, :3].float(),
        grid,
        mode="bicubic",
        padding_mode="reflection",
        align_corners=False,
    ).clamp(-1.0, 1.0)
    condition_masks = F.grid_sample(
        reference_cond[:, 3:].float(),
        grid,
        mode="nearest",
        padding_mode="reflection",
        align_corners=False,
    )
    rotated_condition = torch.cat([rgb, condition_masks], dim=1).to(dtype=reference_cond.dtype)
    return RotatedReferenceBundle(
        reference_cond=rotated_condition,
        reference_region=_rotate_labels(region, grid),
        reference_tissue_mask=_rotate_labels(tissue, grid),
        reference_nuclei_mask=_rotate_labels(nuclei, grid),
        angles_degrees=angles,
    )

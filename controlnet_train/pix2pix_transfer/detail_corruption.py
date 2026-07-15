"""Train-only I0 detail dropout and context counterfactual utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .losses import gaussian_blur_image


@dataclass(frozen=True)
class DetailDropoutResult:
    image: torch.Tensor
    mask: torch.Tensor
    active: torch.Tensor
    sigma: torch.Tensor


def _randint(low: int, high: int, generator: torch.Generator | None) -> int:
    if high <= low:
        return int(low)
    return int(torch.randint(low, high + 1, (1,), generator=generator).item())


def _uniform(low: float, high: float, generator: torch.Generator | None) -> float:
    if high <= low:
        return float(low)
    value = float(torch.rand((), generator=generator).item())
    return float(low) + value * (float(high) - float(low))


def rotate_batch_d4(tensor: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """Apply an independently selected exact D4 transform to each sample."""

    if tensor.ndim < 3:
        raise ValueError(f"tensor must have a batch and spatial dimensions, got {tuple(tensor.shape)}")
    if codes.ndim != 1 or codes.shape[0] != tensor.shape[0]:
        raise ValueError(f"codes must have shape [{tensor.shape[0]}], got {tuple(codes.shape)}")
    transformed = []
    for sample, raw_code in zip(tensor, codes.detach().cpu().tolist(), strict=True):
        code = int(raw_code) % 8
        value = torch.rot90(sample, code % 4, dims=(-2, -1))
        if code >= 4:
            value = torch.flip(value, dims=(-1,))
        transformed.append(value)
    return torch.stack(transformed, dim=0)


def sample_nonzero_d4_codes(
    batch_size: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    codes = torch.randint(1, 8, (int(batch_size),), generator=generator, device="cpu")
    return codes.to(device=device) if device is not None else codes


def apply_local_detail_dropout(
    i0: torch.Tensor,
    target_region: torch.Tensor,
    *,
    probability: float = 0.20,
    min_diameter: int = 32,
    max_diameter: int = 96,
    sigma_min: float = 1.2,
    sigma_max: float = 2.5,
    feather_radius: int = 5,
    eligible_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> DetailDropoutResult:
    """Remove local I0 detail while preserving its low-frequency content."""

    if i0.ndim != 4 or i0.shape[1] != 3:
        raise ValueError(f"i0 must have shape [B,3,H,W], got {tuple(i0.shape)}")
    if target_region.ndim == 3:
        target_region = target_region.unsqueeze(1)
    if target_region.ndim != 4 or target_region.shape[1] != 1:
        raise ValueError(
            f"target_region must have shape [B,1,H,W] or [B,H,W], got {tuple(target_region.shape)}"
        )
    if tuple(target_region.shape[-2:]) != tuple(i0.shape[-2:]):
        raise ValueError("target_region and i0 must have identical spatial dimensions")
    if eligible_mask is not None:
        if eligible_mask.ndim == 3:
            eligible_mask = eligible_mask.unsqueeze(1)
        if eligible_mask.ndim != 4 or eligible_mask.shape[:2] != (i0.shape[0], 1):
            raise ValueError("eligible_mask must have shape [B,1,H,W] or [B,H,W]")
        if tuple(eligible_mask.shape[-2:]) != tuple(i0.shape[-2:]):
            raise ValueError("eligible_mask and i0 must have identical spatial dimensions")
        eligible_mask = eligible_mask.to(device=i0.device).gt(0)

    batch, _, height, width = i0.shape
    output = i0.clone()
    masks = i0.new_zeros((batch, 1, height, width))
    active = torch.zeros(batch, dtype=torch.bool, device=i0.device)
    sigmas = torch.zeros(batch, dtype=torch.float32, device=i0.device)
    yy = torch.arange(height, device=i0.device, dtype=torch.float32).view(height, 1)
    xx = torch.arange(width, device=i0.device, dtype=torch.float32).view(1, width)

    for index in range(batch):
        if float(torch.rand((), generator=generator).item()) >= float(probability):
            continue
        if eligible_mask is None:
            labels = torch.unique(target_region[index, 0].detach().cpu())
            labels = labels[labels.ne(0)]
            if labels.numel() == 0:
                continue
            label = labels[_randint(0, int(labels.numel()) - 1, generator)].to(
                target_region.device
            )
            sample_eligible = target_region[index, 0].eq(label)
        else:
            sample_eligible = eligible_mask[index, 0]
        coordinates = torch.nonzero(sample_eligible, as_tuple=False)
        if coordinates.numel() == 0:
            continue
        center = coordinates[_randint(0, int(coordinates.shape[0]) - 1, generator)]
        diameter = _randint(
            max(2, min(int(min_diameter), min(height, width))),
            max(2, min(int(max_diameter), min(height, width))),
            generator,
        )
        radius_y = max(1.0, diameter * _uniform(0.35, 0.50, generator))
        radius_x = max(1.0, diameter * _uniform(0.35, 0.50, generator))
        ellipse = (
            ((yy - float(center[0])) / radius_y).square()
            + ((xx - float(center[1])) / radius_x).square()
        ).le(1.0)
        mask = (ellipse & sample_eligible).to(dtype=i0.dtype).view(1, 1, height, width)
        if int(feather_radius) > 0:
            mask = gaussian_blur_image(mask, max(0.5, float(feather_radius) / 2.0))
            mask = (mask / mask.amax().clamp_min(1.0e-6)).clamp(0.0, 1.0)
            mask = mask * sample_eligible.to(dtype=i0.dtype).view(1, 1, height, width)
        sigma = _uniform(sigma_min, sigma_max, generator)
        low = gaussian_blur_image(i0[index : index + 1], sigma)
        output[index : index + 1] = (
            i0[index : index + 1] * (1.0 - mask) + low * mask
        ).clamp(-1.0, 1.0)
        masks[index : index + 1] = mask
        active[index] = True
        sigmas[index] = sigma

    return DetailDropoutResult(image=output, mask=masks, active=active, sigma=sigmas)


def build_context_mismatch_condition(
    region_condition: torch.Tensor,
    corruption_mask: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    ring_radius: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep the corrupted core label but replace only its surrounding context."""

    if region_condition.ndim != 4:
        raise ValueError(f"region_condition must be BCHW, got {tuple(region_condition.shape)}")
    if corruption_mask.ndim == 3:
        corruption_mask = corruption_mask.unsqueeze(1)
    if corruption_mask.ndim != 4 or corruption_mask.shape[1] != 1:
        raise ValueError(f"corruption_mask must be B1HW, got {tuple(corruption_mask.shape)}")
    core = corruption_mask.to(device=region_condition.device).gt(0.5)
    radius = max(1, int(ring_radius))
    dilated = F.max_pool2d(
        core.float(),
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    ).gt(0.0)
    ring = dilated & ~core
    if region_condition.shape[0] > 1:
        source = torch.roll(region_condition, shifts=1, dims=0)
    else:
        codes = sample_nonzero_d4_codes(
            1,
            generator=generator,
            device=region_condition.device,
        )
        source = rotate_batch_d4(region_condition, codes)
    wrong = torch.where(ring.expand_as(region_condition), source, region_condition)
    return wrong, core, ring

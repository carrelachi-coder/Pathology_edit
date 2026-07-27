from __future__ import annotations

import torch
import torch.nn.functional as F


def sanitize_target(target: torch.Tensor, num_classes: int, invalid_to: int = 255) -> torch.Tensor:
    target = target.long()
    invalid = (target < 0) | (target >= num_classes)
    if invalid.any():
        target = target.clone()
        target[invalid] = invalid_to
    return target


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    class_weights: torch.Tensor | None = None,
    invalid_to: int = 7,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = logits.softmax(dim=1)
    target = sanitize_target(target, num_classes=num_classes, invalid_to=invalid_to)
    valid = (target >= 0) & (target < num_classes)
    safe_target = target.clamp(0, num_classes - 1)
    target_1h = F.one_hot(safe_target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid_f = valid.unsqueeze(1).to(dtype=target_1h.dtype)
    probs = probs * valid_f
    target_1h = target_1h * valid_f
    dims = (0, 2, 3)
    intersect = (probs * target_1h).sum(dims)
    denom = probs.sum(dims) + target_1h.sum(dims)
    dice = (2 * intersect + eps) / (denom + eps)
    loss = 1.0 - dice
    if class_weights is not None:
        weights = class_weights.to(loss.device, dtype=loss.dtype)
        return (loss * weights).sum() / weights.sum().clamp_min(eps)
    return loss.mean()


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    class_weights: torch.Tensor | None = None,
    invalid_to: int = 7,
) -> dict[str, torch.Tensor]:
    target = sanitize_target(target, num_classes=num_classes, invalid_to=invalid_to)
    ce = F.cross_entropy(logits, target.long(), weight=class_weights, ignore_index=invalid_to)
    dice = dice_loss(logits, target, num_classes=num_classes, class_weights=class_weights, invalid_to=invalid_to)
    total = ce + dice
    return {"total": total, "ce": ce, "dice": dice}


def masked_segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    class_weights: torch.Tensor | None = None,
    ignore_index: int = 255,
) -> dict[str, torch.Tensor]:
    """Segmentation loss that remains finite when a batch has no fine labels."""
    valid = (target >= 0) & (target < num_classes) & (target != ignore_index)
    if not valid.any():
        # Summing masked logits can overflow because disallowed classes use the
        # dtype minimum. A single-element dependency keeps the zero differentiable.
        zero = logits.reshape(-1)[0] * 0.0
        return {"total": zero, "ce": zero, "dice": zero, "valid_pixels": zero.detach()}
    losses = segmentation_loss(
        logits,
        target,
        num_classes=num_classes,
        class_weights=class_weights,
        invalid_to=ignore_index,
    )
    losses["valid_pixels"] = valid.sum().to(dtype=logits.dtype)
    return losses


def soft_boundary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    width: int = 1,
    ignore_index: int = 255,
) -> torch.Tensor:
    target = sanitize_target(target, num_classes=num_classes, invalid_to=ignore_index)
    valid = target != ignore_index
    safe_target = target.clone().long()
    safe_target[~valid] = 0
    target_one_hot = F.one_hot(safe_target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    probabilities = logits.softmax(dim=1)
    kernel = 2 * width + 1

    def boundary_map(values: torch.Tensor) -> torch.Tensor:
        dilation = F.max_pool2d(values, kernel_size=kernel, stride=1, padding=width)
        erosion = -F.max_pool2d(-values, kernel_size=kernel, stride=1, padding=width)
        return (dilation - erosion).clamp(0.0, 1.0)

    difference = (boundary_map(probabilities) - boundary_map(target_one_hot)).abs()
    valid_map = valid[:, None].expand_as(difference)
    return difference[valid_map].mean() if valid_map.any() else logits.sum() * 0.0


def multi_scale_soft_boundary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    widths: tuple[int, ...] = (2, 4, 8),
    ignore_index: int = 255,
) -> torch.Tensor:
    if not widths or any(width < 1 for width in widths):
        raise ValueError("boundary widths must contain positive integers")
    losses = [
        soft_boundary_loss(
            logits,
            target,
            num_classes,
            width=width,
            ignore_index=ignore_index,
        )
        for width in widths
    ]
    return torch.stack(losses).mean()


def target_boundary_band(
    target: torch.Tensor,
    num_classes: int,
    width: int = 4,
    ignore_index: int = 255,
) -> torch.Tensor:
    if width < 1:
        raise ValueError("boundary width must be positive")
    target = sanitize_target(target, num_classes=num_classes, invalid_to=ignore_index)
    valid = target != ignore_index
    safe_target = target.clone().long()
    safe_target[~valid] = 0
    one_hot = F.one_hot(safe_target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    kernel = 2 * width + 1
    dilation = F.max_pool2d(one_hot, kernel_size=kernel, stride=1, padding=width)
    erosion = -F.max_pool2d(-one_hot, kernel_size=kernel, stride=1, padding=width)
    return ((dilation - erosion).amax(dim=1) > 0) & valid


def boundary_band_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    width: int = 4,
    class_weights: torch.Tensor | None = None,
    ignore_index: int = 255,
) -> torch.Tensor:
    target = sanitize_target(target, num_classes=num_classes, invalid_to=ignore_index)
    band = target_boundary_band(target, num_classes, width=width, ignore_index=ignore_index)
    if not band.any():
        return logits.reshape(-1)[0] * 0.0
    losses = F.cross_entropy(
        logits,
        target.long(),
        weight=class_weights,
        ignore_index=ignore_index,
        reduction="none",
    )
    return losses[band].mean()


def outside_boundary_consistency_loss(
    refined_logits: torch.Tensor,
    base_logits: torch.Tensor,
    refinement_gate: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
) -> torch.Tensor:
    if refinement_gate.ndim != 4 or refinement_gate.shape[1] != 1:
        raise ValueError("refinement gate must have shape [B, 1, H, W]")
    if refinement_gate.shape[-2:] != refined_logits.shape[-2:]:
        refinement_gate = F.interpolate(
            refinement_gate,
            size=refined_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    base_probabilities = base_logits.detach().softmax(dim=1)
    per_pixel_kl = F.kl_div(
        refined_logits.log_softmax(dim=1),
        base_probabilities,
        reduction="none",
    ).sum(dim=1)
    outside_weight = (1.0 - refinement_gate.detach().squeeze(1)).clamp(0.0, 1.0)
    outside_weight = outside_weight * (target != ignore_index).to(outside_weight.dtype)
    return (per_pixel_kl * outside_weight).sum() / outside_weight.sum().clamp_min(1.0)

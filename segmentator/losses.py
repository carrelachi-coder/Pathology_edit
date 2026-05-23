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

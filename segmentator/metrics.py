from __future__ import annotations

import torch.nn.functional as F
import torch


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    pred = pred.view(-1).long()
    target = target.view(-1).long()
    valid = (target >= 0) & (target < num_classes)
    indices = target[valid] * num_classes + pred[valid]
    mat = torch.bincount(indices, minlength=num_classes * num_classes)
    return mat.reshape(num_classes, num_classes)


def _safe_mean(values: torch.Tensor) -> float:
    valid = torch.isfinite(values)
    if not valid.any():
        return 0.0
    return float(values[valid].mean().item())


def _boundary(mask: torch.Tensor, width: int = 2) -> torch.Tensor:
    mask = mask.clamp_min(0)
    mask = mask.float().unsqueeze(1)
    max_pool = F.max_pool2d(mask, kernel_size=width * 2 + 1, stride=1, padding=width)
    min_pool = -F.max_pool2d(-mask, kernel_size=width * 2 + 1, stride=1, padding=width)
    return (max_pool != min_pool).squeeze(1)


def boundary_f1(pred: torch.Tensor, target: torch.Tensor, width: int = 2) -> float:
    pred_b = _boundary(pred, width=width)
    target_b = _boundary(target, width=width)
    tp = (pred_b & target_b).sum().float()
    precision = tp / pred_b.sum().float().clamp_min(1.0)
    recall = tp / target_b.sum().float().clamp_min(1.0)
    return float((2 * precision * recall / (precision + recall).clamp_min(1e-6)).item())


def segmentation_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    class_names: tuple[str, ...] | None = None,
    boundary_width: int = 2,
) -> dict[str, object]:
    mat = confusion_matrix(pred, target, num_classes)
    tp = mat.diag().float()
    fp = mat.sum(0).float() - tp
    fn = mat.sum(1).float() - tp
    iou = tp / (tp + fp + fn).clamp_min(1e-6)
    dice = (2 * tp) / (2 * tp + fp + fn).clamp_min(1e-6)
    recall = tp / (tp + fn).clamp_min(1e-6)
    names = class_names or tuple(f"class_{idx}" for idx in range(num_classes))
    per_class = {
        names[idx]: {
            "iou": float(iou[idx].item()),
            "dice": float(dice[idx].item()),
            "recall": float(recall[idx].item()),
            "support_pixels": int(mat[idx].sum().item()),
        }
        for idx in range(num_classes)
    }
    tissue_pred = (pred > 0).long()
    tissue_target = (target > 0).long()
    tissue_mat = confusion_matrix(tissue_pred, tissue_target, 2).float()
    tissue_tp = tissue_mat[1, 1]
    tissue_fp = tissue_mat[0, 1]
    tissue_fn = tissue_mat[1, 0]
    boundary_classes = [1, 2, 3]
    core_5_classes = [1, 2, 3, 4, 6]
    rare_classes = [4, 5, 6]
    return {
        "mIoU": _safe_mean(iou),
        "mDice": _safe_mean(dice),
        "foreground_recall": float(tp[1:].sum().div((tp[1:] + fn[1:]).sum().clamp_min(1e-6)).item()),
        "boundary_f1": boundary_f1(pred, target, width=boundary_width),
        "per_class": per_class,
        "groups": {
            "tissue_vs_background": {
                "iou": float((tissue_tp / (tissue_tp + tissue_fp + tissue_fn).clamp_min(1e-6)).item()),
                "dice": float((2 * tissue_tp / (2 * tissue_tp + tissue_fp + tissue_fn).clamp_min(1e-6)).item()),
            },
            "boundary_classes": {
                "mean_iou": _safe_mean(iou[boundary_classes]),
                "mean_dice": _safe_mean(dice[boundary_classes]),
            },
            "core_5_classes": {
                "classes": [names[idx] for idx in core_5_classes],
                "mean_iou": _safe_mean(iou[core_5_classes]),
                "mean_dice": _safe_mean(dice[core_5_classes]),
            },
            "rare_classes": {
                "mean_iou": _safe_mean(iou[rare_classes]),
                "mean_dice": _safe_mean(dice[rare_classes]),
            },
        },
    }


def mean_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> dict[str, float]:
    metrics = segmentation_metrics(pred, target, num_classes)
    return {
        "mIoU": float(metrics["mIoU"]),
        "mDice": float(metrics["mDice"]),
        "foreground_recall": float(metrics["foreground_recall"]),
    }

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch.nn.functional as F
import torch


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int | None = None) -> torch.Tensor:
    pred = pred.view(-1).long()
    target = target.view(-1).long()
    valid = (target >= 0) & (target < num_classes)
    if ignore_index is not None:
        valid = valid & (target != ignore_index)
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


def boundary_f1(pred: torch.Tensor, target: torch.Tensor, width: int = 2, ignore_index: int | None = None) -> float:
    if ignore_index is not None:
        valid = target != ignore_index
        pred = pred.clone()
        target = target.clone()
        pred[~valid] = 0
        target[~valid] = 0
    pred_b = _boundary(pred, width=width)
    target_b = _boundary(target, width=width)
    tp = (pred_b & target_b).sum().float()
    precision = tp / pred_b.sum().float().clamp_min(1.0)
    recall = tp / target_b.sum().float().clamp_min(1.0)
    return float((2 * precision * recall / (precision + recall).clamp_min(1e-6)).item())


def _thin_boundary(mask: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    result = np.zeros(mask.shape, dtype=bool)
    horizontal = mask[:, 1:] != mask[:, :-1]
    vertical = mask[1:, :] != mask[:-1, :]
    if valid is not None:
        horizontal &= valid[:, 1:] & valid[:, :-1]
        vertical &= valid[1:, :] & valid[:-1, :]
    result[:, 1:] |= horizontal
    result[:, :-1] |= horizontal
    result[1:, :] |= vertical
    result[:-1, :] |= vertical
    return result


def boundary_distance_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    tolerances: tuple[int, ...] = (2, 4, 8),
    ignore_index: int | None = 255,
    sample_limit: int = 0,
) -> dict[str, float | int]:
    try:
        from scipy import ndimage
    except ImportError:
        return {**{f"boundary_f1_{value}": float("nan") for value in tolerances}, "hd95": float("nan"), "samples": 0}

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    count = min(len(pred_np), sample_limit) if sample_limit > 0 else len(pred_np)
    scores: dict[int, list[float]] = defaultdict(list)
    hd95_values: list[float] = []
    for pred_mask, target_mask in zip(pred_np[:count], target_np[:count]):
        valid = target_mask != ignore_index if ignore_index is not None else np.ones(target_mask.shape, dtype=bool)
        pred_boundary = _thin_boundary(pred_mask, valid)
        target_boundary = _thin_boundary(target_mask, valid)
        if not pred_boundary.any() and not target_boundary.any():
            for tolerance in tolerances:
                scores[tolerance].append(1.0)
            hd95_values.append(0.0)
            continue
        pred_distance = ndimage.distance_transform_edt(~pred_boundary)
        target_distance = ndimage.distance_transform_edt(~target_boundary)
        for tolerance in tolerances:
            precision = float(np.mean(target_distance[pred_boundary] <= tolerance)) if pred_boundary.any() else 0.0
            recall = float(np.mean(pred_distance[target_boundary] <= tolerance)) if target_boundary.any() else 0.0
            scores[tolerance].append(2.0 * precision * recall / max(precision + recall, 1e-8))
        distances = []
        if pred_boundary.any():
            distances.append(target_distance[pred_boundary])
        if target_boundary.any():
            distances.append(pred_distance[target_boundary])
        hd95_values.append(float(np.percentile(np.concatenate(distances), 95)) if distances else 0.0)
    return {
        **{f"boundary_f1_{value}": float(np.mean(scores[value])) if scores[value] else float("nan") for value in tolerances},
        "hd95": float(np.mean(hd95_values)) if hd95_values else float("nan"),
        "samples": count,
    }


def fragmentation_metrics(
    pred: torch.Tensor,
    num_classes: int,
    thresholds: tuple[int, ...] = (16, 64),
    sample_limit: int = 0,
) -> dict[str, object]:
    try:
        from scipy import ndimage
    except ImportError:
        return {"samples": 0, "unavailable": "scipy is required"}

    pred_np = pred.detach().cpu().numpy()
    count = min(len(pred_np), sample_limit) if sample_limit > 0 else len(pred_np)
    component_sizes: list[int] = []
    per_class_sizes: dict[int, list[int]] = defaultdict(list)
    for pred_mask in pred_np[:count]:
        for class_id in range(1, num_classes):
            labels, components = ndimage.label(pred_mask == class_id)
            if components == 0:
                continue
            sizes = np.bincount(labels.reshape(-1))[1:]
            values = [int(value) for value in sizes]
            component_sizes.extend(values)
            per_class_sizes[class_id].extend(values)

    def summarize(sizes: list[int]) -> dict[str, float | int]:
        pixels = max(sum(sizes), 1)
        components = max(len(sizes), 1)
        result: dict[str, float | int] = {"components": len(sizes), "pixels": sum(sizes)}
        for threshold in thresholds:
            small = [size for size in sizes if size < threshold]
            result[f"components_lt_{threshold}"] = len(small)
            result[f"component_fraction_lt_{threshold}"] = len(small) / components
            result[f"pixel_fraction_lt_{threshold}"] = sum(small) / pixels
        return result

    return {
        "samples": count,
        "overall": summarize(component_sizes),
        "per_class": {str(class_id): summarize(sizes) for class_id, sizes in sorted(per_class_sizes.items())},
    }


def group_macro_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    group_ids: list[str],
    num_classes: int,
    ignore_index: int | None = 255,
) -> dict[str, float | int]:
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        by_group[group_id].append(index)
    values = []
    for indices in by_group.values():
        index = torch.tensor(indices, dtype=torch.long, device=pred.device)
        matrix = confusion_matrix(pred.index_select(0, index), target.index_select(0, index), num_classes, ignore_index=ignore_index)
        tp = matrix.diag().float()
        denominator = matrix.sum(0).float() + matrix.sum(1).float() - tp
        present = matrix.sum(1) > 0
        if present.any():
            values.append(float((tp[present] / denominator[present].clamp_min(1e-6)).mean().item()))
    return {
        "groups": len(values),
        "mean_mIoU": float(np.mean(values)) if values else float("nan"),
        "median_mIoU": float(np.median(values)) if values else float("nan"),
    }


def fine_segmentation_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    class_names: tuple[str, ...] | None = None,
    ignore_index: int = 255,
) -> dict[str, object]:
    """Report tumor-subtype metrics over pixels with explicit fine supervision."""
    valid = (target >= 0) & (target < num_classes) & (target != ignore_index)
    valid_pixels = int(valid.sum().item())
    names = class_names or tuple(f"fine_{idx}" for idx in range(num_classes))
    if valid_pixels == 0:
        return {"available": False, "valid_pixels": 0, "mIoU": float("nan"), "mDice": float("nan"), "accuracy": float("nan"), "per_class": {}}

    mat = confusion_matrix(pred, target, num_classes, ignore_index=ignore_index)
    tp = mat.diag().float()
    fp = mat.sum(0).float() - tp
    fn = mat.sum(1).float() - tp
    present = mat.sum(1) > 0
    iou = tp / (tp + fp + fn).clamp_min(1e-6)
    dice = (2 * tp) / (2 * tp + fp + fn).clamp_min(1e-6)
    per_class = {
        names[idx]: {
            "iou": float(iou[idx].item()),
            "dice": float(dice[idx].item()),
            "support_pixels": int(mat[idx].sum().item()),
            "predicted_pixels": int(mat[:, idx].sum().item()),
        }
        for idx in range(num_classes)
        if bool(present[idx])
    }
    return {
        "available": True,
        "valid_pixels": valid_pixels,
        "mIoU": _safe_mean(iou[present]),
        "mDice": _safe_mean(dice[present]),
        "accuracy": float(tp.sum().div(mat.sum().clamp_min(1)).item()),
        "per_class": per_class,
    }


def segmentation_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    class_names: tuple[str, ...] | None = None,
    boundary_width: int = 2,
    ignore_index: int | None = 255,
    metric_sample_limit: int = 0,
) -> dict[str, object]:
    mat = confusion_matrix(pred, target, num_classes, ignore_index=ignore_index)
    tp = mat.diag().float()
    fp = mat.sum(0).float() - tp
    fn = mat.sum(1).float() - tp
    iou_denominator = tp + fp + fn
    dice_denominator = 2 * tp + fp + fn
    recall_denominator = tp + fn
    nan = torch.full_like(tp, float("nan"))
    iou = torch.where(iou_denominator > 0, tp / iou_denominator.clamp_min(1e-6), nan)
    dice = torch.where(dice_denominator > 0, (2 * tp) / dice_denominator.clamp_min(1e-6), nan)
    recall = torch.where(recall_denominator > 0, tp / recall_denominator.clamp_min(1e-6), nan)
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
    metric_pred = pred
    metric_target = target
    if ignore_index is not None:
        valid_metric = target != ignore_index
        metric_pred = pred[valid_metric]
        metric_target = target[valid_metric]
    tissue_pred = (metric_pred > 0).long()
    tissue_target = (metric_target > 0).long()
    tissue_mat = confusion_matrix(tissue_pred, tissue_target, 2, ignore_index=None).float()
    tissue_tp = tissue_mat[1, 1]
    tissue_fp = tissue_mat[0, 1]
    tissue_fn = tissue_mat[1, 0]
    boundary_classes = [class_id for class_id in (1, 2, 3) if class_id < num_classes]
    core_5_classes = [class_id for class_id in (1, 2, 3, 4, 6) if class_id < num_classes]
    rare_classes = [class_id for class_id in (4, 5, 6) if class_id < num_classes]
    distance_metrics = boundary_distance_metrics(
        pred,
        target,
        ignore_index=ignore_index,
        sample_limit=metric_sample_limit,
    )
    return {
        "mIoU": _safe_mean(iou),
        "mDice": _safe_mean(dice),
        "foreground_recall": float(tp[1:].sum().div((tp[1:] + fn[1:]).sum().clamp_min(1e-6)).item()),
        "boundary_f1": boundary_f1(pred, target, width=boundary_width, ignore_index=ignore_index),
        **distance_metrics,
        "fragmentation": fragmentation_metrics(pred, num_classes, sample_limit=metric_sample_limit),
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

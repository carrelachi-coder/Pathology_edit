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


def _metric_class_index(
    num_classes: int,
    evaluated_class_ids: tuple[int, ...] | None,
    device: torch.device,
) -> torch.Tensor:
    class_ids = tuple(range(num_classes)) if evaluated_class_ids is None else tuple(evaluated_class_ids)
    if not class_ids:
        raise ValueError("evaluated_class_ids must not be empty")
    if len(set(class_ids)) != len(class_ids):
        raise ValueError("evaluated_class_ids must not contain duplicates")
    if any(class_id < 0 or class_id >= num_classes for class_id in class_ids):
        raise ValueError(f"evaluated_class_ids must be within [0, {num_classes})")
    return torch.tensor(class_ids, dtype=torch.long, device=device)


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
    evaluated_class_ids: tuple[int, ...] | None = None,
) -> dict[str, float | int]:
    matrices, ordered_groups = _group_confusion_matrices(
        pred,
        target,
        group_ids,
        num_classes,
        ignore_index=ignore_index,
    )
    values = []
    metric_index = _metric_class_index(num_classes, evaluated_class_ids, pred.device)
    for matrix in matrices:
        tp = matrix.diag().float()
        denominator = matrix.sum(0).float() + matrix.sum(1).float() - tp
        present = matrix.sum(1) > 0
        evaluated = present.index_select(0, metric_index)
        if evaluated.any():
            selected_tp = tp.index_select(0, metric_index)[evaluated]
            selected_denominator = denominator.index_select(0, metric_index)[evaluated]
            values.append(float((selected_tp / selected_denominator.clamp_min(1e-6)).mean().item()))
    return {
        "groups": len(values),
        "mean_mIoU": float(np.mean(values)) if values else float("nan"),
        "median_mIoU": float(np.median(values)) if values else float("nan"),
        "total_groups": len(ordered_groups),
    }


def dataset_group_macro_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    group_ids: list[str],
    dataset_ids: list[str],
    num_classes: int,
    evaluated_class_ids_by_dataset: dict[str, tuple[int, ...]],
    ignore_index: int | None = 255,
) -> dict[str, float | int]:
    if len(group_ids) != len(dataset_ids) or len(group_ids) != len(pred):
        raise ValueError("predictions, group_ids, and dataset_ids must have matching lengths")
    group_keys = list(zip(dataset_ids, group_ids))
    matrices, ordered_groups = _group_confusion_matrices(
        pred,
        target,
        group_keys,
        num_classes,
        ignore_index=ignore_index,
    )
    values = []
    for (dataset_id, _), matrix in zip(ordered_groups, matrices):
        if dataset_id not in evaluated_class_ids_by_dataset:
            raise ValueError(f"missing evaluated class IDs for dataset {dataset_id!r}")
        tp = matrix.diag().float()
        denominator = matrix.sum(0).float() + matrix.sum(1).float() - tp
        present = matrix.sum(1) > 0
        metric_index = _metric_class_index(
            num_classes,
            evaluated_class_ids_by_dataset[dataset_id],
            pred.device,
        )
        evaluated = present.index_select(0, metric_index)
        if evaluated.any():
            selected_tp = tp.index_select(0, metric_index)[evaluated]
            selected_denominator = denominator.index_select(0, metric_index)[evaluated]
            values.append(float((selected_tp / selected_denominator.clamp_min(1e-6)).mean().item()))
    return {
        "groups": len(values),
        "mean_mIoU": float(np.mean(values)) if values else float("nan"),
        "median_mIoU": float(np.median(values)) if values else float("nan"),
        "total_groups": len(ordered_groups),
    }


def _group_confusion_matrices(
    pred: torch.Tensor,
    target: torch.Tensor,
    group_keys: list[object],
    num_classes: int,
    ignore_index: int | None = 255,
    sample_chunk_size: int = 32,
) -> tuple[torch.Tensor, list[object]]:
    if len(pred) != len(target) or len(pred) != len(group_keys):
        raise ValueError("predictions, targets, and group keys must have matching lengths")
    if sample_chunk_size <= 0:
        raise ValueError("sample_chunk_size must be positive")

    group_to_index: dict[object, int] = {}
    ordered_groups: list[object] = []
    group_indices: list[int] = []
    for group_key in group_keys:
        if group_key not in group_to_index:
            group_to_index[group_key] = len(ordered_groups)
            ordered_groups.append(group_key)
        group_indices.append(group_to_index[group_key])

    group_count = len(ordered_groups)
    matrices = torch.zeros(
        (group_count, num_classes, num_classes),
        dtype=torch.long,
        device=pred.device,
    )
    matrix_size = num_classes * num_classes
    for start in range(0, len(pred), sample_chunk_size):
        end = min(start + sample_chunk_size, len(pred))
        pred_chunk = pred[start:end].reshape(end - start, -1).long()
        target_chunk = target[start:end].reshape(end - start, -1).long()
        valid = (target_chunk >= 0) & (target_chunk < num_classes)
        if ignore_index is not None:
            valid &= target_chunk != ignore_index
        chunk_groups = torch.tensor(
            group_indices[start:end],
            dtype=torch.long,
            device=pred.device,
        ).view(-1, 1).expand_as(target_chunk)
        encoded = (
            chunk_groups[valid] * matrix_size
            + target_chunk[valid] * num_classes
            + pred_chunk[valid]
        )
        matrices += torch.bincount(
            encoded,
            minlength=group_count * matrix_size,
        ).reshape(group_count, num_classes, num_classes)
    return matrices, ordered_groups


def fine_segmentation_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    class_names: tuple[str, ...] | None = None,
    ignore_index: int = 255,
    evaluated_class_ids: tuple[int, ...] | None = None,
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
    metric_index = _metric_class_index(num_classes, evaluated_class_ids, pred.device)
    metric_class_ids = [int(class_id) for class_id in metric_index.tolist()]
    metric_class_id_set = set(metric_class_ids)
    evaluated_present = present.index_select(0, metric_index)
    iou = tp / (tp + fp + fn).clamp_min(1e-6)
    dice = (2 * tp) / (2 * tp + fp + fn).clamp_min(1e-6)
    per_class = {
        names[idx]: {
            "iou": float(iou[idx].item()),
            "dice": float(dice[idx].item()),
            "support_pixels": int(mat[idx].sum().item()),
            "predicted_pixels": int(mat[:, idx].sum().item()),
            "evaluated": idx in metric_class_id_set,
        }
        for idx in range(num_classes)
        if bool(present[idx])
    }
    selected_iou = iou.index_select(0, metric_index)[evaluated_present]
    selected_dice = dice.index_select(0, metric_index)[evaluated_present]
    return {
        "available": True,
        "valid_pixels": valid_pixels,
        "mIoU": _safe_mean(selected_iou) if evaluated_present.any() else float("nan"),
        "mDice": _safe_mean(selected_dice) if evaluated_present.any() else float("nan"),
        "accuracy": float(tp.sum().div(mat.sum().clamp_min(1)).item()),
        "evaluated_class_ids": metric_class_ids,
        "evaluated_classes": [names[idx] for idx in metric_class_ids],
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
    evaluated_class_ids: tuple[int, ...] | None = None,
    include_spatial_metrics: bool = True,
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
    metric_index = _metric_class_index(num_classes, evaluated_class_ids, pred.device)
    metric_class_ids = [int(class_id) for class_id in metric_index.tolist()]
    metric_class_id_set = set(metric_class_ids)
    per_class = {
        names[idx]: {
            "iou": float(iou[idx].item()),
            "dice": float(dice[idx].item()),
            "recall": float(recall[idx].item()),
            "support_pixels": int(mat[idx].sum().item()),
            "evaluated": idx in metric_class_id_set,
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
    boundary_classes = [
        class_id for class_id in (1, 2, 3)
        if class_id < num_classes and class_id in metric_class_id_set
    ]
    core_5_classes = [
        class_id for class_id in (1, 2, 3, 4, 6)
        if class_id < num_classes and class_id in metric_class_id_set
    ]
    rare_classes = [
        class_id for class_id in (4, 5, 6)
        if class_id < num_classes and class_id in metric_class_id_set
    ]
    if include_spatial_metrics:
        boundary_score = boundary_f1(pred, target, width=boundary_width, ignore_index=ignore_index)
        distance_metrics = boundary_distance_metrics(
            pred,
            target,
            ignore_index=ignore_index,
            sample_limit=metric_sample_limit,
        )
        fragmentation = fragmentation_metrics(pred, num_classes, sample_limit=metric_sample_limit)
    else:
        boundary_score = float("nan")
        distance_metrics = {
            "boundary_f1_2": float("nan"),
            "boundary_f1_4": float("nan"),
            "boundary_f1_8": float("nan"),
            "hd95": float("nan"),
            "samples": 0,
        }
        fragmentation = {"samples": 0, "skipped": "spatial metrics disabled"}
    return {
        "mIoU": _safe_mean(iou.index_select(0, metric_index)),
        "mDice": _safe_mean(dice.index_select(0, metric_index)),
        "evaluated_class_ids": metric_class_ids,
        "evaluated_classes": [names[idx] for idx in metric_class_ids],
        "foreground_recall": float(tp[1:].sum().div((tp[1:] + fn[1:]).sum().clamp_min(1e-6)).item()),
        "boundary_f1": boundary_score,
        **distance_metrics,
        "fragmentation": fragmentation,
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


def mean_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    evaluated_class_ids: tuple[int, ...] | None = None,
) -> dict[str, float]:
    metrics = segmentation_metrics(
        pred,
        target,
        num_classes,
        evaluated_class_ids=evaluated_class_ids,
    )
    return {
        "mIoU": float(metrics["mIoU"]),
        "mDice": float(metrics["mDice"]),
        "foreground_recall": float(metrics["foreground_recall"]),
    }

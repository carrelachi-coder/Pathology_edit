"""Metrics for tissue- and nuclei-conditioned generation fidelity."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Detections:
    xy: np.ndarray
    class_ids: np.ndarray
    mpp: float
    image_size: tuple[int, int]

    def __post_init__(self) -> None:
        xy = np.asarray(self.xy, dtype=np.float64)
        class_ids = np.asarray(self.class_ids, dtype=np.int64)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"xy must have shape [N, 2], got {xy.shape}")
        if class_ids.shape != (len(xy),):
            raise ValueError("class_ids must contain one value per detection")
        if self.mpp <= 0 or min(self.image_size) <= 0:
            raise ValueError("mpp and image dimensions must be positive")
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "class_ids", class_ids)


def load_label_mask(
    path: Path, *, expected_size: tuple[int, int] | None = None
) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image)
    if values.ndim == 3:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"expected a rank-2 label mask, got {values.shape}: {path}")
    if expected_size is not None:
        expected_shape = (int(expected_size[1]), int(expected_size[0]))
        if values.shape != expected_shape:
            raise ValueError(
                f"expected label mask size {expected_size}, got "
                f"{(values.shape[1], values.shape[0])}: {path}"
            )
    return values.astype(np.int64, copy=False)


def rescale_detections(
    detections: Detections,
    *,
    image_size: tuple[int, int],
    mpp: float,
) -> Detections:
    """Map detector coordinates onto an equivalent evaluation pixel grid."""
    image_size = (int(image_size[0]), int(image_size[1]))
    source_fov = np.asarray(detections.image_size, dtype=np.float64) * detections.mpp
    target_fov = np.asarray(image_size, dtype=np.float64) * float(mpp)
    if not np.allclose(source_fov, target_fov, rtol=0.0, atol=1e-6):
        raise ValueError(
            "detection rescaling must preserve the physical field: "
            f"source={source_fov.tolist()} um, target={target_fov.tolist()} um"
        )
    scale = np.asarray(image_size, dtype=np.float64) / np.asarray(
        detections.image_size, dtype=np.float64
    )
    return Detections(
        xy=detections.xy * scale,
        class_ids=detections.class_ids.copy(),
        mpp=float(mpp),
        image_size=image_size,
    )


def jensen_shannon_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("JSD inputs must be same-shape vectors")
    if np.any(left < 0) or np.any(right < 0):
        raise ValueError("JSD inputs cannot be negative")
    if left.sum() <= 0 or right.sum() <= 0:
        return 0.0 if left.sum() == right.sum() else 1.0
    left = left / left.sum()
    right = right / right.sum()
    midpoint = (left + right) / 2.0

    def kl(values: np.ndarray) -> float:
        keep = values > 0
        return float(np.sum(values[keep] * np.log2(values[keep] / midpoint[keep])))

    return 0.5 * (kl(left) + kl(right))


def tissue_fidelity_metrics(
    target: np.ndarray,
    predicted: np.ndarray,
    *,
    class_ids: Sequence[int],
    ignore_index: int = 255,
    background_id: int = 0,
    presence_min_fraction: float = 0.005,
) -> dict:
    target = np.asarray(target, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    if target.shape != predicted.shape or target.ndim != 2:
        raise ValueError(
            f"target and predicted masks must share a rank-2 shape: "
            f"{target.shape}, {predicted.shape}"
        )
    valid = target != ignore_index
    if not np.any(valid):
        raise ValueError("target mask contains no evaluable pixels")
    allowed = set(int(value) for value in class_ids)
    unknown_target = set(np.unique(target[valid]).tolist()) - allowed
    unknown_predicted = set(np.unique(predicted[valid]).tolist()) - allowed
    if unknown_target or unknown_predicted:
        raise ValueError(
            f"unknown tissue IDs: target={sorted(unknown_target)}, "
            f"predicted={sorted(unknown_predicted)}"
        )

    total = int(valid.sum())
    per_class = {}
    macro_dice = []
    macro_iou = []
    target_present = []
    predicted_present = []
    area_target = []
    area_predicted = []
    for class_id in class_ids:
        class_id = int(class_id)
        target_class = (target == class_id) & valid
        predicted_class = (predicted == class_id) & valid
        target_count = int(target_class.sum())
        predicted_count = int(predicted_class.sum())
        intersection = int(np.count_nonzero(target_class & predicted_class))
        union = target_count + predicted_count - intersection
        dice = (
            2.0 * intersection / (target_count + predicted_count)
            if target_count + predicted_count
            else None
        )
        iou = intersection / union if union else None
        target_fraction = target_count / total
        predicted_fraction = predicted_count / total
        per_class[str(class_id)] = {
            "target_pixels": target_count,
            "predicted_pixels": predicted_count,
            "target_fraction": target_fraction,
            "predicted_fraction": predicted_fraction,
            "area_fraction_error": predicted_fraction - target_fraction,
            "area_fraction_abs_error": abs(predicted_fraction - target_fraction),
            "dice": dice,
            "iou": iou,
        }
        area_target.append(target_count)
        area_predicted.append(predicted_count)
        if class_id != background_id and union:
            macro_dice.append(float(dice))
            macro_iou.append(float(iou))
        if class_id != background_id:
            target_present.append(target_fraction >= presence_min_fraction)
            predicted_present.append(predicted_fraction >= presence_min_fraction)

    target_present_count = int(sum(target_present))
    presence_hits = int(
        sum(target_flag and pred_flag for target_flag, pred_flag in zip(target_present, predicted_present))
    )
    foreground_errors = [
        values["area_fraction_abs_error"]
        for class_id, values in per_class.items()
        if int(class_id) != background_id
    ]
    return {
        "valid_pixels": total,
        "macro_policy": "non-background classes present in target or prediction",
        "macro_dice": float(np.mean(macro_dice)) if macro_dice else None,
        "macro_miou": float(np.mean(macro_iou)) if macro_iou else None,
        "mean_foreground_area_fraction_abs_error": float(np.mean(foreground_errors)),
        "tissue_area_distribution_jsd": jensen_shannon_divergence(
            np.asarray(area_target), np.asarray(area_predicted)
        ),
        "class_presence_min_fraction": presence_min_fraction,
        "target_present_class_count": target_present_count,
        "class_presence_recall": (
            presence_hits / target_present_count if target_present_count else 1.0
        ),
        "per_class": per_class,
    }


def detections_from_conic(path: Path, *, mpp: float) -> Detections:
    values = np.load(path, allow_pickle=False)
    if values.ndim != 3 or values.shape[2] < 2:
        raise ValueError(f"expected CoNIC [H, W, 2] array, got {values.shape}: {path}")
    instance_map = values[..., 0].astype(np.int64, copy=False)
    type_map = values[..., 1].astype(np.int64, copy=False)
    points = []
    class_ids = []
    for instance_id in np.unique(instance_map):
        if instance_id <= 0:
            continue
        yy, xx = np.nonzero(instance_map == instance_id)
        if not len(xx):
            continue
        labels, counts = np.unique(type_map[yy, xx], return_counts=True)
        foreground = labels > 0
        class_id = int(labels[foreground][np.argmax(counts[foreground])]) if np.any(foreground) else 0
        points.append([float(xx.mean()), float(yy.mean())])
        class_ids.append(class_id)
    return Detections(
        xy=np.asarray(points, dtype=np.float64).reshape(-1, 2),
        class_ids=np.asarray(class_ids, dtype=np.int64),
        mpp=mpp,
        image_size=(int(instance_map.shape[1]), int(instance_map.shape[0])),
    )


def _is_xy_pair(value: object) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return False
    try:
        float(value[0])
        float(value[1])
    except (TypeError, ValueError):
        return False
    return True


def _local_cellvit_points(cell: dict, metadata: dict, width: int, height: int) -> list[list[float]]:
    contour = cell.get("contour")
    if not isinstance(contour, list):
        contour = []
    offset = cell.get("offset_global")
    if not _is_xy_pair(offset):
        return [[float(x), float(y)] for x, y in contour if _is_xy_pair([x, y])]
    patch_size = metadata.get("patch_size")
    if patch_size is not None:
        x_offset = max(float(patch_size) - width, 0.0) + float(offset[1])
        y_offset = max(float(patch_size) - height, 0.0) + float(offset[0])
    else:
        x_offset = float(offset[1])
        y_offset = float(offset[0])
    return [
        [float(point[0]) - x_offset, float(point[1]) - y_offset]
        for point in contour
        if _is_xy_pair(point)
    ]


def detections_from_cellvit_json(
    path: Path, *, mpp: float, image_size: tuple[int, int]
) -> Detections:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise ValueError(f"CellViT JSON has no cells list: {path}")
    metadata = payload.get("wsi_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    width, height = image_size
    points = []
    class_ids = []
    for cell in cells:
        class_id = int(cell.get("type", 0))
        if class_id <= 0:
            continue
        local_contour = _local_cellvit_points(cell, metadata, width, height)
        if local_contour:
            point = np.mean(np.asarray(local_contour, dtype=np.float64), axis=0)
        else:
            centroid = cell.get("centroid")
            if not _is_xy_pair(centroid):
                continue
            point = np.asarray(centroid[:2], dtype=np.float64)
        if -1 <= point[0] <= width and -1 <= point[1] <= height:
            points.append(point.tolist())
            class_ids.append(class_id)
    return Detections(
        xy=np.asarray(points, dtype=np.float64).reshape(-1, 2),
        class_ids=np.asarray(class_ids, dtype=np.int64),
        mpp=mpp,
        image_size=image_size,
    )


def _class_counts(detections: Detections, class_ids: Iterable[int]) -> np.ndarray:
    return np.asarray(
        [np.count_nonzero(detections.class_ids == int(class_id)) for class_id in class_ids],
        dtype=np.int64,
    )


def cell_distribution_metrics(
    target: Detections,
    predicted: Detections,
    *,
    class_ids: Sequence[int],
) -> dict:
    if target.image_size != predicted.image_size or not np.isclose(target.mpp, predicted.mpp):
        raise ValueError("target and predicted detections must use the same physical frame")
    target_counts = _class_counts(target, class_ids)
    predicted_counts = _class_counts(predicted, class_ids)
    area_mm2 = (
        target.image_size[0] * target.mpp * target.image_size[1] * target.mpp / 1_000_000.0
    )
    per_class = {}
    for index, class_id in enumerate(class_ids):
        error = int(predicted_counts[index] - target_counts[index])
        per_class[str(int(class_id))] = {
            "target_count": int(target_counts[index]),
            "predicted_count": int(predicted_counts[index]),
            "count_error": error,
            "count_abs_error": abs(error),
            "density_error_per_mm2": error / area_mm2,
            "density_abs_error_per_mm2": abs(error) / area_mm2,
        }
    total_target = int(target_counts.sum())
    total_predicted = int(predicted_counts.sum())
    return {
        "target_total_count": total_target,
        "predicted_total_count": total_predicted,
        "total_count_error": total_predicted - total_target,
        "total_count_abs_error": abs(total_predicted - total_target),
        "total_density_error_per_mm2": (total_predicted - total_target) / area_mm2,
        "total_density_abs_error_per_mm2": abs(total_predicted - total_target) / area_mm2,
        "total_density_relative_abs_error": (
            abs(total_predicted - total_target) / total_target
            if total_target
            else (0.0 if total_predicted == 0 else None)
        ),
        "mean_class_count_abs_error": float(np.mean(np.abs(predicted_counts - target_counts))),
        "cell_type_distribution_jsd": jensen_shannon_divergence(target_counts, predicted_counts),
        "field_area_mm2": area_mm2,
        "per_class": per_class,
    }


def spatial_matching_metrics(
    target: Detections,
    predicted: Detections,
    *,
    max_distance_um: float,
    class_aware: bool,
) -> dict:
    if target.image_size != predicted.image_size or not np.isclose(target.mpp, predicted.mpp):
        raise ValueError("target and predicted detections must use the same physical frame")
    if max_distance_um <= 0:
        raise ValueError("max_distance_um must be positive")
    from scipy.optimize import linear_sum_assignment

    target_um = target.xy * target.mpp
    predicted_um = predicted.xy * predicted.mpp
    if not len(target_um) or not len(predicted_um):
        matches = np.empty(0, dtype=np.float64)
    else:
        distances = np.linalg.norm(target_um[:, None, :] - predicted_um[None, :, :], axis=2)
        if class_aware:
            mismatched = target.class_ids[:, None] != predicted.class_ids[None, :]
            distances = distances.copy()
            distances[mismatched] = max_distance_um + 1.0
        target_index, predicted_index = linear_sum_assignment(distances)
        matches = distances[target_index, predicted_index]
        matches = matches[matches <= max_distance_um]
    true_positive = int(len(matches))
    false_positive = int(len(predicted.xy) - true_positive)
    false_negative = int(len(target.xy) - true_positive)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "class_aware": class_aware,
        "max_distance_um": max_distance_um,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_distance_mean_um": float(matches.mean()) if len(matches) else None,
        "matched_distance_p95_um": float(np.quantile(matches, 0.95)) if len(matches) else None,
    }

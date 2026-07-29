"""Reusable tissue-evaluator metrics for online semantic self-auditing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ConfidencePolicy:
    status: str = "calibration_required"
    top1_probability_min: float | None = None
    top1_top2_margin_min: float | None = None
    normalized_entropy_max: float | None = None

    @property
    def calibrated(self) -> bool:
        return (
            self.status == "frozen"
            and self.top1_probability_min is not None
            and self.top1_top2_margin_min is not None
            and self.normalized_entropy_max is not None
        )


@dataclass(frozen=True)
class EvaluatorCleanPolicy:
    status: str = "calibration_required"
    source_region_accuracy_min: float | None = None
    source_class_recall_min: float | None = None
    source_mean_normalized_entropy_max: float | None = None
    source_boundary_f1_4_min: float | None = None
    stratification: str = "organ_and_source_class_with_pooled_fallback"

    @property
    def calibrated(self) -> bool:
        return (
            self.status == "frozen"
            and self.source_region_accuracy_min is not None
            and self.source_class_recall_min is not None
            and self.source_mean_normalized_entropy_max is not None
            and self.source_boundary_f1_4_min is not None
        )


def build_edit_regions(
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    ignore_index: int = 255,
    boundary_radius: int = 4,
    semantic_change_region: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    source = np.asarray(source_mask)
    target = np.asarray(target_mask)
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("source and target masks must share a rank-2 shape")
    if boundary_radius < 1:
        raise ValueError("boundary_radius must be positive")
    valid = (source != ignore_index) & (target != ignore_index)
    changed = (
        valid & (source != target)
        if semantic_change_region is None
        else valid & np.asarray(semantic_change_region, dtype=bool)
    )
    if changed.shape != source.shape:
        raise ValueError("semantic_change_region must match source and target")
    if not np.any(changed):
        raise ValueError("G2 semantic change region is empty")
    unchanged = valid & ~changed
    boundary_inside = changed & _binary_dilation(~changed, boundary_radius)
    boundary_outside = unchanged & _binary_dilation(changed, boundary_radius)
    boundary = boundary_inside | boundary_outside
    unchanged_far = unchanged & ~boundary_outside
    return {
        "valid": valid,
        "R": changed,
        "U": unchanged,
        "B_in": boundary_inside,
        "B_out": boundary_outside,
        "B": boundary,
        "U_far": unchanged_far,
    }


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 3:
        raise ValueError("probabilities must have CHW layout")
    class_count = probabilities.shape[0]
    if class_count < 2:
        return np.zeros(probabilities.shape[1:], dtype=np.float64)
    if np.any(probabilities < -1e-6):
        raise ValueError("probabilities cannot be negative")
    sums = probabilities.sum(axis=0)
    if not np.allclose(sums, 1.0, atol=2e-3, rtol=0.0):
        raise ValueError("probability channels must sum to one per pixel")
    terms = np.zeros_like(probabilities)
    positive = probabilities > 0
    terms[positive] = probabilities[positive] * np.log(probabilities[positive])
    return -np.sum(terms, axis=0) / np.log(class_count)


def confidence_maps(
    probabilities: np.ndarray,
    *,
    entropy: np.ndarray | None = None,
    policy: ConfidencePolicy | None = None,
) -> dict[str, np.ndarray | bool | dict]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 3:
        raise ValueError("probabilities must have CHW layout")
    entropy_values = (
        normalized_entropy(probabilities)
        if entropy is None
        else np.asarray(entropy, dtype=np.float64)
    )
    if entropy_values.shape != probabilities.shape[1:]:
        raise ValueError("entropy and probability spatial shapes differ")
    sorted_probabilities = np.sort(probabilities, axis=0)
    top1 = sorted_probabilities[-1]
    top2 = sorted_probabilities[-2]
    margin = top1 - top2
    policy = policy or ConfidencePolicy()
    result: dict[str, np.ndarray | bool | dict] = {
        "top1_probability": top1,
        "top1_top2_margin": margin,
        "normalized_entropy": entropy_values,
        "policy": asdict(policy),
        "policy_calibrated": policy.calibrated,
    }
    if policy.calibrated:
        result["high_confidence"] = (
            (top1 >= float(policy.top1_probability_min))
            & (margin >= float(policy.top1_top2_margin_min))
            & (entropy_values <= float(policy.normalized_entropy_max))
        )
    return result


def region_macro_iou(
    target: np.ndarray,
    predicted: np.ndarray,
    region: np.ndarray,
    *,
    class_ids: Sequence[int],
    include_background: bool,
    background_id: int = 0,
) -> dict:
    target = np.asarray(target)
    predicted = np.asarray(predicted)
    region = np.asarray(region, dtype=bool)
    if not (target.shape == predicted.shape == region.shape):
        raise ValueError("target, predicted, and region shapes must match")
    allowed = {int(value) for value in class_ids}
    labels = sorted(
        int(value)
        for value in (
            (set(np.unique(target[region])) | set(np.unique(predicted[region])))
            & allowed
        )
    )
    if not include_background:
        labels = [value for value in labels if value != background_id]
    per_class = {}
    for class_id in labels:
        target_class = (target == class_id) & region
        predicted_class = (predicted == class_id) & region
        intersection = int(np.count_nonzero(target_class & predicted_class))
        union = int(np.count_nonzero(target_class | predicted_class))
        per_class[str(class_id)] = {
            "intersection_pixels": intersection,
            "union_pixels": union,
            "iou": intersection / union if union else None,
        }
    scores = [
        values["iou"]
        for values in per_class.values()
        if values["iou"] is not None
    ]
    return {
        "macro_policy": "target_or_prediction_present",
        "evaluated_class_ids": labels,
        "per_class": per_class,
        "macro_miou": float(np.mean(scores)) if scores else None,
    }


def source_relative_tissue_metrics(
    *,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    source_prediction: np.ndarray,
    generated_prediction: np.ndarray,
    source_probabilities: np.ndarray,
    generated_probabilities: np.ndarray,
    class_ids: Sequence[int] = tuple(range(8)),
    ignore_index: int = 255,
    boundary_radius: int = 4,
    source_entropy: np.ndarray | None = None,
    generated_entropy: np.ndarray | None = None,
    confidence_policy: ConfidencePolicy | None = None,
    semantic_change_region: np.ndarray | None = None,
) -> dict:
    source = np.asarray(source_mask)
    target = np.asarray(target_mask)
    source_pred = np.asarray(source_prediction)
    generated_pred = np.asarray(generated_prediction)
    if not (
        source.shape
        == target.shape
        == source_pred.shape
        == generated_pred.shape
    ):
        raise ValueError("all tissue masks must share a shape")
    source_probs = _validate_probabilities(source_probabilities, source.shape)
    generated_probs = _validate_probabilities(
        generated_probabilities, source.shape
    )
    regions = build_edit_regions(
        source,
        target,
        ignore_index=ignore_index,
        boundary_radius=boundary_radius,
        semantic_change_region=semantic_change_region,
    )
    changed = regions["R"]
    include_background = bool(
        np.any(changed & ((source == 0) | (target == 0)))
    )
    generated_iou = region_macro_iou(
        target,
        generated_pred,
        changed,
        class_ids=class_ids,
        include_background=include_background,
    )
    source_iou = region_macro_iou(
        target,
        source_pred,
        changed,
        class_ids=class_ids,
        include_background=include_background,
    )
    target_accuracy = float(np.mean(generated_pred[changed] == target[changed]))
    no_edit_accuracy = float(np.mean(source_pred[changed] == target[changed]))
    target_class = target[changed].astype(np.int64)
    source_class = source[changed].astype(np.int64)
    index = np.arange(target_class.size)
    generated_margin_values = (
        generated_probs[:, changed][target_class, index]
        - generated_probs[:, changed][source_class, index]
    )
    source_margin_values = (
        source_probs[:, changed][target_class, index]
        - source_probs[:, changed][source_class, index]
    )
    source_confidence = confidence_maps(
        source_probs,
        entropy=source_entropy,
        policy=confidence_policy,
    )
    generated_confidence = confidence_maps(
        generated_probs,
        entropy=generated_entropy,
        policy=confidence_policy,
    )

    result = {
        "region_pixels": {
            name: int(np.count_nonzero(values))
            for name, values in regions.items()
        },
        "changed_region": {
            "accuracy": target_accuracy,
            "macro_miou": generated_iou["macro_miou"],
            "macro_iou_detail": generated_iou,
            "source_residual_rate": float(
                np.mean(generated_pred[changed] == source[changed])
            ),
            "no_edit_accuracy": no_edit_accuracy,
            "no_edit_macro_miou": source_iou["macro_miou"],
            "target_gain_accuracy": target_accuracy - no_edit_accuracy,
            "target_gain_miou": _difference(
                generated_iou["macro_miou"], source_iou["macro_miou"]
            ),
            "soft_target_source_margin": float(
                np.mean(generated_margin_values)
            ),
            "soft_no_edit_target_source_margin": float(
                np.mean(source_margin_values)
            ),
            "soft_margin_gain": float(
                np.mean(generated_margin_values)
                - np.mean(source_margin_values)
            ),
        },
        "preservation": {
            "prediction_relative_drift_U": _disagreement_rate(
                generated_pred, source_pred, regions["U"]
            ),
            "prediction_relative_drift_U_far": _disagreement_rate(
                generated_pred, source_pred, regions["U_far"]
            ),
            "mask_relative_drift_U": _disagreement_rate(
                generated_pred, source, regions["U"]
            ),
            "outer_ring_spillover": _disagreement_rate(
                generated_pred, source_pred, regions["B_out"]
            ),
            "inner_ring_target_error": _disagreement_rate(
                generated_pred, target, regions["B_in"]
            ),
        },
        "uncertainty": {
            "source": _uncertainty_summary(source_confidence, regions),
            "generated": _uncertainty_summary(generated_confidence, regions),
        },
    }
    return result


def source_evaluator_quality(
    *,
    source_mask: np.ndarray,
    source_prediction: np.ndarray,
    source_probabilities: np.ndarray,
    class_ids: Sequence[int] = tuple(range(8)),
    ignore_index: int = 255,
    policy: EvaluatorCleanPolicy | None = None,
) -> dict:
    source = np.asarray(source_mask)
    prediction = np.asarray(source_prediction)
    if source.shape != prediction.shape or source.ndim != 2:
        raise ValueError("source mask and prediction must share a rank-2 shape")
    probabilities = _validate_probabilities(source_probabilities, source.shape)
    valid = source != ignore_index
    allowed = {int(value) for value in class_ids}
    valid &= np.isin(source, list(allowed))
    if not np.any(valid):
        raise ValueError("source mask contains no evaluable pixels")
    recall = {}
    for class_id in sorted(allowed & set(np.unique(source[valid]))):
        pixels = valid & (source == class_id)
        recall[str(class_id)] = float(np.mean(prediction[pixels] == class_id))
    entropy = normalized_entropy(probabilities)
    metrics = {
        "source_region_accuracy": float(np.mean(prediction[valid] == source[valid])),
        "source_class_recall": recall,
        "source_class_recall_min": min(recall.values()) if recall else None,
        "source_mean_normalized_entropy": float(np.mean(entropy[valid])),
        "source_boundary_f1_4": class_aware_boundary_f1(
            source,
            prediction,
            valid=valid,
            class_ids=class_ids,
            tolerance=4,
        ),
    }
    policy = policy or EvaluatorCleanPolicy()
    reasons = []
    if policy.calibrated:
        if metrics["source_region_accuracy"] < float(
            policy.source_region_accuracy_min
        ):
            reasons.append("low_source_region_accuracy")
        if metrics["source_class_recall_min"] < float(
            policy.source_class_recall_min
        ):
            reasons.append("low_source_class_recall")
        if metrics["source_mean_normalized_entropy"] > float(
            policy.source_mean_normalized_entropy_max
        ):
            reasons.append("high_source_entropy")
        if metrics["source_boundary_f1_4"] < float(
            policy.source_boundary_f1_4_min
        ):
            reasons.append("low_source_boundary_f1_4")
    return {
        "metrics": metrics,
        "policy": asdict(policy),
        "policy_calibrated": policy.calibrated,
        "evaluator_uncertain": bool(reasons) if policy.calibrated else None,
        "uncertainty_reasons": reasons,
        "interpretation": (
            "source_only_flag_frozen_before_generated_metrics"
            if policy.calibrated
            else "uncalibrated_features_only_no_row_exclusion"
        ),
    }


def class_aware_boundary_f1(
    target: np.ndarray,
    predicted: np.ndarray,
    *,
    valid: np.ndarray,
    class_ids: Sequence[int],
    tolerance: int,
) -> float:
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    scores = []
    for class_id in class_ids:
        target_class = (target == int(class_id)) & valid
        predicted_class = (predicted == int(class_id)) & valid
        if not np.any(target_class) and not np.any(predicted_class):
            continue
        target_boundary = _binary_boundary(target_class)
        predicted_boundary = _binary_boundary(predicted_class)
        if not np.any(target_boundary) and not np.any(predicted_boundary):
            scores.append(1.0)
            continue
        if not np.any(target_boundary) or not np.any(predicted_boundary):
            scores.append(0.0)
            continue
        target_near = (
            _binary_dilation(target_boundary, tolerance)
            if tolerance
            else target_boundary
        )
        predicted_near = (
            _binary_dilation(predicted_boundary, tolerance)
            if tolerance
            else predicted_boundary
        )
        precision = (
            float(np.mean(target_near[predicted_boundary]))
            if np.any(predicted_boundary)
            else 0.0
        )
        recall = (
            float(np.mean(predicted_near[target_boundary]))
            if np.any(target_boundary)
            else 0.0
        )
        scores.append(
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return float(np.mean(scores)) if scores else 1.0


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    eroded = mask.copy()
    eroded[1:, :] &= mask[:-1, :]
    eroded[:-1, :] &= mask[1:, :]
    eroded[:, 1:] &= mask[:, :-1]
    eroded[:, :-1] &= mask[:, 1:]
    return mask & ~eroded


def _binary_dilation(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Four-connected binary dilation without a compiled SciPy dependency."""
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(iterations):
        expanded = result.copy()
        expanded[1:, :] |= result[:-1, :]
        expanded[:-1, :] |= result[1:, :]
        expanded[:, 1:] |= result[:, :-1]
        expanded[:, :-1] |= result[:, 1:]
        result = expanded
    return result


def _validate_probabilities(
    probabilities: np.ndarray, spatial_shape: tuple[int, int]
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != spatial_shape:
        raise ValueError(
            f"probabilities must have CHW shape with spatial size {spatial_shape}, "
            f"got {values.shape}"
        )
    normalized_entropy(values)
    return values


def _uncertainty_summary(
    confidence: Mapping[str, np.ndarray | bool | dict],
    regions: Mapping[str, np.ndarray],
) -> dict:
    entropy = np.asarray(confidence["normalized_entropy"])
    top1 = np.asarray(confidence["top1_probability"])
    margin = np.asarray(confidence["top1_top2_margin"])
    summary = {
        "policy_calibrated": bool(confidence["policy_calibrated"]),
        "mean_normalized_entropy_R": _region_mean(entropy, regions["R"]),
        "p95_normalized_entropy_R": _region_percentile(
            entropy, regions["R"], 95
        ),
        "mean_normalized_entropy_B": _region_mean(entropy, regions["B"]),
        "mean_normalized_entropy_U_far": _region_mean(
            entropy, regions["U_far"]
        ),
        "mean_top1_probability_R": _region_mean(top1, regions["R"]),
        "mean_top1_top2_margin_R": _region_mean(margin, regions["R"]),
        "high_confidence_coverage_R": None,
        "high_confidence_coverage_U_far": None,
    }
    if "high_confidence" in confidence:
        high_confidence = np.asarray(confidence["high_confidence"], dtype=bool)
        summary["high_confidence_coverage_R"] = _region_mean(
            high_confidence, regions["R"]
        )
        summary["high_confidence_coverage_U_far"] = _region_mean(
            high_confidence, regions["U_far"]
        )
    return summary


def _region_mean(values: np.ndarray, region: np.ndarray) -> float | None:
    return float(np.mean(values[region])) if np.any(region) else None


def _region_percentile(
    values: np.ndarray, region: np.ndarray, percentile: float
) -> float | None:
    return (
        float(np.percentile(values[region], percentile))
        if np.any(region)
        else None
    )


def _disagreement_rate(
    left: np.ndarray, right: np.ndarray, region: np.ndarray
) -> float | None:
    return float(np.mean(left[region] != right[region])) if np.any(region) else None


def _difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else float(left - right)

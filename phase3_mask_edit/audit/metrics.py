"""Reusable tissue-evaluator metrics for online semantic self-auditing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

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
    preservation_exclusion_region: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    source = np.asarray(source_mask)
    target = np.asarray(target_mask)
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("source and target masks must share a rank-2 shape")
    if boundary_radius < 1:
        raise ValueError("boundary_radius must be positive")
    valid = (source != ignore_index) & (target != ignore_index)
    semantic_region = (
        source != target
        if semantic_change_region is None
        else np.asarray(semantic_change_region, dtype=bool)
    )
    if semantic_region.shape != source.shape:
        raise ValueError("semantic_change_region must match source and target")
    changed = valid & semantic_region
    if not np.any(changed):
        raise ValueError("G2 semantic change region is empty")
    preservation_exclusion = np.array(changed, copy=True)
    if preservation_exclusion_region is not None:
        supplied_exclusion = np.asarray(
            preservation_exclusion_region, dtype=bool
        )
        if supplied_exclusion.shape != source.shape:
            raise ValueError(
                "preservation_exclusion_region must match source and target"
            )
        preservation_exclusion |= valid & supplied_exclusion
    unchanged = valid & ~changed
    boundary_inside = changed & _binary_dilation(~changed, boundary_radius)
    boundary_outside = unchanged & _binary_dilation(changed, boundary_radius)
    boundary = boundary_inside | boundary_outside
    unchanged_far = valid & ~preservation_exclusion & ~boundary_outside
    return {
        "valid": valid,
        "R": changed,
        "U": unchanged,
        "B_in": boundary_inside,
        "B_out": boundary_outside,
        "B": boundary,
        "P_exclude": preservation_exclusion,
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
    include_prediction_only_classes: bool = True,
) -> dict:
    target = np.asarray(target)
    predicted = np.asarray(predicted)
    region = np.asarray(region, dtype=bool)
    if not (target.shape == predicted.shape == region.shape):
        raise ValueError("target, predicted, and region shapes must match")
    allowed = {int(value) for value in class_ids}
    present = set(np.unique(target[region]))
    if include_prediction_only_classes:
        present |= set(np.unique(predicted[region]))
    labels = sorted(int(value) for value in present & allowed)
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
        "macro_policy": (
            "target_or_prediction_present"
            if include_prediction_only_classes
            else "target_present"
        ),
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
    preservation_exclusion_region: np.ndarray | None = None,
    semantic_direction_epsilon: float = 1e-4,
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
        preservation_exclusion_region=preservation_exclusion_region,
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
        include_prediction_only_classes=True,
    )
    source_iou = region_macro_iou(
        target,
        source_pred,
        changed,
        class_ids=class_ids,
        include_background=include_background,
        include_prediction_only_classes=True,
    )
    source_calibration_iou = region_macro_iou(
        source,
        source_pred,
        changed,
        class_ids=class_ids,
        include_background=include_background,
        include_prediction_only_classes=True,
    )
    target_accuracy = float(np.mean(generated_pred[changed] == target[changed]))
    no_edit_accuracy = float(np.mean(source_pred[changed] == target[changed]))
    source_calibration_accuracy = float(
        np.mean(source_pred[changed] == source[changed])
    )
    transition_calibration = transition_evaluator_calibration(
        source_mask=source,
        target_mask=target,
        source_prediction=source_pred,
        change_region=changed,
        class_ids=class_ids,
        ignore_index=ignore_index,
    )
    target_class = target[changed].astype(np.int64)
    source_class = source[changed].astype(np.int64)
    probability_count = int(source_probs.shape[0])
    allowed_direction_ids = {
        int(value)
        for value in class_ids
        if 0 <= int(value) < probability_count
        and int(value) != int(ignore_index)
    }
    is_transition = source_class != target_class
    target_direction_support = is_transition & np.isin(
        target_class, list(allowed_direction_ids)
    )
    source_direction_support = is_transition & np.isin(
        source_class, list(allowed_direction_ids)
    )
    margin_direction_support = (
        target_direction_support & source_direction_support
    )

    target_index = np.flatnonzero(target_direction_support)
    source_index = np.flatnonzero(source_direction_support)
    margin_index = np.flatnonzero(margin_direction_support)
    source_changed_probs = source_probs[:, changed]
    generated_changed_probs = generated_probs[:, changed]

    appearance_calibration = _fit_global_appearance_calibration(
        source_prediction=source_pred,
        source_probabilities=source_probs,
        generated_probabilities=generated_probs,
        region=regions["U_far"],
        class_ids=class_ids,
    )
    calibrated_outer_ring = _apply_global_appearance_calibration(
        source_prediction=source_pred,
        source_probabilities=source_probs,
        generated_probabilities=generated_probs,
        region=regions["B_out"],
        biases=appearance_calibration["biases"],
        class_ids=class_ids,
        min_evaluated_pixels=64,
    )
    changed_bias = _appearance_bias_for_pixels(
        source_prediction=source_pred[changed],
        biases=appearance_calibration["biases"],
        channel_count=probability_count,
    )

    generated_target_probability = generated_changed_probs[
        target_class[target_index], target_index
    ]
    source_target_probability = source_changed_probs[
        target_class[target_index], target_index
    ]
    target_probability_gain_values = (
        generated_target_probability - source_target_probability
    )
    calibrated_target_probability_gain_values = (
        target_probability_gain_values
        - changed_bias[target_class[target_index], target_index]
    )

    generated_source_probability = generated_changed_probs[
        source_class[source_index], source_index
    ]
    source_source_probability = source_changed_probs[
        source_class[source_index], source_index
    ]
    source_probability_suppression_values = (
        source_source_probability - generated_source_probability
    )
    calibrated_source_probability_suppression_values = (
        source_probability_suppression_values
        + changed_bias[source_class[source_index], source_index]
    )

    generated_margin_values = (
        generated_changed_probs[target_class[margin_index], margin_index]
        - generated_changed_probs[source_class[margin_index], margin_index]
    )
    source_margin_values = (
        source_changed_probs[target_class[margin_index], margin_index]
        - source_changed_probs[source_class[margin_index], margin_index]
    )
    margin_gain_values = generated_margin_values - source_margin_values
    calibrated_margin_gain_values = (
        margin_gain_values
        - changed_bias[target_class[margin_index], margin_index]
        + changed_bias[source_class[margin_index], margin_index]
    )
    changed_inner = regions["B_in"][changed]
    boundary_target_values = calibrated_target_probability_gain_values[
        changed_inner[target_index]
    ]
    boundary_source_values = calibrated_source_probability_suppression_values[
        changed_inner[source_index]
    ]
    boundary_margin_values = calibrated_margin_gain_values[
        changed_inner[margin_index]
    ]
    direction_epsilon = max(0.0, float(semantic_direction_epsilon))
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
            "soft_target_source_margin": _mean_or_none(
                generated_margin_values
            ),
            "soft_no_edit_target_source_margin": _mean_or_none(
                source_margin_values
            ),
            "soft_margin_gain": _mean_or_none(margin_gain_values),
            "source_target_probability": _mean_or_none(
                source_target_probability
            ),
            "generated_target_probability": _mean_or_none(
                generated_target_probability
            ),
            "target_probability_gain": _mean_or_none(
                target_probability_gain_values
            ),
            "appearance_calibrated_target_probability_gain": _mean_or_none(
                calibrated_target_probability_gain_values
            ),
            "source_source_probability": _mean_or_none(
                source_source_probability
            ),
            "generated_source_probability": _mean_or_none(
                generated_source_probability
            ),
            "source_probability_suppression": _mean_or_none(
                source_probability_suppression_values
            ),
            "appearance_calibrated_source_probability_suppression": (
                _mean_or_none(
                    calibrated_source_probability_suppression_values
                )
            ),
            "appearance_calibrated_soft_margin_gain": _mean_or_none(
                calibrated_margin_gain_values
            ),
            "target_probability_gain_fraction": _fraction_or_none(
                target_probability_gain_values > direction_epsilon
            ),
            "source_probability_suppression_fraction": _fraction_or_none(
                source_probability_suppression_values > direction_epsilon
            ),
            "margin_gain_fraction": _fraction_or_none(
                margin_gain_values > direction_epsilon
            ),
            "appearance_calibrated_target_probability_gain_fraction": (
                _fraction_or_none(
                    calibrated_target_probability_gain_values
                    > direction_epsilon
                )
            ),
            "appearance_calibrated_source_probability_suppression_fraction": (
                _fraction_or_none(
                    calibrated_source_probability_suppression_values
                    > direction_epsilon
                )
            ),
            "appearance_calibrated_margin_gain_fraction": _fraction_or_none(
                calibrated_margin_gain_values > direction_epsilon
            ),
            "appearance_calibration_coverage": float(
                appearance_calibration["coverage"]
            ),
            "target_direction_support_pixels": int(target_index.size),
            "source_direction_support_pixels": int(source_index.size),
            "margin_direction_support_pixels": int(margin_index.size),
            "semantic_direction_epsilon": direction_epsilon,
        },
        "source_evaluator_calibration": {
            "accuracy": source_calibration_accuracy,
            "macro_miou": source_calibration_iou["macro_miou"],
            "macro_iou_detail": source_calibration_iou,
        },
        "transition_evaluator_calibration": transition_calibration,
        "preservation": {
            "prediction_relative_drift_U": _disagreement_rate(
                generated_pred, source_pred, regions["U"]
            ),
            "prediction_relative_drift_U_far": _disagreement_rate(
                generated_pred, source_pred, regions["U_far"]
            ),
            "appearance_calibrated_prediction_drift_U_far": (
                appearance_calibration["drift"]
            ),
            "appearance_calibrated_probability_tv_U_far": (
                appearance_calibration["probability_tv"]
            ),
            "appearance_calibration_applicable": bool(
                appearance_calibration["applicable"]
            ),
            "appearance_calibration_coverage_U_far": float(
                appearance_calibration["coverage"]
            ),
            "appearance_calibration_evaluated_pixels_U_far": int(
                appearance_calibration["evaluated_pixels"]
            ),
            "appearance_calibration_class_count": int(
                appearance_calibration["class_count"]
            ),
            "appearance_calibration_class_detail": appearance_calibration[
                "class_detail"
            ],
            "global_appearance_probability_shift_l1": float(
                appearance_calibration["global_shift_l1"]
            ),
            "mask_relative_drift_U": _disagreement_rate(
                generated_pred, source, regions["U"]
            ),
            "outer_ring_spillover": _disagreement_rate(
                generated_pred, source_pred, regions["B_out"]
            ),
            "appearance_calibrated_outer_ring_spillover": (
                calibrated_outer_ring["drift"]
            ),
            "appearance_calibrated_outer_ring_applicable": bool(
                calibrated_outer_ring["applicable"]
            ),
            "inner_ring_target_error": _disagreement_rate(
                generated_pred, target, regions["B_in"]
            ),
        },
        "boundary": {
            "class_aware_f1_4": class_aware_boundary_f1(
                target,
                generated_pred,
                valid=regions["B"],
                class_ids=class_ids,
                tolerance=boundary_radius,
            ),
            "relative_inner_target_probability_gain": _mean_or_none(
                boundary_target_values
            ),
            "relative_inner_source_probability_suppression": _mean_or_none(
                boundary_source_values
            ),
            "relative_inner_margin_gain": _mean_or_none(
                boundary_margin_values
            ),
            "relative_inner_target_support_pixels": int(
                boundary_target_values.size
            ),
            "relative_inner_source_support_pixels": int(
                boundary_source_values.size
            ),
            "relative_inner_margin_support_pixels": int(
                boundary_margin_values.size
            ),
            "relative_outer_drift": calibrated_outer_ring["drift"],
            "relative_outer_applicable": bool(
                calibrated_outer_ring["applicable"]
            ),
            "relative_outer_evaluated_pixels": int(
                calibrated_outer_ring["evaluated_pixels"]
            ),
        },
        "uncertainty": {
            "source": _uncertainty_summary(source_confidence, regions),
            "generated": _uncertainty_summary(generated_confidence, regions),
        },
    }
    return result


def _fit_global_appearance_calibration(
    *,
    source_prediction: np.ndarray,
    source_probabilities: np.ndarray,
    generated_probabilities: np.ndarray,
    region: np.ndarray,
    class_ids: Sequence[int],
    min_anchor_pixels: int = 64,
    min_evaluated_pixels: int = 256,
    min_anchor_stability: float = 0.50,
    min_coverage: float = 0.25,
) -> dict:
    """Remove broad class-conditioned appearance bias before drift scoring.

    Cross is allowed to redraw stain and texture over the full image. The
    calibration is estimated only from unchanged pixels whose source and
    generated predictions remain stable. A true broad class replacement has
    too few stable anchors and therefore abstains instead of being calibrated
    away.
    """

    source_pred = np.asarray(source_prediction)
    source_probs = np.asarray(source_probabilities, dtype=np.float64)
    generated_probs = np.asarray(generated_probabilities, dtype=np.float64)
    region_mask = np.asarray(region, dtype=bool)
    source_argmax = np.argmax(source_probs, axis=0)
    generated_argmax = np.argmax(generated_probs, axis=0)
    channel_count = int(source_probs.shape[0])
    allowed = sorted(
        {
            int(value)
            for value in class_ids
            if 0 <= int(value) < channel_count
        }
    )
    evaluable = (
        region_mask
        & np.isin(source_pred, allowed)
        & (source_argmax == source_pred)
    )
    source_sorted = np.sort(source_probs, axis=0)
    source_margin = source_sorted[-1] - source_sorted[-2]
    probability_delta = generated_probs - source_probs
    biases: dict[int, np.ndarray] = {}
    class_detail: dict[str, dict[str, float | int | bool | None]] = {}
    evaluated = np.zeros(region_mask.shape, dtype=bool)
    corrected_prediction = np.array(generated_argmax, copy=True)
    probability_tv_sum = 0.0
    probability_tv_count = 0

    for class_id in allowed:
        support = evaluable & (source_pred == class_id)
        support_count = int(np.count_nonzero(support))
        stable = support & (generated_argmax == class_id)
        stable_count = int(np.count_nonzero(stable))
        stability = stable_count / support_count if support_count else 0.0
        calibratable = bool(
            support_count >= min_anchor_pixels
            and stable_count >= min_anchor_pixels
            and stability >= min_anchor_stability
        )
        detail: dict[str, float | int | bool | None] = {
            "support_pixels": support_count,
            "stable_anchor_pixels": stable_count,
            "stable_anchor_fraction": float(stability),
            "calibratable": calibratable,
            "source_margin_threshold": None,
        }
        if not calibratable:
            class_detail[str(class_id)] = detail
            continue

        bias = np.median(probability_delta[:, stable], axis=1)
        biases[class_id] = bias
        margin_threshold = float(np.quantile(source_margin[support], 0.25))
        class_evaluated = support & (source_margin >= margin_threshold)
        detail["source_margin_threshold"] = margin_threshold
        detail["evaluated_pixels"] = int(np.count_nonzero(class_evaluated))
        class_detail[str(class_id)] = detail
        evaluated |= class_evaluated

        corrected = generated_probs[:, class_evaluated] - bias[:, None]
        corrected = np.clip(corrected, 1e-8, None)
        corrected /= np.sum(corrected, axis=0, keepdims=True)
        corrected_prediction[class_evaluated] = np.argmax(corrected, axis=0)
        source_values = source_probs[:, class_evaluated]
        probability_tv_sum += float(
            np.sum(0.5 * np.sum(np.abs(corrected - source_values), axis=0))
        )
        probability_tv_count += int(corrected.shape[1])

    evaluable_pixels = int(np.count_nonzero(evaluable))
    evaluated_pixels = int(np.count_nonzero(evaluated))
    coverage = evaluated_pixels / evaluable_pixels if evaluable_pixels else 0.0
    applicable = bool(
        evaluated_pixels >= min_evaluated_pixels and coverage >= min_coverage
    )
    drift = (
        float(np.mean(corrected_prediction[evaluated] != source_pred[evaluated]))
        if evaluated_pixels
        else None
    )
    probability_tv = (
        probability_tv_sum / probability_tv_count
        if probability_tv_count
        else None
    )
    global_shift_l1 = (
        float(np.mean(np.abs(probability_delta[:, evaluated])))
        if evaluated_pixels
        else 0.0
    )
    return {
        "applicable": applicable,
        "drift": drift,
        "probability_tv": probability_tv,
        "coverage": float(coverage),
        "evaluated_pixels": evaluated_pixels,
        "class_count": len(biases),
        "class_detail": class_detail,
        "global_shift_l1": global_shift_l1,
        "biases": biases,
    }


def _appearance_bias_for_pixels(
    *,
    source_prediction: np.ndarray,
    biases: Mapping[int, np.ndarray],
    channel_count: int,
) -> np.ndarray:
    prediction = np.asarray(source_prediction)
    result = np.zeros((channel_count, prediction.size), dtype=np.float64)
    flat_prediction = prediction.reshape(-1)
    for class_id, bias in biases.items():
        selected = flat_prediction == int(class_id)
        if np.any(selected):
            result[:, selected] = np.asarray(bias, dtype=np.float64)[:, None]
    return result


def _apply_global_appearance_calibration(
    *,
    source_prediction: np.ndarray,
    source_probabilities: np.ndarray,
    generated_probabilities: np.ndarray,
    region: np.ndarray,
    biases: Mapping[int, np.ndarray],
    class_ids: Sequence[int],
    min_evaluated_pixels: int,
    min_coverage: float = 0.25,
) -> dict[str, float | int | bool | None]:
    source_pred = np.asarray(source_prediction)
    source_probs = np.asarray(source_probabilities, dtype=np.float64)
    generated_probs = np.asarray(generated_probabilities, dtype=np.float64)
    region_mask = np.asarray(region, dtype=bool)
    source_argmax = np.argmax(source_probs, axis=0)
    source_sorted = np.sort(source_probs, axis=0)
    source_margin = source_sorted[-1] - source_sorted[-2]
    allowed = {
        int(value)
        for value in class_ids
        if 0 <= int(value) < source_probs.shape[0]
    }
    evaluable = (
        region_mask
        & np.isin(source_pred, list(allowed))
        & (source_argmax == source_pred)
    )
    evaluated = np.zeros(region_mask.shape, dtype=bool)
    corrected_prediction = np.argmax(generated_probs, axis=0)
    for class_id, bias in biases.items():
        support = evaluable & (source_pred == int(class_id))
        if not np.any(support):
            continue
        margin_threshold = float(np.quantile(source_margin[support], 0.25))
        selected = support & (source_margin >= margin_threshold)
        evaluated |= selected
        corrected = generated_probs[:, selected] - np.asarray(bias)[:, None]
        corrected = np.clip(corrected, 1e-8, None)
        corrected /= np.sum(corrected, axis=0, keepdims=True)
        corrected_prediction[selected] = np.argmax(corrected, axis=0)

    evaluable_pixels = int(np.count_nonzero(evaluable))
    evaluated_pixels = int(np.count_nonzero(evaluated))
    coverage = evaluated_pixels / evaluable_pixels if evaluable_pixels else 0.0
    return {
        "applicable": bool(
            evaluated_pixels >= min_evaluated_pixels
            and coverage >= min_coverage
        ),
        "drift": (
            float(
                np.mean(
                    corrected_prediction[evaluated] != source_pred[evaluated]
                )
            )
            if evaluated_pixels
            else None
        ),
        "coverage": float(coverage),
        "evaluated_pixels": evaluated_pixels,
    }


def transition_evaluator_calibration(
    *,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    source_prediction: np.ndarray,
    change_region: np.ndarray,
    class_ids: Sequence[int],
    ignore_index: int = 255,
    target_reference_min_pixels: int = 256,
) -> dict:
    source = np.asarray(source_mask)
    target = np.asarray(target_mask)
    prediction = np.asarray(source_prediction)
    changed = np.asarray(change_region, dtype=bool)
    if not (source.shape == target.shape == prediction.shape == changed.shape):
        raise ValueError("transition calibration inputs must share a shape")
    allowed = {int(value) for value in class_ids}
    transition = (
        changed
        & (source != target)
        & (source != ignore_index)
        & (target != ignore_index)
        & np.isin(source, list(allowed))
        & np.isin(target, list(allowed))
    )
    source_labels = sorted(int(value) for value in np.unique(source[transition]))
    target_labels = sorted(int(value) for value in np.unique(target[transition]))
    source_recall: dict[str, float] = {}
    source_support: dict[str, int] = {}
    for class_id in source_labels:
        pixels = transition & (source == class_id)
        source_support[str(class_id)] = int(np.count_nonzero(pixels))
        source_recall[str(class_id)] = float(
            np.mean(prediction[pixels] == class_id)
        )

    target_reference_recall: dict[str, float | None] = {}
    target_reference_support: dict[str, int] = {}
    for class_id in target_labels:
        pixels = (~changed) & (source == class_id)
        support = int(np.count_nonzero(pixels))
        target_reference_support[str(class_id)] = support
        target_reference_recall[str(class_id)] = (
            float(np.mean(prediction[pixels] == class_id))
            if support >= target_reference_min_pixels
            else None
        )

    available_target_recalls = [
        float(value)
        for value in target_reference_recall.values()
        if value is not None
    ]
    target_reference_available = bool(
        target_labels
        and len(available_target_recalls) == len(target_labels)
    )
    source_target_confusion = (
        float(np.mean(np.isin(prediction[transition], target_labels)))
        if np.any(transition) and target_labels
        else None
    )
    return {
        "transition_pixels": int(np.count_nonzero(transition)),
        "source_labels": source_labels,
        "target_labels": target_labels,
        "source_class_support": source_support,
        "source_class_recall": source_recall,
        "source_class_recall_min": (
            min(source_recall.values()) if source_recall else None
        ),
        "target_reference_min_pixels": int(target_reference_min_pixels),
        "target_reference_support": target_reference_support,
        "target_reference_class_recall": target_reference_recall,
        "target_reference_recall_min": (
            min(available_target_recalls)
            if target_reference_available
            else None
        ),
        "target_reference_available": target_reference_available,
        "source_to_target_confusion_rate": source_target_confusion,
    }


def source_evaluator_quality(
    *,
    source_mask: np.ndarray,
    source_prediction: np.ndarray,
    source_probabilities: np.ndarray,
    class_ids: Sequence[int] = tuple(range(8)),
    ignore_index: int = 255,
    region: np.ndarray | None = None,
    policy: EvaluatorCleanPolicy | None = None,
) -> dict:
    source = np.asarray(source_mask)
    prediction = np.asarray(source_prediction)
    if source.shape != prediction.shape or source.ndim != 2:
        raise ValueError("source mask and prediction must share a rank-2 shape")
    probabilities = _validate_probabilities(source_probabilities, source.shape)
    valid = source != ignore_index
    if region is not None:
        region_mask = np.asarray(region, dtype=bool)
        if region_mask.shape != source.shape:
            raise ValueError("source evaluator region must match the source mask")
        valid &= region_mask
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


def _mean_or_none(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if values.size else None


def _fraction_or_none(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if values.size else None


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

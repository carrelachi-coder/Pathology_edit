"""Generic immune infiltration mask primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy import ndimage

from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.generic.tumor_burden import (
    PrimitiveEditResult,
    PrimitiveExecutionError,
    _four_neighbor_structure,
    _target_pixels,
)


_DEFAULT_FALLOFF_RADIUS_PX = 96.0
_DEFAULT_IMMUNE_NEIGHBOR_RADIUS_PX = 48.0
_DEFAULT_MIN_TUMOR_FRACTION_FOR_NORMAL_PRIORITY = 0.05
_DEFAULT_MIN_COMPONENT_AREA_PX = 400
_DEFAULT_MAX_COMPONENTS = 6
_DEFAULT_SMOOTHING_RADIUS_PX = 4


@dataclass(frozen=True)
class _TumorModeInfo:
    """Weights for stromal immune candidate scoring."""

    tumor_mode: str
    tumor_fraction: float
    min_tumor_fraction_for_normal_priority: float
    proximity_weight: float
    immune_neighbor_weight: float
    noise_weight: float


@dataclass(frozen=True)
class _ScoreInfo:
    """Stromal immune candidate scoring metadata."""

    score: np.ndarray
    tumor_mode: str
    tumor_fraction: float
    min_tumor_fraction_for_normal_priority: float
    used_soft_peritumoral_priority: bool
    used_existing_immune_neighborhood: bool
    active_weights: dict[str, float]
    radii: dict[str, float]


@dataclass(frozen=True)
class _SelectionInfo:
    """Connected-component selection metadata."""

    change_region: np.ndarray
    score_threshold: float
    threshold_percentile: float
    selected_components: int
    pre_cleanup_pixels: int
    removed_small_component_pixels: int
    smoothing_added_pixels: int
    smoothing_removed_pixels: int
    hole_fill_pixels: int
    smoothing_radius_px: int
    smoothing_applied: bool


def apply_stromal_immune_infiltration(
    old_mask: np.ndarray,
    schema: MaskProfileSchema,
    context: MaskEditContext,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> PrimitiveEditResult:
    """Convert patchy stromal regions into Immune infiltrate."""

    _validate_stromal_immune_request(
        old_mask, schema, context, primitive_config, intent
    )
    normalized = context.normalized_mask
    target_fraction = _target_immune_delta_fraction_for_intent(
        primitive_config, intent
    )

    stroma_ids = schema.resolve_fine_ids("Stroma")
    immune_ids = schema.resolve_fine_ids("Immune infiltrate")
    tumor = np.isin(normalized, schema.tumor_fine_ids)
    stroma = np.isin(normalized, stroma_ids)
    existing_immune = np.isin(normalized, immune_ids)

    stroma_pixels = int(np.count_nonzero(stroma))
    if stroma_pixels == 0:
        raise PrimitiveExecutionError("no_stroma")

    reference_pixels = stroma_pixels + int(np.count_nonzero(existing_immune))
    if reference_pixels == 0:
        raise PrimitiveExecutionError("no_stromal_immune_candidate_region")

    candidate_base = stroma
    candidate_pixels = int(np.count_nonzero(candidate_base))
    target_count = _target_pixels(target_fraction, reference_pixels)
    capped_target_count = min(target_count, candidate_pixels)
    if capped_target_count < 1:
        raise PrimitiveExecutionError("no_stromal_immune_candidate_region")

    score_info = _stromal_immune_score(
        normalized,
        tumor,
        existing_immune,
        candidate_base,
        primitive_config,
        intent,
    )
    min_component_area = _min_component_area_for_intent(intent)
    max_components = _max_components_for_intent(intent)
    selection = _select_patchy_high_score_regions(
        score_info.score,
        candidate_base,
        target_pixels=capped_target_count,
        min_component_area=min_component_area,
        max_components=max_components,
    )
    change_region = selection.change_region
    selected_pixels = int(np.count_nonzero(change_region))
    if selected_pixels == 0:
        raise PrimitiveExecutionError("no_stromal_immune_candidate_region")

    target_mask = np.array(normalized, copy=True)
    immune_label = int(immune_ids[0])
    target_mask[change_region] = immune_label

    changed_area_fraction = selected_pixels / target_mask.size
    ops_log = {
        "primitive": "stromal_immune_infiltration",
        "reference_profile": schema.reference_profile,
        "target_change_fraction": target_fraction,
        "changed_area_fraction": changed_area_fraction,
        "changed_stroma_fraction": selected_pixels / stroma_pixels,
        "changed_stroma_immune_fraction": selected_pixels / reference_pixels,
        "selected_pixels": selected_pixels,
        "immune_label": immune_label,
        "spatial": {
            "method": "soft_peritumoral_stroma_to_immune",
            "target_area_reference": "stroma_plus_immune",
            "target_pixels": int(target_count),
            "capped_target_pixels": int(capped_target_count),
            "reference_pixels": int(reference_pixels),
            "stroma_pixels": stroma_pixels,
            "existing_immune_pixels": int(np.count_nonzero(existing_immune)),
            "candidate_pixels": candidate_pixels,
            "distance_policy": "peritumoral_priority_no_hard_limit",
            "tumor_mode": score_info.tumor_mode,
            "tumor_pixels": int(np.count_nonzero(tumor)),
            "tumor_fraction": score_info.tumor_fraction,
            "min_tumor_fraction_for_normal_priority": (
                score_info.min_tumor_fraction_for_normal_priority
            ),
            "peritumoral_falloff_radius_px": (
                score_info.radii["peritumoral_falloff_radius_px"]
            ),
            "hard_distance_limit_px": None,
            "used_soft_peritumoral_priority": (
                score_info.used_soft_peritumoral_priority
            ),
            "used_existing_immune_neighborhood": (
                score_info.used_existing_immune_neighborhood
            ),
            "active_weights": dict(score_info.active_weights),
            "radii_px": dict(score_info.radii),
            "score_threshold": selection.score_threshold,
            "threshold_percentile": selection.threshold_percentile,
            "selected_components": selection.selected_components,
            "pre_cleanup_pixels": selection.pre_cleanup_pixels,
            "removed_small_component_pixels": (
                selection.removed_small_component_pixels
            ),
            "smoothing_added_pixels": selection.smoothing_added_pixels,
            "smoothing_removed_pixels": selection.smoothing_removed_pixels,
            "hole_fill_pixels": selection.hole_fill_pixels,
            "smoothing_radius_px": selection.smoothing_radius_px,
            "smoothing_applied": selection.smoothing_applied,
            "min_component_area_px": min_component_area,
            "max_components": max_components,
        },
    }

    return PrimitiveEditResult(
        target_mask=target_mask,
        change_region=change_region,
        changed_area_fraction=changed_area_fraction,
        selected_pixels=selected_pixels,
        warnings=(),
        ops_log=ops_log,
    )


def _validate_stromal_immune_request(
    old_mask: np.ndarray,
    schema: MaskProfileSchema,
    context: MaskEditContext,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> None:
    mask = np.asarray(old_mask)
    if mask.ndim != 2:
        raise PrimitiveExecutionError(
            "stromal_immune_infiltration requires a 2D mask."
        )
    if tuple(mask.shape) != context.mask_shape:
        raise PrimitiveExecutionError("old_mask shape must match MaskEditContext.")
    if schema.reference_profile != context.reference_profile:
        raise PrimitiveExecutionError(
            "schema.reference_profile must match context.reference_profile."
        )
    if intent.primitive != "stromal_immune_infiltration":
        raise PrimitiveExecutionError(
            "apply_stromal_immune_infiltration requires a "
            "stromal_immune_infiltration intent."
        )
    if primitive_config.get("name") != "stromal_immune_infiltration":
        raise PrimitiveExecutionError(
            "primitive_config must describe stromal_immune_infiltration."
        )
    if "Stroma" not in schema.readable_labels:
        raise PrimitiveExecutionError("no_stroma")
    if "Immune infiltrate" not in schema.readable_labels:
        raise PrimitiveExecutionError("no_immune_label")


def _target_immune_delta_fraction_for_intent(
    primitive_config: Mapping[str, Any], intent: EditIntent
) -> float:
    if intent.target_change_fraction is not None:
        return intent.target_change_fraction

    intervals = primitive_config.get("parameter_ranges", {}).get(
        "immune_area_delta_fraction", {}
    )
    interval = intervals.get(intent.strength)
    if not isinstance(interval, list) or len(interval) != 2:
        raise PrimitiveExecutionError(
            f"stromal_immune_infiltration does not define strength {intent.strength}."
        )

    lower, upper = float(interval[0]), float(interval[1])
    return (lower + upper) / 2


def _stromal_immune_score(
    mask: np.ndarray,
    tumor: np.ndarray,
    existing_immune: np.ndarray,
    candidate_base: np.ndarray,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> _ScoreInfo:
    tumor_pixels = int(np.count_nonzero(tumor))
    mode = _tumor_mode_for_intent(mask.size, tumor_pixels, intent)
    falloff_radius = _positive_float_parameter(
        intent,
        "peritumoral_falloff_radius_px",
        float(
            primitive_config.get("parameter_ranges", {}).get(
                "peritumoral_falloff_radius_px",
                _DEFAULT_FALLOFF_RADIUS_PX,
            )
        ),
    )
    immune_radius = _positive_float_parameter(
        intent,
        "immune_neighbor_radius_px",
        _DEFAULT_IMMUNE_NEIGHBOR_RADIUS_PX,
    )

    raw_weights = {
        "peritumoral_proximity": mode.proximity_weight,
        "smooth_noise": mode.noise_weight,
    }
    has_existing_immune = bool(np.any(existing_immune))
    if has_existing_immune:
        raw_weights["existing_immune_neighborhood"] = mode.immune_neighbor_weight

    active_weights = _normalize_positive_weights(raw_weights)
    score = np.zeros(mask.shape, dtype=float)

    if "peritumoral_proximity" in active_weights and tumor_pixels > 0:
        dist_to_tumor = ndimage.distance_transform_edt(~tumor)
        proximity_score = np.exp(-dist_to_tumor / falloff_radius)
        score += active_weights["peritumoral_proximity"] * proximity_score

    if "existing_immune_neighborhood" in active_weights:
        dist_to_immune = ndimage.distance_transform_edt(~existing_immune)
        immune_score = np.exp(-dist_to_immune / immune_radius)
        score += active_weights["existing_immune_neighborhood"] * immune_score

    if "smooth_noise" in active_weights:
        noise = _smooth_patchy_noise(mask.shape, seed=intent.seed)
        score += active_weights["smooth_noise"] * noise

    score[~candidate_base] = -np.inf
    return _ScoreInfo(
        score=score,
        tumor_mode=mode.tumor_mode,
        tumor_fraction=mode.tumor_fraction,
        min_tumor_fraction_for_normal_priority=(
            mode.min_tumor_fraction_for_normal_priority
        ),
        used_soft_peritumoral_priority=(
            tumor_pixels > 0 and active_weights.get("peritumoral_proximity", 0.0) > 0
        ),
        used_existing_immune_neighborhood=(
            "existing_immune_neighborhood" in active_weights
        ),
        active_weights=active_weights,
        radii={
            "peritumoral_falloff_radius_px": falloff_radius,
            "immune_neighbor_radius_px": immune_radius,
        },
    )


def _tumor_mode_for_intent(
    total_pixels: int, tumor_pixels: int, intent: EditIntent
) -> _TumorModeInfo:
    min_fraction = _positive_float_parameter(
        intent,
        "min_tumor_fraction_for_normal_priority",
        _DEFAULT_MIN_TUMOR_FRACTION_FOR_NORMAL_PRIORITY,
    )
    if min_fraction > 1.0:
        raise PrimitiveExecutionError(
            "parameters.min_tumor_fraction_for_normal_priority must be <= 1."
        )

    tumor_fraction = tumor_pixels / total_pixels
    if tumor_pixels == 0:
        return _TumorModeInfo(
            tumor_mode="none",
            tumor_fraction=tumor_fraction,
            min_tumor_fraction_for_normal_priority=min_fraction,
            proximity_weight=0.0,
            immune_neighbor_weight=0.20,
            noise_weight=1.0,
        )
    if tumor_fraction < min_fraction:
        return _TumorModeInfo(
            tumor_mode="small",
            tumor_fraction=tumor_fraction,
            min_tumor_fraction_for_normal_priority=min_fraction,
            proximity_weight=0.40,
            immune_neighbor_weight=0.15,
            noise_weight=0.60,
        )
    return _TumorModeInfo(
        tumor_mode="normal",
        tumor_fraction=tumor_fraction,
        min_tumor_fraction_for_normal_priority=min_fraction,
        proximity_weight=0.55,
        immune_neighbor_weight=0.15,
        noise_weight=0.35,
    )


def _select_patchy_high_score_regions(
    score: np.ndarray,
    candidate_base: np.ndarray,
    *,
    target_pixels: int,
    min_component_area: int,
    max_components: int,
) -> _SelectionInfo:
    finite_scores = score[candidate_base & np.isfinite(score)]
    if finite_scores.size == 0:
        raise PrimitiveExecutionError("no_stromal_immune_candidate_region")

    percentile = max(
        0.0,
        100.0 * (1.0 - min(float(target_pixels) / float(finite_scores.size), 1.0)),
    )
    threshold = float(np.percentile(finite_scores, percentile))
    high_score = candidate_base & np.isfinite(score) & (score >= threshold)
    selected = _select_components_by_score(
        high_score,
        score,
        target_pixels,
        max_components=max_components,
    )
    pre_cleanup_pixels = int(np.count_nonzero(selected))
    selected, removed_pixels = _remove_small_components(selected, min_component_area)
    if not np.any(selected):
        threshold = float(np.percentile(finite_scores, 50.0))
        percentile = 50.0
        high_score = candidate_base & np.isfinite(score) & (score >= threshold)
        selected = _select_components_by_score(
            high_score,
            score,
            target_pixels,
            max_components=max_components,
        )
        pre_cleanup_pixels = int(np.count_nonzero(selected))
        selected, removed_pixels = _remove_small_components(
            selected, min_component_area
        )

    selected, smoothing_info = _smooth_patchy_region(
        selected,
        candidate_base,
        min_component_area=min_component_area,
    )
    selected = _limit_selected_by_score(selected, score, target_pixels)
    selected, post_limit_removed = _remove_small_components(
        selected, min_component_area
    )
    removed_pixels += post_limit_removed
    selected_components = int(
        ndimage.label(selected, structure=_four_neighbor_structure())[1]
    )
    return _SelectionInfo(
        change_region=selected,
        score_threshold=threshold,
        threshold_percentile=float(percentile),
        selected_components=selected_components,
        pre_cleanup_pixels=pre_cleanup_pixels,
        removed_small_component_pixels=removed_pixels,
        smoothing_added_pixels=smoothing_info["smoothing_added_pixels"],
        smoothing_removed_pixels=smoothing_info["smoothing_removed_pixels"],
        hole_fill_pixels=smoothing_info["hole_fill_pixels"],
        smoothing_radius_px=smoothing_info["smoothing_radius_px"],
        smoothing_applied=smoothing_info["smoothing_applied"],
    )


def _select_components_by_score(
    high_score: np.ndarray,
    score: np.ndarray,
    target_pixels: int,
    *,
    max_components: int,
) -> np.ndarray:
    labeled, count = ndimage.label(high_score, structure=_four_neighbor_structure())
    if count == 0:
        return np.zeros_like(high_score, dtype=bool)

    components: list[tuple[float, int, int]] = []
    for component_id in range(1, count + 1):
        component = labeled == component_id
        area = int(np.count_nonzero(component))
        mean_score = float(score[component].mean())
        components.append((mean_score, area, component_id))

    components.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = np.zeros_like(high_score, dtype=bool)
    selected_count = 0
    selected_components = 0
    for _, area, component_id in components:
        if selected_components >= max_components or selected_count >= target_pixels:
            break
        component = labeled == component_id
        selected |= component
        selected_count += area
        selected_components += 1
    return selected


def _remove_small_components(
    selected: np.ndarray, min_component_area: int
) -> tuple[np.ndarray, int]:
    if min_component_area <= 1:
        return selected, 0

    labeled, count = ndimage.label(selected, structure=_four_neighbor_structure())
    cleaned = np.zeros_like(selected, dtype=bool)
    removed_pixels = 0
    for component_id in range(1, count + 1):
        component = labeled == component_id
        area = int(np.count_nonzero(component))
        if area >= min_component_area:
            cleaned |= component
        else:
            removed_pixels += area
    return cleaned, removed_pixels


def _smooth_patchy_region(
    selected: np.ndarray,
    candidate_base: np.ndarray,
    *,
    min_component_area: int,
) -> tuple[np.ndarray, dict[str, int | bool]]:
    if not np.any(selected):
        return selected, {
            "smoothing_added_pixels": 0,
            "smoothing_removed_pixels": 0,
            "hole_fill_pixels": 0,
            "smoothing_radius_px": _DEFAULT_SMOOTHING_RADIUS_PX,
            "smoothing_applied": False,
        }

    before = selected.copy()
    structure = _disk_structure(_DEFAULT_SMOOTHING_RADIUS_PX)
    smoothed = np.zeros_like(selected, dtype=bool)
    hole_fill_pixels = 0
    labeled, count = ndimage.label(selected, structure=_four_neighbor_structure())
    for component_id in range(1, count + 1):
        component = labeled == component_id
        closed = ndimage.binary_closing(component, structure=structure)
        filled = ndimage.binary_fill_holes(closed) & candidate_base
        hole_fill_pixels += int(np.count_nonzero(filled & ~closed))
        closed = filled
        opened = ndimage.binary_opening(closed, structure=structure)
        if int(np.count_nonzero(opened)) < min_component_area:
            opened = closed
        smoothed |= opened & candidate_base

    smoothed, _ = _remove_small_components(smoothed, min_component_area)
    added = int(np.count_nonzero(smoothed & ~before))
    removed = int(np.count_nonzero(before & ~smoothed))
    return smoothed, {
        "smoothing_added_pixels": added,
        "smoothing_removed_pixels": removed,
        "hole_fill_pixels": hole_fill_pixels,
        "smoothing_radius_px": _DEFAULT_SMOOTHING_RADIUS_PX,
        "smoothing_applied": bool(
            added > 0 or removed > 0 or hole_fill_pixels > 0
        ),
    }


def _limit_selected_by_score(
    selected: np.ndarray,
    score: np.ndarray,
    max_pixels: int,
) -> np.ndarray:
    current_pixels = int(np.count_nonzero(selected))
    if current_pixels <= max_pixels:
        return selected
    if max_pixels < 1:
        return np.zeros_like(selected, dtype=bool)

    coords = np.argwhere(selected)
    values = score[coords[:, 0], coords[:, 1]]
    order = np.argsort(values, kind="stable")[::-1]
    chosen = coords[order[:max_pixels]]
    limited = np.zeros_like(selected, dtype=bool)
    limited[chosen[:, 0], chosen[:, 1]] = True
    return limited


def _disk_structure(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return (yy * yy + xx * xx) <= radius * radius


def _normalize_positive_weights(raw_weights: Mapping[str, float]) -> dict[str, float]:
    active: dict[str, float] = {}
    for name, value in raw_weights.items():
        if not isinstance(value, (int, float)) or float(value) < 0:
            raise PrimitiveExecutionError(f"invalid stromal immune weight: {name}")
        if float(value) > 0:
            active[str(name)] = float(value)

    total = sum(active.values())
    if total <= 0:
        raise PrimitiveExecutionError("no_stromal_immune_candidate_region")
    return {name: value / total for name, value in active.items()}


def _smooth_patchy_noise(shape: tuple[int, int], *, seed: int | None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = np.zeros(shape, dtype=float)
    for sigma, amplitude in ((36.0, 0.55), (14.0, 0.30), (5.0, 0.15)):
        raw = rng.standard_normal(shape)
        smoothed = ndimage.gaussian_filter(raw, sigma=sigma)
        max_abs = float(np.abs(smoothed).max())
        if max_abs > 0:
            smoothed /= max_abs
        noise += amplitude * smoothed

    noise -= float(noise.min())
    max_val = float(noise.max())
    if max_val > 0:
        noise /= max_val
    return noise


def _positive_float_parameter(
    intent: EditIntent, key: str, default: float
) -> float:
    value = intent.parameters.get(key, default)
    if not isinstance(value, (int, float)) or float(value) <= 0:
        raise PrimitiveExecutionError(f"parameters.{key} must be a positive number.")
    return float(value)


def _min_component_area_for_intent(intent: EditIntent) -> int:
    value = intent.parameters.get(
        "min_stromal_immune_component_area_px",
        _DEFAULT_MIN_COMPONENT_AREA_PX,
    )
    if not isinstance(value, int) or value < 1:
        raise PrimitiveExecutionError(
            "parameters.min_stromal_immune_component_area_px must be a positive integer."
        )
    return value


def _max_components_for_intent(intent: EditIntent) -> int:
    value = intent.parameters.get(
        "max_stromal_immune_components",
        _DEFAULT_MAX_COMPONENTS,
    )
    if not isinstance(value, int) or value < 1:
        raise PrimitiveExecutionError(
            "parameters.max_stromal_immune_components must be a positive integer."
        )
    return value

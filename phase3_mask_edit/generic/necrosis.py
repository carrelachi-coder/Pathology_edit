"""Generic necrosis appearance mask primitive."""

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


_DEFAULT_NECROSIS_NEIGHBOR_RADIUS_PX = 48.0
_DEFAULT_TUMOR_INTERIOR_RADIUS_PX = 64.0
_DEFAULT_VESSEL_AVOIDANCE_RADIUS_PX = 96.0
_DEFAULT_NOISE_WEIGHT = 0.05
_DEFAULT_MIN_COMPONENT_AREA_PX = 12
_DEFAULT_MAX_COMPONENTS_NO_EXISTING_NECROSIS = 1
_DEFAULT_MAX_COMPONENTS_WITH_EXISTING_NECROSIS = 3
_DEFAULT_CLOSING_RADIUS_PX = 3
_MIN_SELECTED_TARGET_FRACTION = 0.75


@dataclass(frozen=True)
class _ScoreInfo:
    """Necrosis candidate scoring metadata."""

    score: np.ndarray
    used_existing_necrosis_neighborhood: bool
    used_blood_vessel_distance: bool
    active_weights: dict[str, float]
    radii: dict[str, float]


@dataclass(frozen=True)
class _SelectionInfo:
    """Connected-component selection metadata."""

    change_region: np.ndarray
    score_threshold: float
    threshold_percentile: float
    selected_components: int
    retry_applied: bool
    pre_cleanup_pixels: int
    removed_small_component_pixels: int
    hole_fill_pixels: int
    closing_added_pixels: int
    morphology_cleanup_applied: bool


def apply_necrosis_appearance(
    old_mask: np.ndarray,
    schema: MaskProfileSchema,
    context: MaskEditContext,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> PrimitiveEditResult:
    """Replace a plausible intratumoral candidate region with Necrosis."""

    _validate_necrosis_appearance_request(
        old_mask, schema, context, primitive_config, intent
    )
    mask = np.asarray(old_mask)
    normalized = context.normalized_mask
    target_fraction = _target_changed_fraction_for_intent(primitive_config, intent)

    necrosis_ids = schema.resolve_fine_ids("Necrosis")
    source_tumor = np.isin(normalized, schema.tumor_fine_ids)
    existing_necrosis = np.isin(normalized, necrosis_ids)
    tumor_pixels = int(np.count_nonzero(source_tumor))
    if tumor_pixels == 0:
        raise PrimitiveExecutionError("no_tumor")

    candidate_base = source_tumor & ~existing_necrosis
    candidate_pixels = int(np.count_nonzero(candidate_base))
    if candidate_pixels == 0:
        raise PrimitiveExecutionError("tumor_too_small")

    max_necrosis_fraction = _max_necrosis_fraction_of_tumor(primitive_config)
    existing_necrosis_pixels = int(np.count_nonzero(existing_necrosis))
    max_necrosis_pixels = int(round(max_necrosis_fraction * tumor_pixels))
    remaining_allowed = max_necrosis_pixels - existing_necrosis_pixels
    if remaining_allowed < 1:
        raise PrimitiveExecutionError("necrosis_fraction_limit_reached")

    target_count = _target_pixels(target_fraction, tumor_pixels)
    capped_target_count = min(target_count, remaining_allowed, candidate_pixels)
    if capped_target_count < 1:
        raise PrimitiveExecutionError("no_high_probability_hypoxic_candidate_region")

    score_info = _necrosis_probability_score(
        normalized,
        schema,
        source_tumor,
        existing_necrosis,
        candidate_base,
        primitive_config,
        intent,
    )
    min_component_area = _min_component_area_for_intent(intent)
    max_components = _max_components_for_intent(
        intent,
        has_existing_necrosis=bool(np.any(existing_necrosis)),
    )
    selection = _select_connected_high_score_region(
        score_info.score,
        candidate_base,
        target_pixels=capped_target_count,
        min_component_area=min_component_area,
        max_components=max_components,
    )
    change_region = selection.change_region
    selected_pixels = int(np.count_nonzero(change_region))
    if selected_pixels == 0:
        raise PrimitiveExecutionError("no_high_probability_hypoxic_candidate_region")

    target_mask = np.array(normalized, copy=True)
    necrosis_label = int(necrosis_ids[0])
    target_mask[change_region] = necrosis_label

    changed_area_fraction = selected_pixels / target_mask.size
    ops_log = {
        "primitive": "necrosis_appearance",
        "reference_profile": schema.reference_profile,
        "target_change_fraction": target_fraction,
        "changed_area_fraction": changed_area_fraction,
        "changed_tumor_fraction": selected_pixels / tumor_pixels,
        "selected_pixels": selected_pixels,
        "necrosis_label": necrosis_label,
        "spatial": {
            "method": "necrosis_probability_intratumoral_replacement",
            "target_area_reference": "tumor",
            "target_pixels": int(target_count),
            "capped_target_pixels": int(capped_target_count),
            "tumor_pixels": tumor_pixels,
            "existing_necrosis_pixels": existing_necrosis_pixels,
            "max_necrosis_fraction_of_tumor": max_necrosis_fraction,
            "necrosis_denominator_policy": "tumor_only",
            "remaining_allowed_necrosis_pixels": int(remaining_allowed),
            "candidate_pixels": candidate_pixels,
            "used_existing_necrosis_neighborhood": (
                score_info.used_existing_necrosis_neighborhood
            ),
            "used_blood_vessel_distance": score_info.used_blood_vessel_distance,
            "active_weights": dict(score_info.active_weights),
            "radii_px": dict(score_info.radii),
            "score_threshold": selection.score_threshold,
            "threshold_percentile": selection.threshold_percentile,
            "selected_components": selection.selected_components,
            "retry_applied": selection.retry_applied,
            "pre_cleanup_pixels": selection.pre_cleanup_pixels,
            "removed_small_component_pixels": selection.removed_small_component_pixels,
            "hole_fill_pixels": selection.hole_fill_pixels,
            "closing_added_pixels": selection.closing_added_pixels,
            "morphology_cleanup_applied": selection.morphology_cleanup_applied,
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


def _validate_necrosis_appearance_request(
    old_mask: np.ndarray,
    schema: MaskProfileSchema,
    context: MaskEditContext,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> None:
    mask = np.asarray(old_mask)
    if mask.ndim != 2:
        raise PrimitiveExecutionError("necrosis_appearance requires a 2D mask.")
    if tuple(mask.shape) != context.mask_shape:
        raise PrimitiveExecutionError("old_mask shape must match MaskEditContext.")
    if schema.reference_profile != context.reference_profile:
        raise PrimitiveExecutionError(
            "schema.reference_profile must match context.reference_profile."
        )
    if intent.primitive != "necrosis_appearance":
        raise PrimitiveExecutionError(
            "apply_necrosis_appearance requires a necrosis_appearance intent."
        )
    if primitive_config.get("name") != "necrosis_appearance":
        raise PrimitiveExecutionError(
            "primitive_config must describe necrosis_appearance."
        )
    if "Necrosis" not in schema.readable_labels:
        raise PrimitiveExecutionError("no_necrosis_label")


def _target_changed_fraction_for_intent(
    primitive_config: Mapping[str, Any], intent: EditIntent
) -> float:
    if intent.target_change_fraction is not None:
        return intent.target_change_fraction

    intervals = (
        primitive_config.get("parameter_ranges", {})
        .get("target_changed_area_fraction", {})
    )
    interval = intervals.get(intent.strength)
    if not isinstance(interval, list) or len(interval) != 2:
        raise PrimitiveExecutionError(
            f"necrosis_appearance does not define strength {intent.strength}."
        )

    lower, upper = float(interval[0]), float(interval[1])
    return (lower + upper) / 2


def _max_necrosis_fraction_of_tumor(primitive_config: Mapping[str, Any]) -> float:
    value = primitive_config.get("parameter_ranges", {}).get(
        "max_necrosis_fraction_of_tumor", 0.60
    )
    if not isinstance(value, (int, float)) or not 0 < float(value) <= 1:
        raise PrimitiveExecutionError("invalid max_necrosis_fraction_of_tumor.")
    return float(value)


def _necrosis_probability_score(
    mask: np.ndarray,
    schema: MaskProfileSchema,
    source_tumor: np.ndarray,
    existing_necrosis: np.ndarray,
    candidate_base: np.ndarray,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> _ScoreInfo:
    shape = mask.shape
    radius_cap = max(1.0, float(min(shape) // 4))
    necrosis_radius = _clamped_positive_float_parameter(
        intent,
        "necrosis_neighbor_radius_px",
        _DEFAULT_NECROSIS_NEIGHBOR_RADIUS_PX,
        radius_cap,
    )
    interior_radius = _clamped_positive_float_parameter(
        intent,
        "tumor_interior_radius_px",
        _DEFAULT_TUMOR_INTERIOR_RADIUS_PX,
        radius_cap,
    )
    vessel_radius = _clamped_positive_float_parameter(
        intent,
        "vessel_avoidance_radius_px",
        _DEFAULT_VESSEL_AVOIDANCE_RADIUS_PX,
        radius_cap,
    )

    has_existing_necrosis = bool(np.any(existing_necrosis))
    vessel_mask = _blood_vessel_mask(mask, schema)
    has_vessel_pixels = bool(np.any(vessel_mask))
    weights = _active_candidate_weights(
        primitive_config,
        has_existing_necrosis=has_existing_necrosis,
        has_vessel_pixels=has_vessel_pixels,
    )

    score = np.zeros(shape, dtype=float)
    if "existing_necrosis_neighborhood" in weights:
        dist_to_necrosis = ndimage.distance_transform_edt(~existing_necrosis)
        score += weights["existing_necrosis_neighborhood"] * np.exp(
            -dist_to_necrosis / necrosis_radius
        )

    if "tumor_interior_far_from_outer_boundary" in weights:
        dist_inside_tumor = _edge_aware_tumor_interior_distance(
            source_tumor,
            pad_width=int(round(interior_radius)),
        )
        interior_score = np.clip(dist_inside_tumor / interior_radius, 0.0, 1.0)
        score += weights["tumor_interior_far_from_outer_boundary"] * interior_score

    if "tumor_region_far_from_blood_vessel" in weights:
        dist_to_vessel = ndimage.distance_transform_edt(~vessel_mask)
        vessel_score = np.clip(dist_to_vessel / vessel_radius, 0.0, 1.0)
        score += weights["tumor_region_far_from_blood_vessel"] * vessel_score

    noise_weight = _nonnegative_float_parameter(
        intent, "necrosis_score_noise_weight", _DEFAULT_NOISE_WEIGHT
    )
    if noise_weight > 0:
        score += noise_weight * _smooth_noise(shape, seed=intent.seed)

    score[~candidate_base] = -np.inf
    return _ScoreInfo(
        score=score,
        used_existing_necrosis_neighborhood=(
            "existing_necrosis_neighborhood" in weights
        ),
        used_blood_vessel_distance=("tumor_region_far_from_blood_vessel" in weights),
        active_weights=weights,
        radii={
            "necrosis_neighbor_radius_px": necrosis_radius,
            "tumor_interior_radius_px": interior_radius,
            "vessel_avoidance_radius_px": vessel_radius,
            "radius_cap_px": radius_cap,
        },
    )


def _active_candidate_weights(
    primitive_config: Mapping[str, Any],
    *,
    has_existing_necrosis: bool,
    has_vessel_pixels: bool,
) -> dict[str, float]:
    spatial = primitive_config.get("spatial_pattern", {})
    key = (
        "candidate_weights"
        if has_existing_necrosis
        else "candidate_weights_no_existing_necrosis"
    )
    raw = spatial.get(key, {}) if isinstance(spatial, Mapping) else {}
    if not isinstance(raw, Mapping):
        raise PrimitiveExecutionError(f"necrosis_appearance missing {key}.")

    active: dict[str, float] = {}
    for name, value in raw.items():
        if name == "existing_necrosis_neighborhood" and not has_existing_necrosis:
            continue
        if name == "tumor_region_far_from_blood_vessel" and not has_vessel_pixels:
            continue
        if not isinstance(value, (int, float)) or float(value) < 0:
            raise PrimitiveExecutionError(f"invalid necrosis candidate weight: {name}")
        if float(value) > 0:
            active[str(name)] = float(value)

    total = sum(active.values())
    if total <= 0:
        raise PrimitiveExecutionError("no_high_probability_hypoxic_candidate_region")
    return {name: value / total for name, value in active.items()}


def _blood_vessel_mask(mask: np.ndarray, schema: MaskProfileSchema) -> np.ndarray:
    if "Blood vessel" not in schema.readable_labels:
        return np.zeros(mask.shape, dtype=bool)
    return np.isin(mask, schema.resolve_fine_ids("Blood vessel"))


def _edge_aware_tumor_interior_distance(
    source_tumor: np.ndarray, *, pad_width: int
) -> np.ndarray:
    """Compute tumor-interior EDT without treating patch crop edges as tumor edges."""

    base_dist = ndimage.distance_transform_edt(source_tumor)
    if pad_width <= 0:
        return base_dist

    source = np.asarray(source_tumor, dtype=bool)
    padded = np.pad(source, pad_width, mode="constant", constant_values=False)

    inner_rows = slice(pad_width, pad_width + source.shape[0])
    inner_cols = slice(pad_width, pad_width + source.shape[1])

    padded[:pad_width, inner_cols] = source[0, :][np.newaxis, :]
    padded[pad_width + source.shape[0] :, inner_cols] = source[-1, :][np.newaxis, :]
    padded[inner_rows, :pad_width] = source[:, 0][:, np.newaxis]
    padded[inner_rows, pad_width + source.shape[1] :] = source[:, -1][:, np.newaxis]

    dist = ndimage.distance_transform_edt(padded)
    # The unpadded SciPy EDT already avoids treating out-of-array space as
    # background.  Keep that as a lower bound so false corner padding never
    # suppresses crop-edge tumor interiors.
    return np.maximum(base_dist, dist[inner_rows, inner_cols])


def _select_connected_high_score_region(
    score: np.ndarray,
    candidate_base: np.ndarray,
    *,
    target_pixels: int,
    min_component_area: int,
    max_components: int,
) -> _SelectionInfo:
    finite_scores = score[candidate_base & np.isfinite(score)]
    if finite_scores.size == 0:
        raise PrimitiveExecutionError("no_high_probability_hypoxic_candidate_region")

    first_percentile = max(
        0.0,
        100.0 * (1.0 - min(float(target_pixels) / float(finite_scores.size), 1.0)),
    )
    attempts = (first_percentile, 50.0)
    first_removed_pixels = 0
    first_pre_cleanup_pixels = 0
    best_info: _SelectionInfo | None = None
    min_acceptable_pixels = max(1, int(round(target_pixels * _MIN_SELECTED_TARGET_FRACTION)))
    for attempt_index, percentile in enumerate(attempts):
        threshold = float(np.percentile(finite_scores, percentile))
        high_score = candidate_base & np.isfinite(score) & (score >= threshold)
        selected = _select_components_by_score(
            high_score,
            score,
            target_pixels,
            max_components=max_components,
        )
        pre_cleanup_pixels = int(np.count_nonzero(selected))
        cleaned, removed_pixels = _remove_small_components(
            selected, min_component_area
        )
        cleaned, morph_info = _solidify_necrosis_region(
            cleaned,
            candidate_base,
            score,
            target_pixels,
        )
        if attempt_index == 0:
            first_pre_cleanup_pixels = pre_cleanup_pixels
            first_removed_pixels = removed_pixels
        selected_pixels = int(np.count_nonzero(cleaned))
        if selected_pixels > 0:
            selected_components = int(
                ndimage.label(cleaned, structure=_four_neighbor_structure())[1]
            )
            info = _SelectionInfo(
                change_region=cleaned,
                score_threshold=threshold,
                threshold_percentile=float(percentile),
                selected_components=selected_components,
                retry_applied=bool(attempt_index > 0),
                pre_cleanup_pixels=pre_cleanup_pixels,
                removed_small_component_pixels=removed_pixels,
                hole_fill_pixels=morph_info["hole_fill_pixels"],
                closing_added_pixels=morph_info["closing_added_pixels"],
                morphology_cleanup_applied=morph_info["morphology_cleanup_applied"],
            )
            if (
                best_info is None
                or selected_pixels > int(np.count_nonzero(best_info.change_region))
            ):
                best_info = info
            if selected_pixels >= min_acceptable_pixels or attempt_index == len(attempts) - 1:
                return info

    if best_info is not None:
        return best_info

    return _SelectionInfo(
        change_region=np.zeros_like(candidate_base, dtype=bool),
        score_threshold=float(np.percentile(finite_scores, attempts[-1])),
        threshold_percentile=float(attempts[-1]),
        selected_components=0,
        retry_applied=True,
        pre_cleanup_pixels=first_pre_cleanup_pixels,
        removed_small_component_pixels=first_removed_pixels,
        hole_fill_pixels=0,
        closing_added_pixels=0,
        morphology_cleanup_applied=False,
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
        if selected_components >= max_components:
            break
        component = labeled == component_id
        remaining = target_pixels - selected_count
        if remaining <= 0:
            break
        if area <= remaining:
            selected |= component
            selected_count += area
            selected_components += 1
            continue

        # Keep the component intact.  Area limiting happens later by peeling
        # low-score boundary pixels, which preserves a solid necrotic focus
        # better than selecting arbitrary high-score interior pixels.
        selected |= component
        selected_count += area
        selected_components += 1
        break
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


def _solidify_necrosis_region(
    selected: np.ndarray,
    candidate_base: np.ndarray,
    score: np.ndarray,
    target_pixels: int,
) -> tuple[np.ndarray, dict[str, int | bool]]:
    if not np.any(selected):
        return selected, {
            "hole_fill_pixels": 0,
            "closing_added_pixels": 0,
            "morphology_cleanup_applied": False,
        }

    before_pixels = int(np.count_nonzero(selected))
    filled = ndimage.binary_fill_holes(selected) & candidate_base
    hole_fill_pixels = int(np.count_nonzero(filled & ~selected))

    structure = _disk_structure(_DEFAULT_CLOSING_RADIUS_PX)
    closed = ndimage.binary_closing(filled, structure=structure) & candidate_base
    closing_added_pixels = int(np.count_nonzero(closed & ~filled))

    limited = _limit_region_by_boundary_distance(
        closed,
        score,
        max_pixels=target_pixels,
    )
    limited = ndimage.binary_fill_holes(limited) & candidate_base

    return limited, {
        "hole_fill_pixels": hole_fill_pixels,
        "closing_added_pixels": closing_added_pixels,
        "morphology_cleanup_applied": bool(
            hole_fill_pixels > 0 or closing_added_pixels > 0
        ),
    }


def _limit_region_by_boundary_distance(
    region: np.ndarray,
    score: np.ndarray,
    *,
    max_pixels: int,
) -> np.ndarray:
    current_pixels = int(np.count_nonzero(region))
    if current_pixels <= max_pixels:
        return region
    if max_pixels < 1:
        return np.zeros_like(region, dtype=bool)

    limited = region.copy()
    while int(np.count_nonzero(limited)) > max_pixels:
        eroded = ndimage.binary_erosion(limited, structure=_four_neighbor_structure())
        boundary = limited & ~eroded
        boundary_count = int(np.count_nonzero(boundary))
        if boundary_count == 0:
            break

        current_pixels = int(np.count_nonzero(limited))
        removable_pixels = current_pixels - max_pixels
        coords = np.argwhere(boundary)
        values = score[coords[:, 0], coords[:, 1]]
        order = np.argsort(values, kind="stable")
        remove_count = min(removable_pixels, boundary_count)
        remove = coords[order[:remove_count]]
        limited[remove[:, 0], remove[:, 1]] = False
        if remove_count < boundary_count:
            break
    return limited


def _disk_structure(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return (yy * yy + xx * xx) <= radius * radius


def _clamped_positive_float_parameter(
    intent: EditIntent, key: str, default: float, cap: float
) -> float:
    value = intent.parameters.get(key, default)
    if not isinstance(value, (int, float)) or float(value) <= 0:
        raise PrimitiveExecutionError(f"parameters.{key} must be a positive number.")
    return min(float(value), float(cap))


def _nonnegative_float_parameter(
    intent: EditIntent, key: str, default: float
) -> float:
    value = intent.parameters.get(key, default)
    if not isinstance(value, (int, float)) or float(value) < 0:
        raise PrimitiveExecutionError(f"parameters.{key} must be a non-negative number.")
    return float(value)


def _min_component_area_for_intent(intent: EditIntent) -> int:
    value = intent.parameters.get(
        "min_necrosis_component_area_px",
        _DEFAULT_MIN_COMPONENT_AREA_PX,
    )
    if not isinstance(value, int) or value < 1:
        raise PrimitiveExecutionError(
            "parameters.min_necrosis_component_area_px must be a positive integer."
        )
    return value


def _max_components_for_intent(
    intent: EditIntent, *, has_existing_necrosis: bool
) -> int:
    default = (
        _DEFAULT_MAX_COMPONENTS_WITH_EXISTING_NECROSIS
        if has_existing_necrosis
        else _DEFAULT_MAX_COMPONENTS_NO_EXISTING_NECROSIS
    )
    value = intent.parameters.get("max_necrosis_components", default)
    if not isinstance(value, int) or value < 1:
        raise PrimitiveExecutionError(
            "parameters.max_necrosis_components must be a positive integer."
        )
    return value


def _smooth_noise(shape: tuple[int, int], *, seed: int | None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(shape)
    sigma = max(3.0, min(shape) / 64.0)
    noise = ndimage.gaussian_filter(raw, sigma=sigma)
    max_abs = float(np.abs(noise).max())
    if max_abs > 0:
        noise /= max_abs
    return noise

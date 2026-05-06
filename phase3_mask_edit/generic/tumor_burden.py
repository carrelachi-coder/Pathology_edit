"""Generic tumor-burden mask primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy import ndimage

from phase3_mask_edit.core.candidates import build_candidate_mask_by_priority
from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema


_EXPAND_NOISE_OCTAVES: tuple[tuple[float, float], ...] = (
    (40.0, 0.55),
    (15.0, 0.30),
    (5.0, 0.15),
)
_SHRINK_NOISE_OCTAVES: tuple[tuple[float, float], ...] = (
    (45.0, 0.60),
    (18.0, 0.28),
    (6.0, 0.12),
)
_DEFAULT_INFLUENCE_RADIUS_PX = 45.0
_DEFAULT_ALPHA_PX = 18.0
_DEFAULT_EXPAND_BETA_MAX_PX = 80.0
_DEFAULT_SHRINK_BETA_MAX_PX = 150.0
_DEFAULT_EDGE_FADE_MARGIN_PX = 40


class PrimitiveExecutionError(ValueError):
    """Raised when a primitive cannot produce a valid mask edit."""


@dataclass(frozen=True)
class PrimitiveEditResult:
    """Output of a Phase 3 mask primitive."""

    target_mask: np.ndarray
    change_region: np.ndarray
    changed_area_fraction: float
    selected_pixels: int
    warnings: tuple[str, ...]
    ops_log: dict[str, Any]


def apply_tumor_burden_increase(
    old_mask: np.ndarray,
    schema: MaskProfileSchema,
    context: MaskEditContext,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> PrimitiveEditResult:
    """Expand Tumor into semantically valid neighboring tissue."""

    _validate_tumor_burden_increase_request(
        old_mask, schema, context, primitive_config, intent
    )
    target_fraction = _target_fraction_for_intent(primitive_config, intent)

    candidate_selection = build_candidate_mask_by_priority(
        context.normalized_mask,
        schema,
        primitive_config,
        intent,
        context=context,
        target_fraction=target_fraction,
    )
    if candidate_selection.candidate_pixels == 0:
        if _only_necrosis_available_near_tumor(context, candidate_selection.excluded_labels):
            raise PrimitiveExecutionError(
                "no_editable_non_tumor_tissue_only_necrosis_available"
            )
        raise PrimitiveExecutionError("no editable candidate tissue for tumor growth.")

    source_mask = np.isin(context.normalized_mask, schema.tumor_fine_ids)
    if not np.any(source_mask):
        raise PrimitiveExecutionError("no tumor source region for tumor growth.")
    geometry_source_mask = _filled_tumor_geometry_mask(source_mask)
    internal_holes = geometry_source_mask & ~source_mask

    change_region, spatial_info = _select_sdf_expansion_region(
        source_mask,
        candidate_selection.candidate_mask & ~internal_holes,
        target_fraction=target_fraction,
        intent=intent,
        geometry_source_mask=geometry_source_mask,
    )
    selected_pixels = int(np.count_nonzero(change_region))
    if selected_pixels == 0:
        raise PrimitiveExecutionError("no editable boundary candidate region for tumor growth.")

    target_mask = np.array(context.normalized_mask, copy=True)
    target_mask[change_region] = _nearest_tumor_fine_ids(
        context.normalized_mask,
        source_mask,
        change_region,
        schema,
    )

    changed_area_fraction = selected_pixels / target_mask.size
    ops_log = {
        "primitive": "tumor_burden_increase",
        "reference_profile": schema.reference_profile,
        "target_change_fraction": target_fraction,
        "changed_area_fraction": changed_area_fraction,
        "selected_pixels": selected_pixels,
        "candidate_labels": list(candidate_selection.included_labels),
        "excluded_labels": list(candidate_selection.excluded_labels),
        "spatial": dict(spatial_info),
    }

    return PrimitiveEditResult(
        target_mask=target_mask,
        change_region=change_region,
        changed_area_fraction=changed_area_fraction,
        selected_pixels=selected_pixels,
        warnings=candidate_selection.warnings,
        ops_log=ops_log,
    )


def apply_tumor_burden_decrease(
    old_mask: np.ndarray,
    schema: MaskProfileSchema,
    context: MaskEditContext,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> PrimitiveEditResult:
    """Shrink Tumor inward and backfill the released region."""

    _validate_tumor_burden_decrease_request(
        old_mask, schema, context, primitive_config, intent
    )
    mask = np.asarray(old_mask)
    target_fraction = _target_decrease_fraction_for_intent(primitive_config, intent)

    source_mask = np.isin(context.normalized_mask, schema.tumor_fine_ids)
    geometry_source_mask = _filled_tumor_geometry_mask(source_mask)
    tumor_pixels = int(np.count_nonzero(source_mask))
    if tumor_pixels == 0:
        raise PrimitiveExecutionError("no_tumor")

    backfill_labels, backfill_mask = _available_backfill_labels_and_mask(
        context.normalized_mask,
        schema,
        primitive_config,
        intent,
    )
    if not backfill_labels:
        raise PrimitiveExecutionError("no_valid_backfill_tissue")

    min_remaining_pixels = _min_remaining_tumor_pixels(
        primitive_config, intent, mask.size
    )
    target_pixels = _target_pixels(target_fraction, mask.size)
    max_removable_pixels = tumor_pixels - min_remaining_pixels
    if max_removable_pixels < 1 or target_pixels > max_removable_pixels:
        raise PrimitiveExecutionError("shrink_would_delete_tumor")

    protected_boundary = _protected_tumor_boundary_mask(
        context.normalized_mask,
        source_mask,
        schema,
    )
    max_removable_mask_pixels = tumor_pixels - min_remaining_pixels
    change_region, spatial_info = _select_sdf_shrink_region(
        source_mask,
        backfill_mask,
        protected_boundary,
        target_fraction=target_fraction,
        intent=intent,
        max_selected_pixels=max_removable_mask_pixels,
        geometry_source_mask=geometry_source_mask,
    )
    selected_pixels = int(np.count_nonzero(change_region))
    if selected_pixels == 0:
        raise PrimitiveExecutionError("tumor_too_small")
    if tumor_pixels - selected_pixels < min_remaining_pixels:
        raise PrimitiveExecutionError("shrink_would_delete_tumor")

    target_mask = np.array(context.normalized_mask, copy=True)
    target_mask[change_region] = _nearest_backfill_fine_ids(
        context.normalized_mask,
        backfill_mask,
        change_region,
    )

    changed_area_fraction = selected_pixels / target_mask.size
    warnings = _semantic_warnings_for_labels(backfill_labels, schema, context)
    ops_log = {
        "primitive": "tumor_burden_decrease",
        "reference_profile": schema.reference_profile,
        "target_change_fraction": target_fraction,
        "changed_area_fraction": changed_area_fraction,
        "selected_pixels": selected_pixels,
        "backfill_labels": list(backfill_labels),
        "spatial": dict(spatial_info),
    }

    return PrimitiveEditResult(
        target_mask=target_mask,
        change_region=change_region,
        changed_area_fraction=changed_area_fraction,
        selected_pixels=selected_pixels,
        warnings=warnings,
        ops_log=ops_log,
    )


def _validate_tumor_burden_increase_request(
    old_mask: np.ndarray,
    schema: MaskProfileSchema,
    context: MaskEditContext,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> None:
    mask = np.asarray(old_mask)
    if mask.ndim != 2:
        raise PrimitiveExecutionError("tumor_burden_increase requires a 2D mask.")
    if tuple(mask.shape) != context.mask_shape:
        raise PrimitiveExecutionError("old_mask shape must match MaskEditContext.")
    if schema.reference_profile != context.reference_profile:
        raise PrimitiveExecutionError(
            "schema.reference_profile must match context.reference_profile."
        )
    if intent.primitive != "tumor_burden_increase":
        raise PrimitiveExecutionError(
            "apply_tumor_burden_increase requires a tumor_burden_increase intent."
        )
    if primitive_config.get("name") != "tumor_burden_increase":
        raise PrimitiveExecutionError(
            "primitive_config must describe tumor_burden_increase."
        )


def _validate_tumor_burden_decrease_request(
    old_mask: np.ndarray,
    schema: MaskProfileSchema,
    context: MaskEditContext,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> None:
    mask = np.asarray(old_mask)
    if mask.ndim != 2:
        raise PrimitiveExecutionError("tumor_burden_decrease requires a 2D mask.")
    if tuple(mask.shape) != context.mask_shape:
        raise PrimitiveExecutionError("old_mask shape must match MaskEditContext.")
    if schema.reference_profile != context.reference_profile:
        raise PrimitiveExecutionError(
            "schema.reference_profile must match context.reference_profile."
        )
    if intent.primitive != "tumor_burden_decrease":
        raise PrimitiveExecutionError(
            "apply_tumor_burden_decrease requires a tumor_burden_decrease intent."
        )
    if primitive_config.get("name") != "tumor_burden_decrease":
        raise PrimitiveExecutionError(
            "primitive_config must describe tumor_burden_decrease."
        )


def _target_fraction_for_intent(
    primitive_config: Mapping[str, Any], intent: EditIntent
) -> float:
    if intent.target_change_fraction is not None:
        return intent.target_change_fraction

    intervals = (
        primitive_config.get("parameter_ranges", {})
        .get("target_area_delta_fraction", {})
    )
    interval = intervals.get(intent.strength)
    if not isinstance(interval, list) or len(interval) != 2:
        raise PrimitiveExecutionError(
            f"tumor_burden_increase does not define strength {intent.strength}."
        )

    lower, upper = float(interval[0]), float(interval[1])
    return (lower + upper) / 2


def _target_decrease_fraction_for_intent(
    primitive_config: Mapping[str, Any], intent: EditIntent
) -> float:
    if intent.target_change_fraction is not None:
        return intent.target_change_fraction

    intervals = (
        primitive_config.get("parameter_ranges", {})
        .get("target_area_decrease_fraction", {})
    )
    interval = intervals.get(intent.strength)
    if not isinstance(interval, list) or len(interval) != 2:
        raise PrimitiveExecutionError(
            f"tumor_burden_decrease does not define strength {intent.strength}."
        )

    lower, upper = float(interval[0]), float(interval[1])
    return (lower + upper) / 2


def _select_sdf_expansion_region(
    source_mask: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    target_fraction: float,
    intent: EditIntent,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    target_count = _target_pixels(target_fraction, source_mask.size)
    if target_count == 0:
        return np.zeros_like(source_mask, dtype=bool), _sdf_spatial_info(
            method="sdf_noise_expansion",
            target_count=target_count,
            selected=np.zeros_like(source_mask, dtype=bool),
            beta=0.0,
            alpha=0.0,
            candidate_mask=candidate_mask,
        )

    influence_radius = _positive_float_parameter(
        intent, "sdf_influence_radius_px", _DEFAULT_INFLUENCE_RADIUS_PX
    )
    alpha = _positive_float_parameter(intent, "sdf_alpha_px", _DEFAULT_ALPHA_PX)
    beta_max = _positive_float_parameter(
        intent, "sdf_beta_max_px", _DEFAULT_EXPAND_BETA_MAX_PX
    )

    sdf = _compute_sdf(source_mask)
    weight = _compute_boundary_weight(source_mask, influence_radius)
    noise = _generate_smooth_noise(
        source_mask.shape,
        seed=intent.seed,
        octaves=_EXPAND_NOISE_OCTAVES,
    )

    def region_for_beta(beta: float) -> np.ndarray:
        shifted = sdf + weight * (alpha * noise + beta)
        return shifted > 0

    beta, selected = _calibrate_sdf_beta(
        region_for_beta=region_for_beta,
        selectable_mask=candidate_mask & ~source_mask,
        target_count=target_count,
        beta_min=0.0,
        beta_max=beta_max,
    )

    selected = _keep_growth_touching_source(selected, source_mask)
    info = _sdf_spatial_info(
        method="sdf_noise_expansion",
        target_count=target_count,
        selected=selected,
        beta=beta,
        alpha=alpha,
        candidate_mask=candidate_mask,
    )
    info["influence_radius_px"] = float(influence_radius)
    return selected, info


def _select_sdf_shrink_region(
    source_mask: np.ndarray,
    backfill_mask: np.ndarray,
    protected_mask: np.ndarray,
    *,
    target_fraction: float,
    intent: EditIntent,
    max_selected_pixels: int | None = None,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    target_count = _target_pixels(target_fraction, source_mask.size)
    shrinkable_mask = source_mask & ~protected_mask
    if target_count == 0 or not np.any(shrinkable_mask):
        selected = np.zeros_like(source_mask, dtype=bool)
        return selected, _sdf_spatial_info(
            method="sdf_noise_shrink",
            target_count=target_count,
            selected=selected,
            beta=0.0,
            alpha=0.0,
            candidate_mask=shrinkable_mask,
        )

    influence_radius = _positive_float_parameter(
        intent, "sdf_influence_radius_px", _DEFAULT_INFLUENCE_RADIUS_PX
    )
    alpha = _positive_float_parameter(intent, "sdf_alpha_px", _DEFAULT_ALPHA_PX)
    beta_max = _positive_float_parameter(
        intent, "sdf_beta_max_px", _DEFAULT_SHRINK_BETA_MAX_PX
    )
    edge_margin = _nonnegative_int_parameter(
        intent, "edge_fade_margin_px", _DEFAULT_EDGE_FADE_MARGIN_PX
    )
    smoothing_enabled = _bool_parameter(intent, "smooth_boundary", True)
    smooth_sigma = _nonnegative_float_parameter(
        intent,
        "sdf_smooth_sigma_px",
        4.0 if smoothing_enabled else 0.0,
    )
    if not smoothing_enabled:
        smooth_sigma = 0.0

    sdf = _compute_sdf(source_mask)
    weight = _compute_boundary_weight(source_mask, influence_radius)
    effective_edge_margin = _effective_edge_fade_margin(source_mask.shape, edge_margin)
    if effective_edge_margin > 0:
        weight *= _compute_edge_fade_mask(source_mask.shape, effective_edge_margin)
    weight[protected_mask] = 0.0

    noise = _generate_smooth_noise(
        source_mask.shape,
        seed=intent.seed,
        octaves=_SHRINK_NOISE_OCTAVES,
    )

    def region_for_beta(beta: float) -> np.ndarray:
        shifted = sdf - weight * (alpha * noise + beta)
        if smooth_sigma > 0:
            shifted = ndimage.gaussian_filter(shifted, sigma=smooth_sigma)
        remaining = (shifted > 0) & source_mask
        remaining |= source_mask & (weight <= 0)
        return source_mask & ~remaining

    beta, selected = _calibrate_sdf_beta(
        region_for_beta=region_for_beta,
        selectable_mask=shrinkable_mask,
        target_count=target_count,
        beta_min=0.0,
        beta_max=beta_max,
        prefer_under_target=True,
    )

    selected &= shrinkable_mask
    selected = _keep_released_region_backfill_reachable(selected, backfill_mask)
    if max_selected_pixels is not None:
        selected = _limit_selected_by_boundary_score(
            selected,
            source_mask,
            backfill_mask,
            max_selected_pixels,
        )
    info = _sdf_spatial_info(
        method="sdf_noise_shrink",
        target_count=target_count,
        selected=selected,
        beta=beta,
        alpha=alpha,
        candidate_mask=shrinkable_mask,
    )
    info["influence_radius_px"] = float(influence_radius)
    info["edge_fade_margin_px"] = int(effective_edge_margin)
    info["requested_edge_fade_margin_px"] = int(edge_margin)
    info["sdf_smooth_sigma_px"] = float(smooth_sigma)
    info["smoothing_applied"] = bool(smooth_sigma > 0)
    info["smoothing_method"] = "sdf_gaussian" if smooth_sigma > 0 else "none"
    info["smoothing_radius"] = int(round(smooth_sigma * 2))
    info["protected_pixels"] = int(np.count_nonzero(protected_mask))
    return selected, info


def _compute_sdf(source_mask: np.ndarray) -> np.ndarray:
    source = np.asarray(source_mask, dtype=bool)
    if not np.any(source):
        return -ndimage.distance_transform_edt(np.ones_like(source, dtype=bool))
    if np.all(source):
        return ndimage.distance_transform_edt(source)
    return ndimage.distance_transform_edt(source) - ndimage.distance_transform_edt(~source)


def _compute_boundary_weight(source_mask: np.ndarray, radius: float) -> np.ndarray:
    source = np.asarray(source_mask, dtype=bool)
    if not np.any(source) or np.all(source):
        return np.zeros(source.shape, dtype=float)

    main_source = _largest_component(source)
    dist_in = ndimage.distance_transform_edt(main_source)
    dist_out = ndimage.distance_transform_edt(~main_source)
    return np.clip(1.0 - np.minimum(dist_in, dist_out) / float(radius), 0.0, 1.0)


def _largest_component(source_mask: np.ndarray) -> np.ndarray:
    labeled, count = ndimage.label(source_mask, structure=_four_neighbor_structure())
    if count <= 1:
        return source_mask.astype(bool, copy=True)

    areas = ndimage.sum(source_mask, labeled, range(1, count + 1))
    largest_label = int(np.argmax(areas)) + 1
    return labeled == largest_label


def _compute_edge_fade_mask(shape: tuple[int, int], margin: int) -> np.ndarray:
    if margin <= 0:
        return np.ones(shape, dtype=float)

    rows = np.arange(shape[0], dtype=float)
    cols = np.arange(shape[1], dtype=float)
    fade = np.ones(shape, dtype=float)
    fade *= np.clip(rows / margin, 0.0, 1.0)[:, np.newaxis]
    fade *= np.clip((shape[0] - 1 - rows) / margin, 0.0, 1.0)[:, np.newaxis]
    fade *= np.clip(cols / margin, 0.0, 1.0)[np.newaxis, :]
    fade *= np.clip((shape[1] - 1 - cols) / margin, 0.0, 1.0)[np.newaxis, :]
    return fade


def _effective_edge_fade_margin(shape: tuple[int, int], requested_margin: int) -> int:
    if requested_margin <= 0:
        return 0
    small_patch_cap = max(1, min(shape) // 6)
    return min(int(requested_margin), small_patch_cap)


def _generate_smooth_noise(
    shape: tuple[int, int],
    *,
    seed: int | None,
    octaves: tuple[tuple[float, float], ...],
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = np.zeros(shape, dtype=float)
    for sigma, amplitude in octaves:
        raw = rng.standard_normal(shape)
        smoothed = ndimage.gaussian_filter(raw, sigma=sigma)
        max_abs = float(np.abs(smoothed).max())
        if max_abs > 0:
            smoothed /= max_abs
        noise += amplitude * smoothed

    max_abs = float(np.abs(noise).max())
    if max_abs > 0:
        noise /= max_abs
    return noise


def _calibrate_sdf_beta(
    *,
    region_for_beta,
    selectable_mask: np.ndarray,
    target_count: int,
    beta_min: float,
    beta_max: float,
    prefer_under_target: bool = False,
) -> tuple[float, np.ndarray]:
    selectable = np.asarray(selectable_mask, dtype=bool)
    best_beta = float(beta_min)
    best_region = np.zeros_like(selectable, dtype=bool)
    best_error: int | None = None

    for _ in range(32):
        beta = (beta_min + beta_max) / 2.0
        region = region_for_beta(beta) & selectable
        count = int(np.count_nonzero(region))
        error = abs(count - target_count)
        is_better_tie = (
            best_error is not None
            and error == best_error
            and prefer_under_target
            and count <= target_count
            and int(np.count_nonzero(best_region)) > target_count
        )
        if best_error is None or error < best_error or is_better_tie:
            best_error = error
            best_beta = float(beta)
            best_region = region
        if count < target_count:
            beta_min = beta
        else:
            beta_max = beta

    return best_beta, best_region


def _keep_growth_touching_source(
    change_region: np.ndarray, source_mask: np.ndarray
) -> np.ndarray:
    labeled, count = ndimage.label(change_region, structure=_four_neighbor_structure())
    if count == 0:
        return np.zeros_like(change_region, dtype=bool)
    touching = ndimage.binary_dilation(source_mask, structure=_four_neighbor_structure())
    result = np.zeros_like(change_region, dtype=bool)
    for component_id in range(1, count + 1):
        component = labeled == component_id
        if np.any(component & touching):
            result |= component
    return result


def _keep_released_region_backfill_reachable(
    change_region: np.ndarray, backfill_mask: np.ndarray
) -> np.ndarray:
    labeled, count = ndimage.label(change_region, structure=_four_neighbor_structure())
    if count == 0:
        return np.zeros_like(change_region, dtype=bool)
    touching = ndimage.binary_dilation(backfill_mask, structure=_four_neighbor_structure())
    result = np.zeros_like(change_region, dtype=bool)
    for component_id in range(1, count + 1):
        component = labeled == component_id
        if np.any(component & touching):
            result |= component
    return result


def _limit_selected_by_boundary_score(
    selected: np.ndarray,
    source_mask: np.ndarray,
    backfill_mask: np.ndarray,
    max_pixels: int,
) -> np.ndarray:
    selected_count = int(np.count_nonzero(selected))
    if max_pixels < 0:
        max_pixels = 0
    if selected_count <= max_pixels:
        return selected
    if max_pixels == 0:
        return np.zeros_like(selected, dtype=bool)

    selected_indices = np.argwhere(selected)
    distance_to_backfill = ndimage.distance_transform_edt(~backfill_mask)
    distance_to_tumor_core = ndimage.distance_transform_edt(source_mask)
    score = distance_to_backfill + 0.01 * distance_to_tumor_core
    values = score[selected_indices[:, 0], selected_indices[:, 1]]
    order = np.argsort(values, kind="stable")
    chosen = selected_indices[order[:max_pixels]]
    limited = np.zeros_like(selected, dtype=bool)
    limited[chosen[:, 0], chosen[:, 1]] = True
    return limited


def _sdf_spatial_info(
    *,
    method: str,
    target_count: int,
    selected: np.ndarray,
    beta: float,
    alpha: float,
    candidate_mask: np.ndarray,
) -> dict[str, float | int | bool | str]:
    selected_pixels = int(np.count_nonzero(selected))
    candidate_pixels = int(np.count_nonzero(candidate_mask))
    return {
        "method": method,
        "target_pixels": int(target_count),
        "selected_pixels": selected_pixels,
        "actual_fraction": selected_pixels / selected.size,
        "target_area_shortfall": selected_pixels < target_count,
        "candidate_pixels": candidate_pixels,
        "candidate_shortfall": candidate_pixels < target_count,
        "alpha_px": float(alpha),
        "beta_px": float(beta),
    }


def _positive_float_parameter(
    intent: EditIntent, key: str, default: float
) -> float:
    value = intent.parameters.get(key, default)
    if not isinstance(value, (int, float)) or float(value) <= 0:
        raise PrimitiveExecutionError(f"parameters.{key} must be a positive number.")
    return float(value)


def _nonnegative_float_parameter(
    intent: EditIntent, key: str, default: float
) -> float:
    value = intent.parameters.get(key, default)
    if not isinstance(value, (int, float)) or float(value) < 0:
        raise PrimitiveExecutionError(f"parameters.{key} must be a non-negative number.")
    return float(value)


def _nonnegative_int_parameter(
    intent: EditIntent, key: str, default: int
) -> int:
    value = intent.parameters.get(key, default)
    if not isinstance(value, int) or value < 0:
        raise PrimitiveExecutionError(f"parameters.{key} must be a non-negative integer.")
    return value


def _bool_parameter(intent: EditIntent, key: str, default: bool) -> bool:
    value = intent.parameters.get(key, default)
    if not isinstance(value, bool):
        raise PrimitiveExecutionError(f"parameters.{key} must be boolean.")
    return value


def _only_necrosis_available_near_tumor(
    context: MaskEditContext, excluded_labels: tuple[str, ...]
) -> bool:
    tumor_neighbors = context.adjacency.get("Tumor", frozenset())
    if "Necrosis" not in tumor_neighbors:
        return False

    non_absent_exclusions = tuple(
        reason
        for reason in excluded_labels
        if not reason.startswith(("label_absent_in_mask:", "label_not_readable:"))
    )
    return not non_absent_exclusions


def _available_backfill_labels_and_mask(
    mask: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> tuple[tuple[str, ...], np.ndarray]:
    priority = _backfill_priority_labels(primitive_config, intent)
    forbidden = set(intent.forbidden_labels) | set(intent.preserve_labels)
    selected_mask = np.zeros(mask.shape, dtype=bool)
    labels: list[str] = []

    for label in priority:
        if label in forbidden or label not in schema.readable_labels:
            continue
        label_mask = np.isin(mask, schema.resolve_fine_ids(label))
        if not np.any(label_mask):
            continue
        selected_mask |= label_mask
        labels.append(label)

    return tuple(labels), selected_mask


def _backfill_priority_labels(
    primitive_config: Mapping[str, Any], intent: EditIntent
) -> tuple[str, ...]:
    if intent.target_label is not None:
        return (intent.target_label,)

    operation = primitive_config.get("mask_operation", {})
    priority = operation.get("backfill_priority", ())
    if not isinstance(priority, list):
        return ()
    return tuple(label for label in priority if isinstance(label, str))


def _min_remaining_tumor_pixels(
    primitive_config: Mapping[str, Any], intent: EditIntent, total_pixels: int
) -> int:
    ranges = primitive_config.get("parameter_ranges", {})
    min_remaining = ranges.get("min_remaining_tumor_fraction", {})
    if not isinstance(min_remaining, Mapping):
        raise PrimitiveExecutionError(
            "tumor_burden_decrease missing min_remaining_tumor_fraction."
        )

    key = "xlarge_deid" if intent.strength == "xlarge_deid" else "default"
    value = min_remaining.get(key, min_remaining.get("default"))
    if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
        raise PrimitiveExecutionError("invalid min_remaining_tumor_fraction.")
    return int(np.ceil(float(value) * total_pixels))


def _protected_tumor_boundary_mask(
    mask: np.ndarray,
    source_mask: np.ndarray,
    schema: MaskProfileSchema,
) -> np.ndarray:
    protected_context = _external_background_mask(mask, schema.skip_fine_ids)
    if "Necrosis" in schema.readable_labels:
        protected_context |= np.isin(mask, schema.resolve_fine_ids("Necrosis"))

    return source_mask & ndimage.binary_dilation(
        protected_context,
        structure=_four_neighbor_structure(),
    )


def _external_background_mask(mask: np.ndarray, skip_fine_ids: frozenset[int]) -> np.ndarray:
    background = np.isin(mask, tuple(skip_fine_ids))
    if not np.any(background):
        return np.zeros(mask.shape, dtype=bool)

    labeled, component_count = ndimage.label(background, structure=_four_neighbor_structure())
    if component_count == 0:
        return np.zeros(mask.shape, dtype=bool)

    border_labels = set(int(label) for label in labeled[0, :] if label)
    border_labels.update(int(label) for label in labeled[-1, :] if label)
    border_labels.update(int(label) for label in labeled[:, 0] if label)
    border_labels.update(int(label) for label in labeled[:, -1] if label)

    if not border_labels:
        return np.zeros(mask.shape, dtype=bool)
    return np.isin(labeled, tuple(border_labels))


def _four_neighbor_structure() -> np.ndarray:
    return np.array(
        [
            [False, True, False],
            [True, True, True],
            [False, True, False],
        ],
        dtype=bool,
    )


def _target_pixels(target_fraction: float, total_pixels: int) -> int:
    if not isinstance(target_fraction, (int, float)):
        raise PrimitiveExecutionError("target_change_fraction must be numeric.")
    target_fraction = float(target_fraction)
    if not 0.0 <= target_fraction <= 1.0:
        raise PrimitiveExecutionError("target_change_fraction must be in [0, 1].")
    if target_fraction == 0:
        return 0
    return max(1, int(round(target_fraction * total_pixels)))


def _nearest_backfill_fine_ids(
    mask: np.ndarray,
    backfill_mask: np.ndarray,
    change_region: np.ndarray,
) -> np.ndarray:
    _, nearest_indices = ndimage.distance_transform_edt(
        ~backfill_mask,
        return_indices=True,
    )
    row_indices, col_indices = nearest_indices
    nearest_ids = mask[row_indices, col_indices]
    return nearest_ids[change_region]


def _semantic_warnings_for_labels(
    labels: tuple[str, ...],
    schema: MaskProfileSchema,
    context: MaskEditContext,
) -> tuple[str, ...]:
    semantic_warnings = dict(schema.semantic_warnings)
    semantic_warnings.update(context.semantic_warnings)
    warnings = [
        f"semantic_warning:{label}"
        for label in labels
        if label in semantic_warnings
    ]
    warnings.extend(f"context_risk:{flag}" for flag in context.risk_flags)
    return tuple(dict.fromkeys(warnings))


def _nearest_tumor_fine_ids(
    mask: np.ndarray,
    source_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
) -> np.ndarray | int:
    if len(schema.tumor_fine_ids) == 1:
        return int(schema.tumor_fine_ids[0])

    _, nearest_indices = ndimage.distance_transform_edt(
        ~source_mask,
        return_indices=True,
    )
    row_indices, col_indices = nearest_indices
    nearest_ids = mask[row_indices, col_indices]
    output = np.full(mask.shape, int(schema.tumor_fine_ids[0]), dtype=mask.dtype)
    output[change_region] = nearest_ids[change_region]
    return output[change_region]

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
from phase3_mask_edit.core.morphology import select_boundary_band_by_fraction


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
    mask = np.asarray(old_mask)
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

    max_radius = _max_radius_for_intent(intent, mask.shape)
    change_region, spatial_info = select_boundary_band_by_fraction(
        source_mask,
        candidate_selection.candidate_mask,
        target_fraction=target_fraction,
        min_radius=1,
        max_radius=max_radius,
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

    max_radius = _max_radius_for_intent(intent, mask.shape)
    protected_boundary = _protected_tumor_boundary_mask(
        context.normalized_mask,
        source_mask,
        schema,
    )
    change_region, spatial_info = _select_inward_boundary_band_by_fraction(
        source_mask,
        backfill_mask,
        protected_boundary,
        target_fraction=target_fraction,
        min_radius=1,
        max_radius=max_radius,
    )
    change_region, smoothing_info = _maybe_smooth_decrease_region(
        change_region,
        source_mask,
        backfill_mask,
        protected_boundary,
        intent,
    )
    spatial_info.update(smoothing_info)
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


def _max_radius_for_intent(intent: EditIntent, mask_shape: tuple[int, int]) -> int:
    value = intent.parameters.get("max_radius", max(mask_shape))
    if not isinstance(value, int) or value < 1:
        raise PrimitiveExecutionError("parameters.max_radius must be a positive integer.")
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


def _select_inward_boundary_band_by_fraction(
    source_mask: np.ndarray,
    backfill_mask: np.ndarray,
    protected_mask: np.ndarray,
    *,
    target_fraction: float,
    min_radius: int,
    max_radius: int,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    target_count = _target_pixels(target_fraction, source_mask.size)
    if max_radius < min_radius:
        raise PrimitiveExecutionError("max_radius must be >= min_radius.")

    best_region = np.zeros_like(source_mask, dtype=bool)
    best_radius = min_radius
    best_error: int | None = None
    reachable = np.asarray(backfill_mask, dtype=bool).copy()
    structure = _four_neighbor_structure()

    for radius in range(min_radius, max_radius + 1):
        grown = ndimage.binary_dilation(reachable, structure=structure)
        reachable |= grown & source_mask & ~protected_mask
        region = reachable & source_mask & ~protected_mask
        selected_pixels = int(np.count_nonzero(region))
        error = abs(selected_pixels - target_count)
        if best_error is None or error < best_error:
            best_error = error
            best_region = region
            best_radius = radius
        if error == 0:
            break

    selected_pixels = int(np.count_nonzero(best_region))
    return best_region, {
        "radius": best_radius,
        "target_pixels": target_count,
        "selected_pixels": selected_pixels,
        "actual_fraction": selected_pixels / source_mask.size,
        "target_area_shortfall": selected_pixels < target_count,
    }


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


def _maybe_smooth_decrease_region(
    change_region: np.ndarray,
    source_mask: np.ndarray,
    backfill_mask: np.ndarray,
    protected_mask: np.ndarray,
    intent: EditIntent,
) -> tuple[np.ndarray, dict[str, int | bool | str]]:
    enabled = intent.parameters.get("smooth_boundary", True)
    if not isinstance(enabled, bool):
        raise PrimitiveExecutionError("parameters.smooth_boundary must be boolean.")

    radius = intent.parameters.get("smooth_radius", 6)
    if not isinstance(radius, int) or radius < 0:
        raise PrimitiveExecutionError("parameters.smooth_radius must be a non-negative integer.")

    if not enabled or radius == 0 or not np.any(change_region):
        return change_region, {
            "smoothing_applied": False,
            "smoothing_method": "none",
            "smoothing_radius": 0,
        }

    shrinkable_mask = source_mask & ~protected_mask
    desired_pixels = int(np.count_nonzero(change_region))
    sigma = max(1.0, radius / 2)
    blurred_change = ndimage.gaussian_filter(change_region.astype(float), sigma=sigma)
    distance_to_backfill = ndimage.distance_transform_edt(~backfill_mask)
    score = blurred_change - 0.015 * distance_to_backfill

    corrected = _select_top_scoring_region(
        score,
        shrinkable_mask,
        desired_pixels,
    )

    return corrected, {
        "smoothing_applied": True,
        "smoothing_method": "gaussian_threshold",
        "smoothing_radius": radius,
    }


def _select_top_scoring_region(
    score: np.ndarray,
    candidate_mask: np.ndarray,
    desired_pixels: int,
) -> np.ndarray:
    selected = np.zeros(candidate_mask.shape, dtype=bool)
    candidate_indices = np.argwhere(candidate_mask)
    if candidate_indices.size == 0 or desired_pixels <= 0:
        return selected

    desired_pixels = min(desired_pixels, len(candidate_indices))
    order = np.argsort(
        -score[candidate_indices[:, 0], candidate_indices[:, 1]],
        kind="stable",
    )
    chosen = candidate_indices[order[:desired_pixels]]
    selected[chosen[:, 0], chosen[:, 1]] = True
    return selected


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

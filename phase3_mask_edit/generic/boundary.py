"""Generic tumor-boundary remodeling primitives."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy import ndimage

from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.generic.tumor_burden import (
    PrimitiveEditResult,
    PrimitiveExecutionError,
    _available_backfill_labels_and_mask,
    _four_neighbor_structure,
    _nearest_backfill_fine_ids,
    _nearest_tumor_fine_ids,
    _semantic_warnings_for_labels,
    _target_pixels,
)


# ── boundary pushing remodel ───────────────────────────────────────────

def apply_boundary_pushing_remodel(
    old_mask: np.ndarray,
    schema: MaskProfileSchema,
    context: MaskEditContext,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> PrimitiveEditResult:
    """Convert a locally irregular Tumor boundary toward a pushing border."""

    _validate_boundary_pushing_request(old_mask, schema, context, primitive_config, intent)
    mask = np.asarray(old_mask)
    source_mask = np.isin(context.normalized_mask, schema.tumor_fine_ids)
    if not np.any(source_mask):
        raise PrimitiveExecutionError("no_tumor")

    backfill_labels, backfill_mask = _available_backfill_labels_and_mask(
        context.normalized_mask,
        schema,
        primitive_config,
        intent,
    )
    if not backfill_labels:
        raise PrimitiveExecutionError("no_valid_backfill_tissue")

    target_fraction = _target_changed_fraction_for_intent(primitive_config, intent)
    target_pixels = _target_pixels(target_fraction, mask.size)
    smooth_radius = _smooth_radius_for_intent(primitive_config, intent)
    max_abs_delta_fraction = _max_abs_tumor_delta_fraction(
        primitive_config,
        intent,
    )
    min_component_area = _min_component_area_for_intent(primitive_config, intent)
    max_abs_delta_pixels = int(round(max_abs_delta_fraction * mask.size))

    raw_added, raw_removed, score = _pushing_boundary_candidates(
        context.normalized_mask,
        source_mask,
        backfill_mask,
        schema,
        smooth_radius=smooth_radius,
    )
    added, removed, selection_info = _select_remodel_components(
        raw_added,
        raw_removed,
        score,
        target_pixels=target_pixels,
        max_abs_delta_pixels=max_abs_delta_pixels,
        min_component_area=min_component_area,
    )
    change_region = added | removed
    selected_pixels = int(np.count_nonzero(change_region))
    if selected_pixels == 0:
        raise PrimitiveExecutionError("tumor_already_pushing_or_smooth")

    target_mask = np.array(context.normalized_mask, copy=True)
    target_mask[added] = _nearest_tumor_fine_ids(
        context.normalized_mask,
        source_mask,
        added,
        schema,
    )
    target_mask[removed] = _nearest_backfill_fine_ids(
        context.normalized_mask,
        backfill_mask,
        removed,
    )

    tumor_area_delta_pixels = int(np.count_nonzero(added)) - int(np.count_nonzero(removed))
    changed_area_fraction = selected_pixels / target_mask.size
    warnings = _semantic_warnings_for_labels(backfill_labels, schema, context)
    ops_log = {
        "primitive": "boundary_pushing_remodel",
        "reference_profile": schema.reference_profile,
        "target_change_fraction": target_fraction,
        "changed_area_fraction": changed_area_fraction,
        "selected_pixels": selected_pixels,
        "added_tumor_pixels": int(np.count_nonzero(added)),
        "removed_tumor_pixels": int(np.count_nonzero(removed)),
        "tumor_area_delta_pixels": tumor_area_delta_pixels,
        "backfill_labels": list(backfill_labels),
        "spatial": {
            "smooth_radius": smooth_radius,
            "target_pixels": target_pixels,
            "min_component_area_px": min_component_area,
            "max_abs_tumor_delta_fraction": max_abs_delta_fraction,
            "max_abs_tumor_delta_pixels": max_abs_delta_pixels,
            **selection_info,
        },
    }

    return PrimitiveEditResult(
        target_mask=target_mask,
        change_region=change_region,
        changed_area_fraction=changed_area_fraction,
        selected_pixels=selected_pixels,
        warnings=warnings,
        ops_log=ops_log,
    )


def _validate_boundary_pushing_request(
    old_mask: np.ndarray,
    schema: MaskProfileSchema,
    context: MaskEditContext,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> None:
    mask = np.asarray(old_mask)
    if mask.ndim != 2:
        raise PrimitiveExecutionError("boundary_pushing_remodel requires a 2D mask.")
    if tuple(mask.shape) != context.mask_shape:
        raise PrimitiveExecutionError("old_mask shape must match MaskEditContext.")
    if schema.reference_profile != context.reference_profile:
        raise PrimitiveExecutionError(
            "schema.reference_profile must match context.reference_profile."
        )
    if intent.primitive != "boundary_pushing_remodel":
        raise PrimitiveExecutionError(
            "apply_boundary_pushing_remodel requires a boundary_pushing_remodel intent."
        )
    if primitive_config.get("name") != "boundary_pushing_remodel":
        raise PrimitiveExecutionError(
            "primitive_config must describe boundary_pushing_remodel."
        )


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
            f"boundary_pushing_remodel does not define strength {intent.strength}."
        )

    lower, upper = float(interval[0]), float(interval[1])
    return (lower + upper) / 2


def _smooth_radius_for_intent(
    primitive_config: Mapping[str, Any], intent: EditIntent
) -> int:
    value = intent.parameters.get(
        "smooth_radius",
        primitive_config.get("parameter_ranges", {}).get("default_smooth_radius_px", 18),
    )
    if not isinstance(value, int) or value < 1:
        raise PrimitiveExecutionError("parameters.smooth_radius must be a positive integer.")
    return value


def _min_component_area_for_intent(
    primitive_config: Mapping[str, Any], intent: EditIntent
) -> int:
    value = intent.parameters.get(
        "min_component_area_px",
        primitive_config.get("parameter_ranges", {}).get("min_component_area_px", 80),
    )
    if not isinstance(value, int) or value < 1:
        raise PrimitiveExecutionError(
            "parameters.min_component_area_px must be a positive integer."
        )
    return value


def _max_abs_tumor_delta_fraction(
    primitive_config: Mapping[str, Any], intent: EditIntent
) -> float:
    value = intent.parameters.get(
        "max_abs_tumor_area_delta_fraction",
        primitive_config.get("parameter_ranges", {}).get(
            "max_abs_tumor_area_delta_fraction",
            0.02,
        ),
    )
    if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
        raise PrimitiveExecutionError(
            "max_abs_tumor_area_delta_fraction must be numeric in [0, 1]."
        )
    return float(value)


def _pushing_boundary_candidates(
    mask: np.ndarray,
    source_mask: np.ndarray,
    backfill_mask: np.ndarray,
    schema: MaskProfileSchema,
    *,
    smooth_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma = float(smooth_radius)
    blurred = ndimage.gaussian_filter(source_mask.astype(float), sigma=sigma)
    smoothed = blurred >= 0.5

    protected_context = np.isin(mask, tuple(schema.skip_fine_ids))
    if "Necrosis" in schema.readable_labels:
        protected_context |= np.isin(mask, schema.resolve_fine_ids("Necrosis"))
    protected_boundary = source_mask & ndimage.binary_dilation(
        protected_context,
        structure=_four_neighbor_structure(),
    )

    raw_added = smoothed & ~source_mask & backfill_mask
    raw_removed = source_mask & ~smoothed & ~protected_boundary
    raw_removed = _keep_components_touching_context(raw_removed, backfill_mask)

    score = np.zeros(mask.shape, dtype=float)
    score[raw_added] = blurred[raw_added]
    score[raw_removed] = 1.0 - blurred[raw_removed]
    return raw_added, raw_removed, score


def _keep_components_touching_context(
    components_mask: np.ndarray, context_mask: np.ndarray
) -> np.ndarray:
    labeled, count = ndimage.label(components_mask, structure=_four_neighbor_structure())
    if count == 0:
        return np.zeros_like(components_mask, dtype=bool)

    touching_context = ndimage.binary_dilation(
        context_mask,
        structure=_four_neighbor_structure(),
    )
    kept = np.zeros_like(components_mask, dtype=bool)
    for component_id in range(1, count + 1):
        component = labeled == component_id
        if np.any(component & touching_context):
            kept |= component
    return kept


def _select_remodel_components(
    raw_added: np.ndarray,
    raw_removed: np.ndarray,
    score: np.ndarray,
    *,
    target_pixels: int,
    max_abs_delta_pixels: int,
    min_component_area: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | bool]]:
    raw_change = raw_added | raw_removed
    if not np.any(raw_change):
        return (
            np.zeros_like(raw_added, dtype=bool),
            np.zeros_like(raw_removed, dtype=bool),
            {"raw_candidate_pixels": 0, "selected_components": 0},
        )

    labeled, count = ndimage.label(raw_change, structure=_four_neighbor_structure())
    components: list[tuple[float, int, int, int, int]] = []
    for component_id in range(1, count + 1):
        component = labeled == component_id
        area = int(np.count_nonzero(component))
        if area < min_component_area:
            continue
        added_count = int(np.count_nonzero(component & raw_added))
        removed_count = int(np.count_nonzero(component & raw_removed))
        delta = added_count - removed_count
        mean_score = float(score[component].mean()) if area else 0.0
        components.append((mean_score, area, delta, added_count, component_id))

    if not components and min_component_area > 1:
        return _select_remodel_components(
            raw_added,
            raw_removed,
            score,
            target_pixels=target_pixels,
            max_abs_delta_pixels=max_abs_delta_pixels,
            min_component_area=1,
        )

    components.sort(key=lambda item: (-item[0], -item[1]))
    selected = np.zeros_like(raw_change, dtype=bool)
    selected_pixels = 0
    selected_delta = 0
    selected_components = 0

    for _, area, delta, _, component_id in components:
        if selected_pixels > 0 and selected_pixels + area > target_pixels:
            continue
        if abs(selected_delta + delta) > max_abs_delta_pixels:
            continue
        component = labeled == component_id
        selected |= component
        selected_pixels += area
        selected_delta += delta
        selected_components += 1
        if selected_pixels >= target_pixels:
            break

    selected_added = selected & raw_added
    selected_removed = selected & raw_removed
    return (
        selected_added,
        selected_removed,
        {
            "raw_candidate_pixels": int(np.count_nonzero(raw_change)),
            "selected_components": selected_components,
            "target_area_shortfall": selected_pixels < target_pixels,
            "tumor_area_delta_within_limit": abs(selected_delta) <= max_abs_delta_pixels,
        },
    )
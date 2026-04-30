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

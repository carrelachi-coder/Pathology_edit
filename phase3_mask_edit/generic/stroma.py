"""Generic coarse stromal expansion primitive."""

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
)


def apply_stroma_increase(
    old_mask: np.ndarray,
    schema: MaskProfileSchema,
    context: MaskEditContext,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> PrimitiveEditResult:
    """Expand existing stroma into adjacent legal non-stromal tissue."""

    mask = np.asarray(old_mask)
    if mask.ndim != 2 or tuple(mask.shape) != context.mask_shape:
        raise PrimitiveExecutionError("stroma_increase requires one matching 2D mask.")
    if intent.primitive != "stroma_increase":
        raise PrimitiveExecutionError("apply_stroma_increase requires stroma_increase.")
    if primitive_config.get("name") != "stroma_increase":
        raise PrimitiveExecutionError("primitive_config must describe stroma_increase.")
    if "Stroma" not in schema.readable_labels:
        raise PrimitiveExecutionError("no_stroma_label")

    normalized = context.normalized_mask
    stroma_ids = schema.resolve_fine_ids("Stroma")
    stroma = np.isin(normalized, stroma_ids)
    if not np.any(stroma):
        raise PrimitiveExecutionError("no_stroma")

    operation = primitive_config.get("mask_operation", {})
    source_labels = tuple(operation.get("primary_sources", ())) + tuple(
        operation.get("secondary_sources", ())
    )
    legal = np.zeros(normalized.shape, dtype=bool)
    for label in source_labels:
        if label in schema.readable_labels:
            legal |= np.isin(normalized, schema.resolve_fine_ids(label))
    legal &= ~np.isin(normalized, tuple(schema.skip_fine_ids))
    if not np.any(legal):
        raise PrimitiveExecutionError("no_editable_non_stroma_tissue")

    ranges = primitive_config.get("parameter_ranges", {})
    interval = ranges.get("target_area_delta_fraction", {}).get(intent.strength)
    if intent.target_change_fraction is not None:
        fraction = float(intent.target_change_fraction)
    elif isinstance(interval, list) and len(interval) == 2:
        fraction = 0.5 * (float(interval[0]) + float(interval[1]))
    else:
        raise PrimitiveExecutionError(
            f"stroma_increase does not define strength {intent.strength}."
        )
    target_pixels = min(
        int(np.ceil(fraction * normalized.size)),
        int(np.count_nonzero(legal)),
    )
    if target_pixels <= 0:
        raise PrimitiveExecutionError("no_editable_non_stroma_tissue")

    radius = float(ranges.get("stroma_neighbor_radius_px", 48.0))
    distance = ndimage.distance_transform_edt(~stroma)
    score = np.exp(-distance / max(radius, 1.0))
    score[~legal] = -np.inf
    order = np.argsort(-score.ravel(), kind="stable")
    finite = order[np.isfinite(score.ravel()[order])]
    selected = np.zeros(normalized.shape, dtype=bool)
    selected.ravel()[finite[:target_pixels]] = True

    target = np.array(normalized, copy=True)
    target[selected] = int(stroma_ids[0])
    selected_pixels = int(np.count_nonzero(selected))
    changed_fraction = selected_pixels / int(normalized.size)
    return PrimitiveEditResult(
        target_mask=target,
        change_region=selected,
        changed_area_fraction=changed_fraction,
        selected_pixels=selected_pixels,
        warnings=(),
        ops_log={
            "primitive": "stroma_increase",
            "reference_profile": schema.reference_profile,
            "source_labels": list(source_labels),
            "target_label": "Stroma",
            "selected_pixels": selected_pixels,
            "target_pixels": target_pixels,
            "changed_area_fraction": changed_fraction,
            "spatial_policy": "expand_from_existing_stroma_without_tumor_requirement",
        },
    )

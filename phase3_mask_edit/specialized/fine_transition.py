"""Fine-label transition primitive for dataset-specialized Phase 3 edits."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.generic.tumor_burden import (
    PrimitiveEditResult,
    PrimitiveExecutionError,
)


def apply_fine_label_transition(
    old_mask: np.ndarray,
    schema: MaskProfileSchema,
    context: MaskEditContext,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> PrimitiveEditResult:
    """Convert selected source fine IDs into one target fine ID.

    This is the shared mask primitive for grade/subtype specials such as
    PANDA Gleason transitions, GlaS differentiation shifts, and BCSS DCIS or
    angioinvasion emphasis. Geometry is intentionally conservative: pixels are
    relabeled in place, preserving the original region footprint.
    """

    del old_mask
    mask_operation = primitive_config.get("mask_operation", {})
    if not isinstance(mask_operation, Mapping):
        raise PrimitiveExecutionError("fine_transition_missing_mask_operation")

    source_ids = _int_tuple(mask_operation.get("source_fine_ids"))
    target_id = mask_operation.get("target_fine_id")
    if not source_ids or not isinstance(target_id, int):
        raise PrimitiveExecutionError("fine_transition_missing_fine_ids")

    legal_ids = _legal_schema_ids(schema)
    illegal_ids = (set(source_ids) | {target_id}) - legal_ids
    if illegal_ids:
        raise PrimitiveExecutionError(
            f"fine_transition_ids_not_in_schema:{sorted(illegal_ids)}"
        )

    candidate_mask = np.isin(context.normalized_mask, source_ids)
    candidate_pixels = int(np.count_nonzero(candidate_mask))
    if candidate_pixels == 0:
        raise PrimitiveExecutionError("fine_transition_source_absent")

    target_fraction = _target_fraction_for_intent(primitive_config, intent)
    target_pixels = max(1, min(candidate_pixels, int(round(target_fraction * candidate_pixels))))
    change_region = _select_transition_pixels(candidate_mask, target_pixels)
    selected_pixels = int(np.count_nonzero(change_region))
    if selected_pixels == 0:
        raise PrimitiveExecutionError("fine_transition_empty_selection")

    target_mask = np.array(context.normalized_mask, copy=True)
    target_mask[change_region] = int(target_id)
    changed_area_fraction = selected_pixels / target_mask.size

    return PrimitiveEditResult(
        target_mask=target_mask,
        change_region=change_region,
        changed_area_fraction=changed_area_fraction,
        selected_pixels=selected_pixels,
        warnings=(),
        ops_log={
            "primitive": primitive_config.get("name", intent.primitive),
            "execution_strategy": "id_transition",
            "operation_type": "fine_label_transition",
            "target_change_fraction_semantics": "source_fine_id_relative_relabel_fraction",
            "target_change_fraction_denominator": "source_fine_id_pixels",
            "requested_source_relative_fraction": target_fraction,
            "reference_profile": schema.reference_profile,
            "source_fine_ids": list(source_ids),
            "target_fine_id": int(target_id),
            "candidate_pixels": candidate_pixels,
            "target_pixels": target_pixels,
            "selected_pixels": selected_pixels,
            "source_relative_fraction": selected_pixels / candidate_pixels,
            "changed_area_fraction": changed_area_fraction,
        },
    )


def _select_transition_pixels(candidate_mask: np.ndarray, target_pixels: int) -> np.ndarray:
    """Select a deterministic central subset of candidate pixels."""

    ys, xs = np.nonzero(candidate_mask)
    if len(ys) <= target_pixels:
        return candidate_mask.copy()

    cy = float(np.mean(ys))
    cx = float(np.mean(xs))
    order = np.lexsort((xs, ys, (ys - cy) ** 2 + (xs - cx) ** 2))
    selected = np.zeros(candidate_mask.shape, dtype=bool)
    keep = order[:target_pixels]
    selected[ys[keep], xs[keep]] = True
    return selected


def _target_fraction_for_intent(
    primitive_config: Mapping[str, Any], intent: EditIntent
) -> float:
    if intent.target_change_fraction is not None:
        return max(0.0, min(1.0, float(intent.target_change_fraction)))

    ranges = primitive_config.get("parameter_ranges", {})
    if not isinstance(ranges, Mapping):
        return 0.25
    transition_ranges = ranges.get("source_area_transition_fraction", {})
    if not isinstance(transition_ranges, Mapping):
        return 0.25
    interval = transition_ranges.get(intent.strength)
    if (
        isinstance(interval, list)
        and len(interval) == 2
        and all(isinstance(item, (int, float)) for item in interval)
    ):
        return float(interval[0] + interval[1]) / 2.0
    return 0.25


def _int_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,)
    if isinstance(value, (list, tuple)) and all(isinstance(item, int) for item in value):
        return tuple(value)
    return ()


def _legal_schema_ids(schema: MaskProfileSchema) -> set[int]:
    ids: set[int] = set(schema.skip_fine_ids)
    for fine_ids in schema.label_to_fine_ids.values():
        ids.update(int(fine_id) for fine_id in fine_ids)
    return ids

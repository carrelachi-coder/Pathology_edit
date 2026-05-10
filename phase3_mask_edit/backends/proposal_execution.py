"""Shared proposal projection and deterministic label-write helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.generic.tumor_burden import PrimitiveEditResult


class ProposalExecutionError(ValueError):
    """Raised when a backend proposal cannot be executed safely."""


def apply_projected_label_write(
    old_mask: np.ndarray,
    candidate_region: np.ndarray,
    *,
    schema: MaskProfileSchema,
    source_labels: Sequence[str],
    target_label: str,
    preserve_labels: Sequence[str] = (),
    forbidden_labels: Sequence[str] = (),
    backend: str = "proposal",
    raw_payload: Mapping[str, Any] | None = None,
) -> PrimitiveEditResult:
    """Project a proposal onto legal source labels and write a target label.

    Backends may propose broad, approximate regions.  This helper keeps Phase 3
    in charge of label safety by only writing pixels that currently belong to
    the declared source labels and do not belong to preserve/forbidden labels.
    Empty projected regions are returned as empty edits so normal validation can
    produce structured failure checks for repair loops.
    """

    mask = np.asarray(old_mask)
    if mask.ndim != 2:
        raise ProposalExecutionError("old_mask must be a 2D id mask.")

    candidate = np.asarray(candidate_region, dtype=bool)
    if candidate.shape != mask.shape:
        raise ProposalExecutionError(
            "candidate_region shape must match old_mask shape: "
            f"{candidate.shape} != {mask.shape}."
        )

    if not source_labels:
        raise ProposalExecutionError("source_labels must be non-empty.")

    source_mask = _label_mask(mask, schema, source_labels, context="source_labels")
    removal_labels = tuple(dict.fromkeys(tuple(preserve_labels) + tuple(forbidden_labels)))
    removal_mask = (
        _label_mask(mask, schema, removal_labels, context="preserve_or_forbidden_labels")
        if removal_labels
        else np.zeros(mask.shape, dtype=bool)
    )
    background_mask = np.isin(mask, tuple(schema.skip_fine_ids))

    projected_region = candidate & source_mask & ~removal_mask & ~background_mask
    selected_pixels = int(np.count_nonzero(projected_region))

    target_ids = schema.resolve_fine_ids(target_label)
    target_mask = np.array(mask, copy=True)
    if selected_pixels > 0:
        target_mask[projected_region] = int(target_ids[0])

    candidate_pixels = int(np.count_nonzero(candidate))
    source_projected_pixels = int(np.count_nonzero(candidate & source_mask))
    removed_pixels = int(np.count_nonzero(candidate & (removal_mask | background_mask)))
    retained_fraction = (
        selected_pixels / candidate_pixels if candidate_pixels > 0 else 0.0
    )
    changed_area_fraction = selected_pixels / int(mask.size)

    warnings: list[str] = []
    if selected_pixels == 0:
        warnings.append("proposal_projected_region_empty")

    ops_log = {
        "backend": backend,
        "method": "source_label_projection_and_deterministic_write",
        "reference_profile": schema.reference_profile,
        "source_labels": list(source_labels),
        "target_label": target_label,
        "target_fine_id": int(target_ids[0]),
        "preserve_labels": list(preserve_labels),
        "forbidden_labels": list(forbidden_labels),
        "candidate_pixels": candidate_pixels,
        "source_projected_pixels": source_projected_pixels,
        "projected_pixels": selected_pixels,
        "removed_pixels": removed_pixels,
        "selected_pixels": selected_pixels,
        "projection_retained_fraction": retained_fraction,
        "changed_area_fraction": changed_area_fraction,
    }
    if raw_payload is not None:
        ops_log["raw_payload"] = dict(raw_payload)

    return PrimitiveEditResult(
        target_mask=target_mask,
        change_region=projected_region,
        changed_area_fraction=changed_area_fraction,
        selected_pixels=selected_pixels,
        warnings=tuple(warnings),
        ops_log=ops_log,
    )


def _label_mask(
    mask: np.ndarray,
    schema: MaskProfileSchema,
    labels: Sequence[str],
    *,
    context: str,
) -> np.ndarray:
    result = np.zeros(mask.shape, dtype=bool)
    for label in labels:
        if not isinstance(label, str) or not label:
            raise ProposalExecutionError(f"{context} must contain non-empty strings.")
        result |= np.isin(mask, schema.resolve_fine_ids(label))
    return result

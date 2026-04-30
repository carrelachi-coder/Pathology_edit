"""Semantic candidate-mask selection for Phase 3 primitive execution."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Mapping

import numpy as np

from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema


class CandidateSelectionError(ValueError):
    """Raised when candidate selection receives an invalid request."""


@dataclass(frozen=True)
class CandidateSelection:
    """Candidate pixels that a primitive may pass to geometry selection."""

    candidate_mask: np.ndarray
    included_labels: tuple[str, ...]
    excluded_labels: tuple[str, ...]
    priority_labels: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @property
    def candidate_pixels(self) -> int:
        """Number of candidate pixels selected by semantic filtering."""

        return int(np.count_nonzero(self.candidate_mask))


def build_candidate_mask_by_priority(
    mask: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
    *,
    context: MaskEditContext | None = None,
    target_fraction: float | None = None,
) -> CandidateSelection:
    """Build a semantic source-candidate mask from recipe priority rules.

    This function does not choose a spatially connected edit region and does
    not modify the mask. It only answers: which current-mask tissue pixels are
    semantically legal sources for this primitive?
    """

    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise CandidateSelectionError("Candidate selection requires a 2D mask.")
    if context is not None and context.reference_profile != schema.reference_profile:
        raise CandidateSelectionError(
            "context.reference_profile must match schema.reference_profile."
        )

    priority_labels = _candidate_priority_labels(primitive_config)
    forbidden_labels = _forbidden_labels(primitive_config) | set(intent.forbidden_labels)
    preserve_labels = set(intent.preserve_labels)
    source_label_filter = set(intent.source_labels)
    requested_fraction = _requested_fraction(intent, target_fraction)
    target_pixels = (
        ceil(requested_fraction * mask_array.size)
        if requested_fraction is not None
        else None
    )

    selected = np.zeros(mask_array.shape, dtype=bool)
    included: list[str] = []
    excluded: list[str] = []
    warnings: list[str] = []

    if source_label_filter:
        warnings.append("source_label_filter_applied")

    for label in priority_labels:
        exclusion = _label_exclusion_reason(
            label=label,
            schema=schema,
            source_label_filter=source_label_filter,
            preserve_labels=preserve_labels,
            forbidden_labels=forbidden_labels,
        )
        if exclusion is not None:
            excluded.append(exclusion)
            continue

        label_mask = np.isin(mask_array, schema.resolve_fine_ids(label))
        if not np.any(label_mask):
            excluded.append(f"label_absent_in_mask:{label}")
            continue

        selected |= label_mask
        included.append(label)
        _append_semantic_warning(label, schema, context, warnings)

        if target_pixels is None or int(np.count_nonzero(selected)) >= target_pixels:
            break

    if target_pixels is not None and int(np.count_nonzero(selected)) < target_pixels:
        warnings.append(
            "candidate_area_below_target_fraction:"
            f"requested={target_pixels},available={int(np.count_nonzero(selected))}"
        )
    if context is not None:
        warnings.extend(f"context_risk:{flag}" for flag in context.risk_flags)
    if not included:
        warnings.append("no_candidate_labels_available")

    return CandidateSelection(
        candidate_mask=selected,
        included_labels=tuple(included),
        excluded_labels=tuple(excluded),
        priority_labels=priority_labels,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _candidate_priority_labels(primitive_config: Mapping[str, Any]) -> tuple[str, ...]:
    operation = _mapping_value(primitive_config, "mask_operation")

    if isinstance(operation.get("target_priority"), list):
        return tuple(str(label) for label in operation["target_priority"])

    primary_sources = operation.get("primary_sources")
    secondary_sources = operation.get("secondary_sources")
    if isinstance(primary_sources, list) or isinstance(secondary_sources, list):
        labels: list[str] = []
        if isinstance(primary_sources, list):
            labels.extend(str(label) for label in primary_sources)
        if isinstance(secondary_sources, list):
            labels.extend(str(label) for label in secondary_sources)
        return tuple(labels)

    source = operation.get("source")
    target = operation.get("target")
    if isinstance(source, str) and isinstance(target, str):
        return (source,)

    return ()


def _forbidden_labels(primitive_config: Mapping[str, Any]) -> set[str]:
    operation = _mapping_value(primitive_config, "mask_operation")
    spatial_pattern = _mapping_value(primitive_config, "spatial_pattern")
    forbidden: set[str] = set()

    for key in ("forbid_targets", "forbid_sources"):
        forbidden.update(_string_list(operation.get(key)))
        forbidden.update(_string_list(spatial_pattern.get(key)))

    return forbidden


def _label_exclusion_reason(
    *,
    label: str,
    schema: MaskProfileSchema,
    source_label_filter: set[str],
    preserve_labels: set[str],
    forbidden_labels: set[str],
) -> str | None:
    if source_label_filter and label not in source_label_filter:
        return f"label_not_requested:{label}"
    if label in preserve_labels:
        return f"label_preserved:{label}"
    if label in forbidden_labels:
        return f"label_forbidden:{label}"
    if label not in schema.readable_labels:
        return f"label_not_readable:{label}"
    return None


def _append_semantic_warning(
    label: str,
    schema: MaskProfileSchema,
    context: MaskEditContext | None,
    warnings: list[str],
) -> None:
    semantic_warnings = dict(schema.semantic_warnings)
    if context is not None:
        semantic_warnings.update(context.semantic_warnings)
    if label in semantic_warnings:
        warnings.append(f"semantic_warning:{label}")


def _requested_fraction(
    intent: EditIntent, target_fraction: float | None
) -> float | None:
    requested = target_fraction
    if requested is None:
        requested = intent.target_change_fraction
    if requested is None:
        return None
    if not 0.0 <= requested <= 1.0:
        raise CandidateSelectionError("target_fraction must be in [0, 1].")
    return float(requested)


def _mapping_value(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    nested = value.get(key, {})
    if isinstance(nested, Mapping):
        return nested
    return {}


def _string_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))

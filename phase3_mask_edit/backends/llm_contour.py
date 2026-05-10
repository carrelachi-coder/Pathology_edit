"""LLM contour proposal schema validation and rasterization.

This module only handles Milestone A responsibilities: validate an LLM-returned
polygon JSON payload and convert valid polygons into a binary candidate region.
Projection, label writing, and Phase 3 validation live in later execution steps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from phase3_mask_edit.backends.proposal_execution import apply_projected_label_write
from phase3_mask_edit.core.labels import MaskProfileSchema, MaskProfileSchemaError
from phase3_mask_edit.generic.tumor_burden import PrimitiveEditResult


CONTOUR_PROPOSAL_SCHEMA_VERSION = "0.1"
CONTOUR_PROPOSAL_BACKEND = "llm_contour_proposal"


class ContourProposalValidationError(ValueError):
    """Raised when an LLM contour proposal is not safe to rasterize."""


@dataclass(frozen=True)
class ContourRegion:
    """One validated polygon region from an LLM contour proposal."""

    region_id: str
    source_labels: tuple[str, ...]
    points: tuple[tuple[float, float], ...]
    confidence: float | None = None


@dataclass(frozen=True)
class ContourProposal:
    """Validated LLM contour proposal ready for rasterization."""

    primitive: str
    reference_profile: str
    target_label: str
    width: int
    height: int
    regions: tuple[ContourRegion, ...]
    raw_payload: dict[str, Any]


def load_contour_proposal_json(path: str | Path) -> dict[str, Any]:
    """Load an LLM-returned contour proposal JSON file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ContourProposalValidationError("contour proposal JSON must be an object.")
    return payload


def validate_contour_proposal(
    payload: Mapping[str, Any],
    *,
    schema: MaskProfileSchema,
    mask_shape: tuple[int, int],
    primitive: str | None = None,
    reference_profile: str | None = None,
    target_label: str | None = None,
    allowed_source_labels: Sequence[str] | None = None,
    max_regions: int = 8,
    max_points_per_region: int = 64,
) -> ContourProposal:
    """Validate an LLM contour proposal before rasterization.

    Coordinates are interpreted as original mask-space ``[x, y]`` values where
    x is the column and y is the row, with origin at the top-left.
    """

    if not isinstance(payload, Mapping):
        raise ContourProposalValidationError("contour proposal must be a mapping.")
    height, width = _validate_mask_shape(mask_shape)

    _require_equal(
        payload.get("schema_version"),
        CONTOUR_PROPOSAL_SCHEMA_VERSION,
        "schema_version",
    )
    _require_equal(payload.get("backend"), CONTOUR_PROPOSAL_BACKEND, "backend")

    payload_primitive = _required_string(payload, "primitive")
    if primitive is not None and payload_primitive != primitive:
        raise ContourProposalValidationError(
            f"primitive mismatch: expected {primitive!r}, got {payload_primitive!r}."
        )

    payload_profile = _required_string(payload, "reference_profile")
    expected_profile = reference_profile or schema.reference_profile
    if payload_profile != expected_profile:
        raise ContourProposalValidationError(
            "reference_profile mismatch: "
            f"expected {expected_profile!r}, got {payload_profile!r}."
        )

    payload_target = _required_string(payload, "target_label")
    if target_label is not None and payload_target != target_label:
        raise ContourProposalValidationError(
            f"target_label mismatch: expected {target_label!r}, got {payload_target!r}."
        )
    _ensure_label(payload_target, schema, context="target_label")

    _validate_coordinate_system(payload.get("coordinate_system"), width=width, height=height)

    raw_regions = payload.get("regions")
    if not isinstance(raw_regions, list) or not raw_regions:
        raise ContourProposalValidationError("regions must be a non-empty list.")
    if len(raw_regions) > max_regions:
        raise ContourProposalValidationError(
            f"regions has {len(raw_regions)} items; maximum is {max_regions}."
        )

    allowed_sources = (
        frozenset(allowed_source_labels) if allowed_source_labels is not None else None
    )
    regions = tuple(
        _validate_region(
            region,
            schema=schema,
            width=width,
            height=height,
            allowed_source_labels=allowed_sources,
            max_points=max_points_per_region,
        )
        for region in raw_regions
    )

    return ContourProposal(
        primitive=payload_primitive,
        reference_profile=payload_profile,
        target_label=payload_target,
        width=width,
        height=height,
        regions=regions,
        raw_payload=dict(payload),
    )


def rasterize_polygon(
    points: Sequence[Sequence[float]],
    *,
    mask_shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize one validated polygon into a 2D boolean candidate region."""

    height, width = _validate_mask_shape(mask_shape)
    if len(points) < 3:
        raise ContourProposalValidationError("polygon requires at least 3 points.")
    polygon = [_coerce_point(point, width=width, height=height) for point in points]

    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    draw.polygon(polygon, fill=1)
    return np.asarray(image, dtype=np.uint8).astype(bool)


def rasterize_contour_proposal(proposal: ContourProposal) -> np.ndarray:
    """Rasterize all regions in a validated proposal into one candidate mask."""

    candidate = np.zeros((proposal.height, proposal.width), dtype=bool)
    for region in proposal.regions:
        candidate |= rasterize_polygon(
            region.points,
            mask_shape=(proposal.height, proposal.width),
        )
    return candidate


def execute_contour_proposal_write(
    old_mask: np.ndarray,
    proposal: ContourProposal,
    *,
    schema: MaskProfileSchema,
    preserve_labels: Sequence[str] = (),
    forbidden_labels: Sequence[str] = (),
) -> PrimitiveEditResult:
    """Project an LLM contour proposal and deterministically write its target.

    Each region is projected using its own ``source_labels`` before being merged,
    so labels declared for one polygon never authorize writes in another polygon.
    """

    if tuple(old_mask.shape) != (proposal.height, proposal.width):
        raise ContourProposalValidationError(
            "old_mask shape must match proposal coordinate system: "
            f"{tuple(old_mask.shape)} != {(proposal.height, proposal.width)}."
        )

    projected_candidate = np.zeros((proposal.height, proposal.width), dtype=bool)
    region_logs: list[dict[str, Any]] = []
    for region in proposal.regions:
        raw_region = rasterize_polygon(
            region.points,
            mask_shape=(proposal.height, proposal.width),
        )
        region_result = apply_projected_label_write(
            old_mask,
            raw_region,
            schema=schema,
            source_labels=region.source_labels,
            target_label=proposal.target_label,
            preserve_labels=preserve_labels,
            forbidden_labels=forbidden_labels,
            backend=CONTOUR_PROPOSAL_BACKEND,
        )
        projected_candidate |= region_result.change_region
        region_logs.append(
            {
                "region_id": region.region_id,
                "source_labels": list(region.source_labels),
                "confidence": region.confidence,
                "candidate_pixels": region_result.ops_log["candidate_pixels"],
                "projected_pixels": region_result.ops_log["projected_pixels"],
                "selected_pixels": region_result.selected_pixels,
                "projection_retained_fraction": region_result.ops_log[
                    "projection_retained_fraction"
                ],
            }
        )

    selected_pixels = int(np.count_nonzero(projected_candidate))
    target_ids = schema.resolve_fine_ids(proposal.target_label)
    target_mask = np.array(old_mask, copy=True)
    if selected_pixels > 0:
        target_mask[projected_candidate] = int(target_ids[0])

    raw_candidate = rasterize_contour_proposal(proposal)
    candidate_pixels = int(np.count_nonzero(raw_candidate))
    changed_area_fraction = selected_pixels / int(old_mask.size)
    warnings = ("proposal_projected_region_empty",) if selected_pixels == 0 else ()
    ops_log = {
        "backend": CONTOUR_PROPOSAL_BACKEND,
        "method": "per_region_source_label_projection_and_deterministic_write",
        "primitive": proposal.primitive,
        "reference_profile": schema.reference_profile,
        "target_label": proposal.target_label,
        "target_fine_id": int(target_ids[0]),
        "candidate_pixels": candidate_pixels,
        "projected_pixels": selected_pixels,
        "selected_pixels": selected_pixels,
        "projection_retained_fraction": (
            selected_pixels / candidate_pixels if candidate_pixels > 0 else 0.0
        ),
        "changed_area_fraction": changed_area_fraction,
        "preserve_labels": list(preserve_labels),
        "forbidden_labels": list(forbidden_labels),
        "raw_payload": dict(proposal.raw_payload),
    }
    ops_log["region_projection"] = region_logs
    ops_log["source_labels"] = sorted(
        {label for region in proposal.regions for label in region.source_labels}
    )

    return PrimitiveEditResult(
        target_mask=target_mask,
        change_region=projected_candidate,
        changed_area_fraction=changed_area_fraction,
        selected_pixels=selected_pixels,
        warnings=warnings,
        ops_log=ops_log,
    )


def _validate_region(
    region: Any,
    *,
    schema: MaskProfileSchema,
    width: int,
    height: int,
    allowed_source_labels: frozenset[str] | None,
    max_points: int,
) -> ContourRegion:
    if not isinstance(region, Mapping):
        raise ContourProposalValidationError("each region must be a mapping.")
    _require_equal(region.get("type"), "polygon", "region.type")

    region_id = _required_string(region, "region_id")
    source_labels = _string_tuple(region.get("source_labels"), "source_labels")
    if not source_labels:
        raise ContourProposalValidationError("source_labels must be non-empty.")
    for label in source_labels:
        _ensure_label(label, schema, context="source_labels")
        if allowed_source_labels is not None and label not in allowed_source_labels:
            raise ContourProposalValidationError(
                f"source label {label!r} is not allowed for this proposal."
            )

    raw_points = region.get("points")
    if not isinstance(raw_points, list):
        raise ContourProposalValidationError("points must be a list.")
    if len(raw_points) < 3:
        raise ContourProposalValidationError("polygon requires at least 3 points.")
    if len(raw_points) > max_points:
        raise ContourProposalValidationError(
            f"polygon has {len(raw_points)} points; maximum is {max_points}."
        )
    points = tuple(_coerce_point(point, width=width, height=height) for point in raw_points)

    confidence = region.get("confidence")
    if confidence is not None:
        if not isinstance(confidence, (int, float)):
            raise ContourProposalValidationError("confidence must be numeric.")
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ContourProposalValidationError("confidence must be in [0, 1].")

    return ContourRegion(
        region_id=region_id,
        source_labels=source_labels,
        points=points,
        confidence=confidence,
    )


def _validate_coordinate_system(value: Any, *, width: int, height: int) -> None:
    if not isinstance(value, Mapping):
        raise ContourProposalValidationError("coordinate_system must be a mapping.")
    expected = {
        "origin": "top_left",
        "point_format": "[x, y]",
        "x_axis": "horizontal_column_right",
        "y_axis": "vertical_row_down",
    }
    for key, expected_value in expected.items():
        _require_equal(value.get(key), expected_value, f"coordinate_system.{key}")
    _require_equal(value.get("width"), width, "coordinate_system.width")
    _require_equal(value.get("height"), height, "coordinate_system.height")


def _coerce_point(point: Any, *, width: int, height: int) -> tuple[float, float]:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise ContourProposalValidationError("each point must be [x, y].")
    x_raw, y_raw = point
    if not isinstance(x_raw, (int, float)) or not isinstance(y_raw, (int, float)):
        raise ContourProposalValidationError("point coordinates must be numeric.")
    x = float(x_raw)
    y = float(y_raw)
    if not np.isfinite(x) or not np.isfinite(y):
        raise ContourProposalValidationError("point coordinates must be finite.")
    if x < 0 or y < 0 or x > width - 1 or y > height - 1:
        raise ContourProposalValidationError(
            f"point {[x_raw, y_raw]!r} is outside mask bounds {width}x{height}."
        )
    return (x, y)


def _validate_mask_shape(mask_shape: tuple[int, int]) -> tuple[int, int]:
    if len(mask_shape) != 2:
        raise ContourProposalValidationError("mask_shape must be (height, width).")
    height, width = int(mask_shape[0]), int(mask_shape[1])
    if height <= 0 or width <= 0:
        raise ContourProposalValidationError("mask_shape dimensions must be positive.")
    return height, width


def _ensure_label(label: str, schema: MaskProfileSchema, *, context: str) -> None:
    try:
        schema.resolve_fine_ids(label)
    except MaskProfileSchemaError as exc:
        raise ContourProposalValidationError(
            f"{context} contains unknown label: {label!r}."
        ) from exc


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ContourProposalValidationError(f"{key} is required and must be a string.")
    return value


def _string_tuple(value: Any, key: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list):
        raise ContourProposalValidationError(f"{key} must be a list of strings.")
    labels = tuple(value)
    if not all(isinstance(label, str) and label for label in labels):
        raise ContourProposalValidationError(f"{key} must contain only strings.")
    return labels


def _require_equal(value: Any, expected: Any, key: str) -> None:
    if value != expected:
        raise ContourProposalValidationError(
            f"{key} must be {expected!r}; got {value!r}."
        )

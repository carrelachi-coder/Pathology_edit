"""Unwired research candidate for subdividing one directional interface anchor.

This module is deliberately not imported by the production scene builder or
planner.  It supports local synthetic evaluation of a possible way to retain
an unedited boundary witness on interfaces that currently have one anchor.
Any future integration must rebuild the scene and rerun nuclei preflight; an
old feasible-anchor result is not valid for the returned subanchors.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace

import numpy as np

from phase3_mask_edit_refine.models import SceneAnchorSegment
from phase3_mask_edit_refine.scene import SceneAnalysis

CANDIDATE_VERSION = "directional-single-anchor-balanced-geodesic-bisection-v3"


def minimum_witness_contact_pixels(reference_area_p95: float) -> int:
    """Return one source-reference equivalent diameter in raster pixels."""

    if not np.isfinite(reference_area_p95) or reference_area_p95 <= 0.0:
        raise ValueError("reference area p95 must be positive and finite")
    return int(np.ceil(np.sqrt(4.0 * float(reference_area_p95) / np.pi)))


def geodesic_span_steps(anchor_mask: np.ndarray) -> int:
    """Return the exact 8-neighbour diameter used by the reviewed policy."""

    return _geodesic_diameter_steps(np.asarray(anchor_mask, dtype=bool))


def subdivision_eligibility_for_underfill_replan(
    scene: SceneAnalysis,
    *,
    interface_id: str,
    cell_feasible_anchor_ids: tuple[str, ...],
    remaining_feasible_interface_ids: tuple[str, ...],
    prior_topology_passed: bool,
    area_underfill_replan: bool,
    maximum_selected_anchor_fraction: float,
    minimum_unselected_anchor_count: int,
    minimum_subanchor_pixels: int,
) -> tuple[bool, tuple[str, ...]]:
    """Fail-closed eligibility for proposing subdivision after safe underfill.

    This policy never runs on the initial compile and never converts a failed
    topology trial into an admissible replan.  It only identifies an interface
    whose parent anchor was cell-feasible but was excluded solely because the
    unchanged directional witness rule could select zero of one anchors.
    """

    reasons: list[str] = []
    if minimum_subanchor_pixels < 1:
        reasons.append("invalid_minimum_subanchor_pixels")
    if (
        not np.isfinite(maximum_selected_anchor_fraction)
        or maximum_selected_anchor_fraction < 0.0
        or maximum_selected_anchor_fraction > 1.0
    ):
        reasons.append("invalid_maximum_selected_anchor_fraction")
    if minimum_unselected_anchor_count < 0:
        reasons.append("invalid_minimum_unselected_anchor_count")
    if reasons:
        return False, tuple(reasons)
    interface = next(
        (item for item in scene.graph.interfaces if item.interface_id == interface_id),
        None,
    )
    if interface is None:
        return False, ("unknown_interface",)
    if not area_underfill_replan:
        reasons.append("not_area_underfill_replan")
    if not prior_topology_passed:
        reasons.append("prior_topology_not_passed")
    if interface_id not in set(remaining_feasible_interface_ids):
        reasons.append("interface_not_remaining_feasible")
    if len(interface.anchor_segment_ids) != 1:
        reasons.append("interface_not_single_anchor")
        return False, tuple(reasons)
    anchor_id = interface.anchor_segment_ids[0]
    if anchor_id not in scene.anchor_masks:
        reasons.append("parent_anchor_mask_missing")
        return False, tuple(reasons)
    feasible = set(cell_feasible_anchor_ids)
    if anchor_id not in feasible:
        reasons.append("parent_anchor_not_cell_feasible")
    total = 1
    selection_limit = max(
        0,
        min(
            1 if anchor_id in feasible else 0,
            int(np.floor(total * maximum_selected_anchor_fraction)),
            total - minimum_unselected_anchor_count,
        ),
    )
    if selection_limit != 0:
        reasons.append("parent_anchor_already_selectable")
    anchor_mask = scene.anchor_masks[anchor_id]
    if int(np.count_nonzero(anchor_mask)) < 2 * int(
        minimum_subanchor_pixels
    ):
        reasons.append("parent_anchor_too_small")
    elif not reasons:
        try:
            bisect_connected_interface_anchor(
                anchor_mask,
                minimum_subanchor_pixels=minimum_subanchor_pixels,
            )
        except ValueError:
            reasons.append("parent_anchor_not_bisectable")
    return not reasons, tuple(reasons)


def subdivision_eligibility_for_directional_front_block(
    scene: SceneAnalysis,
    *,
    interface_id: str,
    cell_feasible_anchor_ids: tuple[str, ...],
    maximum_selected_anchor_fraction: float,
    minimum_unselected_anchor_count: int,
    minimum_subanchor_pixels: int,
) -> tuple[bool, tuple[str, ...]]:
    """Fail closed when a valid single anchor is blocked only by sector count.

    This is the pre-planner counterpart of the underfill policy above.  It is
    eligible only when nuclei preflight certified the sole parent anchor but
    the unchanged directional-front count contract permits selecting zero of
    one anchors.  The caller must rebuild the scene and rerun nuclei preflight.
    """

    reasons: list[str] = []
    if minimum_subanchor_pixels < 1:
        reasons.append("invalid_minimum_subanchor_pixels")
    if (
        not np.isfinite(maximum_selected_anchor_fraction)
        or maximum_selected_anchor_fraction < 0.0
        or maximum_selected_anchor_fraction > 1.0
    ):
        reasons.append("invalid_maximum_selected_anchor_fraction")
    if minimum_unselected_anchor_count < 0:
        reasons.append("invalid_minimum_unselected_anchor_count")
    if reasons:
        return False, tuple(reasons)
    interface = next(
        (item for item in scene.graph.interfaces if item.interface_id == interface_id),
        None,
    )
    if interface is None:
        return False, ("unknown_interface",)
    if len(interface.anchor_segment_ids) != 1:
        return False, ("interface_not_single_anchor",)
    anchor_id = interface.anchor_segment_ids[0]
    if anchor_id not in scene.anchor_masks:
        return False, ("parent_anchor_mask_missing",)
    feasible = set(cell_feasible_anchor_ids)
    if anchor_id not in feasible:
        reasons.append("parent_anchor_not_cell_feasible")
    selection_limit = max(
        0,
        min(
            1 if anchor_id in feasible else 0,
            int(np.floor(maximum_selected_anchor_fraction)),
            1 - minimum_unselected_anchor_count,
        ),
    )
    if selection_limit != 0:
        reasons.append("parent_anchor_already_selectable")
    anchor_mask = scene.anchor_masks[anchor_id]
    if int(np.count_nonzero(anchor_mask)) < 2 * minimum_subanchor_pixels:
        reasons.append("parent_anchor_too_small")
    elif not reasons:
        try:
            bisect_connected_interface_anchor(
                anchor_mask,
                minimum_subanchor_pixels=minimum_subanchor_pixels,
            )
        except ValueError:
            reasons.append("parent_anchor_not_bisectable")
    return not reasons, tuple(reasons)


def rebuild_scene_with_bisected_single_anchor(
    scene: SceneAnalysis,
    *,
    interface_id: str,
    minimum_subanchor_pixels: int,
) -> SceneAnalysis:
    """Return a new scene whose named single-anchor interface has two anchors.

    All graph and mask identities are regenerated in the returned immutable
    scene.  The function does not carry any nuclei-preflight result across the
    identity change; callers must recompute preflight against the new scene.
    """

    interface = next(
        (item for item in scene.graph.interfaces if item.interface_id == interface_id),
        None,
    )
    if interface is None:
        raise ValueError(f"unknown interface: {interface_id}")
    if len(interface.anchor_segment_ids) != 1:
        raise ValueError("candidate applies only to a single-anchor interface")
    old_anchor_id = interface.anchor_segment_ids[0]
    parts = bisect_connected_interface_anchor(
        scene.anchor_masks[old_anchor_id],
        minimum_subanchor_pixels=minimum_subanchor_pixels,
    )
    new_ids = tuple(
        f"{old_anchor_id}:sub:{index:04d}" for index in range(1, 3)
    )

    anchor_masks = dict(scene.anchor_masks)
    anchor_masks.pop(old_anchor_id)
    anchor_masks.update(zip(new_ids, parts))
    new_segments: list[SceneAnchorSegment] = []
    for segment in scene.graph.anchor_segments:
        if segment.anchor_segment_id != old_anchor_id:
            new_segments.append(segment)
            continue
        for anchor_id, part in zip(new_ids, parts):
            rows, cols = np.where(part)
            new_segments.append(
                replace(
                    segment,
                    anchor_segment_id=anchor_id,
                    contact_pixels=int(rows.size),
                    bbox_xyxy=(
                        int(cols.min()),
                        int(rows.min()),
                        int(cols.max()) + 1,
                        int(rows.max()) + 1,
                    ),
                    centroid_xy=(float(np.mean(cols)), float(np.mean(rows))),
                )
            )
    new_segments = [
        replace(item, display_index=index)
        for index, item in enumerate(new_segments, start=1)
    ]
    interfaces = tuple(
        replace(item, anchor_segment_ids=new_ids)
        if item.interface_id == interface_id
        else item
        for item in scene.graph.interfaces
    )
    return replace(
        scene,
        graph=replace(
            scene.graph,
            interfaces=interfaces,
            anchor_segments=tuple(new_segments),
        ),
        anchor_masks=anchor_masks,
    )


def bisect_connected_interface_anchor(
    anchor_mask: np.ndarray,
    *,
    minimum_subanchor_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic, exhaustive 8-connected geodesic bisection.

    The two seeds are deterministic two-sweep geodesic endpoints: first take
    the pixel farthest from the lexicographically first mask pixel, then take
    the pixel farthest from that endpoint.  Starting from an endpoint avoids
    the severe imbalance produced when the lexicographically first pixel lies
    near the middle of a branched or folded interface.  Among every integer
    threshold of the signed endpoint-distance field, the function selects the
    most balanced exact partition whose two children are connected and both
    satisfy the frozen size and span requirements.  This criterion is mask-only
    and does not inspect generated H&E or downstream qualification outcomes.
    """

    mask = np.asarray(anchor_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("anchor mask must be two-dimensional")
    if minimum_subanchor_pixels < 1:
        raise ValueError("minimum subanchor pixels must be positive")
    coordinates = np.argwhere(mask)
    if coordinates.shape[0] < 2 * int(minimum_subanchor_pixels):
        raise ValueError("anchor is too small for the requested bisection")

    lexical_seed = tuple(int(value) for value in coordinates[0])
    lexical_distance = _geodesic_distance(mask, lexical_seed)
    first_value = int(np.max(lexical_distance[mask]))
    first_coordinates = np.argwhere(mask & (lexical_distance == first_value))
    first_seed = tuple(int(value) for value in first_coordinates[0])
    first_distance = _geodesic_distance(mask, first_seed)
    second_value = int(np.max(first_distance[mask]))
    second_coordinates = np.argwhere(mask & (first_distance == second_value))
    second_seed = tuple(int(value) for value in second_coordinates[0])
    second_distance = _geodesic_distance(mask, second_seed)

    signed_distance = first_distance.astype(np.int64) - second_distance.astype(
        np.int64
    )
    candidates: list[tuple[tuple[int, int], np.ndarray, np.ndarray]] = []
    thresholds = sorted({int(value) for value in signed_distance[mask].tolist()})
    for threshold in thresholds:
        first = mask & (signed_distance <= threshold)
        second = mask & ~first
        sizes = (int(np.count_nonzero(first)), int(np.count_nonzero(second)))
        if min(sizes) < minimum_subanchor_pixels:
            continue
        try:
            spans = tuple(
                _geodesic_diameter_steps(part) for part in (first, second)
            )
        except ValueError:
            continue
        if min(spans) < minimum_subanchor_pixels - 1:
            continue
        candidates.append(
            ((abs(sizes[0] - sizes[1]), threshold), first, second)
        )
    if not candidates:
        raise ValueError(
            "geodesic bisection has no connected size-and-span-valid partition"
        )
    _rank, first, second = min(candidates, key=lambda item: item[0])
    if not np.array_equal(first | second, mask) or np.any(first & second):
        raise RuntimeError("geodesic bisection is not an exact partition")
    return first, second


def _geodesic_diameter_steps(mask: np.ndarray) -> int:
    """Return the exact 8-neighbour graph diameter for a small anchor mask."""

    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return 0
    diameter = 0
    for coordinate in coordinates:
        distance = _geodesic_distance(
            mask,
            tuple(int(value) for value in coordinate),
        )
        diameter = max(diameter, int(np.max(distance[np.asarray(mask, dtype=bool)])))
    return diameter


def _geodesic_distance(mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    interface = np.asarray(mask, dtype=bool)
    if not interface[seed]:
        raise ValueError("seed must lie inside the anchor")
    unreachable = np.iinfo(np.int32).max
    distance = np.full(interface.shape, unreachable, dtype=np.int32)
    distance[seed] = 0
    queue: deque[tuple[int, int]] = deque([seed])
    height, width = interface.shape
    while queue:
        row, col = queue.popleft()
        next_distance = int(distance[row, col]) + 1
        for delta_row in (-1, 0, 1):
            for delta_col in (-1, 0, 1):
                if delta_row == 0 and delta_col == 0:
                    continue
                next_row = row + delta_row
                next_col = col + delta_col
                if (
                    0 <= next_row < height
                    and 0 <= next_col < width
                    and interface[next_row, next_col]
                    and next_distance < distance[next_row, next_col]
                ):
                    distance[next_row, next_col] = next_distance
                    queue.append((next_row, next_col))
    if np.any(distance[interface] == unreachable):
        raise ValueError("anchor mask must be one 8-connected component")
    return distance

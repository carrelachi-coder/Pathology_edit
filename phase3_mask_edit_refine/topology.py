"""Topology-preserving deterministic front growth for mask-edit refine."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class TopologyGrowAudit:
    requested_pixels: int
    realized_pixels: int
    rejected_source_connectivity: int
    rejected_source_hole_change: int
    rejected_target_hole_change: int
    rejected_target_island: int
    rejected_unselected_target_contact: int
    rejected_source_retention: int

    def to_metadata(self) -> dict[str, int]:
        return {
            "requested_pixels": self.requested_pixels,
            "realized_pixels": self.realized_pixels,
            "rejected_source_connectivity": self.rejected_source_connectivity,
            "rejected_source_hole_change": self.rejected_source_hole_change,
            "rejected_target_hole_change": self.rejected_target_hole_change,
            "rejected_target_island": self.rejected_target_island,
            "rejected_unselected_target_contact": self.rejected_unselected_target_contact,
            "rejected_source_retention": self.rejected_source_retention,
        }


def topology_safe_priority_grow(
    allowed: np.ndarray,
    *,
    interface_mask: np.ndarray,
    target_pixels: int,
    priority: np.ndarray,
    source_component_state: np.ndarray,
    target_state: np.ndarray,
    unselected_target: np.ndarray,
    maximum_source_deletions: int,
    already_deleted_from_source: int,
    protected_source_necks: np.ndarray | None = None,
    seed: int,
) -> tuple[np.ndarray, TopologyGrowAudit]:
    """Grow a front while preserving source/target digital topology.

    A conservative medial-neck raster protects obvious source articulation
    zones before growth. Target additions must remain attached to a selected
    pre-edit target and may not touch an unselected target component. The
    authoritative whole-mask component/hole checks still run after drawing.
    Arrays passed as ``source_component_state`` and ``target_state`` are updated
    in place so several planned interfaces share one auditable topology state.
    """

    legal = np.asarray(allowed, dtype=bool)
    selected = np.zeros_like(legal, dtype=bool)
    requested = max(0, int(target_pixels))
    counters = {
        "source_connectivity": 0,
        "source_hole_change": 0,
        "target_hole_change": 0,
        "target_island": 0,
        "unselected_target_contact": 0,
        "source_retention": 0,
    }
    if requested <= 0 or not np.any(legal):
        return selected, _audit(requested, 0, counters)

    seed_region = legal & (
        np.asarray(interface_mask, dtype=bool)
        | _binary_dilation_8(np.asarray(interface_mask, dtype=bool))
    )
    seeds = np.argwhere(seed_region)
    if seeds.size == 0:
        return selected, _audit(requested, 0, counters)

    rng = np.random.default_rng(seed)
    seen = np.zeros_like(legal, dtype=bool)
    heap: list[tuple[float, float, int, int]] = []
    for row, col in seeds:
        heapq.heappush(
            heap,
            (
                float(priority[row, col]),
                float(rng.random()),
                int(row),
                int(col),
            ),
        )
        seen[row, col] = True

    selected_count = 0
    neighbor_offsets = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )
    while heap and selected_count < requested:
        _, _, row, col = heapq.heappop(heap)
        if not legal[row, col] or not source_component_state[row, col]:
            continue
        if protected_source_necks is not None and protected_source_necks[row, col]:
            counters["source_connectivity"] += 1
            continue
        if already_deleted_from_source + selected_count >= maximum_source_deletions:
            counters["source_retention"] += 1
            break

        reason = _topology_rejection_reason(
            row,
            col,
            target_state=target_state,
            unselected_target=unselected_target,
        )
        if reason is not None:
            counters[reason] += 1
            continue

        selected[row, col] = True
        source_component_state[row, col] = False
        target_state[row, col] = True
        selected_count += 1
        for row_offset, col_offset in neighbor_offsets:
            next_row, next_col = row + row_offset, col + col_offset
            if not (
                0 <= next_row < legal.shape[0]
                and 0 <= next_col < legal.shape[1]
            ):
                continue
            if seen[next_row, next_col] or not legal[next_row, next_col]:
                continue
            seen[next_row, next_col] = True
            heapq.heappush(
                heap,
                (
                    float(priority[next_row, next_col]),
                    float(rng.random()),
                    int(next_row),
                    int(next_col),
                ),
            )
    return selected, _audit(requested, selected_count, counters)


def source_deletion_limit(
    component_area: int,
    *,
    maximum_changed_fraction: float,
    minimum_remaining_pixels: int,
) -> int:
    """Return the shared hard deletion ceiling for one source component."""

    area = max(0, int(component_area))
    fraction_limit = int(np.floor(area * float(maximum_changed_fraction)))
    retention_limit = max(0, area - int(minimum_remaining_pixels))
    return max(0, min(fraction_limit, retention_limit))


def protected_narrow_necks(
    component: np.ndarray, *, maximum_medial_radius_px: float = 3.0
) -> np.ndarray:
    """Conservatively protect thin medial bridges before front growth.

    The final topology gate remains authoritative.  This raster guard prevents
    the generator from spending its area budget through obvious narrow necks,
    which is both faster and more stable than discovering every severed bridge
    only after a complete candidate has been drawn.
    """

    region = np.asarray(component, dtype=bool)
    if not np.any(region):
        return np.zeros_like(region, dtype=bool)
    distance = ndimage.distance_transform_edt(region)
    local_maximum = distance >= (
        ndimage.maximum_filter(distance, size=3, mode="constant") - 1e-6
    )
    thin_medial = (
        region
        & local_maximum
        & (distance > 0)
        & (distance <= float(maximum_medial_radius_px))
    )
    protected = ndimage.binary_dilation(
        thin_medial,
        structure=np.ones((3, 3), dtype=bool),
        iterations=max(1, int(np.ceil(maximum_medial_radius_px))),
    )
    return protected & region


def _topology_rejection_reason(
    row: int,
    col: int,
    *,
    target_state: np.ndarray,
    unselected_target: np.ndarray,
) -> str | None:
    target_patch = _patch3(target_state, row, col, outside=False)
    target_patch[1, 1] = False
    if not np.any(target_patch):
        return "target_island"

    unselected_patch = _patch3(unselected_target, row, col, outside=False)
    unselected_patch[1, 1] = False
    if np.any(unselected_patch):
        return "unselected_target_contact"
    return None


def _patch3(array: np.ndarray, row: int, col: int, *, outside: bool) -> np.ndarray:
    result = np.full((3, 3), outside, dtype=bool)
    row_start = max(0, row - 1)
    row_stop = min(array.shape[0], row + 2)
    col_start = max(0, col - 1)
    col_stop = min(array.shape[1], col + 2)
    result[
        1 - (row - row_start) : 1 + (row_stop - row),
        1 - (col - col_start) : 1 + (col_stop - col),
    ] = array[row_start:row_stop, col_start:col_stop]
    return result


def _binary_dilation_8(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    result = np.zeros_like(mask, dtype=bool)
    for row_offset in range(3):
        for col_offset in range(3):
            result |= padded[
                row_offset : row_offset + mask.shape[0],
                col_offset : col_offset + mask.shape[1],
            ]
    return result


def _audit(
    requested: int, realized: int, counters: dict[str, int]
) -> TopologyGrowAudit:
    return TopologyGrowAudit(
        requested_pixels=int(requested),
        realized_pixels=int(realized),
        rejected_source_connectivity=int(counters["source_connectivity"]),
        rejected_source_hole_change=int(counters["source_hole_change"]),
        rejected_target_hole_change=int(counters["target_hole_change"]),
        rejected_target_island=int(counters["target_island"]),
        rejected_unselected_target_contact=int(
            counters["unselected_target_contact"]
        ),
        rejected_source_retention=int(counters["source_retention"]),
    )

"""Topology-preserving deterministic front growth for mask-edit refine."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from functools import lru_cache

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
    allow_source_component_resolution: bool = False,
    allow_target_hole_resolution: bool = False,
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
    deferred = np.zeros_like(legal, dtype=bool)
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
            source_component_state=source_component_state,
            target_state=target_state,
            unselected_target=unselected_target,
            allow_source_component_resolution=(
                allow_source_component_resolution
            ),
            allow_target_hole_resolution=allow_target_hole_resolution,
        )
        if reason is not None:
            counters[reason] += 1
            if reason != "unselected_target_contact":
                # Digital simple-point status depends only on this 3x3
                # neighborhood. Defer the pixel until (and only until) an
                # adjacent accepted pixel changes that neighborhood. This is
                # the fixed-point behavior the old multi-call compiler relied
                # on, without repeatedly rescanning every rejected pixel.
                deferred[row, col] = True
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
            if deferred[next_row, next_col]:
                deferred[next_row, next_col] = False
                heapq.heappush(
                    heap,
                    (
                        float(priority[next_row, next_col]),
                        float(rng.random()),
                        int(next_row),
                        int(next_col),
                    ),
                )
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
    source_component_state: np.ndarray,
    target_state: np.ndarray,
    unselected_target: np.ndarray,
    allow_source_component_resolution: bool = False,
    allow_target_hole_resolution: bool = False,
) -> str | None:
    """Return the first local digital-topology violation for one conversion.

    The previous grower only required a new target pixel to touch *some*
    target.  Consequently a legal-looking distance band could cut a source
    corridor or let two arms of the same target wrap around a residual island;
    the expensive whole-mask audit discovered the split/hole only after the
    complete area had been drawn.  A pixel is now accepted only when it is a
    simple point for both sides of the source->target transition under the
    same 8-connected foreground / 4-connected background convention used by
    the whole-mask gates.

    Target-component merges remain an explicit Planner/gate capability, but
    this low-level grower deliberately does not create them.  Independent
    fronts may approach one another while retaining a one-pixel legal
    corridor.  A separate, auditable merge program can be added when a skill
    positively requests coalescence; silently merging during generic burden
    growth is not safe.
    """

    if not source_component_state[row, col]:
        return "source_connectivity"
    source_pattern = _neighbor_pattern_at(
        source_component_state, row, col, outside=False
    )
    source_neighbor_components = _cached_local_component_count(
        source_pattern, 8, False
    )
    if source_neighbor_components != 1 and not (
        allow_source_component_resolution and source_neighbor_components == 0
    ):
        return "source_connectivity"

    # Removing a source pixel adds one background pixel.  Zero adjacent
    # background components creates a new source hole; more than one joins
    # previously separate background regions and removes a protected hole.
    source_background_pattern = (~source_pattern) & 0xFF
    source_background_components = _cached_local_component_count(
        source_background_pattern, 4, True
    )
    # Zero background neighbours means removing the center creates a new
    # source hole and is never legal. More than one joins existing background
    # regions and therefore resolves a source hole; that is legal only for an
    # explicitly authorized source-component resolution primitive.
    if source_background_components == 0 or (
        source_background_components > 1
        and not allow_source_component_resolution
    ):
        return "source_hole_change"

    target_pattern = _neighbor_pattern_at(target_state, row, col, outside=False)
    target_neighbor_components = _cached_local_component_count(
        target_pattern, 8, False
    )
    if target_neighbor_components == 0:
        return "target_island"
    if target_neighbor_components != 1 and not allow_target_hole_resolution:
        return "target_hole_change"

    # Adding a target pixel removes one background pixel.  If that pixel is a
    # local articulation of target background, the new target front closes a
    # ring and creates a target hole.  If it is an isolated background pixel,
    # filling it silently removes a pre-existing target hole.  Both are
    # forbidden by the generic topology contract.
    target_background_pattern = (~target_pattern) & 0xFF
    target_background_components = _cached_local_component_count(
        target_background_pattern, 4, True
    )
    # Zero background neighbours removes an existing one-pixel target hole;
    # more than one splits background and creates a new target hole. The
    # latter remains forbidden even when hole *resolution* is authorized.
    if target_background_components > 1 or (
        target_background_components == 0
        and not allow_target_hole_resolution
    ):
        return "target_hole_change"

    if _neighbor_pattern_at(unselected_target, row, col, outside=False):
        return "unselected_target_contact"
    return None


def _neighbor_component_count(mask: np.ndarray, *, connectivity: int) -> int:
    """Count foreground components in a 3x3 neighborhood without its center."""

    return _cached_local_component_count(
        _neighbor_pattern(mask), connectivity, False
    )


def _center_adjacent_component_count(
    mask: np.ndarray, *, connectivity: int
) -> int:
    """Count neighbor components that the center would join.

    For 4-connectivity, diagonal-only regions are intentionally ignored: a
    newly added center pixel does not connect to them in the topology model.
    """

    return _cached_local_component_count(
        _neighbor_pattern(mask), connectivity, True
    )


_NEIGHBOR_POSITIONS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
)

_NEIGHBOR_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _neighbor_pattern_at(
    array: np.ndarray, row: int, col: int, *, outside: bool
) -> int:
    pattern = 0
    height, width = array.shape
    for bit, (row_offset, col_offset) in enumerate(_NEIGHBOR_OFFSETS):
        current_row = row + row_offset
        current_col = col + col_offset
        value = (
            bool(array[current_row, current_col])
            if 0 <= current_row < height and 0 <= current_col < width
            else outside
        )
        pattern |= int(value) << bit
    return pattern


def _neighbor_pattern(mask: np.ndarray) -> int:
    neighborhood = np.asarray(mask, dtype=bool)
    pattern = 0
    for bit, (row, col) in enumerate(_NEIGHBOR_POSITIONS):
        pattern |= int(bool(neighborhood[row, col])) << bit
    return pattern


@lru_cache(maxsize=1024)
def _cached_local_component_count(
    pattern: int, connectivity: int, center_adjacent_only: bool
) -> int:
    neighborhood = np.zeros((3, 3), dtype=bool)
    for bit, (row, col) in enumerate(_NEIGHBOR_POSITIONS):
        neighborhood[row, col] = bool(pattern & (1 << bit))
    structure = (
        np.ones((3, 3), dtype=bool)
        if connectivity == 8
        else np.asarray(
            [[False, True, False], [True, True, True], [False, True, False]],
            dtype=bool,
        )
    )
    labeled, count = ndimage.label(neighborhood, structure=structure)
    if not center_adjacent_only:
        return int(count)
    positions = (
        _NEIGHBOR_POSITIONS
        if connectivity == 8
        else ((0, 1), (1, 0), (1, 2), (2, 1))
    )
    return len(
        {
            int(labeled[position])
            for position in positions
            if int(labeled[position]) > 0
        }
    )


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

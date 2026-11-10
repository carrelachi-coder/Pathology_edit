"""Unwired research candidate for subdividing one directional interface anchor.

This module is deliberately not imported by the production scene builder or
planner.  It supports local synthetic evaluation of a possible way to retain
an unedited boundary witness on interfaces that currently have one anchor.
Any future integration must rebuild the scene and rerun nuclei preflight; an
old feasible-anchor result is not valid for the returned subanchors.
"""

from __future__ import annotations

from collections import deque

import numpy as np


CANDIDATE_VERSION = "directional-single-anchor-geodesic-bisection-v1"


def bisect_connected_interface_anchor(
    anchor_mask: np.ndarray,
    *,
    minimum_subanchor_pixels: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic, exhaustive 8-connected geodesic bisection.

    The two seeds are the lexicographically first mask pixel and the pixel
    farthest from it along the interface graph.  Equal-distance pixels are
    assigned to the first seed.  The function fails closed if either result
    is too small or disconnected.
    """

    mask = np.asarray(anchor_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("anchor mask must be two-dimensional")
    coordinates = np.argwhere(mask)
    if coordinates.shape[0] < 2 * int(minimum_subanchor_pixels):
        raise ValueError("anchor is too small for the requested bisection")

    first_seed = tuple(int(value) for value in coordinates[0])
    first_distance = _geodesic_distance(mask, first_seed)
    farthest_value = int(np.max(first_distance[mask]))
    farthest_coordinates = np.argwhere(mask & (first_distance == farthest_value))
    second_seed = tuple(int(value) for value in farthest_coordinates[0])
    second_distance = _geodesic_distance(mask, second_seed)

    first = mask & (first_distance <= second_distance)
    second = mask & ~first
    parts = (first, second)
    if any(np.count_nonzero(part) < minimum_subanchor_pixels for part in parts):
        raise ValueError("geodesic bisection produced an undersized subanchor")
    if not np.array_equal(first | second, mask) or np.any(first & second):
        raise RuntimeError("geodesic bisection is not an exact partition")
    for part in parts:
        _geodesic_distance(part, tuple(int(value) for value in np.argwhere(part)[0]))
    return parts


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

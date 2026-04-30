"""Small NumPy morphology helpers shared by Phase 3 primitives."""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy import ndimage


def binary_dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Dilate a 2D boolean mask with a square structuring element."""

    mask_bool = _as_2d_bool(mask)
    _validate_radius(radius)
    if radius == 0:
        return mask_bool.copy()

    padded = np.pad(mask_bool, radius, mode="constant", constant_values=False)
    result = np.zeros_like(mask_bool, dtype=bool)
    size = 2 * radius + 1
    for row_offset in range(size):
        for col_offset in range(size):
            result |= padded[
                row_offset : row_offset + mask_bool.shape[0],
                col_offset : col_offset + mask_bool.shape[1],
            ]
    return result


def binary_erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Erode a 2D boolean mask with a square structuring element."""

    mask_bool = _as_2d_bool(mask)
    _validate_radius(radius)
    if radius == 0:
        return mask_bool.copy()

    padded = np.pad(mask_bool, radius, mode="constant", constant_values=False)
    result = np.ones_like(mask_bool, dtype=bool)
    size = 2 * radius + 1
    for row_offset in range(size):
        for col_offset in range(size):
            result &= padded[
                row_offset : row_offset + mask_bool.shape[0],
                col_offset : col_offset + mask_bool.shape[1],
            ]
    return result


def boundary_ring(
    source_mask: np.ndarray, candidate_mask: np.ndarray, radius: int = 1
) -> np.ndarray:
    """Return the outward source boundary ring restricted to candidates."""

    source = _as_2d_bool(source_mask)
    candidate = _as_2d_bool(candidate_mask)
    if source.shape != candidate.shape:
        raise ValueError("source_mask and candidate_mask must have the same shape.")

    return binary_dilate(source, radius=radius) & ~source & candidate


def select_boundary_band_by_fraction(
    source_mask: np.ndarray,
    candidate_mask: np.ndarray,
    target_fraction: float,
    min_radius: int = 1,
    max_radius: int = 128,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Search outward boundary-band radius whose area best matches target."""

    source = _as_2d_bool(source_mask)
    candidate = _as_2d_bool(candidate_mask)
    if source.shape != candidate.shape:
        raise ValueError("source_mask and candidate_mask must have the same shape.")
    _validate_radius(min_radius)
    _validate_radius(max_radius)
    if min_radius < 1:
        raise ValueError("min_radius must be >= 1.")
    if max_radius < min_radius:
        raise ValueError("max_radius must be >= min_radius.")

    target_count = _target_count_for_fraction(candidate, target_fraction)
    distance_to_source = ndimage.distance_transform_cdt(~source, metric="chessboard")
    candidate_distances = distance_to_source[candidate & (distance_to_source > 0)]
    if candidate_distances.size == 0:
        info: dict[str, float | int | bool] = {
            "radius": min_radius,
            "target_pixels": target_count,
            "selected_pixels": 0,
            "actual_fraction": 0.0,
            "candidate_shortfall": target_count > 0,
        }
        return np.zeros_like(candidate, dtype=bool), info

    best_region = np.zeros_like(candidate, dtype=bool)
    best_radius = min_radius
    best_error: int | None = None

    for radius in range(min_radius, max_radius + 1):
        region = (distance_to_source <= radius) & (distance_to_source > 0) & candidate
        selected_pixels = int(region.sum())
        error = abs(selected_pixels - target_count)
        if best_error is None or error < best_error:
            best_error = error
            best_region = region
            best_radius = radius
        if error == 0:
            break

    selected_pixels = int(best_region.sum())
    info: dict[str, float | int | bool] = {
        "radius": best_radius,
        "target_pixels": target_count,
        "selected_pixels": selected_pixels,
        "actual_fraction": selected_pixels / candidate.size,
        "candidate_shortfall": selected_pixels < target_count,
    }
    return best_region, info


def select_region_by_fraction(
    candidate_mask: np.ndarray, target_fraction: float, seed: int | None = None
) -> np.ndarray:
    """Select candidate pixels covering a target fraction of the full patch."""

    candidate = _as_2d_bool(candidate_mask)
    if not isinstance(target_fraction, (int, float)):
        raise ValueError("target_fraction must be numeric.")
    target_fraction = float(target_fraction)
    if not 0.0 <= target_fraction <= 1.0:
        raise ValueError("target_fraction must be in [0, 1].")

    candidate_indices = np.argwhere(candidate)
    selected = np.zeros_like(candidate, dtype=bool)
    if candidate_indices.size == 0 or target_fraction == 0:
        return selected

    target_count = int(round(candidate.size * target_fraction))
    target_count = max(1, target_count)
    target_count = min(target_count, len(candidate_indices))

    rng = np.random.default_rng(seed)
    chosen_positions = rng.choice(len(candidate_indices), size=target_count, replace=False)
    chosen_indices = candidate_indices[chosen_positions]
    selected[chosen_indices[:, 0], chosen_indices[:, 1]] = True
    return selected


def select_connected_region_by_fraction(
    candidate_mask: np.ndarray,
    seed_mask: np.ndarray,
    target_fraction: float,
    seed_value: int | None = None,
) -> np.ndarray:
    """Grow one connected region inside candidates from a seed boundary."""

    candidate = _as_2d_bool(candidate_mask)
    seed = _as_2d_bool(seed_mask)
    if candidate.shape != seed.shape:
        raise ValueError("candidate_mask and seed_mask must have the same shape.")
    target_count = _target_count_for_fraction(candidate, target_fraction)

    valid_seed = candidate & seed
    seed_indices = np.argwhere(valid_seed)
    if seed_indices.size == 0:
        raise ValueError("seed_mask must overlap candidate_mask.")

    rng = np.random.default_rng(seed_value)
    start = tuple(seed_indices[rng.integers(0, len(seed_indices))])

    selected = np.zeros_like(candidate, dtype=bool)
    seen = np.zeros_like(candidate, dtype=bool)
    queue: deque[tuple[int, int]] = deque([start])
    seen[start] = True

    while queue and int(selected.sum()) < target_count:
        row, col = queue.popleft()
        if not candidate[row, col]:
            continue
        selected[row, col] = True

        neighbors = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ]
        rng.shuffle(neighbors)
        for next_row, next_col in neighbors:
            if not (0 <= next_row < candidate.shape[0] and 0 <= next_col < candidate.shape[1]):
                continue
            if seen[next_row, next_col] or not candidate[next_row, next_col]:
                continue
            seen[next_row, next_col] = True
            queue.append((next_row, next_col))

    return selected


def _as_2d_bool(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("mask must be a 2D array.")
    return array.astype(bool, copy=False)


def _validate_radius(radius: int) -> None:
    if not isinstance(radius, int) or radius < 0:
        raise ValueError("radius must be a non-negative integer.")


def _target_count_for_fraction(candidate: np.ndarray, target_fraction: float) -> int:
    if not isinstance(target_fraction, (int, float)):
        raise ValueError("target_fraction must be numeric.")
    target_fraction = float(target_fraction)
    if not 0.0 <= target_fraction <= 1.0:
        raise ValueError("target_fraction must be in [0, 1].")
    if not np.any(candidate) or target_fraction == 0:
        return 0
    target_count = int(round(candidate.size * target_fraction))
    return max(1, target_count)

"""Small NumPy morphology helpers shared by Phase 3 primitives."""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np
from scipy import ndimage


# ── binary dilation / erosion ──────────────────────────────────────

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


# ── SDF / distance map ────────────────────────────────────────────

def signed_distance_field(
    mask: np.ndarray, metric: str = "euclidean"
) -> np.ndarray:
    """Compute signed distance field of a 2D boolean mask.

    Positive values outside the mask, negative inside.
    """

    arr = _as_2d_bool(mask)
    if metric == "euclidean":
        outside_dist = ndimage.distance_transform_edt(~arr)
        inside_dist = ndimage.distance_transform_edt(arr)
    elif metric == "chessboard":
        outside_dist = ndimage.distance_transform_cdt(~arr, metric="chessboard")
        inside_dist = ndimage.distance_transform_cdt(arr, metric="chessboard")
    elif metric == "taxicab":
        outside_dist = ndimage.distance_transform_cdt(~arr, metric="taxicab")
        inside_dist = ndimage.distance_transform_cdt(arr, metric="taxicab")
    else:
        raise ValueError(f"unknown metric: {metric}. Use 'euclidean', 'chessboard', or 'taxicab'.")

    sdf = outside_dist.copy()
    sdf[arr] = -inside_dist[arr]
    return sdf


def distance_to_boundary(
    mask: np.ndarray, metric: str = "euclidean"
) -> np.ndarray:
    """Compute unsigned distance to the boundary of a 2D boolean mask.

    Every pixel gets the Euclidean (or chess/taxicab) distance to the
    nearest boundary pixel of the mask, regardless of being inside or
    outside.
    """

    sdf = signed_distance_field(mask, metric=metric)
    return np.abs(sdf)


def distance_to_label(
    id_mask: np.ndarray, target_ids: Sequence[int], metric: str = "euclidean"
) -> np.ndarray:
    """Compute distance from every pixel to the nearest target label pixel.

    Args:
        id_mask: 2D int mask with fine ids.
        target_ids: label ids whose pixels define the distance origin.
        metric: 'euclidean', 'chessboard', or 'taxicab'.

    Returns:
        (H, W) float64 distance map. Pixels inside target_ids have distance 0.
    """

    arr = np.asarray(id_mask)
    if arr.ndim != 2:
        raise ValueError("id_mask must be 2D.")
    target = np.isin(arr, target_ids)
    if metric == "euclidean":
        return ndimage.distance_transform_edt(~target)
    elif metric == "chessboard":
        return ndimage.distance_transform_cdt(~target, metric="chessboard").astype(float)
    elif metric == "taxicab":
        return ndimage.distance_transform_cdt(~target, metric="taxicab").astype(float)
    raise ValueError(f"unknown metric: {metric}.")


# ── multi-scale smooth noise ───────────────────────────────────────

def multi_scale_smooth_noise(
    shape: tuple[int, int],
    scales: tuple[float, ...] = (2.0, 8.0, 32.0),
    amplitudes: tuple[float, ...] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Generate multi-scale smooth noise as a weighted sum of Gaussians.

    Each scale produces a Gaussian-blurred random field at that sigma.
    The sum gives a noise pattern with both large-scale structure and
    fine detail, suitable for irregular boundary growth.
    """

    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError("shape must be a 2-element tuple.")
    if len(scales) == 0:
        raise ValueError("scales must be non-empty.")

    if amplitudes is None:
        amplitudes = tuple(1.0 / len(scales) for _ in scales)
    if not isinstance(amplitudes, (tuple, list)):
        raise ValueError("amplitudes must be a tuple or list.")
    if len(amplitudes) != len(scales):
        raise ValueError("amplitudes and scales must have the same length.")

    rng = np.random.default_rng(seed)
    result = np.zeros(shape, dtype=float)

    for sigma, amp in zip(scales, amplitudes):
        raw = rng.standard_normal(shape)
        blurred = ndimage.gaussian_filter(raw, sigma=sigma)
        result += amp * blurred

    return result


# ── island generation ──────────────────────────────────────────────

def generate_islands(
    candidate_mask: np.ndarray,
    source_mask: np.ndarray,
    *,
    max_distance_px: int,
    max_island_area_px: int,
    max_islands: int,
    target_fraction: float,
    seed: int | None = None,
) -> tuple[np.ndarray, dict[str, int | bool]]:
    """Generate small island patches near the source mask within candidates.

    Each island is a connected region grown from a seed point. Islands
    are constrained by distance from source, maximum area, and maximum
    count. Used for tumor budding in boundary_infiltration.
    """

    candidate = _as_2d_bool(candidate_mask)
    source = _as_2d_bool(source_mask)
    if candidate.shape != source.shape:
        raise ValueError("candidate_mask and source_mask must have the same shape.")

    rng = np.random.default_rng(seed)

    dist_to_source = ndimage.distance_transform_edt(~source)
    eligible = candidate & (dist_to_source <= max_distance_px) & ~source
    eligible_indices = np.argwhere(eligible)
    if eligible_indices.size == 0:
        return np.zeros_like(candidate, dtype=bool), {
            "islands_generated": 0,
            "total_island_pixels": 0,
            "target_fraction_shortfall": target_fraction > 0,
        }

    target_pixels = max(1, int(round(candidate.size * target_fraction)))
    total_island_pixels = 0
    islands_generated = 0
    result = np.zeros_like(candidate, dtype=bool)
    used_seeds = np.zeros_like(candidate, dtype=bool)

    max_attempts = max_islands * 5
    for _ in range(max_attempts):
        if islands_generated >= max_islands:
            break
        if total_island_pixels >= target_pixels:
            break

        remaining = eligible & ~used_seeds & ~result
        remaining_indices = np.argwhere(remaining)
        if remaining_indices.size == 0:
            break

        idx = rng.integers(0, len(remaining_indices))
        seed_point = tuple(remaining_indices[idx])

        island = _grow_single_island(
            candidate=eligible,
            seed_point=seed_point,
            max_area=max_island_area_px,
            rng=rng,
            existing=result,
        )
        island_area = int(np.count_nonzero(island))
        if island_area < 1:
            continue

        result |= island
        used_seeds |= binary_dilate(island, radius=2)
        total_island_pixels += island_area
        islands_generated += 1

    return result, {
        "islands_generated": islands_generated,
        "total_island_pixels": total_island_pixels,
        "target_fraction_shortfall": total_island_pixels < target_pixels,
    }


def _grow_single_island(
    *,
    candidate: np.ndarray,
    seed_point: tuple[int, int],
    max_area: int,
    rng: np.random.default_rng,
    existing: np.ndarray,
) -> np.ndarray:
    """Grow one connected island from a seed point up to max_area pixels."""

    island = np.zeros_like(candidate, dtype=bool)
    row, col = seed_point
    if not candidate[row, col]:
        return island

    queue: deque[tuple[int, int]] = deque([(row, col)])
    seen = np.zeros_like(candidate, dtype=bool)
    seen[row, col] = True
    area = 0

    while queue and area < max_area:
        r, c = queue.popleft()
        if not candidate[r, c] or existing[r, c]:
            continue
        island[r, c] = True
        area += 1

        neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        rng.shuffle(neighbors)
        for nr, nc in neighbors:
            if not (0 <= nr < candidate.shape[0] and 0 <= nc < candidate.shape[1]):
                continue
            if seen[nr, nc]:
                continue
            seen[nr, nc] = True
            queue.append((nr, nc))

    return island


# ── topology cleanup ───────────────────────────────────────────────

def remove_small_components(
    mask: np.ndarray, min_area_px: int = 1, structure: np.ndarray | None = None
) -> np.ndarray:
    """Remove connected components smaller than min_area_px."""

    arr = _as_2d_bool(mask)
    if min_area_px < 1:
        return arr.copy()
    if structure is None:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

    labeled, count = ndimage.label(arr, structure=structure)
    if count == 0:
        return arr.copy()

    component_sizes = ndimage.sum(arr, labeled, range(1, count + 1))
    keep = np.zeros_like(arr, dtype=bool)
    for component_id, size in zip(range(1, count + 1), component_sizes):
        if int(size) >= min_area_px:
            keep |= labeled == component_id

    return keep


def fill_small_holes(
    mask: np.ndarray, max_hole_area_px: int = 1, structure: np.ndarray | None = None
) -> np.ndarray:
    """Fill holes (connected False regions) inside a True mask up to max area."""

    arr = _as_2d_bool(mask)
    if max_hole_area_px < 1:
        return arr.copy()
    if structure is None:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

    inverted = ~arr
    labeled, count = ndimage.label(inverted, structure=structure)
    if count == 0:
        return arr.copy()

    component_sizes = ndimage.sum(inverted, labeled, range(1, count + 1))
    result = arr.copy()
    for component_id, size in zip(range(1, count + 1), component_sizes):
        if int(size) <= max_hole_area_px:
            hole = labeled == component_id
            result[hole] = True

    return result


def keep_only_touching(
    components_mask: np.ndarray,
    context_mask: np.ndarray,
    structure: np.ndarray | None = None,
) -> np.ndarray:
    """Keep only connected components that touch the context mask."""

    comp = _as_2d_bool(components_mask)
    ctx = _as_2d_bool(context_mask)
    if structure is None:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

    labeled, count = ndimage.label(comp, structure=structure)
    if count == 0:
        return np.zeros_like(comp, dtype=bool)

    touching = ndimage.binary_dilation(ctx, structure=structure)
    kept = np.zeros_like(comp, dtype=bool)
    for component_id in range(1, count + 1):
        component = labeled == component_id
        if np.any(component & touching):
            kept |= component

    return kept


# ── nearest / context backfill ─────────────────────────────────────

def nearest_label_backfill(
    id_mask: np.ndarray,
    source_labels: Sequence[int],
    change_region: np.ndarray,
) -> np.ndarray:
    """Assign each change-region pixel the fine id of its nearest source pixel.

    Uses Euclidean distance transform to find the closest source label
    pixel for each changed pixel, then copies that pixel's id.
    """

    arr = np.asarray(id_mask)
    if arr.ndim != 2:
        raise ValueError("id_mask must be 2D.")
    change = _as_2d_bool(change_region)
    if arr.shape != change.shape:
        raise ValueError("id_mask and change_region must have the same shape.")

    source_mask = np.isin(arr, source_labels)
    if not np.any(source_mask):
        raise ValueError("source_labels not found in id_mask.")

    _, nearest_indices = ndimage.distance_transform_edt(
        ~source_mask, return_indices=True
    )
    row_indices, col_indices = nearest_indices
    nearest_ids = arr[row_indices, col_indices]
    return nearest_ids[change]


# ── internal helpers ───────────────────────────────────────────────

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
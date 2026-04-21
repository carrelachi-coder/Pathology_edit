"""Synthetic inpaint metadata builder for Phase 5."""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image
from dataset_config import get_config

from .common import (
    default_prompt_for_dataset,
    load_layered_dataset_samples,
    load_mask_array,
    split_records_by_case,
    write_jsonl,
)


_VALID_GEOMETRY_BUCKETS = {"expand_band", "shrink_band", "replace_like_blob"}
_VALID_FORCED_MODES = {"identity", "near_identity", "mixed"} | _VALID_GEOMETRY_BUCKETS
_VALID_SIZE_BUCKETS = {"identity", "small", "medium", "large"}
_MODE_WEIGHTS = {
    "identity": 10,
    "near_identity": 10,
    "expand_band": 30,
    "shrink_band": 25,
    "replace_like_blob": 25,
}
_GEOMETRY_SIZE_BUCKET_WEIGHTS = {
    "small": 70,
    "large": 30,
}
_MIN_GEOMETRY_THICKNESS = {
    "small": 90,
    "medium": 130,
    "large": 180,
}
_GEOMETRY_ARC_RADIUS = {
    "small": 90,
    "medium": 120,
    "large": 150,
}
_TARGET_CHANGE_RATIO_RANGES = {
    "small": (0.08, 0.14),
    "medium": (0.14, 0.24),
    "large": (0.24, 0.40),
}
_REPLACE_COMPONENT_FRACTION_RANGES = {
    "small": (0.55, 0.70),
    "medium": (0.70, 0.82),
    "large": (0.82, 0.92),
}
_SIZE_BUCKET_ORDER = {
    "identity": 0,
    "small": 1,
    "medium": 2,
    "large": 3,
}
_MAX_CHANGE_RATIO = 0.70
_PATCH_FRAME_MARGIN = 1

_DEFAULT_NEAR_IDENTITY_CHANGE_PIXELS = 1


@dataclass(frozen=True)
class _SyntheticInpaintConfig:
    forced_mode: str
    forced_bucket: str | None = None
    seed: int = 42
    near_identity_change_pixels: int = _DEFAULT_NEAR_IDENTITY_CHANGE_PIXELS


def _sample_effective_mode(
    *,
    config: _SyntheticInpaintConfig,
    sample,
    attempt_seed: int | None,
    variant_index: int = 0,
    excluded_modes: tuple[str, ...] = (),
) -> str:
    if config.forced_mode != "mixed":
        return config.forced_mode

    candidate_modes = _candidate_modes_for_config(config)
    filtered_modes = [mode for mode in candidate_modes if mode not in excluded_modes]
    if filtered_modes:
        candidate_modes = filtered_modes

    weights = [_MODE_WEIGHTS[mode] for mode in candidate_modes]
    seed_value = (
        f"{config.seed}::{sample.dataset_name}::{sample.sample_id}::{attempt_seed}::{variant_index}"
    )
    rng = random.Random(seed_value)
    return rng.choices(candidate_modes, weights=weights, k=1)[0]


def _candidate_modes_for_config(config: _SyntheticInpaintConfig) -> list[str]:
    if config.forced_mode != "mixed":
        return [config.forced_mode]

    if config.forced_bucket == "identity":
        return ["identity"]

    if config.forced_bucket == "small":
        return ["near_identity", *sorted(_VALID_GEOMETRY_BUCKETS)]
    if config.forced_bucket in {"medium", "large"}:
        return sorted(_VALID_GEOMETRY_BUCKETS)
    return list(_MODE_WEIGHTS.keys())


def _normalize_binary_mask(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask) > 0


def _iter_neighborhood(y: int, x: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = shape
    neighbors: list[tuple[int, int]] = []
    if y > 0:
        neighbors.append((y - 1, x))
    if y + 1 < height:
        neighbors.append((y + 1, x))
    if x > 0:
        neighbors.append((y, x - 1))
    if x + 1 < width:
        neighbors.append((y, x + 1))
    return neighbors


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = _normalize_binary_mask(mask)
    visited = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue

        component = np.zeros_like(mask, dtype=bool)
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True

        while stack:
            y, x = stack.pop()
            component[y, x] = True
            for ny, nx in _iter_neighborhood(y, x, mask.shape):
                if mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))

        components.append(component)

    return components


def _select_single_component(
    mask: np.ndarray,
    seed: int | None = None,
    preferred_labels: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Pick one connected component from a possibly multi-component tissue mask.

    The public geometry helpers intentionally work on a single component even if
    the caller passes a full tissue mask with several disjoint regions. When the
    mask carries multiple tissue labels, we first choose the dominant nonzero
    label so structured synthesis stays tissue-aware instead of collapsing all
    foreground labels together. Within that label, we choose the largest
    connected component in discovery order; if several components tie for the
    largest area, ``seed`` controls which tied component is selected.
    """
    raw_mask = np.asarray(mask)
    foreground_labels = [int(label) for label in np.unique(raw_mask) if int(label) > 0]
    if not foreground_labels:
        return np.zeros_like(raw_mask, dtype=bool)

    if preferred_labels:
        preferred = [label for label in foreground_labels if label in set(preferred_labels)]
        if preferred:
            foreground_labels = preferred

    label_areas = {label: int(np.count_nonzero(raw_mask == label)) for label in foreground_labels}
    max_area = max(label_areas.values())
    dominant_labels = [label for label, area in label_areas.items() if area == max_area]
    if len(dominant_labels) == 1 or seed is None:
        selected_label = dominant_labels[0]
    else:
        rng = np.random.default_rng(seed)
        selected_label = dominant_labels[int(rng.integers(0, len(dominant_labels)))]

    components = _connected_components(raw_mask == selected_label)
    if not components:
        return np.zeros_like(raw_mask, dtype=bool)
    components.sort(key=lambda component: int(component.sum()), reverse=True)
    if seed is None or len(components) == 1:
        return components[0]
    rng = np.random.default_rng(seed)
    top_components = [component for component in components if component.sum() == components[0].sum()]
    return top_components[int(rng.integers(0, len(top_components)))]


def _dilate(mask: np.ndarray, steps: int = 1) -> np.ndarray:
    result = _normalize_binary_mask(mask)
    for _ in range(max(steps, 0)):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
        )
    return result


def _erode(mask: np.ndarray, steps: int = 1) -> np.ndarray:
    result = _normalize_binary_mask(mask)
    for _ in range(max(steps, 0)):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
    return result


def _component_boundary(component: np.ndarray) -> np.ndarray:
    component = _normalize_binary_mask(component)
    if not component.any():
        return np.zeros_like(component, dtype=bool)
    return component & ~_erode(component, steps=1)


def _frame_mask(shape: tuple[int, int], margin: int = _PATCH_FRAME_MARGIN) -> np.ndarray:
    frame = np.zeros(shape, dtype=bool)
    margin = max(1, margin)
    frame[:margin, :] = True
    frame[-margin:, :] = True
    frame[:, :margin] = True
    frame[:, -margin:] = True
    return frame


def _valid_component_boundary(component: np.ndarray) -> np.ndarray:
    return _component_boundary(component) & ~_frame_mask(component.shape)


def _component_core(component: np.ndarray, seed: int | None = None) -> np.ndarray:
    component = _normalize_binary_mask(component)
    if not component.any():
        return np.zeros_like(component, dtype=bool)

    core = _erode(component, steps=1)
    if core.any():
        return core

    coords = np.argwhere(component)
    if len(coords) == 0:
        return np.zeros_like(component, dtype=bool)
    if len(coords) == 1:
        return np.zeros_like(component, dtype=bool)

    centroid = coords.mean(axis=0)
    distances = np.sum((coords - centroid) ** 2, axis=1)
    min_distance = float(distances.min())
    tied_indices = np.flatnonzero(distances == min_distance)
    rng = np.random.default_rng(seed)
    chosen_index = int(tied_indices[int(rng.integers(0, len(tied_indices)))])
    core_mask = np.zeros_like(component, dtype=bool)
    y, x = map(int, coords[chosen_index])
    core_mask[y, x] = True
    return core_mask


def _sample_geometry_size_bucket(seed: int | None = None) -> str:
    rng = random.Random(str(seed))
    candidate_buckets = list(_GEOMETRY_SIZE_BUCKET_WEIGHTS.keys())
    weights = [_GEOMETRY_SIZE_BUCKET_WEIGHTS[bucket] for bucket in candidate_buckets]
    return rng.choices(candidate_buckets, weights=weights, k=1)[0]


def _component_bbox(component: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(component)
    if coords.size == 0:
        return None
    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)
    return int(min_y), int(min_x), int(max_y), int(max_x)


def _component_centroid(component: np.ndarray) -> tuple[float, float] | None:
    coords = np.argwhere(component)
    if coords.size == 0:
        return None
    centroid = coords.mean(axis=0)
    return float(centroid[0]), float(centroid[1])


def _ellipse_mask(
    shape: tuple[int, int],
    *,
    center_y: float,
    center_x: float,
    radius_y: float,
    radius_x: float,
) -> np.ndarray:
    if radius_y <= 0 or radius_x <= 0:
        return np.zeros(shape, dtype=bool)
    yy, xx = np.indices(shape)
    normalized = ((yy - center_y) / radius_y) ** 2 + ((xx - center_x) / radius_x) ** 2
    return normalized <= 1.0


def _component_axis_fractions(size_bucket: str) -> tuple[float, float]:
    if size_bucket == "small":
        return 0.60, 0.72
    if size_bucket == "medium":
        return 0.72, 0.85
    return 0.85, 1.00


def _sample_component_axes(
    component: np.ndarray,
    *,
    size_bucket: str,
    seed: int | None = None,
) -> tuple[float, float]:
    bbox = _component_bbox(component)
    if bbox is None:
        return 0.0, 0.0
    min_y, min_x, max_y, max_x = bbox
    height = max(1, max_y - min_y + 1)
    width = max(1, max_x - min_x + 1)
    low_fraction, high_fraction = _component_axis_fractions(size_bucket)
    rng = np.random.default_rng(seed)
    frac_y = float(rng.uniform(low_fraction, high_fraction))
    frac_x = float(rng.uniform(low_fraction, high_fraction))
    radius_y = max(1.0, height * frac_y / 2.0)
    radius_x = max(1.0, width * frac_x / 2.0)
    return radius_y, radius_x


def _sample_interior_center(component: np.ndarray, seed: int | None = None) -> tuple[int, int] | None:
    depth_map = _boundary_depth_map(component)
    max_depth = int(depth_map.max())
    if max_depth <= 0:
        return None
    threshold = max(1, int(np.ceil(max_depth * 0.6)))
    interior = depth_map >= threshold
    if not interior.any():
        interior = depth_map == max_depth
    coords = np.argwhere(interior)
    if coords.size == 0:
        coords = np.argwhere(component)
    if coords.size == 0:
        return None
    rng = np.random.default_rng(seed)
    y, x = coords[int(rng.integers(0, len(coords)))]
    return int(y), int(x)


def _sample_boundary_ellipse_mask(
    component: np.ndarray,
    *,
    size_bucket: str,
    seed: int | None = None,
) -> np.ndarray:
    valid_boundary = _valid_component_boundary(component)
    coords = np.argwhere(valid_boundary)
    if coords.size == 0:
        return _sample_interior_ellipse_mask(component, size_bucket=size_bucket, seed=seed)

    rng = np.random.default_rng(seed)
    start_y, start_x = coords[int(rng.integers(0, len(coords)))]
    radius_y, radius_x = _sample_component_axes(component, size_bucket=size_bucket, seed=seed)
    centroid = _component_centroid(component)
    if centroid is None:
        return np.zeros_like(component, dtype=bool)

    centroid_y, centroid_x = centroid
    shift_y = (centroid_y - float(start_y)) * 0.45
    shift_x = (centroid_x - float(start_x)) * 0.45
    mask = _ellipse_mask(
        component.shape,
        center_y=float(start_y) + shift_y,
        center_x=float(start_x) + shift_x,
        radius_y=radius_y,
        radius_x=radius_x,
    )
    return mask & component


def _sample_interior_ellipse_mask(
    component: np.ndarray,
    *,
    size_bucket: str,
    seed: int | None = None,
) -> np.ndarray:
    center = _sample_interior_center(component, seed=seed)
    if center is None:
        return np.zeros_like(component, dtype=bool)
    radius_y, radius_x = _sample_component_axes(component, size_bucket=size_bucket, seed=seed)
    center_y, center_x = center
    mask = _ellipse_mask(
        component.shape,
        center_y=float(center_y),
        center_x=float(center_x),
        radius_y=radius_y,
        radius_x=radius_x,
    )
    return mask & component


def _resolve_geometry_size_bucket(size_bucket: str | None, seed: int | None = None) -> str:
    if size_bucket is None:
        return _sample_geometry_size_bucket(seed)
    if size_bucket not in {"small", "medium", "large"}:
        raise ValueError(f"Unsupported geometry size bucket: {size_bucket}")
    return size_bucket


def _choose_target_change_pixels(
    *,
    shape: tuple[int, int],
    candidate_limit: int,
    size_bucket: str,
    seed: int | None = None,
) -> int:
    low_ratio, high_ratio = _TARGET_CHANGE_RATIO_RANGES[size_bucket]
    total_pixels = int(np.prod(shape))
    low_pixels = max(1, int(np.ceil(total_pixels * low_ratio)))
    high_pixels = max(low_pixels, int(np.floor(total_pixels * high_ratio)))
    low_pixels = min(low_pixels, candidate_limit)
    high_pixels = min(high_pixels, candidate_limit)
    if high_pixels <= 0:
        return 0
    if low_pixels >= high_pixels:
        return high_pixels
    rng = np.random.default_rng(seed)
    return int(rng.integers(low_pixels, high_pixels + 1))


def _boundary_depth_map(component: np.ndarray) -> np.ndarray:
    component = _normalize_binary_mask(component)
    depth = np.zeros_like(component, dtype=np.int32)
    remaining = component.copy()
    current_depth = 1
    while remaining.any():
        inner = _erode(remaining, steps=1)
        layer = remaining & ~inner
        depth[layer] = current_depth
        if np.array_equal(inner, remaining):
            break
        remaining = inner
        current_depth += 1
    return depth


def _boundary_band_for_thickness(component: np.ndarray, thickness: int) -> np.ndarray:
    if thickness <= 0:
        return np.zeros_like(component, dtype=bool)
    depth_map = _boundary_depth_map(component)
    return (depth_map > 0) & (depth_map <= thickness)


def _thickness_for_target_area(
    component: np.ndarray,
    *,
    target_pixels: int,
    preserve_core: bool = True,
) -> int:
    depth_map = _boundary_depth_map(component)
    max_depth = int(depth_map.max())
    if max_depth <= 0:
        return 0

    max_allowed_thickness = max_depth - 1 if preserve_core and max_depth > 1 else max_depth
    max_allowed_thickness = max(1, max_allowed_thickness)

    chosen_thickness = max_allowed_thickness
    for thickness in range(1, max_allowed_thickness + 1):
        area = int(np.count_nonzero((depth_map > 0) & (depth_map <= thickness)))
        chosen_thickness = thickness
        if area >= target_pixels:
            break
    return chosen_thickness


def _select_start_coords(allowed_mask: np.ndarray, seed: int | None = None) -> tuple[int, int] | None:
    boundary = _component_boundary(allowed_mask)
    coords = np.argwhere(boundary)
    if coords.size == 0:
        coords = np.argwhere(allowed_mask)
    if coords.size == 0:
        return None
    rng = np.random.default_rng(seed)
    y, x = coords[int(rng.integers(0, len(coords)))]
    return int(y), int(x)


def _low_frequency_noise(shape: tuple[int, int], seed: int | None = None, block_size: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coarse_h = max(1, int(np.ceil(shape[0] / block_size)))
    coarse_w = max(1, int(np.ceil(shape[1] / block_size)))
    coarse = rng.random((coarse_h, coarse_w))
    noise = np.repeat(np.repeat(coarse, block_size, axis=0), block_size, axis=1)
    return noise[: shape[0], : shape[1]]


def _sample_boundary_arc(boundary_mask: np.ndarray, *, size_bucket: str, seed: int | None = None) -> np.ndarray:
    boundary_mask = _normalize_binary_mask(boundary_mask)
    coords = np.argwhere(boundary_mask)
    if coords.size == 0:
        return np.zeros_like(boundary_mask, dtype=bool)

    rng = np.random.default_rng(seed)
    start_y, start_x = coords[int(rng.integers(0, len(coords)))]
    radius = _GEOMETRY_ARC_RADIUS[size_bucket]
    yy, xx = np.indices(boundary_mask.shape)
    arc = boundary_mask & (((yy - int(start_y)) ** 2 + (xx - int(start_x)) ** 2) <= radius**2)
    if not arc.any():
        arc[int(start_y), int(start_x)] = True
    return arc


def _geodesic_distance_within_mask(mask: np.ndarray, starts_mask: np.ndarray) -> np.ndarray:
    mask = _normalize_binary_mask(mask)
    starts_mask = _normalize_binary_mask(starts_mask) & mask
    distance = np.full(mask.shape, -1, dtype=np.int32)
    queue: list[tuple[int, int]] = []
    for y, x in np.argwhere(starts_mask):
        distance[int(y), int(x)] = 0
        queue.append((int(y), int(x)))

    head = 0
    while head < len(queue):
        y, x = queue[head]
        head += 1
        for ny, nx in _iter_neighborhood(y, x, mask.shape):
            if mask[ny, nx] and distance[ny, nx] < 0:
                distance[ny, nx] = distance[y, x] + 1
                queue.append((ny, nx))
    return distance


def _select_pixels_by_score(
    candidate_mask: np.ndarray,
    *,
    score_map: np.ndarray,
    target_pixels: int,
) -> np.ndarray:
    candidate_mask = _normalize_binary_mask(candidate_mask)
    if not candidate_mask.any() or target_pixels <= 0:
        return np.zeros_like(candidate_mask, dtype=bool)

    coords = np.argwhere(candidate_mask)
    if len(coords) <= target_pixels:
        return candidate_mask.copy()

    order = sorted(coords.tolist(), key=lambda coord: float(score_map[int(coord[0]), int(coord[1])]))
    selected = np.zeros_like(candidate_mask, dtype=bool)
    for y, x in order[:target_pixels]:
        selected[int(y), int(x)] = True
    return selected


def _grow_scored_connected_region(
    allowed_mask: np.ndarray,
    *,
    target_pixels: int,
    seed_coord: tuple[int, int] | None,
    score_map: np.ndarray,
) -> np.ndarray:
    allowed_mask = _normalize_binary_mask(allowed_mask)
    if not allowed_mask.any() or target_pixels <= 0 or seed_coord is None:
        return np.zeros_like(allowed_mask, dtype=bool)

    sy, sx = seed_coord
    if not allowed_mask[sy, sx]:
        return np.zeros_like(allowed_mask, dtype=bool)

    region = np.zeros_like(allowed_mask, dtype=bool)
    visited = {(sy, sx)}
    heap: list[tuple[float, int, int]] = [(float(score_map[sy, sx]), sy, sx)]

    while heap and int(region.sum()) < target_pixels:
        _, y, x = heapq.heappop(heap)
        if region[y, x]:
            continue
        region[y, x] = True
        for ny, nx in _iter_neighborhood(y, x, allowed_mask.shape):
            if allowed_mask[ny, nx] and (ny, nx) not in visited:
                visited.add((ny, nx))
                heapq.heappush(heap, (float(score_map[ny, nx]), ny, nx))

    return region


def _grow_connected_region(
    allowed_mask: np.ndarray,
    *,
    target_pixels: int,
    seed: int | None = None,
    depth_map: np.ndarray | None = None,
) -> np.ndarray:
    allowed_mask = _normalize_binary_mask(allowed_mask)
    if not allowed_mask.any() or target_pixels <= 0:
        return np.zeros_like(allowed_mask, dtype=bool)

    start = _select_start_coords(allowed_mask, seed=seed)
    if start is None:
        return np.zeros_like(allowed_mask, dtype=bool)

    rng = np.random.default_rng(seed)
    region = np.zeros_like(allowed_mask, dtype=bool)
    queue: list[tuple[int, int]] = [start]
    visited = {start}
    region[start] = True

    while queue and int(region.sum()) < target_pixels:
        y, x = queue.pop(0)
        neighbors = [
            (ny, nx)
            for ny, nx in _iter_neighborhood(y, x, allowed_mask.shape)
            if allowed_mask[ny, nx] and (ny, nx) not in visited
        ]
        if depth_map is not None and len(neighbors) > 1:
            tie_breakers = rng.random(len(neighbors))
            ordered = sorted(
                enumerate(neighbors),
                key=lambda item: (int(depth_map[item[1]]), float(tie_breakers[item[0]])),
            )
            neighbors = [coord for _, coord in ordered]
        elif len(neighbors) > 1:
            rng.shuffle(neighbors)
        for coord in neighbors:
            visited.add(coord)
            region[coord] = True
            queue.append(coord)
            if int(region.sum()) >= target_pixels:
                break

    return region


def _geometry_thickness_and_target(
    component: np.ndarray,
    *,
    size_bucket: str,
    seed: int | None = None,
) -> tuple[int, int, np.ndarray]:
    depth_map = _boundary_depth_map(component)
    max_depth = int(depth_map.max())
    if max_depth <= 0:
        return 0, 0, depth_map

    min_thickness = _MIN_GEOMETRY_THICKNESS[size_bucket]
    preserve_core = max_depth > 1
    max_allowed = max_depth - 1 if preserve_core else max_depth
    max_allowed = max(1, max_allowed)
    thickness = min(max_allowed, max(min_thickness, 1))

    candidate = _boundary_band_for_thickness(component, thickness)
    target_pixels = _choose_target_change_pixels(
        shape=component.shape,
        candidate_limit=int(candidate.sum()),
        size_bucket=size_bucket,
        seed=seed,
    )
    return thickness, target_pixels, depth_map


def _effective_geometry_thickness(component: np.ndarray, *, size_bucket: str) -> int:
    depth_map = _boundary_depth_map(component)
    max_depth = int(depth_map.max())
    if max_depth <= 0:
        return 0
    preserve_core = max_depth > 1
    max_allowed = max_depth - 1 if preserve_core else max_depth
    max_allowed = max(1, max_allowed)
    return min(_MIN_GEOMETRY_THICKNESS[size_bucket], max_allowed)


def _choose_replace_target_pixels(
    *,
    shape: tuple[int, int],
    component_pixels: int,
    candidate_limit: int,
    size_bucket: str,
    seed: int | None = None,
) -> int:
    total_pixels = int(np.prod(shape))
    total_low_ratio, total_high_ratio = _TARGET_CHANGE_RATIO_RANGES[size_bucket]
    comp_low_ratio, comp_high_ratio = _REPLACE_COMPONENT_FRACTION_RANGES[size_bucket]

    total_low = max(1, int(np.ceil(total_pixels * total_low_ratio)))
    total_high = max(total_low, int(np.floor(total_pixels * total_high_ratio)))
    comp_low = max(1, int(np.ceil(component_pixels * comp_low_ratio)))
    comp_high = max(comp_low, int(np.floor(component_pixels * comp_high_ratio)))

    feasible_low = max(total_low, comp_low)
    feasible_high = min(total_high, comp_high, candidate_limit)

    rng = np.random.default_rng(seed)
    if feasible_low <= feasible_high:
        return int(rng.integers(feasible_low, feasible_high + 1))

    fallback_low = max(1, min(total_low, candidate_limit))
    fallback_high = max(fallback_low, min(total_high, candidate_limit))
    if fallback_low <= fallback_high:
        return int(rng.integers(fallback_low, fallback_high + 1))
    return candidate_limit


def expand_band(
    tissue_mask: np.ndarray,
    seed: int | None = None,
    size_bucket: str | None = "small",
    preferred_labels: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Create a large boundary-crossing ellipse within one connected tissue component."""
    component = _select_single_component(tissue_mask, seed=seed, preferred_labels=preferred_labels)
    if not component.any():
        return np.zeros_like(component, dtype=np.uint8)

    resolved_bucket = _resolve_geometry_size_bucket(size_bucket, seed)
    mask = _sample_boundary_ellipse_mask(component, size_bucket=resolved_bucket, seed=seed)
    return (mask.astype(np.uint8) * 255)


def shrink_band(
    tissue_mask: np.ndarray,
    seed: int | None = None,
    size_bucket: str | None = "small",
    preferred_labels: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Create a large interior ellipse within one connected tissue component."""
    component = _select_single_component(tissue_mask, seed=seed, preferred_labels=preferred_labels)
    if not component.any():
        return np.zeros_like(component, dtype=np.uint8)

    resolved_bucket = _resolve_geometry_size_bucket(size_bucket, seed)
    mask = _sample_interior_ellipse_mask(component, size_bucket=resolved_bucket, seed=seed)
    return (mask.astype(np.uint8) * 255)


def replace_like_blob(
    tissue_mask: np.ndarray,
    seed: int | None = None,
    size_bucket: str | None = "small",
    preferred_labels: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Erase one whole connected tissue component."""
    component = _select_single_component(tissue_mask, seed=seed, preferred_labels=preferred_labels)
    if not component.any():
        return np.zeros_like(component, dtype=np.uint8)
    return component.astype(np.uint8) * 255


def synthesize_change_region(
    tissue_mask: np.ndarray,
    forced_bucket: str | None = None,
    size_bucket: str | None = None,
    seed: int | None = None,
    preferred_labels: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, str]:
    bucket = forced_bucket
    if bucket is None:
        rng = np.random.default_rng(seed)
        bucket = str(rng.choice(sorted(_VALID_GEOMETRY_BUCKETS)))
    if bucket not in _VALID_GEOMETRY_BUCKETS:
        raise ValueError(f"Unsupported forced_bucket: {bucket}")

    if bucket == "expand_band":
        return expand_band(tissue_mask, seed=seed, size_bucket=size_bucket, preferred_labels=preferred_labels), bucket
    if bucket == "shrink_band":
        return shrink_band(tissue_mask, seed=seed, size_bucket=size_bucket, preferred_labels=preferred_labels), bucket
    return replace_like_blob(tissue_mask, seed=seed, size_bucket=size_bucket, preferred_labels=preferred_labels), bucket


def _size_bucket_for_change_ratio(change_ratio: float) -> str:
    if change_ratio <= 0.0:
        return "identity"
    if change_ratio <= 0.14:
        return "small"
    if change_ratio <= 0.24:
        return "medium"
    return "large"


def _validate_synthesized_change_region(
    *,
    mask_mode: str,
    change_region_mask: np.ndarray,
    expected_bucket: str | None = None,
) -> tuple[float, str]:
    change_pixels = int((change_region_mask > 0).sum())
    if mask_mode != "identity" and change_pixels <= 0:
        raise ValueError(f"Synthesized change mask for {mask_mode} must be non-empty")

    change_ratio = float(change_pixels / change_region_mask.size)
    if mask_mode != "identity" and change_ratio > _MAX_CHANGE_RATIO:
        raise ValueError(
            f"Synthesized change mask for {mask_mode} exceeded max change ratio {_MAX_CHANGE_RATIO:.2f}: {change_ratio:.4f}"
        )
    size_bucket = _size_bucket_for_change_ratio(change_ratio)
    if size_bucket not in _VALID_SIZE_BUCKETS:
        raise ValueError(f"Unsupported synthesized size bucket: {size_bucket}")
    if expected_bucket is not None and size_bucket != expected_bucket:
        if (
            mask_mode in _VALID_GEOMETRY_BUCKETS
            and expected_bucket in _SIZE_BUCKET_ORDER
            and size_bucket in _SIZE_BUCKET_ORDER
            and _SIZE_BUCKET_ORDER[size_bucket] >= _SIZE_BUCKET_ORDER[expected_bucket]
        ):
            return change_ratio, size_bucket
        raise ValueError(
            f"Synthesized change mask for {mask_mode} landed in {size_bucket}, expected {expected_bucket}"
        )
    return change_ratio, size_bucket


def build_synthetic_inpaint_metadata(
    dataset_roots: Mapping[str, str | Path],
    output_dir: str | Path,
    forced_mode: str = "mixed",
    forced_bucket: str | None = None,
    val_ratio: float = 0.1,
    seed: int = 42,
    samples_per_dataset: int | None = None,
    max_attempts_per_sample: int | None = None,
) -> tuple[Path, Path]:
    if forced_mode not in _VALID_FORCED_MODES:
        raise ValueError(
            f"Unsupported forced_mode for synthetic inpaint metadata: {forced_mode}"
        )
    if samples_per_dataset is not None and samples_per_dataset <= 0:
        raise ValueError(f"samples_per_dataset must be positive, got {samples_per_dataset}")
    if max_attempts_per_sample is not None and max_attempts_per_sample <= 0:
        raise ValueError(
            f"max_attempts_per_sample must be positive, got {max_attempts_per_sample}"
        )
    if forced_bucket is not None and forced_bucket not in _VALID_SIZE_BUCKETS:
        raise ValueError(f"Unsupported forced_bucket for synthetic inpaint metadata: {forced_bucket}")

    config = _SyntheticInpaintConfig(forced_mode=forced_mode, forced_bucket=forced_bucket, seed=seed)
    output_dir = Path(output_dir)
    attempt_limit = max_attempts_per_sample or 1

    records: list[dict] = []
    for dataset_name, dataset_root in dataset_roots.items():
        samples = load_layered_dataset_samples(dataset_name, dataset_root)
        selected_samples = _select_samples(samples, samples_per_dataset, seed, dataset_name)
        for sample in selected_samples:
            try:
                prior_modes: list[str] = []
                for variant_index in range(_variant_count_for_sample(sample=sample, config=config)):
                    record = _build_synthetic_record_with_attempts(
                        sample=sample,
                        output_dir=output_dir,
                        config=config,
                        attempts=attempt_limit,
                        variant_index=variant_index,
                        excluded_modes=tuple(prior_modes),
                    )
                    if "mask_mode" in record:
                        prior_modes.append(record["mask_mode"])
                    records.append(record)
            except OSError as exc:
                print(f"Skipping unreadable sample {sample.sample_id} from {dataset_name}: {exc}")

    train_records, val_records = split_records_by_case(
        records,
        case_id_getter=lambda record: f"{record['dataset']}::{record['case_id']}",
        val_ratio=val_ratio,
        seed=seed,
    )

    train_path = write_jsonl(output_dir / "metadata_inpaint_train.jsonl", train_records)
    val_path = write_jsonl(output_dir / "metadata_inpaint_val.jsonl", val_records)
    return train_path, val_path


def _select_samples(
    samples: list,
    samples_per_dataset: int | None,
    seed: int,
    dataset_name: str,
) -> list:
    if samples_per_dataset is None or samples_per_dataset >= len(samples):
        return list(samples)

    dataset_rng = random.Random(f"{seed}::{dataset_name}")
    selected_indexes = sorted(dataset_rng.sample(range(len(samples)), k=samples_per_dataset))
    return [samples[index] for index in selected_indexes]


def _build_synthetic_record_with_attempts(
    *,
    sample,
    output_dir: Path,
    config: _SyntheticInpaintConfig,
    attempts: int,
    variant_index: int = 0,
    excluded_modes: tuple[str, ...] = (),
) -> dict:
    last_error: Exception | None = None
    for attempt_index in range(attempts):
        try:
            return _build_synthetic_record(
                sample=sample,
                output_dir=output_dir,
                config=config,
                attempt_seed=config.seed + attempt_index,
                variant_index=variant_index,
                excluded_modes=excluded_modes,
            )
        except Exception as exc:  # pragma: no cover - exercised through retry tests
            last_error = exc
    assert last_error is not None
    raise last_error


def _build_synthetic_record(
    *,
    sample,
    output_dir: Path,
    config: _SyntheticInpaintConfig,
    attempt_seed: int | None = None,
    variant_index: int = 0,
    excluded_modes: tuple[str, ...] = (),
) -> dict:
    dataset_name = sample.dataset_name
    dataset_config = get_config(dataset_name)
    source_image = sample.image_path
    target_image = sample.image_path
    target_tissue_mask = sample.tissue_mask_path
    target_nuclei_mask = sample.nuclei_mask_path
    tissue_mask_array = load_mask_array(sample.tissue_mask_path)
    mask_mode = _sample_effective_mode(
        config=config,
        sample=sample,
        attempt_seed=attempt_seed,
        variant_index=variant_index,
        excluded_modes=excluded_modes,
    )

    if mask_mode == "identity":
        change_region_mask_array = np.zeros_like(tissue_mask_array, dtype=np.uint8)
        change_ratio, size_bucket = _validate_synthesized_change_region(
            mask_mode=mask_mode,
            change_region_mask=change_region_mask_array,
            expected_bucket=config.forced_bucket,
        )
        change_region_mask = _write_change_region_mask(
            output_dir=output_dir,
            dataset_name=dataset_name,
            sample_id=sample.sample_id,
            mask=change_region_mask_array,
            variant_index=variant_index,
        )
        erased_source_image = source_image
    elif mask_mode == "near_identity":
        change_region_mask_array = _build_near_identity_mask(
            tissue_mask_array,
            change_pixels=config.near_identity_change_pixels,
        )
        change_ratio, size_bucket = _validate_synthesized_change_region(
            mask_mode=mask_mode,
            change_region_mask=change_region_mask_array,
            expected_bucket=config.forced_bucket,
        )
        change_region_mask = _write_change_region_mask(
            output_dir=output_dir,
            dataset_name=dataset_name,
            sample_id=sample.sample_id,
            mask=change_region_mask_array,
            variant_index=variant_index,
        )
        erased_source_image = _materialize_erased_source_image(
            dataset_name=dataset_name,
            sample_id=sample.sample_id,
            source_image=source_image,
            change_region_mask=change_region_mask,
            output_dir=output_dir,
            variant_index=variant_index,
        )
    else:
        preferred_labels = dataset_config.tumor_ids if dataset_config.tumor_ids else None
        change_region_mask_array, mask_mode = synthesize_change_region(
            tissue_mask_array,
            forced_bucket=mask_mode,
            size_bucket=config.forced_bucket,
            seed=attempt_seed if attempt_seed is not None else config.seed,
            preferred_labels=preferred_labels,
        )
        change_ratio, size_bucket = _validate_synthesized_change_region(
            mask_mode=mask_mode,
            change_region_mask=change_region_mask_array,
            expected_bucket=None if mask_mode == "replace_like_blob" else config.forced_bucket,
        )
        change_region_mask = _write_change_region_mask(
            output_dir=output_dir,
            dataset_name=dataset_name,
            sample_id=sample.sample_id,
            mask=change_region_mask_array,
            variant_index=variant_index,
        )
        erased_source_image = _materialize_erased_source_image(
            dataset_name=dataset_name,
            sample_id=sample.sample_id,
            source_image=source_image,
            change_region_mask=change_region_mask,
            output_dir=output_dir,
            variant_index=variant_index,
        )

    return {
        "dataset": dataset_name,
        "sample_id": sample.sample_id,
        "case_id": sample.case_id,
        "source_image": str(source_image),
        "erased_source_image": str(erased_source_image),
        "target_image": str(target_image),
        "target_tissue_mask": str(target_tissue_mask),
        "target_nuclei_mask": str(target_nuclei_mask),
        "change_region_mask": str(change_region_mask),
        "prompt": sample.prompt or default_prompt_for_dataset(dataset_name),
        "edit_type": mask_mode,
        "change_ratio": change_ratio,
        "mask_mode": mask_mode,
        "size_bucket": size_bucket,
        "variant_index": variant_index,
    }


def _variant_count_for_sample(*, sample, config: _SyntheticInpaintConfig) -> int:
    if config.forced_mode != "mixed":
        return 1
    if len(_candidate_modes_for_config(config)) <= 1:
        return 1
    return 2 if _is_high_value_patch(sample) else 1


def _is_high_value_patch(sample) -> bool:
    config = get_config(sample.dataset_name)
    tissue_mask = load_mask_array(sample.tissue_mask_path)
    foreground_labels = {
        int(label) for label in np.unique(tissue_mask) if int(label) not in config.skip_tissues
    }
    if not foreground_labels:
        return False
    has_tumor = any(label in config.tumor_ids for label in foreground_labels)
    has_other_tissue = any(label not in config.tumor_ids for label in foreground_labels)
    return has_tumor and has_other_tissue


def _build_near_identity_mask(tissue_mask: np.ndarray, change_pixels: int) -> np.ndarray:
    mask = np.zeros_like(tissue_mask, dtype=np.uint8)
    if change_pixels <= 0:
        return mask

    foreground = [tuple(coord) for coord in np.argwhere(tissue_mask > 0)]
    if not foreground:
        foreground = list(np.ndindex(tissue_mask.shape))

    selected: list[tuple[int, int]] = []
    for coord in foreground:
        if coord not in selected:
            selected.append(coord)
        if len(selected) == change_pixels:
            break

    if len(selected) < change_pixels:
        for coord in np.ndindex(tissue_mask.shape):
            if coord not in selected:
                selected.append(coord)
            if len(selected) == change_pixels:
                break

    for y, x in selected[:change_pixels]:
        mask[y, x] = 255
    return mask


def _materialize_erased_source_image(
    *,
    dataset_name: str,
    sample_id: str,
    source_image: Path,
    change_region_mask: Path,
    output_dir: Path,
    variant_index: int = 0,
) -> Path:
    erased_dir = output_dir / "erased_source_images" / dataset_name
    erased_dir.mkdir(parents=True, exist_ok=True)
    erased_path = erased_dir / _variant_filename(sample_id, variant_index)

    source = np.asarray(Image.open(source_image).convert("RGB"), dtype=np.uint8)
    change_mask = np.asarray(Image.open(change_region_mask))
    if change_mask.ndim == 3:
        changed = np.any(change_mask > 0, axis=-1)
    else:
        changed = change_mask > 0

    erased = source.copy()
    erased[changed] = 128
    Image.fromarray(erased).save(erased_path)
    return erased_path


def _write_change_region_mask(
    *,
    output_dir: Path,
    dataset_name: str,
    sample_id: str,
    mask: np.ndarray,
    variant_index: int = 0,
) -> Path:
    mask_dir = output_dir / "change_region_masks" / dataset_name
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / _variant_filename(sample_id, variant_index)
    Image.fromarray(mask.astype(np.uint8)).save(mask_path)
    return mask_path


def _variant_filename(sample_id: str, variant_index: int) -> str:
    suffix = "" if variant_index == 0 else f"__v{variant_index}"
    return f"{sample_id}{suffix}.png"

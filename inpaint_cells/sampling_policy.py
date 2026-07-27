"""Shared frozen nuclei-sampling support-mask policy."""

from __future__ import annotations

import math
import random
from types import SimpleNamespace
from typing import Any

import numpy as np
from skimage.morphology import binary_dilation, disk, skeletonize


def widen_locally_thin_mask(
    semantic_mask: np.ndarray,
    allowed_region: np.ndarray,
    minimum_width: int = 33,
) -> np.ndarray:
    """Preserve thick edits and widen locally thin branches around the skeleton."""
    if minimum_width < 1 or minimum_width % 2 == 0:
        raise ValueError("minimum_width must be a positive odd integer")
    semantic = np.asarray(semantic_mask, dtype=bool)
    allowed = np.asarray(allowed_region, dtype=bool)
    if semantic.shape != allowed.shape:
        raise ValueError("semantic_mask and allowed_region must share one shape")
    if not np.any(semantic):
        raise ValueError("semantic mask is empty")
    if np.any(semantic & ~allowed):
        raise ValueError("semantic mask extends outside biological foreground")
    radius = int(minimum_width) // 2
    medial = skeletonize(semantic)
    widened = (semantic | binary_dilation(medial, footprint=disk(radius))) & allowed
    if np.any(semantic & ~widened):
        raise AssertionError("widening must preserve all semantic pixels")
    return widened


def valid_biological_tissue_mask(
    tissue_map: np.ndarray,
    skip_tissue_ids: set[int] | frozenset[int] | tuple[int, ...] | list[int],
) -> np.ndarray:
    """Return tissue support that may contain a complete generated nucleus."""

    tissue = np.asarray(tissue_map)
    allowed = tissue != 0
    skipped = tuple(sorted(int(value) for value in skip_tissue_ids))
    if skipped:
        allowed &= ~np.isin(tissue, skipped)
    return np.asarray(allowed, dtype=bool)


def retry_pool_target(
    *,
    quota: int,
    component_area: int,
    expected_nucleus_area: float,
    args: SimpleNamespace,
) -> tuple[int, bool, float]:
    """Choose the frozen retry-pool size, with extra coverage when packing is dense."""

    quota = max(0, int(quota))
    component_area = max(1, int(component_area))
    occupancy = quota * max(float(expected_nucleus_area), 1.0) / component_area
    dense = bool(
        quota >= int(args.dense_retry_quota_threshold)
        or occupancy >= float(args.dense_retry_occupancy_threshold)
    )
    if dense:
        multiplier = float(args.dense_retry_candidate_multiplier)
        floor = int(args.dense_retry_candidate_floor)
    else:
        multiplier = float(args.retry_candidate_multiplier)
        floor = int(args.retry_candidate_floor)
    target = max(floor, int(math.ceil(quota * multiplier)))
    return target, dense, float(occupancy)


def retry_transform_specs(
    args: SimpleNamespace,
    *,
    trial_count: int | None = None,
) -> list[dict[str, Any]]:
    """Create the shared seeded rotation/flip/jitter/scale retry schedule."""

    count = max(
        1,
        int(
            args.placement_transform_trials
            if trial_count is None
            else trial_count
        ),
    )
    configured_scales = tuple(float(value) for value in args.placement_retry_scales)
    if not configured_scales:
        configured_scales = (1.0,)
    max_jitter = max(0, int(args.placement_center_jitter_max))
    offsets = [(0, 0)]
    for radius in range(2, max_jitter + 1, 2):
        offsets.extend(
            [
                (-radius, 0),
                (radius, 0),
                (0, -radius),
                (0, radius),
                (-radius, -radius),
                (-radius, radius),
                (radius, -radius),
                (radius, radius),
            ]
        )
    nonzero_offsets = offsets[1:]
    random.shuffle(nonzero_offsets)
    offsets = [offsets[0], *nonzero_offsets]
    rotations = [0, 1, 2, 3]
    random.shuffle(rotations)
    return [
        {
            "rotation_quarters": rotations[index % len(rotations)],
            "flip_horizontal": bool((index // len(rotations)) % 2),
            "flip_vertical": bool((index // (2 * len(rotations))) % 2),
            "scale": configured_scales[
                min(index, len(configured_scales) - 1)
            ],
            "offset_yx": offsets[index % len(offsets)],
        }
        for index in range(count)
    ]

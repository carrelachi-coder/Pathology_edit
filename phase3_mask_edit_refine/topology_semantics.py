"""Shared whole-mask digital-topology measurements.

The pixel grower uses 8-connected tissue foreground and 4-connected
background.  Whole-mask hole measurements must use the same background
connectivity; otherwise diagonally touching enclosed regions are silently
merged by the audit even though the local simple-point guard keeps them
separate.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


WHOLE_MASK_HOLE_SEMANTICS_VERSION = "digital-topology-8fg-4bg-v2"

_BACKGROUND_4 = np.asarray(
    [[False, True, False], [True, True, True], [False, True, False]],
    dtype=bool,
)


def enclosed_background_4(mask: np.ndarray) -> np.ndarray:
    """Return 4-connected background regions enclosed by ``mask``."""

    foreground = np.asarray(mask, dtype=bool)
    return (
        ndimage.binary_fill_holes(foreground, structure=_BACKGROUND_4)
        & ~foreground
    )


def hole_count_4(mask: np.ndarray) -> int:
    """Count enclosed 4-connected background components."""

    holes = enclosed_background_4(mask)
    return int(ndimage.label(holes, structure=_BACKGROUND_4)[1])


def hole_intersection_count_4(mask: np.ndarray, region: np.ndarray) -> int:
    """Count enclosed 4-connected background components meeting ``region``."""

    holes = enclosed_background_4(mask)
    labels, count = ndimage.label(holes, structure=_BACKGROUND_4)
    target = np.asarray(region, dtype=bool)
    return sum(
        bool(np.any(target & (labels == index)))
        for index in range(1, count + 1)
    )

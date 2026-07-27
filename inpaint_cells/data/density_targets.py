"""Instance-aware erasure and center-density targets for ProbNet."""

from __future__ import annotations

from typing import Iterator, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage


def iter_class_components(
    nuclei_map: np.ndarray,
    num_classes: int = 5,
) -> Iterator[Tuple[int, np.ndarray, Tuple[float, float]]]:
    """Yield class ID, component mask, and ``(y, x)`` centroid.

    ProbNet's prepared semantic masks do not contain instance IDs. The frozen
    fallback definition is therefore an 8-connected component within each
    nucleus class.
    """
    if nuclei_map.ndim != 2:
        raise ValueError(f"nuclei_map must be 2-D, got shape {nuclei_map.shape}.")

    values = nuclei_map.astype(np.int64, copy=False)
    for class_id in range(1, num_classes + 1):
        class_mask = values == class_id
        labels, count = ndimage.label(
            class_mask,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        centroids = ndimage.center_of_mass(
            class_mask,
            labels,
            range(1, count + 1),
        )
        for component_id, (centroid_y, centroid_x) in enumerate(centroids, start=1):
            yield (
                class_id,
                labels == component_id,
                (float(centroid_y), float(centroid_x)),
            )


def expand_edit_mask_to_complete_instances(
    nuclei_map: np.ndarray,
    edit_mask: np.ndarray,
    num_classes: int = 5,
) -> np.ndarray:
    """Expand an edit mask to include every intersecting nucleus component."""
    if nuclei_map.shape != edit_mask.shape:
        raise ValueError(
            f"nuclei_map and edit_mask must have the same shape, got "
            f"{nuclei_map.shape} and {edit_mask.shape}."
        )

    expanded = edit_mask.astype(bool, copy=True)
    for _, component, _ in iter_class_components(nuclei_map, num_classes=num_classes):
        if np.any(component & expanded):
            expanded |= component
    return expanded


def select_instances_by_centroid(
    nuclei_map: np.ndarray,
    edit_mask: np.ndarray,
    num_classes: int = 5,
) -> np.ndarray:
    """Select complete nucleus components whose centroids lie in ``edit_mask``.

    Boundary-crossing nuclei whose centroids remain outside the edit support
    are deliberately kept whole.
    """
    if nuclei_map.shape != edit_mask.shape:
        raise ValueError(
            f"nuclei_map and edit_mask must have the same shape, got "
            f"{nuclei_map.shape} and {edit_mask.shape}."
        )

    selected = np.zeros_like(edit_mask, dtype=bool)
    height, width = edit_mask.shape
    for _, component, (centroid_y, centroid_x) in iter_class_components(
        nuclei_map,
        num_classes=num_classes,
    ):
        row = int(np.clip(round(centroid_y), 0, height - 1))
        col = int(np.clip(round(centroid_x), 0, width - 1))
        if edit_mask[row, col]:
            selected |= component
    return selected


def extract_class_centers(
    nuclei_map: np.ndarray,
    num_classes: int = 5,
) -> list[Tuple[int, float, float]]:
    """Freeze semantic-component centers on the uncropped source patch."""
    return [
        (class_id, center_y, center_x)
        for class_id, _, (center_y, center_x) in iter_class_components(
            nuclei_map,
            num_classes=num_classes,
        )
    ]


def _add_normalized_gaussian(
    target: np.ndarray,
    center_y: float,
    center_x: float,
    support: np.ndarray,
    sigma: float,
) -> None:
    """Add one unit-mass Gaussian restricted to ``support``."""
    height, width = support.shape
    radius = max(1, int(np.ceil(3.0 * sigma)))
    y0 = max(0, int(np.floor(center_y)) - radius)
    y1 = min(height, int(np.floor(center_y)) + radius + 1)
    x0 = max(0, int(np.floor(center_x)) - radius)
    x1 = min(width, int(np.floor(center_x)) + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return

    ys = np.arange(y0, y1, dtype=np.float32)[:, None]
    xs = np.arange(x0, x1, dtype=np.float32)[None, :]
    kernel = np.exp(
        -((ys - center_y) ** 2 + (xs - center_x) ** 2) / (2.0 * sigma * sigma)
    ).astype(np.float32)
    kernel *= support[y0:y1, x0:x1]
    mass = float(kernel.sum())
    if mass <= 0.0:
        center_row = int(np.clip(round(center_y), 0, height - 1))
        center_col = int(np.clip(round(center_x), 0, width - 1))
        if support[center_row, center_col]:
            target[center_row, center_col] += 1.0
        return
    target[y0:y1, x0:x1] += kernel / mass


def build_center_density_targets(
    nuclei_map: np.ndarray,
    tissue_map: np.ndarray,
    edit_mask: np.ndarray,
    sigma: float = 2.0,
    num_classes: int = 5,
    num_tissues: int = 16,
    centers: Optional[Sequence[Tuple[int, float, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build five class-density maps and an exact tissue-by-class count table.

    A component is supervised when its centroid lies in the changed region.
    Its Gaussian is clipped to the corresponding changed tissue region and
    renormalized so the channel integral remains exactly one instance.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    if nuclei_map.shape != tissue_map.shape or nuclei_map.shape != edit_mask.shape:
        raise ValueError("nuclei_map, tissue_map, and edit_mask must share one shape.")

    height, width = nuclei_map.shape
    changed = edit_mask.astype(bool, copy=False)
    density = np.zeros((num_classes, height, width), dtype=np.float32)
    counts = np.zeros((num_tissues, num_classes), dtype=np.float32)

    if centers is None:
        centers = extract_class_centers(nuclei_map, num_classes=num_classes)

    for class_id, center_y, center_x in centers:
        if center_y < 0 or center_y >= height or center_x < 0 or center_x >= width:
            continue
        row = int(np.clip(round(center_y), 0, height - 1))
        col = int(np.clip(round(center_x), 0, width - 1))
        if not changed[row, col]:
            continue

        tissue_id = int(tissue_map[row, col])
        if tissue_id < 0 or tissue_id >= num_tissues:
            raise ValueError(f"Invalid tissue ID {tissue_id} at nucleus centroid.")
        support = changed & (tissue_map == tissue_id)
        _add_normalized_gaussian(
            density[class_id - 1],
            center_y,
            center_x,
            support,
            sigma,
        )
        counts[tissue_id, class_id - 1] += 1.0

    return density, counts

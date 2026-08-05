"""Nuclei I/O and complete-instance utilities shared by joint components."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from .models import JointContractError

RAW_TO_INTERNAL = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 101: 1, 102: 2, 103: 3, 104: 4, 105: 5}
INTERNAL_TO_RAW = {1: 101, 2: 102, 3: 103, 4: 104, 5: 105}


def load_nuclei_mask(path: str | Path) -> np.ndarray:
    array = np.asarray(Image.open(path).convert("L"), dtype=np.int64)
    return normalize_nuclei_mask(array)


def normalize_nuclei_mask(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask)
    if values.ndim != 2:
        raise JointContractError(f"nuclei mask must be 2-D, got {values.shape}")
    unique = {int(value) for value in np.unique(values)}
    unsupported = sorted(unique - set(RAW_TO_INTERNAL))
    if unsupported:
        raise JointContractError(f"unsupported nuclei IDs: {unsupported}")
    result = np.zeros(values.shape, dtype=np.uint8)
    for source, target in RAW_TO_INTERNAL.items():
        result[values == source] = target
    return result


def to_raw_nuclei_mask(mask: np.ndarray) -> np.ndarray:
    internal = normalize_nuclei_mask(mask)
    result = np.zeros(internal.shape, dtype=np.uint8)
    for source, target in INTERNAL_TO_RAW.items():
        result[internal == source] = target
    return result


def iter_instances(mask: np.ndarray) -> Iterator[tuple[str, int, np.ndarray]]:
    """Yield stable semantic-fallback instances separated by watershed.

    CellViT semantic rasters frequently join touching same-class nuclei into
    one eight-connected component.  Treating that component as one nucleus
    created implausibly large reference shapes and made REMOVE_WHOLE extend far
    outside the authorized halo.  Marker-controlled distance watershed recovers
    conservative instance identities without retraining a model.  Native JSON
    remains preferred whenever it is available.
    """

    internal = normalize_nuclei_mask(mask)
    for class_id in range(1, 6):
        labeled, count = _semantic_instance_labels(internal == class_id)
        components: list[tuple[float, float, int, np.ndarray]] = []
        for component_id in range(1, count + 1):
            component = labeled == component_id
            if not np.any(component):
                continue
            cy, cx = ndimage.center_of_mass(component)
            components.append((float(cy), float(cx), int(component.sum()), component))
        components.sort(key=lambda item: (item[0], item[1], item[2]))
        for index, (_, _, _, component) in enumerate(components, start=1):
            yield f"nuc-c{class_id}-{index:04d}", class_id, component


def _semantic_instance_labels(region: np.ndarray) -> tuple[np.ndarray, int]:
    binary = np.asarray(region, dtype=bool)
    if not np.any(binary):
        return np.zeros(binary.shape, dtype=np.int32), 0
    distance = ndimage.distance_transform_edt(binary)
    # h=1 suppresses pixel-scale shoulders while retaining distinct centers of
    # touching nuclei. Requiring radius >=1.5 avoids markers on thin bridges.
    maxima = h_maxima(distance, 1.0) & (distance >= 1.5)
    markers, marker_count = ndimage.label(
        maxima,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    if marker_count == 0:
        return ndimage.label(
            binary,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
    labels = watershed(
        -distance,
        markers=markers,
        mask=binary,
        connectivity=np.ones((3, 3), dtype=np.uint8),
        watershed_line=False,
    ).astype(np.int32, copy=False)
    return labels, int(labels.max(initial=0))


def instance_centroid(component: np.ndarray) -> tuple[float, float]:
    cy, cx = ndimage.center_of_mass(np.asarray(component, dtype=bool))
    return float(cx), float(cy)


def instance_bbox(component: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(component)
    if not ys.size:
        raise JointContractError("empty nucleus component has no bounding box")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def touches_border(component: np.ndarray) -> bool:
    value = np.asarray(component, dtype=bool)
    return bool(
        np.any(value[0])
        or np.any(value[-1])
        or np.any(value[:, 0])
        or np.any(value[:, -1])
    )


def complete_instances_intersecting(mask: np.ndarray, region: np.ndarray) -> np.ndarray:
    selected = np.zeros(np.asarray(mask).shape, dtype=bool)
    region_bool = np.asarray(region, dtype=bool)
    for _, _, component in iter_instances(mask):
        if np.any(component & region_bool):
            selected |= component
    return selected


def load_native_instances(
    path: str | Path,
    *,
    shape: tuple[int, int],
    semantic_mask: np.ndarray,
) -> tuple[tuple[str, int, np.ndarray], ...]:
    """Load CellViT ``nuc`` or ``cells`` JSON and verify semantic agreement."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise JointContractError("native nucleus instance JSON root must be an object")
    raw_items = []
    if isinstance(payload.get("nuc"), dict):
        raw_items = [(str(key), value) for key, value in payload["nuc"].items()]
    elif isinstance(payload.get("nuclei"), dict):
        raw_items = [(str(key), value) for key, value in payload["nuclei"].items()]
    elif isinstance(payload.get("cells"), list):
        raw_items = [(str(index), value) for index, value in enumerate(payload["cells"])]
    if not raw_items:
        raise JointContractError("native nucleus instance JSON contains no instances")
    metadata = payload.get("wsi_metadata") if isinstance(payload.get("wsi_metadata"), dict) else {}
    semantic = normalize_nuclei_mask(semantic_mask)
    occupied = np.zeros(shape, dtype=bool)
    result = []
    for raw_id, info in raw_items:
        if not isinstance(info, dict):
            continue
        raw_class = int(info.get("type", 0))
        class_id = RAW_TO_INTERNAL.get(raw_class)
        if class_id not in range(1, 6):
            continue
        contour = info.get("contour")
        if not isinstance(contour, list) or len(contour) < 3:
            continue
        points = _local_contour(contour, info=info, metadata=metadata, shape=shape)
        if len(points) < 3:
            continue
        canvas = Image.new("1", (shape[1], shape[0]), 0)
        ImageDraw.Draw(canvas).polygon(points, fill=1)
        component = np.asarray(canvas, dtype=bool)
        if not np.any(component):
            continue
        if np.any(component & occupied):
            raise JointContractError("native nucleus instances overlap")
        agreement = float(np.mean(semantic[component] == class_id))
        if agreement < 0.80:
            raise JointContractError(
                f"native instance {raw_id} disagrees with semantic nuclei mask"
            )
        occupied |= component
        result.append((f"native-{raw_id}", int(class_id), component))
    if not result:
        raise JointContractError("native nucleus JSON has no representable complete instances")
    result.sort(key=lambda item: (item[1], instance_centroid(item[2])[1], instance_centroid(item[2])[0], item[0]))
    return tuple(result)


def _local_contour(contour, *, info, metadata, shape):
    points = []
    offset = info.get("offset_global")
    x_offset = y_offset = 0.0
    if isinstance(offset, list) and len(offset) >= 2:
        patch_size = metadata.get("patch_size")
        if isinstance(patch_size, (int, float)):
            x_offset = max(float(patch_size) - shape[1], 0.0) + float(offset[1])
            y_offset = max(float(patch_size) - shape[0], 0.0) + float(offset[0])
        else:
            x_offset, y_offset = float(offset[1]), float(offset[0])
    for point in contour:
        if not isinstance(point, list) or len(point) < 2:
            continue
        x, y = float(point[0]) - x_offset, float(point[1]) - y_offset
        if -1 <= x <= shape[1] and -1 <= y <= shape[0]:
            points.append((x, y))
    return points

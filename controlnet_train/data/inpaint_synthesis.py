"""Synthetic inpaint metadata builder for Phase 5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from .common import (
    default_prompt_for_dataset,
    load_layered_dataset_samples,
    load_mask_array,
    split_records_by_case,
    write_jsonl,
)


_VALID_FORCED_MODES = {"identity", "near_identity"}
_VALID_GEOMETRY_BUCKETS = {"expand_band", "shrink_band", "replace_like_blob"}

# Keep the synthetic edits small and local so they behave like inpainting
# patches rather than wholesale shape replacements.
_MIN_BLOB_SIZE = 1
_MAX_BLOB_SIZE = 8
_BLOB_SIZE_DIVISOR = 8
_BLOB_SIZE_OFFSET = 2
_DEFAULT_NEAR_IDENTITY_CHANGE_PIXELS = 1


@dataclass(frozen=True)
class _SyntheticInpaintConfig:
    forced_mode: str
    near_identity_change_pixels: int = _DEFAULT_NEAR_IDENTITY_CHANGE_PIXELS


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


def _select_single_component(mask: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Pick one connected component from a possibly multi-component tissue mask.

    The public geometry helpers intentionally work on a single component even if
    the caller passes a full tissue mask with several disjoint regions. By
    default we choose the largest component in discovery order; if several
    components tie for the largest area, ``seed`` controls which tied component
    is selected.
    """
    components = _connected_components(mask)
    if not components:
        return np.zeros_like(mask, dtype=bool)
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


def _boundary_attached_blob(component: np.ndarray, seed: int | None = None) -> np.ndarray:
    component = _normalize_binary_mask(component)
    if not component.any():
        return np.zeros_like(component, dtype=bool)

    core = _component_core(component, seed=seed)
    candidate_region = component & ~core
    if not candidate_region.any():
        return np.zeros_like(component, dtype=bool)

    boundary_coords = np.argwhere(_component_boundary(candidate_region))
    if boundary_coords.size == 0:
        boundary_coords = np.argwhere(candidate_region)
    if boundary_coords.size == 0:
        return np.zeros_like(component, dtype=bool)

    rng = np.random.default_rng(seed)
    start_y, start_x = boundary_coords[int(rng.integers(0, len(boundary_coords)))]
    # The blob is meant to stay compact: start from a tiny patch and grow only
    # enough to look like a local synthetic edit.
    target_size = max(
        _MIN_BLOB_SIZE,
        min(
            _MAX_BLOB_SIZE,
            int(candidate_region.sum() // _BLOB_SIZE_DIVISOR) + _BLOB_SIZE_OFFSET,
        ),
    )
    target_size = min(target_size, int(candidate_region.sum()))

    blob = {(int(start_y), int(start_x))}
    frontier = [(int(start_y), int(start_x))]
    visited = set(blob)

    while frontier and len(blob) < target_size:
        y, x = frontier.pop(0)
        neighbors = [
            (ny, nx)
            for ny, nx in _iter_neighborhood(y, x, component.shape)
            if candidate_region[ny, nx] and (ny, nx) not in visited
        ]
        if len(neighbors) > 1:
            rng.shuffle(neighbors)
        for ny, nx in neighbors:
            visited.add((ny, nx))
            blob.add((ny, nx))
            frontier.append((ny, nx))
            if len(blob) >= target_size:
                break

    mask = np.zeros_like(component, dtype=bool)
    for y, x in blob:
        mask[y, x] = True
    return mask


def expand_band(tissue_mask: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Create an exterior band around one connected tissue component."""
    component = _select_single_component(tissue_mask, seed=seed)
    if not component.any():
        return np.zeros_like(component, dtype=np.uint8)

    band = _dilate(component, steps=1) & ~component
    return band.astype(np.uint8) * 255


def shrink_band(tissue_mask: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Create an interior band around one connected tissue component."""
    component = _select_single_component(tissue_mask, seed=seed)
    if not component.any():
        return np.zeros_like(component, dtype=np.uint8)

    band = component & ~_erode(component, steps=1)
    if not band.any():
        band = _component_boundary(component)
    if np.array_equal(band, component):
        core = _component_core(component, seed=seed)
        band = component & ~core if core.any() else np.zeros_like(component, dtype=bool)
    return band.astype(np.uint8) * 255


def replace_like_blob(tissue_mask: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Create a compact blob attached to one connected tissue component."""
    component = _select_single_component(tissue_mask, seed=seed)
    if not component.any():
        return np.zeros_like(component, dtype=np.uint8)

    blob = _boundary_attached_blob(component, seed=seed)
    if not blob.any():
        blob = _component_boundary(component)
    return blob.astype(np.uint8) * 255


def synthesize_change_region(
    tissue_mask: np.ndarray,
    forced_bucket: str | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, str]:
    bucket = forced_bucket
    if bucket is None:
        rng = np.random.default_rng(seed)
        bucket = str(rng.choice(sorted(_VALID_GEOMETRY_BUCKETS)))
    if bucket not in _VALID_GEOMETRY_BUCKETS:
        raise ValueError(f"Unsupported forced_bucket: {bucket}")

    if bucket == "expand_band":
        return expand_band(tissue_mask, seed=seed), bucket
    if bucket == "shrink_band":
        return shrink_band(tissue_mask, seed=seed), bucket
    return replace_like_blob(tissue_mask, seed=seed), bucket


def build_synthetic_inpaint_metadata(
    dataset_roots: Mapping[str, str | Path],
    output_dir: str | Path,
    forced_mode: str,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Path, Path]:
    if forced_mode not in _VALID_FORCED_MODES:
        raise ValueError(
            f"Unsupported forced_mode for synthetic inpaint metadata: {forced_mode}"
        )

    config = _SyntheticInpaintConfig(forced_mode=forced_mode)
    output_dir = Path(output_dir)

    records: list[dict] = []
    for dataset_name, dataset_root in dataset_roots.items():
        samples = load_layered_dataset_samples(dataset_name, dataset_root)
        for sample in samples:
            records.append(
                _build_synthetic_record(
                    sample=sample,
                    output_dir=output_dir,
                    config=config,
                )
            )

    train_records, val_records = split_records_by_case(
        records,
        case_id_getter=lambda record: f"{record['dataset']}::{record['case_id']}",
        val_ratio=val_ratio,
        seed=seed,
    )

    train_path = write_jsonl(output_dir / "metadata_inpaint_train.jsonl", train_records)
    val_path = write_jsonl(output_dir / "metadata_inpaint_val.jsonl", val_records)
    return train_path, val_path


def _build_synthetic_record(*, sample, output_dir: Path, config: _SyntheticInpaintConfig) -> dict:
    dataset_name = sample.dataset_name
    source_image = sample.image_path
    target_image = sample.image_path
    target_tissue_mask = sample.tissue_mask_path
    target_nuclei_mask = sample.nuclei_mask_path

    if config.forced_mode == "identity":
        change_region_mask = _write_change_region_mask(
            output_dir=output_dir,
            dataset_name=dataset_name,
            sample_id=sample.sample_id,
            mask=np.zeros_like(load_mask_array(sample.tissue_mask_path), dtype=np.uint8),
        )
        erased_source_image = source_image
        change_ratio = 0.0
        size_bucket = "identity"
    elif config.forced_mode == "near_identity":
        change_region_mask_array = _build_near_identity_mask(
            load_mask_array(sample.tissue_mask_path),
            change_pixels=config.near_identity_change_pixels,
        )
        change_region_mask = _write_change_region_mask(
            output_dir=output_dir,
            dataset_name=dataset_name,
            sample_id=sample.sample_id,
            mask=change_region_mask_array,
        )
        erased_source_image = _materialize_erased_source_image(
            dataset_name=dataset_name,
            sample_id=sample.sample_id,
            source_image=source_image,
            change_region_mask=change_region_mask,
            output_dir=output_dir,
        )
        change_ratio = float((change_region_mask_array > 0).sum() / change_region_mask_array.size)
        size_bucket = "small"
    else:
        raise ValueError(f"Unsupported forced_mode for synthetic inpaint metadata: {config.forced_mode}")

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
        "edit_type": config.forced_mode,
        "change_ratio": change_ratio,
        "mask_mode": config.forced_mode,
        "size_bucket": size_bucket,
    }


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
) -> Path:
    erased_dir = output_dir / "erased_source_images" / dataset_name
    erased_dir.mkdir(parents=True, exist_ok=True)
    erased_path = erased_dir / f"{sample_id}.png"

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


def _write_change_region_mask(*, output_dir: Path, dataset_name: str, sample_id: str, mask: np.ndarray) -> Path:
    mask_dir = output_dir / "change_region_masks" / dataset_name
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / f"{sample_id}.png"
    Image.fromarray(mask.astype(np.uint8)).save(mask_path)
    return mask_path

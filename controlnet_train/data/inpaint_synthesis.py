"""Synthetic inpaint metadata builder for Phase 5."""

from __future__ import annotations

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


def _select_single_component(mask: np.ndarray, seed: int | None = None) -> np.ndarray:
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
    """Create a tissue-side boundary band around one connected tissue component."""
    component = _select_single_component(tissue_mask, seed=seed)
    if not component.any():
        return np.zeros_like(component, dtype=np.uint8)

    band = _component_boundary(component)
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


def _size_bucket_for_change_ratio(change_ratio: float) -> str:
    if change_ratio <= 0.0:
        return "identity"
    if change_ratio <= 0.12:
        return "small"
    if change_ratio <= 0.30:
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
    size_bucket = _size_bucket_for_change_ratio(change_ratio)
    if size_bucket not in _VALID_SIZE_BUCKETS:
        raise ValueError(f"Unsupported synthesized size bucket: {size_bucket}")
    if expected_bucket is not None and size_bucket != expected_bucket:
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
    source_image = sample.image_path
    target_image = sample.image_path
    target_tissue_mask = sample.tissue_mask_path
    target_nuclei_mask = sample.nuclei_mask_path
    mask_mode = _sample_effective_mode(
        config=config,
        sample=sample,
        attempt_seed=attempt_seed,
        variant_index=variant_index,
        excluded_modes=excluded_modes,
    )

    if mask_mode == "identity":
        change_region_mask_array = np.zeros_like(load_mask_array(sample.tissue_mask_path), dtype=np.uint8)
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
            load_mask_array(sample.tissue_mask_path),
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
        change_region_mask_array, mask_mode = synthesize_change_region(
            load_mask_array(sample.tissue_mask_path),
            forced_bucket=mask_mode,
            seed=attempt_seed if attempt_seed is not None else config.seed,
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

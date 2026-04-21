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


@dataclass(frozen=True)
class _SyntheticInpaintConfig:
    forced_mode: str
    near_identity_change_pixels: int = 1


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

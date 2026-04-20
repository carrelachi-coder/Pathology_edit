"""Inpaint metadata builder and dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from .common import (
    default_prompt_for_dataset,
    load_binary_mask,
    load_image_tensor,
    load_nuclei_mask,
    load_tissue_mask,
    parse_sample_identity,
    read_jsonl,
    resolve_path,
    split_records_by_case,
    write_jsonl,
)


def build_inpaint_metadata(
    input_jsonl_paths: Sequence[str | Path],
    output_dir: str | Path,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    normalized_records: list[dict] = []

    for input_path in input_jsonl_paths:
        input_path = Path(input_path)
        for row in read_jsonl(input_path):
            normalized_records.append(
                _normalize_inpaint_record(row, base_dir=input_path.parent, output_dir=output_dir)
            )

    train_records, val_records = split_records_by_case(
        normalized_records,
        case_id_getter=lambda record: f"{record['dataset']}::{record['case_id']}",
        val_ratio=val_ratio,
        seed=seed,
    )

    train_path = write_jsonl(output_dir / "metadata_inpaint_train.jsonl", train_records)
    val_path = write_jsonl(output_dir / "metadata_inpaint_val.jsonl", val_records)
    return train_path, val_path


class InpaintDataset(torch.utils.data.Dataset):
    """Load normalized inpaint metadata and layered mask conditions."""

    def __init__(self, metadata_path: str | Path) -> None:
        self.metadata_path = Path(metadata_path)
        self.records = read_jsonl(self.metadata_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        return {
            "dataset": record["dataset"],
            "sample_id": record["sample_id"],
            "case_id": record["case_id"],
            "source_image": load_image_tensor(record["source_image"]),
            "erased_source_image": load_image_tensor(record["erased_source_image"]),
            "target_image": load_image_tensor(record["target_image"]),
            "target_tissue_mask": load_tissue_mask(record["target_tissue_mask"]),
            "target_nuclei_mask": load_nuclei_mask(record["target_nuclei_mask"], remap=True),
            "change_region_mask": load_binary_mask(record["change_region_mask"]),
            "prompt": record["prompt"],
            "edit_type": record["edit_type"],
            "change_ratio": float(record["change_ratio"]),
            "erased_source_image_path": record["erased_source_image"],
        }


def _normalize_inpaint_record(row: dict, base_dir: Path, output_dir: Path) -> dict:
    required_keys = (
        "dataset",
        "source_image",
        "target_image",
        "target_tissue_mask",
        "target_nuclei_mask",
        "change_region_mask",
    )
    missing = [key for key in required_keys if key not in row]
    if missing:
        raise KeyError(f"Inpaint metadata row missing required keys: {missing}")

    dataset_name = str(row["dataset"]).upper()
    source_image = resolve_path(row["source_image"], base_dir)
    target_image = resolve_path(row["target_image"], base_dir)
    target_tissue_mask = resolve_path(row["target_tissue_mask"], base_dir)
    target_nuclei_mask = resolve_path(row["target_nuclei_mask"], base_dir)
    change_region_mask = resolve_path(row["change_region_mask"], base_dir)

    sample_id = row.get("sample_id") or target_image.stem
    case_id, _, _ = parse_sample_identity(sample_id)
    erased_source_image = row.get("erased_source_image")
    if erased_source_image:
        erased_source_image_path = resolve_path(erased_source_image, base_dir)
    else:
        erased_source_image_path = _materialize_erased_source_image(
            dataset_name=dataset_name,
            sample_id=sample_id,
            source_image=source_image,
            change_region_mask=change_region_mask,
            output_dir=output_dir,
        )

    return {
        "dataset": dataset_name,
        "sample_id": sample_id,
        "case_id": row.get("case_id", case_id),
        "source_image": str(source_image),
        "erased_source_image": str(erased_source_image_path),
        "target_image": str(target_image),
        "target_tissue_mask": str(target_tissue_mask),
        "target_nuclei_mask": str(target_nuclei_mask),
        "change_region_mask": str(change_region_mask),
        "prompt": row.get("prompt") or default_prompt_for_dataset(dataset_name),
        "edit_type": row.get("edit_type", "unspecified"),
        "change_ratio": float(row.get("change_ratio", 0.0)),
    }


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

"""Shared layered-data utilities for Phase 5 ControlNet training."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from PIL import Image

from dataset_config import get_config

_PATCH_COORDS_RE = re.compile(r"^(?P<case_id>.+)_py(?P<py>\d+)_px(?P<px>\d+)$")

_PROMPT_BY_CANCER_TYPE = {
    "breast": "H&E stained breast cancer histopathology at 40x magnification",
    "prostate": "H&E stained prostate cancer histopathology at 40x magnification",
    "colorectal": "H&E stained colorectal cancer histopathology at 40x magnification",
    "lung": "H&E stained lung cancer histopathology at 40x magnification",
    "melanoma": "H&E stained melanoma histopathology at 40x magnification",
    "oral": "H&E stained oral squamous cell carcinoma histopathology at 40x magnification",
}

_NUCLEI_ID_LOOKUP = np.full(106, -1, dtype=np.int64)
for raw_id, internal_id in {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    101: 1,
    102: 2,
    103: 3,
    104: 4,
    105: 5,
}.items():
    _NUCLEI_ID_LOOKUP[raw_id] = internal_id


@dataclass(frozen=True)
class LayeredSample:
    dataset_name: str
    dataset_root: Path
    sample_id: str
    case_id: str
    image_path: Path
    tissue_mask_path: Path
    nuclei_mask_path: Path
    prompt: str
    patch_y: int | None
    patch_x: int | None


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Sequence[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def write_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
    return path


def normalize_metadata_path_value(path_value: str | Path) -> str:
    """Normalize JSON metadata paths written on Windows for POSIX servers."""
    return str(path_value).replace("\\", "/")


def resolve_path(path_value: str | Path, base_dir: str | Path) -> Path:
    candidate = Path(normalize_metadata_path_value(path_value))
    if candidate.is_absolute():
        return candidate
    return Path(base_dir) / candidate


def default_prompt_for_dataset(dataset_name: str) -> str:
    config = get_config(dataset_name)
    return _PROMPT_BY_CANCER_TYPE.get(
        config.cancer_type,
        f"H&E stained {config.cancer_type} histopathology at 40x magnification",
    )


def parse_sample_identity(sample_id: str) -> tuple[str, int | None, int | None]:
    match = _PATCH_COORDS_RE.match(sample_id)
    if not match:
        return sample_id, None, None
    return (
        match.group("case_id"),
        int(match.group("py")),
        int(match.group("px")),
    )


def load_image_tensor(path: str | Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_mask_array(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path))


def load_tissue_mask(path: str | Path) -> torch.Tensor:
    return torch.from_numpy(load_mask_array(path).astype(np.int64))


def remap_nuclei_ids_array(nuclei_mask: np.ndarray) -> np.ndarray:
    nuclei_mask = nuclei_mask.astype(np.int64, copy=False)
    if nuclei_mask.size == 0:
        return nuclei_mask

    min_id = int(nuclei_mask.min())
    max_id = int(nuclei_mask.max())
    if min_id < 0 or max_id >= _NUCLEI_ID_LOOKUP.shape[0]:
        raise ValueError(
            f"nuclei ids out of range: got [{min_id}, {max_id}], expected values within [0, 105]."
        )

    remapped = _NUCLEI_ID_LOOKUP[nuclei_mask]
    if (remapped < 0).any():
        invalid = np.unique(nuclei_mask[remapped < 0]).tolist()
        raise ValueError(f"Unsupported nuclei ids encountered: {invalid}")
    return remapped


def load_nuclei_mask(path: str | Path, remap: bool = True) -> torch.Tensor:
    array = load_mask_array(path)
    if remap:
        array = remap_nuclei_ids_array(array)
    return torch.from_numpy(array.astype(np.int64, copy=False))


def load_binary_mask(path: str | Path) -> torch.Tensor:
    array = np.asarray(Image.open(path))
    if array.ndim == 3:
        array = np.any(array > 0, axis=-1).astype(np.float32)
    else:
        array = (array > 0).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0)


def load_layered_dataset_samples(
    dataset_name: str,
    dataset_root: str | Path,
    metadata_path: str | Path | None = None,
    require_images: bool = True,
) -> list[LayeredSample]:
    dataset_name = dataset_name.upper()
    dataset_root = Path(dataset_root)
    metadata_path = Path(metadata_path) if metadata_path is not None else dataset_root / "metadata.jsonl"

    samples: list[LayeredSample] = []
    if metadata_path.exists():
        for row in read_jsonl(metadata_path):
            image_path = resolve_path(row["image"], dataset_root)
            sample_id = image_path.stem
            tissue_mask_path = dataset_root / "tissue_masks" / f"{sample_id}.png"
            nuclei_mask_path = dataset_root / "nuclei_masks" / f"{sample_id}.png"
            case_id, patch_y, patch_x = parse_sample_identity(sample_id)
            sample = LayeredSample(
                dataset_name=dataset_name,
                dataset_root=dataset_root,
                sample_id=sample_id,
                case_id=case_id,
                image_path=image_path,
                tissue_mask_path=tissue_mask_path,
                nuclei_mask_path=nuclei_mask_path,
                prompt=row.get("text") or default_prompt_for_dataset(dataset_name),
                patch_y=patch_y,
                patch_x=patch_x,
            )
            _validate_sample_paths(sample, require_images=require_images)
            samples.append(sample)
        return samples

    for tissue_mask_path in sorted((dataset_root / "tissue_masks").glob("*.png")):
        sample_id = tissue_mask_path.stem
        image_path = dataset_root / "images" / f"{sample_id}.png"
        nuclei_mask_path = dataset_root / "nuclei_masks" / f"{sample_id}.png"
        case_id, patch_y, patch_x = parse_sample_identity(sample_id)
        sample = LayeredSample(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            sample_id=sample_id,
            case_id=case_id,
            image_path=image_path,
            tissue_mask_path=tissue_mask_path,
            nuclei_mask_path=nuclei_mask_path,
            prompt=default_prompt_for_dataset(dataset_name),
            patch_y=patch_y,
            patch_x=patch_x,
        )
        _validate_sample_paths(sample, require_images=require_images)
        samples.append(sample)
    return samples


def split_records_by_case(
    records: Sequence[dict],
    case_id_getter: Callable[[dict], str],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    if not records:
        return [], []
    if val_ratio <= 0:
        return list(records), []

    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(case_id_getter(record), []).append(record)

    case_ids = list(grouped.keys())
    random.Random(seed).shuffle(case_ids)
    val_case_count = int(round(len(case_ids) * val_ratio))
    if val_case_count <= 0 and len(case_ids) > 1:
        val_case_count = 1
    val_case_ids = set(case_ids[:val_case_count])

    train_records: list[dict] = []
    val_records: list[dict] = []
    for case_id, case_records in grouped.items():
        if case_id in val_case_ids:
            val_records.extend(case_records)
        else:
            train_records.extend(case_records)
    return train_records, val_records


def _validate_sample_paths(sample: LayeredSample, require_images: bool) -> None:
    missing: list[str] = []
    if require_images and not sample.image_path.exists():
        missing.append(str(sample.image_path))
    if not sample.tissue_mask_path.exists():
        missing.append(str(sample.tissue_mask_path))
    if not sample.nuclei_mask_path.exists():
        missing.append(str(sample.nuclei_mask_path))
    if missing:
        raise FileNotFoundError(f"Missing layered sample files for {sample.sample_id}: {missing}")

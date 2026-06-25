"""Dataset for supervised I0 + reference -> target texture transfer.

Expected metadata records use the same fields as the existing cross metadata:
target_image, target_tissue_mask, target_nuclei_mask, reference_image,
reference_tissue_mask, reference_nuclei_mask.

I0 is the ControlNet output with target structure but wrong appearance. It is
the image input to the pix2pix model. It can be read from an I0 image field in
the metadata or from a cache directory produced by precompute_i0.py. The target
image is used only as the supervision target.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from dataset_config import NUM_CELL_CLASSES, NUM_FINE

NUCLEI_LABEL_OFFSET = 256

_NUCLEI_ID_LOOKUP = np.full(106, -1, dtype=np.int64)
for _raw_id, _internal_id in {
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
    _NUCLEI_ID_LOOKUP[_raw_id] = _internal_id


def read_metadata(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf8"))
    if isinstance(payload, dict):
        records = payload.get("pairs") or payload.get("samples") or payload.get("records")
        if not isinstance(records, list):
            raise ValueError(
                f"Metadata dict must contain a pairs/samples/records list: {path}"
            )
        return [dict(row) for row in records]
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    raise TypeError(f"Unsupported metadata payload type: {type(payload)!r}")


def resolve_path(value: str | Path, *, metadata_root: Path | None = None) -> Path:
    path = Path(str(value).replace("\\", "/")).expanduser()
    if path.is_absolute() or metadata_root is None:
        return path
    return metadata_root / path


def metadata_cache_id(record: dict[str, Any], index: int) -> int:
    value = record.get("metadata_index", index)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(index)


def i0_cache_path(cache_dir: str | Path, record: dict[str, Any], index: int) -> Path:
    return Path(cache_dir) / f"{metadata_cache_id(record, index):08d}.png"


def load_rgb(path: str | Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = ImageOps.exif_transpose(image)
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_label_mask(path: str | Path, image_size: int) -> torch.Tensor:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    image = image.resize((image_size, image_size), Image.Resampling.NEAREST)
    array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    return torch.from_numpy(array.astype(np.int64, copy=False)).unsqueeze(0)


def remap_nuclei_mask(mask: torch.Tensor) -> torch.Tensor:
    values = mask.long()
    min_id = int(values.min().item()) if values.numel() else 0
    max_id = int(values.max().item()) if values.numel() else 0
    if min_id < 0 or max_id >= _NUCLEI_ID_LOOKUP.shape[0]:
        raise ValueError(
            f"Nuclei ids out of range [{min_id}, {max_id}], expected 0..5 or 101..105."
        )
    mapped = _NUCLEI_ID_LOOKUP[values.cpu().numpy()]
    if (mapped < 0).any():
        invalid = np.unique(values.cpu().numpy()[mapped < 0]).tolist()
        raise ValueError(f"Unsupported nuclei ids encountered: {invalid}")
    return torch.from_numpy(mapped.astype(np.int64, copy=False))


def one_hot_mask(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    labels = mask.squeeze(0).long().clamp(min=0, max=num_classes - 1)
    return F.one_hot(labels, num_classes=num_classes).permute(2, 0, 1).float()


def tissue_nuclei_region_labels(
    tissue: torch.Tensor,
    nuclei: torch.Tensor,
    *,
    label_mode: str,
) -> torch.Tensor:
    tissue = tissue.long()
    nuclei = nuclei.long()
    mode = str(label_mode or "tissue_nuclei").strip().lower()
    if mode == "tissue":
        return tissue
    if mode == "nuclei":
        return nuclei
    if mode != "tissue_nuclei":
        raise ValueError(
            "--region-label-mode must be one of: tissue, nuclei, tissue_nuclei"
        )
    region = tissue.clone()
    nuclei_pixels = nuclei != 0
    region[nuclei_pixels] = nuclei[nuclei_pixels] + NUCLEI_LABEL_OFFSET
    return region


def maybe_flip(
    image: torch.Tensor,
    tissue: torch.Tensor,
    nuclei: torch.Tensor,
    *,
    hflip: bool,
    vflip: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if hflip:
        image = torch.flip(image, dims=(-1,))
        tissue = torch.flip(tissue, dims=(-1,))
        nuclei = torch.flip(nuclei, dims=(-1,))
    if vflip:
        image = torch.flip(image, dims=(-2,))
        tissue = torch.flip(tissue, dims=(-2,))
        nuclei = torch.flip(nuclei, dims=(-2,))
    return image, tissue, nuclei


def maybe_flip_image(image: torch.Tensor, *, hflip: bool, vflip: bool) -> torch.Tensor:
    if hflip:
        image = torch.flip(image, dims=(-1,))
    if vflip:
        image = torch.flip(image, dims=(-2,))
    return image


class I0ReferenceTextureDataset(Dataset):
    """Load I0/reference/target triplets for supervised texture transfer."""

    def __init__(
        self,
        metadata_path: str | Path,
        *,
        image_size: int = 256,
        i0_field: str = "i0_image",
        i0_cache_dir: str | Path | None = None,
        allow_missing_i0: bool = False,
        metadata_root: str | Path | None = None,
        max_samples: int | None = None,
        region_label_mode: str = "tissue_nuclei",
        augment_flips: bool = False,
        split: str = "train",
    ) -> None:
        self.metadata_path = Path(metadata_path)
        self.metadata_root = (
            Path(metadata_root)
            if metadata_root is not None
            else self.metadata_path.parent
        )
        self.records = read_metadata(self.metadata_path)
        for original_index, record in enumerate(self.records):
            record.setdefault("metadata_index", original_index)
        if max_samples is not None:
            self.records = self.records[: int(max_samples)]
        self.image_size = int(image_size)
        self.i0_field = str(i0_field)
        self.i0_cache_dir = Path(i0_cache_dir) if i0_cache_dir is not None else None
        self.allow_missing_i0 = bool(allow_missing_i0)
        self.region_label_mode = str(region_label_mode)
        self.augment_flips = bool(augment_flips)
        print(
            f"[{split}] loaded {len(self.records)} pix2pix transfer samples "
            f"from {self.metadata_path}"
        )

    def __len__(self) -> int:
        return len(self.records)

    def _path(self, record: dict[str, Any], field: str) -> Path:
        if field not in record or not record[field]:
            raise KeyError(
                f"Metadata record is missing {field!r}. Available fields: "
                f"{sorted(record.keys())}"
            )
        return resolve_path(record[field], metadata_root=self.metadata_root)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        s = self.image_size

        if self.i0_cache_dir is not None:
            i0_path = i0_cache_path(self.i0_cache_dir, record, index)
            i0_missing = not i0_path.exists()
            if i0_missing and not self.allow_missing_i0:
                raise FileNotFoundError(
                    f"Missing cached I0 image: {i0_path}. "
                    "Run python -m controlnet_train.pix2pix_transfer.precompute_i0 first."
                )
        else:
            i0_path = self._path(record, self.i0_field)
            i0_missing = False
        i0 = torch.zeros(3, s, s) if i0_missing else load_rgb(i0_path, s)
        target = load_rgb(self._path(record, "target_image"), s)
        reference = load_rgb(self._path(record, "reference_image"), s)

        target_tissue = load_label_mask(self._path(record, "target_tissue_mask"), s)
        reference_tissue = load_label_mask(self._path(record, "reference_tissue_mask"), s)
        target_nuclei = remap_nuclei_mask(
            load_label_mask(self._path(record, "target_nuclei_mask"), s)
        )
        reference_nuclei = remap_nuclei_mask(
            load_label_mask(self._path(record, "reference_nuclei_mask"), s)
        )

        if self.augment_flips:
            target_hflip = random.random() < 0.5
            target_vflip = random.random() < 0.5
            ref_hflip = random.random() < 0.5
            ref_vflip = random.random() < 0.5
            i0, target_tissue, target_nuclei = maybe_flip(
                i0,
                target_tissue,
                target_nuclei,
                hflip=target_hflip,
                vflip=target_vflip,
            )
            target = maybe_flip_image(target, hflip=target_hflip, vflip=target_vflip)
            reference, reference_tissue, reference_nuclei = maybe_flip(
                reference,
                reference_tissue,
                reference_nuclei,
                hflip=ref_hflip,
                vflip=ref_vflip,
            )

        target_cond = torch.cat(
            [
                i0,
                one_hot_mask(target_tissue, NUM_FINE),
                one_hot_mask(target_nuclei, NUM_CELL_CLASSES + 1),
            ],
            dim=0,
        )
        reference_cond = torch.cat(
            [
                reference,
                one_hot_mask(reference_tissue, NUM_FINE),
                one_hot_mask(reference_nuclei, NUM_CELL_CLASSES + 1),
            ],
            dim=0,
        )

        target_region = tissue_nuclei_region_labels(
            target_tissue,
            target_nuclei,
            label_mode=self.region_label_mode,
        )
        reference_region = tissue_nuclei_region_labels(
            reference_tissue,
            reference_nuclei,
            label_mode=self.region_label_mode,
        )

        return {
            "target_cond": target_cond,
            "reference_cond": reference_cond,
            "i0": i0,
            "reference_image": reference,
            "target_image": target,
            "target_region": target_region,
            "reference_region": reference_region,
            "target_tissue_mask": target_tissue,
            "target_nuclei_mask": target_nuclei,
            "reference_tissue_mask": reference_tissue,
            "reference_nuclei_mask": reference_nuclei,
            "sample_id": str(record.get("sample_id") or index),
            "reference_sample_id": str(record.get("reference_sample_id") or ""),
            "dataset": str(record.get("dataset") or ""),
            "prompt": str(record.get("prompt") or ""),
            "metadata_index": metadata_cache_id(record, index),
            "i0_missing": bool(i0_missing),
            "i0_cache_path": str(i0_path),
        }

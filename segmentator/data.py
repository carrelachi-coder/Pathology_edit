from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from .config import DatasetManifest, SampleRecord


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_image_tensor(image: torch.Tensor) -> torch.Tensor:
    return TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD)


def load_mask(path: Path) -> torch.Tensor:
    mask = Image.open(path).convert("L")
    return torch.from_numpy(np.array(mask, dtype=np.int64))


class TissueSegmentationDataset(Dataset):
    def __init__(self, records: list[SampleRecord], image_size: int, augment: bool = False, num_classes: int = 8, remap_invalid_to: int = 7) -> None:
        self.records = records
        self.image_size = image_size
        self.augment = augment
        self.num_classes = num_classes
        self.remap_invalid_to = remap_invalid_to

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        record = self.records[idx]
        image = Image.open(record.image_path).convert("RGB")
        mask = Image.open(record.mask_path).convert("L")

        if self.augment and random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image = TF.resize(image, [self.image_size, self.image_size])
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)

        image_t = normalize_image_tensor(TF.to_tensor(image))
        mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))
        invalid = (mask_t < 0) | (mask_t >= self.num_classes)
        mask_t[invalid] = self.remap_invalid_to
        return {"image": image_t, "mask": mask_t, "sample_id": record.sample_id}


def _sorted_pngs(path: Path) -> list[Path]:
    return sorted([p for p in path.iterdir() if p.suffix.lower() == ".png"])


def build_manifest(root: Path, train_count: int, val_count: int, seed: int = 42) -> DatasetManifest:
    if (root / "patches" / "images").exists():
        images_dir = root / "patches" / "images"
        masks_dir = root / "patches" / "tissue_masks"
    else:
        images_dir = root / "images"
        masks_dir = root / "tissue_masks"
    if not images_dir.exists():
        raise FileNotFoundError(images_dir)
    if not masks_dir.exists():
        raise FileNotFoundError(masks_dir)
    image_paths = _sorted_pngs(images_dir)
    if len(image_paths) < train_count + val_count:
        raise ValueError(f"not enough images under {images_dir}: need {train_count + val_count}, found {len(image_paths)}")

    rng = random.Random(seed)
    shuffled = image_paths[:]
    rng.shuffle(shuffled)
    selected = shuffled[: train_count + val_count]
    train_paths = selected[:train_count]
    val_paths = selected[train_count:]

    def make_record(path: Path) -> SampleRecord:
        mask_path = masks_dir / path.name
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        return SampleRecord(image_path=path, mask_path=mask_path, sample_id=path.stem)

    return DatasetManifest(
        root=root,
        train=tuple(make_record(p) for p in train_paths),
        val=tuple(make_record(p) for p in val_paths),
    )


def load_manifest(manifest_path: Path, root: Path | None = None) -> DatasetManifest:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_root = root or Path(payload["dataset_root"])
    if (manifest_root / "patches" / "images").exists():
        images_dir = manifest_root / "patches" / "images"
        masks_dir = manifest_root / "patches" / "tissue_masks"
    else:
        images_dir = manifest_root / "images"
        masks_dir = manifest_root / "tissue_masks"

    def make_record(name: str) -> SampleRecord:
        image_path = images_dir / name
        mask_path = masks_dir / name
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        return SampleRecord(image_path=image_path, mask_path=mask_path, sample_id=image_path.stem)

    return DatasetManifest(
        root=manifest_root,
        train=tuple(make_record(name) for name in payload["train"]),
        val=tuple(make_record(name) for name in payload["val"]),
    )

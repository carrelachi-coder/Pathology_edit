from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from collections import Counter
import warnings

from PIL import Image, UnidentifiedImageError
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from .config import DatasetManifest, SampleRecord
from dataset_config.unified_labels import FINE_TO_PARENT


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_image_tensor(image: torch.Tensor) -> torch.Tensor:
    return TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD)


def _open_image(path: Path, mode: str) -> Image.Image:
    try:
        with Image.open(path) as image:
            loaded = image.convert(mode)
            loaded.load()
            return loaded
    except (OSError, UnidentifiedImageError) as exc:
        raise OSError(f"failed to load image file {path}: {exc}") from exc


def load_mask(path: Path) -> torch.Tensor:
    mask = _open_image(path, "L")
    return torch.from_numpy(np.array(mask, dtype=np.int64))


def infer_dataset_id(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    if "bcss_patches" in parts or "bcss" in parts or name.startswith("bcss"):
        return "bcss"
    if "ignite_patches" in parts or "ignite" in parts or name.startswith("ignite"):
        return "ignite"
    if "orca_patches" in parts or "orca" in parts or name.startswith("orca"):
        return "orca"
    if "panda_patches" in parts or "panda" in parts or name.startswith("panda"):
        return "panda"
    if "glas_patches" in parts or "glas" in parts or "glas" in name:
        return "glas"
    if "puma_patches" in parts or "puma" in parts or name.startswith("puma"):
        return "puma"
    return "default"


def coarse_remap_table(strategy: str = "auto", num_classes: int = 8, ignore_index: int = 255) -> torch.Tensor:
    table = torch.full((256,), int(ignore_index), dtype=torch.long)
    if strategy in {"auto", "fine_to_coarse"}:
        for fine_id, coarse_id in FINE_TO_PARENT.items():
            if 0 <= coarse_id < num_classes:
                table[int(fine_id)] = int(coarse_id)
    elif strategy == "coarse":
        for idx in range(num_classes):
            table[idx] = idx
    elif strategy == "ignore_invalid":
        for idx in range(num_classes):
            table[idx] = idx
    else:
        raise ValueError(f"unsupported mask remap strategy: {strategy}")
    table[ignore_index] = ignore_index
    return table


def remap_mask_to_coarse(mask: torch.Tensor, table: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    mask = mask.long()
    remapped = torch.full_like(mask, int(ignore_index))
    valid = (mask >= 0) & (mask < table.numel())
    remapped[valid] = table[mask[valid]]
    return remapped


class TissueSegmentationDataset(Dataset):
    def __init__(
        self,
        records: list[SampleRecord],
        image_size: int,
        augment: bool = False,
        num_classes: int = 8,
        remap_invalid_to: int = 7,
        ignore_index: int = 255,
        mask_remap: str = "auto",
    ) -> None:
        self.records = records
        self.image_size = image_size
        self.augment = augment
        self.num_classes = num_classes
        self.remap_invalid_to = remap_invalid_to
        self.ignore_index = ignore_index
        self.mask_remap = mask_remap
        self._remap_table = coarse_remap_table(mask_remap, num_classes=num_classes, ignore_index=ignore_index)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        if not self.records:
            raise IndexError("TissueSegmentationDataset is empty")

        start_idx = idx % len(self.records)
        last_error: Exception | None = None
        for offset in range(len(self.records)):
            record = self.records[(start_idx + offset) % len(self.records)]
            try:
                return self._load_item(record)
            except OSError as exc:
                last_error = exc
                warnings.warn(
                    f"Skipping unreadable segmentator sample {record.sample_id}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        raise RuntimeError("no readable segmentator samples remain") from last_error

    def _load_item(self, record: SampleRecord) -> dict[str, torch.Tensor | str]:
        image = _open_image(record.image_path, "RGB")
        mask = _open_image(record.mask_path, "L")

        if self.augment and random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image = TF.resize(image, [self.image_size, self.image_size])
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)

        image_t = normalize_image_tensor(TF.to_tensor(image))
        mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))
        mask_t = remap_mask_to_coarse(mask_t, self._remap_table, ignore_index=self.ignore_index)
        return {"image": image_t, "mask": mask_t, "sample_id": record.sample_id, "dataset_id": record.dataset_id}


def dataset_balanced_weights(records: list[SampleRecord]) -> torch.DoubleTensor:
    counts = Counter(record.dataset_id for record in records)
    weights = [1.0 / counts[record.dataset_id] for record in records]
    return torch.DoubleTensor(weights)


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

    dataset_id = infer_dataset_id(root)

    def make_record(path: Path) -> SampleRecord:
        mask_path = masks_dir / path.name
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        return SampleRecord(image_path=path, mask_path=mask_path, sample_id=path.stem, dataset_id=dataset_id)

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

    def make_record(item: str | dict[str, str]) -> SampleRecord:
        if isinstance(item, dict):
            dataset_root = Path(item.get("dataset_root", manifest_root))
            if not dataset_root.is_absolute():
                dataset_root = manifest_root / dataset_root
            item_images_dir = Path(item.get("images_dir", "images"))
            item_masks_dir = Path(item.get("masks_dir", "tissue_masks"))
            if not item_images_dir.is_absolute():
                item_images_dir = dataset_root / item_images_dir
            if not item_masks_dir.is_absolute():
                item_masks_dir = dataset_root / item_masks_dir
            image_name = item["image"]
            mask_name = item.get("mask", Path(image_name).name)
            image_path = item_images_dir / image_name
            mask_path = item_masks_dir / mask_name
            dataset_id = item.get("dataset_id", infer_dataset_id(dataset_root))
            sample_id = item.get("sample_id", f"{dataset_id}:{Path(image_name).stem}")
        else:
            image_path = images_dir / item
            mask_path = masks_dir / item
            dataset_id = infer_dataset_id(manifest_root)
            sample_id = image_path.stem
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        return SampleRecord(image_path=image_path, mask_path=mask_path, sample_id=sample_id, dataset_id=dataset_id)

    return DatasetManifest(
        root=manifest_root,
        train=tuple(make_record(name) for name in payload["train"]),
        val=tuple(make_record(name) for name in payload["val"]),
    )

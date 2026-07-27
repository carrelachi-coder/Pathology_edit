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
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from .config import DatasetManifest, SEGMENTATOR_CLASSES, SampleRecord
from .stain_augmentation import StainAugmentationConfig, build_stain_augmenter, maybe_apply_stain_augmentation
from dataset_config import get_config
from dataset_config.unified_labels import FINE_TO_PARENT, NUM_COARSE, NUM_FINE


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


CELL_IDS = (101, 102, 103, 104, 105)


def nuclei_mask_to_density(mask: torch.Tensor, sigma: float = 8.0) -> torch.Tensor:
    """Convert CellViT semantic nuclei masks into center-based Gaussian maps."""
    from scipy import ndimage

    mask_array = mask.detach().cpu().numpy()
    impulses = np.zeros((len(CELL_IDS), *mask_array.shape), dtype=np.float32)
    connectivity = ndimage.generate_binary_structure(2, 1)
    for channel, cell_id in enumerate(CELL_IDS):
        binary = mask_array == cell_id
        components, count = ndimage.label(binary, structure=connectivity)
        if count == 0:
            continue
        centers = ndimage.center_of_mass(binary, components, range(1, count + 1))
        for y, x in centers:
            row = int(np.clip(round(y), 0, mask_array.shape[0] - 1))
            column = int(np.clip(round(x), 0, mask_array.shape[1] - 1))
            impulses[channel, row, column] += 1.0
    channels = torch.from_numpy(impulses)
    if sigma > 0:
        radius = max(1, int(round(3.0 * sigma)))
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-(coords**2) / (2.0 * sigma**2))
        channels = F.conv2d(
            channels.unsqueeze(0),
            kernel.view(1, 1, 1, -1).expand(len(CELL_IDS), 1, 1, -1),
            padding=(0, radius),
            groups=len(CELL_IDS),
        )
        channels = F.conv2d(
            channels,
            kernel.view(1, 1, -1, 1).expand(len(CELL_IDS), 1, -1, 1),
            padding=(radius, 0),
            groups=len(CELL_IDS),
        ).squeeze(0)
    channels = channels.clamp(0.0, 1.0)
    total = channels.sum(dim=0, keepdim=True).clamp(0.0, 1.0)
    return torch.cat([channels, total], dim=0)


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


def fine_supervision_for_dataset(dataset_id: str) -> torch.Tensor:
    """Return dataset-specific fine children for every coarse parent class."""
    allowed = torch.zeros((NUM_COARSE, NUM_FINE), dtype=torch.bool)
    allowed[torch.arange(NUM_COARSE), torch.arange(NUM_COARSE)] = True
    try:
        dataset_config = get_config(dataset_id)
    except KeyError:
        return allowed
    fine_ids_by_parent: dict[int, set[int]] = {}
    for fine_id in dataset_config.to_fine_map.values():
        fine_id = int(fine_id)
        if 0 <= fine_id < NUM_FINE:
            fine_ids_by_parent.setdefault(FINE_TO_PARENT[fine_id], set()).add(fine_id)
    for parent_id, fine_ids in fine_ids_by_parent.items():
        allowed[parent_id].zero_()
        allowed[parent_id, sorted(fine_ids)] = True
    return allowed


def build_fine_target(mask: torch.Tensor, dataset_id: str, ignore_index: int = 255) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep labels only where the dataset distinguishes multiple children of a parent."""
    allowed = fine_supervision_for_dataset(dataset_id)
    target = torch.full_like(mask.long(), int(ignore_index))
    valid = (mask >= 0) & (mask < NUM_FINE)
    if valid.any():
        parent_lookup = torch.tensor([FINE_TO_PARENT[idx] for idx in range(NUM_FINE)], dtype=torch.long)
        safe_mask = mask.clamp(0, NUM_FINE - 1).long()
        parent = parent_lookup[safe_mask]
        branching_parent = allowed.sum(dim=1) > 1
        supervised = valid & branching_parent[parent]
        target[supervised] = safe_mask[supervised]
    return target, allowed


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
        stain_augmentation: StainAugmentationConfig | None = None,
        cellvit_mode: str = "none",
        cell_density_sigma: float = 8.0,
        augment_vflip: bool = False,
        augment_rot90: bool = False,
        augment_scale_crop: float = 0.0,
        hierarchical_fine: bool = False,
    ) -> None:
        self.records = records
        self.image_size = image_size
        self.augment = augment
        self.num_classes = num_classes
        self.remap_invalid_to = remap_invalid_to
        self.ignore_index = ignore_index
        self.mask_remap = mask_remap
        self._remap_table = coarse_remap_table(mask_remap, num_classes=num_classes, ignore_index=ignore_index)
        self.stain_augmentation = stain_augmentation or StainAugmentationConfig()
        self._stain_augmenter = build_stain_augmenter(self.stain_augmentation) if augment else None
        self.cellvit_mode = cellvit_mode
        self.cell_density_sigma = cell_density_sigma
        self.augment_vflip = augment_vflip
        self.augment_rot90 = augment_rot90
        self.augment_scale_crop = augment_scale_crop
        self.hierarchical_fine = hierarchical_fine

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
        nuclei = _open_image(record.nuclei_path, "L") if record.nuclei_path is not None and record.nuclei_path.exists() else None

        if self.augment and random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            if nuclei is not None:
                nuclei = TF.hflip(nuclei)
        if self.augment and self.augment_vflip and random.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            if nuclei is not None:
                nuclei = TF.vflip(nuclei)
        if self.augment and self.augment_rot90:
            angle = random.choice((0, 90, 180, 270))
            if angle:
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
                if nuclei is not None:
                    nuclei = TF.rotate(nuclei, angle, interpolation=TF.InterpolationMode.NEAREST)
        if self.augment and self.augment_scale_crop > 0:
            crop_fraction = random.uniform(max(0.5, 1.0 - self.augment_scale_crop), 1.0)
            crop_size = max(1, int(round(min(image.size) * crop_fraction)))
            top = random.randint(0, max(image.height - crop_size, 0))
            left = random.randint(0, max(image.width - crop_size, 0))
            image = TF.crop(image, top, left, crop_size, crop_size)
            mask = TF.crop(mask, top, left, crop_size, crop_size)
            if nuclei is not None:
                nuclei = TF.crop(nuclei, top, left, crop_size, crop_size)

        image = TF.resize(image, [self.image_size, self.image_size])
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)
        if nuclei is not None:
            nuclei = TF.resize(nuclei, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)

        if self.augment:
            image = maybe_apply_stain_augmentation(
                image,
                self._stain_augmenter,
                self.stain_augmentation.probability,
            )

        image_t = normalize_image_tensor(TF.to_tensor(image))
        raw_mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))
        mask_t = remap_mask_to_coarse(raw_mask_t, self._remap_table, ignore_index=self.ignore_index)
        result: dict[str, torch.Tensor | str] = {
            "image": image_t,
            "mask": mask_t,
            "sample_id": record.sample_id,
            "dataset_id": record.dataset_id,
            "group_id": record.group_id or record.sample_id,
        }
        if self.hierarchical_fine:
            fine_mask, fine_allowed = build_fine_target(raw_mask_t, record.dataset_id, ignore_index=self.ignore_index)
            result["fine_mask"] = fine_mask
            result["fine_allowed"] = fine_allowed
        if self.cellvit_mode != "none":
            nuclei_t = torch.from_numpy(np.array(nuclei, dtype=np.int64)) if nuclei is not None else torch.zeros_like(mask_t)
            result["nuclei_density"] = nuclei_mask_to_density(nuclei_t, sigma=self.cell_density_sigma)
            has_cells = any(bool((nuclei_t == cell_id).any()) for cell_id in CELL_IDS)
            result["nuclei_available"] = torch.tensor(float(nuclei is not None and has_cells))
        return result


def dataset_balanced_weights(records: list[SampleRecord], temperature: float = 0.0) -> torch.DoubleTensor:
    if not 0.0 <= temperature <= 1.0:
        raise ValueError("dataset sampling temperature must be in [0, 1]")
    counts = Counter(record.dataset_id for record in records)
    weights = [counts[record.dataset_id] ** (temperature - 1.0) for record in records]
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
        nuclei_path = masks_dir.parent / "nuclei_masks" / path.name
        return SampleRecord(
            image_path=path,
            mask_path=mask_path,
            sample_id=path.stem,
            dataset_id=dataset_id,
            group_id=path.stem,
            nuclei_path=nuclei_path if nuclei_path.exists() else None,
        )

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
            item_nuclei_dir = Path(item.get("nuclei_dir", "nuclei_masks"))
            if not item_images_dir.is_absolute():
                item_images_dir = dataset_root / item_images_dir
            if not item_masks_dir.is_absolute():
                item_masks_dir = dataset_root / item_masks_dir
            if not item_nuclei_dir.is_absolute():
                item_nuclei_dir = dataset_root / item_nuclei_dir
            image_name = item["image"]
            mask_name = item.get("mask", Path(image_name).name)
            nuclei_name = item.get("nuclei", Path(image_name).name)
            image_path = item_images_dir / image_name
            mask_path = item_masks_dir / mask_name
            nuclei_path = item_nuclei_dir / nuclei_name
            dataset_id = item.get("dataset_id", infer_dataset_id(dataset_root))
            sample_id = item.get("sample_id", f"{dataset_id}:{Path(image_name).stem}")
            group_id = item.get("group_id", sample_id)
        else:
            image_path = images_dir / item
            mask_path = masks_dir / item
            nuclei_path = masks_dir.parent / "nuclei_masks" / item
            dataset_id = infer_dataset_id(manifest_root)
            sample_id = image_path.stem
            group_id = sample_id
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        return SampleRecord(
            image_path=image_path,
            mask_path=mask_path,
            sample_id=sample_id,
            dataset_id=dataset_id,
            group_id=group_id,
            nuclei_path=nuclei_path if nuclei_path.exists() else None,
        )

    return DatasetManifest(
        root=manifest_root,
        train=tuple(make_record(name) for name in payload["train"]),
        val=tuple(make_record(name) for name in payload["val"]),
        test=tuple(make_record(name) for name in payload.get("test", [])),
        classes=tuple(payload.get("classes") or SEGMENTATOR_CLASSES),
    )

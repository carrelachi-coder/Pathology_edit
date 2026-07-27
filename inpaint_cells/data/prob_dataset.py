"""
NucleiProbDataset — ProbNet training dataset (Phase 4.1).

Supports two data loading modes:

1. **Layered storage (AD-1, preferred)**:
   Each sample directory contains separate tissue_mask.png + nuclei_mask.png.
   Loaded directly as integer arrays — no RGB conversion needed.

2. **Legacy LaMa format (backward compatible)**:
   RGB combined masks in ground_truth/ + train/ + *_mask001.png format.
   Auto-detected when layered storage is not available.

Data flow (Phase 4.1):
  tissue_mask.png → tissue_map (int64, 0-15)
  nuclei_mask.png → cell_map (int64, 0-5, edit region zeroed)
  edit_mask → mask (float32, 0/1)
  DatasetConfig → cancer_id (int64, 0-5)
  → ProbNetInputEncoder (Embedding lookup) → (B, 17, H, W) → UNet

Multi-dataset support:
  - Accepts list of (data_dir, dataset_name) pairs
  - Each sample carries cancer_id from DatasetConfig.cancer_type_index
  - Sampling weights proportional to dataset size (configurable)
"""

import os
import glob
import random
import logging
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler

from ..utils.mask_utils import (
    load_tissue_mask, load_nuclei_mask,
    NUM_TISSUE, NUM_NUCLEI, NUCLEI_RAW_TO_INDEX,
)
from .density_targets import (
    build_center_density_targets,
    expand_edit_mask_to_complete_instances,
    extract_class_centers,
)

logger = logging.getLogger(__name__)


def _choose_crop_origin(
    h: int,
    w: int,
    out_size: int,
    edit_mask: np.ndarray,
    mode: str,
    deterministic: bool = False,
) -> Tuple[int, int]:
    """Choose a crop origin. Mask mode keeps the erased region visible when possible."""
    max_y = max(0, h - out_size)
    max_x = max(0, w - out_size)
    if mode == 'mask' and edit_mask is not None and edit_mask.any():
        ys, xs = np.where(edit_mask > 0.5)
        if deterministic:
            cy = (int(ys.min()) + int(ys.max())) // 2
            cx = (int(xs.min()) + int(xs.max())) // 2
        else:
            idx = random.randrange(len(ys))
            cy = int(ys[idx])
            cx = int(xs[idx])
        y = min(max(cy - out_size // 2, 0), max_y)
        x = min(max(cx - out_size // 2, 0), max_x)
        return y, x
    if deterministic:
        return max_y // 2, max_x // 2
    return random.randint(0, max_y), random.randint(0, max_x)


# ============================================================
#  Single-dataset loader (layered storage, AD-1)
# ============================================================

class NucleiProbDatasetLayered(Dataset):
    """
    ProbNet dataset for layered storage format (AD-1).

    Each sample directory structure:
        {sample_dir}/
            tissue_mask.png     — uint8, values 0-15 (unified fine IDs)
            nuclei_mask.png     — uint8, values 0/101-105
            edit_mask.png       — uint8, binary (255=edit region)

    Or each sample is a flat triplet:
        {data_dir}/tissue/{name}_tissue.png
        {data_dir}/nuclei/{name}_nuclei.png
        {data_dir}/masks/{name}_mask.png

    Args:
        data_dir: root directory containing samples
        cancer_type_index: cancer type index (0-5) for this dataset
        out_size: crop size (default 256)
        augment: enable random flip/rotation augmentation
        gt_tissue_dir: optional separate GT tissue dir (for training with erased inputs)
        gt_nuclei_dir: optional separate GT nuclei dir
    """

    def __init__(
        self,
        data_dir: str,
        cancer_type_index: int,
        out_size: int = 256,
        augment: bool = True,
        crop_mode: str = 'mask',
        dataset_name: str = 'unknown',
        center_density_targets: bool = False,
        density_sigma: float = 2.0,
        complete_instance_erasure: bool = True,
    ):
        self.data_dir = data_dir
        self.cancer_type_index = cancer_type_index
        self.out_size = out_size
        self.augment = augment
        self.crop_mode = crop_mode
        self.dataset_name = dataset_name
        self.center_density_targets = center_density_targets
        self.density_sigma = density_sigma
        self.complete_instance_erasure = complete_instance_erasure

        self.samples = self._discover_samples()
        logger.info(
            f"NucleiProbDatasetLayered: {len(self.samples)} samples "
            f"from {data_dir} (cancer_type={cancer_type_index})"
        )

    def _discover_samples(self) -> List[Dict[str, str]]:
        """Auto-discover samples from directory structure."""
        samples = []

        # Pattern 1: subdirectory per sample
        #   {data_dir}/{sample_name}/tissue_mask.png, nuclei_mask.png, edit_mask.png
        subdirs = sorted(glob.glob(os.path.join(self.data_dir, '*', 'tissue_mask.png')))
        if subdirs:
            for tissue_path in subdirs:
                sample_dir = os.path.dirname(tissue_path)
                nuclei_path = os.path.join(sample_dir, 'nuclei_mask.png')
                mask_path = os.path.join(sample_dir, 'edit_mask.png')
                if os.path.exists(nuclei_path) and os.path.exists(mask_path):
                    samples.append({
                        'name': os.path.basename(sample_dir),
                        'gt_tissue': tissue_path,
                        'gt_nuclei': nuclei_path,
                        'input_tissue': tissue_path,  # same as GT (edit region defined by mask)
                        'input_nuclei': nuclei_path,
                        'mask': mask_path,
                    })
            return samples

        # Pattern 2: flat directory with naming convention
        #   {data_dir}/gt_tissue/{name}.png
        #   {data_dir}/gt_nuclei/{name}.png
        #   {data_dir}/input_tissue/{name}.png  (= gt_tissue for edit training)
        #   {data_dir}/input_nuclei/{name}.png  (nuclei with edit region zeroed)
        #   {data_dir}/masks/{name}.png
        gt_tissue_dir = os.path.join(self.data_dir, 'gt_tissue')
        gt_nuclei_dir = os.path.join(self.data_dir, 'gt_nuclei')
        input_tissue_dir = os.path.join(self.data_dir, 'input_tissue')
        input_nuclei_dir = os.path.join(self.data_dir, 'input_nuclei')
        masks_dir = os.path.join(self.data_dir, 'masks')

        if os.path.isdir(gt_tissue_dir) and os.path.isdir(masks_dir):
            gt_tissue_files = sorted(glob.glob(os.path.join(gt_tissue_dir, '*.png')))
            for gt_t_path in gt_tissue_files:
                name = os.path.basename(gt_t_path)
                gt_n_path = os.path.join(gt_nuclei_dir, name)
                mask_path = os.path.join(masks_dir, name)

                # Input tissue defaults to GT tissue (edit region tissue is already updated)
                in_t_path = os.path.join(input_tissue_dir, name) if os.path.isdir(input_tissue_dir) else gt_t_path
                if not os.path.exists(in_t_path):
                    in_t_path = gt_t_path

                # Input nuclei defaults to GT nuclei (will be zeroed by mask at runtime)
                in_n_path = os.path.join(input_nuclei_dir, name) if os.path.isdir(input_nuclei_dir) else gt_n_path
                if not os.path.exists(in_n_path):
                    in_n_path = gt_n_path

                if os.path.exists(gt_n_path) and os.path.exists(mask_path):
                    samples.append({
                        'name': os.path.splitext(name)[0],
                        'gt_tissue': gt_t_path,
                        'gt_nuclei': gt_n_path,
                        'input_tissue': in_t_path,
                        'input_nuclei': in_n_path,
                        'mask': mask_path,
                    })
            return samples

        logger.warning(f"No samples found in {self.data_dir}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Load integer maps directly (AD-1: layered storage)
        gt_tissue = load_tissue_mask(s['gt_tissue'])          # (H, W) int64, 0-15
        gt_nuclei = load_nuclei_mask(s['gt_nuclei'], remap=True)  # (H, W) int64, 0-5
        input_tissue = load_tissue_mask(s['input_tissue'])
        input_nuclei = load_nuclei_mask(s['input_nuclei'], remap=True)

        mask_bin = cv2.imread(s['mask'], cv2.IMREAD_GRAYSCALE)
        edit_mask = (mask_bin > 128).astype(np.float32)
        density_centers = (
            extract_class_centers(gt_nuclei)
            if self.center_density_targets
            else None
        )

        if self.complete_instance_erasure:
            erasure_mask = expand_edit_mask_to_complete_instances(
                gt_nuclei,
                edit_mask,
            ).astype(np.float32)
        else:
            erasure_mask = edit_mask.copy()

        h, w = gt_tissue.shape[:2]

        # Crop to the training size. Prefer edit-mask-centered crops so the
        # supervised erased region remains visible after cropping.
        if h > self.out_size or w > self.out_size:
            y, x = _choose_crop_origin(
                h,
                w,
                self.out_size,
                edit_mask,
                self.crop_mode,
                deterministic=not self.augment,
            )
            gt_tissue = gt_tissue[y:y+self.out_size, x:x+self.out_size]
            gt_nuclei = gt_nuclei[y:y+self.out_size, x:x+self.out_size]
            input_tissue = input_tissue[y:y+self.out_size, x:x+self.out_size]
            input_nuclei = input_nuclei[y:y+self.out_size, x:x+self.out_size]
            edit_mask = edit_mask[y:y+self.out_size, x:x+self.out_size]
            erasure_mask = erasure_mask[y:y+self.out_size, x:x+self.out_size]
            if density_centers is not None:
                density_centers = [
                    (class_id, center_y - y, center_x - x)
                    for class_id, center_y, center_x in density_centers
                ]

        # Zero nuclei in edit region (AD-3: generate from scratch)
        input_nuclei_masked = input_nuclei.copy()
        input_nuclei_masked[erasure_mask > 0.5] = 0

        density_target = None
        target_count_table = None
        if self.center_density_targets:
            density_target, target_count_table = build_center_density_targets(
                gt_nuclei,
                input_tissue,
                edit_mask,
                sigma=self.density_sigma,
                centers=density_centers,
            )

        # Data augmentation
        if self.augment:
            (
                gt_tissue,
                gt_nuclei,
                input_tissue,
                input_nuclei_masked,
                edit_mask,
                erasure_mask,
                density_target,
            ) = self._augment(
                gt_tissue,
                gt_nuclei,
                input_tissue,
                input_nuclei_masked,
                edit_mask,
                erasure_mask,
                density_target,
            )

        # Target: GT nuclei for supervision
        target = gt_nuclei.astype(np.int64)

        # Convert to tensors
        # tissue_map and cell_map are int64 for Embedding lookup (AD-4)
        edit_mask = edit_mask[np.newaxis, :, :]  # (1, H, W)
        erasure_mask = erasure_mask[np.newaxis, :, :]

        result = {
            'tissue_map': torch.from_numpy(input_tissue.astype(np.int64)),       # (H, W)
            'cell_map': torch.from_numpy(input_nuclei_masked.astype(np.int64)),  # (H, W)
            'mask': torch.from_numpy(edit_mask).float(),                          # (1, H, W)
            'erasure_mask': torch.from_numpy(erasure_mask).float(),                # (1, H, W)
            'cancer_id': torch.tensor(self.cancer_type_index, dtype=torch.int64), # scalar
            'target': torch.from_numpy(target),                                   # (H, W)
            'dataset_name': self.dataset_name,
            'sample_id': s['name'],
        }
        if density_target is not None:
            result['density_target'] = torch.from_numpy(density_target)
            result['target_count_table'] = torch.from_numpy(target_count_table)
        return result

    def _augment(
        self,
        gt_tissue,
        gt_nuclei,
        input_tissue,
        input_nuclei,
        edit_mask,
        erasure_mask,
        density_target,
    ):
        """Random flip and rotation augmentation for integer maps."""
        # Horizontal flip
        if random.random() > 0.5:
            gt_tissue = gt_tissue[:, ::-1].copy()
            gt_nuclei = gt_nuclei[:, ::-1].copy()
            input_tissue = input_tissue[:, ::-1].copy()
            input_nuclei = input_nuclei[:, ::-1].copy()
            edit_mask = edit_mask[:, ::-1].copy()
            erasure_mask = erasure_mask[:, ::-1].copy()
            if density_target is not None:
                density_target = density_target[:, :, ::-1].copy()

        # Vertical flip
        if random.random() > 0.5:
            gt_tissue = gt_tissue[::-1, :].copy()
            gt_nuclei = gt_nuclei[::-1, :].copy()
            input_tissue = input_tissue[::-1, :].copy()
            input_nuclei = input_nuclei[::-1, :].copy()
            edit_mask = edit_mask[::-1, :].copy()
            erasure_mask = erasure_mask[::-1, :].copy()
            if density_target is not None:
                density_target = density_target[:, ::-1, :].copy()

        # 90-degree rotation
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            gt_tissue = np.rot90(gt_tissue, k).copy()
            gt_nuclei = np.rot90(gt_nuclei, k).copy()
            input_tissue = np.rot90(input_tissue, k).copy()
            input_nuclei = np.rot90(input_nuclei, k).copy()
            edit_mask = np.rot90(edit_mask, k).copy()
            erasure_mask = np.rot90(erasure_mask, k).copy()
            if density_target is not None:
                density_target = np.rot90(
                    density_target,
                    k,
                    axes=(1, 2),
                ).copy()

        return (
            gt_tissue,
            gt_nuclei,
            input_tissue,
            input_nuclei,
            edit_mask,
            erasure_mask,
            density_target,
        )


# ============================================================
#  Legacy LaMa format loader (backward compatible)
# ============================================================

class NucleiProbDatasetLegacy(Dataset):
    """
    ProbNet dataset for legacy LaMa format (RGB combined masks).

    Directory structure:
        {root}/ground_truth/{name}.png  — GT RGB mask (tissue + nuclei)
        {root}/train/{name}.png         — erased RGB mask
        {root}/train/{name}_mask001.png — edit region binary mask

    Converts RGB → integer maps at load time for Embedding input.

    Args:
        gt_dir: ground truth PNG directory
        train_dir: erased PNG + mask directory (train/ or val/)
        cancer_type_index: cancer type index (0-5)
        out_size: crop size (default 256)
        augment: enable augmentation
    """

    def __init__(
        self,
        gt_dir: str,
        train_dir: str,
        cancer_type_index: int = 0,
        out_size: int = 256,
        augment: bool = True,
        crop_mode: str = 'mask',
        dataset_name: str = 'unknown',
        center_density_targets: bool = False,
        density_sigma: float = 2.0,
        complete_instance_erasure: bool = True,
    ):
        self.gt_dir = gt_dir
        self.train_dir = train_dir
        self.cancer_type_index = cancer_type_index
        self.out_size = out_size
        self.augment = augment
        self.crop_mode = crop_mode
        self.dataset_name = dataset_name
        self.center_density_targets = center_density_targets
        self.density_sigma = density_sigma
        self.complete_instance_erasure = complete_instance_erasure

        all_gt = sorted(glob.glob(os.path.join(gt_dir, '*.png')))
        self.samples = []
        for gt_path in all_gt:
            fname = os.path.basename(gt_path)
            train_path = os.path.join(train_dir, fname)
            mask_path = os.path.join(train_dir, fname.replace('.png', '_mask001.png'))
            if os.path.exists(train_path) and os.path.exists(mask_path):
                self.samples.append({
                    'name': os.path.splitext(fname)[0],
                    'gt': gt_path,
                    'input': train_path,
                    'mask': mask_path,
                })

        logger.info(
            f"NucleiProbDatasetLegacy: {len(self.samples)} samples "
            f"(cancer_type={cancer_type_index})"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        gt_rgb = cv2.cvtColor(cv2.imread(s['gt']), cv2.COLOR_BGR2RGB)
        input_rgb = cv2.cvtColor(cv2.imread(s['input']), cv2.COLOR_BGR2RGB)
        mask_bin = cv2.imread(s['mask'], cv2.IMREAD_GRAYSCALE)

        h, w = gt_rgb.shape[:2]

        edit_mask_full = (mask_bin > 128).astype(np.float32)

        # Crop to the training size. Prefer edit-mask-centered crops so the
        # supervised erased region remains visible after cropping.
        if h > self.out_size or w > self.out_size:
            y, x = _choose_crop_origin(
                h,
                w,
                self.out_size,
                edit_mask_full,
                self.crop_mode,
                deterministic=not self.augment,
            )
            gt_rgb = gt_rgb[y:y+self.out_size, x:x+self.out_size]
            input_rgb = input_rgb[y:y+self.out_size, x:x+self.out_size]
            mask_bin = mask_bin[y:y+self.out_size, x:x+self.out_size]

        # Convert RGB → integer maps
        gt_tissue, gt_nuclei = self._rgb_to_layers(gt_rgb)
        input_tissue, input_nuclei = self._rgb_to_layers(input_rgb)

        edit_mask = (mask_bin > 128).astype(np.float32)

        if self.complete_instance_erasure:
            erasure_mask = expand_edit_mask_to_complete_instances(
                gt_nuclei,
                edit_mask,
            ).astype(np.float32)
        else:
            erasure_mask = edit_mask.copy()

        # Zero nuclei in edit region (AD-3)
        input_nuclei[erasure_mask > 0.5] = 0

        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                gt_tissue = gt_tissue[:, ::-1].copy()
                gt_nuclei = gt_nuclei[:, ::-1].copy()
                input_tissue = input_tissue[:, ::-1].copy()
                input_nuclei = input_nuclei[:, ::-1].copy()
                edit_mask = edit_mask[:, ::-1].copy()
                erasure_mask = erasure_mask[:, ::-1].copy()
            if random.random() > 0.5:
                gt_tissue = gt_tissue[::-1, :].copy()
                gt_nuclei = gt_nuclei[::-1, :].copy()
                input_tissue = input_tissue[::-1, :].copy()
                input_nuclei = input_nuclei[::-1, :].copy()
                edit_mask = edit_mask[::-1, :].copy()
                erasure_mask = erasure_mask[::-1, :].copy()
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])
                gt_tissue = np.rot90(gt_tissue, k).copy()
                gt_nuclei = np.rot90(gt_nuclei, k).copy()
                input_tissue = np.rot90(input_tissue, k).copy()
                input_nuclei = np.rot90(input_nuclei, k).copy()
                edit_mask = np.rot90(edit_mask, k).copy()
                erasure_mask = np.rot90(erasure_mask, k).copy()

        target = gt_nuclei.astype(np.int64)
        density_target = None
        target_count_table = None
        if self.center_density_targets:
            density_target, target_count_table = build_center_density_targets(
                target,
                input_tissue,
                edit_mask,
                sigma=self.density_sigma,
            )
        edit_mask = edit_mask[np.newaxis, :, :]
        erasure_mask = erasure_mask[np.newaxis, :, :]

        result = {
            'tissue_map': torch.from_numpy(input_tissue.astype(np.int64)),
            'cell_map': torch.from_numpy(input_nuclei.astype(np.int64)),
            'mask': torch.from_numpy(edit_mask).float(),
            'erasure_mask': torch.from_numpy(erasure_mask).float(),
            'cancer_id': torch.tensor(self.cancer_type_index, dtype=torch.int64),
            'target': torch.from_numpy(target),
            'dataset_name': self.dataset_name,
            'sample_id': s['name'],
        }
        if density_target is not None:
            result['density_target'] = torch.from_numpy(density_target)
            result['target_count_table'] = torch.from_numpy(target_count_table)
        return result

    @staticmethod
    def _rgb_to_layers(rgb_img):
        """
        Convert legacy RGB combined mask to tissue + nuclei integer layers.
        Uses EDT to infer tissue under nuclei pixels (legacy workaround).
        """
        from ..utils.mask_utils import COLOR_MAP

        # Build RGB -> value lookup
        rgb_to_val = {}
        for val, rgb in COLOR_MAP.items():
            key = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
            rgb_to_val[key] = val

        encoded = (rgb_img[:, :, 0].astype(np.int64) * 65536
                   + rgb_img[:, :, 1].astype(np.int64) * 256
                   + rgb_img[:, :, 2].astype(np.int64))

        class_map = np.zeros(rgb_img.shape[:2], dtype=np.int64)
        for key, val in rgb_to_val.items():
            class_map[encoded == key] = val

        # Split tissue and nuclei
        tissue = class_map.copy()
        nuclei = np.zeros_like(class_map)

        nuclei_classes = [101, 102, 103, 104, 105]
        for i, nuc_val in enumerate(nuclei_classes):
            mask = class_map == nuc_val
            nuclei[mask] = i + 1

        nuc_mask = class_map >= 100
        if nuc_mask.any():
            from scipy.ndimage import distance_transform_edt
            _, nearest_idx = distance_transform_edt(
                nuc_mask, return_distances=True, return_indices=True
            )
            tissue[nuc_mask] = class_map[nearest_idx[0][nuc_mask], nearest_idx[1][nuc_mask]]

        # Clamp tissue to valid range [0, 15] for unified labels
        tissue = np.clip(tissue, 0, NUM_TISSUE - 1)

        return tissue, nuclei


# ============================================================
#  Multi-dataset combined loader
# ============================================================

def build_multi_dataset(
    dataset_configs: List[Dict],
    split: str = 'train',
    out_size: int = 256,
    augment: bool = True,
    crop_mode: str = 'mask',
    center_density_targets: bool = False,
    density_sigma: float = 2.0,
    complete_instance_erasure: bool = True,
) -> Tuple[ConcatDataset, WeightedRandomSampler]:
    """
    Build a combined dataset from multiple data sources with weighted sampling.

    Args:
        dataset_configs: list of dicts, each with:
            - 'data_dir': path to dataset directory
            - 'dataset_name': name for get_config() lookup
            - 'format': 'layered' or 'legacy' (default: auto-detect)
        split: 'train' or 'val'
        out_size: crop size
        augment: enable augmentation

    Returns:
        (combined_dataset, weighted_sampler)

    Example:
        configs = [
            {'data_dir': '/data/bcss_probnet', 'dataset_name': 'BCSS'},
            {'data_dir': '/data/panda_probnet', 'dataset_name': 'PANDA'},
        ]
        dataset, sampler = build_multi_dataset(configs, split='train')
        loader = DataLoader(dataset, batch_size=16, sampler=sampler)
    """
    from dataset_config import get_config

    datasets = []
    sizes = []

    for cfg_dict in dataset_configs:
        data_dir = cfg_dict['data_dir']
        ds_name = cfg_dict['dataset_name']
        fmt = cfg_dict.get('format', 'auto')

        ds_config = get_config(ds_name)
        cancer_idx = ds_config.cancer_type_index

        # Auto-detect format
        if fmt == 'auto':
            # Check for layered storage markers
            has_layered = (
                os.path.isdir(os.path.join(data_dir, 'gt_tissue'))
                or os.path.isdir(os.path.join(data_dir, split, 'gt_tissue'))
                or os.path.isdir(os.path.join(data_dir, 'train', 'gt_tissue'))
                or os.path.isdir(os.path.join(data_dir, 'val', 'gt_tissue'))
                or len(glob.glob(os.path.join(data_dir, '*', 'tissue_mask.png'))) > 0
            )
            fmt = 'layered' if has_layered else 'legacy'

        if fmt == 'layered':
            split_dir = os.path.join(data_dir, split) if os.path.isdir(os.path.join(data_dir, split)) else data_dir
            ds = NucleiProbDatasetLayered(
                data_dir=split_dir,
                cancer_type_index=cancer_idx,
                out_size=out_size,
                augment=augment and (split == 'train'),
                crop_mode=crop_mode,
                dataset_name=ds_name,
                center_density_targets=center_density_targets,
                density_sigma=density_sigma,
                complete_instance_erasure=complete_instance_erasure,
            )
        else:
            # Legacy LaMa format
            ds = NucleiProbDatasetLegacy(
                gt_dir=os.path.join(data_dir, 'ground_truth'),
                train_dir=os.path.join(data_dir, split),
                cancer_type_index=cancer_idx,
                out_size=out_size,
                augment=augment and (split == 'train'),
                crop_mode=crop_mode,
                dataset_name=ds_name,
                center_density_targets=center_density_targets,
                density_sigma=density_sigma,
                complete_instance_erasure=complete_instance_erasure,
            )

        if len(ds) > 0:
            datasets.append(ds)
            sizes.append(len(ds))
            logger.info(f"  {ds_name} ({fmt}): {len(ds)} {split} samples, cancer_idx={cancer_idx}")

    if not datasets:
        raise RuntimeError(f"No valid datasets found for split '{split}'")

    combined = ConcatDataset(datasets)

    # Weighted sampling: probability proportional to sqrt(dataset_size)
    # sqrt balances between equal-weight and size-proportional
    total_sqrt = sum(s ** 0.5 for s in sizes)
    weights_per_dataset = [(s ** 0.5) / total_sqrt for s in sizes]

    sample_weights = []
    for ds_idx, ds in enumerate(datasets):
        w = weights_per_dataset[ds_idx] / sizes[ds_idx]
        sample_weights.extend([w] * sizes[ds_idx])

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(combined),
        replacement=True,
    )

    logger.info(
        f"Combined dataset: {len(combined)} total samples, "
        f"{len(datasets)} datasets, "
        f"weights={[f'{w:.3f}' for w in weights_per_dataset]}"
    )

    return combined, sampler


# ============================================================
#  Backward-compatible alias
# ============================================================

# For scripts that import the old class name
NucleiProbDataset = NucleiProbDatasetLegacy

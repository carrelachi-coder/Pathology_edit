"""
ProbNet 训练数据准备脚本 — Phase 4.4 多数据集适配
====================================================

Phase 4.4 相比旧版 (v2) 的改动:
    1. 支持分层存储 (AD-1): 直接读取 tissue_mask.png + nuclei_mask.png
    2. 使用统一 16 类 fine 标签, 替代 BCSS 22 类硬编码
    3. 添加 --dataset 参数, 从 DatasetConfig 读取 skip_tissues / cancer_type
    4. 输出分层格式: gt_tissue/ + gt_nuclei/ + masks/ + meta.json
    5. 删除 is_cell_pixel() / get_tissue_background() 等 RGB workaround 函数
    6. 保留 legacy RGB 输入兼容 (自动检测)

擦除模式概率 (不变):
    - 20% 负样本 (无细胞区域, 教模型"这里不该填细胞")
    - 20% 全图擦除 (所有细胞清零)
    - 20% 大区域擦除 (多种组织全部擦除)
    - 40% 局部擦除 (原有的 full/partial 模式)

输出目录结构 (layered):
    {output_dir}/
        train/
            gt_tissue/{name}.png      — tissue mask (uint8, 0-15)
            gt_nuclei/{name}.png      — nuclei mask (uint8, 0/101-105)
            masks/{name}.png          — edit region binary mask (uint8, 0/255)
        val/
            gt_tissue/{name}.png
            gt_nuclei/{name}.png
            masks/{name}.png
        meta.json                     — 数据集元信息 (dataset, cancer_type, etc.)

用法:
    # 分层存储输入 (推荐)
    python inpaint_cells/data/prepare_dataset.py \\
        --dataset BCSS \\
        --input-dir /data/BCSS_patches \\
        --output-dir /data/probnet_training_BCSS \\
        --n-augmentations 3

    # 旧 RGB 合并输入 (自动检测)
    python inpaint_cells/data/prepare_dataset.py \\
        --dataset BCSS \\
        --input-dir /data/BCSS_dataset/conditioning \\
        --output-dir /data/probnet_training_BCSS \\
        --format legacy

    # 多数据集批量准备
    python inpaint_cells/data/prepare_dataset.py \\
        --dataset PANDA \\
        --input-dir /data/PANDA_patches \\
        --output-dir /data/probnet_training_PANDA
"""

import os
import sys
import json
import hashlib
import argparse
import glob
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataset_config import get_config
from dataset_config.unified_labels import (
    NUM_FINE, CELL_IDS, FULL_COLOR_MAP,
)

# ============================================================
# 常量
# ============================================================

NUCLEI_CLASSES = CELL_IDS  # [101, 102, 103, 104, 105]

MIN_CELL_PIXELS = 50
MIN_REGION_PIXELS = 200

# 擦除模式概率
PROB_NEGATIVE = 0.20
PROB_FULL_IMAGE = 0.20
PROB_LARGE_REGION = 0.20
# 剩余 0.40 = 局部擦除


# ============================================================
# 输入加载: 分层存储 (AD-1, 推荐)
# ============================================================

def load_sample_layered(tissue_path, nuclei_path):
    """
    从分层存储加载 tissue_map 和 nuclei_map.

    Returns:
        tissue_map: (H, W) int64, 0-15 (unified fine IDs)
        nuclei_map: (H, W) int64, 0/101-105 (raw CellViT IDs)
        cell_mask: (H, W) bool, True where nuclei exist
    """
    tissue_img = cv2.imread(tissue_path, cv2.IMREAD_GRAYSCALE)
    if tissue_img is None:
        raise FileNotFoundError(f"Cannot load tissue mask: {tissue_path}")
    nuclei_img = cv2.imread(nuclei_path, cv2.IMREAD_GRAYSCALE)
    if nuclei_img is None:
        raise FileNotFoundError(f"Cannot load nuclei mask: {nuclei_path}")

    tissue_map = tissue_img.astype(np.int64)
    nuclei_map = nuclei_img.astype(np.int64)
    cell_mask = nuclei_map >= 101
    return tissue_map, nuclei_map, cell_mask


# ============================================================
# 输入加载: Legacy RGB (向后兼容)
# ============================================================

# RGB → class value 查找表
_rgb_to_val = {}
for _val, _rgb in FULL_COLOR_MAP.items():
    _key = _rgb[0] * 65536 + _rgb[1] * 256 + _rgb[2]
    _rgb_to_val[_key] = _val


def load_sample_legacy(rgb_path):
    """
    从旧 RGB 合并 mask 加载, 拆分为 tissue_map + nuclei_map.

    使用 EDT 反推核下面的组织类型 (legacy workaround).

    Returns:
        tissue_map: (H, W) int64, 0-15
        nuclei_map: (H, W) int64, 0/101-105
        cell_mask: (H, W) bool
    """
    bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None, None
    rgb_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # RGB → class value map
    encoded = (rgb_img[:, :, 0].astype(np.int64) * 65536
               + rgb_img[:, :, 1].astype(np.int64) * 256
               + rgb_img[:, :, 2].astype(np.int64))
    class_map = np.zeros(rgb_img.shape[:2], dtype=np.int64)
    for key, val in _rgb_to_val.items():
        class_map[encoded == key] = val

    # 拆分 tissue 和 nuclei
    nuclei_map = np.zeros_like(class_map)
    cell_mask = np.zeros(class_map.shape, dtype=bool)
    for nuc_val in NUCLEI_CLASSES:
        mask = class_map == nuc_val
        nuclei_map[mask] = nuc_val
        cell_mask |= mask

    # tissue: 用 EDT 反推核下面的组织
    tissue_map = class_map.copy()
    if cell_mask.any():
        _, nearest_idx = ndimage.distance_transform_edt(
            cell_mask, return_distances=True, return_indices=True
        )
        tissue_map[cell_mask] = class_map[nearest_idx[0][cell_mask],
                                           nearest_idx[1][cell_mask]]

    # clamp to valid fine range
    tissue_map = np.clip(tissue_map, 0, NUM_FINE - 1)

    return tissue_map, nuclei_map, cell_mask


# ============================================================
# 组织区域识别 (直接基于整数 tissue_map)
# ============================================================

def identify_tissue_regions(tissue_map, skip_tissues):
    """
    识别 tissue_map 中各组织类型区域.

    Args:
        tissue_map: (H, W) int64, 0-15
        skip_tissues: set of tissue IDs to skip

    Returns:
        dict: {tissue_id: bool_mask}
    """
    regions = {}
    for tid in range(NUM_FINE):
        if tid in skip_tissues:
            continue
        mask = tissue_map == tid
        if mask.sum() > 0:
            regions[tid] = mask
    return regions


# ============================================================
# 擦除区域生成 (基于整数 map, 不依赖 RGB)
# ============================================================

def _draw_random_ellipses(
    h: int, w: int,
    constraint_region: np.ndarray,
    target_area: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """在 constraint_region 内画随机椭圆"""
    coords = np.argwhere(constraint_region)
    if len(coords) == 0:
        return np.zeros((h, w), dtype=bool)

    n_ellipses = int(rng.integers(1, 4))
    result = np.zeros((h, w), dtype=bool)

    for _ in range(n_ellipses):
        center = coords[rng.integers(0, len(coords))]
        per_area = target_area / n_ellipses
        aspect = rng.uniform(0.5, 2.0)
        r = np.sqrt(max(per_area, 25) / (np.pi * aspect))
        a = max(int(r * aspect), 5)
        b = max(int(r), 5)
        angle = rng.uniform(0, 360)

        ellipse_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            ellipse_mask,
            center=(int(center[1]), int(center[0])),
            axes=(a, b),
            angle=angle,
            startAngle=0, endAngle=360,
            color=1, thickness=-1,
        )
        result |= ellipse_mask.astype(bool)

    result &= constraint_region
    return result


def generate_full_image_erasure(
    tissue_map: np.ndarray,
    cell_mask: np.ndarray,
    skip_tissues: set,
    rng: np.random.Generator,
) -> Optional[Tuple[np.ndarray, str]]:
    """全图擦除: mask = 所有有生物学意义的组织区域"""
    if cell_mask.sum() < MIN_CELL_PIXELS:
        return None

    tissue_regions = identify_tissue_regions(tissue_map, skip_tissues)
    erasure = np.zeros_like(cell_mask)
    for _tid, region in tissue_regions.items():
        erasure |= region

    if erasure.sum() < MIN_REGION_PIXELS:
        return None

    return (erasure, "full_image")


def generate_large_region_erasure(
    tissue_map: np.ndarray,
    cell_mask: np.ndarray,
    skip_tissues: set,
    rng: np.random.Generator,
) -> Optional[Tuple[np.ndarray, str]]:
    """大区域擦除: 随机选 2-4 种组织类型, 擦除它们全部区域的细胞"""
    tissue_regions = identify_tissue_regions(tissue_map, skip_tissues)

    candidates = []
    for tid, region in tissue_regions.items():
        if (cell_mask & region).sum() >= MIN_CELL_PIXELS:
            candidates.append(tid)

    if len(candidates) < 1:
        return None

    n_pick = min(int(rng.integers(2, 5)), len(candidates))
    chosen = rng.choice(candidates, size=n_pick, replace=False).tolist()

    erasure = np.zeros_like(cell_mask)
    for tid in chosen:
        erasure |= tissue_regions[tid]

    if erasure.sum() < MIN_REGION_PIXELS:
        return None

    return (erasure, "large_region")


def generate_local_erasure(
    tissue_map: np.ndarray,
    cell_mask: np.ndarray,
    skip_tissues: set,
    rng: np.random.Generator,
) -> Optional[Tuple[np.ndarray, str]]:
    """局部擦除: 在单个组织区域内 full 或 partial 擦除"""
    h, w = tissue_map.shape
    tissue_regions = identify_tissue_regions(tissue_map, skip_tissues)

    positive_candidates = {}
    for tid, region in tissue_regions.items():
        if (cell_mask & region).sum() >= MIN_CELL_PIXELS:
            positive_candidates[tid] = region

    if not positive_candidates:
        return None

    pos_tid = rng.choice(list(positive_candidates.keys()))
    selected_region = positive_candidates[pos_tid]

    erase_mode = rng.choice(["full", "partial"], p=[0.3, 0.7])

    if erase_mode == "full":
        erasure = selected_region.copy()
    else:
        target_area = rng.uniform(0.05, 0.50) * selected_region.sum()
        erasure = _draw_random_ellipses(h, w, selected_region, target_area, rng)

    effective = erasure & cell_mask
    if effective.sum() < MIN_CELL_PIXELS:
        erasure = selected_region.copy()
        effective = erasure & cell_mask
        if effective.sum() < MIN_CELL_PIXELS:
            return None

    return (erasure, "local")


def generate_negative_erasure(
    tissue_map: np.ndarray,
    cell_mask: np.ndarray,
    skip_tissues: set,
    rng: np.random.Generator,
) -> Optional[Tuple[np.ndarray, str]]:
    """负样本: 在无细胞的组织区域擦除, GT 和输入一样"""
    h, w = tissue_map.shape
    tissue_regions = identify_tissue_regions(tissue_map, skip_tissues)

    negative_candidates = {}
    for tid, region in tissue_regions.items():
        if (cell_mask & region).sum() < MIN_CELL_PIXELS and region.sum() >= MIN_REGION_PIXELS:
            negative_candidates[tid] = region

    if not negative_candidates:
        return None

    neg_tid = rng.choice(list(negative_candidates.keys()))
    selected_region = negative_candidates[neg_tid]

    target_area = rng.uniform(0.05, 0.50) * selected_region.sum()
    erasure = _draw_random_ellipses(h, w, selected_region, target_area, rng)

    if erasure.sum() < MIN_REGION_PIXELS:
        erasure = selected_region.copy()

    return (erasure, "negative")


# ============================================================
# 统一调度
# ============================================================

def generate_erasure_region(
    tissue_map: np.ndarray,
    cell_mask: np.ndarray,
    skip_tissues: set,
    rng: np.random.Generator,
) -> Optional[Tuple[np.ndarray, str]]:
    """按概率选择擦除模式, 返回 (erasure_mask, mode_name) 或 None"""
    roll = rng.random()

    if roll < PROB_NEGATIVE:
        result = generate_negative_erasure(tissue_map, cell_mask, skip_tissues, rng)
        if result is not None:
            return result
        return generate_local_erasure(tissue_map, cell_mask, skip_tissues, rng)

    elif roll < PROB_NEGATIVE + PROB_FULL_IMAGE:
        result = generate_full_image_erasure(tissue_map, cell_mask, skip_tissues, rng)
        if result is not None:
            return result
        return generate_local_erasure(tissue_map, cell_mask, skip_tissues, rng)

    elif roll < PROB_NEGATIVE + PROB_FULL_IMAGE + PROB_LARGE_REGION:
        result = generate_large_region_erasure(tissue_map, cell_mask, skip_tissues, rng)
        if result is not None:
            return result
        return generate_local_erasure(tissue_map, cell_mask, skip_tissues, rng)

    else:
        return generate_local_erasure(tissue_map, cell_mask, skip_tissues, rng)


# ============================================================
# 单样本处理
# ============================================================

def process_single_sample(
    tissue_map: np.ndarray,
    nuclei_map: np.ndarray,
    cell_mask: np.ndarray,
    skip_tissues: set,
    rng: np.random.Generator,
    n_augmentations: int = 3,
) -> list:
    """
    对一个样本的 tissue_map + nuclei_map 生成多组训练数据.

    Returns:
        list of (tissue_map, nuclei_map, erasure_mask, mode_name)
        tissue_map 和 nuclei_map 是原始整数 map (不变),
        erasure_mask 是二值 mask (bool), mode_name 是擦除模式名.
    """
    results = []
    for _ in range(n_augmentations):
        result = generate_erasure_region(tissue_map, cell_mask, skip_tissues, rng)
        if result is None:
            continue

        erasure_mask, mode_name = result
        results.append((tissue_map, nuclei_map, erasure_mask, mode_name))

    return results


# ============================================================
# 批量处理: 分层存储输入
# ============================================================

def _discover_layered_samples(input_dir):
    """
    自动发现分层存储格式的样本.

    支持两种目录结构:
        Pattern 1: {input_dir}/{sample_name}/tissue_mask.png + nuclei_mask.png
        Pattern 2: {input_dir}/gt_tissue/{name}.png + gt_nuclei/{name}.png
    """
    samples = []

    # Pattern 0: upload staging/edit_datasets format:
    #   {input_dir}/tissue_masks/{name}.png + nuclei_masks/{name}.png
    tissue_masks_dir = os.path.join(input_dir, 'tissue_masks')
    nuclei_masks_dir = os.path.join(input_dir, 'nuclei_masks')
    if os.path.isdir(tissue_masks_dir) and os.path.isdir(nuclei_masks_dir):
        tissue_files = sorted(glob.glob(os.path.join(tissue_masks_dir, '*.png')))
        for t_path in tissue_files:
            name = os.path.splitext(os.path.basename(t_path))[0]
            n_path = os.path.join(nuclei_masks_dir, os.path.basename(t_path))
            if os.path.exists(n_path):
                samples.append({
                    'name': name,
                    'tissue': t_path,
                    'nuclei': n_path,
                })
        return samples

    # Pattern 1: subdirectory per sample
    subdirs = sorted(glob.glob(os.path.join(input_dir, '*', 'tissue_mask.png')))
    if subdirs:
        for tissue_path in subdirs:
            sample_dir = os.path.dirname(tissue_path)
            nuclei_path = os.path.join(sample_dir, 'nuclei_mask.png')
            if os.path.exists(nuclei_path):
                name = os.path.basename(sample_dir)
                samples.append({
                    'name': name,
                    'tissue': tissue_path,
                    'nuclei': nuclei_path,
                })
        return samples

    # Pattern 2: flat directory
    gt_tissue_dir = os.path.join(input_dir, 'gt_tissue')
    gt_nuclei_dir = os.path.join(input_dir, 'gt_nuclei')

    if os.path.isdir(gt_tissue_dir) and os.path.isdir(gt_nuclei_dir):
        tissue_files = sorted(glob.glob(os.path.join(gt_tissue_dir, '*.png')))
        for t_path in tissue_files:
            name = os.path.splitext(os.path.basename(t_path))[0]
            n_path = os.path.join(gt_nuclei_dir, os.path.basename(t_path))
            if os.path.exists(n_path):
                samples.append({
                    'name': name,
                    'tissue': t_path,
                    'nuclei': n_path,
                })
        return samples

    return samples


def _discover_legacy_samples(input_dir):
    """发现旧 RGB 合并 mask 文件."""
    samples = []
    mask_files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    if not mask_files:
        mask_files = sorted(glob.glob(os.path.join(input_dir, '*.jpg')))
    for f in mask_files:
        name = os.path.splitext(os.path.basename(f))[0]
        samples.append({'name': name, 'rgb_path': f})
    return samples


def _load_grouped_split_assignments(
    manifest_path: str,
    dataset_name: str,
) -> Tuple[Dict[str, str], Dict[str, int], str]:
    """Load train/val/test source assignments from a grouped Segmentator manifest."""
    with open(manifest_path, 'rb') as handle:
        raw = handle.read()
    payload = json.loads(raw)
    dataset_id = dataset_name.lower()
    assignments: Dict[str, str] = {}
    group_sets = {'train': set(), 'val': set(), 'test': set()}

    for split in ('train', 'val', 'test'):
        for record in payload.get(split, []):
            if str(record.get('dataset_id', '')).lower() != dataset_id:
                continue
            sample_name = Path(str(record.get('image', record.get('sample_id', '')))).stem
            if not sample_name:
                raise ValueError(f"Grouped split record has no image/sample_id: {record}")
            previous = assignments.get(sample_name)
            if previous is not None and previous != split:
                raise ValueError(
                    f"Sample {dataset_name}/{sample_name} occurs in both {previous} and {split}."
                )
            assignments[sample_name] = split
            group_id = record.get('group_id')
            if group_id is not None:
                group_sets[split].add(str(group_id))

    if not assignments:
        raise ValueError(
            f"No records for dataset {dataset_name!r} were found in split manifest {manifest_path}."
        )
    if not any(split == 'train' for split in assignments.values()):
        raise ValueError(f"Grouped split has no train records for {dataset_name}.")
    if not any(split == 'val' for split in assignments.values()):
        raise ValueError(f"Grouped split has no val records for {dataset_name}.")

    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        overlap = group_sets[left] & group_sets[right]
        if overlap:
            raise ValueError(
                f"Grouped manifest leaks {dataset_name} groups across {left}/{right}: "
                f"{sorted(overlap)[:5]}"
            )

    group_counts = {split: len(groups) for split, groups in group_sets.items()}
    return assignments, group_counts, hashlib.sha256(raw).hexdigest()


def prepare_dataset(
    input_dir: str,
    output_dir: str,
    dataset_name: str,
    fmt: str = 'auto',
    val_ratio: float = 0.1,
    n_augmentations: int = 3,
    seed: int = 42,
    split_manifest: Optional[str] = None,
):
    """
    准备 ProbNet 训练数据.

    Args:
        input_dir: 输入目录 (分层或 legacy)
        output_dir: 输出目录
        dataset_name: 数据集名称 (用于获取配置)
        fmt: 输入格式 ('auto', 'layered', 'legacy')
        val_ratio: 验证集比例
        n_augmentations: 每个样本的擦除增强数量
        seed: 随机种子
    """
    rng = np.random.default_rng(seed)

    # 加载数据集配置
    config = get_config(dataset_name)
    skip_tissues = set(config.skip_tissues)
    cancer_type_index = config.cancer_type_index

    logger.info(f"Dataset: {config.name} ({config.cancer_type})")
    logger.info(f"  cancer_type_index: {cancer_type_index}")
    logger.info(f"  skip_tissues: {skip_tissues}")
    logger.info(f"  label_granularity: {config.label_granularity}")

    # 自动检测输入格式
    if fmt == 'auto':
        layered_samples = _discover_layered_samples(input_dir)
        fmt = 'layered' if layered_samples else 'legacy'
        logger.info(f"  Auto-detected format: {fmt}")

    # 发现样本
    if fmt == 'layered':
        all_samples = _discover_layered_samples(input_dir)
    else:
        all_samples = _discover_legacy_samples(input_dir)

    if not all_samples:
        logger.error(f"No samples found in {input_dir}")
        return

    logger.info(f"Found {len(all_samples)} input samples")

    # Train/val split. A grouped manifest is required for leakage-safe formal runs.
    split_assignments = None
    split_group_counts = None
    split_manifest_sha256 = None
    if split_manifest:
        split_assignments, split_group_counts, split_manifest_sha256 = \
            _load_grouped_split_assignments(split_manifest, dataset_name)
        discovered_names = {sample['name'] for sample in all_samples}
        missing = sorted(discovered_names - set(split_assignments))
        if missing:
            raise ValueError(
                f"Grouped split manifest does not cover {len(missing)} discovered {dataset_name} "
                f"samples; examples: {missing[:5]}"
            )
        logger.info(
            f"Using grouped split manifest: {split_manifest} "
            f"(groups={split_group_counts})"
        )
        val_indices = set()
    else:
        logger.warning(
            "No --split-manifest supplied; falling back to patch-level random split. "
            "Do not use this fallback for formal evaluation."
        )
        n_val = max(int(len(all_samples) * val_ratio), 1)
        indices = rng.permutation(len(all_samples))
        val_indices = set(indices[:n_val].tolist())

    # 创建输出目录 (分层格式)
    output_path = Path(output_dir)
    for split in ['train', 'val']:
        (output_path / split / 'gt_tissue').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'gt_nuclei').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'masks').mkdir(parents=True, exist_ok=True)

    # 统计
    stats = {"train": 0, "val": 0}
    mode_counts = {"negative": 0, "full_image": 0, "large_region": 0, "local": 0}
    skipped = 0
    held_out_test_sources = 0

    for file_idx, sample in enumerate(all_samples):
        # 加载样本
        if fmt == 'layered':
            tissue_map, nuclei_map, cell_mask = load_sample_layered(
                sample['tissue'], sample['nuclei'])
        else:
            tissue_map, nuclei_map, cell_mask = load_sample_legacy(sample['rgb_path'])
            if tissue_map is None:
                logger.warning(f"Cannot read {sample['rgb_path']}, skipping.")
                skipped += 1
                continue

        if split_assignments is not None:
            split = split_assignments[sample['name']]
            if split == 'test':
                held_out_test_sources += 1
                continue
        else:
            split = "val" if file_idx in val_indices else "train"

        # 生成擦除样本
        augmented = process_single_sample(
            tissue_map, nuclei_map, cell_mask,
            skip_tissues, rng, n_augmentations)

        if not augmented:
            skipped += 1
            continue

        sample_name = sample['name']

        for aug_idx, (t_map, n_map, erasure_mask, mode_name) in enumerate(augmented):
            out_name = f"{sample_name}_{aug_idx:03d}.png"

            # 保存 tissue mask (uint8, 0-15)
            cv2.imwrite(
                str(output_path / split / 'gt_tissue' / out_name),
                t_map.astype(np.uint8),
            )

            # 保存 nuclei mask (uint8, 0/101-105, raw IDs)
            cv2.imwrite(
                str(output_path / split / 'gt_nuclei' / out_name),
                n_map.astype(np.uint8),
            )

            # 保存 edit region mask (uint8, 0/255)
            cv2.imwrite(
                str(output_path / split / 'masks' / out_name),
                erasure_mask.astype(np.uint8) * 255,
            )

            stats[split] += 1
            mode_counts[mode_name] += 1

        if (file_idx + 1) % 500 == 0:
            logger.info(
                f"Processed {file_idx + 1}/{len(all_samples)} files | "
                f"train={stats['train']}, val={stats['val']}, skip={skipped}"
            )

    total = stats["train"] + stats["val"]

    # 保存元信息
    meta = {
        'dataset': config.name,
        'cancer_type': config.cancer_type,
        'cancer_type_index': cancer_type_index,
        'label_space': 'unified_fine_16',
        'input_format': fmt,
        'n_augmentations': n_augmentations,
        'seed': seed,
        'split_strategy': (
            'group_manifest_train_val_with_test_held_out'
            if split_assignments is not None
            else 'patch_random'
        ),
        'split_manifest': split_manifest,
        'split_manifest_sha256': split_manifest_sha256,
        'split_group_counts': split_group_counts,
        'held_out_test_sources': held_out_test_sources,
        'total_samples': total,
        'train_samples': stats['train'],
        'val_samples': stats['val'],
        'skipped': skipped,
        'mode_distribution': mode_counts,
        'complete_instance_erasure': 'runtime_derived_from_gt_nuclei',
        'stored_mask_role': 'conditioning_changed_region',
        'instance_definition': 'per_class_8_connected_components',
        'output_format': 'layered',
        'output_structure': {
            'gt_tissue': 'uint8 PNG, values 0-15 (unified fine tissue IDs)',
            'gt_nuclei': 'uint8 PNG, values 0/101-105 (CellViT raw nuclei IDs)',
            'masks': 'uint8 PNG, values 0/255 (edit region binary mask)',
        },
    }

    with open(str(output_path / 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Done! Total: {total} (train={stats['train']}, val={stats['val']})")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"\nMode distribution:")
    for mode, count in sorted(mode_counts.items()):
        pct = count / total * 100 if total > 0 else 0
        logger.info(f"  {mode:<15s}: {count:>6d} ({pct:5.1f}%)")
    logger.info(f"\nDataset: {config.name} (cancer_type_index={cancer_type_index})")
    logger.info(f"Output: {output_path}")
    logger.info(f"Meta: {output_path / 'meta.json'}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare ProbNet training dataset (Phase 4.4 multi-dataset)"
    )
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (BCSS, PANDA, GlaS, IGNITE, PUMA, ORCA)")
    parser.add_argument("--input-dir", required=True,
                        help="Input directory (layered: tissue_mask.png+nuclei_mask.png; "
                             "legacy: RGB combined mask PNGs)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for ProbNet training data")
    parser.add_argument("--format", choices=['auto', 'layered', 'legacy'], default='auto',
                        help="Input format (default: auto-detect)")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--n-augmentations", type=int, default=3,
                        help="Number of erasure augmentations per image (default: 3)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-manifest",
        default=None,
        help="Grouped Segmentator manifest whose train/val/test assignments are reused",
    )
    args = parser.parse_args()

    prepare_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        fmt=args.format,
        val_ratio=args.val_ratio,
        n_augmentations=args.n_augmentations,
        seed=args.seed,
        split_manifest=args.split_manifest,
    )

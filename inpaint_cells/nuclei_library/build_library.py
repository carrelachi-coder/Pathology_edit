#!/usr/bin/env python3
"""
阶段一：预处理 —— 从 GT mask 中提取核实例并建立统计库 (Phase 4.2 多数据集适配)

支持两种输入格式:
  1. 分层存储 (AD-1, 推荐): 每个 patch 有独立的 tissue_mask.png + nuclei_mask.png
  2. 旧 RGB 合并格式 (legacy): 单张 RGB PNG, tissue+nuclei 混合编码

输出:
    {output_dir}/
        nuclei_instances/          # 按统一 fine 组织类型分桶的核实例
            tissue_01_Tumor/       # 每个核: {id}.npz (mask, type, area)
            tissue_02_Stroma/
            ...
        statistics.json            # 每种组织类型的核密度、类型分布等
        summary.txt                # 可读的统计摘要

用法:
    # 分层存储格式 (推荐)
    python inpaint_cells/nuclei_library/build_library.py \\
        --dataset BCSS \\
        --gt-dir /data/BCSS_patches \\
        --output-dir /data/nuclei_library_BCSS

    # 旧 RGB 合并格式
    python inpaint_cells/nuclei_library/build_library.py \\
        --dataset BCSS \\
        --gt-dir /data/BCSS_dataset/conditioning \\
        --format legacy \\
        --output-dir /data/nuclei_library_BCSS
"""

import os
import sys
import json
import argparse
import glob
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm
from scipy import ndimage

# 项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataset_config import get_config
from dataset_config.unified_labels import (
    FINE_LABELS, NUM_FINE, UNIFIED_COLOR_MAP,
    CELL_CLASSES, CELL_IDS, CELL_COLOR_MAP, FULL_COLOR_MAP, NON_BIO_IDS,
)

# ============================================================
#  常量 (统一标签空间)
# ============================================================

NUCLEI_NAMES = {101: 'neoplastic', 102: 'inflammatory', 103: 'connective',
                104: 'dead', 105: 'epithelial'}
NUCLEI_CLASSES = CELL_IDS  # [101, 102, 103, 104, 105]

# RGB → class value 查找表 (用于 legacy 格式)
_rgb_to_val = {}
for _val, _rgb in FULL_COLOR_MAP.items():
    _key = _rgb[0] * 65536 + _rgb[1] * 256 + _rgb[2]
    _rgb_to_val[_key] = _val


def rgb_to_class_map(rgb_img):
    """RGB → class value map (H, W) — legacy 格式用"""
    encoded = (rgb_img[:, :, 0].astype(np.int64) * 65536
               + rgb_img[:, :, 1].astype(np.int64) * 256
               + rgb_img[:, :, 2].astype(np.int64))
    result = np.zeros(rgb_img.shape[:2], dtype=np.int64)
    for key, val in _rgb_to_val.items():
        result[encoded == key] = val
    return result


# ============================================================
#  核提取: 分层存储模式 (AD-1)
# ============================================================

def extract_nuclei_from_layered(tissue_map, nuclei_map, min_area=10, max_area=5000):
    """
    从分层存储的 tissue_mask + nuclei_mask 中提取所有核实例。

    Args:
        tissue_map: (H, W) int, tissue fine IDs (0-15)
        nuclei_map: (H, W) int, nuclei raw IDs (0/101-105)
        min_area, max_area: 核面积过滤

    Returns:
        list of dict: [{
            'type': 101-105,
            'tissue': 0-15,      # 核所在的组织类型 (直接从 tissue_map 读取)
            'mask': np.array,    # 核的二值 mask (bbox 大小)
            'area': int,
        }, ...]
    """
    instances = []
    H, W = tissue_map.shape

    for nuc_class in NUCLEI_CLASSES:
        binary = (nuclei_map == nuc_class).astype(np.uint8)
        if binary.sum() == 0:
            continue

        num_labels, labels = cv2.connectedComponents(binary, connectivity=8)

        for label_id in range(1, num_labels):
            component_mask = labels == label_id
            area = component_mask.sum()

            if area < min_area or area > max_area:
                continue

            ys, xs = np.where(component_mask)
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()

            # 排除边界核
            if y_min == 0 or x_min == 0 or y_max == H - 1 or x_max == W - 1:
                continue

            # 提取 bbox 区域的 mask
            local_mask = component_mask[y_min:y_max + 1, x_min:x_max + 1]

            # 核所在的组织类型: 直接从 tissue_map 读取核像素对应位置的众数
            # (AD-1: 不需要 EDT 反推)
            tissue_values = tissue_map[component_mask]
            counts = np.bincount(tissue_values.astype(np.int64), minlength=NUM_FINE)
            tissue_type = int(np.argmax(counts))

            instances.append({
                'type': nuc_class,
                'tissue': tissue_type,
                'mask': local_mask,
                'area': int(area),
            })

    return instances


# ============================================================
#  核提取: legacy RGB 合并格式 (backward compatible)
# ============================================================

def get_tissue_under_nucleus(class_map, nuc_mask):
    """
    获取核下面的组织类型（用核周围的组织像素推断）— legacy 格式用。
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(nuc_mask.astype(np.uint8), kernel)
    ring = (dilated > 0) & (~nuc_mask)

    if ring.sum() == 0:
        return -1

    ring_values = class_map[ring]
    ring_values = ring_values[(ring_values < 100) & (ring_values < NUM_FINE)]

    if len(ring_values) == 0:
        return -1

    counts = np.bincount(ring_values.astype(np.int64), minlength=NUM_FINE)
    return int(np.argmax(counts))


def extract_nuclei_from_classmap(class_map, min_area=10, max_area=5000):
    """
    从合并的 class_map 中提取所有核实例 — legacy 格式用。
    """
    instances = []
    H, W = class_map.shape

    for nuc_class in NUCLEI_CLASSES:
        binary = (class_map == nuc_class).astype(np.uint8)
        if binary.sum() == 0:
            continue

        num_labels, labels = cv2.connectedComponents(binary, connectivity=8)

        for label_id in range(1, num_labels):
            component_mask = labels == label_id
            area = component_mask.sum()

            if area < min_area or area > max_area:
                continue

            ys, xs = np.where(component_mask)
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()

            if y_min == 0 or x_min == 0 or y_max == H - 1 or x_max == W - 1:
                continue

            local_mask = component_mask[y_min:y_max + 1, x_min:x_max + 1]
            tissue_type = get_tissue_under_nucleus(class_map, component_mask)

            if tissue_type < 0:
                continue

            instances.append({
                'type': nuc_class,
                'tissue': tissue_type,
                'mask': local_mask,
                'area': int(area),
            })

    return instances


# ============================================================
#  主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Build nuclei instance library (Phase 4.2 multi-dataset)')
    parser.add_argument('--dataset', required=True,
                        help='Dataset name (BCSS, PANDA, GlaS, IGNITE, PUMA, ORCA)')
    parser.add_argument('--gt-dir', required=True,
                        help='GT directory (layered: contains tissue_mask.png+nuclei_mask.png; '
                             'legacy: contains RGB mask PNGs)')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for nuclei library')
    parser.add_argument('--format', choices=['auto', 'layered', 'legacy'], default='auto',
                        help='Input format (default: auto-detect)')
    parser.add_argument('--min-area', type=int, default=10,
                        help='Min nucleus area in pixels')
    parser.add_argument('--max-area', type=int, default=5000,
                        help='Max nucleus area in pixels')
    parser.add_argument('--max-instances-per-bucket', type=int, default=10000,
                        help='Max instances to store per tissue-type bucket')
    args = parser.parse_args()

    # 加载数据集配置
    config = get_config(args.dataset)
    skip_tissues = config.skip_tissues
    print(f"Dataset: {config.name} ({config.cancer_type})")
    print(f"  cancer_type_index: {config.cancer_type_index}")
    print(f"  has_cell_annotations: {config.has_cell_annotations}")
    print(f"  skip_tissues: {skip_tissues}")

    # 自动检测输入格式
    fmt = args.format
    if fmt == 'auto':
        # 检查是否有分层存储标记
        has_tissue = (
            len(glob.glob(os.path.join(args.gt_dir, '*', 'tissue_mask.png'))) > 0
            or len(glob.glob(os.path.join(args.gt_dir, '**', 'tissue_mask.png'), recursive=True)) > 0
        )
        has_nuclei = (
            len(glob.glob(os.path.join(args.gt_dir, '*', 'nuclei_mask.png'))) > 0
            or len(glob.glob(os.path.join(args.gt_dir, '**', 'nuclei_mask.png'), recursive=True)) > 0
        )
        fmt = 'layered' if (has_tissue and has_nuclei) else 'legacy'
        print(f"  Auto-detected format: {fmt}")

    os.makedirs(args.output_dir, exist_ok=True)
    instances_dir = os.path.join(args.output_dir, 'nuclei_instances')
    os.makedirs(instances_dir, exist_ok=True)

    # 为每种统一 fine 组织类型创建目录 (排除 skip_tissues)
    for tissue_id, tissue_name in FINE_LABELS.items():
        if tissue_id in skip_tissues:
            continue
        bucket_dir = os.path.join(instances_dir, f'tissue_{tissue_id:02d}_{tissue_name}')
        os.makedirs(bucket_dir, exist_ok=True)

    # 统计数据
    tissue_stats = defaultdict(lambda: {
        'total_area': 0,
        'nuclei_counts': defaultdict(int),
        'nuclei_areas': defaultdict(list),
    })
    bucket_counts = defaultdict(int)

    total_instances = 0

    if fmt == 'layered':
        total_instances = _process_layered(
            args, config, skip_tissues, instances_dir,
            tissue_stats, bucket_counts)
    else:
        total_instances = _process_legacy(
            args, config, skip_tissues, instances_dir,
            tissue_stats, bucket_counts)

    print(f"\nTotal instances extracted: {total_instances}")

    # 计算并保存统计数据
    _save_statistics(args, tissue_stats, bucket_counts, total_instances,
                     skip_tissues, config)

    print(f"\nLibrary saved to {args.output_dir}")


def _process_layered(args, config, skip_tissues, instances_dir,
                     tissue_stats, bucket_counts):
    """处理分层存储格式 (AD-1)"""
    # 查找所有 patch 目录 (含 tissue_mask.png + nuclei_mask.png)
    tissue_files = sorted(glob.glob(os.path.join(args.gt_dir, '**', 'tissue_mask.png'),
                                    recursive=True))
    if not tissue_files:
        # 也尝试平面目录: gt_tissue/ + gt_nuclei/
        gt_tissue_dir = os.path.join(args.gt_dir, 'gt_tissue')
        gt_nuclei_dir = os.path.join(args.gt_dir, 'gt_nuclei')
        if os.path.isdir(gt_tissue_dir) and os.path.isdir(gt_nuclei_dir):
            tissue_files = sorted(glob.glob(os.path.join(gt_tissue_dir, '*.png')))
        else:
            print(f"No layered samples found in {args.gt_dir}")
            return 0

    print(f"Processing {len(tissue_files)} layered patches...")
    total_instances = 0

    for tissue_path in tqdm(tissue_files, desc='Extracting nuclei (layered)'):
        # 定位 nuclei 文件
        parent = os.path.dirname(tissue_path)
        basename = os.path.basename(tissue_path)

        if basename == 'tissue_mask.png':
            nuclei_path = os.path.join(parent, 'nuclei_mask.png')
        else:
            # 平面目录模式: gt_tissue/xxx.png → gt_nuclei/xxx.png
            gt_nuclei_dir = os.path.join(os.path.dirname(parent), 'gt_nuclei')
            nuclei_path = os.path.join(gt_nuclei_dir, basename)

        if not os.path.exists(nuclei_path):
            continue

        # 读取分层 mask (AD-1: 直接读取, 无需 RGB 转换)
        tissue_map = cv2.imread(tissue_path, cv2.IMREAD_GRAYSCALE).astype(np.int64)
        nuclei_map = cv2.imread(nuclei_path, cv2.IMREAD_GRAYSCALE).astype(np.int64)

        # 统计组织面积
        for tissue_id in range(NUM_FINE):
            if tissue_id in skip_tissues:
                continue
            area = (tissue_map == tissue_id).sum()
            if area > 0:
                tissue_stats[tissue_id]['total_area'] += int(area)

        # 提取核实例 (AD-1: 直接从 tissue_map 读取组织类型)
        instances = extract_nuclei_from_layered(
            tissue_map, nuclei_map, args.min_area, args.max_area)

        for inst in instances:
            tissue_id = inst['tissue']
            if tissue_id in skip_tissues:
                continue
            nuc_type = inst['type']

            tissue_stats[tissue_id]['nuclei_counts'][nuc_type] += 1
            tissue_stats[tissue_id]['nuclei_areas'][nuc_type].append(inst['area'])

            if bucket_counts[tissue_id] < args.max_instances_per_bucket:
                tissue_name = FINE_LABELS.get(tissue_id, f'tissue_{tissue_id}')
                bucket_dir = os.path.join(instances_dir,
                                          f'tissue_{tissue_id:02d}_{tissue_name}')
                os.makedirs(bucket_dir, exist_ok=True)
                inst_id = bucket_counts[tissue_id]

                np.savez_compressed(
                    os.path.join(bucket_dir, f'{inst_id:06d}.npz'),
                    mask=inst['mask'].astype(np.bool_),
                    type=np.array(nuc_type, dtype=np.int32),
                    area=np.array(inst['area'], dtype=np.int32),
                )
                bucket_counts[tissue_id] += 1

            total_instances += 1

    return total_instances


def _process_legacy(args, config, skip_tissues, instances_dir,
                    tissue_stats, bucket_counts):
    """处理旧 RGB 合并格式 (backward compatible)"""
    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, '*.png')))
    print(f"Processing {len(gt_files)} legacy GT files...")
    total_instances = 0

    for gt_path in tqdm(gt_files, desc='Extracting nuclei (legacy)'):
        rgb = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)
        class_map = rgb_to_class_map(rgb)

        # 统计组织面积
        for tissue_id in range(NUM_FINE):
            if tissue_id in skip_tissues:
                continue
            area = (class_map == tissue_id).sum()
            if area > 0:
                tissue_stats[tissue_id]['total_area'] += int(area)

        # 提取核实例
        instances = extract_nuclei_from_classmap(
            class_map, args.min_area, args.max_area)

        for inst in instances:
            tissue_id = inst['tissue']
            if tissue_id in skip_tissues:
                continue
            nuc_type = inst['type']

            tissue_stats[tissue_id]['nuclei_counts'][nuc_type] += 1
            tissue_stats[tissue_id]['nuclei_areas'][nuc_type].append(inst['area'])

            if bucket_counts[tissue_id] < args.max_instances_per_bucket:
                tissue_name = FINE_LABELS.get(tissue_id, f'tissue_{tissue_id}')
                bucket_dir = os.path.join(instances_dir,
                                          f'tissue_{tissue_id:02d}_{tissue_name}')
                os.makedirs(bucket_dir, exist_ok=True)
                inst_id = bucket_counts[tissue_id]

                np.savez_compressed(
                    os.path.join(bucket_dir, f'{inst_id:06d}.npz'),
                    mask=inst['mask'].astype(np.bool_),
                    type=np.array(nuc_type, dtype=np.int32),
                    area=np.array(inst['area'], dtype=np.int32),
                )
                bucket_counts[tissue_id] += 1

            total_instances += 1

    return total_instances


def _save_statistics(args, tissue_stats, bucket_counts, total_instances,
                     skip_tissues, config):
    """保存统计数据和摘要"""
    stats_output = {}
    summary_lines = []
    summary_lines.append(f"{'=' * 80}")
    summary_lines.append(f"Nuclei Library Statistics — {config.name} ({config.cancer_type})")
    summary_lines.append(f"{'=' * 80}")
    summary_lines.append(f"Total nuclei instances: {total_instances}")
    summary_lines.append(f"Cancer type index: {config.cancer_type_index}")
    summary_lines.append(f"Label space: unified {NUM_FINE}-class fine labels")
    summary_lines.append("")

    for tissue_id in range(NUM_FINE):
        if tissue_id in skip_tissues:
            continue

        ts = tissue_stats[tissue_id]
        total_area = ts['total_area']

        if total_area == 0:
            continue

        tissue_name = FINE_LABELS[tissue_id]

        total_nuclei = sum(ts['nuclei_counts'].values())
        density = total_nuclei / (total_area / 10000.0) if total_area > 0 else 0

        type_dist = {}
        for nuc_type in NUCLEI_CLASSES:
            count = ts['nuclei_counts'].get(nuc_type, 0)
            frac = count / total_nuclei if total_nuclei > 0 else 0
            areas = ts['nuclei_areas'].get(nuc_type, [])
            mean_area = np.mean(areas) if areas else 0
            std_area = np.std(areas) if areas else 0

            type_dist[str(nuc_type)] = {
                'count': int(count),
                'fraction': round(frac, 4),
                'mean_area': round(float(mean_area), 1),
                'std_area': round(float(std_area), 1),
            }

        stats_output[str(tissue_id)] = {
            'name': tissue_name,
            'total_area_pixels': int(total_area),
            'total_nuclei': int(total_nuclei),
            'density_per_10k_px': round(float(density), 2),
            'nuclei_types': type_dist,
            'stored_instances': int(bucket_counts.get(tissue_id, 0)),
        }

        summary_lines.append(f"Tissue {tissue_id:2d} ({tissue_name}):")
        summary_lines.append(f"  Total area: {total_area:,} px")
        summary_lines.append(f"  Total nuclei: {total_nuclei:,}")
        summary_lines.append(f"  Density: {density:.2f} per 10k px^2")
        summary_lines.append(f"  Stored instances: {bucket_counts.get(tissue_id, 0)}")
        for nuc_type in NUCLEI_CLASSES:
            td = type_dist[str(nuc_type)]
            if td['count'] > 0:
                summary_lines.append(
                    f"    {NUCLEI_NAMES[nuc_type]:15s}: {td['count']:6d} ({td['fraction'] * 100:5.1f}%) "
                    f"area={td['mean_area']:.0f}+/-{td['std_area']:.0f}")
        summary_lines.append("")

    # 保存 metadata
    meta = {
        'dataset': config.name,
        'cancer_type': config.cancer_type,
        'cancer_type_index': config.cancer_type_index,
        'label_space': 'unified_fine_16',
        'statistics': stats_output,
    }

    with open(os.path.join(args.output_dir, 'statistics.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    summary_text = '\n'.join(summary_lines)
    with open(os.path.join(args.output_dir, 'summary.txt'), 'w') as f:
        f.write(summary_text)

    print(summary_text)


if __name__ == '__main__':
    main()

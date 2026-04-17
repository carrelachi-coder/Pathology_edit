#!/usr/bin/env python3
"""
阶段一：预处理 —— 从 GT mask 中提取核实例并建立统计库 (Phase 4.2/4.3 多数据集适配)

支持三种输入格式:
  1. 分层存储 (AD-1, 推荐): 每个 patch 有独立的 tissue_mask.png + nuclei_mask.png
  2. 旧 RGB 合并格式 (legacy): 单张 RGB PNG, tissue+nuclei 混合编码
  3. CellViT JSON (Phase 4.3): tissue_mask.png + CellViT 推理输出的 JSON/GeoJSON

核来源 (Phase 4.3):
  - PUMA：原始含 10 类细胞核 GeoJSON，映射为 CellViT 5 类 → 直接建库
  - BCSS：原始含 nuclei 标注 → 直接建库
  - IGNITE/PANDA/GlaS/ORCA：需先用 CellViT 推理获取细胞核 → 建库
    CellViT 推理后将结果存为 nuclei_mask.png (layered) 或 JSON 格式

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

    # CellViT JSON 格式 (Phase 4.3: 适用于无原始核标注的数据集)
    python inpaint_cells/nuclei_library/build_library.py \\
        --dataset PANDA \\
        --gt-dir /data/PANDA_patches \\
        --format cellvit-json \\
        --cellvit-dir /data/PANDA_cellvit_output \\
        --output-dir /data/nuclei_library_PANDA

    # PUMA GeoJSON (Phase 4.3: 10类核 → CellViT 5类)
    python inpaint_cells/nuclei_library/build_library.py \\
        --dataset PUMA \\
        --gt-dir /data/PUMA_patches \\
        --format geojson \\
        --geojson-dir /data/PUMA_nuclei_geojson \\
        --output-dir /data/nuclei_library_PUMA
"""

import os
import sys
import json
import argparse
import glob
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from scipy import ndimage

# 项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataset_config import get_config
from dataset_config.unified_labels import (
    FINE_LABELS, NUM_FINE, CELL_IDS, FULL_COLOR_MAP,
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
#  核提取: CellViT JSON 格式 (Phase 4.3)
# ============================================================

# CellViT 推理输出 JSON 中的类型 ID → CellViT 5 类映射
CELLVIT_TYPE_MAP = {
    1: 101,  # neoplastic
    2: 102,  # inflammatory
    3: 103,  # connective
    4: 104,  # dead
    5: 105,  # epithelial
}


def extract_nuclei_from_cellvit_json(tissue_map, json_path, min_area=10, max_area=5000):
    """
    从 CellViT 推理输出的 JSON 中提取核实例。

    CellViT JSON 格式 (每个 patch 一个 JSON):
        {
            "nuc": {
                "0": {"bbox": [y1,x1,y2,x2], "centroid": [y,x], "contour": [[x,y],...], "type": 1},
                "1": {...},
                ...
            },
            "mag": 40,
            ...
        }

    Args:
        tissue_map: (H, W) int, tissue fine IDs (0-15)
        json_path: path to CellViT output JSON
        min_area, max_area: 核面积过滤

    Returns:
        list of dict (same format as extract_nuclei_from_layered)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    nuc_dict = data.get('nuc', data.get('nuclei', {}))
    if not nuc_dict:
        return []

    instances = []
    H, W = tissue_map.shape

    for nuc_id, nuc_info in nuc_dict.items():
        nuc_type_cv = nuc_info.get('type', 0)
        if nuc_type_cv not in CELLVIT_TYPE_MAP:
            continue
        nuc_class = CELLVIT_TYPE_MAP[nuc_type_cv]

        contour = nuc_info.get('contour', None)
        if contour is None:
            continue

        contour_np = np.array(contour, dtype=np.int32)  # [[x, y], ...]
        if len(contour_np) < 3:
            continue

        # 绘制核 mask
        nuc_mask_full = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(nuc_mask_full, [contour_np], 1)

        area = nuc_mask_full.sum()
        if area < min_area or area > max_area:
            continue

        ys, xs = np.where(nuc_mask_full > 0)
        if len(ys) == 0:
            continue

        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        # 排除边界核
        if y_min == 0 or x_min == 0 or y_max >= H - 1 or x_max >= W - 1:
            continue

        local_mask = nuc_mask_full[y_min:y_max + 1, x_min:x_max + 1].astype(bool)

        # 核所在组织类型: 从 tissue_map 取众数
        tissue_values = tissue_map[nuc_mask_full > 0]
        counts = np.bincount(tissue_values.astype(np.int64), minlength=NUM_FINE)
        tissue_type = int(np.argmax(counts))

        instances.append({
            'type': nuc_class,
            'tissue': tissue_type,
            'mask': local_mask,
            'area': int(area),
        })

    return instances


# PUMA 10 类核 → CellViT 5 类映射
PUMA_NUC_TYPE_MAP = {
    # PUMA 原始类别 → CellViT 类别
    'tumor': 101,
    'lymphocyte': 102,
    'plasma_cell': 102,       # → inflammatory
    'macrophage': 102,        # → inflammatory
    'neutrophil': 102,        # → inflammatory
    'fibroblast': 103,        # → connective
    'endothelial': 103,       # → connective
    'apoptotic': 104,         # → dead
    'epithelial': 105,
    'melanocyte': 101,        # → neoplastic (melanoma context)
}

# PUMA 数值类型 ID 映射 (若 GeoJSON 使用数字 type)
PUMA_NUC_TYPE_ID_MAP = {
    0: 101,   # tumor / neoplastic
    1: 102,   # lymphocyte / inflammatory
    2: 102,   # plasma_cell → inflammatory
    3: 102,   # macrophage → inflammatory
    4: 102,   # neutrophil → inflammatory
    5: 103,   # fibroblast → connective
    6: 103,   # endothelial → connective
    7: 104,   # apoptotic → dead
    8: 105,   # epithelial
    9: 101,   # melanocyte → neoplastic
}


def extract_nuclei_from_geojson(tissue_map, geojson_path, min_area=10, max_area=5000):
    """
    从 GeoJSON 核标注中提取实例 (PUMA 等有 GeoJSON 核标注的数据集).

    GeoJSON 格式:
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [[[x, y], ...]]},
                    "properties": {"classification": {"name": "tumor"}}
                }, ...
            ]
        }

    Args:
        tissue_map: (H, W) int, tissue fine IDs (0-15)
        geojson_path: path to GeoJSON file
        min_area, max_area: filter thresholds

    Returns:
        list of dict (same format as extract_nuclei_from_layered)
    """
    with open(geojson_path, 'r') as f:
        geo = json.load(f)

    features = geo.get('features', [])
    if not features:
        return []

    instances = []
    H, W = tissue_map.shape

    for feat in features:
        geom = feat.get('geometry', {})
        props = feat.get('properties', {})

        if geom.get('type') != 'Polygon':
            continue

        coords = geom.get('coordinates', [[]])[0]  # 外环
        if len(coords) < 3:
            continue

        # 确定核类型
        classification = props.get('classification', {})
        type_name = classification.get('name', '').lower().strip()
        type_id = classification.get('type_id', None)

        nuc_class = None
        if type_name and type_name in PUMA_NUC_TYPE_MAP:
            nuc_class = PUMA_NUC_TYPE_MAP[type_name]
        elif type_id is not None and type_id in PUMA_NUC_TYPE_ID_MAP:
            nuc_class = PUMA_NUC_TYPE_ID_MAP[type_id]

        if nuc_class is None:
            continue

        # GeoJSON coordinates: [[x, y], ...] → numpy int32
        contour_np = np.array(coords, dtype=np.float64)
        contour_int = np.round(contour_np).astype(np.int32)

        nuc_mask_full = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(nuc_mask_full, [contour_int], 1)

        area = nuc_mask_full.sum()
        if area < min_area or area > max_area:
            continue

        ys, xs = np.where(nuc_mask_full > 0)
        if len(ys) == 0:
            continue

        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        if y_min == 0 or x_min == 0 or y_max >= H - 1 or x_max >= W - 1:
            continue

        local_mask = nuc_mask_full[y_min:y_max + 1, x_min:x_max + 1].astype(bool)

        tissue_values = tissue_map[nuc_mask_full > 0]
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
#  主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Build nuclei instance library (Phase 4.2/4.3 multi-dataset)')
    parser.add_argument('--dataset', required=True,
                        help='Dataset name (BCSS, PANDA, GlaS, IGNITE, PUMA, ORCA)')
    parser.add_argument('--gt-dir', required=True,
                        help='GT directory (layered: contains tissue_mask.png+nuclei_mask.png; '
                             'legacy: contains RGB mask PNGs)')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for nuclei library')
    parser.add_argument('--format', choices=['auto', 'layered', 'legacy',
                                             'cellvit-json', 'geojson'], default='auto',
                        help='Input format (default: auto-detect)')
    parser.add_argument('--cellvit-dir', default=None,
                        help='[cellvit-json] Directory containing CellViT inference JSON files. '
                             'Each JSON corresponds to a tissue_mask.png in gt-dir.')
    parser.add_argument('--geojson-dir', default=None,
                        help='[geojson] Directory containing nuclei GeoJSON files (PUMA etc.).')
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
        has_flat_tissue = os.path.isdir(os.path.join(args.gt_dir, 'tissue_masks'))
        has_flat_nuclei = os.path.isdir(os.path.join(args.gt_dir, 'nuclei_masks'))
        has_tissue = (
            has_flat_tissue
            or len(glob.glob(os.path.join(args.gt_dir, '*', 'tissue_mask.png'))) > 0
            or len(glob.glob(os.path.join(args.gt_dir, '**', 'tissue_mask.png'), recursive=True)) > 0
            or os.path.isdir(os.path.join(args.gt_dir, 'gt_tissue'))
        )
        has_nuclei = (
            has_flat_nuclei
            or len(glob.glob(os.path.join(args.gt_dir, '*', 'nuclei_mask.png'))) > 0
            or len(glob.glob(os.path.join(args.gt_dir, '**', 'nuclei_mask.png'), recursive=True)) > 0
            or os.path.isdir(os.path.join(args.gt_dir, 'gt_nuclei'))
        )
        if has_tissue and has_nuclei:
            fmt = 'layered'
        elif has_tissue and args.cellvit_dir:
            fmt = 'cellvit-json'
        elif has_tissue and args.geojson_dir:
            fmt = 'geojson'
        else:
            fmt = 'legacy'
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
    elif fmt == 'cellvit-json':
        total_instances = _process_cellvit_json(
            args, config, skip_tissues, instances_dir,
            tissue_stats, bucket_counts)
    elif fmt == 'geojson':
        total_instances = _process_geojson(
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
    flat_tissue_dir = os.path.join(args.gt_dir, 'tissue_masks')
    flat_nuclei_dir = os.path.join(args.gt_dir, 'nuclei_masks')
    layered_mode = 'sample_dirs'
    if os.path.isdir(flat_tissue_dir) and os.path.isdir(flat_nuclei_dir):
        tissue_files = sorted(glob.glob(os.path.join(flat_tissue_dir, '*.png')))
        layered_mode = 'tissue_masks'
    else:
        tissue_files = sorted(glob.glob(os.path.join(args.gt_dir, '**', 'tissue_mask.png'),
                                        recursive=True))
    if not tissue_files:
        # 也尝试平面目录: gt_tissue/ + gt_nuclei/
        gt_tissue_dir = os.path.join(args.gt_dir, 'gt_tissue')
        gt_nuclei_dir = os.path.join(args.gt_dir, 'gt_nuclei')
        if os.path.isdir(gt_tissue_dir) and os.path.isdir(gt_nuclei_dir):
            tissue_files = sorted(glob.glob(os.path.join(gt_tissue_dir, '*.png')))
            layered_mode = 'gt_tissue'
        else:
            print(f"No layered samples found in {args.gt_dir}")
            return 0

    print(f"Processing {len(tissue_files)} layered patches...")
    total_instances = 0

    for tissue_path in tqdm(tissue_files, desc='Extracting nuclei (layered)'):
        # 定位 nuclei 文件
        parent = os.path.dirname(tissue_path)
        basename = os.path.basename(tissue_path)

        if layered_mode == 'tissue_masks':
            nuclei_path = os.path.join(flat_nuclei_dir, basename)
        elif basename == 'tissue_mask.png':
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

        total_instances += _save_instances(
            instances, skip_tissues, instances_dir,
            tissue_stats, bucket_counts, args.max_instances_per_bucket)

    return total_instances


def _process_cellvit_json(args, config, skip_tissues, instances_dir,
                          tissue_stats, bucket_counts):
    """处理 CellViT JSON 格式 (Phase 4.3: 无原始核标注的数据集)"""
    if not args.cellvit_dir:
        print("Error: --cellvit-dir is required for cellvit-json format")
        return 0

    # 查找 tissue_mask 文件
    tissue_files = sorted(glob.glob(os.path.join(args.gt_dir, '**', 'tissue_mask.png'),
                                    recursive=True))
    if not tissue_files:
        gt_tissue_dir = os.path.join(args.gt_dir, 'gt_tissue')
        if os.path.isdir(gt_tissue_dir):
            tissue_files = sorted(glob.glob(os.path.join(gt_tissue_dir, '*.png')))

    if not tissue_files:
        print(f"No tissue mask files found in {args.gt_dir}")
        return 0

    print(f"Processing {len(tissue_files)} patches with CellViT JSON nuclei...")
    total_instances = 0
    cellvit_dir = Path(args.cellvit_dir)

    for tissue_path in tqdm(tissue_files, desc='Extracting nuclei (cellvit-json)'):
        tissue_map = cv2.imread(tissue_path, cv2.IMREAD_GRAYSCALE).astype(np.int64)

        # 查找对应的 CellViT JSON
        # 匹配策略: tissue_mask.png 所在目录名或文件名 → 同名 .json
        parent = os.path.dirname(tissue_path)
        basename = os.path.basename(tissue_path)

        if basename == 'tissue_mask.png':
            sample_name = os.path.basename(parent)
        else:
            sample_name = os.path.splitext(basename)[0]

        json_path = None
        for ext in ['.json', '.geojson']:
            candidate = cellvit_dir / f'{sample_name}{ext}'
            if candidate.exists():
                json_path = str(candidate)
                break

        if json_path is None:
            continue

        # 统计组织面积
        for tissue_id in range(NUM_FINE):
            if tissue_id in skip_tissues:
                continue
            area = (tissue_map == tissue_id).sum()
            if area > 0:
                tissue_stats[tissue_id]['total_area'] += int(area)

        # 从 CellViT JSON 提取核实例
        instances = extract_nuclei_from_cellvit_json(
            tissue_map, json_path, args.min_area, args.max_area)

        total_instances += _save_instances(
            instances, skip_tissues, instances_dir,
            tissue_stats, bucket_counts, args.max_instances_per_bucket)

    return total_instances


def _process_geojson(args, config, skip_tissues, instances_dir,
                     tissue_stats, bucket_counts):
    """处理 GeoJSON 核标注格式 (Phase 4.3: PUMA 等有 GeoJSON 标注的数据集)"""
    if not args.geojson_dir:
        print("Error: --geojson-dir is required for geojson format")
        return 0

    # 查找 tissue_mask 文件
    tissue_files = sorted(glob.glob(os.path.join(args.gt_dir, '**', 'tissue_mask.png'),
                                    recursive=True))
    if not tissue_files:
        gt_tissue_dir = os.path.join(args.gt_dir, 'gt_tissue')
        if os.path.isdir(gt_tissue_dir):
            tissue_files = sorted(glob.glob(os.path.join(gt_tissue_dir, '*.png')))

    if not tissue_files:
        print(f"No tissue mask files found in {args.gt_dir}")
        return 0

    print(f"Processing {len(tissue_files)} patches with GeoJSON nuclei...")
    total_instances = 0
    geojson_dir = Path(args.geojson_dir)

    for tissue_path in tqdm(tissue_files, desc='Extracting nuclei (geojson)'):
        tissue_map = cv2.imread(tissue_path, cv2.IMREAD_GRAYSCALE).astype(np.int64)

        parent = os.path.dirname(tissue_path)
        basename = os.path.basename(tissue_path)

        if basename == 'tissue_mask.png':
            sample_name = os.path.basename(parent)
        else:
            sample_name = os.path.splitext(basename)[0]

        geojson_path = None
        for ext in ['.geojson', '.json']:
            candidate = geojson_dir / f'{sample_name}{ext}'
            if candidate.exists():
                geojson_path = str(candidate)
                break

        if geojson_path is None:
            continue

        # 统计组织面积
        for tissue_id in range(NUM_FINE):
            if tissue_id in skip_tissues:
                continue
            area = (tissue_map == tissue_id).sum()
            if area > 0:
                tissue_stats[tissue_id]['total_area'] += int(area)

        instances = extract_nuclei_from_geojson(
            tissue_map, geojson_path, args.min_area, args.max_area)

        total_instances += _save_instances(
            instances, skip_tissues, instances_dir,
            tissue_stats, bucket_counts, args.max_instances_per_bucket)

    return total_instances


def _save_instances(instances, skip_tissues, instances_dir,
                    tissue_stats, bucket_counts, max_per_bucket):
    """将提取的核实例保存到桶中 (公共函数, 供所有格式使用)"""
    saved = 0
    for inst in instances:
        tissue_id = inst['tissue']
        if tissue_id in skip_tissues:
            continue
        nuc_type = inst['type']

        tissue_stats[tissue_id]['nuclei_counts'][nuc_type] += 1
        tissue_stats[tissue_id]['nuclei_areas'][nuc_type].append(inst['area'])

        if bucket_counts[tissue_id] < max_per_bucket:
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

        saved += 1
    return saved


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

        total_instances += _save_instances(
            instances, skip_tissues, instances_dir,
            tissue_stats, bucket_counts, args.max_instances_per_bucket)

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

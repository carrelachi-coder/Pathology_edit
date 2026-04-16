#!/usr/bin/env python3
"""
阶段一：预处理 —— 从 26K GT mask 中提取核实例并建立统计库

输出:
    {output_dir}/
        nuclei_instances/          # 按组织类型分桶的核实例
            tissue_01_tumor/       # 每个核: {id}.npz (mask, type, area, bbox_size)
            tissue_02_stroma/
            ...
        statistics.json            # 每种组织类型的核密度、类型分布等
        summary.txt                # 可读的统计摘要

用法:
    python /home/lyw/wqx-DL/flow-edit/FlowEdit-main/inpaint_cells/DDPM+Cell_inpaint/build_nuclei_library.py\
        --gt-dir /home/lyw/wqx-DL/flow-edit/FlowEdit-main/inpaint_cells/lama_dataset/ground_truth \
        --output-dir /data/huggingface/pathology_edit/nuclei_library
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


# ============================================================
#  颜色/类别定义
# ============================================================

COLOR_MAP = {
    0: [30,30,30], 1: [180,60,60], 2: [60,150,60], 3: [140,60,180],
    4: [60,60,180], 5: [180,180,80], 6: [160,40,40], 7: [40,40,40],
    8: [80,150,150], 9: [200,170,100], 10: [180,120,150], 11: [120,120,190],
    12: [100,190,190], 13: [200,140,60], 14: [140,200,100], 15: [140,140,140],
    16: [200,200,130], 17: [150,80,60], 18: [60,140,100], 19: [190,40,40],
    20: [80,60,150], 21: [170,170,170],
    101: [255,0,0], 102: [0,255,0], 103: [0,80,255], 104: [255,255,0], 105: [255,0,255],
}

TISSUE_NAMES = {
    0: 'outside_roi', 1: 'tumor', 2: 'stroma', 3: 'lymphocytic_infiltrate',
    4: 'necrosis_or_debris', 5: 'glandular_secretions', 6: 'blood', 7: 'exclude',
    8: 'metaplasia_NOS', 9: 'fat', 10: 'plasma_cells', 11: 'other_immune_infiltrate',
    12: 'mucoid_material', 13: 'normal_acinus_or_duct', 14: 'lymphatics',
    15: 'undetermined', 16: 'nerve', 17: 'skin_adnexa', 18: 'blood_vessel',
    19: 'angioinvasion', 20: 'dcis', 21: 'other',
}

NUCLEI_NAMES = {101: 'neoplastic', 102: 'inflammatory', 103: 'connective', 
                104: 'dead', 105: 'epithelial'}
NUCLEI_CLASSES = [101, 102, 103, 104, 105]

# RGB → class value 查找表
_rgb_to_val = {}
for val, rgb in COLOR_MAP.items():
    key = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
    _rgb_to_val[key] = val


def rgb_to_class_map(rgb_img):
    """RGB → class value map (H, W)"""
    encoded = rgb_img[:,:,0].astype(np.int64)*65536 + rgb_img[:,:,1].astype(np.int64)*256 + rgb_img[:,:,2].astype(np.int64)
    result = np.zeros(rgb_img.shape[:2], dtype=np.int64)
    for key, val in _rgb_to_val.items():
        result[encoded == key] = val
    return result


def get_tissue_under_nucleus(class_map, nuc_mask):
    """
    获取核下面的组织类型（用核周围的组织像素推断）
    """
    # 膨胀核 mask 3 像素，取环形区域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(nuc_mask.astype(np.uint8), kernel)
    ring = (dilated > 0) & (~nuc_mask)
    
    if ring.sum() == 0:
        return -1  # 无法判断
    
    # 取环形区域中最常见的组织类型（排除核类型 >= 100）
    ring_values = class_map[ring]
    ring_values = ring_values[ring_values < 100]
    
    if len(ring_values) == 0:
        return -1
    
    # 众数
    counts = np.bincount(ring_values.astype(np.int64), minlength=22)
    return int(np.argmax(counts))


def extract_nuclei_from_patch(class_map, min_area=10, max_area=5000):
    """
    从一个 patch 的 class_map 中提取所有核实例
    
    Returns:
        list of dict: [{
            'type': 101-105,
            'tissue': 0-21,      # 核所在的组织类型
            'mask': np.array,    # 核的二值 mask (bbox 大小)
            'area': int,
            'bbox': (y, x, h, w),
            'centroid': (cy, cx),
        }, ...]
    """
    instances = []
    
    for nuc_class in NUCLEI_CLASSES:
        # 该类型核的二值 mask
        binary = (class_map == nuc_class).astype(np.uint8)
        
        if binary.sum() == 0:
            continue
        
        # Connected components
        num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
        
        for label_id in range(1, num_labels):
            component_mask = labels == label_id
            area = component_mask.sum()
            
            if area < min_area or area > max_area:
                continue
            
            # Bounding box
            ys, xs = np.where(component_mask)
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()

            H, W = class_map.shape
            if y_min == 0 or x_min == 0 or y_max == H - 1 or x_max == W - 1:
                continue
            h = y_max - y_min + 1
            w = x_max - x_min + 1
            
            # 提取 bbox 区域的 mask
            local_mask = component_mask[y_min:y_max+1, x_min:x_max+1]
            
            # 质心
            cy = ys.mean()
            cx = xs.mean()
            
            # 核所在的组织类型
            tissue_type = get_tissue_under_nucleus(class_map, component_mask)
            
            if tissue_type < 0:
                continue
            
            instances.append({
                'type': nuc_class,
                'tissue': tissue_type,
                'mask': local_mask,
                'area': int(area),
                'bbox': (int(y_min), int(x_min), int(h), int(w)),
                'centroid': (float(cy), float(cx)),
                'bbox_h': int(h),
                'bbox_w': int(w),
            })
    
    return instances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-dir', required=True, help='Ground truth directory with RGB mask PNGs')
    parser.add_argument('--output-dir', required=True, help='Output directory for nuclei library')
    parser.add_argument('--min-area', type=int, default=10, help='Min nucleus area in pixels')
    parser.add_argument('--max-area', type=int, default=5000, help='Max nucleus area in pixels')
    parser.add_argument('--max-instances-per-bucket', type=int, default=10000,
                        help='Max instances to store per tissue-type bucket')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    instances_dir = os.path.join(args.output_dir, 'nuclei_instances')
    os.makedirs(instances_dir, exist_ok=True)
    
    # 为每种组织类型创建目录
    for tissue_id, tissue_name in TISSUE_NAMES.items():
        bucket_dir = os.path.join(instances_dir, f'tissue_{tissue_id:02d}_{tissue_name}')
        os.makedirs(bucket_dir, exist_ok=True)
    
    # 统计数据
    # tissue_stats[tissue_id] = {
    #   'total_area': int,           # 该组织类型的总面积（像素数）
    #   'nuclei_counts': {101: n, 102: n, ...},  # 各类核的总个数
    #   'nuclei_areas': {101: [areas], ...},      # 各类核的面积列表
    # }
    tissue_stats = defaultdict(lambda: {
        'total_area': 0,
        'nuclei_counts': defaultdict(int),
        'nuclei_areas': defaultdict(list),
    })
    
    # 核实例计数
    bucket_counts = defaultdict(int)
    
    # 遍历所有 GT 文件
    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, '*.png')))
    print(f"Processing {len(gt_files)} GT files...")
    
    total_instances = 0
    
    for gt_path in tqdm(gt_files, desc='Extracting nuclei'):
        # 读取
        rgb = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)
        class_map = rgb_to_class_map(rgb)
        
        # 统计每种组织类型的面积
        for tissue_id in range(22):
            area = (class_map == tissue_id).sum()
            if area > 0:
                tissue_stats[tissue_id]['total_area'] += int(area)
        
        # 提取核实例
        instances = extract_nuclei_from_patch(class_map, args.min_area, args.max_area)
        
        for inst in instances:
            tissue_id = inst['tissue']
            nuc_type = inst['type']
            
            # 统计
            tissue_stats[tissue_id]['nuclei_counts'][nuc_type] += 1
            tissue_stats[tissue_id]['nuclei_areas'][nuc_type].append(inst['area'])
            
            # 保存核实例（如果该桶还没满）
            bucket_key = tissue_id
            if bucket_counts[bucket_key] < args.max_instances_per_bucket:
                bucket_dir = os.path.join(instances_dir, 
                    f'tissue_{tissue_id:02d}_{TISSUE_NAMES[tissue_id]}')
                inst_id = bucket_counts[bucket_key]
                
                np.savez_compressed(
                    os.path.join(bucket_dir, f'{inst_id:06d}.npz'),
                    mask=inst['mask'].astype(np.bool_),
                    type=np.array(nuc_type, dtype=np.int32),
                    area=np.array(inst['area'], dtype=np.int32),
                )
                
                bucket_counts[bucket_key] += 1
            
            total_instances += 1
    
    print(f"\nTotal instances extracted: {total_instances}")
    
    # 计算统计数据
    stats_output = {}
    summary_lines = []
    summary_lines.append(f"{'='*80}")
    summary_lines.append(f"Nuclei Library Statistics")
    summary_lines.append(f"{'='*80}")
    summary_lines.append(f"Total GT files: {len(gt_files)}")
    summary_lines.append(f"Total nuclei instances: {total_instances}")
    summary_lines.append(f"")
    
    for tissue_id in range(22):
        ts = tissue_stats[tissue_id]
        total_area = ts['total_area']
        
        if total_area == 0:
            continue
        
        tissue_name = TISSUE_NAMES[tissue_id]
        
        # 核密度：每 10000 像素² 的核数
        total_nuclei = sum(ts['nuclei_counts'].values())
        density = total_nuclei / (total_area / 10000.0) if total_area > 0 else 0
        
        # 核类型分布
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
        
        # Summary
        summary_lines.append(f"Tissue {tissue_id:2d} ({tissue_name}):")
        summary_lines.append(f"  Total area: {total_area:,} px")
        summary_lines.append(f"  Total nuclei: {total_nuclei:,}")
        summary_lines.append(f"  Density: {density:.2f} per 10k px²")
        summary_lines.append(f"  Stored instances: {bucket_counts.get(tissue_id, 0)}")
        for nuc_type in NUCLEI_CLASSES:
            td = type_dist[str(nuc_type)]
            if td['count'] > 0:
                summary_lines.append(
                    f"    {NUCLEI_NAMES[nuc_type]:15s}: {td['count']:6d} ({td['fraction']*100:5.1f}%) "
                    f"area={td['mean_area']:.0f}±{td['std_area']:.0f}")
        summary_lines.append("")
    
    # 保存
    with open(os.path.join(args.output_dir, 'statistics.json'), 'w') as f:
        json.dump(stats_output, f, indent=2)
    
    summary_text = '\n'.join(summary_lines)
    with open(os.path.join(args.output_dir, 'summary.txt'), 'w') as f:
        f.write(summary_text)
    
    print(summary_text)
    print(f"\nLibrary saved to {args.output_dir}")


if __name__ == '__main__':
    main()
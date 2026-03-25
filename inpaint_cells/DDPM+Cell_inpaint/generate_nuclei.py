#!/usr/bin/env python3
"""
阶段二：推理 —— 在用户编辑的区域填充合理的细胞核

输入:
    - 编辑后的组织 mask（RGB PNG）
    - 原始完整 mask（RGB PNG）
    - 或者：编辑后的组织 mask + 二值编辑区域 mask

输出:
    - 填充了细胞核的完整 mask（RGB PNG）

用法:
    python generate_nuclei.py \
        --library /data/huggingface/pathology_edit/nuclei_library \
        --input-mask /path/to/edited_tissue_mask.png \
        --edit-region /path/to/edit_region_mask.png \
        --original-mask /path/to/original_full_mask.png \
        --output /path/to/output.png

    # 或者批量测试（用 val 数据）
    python generate_nuclei.py \
        --library /data/huggingface/pathology_edit/nuclei_library \
        --test-dir /home/lyw/.../lama_dataset \
        --output-dir /data/huggingface/pathology_edit/nuclei_gen_results \
        --n 10
"""

import os
import json
import argparse
import glob
import random
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm


# ============================================================
#  颜色/类别定义（和 build_nuclei_library.py 一致）
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

NUCLEI_CLASSES = [101, 102, 103, 104, 105]

_rgb_to_val = {}
for val, rgb in COLOR_MAP.items():
    key = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
    _rgb_to_val[key] = val

_val_to_rgb = {v: rgb for v, rgb in COLOR_MAP.items()}


def rgb_to_class_map(rgb_img):
    encoded = rgb_img[:,:,0].astype(np.int64)*65536 + rgb_img[:,:,1].astype(np.int64)*256 + rgb_img[:,:,2].astype(np.int64)
    result = np.zeros(rgb_img.shape[:2], dtype=np.int64)
    for key, val in _rgb_to_val.items():
        result[encoded == key] = val
    return result


def class_map_to_rgb(class_map):
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for val, color in _val_to_rgb.items():
        rgb[class_map == val] = color
    return rgb


# ============================================================
#  核实例库
# ============================================================

class NucleiLibrary:
    def __init__(self, library_dir):
        self.library_dir = library_dir
        
        # 加载统计数据
        with open(os.path.join(library_dir, 'statistics.json'), 'r') as f:
            self.stats = json.load(f)
        
        # 加载核实例（按组织类型分桶）
        self.instances = defaultdict(list)  # tissue_id -> list of (mask, nuc_type, area)
        instances_dir = os.path.join(library_dir, 'nuclei_instances')
        
        for tissue_id in range(22):
            tissue_name = TISSUE_NAMES[tissue_id]
            bucket_dir = os.path.join(instances_dir, f'tissue_{tissue_id:02d}_{tissue_name}')
            
            if not os.path.isdir(bucket_dir):
                continue
            
            npz_files = sorted(glob.glob(os.path.join(bucket_dir, '*.npz')))
            for npz_path in npz_files:
                data = np.load(npz_path, allow_pickle=True)
                self.instances[tissue_id].append({
                    'mask': data['mask'],
                    'type': int(data['type']),
                    'area': int(data['area']),
                })
        
        total_loaded = sum(len(v) for v in self.instances.values())
        print(f"Loaded {total_loaded} nuclei instances from {library_dir}")
    
    def get_density(self, tissue_id):
        """获取该组织类型的核密度（每 10000 px²）"""
        key = str(tissue_id)
        if key not in self.stats:
            return 0.0
        return self.stats[key].get('density_per_10k_px', 0.0)
    
    def get_type_distribution(self, tissue_id):
        """获取该组织类型的核类型分布 {nuc_type: fraction}"""
        key = str(tissue_id)
        if key not in self.stats:
            return {}
        
        dist = {}
        for nuc_type_str, info in self.stats[key].get('nuclei_types', {}).items():
            nuc_type = int(nuc_type_str)
            if info['fraction'] > 0:
                dist[nuc_type] = info['fraction']
        return dist
    
    def sample_instance(self, tissue_id, nuc_type=None):
        """从指定组织类型的桶中随机抽一个核实例"""
        candidates = self.instances.get(tissue_id, [])
        
        if nuc_type is not None:
            candidates = [c for c in candidates if c['type'] == nuc_type]
        
        if not candidates:
            # fallback: 从任意组织类型中找该核类型
            if nuc_type is not None:
                for tid in range(22):
                    fallback = [c for c in self.instances.get(tid, []) if c['type'] == nuc_type]
                    if fallback:
                        return random.choice(fallback)
            return None
        
        return random.choice(candidates)


# ============================================================
#  泊松盘采样
# ============================================================

def poisson_disk_sampling(region_mask, min_distance, max_attempts=30):
    """
    在 region_mask 为 True 的区域内做泊松盘采样
    
    Args:
        region_mask: (H, W) bool
        min_distance: 点之间的最小距离
        max_attempts: 每个活跃点的最大尝试次数
    
    Returns:
        list of (y, x) 坐标
    """
    h, w = region_mask.shape
    
    # 有效像素的坐标
    valid_ys, valid_xs = np.where(region_mask)
    if len(valid_ys) == 0:
        return []
    
    cell_size = min_distance / np.sqrt(2)
    grid_h = int(np.ceil(h / cell_size))
    grid_w = int(np.ceil(w / cell_size))
    grid = -np.ones((grid_h, grid_w), dtype=np.int64)  # -1 = empty
    
    points = []
    active = []
    
    # 随机选一个起始点
    idx = random.randint(0, len(valid_ys) - 1)
    start = (int(valid_ys[idx]), int(valid_xs[idx]))
    points.append(start)
    active.append(0)
    
    gy, gx = int(start[0] / cell_size), int(start[1] / cell_size)
    grid[gy, gx] = 0
    
    while active:
        # 随机选一个活跃点
        active_idx = random.randint(0, len(active) - 1)
        point_idx = active[active_idx]
        py, px = points[point_idx]
        
        found = False
        for _ in range(max_attempts):
            # 在 min_distance 到 2*min_distance 的环形区域随机采样
            angle = random.uniform(0, 2 * np.pi)
            dist = random.uniform(min_distance, 2 * min_distance)
            ny = int(py + dist * np.sin(angle))
            nx = int(px + dist * np.cos(angle))
            
            # 边界检查
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            
            # 是否在有效区域内
            if not region_mask[ny, nx]:
                continue
            
            # 网格检查
            ngy, ngx = int(ny / cell_size), int(nx / cell_size)
            
            # 检查周围 5x5 网格有没有太近的点
            too_close = False
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    cgy, cgx = ngy + dy, ngx + dx
                    if 0 <= cgy < grid_h and 0 <= cgx < grid_w:
                        if grid[cgy, cgx] >= 0:
                            ey, ex = points[grid[cgy, cgx]]
                            if (ny - ey)**2 + (nx - ex)**2 < min_distance**2:
                                too_close = True
                                break
                if too_close:
                    break
            
            if not too_close:
                new_idx = len(points)
                points.append((ny, nx))
                active.append(new_idx)
                grid[ngy, ngx] = new_idx
                found = True
                break
        
        if not found:
            active.pop(active_idx)
    
    return points


# ============================================================
#  核填充
# ============================================================

def place_nucleus(output_map, center_y, center_x, nuc_instance, augment=True):
    """
    把一个核实例贴到 output_map 上
    
    Returns:
        True if placed successfully, False if overlaps
    """
    nuc_mask = nuc_instance['mask'].copy()
    nuc_type = nuc_instance['type']
    
    # 随机增强
    if augment:
        # 随机旋转 0/90/180/270
        k = random.randint(0, 3)
        nuc_mask = np.rot90(nuc_mask, k)
        
        # 随机翻转
        if random.random() > 0.5:
            nuc_mask = np.fliplr(nuc_mask)
        if random.random() > 0.5:
            nuc_mask = np.flipud(nuc_mask)
        
        # 轻微缩放 (0.8x ~ 1.2x)
        scale = random.uniform(0.8, 1.2)
        if scale != 1.0:
            new_h = max(1, int(nuc_mask.shape[0] * scale))
            new_w = max(1, int(nuc_mask.shape[1] * scale))
            nuc_mask = cv2.resize(nuc_mask.astype(np.uint8), (new_w, new_h), 
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
    
    h, w = nuc_mask.shape
    H, W = output_map.shape
    
    # 计算放置位置
    y1 = center_y - h // 2
    x1 = center_x - w // 2
    y2 = y1 + h
    x2 = x1 + w
    
    # 裁剪到图像边界
    src_y1 = max(0, -y1)
    src_x1 = max(0, -x1)
    src_y2 = h - max(0, y2 - H)
    src_x2 = w - max(0, x2 - W)
    
    dst_y1 = max(0, y1)
    dst_x1 = max(0, x1)
    dst_y2 = min(H, y2)
    dst_x2 = min(W, x2)
    
    if dst_y2 <= dst_y1 or dst_x2 <= dst_x1:
        return False
    
    local_mask = nuc_mask[src_y1:src_y2, src_x1:src_x2]
    
    # 检查重叠：目标区域是否已经有核
    target_region = output_map[dst_y1:dst_y2, dst_x1:dst_x2]
    overlap = (target_region >= 100) & local_mask
    if overlap.sum() > local_mask.sum() * 0.2:  # 超过 20% 重叠就跳过
        return False
    
    # 放置
    output_map[dst_y1:dst_y2, dst_x1:dst_x2][local_mask] = nuc_type
    return True


def fill_nuclei_in_region(output_map, edit_mask, library):
    """
    在 edit_mask 标记的区域内，根据组织类型填充合理的细胞核
    
    Args:
        output_map: (H, W) int64, 当前的 class map（组织层）
        edit_mask: (H, W) bool, 需要填充核的区域
        library: NucleiLibrary
    
    Modifies output_map in-place
    """
    H, W = output_map.shape
    
    # 对编辑区域内的每种组织类型分别处理
    tissue_types_in_region = np.unique(output_map[edit_mask])
    tissue_types_in_region = tissue_types_in_region[tissue_types_in_region < 100]
    
    total_placed = 0
    
    for tissue_id in tissue_types_in_region:
        tissue_id = int(tissue_id)
        
        # 该组织类型在编辑区域内的 mask
        tissue_region = edit_mask & (output_map == tissue_id)
        region_area = tissue_region.sum()
        
        if region_area < 50:  # 太小的区域跳过
            continue
        
        # 查表：核密度和类型分布
        density = library.get_density(tissue_id)
        type_dist = library.get_type_distribution(tissue_id)
        
        if density == 0 or not type_dist:
            continue
        
        # 计算需要放多少个核
        num_nuclei = int(density * region_area / 10000.0)
        
        # 加一点随机性
        num_nuclei = max(0, int(num_nuclei * random.uniform(0.7, 1.3)))
        
        if num_nuclei == 0:
            continue
        
        # 估算核的平均直径（用于泊松盘间距）
        # 从统计数据中获取平均面积
        stats = library.stats.get(str(tissue_id), {})
        mean_areas = []
        for nuc_str, info in stats.get('nuclei_types', {}).items():
            if info.get('mean_area', 0) > 0:
                mean_areas.append(info['mean_area'])
        avg_area = np.mean(mean_areas) if mean_areas else 100
        avg_diameter = np.sqrt(avg_area / np.pi) * 2
        min_distance = max(avg_diameter * 1.5, 8)  # 核间最小距离
        
        # 泊松盘采样确定核中心点
        centers = poisson_disk_sampling(tissue_region, min_distance)
        
        # 如果采样点太多，截断
        if len(centers) > num_nuclei:
            random.shuffle(centers)
            centers = centers[:num_nuclei]
        
        # 按类型分布确定每个核的类型
        nuc_types_list = []
        for nuc_type, frac in type_dist.items():
            count = max(1, int(len(centers) * frac))
            nuc_types_list.extend([nuc_type] * count)
        
        random.shuffle(nuc_types_list)
        
        # 放置核
        placed = 0
        for i, (cy, cx) in enumerate(centers):
            nuc_type = nuc_types_list[i % len(nuc_types_list)] if nuc_types_list else 101
            
            instance = library.sample_instance(tissue_id, nuc_type)
            if instance is None:
                continue
            
            if place_nucleus(output_map, cy, cx, instance, augment=True):
                placed += 1
        
        total_placed += placed
    
    return total_placed


# ============================================================
#  主函数
# ============================================================

def test_on_val(library, data_dir, output_dir, n=10):
    """在验证集上测试"""
    gt_dir = os.path.join(data_dir, 'ground_truth')
    val_dir = os.path.join(data_dir, 'val')
    os.makedirs(output_dir, exist_ok=True)
    
    # 找验证集文件
    val_files = sorted([f for f in glob.glob(os.path.join(val_dir, '*.png')) if '_mask' not in f])
    
    for idx in range(min(n, len(val_files))):
        val_path = val_files[idx]
        fname = os.path.basename(val_path)
        gt_path = os.path.join(gt_dir, fname)
        mask_path = val_path.replace('.png', '_mask001.png')
        
        if not os.path.exists(gt_path) or not os.path.exists(mask_path):
            continue
        
        print(f"[{idx+1}/{n}] {fname}")
        
        # 读取
        gt_rgb = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)
        input_rgb = cv2.cvtColor(cv2.imread(val_path), cv2.COLOR_BGR2RGB)
        edit_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 128
        
        gt_map = rgb_to_class_map(gt_rgb)
        input_map = rgb_to_class_map(input_rgb)
        
        # 在输入 map 上填充核
        output_map = input_map.copy()
        
        # 先清掉编辑区域内残留的核（如果有的话）
        output_map[edit_mask & (output_map >= 100)] = 0  # 临时设为 0
        
        # 恢复编辑区域的组织层（从 input 中获取，因为 input 保留了组织信息）
        # input_map 中 < 100 的值就是组织类型
        tissue_in_edit = input_map.copy()
        tissue_in_edit[tissue_in_edit >= 100] = 0  # 去掉残留核
        output_map[edit_mask] = tissue_in_edit[edit_mask]
        
        # 填充核
        placed = fill_nuclei_in_region(output_map, edit_mask, library)
        print(f"  Placed {placed} nuclei")
        
        # 转回 RGB
        output_rgb = class_map_to_rgb(output_map)
        gt_rgb_vis = gt_rgb.copy()
        input_rgb_vis = input_rgb.copy()
        
        # mask 轮廓
        mask_uint8 = edit_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for img in [input_rgb_vis, gt_rgb_vis, output_rgb]:
            cv2.drawContours(img, contours, -1, (255, 255, 255), 2)
        
        # 拼接：Input | GT | Generated
        h = gt_rgb.shape[0]
        w = gt_rgb.shape[1]
        row = np.concatenate([input_rgb_vis, gt_rgb_vis, output_rgb], axis=1)
        
        labeled = np.zeros((h + 30, row.shape[1], 3), dtype=np.uint8)
        labeled[30:] = row
        labeled[:30] = 40
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(labeled, 'Input (erased)', (5, 22), font, 0.5, (255,255,255), 1)
        cv2.putText(labeled, 'GT', (w+5, 22), font, 0.5, (255,255,255), 1)
        cv2.putText(labeled, f'Generated ({placed} nuclei)', (w*2+5, 22), font, 0.5, (255,255,255), 1)
        
        out_path = os.path.join(output_dir, f'gen_{idx:03d}_{fname}')
        cv2.imwrite(out_path, cv2.cvtColor(labeled, cv2.COLOR_RGB2BGR))
    
    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', required=True, help='Path to nuclei library directory')
    parser.add_argument('--test-dir', default=None, help='lama_dataset directory for testing')
    parser.add_argument('--output-dir', default='./nuclei_gen_results')
    parser.add_argument('--n', type=int, default=10, help='Number of test samples')
    # 单张推理参数
    parser.add_argument('--input-mask', default=None, help='Edited tissue mask (RGB PNG)')
    parser.add_argument('--edit-region', default=None, help='Edit region binary mask')
    parser.add_argument('--original-mask', default=None, help='Original full mask')
    parser.add_argument('--output', default=None, help='Output path')
    args = parser.parse_args()
    
    print("Loading nuclei library...")
    library = NucleiLibrary(args.library)
    
    if args.test_dir:
        test_on_val(library, args.test_dir, args.output_dir, args.n)
    elif args.input_mask and args.edit_region:
        # 单张推理
        input_rgb = cv2.cvtColor(cv2.imread(args.input_mask), cv2.COLOR_BGR2RGB)
        edit_mask = cv2.imread(args.edit_region, cv2.IMREAD_GRAYSCALE) > 128
        
        output_map = rgb_to_class_map(input_rgb)
        output_map[edit_mask & (output_map >= 100)] = 0
        
        placed = fill_nuclei_in_region(output_map, edit_mask, library)
        print(f"Placed {placed} nuclei")
        
        output_rgb = class_map_to_rgb(output_map)
        out_path = args.output or 'generated_mask.png'
        cv2.imwrite(out_path, cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR))
        print(f"Saved to {out_path}")
    else:
        print("Please specify --test-dir or --input-mask + --edit-region")


if __name__ == '__main__':
    main()
"""
NucleiLibrary — 细胞核实例库 + 采样/放置工具

包含：
  - NucleiLibrary: 从 build_library.py 建好的库中加载实例，按组织类型分桶
  - poisson_disk_sampling: 在指定区域内做泊松盘采样，确定核中心点
  - place_nucleus: 将一个核实例贴到 output map 上（带增强和重叠检测）
  - fill_nuclei_in_region: 对编辑区域按组织类型自动填充核（统一调度）
"""

import os
import json
import glob
import random
from collections import defaultdict

import cv2
import numpy as np

from ..utils.mask_utils import TISSUE_NAMES, NUCLEI_CLASSES


# ============================================================
#  核实例库
# ============================================================

class NucleiLibrary:
    """
    管理按组织类型分桶的细胞核实例库。

    目录结构：
        {library_dir}/
            statistics.json
            nuclei_instances/
                tissue_01_tumor/  → {id}.npz (mask, type, area)
                tissue_02_stroma/
                ...
    """
    def __init__(self, library_dir):
        self.library_dir = library_dir

        with open(os.path.join(library_dir, 'statistics.json'), 'r') as f:
            self.stats = json.load(f)

        self.instances = defaultdict(list)
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
    在 region_mask 为 True 的区域内做泊松盘采样。

    Args:
        region_mask: (H, W) bool
        min_distance: 点之间的最小距离
        max_attempts: 每个活跃点的最大尝试次数

    Returns:
        list of (y, x) 坐标
    """
    h, w = region_mask.shape
    valid_ys, valid_xs = np.where(region_mask)
    if len(valid_ys) == 0:
        return []

    cell_size = min_distance / np.sqrt(2)
    grid_h = int(np.ceil(h / cell_size))
    grid_w = int(np.ceil(w / cell_size))
    grid = -np.ones((grid_h, grid_w), dtype=np.int64)

    points = []
    active = []

    idx = random.randint(0, len(valid_ys) - 1)
    start = (int(valid_ys[idx]), int(valid_xs[idx]))
    points.append(start)
    active.append(0)

    gy, gx = int(start[0] / cell_size), int(start[1] / cell_size)
    grid[gy, gx] = 0

    while active:
        active_idx = random.randint(0, len(active) - 1)
        point_idx = active[active_idx]
        py, px = points[point_idx]

        found = False
        for _ in range(max_attempts):
            angle = random.uniform(0, 2 * np.pi)
            dist = random.uniform(min_distance, 2 * min_distance)
            ny = int(py + dist * np.sin(angle))
            nx = int(px + dist * np.cos(angle))

            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            if not region_mask[ny, nx]:
                continue

            ngy, ngx = int(ny / cell_size), int(nx / cell_size)

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
#  核放置
# ============================================================

def place_nucleus(output_map, center_y, center_x, nuc_instance, augment=True):
    """
    把一个核实例贴到 output_map 上。

    Args:
        output_map: (H, W) int64 — 输出 map (原位修改)
        center_y, center_x: 放置中心坐标
        nuc_instance: dict with 'mask' (bool), 'type' (int)
        augment: 是否随机旋转/翻转/缩放

    Returns:
        True if placed successfully, False if overlap too large
    """
    nuc_mask = nuc_instance['mask'].copy()
    nuc_type = nuc_instance['type']

    if augment:
        k = random.randint(0, 3)
        nuc_mask = np.rot90(nuc_mask, k)
        if random.random() > 0.5:
            nuc_mask = np.fliplr(nuc_mask)
        if random.random() > 0.5:
            nuc_mask = np.flipud(nuc_mask)
        scale = random.uniform(0.8, 1.2)
        if abs(scale - 1.0) > 0.05:
            new_h = max(1, int(nuc_mask.shape[0] * scale))
            new_w = max(1, int(nuc_mask.shape[1] * scale))
            nuc_mask = cv2.resize(
                nuc_mask.astype(np.uint8), (new_w, new_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

    h, w = nuc_mask.shape
    H, W = output_map.shape

    y1 = center_y - h // 2
    x1 = center_x - w // 2
    y2 = y1 + h
    x2 = x1 + w

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
    target_region = output_map[dst_y1:dst_y2, dst_x1:dst_x2]

    overlap = (target_region >= 100) & local_mask
    if overlap.sum() > local_mask.sum() * 0.2:
        return False

    output_map[dst_y1:dst_y2, dst_x1:dst_x2][local_mask] = nuc_type
    return True


# ============================================================
#  区域填充调度
# ============================================================

def fill_nuclei_in_region(output_map, edit_mask, library):
    """
    在 edit_mask 标记的区域内，根据组织类型填充合理的细胞核。

    Args:
        output_map: (H, W) int64, 当前的 class map（组织层）
        edit_mask: (H, W) bool, 需要填充核的区域
        library: NucleiLibrary

    Modifies output_map in-place.
    Returns: int — 成功放置的核数量
    """
    tissue_types_in_region = np.unique(output_map[edit_mask])
    tissue_types_in_region = tissue_types_in_region[tissue_types_in_region < 100]

    total_placed = 0

    for tissue_id in tissue_types_in_region:
        tissue_id = int(tissue_id)
        tissue_region = edit_mask & (output_map == tissue_id)
        region_area = tissue_region.sum()

        if region_area < 50:
            continue

        density = library.get_density(tissue_id)
        type_dist = library.get_type_distribution(tissue_id)

        if density == 0 or not type_dist:
            continue

        num_nuclei = int(density * region_area / 10000.0)
        num_nuclei = max(0, int(num_nuclei * random.uniform(0.7, 1.3)))

        if num_nuclei == 0:
            continue

        stats = library.stats.get(str(tissue_id), {})
        mean_areas = [info['mean_area'] for nuc_str, info
                      in stats.get('nuclei_types', {}).items()
                      if info.get('mean_area', 0) > 0]
        avg_area = np.mean(mean_areas) if mean_areas else 100
        avg_diameter = np.sqrt(avg_area / np.pi) * 2
        min_distance = max(avg_diameter * 1.5, 8)

        centers = poisson_disk_sampling(tissue_region, min_distance)

        if len(centers) > num_nuclei:
            random.shuffle(centers)
            centers = centers[:num_nuclei]

        nuc_types_list = []
        for nuc_type, frac in type_dist.items():
            count = max(1, int(len(centers) * frac))
            nuc_types_list.extend([nuc_type] * count)
        random.shuffle(nuc_types_list)

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

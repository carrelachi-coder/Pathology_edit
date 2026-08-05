"""
NucleiLibrary — 细胞核实例库 + 采样/放置工具 (Phase 4.2 多数据集适配)

包含：
  - NucleiLibrary: 从 build_library.py 建好的库中加载实例，按统一 fine 组织类型分桶
  - poisson_disk_sampling: 在指定区域内做泊松盘采样，确定核中心点
  - place_nucleus: 将一个核实例贴到 output map 上（带增强和重叠检测）
  - fill_nuclei_in_region: 对编辑区域按组织类型自动填充核（统一调度）

Phase 4.2 changes:
  - NucleiLibrary 支持 dataset 参数，自动查找 per-dataset 细胞库
  - 桶名使用统一 fine ID（如 tissue_01_Tumor/、tissue_08_Gleason 3/）
  - statistics.json 新格式: 顶层含 dataset/cancer_type 元数据, statistics 在子字段中
  - 兼容旧格式 statistics.json (Phase 4.1 及更早)
"""

import os
import json
import glob
import random
from collections import defaultdict

import cv2
import numpy as np

from ..utils.mask_utils import (
    TISSUE_NAMES, NUCLEI_CLASSES, NUM_TISSUE,
    NUCLEI_RAW_TO_INDEX, NUCLEI_INDEX_TO_RAW,
)


# ============================================================
#  核实例库
# ============================================================

class NucleiLibrary:
    """
    管理按组织类型分桶的细胞核实例库。

    目录结构：
        {library_dir}/
            statistics.json         — 统计数据 (新格式含 dataset 元数据)
            nuclei_instances/
                tissue_01_Tumor/    → {id}.npz (mask, type, area)
                tissue_02_Stroma/
                ...

    Args:
        library_dir: 细胞库根目录
        dataset: 可选, 数据集名称 (用于日志)
    """
    def __init__(self, library_dir, dataset=None):
        self.library_dir = library_dir
        self.dataset = dataset

        with open(os.path.join(library_dir, 'statistics.json'), 'r') as f:
            raw_stats = json.load(f)

        # 兼容新旧格式:
        #   新格式 (Phase 4.2): {'dataset': ..., 'statistics': {tissue_id: {...}}}
        #   旧格式 (Phase 4.1): {tissue_id: {...}} 直接平铺
        if 'statistics' in raw_stats and isinstance(raw_stats['statistics'], dict):
            self.meta = {k: v for k, v in raw_stats.items() if k != 'statistics'}
            self.stats = raw_stats['statistics']
        else:
            self.meta = {}
            self.stats = raw_stats

        self.instances = defaultdict(list)
        instances_dir = os.path.join(library_dir, 'nuclei_instances')

        # 动态扫描所有 tissue_XX_xxx 子目录 (不假设固定数量)
        if os.path.isdir(instances_dir):
            for entry in sorted(os.listdir(instances_dir)):
                bucket_path = os.path.join(instances_dir, entry)
                if not os.path.isdir(bucket_path):
                    continue
                if not entry.startswith('tissue_'):
                    continue

                # 解析 tissue_id: tissue_01_Tumor → 1
                try:
                    tissue_id = int(entry.split('_')[1])
                except (IndexError, ValueError):
                    continue

                npz_files = sorted(glob.glob(os.path.join(bucket_path, '*.npz')))
                for npz_path in npz_files:
                    data = np.load(npz_path, allow_pickle=True)
                    self.instances[tissue_id].append({
                        'mask': data['mask'],
                        'type': int(data['type']),
                        'area': int(data['area']),
                    })

        total_loaded = sum(len(v) for v in self.instances.values())
        ds_info = f" ({dataset})" if dataset else ""
        print(f"Loaded {total_loaded} nuclei instances from {library_dir}{ds_info}")

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

    def sample_instance(self, tissue_id, nuc_type=None, allow_cross_tissue=True):
        """从指定组织类型的桶中随机抽一个核实例"""
        candidates = self.instances.get(tissue_id, [])
        if nuc_type is not None:
            candidates = [c for c in candidates if c['type'] == nuc_type]
        if not candidates:
            if nuc_type is not None and allow_cross_tissue:
                # Fallback: 从所有桶中找该类型的核
                for tid in self.instances:
                    fallback = [c for c in self.instances[tid] if c['type'] == nuc_type]
                    if fallback:
                        return random.choice(fallback)
            return None
        return random.choice(candidates)


class ReferenceNucleiInstancePool:
    """Extract reusable, class-preserving nucleus shapes from one reference mask.

    The reference mask may use raw CellViT IDs (101-105) or the internal
    indices (1-5).  Components touching the patch boundary are excluded by
    default because their full shape is not observable.  Optional per-class
    area-outlier filtering is available for masks with merged components, but
    is disabled by default so genuine pleomorphic nuclei are preserved.
    """

    def __init__(self, instances=None, rejected=None):
        self.instances = defaultdict(list)
        for nuc_type, values in (instances or {}).items():
            self.instances[int(nuc_type)].extend(values)
        self.rejected = {
            "border": 0,
            "too_small": 0,
            "area_outlier": 0,
            **(rejected or {}),
        }

    @classmethod
    def from_mask(
        cls,
        nuclei_mask,
        *,
        min_area=8,
        exclude_border=True,
        max_area_ratio_to_median=0.0,
    ):
        mask = np.asarray(nuclei_mask)
        if mask.ndim != 2:
            raise ValueError(f"reference nuclei mask must be 2D, got {mask.shape}")

        # Accept either raw mask IDs (101-105) or model indices (1-5).
        positive_values = set(int(v) for v in np.unique(mask) if int(v) > 0)
        internal_values = set(NUCLEI_INDEX_TO_RAW)
        if positive_values and positive_values.issubset(internal_values):
            raw_mask = np.zeros(mask.shape, dtype=np.int64)
            for index, raw_id in NUCLEI_INDEX_TO_RAW.items():
                raw_mask[mask == index] = raw_id
        else:
            raw_mask = mask.astype(np.int64, copy=False)

        instances = defaultdict(list)
        rejected = {"border": 0, "too_small": 0, "area_outlier": 0}
        height, width = raw_mask.shape

        for nuc_type in NUCLEI_CLASSES:
            count, labels, stats, centroids = cv2.connectedComponentsWithStats(
                (raw_mask == nuc_type).astype(np.uint8),
                connectivity=8,
            )
            candidates = []
            for component_id in range(1, count):
                x, y, component_width, component_height, area = (
                    int(value) for value in stats[component_id]
                )
                touches_border = (
                    x == 0
                    or y == 0
                    or x + component_width == width
                    or y + component_height == height
                )
                if exclude_border and touches_border:
                    rejected["border"] += 1
                    continue
                if area < min_area:
                    rejected["too_small"] += 1
                    continue
                candidates.append(
                    (component_id, x, y, component_width, component_height, area)
                )

            max_area = None
            if len(candidates) >= 5 and max_area_ratio_to_median > 0:
                median_area = float(np.median([item[-1] for item in candidates]))
                max_area = median_area * float(max_area_ratio_to_median)

            for component_id, x, y, component_width, component_height, area in candidates:
                if max_area is not None and area > max_area:
                    rejected["area_outlier"] += 1
                    continue
                crop = labels[
                    y:y + component_height,
                    x:x + component_width,
                ] == component_id
                instances[int(nuc_type)].append(
                    {
                        "mask": np.ascontiguousarray(crop, dtype=bool),
                        "type": int(nuc_type),
                        "area": int(area),
                        "source": "reference",
                        "center_y": float(centroids[component_id][1]),
                        "center_x": float(centroids[component_id][0]),
                    }
                )

        return cls(instances=instances, rejected=rejected)

    def subset_by_center_region(self, region_mask):
        """Return reference instances whose original centroids lie in a region."""

        region = np.asarray(region_mask, dtype=bool)
        if region.ndim != 2:
            raise ValueError("reference subset region must be 2D")
        height, width = region.shape
        instances = defaultdict(list)
        for nuc_type, values in self.instances.items():
            for instance in values:
                if "center_y" not in instance or "center_x" not in instance:
                    continue
                row = int(
                    np.clip(round(float(instance["center_y"])), 0, height - 1)
                )
                col = int(
                    np.clip(round(float(instance["center_x"])), 0, width - 1)
                )
                if region[row, col]:
                    instances[int(nuc_type)].append(instance)
        return ReferenceNucleiInstancePool(instances=instances)

    def counts(self):
        return {
            int(nuc_type): len(self.instances.get(int(nuc_type), []))
            for nuc_type in NUCLEI_CLASSES
        }

    def area_samples(self, nuc_type):
        """Return accepted reference-instance areas for one CellViT class."""
        values = []
        for instance in self.instances.get(int(nuc_type), []):
            area = int(instance.get("area", np.count_nonzero(instance["mask"])))
            if area > 0:
                values.append(area)
        return values

    def area_statistics(self):
        """Summarize the patch-local size distribution for every nucleus type."""
        summaries = {}
        for nuc_type in NUCLEI_CLASSES:
            values = np.asarray(self.area_samples(nuc_type), dtype=np.float64)
            if values.size == 0:
                continue
            summaries[str(int(nuc_type))] = {
                "count": int(values.size),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "q25": float(np.percentile(values, 25)),
                "median": float(np.median(values)),
                "q75": float(np.percentile(values, 75)),
                "max": float(np.max(values)),
            }
        return summaries

    def describe(self):
        counts = self.counts()
        return {
            "counts_by_type": {str(key): int(value) for key, value in counts.items()},
            "total": int(sum(counts.values())),
            "area_statistics_by_type": self.area_statistics(),
            "rejected": {key: int(value) for key, value in self.rejected.items()},
        }


class ReferenceFirstNucleiSampler:
    """Use same-class reference shapes once, then fall back to the library.

    Library shapes are resized to an empirical area sampled from the current
    patch's same-class reference nuclei. This retains library morphology while
    matching patch-local cell-size statistics. If the patch contains no
    accepted nucleus of that class, the library shape remains uncalibrated and
    the fallback is recorded explicitly.
    """

    def __init__(
        self,
        library,
        reference_pool=None,
        *,
        calibrate_library_size=True,
        library_size_min_scale=0.5,
        library_size_max_scale=2.0,
        library_size_log_area_jitter=0.05,
    ):
        if library_size_min_scale <= 0:
            raise ValueError("library_size_min_scale must be positive")
        if library_size_max_scale < library_size_min_scale:
            raise ValueError("library_size_max_scale must be >= library_size_min_scale")
        if library_size_log_area_jitter < 0:
            raise ValueError("library_size_log_area_jitter must be non-negative")
        self.library = library
        self.reference_areas_by_type = {
            int(nuc_type): (
                reference_pool.area_samples(int(nuc_type))
                if reference_pool is not None
                else []
            )
            for nuc_type in NUCLEI_CLASSES
        }
        self.calibrate_library_size = bool(calibrate_library_size)
        self.library_size_min_scale = float(library_size_min_scale)
        self.library_size_max_scale = float(library_size_max_scale)
        self.library_size_log_area_jitter = float(library_size_log_area_jitter)
        self.initial_counts = (
            reference_pool.counts()
            if reference_pool is not None
            else {int(nuc_type): 0 for nuc_type in NUCLEI_CLASSES}
        )
        self.remaining = {}
        for nuc_type in NUCLEI_CLASSES:
            items = list(
                reference_pool.instances.get(int(nuc_type), [])
                if reference_pool is not None
                else []
            )
            random.shuffle(items)
            self.remaining[int(nuc_type)] = items
        self.requested_by_type = defaultdict(int)
        self.selected_by_source = defaultdict(int)
        self.library_fallback_by_type = defaultdict(int)
        self.library_size_calibrated_by_type = defaultdict(int)
        self.library_size_uncalibrated_no_reference_by_type = defaultdict(int)
        self.library_size_scale_clamped_by_type = defaultdict(int)
        self.library_size_records_by_type = defaultdict(list)

    def _calibrate_library_instance(self, instance):
        if instance is None or not self.calibrate_library_size:
            return instance

        nuc_type = int(instance.get("type", 0))
        reference_areas = self.reference_areas_by_type.get(nuc_type, [])
        if not reference_areas:
            self.library_size_uncalibrated_no_reference_by_type[nuc_type] += 1
            return instance

        source_mask = np.asarray(instance["mask"], dtype=bool)
        source_area = int(np.count_nonzero(source_mask))
        if source_area <= 0:
            return instance

        empirical_area = float(random.choice(reference_areas))
        if self.library_size_log_area_jitter > 0:
            empirical_area *= float(
                np.exp(random.gauss(0.0, self.library_size_log_area_jitter))
            )
        target_area = max(1, int(round(empirical_area)))
        requested_scale = float(np.sqrt(target_area / source_area))
        applied_scale = float(
            np.clip(
                requested_scale,
                self.library_size_min_scale,
                self.library_size_max_scale,
            )
        )
        scale_clamped = not np.isclose(requested_scale, applied_scale)

        new_height = max(1, int(round(source_mask.shape[0] * applied_scale)))
        new_width = max(1, int(round(source_mask.shape[1] * applied_scale)))
        resized = cv2.resize(
            source_mask.astype(np.uint8),
            (new_width, new_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        if np.any(resized):
            ys, xs = np.where(resized)
            resized = resized[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        actual_area = int(np.count_nonzero(resized))

        calibrated = dict(instance)
        calibrated.update(
            {
                "mask": np.ascontiguousarray(resized, dtype=bool),
                "area": actual_area,
                "size_calibrated": True,
                "size_calibration": {
                    "basis": "same_class_reference_empirical_area",
                    "source_area": source_area,
                    "target_area": target_area,
                    "actual_area": actual_area,
                    "requested_scale": requested_scale,
                    "applied_scale": applied_scale,
                    "scale_clamped": bool(scale_clamped),
                    "reference_sample_count": len(reference_areas),
                },
            }
        )
        self.library_size_calibrated_by_type[nuc_type] += 1
        if scale_clamped:
            self.library_size_scale_clamped_by_type[nuc_type] += 1
        self.library_size_records_by_type[nuc_type].append(
            calibrated["size_calibration"]
        )
        return calibrated

    def sample_instance(self, tissue_id, nuc_type, allow_cross_tissue=True):
        """Return ``(instance, source)`` for an exact requested nucleus type."""
        nuc_type = int(nuc_type)
        self.requested_by_type[nuc_type] += 1
        available = self.remaining.get(nuc_type, [])
        if available:
            selected_index = random.randrange(len(available))
            self.selected_by_source["reference"] += 1
            return available.pop(selected_index), "reference"

        instance = self.library.sample_instance(
            tissue_id,
            nuc_type,
            allow_cross_tissue=allow_cross_tissue,
        )
        if instance is not None:
            instance = self._calibrate_library_instance(instance)
            self.selected_by_source["library"] += 1
            self.library_fallback_by_type[nuc_type] += 1
            return instance, "library"
        return None, None

    def release_failed_instance(self, instance, source):
        """Return an unsuccessfully placed reference shape to the pool."""

        if instance is None or str(source) != "reference":
            return
        nuc_type = int(instance.get("type", 0))
        if nuc_type not in self.remaining:
            return
        self.remaining[nuc_type].append(instance)
        self.selected_by_source["reference"] = max(
            0,
            int(self.selected_by_source.get("reference", 0)) - 1,
        )

    def sample_library_instance(
        self,
        tissue_id,
        nuc_type=None,
        *,
        allow_cross_tissue=True,
        requested_type=None,
        calibrate_size=True,
    ):
        """Use the legacy library fallback while keeping provenance counts."""
        instance = self.library.sample_instance(
            tissue_id,
            nuc_type,
            allow_cross_tissue=allow_cross_tissue,
        )
        if instance is None:
            return None, None
        if calibrate_size:
            instance = self._calibrate_library_instance(instance)
        self.selected_by_source["library"] += 1
        key = requested_type if requested_type is not None else instance.get("type", nuc_type)
        if key is not None:
            self.library_fallback_by_type[int(key)] += 1
        return instance, "library"

    def diagnostics(self):
        size_records = {}
        for nuc_type, records in self.library_size_records_by_type.items():
            if not records:
                continue
            size_records[str(int(nuc_type))] = {
                "count": len(records),
                "mean_source_area": float(np.mean([item["source_area"] for item in records])),
                "mean_target_area": float(np.mean([item["target_area"] for item in records])),
                "mean_actual_area": float(np.mean([item["actual_area"] for item in records])),
                "mean_applied_scale": float(np.mean([item["applied_scale"] for item in records])),
                "scale_clamped": int(sum(item["scale_clamped"] for item in records)),
            }
        return {
            "policy": "same_class_reference_without_replacement_then_library",
            "initial_reference_counts_by_type": {
                str(key): int(value) for key, value in self.initial_counts.items()
            },
            "remaining_reference_counts_by_type": {
                str(key): len(value) for key, value in self.remaining.items()
            },
            "requested_by_type": {
                str(key): int(value) for key, value in self.requested_by_type.items()
            },
            "selected_by_source": {
                "reference": int(self.selected_by_source.get("reference", 0)),
                "library": int(self.selected_by_source.get("library", 0)),
            },
            "library_fallback_by_type": {
                str(key): int(value) for key, value in self.library_fallback_by_type.items()
            },
            "library_size_calibration": {
                "enabled": self.calibrate_library_size,
                "policy": "same_class_reference_empirical_area_bootstrap",
                "min_scale": self.library_size_min_scale,
                "max_scale": self.library_size_max_scale,
                "log_area_jitter": self.library_size_log_area_jitter,
                "calibrated_by_type": {
                    str(key): int(value)
                    for key, value in self.library_size_calibrated_by_type.items()
                },
                "uncalibrated_no_reference_by_type": {
                    str(key): int(value)
                    for key, value in self.library_size_uncalibrated_no_reference_by_type.items()
                },
                "scale_clamped_by_type": {
                    str(key): int(value)
                    for key, value in self.library_size_scale_clamped_by_type.items()
                },
                "summary_by_type": size_records,
            },
        }


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


# ============================================================
#  Layered storage variants (AD-1)
# ============================================================

def _largest_eight_connected_component(mask):
    binary = np.asarray(mask, dtype=np.uint8)
    component_count, labels = cv2.connectedComponents(binary, connectivity=8)
    if component_count <= 1:
        return binary.astype(bool)
    areas = np.bincount(labels.ravel(), minlength=component_count)
    areas[0] = 0
    return labels == int(np.argmax(areas))


def place_nucleus_layered(
    nuclei_map,
    center_y,
    center_x,
    nuc_instance,
    augment=True,
    max_overlap_fraction=0.0,
    valid_tissue_mask=None,
    require_full_tissue_containment=False,
    rotation_quarters=None,
    flip_horizontal=None,
    flip_vertical=None,
    scale=None,
    minimum_separation_px=0,
):
    """
    把一个核实例贴到 nuclei_map 上 (AD-1: 分层存储, nuclei_map 值域 0-5 internal index)。

    Args:
        nuclei_map: (H, W) int64, values [0, 5] — 独立 nuclei 层 (原位修改)
        center_y, center_x: 放置中心坐标
        nuc_instance: dict with 'mask' (bool), 'type' (int, raw 101-105)
        augment: 是否随机旋转/翻转/缩放
        max_overlap_fraction: Maximum fraction of the proposed nucleus that may
            overlap an existing nucleus. Production reference-preserving
            sampling passes ``0.0`` so retained and newly placed nuclei remain
            bitwise disjoint.
        valid_tissue_mask: Optional boolean mask defining biological support.
        require_full_tissue_containment: Reject a truncated proposal or any
            proposal pixel outside ``valid_tissue_mask``.
        minimum_separation_px: Empty-pixel margin required between the proposal
            and every retained or newly generated nucleus.

    Returns:
        True if placed successfully
    """
    nuc_mask = nuc_instance['mask'].copy()
    nuc_type_raw = nuc_instance['type']
    nuc_type_idx = NUCLEI_RAW_TO_INDEX.get(nuc_type_raw, 0)

    if augment:
        k = (
            random.randint(0, 3)
            if rotation_quarters is None
            else int(rotation_quarters) % 4
        )
        nuc_mask = np.rot90(nuc_mask, k)
        horizontal = (
            random.random() > 0.5
            if flip_horizontal is None
            else bool(flip_horizontal)
        )
        vertical = (
            random.random() > 0.5
            if flip_vertical is None
            else bool(flip_vertical)
        )
        if horizontal:
            nuc_mask = np.fliplr(nuc_mask)
        if vertical:
            nuc_mask = np.flipud(nuc_mask)
        preserve_patch_size = (
            nuc_instance.get("source") == "reference"
            or bool(nuc_instance.get("size_calibrated", False))
        )
        applied_scale = (
            1.0
            if preserve_patch_size
            else (
                scale
                if scale is not None
                else random.uniform(0.8, 1.2)
            )
        )
        resize_threshold = 1e-6 if scale is not None else 0.05
        if abs(float(applied_scale) - 1.0) > resize_threshold:
            new_h = max(1, int(nuc_mask.shape[0] * float(applied_scale)))
            new_w = max(1, int(nuc_mask.shape[1] * float(applied_scale)))
            nuc_mask = cv2.resize(
                nuc_mask.astype(np.uint8), (new_w, new_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

    nuc_mask = _largest_eight_connected_component(nuc_mask)
    if not np.any(nuc_mask):
        return False

    h, w = nuc_mask.shape
    H, W = nuclei_map.shape

    y1 = center_y - h // 2
    x1 = center_x - w // 2
    y2 = y1 + h
    x2 = x1 + w

    boundary_truncated = y1 < 0 or x1 < 0 or y2 > H or x2 > W
    if require_full_tissue_containment and boundary_truncated:
        return False

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
    target_region = nuclei_map[dst_y1:dst_y2, dst_x1:dst_x2]

    if require_full_tissue_containment:
        if valid_tissue_mask is None:
            raise ValueError(
                "valid_tissue_mask is required for full tissue containment"
            )
        allowed = np.asarray(valid_tissue_mask, dtype=bool)
        if allowed.shape != nuclei_map.shape:
            raise ValueError(
                "valid_tissue_mask and nuclei_map must share one shape"
            )
        local_allowed = allowed[dst_y1:dst_y2, dst_x1:dst_x2]
        if np.any(local_mask & ~local_allowed):
            return False

    # Overlap check: nuclei_map uses index 0-5 (0=background, 1-5=cell types)
    overlap = (target_region > 0) & local_mask
    maximum_overlap = max(0.0, float(max_overlap_fraction))
    if overlap.sum() > local_mask.sum() * maximum_overlap:
        return False
    separation = max(0, int(minimum_separation_px))
    if separation > 0:
        extended_y1 = max(0, dst_y1 - separation)
        extended_x1 = max(0, dst_x1 - separation)
        extended_y2 = min(H, dst_y2 + separation)
        extended_x2 = min(W, dst_x2 + separation)
        proposal = np.zeros(
            (extended_y2 - extended_y1, extended_x2 - extended_x1),
            dtype=np.uint8,
        )
        proposal[
            dst_y1 - extended_y1 : dst_y2 - extended_y1,
            dst_x1 - extended_x1 : dst_x2 - extended_x1,
        ][local_mask] = 1
        proposal = cv2.dilate(
            proposal,
            np.ones(
                (2 * separation + 1, 2 * separation + 1),
                dtype=np.uint8,
            ),
        )
        occupied = (
            nuclei_map[
                extended_y1:extended_y2,
                extended_x1:extended_x2,
            ]
            > 0
        )
        if np.any((proposal > 0) & occupied):
            return False

    nuclei_map[dst_y1:dst_y2, dst_x1:dst_x2][local_mask] = nuc_type_idx
    return True


def fill_nuclei_in_region_layered(nuclei_map, tissue_map, edit_mask, library):
    """
    在 edit_mask 标记的区域内填充合理的细胞核 (AD-1: 分层存储)。

    与 fill_nuclei_in_region 不同:
      - nuclei_map 和 tissue_map 是独立的两层
      - nuclei_map 使用 internal index (0-5)
      - tissue_map 用于确定各区域的组织类型

    Args:
        nuclei_map: (H, W) int64, values [0, 5], 独立 nuclei 层 (原位修改)
        tissue_map: (H, W) int64, values [0, 15], 独立 tissue 层 (只读)
        edit_mask: (H, W) bool, 需要填充核的区域
        library: NucleiLibrary

    Modifies nuclei_map in-place.
    Returns: int — 成功放置的核数量
    """
    tissue_types_in_region = np.unique(tissue_map[edit_mask])

    total_placed = 0

    for tissue_id in tissue_types_in_region:
        tissue_id = int(tissue_id)
        tissue_region = edit_mask & (tissue_map == tissue_id)
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
            if place_nucleus_layered(
                nuclei_map,
                cy,
                cx,
                instance,
                augment=True,
                max_overlap_fraction=0.0,
            ):
                placed += 1

        total_placed += placed

    return total_placed

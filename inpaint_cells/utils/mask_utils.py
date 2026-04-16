"""
共享的 mask 工具函数

提供 RGB ↔ class ID 转换、tissue/nuclei 分离、one-hot 编码等基础操作。
三个原始文件 (train_prob_net.py, build_nuclei_library.py, generate_nuclei.py)
都在重复定义这些常量和函数，统一到此处消除重复。

NOTE: 当前使用 BCSS 22 类硬编码。
Phase 4 适配时将改为从 dataset_config 的统一 16 类标签读取。
"""

import numpy as np
import cv2


# ============================================================
#  颜色/类别常量 (BCSS 22 类 + CellViT 5 类细胞核)
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

NUCLEI_NAMES = {
    101: 'neoplastic', 102: 'inflammatory', 103: 'connective',
    104: 'dead', 105: 'epithelial',
}

NUCLEI_CLASSES = [101, 102, 103, 104, 105]
NUM_TISSUE = 22
NUM_NUCLEI = 6  # 背景(0) + 5类核


# ============================================================
#  RGB → class value 查找表
# ============================================================

_rgb_to_val = {}
for _val, _rgb in COLOR_MAP.items():
    _key = _rgb[0] * 65536 + _rgb[1] * 256 + _rgb[2]
    _rgb_to_val[_key] = _val

_val_to_rgb = {v: rgb for v, rgb in COLOR_MAP.items()}


# ============================================================
#  转换函数
# ============================================================

def rgb_to_class_map(rgb_img):
    """RGB 图像 → class value map (H, W), int64"""
    encoded = (rgb_img[:,:,0].astype(np.int64) * 65536
             + rgb_img[:,:,1].astype(np.int64) * 256
             + rgb_img[:,:,2].astype(np.int64))
    result = np.zeros(rgb_img.shape[:2], dtype=np.int64)
    for key, val in _rgb_to_val.items():
        result[encoded == key] = val
    return result


def class_map_to_rgb(class_map):
    """class value map (H, W) → RGB 图像"""
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for val, color in _val_to_rgb.items():
        rgb[class_map == val] = color
    return rgb


def split_tissue_nuclei(class_map):
    """
    将合并的 class map 拆分为 tissue layer 和 nuclei layer。
    tissue 层中细胞核像素用 EDT 推断其下方组织。
    nuclei 层值域: 0=背景, 1-5=五类核。
    """
    tissue = class_map.copy()
    nuclei = np.zeros_like(class_map)

    for i, nuc_val in enumerate(NUCLEI_CLASSES):
        mask = class_map == nuc_val
        nuclei[mask] = i + 1

    nuc_mask = class_map >= 100
    if nuc_mask.any():
        from scipy.ndimage import distance_transform_edt
        _, nearest_idx = distance_transform_edt(
            nuc_mask, return_distances=True, return_indices=True
        )
        tissue[nuc_mask] = class_map[nearest_idx[0][nuc_mask], nearest_idx[1][nuc_mask]]
        tissue = np.clip(tissue, 0, 21)

    return tissue, nuclei


def to_onehot(index_map, num_classes):
    """index map → one-hot tensor (num_classes, H, W), float32"""
    oh = np.zeros((num_classes, index_map.shape[0], index_map.shape[1]), dtype=np.float32)
    for c in range(num_classes):
        oh[c] = (index_map == c).astype(np.float32)
    return oh


# ============================================================
#  可视化辅助
# ============================================================

NUCLEI_RGB = {
    0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 255, 0],
    3: [0, 80, 255], 4: [255, 255, 0], 5: [255, 0, 255],
}

TISSUE_RGB_MAP = {i: COLOR_MAP[i] for i in range(22)}


def index_to_rgb(index_map, color_map):
    """通用的 index → RGB 可视化"""
    h, w = index_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in color_map.items():
        rgb[index_map == idx] = color
    return rgb


def overlay(tissue_map, nuclei_map):
    """将 tissue 和 nuclei 两层叠加渲染为 RGB（nuclei 覆盖 tissue）"""
    tissue_rgb = index_to_rgb(tissue_map, TISSUE_RGB_MAP)
    nuc_rgb = index_to_rgb(nuclei_map, NUCLEI_RGB)
    result = tissue_rgb.copy()
    result[nuclei_map > 0] = nuc_rgb[nuclei_map > 0]
    return result

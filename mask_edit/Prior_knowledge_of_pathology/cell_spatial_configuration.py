import numpy as np
import cv2
import json
import os
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.spatial import KDTree
from tqdm import tqdm
import matplotlib.pyplot as plt
# --- 1. 配置与映射 ---
COLOR_MAP = {
    0: [30, 30, 30], 1: [180, 60, 60], 2: [60, 150, 60], 3: [140, 60, 180],
    4: [60, 60, 180], 5: [180, 180, 80], 6: [160, 40, 40], 7: [40, 40, 40],
    8: [80, 150, 150], 9: [200, 170, 100], 10: [180, 120, 150], 11: [120, 120, 190],
    12: [100, 190, 190], 13: [200, 140, 60], 14: [140, 200, 100], 15: [140, 140, 140],
    16: [200, 200, 130], 17: [150, 80, 60], 18: [60, 140, 100], 19: [190, 40, 40],
    20: [80, 60, 150], 21: [170, 170, 170],
    101: [255, 0, 0], 102: [0, 255, 0], 103: [0, 80, 255], 104: [255, 255, 0], 105: [255, 0, 255]
}


TISSUE_NAME_MAP = {
    0: "outside_roi", 1: "tumor", 2: "stroma", 3: "lymphocytic_infiltrate",
    4: "necrosis_or_debris", 5: "glandular_secretions", 6: "blood", 7: "exclude",
    8: "metaplasia_NOS", 9: "fat", 10: "plasma_cells", 11: "other_immune_infiltrate",
    12: "mucoid_material", 13: "normal_acinus_or_duct", 14: "lymphatics",
    15: "undetermined", 16: "nerve", 17: "skin_adnexa", 18: "blood_vessel",
    19: "angioinvasion", 20: "dcis", 21: "other"
}

CELL_NAME_MAP = {
    101: "neoplastic", 102: "inflammatory", 103: "connective", 
    104: "dead", 105: "epithelial"
}
TISSUE_IDS = list(range(22))
CELL_IDS = [101, 102, 103, 104, 105]


IS_FIRST_RUN = True # 控制仅可视化第一张图

# --- 2. 工具函数 ---
def rgb_to_id_mask(img_array):
    h, w, _ = img_array.shape
    id_mask = np.full((h, w), -1, dtype=np.int16)
    for idx, color in COLOR_MAP.items():
        id_mask[np.all(img_array == color, axis=-1)] = idx
    return id_mask

def id_to_rgb(id_mask):
    h, w = id_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in COLOR_MAP.items():
        if idx <= 21: rgb[id_mask == idx] = color
    return rgb

# --- 3. 核心统计逻辑 ---
def get_spatial_features(mask_path, k=5):
    global IS_FIRST_RUN
    img_np = np.array(Image.open(mask_path).convert("RGB"))
    h, w = img_np.shape[:2]
    id_mask = rgb_to_id_mask(img_np)
    
    # 组织修复逻辑
    is_tissue = (id_mask >= 0) & (id_mask <= 21)
    _, indices = ndimage.distance_transform_edt(~is_tissue, return_indices=True)
    full_tissue_ids = id_mask[indices[0], indices[1]]
    
    # Bug 2 修复逻辑：按 (Tid, Cid) 二元组存储坐标
    # 结构：{ tid: { cid: [coords] } }
    tissue_cell_coords = {tid: {cid: [] for cid in CELL_IDS} for tid in TISSUE_IDS}
    
    for cid in CELL_IDS:
        cell_pixels = (id_mask == cid).astype(np.uint8)
        if not np.any(cell_pixels): continue
        labeled, num = ndimage.label(cell_pixels)
        centroids = ndimage.center_of_mass(cell_pixels, labeled, range(1, num + 1))
        if not isinstance(centroids, list): centroids = [centroids]
        
        for pt in centroids:
            if not isinstance(pt, (tuple, list)) or len(pt) != 2: continue
            cy, cx = pt
            if np.isnan(cy): continue
            # 确定质心落在哪个组织上
            tid = full_tissue_ids[int(np.clip(cy, 0, h-1)), int(np.clip(cx, 0, w-1))]
            tissue_cell_coords[tid][cid].append((cy, cx))

    # 计算分类 NND
    batch_results = [] # 存储本图中所有有效的 (tid, cid, nnds)
    vis_target = None # 可视化用的目标数据

    for tid in TISSUE_IDS:
        for cid in CELL_IDS:
            coords = tissue_cell_coords[tid][cid]
            if len(coords) < k + 1: continue # 细胞数必须大于K邻居数
            
            coords_array = np.array(coords)
            tree = KDTree(coords_array)
            # 计算同类细胞间的最近邻距离
            dists, idxs = tree.query(coords_array, k=k+1)
            avg_k_nnds = dists[:, 1:].mean(axis=1)
            
            batch_results.append((tid, cid, avg_k_nnds.tolist()))

            # 选择“最密集”的一个类别进行可视化展示原理
            if IS_FIRST_RUN:
                if vis_target is None or len(coords) > len(vis_target['coords']):
                    vis_target = {
                        'tid': tid, 'cid': cid,
                        'coords': coords_array, 'idxs': idxs, 'nnds': avg_k_nnds
                    }

    # --- 可视化逻辑 (Panel 2 展示的是同类细胞间的拓扑连线) ---
    if IS_FIRST_RUN and vis_target:
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        # Panel 1: 原图
        axes[0].imshow(img_np); axes[0].set_title("1. Original Multi-type Mask"); axes[0].axis('off')
        
        # Panel 2: 同类细胞 K-NN 网 (展示 Bug 2 修复后的效果)
        repaired_bg = id_to_rgb(full_tissue_ids)
        axes[1].imshow(repaired_bg)
        c_raw = vis_target['coords']
        i_raw = vis_target['idxs']
        # 画出同类细胞之间的连线
        for i in range(len(c_raw)):
            for n_idx in i_raw[i, 1:]:
                p1, p2 = c_raw[i], c_raw[n_idx]
                axes[1].plot([p1[1], p2[1]], [p1[0], p2[0]], color='white', alpha=0.4, linewidth=0.6)
        # 画出这些点
        axes[1].scatter(c_raw[:, 1], c_raw[:, 0], c='yellow', s=12, edgecolors='black', zorder=3)
        t_n, c_n = TISSUE_NAME_MAP.get(vis_target['tid'], 'N/A'), CELL_NAME_MAP.get(vis_target['cid'], 'N/A')
        axes[1].set_title(f"2. {c_n} patterns within {t_n} (k={k})", fontsize=14)
        axes[1].axis('off')

        # Panel 3: 分类 NND 直方图
        axes[2].hist(vis_target['nnds'], bins=25, color='orange', alpha=0.7, edgecolor='black')
        axes[2].axvline(np.mean(vis_target['nnds']), color='red', linestyle='--', label=f"Mean NND: {np.mean(vis_target['nnds']):.2f}")
        axes[2].set_title(f"3. {c_n} Homotypic NND Distribution", fontsize=14); axes[2].legend()

        plt.tight_layout()
        plt.savefig("spatial_stats_corrected_verify.png", dpi=200)
        print(f"\n[验证] 修复语义后的可视化已保存至: {os.path.abspath('spatial_stats_corrected_verify.png')}")
        IS_FIRST_RUN = False

    return batch_results

# --- 4. 主循环汇总 ---
def main(jsonl_path):
    # 全局字典：{ (tid, cid): [all_nnds] }
    global_stats = {}
    
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Processing Spatial Configuration"):
        try:
            data = json.loads(line)
            mask_path = data["conditioning_image"]
            if not mask_path.startswith('/'):
                mask_path = os.path.join(os.path.dirname(jsonl_path), mask_path)
            
            res = get_spatial_features(mask_path)
            for tid, cid, nnds in res:
                key = (tid, cid)
                if key not in global_stats: global_stats[key] = []
                global_stats[key].extend(nnds)
        except Exception as e:
            print(f"[WARN] Skipped: {e}")
            continue

    # 结果封装
    prior_db = {}
    for (tid, cid), all_vals in global_stats.items():
        if len(all_vals) < 50: continue # 过滤小样本
        
        t_name = TISSUE_NAME_MAP.get(tid, f"ID_{tid}")
        c_name = CELL_NAME_MAP.get(cid, f"ID_{cid}")
        
        if t_name not in prior_db: prior_db[t_name] = {}
        
        prior_db[t_name][c_name] = {
            "nnd_mean": round(float(np.mean(all_vals)), 4),
            "nnd_std": round(float(np.std(all_vals)), 4),
            "sample_count": len(all_vals)
        }

    with open("spatial_prior_knowledge_per_type.json", "w") as f:
        json.dump(prior_db, f, indent=4)
    print("\n[完成] 按(组织, 细胞)二元组统计的知识库已保存！")

if __name__ == "__main__":
    DATA_PATH = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/BCSS_dataset/metadata.jsonl"
    main(DATA_PATH)
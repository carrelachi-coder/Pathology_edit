import numpy as np
import cv2
import json
import os
from PIL import Image
from scipy import ndimage
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

COLOR_MAP = {
    # Tissue types (0-21)
    0:  [30, 30, 30],        # outside_roi
    1:  [180, 60, 60],       # tumor
    2:  [60, 150, 60],       # stroma
    3:  [140, 60, 180],      # lymphocytic_infiltrate
    4:  [60, 60, 180],       # necrosis_or_debris
    5:  [180, 180, 80],      # glandular_secretions
    6:  [160, 40, 40],       # blood
    7:  [40, 40, 40],        # exclude
    8:  [80, 150, 150],      # metaplasia_NOS
    9:  [200, 170, 100],     # fat
    10: [180, 120, 150],     # plasma_cells
    11: [120, 120, 190],     # other_immune_infiltrate
    12: [100, 190, 190],     # mucoid_material
    13: [200, 140, 60],      # normal_acinus_or_duct
    14: [140, 200, 100],     # lymphatics
    15: [140, 140, 140],     # undetermined
    16: [200, 200, 130],     # nerve
    17: [150, 80, 60],       # skin_adnexa
    18: [60, 140, 100],      # blood_vessel
    19: [190, 40, 40],       # angioinvasion
    20: [80, 60, 150],       # dcis
    21: [170, 170, 170],     # other
    # Nuclei types (101-105) - bright saturated
    101: [255, 0, 0],        # neoplastic
    102: [0, 255, 0],        # inflammatory
    103: [0, 80, 255],       # connective
    104: [255, 255, 0],      # dead
    105: [255, 0, 255],      # epithelial
}
TISSUE_NAME_MAP = {
    0: "outside_roi", 1: "tumor", 2: "stroma",
    3: "lymphocytic_infiltrate", 4: "necrosis_or_debris",
    5: "glandular_secretions", 6: "blood", 7: "exclude",
    8: "metaplasia_NOS", 9: "fat", 10: "plasma_cells",
    11: "other_immune_infiltrate", 12: "mucoid_material",
    13: "normal_acinus_or_duct", 14: "lymphatics",
    15: "undetermined", 16: "nerve", 17: "skin_adnexa",
    18: "blood_vessel", 19: "angioinvasion", 20: "dcis", 21: "other",
}

CELL_NAME_MAP = {
    
    101: "Neoplastic",
    102: "Inflammatory",
    103: "Connective",
    104: "Dead",
    105: "Epithelial"

}

TISSUE_IDS = list(range(22))
CELL_IDS = list(range(101, 106))

def rgb_to_id_mask(img_array, color_map):
    """精确将RGB图像映射为ID，未匹配像素标记为 -1"""
    h, w, _ = img_array.shape
   
    id_mask = np.full((h, w), -1, dtype=np.int16)
    
    for idx, color in color_map.items():
        match = np.all(img_array == color, axis=-1)
        id_mask[match] = idx
        
    return id_mask

def id_to_rgb(id_mask):
    """将ID Mask转换回RGB进行可视化"""
    h, w = id_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in COLOR_MAP.items():
        rgb[id_mask == idx] = color
    return rgb


def test_mask_processing(mask_path, output_path="test_result.png"):
    # --- A. 加载并转换图像 ---
    img = np.array(Image.open(mask_path).convert("RGB"))
    h, w, _ = img.shape
    id_mask = rgb_to_id_mask(img, COLOR_MAP)
    
   
   
    is_tissue = (id_mask >= 0) & (id_mask <= 21)
    
    # 2. 对“非组织区域”进行距离变换，找到最近的组织像素坐标
    # return_indices=True 会返回一个 (2, H, W) 的数组，包含了最近的源像素 y, x 坐标
    dist, indices = ndimage.distance_transform_edt(~is_tissue, return_indices=True)
    
    # 3. 使用这些坐标直接从原 ID 图中采样
    # 这一步保证了原本是细胞的地方，被它最近的组织 ID 替换
    full_tissue_ids = id_mask[indices[0], indices[1]]
    
    repaired_tissue_rgb = id_to_rgb(full_tissue_ids)

    # --- C. 细胞质心提取与叠加 (Centroid Overlay) ---
    centroid_vis = repaired_tissue_rgb.copy()
    
    for cid in CELL_IDS:
        cell_pixels = (id_mask == cid).astype(np.uint8)
        if not np.any(cell_pixels): continue
        
        
        labeled_cells, num_features = ndimage.label(cell_pixels)
   
        if num_features == 0: continue
        
        centroids = ndimage.center_of_mass(cell_pixels, labeled_cells, range(1, num_features + 1))
         
        # 在修复后的组织背景上画出细胞质心
        for cy, cx in centroids:
            # 画一个小圆点代表质心，颜色对应其细胞类型
            cv2.circle(centroid_vis, (int(cx), int(cy)), 3, COLOR_MAP[cid], -1)
            # 加一个小黑圈边框，让点更清晰
            cv2.circle(centroid_vis, (int(cx), int(cy)), 3, (0, 0, 0), 1)

    # --- D. 可视化摆放 ---
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Mask (Tissue + Nuclei)")
    plt.imshow(img)
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.title("Repaired Tissue Mask (Filled)")
    plt.imshow(repaired_tissue_rgb)
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.title("Centroids on Tissue Background")
    plt.imshow(centroid_vis)
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()
    print(f"测试完成，请查看: {output_path}")

# 使用你本地的一张图片路径运行
#test_mask_processing("/home/lyw/wqx-DL/flow-edit/FlowEdit-main/BCSS_dataset/conditioning/TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500_y256_x2304.png")


# 全局变量，用于控制仅在第一次运行脚本时进行可视化输出
IS_FIRST_RUN = True

def process_single_image(mask_path):
    global IS_FIRST_RUN
    
    # --- A. 加载与初始化 ---
    img_pil = Image.open(mask_path).convert("RGB")
    img = np.array(img_pil)
    h, w, _ = img.shape
    id_mask = rgb_to_id_mask(img, COLOR_MAP) # 假设该函数已定义
    
    # 初始化统计字典
    # TISSUE_IDS: 0-21, CELL_IDS: 101-105
    counts = {tid: {cid: 0 for cid in CELL_IDS} for tid in TISSUE_IDS}
    
    # --- B. 组织修复 (逻辑不改动，确保 full_tissue_ids 正确) ---
    is_tissue = (id_mask >= 0) & (id_mask <= 21)
    dist, indices = ndimage.distance_transform_edt(~is_tissue, return_indices=True)
    full_tissue_ids = id_mask[indices[0], indices[1]]
    
    # 用于可视化的底图
    if IS_FIRST_RUN:
        repaired_tissue_rgb = id_to_rgb(full_tissue_ids)
        centroid_vis = repaired_tissue_rgb.copy()

    # --- C. 细胞质心提取与统计 ---
    for cid in CELL_IDS:
        cell_pixels = (id_mask == cid).astype(np.uint8)
        if not np.any(cell_pixels): 
            continue
        
        # 连通域分析识别个体
        labeled_cells, num_features = ndimage.label(cell_pixels)
        if num_features == 0: 
            continue
        
        # 获取所有细胞个体的质心 (y, x)
        centroids = ndimage.center_of_mass(cell_pixels, labeled_cells, range(1, num_features + 1))
        
        # 如果只有一个细胞，center_of_mass 返回单对坐标，需统一为列表
        if num_features == 1:
            centroids = [centroids]
         
        for pt in centroids:
            # 再次确认解包安全性
            if not isinstance(pt, (tuple, list, np.ndarray)) or len(pt) != 2:
                continue
                
            cy, cx = pt
            if np.isnan(cy) or np.isnan(cx):
                continue
            
            # 坐标取整并增加越界保护 (防止 512.0 这种点)
            iy = int(np.clip(cy, 0, h - 1))
            ix = int(np.clip(cx, 0, w - 1))
            
            # 核心计数：查找该细胞中心点下方的组织 ID
            tid = full_tissue_ids[iy, ix]
            if tid in counts:
                counts[tid][cid] += 1
            
            if IS_FIRST_RUN:
                cv2.circle(centroid_vis, (ix, iy), 3, COLOR_MAP[cid], -1)
                cv2.circle(centroid_vis, (ix, iy), 3, (0, 0, 0), 1)

    # --- D. 第一次运行的可视化输出 ---
    if IS_FIRST_RUN:
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.title("Original Mask")
        plt.imshow(img)
        plt.axis("off")
        
        plt.subplot(1, 3, 2)
        plt.title("Repaired Tissue Mask")
        plt.imshow(id_to_rgb(full_tissue_ids))
        plt.axis("off")
        
        plt.subplot(1, 3, 3)
        plt.title("Extracted Centroids on Tissue")
        plt.imshow(centroid_vis)
        plt.axis("off")
        
        plt.tight_layout()
        save_path = "first_run_verification.png"
        plt.savefig(save_path, dpi=200)
        print(f"\n[验证] 第一次运行可视化已保存至: {os.path.abspath(save_path)}")
        
        # 设为 False，后续循环不再运行可视化
        IS_FIRST_RUN = False

# 在 process_single_image 的返回中增加面积信息
    tissue_areas = {}
    for tid in TISSUE_IDS:
        tissue_areas[tid] = int((full_tissue_ids == tid).sum())

    return counts, tissue_areas

def main(jsonl_path):
    raw_rows = []
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
        
    for line in tqdm(lines, desc="Analyzing microenvironment"):
        try:
            data = json.loads(line)
            mask_path = data["conditioning_image"]
            if not mask_path.startswith('/'):
                mask_path = os.path.join(os.path.dirname(jsonl_path), mask_path)
            
            img_counts, img_areas = process_single_image(mask_path)
            if img_counts is None: continue

            for tid in TISSUE_IDS:
                area = img_areas.get(tid, 0)
                if area > 0:
                    total_cells = sum(img_counts[tid].values())
                    row = {
                        'tissue_type': tid,
                        'density': total_cells / area, # 原始比例 (cells/px)
                        'total_cells': total_cells,
                        'area': area
                    }
                    for cid in CELL_IDS:
                        # 存每一类细胞在该样本中的绝对密度和占比
                        count = img_counts[tid][cid]
                        row[f'prop_{cid}'] = count / total_cells if total_cells > 0 else 0.0
                        row[f'abs_dens_{cid}'] = count / area
                    raw_rows.append(row)
        except Exception: continue

        # 转为原始 DataFrame 并备份
    df_raw = pd.DataFrame(raw_rows)
    df_raw.to_csv("raw_sample_stats.csv", index=False)

    # --- 构造字典结构 (Prior DB) ---
    prior_db = {}
    # --- 构造汇总表格 (Human Readable) ---
    summary_list = []

    for tid in TISSUE_IDS:
        group = df_raw[df_raw['tissue_type'] == tid]
        if group.empty: continue
        
        t_name = TISSUE_NAME_MAP.get(tid, f"id_{tid}")
        
        # A. 计算 Cell Type Dist: {name: [mean, std]}
        dist_dict = {}
        for cid, c_name in CELL_NAME_MAP.items():
            dist_dict[c_name] = [
                round(float(group[f'prop_{cid}'].mean()), 4),
                round(float(group[f'prop_{cid}'].std()), 4)
            ]

        # B. 计算 Density Stats (转为 cells/1e4px 以便人类阅读)
        dens_px = group['density']
        density_stats = {
            "mean": round(float(dens_px.mean()), 6),
            "std":  round(float(dens_px.std()), 6),
            "min":  round(float(dens_px.min()), 6),
            "max":  round(float(dens_px.max()), 6)
        }

        # C. 计算 Cell Density Per Type (严谨计算 Std)
        cell_dens_dict = {}
        for cid, c_name in CELL_NAME_MAP.items():
            abs_dens_col = group[f'abs_dens_{cid}']
            cell_dens_dict[c_name] = [
                round(float(abs_dens_col.mean()), 6),
                round(float(abs_dens_col.std()), 6)
            ]

        # 写入 JSON 结构
        prior_db[t_name] = {
            "cell_type_dist": dist_dict,
            "density": density_stats,
            "cell_density_per_type": cell_dens_dict
        }

        # 写入 CSV 结构 (转为 1e4px 直观数值)
        csv_row = {
            "Tissue_ID": tid,
            "Tissue_Name": t_name,
            "Density (cells/1e4px)": f"{density_stats['mean']*10000:.4f}±{density_stats['std']*10000:.4f}"
        }
        for cid, c_name in CELL_NAME_MAP.items():
            m = dist_dict[c_name][0]
            s = dist_dict[c_name][1]
            csv_row[f"{cid}_{c_name}"] = f"{m:.3f}±{s:.3f}"
        summary_list.append(csv_row)

    # 保存 JSON
    with open("pathology_prior_db.json", 'w', encoding='utf-8') as f:
        json.dump(prior_db, f, indent=4, ensure_ascii=False)
    
    # 保存 CSV
    df_summary = pd.DataFrame(summary_list)
    df_summary.to_csv("cell_distribution_matrix.csv", index=False)

    return df_summary

# ========== 4. 执行 ==========
if __name__ == "__main__":
    PATH = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/BCSS_dataset/metadata.jsonl"
    res = main(PATH)
    print("\n[Done] 统计完成！")
    print(res.to_string())
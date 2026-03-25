import numpy as np
import json
import os
import pandas as pd
from PIL import Image
from scipy import ndimage
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 配置 ---

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

def rgb_to_id_mask(img_array):
    h, w, _ = img_array.shape
    id_mask = np.full((h, w), -1, dtype=np.int16)
    for idx, color in COLOR_MAP.items():
        id_mask[np.all(img_array == color, axis=-1)] = idx
    return id_mask

def repair_tissue_layer(id_mask):
    """去除细胞干扰，获取纯组织分布"""
    is_tissue = (id_mask >= 0) & (id_mask <= 21)
    if not np.any(is_tissue): return id_mask
    _, indices = ndimage.distance_transform_edt(~is_tissue, return_indices=True)
    return id_mask[indices[0], indices[1]]

# --- 2. 主统计函数 ---
def main(jsonl_path):
    all_patch_ratios = []
    
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
        
    for line in tqdm(lines, desc="统计组织面积比例"):
        try:
            data = json.loads(line)
            mask_path = data["conditioning_image"]
            if not mask_path.startswith('/'):
                mask_path = os.path.join(os.path.dirname(jsonl_path), mask_path)
            
            img_np = np.array(Image.open(mask_path).convert("RGB"))
            id_mask = rgb_to_id_mask(img_np)
            full_tissue_ids = repair_tissue_layer(id_mask)
            
            # 计算每种组织的像素数
            unique, counts = np.unique(full_tissue_ids, return_counts=True)
            counts_dict = dict(zip(unique, counts))
            
            # 计算占比 (忽略 ID -1)
            total_pixels = full_tissue_ids.size
            patch_stat = {tid: 0.0 for tid in TISSUE_IDS}
            for tid, count in counts_dict.items():
                if 0 <= tid <= 21:
                    patch_stat[tid] = count / total_pixels
            
            all_patch_ratios.append(patch_stat)
        except Exception as e: print(f"[WARN] {e}"); continue

    # --- 3. 数据分析 ---
    df = pd.DataFrame(all_patch_ratios)
    df.columns = [TISSUE_NAME_MAP.get(c, f"ID_{c}") for c in df.columns]
    
    occurrence_threshold = 0.1
    frequent_cols = [c for c in df.columns if (df[c] > 0).mean() > occurrence_threshold]
    
    # 如果符合条件的组织太少（比如只剩下 tumor 和 stroma），可以适当降低阈值到 0.05
    if len(frequent_cols) < 3:
        occurrence_threshold = 0.05
        frequent_cols = [c for c in df.columns if (df[c] > 0).mean() > occurrence_threshold]

    # 【优化 1.2】 使用 Spearman 秩相关
    # 相比 Pearson，Spearman 对非正态分布和零值较多的数据更稳健
    correlation_matrix = df[frequent_cols].corr(method='spearman')

    # A. 计算基础统计量 (min, max, mean, std)
    area_constraints = {}
    for col in df.columns:
        # 只针对在数据集中出现过的组织
        if df[col].max() > 0:
            area_constraints[col] = {
                "mean": round(float(df[col].mean()), 4),
                "std":  round(float(df[col].std()), 4),
                "occurrence_rate": round(float((df[col] > 0).mean()), 4),
                "max_observed": round(float(df[col].max()), 4)
            }

    area_constraints["_tissue_cooccurrence"] = correlation_matrix.to_dict()

    
    # C. 专门分析 Tumor-Stroma 比例 (临床金标准)
    if 'tumor' in df.columns and 'stroma' in df.columns:
        # 只取两者共存的 Patch
        ts_df = df[(df['tumor'] > 0.05) & (df['stroma'] > 0.05)]
        ts_ratio = (ts_df['stroma'] / ts_df['tumor']).clip(upper=20)  # 防极端值
        area_constraints["_pathology_rules"] = {
            "stroma_to_tumor_ratio": {
                "mean": round(float(ts_ratio.mean()), 4),
                "median": round(float(ts_ratio.median()), 4),  # 中位数更稳健
                "std": round(float(ts_ratio.std()), 4),
            }
        }

    # --- 4. 保存结果 ---
    # 保存统计 JSON
    with open("tissue_area_prior.json", "w") as f:
        json.dump(area_constraints, f, indent=4)
        
    # 保存共现相关性热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap="vlag", center=0)
    plt.title("Tissue Area Co-occurrence Correlation (Pearson)")
    plt.tight_layout()
    plt.savefig("tissue_cooccurrence_heatmap.png", dpi=200)

    print("\n[完成] 组织比例约束已保存至 tissue_area_prior.json")
    print("[完成] 组织共现热力图已保存至 tissue_cooccurrence_heatmap.png")

if __name__ == "__main__":
    DATA_PATH = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/BCSS_dataset/metadata.jsonl"
    main(DATA_PATH)
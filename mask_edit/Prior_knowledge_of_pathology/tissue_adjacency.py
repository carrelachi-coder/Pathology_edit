import numpy as np
import cv2
import json
import os
import pandas as pd
from PIL import Image
from scipy import ndimage
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

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
IS_FIRST_RUN = True

# --- 2. 工具函数 ---
def rgb_to_id_mask(img_array):
    h, w, _ = img_array.shape
    id_mask = np.full((h, w), -1, dtype=np.int16)
    for idx, color in COLOR_MAP.items():
        id_mask[np.all(img_array == color, axis=-1)] = idx
    return id_mask

def repair_tissue_layer(id_mask):
    """使用EDT修复组织层（去除细胞干扰）"""
    is_tissue = (id_mask >= 0) & (id_mask <= 21)
    # 如果全图没有组织（只有细胞或背景），返回原图
    if not np.any(is_tissue): return id_mask
    _, indices = ndimage.distance_transform_edt(~is_tissue, return_indices=True)
    return id_mask[indices[0], indices[1]]

# --- 3. 核心统计：偏移比对法 ---
def count_adjacency(full_tissue_ids):
    """
    通过图像偏移计算相邻像素的类别对。
    返回一个 (22, 22) 的计数矩阵。
    """
    adj_matrix = np.zeros((22, 22), dtype=np.int64)
    
    # 获取上下左右四个方向的偏移图像
    # 我们只关心 ID 0-21 之间的转换
    mask = full_tissue_ids.astype(np.int32)
    
    # 垂直边界 (H-1, W)
    v_edges_mask = (mask[:-1, :] != mask[1:, :])
    v_pairs_i = mask[:-1, :][v_edges_mask]
    v_pairs_j = mask[1:, :][v_edges_mask]
    
    # 水平边界 (H, W-1)
    h_edges_mask = (mask[:, :-1] != mask[:, 1:])
    h_pairs_i = mask[:, :-1][h_edges_mask]
    h_pairs_j = mask[:, 1:][h_edges_mask]
    
    # 合并所有相邻对
    pairs_i = np.concatenate([v_pairs_i, h_pairs_i])
    pairs_j = np.concatenate([v_pairs_j, h_pairs_j])
    
    # 填充对称矩阵（i与j相邻 等同于 j与i相邻）
    for i, j in zip(pairs_i, pairs_j):
        if 0 <= i < 22 and 0 <= j < 22:
            adj_matrix[i, j] += 1
            adj_matrix[j, i] += 1
            
    return adj_matrix

# --- 4. 可视化函数 ---
def visualize_adjacency_logic(original_img, full_tissue_ids, adj_matrix_normalized):
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    # Panel 1: 修复后的组织图
    h, w = full_tissue_ids.shape
    bg_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in COLOR_MAP.items():
        bg_rgb[full_tissue_ids == idx] = color
    axes[0].imshow(bg_rgb)
    axes[0].set_title("1. Clean Tissue Background (Repaired)", fontsize=14)
    axes[0].axis('off')
    
    # Panel 2: 提取出的边界线（可视化哪些地方在贡献统计）
    edges = cv2.Canny(bg_rgb, 10, 100)
    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title("2. Extracted Tissue Boundaries", fontsize=14)
    axes[1].axis('off')
    
    # Panel 3: 邻接热力图
    # 选出出现频率最高的前10种组织显示，否则矩阵太大看不清
    labels = [TISSUE_NAME_MAP.get(i, f"ID_{i}") for i in TISSUE_IDS]
    sns.heatmap(adj_matrix_normalized, annot=True, fmt=".2f", cmap="YlGnBu", 
                xticklabels=labels, yticklabels=labels, ax=axes[2], cbar=False)
    axes[2].set_title("3. Adjacency Probability Matrix A(i, j)", fontsize=14)
    
    plt.tight_layout()
    plt.savefig("tissue_adjacency_verify.png", dpi=200)
    print(f"\n[验证] 邻接关系可视化已保存至: {os.path.abspath('tissue_adjacency_verify.png')}")

# --- 5. 主程序 ---
def main(jsonl_path):
    global IS_FIRST_RUN
    global_adj_counts = np.zeros((22, 22), dtype=np.int64)
    
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
        
    for line in tqdm(lines, desc="Analyzing Tissue Adjacency"):
        try:
            data = json.loads(line)
            mask_path = data["conditioning_image"]
            if not mask_path.startswith('/'):
                mask_path = os.path.join(os.path.dirname(jsonl_path), mask_path)
            
            img_np = np.array(Image.open(mask_path).convert("RGB"))
            id_mask = rgb_to_id_mask(img_np)
            full_tissue_ids = repair_tissue_layer(id_mask)
            
            # 统计单张图的边界长度
            patch_adj = count_adjacency(full_tissue_ids)
            global_adj_counts += patch_adj
            
            # 第一次运行可视化
            if IS_FIRST_RUN:
                # 先做个归一化用于显示
                row_sums = patch_adj.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                vis_matrix = patch_adj / row_sums
                visualize_adjacency_logic(img_np, full_tissue_ids, vis_matrix)
                IS_FIRST_RUN = False
                
        except Exception as e:
            continue

    # --- 最终归一化 ---
    # A(i, j) = 组织 i 的所有边界中，有多少比例是给了 组织 j
    row_sums = global_adj_counts.sum(axis=1, keepdims=True)
    # 防止除以0
    row_sums[row_sums == 0] = 1
    adj_probability_matrix = global_adj_counts / row_sums

    # 保存结果
    # 1. 保存为 CSV
    df_labels = [TISSUE_NAME_MAP.get(i, f"ID_{i}") for i in TISSUE_IDS]
    df = pd.DataFrame(adj_probability_matrix, index=df_labels, columns=df_labels)
    df.to_csv("tissue_adjacency_matrix.csv")
    
    # 2. 保存为 JSON 字典结构
    adj_dict = {}
    for i in TISSUE_IDS:
        i_name = TISSUE_NAME_MAP.get(i, f"ID_{i}")
        adj_dict[i_name] = {}
        for j in TISSUE_IDS:
            j_name = TISSUE_NAME_MAP.get(j, f"ID_{j}")
            prob = adj_probability_matrix[i, j]
            if prob > 0.001: # 只记录有意义的邻接
                adj_dict[i_name][j_name] = round(float(prob), 4)

    with open("tissue_adjacency_prior.json", "w") as f:
        json.dump(adj_dict, f, indent=4)

    print("\n[完成] 组织邻接矩阵已保存！")
    print(f"核心发现示例：肿瘤(Tumor)最常邻接的组织是: {df.loc['tumor'].idxmax()} ({df.loc['tumor'].max():.2%})")

if __name__ == "__main__":
    DATA_PATH = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/BCSS_dataset/metadata.jsonl"
    main(DATA_PATH)
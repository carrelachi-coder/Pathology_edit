"""
generate_mask_pairs.py
=====================
完整 Pipeline：组织编辑 → 细胞保留 → 概率图推理 → 细胞填充

支持通过命令行选择编辑操作:
  --ops tumor_dilate              # 只跑肿瘤膨胀
  --ops tumor_shrink              # 只跑肿瘤收缩
  --ops lymph_dilate              # 只跑淋巴扩散
  --ops necrosis_replace          # 只跑坏死替换
  --ops necrosis_fibrosis         # 只跑坏死纤维化
  --ops tumor_dilate tumor_shrink # 跑多个
  --ops all                       # 跑全部 (默认)

其他参数:
  --mask <path>                   # 指定 mask 文件
  --output <dir>                  # 输出目录
  --n-variants <N>                # 每种操作生成几个变体
  --no-probnet                    # 不用 ProbNet，纯实例库 fallback
  --device <cuda:X>               # GPU

用法:
  cd mask_data_generate

  # 只跑 necrosis_fibrosis
  python generate_mask_pairs.py --ops necrosis_fibrosis

  # 跑 tumor_dilate + tumor_shrink 对比
  python generate_mask_pairs.py --ops tumor_dilate tumor_shrink

  # 指定 mask 文件
  python generate_mask_pairs.py --ops tumor_shrink --mask /path/to/mask.png

  # 全部操作
  python generate_mask_pairs.py --ops all
"""

import numpy as np
import os
import sys
import glob
import random
import logging
import argparse
from PIL import Image
from scipy import ndimage
from collections import Counter
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

# ============================================================
#  导入编辑器和验证器
# ============================================================
from mask_validator import MaskValidator
from boundary_deform import TumorBoundaryTransform
from lymphocyte_infiltration import LymphocyteInfiltrationTransform
from tumor_to_necrosis import NecrosisReplacementTransform
from tumor_shrink import TumorShrinkTransform
from necrosis_fibrosis import NecrosisFibrosisTransform
from stromal_fibrosis import StromalFibrosisTransform

# ============================================================
#  导入概率图网络和细胞生成器
# ============================================================
INPAINT_CELLS_DIR = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/inpaint_cells/DDPM+Cell_inpaint"
sys.path.insert(0, INPAINT_CELLS_DIR)

from train_prob_net import ProbUNet, NUM_TISSUE, NUM_NUCLEI, NUCLEI_CLASSES
from generate_nuclei import NucleiLibrary, poisson_disk_sampling, place_nucleus

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
#  颜色映射 & 常量
# ============================================================
COLOR_MAP = {
    0: [30, 30, 30],    1: [180, 60, 60],   2: [60, 150, 60],   3: [140, 60, 180],
    4: [60, 60, 180],   5: [180, 180, 80],  6: [160, 40, 40],   7: [40, 40, 40],
    8: [80, 150, 150],  9: [200, 170, 100], 10: [180, 120, 150],11: [120, 120, 190],
    12: [100, 190, 190],13: [200, 140, 60], 14: [140, 200, 100],15: [140, 140, 140],
    16: [200, 200, 130],17: [150, 80, 60],  18: [60, 140, 100], 19: [190, 40, 40],
    20: [80, 60, 150],  21: [170, 170, 170],
    101: [255, 0, 0],   102: [0, 255, 0],   103: [0, 80, 255],
    104: [255, 255, 0], 105: [255, 0, 255],
}

TISSUE_IDS = set(range(0, 22))
CELL_IDS   = {101, 102, 103, 104, 105}

# 所有可用操作名
ALL_OPS = ['tumor_dilate', 'tumor_shrink', 'lymph_dilate', 'necrosis_replace', 'necrosis_fibrosis', 'stromal_fibrosis']


# ============================================================
#  工具函数 (和原版一致)
# ============================================================

def rgb_to_id(img_rgb):
    h, w, _ = img_rgb.shape
    id_mask = np.full((h, w), -1, dtype=np.int16)
    for idx, color in COLOR_MAP.items():
        match = np.all(img_rgb == np.array(color, dtype=np.uint8), axis=-1)
        id_mask[match] = idx
    return id_mask


def id_to_rgb(id_mask, include_cells=True):
    h, w = id_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for tid, color in COLOR_MAP.items():
        if not include_cells and tid in CELL_IDS:
            continue
        rgb[id_mask == tid] = color
    return rgb


def extract_tissue_mask(id_mask):
    is_tissue = np.isin(id_mask, list(TISSUE_IDS))
    if not np.any(is_tissue):
        return id_mask.copy()
    _, nearest_idx = ndimage.distance_transform_edt(~is_tissue, return_indices=True)
    return id_mask[nearest_idx[0], nearest_idx[1]]


def extract_cell_instances(id_mask):
    cells = []
    cell_mask = np.isin(id_mask, list(CELL_IDS))
    if not np.any(cell_mask):
        return cells
    labeled, n_cells = ndimage.label(cell_mask)
    for i in range(1, n_cells + 1):
        pixels = np.argwhere(labeled == i)
        if len(pixels) == 0:
            continue
        class_ids = id_mask[pixels[:, 0], pixels[:, 1]]
        class_id = int(np.bincount(class_ids[class_ids > 0]).argmax())
        centroid = pixels.mean(axis=0)
        rmin, cmin = pixels.min(axis=0)
        rmax, cmax = pixels.max(axis=0)
        cells.append({
            'class_id': class_id, 'pixels': pixels,
            'centroid': centroid,
            'bbox': (int(rmin), int(rmax), int(cmin), int(cmax)),
        })
    return cells


# ============================================================
#  mask 编辑后的细胞处理 (和原版一致)
# ============================================================

def get_unchanged_tissue_mask(tissue_before, tissue_after):
    return tissue_before == tissue_after


def filter_retained_cells(cells, unchanged_mask):
    retained, removed = [], []
    h, w = unchanged_mask.shape
    for cell in cells:
        cr = int(np.clip(round(cell['centroid'][0]), 0, h - 1))
        cc = int(np.clip(round(cell['centroid'][1]), 0, w - 1))
        if unchanged_mask[cr, cc]:
            retained.append(cell)
        else:
            removed.append(cell)
    return retained, removed


def build_retained_cell_bool_mask(retained_cells, shape):
    mask = np.zeros(shape, dtype=bool)
    for cell in retained_cells:
        mask[cell['pixels'][:, 0], cell['pixels'][:, 1]] = True
    return mask


def compose_edited_mask(edited_tissue, retained_cells):
    result = edited_tissue.copy()
    for cell in retained_cells:
        result[cell['pixels'][:, 0], cell['pixels'][:, 1]] = cell['class_id']
    return result


def build_change_region_mask(tissue_before, tissue_after, retained_cells):
    h, w = tissue_before.shape
    changed = tissue_before != tissue_after
    retained_coverage = build_retained_cell_bool_mask(retained_cells, (h, w))
    need_generation = changed & (~retained_coverage)
    change_mask_id = np.full((h, w), -1, dtype=np.int16)
    change_mask_id[need_generation] = tissue_after[need_generation]
    return change_mask_id, need_generation


def change_region_to_rgb(change_mask_id):
    h, w = change_mask_id.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for tid, color in COLOR_MAP.items():
        if 0 <= tid <= 21:
            rgb[change_mask_id == tid] = color
    return rgb


# ============================================================
#  概率图推理 + 细胞生成 (和原版一致)
# ============================================================

def to_onehot_tensor(index_map, num_classes):
    oh = np.zeros((num_classes, index_map.shape[0], index_map.shape[1]), dtype=np.float32)
    for c in range(num_classes):
        oh[c] = (index_map == c).astype(np.float32)
    return torch.from_numpy(oh).unsqueeze(0)


def tissue_id_to_nuclei_input(edited_combined, change_mask_bool):
    h, w = edited_combined.shape
    nuclei_map = np.zeros((h, w), dtype=np.int64)
    for i, nuc_val in enumerate(NUCLEI_CLASSES):
        nuclei_map[edited_combined == nuc_val] = i + 1
    nuclei_map[change_mask_bool] = 0
    return nuclei_map


@torch.no_grad()
def predict_prob_map(model, tissue_map, nuclei_input_map, change_mask_bool, device):
    H, W = tissue_map.shape
    tissue_oh = to_onehot_tensor(np.clip(tissue_map, 0, NUM_TISSUE - 1), NUM_TISSUE).to(device)
    nuclei_oh = to_onehot_tensor(nuclei_input_map, NUM_NUCLEI).to(device)
    mask_t = torch.from_numpy(change_mask_bool.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    if pad_h > 0 or pad_w > 0:
        tissue_oh = F.pad(tissue_oh, (0, pad_w, 0, pad_h), mode='reflect')
        nuclei_oh = F.pad(nuclei_oh, (0, pad_w, 0, pad_h), mode='reflect')
        mask_t = F.pad(mask_t, (0, pad_w, 0, pad_h), mode='constant', value=0)

    logits = model(tissue_oh, nuclei_oh, mask_t)
    prob = F.softmax(logits, dim=1)[0].cpu().numpy()
    return prob[:, :H, :W]


def generate_cells_from_prob(prob_map, tissue_map, change_mask_bool, library,
                            type_bias_map=None, density_scale_map=None):
    H, W = tissue_map.shape
    output_map = tissue_map.astype(np.int64).copy()
    total_placed = 0

    # type_bias_map name → index mapping
    _BIAS_NAME_TO_IDX = {
        "neoplastic": 0, "inflammatory": 1,
        "connective": 2, "dead": 3, "epithelial": 4,
    }

    tissue_types = np.unique(tissue_map[change_mask_bool])
    tissue_types = tissue_types[(tissue_types >= 0) & (tissue_types <= 21)]

    for tissue_id in tissue_types:
        tissue_id = int(tissue_id)
        tissue_region = change_mask_bool & (tissue_map == tissue_id)
        region_area = tissue_region.sum()
        if region_area < 50:
            continue

        nuc_prob = 1.0 - prob_map[0]
        avg_nuc_prob = nuc_prob[tissue_region].mean()
        num_nuclei = int(avg_nuc_prob * region_area / 80)
        num_nuclei = max(0, int(num_nuclei * random.uniform(0.8, 1.2)))
        if num_nuclei == 0:
            continue

        stats = library.stats.get(str(tissue_id), {})
        mean_areas = [info['mean_area'] for info in stats.get('nuclei_types', {}).values()
                     if info.get('mean_area', 0) > 0]
        avg_area = np.mean(mean_areas) if mean_areas else 100
        min_distance = max(np.sqrt(avg_area / np.pi) * 3, 10)

        # 应用密度缩放
        if density_scale_map and tissue_id in density_scale_map:
            min_distance *= density_scale_map[tissue_id]
            min_distance = max(min_distance, 5)

        centers = poisson_disk_sampling(tissue_region, min_distance)
        if len(centers) > num_nuclei:
            random.shuffle(centers)
            centers = centers[:num_nuclei]

        placed = 0
        for cy, cx in centers:
            type_probs = prob_map[1:, cy, cx].copy()
            if type_probs.sum() < 0.05:
                continue
            # 应用 type_bias_map
            if type_bias_map and tissue_id in type_bias_map:
                for tname, mult in type_bias_map[tissue_id].items():
                    if tname in _BIAS_NAME_TO_IDX:
                        type_probs[_BIAS_NAME_TO_IDX[tname]] *= mult
            type_probs = type_probs / type_probs.sum()
            nuc_type_idx = np.random.choice(5, p=type_probs)
            nuc_type = NUCLEI_CLASSES[nuc_type_idx]
            instance = library.sample_instance(tissue_id, nuc_type)
            if instance is None:
                continue
            if place_nucleus(output_map, cy, cx, instance, augment=True):
                placed += 1

        total_placed += placed
        logger.info(f"    组织 {tissue_id}: 区域 {region_area} px, "
                    f"采样 {len(centers)} 点, 放置 {placed} 核")

    return output_map, total_placed


def generate_cells_from_library_only(tissue_map, change_mask_bool, library,
                                    type_bias_map=None, density_scale_map=None):
    H, W = tissue_map.shape
    output_map = tissue_map.astype(np.int64).copy()
    total_placed = 0

    # type_bias_map name → nuclei class value mapping
    _BIAS_TYPE_TO_VAL = {
        "neoplastic": 101, "inflammatory": 102,
        "connective": 103, "dead": 104, "epithelial": 105,
    }

    tissue_types = np.unique(tissue_map[change_mask_bool])
    tissue_types = tissue_types[(tissue_types >= 0) & (tissue_types <= 21)]

    for tissue_id in tissue_types:
        tissue_id = int(tissue_id)
        tissue_region = change_mask_bool & (tissue_map == tissue_id)
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
        mean_areas = [info['mean_area'] for info in stats.get('nuclei_types', {}).values()
                     if info.get('mean_area', 0) > 0]
        avg_area = np.mean(mean_areas) if mean_areas else 100
        avg_diameter = np.sqrt(avg_area / np.pi) * 2
        min_distance = max(avg_diameter * 1.5, 8)

        # 应用密度缩放
        if density_scale_map and tissue_id in density_scale_map:
            min_distance *= density_scale_map[tissue_id]
            min_distance = max(min_distance, 5)

        centers = poisson_disk_sampling(tissue_region, min_distance)
        if len(centers) > num_nuclei:
            random.shuffle(centers)
            centers = centers[:num_nuclei]

        nuc_types_list = []
        # 应用 type_bias_map 到类型分布
        adjusted_dist = dict(type_dist)
        if type_bias_map and tissue_id in type_bias_map:
            for tname, mult in type_bias_map[tissue_id].items():
                val = _BIAS_TYPE_TO_VAL.get(tname)
                if val is not None and val in adjusted_dist:
                    adjusted_dist[val] *= mult
            # 归一化
            total_frac = sum(adjusted_dist.values())
            if total_frac > 0:
                adjusted_dist = {k: v / total_frac for k, v in adjusted_dist.items()}
        for nuc_type, frac in adjusted_dist.items():
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
        logger.info(f"    组织 {tissue_id}: 区域 {region_area} px, "
                    f"采样 {len(centers)} 点, 放置 {placed} 核")

    return output_map, total_placed


# ============================================================
#  单种编辑方式的完整处理流程 (和原版一致)
# ============================================================

def process_single_edit(original_id_mask, pure_tissue_before, cells,
                        transform, method_name,
                        prob_model, library, device,
                        n_variants=1, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  处理编辑方式: {method_name}")
    print(f"{'='*60}")

    print(f"  [1/5] 生成组织编辑变体...")
    results = transform.generate_variants(pure_tissue_before, n_variants=n_variants)

    if not results:
        print(f"  [错误] {method_name}: 未能生成任何变体，跳过。")
        return None

    edited_tissue, edit_log = results[0]
    print(f"  编辑日志: area_change={edit_log.area_change}")

    print(f"  [2/5] 筛选保留细胞...")
    unchanged_mask = get_unchanged_tissue_mask(pure_tissue_before, edited_tissue)
    retained_cells, removed_cells = filter_retained_cells(cells, unchanged_mask)
    print(f"  原始细胞: {len(cells)}, 保留: {len(retained_cells)}, 移除: {len(removed_cells)}")

    print(f"  [3/5] 合成编辑后 combined mask...")
    edited_combined = compose_edited_mask(edited_tissue, retained_cells)

    print(f"  [4/5] 生成变化区域 mask...")
    change_mask_id, change_mask_bool = build_change_region_mask(
        pure_tissue_before, edited_tissue, retained_cells
    )

    changed_total = int(np.sum(pure_tissue_before != edited_tissue))
    retained_overlap = int(np.sum(
        build_retained_cell_bool_mask(retained_cells, pure_tissue_before.shape)
        & (pure_tissue_before != edited_tissue)
    ))
    need_gen = int(np.sum(change_mask_bool))
    print(f"  变化像素: {changed_total}, 被保留细胞覆盖: {retained_overlap}, "
          f"最终需生成: {need_gen}")

    print(f"  [5/5] 在变化区域内生成细胞核...")

    if prob_model is not None and need_gen > 0:
        print(f"    使用 ProbUNet 预测核类型概率图...")
        nuclei_input_map = tissue_id_to_nuclei_input(edited_combined, change_mask_bool)
        prob_map = predict_prob_map(
            prob_model, edited_tissue.astype(np.int64),
            nuclei_input_map, change_mask_bool, device
        )
        gen_map, n_placed = generate_cells_from_prob(
            prob_map, edited_tissue.astype(np.int64), change_mask_bool, library
        )
    elif library is not None and need_gen > 0:
        print(f"    ProbUNet 不可用，使用实例库 fallback...")
        gen_map, n_placed = generate_cells_from_library_only(
            edited_tissue.astype(np.int64), change_mask_bool, library
        )
    else:
        gen_map = edited_tissue.astype(np.int64).copy()
        n_placed = 0

    print(f"  共放置 {n_placed} 个新细胞核")

    final_combined = gen_map.copy()
    for cell in retained_cells:
        final_combined[cell['pixels'][:, 0], cell['pixels'][:, 1]] = cell['class_id']

    # --- 保存 ---
    original_rgb = id_to_rgb(original_id_mask, include_cells=True)
    edited_rgb = id_to_rgb(edited_combined, include_cells=True)
    change_rgb = change_region_to_rgb(change_mask_id)
    final_rgb = id_to_rgb(final_combined.astype(np.int16), include_cells=True)

    Image.fromarray(original_rgb).save(os.path.join(output_dir, "original_with_cells.png"))
    Image.fromarray(edited_rgb).save(os.path.join(output_dir, "edited_with_retained_cells.png"))
    Image.fromarray(change_rgb).save(os.path.join(output_dir, "change_region_mask.png"))
    Image.fromarray(final_rgb).save(os.path.join(output_dir, "final_with_generated_cells.png"))

    np.save(os.path.join(output_dir, "edited_tissue.npy"), edited_tissue)
    np.save(os.path.join(output_dir, "final_combined.npy"), final_combined)
    np.save(os.path.join(output_dir, "change_region.npy"), change_mask_id)

    # --- 可视化 ---
    fig, axes = plt.subplots(1, 6, figsize=(30, 6))
    titles = [
        "Original\n(Combined)",
        "Pure Tissue\n(Before)",
        "Edited Tissue\n(After)",
        "Edited +\nRetained Cells",
        "Change Region\n(excl. retained)",
        f"Final Result\n({n_placed} new cells)",
    ]
    images = [
        original_rgb,
        id_to_rgb(pure_tissue_before, include_cells=False),
        id_to_rgb(edited_tissue, include_cells=False),
        edited_rgb,
        change_rgb,
        final_rgb,
    ]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.suptitle(f"Edit Method: {method_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    vis_path = os.path.join(output_dir, "visualization.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  可视化已保存: {vis_path}")

    return {
        'edited_tissue':    edited_tissue,
        'edited_combined':  edited_combined,
        'final_combined':   final_combined,
        'change_mask_id':   change_mask_id,
        'change_mask_bool': change_mask_bool,
        'retained_cells':   retained_cells,
        'removed_cells':    removed_cells,
        'n_generated':      n_placed,
        'edit_log':         edit_log,
    }


# ============================================================
#  加载模型
# ============================================================

def load_prob_model(ckpt_path, device):
    if ckpt_path is None or not os.path.exists(ckpt_path):
        logger.warning(f"ProbUNet checkpoint 不存在: {ckpt_path}，将使用实例库 fallback")
        return None

    model = ProbUNet(
        in_ch=NUM_TISSUE + NUM_NUCLEI + 1,
        out_ch=NUM_NUCLEI,
        base_ch=64,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()

    val_loss = ckpt.get('val_loss', 'N/A')
    logger.info(f"已加载 ProbUNet: {ckpt_path} (val_loss={val_loss})")
    return model


# ============================================================
#  主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Mask编辑 + 细胞填充 Pipeline")
    parser.add_argument("--ops", nargs="+", default=["all"],
                        choices=ALL_OPS + ["all"],
                        help=f"选择编辑操作: {ALL_OPS} 或 all")
    parser.add_argument("--mask", type=str, default=None,
                        help="指定 mask 文件路径 (默认用数据集第一张)")
    parser.add_argument("--output", type=str, default="./mask_edit_output",
                        help="输出目录")
    parser.add_argument("--n-variants", type=int, default=1,
                        help="每种操作生成几个变体")
    parser.add_argument("--no-probnet", action="store_true",
                        help="不用 ProbNet, 纯实例库 fallback")
    parser.add_argument("--device", type=str, default="cuda",
                        help="GPU 设备")
    args = parser.parse_args()

    # 解析要跑哪些操作
    if "all" in args.ops:
        selected_ops = ALL_OPS
    else:
        selected_ops = args.ops
    print(f">>> 选择的编辑操作: {selected_ops}")

    # ====== 配置 ======
    JSON_DB      = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/mask_edit/Prior_knowledge_of_pathology/prior_db.json"
    DATASET_DIR  = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/BCSS_dataset/conditioning"
    PROB_NET_CKPT = None if args.no_probnet else "/data/huggingface/pathology_edit/prob_net/checkpoints/best.pt"
    NUCLEI_LIB_DIR = "/data/huggingface/pathology_edit/nuclei_library"

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f">>> 使用设备: {device}")

    # ====== 加载模型 ======
    prob_model = load_prob_model(PROB_NET_CKPT, device)

    if os.path.exists(NUCLEI_LIB_DIR):
        library = NucleiLibrary(NUCLEI_LIB_DIR)
    else:
        logger.warning(f"核实例库不存在: {NUCLEI_LIB_DIR}")
        library = None

    if prob_model is None and library is None:
        logger.error("概率图模型和实例库都不可用，无法生成细胞！")
        return

    # ====== 获取 mask ======
    if args.mask:
        mask_path = args.mask
    else:
        mask_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.png")))
        if not mask_files:
            print(f"错误：在 {DATASET_DIR} 中未找到 png 文件")
            return
        mask_path = mask_files[0]

    print(f">>> 处理 mask: {os.path.basename(mask_path)}")

    img_rgb = np.array(Image.open(mask_path).convert("RGB"))
    original_id_mask = rgb_to_id(img_rgb)
    print(f">>> 图像尺寸: {img_rgb.shape[:2]}")

    pure_tissue = extract_tissue_mask(original_id_mask)
    cells = extract_cell_instances(original_id_mask)
    print(f">>> 共提取 {len(cells)} 个细胞实例")

    class_counts = Counter(c['class_id'] for c in cells)
    for cid, cnt in sorted(class_counts.items()):
        print(f"    类别 {cid}: {cnt} 个")

    # ====== 实例化编辑器 ======
    validator = MaskValidator(JSON_DB)

    # 操作名 → (编辑器实例, 显示名)
    editor_registry = {
        'tumor_dilate':      (TumorBoundaryTransform(JSON_DB, validator),          "Tumor Boundary Dilation"),
        'tumor_shrink':      (TumorShrinkTransform(JSON_DB, validator),            "Tumor Shrink (Treatment Response)"),
        'lymph_dilate':      (LymphocyteInfiltrationTransform(JSON_DB, validator), "Lymphocyte Infiltration"),
        'necrosis_replace':  (NecrosisReplacementTransform(JSON_DB, validator),    "Necrosis Replacement"),
        'necrosis_fibrosis': (NecrosisFibrosisTransform(JSON_DB, validator),       "Necrosis Fibrosis"),
        'stromal_fibrosis':  (StromalFibrosisTransform(JSON_DB, validator),        "Stromal Fibrosis"),
    }

    # ====== 执行选中的操作 ======
    all_results = {}
    for op_name in selected_ops:
        if op_name not in editor_registry:
            print(f"  [警告] 未知操作: {op_name}, 跳过")
            continue

        transform, display_name = editor_registry[op_name]
        out_dir = os.path.join(args.output, op_name)

        result = process_single_edit(
            original_id_mask   = original_id_mask,
            pure_tissue_before = pure_tissue,
            cells              = cells,
            transform          = transform,
            method_name        = display_name,
            prob_model         = prob_model,
            library            = library,
            device             = device,
            n_variants         = args.n_variants,
            output_dir         = out_dir,
        )
        if result is not None:
            all_results[op_name] = result

    # ====== 汇总 ======
    print(f"\n{'='*60}")
    print(f"  全部处理完成！结果: {args.output}")
    print(f"{'='*60}")

    for name, res in all_results.items():
        print(f"\n  [{name}]")
        print(f"    保留细胞: {len(res['retained_cells'])}, "
              f"移除细胞: {len(res['removed_cells'])}")
        print(f"    变化区域像素: {int(np.sum(res['change_mask_bool']))}")
        print(f"    新生成细胞核: {res['n_generated']}")


if __name__ == "__main__":
    main()
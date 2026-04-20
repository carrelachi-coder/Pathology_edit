"""
生成 cross-reconstruction 训练对 JSON (v2 - 类别覆盖度过滤)

规则: target mask 中出现的所有组织/细胞类别, 必须在 reference mask 中也出现。
这样模型训练时, reference 中一定包含 target 需要的所有类别的外观参照。

用法: python generate_training_pairs_v2.py

输出: training_pairs.json
"""

import os
import json
import random
import argparse
import numpy as np
from collections import defaultdict
from PIL import Image
from tqdm import tqdm


IMAGE_DIR = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/BCSS_dataset/images"
MASK_DIR = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/BCSS_dataset/conditioning"
OUTPUT_JSON = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/controlnet_reference_train/training_pairs.json"
NUM_REF_PER_TARGET = 2
SEED = 42

# ============================================================
# 从你的 COLOR_MAP 反向构建: RGB → class_id
# ============================================================
COLOR_MAP = {
    0: [30,30,30], 1: [180,60,60], 2: [60,150,60], 3: [140,60,180],
    4: [60,60,180], 5: [180,180,80], 6: [160,40,40], 7: [40,40,40],
    8: [80,150,150], 9: [200,170,100], 10: [180,120,150], 11: [120,120,190],
    12: [100,190,190], 13: [200,140,60], 14: [140,200,100], 15: [140,140,140],
    16: [200,200,130], 17: [150,80,60], 18: [60,140,100], 19: [190,40,40],
    20: [80,60,150], 21: [170,170,170],
    101: [255,0,0], 102: [0,255,0], 103: [0,80,255],
    104: [255,255,0], 105: [255,0,255],
}

# 跳过的类别 (outside_roi, exclude, undetermined, other) — 不参与覆盖度判断
SKIP_CLASSES = {0, 7, 15, 21}

# 构建 RGB tuple → class_id 的查找表
RGB_TO_CLASS = {}
for cls_id, rgb in COLOR_MAP.items():
    RGB_TO_CLASS[tuple(rgb)] = cls_id


def extract_classes_from_mask(mask_path):
    """
    从 mask PNG 中提取出现的所有类别 ID (排除 SKIP_CLASSES)。
    通过采样像素来加速, 不需要遍历每个像素。
    """
    img = Image.open(mask_path).convert("RGB")
    arr = np.array(img)  # (H, W, 3)

    # 采样: 每隔几个像素取一次, 大幅加速
    H, W = arr.shape[:2]
    step = max(1, min(H, W) // 64)  # 采样步长
    sampled = arr[::step, ::step].reshape(-1, 3)

    # 找唯一颜色
    unique_colors = np.unique(sampled, axis=0)

    classes = set()
    for color in unique_colors:
        color_tuple = tuple(color)
        if color_tuple in RGB_TO_CLASS:
            cls_id = RGB_TO_CLASS[color_tuple]
            if cls_id not in SKIP_CLASSES:
                classes.add(cls_id)

    return classes


def parse_filename(filename):
    """解析文件名, 提取 WSI ID 和 patch 坐标"""
    name = os.path.splitext(filename)[0]
    parts = name.split("_")

    xmin_idx = None
    for i, p in enumerate(parts):
        if p.startswith("xmin"):
            xmin_idx = i
            break
    if xmin_idx is None:
        return None

    wsi_id = "_".join(parts[:xmin_idx])

    try:
        patch_y = int(parts[-2].replace("y", ""))
        patch_x = int(parts[-1].replace("x", ""))
    except ValueError:
        return None

    return {
        "wsi_id": wsi_id,
        "patch_y": patch_y,
        "patch_x": patch_x,
        "filename": filename,
    }


def compute_distance(info_a, info_b):
    return abs(info_a["patch_x"] - info_b["patch_x"]) + abs(info_a["patch_y"] - info_b["patch_y"])


def main():
    random.seed(SEED)

    # 收集文件
    all_files = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ])
    print(f"总文件数: {len(all_files)}")

    # 验证 mask 存在
    missing_masks = [f for f in all_files if not os.path.exists(os.path.join(MASK_DIR, f))]
    if missing_masks:
        print(f"警告: {len(missing_masks)} 个文件缺少 mask, 将跳过")

    # ============================================================
    # Step 1: 提取每个 patch 的类别集合
    # ============================================================
    print("\n[Step 1] 提取每个 mask 的类别集合...")
    patch_classes = {}  # filename → set of class ids
    valid_files = [f for f in all_files if f not in missing_masks]

    for f in tqdm(valid_files, desc="  Parsing masks"):
        mask_path = os.path.join(MASK_DIR, f)
        classes = extract_classes_from_mask(mask_path)
        patch_classes[f] = classes

    # 统计
    all_class_counts = defaultdict(int)
    for classes in patch_classes.values():
        for c in classes:
            all_class_counts[c] += 1

    TISSUE_NAMES = {
        0: 'outside_roi', 1: 'tumor', 2: 'stroma', 3: 'lymphocytic_infiltrate',
        4: 'necrosis', 5: 'glandular_secretions', 6: 'blood', 7: 'exclude',
        8: 'metaplasia', 9: 'fat', 10: 'plasma_cells', 11: 'other_immune_infiltrate',
        12: 'mucoid_material', 13: 'normal_acinus_or_duct', 14: 'lymphatics',
        15: 'undetermined', 16: 'nerve', 17: 'skin_adnexa', 18: 'blood_vessel',
        19: 'angioinvasion', 20: 'dcis', 21: 'other',
    }
    NUCLEI_NAMES = {101: 'neoplastic', 102: 'inflammatory', 103: 'connective',
                    104: 'dead', 105: 'epithelial'}
    ALL_NAMES = {**TISSUE_NAMES, **NUCLEI_NAMES}

    print("\n  类别分布 (排除 SKIP):")
    for cls_id, count in sorted(all_class_counts.items()):
        name = ALL_NAMES.get(cls_id, f"unknown_{cls_id}")
        print(f"    {cls_id:>3d} ({name:<25s}): {count:>6d} patches")

    # ============================================================
    # Step 2: 按 WSI 分组
    # ============================================================
    wsi_patches = defaultdict(list)
    for f in valid_files:
        info = parse_filename(f)
        if info is None:
            continue
        info["classes"] = patch_classes[f]
        wsi_patches[info["wsi_id"]].append(info)

    print(f"\n可用 WSI 数: {len(wsi_patches)}")

    # ============================================================
    # Step 3: 生成训练对 (带类别覆盖度过滤)
    # ============================================================
    print("\n[Step 2] 生成训练对 (类别覆盖度过滤)...")

    training_pairs = []
    skipped_no_ref = 0
    skipped_partial = 0

    for wsi_id, patches in tqdm(wsi_patches.items(), desc="  Processing WSIs"):
        if len(patches) < 2:
            continue

        for target in patches:
            target_classes = target["classes"]
            if not target_classes:
                continue

            # 找所有类别覆盖的候选 reference
            # 条件: ref 的类别集合 是 target 类别集合的超集 (或相等)
            candidates = []
            for p in patches:
                if p["filename"] == target["filename"]:
                    continue
                ref_classes = p["classes"]
                if target_classes.issubset(ref_classes):
                    candidates.append(p)

            if not candidates:
                skipped_no_ref += 1
                continue

            # 按距离排序, 从最近的 top-k 中随机选
            candidates.sort(key=lambda c: compute_distance(target, c))
            top_k = min(max(NUM_REF_PER_TARGET * 3, 6), len(candidates))
            pool = candidates[:top_k]
            refs = random.sample(pool, min(NUM_REF_PER_TARGET, len(pool)))

            for ref in refs:
                pair = {
                    "target_image": os.path.join(IMAGE_DIR, target["filename"]),
                    "target_mask": os.path.join(MASK_DIR, target["filename"]),
                    "reference_image": os.path.join(IMAGE_DIR, ref["filename"]),
                    "reference_mask": os.path.join(MASK_DIR, ref["filename"]),
                    "wsi_id": wsi_id,
                    "distance": compute_distance(target, ref),
                    "num_target_classes": len(target_classes),
                }
                training_pairs.append(pair)

    random.shuffle(training_pairs)

    # ============================================================
    # 统计
    # ============================================================
    distances = [p["distance"] for p in training_pairs]
    n_classes = [p["num_target_classes"] for p in training_pairs]

    print(f"\n{'='*60}")
    print(f"结果统计:")
    print(f"{'='*60}")
    print(f"  生成训练对数:       {len(training_pairs)}")
    print(f"  跳过 (无合格ref):   {skipped_no_ref}")
    print(f"  原始 patch 数:      {len(valid_files)}")
    print(f"")
    print(f"  距离 (像素):")
    print(f"    平均: {sum(distances)/len(distances):.0f}")
    print(f"    中位: {sorted(distances)[len(distances)//2]}")
    print(f"    最近: {min(distances)}, 最远: {max(distances)}")
    print(f"")
    print(f"  Target 类别数:")
    print(f"    平均: {sum(n_classes)/len(n_classes):.1f}")
    print(f"    最多: {max(n_classes)}, 最少: {min(n_classes)}")

    # 保存
    output = {
        "description": "Cross-reconstruction training pairs with class coverage filtering",
        "image_dir": IMAGE_DIR,
        "mask_dir": MASK_DIR,
        "num_pairs": len(training_pairs),
        "num_ref_per_target": NUM_REF_PER_TARGET,
        "filtering": "target_classes ⊆ reference_classes",
        "skip_classes": list(SKIP_CLASSES),
        "pairs": training_pairs,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n已保存到: {OUTPUT_JSON}")

    # 样本展示
    print(f"\n样本展示 (前3个):")
    for p in training_pairs[:3]:
        t = os.path.basename(p["target_image"])
        r = os.path.basename(p["reference_image"])
        print(f"  target: {t}")
        print(f"  ref:    {r}")
        print(f"  dist:   {p['distance']}px, classes: {p['num_target_classes']}")
        print()


if __name__ == "__main__":
    main()
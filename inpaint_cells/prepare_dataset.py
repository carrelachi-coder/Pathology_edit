"""
LaMa 训练数据准备脚本 v2 — 适配 ProbNet 编辑场景
====================================================

相比 v1 的改动:
    1. 新增 "全图擦除" 模式 (~15%): mask覆盖整张图, 所有细胞清零
       → 训练 ProbNet 纯靠组织layout预测细胞分布的能力
    2. 新增 "大区域擦除" 模式 (~15%): 随机选2-4种组织, 擦除这些组织上的全部细胞
       → 模拟组织编辑后大面积需要重新预测细胞的场景
    3. 保留原有 "局部擦除" (~40%) 和 "负样本" (~20%)

擦除模式概率:
    - 20% 负样本 (无细胞区域, 教模型"这里不该填细胞")
    - 20% 全图擦除 (所有细胞清零)
    - 20% 大区域擦除 (多种组织全部擦除)
    - 40% 局部擦除 (原有的 full/partial 模式)

输出格式不变, 兼容原有 ProbNet 训练代码。

用法:
    python /home/lyw/wqx-DL/flow-edit/FlowEdit-main/inpaint_cells/prepare_lama_dataset.py \
        --input-dir /home/lyw/wqx-DL/flow-edit/FlowEdit-main/BCSS_dataset/conditioning \
        --output-dir /data/huggingface/dataset_for_mask_edit \
        --val-ratio 0.1 \
        --n-augmentations 3
"""

import numpy as np
import cv2
import argparse
from pathlib import Path
from scipy import ndimage
import logging
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# 配置
# ============================================================

COLOR_MAP = {
    0: [30, 30, 30], 1: [180, 60, 60], 2: [60, 150, 60], 3: [140, 60, 180],
    4: [60, 60, 180], 5: [180, 180, 80], 6: [160, 40, 40], 7: [40, 40, 40],
    8: [80, 150, 150], 9: [200, 170, 100], 10: [180, 120, 150], 11: [120, 120, 190],
    12: [100, 190, 190], 13: [200, 140, 60], 14: [140, 200, 100], 15: [140, 140, 140],
    16: [200, 200, 130], 17: [150, 80, 60], 18: [60, 140, 100], 19: [190, 40, 40],
    20: [80, 60, 150], 21: [170, 170, 170],
    101: [255, 0, 0], 102: [0, 255, 0], 103: [0, 80, 255],
    104: [255, 255, 0], 105: [255, 0, 255],
}

TISSUE_COLORS = {cid: np.array(rgb, dtype=np.uint8) for cid, rgb in COLOR_MAP.items() if cid <= 21}
CELL_COLORS = {cid: np.array(rgb, dtype=np.uint8) for cid, rgb in COLOR_MAP.items() if cid >= 101}
CELL_COLOR_SET = np.array([rgb for rgb in CELL_COLORS.values()], dtype=np.uint8)

# 无生物学意义的组织, 不参与擦除
SKIP_TISSUES = {0, 7, 15, 21}

MIN_CELL_PIXELS = 50
MIN_REGION_PIXELS = 200

# 擦除模式概率
PROB_NEGATIVE = 0.20    # 负样本
PROB_FULL_IMAGE = 0.20  # 全图擦除
PROB_LARGE_REGION = 0.20  # 大区域擦除
# 剩余 0.50 = 局部擦除


# ============================================================
# 基础工具 (和 v1 一致)
# ============================================================

def is_cell_pixel(rgb_img: np.ndarray) -> np.ndarray:
    cell_mask = np.zeros(rgb_img.shape[:2], dtype=bool)
    for cell_rgb in CELL_COLOR_SET:
        match = np.all(rgb_img == cell_rgb[None, None, :], axis=2)
        cell_mask |= match
    return cell_mask


def get_tissue_background(rgb_img: np.ndarray, cell_mask: np.ndarray) -> np.ndarray:
    if not np.any(cell_mask):
        return rgb_img.copy()
    _, indices = ndimage.distance_transform_edt(cell_mask, return_indices=True)
    tissue_bg = rgb_img[indices[0], indices[1]]
    return tissue_bg


def identify_tissue_regions(tissue_bg: np.ndarray) -> dict:
    regions = {}
    for tid, rgb in TISSUE_COLORS.items():
        if tid in SKIP_TISSUES:
            continue
        match = np.all(tissue_bg == rgb[None, None, :], axis=2)
        if match.sum() > 0:
            regions[tid] = match
    return regions


def _draw_random_ellipses(
    h: int, w: int,
    constraint_region: np.ndarray,
    target_area: float,
    rng: np.random.Generator,
) -> np.ndarray:
    coords = np.argwhere(constraint_region)
    if len(coords) == 0:
        return np.zeros((h, w), dtype=bool)

    n_ellipses = int(rng.integers(1, 4))
    result = np.zeros((h, w), dtype=bool)

    for _ in range(n_ellipses):
        center = coords[rng.integers(0, len(coords))]
        per_area = target_area / n_ellipses
        aspect = rng.uniform(0.5, 2.0)
        r = np.sqrt(max(per_area, 25) / (np.pi * aspect))
        a = max(int(r * aspect), 5)
        b = max(int(r), 5)
        angle = rng.uniform(0, 360)

        ellipse_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            ellipse_mask,
            center=(int(center[1]), int(center[0])),
            axes=(a, b),
            angle=angle,
            startAngle=0, endAngle=360,
            color=1, thickness=-1,
        )
        result |= ellipse_mask.astype(bool)

    result &= constraint_region
    return result


def apply_erasure(
    rgb_img: np.ndarray,
    erasure_mask: np.ndarray,
    tissue_bg: np.ndarray,
) -> np.ndarray:
    erased = rgb_img.copy()
    erased[erasure_mask] = tissue_bg[erasure_mask]
    return erased


# ============================================================
# 擦除模式
# ============================================================

def generate_full_image_erasure(
    rgb_img: np.ndarray,
    cell_mask: np.ndarray,
    tissue_bg: np.ndarray,
    rng: np.random.Generator,
) -> Optional[Tuple[np.ndarray, str]]:
    """
    全图擦除: mask = 所有有生物学意义的组织区域
    训练 ProbNet 在零上下文时纯靠组织layout预测细胞
    """
    if cell_mask.sum() < MIN_CELL_PIXELS:
        return None

    tissue_regions = identify_tissue_regions(tissue_bg)
    # mask = 所有非SKIP组织的联合区域
    erasure = np.zeros_like(cell_mask)
    for tid, region in tissue_regions.items():
        erasure |= region

    if erasure.sum() < MIN_REGION_PIXELS:
        return None

    return (erasure, "full_image")


def generate_large_region_erasure(
    rgb_img: np.ndarray,
    cell_mask: np.ndarray,
    tissue_bg: np.ndarray,
    rng: np.random.Generator,
) -> Optional[Tuple[np.ndarray, str]]:
    """
    大区域擦除: 随机选 2-4 种组织类型, 擦除它们全部区域的细胞
    模拟组织编辑后多种组织区域同时需要重新预测细胞的场景
    """
    tissue_regions = identify_tissue_regions(tissue_bg)

    # 只选有细胞的组织
    candidates = []
    for tid, region in tissue_regions.items():
        if (cell_mask & region).sum() >= MIN_CELL_PIXELS:
            candidates.append(tid)

    if len(candidates) < 1:
        return None

    # 选 2-4 种 (或全部, 如果不够)
    n_pick = min(int(rng.integers(2, 5)), len(candidates))
    chosen = rng.choice(candidates, size=n_pick, replace=False).tolist()

    erasure = np.zeros_like(cell_mask)
    for tid in chosen:
        erasure |= tissue_regions[tid]

    if erasure.sum() < MIN_REGION_PIXELS:
        return None

    return (erasure, "large_region")


def generate_local_erasure(
    rgb_img: np.ndarray,
    cell_mask: np.ndarray,
    tissue_bg: np.ndarray,
    rng: np.random.Generator,
) -> Optional[Tuple[np.ndarray, str]]:
    """
    局部擦除 (原有逻辑): 在单个组织区域内 full 或 partial 擦除
    """
    h, w = rgb_img.shape[:2]
    tissue_regions = identify_tissue_regions(tissue_bg)

    positive_candidates = {}
    for tid, region in tissue_regions.items():
        if (cell_mask & region).sum() >= MIN_CELL_PIXELS:
            positive_candidates[tid] = region

    if not positive_candidates:
        return None

    pos_tid = rng.choice(list(positive_candidates.keys()))
    selected_region = positive_candidates[pos_tid]

    erase_mode = rng.choice(["full", "partial"], p=[0.3, 0.7])

    if erase_mode == "full":
        erasure = selected_region.copy()
    else:
        target_area = rng.uniform(0.05, 0.50) * selected_region.sum()
        erasure = _draw_random_ellipses(h, w, selected_region, target_area, rng)

    effective = erasure & cell_mask
    if effective.sum() < MIN_CELL_PIXELS:
        erasure = selected_region.copy()
        effective = erasure & cell_mask
        if effective.sum() < MIN_CELL_PIXELS:
            return None

    return (erasure, "local")


def generate_negative_erasure(
    rgb_img: np.ndarray,
    cell_mask: np.ndarray,
    tissue_bg: np.ndarray,
    rng: np.random.Generator,
) -> Optional[Tuple[np.ndarray, str]]:
    """
    负样本: 在无细胞的组织区域擦除, GT和输入一样
    """
    h, w = rgb_img.shape[:2]
    tissue_regions = identify_tissue_regions(tissue_bg)

    negative_candidates = {}
    for tid, region in tissue_regions.items():
        if (cell_mask & region).sum() < MIN_CELL_PIXELS and region.sum() >= MIN_REGION_PIXELS:
            negative_candidates[tid] = region

    if not negative_candidates:
        return None

    neg_tid = rng.choice(list(negative_candidates.keys()))
    selected_region = negative_candidates[neg_tid]

    target_area = rng.uniform(0.05, 0.50) * selected_region.sum()
    erasure = _draw_random_ellipses(h, w, selected_region, target_area, rng)

    if erasure.sum() < MIN_REGION_PIXELS:
        erasure = selected_region.copy()

    return (erasure, "negative")


# ============================================================
# 统一调度
# ============================================================

def generate_erasure_region(
    rgb_img: np.ndarray,
    cell_mask: np.ndarray,
    tissue_bg: np.ndarray,
    rng: np.random.Generator,
) -> Optional[Tuple[np.ndarray, str]]:
    """
    按概率选择擦除模式, 返回 (erasure_mask, mode_name) 或 None
    """
    roll = rng.random()

    if roll < PROB_NEGATIVE:
        # 负样本
        result = generate_negative_erasure(rgb_img, cell_mask, tissue_bg, rng)
        if result is not None:
            return result
        # fallback to local
        return generate_local_erasure(rgb_img, cell_mask, tissue_bg, rng)

    elif roll < PROB_NEGATIVE + PROB_FULL_IMAGE:
        # 全图擦除
        result = generate_full_image_erasure(rgb_img, cell_mask, tissue_bg, rng)
        if result is not None:
            return result
        return generate_local_erasure(rgb_img, cell_mask, tissue_bg, rng)

    elif roll < PROB_NEGATIVE + PROB_FULL_IMAGE + PROB_LARGE_REGION:
        # 大区域擦除
        result = generate_large_region_erasure(rgb_img, cell_mask, tissue_bg, rng)
        if result is not None:
            return result
        return generate_local_erasure(rgb_img, cell_mask, tissue_bg, rng)

    else:
        # 局部擦除
        return generate_local_erasure(rgb_img, cell_mask, tissue_bg, rng)


# ============================================================
# 批量处理
# ============================================================

def process_single_image(
    rgb_img: np.ndarray,
    rng: np.random.Generator,
    n_augmentations: int = 3,
) -> list:
    """
    对一张 RGB combined mask 生成多组训练数据。
    Returns: list of (gt_rgb, erased_rgb, binary_mask, mode_name)
    """
    cell_mask = is_cell_pixel(rgb_img)
    tissue_bg = get_tissue_background(rgb_img, cell_mask)

    results = []
    for _ in range(n_augmentations):
        result = generate_erasure_region(rgb_img, cell_mask, tissue_bg, rng)
        if result is None:
            continue

        erasure_mask, mode_name = result

        gt_rgb = rgb_img.copy()
        erased_rgb = apply_erasure(rgb_img, erasure_mask, tissue_bg)
        binary_mask = (erasure_mask.astype(np.uint8) * 255)

        results.append((gt_rgb, erased_rgb, binary_mask, mode_name))

    return results


def prepare_dataset(
    input_dir: str,
    output_dir: str,
    val_ratio: float = 0.1,
    n_augmentations: int = 3,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    mask_files = sorted(list(input_path.glob("*.png")))
    if not mask_files:
        mask_files = sorted(list(input_path.glob("*.jpg")))
    logger.info(f"Found {len(mask_files)} image files in {input_dir}")

    if len(mask_files) == 0:
        logger.error("No image files found!")
        return

    n_val = max(int(len(mask_files) * val_ratio), 1)
    indices = rng.permutation(len(mask_files))
    val_indices = set(indices[:n_val].tolist())

    for split in ["train", "val"]:
        (output_path / split).mkdir(parents=True, exist_ok=True)
    (output_path / "ground_truth").mkdir(parents=True, exist_ok=True)

    # 统计
    stats = {"train": 0, "val": 0}
    mode_counts = {"negative": 0, "full_image": 0, "large_region": 0, "local": 0}
    skipped = 0

    for file_idx, img_file in enumerate(mask_files):
        bgr = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
        if bgr is None:
            logger.warning(f"Cannot read {img_file}, skipping.")
            skipped += 1
            continue
        rgb_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        split = "val" if file_idx in val_indices else "train"

        samples = process_single_image(rgb_img, rng, n_augmentations)

        if not samples:
            skipped += 1
            continue

        for aug_idx, (gt_rgb, erased_rgb, binary_mask, mode_name) in enumerate(samples):
            sample_name = f"{img_file.stem}_{aug_idx:03d}"

            cv2.imwrite(
                str(output_path / split / f"{sample_name}.png"),
                cv2.cvtColor(erased_rgb, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                str(output_path / split / f"{sample_name}_mask001.png"),
                binary_mask,
            )
            cv2.imwrite(
                str(output_path / "ground_truth" / f"{sample_name}.png"),
                cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR),
            )

            stats[split] += 1
            mode_counts[mode_name] += 1

        if (file_idx + 1) % 500 == 0:
            logger.info(
                f"Processed {file_idx + 1}/{len(mask_files)} files | "
                f"train={stats['train']}, val={stats['val']}, skip={skipped}"
            )

    total = stats["train"] + stats["val"]
    logger.info(f"\n{'='*60}")
    logger.info(f"Done! Total: {total} (train={stats['train']}, val={stats['val']})")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"\nMode distribution:")
    for mode, count in sorted(mode_counts.items()):
        pct = count / total * 100 if total > 0 else 0
        logger.info(f"  {mode:<15s}: {count:>6d} ({pct:5.1f}%)")
    logger.info(f"\nOutput: {output_path}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare LaMa training dataset v2 (with full-image & large-region erasure)"
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing RGB combined mask PNG files")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for LaMa dataset")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--n-augmentations", type=int, default=3,
                        help="Number of erasure augmentations per image (default: 3, 建议比v1多1个)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        n_augmentations=args.n_augmentations,
        seed=args.seed,
    )
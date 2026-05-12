"""
build_inpaint_dataset.py (多进程并行 + skip existing)
=====================================================
用法:
  python build_inpaint_dataset.py \
      --bcss_image_dir .../BCSS_dataset/images \
      --bcss_mask_dir .../BCSS_dataset/conditioning \
      --prior_db .../prior_db.json \
      --output_dir /data/huggingface/pathology_edit/inpaint_dataset \
      --num_variants 2 \
      --num_workers 32
"""

import os
import sys
import glob
import json
import argparse
import random
import logging
from multiprocessing import Pool
from collections import Counter

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import binary_dilation
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

MASK_EDIT_DIR = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/mask_data_generate"
sys.path.insert(0, MASK_EDIT_DIR)

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
TISSUE_IDS = set(range(0, 22))
SKIP_TISSUES = {0, 7, 15, 21}
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


def rgb_to_id(img_rgb):
    h, w, _ = img_rgb.shape
    id_mask = np.full((h, w), -1, dtype=np.int16)
    for idx, color in COLOR_MAP.items():
        match = np.all(img_rgb == np.array(color, dtype=np.uint8), axis=-1)
        id_mask[match] = idx
    return id_mask


def id_to_rgb(id_mask):
    h, w = id_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for tid, color in COLOR_MAP.items():
        rgb[id_mask == tid] = color
    return rgb


def extract_tissue_mask(id_mask):
    is_tissue = np.isin(id_mask, list(TISSUE_IDS))
    if not np.any(is_tissue):
        return id_mask.copy()
    _, nearest_idx = ndimage.distance_transform_edt(~is_tissue, return_indices=True)
    return id_mask[nearest_idx[0], nearest_idx[1]]


def mask_to_prompt(mask_rgb):
    id_mask = rgb_to_id(mask_rgb)
    h, w = id_mask.shape
    total = h * w
    tissue_parts = []
    for tid in range(22):
        if tid in SKIP_TISSUES: continue
        pct = (id_mask == tid).sum() / total * 100
        if pct >= 1.0:
            tissue_parts.append((TISSUE_NAMES[tid], pct))
    tissue_parts.sort(key=lambda x: -x[1])
    nuclei_parts = [NUCLEI_NAMES[nid] for nid in [101,102,103,104,105]
                     if (id_mask == nid).sum() > 0]
    prompt = "H&E stained breast cancer histopathology at 40x magnification"
    if tissue_parts:
        prompt += ", showing " + ", ".join(f"{n} ({p:.0f}%)" for n, p in tissue_parts)
    if nuclei_parts:
        prompt += ", with " + " and ".join(nuclei_parts) + " nuclei"
    return prompt


def _random_region(h, w, min_ratio=0.05, max_ratio=0.4):
    mask = np.zeros((h, w), dtype=bool)
    if random.random() < 0.5:
        rh = random.randint(int(h * min_ratio**0.5), int(h * max_ratio**0.5))
        rw = random.randint(int(w * min_ratio**0.5), int(w * max_ratio**0.5))
        ry = random.randint(0, max(0, h - rh))
        rx = random.randint(0, max(0, w - rw))
        mask[ry:ry+rh, rx:rx+rw] = True
    else:
        cy, cx = random.randint(h//4, 3*h//4), random.randint(w//4, 3*w//4)
        ry = random.randint(int(h*0.1), int(h*0.35))
        rx = random.randint(int(w*0.1), int(w*0.35))
        yy, xx = np.ogrid[:h, :w]
        mask[((yy-cy)**2/max(ry**2,1) + (xx-cx)**2/max(rx**2,1)) < 1] = True
    return mask


def process_one_image(args_tuple):
    img_path, mask_path, prior_db_path, output_dirs, num_variants, worker_id = args_tuple

    # 每个 worker 独立加载编辑工具
    if not hasattr(process_one_image, '_editors'):
        from mask_validator import MaskValidator
        from boundary_deform import TumorBoundaryTransform
        from lymphocyte_infiltration import LymphocyteInfiltrationTransform
        from tumor_to_necrosis import NecrosisReplacementTransform
        validator = MaskValidator(prior_db_path)
        process_one_image._editors = {
            'boundary_deform': TumorBoundaryTransform(prior_db_path, validator),
            'lymphocyte_infiltration': LymphocyteInfiltrationTransform(prior_db_path, validator),
            'necrosis_expansion': NecrosisReplacementTransform(prior_db_path, validator),
        }

    editors = process_one_image._editors
    basename = os.path.splitext(os.path.basename(img_path))[0]

    try:
        src_image = np.array(Image.open(img_path).convert("RGB"))
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
        id_mask = rgb_to_id(mask_rgb)
        pure_tissue = extract_tissue_mask(id_mask)
        h, w = src_image.shape[:2]
        prompt = mask_to_prompt(mask_rgb)

        records = []
        n_skipped = 0

        # --- 编辑样本 ---
        for edit_name, editor in editors.items():
            try:
                results = editor.generate_variants(pure_tissue, n_variants=num_variants)
            except Exception:
                continue
            if not results:
                continue

            for vi, (edited_tissue, _) in enumerate(results):
                change_bool = (pure_tissue != edited_tissue)
                if change_bool.sum() < 100:
                    continue

                sample_name = f"{basename}_{edit_name}_v{vi}"
                gt_path = os.path.join(output_dirs['images'], f"{sample_name}.png")
                erased_path = os.path.join(output_dirs['erased_images'], f"{sample_name}.png")
                mask_save_path = os.path.join(output_dirs['masks'], f"{sample_name}.png")

                # === SKIP 已存在 ===
                if os.path.exists(gt_path) and os.path.exists(erased_path) and os.path.exists(mask_save_path):
                    records.append({
                        "image": gt_path,
                        "erased_image": erased_path,
                        "mask_image": mask_save_path,
                        "text": prompt,
                        "edit_type": edit_name,
                        "edit_ratio": round(float(change_bool.sum() / (h * w)), 4),
                    })
                    n_skipped += 1
                    continue

                change_dilated = binary_dilation(change_bool, iterations=3)
                erased = src_image.copy()
                erased[change_dilated] = 128

                Image.fromarray(src_image).save(gt_path)
                Image.fromarray(erased).save(erased_path)
                Image.fromarray(mask_rgb).save(mask_save_path)

                records.append({
                    "image": gt_path,
                    "erased_image": erased_path,
                    "mask_image": mask_save_path,
                    "text": prompt,
                    "edit_type": edit_name,
                    "edit_ratio": round(float(change_bool.sum() / (h * w)), 4),
                })

        # --- 随机区域样本 ---
        for ri in range(random.randint(1, 2)):
            sample_name = f"{basename}_random_v{ri}"
            gt_path = os.path.join(output_dirs['images'], f"{sample_name}.png")
            erased_path = os.path.join(output_dirs['erased_images'], f"{sample_name}.png")
            mask_save_path = os.path.join(output_dirs['masks'], f"{sample_name}.png")

            # === SKIP 已存在 ===
            if os.path.exists(gt_path) and os.path.exists(erased_path) and os.path.exists(mask_save_path):
                records.append({
                    "image": gt_path,
                    "erased_image": erased_path,
                    "mask_image": mask_save_path,
                    "text": prompt,
                    "edit_type": "random",
                    "edit_ratio": 0.1,  # 近似值，skip时不重新计算
                })
                n_skipped += 1
                continue

            region = _random_region(h, w)
            if region.sum() < 100:
                continue

            erased = src_image.copy()
            erased[region] = 128

            Image.fromarray(src_image).save(gt_path)
            Image.fromarray(erased).save(erased_path)
            Image.fromarray(mask_rgb).save(mask_save_path)

            records.append({
                "image": gt_path,
                "erased_image": erased_path,
                "mask_image": mask_save_path,
                "text": prompt,
                "edit_type": "random",
                "edit_ratio": round(float(region.sum() / (h * w)), 4),
            })

        # --- 不编辑样本 ---
        sample_name = f"{basename}_no_edit"
        gt_path = os.path.join(output_dirs['images'], f"{sample_name}.png")
        erased_path = os.path.join(output_dirs['erased_images'], f"{sample_name}.png")
        mask_save_path = os.path.join(output_dirs['masks'], f"{sample_name}.png")

        # === SKIP 已存在 ===
        if os.path.exists(gt_path) and os.path.exists(erased_path) and os.path.exists(mask_save_path):
            records.append({
                "image": gt_path,
                "erased_image": erased_path,
                "mask_image": mask_save_path,
                "text": prompt,
                "edit_type": "no_edit",
                "edit_ratio": 0.0,
            })
            n_skipped += 1
        else:
            Image.fromarray(src_image).save(gt_path)
            Image.fromarray(src_image).save(erased_path)
            Image.fromarray(mask_rgb).save(mask_save_path)

            records.append({
                "image": gt_path,
                "erased_image": erased_path,
                "mask_image": mask_save_path,
                "text": prompt,
                "edit_type": "no_edit",
                "edit_ratio": 0.0,
            })

        return records

    except Exception as e:
        return []


def main():
    parser = argparse.ArgumentParser(description="构造 Inpainting ControlNet 训练数据 (并行+skip)")
    parser.add_argument("--bcss_image_dir", type=str, required=True)
    parser.add_argument("--bcss_mask_dir", type=str, required=True)
    parser.add_argument("--prior_db", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_variants", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dirs = {
        'images': os.path.join(args.output_dir, 'images'),
        'erased_images': os.path.join(args.output_dir, 'erased_images'),
        'masks': os.path.join(args.output_dir, 'masks'),
    }
    for d in output_dirs.values():
        os.makedirs(d, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(args.bcss_image_dir, "*.png")))
    logger.info(f"找到 {len(image_files)} 张图像")

    tasks = []
    for i, img_path in enumerate(image_files):
        basename = os.path.basename(img_path)
        mask_path = os.path.join(args.bcss_mask_dir, basename)
        if not os.path.exists(mask_path):
            candidates = glob.glob(os.path.join(
                args.bcss_mask_dir, os.path.splitext(basename)[0] + "*"))
            if candidates:
                mask_path = candidates[0]
            else:
                continue
        tasks.append((img_path, mask_path, args.prior_db, output_dirs,
                       args.num_variants, i % args.num_workers))

    logger.info(f"共 {len(tasks)} 个任务, {args.num_workers} 个 worker")

    all_records = []
    with Pool(processes=args.num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_one_image, tasks),
            total=len(tasks), desc="构造数据",
        ))
    for records in results:
        all_records.extend(records)

    logger.info(f"共 {len(all_records)} 个样本")

    random.shuffle(all_records)
    n = len(all_records)
    n_train = int(n * 0.9)
    train_records = all_records[:n_train]
    val_records = all_records[n_train:]

    for split, records in [("train", train_records), ("val", val_records)]:
        jsonl_path = os.path.join(args.output_dir, f"metadata_{split}.jsonl")
        with open(jsonl_path, 'w') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        logger.info(f"{split}: {len(records)} 样本")

    edit_counts = Counter(r['edit_type'] for r in all_records)
    ratios = [r['edit_ratio'] for r in all_records if r['edit_ratio'] > 0]
    logger.info(f"\n{'='*50}")
    logger.info(f"完成！总样本: {len(all_records)}")
    if ratios:
        logger.info(f"平均编辑比例: {np.mean(ratios):.2%}")
    logger.info(f"类型分布:")
    for et, cnt in edit_counts.most_common():
        logger.info(f"  {et}: {cnt}")
    logger.info(f"输出: {args.output_dir}")


if __name__ == "__main__":
    main()

'''
python /home/lyw/wqx-DL/flow-edit/FlowEdit-main/controlnet_train/build_inpaint_dataset.py \
    --bcss_image_dir /home/lyw/wqx-DL/flow-edit/FlowEdit-main/BCSS_dataset/images \
    --bcss_mask_dir /home/lyw/wqx-DL/flow-edit/FlowEdit-main/BCSS_dataset/conditioning \
    --prior_db /home/lyw/wqx-DL/flow-edit/FlowEdit-main/Prior_knowledge_of_pathology/prior_db.json \
    --output_dir /data/huggingface/pathology_edit/inpaint_dataset \
    --num_variants 2 \
    --num_workers 32
'''
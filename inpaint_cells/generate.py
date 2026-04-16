#!/usr/bin/env python3
"""
细胞核填充推理入口 — 基于统计库的规则填充 (Phase 4.2 多数据集适配)

Phase 4.2 changes:
  - 添加 --dataset 参数, 使用统一标签体系
  - 推理时自动传入对应的 cancer_id
  - AD-1: 输入 edited_tissue_mask.png(只读) + change_region_mask → 输出独立的 nuclei_mask.png
  - 细胞库按数据集独立加载

用法:
    # 批量测试（用 layered 格式 val 数据）
    python inpaint_cells/generate.py \\
        --dataset BCSS \\
        --library /data/nuclei_library_BCSS \\
        --test-dir /path/to/layered_dataset \\
        --output-dir /path/to/results \\
        --n 10

    # 批量测试（用 legacy LaMa 格式 val 数据）
    python inpaint_cells/generate.py \\
        --dataset BCSS \\
        --library /data/nuclei_library_BCSS \\
        --test-dir /path/to/lama_dataset \\
        --format legacy \\
        --output-dir /path/to/results

    # 单张推理（分层存储）
    python inpaint_cells/generate.py \\
        --dataset PANDA \\
        --library /data/nuclei_library_PANDA \\
        --input-tissue /path/to/edited_tissue_mask.png \\
        --edit-region /path/to/edit_region_mask.png \\
        --output /path/to/nuclei_mask.png
"""

import os
import sys
import glob
import argparse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inpaint_cells.nuclei_library.library import (
    NucleiLibrary, fill_nuclei_in_region, fill_nuclei_in_region_layered,
)
from inpaint_cells.utils.mask_utils import (
    load_tissue_mask, load_nuclei_mask, save_nuclei_mask,
    overlay, rgb_to_class_map, class_map_to_rgb,
)


def test_on_val_layered(library, data_dir, output_dir, n=10, dataset_name=None):
    """
    在分层存储格式的验证集上测试。

    期望目录结构:
        {data_dir}/val/ 或 {data_dir}/ 下有
            gt_tissue/{name}.png  — tissue mask (uint8, 0-15)
            gt_nuclei/{name}.png  — nuclei mask (uint8, 0/101-105)
            masks/{name}.png      — edit region binary mask
    """
    os.makedirs(output_dir, exist_ok=True)

    ds_info = f" [{dataset_name}]" if dataset_name else ""
    print(f"Running layered validation{ds_info}...")

    # 尝试 val 子目录
    val_dir = os.path.join(data_dir, 'val')
    if not os.path.isdir(val_dir):
        val_dir = data_dir

    gt_tissue_dir = os.path.join(val_dir, 'gt_tissue')
    gt_nuclei_dir = os.path.join(val_dir, 'gt_nuclei')
    masks_dir = os.path.join(val_dir, 'masks')

    if not os.path.isdir(gt_tissue_dir):
        print(f"No gt_tissue/ dir found in {val_dir}, trying subdirectory pattern...")
        # Pattern: {val_dir}/{sample_name}/tissue_mask.png
        subdirs = sorted(glob.glob(os.path.join(val_dir, '*', 'tissue_mask.png')))
        if not subdirs:
            print(f"No layered samples found in {val_dir}")
            return
        _test_subdir_pattern(library, subdirs[:n], output_dir)
        return

    tissue_files = sorted(glob.glob(os.path.join(gt_tissue_dir, '*.png')))
    print(f"Found {len(tissue_files)} tissue mask files in {gt_tissue_dir}")

    for idx in range(min(n, len(tissue_files))):
        tissue_path = tissue_files[idx]
        fname = os.path.basename(tissue_path)
        nuclei_path = os.path.join(gt_nuclei_dir, fname)
        mask_path = os.path.join(masks_dir, fname)

        if not os.path.exists(nuclei_path) or not os.path.exists(mask_path):
            continue

        print(f"[{idx+1}/{n}] {fname}")

        # Load layered storage
        tissue = load_tissue_mask(tissue_path)                    # (H, W) int64, 0-15
        gt_nuclei = load_nuclei_mask(nuclei_path, remap=True)    # (H, W) int64, 0-5
        edit_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 128

        # Fill nuclei in edit region (operates on index 0-5)
        output_nuclei = gt_nuclei.copy()
        output_nuclei[edit_mask] = 0  # clear edit region
        placed = fill_nuclei_in_region_layered(output_nuclei, tissue, edit_mask, library)
        print(f"  Placed {placed} nuclei")

        # Visualize
        vis_input_nuc = gt_nuclei.copy()
        vis_input_nuc[edit_mask] = 0
        vis_input = overlay(tissue, vis_input_nuc)
        vis_gt = overlay(tissue, gt_nuclei)
        vis_pred = overlay(tissue, output_nuclei)

        mask_uint8 = edit_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for img in [vis_input, vis_gt, vis_pred]:
            cv2.drawContours(img, contours, -1, (255, 255, 255), 2)

        h, w = tissue.shape
        row = np.concatenate([vis_input, vis_gt, vis_pred], axis=1)

        labeled = np.zeros((h + 30, row.shape[1], 3), dtype=np.uint8)
        labeled[30:] = row
        labeled[:30] = 40

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(labeled, 'Input (erased)', (5, 22), font, 0.5, (255,255,255), 1)
        cv2.putText(labeled, 'GT', (w+5, 22), font, 0.5, (255,255,255), 1)
        cv2.putText(labeled, f'Generated ({placed} nuclei)', (w*2+5, 22), font, 0.5, (255,255,255), 1)

        out_path = os.path.join(output_dir, f'gen_{idx:03d}_{fname}')
        cv2.imwrite(out_path, cv2.cvtColor(labeled, cv2.COLOR_RGB2BGR))

    print(f"\nResults saved to {output_dir}")


def _test_subdir_pattern(library, tissue_paths, output_dir):
    """Handle subdirectory per sample: {sample_dir}/tissue_mask.png, nuclei_mask.png, edit_mask.png"""
    for idx, tissue_path in enumerate(tissue_paths):
        sample_dir = os.path.dirname(tissue_path)
        nuclei_path = os.path.join(sample_dir, 'nuclei_mask.png')
        mask_path = os.path.join(sample_dir, 'edit_mask.png')

        if not os.path.exists(nuclei_path) or not os.path.exists(mask_path):
            continue

        fname = os.path.basename(sample_dir)
        print(f"[{idx+1}] {fname}")

        tissue = load_tissue_mask(tissue_path)
        gt_nuclei = load_nuclei_mask(nuclei_path, remap=True)
        edit_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 128

        output_nuclei = gt_nuclei.copy()
        output_nuclei[edit_mask] = 0
        placed = fill_nuclei_in_region_layered(output_nuclei, tissue, edit_mask, library)
        print(f"  Placed {placed} nuclei")

        # Save result nuclei mask
        save_nuclei_mask(output_nuclei, os.path.join(output_dir, f'{fname}_nuclei.png'))

        # Visualization
        vis_input_nuc = gt_nuclei.copy()
        vis_input_nuc[edit_mask] = 0
        vis_input = overlay(tissue, vis_input_nuc)
        vis_gt = overlay(tissue, gt_nuclei)
        vis_pred = overlay(tissue, output_nuclei)

        mask_uint8 = edit_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for img in [vis_input, vis_gt, vis_pred]:
            cv2.drawContours(img, contours, -1, (255, 255, 255), 2)

        row = np.concatenate([vis_input, vis_gt, vis_pred], axis=1)
        cv2.imwrite(os.path.join(output_dir, f'gen_{idx:03d}_{fname}.png'),
                    cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

    print(f"\nResults saved to {output_dir}")


def test_on_val_legacy(library, data_dir, output_dir, n=10, dataset_name=None):
    """在旧 LaMa 格式的验证集上测试 (backward compatible)"""
    gt_dir = os.path.join(data_dir, 'ground_truth')
    val_dir = os.path.join(data_dir, 'val')
    os.makedirs(output_dir, exist_ok=True)

    ds_info = f" [{dataset_name}]" if dataset_name else ""
    print(f"Running legacy validation{ds_info}...")

    val_files = sorted([f for f in glob.glob(os.path.join(val_dir, '*.png')) if '_mask' not in f])

    for idx in range(min(n, len(val_files))):
        val_path = val_files[idx]
        fname = os.path.basename(val_path)
        gt_path = os.path.join(gt_dir, fname)
        mask_path = val_path.replace('.png', '_mask001.png')

        if not os.path.exists(gt_path) or not os.path.exists(mask_path):
            continue

        print(f"[{idx+1}/{n}] {fname}")

        gt_rgb = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)
        input_rgb = cv2.cvtColor(cv2.imread(val_path), cv2.COLOR_BGR2RGB)
        edit_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 128

        gt_map = rgb_to_class_map(gt_rgb)
        input_map = rgb_to_class_map(input_rgb)

        output_map = input_map.copy()
        output_map[edit_mask & (output_map >= 100)] = 0

        tissue_in_edit = input_map.copy()
        tissue_in_edit[tissue_in_edit >= 100] = 0
        output_map[edit_mask] = tissue_in_edit[edit_mask]

        placed = fill_nuclei_in_region(output_map, edit_mask, library)
        print(f"  Placed {placed} nuclei")

        # 可视化
        output_rgb = class_map_to_rgb(output_map)
        gt_rgb_vis = gt_rgb.copy()
        input_rgb_vis = input_rgb.copy()

        mask_uint8 = edit_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for img in [input_rgb_vis, gt_rgb_vis, output_rgb]:
            cv2.drawContours(img, contours, -1, (255, 255, 255), 2)

        h, w = gt_rgb.shape[:2]
        row = np.concatenate([input_rgb_vis, gt_rgb_vis, output_rgb], axis=1)

        labeled = np.zeros((h + 30, row.shape[1], 3), dtype=np.uint8)
        labeled[30:] = row
        labeled[:30] = 40

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(labeled, 'Input (erased)', (5, 22), font, 0.5, (255,255,255), 1)
        cv2.putText(labeled, 'GT', (w+5, 22), font, 0.5, (255,255,255), 1)
        cv2.putText(labeled, f'Generated ({placed} nuclei)', (w*2+5, 22), font, 0.5, (255,255,255), 1)

        out_path = os.path.join(output_dir, f'gen_{idx:03d}_{fname}')
        cv2.imwrite(out_path, cv2.cvtColor(labeled, cv2.COLOR_RGB2BGR))

    print(f"\nResults saved to {output_dir}")


def single_inference_layered(library, tissue_path, edit_region_path, output_path,
                             dataset_name=None):
    """
    单张推理 — 分层存储模式。
    输入: edited_tissue_mask.png (只读) + edit_region_mask
    输出: 独立的 nuclei_mask.png (0/101-105)
    """
    ds_info = f" [{dataset_name}]" if dataset_name else ""
    print(f"Single inference{ds_info}")

    tissue = load_tissue_mask(tissue_path)  # (H, W) int64, 0-15
    edit_mask = cv2.imread(edit_region_path, cv2.IMREAD_GRAYSCALE) > 128

    # 从零生成 — edit 区域外也填零 (无已有 nuclei 信息时)
    output_nuclei = np.zeros_like(tissue, dtype=np.int64)
    placed = fill_nuclei_in_region_layered(output_nuclei, tissue, edit_mask, library)
    print(f"Placed {placed} nuclei")

    # Save as raw nuclei mask (0/101-105)
    save_nuclei_mask(output_nuclei, output_path)
    print(f"Saved nuclei mask to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='细胞核填充推理 (Phase 4.2 多数据集适配)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (BCSS, PANDA, GlaS, IGNITE, PUMA, ORCA). '
                             'Used for logging and auto cancer_id lookup.')
    parser.add_argument('--library', required=True,
                        help='Path to nuclei library directory')
    parser.add_argument('--test-dir', default=None,
                        help='Dataset directory for testing')
    parser.add_argument('--format', choices=['auto', 'layered', 'legacy'], default='auto',
                        help='Data format: layered (AD-1), legacy (RGB combined), or auto-detect')
    parser.add_argument('--output-dir', default='./nuclei_gen_results')
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--input-tissue', default=None,
                        help='Edited tissue mask (uint8 PNG, 0-15)')
    parser.add_argument('--edit-region', default=None,
                        help='Edit region binary mask')
    parser.add_argument('--output', default=None,
                        help='Output nuclei mask path')
    args = parser.parse_args()

    # 数据集信息 (用于日志)
    dataset_name = args.dataset
    if dataset_name:
        try:
            from dataset_config import get_config
            config = get_config(dataset_name)
            print(f"Dataset: {config.name} ({config.cancer_type})")
            print(f"  cancer_type_index: {config.cancer_type_index}")
        except Exception as e:
            print(f"Warning: Could not load config for '{dataset_name}': {e}")

    print("Loading nuclei library...")
    library = NucleiLibrary(args.library, dataset=dataset_name)

    if args.test_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        # Auto-detect format
        fmt = args.format
        if fmt == 'auto':
            has_layered = (
                os.path.isdir(os.path.join(args.test_dir, 'gt_tissue'))
                or os.path.isdir(os.path.join(args.test_dir, 'val', 'gt_tissue'))
                or len(glob.glob(os.path.join(args.test_dir, '*', 'tissue_mask.png'))) > 0
                or len(glob.glob(os.path.join(args.test_dir, 'val', '*', 'tissue_mask.png'))) > 0
            )
            fmt = 'layered' if has_layered else 'legacy'
            print(f"Auto-detected format: {fmt}")

        if fmt == 'layered':
            test_on_val_layered(library, args.test_dir, args.output_dir, args.n,
                                dataset_name=dataset_name)
        else:
            test_on_val_legacy(library, args.test_dir, args.output_dir, args.n,
                               dataset_name=dataset_name)

    elif args.input_tissue and args.edit_region:
        out_path = args.output or 'nuclei_mask.png'
        single_inference_layered(library, args.input_tissue, args.edit_region,
                                 out_path, dataset_name=dataset_name)

    else:
        print("Please specify --test-dir or --input-tissue + --edit-region")


if __name__ == '__main__':
    main()

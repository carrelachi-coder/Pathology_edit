#!/usr/bin/env python3
"""
细胞核填充推理入口 — 基于统计库的规则填充 (不需要 ProbNet)

在编辑区域内，根据组织类型查表获取核密度/类型分布，
通过泊松盘采样放置核实例。

用法:
    # 批量测试（用 val 数据）
    python inpaint_cells/generate.py \
        --library /data/huggingface/pathology_edit/nuclei_library \
        --test-dir /path/to/lama_dataset \
        --output-dir /path/to/results \
        --n 10

    # 单张推理
    python inpaint_cells/generate.py \
        --library /data/huggingface/pathology_edit/nuclei_library \
        --input-mask /path/to/edited_tissue_mask.png \
        --edit-region /path/to/edit_region_mask.png \
        --output /path/to/output.png
"""

import os
import sys
import glob
import argparse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inpaint_cells.nuclei_library.library import (
    NucleiLibrary, fill_nuclei_in_region,
)
from inpaint_cells.utils.mask_utils import rgb_to_class_map, class_map_to_rgb


def test_on_val(library, data_dir, output_dir, n=10):
    """在验证集上测试"""
    gt_dir = os.path.join(data_dir, 'ground_truth')
    val_dir = os.path.join(data_dir, 'val')
    os.makedirs(output_dir, exist_ok=True)

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


def main():
    parser = argparse.ArgumentParser(description='细胞核填充推理')
    parser.add_argument('--library', required=True, help='Path to nuclei library directory')
    parser.add_argument('--test-dir', default=None, help='lama_dataset directory for testing')
    parser.add_argument('--output-dir', default='./nuclei_gen_results')
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--input-mask', default=None, help='Edited tissue mask (RGB PNG)')
    parser.add_argument('--edit-region', default=None, help='Edit region binary mask')
    parser.add_argument('--output', default=None, help='Output path')
    args = parser.parse_args()

    print("Loading nuclei library...")
    library = NucleiLibrary(args.library)

    if args.test_dir:
        test_on_val(library, args.test_dir, args.output_dir, args.n)
    elif args.input_mask and args.edit_region:
        input_rgb = cv2.cvtColor(cv2.imread(args.input_mask), cv2.COLOR_BGR2RGB)
        edit_mask = cv2.imread(args.edit_region, cv2.IMREAD_GRAYSCALE) > 128
        output_map = rgb_to_class_map(input_rgb)
        output_map[edit_mask & (output_map >= 100)] = 0
        placed = fill_nuclei_in_region(output_map, edit_mask, library)
        print(f"Placed {placed} nuclei")
        output_rgb = class_map_to_rgb(output_map)
        out_path = args.output or 'generated_mask.png'
        cv2.imwrite(out_path, cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR))
        print(f"Saved to {out_path}")
    else:
        print("Please specify --test-dir or --input-mask + --edit-region")


if __name__ == '__main__':
    main()

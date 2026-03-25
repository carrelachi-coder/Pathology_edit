#!/usr/bin/env python3
"""
小型调试脚本: LLM解析 + Mask编辑, 逐层输出中间结果

输出文件:
  01_original_tissue.png        — 原mask的纯组织层 (去掉细胞)
  02_edited_tissue.png          — 编辑后的纯组织层
  03_tissue_change_region.png   — 组织层变化区域 (白=变化)
  04_retained_cells.png         — 保留的细胞层 (质心不在变化区域的原始细胞)
  05_new_cells.png              — 新生成的细胞层 (ProbNet在变化区域生成)
  06_combined_cells.png         — 保留细胞 + 新细胞 合并
  07_tar_mask.png               — 最终完整mask (组织 + 所有细胞)
  08_original_full.png          — 原始完整mask (对照)
  09_compare.png                — 横向拼接对比

用法:
  CUDA_VISIBLE_DEVICES=0 python test_mask_debug.py \
      --mask /path/to/mask.png \
      --original-report "..." \
      --edited-report "..." \
      --output ./debug_output
"""

import sys
import os
import json
import argparse
import numpy as np
from PIL import Image

# =============================================================================
# 路径
# =============================================================================
RULE_ENGINE_DIR = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/mask_edit/rule_engine"
MASK_GEN_DIR = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/mask_edit/mask_data_generate"
sys.path.insert(0, RULE_ENGINE_DIR)
sys.path.insert(0, MASK_GEN_DIR)

PRIOR_DB_PATH = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/mask_edit/Prior_knowledge_of_pathology/prior_db.json"
PROB_NET_CKPT = "/data/huggingface/pathology_edit/prob_net/checkpoints/best.pt"
NUCLEI_LIBRARY = "/data/huggingface/pathology_edit/nuclei_library"
LLM_PATH = "/data/huggingface/Qwen2.5-VL-7B-Instruct"

DEFAULT_MASK = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/BCSS_dataset/conditioning/TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500_y0_x512.png"

# =============================================================================
# 颜色映射
# =============================================================================
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

_rgb_to_val = {}
for val, rgb in COLOR_MAP.items():
    _rgb_to_val[rgb[0] * 65536 + rgb[1] * 256 + rgb[2]] = val


def load_mask_from_png(path):
    img = np.array(Image.open(path).convert("RGB"))
    encoded = img[:,:,0].astype(np.int64)*65536 + img[:,:,1].astype(np.int64)*256 + img[:,:,2].astype(np.int64)
    result = np.zeros(img.shape[:2], dtype=np.int64)
    for key, val in _rgb_to_val.items():
        result[encoded == key] = val
    return result


def class_map_to_rgb(class_map):
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for val, color in COLOR_MAP.items():
        rgb[class_map == val] = color
    return rgb


def save_layer(arr, path, is_bool=False):
    """保存一个层为 PNG"""
    if is_bool:
        img = np.zeros(arr.shape, dtype=np.uint8)
        img[arr] = 255
        Image.fromarray(img).save(path)
    else:
        Image.fromarray(class_map_to_rgb(arr)).save(path)


def main():
    parser = argparse.ArgumentParser(description="Debug mask editing layers")
    parser.add_argument("--mask", default=DEFAULT_MASK)
    parser.add_argument("--output", default="./debug_output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # 两种输入方式: 报告对 或 直接给 diff
    parser.add_argument("--original-report", default=None)
    parser.add_argument("--edited-report", default=None)
    parser.add_argument("--diff", default=None, help="JSON string of semantic_diff")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # --- 加载 mask ---
    print(f"Loading mask: {args.mask}")
    combined_mask = load_mask_from_png(args.mask)
    print(f"Shape: {combined_mask.shape}")

    # --- 获取 semantic_diff ---
    if args.diff:
        semantic_diff = json.loads(args.diff)
        print(f"Using provided diff: {json.dumps(semantic_diff, indent=2)}")
    elif args.original_report and args.edited_report:
        print(f"Original: {args.original_report[:80]}...")
        print(f"Edited:   {args.edited_report[:80]}...")
        from llm_parser import LLMParser
        llm = LLMParser(model_path=LLM_PATH, device=args.device)
        semantic_diff = llm.parse(args.original_report, args.edited_report)
        del llm
        import torch, gc
        torch.cuda.empty_cache()
        gc.collect()
        print(f"LLM output: {json.dumps(semantic_diff, indent=2)}")
    else:
        print("ERROR: provide --diff or (--original-report + --edited-report)")
        return

    # 保存 semantic_diff
    with open(os.path.join(args.output, "00_semantic_diff.json"), "w") as f:
        json.dump(semantic_diff, f, indent=2)

    # --- 手动执行 edit 流程, 逐步保存中间结果 ---
    from rule_engine import MaskEditor, MaskAnalyzer, RuleEngine

    editor = MaskEditor(
        prior_db_path=PRIOR_DB_PATH,
        prob_net_ckpt=PROB_NET_CKPT if os.path.exists(PROB_NET_CKPT) else None,
        nuclei_library_path=NUCLEI_LIBRARY if os.path.exists(NUCLEI_LIBRARY) else None,
        seed=args.seed,
    )
    editor._ensure_transforms()

    # Step 1: 拆分
    original_tissue = editor._get_tissue_only(combined_mask)
    cell_instances = editor._get_cell_instances(combined_mask)
    print(f"\nExtracted {len(cell_instances)} cell instances")
    print(f"Original tissue: {MaskAnalyzer(original_tissue).summary()}")

    save_layer(original_tissue, os.path.join(args.output, "01_original_tissue.png"))

    # Step 2: 规则引擎
    ops = editor.rule_engine.plan(semantic_diff, original_tissue)
    type_bias_map, density_scale_map = editor.rule_engine.compute_cell_adjustments(semantic_diff)
    print(f"Planned ops: {len(ops)}")
    for op in ops:
        print(f"  {op['op']}: {op['params']}")
    if type_bias_map:
        print(f"Type bias: {type_bias_map}")
    if density_scale_map:
        print(f"Density scale: {density_scale_map}")

    # Step 3: 组织层编辑
    current_tissue = original_tissue.copy()
    for op_spec in ops:
        op_name = op_spec["op"]
        params = op_spec["params"]
        print(f"\nExecuting: {op_name} with {params}")

        for attempt in range(5):
            if op_name == "tumor_dilate":
                new_tissue, log = editor._tumor_transform.apply(
                    current_tissue, target_delta=abs(params.get("target_delta", 0.10)))
            elif op_name == "tumor_shrink":
                new_tissue, log = editor._tumor_shrink_transform.apply(
                    current_tissue, target_delta=abs(params.get("target_delta", 0.08)))
            elif op_name == "lymph_dilate":
                new_tissue, log = editor._lymph_transform.apply(
                    current_tissue, target_delta=abs(params.get("target_delta", 0.05)))
            elif op_name == "necrosis_replace":
                new_tissue, log = editor._necrosis_transform.apply(
                    current_tissue, n_pick=params.get("n_pick", 1))
            elif op_name == "necrosis_fibrosis":
                new_tissue, log = editor._necrosis_fibrosis_transform.apply(
                    current_tissue, target_delta=abs(params.get("target_delta", 0.08)))
            elif op_name == "stromal_fibrosis":
                new_tissue, log = editor._stromal_fibrosis_transform.apply(
                    current_tissue, target_delta=abs(params.get("target_delta", 0.08)))
            else:
                print(f"  Unknown op: {op_name}")
                break

            if log.accepted:
                current_tissue = new_tissue
                print(f"  Accepted on attempt {attempt + 1}")
                break
            else:
                print(f"  Attempt {attempt + 1} rejected: {log.rejection_reason}")

    save_layer(current_tissue, os.path.join(args.output, "02_edited_tissue.png"))
    print(f"\nEdited tissue: {MaskAnalyzer(current_tissue).summary()}")

    # Step 4: 组织变化区域
    tissue_change_region = (current_tissue != original_tissue)
    print(f"Tissue change region: {tissue_change_region.sum()} pixels "
          f"({tissue_change_region.sum() / tissue_change_region.size * 100:.1f}%)")
    save_layer(tissue_change_region, os.path.join(args.output, "03_tissue_change_region.png"), is_bool=True)

    # Step 5: 保留细胞 (质心不在变化区域)
    retained_nuclei = editor._retain_cells_outside_change(
        cell_instances, tissue_change_region, current_tissue)
    save_layer(retained_nuclei, os.path.join(args.output, "04_retained_cells.png"))
    print(f"Retained cells: {(retained_nuclei > 0).sum()} pixels")

    # Step 6: 在变化区域生成新细胞
    # 先算 fill_region
    current_combined = editor._merge_tissue_and_cells(current_tissue, retained_nuclei)
    fill_region = (current_combined != combined_mask)
    print(f"Fill region: {fill_region.sum()} pixels "
          f"({fill_region.sum() / fill_region.size * 100:.1f}%)")

    new_cells = np.zeros_like(current_tissue, dtype=np.int64)
    if editor.nuclei_library_path and fill_region.any():
        print("Filling new cells with ProbNet + Library...")
        new_cells = editor._fill_cells_probnet_and_library(
            current_tissue, retained_nuclei, fill_region,
            type_bias_map=type_bias_map,
            density_scale_map=density_scale_map)
        print(f"New cells: {(new_cells > 0).sum()} pixels")
    save_layer(new_cells, os.path.join(args.output, "05_new_cells.png"))

    # Step 7: 合并细胞层
    combined_cells = retained_nuclei.copy()
    combined_cells[new_cells > 0] = new_cells[new_cells > 0]
    save_layer(combined_cells, os.path.join(args.output, "06_combined_cells.png"))

    # Step 8: 合并得到完整 tar_mask
    tar_mask = editor._merge_tissue_and_cells(current_tissue, combined_cells)
    save_layer(tar_mask, os.path.join(args.output, "07_tar_mask_before_adjust.png"))

    # Step 8.5: 在完整的 tar_mask 上做细胞调整 (type bias / density)
    has_cell_adjustments = bool(type_bias_map or density_scale_map)
    if has_cell_adjustments:
        print("\nApplying cell adjustments on final mask...")
        final_cell_instances = editor._get_cell_instances(tar_mask)
        print(f"  Final mask has {len(final_cell_instances)} cell instances")
        adjusted_nuclei = editor._apply_cell_only_adjustments(
            final_cell_instances, current_tissue, type_bias_map, density_scale_map)
        tar_mask = editor._merge_tissue_and_cells(current_tissue, adjusted_nuclei)
        save_layer(adjusted_nuclei, os.path.join(args.output, "07b_adjusted_cells.png"))

    save_layer(tar_mask, os.path.join(args.output, "08_tar_mask_final.png"))

    # Step 9: 原始完整 mask (对照)
    save_layer(combined_mask, os.path.join(args.output, "09_original_full.png"))

    # Step 10: 横向拼接对比
    compare = np.concatenate([
        class_map_to_rgb(combined_mask),      # 原始
        class_map_to_rgb(original_tissue),    # 原组织
        class_map_to_rgb(current_tissue),     # 新组织
        class_map_to_rgb(tar_mask),           # 最终
    ], axis=1)
    Image.fromarray(compare).save(os.path.join(args.output, "10_compare.png"))

    print(f"\nAll layers saved to {args.output}/")
    print("Files:")
    for f in sorted(os.listdir(args.output)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
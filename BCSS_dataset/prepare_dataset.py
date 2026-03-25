"""
All-in-one dataset preparation for FLUX ControlNet training.

Step 1: Cut 512x512 patches from ROI images + combined masks (synced)
Step 2: Generate text prompts from mask statistics
Step 3: Build HuggingFace-compatible dataset with metadata.jsonl

Input:
    images/          → H&E ROI images (PNG)
    combined_masks/  → Combined semantic maps (NPY, encoding: 0-21 tissue, 101-105 nuclei)

Output:
    BCSS_dataset/
    ├── images/           → target H&E patches (PNG)
    ├── conditioning/     → RGB semantic map patches (PNG, ControlNet input)
    ├── metadata.jsonl    → {"image", "conditioning_image", "text"}
    └── stats.txt         → dataset statistics

Usage:
    python /home/lyw/wqx-DL/flow-edit/FlowEdit-main/BCSS_dataset/prepare_dataset.py \
        --image_dir /data/huggingface/pathology_edit/CellViT/CellViT-plus-plus-main/images \
        --mask_dir  /data/huggingface/pathology_edit/CellViT/CellViT-plus-plus-main/combined_masks \
        --output_dir /home/lyw/wqx-DL/flow-edit/FlowEdit-main/BCSS_dataset \
        --patch_size 512 \
        --stride 256 \
        --min_tissue_ratio 0.5
"""

import numpy as np
from PIL import Image
import os
import glob
import json
import argparse
import time


# ========== Color Mapping for RGB Semantic Map ==========
# Tissue layer: muted colors (background regions)
# Nuclei layer: bright saturated colors (foreground, need to stand out)

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

# Build lookup table for fast color mapping
MAX_CODE = max(COLOR_MAP.keys()) + 1
COLOR_LUT = np.zeros((MAX_CODE, 3), dtype=np.uint8)
for code, color in COLOR_MAP.items():
    COLOR_LUT[code] = color


# ========== Name mappings for prompt generation ==========
TISSUE_NAMES = {
    1: 'tumor', 2: 'stroma', 3: 'lymphocytic infiltrate',
    4: 'necrosis', 5: 'glandular secretions', 6: 'blood',
    8: 'metaplasia', 9: 'fat', 10: 'plasma cells',
    11: 'immune infiltrate', 12: 'mucoid material',
    13: 'normal acinus or duct', 14: 'lymphatics', 16: 'nerve',
    17: 'skin adnexa', 18: 'blood vessel', 19: 'angioinvasion',
    20: 'DCIS', 21: 'other tissue'
}

NUCLEI_NAMES = {
    101: 'neoplastic', 102: 'inflammatory', 103: 'connective',
    104: 'dead', 105: 'epithelial'
}


def mask_to_rgb(mask):
    """Convert numeric combined mask to RGB image using color LUT."""
    # Clip values to valid range
    mask_clipped = np.clip(mask, 0, MAX_CODE - 1).astype(np.int32)
    return COLOR_LUT[mask_clipped]


def generate_prompt(mask_patch):
    """
    Generate text prompt from mask patch statistics.
    Describes tissue composition and nuclei types present.
    """
    total_pixels = mask_patch.size
    unique, counts = np.unique(mask_patch, return_counts=True)
    stats = dict(zip(unique.tolist(), counts.tolist()))

    # Separate tissue and nuclei stats
    tissue_stats = {}
    nuclei_types = []

    for code, count in stats.items():
        if code == 0 or code == 7:  # outside_roi, exclude
            continue
        pct = count / total_pixels * 100
        if code < 100 and code in TISSUE_NAMES and pct >= 2.0:
            tissue_stats[TISSUE_NAMES[code]] = pct
        elif code >= 101 and code in NUCLEI_NAMES and pct >= 0.5:
            nuclei_types.append(NUCLEI_NAMES[code])

    # Build prompt
    parts = ["H&E stained breast cancer histopathology at 40x magnification"]

    # Tissue composition (sorted by percentage, top components)
    if tissue_stats:
        sorted_tissues = sorted(tissue_stats.items(), key=lambda x: -x[1])
        tissue_descs = []
        for name, pct in sorted_tissues[:5]:  # top 5
            if pct >= 10:
                tissue_descs.append(f"{name} ({pct:.0f}%)")
            else:
                tissue_descs.append(name)
        parts.append("showing " + ", ".join(tissue_descs))

    # Nuclei types
    if nuclei_types:
        parts.append("with " + " and ".join(nuclei_types) + " nuclei")

    return ", ".join(parts)


def extract_patches(image_path, mask_path, patch_size, stride, min_tissue_ratio):
    """
    Extract synced patches from image and combined mask.

    Returns list of (image_patch, mask_patch) tuples that pass quality filter.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    mask = np.load(mask_path)

    # Handle size mismatch
    if img.shape[:2] != mask.shape[:2]:
        mask_pil = Image.fromarray(mask.astype(np.uint8))
        mask_pil = mask_pil.resize((img.shape[1], img.shape[0]), Image.NEAREST)
        mask = np.array(mask_pil)

    H, W = img.shape[:2]
    patches = []

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            img_patch = img[y:y+patch_size, x:x+patch_size]
            mask_patch = mask[y:y+patch_size, x:x+patch_size]

            # Quality filter: skip patches with too much outside_roi (code=0) or exclude (code=7)
            n_invalid = np.sum((mask_patch == 0) | (mask_patch == 7))
            tissue_ratio = 1.0 - n_invalid / mask_patch.size

            if tissue_ratio >= min_tissue_ratio:
                patches.append((img_patch, mask_patch, y, x))

    return patches


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for FLUX ControlNet training")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing H&E ROI images (PNG)")
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="Directory containing combined semantic maps (NPY)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output dataset directory")
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256,
                        help="Stride for patch extraction (256 = 50%% overlap)")
    parser.add_argument("--min_tissue_ratio", type=float, default=0.5,
                        help="Minimum ratio of valid tissue pixels (filter out mostly-blank patches)")
    args = parser.parse_args()

    # Create output directories
    img_out_dir = os.path.join(args.output_dir, "images")
    cond_out_dir = os.path.join(args.output_dir, "conditioning")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(cond_out_dir, exist_ok=True)

    # Find matching pairs
    image_files = sorted(glob.glob(os.path.join(args.image_dir, "*.png")))
    mask_index = {}
    for f in glob.glob(os.path.join(args.mask_dir, "*.npy")):
        basename = os.path.splitext(os.path.basename(f))[0]
        mask_index[basename] = f

    matched = []
    for img_path in image_files:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        if basename in mask_index:
            matched.append((img_path, mask_index[basename], basename))

    print(f"Found {len(image_files)} images, {len(mask_index)} masks, {len(matched)} matched pairs")

    if len(matched) == 0:
        print("No matched pairs found. Check filenames.")
        return

    # Process all pairs
    metadata = []
    global_patch_id = 0
    total_patches_before_filter = 0
    t_start = time.time()

    # Track tissue/nuclei distribution
    tissue_counter = {}
    nuclei_counter = {}

    for roi_idx, (img_path, mask_path, basename) in enumerate(matched):
        patches = extract_patches(
            img_path, mask_path,
            args.patch_size, args.stride, args.min_tissue_ratio
        )

        roi_patch_count = 0

        for img_patch, mask_patch, y, x in patches:
            patch_name = f"{basename}_y{y}_x{x}"

            # Save image patch
            img_out_path = os.path.join("images", f"{patch_name}.png")
            Image.fromarray(img_patch).save(os.path.join(args.output_dir, img_out_path))

            # Save RGB conditioning image (semantic map → color)
            cond_rgb = mask_to_rgb(mask_patch)
            cond_out_path = os.path.join("conditioning", f"{patch_name}.png")
            Image.fromarray(cond_rgb).save(os.path.join(args.output_dir, cond_out_path))

            # Generate prompt
            prompt = generate_prompt(mask_patch)

            # Metadata entry
            metadata.append({
                "image": img_out_path,
                "conditioning_image": cond_out_path,
                "text": prompt,
            })

            # Track statistics
            unique_vals = np.unique(mask_patch)
            for v in unique_vals:
                if 1 <= v <= 21:
                    tissue_counter[v] = tissue_counter.get(v, 0) + 1
                elif 101 <= v <= 105:
                    nuclei_counter[v] = nuclei_counter.get(v, 0) + 1

            global_patch_id += 1
            roi_patch_count += 1

        elapsed = time.time() - t_start
        print(f"[{roi_idx+1}/{len(matched)}] {basename}: "
              f"{roi_patch_count} patches  (total: {global_patch_id})  "
              f"time: {elapsed:.0f}s")

    # Write metadata.jsonl
    jsonl_path = os.path.join(args.output_dir, "metadata.jsonl")
    with open(jsonl_path, "w") as f:
        for entry in metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Write statistics
    stats_path = os.path.join(args.output_dir, "stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"Dataset Statistics\n")
        f.write(f"{'='*50}\n")
        f.write(f"ROI images: {len(matched)}\n")
        f.write(f"Total patches: {global_patch_id}\n")
        f.write(f"Patch size: {args.patch_size}x{args.patch_size}\n")
        f.write(f"Stride: {args.stride}\n")
        f.write(f"Min tissue ratio: {args.min_tissue_ratio}\n")
        f.write(f"\nTissue types present (by patch count):\n")
        for code in sorted(tissue_counter.keys()):
            name = TISSUE_NAMES.get(code, f"code_{code}")
            f.write(f"  {name:30s}: {tissue_counter[code]:6d} patches\n")
        f.write(f"\nNuclei types present (by patch count):\n")
        for code in sorted(nuclei_counter.keys()):
            name = NUCLEI_NAMES.get(code, f"code_{code}")
            f.write(f"  {name:15s}: {nuclei_counter[code]:6d} patches\n")

    # Print summary
    elapsed = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"  Total patches: {global_patch_id}")
    print(f"  Output: {args.output_dir}")
    print(f"  metadata.jsonl: {jsonl_path}")
    print(f"  stats.txt: {stats_path}")

    # Show example prompt
    if metadata:
        print(f"\nExample prompt:")
        print(f"  {metadata[0]['text']}")
    if len(metadata) > len(metadata)//2:
        print(f"  {metadata[len(metadata)//2]['text']}")


if __name__ == "__main__":
    main()
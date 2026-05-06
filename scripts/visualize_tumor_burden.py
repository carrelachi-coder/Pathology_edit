"""Visualize tumor_burden_increase and tumor_burden_decrease on real masks."""

import json
import re
from pathlib import Path

import numpy as np

from phase3_mask_edit.core.config import load_recipe
from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.core.mask_io import load_id_mask, save_rgb_mask, save_change_region
from phase3_mask_edit.generic.tumor_burden import (
    apply_tumor_burden_increase,
    apply_tumor_burden_decrease,
)

RECIPE_PATH = Path("phase3_mask_edit/recipes/generic.yaml")
OUT_DIR = Path("phase3_mask_edit/previews/tumor_burden")
DATA_ROOT = Path("edit_datasets")


def _primitive(recipe, name):
    return next(p for p in recipe["primitives"] if p["name"] == name)


def _pick_samples(profile, target_burden_range=(15, 50), count=2):
    """Pick real mask samples with tumor burden in target range."""
    meta_path = DATA_ROOT / profile / "metadata.jsonl"
    results = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            text = d["text"]
            m = re.search(r"tumor burden (\d+)%", text)
            if not m:
                continue
            tb = int(m.group(1))
            if not (target_burden_range[0] <= tb <= target_burden_range[1]):
                continue
            mask_rel = d.get("conditioning_image", "").replace("\\", "/")
            mask_file = mask_rel.split("/")[-1]
            mask_path = DATA_ROOT / profile / "tissue_masks" / mask_file
            if mask_path.exists():
                results.append((tb, mask_path))
                if len(results) >= count:
                    break
    return results


def _target_fraction(primitive_config, strength):
    intervals = primitive_config["parameter_ranges"]["target_area_delta_fraction"]
    lo, hi = intervals[strength]
    return (lo + hi) / 2


def _target_decrease_fraction(primitive_config, strength):
    intervals = primitive_config["parameter_ranges"]["target_area_decrease_fraction"]
    lo, hi = intervals[strength]
    return (lo + hi) / 2


def _make_comparison_panel(src_rgb, tgt_rgb, change_region_rgb, label):
    """Concatenate src, tgt, change_region side by side."""
    from PIL import Image, ImageDraw

    h, w = src_rgb.shape[:2]
    panel = np.zeros((h, w * 3 + 60, 3), dtype=np.uint8)
    panel[:, :w] = src_rgb
    panel[:, w:w*2] = tgt_rgb
    panel[:, w*2:w*3] = change_region_rgb
    # White separator strips
    panel[:, w-2:w+2] = 255
    panel[:, w*2-2:w*2+2] = 255

    img = Image.fromarray(panel)
    draw = ImageDraw.Draw(img)
    draw.text((5, 2), "Source", fill=(255, 255, 255))
    draw.text((w+5, 2), "Target", fill=(255, 255, 255))
    draw.text((w*2+5, 2), "Change", fill=(255, 255, 255))
    draw.text((5, h + 5), label, fill=(255, 255, 255))

    return np.asarray(img)


def main():
    recipe = load_recipe(RECIPE_PATH)
    increase_config = _primitive(recipe, "tumor_burden_increase")
    decrease_config = _primitive(recipe, "tumor_burden_decrease")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cases = [
        ("BCSS", "mild"),
        ("BCSS", "moderate"),
        ("PANDA", "mild"),
        ("PANDA", "moderate"),
        ("ORCA", "mild"),
        ("ORCA", "moderate"),
    ]

    for profile, strength in cases:
        schema = MaskProfileSchema.from_reference_profile(profile)
        samples = _pick_samples(profile)
        if not samples:
            print(f"  [{profile}/{strength}] No real mask samples found.")
            continue

        for idx, (tb, mask_path) in enumerate(samples):
            tag = f"{profile}_{strength}_real{idx}_tb{tb}"

            # Load real mask
            try:
                old_mask = load_id_mask(mask_path)
            except Exception as e:
                print(f"  [{profile}/{strength}] Failed to load {mask_path}: {e}")
                continue

            context = MaskEditContext.from_mask(old_mask, schema)

            # ── Increase ──
            try:
                target_frac = _target_fraction(increase_config, strength)
                intent = EditIntent.from_mapping({
                    "primitive": "tumor_burden_increase",
                    "reference_profile": profile,
                    "target_change_fraction": target_frac,
                    "seed": idx,
                })
                result = apply_tumor_burden_increase(old_mask, schema, context, increase_config, intent)

                src_rgb = save_rgb_mask(old_mask, OUT_DIR / f"{tag}_increase_src.png")
                tgt_rgb = save_rgb_mask(result.target_mask, OUT_DIR / f"{tag}_increase_tgt.png")
                save_change_region(result.change_region, OUT_DIR / f"{tag}_increase_change.png")

                src_tumor = int(np.count_nonzero(np.isin(old_mask, schema.tumor_fine_ids)))
                tgt_tumor = int(np.count_nonzero(np.isin(result.target_mask, schema.tumor_fine_ids)))
                print(f"  [{profile}/{strength}] {tag}_increase  src_tb={tb}% +{tgt_tumor-src_tumor}px frac={result.changed_area_fraction:.3f}")
            except Exception as e:
                print(f"  [{profile}/{strength}] {tag}_increase FAILED: {e}")

            # ── Decrease ──
            try:
                target_frac = _target_decrease_fraction(decrease_config, strength)
                intent = EditIntent.from_mapping({
                    "primitive": "tumor_burden_decrease",
                    "reference_profile": profile,
                    "target_change_fraction": target_frac,
                    "seed": idx,
                })
                result = apply_tumor_burden_decrease(old_mask, schema, context, decrease_config, intent)

                src_rgb = save_rgb_mask(old_mask, OUT_DIR / f"{tag}_decrease_src.png")
                tgt_rgb = save_rgb_mask(result.target_mask, OUT_DIR / f"{tag}_decrease_tgt.png")
                save_change_region(result.change_region, OUT_DIR / f"{tag}_decrease_change.png")

                src_tumor = int(np.count_nonzero(np.isin(old_mask, schema.tumor_fine_ids)))
                tgt_tumor = int(np.count_nonzero(np.isin(result.target_mask, schema.tumor_fine_ids)))
                print(f"  [{profile}/{strength}] {tag}_decrease  src_tb={tb}% -{src_tumor-tgt_tumor}px frac={result.changed_area_fraction:.3f}")
            except Exception as e:
                print(f"  [{profile}/{strength}] {tag}_decrease FAILED: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
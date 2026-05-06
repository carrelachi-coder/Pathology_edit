"""Visualize boundary_infiltration on carefully-selected real masks.

Picks masks with meaningful tumor + stroma boundary structure,
applies boundary_infiltration at mild/moderate strengths,
and saves a clear comparison grid.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from phase3_mask_edit.core.config import load_recipe
from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.core.mask_io import id_to_rgb
from phase3_mask_edit.generic.boundary import apply_boundary_infiltration
from phase3_mask_edit.generic.tumor_burden import PrimitiveExecutionError

from dataset_config.unified_labels import COARSE_LABELS

DATASETS_DIR = Path("edit_datasets")
RECIPE_PATH = Path("phase3_mask_edit/recipes/generic.yaml")
OUTPUT_DIR = Path("phase3_mask_edit/previews/boundary_infiltration")


def _find_good_masks(dataset: str, schema: MaskProfileSchema, max_count: int, scan_limit: int = 200):
    """Find masks with good tumor/stroma boundary structure."""
    mask_dir = DATASETS_DIR / dataset / "tissue_masks"
    files = sorted(mask_dir.glob("*.png"))
    results = []

    for f in files[:scan_limit]:
        img = np.asarray(Image.open(f), dtype=np.int64)
        tumor_count = int(np.count_nonzero(np.isin(img, schema.tumor_fine_ids)))
        total_tissue = int(np.count_nonzero(img > 0))
        tissue_fraction = total_tissue / img.size

        # Want: meaningful tumor (5-40% of tissue), stroma/other to infiltrate,
        # and enough tissue coverage (not mostly background).
        if tumor_count < 0.05 * total_tissue or tumor_count > 0.40 * total_tissue:
            continue
        if tissue_fraction < 0.4:
            continue

        non_tumor_editable = total_tissue - tumor_count
        # Ensure tumor has a meaningful boundary (not just isolated islands).
        from scipy import ndimage
        struct = np.ones((3, 3), dtype=bool)
        tumor_dilated = ndimage.binary_dilation(np.isin(img, schema.tumor_fine_ids), structure=struct)
        boundary_contact = int(np.count_nonzero(tumor_dilated & ~np.isin(img, schema.tumor_fine_ids) & (img > 0)))
        if boundary_contact < 50:
            continue

        results.append((f, tumor_count, img, boundary_contact))
        if len(results) >= max_count:
            break

    return results


def _overlay_change(rgb: np.ndarray, change: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = rgb.copy()
    red = np.array([220, 40, 40], dtype=np.uint8)
    mask3 = np.stack([change]*3, axis=-1)
    out = np.where(mask3,
                   (rgb.astype(np.float32) * (1-alpha) + red.astype(np.float32) * alpha).astype(np.uint8),
                   out)
    return out


def _panel_label(draw, y, w, text, font=None):
    draw.text((4, y), text, fill=(0, 0, 0))


def main():
    recipe = load_recipe(RECIPE_PATH)
    infiltration_config = next(
        p for p in recipe["primitives"] if p["name"] == "boundary_infiltration"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("BCSS", "mild"),
        ("BCSS", "moderate"),
        ("ORCA", "mild"),
        ("PANDA", "moderate"),
    ]

    for dataset, strength in datasets:
        schema = MaskProfileSchema.from_reference_profile(dataset)
        masks = _find_good_masks(dataset, schema, max_count=2, scan_limit=500)

        if not masks:
            print(f"[{dataset}] No suitable masks found.")
            continue

        for idx, (path, tumor_px, mask, boundary_px) in enumerate(masks):
            context = MaskEditContext.from_mask(mask, schema)

            intent = EditIntent.from_mapping({
                "primitive": "boundary_infiltration",
                "reference_profile": dataset,
                "strength": strength,
                "seed": idx * 13 + 7,
            })

            try:
                result = apply_boundary_infiltration(
                    mask, schema, context, infiltration_config, intent,
                )
            except PrimitiveExecutionError as e:
                print(f"  [{dataset}] {path.name}: SKIPPED ({e})")
                continue

            src_rgb = id_to_rgb(mask)
            tgt_rgb = id_to_rgb(result.target_mask)
            overlay_rgb = _overlay_change(src_rgb, result.change_region)

            h, w = src_rgb.shape[:2]
            gap = 6
            total_w = 3 * w + 2 * gap

            # Build composite image with labels
            composite = np.full((h + 50, total_w, 3), 240, dtype=np.uint8)
            composite[:h, :w] = src_rgb
            composite[:h, w+gap:2*w+gap] = tgt_rgb
            composite[:h, 2*w+2*gap:] = overlay_rgb

            img = Image.fromarray(composite)
            draw = ImageDraw.Draw(img)

            # Panel headers
            draw.text((4, h+4), "Source", fill=(0,0,0))
            draw.text((w+gap+4, h+4), "Target", fill=(0,0,0))
            draw.text((2*w+2*gap+4, h+4), "Change overlay", fill=(0,0,0))

            # Composition stats
            tumor_ids = schema.tumor_fine_ids
            src_tumor = np.count_nonzero(np.isin(mask, tumor_ids))
            tgt_tumor = np.count_nonzero(np.isin(result.target_mask, tumor_ids))

            title = (
                f"{dataset} {strength} | {path.stem[:30]} | "
                f"tumor: {src_tumor}→{tgt_tumor} (+{tgt_tumor-src_tumor}px) | "
                f"Δfrac={result.changed_area_fraction:.3f} | "
                f"protrusions={result.ops_log['boundary_protrusion_pixels']} "
                f"islands={result.ops_log['island_pixels']}"
            )
            draw.text((4, h+24), title, fill=(50,50,50))

            out_name = f"{dataset}_{strength}_{idx}_infiltration.png"
            img.save(OUTPUT_DIR / out_name)
            print(f"  [{dataset}/{strength}] Saved {out_name}  "
                  f"(+{tgt_tumor-src_tumor}px tumor, Δ={result.changed_area_fraction:.3f})")

    # Also generate a legend showing the color map
    legend_h = 200
    legend_w = 400
    legend = np.full((legend_h, legend_w, 3), 240, dtype=np.uint8)
    from dataset_config.unified_labels import UNIFIED_COLOR_MAP
    img = Image.fromarray(legend)
    draw = ImageDraw.Draw(img)
    y = 10
    for label_name, fine_ids in schema.label_to_fine_ids.items():
        fid = fine_ids[0] if fine_ids else 0
        color = tuple(int(c) for c in UNIFIED_COLOR_MAP.get(fid, (128,128,128)))
        draw.rectangle([10, y, 30, y+14], fill=color)
        draw.text((35, y), f"{label_name} (id={fid})", fill=(0,0,0))
        y += 18
    img.save(OUTPUT_DIR / "legend.png")
    print("Saved legend.png")


if __name__ == "__main__":
    main()
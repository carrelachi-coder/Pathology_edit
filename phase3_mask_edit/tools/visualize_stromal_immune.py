"""Generate synthetic preview panels for stromal immune infiltration."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from phase3_mask_edit.core.config import load_recipe
from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.generic.immune import apply_stromal_immune_infiltration


OUT_DIR = Path("phase3_mask_edit/previews/stromal_immune/synthetic_tumor_modes")
RECIPE_PATH = Path("phase3_mask_edit/recipes/generic.yaml")

COLORS = {
    0: (238, 238, 238),  # Background
    1: (214, 52, 86),    # Tumor
    2: (92, 158, 106),   # Stroma
    3: (116, 83, 153),   # Necrosis
    4: (45, 111, 196),   # Immune infiltrate
    5: (238, 186, 85),   # Normal epithelium
    6: (42, 160, 160),   # Blood vessel
    7: (164, 164, 164),  # Other tissue
}


def _primitive(recipe, name):
    return next(p for p in recipe["primitives"] if p["name"] == name)


def _rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    for label, color in COLORS.items():
        rgb[mask == label] = color
    return rgb


def _save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _base_stroma(size: int = 256) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.int64)
    mask[18:-18, 18:-18] = 2
    return mask


def _no_tumor_mask() -> np.ndarray:
    return _base_stroma()


def _small_tumor_mask() -> np.ndarray:
    mask = _base_stroma()
    mask[30:50, 30:50] = 1
    return mask


def _normal_tumor_mask() -> np.ndarray:
    mask = _base_stroma()
    yy, xx = np.mgrid[: mask.shape[0], : mask.shape[1]]
    tumor = (yy - 128) ** 2 + (xx - 128) ** 2 <= 54**2
    mask[tumor] = 1
    return mask


def _panel(case_name: str, old_mask: np.ndarray, result, out_dir: Path) -> None:
    src_rgb = _rgb(old_mask)
    target_rgb = _rgb(result.target_mask)
    change = np.zeros_like(src_rgb)
    change[result.change_region] = (255, 255, 255)

    overlay = src_rgb.copy()
    overlay[result.change_region] = (
        0.35 * overlay[result.change_region] + 0.65 * np.array([45, 111, 196])
    ).astype(np.uint8)

    _save_png(out_dir / "src_mask_rgb.png", src_rgb)
    _save_png(out_dir / "change_region.png", change)
    _save_png(out_dir / "target_mask_rgb.png", target_rgb)
    _save_png(out_dir / "overlay.png", overlay)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    panels = (
        ("Source", src_rgb),
        ("Change", change),
        ("Overlay", overlay),
        ("Target", target_rgb),
    )
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(case_name)
    fig.savefig(out_dir / "panel.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    recipe = load_recipe(RECIPE_PATH)
    primitive = _primitive(recipe, "stromal_immune_infiltration")
    schema = MaskProfileSchema.from_reference_profile("BCSS")

    cases = [
        {
            "name": "no_tumor_random_patchy",
            "mask": _no_tumor_mask(),
            "target_change_fraction": 0.12,
            "seed": 17,
        },
        {
            "name": "small_tumor_blended_priority",
            "mask": _small_tumor_mask(),
            "target_change_fraction": 0.16,
            "seed": 19,
        },
        {
            "name": "normal_tumor_peritumoral",
            "mask": _normal_tumor_mask(),
            "target_change_fraction": 0.16,
            "seed": 23,
        },
    ]

    summary = []
    overview_images = []
    for case in cases:
        old_mask = case["mask"]
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "stromal_immune_infiltration",
                "reference_profile": "BCSS",
                "target_change_fraction": case["target_change_fraction"],
                "parameters": {
                    "max_stromal_immune_components": 6,
                    "min_stromal_immune_component_area_px": 400,
                },
                "seed": case["seed"],
            }
        )
        result = apply_stromal_immune_infiltration(
            old_mask,
            schema,
            context,
            primitive,
            intent,
        )

        case_dir = OUT_DIR / case["name"]
        case_dir.mkdir(parents=True, exist_ok=True)
        _panel(case["name"], old_mask, result, case_dir)

        spatial = result.ops_log["spatial"]
        metadata = {
            "case": case["name"],
            "target_change_fraction": case["target_change_fraction"],
            "changed_area_fraction": result.changed_area_fraction,
            "changed_stroma_fraction": result.ops_log["changed_stroma_fraction"],
            "changed_stroma_immune_fraction": (
                result.ops_log["changed_stroma_immune_fraction"]
            ),
            "selected_pixels": result.selected_pixels,
            "tumor_mode": spatial["tumor_mode"],
            "tumor_fraction": spatial["tumor_fraction"],
            "selected_components": spatial["selected_components"],
            "active_weights": spatial["active_weights"],
            "hard_distance_limit_px": spatial["hard_distance_limit_px"],
            "panel": str(case_dir / "panel.png"),
        }
        (case_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        summary.append(metadata)
        overview_images.append((case["name"], case_dir / "panel.png"))

    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(len(overview_images), 1, figsize=(14, 11))
    for ax, (name, panel_path) in zip(axes, overview_images):
        ax.imshow(np.asarray(Image.open(panel_path)))
        ax.set_title(name)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "overview.png", dpi=150)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

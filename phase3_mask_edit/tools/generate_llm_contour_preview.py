"""Generate LLM contour proposal RGB previews for manual inspection."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from phase3_mask_edit.backends.llm_preview import (
    add_coordinate_grid_overlay,
    id_mask_to_llm_preview_rgb,
)
from phase3_mask_edit.core.mask_io import load_id_mask, load_rgb_mask


def generate_llm_contour_preview(
    mask_path: Path,
    output_dir: Path,
    *,
    grid_spacing_px: int = 64,
) -> dict[str, Path]:
    """Save source RGB and grid-overlaid LLM preview images."""

    mask = _load_mask_auto(mask_path)
    rgb = id_mask_to_llm_preview_rgb(mask)
    grid = add_coordinate_grid_overlay(rgb, grid_spacing_px=grid_spacing_px)

    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / "source_mask_rgb.png"
    grid_path = output_dir / "source_mask_rgb_grid.png"
    Image.fromarray(rgb, mode="RGB").save(source_path)
    Image.fromarray(grid, mode="RGB").save(grid_path)
    return {"source_mask_rgb": source_path, "source_mask_rgb_grid": grid_path}


def _load_mask_auto(mask_path: Path):
    image = Image.open(mask_path)
    if image.mode in {"RGB", "RGBA"}:
        return load_rgb_mask(mask_path)
    return load_id_mask(mask_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LLM contour proposal preview images from an id mask."
    )
    parser.add_argument(
        "--mask",
        required=True,
        type=Path,
        help="Input id-mask PNG or RGB mask preview using the unified palette.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output directory.")
    parser.add_argument("--grid-spacing-px", type=int, default=64)
    args = parser.parse_args()

    paths = generate_llm_contour_preview(
        args.mask,
        args.output,
        grid_spacing_px=args.grid_spacing_px,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

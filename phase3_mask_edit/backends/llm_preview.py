"""Preview helpers for LLM contour proposal inputs."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from dataset_config.unified_labels import COARSE_LABELS, FINE_TO_PARENT, UNIFIED_COLOR_MAP
from phase3_mask_edit.core.mask_io import MaskIOError


def id_mask_to_llm_preview_rgb(mask: np.ndarray) -> np.ndarray:
    """Render an id mask with the dataset-config unified palette.

    Fine tumor subtype ids are rendered through their coarse parent color so the
    LLM sees stable coarse tissue semantics across profiles.
    """

    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise MaskIOError(f"id mask must be 2D, got shape {arr.shape}.")

    rgb = np.full(arr.shape + (3,), 255, dtype=np.uint8)
    for fine_id, parent_id in FINE_TO_PARENT.items():
        color = UNIFIED_COLOR_MAP.get(parent_id, [255, 255, 255])
        rgb[arr == fine_id] = color
    return rgb


def llm_palette_legend() -> dict[str, list[int]]:
    """Return the coarse tissue palette exposed to LLM context JSON."""

    return {
        label: list(UNIFIED_COLOR_MAP.get(label_id, [255, 255, 255]))
        for label_id, label in COARSE_LABELS.items()
    }


def add_coordinate_grid_overlay(
    rgb: np.ndarray,
    *,
    grid_spacing_px: int = 64,
    line_color: tuple[int, int, int] = (235, 235, 235),
    text_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Overlay x/y ticks and a regular coordinate grid on an RGB preview."""

    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise MaskIOError(f"RGB preview must be (H, W, 3), got shape {arr.shape}.")
    if grid_spacing_px <= 0:
        raise ValueError("grid_spacing_px must be positive.")

    image = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image, mode="RGBA")
    width, height = image.size

    grid_rgba = (*line_color, 150)
    axis_rgba = (*text_color, 230)

    for x in range(0, width, grid_spacing_px):
        draw.line([(x, 0), (x, height - 1)], fill=grid_rgba, width=1)
        draw.text((min(x + 2, width - 1), 2), str(x), fill=axis_rgba)
    for y in range(0, height, grid_spacing_px):
        draw.line([(0, y), (width - 1, y)], fill=grid_rgba, width=1)
        draw.text((2, min(y + 2, height - 1)), str(y), fill=axis_rgba)

    draw.text((2, max(height - 12, 0)), "x ->", fill=axis_rgba)
    draw.text((max(width - 28, 0), 2), "y down", fill=axis_rgba)
    return np.asarray(image, dtype=np.uint8)


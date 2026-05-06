"""Mask I/O utilities for Phase 3 edit-time mask executor.

Reads/writes id masks, RGB masks, change regions and metadata using
PIL and NumPy — no cv2 dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dataset_config.unified_labels import UNIFIED_COLOR_MAP


class MaskIOError(ValueError):
    """Raised when a mask file cannot be loaded or saved."""


# ── id mask (uint8 PNG, values 0-15) ──────────────────────────────

def load_id_mask(path: str | Path) -> np.ndarray:
    """Load a 2D unified fine-id mask from a grayscale PNG.

    Returns (H, W) int64 array with values in [0, 15].
    """

    img = _load_grayscale(path)
    mask = img.astype(np.int64)
    if mask.ndim != 2:
        raise MaskIOError(f"id mask must be 2D, got shape {mask.shape}.")
    return mask


def save_id_mask(mask: np.ndarray, path: str | Path) -> Path:
    """Save a 2D id mask as uint8 grayscale PNG.

    Values are clipped to [0, 255]; for unified fine labels (0-15) this
    is lossless.
    """

    out = np.clip(np.asarray(mask), 0, 255).astype(np.uint8)
    if out.ndim != 2:
        raise MaskIOError(f"id mask must be 2D, got shape {mask.shape}.")
    return _save_grayscale(out, path)


# ── RGB mask (colour visualization) ────────────────────────────────

def load_rgb_mask(path: str | Path) -> np.ndarray:
    """Load an RGB tissue mask PNG and convert to id mask.

    Returns (H, W) int64 array with unified fine ids.
    """

    p = Path(path)
    if not p.exists():
        raise MaskIOError(f"mask file not found: {p}")
    img = Image.open(p)
    rgb = np.asarray(img)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise MaskIOError(f"RGB mask must be (H, W, 3), got shape {rgb.shape}.")
    return rgb_to_id(rgb)


def save_rgb_mask(mask: np.ndarray, path: str | Path) -> Path:
    """Save a 2D id mask as a colour PNG using UNIFIED_COLOR_MAP.

    Unknown ids are rendered as white (255, 255, 255).
    """

    rgb = id_to_rgb(mask)
    img = Image.fromarray(rgb, mode="RGB")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path


def id_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert a 2D id mask to an (H, W, 3) uint8 RGB array."""

    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise MaskIOError(f"id mask must be 2D, got shape {arr.shape}.")

    h, w = arr.shape
    rgb = np.full((h, w, 3), 255, dtype=np.uint8)

    for id_val, color in UNIFIED_COLOR_MAP.items():
        rgb[arr == id_val] = color

    return rgb


def rgb_to_id(rgb: np.ndarray) -> np.ndarray:
    """Convert an (H, W, 3) uint8 RGB array to a 2D id mask.

    Builds a lookup table from UNIFIED_COLOR_MAP so the conversion is
    O(H*W) with a single hash per pixel.
    """

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise MaskIOError(f"RGB mask must be (H, W, 3), got shape {rgb.shape}.")

    encoded = (
        rgb[:, :, 0].astype(np.int64) * 65536
        + rgb[:, :, 1].astype(np.int64) * 256
        + rgb[:, :, 2].astype(np.int64)
    )

    lut: dict[int, int] = {}
    for id_val, color in UNIFIED_COLOR_MAP.items():
        key = color[0] * 65536 + color[1] * 256 + color[2]
        lut[key] = id_val

    result = np.zeros(rgb.shape[:2], dtype=np.int64)
    for key, id_val in lut.items():
        result[encoded == key] = id_val

    return result


# ── change region (boolean / uint8 PNG) ────────────────────────────

def load_change_region(path: str | Path) -> np.ndarray:
    """Load a change region from grayscale PNG.

    Any pixel > 0 is considered changed.  Returns (H, W) bool array.
    """

    img = _load_grayscale(path)
    return img > 0


def save_change_region(change_region: np.ndarray, path: str | Path) -> Path:
    """Save a change region as uint8 PNG.

    Boolean or numeric input: True / >0 becomes 255, else 0.
    """

    arr = np.asarray(change_region)
    out = np.where(arr, 255, 0).astype(np.uint8)
    if out.ndim != 2:
        raise MaskIOError(f"change region must be 2D, got shape {arr.shape}.")
    return _save_grayscale(out, path)


# ── metadata JSON ──────────────────────────────────────────────────

def load_metadata(path: str | Path) -> dict[str, Any]:
    """Load metadata from a JSON file."""

    p = Path(path)
    if not p.exists():
        raise MaskIOError(f"metadata file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_metadata(metadata: dict[str, Any], path: str | Path) -> Path:
    """Save metadata dict to a JSON file.

    NumPy arrays in the dict are converted to lists so JSON can
    serialize them.
    """

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    serializable = _make_json_serializable(metadata)
    with p.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    return p


# ── convenience: save full primitive edit output ───────────────────

def save_edit_output(
    src_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    ops_log: dict[str, Any],
    output_dir: str | Path,
    warnings: tuple[str, ...] = (),
) -> dict[str, Path]:
    """Write the standard Phase 3 primitive output bundle to a directory.

    Creates:
      - src_mask.png          (id mask, grayscale)
      - tar_mask.png          (id mask, grayscale)
      - change_region.png     (bool → 0/255 grayscale)
      - src_mask_rgb.png      (colour visualization)
      - tar_mask_rgb.png      (colour visualization)
      - metadata.json         (ops_log + warnings + pixel counts)

    Returns a dict mapping each key to its output Path.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    paths["src_mask"] = save_id_mask(src_mask, out / "src_mask.png")
    paths["tar_mask"] = save_id_mask(target_mask, out / "tar_mask.png")
    paths["change_region"] = save_change_region(change_region, out / "change_region.png")
    paths["src_mask_rgb"] = save_rgb_mask(src_mask, out / "src_mask_rgb.png")
    paths["tar_mask_rgb"] = save_rgb_mask(target_mask, out / "tar_mask_rgb.png")

    metadata = {
        "ops_log": ops_log,
        "warnings": list(warnings),
        "src_mask_pixels": int(np.count_nonzero(src_mask)),
        "tar_mask_pixels": int(np.count_nonzero(target_mask)),
        "change_region_pixels": int(np.count_nonzero(change_region)),
        "changed_area_fraction": float(np.count_nonzero(change_region)) / int(change_region.size),
    }
    paths["metadata"] = save_metadata(metadata, out / "metadata.json")

    return paths


# ── internal helpers ───────────────────────────────────────────────

def _load_grayscale(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise MaskIOError(f"mask file not found: {p}")
    img = Image.open(p)
    if img.mode == "RGB":
        arr = np.asarray(img.convert("L"))
    elif img.mode == "RGBA":
        arr = np.asarray(img.convert("L"))
    else:
        arr = np.asarray(img)
    if arr.ndim != 2:
        raise MaskIOError(f"expected 2D grayscale, got shape {arr.shape} from {p}.")
    return arr


def _save_grayscale(arr: np.ndarray, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr, mode="L")
    img.save(p)
    return p


def _make_json_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    return obj
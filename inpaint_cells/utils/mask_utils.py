"""
Shared mask utility functions for the cell filling pipeline.

Provides:
  - Constants for unified 16-class tissue labels and Embedding dimensions
  - Tissue/nuclei mask I/O (layered storage: separate tissue_mask.png + nuclei_mask.png)
  - RGB <-> class ID conversion for visualization
  - Overlay rendering

Phase 4.1 changes:
  - Replaced BCSS 22-class hardcoded constants with unified 16-class fine labels
  - Added Embedding dimension constants (AD-4: Embedding lookup replaces one-hot)
  - Simplified split_tissue_nuclei() -> direct file reads (AD-1: layered storage)
  - Removed to_onehot() (no longer needed with Embedding input)
  - Added load_tissue_mask() / load_nuclei_mask() for layered storage
"""

import os
import numpy as np
import cv2

# ============================================================
#  Import from unified dataset_config
# ============================================================
from dataset_config.unified_labels import (
    FINE_LABELS,
    NUM_FINE,
    UNIFIED_COLOR_MAP,
    CELL_CLASSES,
    CELL_COLOR_MAP,
    CELL_IDS,
    NUM_CELL_CLASSES,
    FULL_COLOR_MAP,
)

# ============================================================
#  Constants
# ============================================================

# Tissue: 16 unified fine classes (IDs 0-15)
NUM_TISSUE = NUM_FINE  # 16

# Nuclei: background(0) + 5 CellViT classes = 6 total embedding rows
NUM_NUCLEI = NUM_CELL_CLASSES + 1  # 6

# Cancer types: 6 datasets
NUM_CANCER_TYPES = 6

# Raw nuclei IDs in mask files (101-105) -> internal indices (1-5)
NUCLEI_CLASSES = CELL_IDS  # [101, 102, 103, 104, 105]

NUCLEI_RAW_TO_INDEX = {raw: i + 1 for i, raw in enumerate(NUCLEI_CLASSES)}
# {101: 1, 102: 2, 103: 3, 104: 4, 105: 5}

NUCLEI_INDEX_TO_RAW = {i + 1: raw for i, raw in enumerate(NUCLEI_CLASSES)}
# {1: 101, 2: 102, 3: 103, 4: 104, 5: 105}

# ----- AD-4: Embedding dimensions -----
TISSUE_EMB_DIM = 8
CELL_EMB_DIM = 4
CANCER_EMB_DIM = 4
PROBNET_IN_CH = TISSUE_EMB_DIM + CELL_EMB_DIM + 1 + CANCER_EMB_DIM  # 17

# Tissue and nuclei name maps (from unified labels)
TISSUE_NAMES = FINE_LABELS

NUCLEI_NAMES = {
    0: 'background',
    1: 'neoplastic',
    2: 'inflammatory',
    3: 'connective',
    4: 'dead',
    5: 'epithelial',
}


# ============================================================
#  Color Maps for visualization
# ============================================================

# Tissue RGB map: unified 16 classes
TISSUE_RGB_MAP = {k: v for k, v in UNIFIED_COLOR_MAP.items()}

# Nuclei RGB map: index-based (0-5)
NUCLEI_RGB = {
    0: [0, 0, 0],
    1: [255, 0, 0],      # neoplastic
    2: [0, 255, 0],      # inflammatory
    3: [0, 80, 255],     # connective
    4: [255, 255, 0],    # dead
    5: [255, 0, 255],    # epithelial
}

# Combined color map (tissue IDs 0-15 + raw nuclei IDs 101-105)
COLOR_MAP = {**UNIFIED_COLOR_MAP, **CELL_COLOR_MAP}

# RGB -> value lookup table
_rgb_to_val = {}
for _val, _rgb in COLOR_MAP.items():
    _key = _rgb[0] * 65536 + _rgb[1] * 256 + _rgb[2]
    _rgb_to_val[_key] = _val

_val_to_rgb = {v: rgb for v, rgb in COLOR_MAP.items()}


# ============================================================
#  Layered Storage I/O (AD-1)
# ============================================================

def load_tissue_mask(path):
    """
    Load tissue mask from a uint8 PNG file.

    Args:
        path: path to tissue_mask.png (uint8, values 0-15)

    Returns:
        numpy array (H, W), int64, values in [0, 15]
    """
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot load tissue mask: {path}")
    return mask.astype(np.int64)


def load_nuclei_mask(path, remap=True):
    """
    Load nuclei mask from a uint8 PNG file.

    Args:
        path: path to nuclei_mask.png (uint8, values 0/101-105)
        remap: if True, remap raw IDs (101-105) to internal indices (1-5)

    Returns:
        numpy array (H, W), int64
        - If remap=True: values in [0, 5] (0=background, 1-5=cell types)
        - If remap=False: values in {0, 101, 102, 103, 104, 105}
    """
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot load nuclei mask: {path}")
    mask = mask.astype(np.int64)

    if remap:
        remapped = np.zeros_like(mask)
        for raw_id, idx in NUCLEI_RAW_TO_INDEX.items():
            remapped[mask == raw_id] = idx
        return remapped

    return mask


def save_nuclei_mask(mask, path, from_index=True):
    """
    Save nuclei mask to a uint8 PNG file.

    Args:
        mask: numpy array (H, W), nuclei mask
        path: output path
        from_index: if True, input uses internal indices (0-5), convert to raw IDs (0/101-105)
    """
    if from_index:
        output = np.zeros_like(mask, dtype=np.uint8)
        for idx, raw_id in NUCLEI_INDEX_TO_RAW.items():
            output[mask == idx] = raw_id
    else:
        output = mask.astype(np.uint8)

    cv2.imwrite(path, output)


# ============================================================
#  RGB <-> Class conversion (for legacy / visualization)
# ============================================================

def rgb_to_class_map(rgb_img):
    """RGB image -> class value map (H, W), int64."""
    encoded = (rgb_img[:, :, 0].astype(np.int64) * 65536
               + rgb_img[:, :, 1].astype(np.int64) * 256
               + rgb_img[:, :, 2].astype(np.int64))
    result = np.zeros(rgb_img.shape[:2], dtype=np.int64)
    for key, val in _rgb_to_val.items():
        result[encoded == key] = val
    return result


def class_map_to_rgb(class_map):
    """Class value map (H, W) -> RGB image."""
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for val, color in _val_to_rgb.items():
        rgb[class_map == val] = color
    return rgb


def split_tissue_nuclei(tissue_path, nuclei_path):
    """
    Load tissue and nuclei layers from separate files (AD-1 layered storage).

    This replaces the old split_tissue_nuclei() which used EDT to infer
    tissue under nuclei pixels from a merged class map.

    Args:
        tissue_path: path to tissue_mask.png (uint8, 0-15)
        nuclei_path: path to nuclei_mask.png (uint8, 0/101-105)

    Returns:
        tissue: (H, W) int64, values [0, 15]
        nuclei: (H, W) int64, values [0, 5] (remapped)
    """
    tissue = load_tissue_mask(tissue_path)
    nuclei = load_nuclei_mask(nuclei_path, remap=True)
    return tissue, nuclei


# ============================================================
#  Visualization Helpers
# ============================================================

def index_to_rgb(index_map, color_map):
    """Generic index map -> RGB visualization."""
    h, w = index_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in color_map.items():
        rgb[index_map == idx] = color
    return rgb


def overlay(tissue_map, nuclei_map):
    """
    Overlay tissue and nuclei layers into a single RGB image.
    Nuclei pixels overwrite tissue pixels.

    Args:
        tissue_map: (H, W) int, tissue fine IDs (0-15)
        nuclei_map: (H, W) int, nuclei internal indices (0-5)

    Returns:
        (H, W, 3) uint8 RGB
    """
    tissue_rgb = index_to_rgb(tissue_map, TISSUE_RGB_MAP)
    nuc_rgb = index_to_rgb(nuclei_map, NUCLEI_RGB)
    result = tissue_rgb.copy()
    result[nuclei_map > 0] = nuc_rgb[nuclei_map > 0]
    return result

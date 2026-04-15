"""
Unified Hierarchical Label Space for Multi-Dataset Pathology Editing.

Two-layer label system:
  - Coarse level: 8 classes shared across all datasets
  - Fine level: 16 classes (coarse + dataset-specific sub-types)

Each fine class is a strict subset of its parent coarse class:
    coarse(x) = FINE_TO_PARENT[fine(x)]

Reference: .claude/Dataset.md "Label Space Design - Hierarchical Tissue Embedding (HTE)"
"""

from typing import Dict, List, Tuple

# ──────────────────────────────────────────────
# Coarse Level: 8 classes (all datasets share)
# ──────────────────────────────────────────────
COARSE_LABELS: Dict[int, str] = {
    0: "Background",
    1: "Tumor",
    2: "Stroma",
    3: "Necrosis",
    4: "Immune infiltrate",
    5: "Normal epithelium",
    6: "Blood vessel",
    7: "Other tissue",
}

NUM_COARSE = len(COARSE_LABELS)  # 8

# ──────────────────────────────────────────────
# Fine Level: 16 classes
#   IDs 0-7 mirror coarse (self-parent)
#   IDs 8-15 are dataset-specific sub-types
# ──────────────────────────────────────────────
FINE_LABELS: Dict[int, str] = {
    # === Coarse-inherited (same as coarse 0-7) ===
    0:  "Background",
    1:  "Tumor",
    2:  "Stroma",
    3:  "Necrosis",
    4:  "Immune infiltrate",
    5:  "Normal epithelium",
    6:  "Blood vessel",
    7:  "Other tissue",
    # === PANDA (Prostate) fine sub-types of Tumor ===
    8:  "Gleason 3",
    9:  "Gleason 4",
    10: "Gleason 5",
    # === GlaS (Colorectal) fine sub-types of Tumor ===
    11: "Adenomatous gland",
    12: "Moderately differentiated",
    13: "Poorly differentiated",
    # === BCSS (Breast) fine sub-types of Tumor ===
    14: "DCIS",
    15: "Angioinvasion",
}

NUM_FINE = len(FINE_LABELS)  # 16

# ──────────────────────────────────────────────
# Fine -> Coarse parent mapping
# ──────────────────────────────────────────────
FINE_TO_PARENT: Dict[int, int] = {
    0:  0,   # Background   -> Background
    1:  1,   # Tumor        -> Tumor
    2:  2,   # Stroma       -> Stroma
    3:  3,   # Necrosis     -> Necrosis
    4:  4,   # Immune       -> Immune
    5:  5,   # Normal epi   -> Normal epi
    6:  6,   # Blood vessel -> Blood vessel
    7:  7,   # Other tissue -> Other tissue
    8:  1,   # Gleason 3    -> Tumor
    9:  1,   # Gleason 4    -> Tumor
    10: 1,   # Gleason 5    -> Tumor
    11: 1,   # Adenomatous  -> Tumor
    12: 1,   # Mod diff     -> Tumor
    13: 1,   # Poor diff    -> Tumor
    14: 1,   # DCIS         -> Tumor
    15: 1,   # Angioinvasion-> Tumor
}

# Coarse -> list of fine children
COARSE_TO_FINE: Dict[int, List[int]] = {
    0: [0],
    1: [1, 8, 9, 10, 11, 12, 13, 14, 15],
    2: [2],
    3: [3],
    4: [4],
    5: [5],
    6: [6],
    7: [7],
}

# ──────────────────────────────────────────────
# Unified 16-class Color Map (for visualization)
# ──────────────────────────────────────────────
UNIFIED_COLOR_MAP: Dict[int, List[int]] = {
    # --- Coarse-level colors ---
    0:  [30,  30,  30],    # Background        - very dark gray
    1:  [180, 60,  60],    # Tumor             - muted red
    2:  [60,  150, 60],    # Stroma            - muted green
    3:  [60,  60,  180],   # Necrosis          - blue
    4:  [140, 60,  180],   # Immune infiltrate - purple
    5:  [180, 180, 80],    # Normal epithelium - yellow-green
    6:  [60,  140, 100],   # Blood vessel      - dark green
    7:  [170, 170, 170],   # Other tissue      - light gray
    # --- PANDA fine (Gleason patterns: red gradient) ---
    8:  [220, 120, 80],    # Gleason 3         - light coral
    9:  [200, 80,  50],    # Gleason 4         - orange-red
    10: [160, 40,  40],    # Gleason 5         - dark red
    # --- GlaS fine (differentiation: warm gradient) ---
    11: [200, 170, 100],   # Adenomatous gland - tan/gold
    12: [200, 140, 60],    # Moderately diff   - orange
    13: [190, 90,  40],    # Poorly diff       - dark orange
    # --- BCSS fine ---
    14: [80,  60,  150],   # DCIS              - indigo
    15: [190, 40,  40],    # Angioinvasion     - bright red
}

# ──────────────────────────────────────────────
# CellViT 5-class Nuclei Definitions
# ──────────────────────────────────────────────
CELL_CLASSES: Dict[int, str] = {
    101: "Neoplastic",
    102: "Inflammatory",
    103: "Connective",
    104: "Dead",
    105: "Epithelial",
}

CELL_COLOR_MAP: Dict[int, List[int]] = {
    101: [255, 0,   0],    # Neoplastic  - pure red
    102: [0,   255, 0],    # Inflammatory- pure green
    103: [0,   80,  255],  # Connective  - bright blue
    104: [255, 255, 0],    # Dead        - yellow
    105: [255, 0,   255],  # Epithelial  - magenta
}

CELL_IDS: List[int] = [101, 102, 103, 104, 105]
NUM_CELL_CLASSES = len(CELL_IDS)  # 5 (plus background = 6 channels)

# ──────────────────────────────────────────────
# Combined Color Map (tissue + cells)
# ──────────────────────────────────────────────
FULL_COLOR_MAP: Dict[int, List[int]] = {**UNIFIED_COLOR_MAP, **CELL_COLOR_MAP}

# ──────────────────────────────────────────────
# Convenience Sets
# ──────────────────────────────────────────────
ALL_FINE_IDS = set(range(NUM_FINE))           # {0, 1, ..., 15}
ALL_COARSE_IDS = set(range(NUM_COARSE))       # {0, 1, ..., 7}
ALL_TUMOR_FINE_IDS = set(COARSE_TO_FINE[1])   # {1, 8, 9, 10, 11, 12, 13, 14, 15}
NON_BIO_IDS = {0}                             # Background only in unified space

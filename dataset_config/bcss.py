"""
BCSS (Breast Cancer Semantic Segmentation) Dataset Configuration.

Cancer type: Breast cancer
Original labels: 22 classes (BCSS original annotation)
Fine sub-types: DCIS (14), Angioinvasion (15)
Label granularity: Fine

Reference: .claude/Dataset.md "BCSS (Breast Cancer)"
"""

from .registry import DatasetConfig

# ──────────────────────────────────────────────
# BCSS Original 22-class label system
# ──────────────────────────────────────────────
_ORIGINAL_LABEL_MAP = {
    0:  "outside_roi",
    1:  "tumor",
    2:  "stroma",
    3:  "lymphocytic_infiltrate",
    4:  "necrosis_or_debris",
    5:  "glandular_secretions",
    6:  "blood",
    7:  "exclude",
    8:  "metaplasia_NOS",
    9:  "fat",
    10: "plasma_cells",
    11: "other_immune_infiltrate",
    12: "mucoid_material",
    13: "normal_acinus_or_duct",
    14: "lymphatics",
    15: "undetermined",
    16: "nerve",
    17: "skin_adnexa",
    18: "blood_vessel",
    19: "angioinvasion",
    20: "dcis",
    21: "other",
}

_ORIGINAL_COLOR_MAP = {
    0:  [30,  30,  30],
    1:  [180, 60,  60],
    2:  [60,  150, 60],
    3:  [140, 60,  180],
    4:  [60,  60,  180],
    5:  [180, 180, 80],
    6:  [160, 40,  40],
    7:  [40,  40,  40],
    8:  [80,  150, 150],
    9:  [200, 170, 100],
    10: [180, 120, 150],
    11: [120, 120, 190],
    12: [100, 190, 190],
    13: [200, 140, 60],
    14: [140, 200, 100],
    15: [140, 140, 140],
    16: [200, 200, 130],
    17: [150, 80,  60],
    18: [60,  140, 100],
    19: [190, 40,  40],
    20: [80,  60,  150],
    21: [170, 170, 170],
}

# ──────────────────────────────────────────────
# Original -> Unified Coarse (22 -> 8)
# ──────────────────────────────────────────────
_TO_COARSE = {
    0:  0,  # outside_roi           -> Background
    1:  1,  # tumor                 -> Tumor
    2:  2,  # stroma                -> Stroma
    3:  4,  # lymphocytic_infiltrate-> Immune infiltrate
    4:  3,  # necrosis_or_debris    -> Necrosis
    5:  2,  # glandular_secretions  -> Stroma
    6:  7,  # blood                 -> Other tissue
    7:  0,  # exclude               -> Background
    8:  7,  # metaplasia_NOS        -> Other tissue
    9:  7,  # fat                   -> Other tissue
    10: 4,  # plasma_cells          -> Immune infiltrate
    11: 4,  # other_immune          -> Immune infiltrate
    12: 7,  # mucoid_material       -> Other tissue
    13: 5,  # normal_acinus_or_duct -> Normal epithelium
    14: 6,  # lymphatics            -> Blood vessel
    15: 0,  # undetermined          -> Background
    16: 7,  # nerve                 -> Other tissue
    17: 7,  # skin_adnexa           -> Other tissue
    18: 6,  # blood_vessel          -> Blood vessel
    19: 1,  # angioinvasion         -> Tumor (fine: 15)
    20: 1,  # dcis                  -> Tumor (fine: 14)
    21: 7,  # other                 -> Other tissue
}

# ──────────────────────────────────────────────
# Original -> Unified Fine (22 -> 16)
# Most same as coarse; DCIS and Angioinvasion get fine IDs
# ──────────────────────────────────────────────
_TO_FINE = {
    0:  0,   # outside_roi           -> Background
    1:  1,   # tumor                 -> Tumor
    2:  2,   # stroma                -> Stroma
    3:  4,   # lymphocytic_infiltrate-> Immune infiltrate
    4:  3,   # necrosis_or_debris    -> Necrosis
    5:  2,   # glandular_secretions  -> Stroma
    6:  7,   # blood                 -> Other tissue
    7:  0,   # exclude               -> Background
    8:  7,   # metaplasia_NOS        -> Other tissue
    9:  7,   # fat                   -> Other tissue
    10: 4,   # plasma_cells          -> Immune infiltrate
    11: 4,   # other_immune          -> Immune infiltrate
    12: 7,   # mucoid_material       -> Other tissue
    13: 5,   # normal_acinus_or_duct -> Normal epithelium
    14: 6,   # lymphatics            -> Blood vessel
    15: 0,   # undetermined          -> Background
    16: 7,   # nerve                 -> Other tissue
    17: 7,   # skin_adnexa           -> Other tissue
    18: 6,   # blood_vessel          -> Blood vessel
    19: 15,  # angioinvasion         -> Angioinvasion (fine, parent: Tumor)
    20: 14,  # dcis                  -> DCIS (fine, parent: Tumor)
    21: 7,   # other                 -> Other tissue
}

# ──────────────────────────────────────────────
# Reverse: coarse -> original IDs
# ──────────────────────────────────────────────
_COARSE_TO_ORIG = {
    0: [0, 7, 15],           # Background
    1: [1, 19, 20],          # Tumor (incl. angioinvasion, dcis)
    2: [2, 5],               # Stroma
    3: [4],                  # Necrosis
    4: [3, 10, 11],          # Immune infiltrate
    5: [13],                 # Normal epithelium
    6: [14, 18],             # Blood vessel
    7: [6, 8, 9, 12, 16, 17, 21],  # Other tissue
}

# ──────────────────────────────────────────────
# Editing Configuration (all IDs in unified fine space)
# ──────────────────────────────────────────────
CONFIG = DatasetConfig(
    name="BCSS",
    cancer_type="breast",

    original_label_map=_ORIGINAL_LABEL_MAP,
    original_color_map=_ORIGINAL_COLOR_MAP,

    to_coarse_map=_TO_COARSE,
    to_fine_map=_TO_FINE,
    coarse_to_original=_COARSE_TO_ORIG,

    # In unified fine space
    tumor_ids=(1, 14, 15),           # Tumor, DCIS, Angioinvasion
    stroma_ids=(2,),
    necrosis_ids=(3,),
    immune_ids=(4,),
    normal_epi_ids=(5,),
    vessel_ids=(6,),
    skip_tissues=frozenset({0}),     # Background only in unified space

    available_edits=(
        "tumor_dilation",
        "tumor_shrink",
        "dcis_invasion",
        "necrosis_appear",
        "TIL_increase",
        "TIL_decrease",
        "stromal_desmoplasia",
    ),

    expansion_targets={
        "tumor_dilation":      (2, 4, 5, 7),       # stroma, immune, normal_epi, other
        "tumor_shrink":        (2,),                # shrink -> stroma
        "dcis_invasion":       (1, 2),              # DCIS -> invasive tumor + stroma change
        "necrosis_appear":     (3,),                # tumor center -> necrosis
        "TIL_increase":        (4,),                # stroma/tumor edge -> immune
        "TIL_decrease":        (2,),                # immune -> stroma
        "stromal_desmoplasia": (2,),                # tumor periphery -> stroma
    },

    has_cell_annotations=True,       # BCSS has original nuclei annotations
    cancer_type_index=0,

    data_dir="BCSS",
    prior_db_path="mask_edit/Prior_knowledge_of_pathology/bcss_prior_db.json",
    label_granularity="fine",
)

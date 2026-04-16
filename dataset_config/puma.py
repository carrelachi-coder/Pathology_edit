"""
PUMA (Melanoma) Dataset Configuration.

Cancer type: Melanoma (skin)
Original labels: 5 classes (coarse subset: no Immune, no Other)
Label granularity: Coarse
Cell annotations: Yes (10 types GeoJSON -> mapped to CellViT 5 types)

Reference: .claude/Dataset.md "PUMA (Melanoma)"

Note: PUMA is the smallest dataset (1,844 patches).
Epidermis (2.10%) and blood vessel (1.10%) are rare.
"""

from .registry import DatasetConfig

# ──────────────────────────────────────────────
# PUMA Original 5-class system (after preprocessing)
# Note: PUMA doesn't have Immune infiltrate or Other tissue classes
# ──────────────────────────────────────────────
_ORIGINAL_LABEL_MAP = {
    0: "background",
    1: "tumor",
    2: "stroma",
    3: "necrosis",
    5: "epidermis",         # maps to Normal epithelium
    6: "blood_vessel",
}

_ORIGINAL_COLOR_MAP = {
    0: [30,  30,  30],     # background
    1: [180, 60,  60],     # tumor          - red
    2: [60,  150, 60],     # stroma         - green
    3: [60,  60,  180],    # necrosis       - blue
    5: [180, 180, 80],     # epidermis      - yellow-green
    6: [60,  140, 100],    # blood vessel   - dark green
}

# ──────────────────────────────────────────────
# Original -> Unified
# PUMA uses a subset of coarse IDs: {0, 1, 2, 3, 5, 6}
# No Immune (4), no Other (7)
# ──────────────────────────────────────────────
_TO_COARSE = {
    0: 0,  # background  -> Background
    1: 1,  # tumor       -> Tumor
    2: 2,  # stroma      -> Stroma
    3: 3,  # necrosis    -> Necrosis
    5: 5,  # epidermis   -> Normal epithelium
    6: 6,  # blood_vessel-> Blood vessel
}

_TO_FINE = {
    0: 0,  # background  -> Background
    1: 1,  # tumor       -> Tumor (coarse-level only)
    2: 2,  # stroma      -> Stroma
    3: 3,  # necrosis    -> Necrosis
    5: 5,  # epidermis   -> Normal epithelium
    6: 6,  # blood_vessel-> Blood vessel
}

_COARSE_TO_ORIG = {
    0: [0],     # Background
    1: [1],     # Tumor
    2: [2],     # Stroma
    3: [3],     # Necrosis
    4: [],      # Immune (not in PUMA)
    5: [5],     # Normal epithelium (epidermis)
    6: [6],     # Blood vessel
    7: [],      # Other tissue (not in PUMA)
}

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
CONFIG = DatasetConfig(
    name="PUMA",
    cancer_type="melanoma",

    original_label_map=_ORIGINAL_LABEL_MAP,
    original_color_map=_ORIGINAL_COLOR_MAP,

    to_coarse_map=_TO_COARSE,
    to_fine_map=_TO_FINE,
    coarse_to_original=_COARSE_TO_ORIG,

    # In unified fine space
    tumor_ids=(1,),
    stroma_ids=(2,),
    necrosis_ids=(3,),
    immune_ids=(),                    # not annotated in PUMA
    normal_epi_ids=(5,),              # epidermis
    vessel_ids=(6,),
    skip_tissues=frozenset({0}),

    available_edits=(
        "tumor_epidermal_invasion",
        "epidermis_ulceration",
        "tumor_regression",
        "necrosis_appear",
        "perivascular_invasion",
    ),

    expansion_targets={
        "tumor_epidermal_invasion": (5, 2),       # tumor -> epidermis (partial), stroma
        "epidermis_ulceration":     (1,),          # epidermis -> completely replaced by tumor
        "tumor_regression":         (2,),          # tumor shrink -> stroma
        "necrosis_appear":          (3,),          # tumor center -> necrosis
        "perivascular_invasion":    (6,),          # tumor -> toward blood vessel
    },

    has_cell_annotations=True,        # 10-type GeoJSON -> CellViT 5-type mapping
    cancer_type_index=4,

    data_dir="PUMA",
    prior_db_path="mask_edit/Prior_knowledge_of_pathology/puma_prior_db.json",
    label_granularity="coarse",
)

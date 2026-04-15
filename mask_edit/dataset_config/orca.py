"""
ORCA (Oral Squamous Cell Carcinoma) Dataset Configuration.

Cancer type: Oral SCC (Head & Neck)
Original labels: 3 classes (BG / Tumor / Non-tumor tissue)
Label granularity: Coarse (minimal — only tumor boundary edits)

Source: Martino et al., Appl. Sci. 2020
  - 196 cores from TCGA head/neck OSCC slides
  - Pixel-level annotation by two board-certified pathologists
  - 29,606 patches (512x512 at 40x)

Reference: .claude/Dataset.md "ORCA (Oral SCC)"
"""

from .registry import DatasetConfig

# ──────────────────────────────────────────────
# ORCA Original 3-class system
# ──────────────────────────────────────────────
_ORIGINAL_LABEL_MAP = {
    0: "non_tissue",           # background (black)
    1: "tissue_non_carcinoma", # non-tumor tissue (gray)
    2: "carcinoma",            # tumor (white)
}

_ORIGINAL_COLOR_MAP = {
    0: [30,  30,  30],     # non_tissue     - dark gray
    1: [170, 170, 170],    # non-carcinoma  - light gray
    2: [180, 60,  60],     # carcinoma      - red
}

# ──────────────────────────────────────────────
# Original -> Unified
# ORCA has no stroma/necrosis/immune/epi/vessel distinction
# Non-carcinoma tissue -> Other tissue (7) since we can't differentiate
# ──────────────────────────────────────────────
_TO_COARSE = {
    0: 0,  # non_tissue           -> Background
    1: 7,  # tissue_non_carcinoma -> Other tissue
    2: 1,  # carcinoma            -> Tumor
}

_TO_FINE = {
    0: 0,  # non_tissue           -> Background
    1: 7,  # tissue_non_carcinoma -> Other tissue
    2: 1,  # carcinoma            -> Tumor (coarse-level)
}

_COARSE_TO_ORIG = {
    0: [0],     # Background
    1: [2],     # Tumor (carcinoma)
    2: [],      # Stroma (not distinguished)
    3: [],      # Necrosis (not distinguished)
    4: [],      # Immune (not distinguished)
    5: [],      # Normal epithelium (not distinguished)
    6: [],      # Blood vessel (not distinguished)
    7: [1],     # Other tissue (all non-carcinoma tissue)
}

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
CONFIG = DatasetConfig(
    name="ORCA",
    cancer_type="oral_scc",

    original_label_map=_ORIGINAL_LABEL_MAP,
    original_color_map=_ORIGINAL_COLOR_MAP,

    to_coarse_map=_TO_COARSE,
    to_fine_map=_TO_FINE,
    coarse_to_original=_COARSE_TO_ORIG,

    # In unified fine space
    tumor_ids=(1,),
    stroma_ids=(),                    # not distinguished in ORCA
    necrosis_ids=(),
    immune_ids=(),
    normal_epi_ids=(),
    vessel_ids=(),
    skip_tissues=frozenset({0}),

    available_edits=(
        "tumor_invasion",
        "tumor_regression",
    ),

    expansion_targets={
        "tumor_invasion":   (7,),     # tumor -> other tissue (only option)
        "tumor_regression": (7,),     # tumor shrink -> other tissue
    },

    has_cell_annotations=False,       # need CellViT inference
    cancer_type_index=5,

    data_dir="ORCA",
    prior_db_path="mask_edit/Prior_knowledge_of_pathology/orca_prior_db.json",
    label_granularity="coarse",
)

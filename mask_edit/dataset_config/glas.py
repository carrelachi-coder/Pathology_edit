"""
GlaS (Gland Segmentation) Dataset Configuration.

Cancer type: Colorectal cancer
Original labels: Instance-level gland segmentation + patch-level grade
Fine sub-types: Adenomatous (11), Moderately diff (12), Poorly diff (13)
Label granularity: Fine

Reference: .claude/Dataset.md "GlaS (Colorectal Cancer)"

Note: GlaS original annotation is instance-level glands + patch-level grade.
During preprocessing, patch-level grade is transferred to instance-level:
  - healthy patch -> glands become fine 5 (Normal epithelium)
  - adenomatous patch -> glands become fine 11 (Adenomatous)
  - moderately diff patch -> glands become fine 12
  - poorly diff patch -> glands become fine 13
  - non-gland area -> fine 2 (Stroma)
"""

from .registry import DatasetConfig

# ──────────────────────────────────────────────
# GlaS labels after preprocessing
# (original is instance masks; this is the semantic mapping)
# ──────────────────────────────────────────────
_ORIGINAL_LABEL_MAP = {
    0: "background",
    1: "stroma",               # non-gland area (all tissue is valid in GlaS)
    2: "normal_gland",         # healthy patch glands
    3: "adenomatous_gland",    # adenomatous patch glands
    4: "moderately_diff",      # mod diff patch glands
    5: "poorly_diff",          # poor diff patch glands (merged mod-to-poor + poor)
}

_ORIGINAL_COLOR_MAP = {
    0: [30,  30,  30],     # background
    1: [60,  150, 60],     # stroma         - green
    2: [180, 180, 80],     # normal gland   - yellow-green
    3: [200, 170, 100],    # adenomatous    - tan/gold
    4: [200, 140, 60],     # mod diff       - orange
    5: [190, 90,  40],     # poor diff      - dark orange
}

# ──────────────────────────────────────────────
# Original -> Unified Coarse
# ──────────────────────────────────────────────
_TO_COARSE = {
    0: 0,  # background        -> Background
    1: 2,  # stroma            -> Stroma
    2: 5,  # normal_gland      -> Normal epithelium
    3: 1,  # adenomatous_gland -> Tumor
    4: 1,  # moderately_diff   -> Tumor
    5: 1,  # poorly_diff       -> Tumor
}

# ──────────────────────────────────────────────
# Original -> Unified Fine
# ──────────────────────────────────────────────
_TO_FINE = {
    0: 0,   # background        -> Background
    1: 2,   # stroma            -> Stroma
    2: 5,   # normal_gland      -> Normal epithelium
    3: 11,  # adenomatous_gland -> Adenomatous gland (fine, parent: Tumor)
    4: 12,  # moderately_diff   -> Moderately differentiated (fine, parent: Tumor)
    5: 13,  # poorly_diff       -> Poorly differentiated (fine, parent: Tumor)
}

# ──────────────────────────────────────────────
# Reverse: coarse -> original IDs
# ──────────────────────────────────────────────
_COARSE_TO_ORIG = {
    0: [0],          # Background
    1: [3, 4, 5],    # Tumor (adenomatous + mod + poor)
    2: [1],          # Stroma
    3: [],           # Necrosis (not present)
    4: [],           # Immune (not annotated)
    5: [2],          # Normal epithelium
    6: [],           # Blood vessel (not present)
    7: [],           # Other tissue (not present)
}

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
CONFIG = DatasetConfig(
    name="GlaS",
    cancer_type="colorectal",

    original_label_map=_ORIGINAL_LABEL_MAP,
    original_color_map=_ORIGINAL_COLOR_MAP,

    to_coarse_map=_TO_COARSE,
    to_fine_map=_TO_FINE,
    coarse_to_original=_COARSE_TO_ORIG,

    # In unified fine space
    tumor_ids=(11, 12, 13),           # Adenomatous, Mod diff, Poor diff
    stroma_ids=(2,),
    necrosis_ids=(),                  # not present
    immune_ids=(),                    # not annotated
    normal_epi_ids=(5,),
    vessel_ids=(),                    # not present
    skip_tissues=frozenset({0}),

    available_edits=(
        "normal_to_adenomatous",
        "adenoma_to_carcinoma",
        "grade_upgrade",
        "treatment_dedifferentiation",
        "tumor_gland_growth",
        "tumor_gland_regression",
    ),

    expansion_targets={
        "normal_to_adenomatous":      (11,),        # in-place: fine 5 -> 11
        "adenoma_to_carcinoma":       (12,),        # in-place: fine 11 -> 12
        "grade_upgrade":              (13,),        # in-place: fine 12 -> 13
        "treatment_dedifferentiation":(12,),        # in-place: fine 13 -> 12
        "tumor_gland_growth":         (2,),         # malignant gland -> cover stroma
        "tumor_gland_regression":     (2,),         # malignant gland shrink -> stroma
    },

    has_cell_annotations=False,       # need CellViT inference
    cancer_type_index=2,

    data_dir="GlaS",
    prior_db_path="mask_edit/Prior_knowledge_of_pathology/glas_prior_db.json",
    label_granularity="fine",
)

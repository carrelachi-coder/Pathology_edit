"""
PANDA (Prostate cANcer graDe Assessment) Dataset Configuration.

Cancer type: Prostate cancer
Original labels: 6 classes
Fine sub-types: Gleason 3 (8), Gleason 4 (9), Gleason 5 (10)
Label granularity: Fine

Reference: .claude/Dataset.md "PANDA (Prostate Cancer)"
"""

from .registry import DatasetConfig

# ──────────────────────────────────────────────
# PANDA Original 6-class label system
# ──────────────────────────────────────────────
_ORIGINAL_LABEL_MAP = {
    0: "background",
    1: "stroma",
    2: "benign_epithelium",
    3: "gleason_3",
    4: "gleason_4",
    5: "gleason_5",
}

_ORIGINAL_COLOR_MAP = {
    0: [30,  30,  30],     # background      - dark gray
    1: [60,  150, 60],     # stroma          - green
    2: [180, 180, 80],     # benign epi      - yellow-green
    3: [220, 120, 80],     # Gleason 3       - light coral
    4: [200, 80,  50],     # Gleason 4       - orange-red
    5: [160, 40,  40],     # Gleason 5       - dark red
}

# ──────────────────────────────────────────────
# Original -> Unified Coarse (6 -> 8)
# ──────────────────────────────────────────────
_TO_COARSE = {
    0: 0,  # background       -> Background
    1: 2,  # stroma           -> Stroma
    2: 5,  # benign_epithelium-> Normal epithelium
    3: 1,  # gleason_3        -> Tumor
    4: 1,  # gleason_4        -> Tumor
    5: 1,  # gleason_5        -> Tumor
}

# ──────────────────────────────────────────────
# Original -> Unified Fine (6 -> 16)
# Gleason patterns get fine IDs 8, 9, 10
# ──────────────────────────────────────────────
_TO_FINE = {
    0: 0,   # background       -> Background
    1: 2,   # stroma           -> Stroma
    2: 5,   # benign_epithelium-> Normal epithelium
    3: 8,   # gleason_3        -> Gleason 3 (fine, parent: Tumor)
    4: 9,   # gleason_4        -> Gleason 4 (fine, parent: Tumor)
    5: 10,  # gleason_5        -> Gleason 5 (fine, parent: Tumor)
}

# ──────────────────────────────────────────────
# Reverse: coarse -> original IDs
# ──────────────────────────────────────────────
_COARSE_TO_ORIG = {
    0: [0],          # Background
    1: [3, 4, 5],    # Tumor (all Gleason patterns)
    2: [1],          # Stroma
    3: [],           # Necrosis (not annotated)
    4: [],           # Immune (not annotated)
    5: [2],          # Normal epithelium
    6: [],           # Blood vessel (not annotated)
    7: [],           # Other tissue (not annotated)
}

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
CONFIG = DatasetConfig(
    name="PANDA",
    cancer_type="prostate",

    original_label_map=_ORIGINAL_LABEL_MAP,
    original_color_map=_ORIGINAL_COLOR_MAP,

    to_coarse_map=_TO_COARSE,
    to_fine_map=_TO_FINE,
    coarse_to_original=_COARSE_TO_ORIG,

    # In unified fine space
    tumor_ids=(8, 9, 10),             # Gleason 3/4/5
    stroma_ids=(2,),
    necrosis_ids=(),                  # not annotated
    immune_ids=(),                    # not annotated
    normal_epi_ids=(5,),              # benign epithelium
    vessel_ids=(),                    # not annotated
    skip_tissues=frozenset({0}),

    available_edits=(
        "gleason_upgrade_3to4",
        "gleason_upgrade_4to5",
        "gleason_downgrade_4to3",
        "tumor_volume_increase",
        "tumor_volume_decrease",
        "benign_to_gleason3",
        "benign_atrophy",
    ),

    expansion_targets={
        "tumor_volume_increase":   (2,),           # Gleason -> cover stroma
        "tumor_volume_decrease":   (2,),           # shrink -> stroma
        "gleason_upgrade_3to4":    (9,),           # in-place: fine 8 -> 9
        "gleason_upgrade_4to5":    (10,),          # in-place: fine 9 -> 10
        "gleason_downgrade_4to3":  (8,),           # in-place: fine 9 -> 8
        "benign_to_gleason3":      (8,),           # in-place: fine 5 -> 8
        "benign_atrophy":          (2,),           # in-place: fine 5 -> 2
    },

    has_cell_annotations=False,       # need CellViT inference
    cancer_type_index=1,

    data_dir="PANDA",
    prior_db_path="mask_edit/Prior_knowledge_of_pathology/panda_prior_db.json",
    label_granularity="fine",
)

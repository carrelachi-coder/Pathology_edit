"""
IGNITE (Lung Cancer) Dataset Configuration.

Cancer type: Lung cancer (adenocarcinoma / squamous / adenosquamous)
Original labels: 8 classes (coarse-level, maps directly to unified coarse)
Label granularity: Coarse

Note: IGNITE's original Erythrocytes(9) is mapped to Other tissue(7),
not Blood vessel(6), because erythrocytes/blood are not vascular structures.

Reference: .claude/Dataset.md "IGNITE (Lung Cancer)"
"""

from .registry import DatasetConfig

# ──────────────────────────────────────────────
# IGNITE labels (already mapped to coarse during preprocessing)
# The 8 classes correspond directly to the unified coarse IDs.
# Original IGNITE had additional classes (e.g., Erythrocytes=9)
# that were mapped during preprocessing.
# ──────────────────────────────────────────────
_ORIGINAL_LABEL_MAP = {
    0: "background",
    1: "tumor",
    2: "stroma",
    3: "necrosis",
    4: "immune_infiltrate",
    5: "normal_epithelium",
    6: "blood_vessel",
    7: "other_tissue",
}

_ORIGINAL_COLOR_MAP = {
    0: [30,  30,  30],     # background
    1: [180, 60,  60],     # tumor          - red
    2: [60,  150, 60],     # stroma         - green
    3: [60,  60,  180],    # necrosis       - blue
    4: [140, 60,  180],    # immune         - purple
    5: [180, 180, 80],     # normal epi     - yellow-green
    6: [60,  140, 100],    # blood vessel   - dark green
    7: [170, 170, 170],    # other tissue   - light gray
}

# ──────────────────────────────────────────────
# Original -> Unified (identity mapping, already coarse)
# ──────────────────────────────────────────────
_TO_COARSE = {i: i for i in range(8)}
_TO_FINE = {i: i for i in range(8)}   # coarse-only -> fine == coarse

_COARSE_TO_ORIG = {i: [i] for i in range(8)}

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
CONFIG = DatasetConfig(
    name="IGNITE",
    cancer_type="lung",

    original_label_map=_ORIGINAL_LABEL_MAP,
    original_color_map=_ORIGINAL_COLOR_MAP,

    to_coarse_map=_TO_COARSE,
    to_fine_map=_TO_FINE,
    coarse_to_original=_COARSE_TO_ORIG,

    # In unified fine space (same as coarse for IGNITE)
    tumor_ids=(1,),
    stroma_ids=(2,),
    necrosis_ids=(3,),
    immune_ids=(4,),
    normal_epi_ids=(5,),
    vessel_ids=(6,),
    skip_tissues=frozenset({0}),

    available_edits=(
        "tumor_invasion",
        "tumor_regression",
        "necrosis_appear",
        "TIL_increase",
        "TIL_decrease",
        "stromal_desmoplasia",
        "angiogenesis",
    ),

    expansion_targets={
        "tumor_invasion":      (2, 4, 5, 7),     # tumor -> stroma, immune, normal_epi, other
        "tumor_regression":    (2,),              # shrink -> stroma
        "necrosis_appear":     (3,),              # tumor center -> necrosis
        "TIL_increase":        (4,),              # stroma -> immune
        "TIL_decrease":        (2,),              # immune -> stroma
        "stromal_desmoplasia": (2,),              # tumor periphery -> stroma
        "angiogenesis":        (6,),              # stroma -> blood vessel
    },

    has_cell_annotations=False,       # need CellViT inference
    cancer_type_index=3,

    data_dir="IGNITE",
    prior_db_path="mask_edit/Prior_knowledge_of_pathology/ignite_prior_db.json",
    label_granularity="coarse",
)

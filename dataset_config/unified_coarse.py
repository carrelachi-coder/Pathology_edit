"""Organ-agnostic unified coarse mask profile.

This profile is for masks produced directly in the shared 8-class Segmentator
label space.  It deliberately describes label encoding only; it must not be
used as image/dataset provenance or as an organ-conditioning profile.
"""

from .registry import DatasetConfig
from .unified_labels import COARSE_LABELS, UNIFIED_COLOR_MAP


_IDS = tuple(sorted(COARSE_LABELS))

CONFIG = DatasetConfig(
    name="UNIFIED_COARSE",
    cancer_type="organ_agnostic",
    original_label_map={idx: label for idx, label in COARSE_LABELS.items()},
    original_color_map={idx: list(UNIFIED_COLOR_MAP[idx]) for idx in _IDS},
    to_coarse_map={idx: idx for idx in _IDS},
    to_fine_map={idx: idx for idx in _IDS},
    coarse_to_original={idx: [idx] for idx in _IDS},
    tumor_ids=(1,),
    stroma_ids=(2,),
    necrosis_ids=(3,),
    immune_ids=(4,),
    normal_epi_ids=(5,),
    vessel_ids=(6,),
    skip_tissues=frozenset({0}),
    available_edits=(
        "tumor_burden_increase",
        "tumor_burden_decrease",
        "tumor_boundary_remodel",
        "necrosis_appearance",
        "necrosis_resolution",
        "stromal_immune_infiltration",
        "stromal_desmoplasia",
    ),
    expansion_targets={
        "tumor_burden_increase": (2, 4, 5, 7),
        "tumor_burden_decrease": (2,),
        "necrosis_appearance": (3,),
        "necrosis_resolution": (2,),
        "stromal_immune_infiltration": (4,),
        "stromal_desmoplasia": (2,),
    },
    has_cell_annotations=False,
    # This profile must not select a ProbNet cancer-type condition. Callers
    # that need nuclei generation must provide image provenance separately.
    cancer_type_index=-1,
    data_dir="",
    prior_db_path="",
    label_granularity="coarse",
)

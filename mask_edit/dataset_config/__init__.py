"""
Dataset Configuration Package — Unified Multi-Dataset Label System.

Central registry for all pathology dataset configurations. Replaces
hardcoded BCSS-specific constants scattered across the codebase.

Quick Start:
    from mask_edit.dataset_config import get_config, FINE_LABELS, UNIFIED_COLOR_MAP

    # Get a specific dataset config
    cfg = get_config("BCSS")
    print(cfg.tumor_ids)         # (1, 14, 15)
    print(cfg.available_edits)   # ("tumor_dilation", "tumor_shrink", ...)

    # List all datasets
    from mask_edit.dataset_config import list_datasets
    print(list_datasets())       # ['BCSS', 'GLAS', 'IGNITE', 'ORCA', 'PANDA', 'PUMA']

    # Use unified labels
    print(FINE_LABELS[8])        # "Gleason 3"
    print(UNIFIED_COLOR_MAP[8])  # [220, 120, 80]

Modules:
    unified_labels : Coarse (8) / Fine (16) label definitions, colors, cells
    registry       : DatasetConfig dataclass + get_config() factory
    bcss           : BCSS (breast) config — 22 original classes, fine IDs 14/15
    panda          : PANDA (prostate) config — Gleason 3/4/5, fine IDs 8/9/10
    glas           : GlaS (colorectal) config — differentiation grades, fine IDs 11/12/13
    ignite         : IGNITE (lung) config — 8 coarse classes
    puma           : PUMA (melanoma) config — 5 coarse classes
    orca           : ORCA (oral SCC) config — 3 classes (BG/Tumor/Other)
"""

# Re-export main API
from .registry import (
    DatasetConfig,
    get_config,
    list_datasets,
    get_all_configs,
)

from .unified_labels import (
    # Coarse labels
    COARSE_LABELS,
    NUM_COARSE,
    # Fine labels
    FINE_LABELS,
    NUM_FINE,
    # Mappings
    FINE_TO_PARENT,
    COARSE_TO_FINE,
    # Colors
    UNIFIED_COLOR_MAP,
    FULL_COLOR_MAP,
    # Cells
    CELL_CLASSES,
    CELL_COLOR_MAP,
    CELL_IDS,
    NUM_CELL_CLASSES,
    # Convenience sets
    ALL_FINE_IDS,
    ALL_COARSE_IDS,
    ALL_TUMOR_FINE_IDS,
    NON_BIO_IDS,
)

__all__ = [
    # Registry
    "DatasetConfig",
    "get_config",
    "list_datasets",
    "get_all_configs",
    # Unified labels
    "COARSE_LABELS",
    "NUM_COARSE",
    "FINE_LABELS",
    "NUM_FINE",
    "FINE_TO_PARENT",
    "COARSE_TO_FINE",
    "UNIFIED_COLOR_MAP",
    "FULL_COLOR_MAP",
    "CELL_CLASSES",
    "CELL_COLOR_MAP",
    "CELL_IDS",
    "NUM_CELL_CLASSES",
    "ALL_FINE_IDS",
    "ALL_COARSE_IDS",
    "ALL_TUMOR_FINE_IDS",
    "NON_BIO_IDS",
]

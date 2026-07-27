"""
Dataset Configuration Registry.

Central registry for all dataset configurations. Each dataset has its own
config file (bcss.py, panda.py, etc.) that defines a DatasetConfig instance.
Use `get_config(dataset_name)` to retrieve the configuration for a dataset.

Example:
    from dataset_config import get_config
    cfg = get_config("BCSS")
    print(cfg.tumor_ids)            # [1, 14, 15]
    print(cfg.available_edits)      # ["tumor_dilation", "tumor_shrink", ...]
    print(cfg.to_fine_map)          # {1: 1, 2: 2, 20: 14, ...}
"""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


@dataclass(frozen=True)
class DatasetConfig:
    """Immutable configuration for a single dataset."""

    # ---- Identity ----
    name: str                                   # "BCSS", "PANDA", ...
    cancer_type: str                            # "breast", "prostate", ...

    # ---- Original label system ----
    original_label_map: Dict[int, str]          # {orig_id: name, ...}
    original_color_map: Dict[int, List[int]]    # {orig_id: [R,G,B], ...}

    # ---- Mapping: original -> unified ----
    to_coarse_map: Dict[int, int]               # orig_id -> coarse ID (0-7)
    to_fine_map: Dict[int, int]                 # orig_id -> fine ID (0-15)

    # ---- Reverse mapping: unified -> original ----
    coarse_to_original: Dict[int, List[int]]    # coarse_id -> [orig_id, ...]

    # ---- Editing-related (in unified fine label space) ----
    tumor_ids: Tuple[int, ...]                  # fine IDs that count as tumor
    stroma_ids: Tuple[int, ...]                 # fine IDs that count as stroma
    necrosis_ids: Tuple[int, ...]               # fine IDs for necrosis
    immune_ids: Tuple[int, ...]                 # fine IDs for immune infiltrate
    normal_epi_ids: Tuple[int, ...]             # fine IDs for normal epithelium
    vessel_ids: Tuple[int, ...]                 # fine IDs for blood vessel
    skip_tissues: FrozenSet[int]                # non-biological fine IDs (BG, etc.)

    available_edits: Tuple[str, ...]            # edit operation names this dataset supports
    expansion_targets: Dict[str, Tuple[int, ...]]  # edit_name -> target fine IDs

    # ---- Cell-related ----
    has_cell_annotations: bool                  # original cell annotations available?
    cancer_type_index: int                      # one-hot index for ProbNet (0-5)

    # ---- Paths ----
    data_dir: str                               # dataset patch directory name
    prior_db_path: str                          # prior_db.json path

    # ---- Label granularity ----
    label_granularity: str = "coarse"           # "coarse" or "fine"
    # Fine-grained datasets (PANDA, GlaS, BCSS) have dataset-specific fine IDs
    # Coarse-only datasets (IGNITE, PUMA, ORCA) use only coarse IDs 0-7

    def get_fine_id(self, original_id: int) -> int:
        """Convert original dataset label to unified fine ID."""
        return self.to_fine_map[original_id]

    def get_coarse_id(self, original_id: int) -> int:
        """Convert original dataset label to unified coarse ID."""
        return self.to_coarse_map[original_id]

    def is_tumor(self, fine_id: int) -> bool:
        """Check if a unified fine ID is a tumor type."""
        return fine_id in self.tumor_ids

    def is_editable(self, edit_name: str) -> bool:
        """Check if this dataset supports a specific edit operation."""
        return edit_name in self.available_edits


# ──────────────────────────────────────────────
# Registry: lazy-loaded, cached
# ──────────────────────────────────────────────
_REGISTRY: Dict[str, DatasetConfig] = {}


def _load_registry() -> None:
    """Import all dataset config modules and populate the registry."""
    if _REGISTRY:
        return  # already loaded

    from . import bcss, panda, glas, ignite, puma, orca, unified_coarse

    for module in (bcss, panda, glas, ignite, puma, orca, unified_coarse):
        cfg: DatasetConfig = module.CONFIG
        _REGISTRY[cfg.name.upper()] = cfg


def get_config(dataset_name: str) -> DatasetConfig:
    """
    Retrieve the DatasetConfig for a given dataset.

    Args:
        dataset_name: Dataset name (case-insensitive), e.g. "BCSS", "panda"

    Returns:
        DatasetConfig instance

    Raises:
        KeyError: if dataset_name is not registered
    """
    _load_registry()
    key = dataset_name.upper()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. Available: {available}"
        )
    return _REGISTRY[key]


def list_datasets() -> List[str]:
    """Return sorted list of registered dataset names."""
    _load_registry()
    return sorted(_REGISTRY.keys())


def get_all_configs() -> Dict[str, DatasetConfig]:
    """Return all registered DatasetConfigs."""
    _load_registry()
    return dict(_REGISTRY)

"""Dataset-native label contracts for online semantic auditing."""

from __future__ import annotations

import numpy as np

from dataset_config import get_config
from dataset_config.unified_labels import FINE_TO_PARENT


UNSUPPORTED_FINE_IDS = {
    "BCSS": frozenset({14}),  # DCIS is not supported by the release.
}


def to_coarse_mask(mask: np.ndarray, *, ignore_index: int = 255) -> np.ndarray:
    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError("mask must be rank 2")
    coarse = np.array(values, copy=True)
    for fine_id, parent_id in FINE_TO_PARENT.items():
        coarse[values == fine_id] = parent_id
    unknown = ~np.isin(values, [*FINE_TO_PARENT, ignore_index])
    if np.any(unknown):
        bad = sorted(int(value) for value in np.unique(values[unknown]))
        raise ValueError(f"mask contains unsupported unified ids: {bad}")
    return coarse


def dataset_native_metric_class_ids(
    profile: str,
    *,
    level: str,
    exclude_background: bool = True,
    exclude_other: bool = True,
) -> tuple[int, ...]:
    """Return only classes the dataset actually annotates at this level."""

    config = get_config(profile)
    if level == "coarse":
        ids = set(int(value) for value in config.to_coarse_map.values())
    elif level == "fine":
        ids = set(int(value) for value in config.to_fine_map.values())
        ids -= set(UNSUPPORTED_FINE_IDS.get(config.name.upper(), ()))
    else:
        raise ValueError("level must be 'coarse' or 'fine'")
    if exclude_background:
        ids.discard(0)
    if exclude_other:
        ids.discard(7)
    return tuple(sorted(ids))


def profile_supports_fine(profile: str) -> bool:
    return get_config(profile).label_granularity == "fine"


__all__ = [
    "UNSUPPORTED_FINE_IDS",
    "dataset_native_metric_class_ids",
    "profile_supports_fine",
    "to_coarse_mask",
]

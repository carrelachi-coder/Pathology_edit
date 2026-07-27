"""Dataset-specific structural rewrite regions for gland edits."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy import ndimage


GLAS_GLAND_FINE_IDS = frozenset({5, 11, 12, 13})
GLAS_WHOLE_GLAND_POLICY_VERSION = "glas_whole_gland_instance_v1"


def expand_region_to_intersecting_components(
    region: np.ndarray,
    component_mask: np.ndarray,
    *,
    connectivity: int = 8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Add every complete binary component touched by ``region``."""

    selected = np.asarray(region, dtype=bool)
    components = np.asarray(component_mask, dtype=bool)
    if selected.shape != components.shape:
        raise ValueError("region and component_mask must have the same shape")
    structure = (
        np.ones((3, 3), dtype=bool)
        if connectivity == 8
        else ndimage.generate_binary_structure(2, 1)
    )
    labeled, component_count = ndimage.label(components, structure=structure)
    touched_ids = sorted(
        int(value)
        for value in np.unique(labeled[selected & components])
        if int(value) > 0
    )
    expanded = selected.copy()
    if touched_ids:
        expanded |= np.isin(labeled, touched_ids)
    return expanded, {
        "component_count": int(component_count),
        "touched_component_ids": touched_ids,
        "touched_component_count": len(touched_ids),
        "input_pixels": int(np.count_nonzero(selected)),
        "expanded_pixels": int(np.count_nonzero(expanded)),
        "added_component_pixels": int(np.count_nonzero(expanded & ~selected)),
    }


def glas_gland_mask(
    tissue_map: np.ndarray,
    *,
    gland_ids: Iterable[int] = GLAS_GLAND_FINE_IDS,
) -> np.ndarray:
    """Return the GlaS gland-object mask in unified fine-label space."""

    return np.isin(np.asarray(tissue_map), tuple(int(value) for value in gland_ids))


def glas_whole_gland_generation_region(
    reference_tissue: np.ndarray,
    target_tissue: np.ndarray,
    semantic_change_region: np.ndarray,
    *,
    profile: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Expand a GlaS boundary edit to the complete affected gland instance.

    Fine-label changes that preserve the gland footprint do not trigger an
    additional expansion. When the binary gland footprint changes, all old and
    new gland pixels in each touched union component enter the structural
    generation region. Other glands in the patch remain untouched.
    """

    source = np.asarray(reference_tissue)
    target = np.asarray(target_tissue)
    semantic = np.asarray(semantic_change_region, dtype=bool)
    if source.shape != target.shape or source.shape != semantic.shape:
        raise ValueError("reference, target, and semantic change masks must align")

    base = {
        "policy_version": GLAS_WHOLE_GLAND_POLICY_VERSION,
        "profile": str(profile),
        "gland_fine_ids": sorted(GLAS_GLAND_FINE_IDS),
        "semantic_change_pixels": int(np.count_nonzero(semantic)),
    }
    if str(profile).upper() != "GLAS":
        return semantic.copy(), {
            **base,
            "applied": False,
            "reason": "non_glas_profile",
            "generation_change_pixels": int(np.count_nonzero(semantic)),
        }

    source_gland = glas_gland_mask(source)
    target_gland = glas_gland_mask(target)
    boundary_delta = source_gland ^ target_gland
    boundary_delta_pixels = int(np.count_nonzero(boundary_delta))
    if boundary_delta_pixels == 0:
        return semantic.copy(), {
            **base,
            "applied": False,
            "reason": "gland_footprint_unchanged",
            "boundary_delta_pixels": 0,
            "generation_change_pixels": int(np.count_nonzero(semantic)),
        }

    generation, component_info = expand_region_to_intersecting_components(
        semantic | boundary_delta,
        source_gland | target_gland,
        connectivity=8,
    )
    return generation, {
        **base,
        "applied": bool(component_info["touched_component_count"]),
        "reason": "gland_boundary_changed",
        "boundary_delta_pixels": boundary_delta_pixels,
        "generation_change_pixels": int(np.count_nonzero(generation)),
        **component_info,
    }

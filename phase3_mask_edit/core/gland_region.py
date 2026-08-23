"""Dataset-specific structural rewrite regions for gland edits."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy import ndimage


GLAS_GLAND_FINE_IDS = frozenset({5, 11, 12, 13})
GLAS_WHOLE_GLAND_POLICY_VERSION = "glas_whole_gland_instance_v2"
GLAS_WHOLE_GLAND_CELL_REGION_POLICY = "whole_glas_connected_component"
SEMANTIC_CELL_DELETION_REGION_POLICY = "semantic_change_region"
SEMANTIC_NUCLEI_GENERATION_REGION_POLICY = (
    "semantic_change_region_plus_complete_intersecting_instances"
)
# Generator-only context is allowed to match the semantic edit area.  The old
# 25% ceiling left only a roughly three-pixel collar around long, narrow edits
# (for example an invasive cord), so Inpaint had no stromal texture domain in
# which to blend the new structure.  A 100% ceiling remains bounded: the final
# generation region can be at most twice the semantic edit area.
GENERATION_CONTEXT_MAX_EXTRA_FRACTION = 1.0
GENERATION_CONTEXT_MIN_EXTRA_PIXELS = 32
PRIMITIVE_GENERATION_CONTEXT_MAX_EXTRA_FRACTIONS = {
    # A cord is a long, thin cellular architecture embedded in stroma.  It
    # needs more generator-only collagen context than a compact semantic edit
    # so the Inpaint boundary does not trace the cellular support itself.
    "invasive-cord-formation-v1": 1.5,
    "infiltrative-nest-cord-extension-v1": 1.5,
    # Nucleus annotations commonly cover less than the full hematoxylin-stained
    # footprint.  Cell-removal generation therefore needs a wider local collar
    # so the image model removes the peripheral chromatin rim instead of
    # preserving a conspicuous ring around the cleared semantic mask.
    "cell-type-abundance-decrease-v1": 2.0,
    "generic-inflammatory-cell-abundance-decrease-v1": 2.0,
    "neoplastic-cell-abundance-decrease-v1": 2.0,
    "cellularity-decrease-v1": 2.0,
}
PRIMITIVE_GENERATION_CONTEXT_MINIMUM_DILATION_PIXELS = {
    "cell-type-abundance-decrease-v1": 4,
    "generic-inflammatory-cell-abundance-decrease-v1": 4,
    "neoplastic-cell-abundance-decrease-v1": 4,
    "cellularity-decrease-v1": 4,
}


def generation_context_max_extra_fraction(primitive_id: str | None) -> float:
    normalized = str(primitive_id or "").strip().lower()
    return float(
        PRIMITIVE_GENERATION_CONTEXT_MAX_EXTRA_FRACTIONS.get(
            normalized,
            GENERATION_CONTEXT_MAX_EXTRA_FRACTION,
        )
    )


def bound_generation_context_region(
    semantic_change_region: np.ndarray,
    candidate_generation_region: np.ndarray,
    *,
    max_extra_fraction: float | None = None,
    min_extra_pixels: int = GENERATION_CONTEXT_MIN_EXTRA_PIXELS,
    primitive_id: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep requested edits while bounding extra generator-only context.

    Candidate context is retained in increasing distance from the semantic
    edit. This prevents a touched, highly connected structure from turning a
    local request into an unbounded whole-patch rewrite.
    """

    semantic = np.asarray(semantic_change_region, dtype=bool)
    candidate = np.asarray(candidate_generation_region, dtype=bool)
    if max_extra_fraction is None:
        max_extra_fraction = generation_context_max_extra_fraction(
            primitive_id
        )
    if semantic.shape != candidate.shape:
        raise ValueError("semantic and candidate generation regions must align")
    if max_extra_fraction < 0:
        raise ValueError("max_extra_fraction must be non-negative")
    if min_extra_pixels < 0:
        raise ValueError("min_extra_pixels must be non-negative")
    missing = semantic & ~candidate
    if np.any(missing):
        raise ValueError(
            "candidate generation region must contain every semantic change pixel"
        )

    semantic_pixels = int(np.count_nonzero(semantic))
    candidate_pixels = int(np.count_nonzero(candidate))
    candidate_extra = candidate & ~semantic
    candidate_extra_pixels = int(np.count_nonzero(candidate_extra))
    minimum_dilation_pixels = int(
        PRIMITIVE_GENERATION_CONTEXT_MINIMUM_DILATION_PIXELS.get(
            str(primitive_id or "").strip().lower(),
            0,
        )
    )
    required_context = semantic.copy()
    if minimum_dilation_pixels and semantic_pixels:
        required_context = (
            ndimage.binary_dilation(
                semantic,
                structure=np.ones((3, 3), dtype=bool),
                iterations=minimum_dilation_pixels,
            )
            & candidate
        )
    required_extra_pixels = int(
        np.count_nonzero(required_context & ~semantic)
    )
    extra_budget = (
        max(
            int(min_extra_pixels),
            int(np.floor(float(max_extra_fraction) * semantic_pixels)),
            required_extra_pixels,
        )
        if semantic_pixels
        else 0
    )
    retained_extra_pixels = min(candidate_extra_pixels, extra_budget)
    capped = candidate_extra_pixels > retained_extra_pixels

    if not capped:
        bounded = candidate.copy()
    else:
        # Freeze the primitive-specific minimum collar first, then spend any
        # remaining bounded budget in stable distance order.  Counting the
        # collar in the budget without freezing it could drop diagonal rim
        # pixels at equal-distance ties and recreate a partial H&E ring.
        bounded = required_context.copy()
        distance_to_semantic = ndimage.distance_transform_edt(~semantic)
        optional_extra = candidate_extra & ~bounded
        coordinates = np.argwhere(optional_extra)
        distances = distance_to_semantic[
            coordinates[:, 0],
            coordinates[:, 1],
        ]
        order = np.lexsort(
            (
                coordinates[:, 1],
                coordinates[:, 0],
                distances,
            )
        )
        remaining_budget = max(
            0,
            retained_extra_pixels - required_extra_pixels,
        )
        keep = coordinates[order[:remaining_budget]]
        if len(keep):
            bounded[keep[:, 0], keep[:, 1]] = True

    return bounded, {
        "policy": "bounded_generation_context_v2",
        "primitive_id": primitive_id,
        "max_extra_fraction": float(max_extra_fraction),
        "min_extra_pixels": int(min_extra_pixels),
        "semantic_pixels": semantic_pixels,
        "candidate_pixels": candidate_pixels,
        "candidate_extra_pixels": candidate_extra_pixels,
        "extra_budget_pixels": int(extra_budget),
        "minimum_dilation_pixels": minimum_dilation_pixels,
        "required_dilation_extra_pixels": required_extra_pixels,
        "retained_extra_pixels": int(retained_extra_pixels),
        "generation_pixels": int(np.count_nonzero(bounded)),
        "capped": bool(capped),
        "selection": "nearest_to_semantic_stable",
    }


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
    touched_gland_change = semantic & (source_gland | target_gland)
    if not np.any(touched_gland_change):
        return semantic.copy(), {
            **base,
            "applied": False,
            "reason": "semantic_change_does_not_touch_gland",
            "boundary_delta_pixels": boundary_delta_pixels,
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
        "reason": "gland_change_whole_connected_component",
        "boundary_delta_pixels": boundary_delta_pixels,
        "generation_change_pixels": int(np.count_nonzero(generation)),
        "component_expansion": component_info,
        "context_bound": {
            "policy": "disabled_for_glas_whole_connected_component",
            "capped": False,
        },
    }

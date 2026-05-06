"""Generic tumor-boundary remodeling primitives."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy import ndimage

from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.core.morphology import (
    binary_dilate,
    binary_erode,
    fill_small_holes,
    generate_islands,
    keep_only_touching,
    multi_scale_smooth_noise,
    remove_small_components,
    select_boundary_band_by_fraction,
)
from phase3_mask_edit.generic.tumor_burden import (
    PrimitiveEditResult,
    PrimitiveExecutionError,
    _available_backfill_labels_and_mask,
    _four_neighbor_structure,
    _nearest_backfill_fine_ids,
    _nearest_tumor_fine_ids,
    _semantic_warnings_for_labels,
    _target_pixels,
)


def _build_infiltration_candidate_mask(
    normalized_mask: np.ndarray,
    source_mask: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build candidate mask of editable non-tumor tissue near tumor boundary.

    For boundary_infiltration, the infiltrating region is all non-tumor,
    non-background tissue that neighbors the tumor.  Unlike
    tumor_burden_increase which uses a fixed target_priority, infiltration
    can invade any neighboring tissue type.
    """
    mask_operation = primitive_config.get("mask_operation", {})
    forbid_targets = mask_operation.get("forbid_targets", [])
    forbid_ids: set[int] = set()
    for label in forbid_targets:
        if label in schema.label_to_fine_ids:
            forbid_ids.update(schema.label_to_fine_ids[label])

    # Candidate: non-tumor, non-background, non-forbidden tissue.
    is_background = np.isin(normalized_mask, tuple(schema.skip_fine_ids))
    is_forbidden = np.isin(normalized_mask, tuple(forbid_ids)) if forbid_ids else np.zeros_like(source_mask, dtype=bool)

    candidate_mask = ~is_background & ~source_mask & ~is_forbidden

    # Collect the distinct fine IDs present in candidate region for reporting.
    candidate_ids = np.unique(normalized_mask[candidate_mask])
    included_labels = tuple(
        label for label, fine_ids in schema.label_to_fine_ids.items()
        if any(fid in candidate_ids for fid in fine_ids)
    )

    return candidate_mask, included_labels


_DEFAULT_TISSUE_PREFERENCE = {
    "Stroma": 1.0,
    "Necrosis": 0.85,
    "Other tissue": 0.55,
    "Immune infiltrate": 0.60,
    "Normal epithelium": 0.30,
    "Blood vessel": 0.05,
}


def _build_tissue_preference_map(
    normalized_mask: np.ndarray,
    source_mask: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
) -> np.ndarray:
    """Build per-pixel preference weight based on tissue type.

    Real tumor infiltration is selective: tumors preferentially invade
    stroma (least resistance) and avoid blood vessels (physical barrier).
    This map weights candidate pixels accordingly so that the infiltration
    algorithm naturally concentrates growth into preferred tissue.
    """
    config_prefs = primitive_config.get("spatial_pattern", {}).get("tissue_preference", {})
    preferences = {**_DEFAULT_TISSUE_PREFERENCE, **config_prefs}

    preference_map = np.ones(normalized_mask.shape, dtype=float)

    for label, weight in preferences.items():
        if label not in schema.label_to_fine_ids:
            continue
        fine_ids = schema.label_to_fine_ids[label]
        is_label = np.isin(normalized_mask, tuple(fine_ids))
        if label == "Necrosis":
            tumor_boundary_ring = binary_dilate(source_mask, radius=2) & ~source_mask
            near_boundary = is_label & tumor_boundary_ring
            preference_map[near_boundary] = weight
        else:
            preference_map[is_label] = weight

    return preference_map


# ── boundary infiltration ──────────────────────────────────────────

def apply_boundary_infiltration(
    old_mask: np.ndarray,
    schema: MaskProfileSchema,
    context: MaskEditContext,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> PrimitiveEditResult:
    """Create infiltrative tumor boundary with tongues and protrusions."""

    _validate_boundary_infiltration_request(old_mask, schema, context, primitive_config, intent)
    mask = np.asarray(old_mask)
    source_mask = np.isin(context.normalized_mask, schema.tumor_fine_ids)
    if not np.any(source_mask):
        raise PrimitiveExecutionError("no_tumor")

    tumor_pixels = int(np.count_nonzero(source_mask))

    # Build candidate mask: all editable non-tumor tissue near the tumor boundary.
    candidate_mask, included_labels = _build_infiltration_candidate_mask(
        context.normalized_mask, source_mask, schema, primitive_config,
    )
    if not np.any(candidate_mask):
        raise PrimitiveExecutionError("no_neighboring_editable_tissue")

    target_fraction = _target_infiltration_fraction_for_intent(primitive_config, intent)

    # ── step 1: boundary protrusions via noise-weighted band selection ─
    boundary_protrusions, band_info = _select_infiltration_protrusions(
        source_mask,
        candidate_mask,
        mask,
        target_fraction=target_fraction,
        max_fraction_cap=_max_changed_area_fraction(primitive_config),
        intent=intent,
        primitive_config=primitive_config,
    )

    # ── step 2: budding islands (optional, disabled by default) ──
    island_policy = _island_policy_from_config(primitive_config, intent)
    if island_policy["max_islands_per_patch"] > 0 and island_policy["max_island_fraction"] > 0:
        remaining_fraction = target_fraction - (int(np.count_nonzero(boundary_protrusions)) / mask.size)
        island_fraction = max(0.0, min(remaining_fraction, island_policy["max_island_fraction"]))
        islands, island_info = generate_islands(
            candidate_mask,
            source_mask,
            max_distance_px=island_policy["max_distance_from_tumor_px"],
            max_island_area_px=island_policy["max_island_area_px"],
            max_islands=island_policy["max_islands_per_patch"],
            target_fraction=island_fraction,
            seed=intent.seed if intent.seed is not None else 42,
            protrusion_mask=boundary_protrusions,
            min_island_area_px=island_policy.get("min_island_area_px", 12),
        )
        change_region = boundary_protrusions | islands
        change_region = _enforce_change_connectivity(change_region, source_mask)
        island_px = int(np.count_nonzero(change_region & ~boundary_protrusions))
    else:
        island_info = {"islands_generated": 0, "total_island_pixels": 0, "target_fraction_shortfall": 0.0}
        change_region = boundary_protrusions
        island_px = 0

    selected_pixels = int(np.count_nonzero(change_region))
    if selected_pixels == 0:
        raise PrimitiveExecutionError("tumor_boundary_too_short")

    # ── enforce max changed area fraction ──────────────────────────
    actual_fraction = selected_pixels / mask.size
    max_frac = _max_changed_area_fraction(primitive_config)
    if actual_fraction > max_frac:
        change_region = _trim_to_max_fraction(change_region, source_mask, max_frac, mask.size)
        selected_pixels = int(np.count_nonzero(change_region))
        if selected_pixels == 0:
            raise PrimitiveExecutionError("tumor_boundary_too_short")
        actual_fraction = selected_pixels / mask.size

    # ── step 3: sculpt infiltrative edge ──
    change_region = _sculpt_infiltrative_edge(
        change_region, source_mask, candidate_mask,
        context.normalized_mask, schema, primitive_config, intent,
    )
    selected_pixels = int(np.count_nonzero(change_region))

    # ── write target mask ──────────────────────────────────────────
    target_mask = np.array(context.normalized_mask, copy=True)
    target_mask[change_region] = _nearest_tumor_fine_ids(
        context.normalized_mask,
        source_mask,
        change_region,
        schema,
    )

    changed_area_fraction = selected_pixels / target_mask.size
    warnings = _semantic_warnings_for_labels(
        included_labels, schema, context,
    )
    ops_log = {
        "primitive": "boundary_infiltration",
        "reference_profile": schema.reference_profile,
        "target_change_fraction": target_fraction,
        "changed_area_fraction": changed_area_fraction,
        "selected_pixels": selected_pixels,
        "boundary_protrusion_pixels": int(np.count_nonzero(boundary_protrusions)),
        "island_pixels": island_px,
        "islands_generated": island_info["islands_generated"],
        "candidate_labels": list(included_labels),
        "spatial": {
            **band_info,
            **island_info,
        },
    }

    return PrimitiveEditResult(
        target_mask=target_mask,
        change_region=change_region,
        changed_area_fraction=changed_area_fraction,
        selected_pixels=selected_pixels,
        warnings=warnings,
        ops_log=ops_log,
    )


def _validate_boundary_infiltration_request(
    old_mask: np.ndarray,
    schema: MaskProfileSchema,
    context: MaskEditContext,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> None:
    mask = np.asarray(old_mask)
    if mask.ndim != 2:
        raise PrimitiveExecutionError("boundary_infiltration requires a 2D mask.")
    if tuple(mask.shape) != context.mask_shape:
        raise PrimitiveExecutionError("old_mask shape must match MaskEditContext.")
    if schema.reference_profile != context.reference_profile:
        raise PrimitiveExecutionError(
            "schema.reference_profile must match context.reference_profile."
        )
    if intent.primitive != "boundary_infiltration":
        raise PrimitiveExecutionError(
            "apply_boundary_infiltration requires a boundary_infiltration intent."
        )
    if primitive_config.get("name") != "boundary_infiltration":
        raise PrimitiveExecutionError(
            "primitive_config must describe boundary_infiltration."
        )


def _target_infiltration_fraction_for_intent(
    primitive_config: Mapping[str, Any], intent: EditIntent
) -> float:
    if intent.target_change_fraction is not None:
        return intent.target_change_fraction

    intervals = (
        primitive_config.get("parameter_ranges", {})
        .get("total_new_tumor_area_fraction", {})
    )
    interval = intervals.get(intent.strength)
    if not isinstance(interval, list) or len(interval) != 2:
        raise PrimitiveExecutionError(
            f"boundary_infiltration does not define strength {intent.strength}."
        )
    lower, upper = float(interval[0]), float(interval[1])
    return (lower + upper) / 2


def _max_changed_area_fraction(primitive_config: Mapping[str, Any]) -> float:
    value = primitive_config.get("parameter_ranges", {}).get("max_changed_area_fraction", 0.30)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.30


def _island_policy_from_config(
    primitive_config: Mapping[str, Any], intent: EditIntent
) -> dict[str, Any]:
    spatial = primitive_config.get("spatial_pattern", {})
    if not isinstance(spatial, dict):
        spatial = {}
    island_raw = spatial.get("island_policy", {})
    if not isinstance(island_raw, dict):
        island_raw = {}

    max_distance = intent.parameters.get(
        "max_distance_from_tumor_px",
        island_raw.get("max_distance_from_tumor_px", 120),
    )
    max_area = intent.parameters.get(
        "max_island_area_px",
        island_raw.get("max_island_area_px", 50),
    )
    max_count = intent.parameters.get(
        "max_islands_per_patch",
        island_raw.get("max_islands_per_patch", 0),
    )
    max_fraction = intent.parameters.get(
        "max_island_fraction",
        island_raw.get("max_island_fraction", 0.0),
    )
    min_area = intent.parameters.get(
        "min_island_area_px",
        island_raw.get("min_island_area_px", 12),
    )

    return {
        "max_distance_from_tumor_px": int(max_distance),
        "max_island_area_px": int(max_area),
        "max_islands_per_patch": int(max_count),
        "max_island_fraction": float(max_fraction),
        "min_island_area_px": int(min_area),
    }


def _select_infiltration_protrusions(
    source_mask: np.ndarray,
    candidate_mask: np.ndarray,
    id_mask: np.ndarray,
    *,
    target_fraction: float,
    max_fraction_cap: float,
    intent: EditIntent,
    primitive_config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create connected, tongue-like infiltration protrusions.

    Strategy: noise-thresholded distance expansion + morphological cleanup.
    Instead of picking individual pixels by score (which produces dots),
    we use multi-scale noise to irregularly threshold a distance-based
    expansion.  This yields connected protrusions because the noise field
    is smooth: pixels near each other share similar noise values, so the
    threshold naturally selects entire tongue-shaped regions.
    """

    if not np.any(source_mask) or not np.any(candidate_mask):
        return np.zeros_like(source_mask, dtype=bool), {
            "boundary_protrusion_pixels": 0,
            "noise_used": False,
        }

    target_pixels = max(1, int(round(source_mask.size * target_fraction)))
    max_pixels = int(round(source_mask.size * max_fraction_cap))
    seed = intent.seed if intent.seed is not None else 0

    # ── step 1: distance-based expansion with noise threshold ──────
    dist_from_tumor = ndimage.distance_transform_edt(~source_mask)

    # Multi-scale noise: smooth fields create connected regions.
    noise = multi_scale_smooth_noise(
        source_mask.shape,
        scales=(4.0, 12.0, 28.0),
        amplitudes=(0.25, 0.45, 0.30),
        seed=seed,
    )

    # Normalize noise to [0, 1] range for thresholding.
    noise_min, noise_max = float(noise.min()), float(noise.max())
    if noise_max - noise_min < 1e-8:
        noise_norm = np.full_like(noise, 0.5)
    else:
        noise_norm = (noise - noise_min) / (noise_max - noise_min)

    # Distance decay: protrusions thin out as they extend further.
    max_expansion_radius = _expansion_radius_for_target(
        candidate_mask, dist_from_tumor, target_pixels,
    )
    dist_decay = np.zeros_like(dist_from_tumor)
    within_radius = candidate_mask & (dist_from_tumor <= max_expansion_radius)
    if np.any(within_radius):
        dist_decay[within_radius] = 1.0 - dist_from_tumor[within_radius] / max_expansion_radius

    # Selection criterion: select where noise_norm < dist_decay.
    raw_expansion = (noise_norm < dist_decay) & candidate_mask & ~source_mask

    # ── step 2: morphological cleanup for connected protrusions ────
    min_protrusion_area = intent.parameters.get(
        "min_protrusion_area_px",
        primitive_config.get("parameter_ranges", {}).get("min_protrusion_area_px", 20),
    )
    protrusions = remove_small_components(raw_expansion, min_area_px=min_protrusion_area)
    protrusions = fill_small_holes(protrusions, max_hole_area_px=min_protrusion_area)

    # ── step 3: adjust total area to match target ──────────────────
    current_px = int(np.count_nonzero(protrusions))

    if current_px < target_pixels:
        protrusions = _expand_protrusions_to_target(
            protrusions, noise_norm, dist_decay, candidate_mask, source_mask,
            target_pixels, max_pixels, min_protrusion_area,
        )
    elif current_px > max_pixels:
        protrusions = _shrink_protrusions_to_max(
            protrusions, dist_from_tumor, max_pixels,
            min_protrusion_area,
        )

    protrusion_px = int(np.count_nonzero(protrusions))

    return protrusions, {
        "boundary_protrusion_pixels": protrusion_px,
        "noise_used": True,
        "max_expansion_radius_px": max_expansion_radius,
        "min_protrusion_area_px": min_protrusion_area,
    }


def _expansion_radius_for_target(
    candidate_mask: np.ndarray,
    dist_from_tumor: np.ndarray,
    target_pixels: int,
) -> float:
    """Choose an expansion radius that roughly hits the target pixel count."""

    max_dist = float(np.max(dist_from_tumor[candidate_mask])) if np.any(candidate_mask) else 1.0

    lo, hi = 1.0, max_dist
    for _ in range(20):
        mid = (lo + hi) / 2
        count = int(np.count_nonzero(candidate_mask & (dist_from_tumor <= mid)))
        if count < target_pixels:
            lo = mid
        else:
            hi = mid

    return max(2.0, (lo + hi) / 2 * 1.5)


def _expand_protrusions_to_target(
    protrusions: np.ndarray,
    noise_norm: np.ndarray,
    dist_decay: np.ndarray,
    candidate_mask: np.ndarray,
    source_mask: np.ndarray,
    target_pixels: int,
    max_pixels: int,
    min_protrusion_area: int,
) -> np.ndarray:
    """Expand protrusions by relaxing the noise threshold until target is met."""

    current = int(np.count_nonzero(protrusions))
    if current >= target_pixels:
        return protrusions

    slack = 0.0
    step = 0.05
    for _ in range(30):
        slack += step
        expanded = (noise_norm < dist_decay + slack) & candidate_mask & ~source_mask
        cleaned = remove_small_components(expanded, min_area_px=min_protrusion_area)
        cleaned = fill_small_holes(cleaned, max_hole_area_px=min_protrusion_area)
        px = int(np.count_nonzero(cleaned))
        if px >= target_pixels or px >= max_pixels:
            return cleaned

    return protrusions


def _shrink_protrusions_to_max(
    protrusions: np.ndarray,
    dist_from_tumor: np.ndarray,
    max_pixels: int,
    min_protrusion_area: int,
) -> np.ndarray:
    """Remove farthest protrusion components until under max_pixels."""

    labeled, count = ndimage.label(protrusions, structure=_four_neighbor_structure())
    if count == 0:
        return protrusions

    components = []
    for cid in range(1, count + 1):
        comp = labeled == cid
        area = int(np.count_nonzero(comp))
        mean_dist = float(np.mean(dist_from_tumor[comp]))
        components.append((mean_dist, area, cid))

    components.sort(key=lambda x: -x[0])
    total_px = int(np.count_nonzero(protrusions))
    kept = protrusions.copy()

    for mean_dist, area, cid in components:
        if total_px <= max_pixels:
            break
        comp = labeled == cid
        kept[comp] = False
        total_px -= area

    kept = remove_small_components(kept, min_area_px=min_protrusion_area)
    return kept


def _enforce_change_connectivity(
    change_region: np.ndarray,
    source_mask: np.ndarray,
) -> np.ndarray:
    """Remove disconnected change-region components that don't touch source.

    All change pixels must belong to a connected component that also
    contains at least one source (tumor) pixel.
    """
    if int(np.count_nonzero(change_region)) == 0:
        return change_region

    combined = source_mask | change_region
    struct4 = _four_neighbor_structure()
    labeled, count = ndimage.label(combined, structure=struct4)
    if count == 0:
        return np.zeros_like(change_region, dtype=bool)

    source_labels = set(np.unique(labeled[source_mask])) - {0}

    result = np.zeros_like(change_region, dtype=bool)
    for src_label in source_labels:
        result |= (labeled == src_label) & change_region

    return result


def _trim_to_max_fraction(
    change_region: np.ndarray,
    source_mask: np.ndarray,
    max_fraction: float,
    total_pixels: int,
) -> np.ndarray:
    max_pixels = int(round(max_fraction * total_pixels))
    current_pixels = int(np.count_nonzero(change_region))
    if current_pixels <= max_pixels:
        return change_region

    dist = ndimage.distance_transform_edt(~source_mask)
    change_distances = dist[change_region]
    if change_distances.size == 0:
        return change_region

    threshold_distance = np.sort(change_distances)[::-1][max_pixels - 1]
    kept = change_region & (dist <= threshold_distance)

    kept_pixels = int(np.count_nonzero(kept))
    if kept_pixels > max_pixels:
        kept_indices = np.argwhere(kept)
        rng = np.random.default_rng(0)
        remove_count = kept_pixels - max_pixels
        remove_idx = rng.choice(len(kept_indices), size=remove_count, replace=False)
        for idx in remove_idx:
            kept[kept_indices[idx][0], kept_indices[idx][1]] = False

    return kept


# ── edge sculpting ──────────────────────────────────────────────────

def _sculpt_infiltrative_edge(
    change_region: np.ndarray,
    source_mask: np.ndarray,
    candidate_mask: np.ndarray,
    normalized_mask: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> np.ndarray:
    """Deform change_region edge to create coarse infiltrative pattern.

    1. Deterministic directional dilation (filtered by tissue preference) → coherent tongue
    2. Noise + distance-decay tapering → wide base, thin tip (not circular blob)
    3. Gaussian smooth → organic edges
    4. Contact-width filter → remove protrusions with needle-thin connections
    5. Morphological erosion in angular sectors → coherent band-shaped notches
    6. Fill holes + contact-width + reconnect → no interior holes or thin bridges
    """
    rng = np.random.default_rng(intent.seed if intent.seed is not None else 0)
    seed_val = intent.seed if intent.seed is not None else 0

    spatial = primitive_config.get("spatial_pattern", {})
    edge_cfg = spatial.get("edge_sculpt", {})

    n_tongues_lo = intent.parameters.get("edge_tongue_count_lo", edge_cfg.get("tongue_count_lo", 3))
    n_tongues_hi = intent.parameters.get("edge_tongue_count_hi", edge_cfg.get("tongue_count_hi", 6))
    tongue_depth = intent.parameters.get("edge_tongue_depth_px", edge_cfg.get("tongue_depth_px", 25))
    angular_width = intent.parameters.get("edge_tongue_width_rad", edge_cfg.get("tongue_width_rad", 0.45))
    min_sep = intent.parameters.get("edge_tongue_sep_rad", edge_cfg.get("tongue_sep_rad", 0.45))
    tongue_smooth = intent.parameters.get("edge_tongue_smooth_sigma_px", edge_cfg.get("tongue_smooth_sigma_px", 3.0))
    min_tissue_pref = intent.parameters.get("edge_min_tissue_pref", edge_cfg.get("min_tissue_pref", 0.20))
    min_contact_px = intent.parameters.get("edge_min_contact_px", edge_cfg.get("min_contact_px", 5))
    erosion_depth = intent.parameters.get("edge_erosion_depth_px", edge_cfg.get("erosion_depth_px", 3))
    erosion_fraction = intent.parameters.get("edge_erosion_fraction", edge_cfg.get("erosion_fraction", 0.20))

    tissue_pref = _build_tissue_preference_map(
        normalized_mask, source_mask, schema, primitive_config,
    )

    struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

    # ── Extract outer boundary ──
    combined = change_region | source_mask
    outside = ~combined
    outer_boundary = change_region & ~source_mask & ndimage.binary_dilation(outside, structure=struct4)

    bcoords = np.argwhere(outer_boundary)
    if len(bcoords) < 10:
        return change_region

    # ── Compute angles relative to tumor centroid ──
    tumor_coords = np.argwhere(source_mask)
    centroid = tumor_coords.mean(axis=0)
    bangles = np.arctan2(bcoords[:, 0] - centroid[0], bcoords[:, 1] - centroid[1])
    bprefs = tissue_pref[bcoords[:, 0], bcoords[:, 1]]

    # ── Sector setup (shared by tongues and erosion) ──
    n_sectors = 24
    sector_edges = np.linspace(-np.pi, np.pi, n_sectors + 1)

    # ── Pick tongue directions biased toward high-preference sectors ──
    num_tongues = rng.integers(n_tongues_lo, n_tongues_hi + 1)
    tongue_angles = _pick_edge_tongue_angles(bangles, bprefs, num_tongues, min_sep, rng)

    # ── Grow tongue protrusions ──
    tongue_mask = np.zeros_like(change_region, dtype=bool)
    dist_from_tumor = ndimage.distance_transform_edt(~source_mask)
    pref_passable = tissue_pref >= min_tissue_pref

    for t_idx, t_angle in enumerate(tongue_angles):
        # Seeds: boundary pixels within angular cone AND with sufficient tissue preference
        angle_diff = np.abs(np.arctan2(np.sin(bangles - t_angle), np.cos(bangles - t_angle)))
        near_seeds = bcoords[angle_diff < angular_width * 0.5]
        if len(near_seeds) == 0:
            continue

        # Direction mask: candidate pixels within angular cone with sufficient preference
        cand_coords = np.argwhere(candidate_mask & ~combined & pref_passable)
        if len(cand_coords) == 0:
            continue
        cand_angles = np.arctan2(cand_coords[:, 0] - centroid[0], cand_coords[:, 1] - centroid[1])
        cand_diff = np.abs(np.arctan2(np.sin(cand_angles - t_angle), np.cos(cand_angles - t_angle)))
        dir_mask = np.zeros_like(candidate_mask, dtype=bool)
        in_cone = cand_diff < angular_width
        dir_mask[cand_coords[in_cone, 0], cand_coords[in_cone, 1]] = True

        # ── Phase A: deterministic directional dilation (filtered by preference) ──
        tongue = np.zeros_like(change_region, dtype=bool)
        for y, x in near_seeds:
            tongue[int(y), int(x)] = True

        max_tongue_dist = float(np.max(dist_from_tumor[near_seeds])) + tongue_depth

        for _ in range(tongue_depth):
            dilated = ndimage.binary_dilation(tongue, structure=struct4)
            new_px = dilated & ~tongue & candidate_mask & ~source_mask & dir_mask & pref_passable
            new_px &= dist_from_tumor <= max_tongue_dist
            tongue |= new_px

        # ── Phase B: noise-tapered tip (wide base, thin tip) ──
        tongue_non_source = tongue & ~source_mask
        if np.any(tongue_non_source):
            t_seed = seed_val + t_idx * 100
            t_noise = multi_scale_smooth_noise(
                tongue.shape,
                scales=(3.0, 8.0, 16.0),
                amplitudes=(0.30, 0.50, 0.20),
                seed=t_seed,
            )
            t_noise_min, t_noise_max = float(t_noise.min()), float(t_noise.max())
            if t_noise_max - t_noise_min < 1e-8:
                t_noise_norm = np.full_like(t_noise, 0.5)
            else:
                t_noise_norm = (t_noise - t_noise_min) / (t_noise_max - t_noise_min)

            base_dist = float(np.min(dist_from_tumor[tongue_non_source]))
            tip_dist = float(np.max(dist_from_tumor[tongue_non_source]))
            if tip_dist - base_dist > 1.0:
                t_decay = np.zeros_like(dist_from_tumor)
                t_within = tongue_non_source & (dist_from_tumor <= tip_dist)
                t_decay[t_within] = 1.0 - (dist_from_tumor[t_within] - base_dist) / (tip_dist - base_dist)
                # Tighter taper: slack=0.08, source always kept
                taper_mask = (t_noise_norm < t_decay + 0.08) | source_mask
                tongue = tongue & taper_mask

        # ── Phase C: Gaussian smooth for organic edges ──
        if tongue_smooth > 0 and np.any(tongue):
            tongue_float = tongue.astype(float)
            blurred = ndimage.gaussian_filter(tongue_float, sigma=tongue_smooth)
            tongue = (blurred >= 0.35) & ~source_mask & candidate_mask & pref_passable
            tongue = keep_only_touching(tongue, source_mask | change_region)

        # ── Phase D: contact-width filter → remove needle-thin connections ──
        tongue = _filter_change_contact_width(tongue, source_mask | change_region, min_contact_px)

        tongue_mask |= tongue

    # ── Inward erosion via morphological erosion in angular sectors ──
    used_sectors = set()
    for ta in tongue_angles:
        center = int(np.round((ta + np.pi) / (2 * np.pi) * n_sectors)) % n_sectors
        for offset in (-1, 0, 1):
            used_sectors.add((center + offset) % n_sectors)

    bsector_idx = np.digitize(bangles, sector_edges) - 1
    bsector_idx = np.clip(bsector_idx, 0, n_sectors - 1)

    sector_prefs = {}
    for i in range(n_sectors):
        in_s = bsector_idx == i
        if np.any(in_s):
            sector_prefs[i] = float(np.mean(bprefs[in_s]))

    erosion_candidates = [
        (p, i) for i, p in sector_prefs.items()
        if i not in used_sectors and p < 0.75
    ]
    erosion_candidates.sort(key=lambda x: x[0])
    n_erosions = max(3, int(len(erosion_candidates) * erosion_fraction))

    cr_coords = np.argwhere(change_region & ~source_mask)
    erosion_angular_mask = np.zeros_like(change_region, dtype=bool)
    if len(cr_coords) > 0 and n_erosions > 0:
        cr_angles = np.arctan2(cr_coords[:, 0] - centroid[0], cr_coords[:, 1] - centroid[1])
        cr_sector_idx = np.digitize(cr_angles, sector_edges) - 1
        cr_sector_idx = np.clip(cr_sector_idx, 0, n_sectors - 1)
        for _, es in erosion_candidates[:n_erosions]:
            for offset in (-1, 0, 1):
                si = (es + offset) % n_sectors
                in_s = cr_sector_idx == si
                erosion_angular_mask[cr_coords[in_s, 0], cr_coords[in_s, 1]] = True

    change_only = change_region & ~source_mask
    eroded_inner = binary_erode(change_only, radius=erosion_depth)
    would_remove = change_only & ~eroded_inner
    actual_erosion = would_remove & erosion_angular_mask

    # ── Combine ──
    result = (change_region | tongue_mask) & ~actual_erosion
    result = fill_small_holes(result, max_hole_area_px=80)
    result = keep_only_touching(result, source_mask)
    result = remove_small_components(result, min_area_px=15)
    result = _filter_change_contact_width(result, source_mask, min_contact_px)
    result = _enforce_change_connectivity(result, source_mask)

    return result


def _filter_change_contact_width(
    change_region: np.ndarray,
    source_mask: np.ndarray,
    min_contact_px: int,
) -> np.ndarray:
    """Remove change-region components with insufficient contact with source.

    Prevents needle-thin connections: a protrusion or detached fragment
    that only touches the tumor body through < min_contact_px pixels
    is pathological and should be removed entirely.
    """
    change_only = change_region & ~source_mask
    if not np.any(change_only):
        return change_region

    struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    labeled, count = ndimage.label(change_only, structure=struct4)
    if count == 0:
        return change_region

    source_ring = ndimage.binary_dilation(source_mask, structure=struct4)

    result = change_region.copy()
    for cid in range(1, count + 1):
        comp = labeled == cid
        contact = comp & source_ring
        if int(np.count_nonzero(contact)) < min_contact_px:
            result[comp] = False

    return result


def _pick_edge_tongue_angles(
    angles: np.ndarray,
    prefs: np.ndarray,
    num_tongues: int,
    min_sep: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick well-separated tongue angles biased toward high-preference sectors."""
    n_bins = 36
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_idx = np.digitize(angles, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    bin_scores = np.zeros(n_bins)
    for i in range(n_bins):
        in_bin = bin_idx == i
        if np.any(in_bin):
            bin_scores[i] = float(np.mean(prefs[in_bin]))

    chosen: list[float] = []
    order = np.argsort(-bin_scores)

    for idx in order:
        if len(chosen) >= num_tongues:
            break
        if bin_scores[idx] <= 0:
            continue
        angle = float(bin_centers[idx]) + rng.uniform(-0.10, 0.10)
        too_close = any(
            abs(np.arctan2(np.sin(angle - ex), np.cos(angle - ex))) < min_sep
            for ex in chosen
        )
        if too_close:
            continue
        chosen.append(angle)

    # Relax separation if not enough tongues
    if len(chosen) < num_tongues:
        for idx in order:
            if len(chosen) >= num_tongues:
                break
            if bin_scores[idx] <= 0:
                continue
            angle = float(bin_centers[idx]) + rng.uniform(-0.15, 0.15)
            too_close = any(
                abs(np.arctan2(np.sin(angle - ex), np.cos(angle - ex))) < 0.15
                for ex in chosen
            )
            if not too_close:
                chosen.append(angle)

    return np.array(chosen[:num_tongues])


# ── boundary pushing remodel (unchanged) ───────────────────────────

def apply_boundary_pushing_remodel(
    old_mask: np.ndarray,
    schema: MaskProfileSchema,
    context: MaskEditContext,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> PrimitiveEditResult:
    """Convert a locally irregular Tumor boundary toward a pushing border."""

    _validate_boundary_pushing_request(old_mask, schema, context, primitive_config, intent)
    mask = np.asarray(old_mask)
    source_mask = np.isin(context.normalized_mask, schema.tumor_fine_ids)
    if not np.any(source_mask):
        raise PrimitiveExecutionError("no_tumor")

    backfill_labels, backfill_mask = _available_backfill_labels_and_mask(
        context.normalized_mask,
        schema,
        primitive_config,
        intent,
    )
    if not backfill_labels:
        raise PrimitiveExecutionError("no_valid_backfill_tissue")

    target_fraction = _target_changed_fraction_for_intent(primitive_config, intent)
    target_pixels = _target_pixels(target_fraction, mask.size)
    smooth_radius = _smooth_radius_for_intent(primitive_config, intent)
    max_abs_delta_fraction = _max_abs_tumor_delta_fraction(
        primitive_config,
        intent,
    )
    min_component_area = _min_component_area_for_intent(primitive_config, intent)
    max_abs_delta_pixels = int(round(max_abs_delta_fraction * mask.size))

    raw_added, raw_removed, score = _pushing_boundary_candidates(
        context.normalized_mask,
        source_mask,
        backfill_mask,
        schema,
        smooth_radius=smooth_radius,
    )
    added, removed, selection_info = _select_remodel_components(
        raw_added,
        raw_removed,
        score,
        target_pixels=target_pixels,
        max_abs_delta_pixels=max_abs_delta_pixels,
        min_component_area=min_component_area,
    )
    change_region = added | removed
    selected_pixels = int(np.count_nonzero(change_region))
    if selected_pixels == 0:
        raise PrimitiveExecutionError("tumor_already_pushing_or_smooth")

    target_mask = np.array(context.normalized_mask, copy=True)
    target_mask[added] = _nearest_tumor_fine_ids(
        context.normalized_mask,
        source_mask,
        added,
        schema,
    )
    target_mask[removed] = _nearest_backfill_fine_ids(
        context.normalized_mask,
        backfill_mask,
        removed,
    )

    tumor_area_delta_pixels = int(np.count_nonzero(added)) - int(np.count_nonzero(removed))
    changed_area_fraction = selected_pixels / target_mask.size
    warnings = _semantic_warnings_for_labels(backfill_labels, schema, context)
    ops_log = {
        "primitive": "boundary_pushing_remodel",
        "reference_profile": schema.reference_profile,
        "target_change_fraction": target_fraction,
        "changed_area_fraction": changed_area_fraction,
        "selected_pixels": selected_pixels,
        "added_tumor_pixels": int(np.count_nonzero(added)),
        "removed_tumor_pixels": int(np.count_nonzero(removed)),
        "tumor_area_delta_pixels": tumor_area_delta_pixels,
        "backfill_labels": list(backfill_labels),
        "spatial": {
            "smooth_radius": smooth_radius,
            "target_pixels": target_pixels,
            "min_component_area_px": min_component_area,
            "max_abs_tumor_delta_fraction": max_abs_delta_fraction,
            "max_abs_tumor_delta_pixels": max_abs_delta_pixels,
            **selection_info,
        },
    }

    return PrimitiveEditResult(
        target_mask=target_mask,
        change_region=change_region,
        changed_area_fraction=changed_area_fraction,
        selected_pixels=selected_pixels,
        warnings=warnings,
        ops_log=ops_log,
    )


def _validate_boundary_pushing_request(
    old_mask: np.ndarray,
    schema: MaskProfileSchema,
    context: MaskEditContext,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> None:
    mask = np.asarray(old_mask)
    if mask.ndim != 2:
        raise PrimitiveExecutionError("boundary_pushing_remodel requires a 2D mask.")
    if tuple(mask.shape) != context.mask_shape:
        raise PrimitiveExecutionError("old_mask shape must match MaskEditContext.")
    if schema.reference_profile != context.reference_profile:
        raise PrimitiveExecutionError(
            "schema.reference_profile must match context.reference_profile."
        )
    if intent.primitive != "boundary_pushing_remodel":
        raise PrimitiveExecutionError(
            "apply_boundary_pushing_remodel requires a boundary_pushing_remodel intent."
        )
    if primitive_config.get("name") != "boundary_pushing_remodel":
        raise PrimitiveExecutionError(
            "primitive_config must describe boundary_pushing_remodel."
        )


def _target_changed_fraction_for_intent(
    primitive_config: Mapping[str, Any], intent: EditIntent
) -> float:
    if intent.target_change_fraction is not None:
        return intent.target_change_fraction

    intervals = (
        primitive_config.get("parameter_ranges", {})
        .get("target_changed_area_fraction", {})
    )
    interval = intervals.get(intent.strength)
    if not isinstance(interval, list) or len(interval) != 2:
        raise PrimitiveExecutionError(
            f"boundary_pushing_remodel does not define strength {intent.strength}."
        )

    lower, upper = float(interval[0]), float(interval[1])
    return (lower + upper) / 2


def _smooth_radius_for_intent(
    primitive_config: Mapping[str, Any], intent: EditIntent
) -> int:
    value = intent.parameters.get(
        "smooth_radius",
        primitive_config.get("parameter_ranges", {}).get("default_smooth_radius_px", 18),
    )
    if not isinstance(value, int) or value < 1:
        raise PrimitiveExecutionError("parameters.smooth_radius must be a positive integer.")
    return value


def _min_component_area_for_intent(
    primitive_config: Mapping[str, Any], intent: EditIntent
) -> int:
    value = intent.parameters.get(
        "min_component_area_px",
        primitive_config.get("parameter_ranges", {}).get("min_component_area_px", 80),
    )
    if not isinstance(value, int) or value < 1:
        raise PrimitiveExecutionError(
            "parameters.min_component_area_px must be a positive integer."
        )
    return value


def _max_abs_tumor_delta_fraction(
    primitive_config: Mapping[str, Any], intent: EditIntent
) -> float:
    value = intent.parameters.get(
        "max_abs_tumor_area_delta_fraction",
        primitive_config.get("parameter_ranges", {}).get(
            "max_abs_tumor_area_delta_fraction",
            0.02,
        ),
    )
    if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
        raise PrimitiveExecutionError(
            "max_abs_tumor_area_delta_fraction must be numeric in [0, 1]."
        )
    return float(value)


def _pushing_boundary_candidates(
    mask: np.ndarray,
    source_mask: np.ndarray,
    backfill_mask: np.ndarray,
    schema: MaskProfileSchema,
    *,
    smooth_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma = float(smooth_radius)
    blurred = ndimage.gaussian_filter(source_mask.astype(float), sigma=sigma)
    smoothed = blurred >= 0.5

    protected_context = np.isin(mask, tuple(schema.skip_fine_ids))
    if "Necrosis" in schema.readable_labels:
        protected_context |= np.isin(mask, schema.resolve_fine_ids("Necrosis"))
    protected_boundary = source_mask & ndimage.binary_dilation(
        protected_context,
        structure=_four_neighbor_structure(),
    )

    raw_added = smoothed & ~source_mask & backfill_mask
    raw_removed = source_mask & ~smoothed & ~protected_boundary
    raw_removed = _keep_components_touching_context(raw_removed, backfill_mask)

    score = np.zeros(mask.shape, dtype=float)
    score[raw_added] = blurred[raw_added]
    score[raw_removed] = 1.0 - blurred[raw_removed]
    return raw_added, raw_removed, score


def _keep_components_touching_context(
    components_mask: np.ndarray, context_mask: np.ndarray
) -> np.ndarray:
    labeled, count = ndimage.label(components_mask, structure=_four_neighbor_structure())
    if count == 0:
        return np.zeros_like(components_mask, dtype=bool)

    touching_context = ndimage.binary_dilation(
        context_mask,
        structure=_four_neighbor_structure(),
    )
    kept = np.zeros_like(components_mask, dtype=bool)
    for component_id in range(1, count + 1):
        component = labeled == component_id
        if np.any(component & touching_context):
            kept |= component
    return kept


def _select_remodel_components(
    raw_added: np.ndarray,
    raw_removed: np.ndarray,
    score: np.ndarray,
    *,
    target_pixels: int,
    max_abs_delta_pixels: int,
    min_component_area: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | bool]]:
    raw_change = raw_added | raw_removed
    if not np.any(raw_change):
        return (
            np.zeros_like(raw_added, dtype=bool),
            np.zeros_like(raw_removed, dtype=bool),
            {"raw_candidate_pixels": 0, "selected_components": 0},
        )

    labeled, count = ndimage.label(raw_change, structure=_four_neighbor_structure())
    components: list[tuple[float, int, int, int, int]] = []
    for component_id in range(1, count + 1):
        component = labeled == component_id
        area = int(np.count_nonzero(component))
        if area < min_component_area:
            continue
        added_count = int(np.count_nonzero(component & raw_added))
        removed_count = int(np.count_nonzero(component & raw_removed))
        delta = added_count - removed_count
        mean_score = float(score[component].mean()) if area else 0.0
        components.append((mean_score, area, delta, added_count, component_id))

    if not components and min_component_area > 1:
        return _select_remodel_components(
            raw_added,
            raw_removed,
            score,
            target_pixels=target_pixels,
            max_abs_delta_pixels=max_abs_delta_pixels,
            min_component_area=1,
        )

    components.sort(key=lambda item: (-item[0], -item[1]))
    selected = np.zeros_like(raw_change, dtype=bool)
    selected_pixels = 0
    selected_delta = 0
    selected_components = 0

    for _, area, delta, _, component_id in components:
        if selected_pixels > 0 and selected_pixels + area > target_pixels:
            continue
        if abs(selected_delta + delta) > max_abs_delta_pixels:
            continue
        component = labeled == component_id
        selected |= component
        selected_pixels += area
        selected_delta += delta
        selected_components += 1
        if selected_pixels >= target_pixels:
            break

    selected_added = selected & raw_added
    selected_removed = selected & raw_removed
    return (
        selected_added,
        selected_removed,
        {
            "raw_candidate_pixels": int(np.count_nonzero(raw_change)),
            "selected_components": selected_components,
            "target_area_shortfall": selected_pixels < target_pixels,
            "tumor_area_delta_within_limit": abs(selected_delta) <= max_abs_delta_pixels,
        },
    )
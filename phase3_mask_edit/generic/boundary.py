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
    fill_small_holes,
    generate_islands,
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


# ── boundary infiltration ──────────────────────────────────────────

def apply_boundary_infiltration(
    old_mask: np.ndarray,
    schema: MaskProfileSchema,
    context: MaskEditContext,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> PrimitiveEditResult:
    """Create infiltrative tumor boundary with tongues, protrusions and budding islands."""

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

    # ── step 2: budding islands near the tumor ──────────────────────
    island_policy = _island_policy_from_config(primitive_config, intent)
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
    )

    # ── combine protrusions + islands ──────────────────────────────
    change_region = boundary_protrusions | islands
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

    # ── enforce island constraints ─────────────────────────────────
    _validate_island_constraints(islands, source_mask, island_policy)

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
        "island_pixels": int(np.count_nonzero(islands)),
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
        island_raw.get("max_island_area_px", 200),
    )
    max_count = intent.parameters.get(
        "max_islands_per_patch",
        island_raw.get("max_islands_per_patch", 12),
    )
    max_fraction = intent.parameters.get("max_island_fraction", 0.10)

    return {
        "max_distance_from_tumor_px": int(max_distance),
        "max_island_area_px": int(max_area),
        "max_islands_per_patch": int(max_count),
        "max_island_fraction": float(max_fraction),
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
        scales=(4.0, 12.0, 28.0),   # larger scales → smoother, connected shapes
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
    # Pixels closer to tumor boundary are easier to select.
    # dist_decay = 1 near boundary, falls toward 0 at max_expansion_radius.
    dist_decay = np.zeros_like(dist_from_tumor)
    within_radius = candidate_mask & (dist_from_tumor <= max_expansion_radius)
    if np.any(within_radius):
        dist_decay[within_radius] = 1.0 - dist_from_tumor[within_radius] / max_expansion_radius

    # Selection criterion: select where noise_norm < dist_decay.
    # This means: near the boundary (high dist_decay), almost all noise
    # values pass → broad expansion.  Further away (low dist_decay),
    # only low noise values pass → only selected where noise "dips",
    # creating thin tongues that extend in specific directions.
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
        # Too few pixels — lower the noise threshold to expand further.
        protrusions = _expand_protrusions_to_target(
            protrusions, noise_norm, dist_decay, candidate_mask, source_mask,
            target_pixels, max_pixels, min_protrusion_area,
        )
    elif current_px > max_pixels:
        # Too many pixels — remove farthest protrusion components first.
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

    # Binary search for radius: count candidate pixels within radius.
    lo, hi = 1.0, max_dist
    for _ in range(20):
        mid = (lo + hi) / 2
        count = int(np.count_nonzero(candidate_mask & (dist_from_tumor <= mid)))
        if count < target_pixels:
            lo = mid
        else:
            hi = mid

    # Return a radius that gives ~1.5x target so noise threshold can trim down.
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

    # Relax threshold: add candidate pixels where noise_norm < dist_decay + slack.
    # Increase slack incrementally until we hit the target.
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

    # Compute mean distance of each component from source.
    components = []
    for cid in range(1, count + 1):
        comp = labeled == cid
        area = int(np.count_nonzero(comp))
        mean_dist = float(np.mean(dist_from_tumor[comp]))
        components.append((mean_dist, area, cid))

    # Remove farthest components first.
    components.sort(key=lambda x: -x[0])
    total_px = int(np.count_nonzero(protrusions))
    kept = protrusions.copy()

    for mean_dist, area, cid in components:
        if total_px <= max_pixels:
            break
        if area < min_protrusion_area:
            # Small component — always safe to remove.
            comp = labeled == cid
            kept[comp] = False
            total_px -= area
        else:
            # Large component — only remove if still above max after removal.
            comp = labeled == cid
            kept[comp] = False
            total_px -= area

    # Final cleanup: remove any newly-isolated small fragments.
    kept = remove_small_components(kept, min_area_px=min_protrusion_area)
    return kept


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

    # Remove pixels furthest from source (islands first, then outer protrusions).
    dist = ndimage.distance_transform_edt(~source_mask)
    change_distances = dist[change_region]
    if change_distances.size == 0:
        return change_region

    # Sort by distance descending: remove farthest first.
    threshold_distance = np.sort(change_distances)[::-1][max_pixels - 1]
    kept = change_region & (dist <= threshold_distance)

    # Adjust if threshold cuts too many or too few.
    kept_pixels = int(np.count_nonzero(kept))
    if kept_pixels > max_pixels:
        kept_indices = np.argwhere(kept)
        rng = np.random.default_rng(0)
        remove_count = kept_pixels - max_pixels
        remove_idx = rng.choice(len(kept_indices), size=remove_count, replace=False)
        for idx in remove_idx:
            kept[kept_indices[idx][0], kept_indices[idx][1]] = False

    return kept


def _validate_island_constraints(
    islands: np.ndarray,
    source_mask: np.ndarray,
    island_policy: dict[str, Any],
) -> None:
    """Check island distance, area and count against policy limits."""

    if not np.any(islands):
        return

    max_distance = island_policy["max_distance_from_tumor_px"]
    max_area = island_policy["max_island_area_px"]
    max_count = island_policy["max_islands_per_patch"]

    dist_to_tumor = ndimage.distance_transform_edt(~source_mask)
    far_pixels = int(np.count_nonzero(islands & (dist_to_tumor > max_distance)))
    if far_pixels > 0:
        raise PrimitiveExecutionError(
            f"islands_too_far: {far_pixels} pixels beyond max distance {max_distance}px"
        )

    labeled, count = ndimage.label(islands, structure=_four_neighbor_structure())
    if count > max_count:
        raise PrimitiveExecutionError(
            f"islands_too_large_or_too_many: {count} islands exceeds max {max_count}"
        )

    for component_id in range(1, count + 1):
        component = labeled == component_id
        area = int(np.count_nonzero(component))
        if area > max_area:
            raise PrimitiveExecutionError(
                f"islands_too_large_or_too_many: island area {area} exceeds max {max_area}px"
            )


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
"""Generic necrosis mask primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy import ndimage

from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.generic.tumor_burden import (
    PrimitiveEditResult,
    PrimitiveExecutionError,
    _four_neighbor_structure,
    _target_pixels,
)


_DEFAULT_NECROSIS_NEIGHBOR_RADIUS_PX = 48.0
_DEFAULT_TUMOR_INTERIOR_RADIUS_PX = 64.0
_DEFAULT_VESSEL_AVOIDANCE_RADIUS_PX = 96.0
_DEFAULT_NOISE_WEIGHT = 0.08
_DEFAULT_MIN_COMPONENT_AREA_PX = 256
_DEFAULT_MAX_COMPONENTS_NO_EXISTING_NECROSIS = 3
_DEFAULT_MAX_COMPONENTS_WITH_EXISTING_NECROSIS = 3
_DEFAULT_CLOSING_RADIUS_PX = 10
_DEFAULT_FINAL_SMOOTH_RADIUS_PX = 8
_DEFAULT_INTERIOR_SCORE_WEIGHT_CAP = 0.35
_DEFAULT_MIN_RESOLUTION_COMPONENT_AREA_PX = 64
_MIN_SELECTED_TARGET_FRACTION = 0.75


@dataclass(frozen=True)
class _ScoreInfo:
    """Necrosis candidate scoring metadata."""

    score: np.ndarray
    used_existing_necrosis_neighborhood: bool
    used_blood_vessel_distance: bool
    existing_necrosis_expansion_band_pixels: int
    allow_multifocal_new_foci_when_existing_necrosis: bool
    active_weights: dict[str, float]
    radii: dict[str, float]


@dataclass(frozen=True)
class _SelectionInfo:
    """Connected-component selection metadata."""

    change_region: np.ndarray
    score_threshold: float
    threshold_percentile: float
    selected_components: int
    retry_applied: bool
    pre_cleanup_pixels: int
    removed_small_component_pixels: int
    removed_extra_focus_pixels: int
    hole_fill_pixels: int
    closing_added_pixels: int
    morphology_cleanup_applied: bool


def apply_necrosis_appearance(
    old_mask: np.ndarray,
    schema: MaskProfileSchema,
    context: MaskEditContext,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> PrimitiveEditResult:
    """Replace a plausible intratumoral candidate region with Necrosis."""

    _validate_necrosis_appearance_request(
        old_mask, schema, context, primitive_config, intent
    )
    mask = np.asarray(old_mask)
    normalized = context.normalized_mask
    target_fraction = _target_changed_fraction_for_intent(primitive_config, intent)

    necrosis_ids = schema.resolve_fine_ids("Necrosis")
    source_tumor = np.isin(normalized, schema.tumor_fine_ids)
    existing_necrosis = np.isin(normalized, necrosis_ids)
    tumor_pixels = int(np.count_nonzero(source_tumor))
    if tumor_pixels == 0:
        raise PrimitiveExecutionError("no_tumor")

    raw_candidate_base = source_tumor & ~existing_necrosis
    candidate_base = _necrosis_appearance_candidate_base(
        raw_candidate_base,
        existing_necrosis,
        primitive_config,
        intent,
    )
    candidate_pixels = int(np.count_nonzero(candidate_base))
    if candidate_pixels == 0:
        raise PrimitiveExecutionError("no_high_probability_hypoxic_candidate_region")

    existing_necrosis_pixels = int(np.count_nonzero(existing_necrosis))

    target_count = _target_pixels(target_fraction, tumor_pixels)
    capped_target_count = min(target_count, candidate_pixels)
    if capped_target_count < 1:
        raise PrimitiveExecutionError("no_high_probability_hypoxic_candidate_region")

    score_info = _necrosis_probability_score(
        normalized,
        schema,
        source_tumor,
        existing_necrosis,
        candidate_base,
        primitive_config,
        intent,
    )
    min_component_area = _min_component_area_for_intent(
        intent,
        target_pixels=capped_target_count,
    )
    max_components = _max_components_for_intent(
        intent,
        has_existing_necrosis=bool(np.any(existing_necrosis)),
    )
    selection = _select_connected_high_score_region(
        score_info.score,
        candidate_base,
        target_pixels=capped_target_count,
        min_component_area=min_component_area,
        max_components=max_components,
    )
    change_region, engulfment_log = _engulf_necrosis_intrusions(
        normalized,
        selection.change_region,
        schema=schema,
        primitive_config=primitive_config,
    )
    selected_pixels = int(np.count_nonzero(change_region))
    if selected_pixels == 0:
        raise PrimitiveExecutionError("no_high_probability_hypoxic_candidate_region")

    target_mask = np.array(normalized, copy=True)
    necrosis_label = int(necrosis_ids[0])
    target_mask[change_region] = necrosis_label

    changed_area_fraction = selected_pixels / target_mask.size
    ops_log = {
        "primitive": "necrosis_appearance",
        "reference_profile": schema.reference_profile,
        "target_change_fraction": target_fraction,
        "changed_area_fraction": changed_area_fraction,
        "changed_tumor_fraction": selected_pixels / tumor_pixels,
        "selected_pixels": selected_pixels,
        "necrosis_label": necrosis_label,
        "spatial": {
            "method": "necrosis_probability_intratumoral_replacement",
            "target_area_reference": "tumor",
            "target_pixels": int(target_count),
            "capped_target_pixels": int(capped_target_count),
            "tumor_pixels": tumor_pixels,
            "existing_necrosis_pixels": existing_necrosis_pixels,
            "necrosis_denominator_policy": "tumor_only",
            "candidate_pixels": candidate_pixels,
            "raw_candidate_pixels": int(np.count_nonzero(raw_candidate_base)),
            "candidate_domain_policy": (
                "tumor_near_existing_necrosis_expansion_band"
                if (
                    score_info.used_existing_necrosis_neighborhood
                    and not score_info.allow_multifocal_new_foci_when_existing_necrosis
                )
                else "original_tumor_only_excluding_existing_necrosis"
            ),
            "existing_necrosis_expansion_band_pixels": (
                score_info.existing_necrosis_expansion_band_pixels
            ),
            "allow_multifocal_new_foci_when_existing_necrosis": (
                score_info.allow_multifocal_new_foci_when_existing_necrosis
            ),
            "used_existing_necrosis_neighborhood": (
                score_info.used_existing_necrosis_neighborhood
            ),
            "used_blood_vessel_distance": score_info.used_blood_vessel_distance,
            "active_weights": dict(score_info.active_weights),
            "radii_px": dict(score_info.radii),
            "score_threshold": selection.score_threshold,
            "threshold_percentile": selection.threshold_percentile,
            "selected_components": selection.selected_components,
            "retry_applied": selection.retry_applied,
            "pre_cleanup_pixels": selection.pre_cleanup_pixels,
            "removed_small_component_pixels": selection.removed_small_component_pixels,
            "removed_extra_focus_pixels": selection.removed_extra_focus_pixels,
            "hole_fill_pixels": selection.hole_fill_pixels,
            "closing_added_pixels": selection.closing_added_pixels,
            "morphology_cleanup_applied": selection.morphology_cleanup_applied,
            "min_component_area_px": min_component_area,
            "max_components": max_components,
            "intrusion_engulfment": engulfment_log,
        },
    }

    return PrimitiveEditResult(
        target_mask=target_mask,
        change_region=change_region,
        changed_area_fraction=changed_area_fraction,
        selected_pixels=selected_pixels,
        warnings=(),
        ops_log=ops_log,
    )


def apply_necrosis_resolution(
    old_mask: np.ndarray,
    schema: MaskProfileSchema,
    context: MaskEditContext,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> PrimitiveEditResult:
    """Replace existing Necrosis with nearest Stroma backfill."""

    _validate_necrosis_resolution_request(
        old_mask, schema, context, primitive_config, intent
    )
    normalized = context.normalized_mask
    target_fraction = _target_necrosis_resolution_fraction_for_intent(
        primitive_config, intent
    )

    necrosis_ids = schema.resolve_fine_ids("Necrosis")
    necrosis = np.isin(normalized, necrosis_ids)
    necrosis_pixels = int(np.count_nonzero(necrosis))
    if necrosis_pixels == 0:
        raise PrimitiveExecutionError("no_necrosis")

    target_count = _target_pixels(target_fraction, necrosis_pixels)
    capped_target_count = min(target_count, necrosis_pixels)
    if capped_target_count < 1:
        raise PrimitiveExecutionError("necrosis_area_too_small")

    backfill_labels = _available_resolution_backfill_labels(
        normalized,
        schema,
        primitive_config,
    )
    backfill = np.zeros_like(necrosis, dtype=bool)
    for label in backfill_labels:
        backfill |= np.isin(normalized, schema.resolve_fine_ids(label))
    if not np.any(backfill):
        raise PrimitiveExecutionError("no_valid_backfill_tissue")

    score = _necrosis_resolution_score(necrosis, backfill, intent)
    min_component_area = _min_resolution_component_area_for_intent(intent)
    selection = _select_necrosis_resolution_region(
        score,
        necrosis,
        target_pixels=capped_target_count,
        min_component_area=min_component_area,
    )
    change_region = selection.change_region
    selected_pixels = int(np.count_nonzero(change_region))
    if selected_pixels == 0:
        raise PrimitiveExecutionError("necrosis_area_too_small")

    fill_labels = _nearest_backfill_labels(normalized, backfill)
    target_mask = np.array(normalized, copy=True)
    target_mask[change_region] = fill_labels[change_region]

    changed_area_fraction = selected_pixels / target_mask.size
    ops_log = {
        "primitive": "necrosis_resolution",
        "reference_profile": schema.reference_profile,
        "target_change_fraction": target_fraction,
        "changed_area_fraction": changed_area_fraction,
        "changed_necrosis_fraction": selected_pixels / necrosis_pixels,
        "selected_pixels": selected_pixels,
        "spatial": {
            "method": "nearest_stroma_or_tumor_fallback_backfill",
            "target_area_reference": "necrosis",
            "target_pixels": int(target_count),
            "capped_target_pixels": int(capped_target_count),
            "necrosis_pixels": necrosis_pixels,
            "backfill_labels": list(backfill_labels),
            "backfill_policy": (
                "nearest_tumor_fallback"
                if "Tumor" in backfill_labels
                else "nearest_stroma"
            ),
            "fallback_backfill_to_tumor": "Tumor" in backfill_labels,
            "backfill_pixels": int(np.count_nonzero(backfill)),
            "score_threshold": selection.score_threshold,
            "threshold_percentile": selection.threshold_percentile,
            "selected_components": selection.selected_components,
            "pre_cleanup_pixels": selection.pre_cleanup_pixels,
            "removed_small_component_pixels": selection.removed_small_component_pixels,
            "min_component_area_px": min_component_area,
        },
    }
    warnings = (
        ("necrosis_resolution_fallback_backfill_to_tumor",)
        if "Tumor" in backfill_labels
        else ()
    )

    return PrimitiveEditResult(
        target_mask=target_mask,
        change_region=change_region,
        changed_area_fraction=changed_area_fraction,
        selected_pixels=selected_pixels,
        warnings=warnings,
        ops_log=ops_log,
    )


def _validate_necrosis_appearance_request(
    old_mask: np.ndarray,
    schema: MaskProfileSchema,
    context: MaskEditContext,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> None:
    mask = np.asarray(old_mask)
    if mask.ndim != 2:
        raise PrimitiveExecutionError("necrosis_appearance requires a 2D mask.")
    if tuple(mask.shape) != context.mask_shape:
        raise PrimitiveExecutionError("old_mask shape must match MaskEditContext.")
    if schema.reference_profile != context.reference_profile:
        raise PrimitiveExecutionError(
            "schema.reference_profile must match context.reference_profile."
        )
    if intent.primitive != "necrosis_appearance":
        raise PrimitiveExecutionError(
            "apply_necrosis_appearance requires a necrosis_appearance intent."
        )
    if primitive_config.get("name") != "necrosis_appearance":
        raise PrimitiveExecutionError(
            "primitive_config must describe necrosis_appearance."
        )
    if "Necrosis" not in schema.readable_labels:
        raise PrimitiveExecutionError("no_necrosis_label")


def _validate_necrosis_resolution_request(
    old_mask: np.ndarray,
    schema: MaskProfileSchema,
    context: MaskEditContext,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> None:
    mask = np.asarray(old_mask)
    if mask.ndim != 2:
        raise PrimitiveExecutionError("necrosis_resolution requires a 2D mask.")
    if tuple(mask.shape) != context.mask_shape:
        raise PrimitiveExecutionError("old_mask shape must match MaskEditContext.")
    if schema.reference_profile != context.reference_profile:
        raise PrimitiveExecutionError(
            "schema.reference_profile must match context.reference_profile."
        )
    if intent.primitive != "necrosis_resolution":
        raise PrimitiveExecutionError(
            "apply_necrosis_resolution requires a necrosis_resolution intent."
        )
    if primitive_config.get("name") != "necrosis_resolution":
        raise PrimitiveExecutionError(
            "primitive_config must describe necrosis_resolution."
        )
    if "Necrosis" not in schema.readable_labels:
        raise PrimitiveExecutionError("no_necrosis_label")


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
            f"necrosis_appearance does not define strength {intent.strength}."
        )

    lower, upper = float(interval[0]), float(interval[1])
    return (lower + upper) / 2


def _target_necrosis_resolution_fraction_for_intent(
    primitive_config: Mapping[str, Any], intent: EditIntent
) -> float:
    if intent.target_change_fraction is not None:
        return intent.target_change_fraction

    intervals = primitive_config.get("parameter_ranges", {}).get(
        "necrosis_area_decrease_fraction", {}
    )
    interval = intervals.get(intent.strength)
    if not isinstance(interval, list) or len(interval) != 2:
        raise PrimitiveExecutionError(
            f"necrosis_resolution does not define strength {intent.strength}."
        )

    lower, upper = float(interval[0]), float(interval[1])
    return (lower + upper) / 2


def _available_resolution_backfill_labels(
    mask: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
) -> tuple[str, ...]:
    mask_operation = primitive_config.get("mask_operation", {})
    priority = (
        mask_operation.get("backfill_priority", ())
        if isinstance(mask_operation, Mapping)
        else ()
    )
    if not isinstance(priority, list):
        priority = ["Stroma", "Tumor"]

    # Prefer true stromal repair when any Stroma exists.  Tumor is a fallback
    # for tumor+necrosis-only masks where "resolved necrosis" means viable tumor.
    for allowed in ("Stroma", "Tumor"):
        if allowed not in priority or allowed not in schema.readable_labels:
            continue
        label_mask = np.isin(mask, schema.resolve_fine_ids(allowed))
        if np.any(label_mask):
            return (allowed,)
    raise PrimitiveExecutionError("no_valid_backfill_tissue")


def _necrosis_resolution_score(
    necrosis: np.ndarray,
    backfill: np.ndarray,
    intent: EditIntent,
) -> np.ndarray:
    dist_to_backfill = ndimage.distance_transform_edt(~backfill)
    max_dist = float(dist_to_backfill[necrosis].max()) if np.any(necrosis) else 0.0
    score = -dist_to_backfill.astype(float)
    noise_weight = _nonnegative_float_parameter(
        intent, "necrosis_resolution_noise_weight", 0.02
    )
    if noise_weight > 0 and max_dist > 0:
        score += noise_weight * _smooth_noise(necrosis.shape, seed=intent.seed)
    score[~necrosis] = -np.inf
    return score


def _select_necrosis_resolution_region(
    score: np.ndarray,
    necrosis: np.ndarray,
    *,
    target_pixels: int,
    min_component_area: int,
) -> _SelectionInfo:
    finite_scores = score[necrosis & np.isfinite(score)]
    if finite_scores.size == 0:
        raise PrimitiveExecutionError("necrosis_area_too_small")

    percentile = max(
        0.0,
        100.0 * (1.0 - min(float(target_pixels) / float(finite_scores.size), 1.0)),
    )
    threshold = float(np.percentile(finite_scores, percentile))
    selected = necrosis & np.isfinite(score) & (score >= threshold)
    pre_cleanup_pixels = int(np.count_nonzero(selected))
    selected = _limit_region_by_score(selected, score, target_pixels)
    selected, removed_pixels = _remove_small_components(selected, min_component_area)
    if not np.any(selected):
        selected = necrosis & np.isfinite(score) & (
            score >= float(np.percentile(finite_scores, 50.0))
        )
        percentile = 50.0
        threshold = float(np.percentile(finite_scores, percentile))
        pre_cleanup_pixels = int(np.count_nonzero(selected))
        selected = _limit_region_by_score(selected, score, target_pixels)
        selected, removed_pixels = _remove_small_components(selected, 1)

    selected_components = int(
        ndimage.label(selected, structure=_four_neighbor_structure())[1]
    )
    return _SelectionInfo(
        change_region=selected,
        score_threshold=threshold,
        threshold_percentile=float(percentile),
        selected_components=selected_components,
        retry_applied=False,
        pre_cleanup_pixels=pre_cleanup_pixels,
        removed_small_component_pixels=removed_pixels,
        removed_extra_focus_pixels=0,
        hole_fill_pixels=0,
        closing_added_pixels=0,
        morphology_cleanup_applied=False,
    )


def _nearest_backfill_labels(mask: np.ndarray, backfill: np.ndarray) -> np.ndarray:
    if not np.any(backfill):
        raise PrimitiveExecutionError("no_valid_backfill_tissue")
    _, indices = ndimage.distance_transform_edt(~backfill, return_indices=True)
    return mask[indices[0], indices[1]]


def _necrosis_appearance_candidate_base(
    raw_candidate_base: np.ndarray,
    existing_necrosis: np.ndarray,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> np.ndarray:
    if not np.any(existing_necrosis):
        return raw_candidate_base
    config_allows_multifocal = _config_bool(
        primitive_config.get("parameter_ranges", {}).get(
            "allow_multifocal_new_foci_when_existing_necrosis",
            False,
        )
    )
    if _intent_bool_parameter(
        intent,
        "allow_multifocal_new_foci_when_existing_necrosis",
        config_allows_multifocal,
    ):
        return raw_candidate_base

    radius_cap = max(1.0, float(min(raw_candidate_base.shape) // 4))
    necrosis_radius = _clamped_positive_float_parameter(
        intent,
        "necrosis_neighbor_radius_px",
        _DEFAULT_NECROSIS_NEIGHBOR_RADIUS_PX,
        radius_cap,
    )
    dist_to_necrosis = ndimage.distance_transform_edt(~existing_necrosis)
    return raw_candidate_base & (dist_to_necrosis <= necrosis_radius)


def _necrosis_probability_score(
    mask: np.ndarray,
    schema: MaskProfileSchema,
    source_tumor: np.ndarray,
    existing_necrosis: np.ndarray,
    candidate_base: np.ndarray,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> _ScoreInfo:
    shape = mask.shape
    radius_cap = max(1.0, float(min(shape) // 4))
    necrosis_radius = _clamped_positive_float_parameter(
        intent,
        "necrosis_neighbor_radius_px",
        _DEFAULT_NECROSIS_NEIGHBOR_RADIUS_PX,
        radius_cap,
    )
    interior_radius = _clamped_positive_float_parameter(
        intent,
        "tumor_interior_radius_px",
        _DEFAULT_TUMOR_INTERIOR_RADIUS_PX,
        radius_cap,
    )
    vessel_radius = _clamped_positive_float_parameter(
        intent,
        "vessel_avoidance_radius_px",
        _DEFAULT_VESSEL_AVOIDANCE_RADIUS_PX,
        radius_cap,
    )

    has_existing_necrosis = bool(np.any(existing_necrosis))
    vessel_mask = _blood_vessel_mask(mask, schema)
    has_vessel_pixels = bool(np.any(vessel_mask))
    weights = _active_candidate_weights(
        primitive_config,
        has_existing_necrosis=has_existing_necrosis,
        has_vessel_pixels=has_vessel_pixels,
    )

    score = np.zeros(shape, dtype=float)
    if "existing_necrosis_neighborhood" in weights:
        dist_to_necrosis = ndimage.distance_transform_edt(~existing_necrosis)
        score += weights["existing_necrosis_neighborhood"] * np.exp(
            -dist_to_necrosis / necrosis_radius
        )

    if "tumor_interior_far_from_outer_boundary" in weights:
        dist_inside_tumor = _edge_aware_tumor_interior_distance(
            source_tumor,
            pad_width=int(round(interior_radius)),
        )
        # Interior distance is a plausibility gate, not the shape generator.
        # Saturating it quickly avoids the repeated "mini tumor in the center"
        # failure mode on large tumor components.
        interior_score = 1.0 - np.exp(-dist_inside_tumor / max(interior_radius, 1.0))
        interior_weight = min(
            weights["tumor_interior_far_from_outer_boundary"],
            _nonnegative_float_parameter(
                intent,
                "necrosis_interior_score_weight_cap",
                _DEFAULT_INTERIOR_SCORE_WEIGHT_CAP,
            ),
        )
        score += interior_weight * interior_score

    if "tumor_region_far_from_blood_vessel" in weights:
        dist_to_vessel = ndimage.distance_transform_edt(~vessel_mask)
        vessel_score = np.clip(dist_to_vessel / vessel_radius, 0.0, 1.0)
        score += weights["tumor_region_far_from_blood_vessel"] * vessel_score

    noise_weight = _nonnegative_float_parameter(
        intent, "necrosis_score_noise_weight", _DEFAULT_NOISE_WEIGHT
    )
    if noise_weight > 0:
        score += noise_weight * _smooth_noise(shape, seed=intent.seed)

    score[~candidate_base] = -np.inf
    allow_multifocal_new_foci = _intent_bool_parameter(
        intent,
        "allow_multifocal_new_foci_when_existing_necrosis",
        _config_bool(
            primitive_config.get("parameter_ranges", {}).get(
                "allow_multifocal_new_foci_when_existing_necrosis",
                False,
            )
        ),
    )
    expansion_band_pixels = (
        int(np.count_nonzero(candidate_base))
        if has_existing_necrosis and not allow_multifocal_new_foci
        else 0
    )
    return _ScoreInfo(
        score=score,
        used_existing_necrosis_neighborhood=(
            "existing_necrosis_neighborhood" in weights
        ),
        used_blood_vessel_distance=("tumor_region_far_from_blood_vessel" in weights),
        existing_necrosis_expansion_band_pixels=expansion_band_pixels,
        allow_multifocal_new_foci_when_existing_necrosis=allow_multifocal_new_foci,
        active_weights=weights,
        radii={
            "necrosis_neighbor_radius_px": necrosis_radius,
            "tumor_interior_radius_px": interior_radius,
            "vessel_avoidance_radius_px": vessel_radius,
            "radius_cap_px": radius_cap,
        },
    )


def _active_candidate_weights(
    primitive_config: Mapping[str, Any],
    *,
    has_existing_necrosis: bool,
    has_vessel_pixels: bool,
) -> dict[str, float]:
    spatial = primitive_config.get("spatial_pattern", {})
    key = (
        "candidate_weights"
        if has_existing_necrosis
        else "candidate_weights_no_existing_necrosis"
    )
    raw = spatial.get(key, {}) if isinstance(spatial, Mapping) else {}
    if not isinstance(raw, Mapping):
        raise PrimitiveExecutionError(f"necrosis_appearance missing {key}.")

    active: dict[str, float] = {}
    for name, value in raw.items():
        if name == "existing_necrosis_neighborhood" and not has_existing_necrosis:
            continue
        if name == "tumor_region_far_from_blood_vessel" and not has_vessel_pixels:
            continue
        if not isinstance(value, (int, float)) or float(value) < 0:
            raise PrimitiveExecutionError(f"invalid necrosis candidate weight: {name}")
        if float(value) > 0:
            active[str(name)] = float(value)

    total = sum(active.values())
    if total <= 0:
        raise PrimitiveExecutionError("no_high_probability_hypoxic_candidate_region")
    return {name: value / total for name, value in active.items()}


def _blood_vessel_mask(mask: np.ndarray, schema: MaskProfileSchema) -> np.ndarray:
    if "Blood vessel" not in schema.readable_labels:
        return np.zeros(mask.shape, dtype=bool)
    return np.isin(mask, schema.resolve_fine_ids("Blood vessel"))


def _edge_aware_tumor_interior_distance(
    source_tumor: np.ndarray, *, pad_width: int
) -> np.ndarray:
    """Compute tumor-interior EDT without treating patch crop edges as tumor edges."""

    base_dist = ndimage.distance_transform_edt(source_tumor)
    if pad_width <= 0:
        return base_dist

    source = np.asarray(source_tumor, dtype=bool)
    padded = np.pad(source, pad_width, mode="constant", constant_values=False)

    inner_rows = slice(pad_width, pad_width + source.shape[0])
    inner_cols = slice(pad_width, pad_width + source.shape[1])

    padded[:pad_width, inner_cols] = source[0, :][np.newaxis, :]
    padded[pad_width + source.shape[0] :, inner_cols] = source[-1, :][np.newaxis, :]
    padded[inner_rows, :pad_width] = source[:, 0][:, np.newaxis]
    padded[inner_rows, pad_width + source.shape[1] :] = source[:, -1][:, np.newaxis]

    dist = ndimage.distance_transform_edt(padded)
    # The unpadded SciPy EDT already avoids treating out-of-array space as
    # background.  Keep that as a lower bound so false corner padding never
    # suppresses crop-edge tumor interiors.
    return np.maximum(base_dist, dist[inner_rows, inner_cols])


def _select_connected_high_score_region(
    score: np.ndarray,
    candidate_base: np.ndarray,
    *,
    target_pixels: int,
    min_component_area: int,
    max_components: int,
) -> _SelectionInfo:
    finite_scores = score[candidate_base & np.isfinite(score)]
    if finite_scores.size == 0:
        raise PrimitiveExecutionError("no_high_probability_hypoxic_candidate_region")

    first_percentile = max(
        0.0,
        100.0 * (1.0 - min(float(target_pixels) / float(finite_scores.size), 1.0)),
    )
    attempts = (first_percentile, 50.0)
    first_removed_pixels = 0
    first_pre_cleanup_pixels = 0
    best_info: _SelectionInfo | None = None
    min_acceptable_pixels = max(1, int(round(target_pixels * _MIN_SELECTED_TARGET_FRACTION)))
    for attempt_index, percentile in enumerate(attempts):
        threshold = float(np.percentile(finite_scores, percentile))
        high_score = candidate_base & np.isfinite(score) & (score >= threshold)
        selected = _select_components_by_score(
            high_score,
            score,
            target_pixels,
            max_components=max_components,
        )
        pre_cleanup_pixels = int(np.count_nonzero(selected))
        cleaned, removed_pixels = _remove_small_components(
            selected, min_component_area
        )
        cleaned, morph_info = _solidify_necrosis_region(
            cleaned,
            candidate_base,
            score,
            target_pixels,
        )
        cleaned, extra_removed_pixels = _keep_largest_components(
            cleaned,
            max_components=max_components,
        )
        cleaned, refill_pixels = _refill_to_minimum_target(
            cleaned,
            score,
            candidate_base,
            min_pixels=target_pixels,
        )
        if refill_pixels:
            cleaned, _ = _keep_largest_components(
                cleaned,
                max_components=max_components,
            )
        if attempt_index == 0:
            first_pre_cleanup_pixels = pre_cleanup_pixels
            first_removed_pixels = removed_pixels
        selected_pixels = int(np.count_nonzero(cleaned))
        if selected_pixels > 0:
            selected_components = int(
                ndimage.label(cleaned, structure=_four_neighbor_structure())[1]
            )
            info = _SelectionInfo(
                change_region=cleaned,
                score_threshold=threshold,
                threshold_percentile=float(percentile),
                selected_components=selected_components,
                retry_applied=bool(attempt_index > 0),
                pre_cleanup_pixels=pre_cleanup_pixels,
                removed_small_component_pixels=removed_pixels,
                removed_extra_focus_pixels=extra_removed_pixels,
                hole_fill_pixels=morph_info["hole_fill_pixels"],
                closing_added_pixels=morph_info["closing_added_pixels"],
                morphology_cleanup_applied=morph_info["morphology_cleanup_applied"],
            )
            if (
                best_info is None
                or selected_pixels > int(np.count_nonzero(best_info.change_region))
            ):
                best_info = info
            if selected_pixels >= min_acceptable_pixels or attempt_index == len(attempts) - 1:
                return info

    if best_info is not None:
        return best_info

    return _SelectionInfo(
        change_region=np.zeros_like(candidate_base, dtype=bool),
        score_threshold=float(np.percentile(finite_scores, attempts[-1])),
        threshold_percentile=float(attempts[-1]),
        selected_components=0,
        retry_applied=True,
        pre_cleanup_pixels=first_pre_cleanup_pixels,
        removed_small_component_pixels=first_removed_pixels,
        removed_extra_focus_pixels=0,
        hole_fill_pixels=0,
        closing_added_pixels=0,
        morphology_cleanup_applied=False,
    )


def _select_components_by_score(
    high_score: np.ndarray,
    score: np.ndarray,
    target_pixels: int,
    *,
    max_components: int,
) -> np.ndarray:
    candidate = high_score & np.isfinite(score)
    labeled_candidates, candidate_count = ndimage.label(
        candidate,
        structure=_four_neighbor_structure(),
    )
    if candidate_count == 0:
        return np.zeros_like(high_score, dtype=bool)

    components: list[tuple[float, int, int]] = []
    for component_id in range(1, candidate_count + 1):
        component = labeled_candidates == component_id
        area = int(np.count_nonzero(component))
        if area <= 0:
            continue
        component_scores = score[component]
        # Rank by high percentile instead of mean so a large component with a
        # broad mediocre center does not monopolize every edit.
        rank_score = float(np.percentile(component_scores, 90.0))
        components.append((rank_score, area, component_id))

    components.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if not components:
        return np.zeros_like(high_score, dtype=bool)

    chosen_components = components[:max_components]
    total_area = sum(area for _, area, _ in chosen_components)
    selected = np.zeros_like(high_score, dtype=bool)
    selected_count = 0
    for index, (_, area, component_id) in enumerate(chosen_components):
        component = labeled_candidates == component_id
        if index == len(chosen_components) - 1:
            quota = target_pixels - selected_count
        else:
            quota = int(round(target_pixels * area / max(total_area, 1)))
        quota = max(1, min(quota, area, target_pixels - selected_count))
        if quota <= 0:
            continue
        coords = np.argwhere(component)
        values = score[coords[:, 0], coords[:, 1]]
        order = np.argsort(values, kind="stable")[::-1]
        chosen = coords[order[:quota]]
        selected[chosen[:, 0], chosen[:, 1]] = True
        selected_count += int(chosen.shape[0])
    return selected


def _remove_small_components(
    selected: np.ndarray, min_component_area: int
) -> tuple[np.ndarray, int]:
    if min_component_area <= 1:
        return selected, 0

    labeled, count = ndimage.label(selected, structure=_four_neighbor_structure())
    cleaned = np.zeros_like(selected, dtype=bool)
    removed_pixels = 0
    for component_id in range(1, count + 1):
        component = labeled == component_id
        area = int(np.count_nonzero(component))
        if area >= min_component_area:
            cleaned |= component
        else:
            removed_pixels += area
    return cleaned, removed_pixels


def _keep_largest_components(
    selected: np.ndarray,
    *,
    max_components: int,
) -> tuple[np.ndarray, int]:
    if max_components < 1:
        return np.zeros_like(selected, dtype=bool), int(np.count_nonzero(selected))

    labeled, count = ndimage.label(selected, structure=_four_neighbor_structure())
    if count <= max_components:
        return selected, 0

    components: list[tuple[int, int]] = []
    for component_id in range(1, count + 1):
        area = int(np.count_nonzero(labeled == component_id))
        components.append((area, component_id))
    components.sort(reverse=True)
    keep_ids = {component_id for _, component_id in components[:max_components]}
    kept = np.isin(labeled, list(keep_ids))
    removed = int(np.count_nonzero(selected & ~kept))
    return kept, removed


def _refill_to_minimum_target(
    selected: np.ndarray,
    score: np.ndarray,
    candidate_base: np.ndarray,
    *,
    min_pixels: int,
) -> tuple[np.ndarray, int]:
    current_pixels = int(np.count_nonzero(selected))
    if current_pixels >= min_pixels:
        return selected, 0
    needed = min_pixels - current_pixels
    refill_domain = candidate_base & np.isfinite(score) & ~selected
    if not np.any(refill_domain):
        return selected, 0
    coords = np.argwhere(refill_domain)
    values = score[coords[:, 0], coords[:, 1]]
    order = np.argsort(values, kind="stable")[::-1]
    chosen = coords[order[:needed]]
    refilled = selected.copy()
    refilled[chosen[:, 0], chosen[:, 1]] = True
    return refilled, int(chosen.shape[0])


def _limit_region_by_score(
    selected: np.ndarray,
    score: np.ndarray,
    max_pixels: int,
) -> np.ndarray:
    current_pixels = int(np.count_nonzero(selected))
    if current_pixels <= max_pixels:
        return selected
    if max_pixels < 1:
        return np.zeros_like(selected, dtype=bool)

    coords = np.argwhere(selected)
    values = score[coords[:, 0], coords[:, 1]]
    order = np.argsort(values, kind="stable")[::-1]
    chosen = coords[order[:max_pixels]]
    limited = np.zeros_like(selected, dtype=bool)
    limited[chosen[:, 0], chosen[:, 1]] = True
    return limited


def _solidify_necrosis_region(
    selected: np.ndarray,
    candidate_base: np.ndarray,
    score: np.ndarray,
    target_pixels: int,
) -> tuple[np.ndarray, dict[str, int | bool]]:
    if not np.any(selected):
        return selected, {
            "hole_fill_pixels": 0,
            "closing_added_pixels": 0,
            "morphology_cleanup_applied": False,
        }

    before_pixels = int(np.count_nonzero(selected))
    filled = ndimage.binary_fill_holes(selected) & candidate_base
    hole_fill_pixels = int(np.count_nonzero(filled & ~selected))

    structure = _disk_structure(_DEFAULT_CLOSING_RADIUS_PX)
    closed = _edge_aware_binary_closing(filled, structure=structure) & candidate_base
    closing_added_pixels = int(np.count_nonzero(closed & ~filled))
    smoothed = _smooth_necrosis_boundary(closed, candidate_base)

    limited = _limit_region_by_boundary_distance(
        smoothed,
        score,
        max_pixels=target_pixels,
    )
    limited = ndimage.binary_fill_holes(limited) & candidate_base

    return limited, {
        "hole_fill_pixels": hole_fill_pixels,
        "closing_added_pixels": closing_added_pixels,
        "morphology_cleanup_applied": bool(
            hole_fill_pixels > 0
            or closing_added_pixels > 0
            or not np.array_equal(smoothed, closed)
        ),
    }


def _engulf_necrosis_intrusions(
    mask: np.ndarray,
    selected: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    engulfable = _necrosis_engulfable_intrusion_domain(
        mask,
        schema=schema,
        primitive_config=primitive_config,
    )
    if not np.any(selected) or not np.any(engulfable):
        return selected, {
            "enabled": True,
            "engulfed_pixels": 0,
            "engulfed_components": 0,
            "engulfed_label_pixels": {},
        }

    ranges = primitive_config.get("parameter_ranges", {})
    closing_radius = _nonnegative_float_parameter_from_config(
        ranges.get("necrosis_intrusion_closing_radius_px", 6.0),
        "necrosis_intrusion_closing_radius_px",
    )
    necrosis = np.isin(mask, schema.resolve_fine_ids("Necrosis"))
    necrosis_body = selected | necrosis
    if closing_radius <= 0:
        engulfed = np.zeros(mask.shape, dtype=bool)
    else:
        closed = _edge_aware_binary_closing(
            necrosis_body,
            structure=_disk_structure(int(round(closing_radius))),
        )
        pinched = _pinched_by_necrosis_body(
            necrosis_body,
            radius_px=closing_radius,
        )
        engulfed = (closed | pinched) & ~necrosis_body & engulfable
        engulfed &= ~np.isin(mask, schema.tumor_fine_ids)
    labeled, count = ndimage.label(engulfed, structure=_four_neighbor_structure())
    kept = np.zeros(mask.shape, dtype=bool)
    engulfed_components = 0
    for component_id in range(1, count + 1):
        component = labeled == component_id
        if np.any(component & ~engulfable):
            continue
        kept |= component
        engulfed_components += 1

    updated = selected | kept
    return updated, {
        "enabled": True,
        "closing_radius_px": float(closing_radius),
        "engulfed_pixels": int(np.count_nonzero(kept)),
        "engulfed_components": int(engulfed_components),
        "engulfed_label_pixels": _label_pixel_counts(
            mask,
            kept,
            schema=schema,
            labels=("Tumor", "Immune infiltrate", "Stroma", "Other tissue", "Normal epithelium"),
        ),
    }


def _pinched_by_necrosis_body(necrosis_body: np.ndarray, *, radius_px: float) -> np.ndarray:
    radius = max(1, int(round(float(radius_px))))
    left = np.zeros(necrosis_body.shape, dtype=bool)
    right = np.zeros(necrosis_body.shape, dtype=bool)
    up = np.zeros(necrosis_body.shape, dtype=bool)
    down = np.zeros(necrosis_body.shape, dtype=bool)
    for offset in range(1, radius + 1):
        left[:, offset:] |= necrosis_body[:, :-offset]
        right[:, :-offset] |= necrosis_body[:, offset:]
        up[offset:, :] |= necrosis_body[:-offset, :]
        down[:-offset, :] |= necrosis_body[offset:, :]
    return (left & right) | (up & down)


def _necrosis_engulfable_intrusion_domain(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
) -> np.ndarray:
    spatial = primitive_config.get("spatial_pattern", {})
    labels = (
        spatial.get("necrosis_engulf_intrusion_labels")
        if isinstance(spatial, Mapping)
        else None
    )
    if labels is None:
        mask_operation = primitive_config.get("mask_operation", {})
        labels = (
            mask_operation.get("necrosis_engulf_intrusion_labels")
            if isinstance(mask_operation, Mapping)
            else None
        )
    if labels is None:
        labels = [
            "Tumor",
            "Stroma",
            "Other tissue",
            "Normal epithelium",
        ]
    if not isinstance(labels, list):
        labels = ["Tumor", "Stroma", "Other tissue", "Normal epithelium"]
    domain = np.zeros(mask.shape, dtype=bool)
    for label in labels:
        if isinstance(label, str) and label in schema.readable_labels:
            domain |= np.isin(mask, schema.resolve_fine_ids(label))
    domain &= ~np.isin(mask, tuple(schema.skip_fine_ids))
    return domain


def _label_pixel_counts(
    mask: np.ndarray,
    region: np.ndarray,
    *,
    schema: MaskProfileSchema,
    labels: tuple[str, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        if label not in schema.readable_labels:
            continue
        count = int(
            np.count_nonzero(region & np.isin(mask, schema.resolve_fine_ids(label)))
        )
        if count:
            counts[label] = count
    return counts


def _smooth_necrosis_boundary(
    region: np.ndarray,
    candidate_base: np.ndarray,
) -> np.ndarray:
    if not np.any(region):
        return region
    sigma = float(_DEFAULT_FINAL_SMOOTH_RADIUS_PX)
    smoothed = np.zeros_like(region, dtype=bool)
    labeled, count = ndimage.label(region, structure=_four_neighbor_structure())
    for component_id in range(1, count + 1):
        component = labeled == component_id
        smoothed_component = _edge_aware_sdf_smooth(component, sigma=sigma)
        smoothed_component &= candidate_base
        if not np.any(smoothed_component):
            smoothed_component = component
        smoothed |= smoothed_component
    return ndimage.binary_fill_holes(smoothed) & candidate_base


def _edge_aware_sdf_smooth(mask: np.ndarray, *, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return mask.copy()
    pad_width = max(1, int(round(sigma * 3)))
    padded = np.pad(mask, pad_width, mode="edge")
    signed = (
        ndimage.distance_transform_edt(padded)
        - ndimage.distance_transform_edt(~padded)
    )
    smooth_signed = ndimage.gaussian_filter(signed, sigma=sigma)
    smoothed = smooth_signed >= 0
    rows = slice(pad_width, pad_width + mask.shape[0])
    cols = slice(pad_width, pad_width + mask.shape[1])
    return smoothed[rows, cols]


def _limit_region_by_boundary_distance(
    region: np.ndarray,
    score: np.ndarray,
    *,
    max_pixels: int,
) -> np.ndarray:
    current_pixels = int(np.count_nonzero(region))
    if current_pixels <= max_pixels:
        return region
    if max_pixels < 1:
        return np.zeros_like(region, dtype=bool)

    limited = region.copy()
    while int(np.count_nonzero(limited)) > max_pixels:
        eroded = _edge_aware_binary_erosion(
            limited,
            structure=_four_neighbor_structure(),
        )
        boundary = limited & ~eroded
        boundary_count = int(np.count_nonzero(boundary))
        if boundary_count == 0:
            break

        current_pixels = int(np.count_nonzero(limited))
        removable_pixels = current_pixels - max_pixels
        coords = np.argwhere(boundary)
        values = score[coords[:, 0], coords[:, 1]]
        order = np.argsort(values, kind="stable")
        remove_count = min(removable_pixels, boundary_count)
        remove = coords[order[:remove_count]]
        limited[remove[:, 0], remove[:, 1]] = False
        if remove_count < boundary_count:
            break
    return limited


def _disk_structure(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return (yy * yy + xx * xx) <= radius * radius


def _edge_aware_binary_erosion(
    mask: np.ndarray,
    *,
    structure: np.ndarray,
) -> np.ndarray:
    return ndimage.binary_erosion(mask, structure=structure, border_value=1)


def _edge_aware_binary_closing(
    mask: np.ndarray,
    *,
    structure: np.ndarray,
) -> np.ndarray:
    dilated = ndimage.binary_dilation(mask, structure=structure, border_value=0)
    return ndimage.binary_erosion(dilated, structure=structure, border_value=1)


def _edge_aware_binary_opening(
    mask: np.ndarray,
    *,
    structure: np.ndarray,
) -> np.ndarray:
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=1)
    return ndimage.binary_dilation(eroded, structure=structure, border_value=0)


def _clamped_positive_float_parameter(
    intent: EditIntent, key: str, default: float, cap: float
) -> float:
    value = intent.parameters.get(key, default)
    if not isinstance(value, (int, float)) or float(value) <= 0:
        raise PrimitiveExecutionError(f"parameters.{key} must be a positive number.")
    return min(float(value), float(cap))


def _nonnegative_float_parameter(
    intent: EditIntent, key: str, default: float
) -> float:
    value = intent.parameters.get(key, default)
    if not isinstance(value, (int, float)) or float(value) < 0:
        raise PrimitiveExecutionError(f"parameters.{key} must be a non-negative number.")
    return float(value)


def _nonnegative_float_parameter_from_config(value: Any, key: str) -> float:
    if not isinstance(value, (int, float)) or float(value) < 0:
        raise PrimitiveExecutionError(f"parameter_ranges.{key} must be a non-negative number.")
    return float(value)


def _intent_bool_parameter(intent: EditIntent, key: str, default: bool) -> bool:
    value = intent.parameters.get(key, default)
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    raise PrimitiveExecutionError(f"parameters.{key} must be a boolean.")


def _config_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    return False


def _min_component_area_for_intent(
    intent: EditIntent,
    *,
    target_pixels: int,
) -> int:
    value = intent.parameters.get(
        "min_necrosis_component_area_px",
        max(
            _DEFAULT_MIN_COMPONENT_AREA_PX,
            int(round(0.08 * max(1, target_pixels))),
        ),
    )
    if not isinstance(value, int) or value < 1:
        raise PrimitiveExecutionError(
            "parameters.min_necrosis_component_area_px must be a positive integer."
        )
    return value


def _max_components_for_intent(
    intent: EditIntent, *, has_existing_necrosis: bool
) -> int:
    default = (
        _DEFAULT_MAX_COMPONENTS_WITH_EXISTING_NECROSIS
        if has_existing_necrosis
        else _default_max_components_for_strength(intent.strength)
    )
    value = intent.parameters.get("max_necrosis_components", default)
    if not isinstance(value, int) or value < 1:
        raise PrimitiveExecutionError(
            "parameters.max_necrosis_components must be a positive integer."
        )
    return value


def _min_resolution_component_area_for_intent(intent: EditIntent) -> int:
    value = intent.parameters.get(
        "min_necrosis_resolution_component_area_px",
        _DEFAULT_MIN_RESOLUTION_COMPONENT_AREA_PX,
    )
    if not isinstance(value, int) or value < 1:
        raise PrimitiveExecutionError(
            "parameters.min_necrosis_resolution_component_area_px must be a positive integer."
        )
    return value


def _default_max_components_for_strength(strength: str) -> int:
    if strength == "mild":
        return 1
    if strength == "moderate":
        return 2
    if strength == "significant":
        return 3
    return _DEFAULT_MAX_COMPONENTS_NO_EXISTING_NECROSIS


def _smooth_noise(shape: tuple[int, int], *, seed: int | None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(shape)
    sigma = max(3.0, min(shape) / 64.0)
    noise = ndimage.gaussian_filter(raw, sigma=sigma)
    max_abs = float(np.abs(noise).max())
    if max_abs > 0:
        noise /= max_abs
    return noise

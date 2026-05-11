"""Generic stromal desmoplasia mask primitive."""

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


_DEFAULT_MIN_COMPONENT_AREA_PX = 64
_DEFAULT_NOISE_SIGMA_PX = 18.0
_DEFAULT_STROMA_NEIGHBOR_RADIUS_PX = 24.0
_DEFAULT_TUMOR_FALLOFF_RADIUS_PX = 48.0


@dataclass(frozen=True)
class _DesmoplasiaScoreInfo:
    score: np.ndarray
    active_weights: dict[str, float]
    radii_px: dict[str, float]
    candidate_pixels: int
    primary_candidate_pixels: int
    immune_candidate_pixels: int


def apply_stromal_desmoplasia(
    old_mask: np.ndarray,
    schema: MaskProfileSchema,
    context: MaskEditContext,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> PrimitiveEditResult:
    """Convert peritumoral non-tumor tissue into Stroma."""

    _validate_stromal_desmoplasia_request(
        old_mask, schema, context, primitive_config, intent
    )
    normalized = context.normalized_mask
    target_fraction = _target_stroma_delta_fraction_for_intent(
        primitive_config, intent
    )

    stroma_ids = schema.resolve_fine_ids("Stroma")
    tumor = np.isin(normalized, schema.tumor_fine_ids)
    stroma = np.isin(normalized, stroma_ids)
    if not np.any(tumor):
        raise PrimitiveExecutionError("no_tumor")
    if not np.any(stroma):
        raise PrimitiveExecutionError("no_stroma")

    candidate_base, candidate_log = _desmoplasia_candidate_domain(
        normalized,
        schema,
        primitive_config,
    )
    if not np.any(candidate_base):
        raise PrimitiveExecutionError("no_peritumoral_editable_region")

    score_info = _desmoplasia_score(
        normalized,
        schema,
        tumor=tumor,
        stroma=stroma,
        candidate_base=candidate_base,
        primitive_config=primitive_config,
        intent=intent,
    )
    max_immune_fraction = _max_immune_fraction_of_delta(primitive_config)
    target_count = _target_pixels(target_fraction, int(np.count_nonzero(stroma)))
    capped_target_count = min(target_count, score_info.candidate_pixels)
    if capped_target_count < 1:
        raise PrimitiveExecutionError("no_peritumoral_editable_region")

    selected = _select_high_score_desmoplasia_region(
        score_info.score,
        candidate_base,
        target_pixels=capped_target_count,
        min_component_area=_min_component_area_for_intent(intent),
        immune_mask=_safe_label_mask(normalized, schema, "Immune infiltrate"),
        max_immune_fraction=max_immune_fraction,
    )
    selected_pixels = int(np.count_nonzero(selected))
    if selected_pixels == 0:
        raise PrimitiveExecutionError("no_peritumoral_editable_region")

    target_mask = np.array(normalized, copy=True)
    stroma_label = int(stroma_ids[0])
    target_mask[selected] = stroma_label

    consumed_immune = (
        selected & _safe_label_mask(normalized, schema, "Immune infiltrate")
    )
    changed_area_fraction = selected_pixels / target_mask.size
    ops_log = {
        "primitive": "stromal_desmoplasia",
        "reference_profile": schema.reference_profile,
        "target_change_fraction": target_fraction,
        "changed_area_fraction": changed_area_fraction,
        "changed_stroma_fraction": selected_pixels / int(np.count_nonzero(stroma)),
        "selected_pixels": selected_pixels,
        "stroma_label": stroma_label,
        "spatial": {
            "method": "peritumoral_stroma_expansion_score_field",
            "target_area_reference": "original_stroma",
            "target_pixels": int(target_count),
            "capped_target_pixels": int(capped_target_count),
            "stroma_pixels": int(np.count_nonzero(stroma)),
            "tumor_pixels": int(np.count_nonzero(tumor)),
            "candidate_pixels": score_info.candidate_pixels,
            "primary_candidate_pixels": score_info.primary_candidate_pixels,
            "immune_candidate_pixels": score_info.immune_candidate_pixels,
            "selected_immune_pixels": int(np.count_nonzero(consumed_immune)),
            "max_distance_from_tumor_px": candidate_log["max_distance_from_tumor_px"],
            "max_immune_fraction_of_delta": candidate_log[
                "max_immune_fraction_of_delta"
            ],
            "immune_cap_enforced_in_selection": True,
            "require_direct_stroma_adjacency_for_immune": candidate_log[
                "require_direct_stroma_adjacency_for_immune"
            ],
            "active_weights": dict(score_info.active_weights),
            "radii_px": dict(score_info.radii_px),
            "min_component_area_px": _min_component_area_for_intent(intent),
        },
    }

    return PrimitiveEditResult(
        target_mask=target_mask,
        change_region=selected,
        changed_area_fraction=changed_area_fraction,
        selected_pixels=selected_pixels,
        warnings=(),
        ops_log=ops_log,
    )


def _validate_stromal_desmoplasia_request(
    old_mask: np.ndarray,
    schema: MaskProfileSchema,
    context: MaskEditContext,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> None:
    mask = np.asarray(old_mask)
    if mask.ndim != 2:
        raise PrimitiveExecutionError("stromal_desmoplasia requires a 2D mask.")
    if tuple(mask.shape) != context.mask_shape:
        raise PrimitiveExecutionError("old_mask shape must match MaskEditContext.")
    if schema.reference_profile != context.reference_profile:
        raise PrimitiveExecutionError(
            "schema.reference_profile must match context.reference_profile."
        )
    if intent.primitive != "stromal_desmoplasia":
        raise PrimitiveExecutionError(
            "apply_stromal_desmoplasia requires a stromal_desmoplasia intent."
        )
    if primitive_config.get("name") != "stromal_desmoplasia":
        raise PrimitiveExecutionError(
            "primitive_config must describe stromal_desmoplasia."
        )
    if "Stroma" not in schema.readable_labels:
        raise PrimitiveExecutionError("no_stroma_label")


def _target_stroma_delta_fraction_for_intent(
    primitive_config: Mapping[str, Any], intent: EditIntent
) -> float:
    if intent.target_change_fraction is not None:
        return intent.target_change_fraction

    intervals = primitive_config.get("parameter_ranges", {}).get(
        "stroma_area_delta_fraction", {}
    )
    interval = intervals.get(intent.strength)
    if not isinstance(interval, list) or len(interval) != 2:
        raise PrimitiveExecutionError(
            f"stromal_desmoplasia does not define strength {intent.strength}."
        )
    return (float(interval[0]) + float(interval[1])) / 2


def _desmoplasia_candidate_domain(
    mask: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    ranges = primitive_config.get("parameter_ranges", {})
    max_distance = _positive_float(
        ranges.get("max_distance_from_tumor_px", 64.0),
        "max_distance_from_tumor_px",
    )
    tumor = np.isin(mask, schema.tumor_fine_ids)
    stroma = _safe_label_mask(mask, schema, "Stroma")
    dist_to_tumor = ndimage.distance_transform_edt(~tumor)
    peritumoral = (dist_to_tumor <= max_distance) & ~tumor

    operation = primitive_config.get("mask_operation", {})
    primary_sources = (
        operation.get("primary_sources", ())
        if isinstance(operation, Mapping)
        else ()
    )
    primary = _label_mask(mask, schema, tuple(primary_sources))

    secondary_sources = (
        operation.get("secondary_sources", ())
        if isinstance(operation, Mapping)
        else ()
    )
    secondary = _label_mask(mask, schema, tuple(secondary_sources))

    spatial_pattern = primitive_config.get("spatial_pattern", {})
    constraints = (
        spatial_pattern.get("immune_to_stroma_constraints", {})
        if isinstance(spatial_pattern, Mapping)
        else {}
    )
    if not isinstance(constraints, Mapping):
        constraints = {}
    require_immune_stroma_adjacency = bool(
        constraints.get("require_direct_stroma_adjacency", True)
    )
    if require_immune_stroma_adjacency:
        secondary &= ndimage.binary_dilation(
            stroma, structure=_four_neighbor_structure()
        )

    candidate = peritumoral & (primary | secondary)
    candidate &= ~np.isin(mask, tuple(schema.skip_fine_ids))
    return candidate, {
        "max_distance_from_tumor_px": float(max_distance),
        "max_immune_fraction_of_delta": float(
            constraints.get("max_fraction_of_total_desmoplasia_delta", 0.30)
        ),
        "require_direct_stroma_adjacency_for_immune": require_immune_stroma_adjacency,
    }


def _desmoplasia_score(
    mask: np.ndarray,
    schema: MaskProfileSchema,
    *,
    tumor: np.ndarray,
    stroma: np.ndarray,
    candidate_base: np.ndarray,
    primitive_config: Mapping[str, Any],
    intent: EditIntent,
) -> _DesmoplasiaScoreInfo:
    ranges = primitive_config.get("parameter_ranges", {})
    tumor_radius = _positive_float(
        intent.parameters.get(
            "desmoplasia_tumor_falloff_radius_px",
            ranges.get("desmoplasia_tumor_falloff_radius_px", _DEFAULT_TUMOR_FALLOFF_RADIUS_PX),
        ),
        "desmoplasia_tumor_falloff_radius_px",
    )
    stroma_radius = _positive_float(
        intent.parameters.get(
            "desmoplasia_stroma_neighbor_radius_px",
            ranges.get("desmoplasia_stroma_neighbor_radius_px", _DEFAULT_STROMA_NEIGHBOR_RADIUS_PX),
        ),
        "desmoplasia_stroma_neighbor_radius_px",
    )
    noise_sigma = _positive_float(
        intent.parameters.get(
            "desmoplasia_noise_sigma_px",
            ranges.get("desmoplasia_noise_sigma_px", _DEFAULT_NOISE_SIGMA_PX),
        ),
        "desmoplasia_noise_sigma_px",
    )

    dist_to_tumor = ndimage.distance_transform_edt(~tumor)
    dist_to_stroma = ndimage.distance_transform_edt(~stroma)
    tumor_score = np.exp(-dist_to_tumor / tumor_radius)
    stroma_score = np.exp(-dist_to_stroma / stroma_radius)
    noise = _smooth_noise(mask.shape, seed=int(intent.seed or 0), sigma=noise_sigma)

    weights = _normalize_positive_weights(
        {
            "tumor_proximity": 0.45,
            "existing_stroma_adjacency": 0.45,
            "smooth_noise": 0.10,
        }
    )
    score = (
        weights["tumor_proximity"] * tumor_score
        + weights["existing_stroma_adjacency"] * stroma_score
        + weights["smooth_noise"] * noise
    )
    score[~candidate_base] = -np.inf
    immune = _safe_label_mask(mask, schema, "Immune infiltrate")
    primary = candidate_base & ~immune
    immune_candidate = candidate_base & immune
    return _DesmoplasiaScoreInfo(
        score=score,
        active_weights=weights,
        radii_px={
            "desmoplasia_tumor_falloff_radius_px": tumor_radius,
            "desmoplasia_stroma_neighbor_radius_px": stroma_radius,
            "desmoplasia_noise_sigma_px": noise_sigma,
        },
        candidate_pixels=int(np.count_nonzero(candidate_base)),
        primary_candidate_pixels=int(np.count_nonzero(primary)),
        immune_candidate_pixels=int(np.count_nonzero(immune_candidate)),
    )


def _select_high_score_desmoplasia_region(
    score: np.ndarray,
    candidate_base: np.ndarray,
    *,
    target_pixels: int,
    min_component_area: int,
    immune_mask: np.ndarray,
    max_immune_fraction: float,
) -> np.ndarray:
    if target_pixels < 1:
        return np.zeros(candidate_base.shape, dtype=bool)
    primary_domain = candidate_base & ~immune_mask
    immune_domain = candidate_base & immune_mask
    max_immune_pixels = int(np.floor(target_pixels * max_immune_fraction))
    primary_target = max(0, target_pixels - max_immune_pixels)

    primary_score = score.copy()
    primary_score[~primary_domain] = -np.inf
    selected = _top_k_mask(
        primary_score,
        min(primary_target, int(np.count_nonzero(primary_domain))),
    )

    shortfall = int(target_pixels) - int(np.count_nonzero(selected))
    if shortfall > 0 and max_immune_pixels > 0:
        immune_score = score.copy()
        immune_score[~immune_domain] = -np.inf
        selected |= _top_k_mask(
            immune_score,
            min(shortfall, max_immune_pixels, int(np.count_nonzero(immune_domain))),
        )

    shortfall = int(target_pixels) - int(np.count_nonzero(selected))
    if shortfall > 0:
        remaining_primary_score = score.copy()
        remaining_primary_score[~(primary_domain & ~selected)] = -np.inf
        selected |= _top_k_mask(remaining_primary_score, shortfall)

    selected = _remove_small_components(selected, min_component_area)
    shortfall = int(target_pixels) - int(np.count_nonzero(selected))
    if shortfall > 0:
        selected_immune = int(np.count_nonzero(selected & immune_mask))
        remaining_immune_budget = max(max_immune_pixels - selected_immune, 0)
        refill_domain = candidate_base & ~selected & ~immune_mask
        if remaining_immune_budget > 0:
            refill_domain |= candidate_base & ~selected & immune_mask
        refill_score = score.copy()
        refill_score[~refill_domain] = -np.inf
        refill = _top_k_mask(refill_score, shortfall)
        if remaining_immune_budget <= 0:
            refill &= ~immune_mask
        elif int(np.count_nonzero(refill & immune_mask)) > remaining_immune_budget:
            immune_refill_score = score.copy()
            immune_refill_score[~(refill & immune_mask)] = -np.inf
            kept_immune = _top_k_mask(immune_refill_score, remaining_immune_budget)
            refill = (refill & ~immune_mask) | kept_immune
        selected |= refill
    return selected & candidate_base


def _top_k_mask(score: np.ndarray, k: int) -> np.ndarray:
    result = np.zeros(score.shape, dtype=bool)
    if k <= 0:
        return result
    flat = score.ravel()
    finite = np.isfinite(flat)
    finite_count = int(np.count_nonzero(finite))
    if finite_count == 0:
        return result
    k = min(int(k), finite_count)
    finite_indices = np.flatnonzero(finite)
    finite_scores = flat[finite_indices]
    chosen_local = np.argpartition(finite_scores, -k)[-k:]
    result.ravel()[finite_indices[chosen_local]] = True
    return result


def _remove_small_components(selected: np.ndarray, min_component_area: int) -> np.ndarray:
    if min_component_area <= 1 or not np.any(selected):
        return selected
    labeled, count = ndimage.label(selected, structure=np.ones((3, 3), dtype=bool))
    kept = np.zeros_like(selected, dtype=bool)
    for component_id in range(1, count + 1):
        component = labeled == component_id
        if int(np.count_nonzero(component)) >= min_component_area:
            kept |= component
    return kept


def _min_component_area_for_intent(intent: EditIntent) -> int:
    value = intent.parameters.get(
        "min_desmoplasia_component_area_px",
        _DEFAULT_MIN_COMPONENT_AREA_PX,
    )
    if not isinstance(value, int) or value < 1:
        raise PrimitiveExecutionError(
            "parameters.min_desmoplasia_component_area_px must be a positive integer."
        )
    return value


def _safe_label_mask(
    mask: np.ndarray,
    schema: MaskProfileSchema,
    label: str,
) -> np.ndarray:
    if label not in schema.readable_labels:
        return np.zeros(mask.shape, dtype=bool)
    return np.isin(mask, schema.resolve_fine_ids(label))


def _label_mask(
    mask: np.ndarray,
    schema: MaskProfileSchema,
    labels: tuple[str, ...],
) -> np.ndarray:
    result = np.zeros(mask.shape, dtype=bool)
    for label in labels:
        if label in schema.readable_labels:
            result |= np.isin(mask, schema.resolve_fine_ids(label))
    return result


def _positive_float(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or float(value) <= 0:
        raise PrimitiveExecutionError(f"{name} must be positive.")
    return float(value)


def _max_immune_fraction_of_delta(primitive_config: Mapping[str, Any]) -> float:
    spatial_pattern = primitive_config.get("spatial_pattern", {})
    constraints = (
        spatial_pattern.get("immune_to_stroma_constraints", {})
        if isinstance(spatial_pattern, Mapping)
        else {}
    )
    if not isinstance(constraints, Mapping):
        return 0.30
    value = constraints.get("max_fraction_of_total_desmoplasia_delta", 0.30)
    if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
        raise PrimitiveExecutionError(
            "max_fraction_of_total_desmoplasia_delta must be in [0, 1]."
        )
    return float(value)


def _normalize_positive_weights(weights: Mapping[str, float]) -> dict[str, float]:
    total = sum(float(value) for value in weights.values() if float(value) > 0)
    if total <= 0:
        raise PrimitiveExecutionError("at least one desmoplasia score weight required.")
    return {
        key: float(value) / total
        for key, value in weights.items()
        if float(value) > 0
    }


def _smooth_noise(
    shape: tuple[int, int],
    *,
    seed: int,
    sigma: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=shape)
    smooth = ndimage.gaussian_filter(noise, sigma=sigma)
    min_val = float(smooth.min())
    max_val = float(smooth.max())
    if max_val <= min_val:
        return np.zeros(shape, dtype=float)
    return (smooth - min_val) / (max_val - min_val)

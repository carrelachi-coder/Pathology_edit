"""Organic score-field projection for LLM contour proposals.

This is the V2 MVP path: use the LLM polygon as a soft template, then choose
legal source-label pixels by a deterministic score field with area control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage

from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.generic.tumor_burden import PrimitiveEditResult


ORGANIC_PROJECTION_BACKEND = "organic_score_projection_v2"


@dataclass(frozen=True)
class OrganicProjectionPolicy:
    spatial_score: np.ndarray
    policy_name: str
    policy_params: dict[str, Any]
    legal_domain: np.ndarray
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class OrganicProjectionParams:
    template_sigma: float
    noise_sigma: float
    noise_amplitude: float
    w_template: float
    w_spatial: float
    w_noise: float
    min_component_fraction: float
    fill_holes_max_area_px: int
    decay_px: float
    min_template_legal_overlap_fraction: float
    min_selected_template_iou: float
    template_neighborhood_radius_px: float
    template_spillover_fraction: float


def apply_organic_projected_label_write(
    old_mask: np.ndarray,
    raw_candidate: np.ndarray,
    *,
    schema: MaskProfileSchema,
    source_labels: Sequence[str],
    target_label: str,
    primitive_config: Mapping[str, Any] | None = None,
    preserve_labels: Sequence[str] = (),
    forbidden_labels: Sequence[str] = (),
    seed: int = 0,
    target_pixels: int | None = None,
    template_sigma: float | None = None,
    noise_sigma: float | None = None,
    noise_amplitude: float | None = None,
    w_template: float | None = None,
    w_spatial: float | None = None,
    w_noise: float | None = None,
    decay_px: float | None = None,
) -> PrimitiveEditResult:
    """Project a rough LLM template to legal pixels with organic area control."""

    mask = np.asarray(old_mask)
    if mask.ndim != 2:
        raise ValueError("old_mask must be a 2D id mask.")

    candidate = np.asarray(raw_candidate, dtype=bool)
    if candidate.shape != mask.shape:
        raise ValueError(
            "raw_candidate shape must match old_mask shape: "
            f"{candidate.shape} != {mask.shape}."
        )

    primitive_name = str((primitive_config or {}).get("name", ""))
    params = _projection_params(
        primitive_config or {},
        template_sigma=template_sigma,
        noise_sigma=noise_sigma,
        noise_amplitude=noise_amplitude,
        w_template=w_template,
        w_spatial=w_spatial,
        w_noise=w_noise,
        decay_px=decay_px,
    )
    legal_domain = _legal_domain(
        mask,
        schema=schema,
        source_labels=source_labels,
        preserve_labels=preserve_labels,
        forbidden_labels=forbidden_labels,
    )
    policy = _policy_for_primitive(
        mask,
        schema=schema,
        primitive_name=primitive_name,
        primitive_config=primitive_config or {},
        source_labels=source_labels,
        target_label=target_label,
        legal_domain=legal_domain,
    )
    legal_domain = policy.legal_domain
    legal_pixels = int(np.count_nonzero(legal_domain))
    raw_candidate_pixels = int(np.count_nonzero(candidate))
    raw_legal_overlap = int(np.count_nonzero(candidate & legal_domain))

    if target_pixels is None:
        target_pixels = _target_pixels_from_config(
            mask,
            schema=schema,
            primitive_config=primitive_config or {},
            legal_pixels=legal_pixels,
        )
    target_pixels = max(int(target_pixels or 0), 0)
    if primitive_name == "necrosis_appearance":
        remaining_allowed = int(
            policy.policy_params.get("remaining_allowed_necrosis_pixels", target_pixels)
        )
        target_pixels = min(target_pixels, max(remaining_allowed, 0))
    selected_target_pixels = min(target_pixels, legal_pixels)

    if legal_pixels == 0 or selected_target_pixels == 0:
        return _empty_result(
            mask,
            schema=schema,
            target_label=target_label,
            source_labels=source_labels,
            preserve_labels=preserve_labels,
            forbidden_labels=forbidden_labels,
            primitive_name=primitive_name,
            raw_candidate_pixels=raw_candidate_pixels,
            legal_pixels=legal_pixels,
            target_pixels=target_pixels,
            seed=seed,
            component_policy=policy,
            raw_legal_overlap=raw_legal_overlap,
        )
    if raw_legal_overlap == 0:
        return _empty_result(
            mask,
            schema=schema,
            target_label=target_label,
            source_labels=source_labels,
            preserve_labels=preserve_labels,
            forbidden_labels=forbidden_labels,
            primitive_name=primitive_name,
            raw_candidate_pixels=raw_candidate_pixels,
            legal_pixels=legal_pixels,
            target_pixels=target_pixels,
            seed=seed,
            component_policy=policy,
            raw_legal_overlap=raw_legal_overlap,
            extra_warnings=("organic_projection_template_no_legal_overlap",),
        )
    template_overlap_fraction = (
        raw_legal_overlap / raw_candidate_pixels if raw_candidate_pixels else 0.0
    )
    if template_overlap_fraction < params.min_template_legal_overlap_fraction:
        return _empty_result(
            mask,
            schema=schema,
            target_label=target_label,
            source_labels=source_labels,
            preserve_labels=preserve_labels,
            forbidden_labels=forbidden_labels,
            primitive_name=primitive_name,
            raw_candidate_pixels=raw_candidate_pixels,
            legal_pixels=legal_pixels,
            target_pixels=target_pixels,
            seed=seed,
            component_policy=policy,
            raw_legal_overlap=raw_legal_overlap,
            extra_warnings=("organic_projection_template_legal_overlap_too_low",),
        )

    template_score = _template_score(candidate, sigma=params.template_sigma)
    spatial_score = policy.spatial_score
    noise = _smooth_noise(mask.shape, seed=seed, sigma=params.noise_sigma)

    template_norm = _normalize_on_domain(template_score, legal_domain)
    spatial_norm = _normalize_on_domain(spatial_score, legal_domain)
    noise_norm = _normalize_on_domain(noise, legal_domain)

    final_score = (
        params.w_template * template_norm
        + params.w_spatial * spatial_norm
        + params.w_noise * params.noise_amplitude * noise_norm
    )
    final_score = np.asarray(final_score, dtype=float)
    final_score[~legal_domain] = -np.inf

    selected, selection_log = _template_constrained_top_k_mask(
        final_score,
        legal_domain=legal_domain,
        raw_template=candidate,
        target_pixels=selected_target_pixels,
        neighborhood_radius_px=params.template_neighborhood_radius_px,
        spillover_fraction=params.template_spillover_fraction,
    )
    selected, cleanup_log = _cleanup_and_refill_once(
        selected,
        legal_domain=legal_domain,
        final_score=final_score,
        target_pixels=selected_target_pixels,
        min_component_fraction=params.min_component_fraction,
        fill_holes_max_area_px=params.fill_holes_max_area_px,
    )
    target_ids = schema.resolve_fine_ids(target_label)
    target_mask = np.array(mask, copy=True)
    target_mask[selected] = int(target_ids[0])

    selected_pixels = int(np.count_nonzero(selected))
    changed_area_fraction = selected_pixels / int(mask.size)
    selected_template_intersection = int(np.count_nonzero(selected & candidate))
    selected_template_union = int(np.count_nonzero(selected | candidate))
    selected_template_iou = (
        selected_template_intersection / selected_template_union
        if selected_template_union
        else 0.0
    )
    warnings: list[str] = []
    if selected_pixels == 0:
        warnings.append("proposal_projected_region_empty")
    if selected_pixels < target_pixels:
        warnings.append("organic_projection_area_shortfall")
    if cleanup_log["post_cleanup_pixels"] < cleanup_log["pre_cleanup_pixels"]:
        warnings.append("organic_projection_cleanup_removed_pixels")
    if (
        params.min_selected_template_iou > 0
        and selected_pixels > 0
        and selected_template_iou < params.min_selected_template_iou
    ):
        warnings.append("organic_projection_selected_template_iou_low")
    warnings.extend(policy.warnings)

    ops_log = {
        "backend": ORGANIC_PROJECTION_BACKEND,
        "method": "organic_score_projection_and_deterministic_write",
        "primitive": primitive_name,
        "reference_profile": schema.reference_profile,
        "source_labels": list(source_labels),
        "target_label": target_label,
        "target_fine_id": int(target_ids[0]),
        "preserve_labels": list(preserve_labels),
        "forbidden_labels": list(forbidden_labels),
        "raw_candidate_pixels": raw_candidate_pixels,
        "candidate_pixels": raw_candidate_pixels,
        "raw_candidate_legal_overlap_pixels": raw_legal_overlap,
        "template_overlap_with_legal_domain": (
            raw_legal_overlap / raw_candidate_pixels if raw_candidate_pixels else 0.0
        ),
        "legal_domain_pixels": legal_pixels,
        "target_pixels": target_pixels,
        "projected_pixels": selected_pixels,
        "selected_pixels": selected_pixels,
        "selected_raw_template_intersection_pixels": selected_template_intersection,
        "selected_raw_template_union_pixels": selected_template_union,
        "selected_raw_template_iou": selected_template_iou,
        "selection_policy": {
            "name": "template_neighborhood_constrained_top_k",
            **selection_log,
        },
        "area_shortfall": int(max(target_pixels - selected_pixels, 0)),
        "projection_retained_fraction": (
            selected_pixels / raw_candidate_pixels if raw_candidate_pixels else 0.0
        ),
        "changed_area_fraction": changed_area_fraction,
        "projection_backend": ORGANIC_PROJECTION_BACKEND,
        "noise_seed": int(seed),
        "score_terms": {
            "w_template": float(params.w_template),
            "w_spatial": float(params.w_spatial),
            "w_noise": float(params.w_noise),
            "template_sigma": float(params.template_sigma),
            "noise_sigma": float(params.noise_sigma),
            "noise_amplitude": float(params.noise_amplitude),
            "decay_px": float(params.decay_px),
            "min_template_legal_overlap_fraction": float(
                params.min_template_legal_overlap_fraction
            ),
            "min_selected_template_iou": float(params.min_selected_template_iou),
            "template_neighborhood_radius_px": float(
                params.template_neighborhood_radius_px
            ),
            "template_spillover_fraction": float(params.template_spillover_fraction),
        },
        "component_policy": {
            "policy_name": policy.policy_name,
            "params": policy.policy_params,
            "spatial_score_stats": _score_stats(spatial_norm, legal_domain),
            "template_score_stats": _score_stats(template_norm, legal_domain),
        },
        "cleanup_removed_pixels": cleanup_log["cleanup_removed_pixels"],
        "cleanup_refill_pixels": cleanup_log["cleanup_refill_pixels"],
        "pre_cleanup_pixels": cleanup_log["pre_cleanup_pixels"],
        "post_cleanup_pixels": cleanup_log["post_cleanup_pixels"],
        "cleanup_single_pass": True,
        "cleanup_iteration_limit": 1,
    }

    return PrimitiveEditResult(
        target_mask=target_mask,
        change_region=selected,
        changed_area_fraction=changed_area_fraction,
        selected_pixels=selected_pixels,
        warnings=tuple(warnings),
        ops_log=ops_log,
    )


def _legal_domain(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    source_labels: Sequence[str],
    preserve_labels: Sequence[str],
    forbidden_labels: Sequence[str],
) -> np.ndarray:
    source_mask = _label_mask(mask, schema, source_labels)
    removal_labels = tuple(dict.fromkeys(tuple(preserve_labels) + tuple(forbidden_labels)))
    removal_mask = (
        _label_mask(mask, schema, removal_labels)
        if removal_labels
        else np.zeros(mask.shape, dtype=bool)
    )
    background_mask = np.isin(mask, tuple(schema.skip_fine_ids))
    return source_mask & ~removal_mask & ~background_mask


def _projection_params(
    primitive_config: Mapping[str, Any],
    *,
    template_sigma: float | None,
    noise_sigma: float | None,
    noise_amplitude: float | None,
    w_template: float | None,
    w_spatial: float | None,
    w_noise: float | None,
    decay_px: float | None,
) -> OrganicProjectionParams:
    ranges = primitive_config.get("parameter_ranges", {})
    weights = ranges.get("organic_score_weights", {})
    if not isinstance(weights, Mapping):
        weights = {}
    return OrganicProjectionParams(
        template_sigma=_positive_float(
            template_sigma, ranges.get("organic_template_sigma_px", 3.0)
        ),
        noise_sigma=_positive_float(
            noise_sigma, ranges.get("organic_noise_sigma_px", 18.0)
        ),
        noise_amplitude=_nonnegative_float(
            noise_amplitude, ranges.get("organic_noise_amplitude", 0.18)
        ),
        w_template=_nonnegative_float(
            w_template, weights.get("template", 0.45)
        ),
        w_spatial=_nonnegative_float(
            w_spatial, weights.get("spatial", 0.45)
        ),
        w_noise=_nonnegative_float(w_noise, weights.get("noise", 0.10)),
        min_component_fraction=_nonnegative_float(
            None, ranges.get("organic_min_component_fraction", 0.01)
        ),
        fill_holes_max_area_px=int(
            max(0, ranges.get("organic_fill_holes_max_area_px", 0))
        ),
        decay_px=_positive_float(
            decay_px, ranges.get("peritumoral_falloff_radius_px", 48.0)
        ),
        min_template_legal_overlap_fraction=_nonnegative_float(
            None, ranges.get("organic_min_template_legal_overlap_fraction", 0.05)
        ),
        min_selected_template_iou=_nonnegative_float(
            None, ranges.get("organic_min_selected_template_iou", 0.0)
        ),
        template_neighborhood_radius_px=_positive_float(
            None, ranges.get("organic_template_neighborhood_radius_px", 48.0)
        ),
        template_spillover_fraction=_fraction_float(
            ranges.get("organic_template_spillover_fraction", 0.15)
        ),
    )


def _positive_float(value: float | None, default: Any) -> float:
    raw = default if value is None else value
    if not isinstance(raw, (int, float)) or float(raw) <= 0:
        raise ValueError("organic projection parameter must be positive.")
    return float(raw)


def _nonnegative_float(value: float | None, default: Any) -> float:
    raw = default if value is None else value
    if not isinstance(raw, (int, float)) or float(raw) < 0:
        raise ValueError("organic projection parameter must be non-negative.")
    return float(raw)


def _fraction_float(value: Any) -> float:
    if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
        raise ValueError("organic projection fraction parameter must be in [0, 1].")
    return float(value)


def _policy_for_primitive(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive_name: str,
    primitive_config: Mapping[str, Any],
    source_labels: Sequence[str],
    target_label: str,
    legal_domain: np.ndarray,
) -> OrganicProjectionPolicy:
    if primitive_name == "stromal_immune_infiltration":
        return _stromal_immune_policy(
            mask,
            schema=schema,
            primitive_config=primitive_config,
            legal_domain=legal_domain,
        )
    if primitive_name == "necrosis_appearance":
        return _necrosis_policy(
            mask,
            schema=schema,
            primitive_config=primitive_config,
            legal_domain=legal_domain,
        )
    return _generic_policy(
        mask,
        schema=schema,
        source_labels=source_labels,
        target_label=target_label,
        legal_domain=legal_domain,
    )


def _stromal_immune_policy(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    legal_domain: np.ndarray,
) -> OrganicProjectionPolicy:
    ranges = primitive_config.get("parameter_ranges", {})
    stroma = _safe_label_mask(mask, schema, "Stroma")
    legal = legal_domain & stroma
    tumor = np.isin(mask, schema.tumor_fine_ids)
    decay_px = _positive_float(
        None, ranges.get("peritumoral_falloff_radius_px", 48.0)
    )
    score = np.zeros(mask.shape, dtype=float)
    if np.any(tumor):
        dist_to_tumor = ndimage.distance_transform_edt(~tumor)
        score += np.exp(-dist_to_tumor / decay_px)

    immune_radius = _positive_float(None, ranges.get("immune_neighbor_radius_px", 48.0))
    immune = _safe_label_mask(mask, schema, "Immune infiltrate")
    used_existing_immune = bool(np.any(immune))
    if used_existing_immune:
        dist_to_immune = ndimage.distance_transform_edt(~immune)
        score += 0.25 * np.exp(-dist_to_immune / immune_radius)

    necrosis = _safe_label_mask(mask, schema, "Necrosis")
    penalty_radius = _positive_float(
        None, ranges.get("necrosis_adjacency_penalty_radius_px", 32.0)
    )
    penalty_weight = _nonnegative_float(
        None, ranges.get("necrosis_adjacency_penalty_weight", 0.20)
    )
    used_necrosis_penalty = bool(np.any(necrosis) and penalty_weight > 0)
    if used_necrosis_penalty:
        dist_to_necrosis = ndimage.distance_transform_edt(~necrosis)
        score -= penalty_weight * np.exp(-dist_to_necrosis / penalty_radius)

    score[~legal] = 0.0
    return OrganicProjectionPolicy(
        spatial_score=score,
        policy_name="stromal_immune_peritumoral",
        policy_params={
            "legal_domain_policy": "original_stroma_only",
            "decay_px": decay_px,
            "immune_neighbor_radius_px": immune_radius,
            "used_existing_immune_adjacency": used_existing_immune,
            "necrosis_adjacency_penalty_radius_px": penalty_radius,
            "necrosis_adjacency_penalty_weight": penalty_weight,
            "used_necrosis_adjacency_penalty": used_necrosis_penalty,
        },
        legal_domain=legal,
    )


def _necrosis_policy(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    legal_domain: np.ndarray,
) -> OrganicProjectionPolicy:
    ranges = primitive_config.get("parameter_ranges", {})
    tumor = np.isin(mask, schema.tumor_fine_ids)
    necrosis = _safe_label_mask(mask, schema, "Necrosis")
    legal = legal_domain & tumor & ~necrosis
    score = np.zeros(mask.shape, dtype=float)

    boundary_radius = _positive_float(
        None, ranges.get("tumor_boundary_margin_radius_px", 64.0)
    )
    interior_dist = _edge_aware_tumor_interior_distance(
        tumor,
        pad_width=int(round(boundary_radius)),
    )
    score += np.clip(interior_dist / boundary_radius, 0.0, 1.0)

    necrosis_radius = _positive_float(
        None, ranges.get("necrosis_neighbor_radius_px", 48.0)
    )
    used_existing_necrosis = bool(np.any(necrosis))
    if used_existing_necrosis:
        dist_to_necrosis = ndimage.distance_transform_edt(~necrosis)
        score += 0.45 * np.exp(-dist_to_necrosis / necrosis_radius)

    vessel = _safe_label_mask(mask, schema, "Blood vessel")
    vessel_radius = _positive_float(
        None, ranges.get("vessel_avoidance_radius_px", 96.0)
    )
    vessel_weight = _nonnegative_float(
        None, ranges.get("vessel_avoidance_weight", 0.20)
    )
    used_vessel_avoidance = bool(np.any(vessel) and vessel_weight > 0)
    if used_vessel_avoidance:
        dist_to_vessel = ndimage.distance_transform_edt(~vessel)
        vessel_penalty = 1.0 - np.clip(dist_to_vessel / vessel_radius, 0.0, 1.0)
        score -= vessel_weight * vessel_penalty

    score[~legal] = 0.0
    tumor_pixels = int(np.count_nonzero(tumor))
    existing_necrosis_pixels = int(np.count_nonzero(necrosis))
    max_fraction = _max_necrosis_fraction_of_tumor(primitive_config)
    max_necrosis_pixels = int(round(max_fraction * tumor_pixels))
    return OrganicProjectionPolicy(
        spatial_score=score,
        policy_name="necrosis_intratumoral_hypoxic",
        policy_params={
            "legal_domain_policy": "original_tumor_only_excluding_existing_necrosis",
            "tumor_pixels": tumor_pixels,
            "existing_necrosis_pixels": existing_necrosis_pixels,
            "max_necrosis_fraction_of_tumor": max_fraction,
            "max_necrosis_pixels": max_necrosis_pixels,
            "remaining_allowed_necrosis_pixels": int(
                max(max_necrosis_pixels - existing_necrosis_pixels, 0)
            ),
            "necrosis_denominator_policy": (
                "original_tumor_fine_ids_only_matches_validation"
            ),
            "tumor_boundary_margin_radius_px": boundary_radius,
            "necrosis_neighbor_radius_px": necrosis_radius,
            "used_existing_necrosis_neighborhood": used_existing_necrosis,
            "vessel_avoidance_radius_px": vessel_radius,
            "vessel_avoidance_weight": vessel_weight,
            "used_vessel_avoidance": used_vessel_avoidance,
        },
        legal_domain=legal,
    )


def _generic_policy(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    source_labels: Sequence[str],
    target_label: str,
    legal_domain: np.ndarray,
) -> OrganicProjectionPolicy:
    interior = ndimage.distance_transform_edt(legal_domain)
    return OrganicProjectionPolicy(
        spatial_score=interior.astype(float),
        policy_name="generic_label_safe",
        policy_params={
            "source_labels": list(source_labels),
            "target_label": target_label,
            "legal_domain_policy": "source_labels_only",
            "reference_profile": schema.reference_profile,
        },
        legal_domain=legal_domain,
        warnings=("organic_projection_generic_policy_used",),
    )


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
    labels: Sequence[str],
) -> np.ndarray:
    result = np.zeros(mask.shape, dtype=bool)
    for label in labels:
        result |= np.isin(mask, schema.resolve_fine_ids(label))
    return result


def _template_score(candidate: np.ndarray, *, sigma: float) -> np.ndarray:
    if not np.any(candidate):
        return np.zeros(candidate.shape, dtype=float)
    inside = ndimage.distance_transform_edt(candidate)
    outside = ndimage.distance_transform_edt(~candidate)
    signed = inside - outside
    return ndimage.gaussian_filter(signed.astype(float), sigma=float(sigma))


def _spatial_score(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive_name: str,
    legal_domain: np.ndarray,
    decay_px: float,
) -> np.ndarray:
    if primitive_name == "stromal_immune_infiltration":
        tumor = np.isin(mask, schema.tumor_fine_ids)
        if np.any(tumor):
            dist_to_tumor = ndimage.distance_transform_edt(~tumor)
            score = np.exp(-dist_to_tumor / max(float(decay_px), 1.0))
            score[~legal_domain] = 0.0
            return score.astype(float)
    return legal_domain.astype(float)


def _smooth_noise(
    shape: tuple[int, int],
    *,
    seed: int,
    sigma: float,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    field = rng.normal(size=shape)
    return ndimage.gaussian_filter(field, sigma=float(sigma))


def _normalize_on_domain(values: np.ndarray, domain: np.ndarray) -> np.ndarray:
    result = np.zeros(values.shape, dtype=float)
    data = np.asarray(values, dtype=float)[domain]
    if data.size == 0:
        return result
    low = float(np.percentile(data, 2))
    high = float(np.percentile(data, 98))
    if high <= low:
        std = float(data.std())
        if std <= 1e-8:
            return result
        normalized = (np.asarray(values, dtype=float) - float(data.mean())) / std
        normalized = np.clip((normalized + 2.0) / 4.0, 0.0, 1.0)
    else:
        normalized = (np.asarray(values, dtype=float) - low) / (high - low)
        normalized = np.clip(normalized, 0.0, 1.0)
    result[domain] = normalized[domain]
    return result


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
    selected_local = np.argpartition(finite_scores, -k)[-k:]
    result.ravel()[finite_indices[selected_local]] = True
    return result


def _template_constrained_top_k_mask(
    score: np.ndarray,
    *,
    legal_domain: np.ndarray,
    raw_template: np.ndarray,
    target_pixels: int,
    neighborhood_radius_px: float,
    spillover_fraction: float,
) -> tuple[np.ndarray, dict[str, int | float]]:
    template_dist = ndimage.distance_transform_edt(~np.asarray(raw_template, dtype=bool))
    primary_zone = legal_domain & (template_dist <= float(neighborhood_radius_px))
    primary_budget = int(round(int(target_pixels) * (1.0 - float(spillover_fraction))))
    primary_budget = min(max(primary_budget, 0), int(target_pixels))
    selected = _top_k_mask_for_refill(
        score,
        legal_domain=primary_zone,
        k=primary_budget,
    )
    primary_selected_pixels = int(np.count_nonzero(selected))

    remaining = int(target_pixels) - primary_selected_pixels
    secondary_selected_pixels = 0
    if remaining > 0:
        secondary = _top_k_mask_for_refill(
            score,
            legal_domain=legal_domain & ~selected,
            k=remaining,
        )
        secondary_selected_pixels = int(np.count_nonzero(secondary))
        selected |= secondary

    return selected, {
        "template_neighborhood_radius_px": float(neighborhood_radius_px),
        "template_spillover_fraction": float(spillover_fraction),
        "primary_zone_pixels": int(np.count_nonzero(primary_zone)),
        "primary_budget_pixels": int(primary_budget),
        "primary_selected_pixels": int(primary_selected_pixels),
        "secondary_selected_pixels": int(secondary_selected_pixels),
        "selected_inside_primary_zone_pixels": int(np.count_nonzero(selected & primary_zone)),
        "selected_outside_primary_zone_pixels": int(np.count_nonzero(selected & ~primary_zone)),
    }


def _cleanup_and_refill_once(
    selected: np.ndarray,
    *,
    legal_domain: np.ndarray,
    final_score: np.ndarray,
    target_pixels: int,
    min_component_fraction: float,
    fill_holes_max_area_px: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Apply one cleanup pass and at most one score-ordered refill pass."""

    pre_cleanup_pixels = int(np.count_nonzero(selected))
    min_component_pixels = max(1, int(round(float(target_pixels) * min_component_fraction)))
    cleaned, removed_pixels = _remove_small_components(selected, min_component_pixels)
    cleaned, hole_fill_pixels = _fill_small_holes_once(
        cleaned,
        legal_domain=legal_domain,
        max_area_px=fill_holes_max_area_px,
    )
    cleaned &= legal_domain
    post_cleanup_pixels_before_refill = int(np.count_nonzero(cleaned))

    refill_pixels = 0
    if post_cleanup_pixels_before_refill < target_pixels:
        needed = target_pixels - post_cleanup_pixels_before_refill
        refill = _top_k_mask_for_refill(
            final_score,
            legal_domain=legal_domain & ~cleaned,
            k=needed,
        )
        refill_pixels = int(np.count_nonzero(refill))
        cleaned |= refill
        cleaned &= legal_domain

    return cleaned, {
        "pre_cleanup_pixels": pre_cleanup_pixels,
        "cleanup_removed_pixels": int(removed_pixels),
        "cleanup_filled_hole_pixels": int(hole_fill_pixels),
        "post_cleanup_pixels_before_refill": post_cleanup_pixels_before_refill,
        "cleanup_refill_pixels": refill_pixels,
        "post_cleanup_pixels": int(np.count_nonzero(cleaned)),
    }


def _remove_small_components(
    selected: np.ndarray,
    min_component_pixels: int,
) -> tuple[np.ndarray, int]:
    if min_component_pixels <= 1:
        return selected & True, 0
    labeled, count = ndimage.label(selected, structure=np.ones((3, 3), dtype=bool))
    cleaned = np.zeros_like(selected, dtype=bool)
    removed = 0
    for component_id in range(1, count + 1):
        component = labeled == component_id
        pixels = int(np.count_nonzero(component))
        if pixels >= min_component_pixels:
            cleaned |= component
        else:
            removed += pixels
    return cleaned, removed


def _fill_small_holes_once(
    selected: np.ndarray,
    *,
    legal_domain: np.ndarray,
    max_area_px: int,
) -> tuple[np.ndarray, int]:
    if max_area_px <= 0 or not np.any(selected):
        return selected, 0
    holes = ndimage.binary_fill_holes(selected) & ~selected & legal_domain
    labeled, count = ndimage.label(holes, structure=np.ones((3, 3), dtype=bool))
    filled = selected.copy()
    added = 0
    for component_id in range(1, count + 1):
        component = labeled == component_id
        pixels = int(np.count_nonzero(component))
        if pixels <= max_area_px:
            filled |= component
            added += pixels
    return filled, added


def _top_k_mask_for_refill(
    score: np.ndarray,
    *,
    legal_domain: np.ndarray,
    k: int,
) -> np.ndarray:
    refill_score = np.asarray(score, dtype=float).copy()
    refill_score[~legal_domain] = -np.inf
    return _top_k_mask(refill_score, k)


def _target_pixels_from_config(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    legal_pixels: int,
) -> int:
    name = primitive_config.get("name")
    ranges = primitive_config.get("parameter_ranges", {})
    if name == "stromal_immune_infiltration":
        bucket = _first_interval(ranges.get("immune_area_delta_fraction"))
        stroma = np.isin(mask, schema.resolve_fine_ids("Stroma"))
        immune = np.isin(mask, schema.resolve_fine_ids("Immune infiltrate"))
        reference_pixels = int(np.count_nonzero(stroma | immune))
    elif name == "necrosis_appearance":
        bucket = _first_interval(ranges.get("target_changed_area_fraction"))
        reference_pixels = int(np.count_nonzero(np.isin(mask, schema.tumor_fine_ids)))
    else:
        bucket = None
        reference_pixels = legal_pixels

    if bucket is None:
        return legal_pixels
    lower, _upper = bucket
    return min(int(np.ceil(reference_pixels * lower)), legal_pixels)


def _max_necrosis_fraction_of_tumor(primitive_config: Mapping[str, Any]) -> float:
    value = primitive_config.get("parameter_ranges", {}).get(
        "max_necrosis_fraction_of_tumor", 0.60
    )
    if not isinstance(value, (int, float)) or not 0 < float(value) <= 1:
        raise ValueError("invalid max_necrosis_fraction_of_tumor.")
    return float(value)


def _edge_aware_tumor_interior_distance(
    source_tumor: np.ndarray,
    *,
    pad_width: int,
) -> np.ndarray:
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
    return np.maximum(base_dist, dist[inner_rows, inner_cols])


def _score_stats(values: np.ndarray, domain: np.ndarray) -> dict[str, float | int]:
    data = np.asarray(values, dtype=float)[domain]
    if data.size == 0:
        return {"pixels": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "pixels": int(data.size),
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": float(data.mean()),
    }


def _first_interval(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        for key in ("mild", "moderate", "significant", "xlarge_deid"):
            interval = value.get(key)
            if _is_interval(interval):
                return float(interval[0]), float(interval[1])
        for interval in value.values():
            if _is_interval(interval):
                return float(interval[0]), float(interval[1])
    if _is_interval(value):
        return float(value[0]), float(value[1])
    return None


def _is_interval(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
    )


def _empty_result(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    target_label: str,
    source_labels: Sequence[str],
    preserve_labels: Sequence[str],
    forbidden_labels: Sequence[str],
    primitive_name: str,
    raw_candidate_pixels: int,
    legal_pixels: int,
    target_pixels: int,
    seed: int,
    component_policy: OrganicProjectionPolicy | None = None,
    raw_legal_overlap: int = 0,
    selected_template_intersection: int = 0,
    selected_template_union: int | None = None,
    selected_template_iou: float = 0.0,
    extra_warnings: Sequence[str] = (),
) -> PrimitiveEditResult:
    target_ids = schema.resolve_fine_ids(target_label)
    empty = np.zeros(mask.shape, dtype=bool)
    ops_log = {
        "backend": ORGANIC_PROJECTION_BACKEND,
        "method": "organic_score_projection_and_deterministic_write",
        "primitive": primitive_name,
        "reference_profile": schema.reference_profile,
        "source_labels": list(source_labels),
        "target_label": target_label,
        "target_fine_id": int(target_ids[0]),
        "preserve_labels": list(preserve_labels),
        "forbidden_labels": list(forbidden_labels),
        "raw_candidate_pixels": int(raw_candidate_pixels),
        "candidate_pixels": int(raw_candidate_pixels),
        "raw_candidate_legal_overlap_pixels": int(raw_legal_overlap),
        "template_overlap_with_legal_domain": (
            raw_legal_overlap / raw_candidate_pixels if raw_candidate_pixels else 0.0
        ),
        "legal_domain_pixels": int(legal_pixels),
        "target_pixels": int(target_pixels),
        "projected_pixels": 0,
        "selected_pixels": 0,
        "selected_raw_template_intersection_pixels": int(selected_template_intersection),
        "selected_raw_template_union_pixels": int(
            raw_candidate_pixels if selected_template_union is None else selected_template_union
        ),
        "selected_raw_template_iou": float(selected_template_iou),
        "area_shortfall": int(target_pixels),
        "projection_retained_fraction": 0.0,
        "changed_area_fraction": 0.0,
        "projection_backend": ORGANIC_PROJECTION_BACKEND,
        "noise_seed": int(seed),
    }
    if component_policy is not None:
        ops_log["component_policy"] = {
            "policy_name": component_policy.policy_name,
            "params": component_policy.policy_params,
        }
    return PrimitiveEditResult(
        target_mask=np.array(mask, copy=True),
        change_region=empty,
        changed_area_fraction=0.0,
        selected_pixels=0,
        warnings=tuple(dict.fromkeys(("proposal_projected_region_empty", *extra_warnings))),
        ops_log=ops_log,
    )

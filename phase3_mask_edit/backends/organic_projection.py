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
    strength: str = "mild",
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
            strength=strength,
        )
    target_pixels = max(int(target_pixels or 0), 0)
    if primitive_name == "necrosis_appearance":
        remaining_allowed = int(
            policy.policy_params.get("remaining_allowed_necrosis_pixels", target_pixels)
        )
        target_pixels = min(target_pixels, max(remaining_allowed, 0))
    if primitive_name == "intratumoral_immune_infiltration":
        remaining_allowed = int(
            policy.policy_params.get(
                "remaining_allowed_intratumoral_immune_pixels", target_pixels
            )
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
        min_component_pixels=_min_component_pixels_for_cleanup(
            primitive_config or {},
            target_pixels=selected_target_pixels,
            min_component_fraction=params.min_component_fraction,
        ),
    )
    spot_policy_log: dict[str, Any] = {"enabled": False}
    if primitive_name == "intratumoral_immune_infiltration":
        selected, spot_policy_log = _apply_intratumoral_immune_spot_policy(
            selected,
            legal_domain=legal_domain,
            final_score=final_score,
            target_pixels=selected_target_pixels,
            primitive_config=primitive_config or {},
        )
    target_ids = schema.resolve_fine_ids(target_label)
    target_mask = np.array(mask, copy=True)
    if primitive_name == "tumor_burden_increase":
        tumor = np.isin(mask, schema.tumor_fine_ids)
        target_mask[selected] = _nearest_source_fine_ids(mask, tumor, selected)
    else:
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
        "strength": str(strength),
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
        "cleanup_min_component_pixels": cleanup_log["cleanup_min_component_pixels"],
        "cleanup_min_component_policy": cleanup_log["cleanup_min_component_policy"],
        "pre_cleanup_pixels": cleanup_log["pre_cleanup_pixels"],
        "post_cleanup_pixels": cleanup_log["post_cleanup_pixels"],
        "cleanup_single_pass": True,
        "cleanup_iteration_limit": 1,
        "spot_policy": spot_policy_log,
    }
    if primitive_name == "tumor_burden_increase":
        ops_log["source_label_contributions"] = _source_label_contributions(
            mask,
            selected,
            schema=schema,
            source_labels=source_labels,
        )

    return PrimitiveEditResult(
        target_mask=target_mask,
        change_region=selected,
        changed_area_fraction=changed_area_fraction,
        selected_pixels=selected_pixels,
        warnings=tuple(warnings),
        ops_log=ops_log,
    )


def apply_organic_tumor_burden_decrease(
    old_mask: np.ndarray,
    raw_candidate: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any] | None = None,
    preserve_labels: Sequence[str] = (),
    forbidden_labels: Sequence[str] = (),
    seed: int = 0,
    strength: str = "mild",
    target_pixels: int | None = None,
    template_sigma: float | None = None,
    noise_sigma: float | None = None,
    noise_amplitude: float | None = None,
    w_template: float | None = None,
    w_spatial: float | None = None,
    w_noise: float | None = None,
) -> PrimitiveEditResult:
    """Select Tumor pixels for regression, then backfill from nearby legal tissue."""

    mask = np.asarray(old_mask)
    if mask.ndim != 2:
        raise ValueError("old_mask must be a 2D id mask.")

    candidate = np.asarray(raw_candidate, dtype=bool)
    if candidate.shape != mask.shape:
        raise ValueError(
            "raw_candidate shape must match old_mask shape: "
            f"{candidate.shape} != {mask.shape}."
        )

    primitive = primitive_config or {}
    params = _projection_params(
        primitive,
        template_sigma=template_sigma,
        noise_sigma=noise_sigma,
        noise_amplitude=noise_amplitude,
        w_template=w_template,
        w_spatial=w_spatial,
        w_noise=w_noise,
        decay_px=None,
    )
    forbidden = tuple(dict.fromkeys(tuple(preserve_labels) + tuple(forbidden_labels)))
    tumor = np.isin(mask, schema.tumor_fine_ids)
    removal_mask = _label_mask(mask, schema, forbidden) if forbidden else np.zeros(mask.shape, dtype=bool)
    protected = _tumor_decrease_protected_boundary(mask, schema=schema, tumor=tumor)
    legal_domain = tumor & ~removal_mask & ~protected & ~np.isin(mask, tuple(schema.skip_fine_ids))
    backfill_labels, backfill_mask = _tumor_decrease_backfill_domain(
        mask,
        schema=schema,
        primitive_config=primitive,
        preserve_labels=preserve_labels,
        forbidden_labels=forbidden_labels,
    )

    raw_candidate_pixels = int(np.count_nonzero(candidate))
    legal_pixels = int(np.count_nonzero(legal_domain))
    raw_legal_overlap = int(np.count_nonzero(candidate & legal_domain))
    if target_pixels is None:
        target_pixels = _tumor_decrease_target_pixels(
            mask,
            primitive_config=primitive,
            legal_pixels=legal_pixels,
            strength=strength,
        )
    target_pixels = max(int(target_pixels or 0), 0)
    max_removable = _tumor_decrease_max_removable_pixels(
        mask,
        primitive_config=primitive,
        tumor_pixels=int(np.count_nonzero(tumor)),
        strength=strength,
    )
    target_pixels = min(target_pixels, max(max_removable, 0))
    selected_target_pixels = min(target_pixels, legal_pixels)

    policy = _tumor_decrease_policy(
        mask,
        schema=schema,
        primitive_config=primitive,
        legal_domain=legal_domain,
        backfill_mask=backfill_mask,
        protected_boundary=protected,
    )

    empty_reason: str | None = None
    if not np.any(tumor):
        empty_reason = "tumor_decrease_no_tumor"
    elif legal_pixels == 0:
        empty_reason = "tumor_decrease_no_shrinkable_tumor_legal_domain"
    elif not np.any(backfill_mask):
        empty_reason = "tumor_decrease_no_valid_backfill_tissue"
    elif selected_target_pixels == 0:
        empty_reason = "tumor_decrease_target_pixels_zero"
    elif raw_legal_overlap == 0:
        empty_reason = "organic_projection_template_no_legal_overlap"
    if empty_reason is not None:
        return _empty_tumor_decrease_result(
            mask,
            schema=schema,
            preserve_labels=preserve_labels,
            forbidden_labels=forbidden_labels,
            raw_candidate_pixels=raw_candidate_pixels,
            legal_pixels=legal_pixels,
            target_pixels=target_pixels,
            seed=seed,
            raw_legal_overlap=raw_legal_overlap,
            backfill_labels=backfill_labels,
            policy=policy,
            warning=empty_reason,
        )

    template_overlap_fraction = raw_legal_overlap / raw_candidate_pixels if raw_candidate_pixels else 0.0
    if template_overlap_fraction < params.min_template_legal_overlap_fraction:
        return _empty_tumor_decrease_result(
            mask,
            schema=schema,
            preserve_labels=preserve_labels,
            forbidden_labels=forbidden_labels,
            raw_candidate_pixels=raw_candidate_pixels,
            legal_pixels=legal_pixels,
            target_pixels=target_pixels,
            seed=seed,
            raw_legal_overlap=raw_legal_overlap,
            backfill_labels=backfill_labels,
            policy=policy,
            warning="organic_projection_template_legal_overlap_too_low",
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

    target_mask = np.array(mask, copy=True)
    target_mask[selected] = _nearest_backfill_fine_ids(mask, backfill_mask, selected)
    selected, target_mask, tumor_cleanup_log = _cleanup_tiny_remaining_tumor_components(
        mask,
        target_mask,
        selected,
        schema=schema,
        backfill_mask=backfill_mask,
        primitive_config=primitive,
    )
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
    if selected_pixels < target_pixels:
        warnings.append("organic_projection_area_shortfall")
    if cleanup_log["post_cleanup_pixels"] < cleanup_log["pre_cleanup_pixels"]:
        warnings.append("organic_projection_cleanup_removed_pixels")
    if tumor_cleanup_log["removed_pixels"] > 0:
        warnings.append("tumor_decrease_removed_tiny_remaining_tumor_components")
    warnings.extend(policy.warnings)

    ops_log = {
        "backend": ORGANIC_PROJECTION_BACKEND,
        "method": "organic_score_projection_and_deterministic_backfill",
        "primitive": "tumor_burden_decrease",
        "reference_profile": schema.reference_profile,
        "source_labels": ["Tumor"],
        "target_label": "nearest_backfill_tissue",
        "backfill_labels": list(backfill_labels),
        "preserve_labels": list(preserve_labels),
        "forbidden_labels": list(forbidden_labels),
        "raw_candidate_pixels": raw_candidate_pixels,
        "candidate_pixels": raw_candidate_pixels,
        "raw_candidate_legal_overlap_pixels": raw_legal_overlap,
        "template_overlap_with_legal_domain": template_overlap_fraction,
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
        "strength": str(strength),
        "score_terms": {
            "w_template": float(params.w_template),
            "w_spatial": float(params.w_spatial),
            "w_noise": float(params.w_noise),
            "template_sigma": float(params.template_sigma),
            "noise_sigma": float(params.noise_sigma),
            "noise_amplitude": float(params.noise_amplitude),
            "min_template_legal_overlap_fraction": float(
                params.min_template_legal_overlap_fraction
            ),
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
        "cleanup_min_component_pixels": cleanup_log["cleanup_min_component_pixels"],
        "cleanup_min_component_policy": cleanup_log["cleanup_min_component_policy"],
        "pre_cleanup_pixels": cleanup_log["pre_cleanup_pixels"],
        "post_cleanup_pixels": cleanup_log["post_cleanup_pixels"],
        "cleanup_single_pass": True,
        "cleanup_iteration_limit": 1,
        "tumor_component_cleanup": tumor_cleanup_log,
        "decrease_semantics": "select_tumor_pixels_then_backfill_from_nearest_legal_tissue",
    }

    return PrimitiveEditResult(
        target_mask=target_mask,
        change_region=selected,
        changed_area_fraction=changed_area_fraction,
        selected_pixels=selected_pixels,
        warnings=tuple(warnings),
        ops_log=ops_log,
    )


def apply_organic_necrosis_resolution(
    old_mask: np.ndarray,
    raw_candidate: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any] | None = None,
    preserve_labels: Sequence[str] = (),
    forbidden_labels: Sequence[str] = (),
    seed: int = 0,
    strength: str = "mild",
    target_pixels: int | None = None,
    template_sigma: float | None = None,
    noise_sigma: float | None = None,
    noise_amplitude: float | None = None,
    w_template: float | None = None,
    w_spatial: float | None = None,
    w_noise: float | None = None,
) -> PrimitiveEditResult:
    """Select Necrosis pixels to resolve, then backfill from nearby viable tissue."""

    mask = np.asarray(old_mask)
    if mask.ndim != 2:
        raise ValueError("old_mask must be a 2D id mask.")

    candidate = np.asarray(raw_candidate, dtype=bool)
    if candidate.shape != mask.shape:
        raise ValueError(
            "raw_candidate shape must match old_mask shape: "
            f"{candidate.shape} != {mask.shape}."
        )

    primitive = primitive_config or {}
    params = _projection_params(
        primitive,
        template_sigma=template_sigma,
        noise_sigma=noise_sigma,
        noise_amplitude=noise_amplitude,
        w_template=w_template,
        w_spatial=w_spatial,
        w_noise=w_noise,
        decay_px=None,
    )
    forbidden = tuple(dict.fromkeys(tuple(preserve_labels) + tuple(forbidden_labels)))
    necrosis = _safe_label_mask(mask, schema, "Necrosis")
    removal_mask = _label_mask(mask, schema, forbidden) if forbidden else np.zeros(mask.shape, dtype=bool)
    legal_domain = necrosis & ~removal_mask & ~np.isin(mask, tuple(schema.skip_fine_ids))
    backfill_labels, backfill_mask = _necrosis_resolution_backfill_domain(
        mask,
        schema=schema,
        primitive_config=primitive,
        preserve_labels=preserve_labels,
        forbidden_labels=forbidden_labels,
    )

    raw_candidate_pixels = int(np.count_nonzero(candidate))
    legal_pixels = int(np.count_nonzero(legal_domain))
    raw_legal_overlap = int(np.count_nonzero(candidate & legal_domain))
    if target_pixels is None:
        target_pixels = _necrosis_resolution_target_pixels(
            mask,
            schema=schema,
            primitive_config=primitive,
            legal_pixels=legal_pixels,
            strength=strength,
        )
    target_pixels = max(int(target_pixels or 0), 0)
    selected_target_pixels = min(target_pixels, legal_pixels)

    policy = _necrosis_resolution_policy(
        mask,
        schema=schema,
        primitive_config=primitive,
        legal_domain=legal_domain,
        backfill_mask=backfill_mask,
    )

    empty_reason: str | None = None
    if legal_pixels == 0:
        empty_reason = "necrosis_resolution_no_necrosis_legal_domain"
    elif not np.any(backfill_mask):
        empty_reason = "necrosis_resolution_no_valid_backfill_tissue"
    elif selected_target_pixels == 0:
        empty_reason = "necrosis_resolution_target_pixels_zero"
    elif raw_legal_overlap == 0:
        empty_reason = "organic_projection_template_no_legal_overlap"
    if empty_reason is not None:
        return _empty_necrosis_resolution_result(
            mask,
            schema=schema,
            preserve_labels=preserve_labels,
            forbidden_labels=forbidden_labels,
            raw_candidate_pixels=raw_candidate_pixels,
            legal_pixels=legal_pixels,
            target_pixels=target_pixels,
            seed=seed,
            raw_legal_overlap=raw_legal_overlap,
            backfill_labels=backfill_labels,
            policy=policy,
            warning=empty_reason,
        )

    template_overlap_fraction = raw_legal_overlap / raw_candidate_pixels if raw_candidate_pixels else 0.0
    if template_overlap_fraction < params.min_template_legal_overlap_fraction:
        return _empty_necrosis_resolution_result(
            mask,
            schema=schema,
            preserve_labels=preserve_labels,
            forbidden_labels=forbidden_labels,
            raw_candidate_pixels=raw_candidate_pixels,
            legal_pixels=legal_pixels,
            target_pixels=target_pixels,
            seed=seed,
            raw_legal_overlap=raw_legal_overlap,
            backfill_labels=backfill_labels,
            policy=policy,
            warning="organic_projection_template_legal_overlap_too_low",
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
        min_component_fraction=0.0,
        fill_holes_max_area_px=0,
        min_component_pixels=_min_necrosis_resolution_component_pixels(
            primitive,
            target_pixels=selected_target_pixels,
        ),
    )

    target_mask = np.array(mask, copy=True)
    target_mask[selected] = _nearest_backfill_fine_ids(mask, backfill_mask, selected)
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
    if selected_pixels < target_pixels:
        warnings.append("organic_projection_area_shortfall")
    if cleanup_log["post_cleanup_pixels"] < cleanup_log["pre_cleanup_pixels"]:
        warnings.append("organic_projection_cleanup_removed_pixels")
    warnings.extend(policy.warnings)

    ops_log = {
        "backend": ORGANIC_PROJECTION_BACKEND,
        "method": "organic_score_projection_and_deterministic_backfill",
        "primitive": "necrosis_resolution",
        "reference_profile": schema.reference_profile,
        "source_labels": ["Necrosis"],
        "target_label": "nearest_backfill_tissue",
        "backfill_labels": list(backfill_labels),
        "preserve_labels": list(preserve_labels),
        "forbidden_labels": list(forbidden_labels),
        "raw_candidate_pixels": raw_candidate_pixels,
        "candidate_pixels": raw_candidate_pixels,
        "raw_candidate_legal_overlap_pixels": raw_legal_overlap,
        "template_overlap_with_legal_domain": template_overlap_fraction,
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
        "changed_area_fraction": changed_area_fraction,
        "changed_necrosis_fraction": (
            selected_pixels / legal_pixels if legal_pixels else 0.0
        ),
        "projection_backend": ORGANIC_PROJECTION_BACKEND,
        "noise_seed": int(seed),
        "strength": str(strength),
        "score_terms": {
            "w_template": float(params.w_template),
            "w_spatial": float(params.w_spatial),
            "w_noise": float(params.w_noise),
            "template_sigma": float(params.template_sigma),
            "noise_sigma": float(params.noise_sigma),
            "noise_amplitude": float(params.noise_amplitude),
            "min_template_legal_overlap_fraction": float(
                params.min_template_legal_overlap_fraction
            ),
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
        "cleanup_min_component_pixels": cleanup_log["cleanup_min_component_pixels"],
        "cleanup_min_component_policy": cleanup_log["cleanup_min_component_policy"],
        "pre_cleanup_pixels": cleanup_log["pre_cleanup_pixels"],
        "post_cleanup_pixels": cleanup_log["post_cleanup_pixels"],
        "cleanup_single_pass": True,
        "cleanup_iteration_limit": 1,
        "resolution_semantics": "select_necrosis_pixels_then_backfill_from_nearest_legal_tissue",
    }
    return PrimitiveEditResult(
        target_mask=target_mask,
        change_region=selected,
        changed_area_fraction=changed_area_fraction,
        selected_pixels=selected_pixels,
        warnings=tuple(warnings),
        ops_log=ops_log,
    )


def apply_organic_immune_infiltration_decrease(
    old_mask: np.ndarray,
    raw_candidate: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any] | None = None,
    preserve_labels: Sequence[str] = (),
    forbidden_labels: Sequence[str] = (),
    seed: int = 0,
    strength: str = "mild",
    target_pixels: int | None = None,
    template_sigma: float | None = None,
    noise_sigma: float | None = None,
    noise_amplitude: float | None = None,
    w_template: float | None = None,
    w_spatial: float | None = None,
    w_noise: float | None = None,
) -> PrimitiveEditResult:
    """Select Immune infiltrate pixels to remove, then backfill from nearby tissue."""

    mask = np.asarray(old_mask)
    if mask.ndim != 2:
        raise ValueError("old_mask must be a 2D id mask.")

    candidate = np.asarray(raw_candidate, dtype=bool)
    if candidate.shape != mask.shape:
        raise ValueError(
            "raw_candidate shape must match old_mask shape: "
            f"{candidate.shape} != {mask.shape}."
        )

    primitive = primitive_config or {}
    params = _projection_params(
        primitive,
        template_sigma=template_sigma,
        noise_sigma=noise_sigma,
        noise_amplitude=noise_amplitude,
        w_template=w_template,
        w_spatial=w_spatial,
        w_noise=w_noise,
        decay_px=None,
    )
    forbidden = tuple(dict.fromkeys(tuple(preserve_labels) + tuple(forbidden_labels)))
    immune = _safe_label_mask(mask, schema, "Immune infiltrate")
    removal_mask = _label_mask(mask, schema, forbidden) if forbidden else np.zeros(mask.shape, dtype=bool)
    legal_domain = immune & ~removal_mask & ~np.isin(mask, tuple(schema.skip_fine_ids))
    backfill_labels, backfill_mask = _immune_decrease_backfill_domain(
        mask,
        schema=schema,
        primitive_config=primitive,
        preserve_labels=preserve_labels,
        forbidden_labels=forbidden_labels,
    )

    raw_candidate_pixels = int(np.count_nonzero(candidate))
    legal_pixels = int(np.count_nonzero(legal_domain))
    raw_legal_overlap = int(np.count_nonzero(candidate & legal_domain))
    if target_pixels is None:
        target_pixels = _immune_decrease_target_pixels(
            mask,
            schema=schema,
            primitive_config=primitive,
            legal_pixels=legal_pixels,
            strength=strength,
        )
    target_pixels = max(int(target_pixels or 0), 0)
    max_removable = _immune_decrease_max_removable_pixels(
        mask,
        schema=schema,
        primitive_config=primitive,
        immune_pixels=legal_pixels,
    )
    target_pixels = min(target_pixels, max(max_removable, 0))
    selected_target_pixels = min(target_pixels, legal_pixels)

    policy = _immune_decrease_policy(
        mask,
        schema=schema,
        primitive_config=primitive,
        legal_domain=legal_domain,
        backfill_mask=backfill_mask,
    )

    empty_reason: str | None = None
    if legal_pixels == 0:
        empty_reason = "immune_decrease_no_immune_legal_domain"
    elif not np.any(backfill_mask):
        empty_reason = "immune_decrease_no_valid_backfill_tissue"
    elif selected_target_pixels == 0:
        empty_reason = "immune_decrease_target_pixels_zero"
    elif raw_legal_overlap == 0:
        empty_reason = "organic_projection_template_no_legal_overlap"
    if empty_reason is not None:
        return _empty_immune_decrease_result(
            mask,
            schema=schema,
            primitive_config=primitive,
            preserve_labels=preserve_labels,
            forbidden_labels=forbidden_labels,
            raw_candidate_pixels=raw_candidate_pixels,
            legal_pixels=legal_pixels,
            target_pixels=target_pixels,
            seed=seed,
            raw_legal_overlap=raw_legal_overlap,
            backfill_labels=backfill_labels,
            policy=policy,
            warning=empty_reason,
        )

    template_overlap_fraction = raw_legal_overlap / raw_candidate_pixels if raw_candidate_pixels else 0.0
    if template_overlap_fraction < params.min_template_legal_overlap_fraction:
        return _empty_immune_decrease_result(
            mask,
            schema=schema,
            primitive_config=primitive,
            preserve_labels=preserve_labels,
            forbidden_labels=forbidden_labels,
            raw_candidate_pixels=raw_candidate_pixels,
            legal_pixels=legal_pixels,
            target_pixels=target_pixels,
            seed=seed,
            raw_legal_overlap=raw_legal_overlap,
            backfill_labels=backfill_labels,
            policy=policy,
            warning="organic_projection_template_legal_overlap_too_low",
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
        min_component_fraction=0.0,
        fill_holes_max_area_px=0,
        min_component_pixels=(1, "immune_decrease_preserve_small_islands"),
    )

    target_mask = np.array(mask, copy=True)
    target_mask[selected] = _nearest_backfill_fine_ids(mask, backfill_mask, selected)
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
    if selected_pixels < target_pixels:
        warnings.append("organic_projection_area_shortfall")
    warnings.extend(policy.warnings)

    ops_log = {
        "backend": ORGANIC_PROJECTION_BACKEND,
        "method": "organic_score_projection_and_deterministic_backfill",
        "primitive": "immune_infiltration_decrease",
        "reference_profile": schema.reference_profile,
        "source_labels": ["Immune infiltrate"],
        "target_label": "nearest_backfill_tissue",
        "backfill_labels": list(backfill_labels),
        "preserve_labels": list(preserve_labels),
        "forbidden_labels": list(forbidden_labels),
        "raw_candidate_pixels": raw_candidate_pixels,
        "candidate_pixels": raw_candidate_pixels,
        "raw_candidate_legal_overlap_pixels": raw_legal_overlap,
        "template_overlap_with_legal_domain": template_overlap_fraction,
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
        "changed_area_fraction": changed_area_fraction,
        "projection_backend": ORGANIC_PROJECTION_BACKEND,
        "noise_seed": int(seed),
        "strength": str(strength),
        "score_terms": {
            "w_template": float(params.w_template),
            "w_spatial": float(params.w_spatial),
            "w_noise": float(params.w_noise),
            "template_sigma": float(params.template_sigma),
            "noise_sigma": float(params.noise_sigma),
            "noise_amplitude": float(params.noise_amplitude),
            "min_template_legal_overlap_fraction": float(
                params.min_template_legal_overlap_fraction
            ),
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
        "cleanup_min_component_pixels": cleanup_log["cleanup_min_component_pixels"],
        "cleanup_min_component_policy": cleanup_log["cleanup_min_component_policy"],
        "pre_cleanup_pixels": cleanup_log["pre_cleanup_pixels"],
        "post_cleanup_pixels": cleanup_log["post_cleanup_pixels"],
        "cleanup_single_pass": True,
        "cleanup_iteration_limit": 1,
        "decrease_semantics": "select_immune_pixels_then_backfill_from_nearest_legal_tissue",
    }
    return PrimitiveEditResult(
        target_mask=target_mask,
        change_region=selected,
        changed_area_fraction=changed_area_fraction,
        selected_pixels=selected_pixels,
        warnings=tuple(warnings),
        ops_log=ops_log,
    )


def apply_organic_stroma_decrease(
    old_mask: np.ndarray,
    raw_candidate: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any] | None = None,
    preserve_labels: Sequence[str] = (),
    forbidden_labels: Sequence[str] = (),
    seed: int = 0,
    strength: str = "mild",
    target_pixels: int | None = None,
    template_sigma: float | None = None,
    noise_sigma: float | None = None,
    noise_amplitude: float | None = None,
    w_template: float | None = None,
    w_spatial: float | None = None,
    w_noise: float | None = None,
) -> PrimitiveEditResult:
    """Select Stroma pixels to loosen/reduce, then backfill from nearby tissue."""

    mask = np.asarray(old_mask)
    if mask.ndim != 2:
        raise ValueError("old_mask must be a 2D id mask.")

    candidate = np.asarray(raw_candidate, dtype=bool)
    if candidate.shape != mask.shape:
        raise ValueError(
            "raw_candidate shape must match old_mask shape: "
            f"{candidate.shape} != {mask.shape}."
        )

    primitive = primitive_config or {}
    primitive_name = _canonical_stroma_decrease_name(primitive.get("name"))
    params = _projection_params(
        primitive,
        template_sigma=template_sigma,
        noise_sigma=noise_sigma,
        noise_amplitude=noise_amplitude,
        w_template=w_template,
        w_spatial=w_spatial,
        w_noise=w_noise,
        decay_px=None,
    )
    forbidden = tuple(dict.fromkeys(tuple(preserve_labels) + tuple(forbidden_labels)))
    stroma = _safe_label_mask(mask, schema, "Stroma")
    removal_mask = _label_mask(mask, schema, forbidden) if forbidden else np.zeros(mask.shape, dtype=bool)
    legal_domain = stroma & ~removal_mask & ~np.isin(mask, tuple(schema.skip_fine_ids))
    backfill_labels, backfill_mask = _stroma_decrease_backfill_domain(
        mask,
        schema=schema,
        primitive_config=primitive,
        preserve_labels=preserve_labels,
        forbidden_labels=forbidden_labels,
    )

    raw_candidate_pixels = int(np.count_nonzero(candidate))
    legal_pixels = int(np.count_nonzero(legal_domain))
    raw_legal_overlap = int(np.count_nonzero(candidate & legal_domain))
    if target_pixels is None:
        target_pixels = _stroma_decrease_target_pixels(
            mask,
            schema=schema,
            primitive_config=primitive,
            legal_pixels=legal_pixels,
            strength=strength,
        )
    target_pixels = max(int(target_pixels or 0), 0)
    max_removable = _stroma_decrease_max_removable_pixels(
        mask,
        schema=schema,
        primitive_config=primitive,
        stroma_pixels=legal_pixels,
    )
    target_pixels = min(target_pixels, max(max_removable, 0))
    selected_target_pixels = min(target_pixels, legal_pixels)

    policy = _stroma_decrease_policy(
        mask,
        schema=schema,
        primitive_config=primitive,
        legal_domain=legal_domain,
        backfill_mask=backfill_mask,
    )

    empty_reason: str | None = None
    if legal_pixels == 0:
        empty_reason = "stroma_decrease_no_stroma_legal_domain"
    elif not np.any(backfill_mask):
        empty_reason = "stroma_decrease_no_valid_backfill_tissue"
    elif selected_target_pixels == 0:
        empty_reason = "stroma_decrease_target_pixels_zero"
    elif raw_legal_overlap == 0:
        empty_reason = "organic_projection_template_no_legal_overlap"
    if empty_reason is not None:
        return _empty_stroma_decrease_result(
            mask,
            schema=schema,
            primitive_name=primitive_name,
            preserve_labels=preserve_labels,
            forbidden_labels=forbidden_labels,
            raw_candidate_pixels=raw_candidate_pixels,
            legal_pixels=legal_pixels,
            target_pixels=target_pixels,
            seed=seed,
            raw_legal_overlap=raw_legal_overlap,
            backfill_labels=backfill_labels,
            policy=policy,
            warning=empty_reason,
        )

    template_overlap_fraction = raw_legal_overlap / raw_candidate_pixels if raw_candidate_pixels else 0.0
    if template_overlap_fraction < params.min_template_legal_overlap_fraction:
        return _empty_stroma_decrease_result(
            mask,
            schema=schema,
            primitive_name=primitive_name,
            preserve_labels=preserve_labels,
            forbidden_labels=forbidden_labels,
            raw_candidate_pixels=raw_candidate_pixels,
            legal_pixels=legal_pixels,
            target_pixels=target_pixels,
            seed=seed,
            raw_legal_overlap=raw_legal_overlap,
            backfill_labels=backfill_labels,
            policy=policy,
            warning="organic_projection_template_legal_overlap_too_low",
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

    target_mask = np.array(mask, copy=True)
    target_mask[selected] = _nearest_backfill_fine_ids(mask, backfill_mask, selected)
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
    if selected_pixels < target_pixels:
        warnings.append("organic_projection_area_shortfall")
    warnings.extend(policy.warnings)

    ops_log = {
        "backend": ORGANIC_PROJECTION_BACKEND,
        "method": "organic_score_projection_and_deterministic_backfill",
        "primitive": primitive_name,
        "reference_profile": schema.reference_profile,
        "source_labels": ["Stroma"],
        "target_label": "nearest_backfill_tissue",
        "backfill_labels": list(backfill_labels),
        "preserve_labels": list(preserve_labels),
        "forbidden_labels": list(forbidden_labels),
        "raw_candidate_pixels": raw_candidate_pixels,
        "candidate_pixels": raw_candidate_pixels,
        "raw_candidate_legal_overlap_pixels": raw_legal_overlap,
        "template_overlap_with_legal_domain": template_overlap_fraction,
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
        "changed_area_fraction": changed_area_fraction,
        "projection_backend": ORGANIC_PROJECTION_BACKEND,
        "noise_seed": int(seed),
        "strength": str(strength),
        "score_terms": {
            "w_template": float(params.w_template),
            "w_spatial": float(params.w_spatial),
            "w_noise": float(params.w_noise),
            "template_sigma": float(params.template_sigma),
            "noise_sigma": float(params.noise_sigma),
            "noise_amplitude": float(params.noise_amplitude),
            "min_template_legal_overlap_fraction": float(
                params.min_template_legal_overlap_fraction
            ),
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
        "cleanup_min_component_pixels": cleanup_log["cleanup_min_component_pixels"],
        "cleanup_min_component_policy": cleanup_log["cleanup_min_component_policy"],
        "pre_cleanup_pixels": cleanup_log["pre_cleanup_pixels"],
        "post_cleanup_pixels": cleanup_log["post_cleanup_pixels"],
        "cleanup_single_pass": True,
        "cleanup_iteration_limit": 1,
        "decrease_semantics": "select_stroma_pixels_then_backfill_from_nearest_legal_tissue",
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


def _nonnegative_int_value(value: Any, default: int = 0) -> int:
    raw = default if value is None else value
    if not isinstance(raw, (int, float)) or int(raw) < 0:
        raise ValueError("organic projection integer parameter must be non-negative.")
    return int(raw)


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
    if primitive_name == "intratumoral_immune_infiltration":
        return _intratumoral_immune_policy(
            mask,
            schema=schema,
            primitive_config=primitive_config,
            legal_domain=legal_domain,
        )
    if primitive_name == "stromal_desmoplasia":
        return _stromal_desmoplasia_policy(
            mask,
            schema=schema,
            primitive_config=primitive_config,
            legal_domain=legal_domain,
        )
    if primitive_name == "tumor_burden_increase":
        return _tumor_increase_policy(
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


def _immune_decrease_policy(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    legal_domain: np.ndarray,
    backfill_mask: np.ndarray,
) -> OrganicProjectionPolicy:
    ranges = primitive_config.get("parameter_ranges", {})
    tumor = np.isin(mask, schema.tumor_fine_ids)
    dist_to_tumor = ndimage.distance_transform_edt(~tumor) if np.any(tumor) else np.zeros(mask.shape)
    if np.any(legal_domain):
        immune_labeled, immune_components = ndimage.label(
            legal_domain,
            structure=np.ones((3, 3), dtype=bool),
        )
        sizes = np.bincount(immune_labeled.ravel())
        component_sizes = sizes[immune_labeled]
        max_size = max(int(component_sizes[legal_domain].max()), 1)
        isolated_component_score = np.zeros(mask.shape, dtype=float)
        isolated_component_score[legal_domain] = 1.0 - (
            component_sizes[legal_domain].astype(float) / float(max_size)
        )
    else:
        immune_components = 0
        isolated_component_score = np.zeros(mask.shape, dtype=float)

    backfill_touching = ndimage.binary_dilation(
        backfill_mask,
        structure=np.ones((3, 3), dtype=bool),
    )
    backfill_reachable_score = (legal_domain & backfill_touching).astype(float)
    decay_px = _positive_config_float(ranges, "immune_decrease_tumor_preserve_radius_px", 48.0)
    far_from_tumor_score = 1.0 - np.exp(-dist_to_tumor / max(decay_px, 1.0))
    spatial = (
        0.45 * far_from_tumor_score
        + 0.35 * isolated_component_score
        + 0.20 * backfill_reachable_score
    )
    spatial[~legal_domain] = 0.0
    return OrganicProjectionPolicy(
        spatial_score=spatial.astype(float),
        policy_name="immune_decrease_remove_isolated_or_distal_immune",
        policy_params={
            "source_label": "Immune infiltrate",
            "backfill_policy": "nearest_legal_tissue",
            "immune_components": int(immune_components),
            "tumor_preserve_radius_px": float(decay_px),
            "selected_backfill_labels": _labels_present_in_mask(
                mask,
                schema=schema,
                labels=_backfill_priority_from_config(primitive_config),
            ),
            "score_prefers": [
                "immune_far_from_tumor",
                "small_or_isolated_immune_components",
                "immune_touching_backfill_tissue",
            ],
        },
        legal_domain=legal_domain,
    )


def _stroma_decrease_policy(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    legal_domain: np.ndarray,
    backfill_mask: np.ndarray,
) -> OrganicProjectionPolicy:
    ranges = primitive_config.get("parameter_ranges", {})
    tumor = np.isin(mask, schema.tumor_fine_ids)
    stroma = _safe_label_mask(mask, schema, "Stroma")
    backfill_radius = _positive_float(
        None, ranges.get("stroma_decrease_backfill_neighbor_radius_px", 48.0)
    )
    tumor_radius = _positive_float(
        None, ranges.get("stroma_decrease_tumor_falloff_radius_px", 96.0)
    )
    score = np.zeros(mask.shape, dtype=float)
    if np.any(backfill_mask):
        dist_to_backfill = ndimage.distance_transform_edt(~backfill_mask)
        score += 0.55 * np.exp(-dist_to_backfill / backfill_radius)
        touches_backfill = ndimage.binary_dilation(
            backfill_mask,
            structure=np.ones((3, 3), dtype=bool),
        )
        score += 0.15 * (legal_domain & touches_backfill).astype(float)
    if np.any(tumor):
        dist_to_tumor = ndimage.distance_transform_edt(~tumor)
        score += 0.30 * np.exp(-dist_to_tumor / tumor_radius)
    score[~legal_domain] = 0.0
    return OrganicProjectionPolicy(
        spatial_score=score.astype(float),
        policy_name="stroma_decrease_microenvironment_loosen",
        policy_params={
            "legal_domain_policy": "current_stroma_excluding_background",
            "backfill_policy": "nearest_legal_tissue",
            "stroma_decrease_backfill_neighbor_radius_px": backfill_radius,
            "stroma_decrease_tumor_falloff_radius_px": tumor_radius,
            "source_stroma_pixels": int(np.count_nonzero(stroma)),
            "selected_backfill_labels": _labels_present_in_mask(
                mask,
                schema=schema,
                labels=_backfill_priority_from_config(primitive_config),
            ),
            "score_prefers": [
                "stroma_near_tumor_microenvironment",
                "stroma_touching_legal_backfill_tissue",
                "stroma_near_other_nonstromal_tissue",
            ],
        },
        legal_domain=legal_domain,
    )


def _tumor_increase_policy(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    legal_domain: np.ndarray,
) -> OrganicProjectionPolicy:
    ranges = primitive_config.get("parameter_ranges", {})
    tumor = np.isin(mask, schema.tumor_fine_ids)
    necrosis = _safe_label_mask(mask, schema, "Necrosis")
    legal = legal_domain & ~tumor & ~necrosis

    tumor_radius = _positive_float(
        None, ranges.get("tumor_growth_falloff_radius_px", 64.0)
    )
    stroma_radius = _positive_float(
        None, ranges.get("tumor_growth_stroma_neighbor_radius_px", 48.0)
    )
    score = np.zeros(mask.shape, dtype=float)
    if np.any(tumor):
        dist_to_tumor = ndimage.distance_transform_edt(~tumor)
        score += 0.70 * np.exp(-dist_to_tumor / tumor_radius)
        tumor_touch = ndimage.binary_dilation(
            tumor,
            structure=np.ones((3, 3), dtype=bool),
        )
        score += 0.20 * (legal & tumor_touch).astype(float)
    stroma = _safe_label_mask(mask, schema, "Stroma")
    if np.any(stroma):
        dist_to_stroma = ndimage.distance_transform_edt(~stroma)
        score += 0.10 * np.exp(-dist_to_stroma / stroma_radius)

    score[~legal] = 0.0
    return OrganicProjectionPolicy(
        spatial_score=score,
        policy_name="tumor_burden_increase_boundary_growth",
        policy_params={
            "legal_domain_policy": (
                "editable_non_tumor_tissue_excluding_necrosis_and_background"
            ),
            "write_policy": "nearest_original_tumor_fine_id",
            "tumor_growth_falloff_radius_px": tumor_radius,
            "tumor_growth_stroma_neighbor_radius_px": stroma_radius,
            "source_tumor_pixels": int(np.count_nonzero(tumor)),
            "excluded_existing_necrosis_pixels": int(np.count_nonzero(necrosis)),
            "score_prefers": [
                "non_tumor_tissue_near_original_tumor",
                "pixels_touching_original_tumor_boundary",
                "stromal_context_when_present",
            ],
        },
        legal_domain=legal,
    )


def _tumor_decrease_policy(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    legal_domain: np.ndarray,
    backfill_mask: np.ndarray,
    protected_boundary: np.ndarray,
) -> OrganicProjectionPolicy:
    ranges = primitive_config.get("parameter_ranges", {})
    tumor = np.isin(mask, schema.tumor_fine_ids)
    interior_radius = _positive_float(
        None, ranges.get("tumor_decrease_boundary_band_radius_px", 80.0)
    )
    dist_inside_tumor = ndimage.distance_transform_edt(tumor)
    boundary_score = 1.0 - np.clip(dist_inside_tumor / interior_radius, 0.0, 1.0)

    backfill_radius = _positive_float(
        None, ranges.get("tumor_decrease_backfill_neighbor_radius_px", 48.0)
    )
    score = 0.65 * boundary_score
    if np.any(backfill_mask):
        dist_to_backfill = ndimage.distance_transform_edt(~backfill_mask)
        score += 0.35 * np.exp(-dist_to_backfill / backfill_radius)
    score[~legal_domain] = 0.0
    return OrganicProjectionPolicy(
        spatial_score=score.astype(float),
        policy_name="tumor_burden_decrease_boundary_regression",
        policy_params={
            "legal_domain_policy": (
                "current_tumor_excluding_background_facing_and_necrosis_adjacent_boundary"
            ),
            "backfill_policy": "nearest_legal_tissue",
            "tumor_decrease_boundary_band_radius_px": interior_radius,
            "tumor_decrease_backfill_neighbor_radius_px": backfill_radius,
            "source_tumor_pixels": int(np.count_nonzero(tumor)),
            "protected_boundary_pixels": int(np.count_nonzero(protected_boundary)),
            "selected_backfill_labels": _labels_present_in_mask(
                mask,
                schema=schema,
                labels=_backfill_priority_from_config(primitive_config),
            ),
            "score_prefers": [
                "tumor_boundary_or_regression_front",
                "tumor_pixels_near_legal_backfill_tissue",
            ],
        },
        legal_domain=legal_domain,
    )


def _necrosis_resolution_policy(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    legal_domain: np.ndarray,
    backfill_mask: np.ndarray,
) -> OrganicProjectionPolicy:
    ranges = primitive_config.get("parameter_ranges", {})
    necrosis = _safe_label_mask(mask, schema, "Necrosis")
    backfill_radius = _positive_float(
        None, ranges.get("necrosis_resolution_backfill_neighbor_radius_px", 48.0)
    )
    score = np.zeros(mask.shape, dtype=float)
    if np.any(backfill_mask):
        dist_to_backfill = ndimage.distance_transform_edt(~backfill_mask)
        score += 0.70 * np.exp(-dist_to_backfill / backfill_radius)
        touches_backfill = ndimage.binary_dilation(
            backfill_mask,
            structure=np.ones((3, 3), dtype=bool),
        )
        score += 0.30 * (legal_domain & touches_backfill).astype(float)
    score[~legal_domain] = 0.0
    return OrganicProjectionPolicy(
        spatial_score=score.astype(float),
        policy_name="necrosis_resolution_viable_backfill_front",
        policy_params={
            "legal_domain_policy": "current_necrosis_excluding_background",
            "backfill_policy": "nearest_tumor_or_stroma",
            "necrosis_pixels": int(np.count_nonzero(necrosis)),
            "backfill_neighbor_radius_px": backfill_radius,
            "selected_backfill_labels": _labels_present_in_mask(
                mask,
                schema=schema,
                labels=_backfill_priority_from_config(primitive_config),
            ),
            "score_prefers": [
                "necrosis_touching_viable_tissue",
                "necrosis_near_legal_backfill_tissue",
            ],
        },
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


def _intratumoral_immune_policy(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    legal_domain: np.ndarray,
) -> OrganicProjectionPolicy:
    ranges = primitive_config.get("parameter_ranges", {})
    spatial_pattern = primitive_config.get("spatial_pattern", {})
    spot_policy = (
        spatial_pattern.get("spot_policy", {})
        if isinstance(spatial_pattern, Mapping)
        else {}
    )
    if not isinstance(spot_policy, Mapping):
        spot_policy = {}

    tumor = np.isin(mask, schema.tumor_fine_ids)
    immune = _safe_label_mask(mask, schema, "Immune infiltrate")
    legal = legal_domain & tumor & ~immune
    score = np.zeros(mask.shape, dtype=float)

    boundary_radius = _positive_float(
        None, ranges.get("tumor_boundary_margin_radius_px", 32.0)
    )
    interior_dist = _edge_aware_tumor_interior_distance(
        tumor,
        pad_width=int(round(boundary_radius)),
    )
    inner_boundary_score = np.exp(-interior_dist / max(boundary_radius, 1.0))
    score += inner_boundary_score

    immune_radius = _positive_float(None, ranges.get("immune_neighbor_radius_px", 36.0))
    used_existing_immune = bool(np.any(immune))
    if used_existing_immune:
        dist_to_immune = ndimage.distance_transform_edt(~immune)
        score += 0.55 * np.exp(-dist_to_immune / immune_radius)

    score[~legal] = 0.0
    tumor_pixels = int(np.count_nonzero(tumor))
    existing_immune_pixels_total = int(np.count_nonzero(immune))
    existing_intratumoral_immune_pixels = int(np.count_nonzero(immune & tumor))
    max_fraction = _max_intratumoral_immune_fraction_of_tumor(primitive_config)
    max_immune_pixels = int(round(max_fraction * tumor_pixels))
    min_spot_area_px = _spot_policy_min_area_px(primitive_config)
    return OrganicProjectionPolicy(
        spatial_score=score,
        policy_name="intratumoral_immune_til_spots",
        policy_params={
            "legal_domain_policy": "current_tumor_only_excluding_existing_immune",
            "tumor_pixels": tumor_pixels,
            "existing_immune_pixels_total": existing_immune_pixels_total,
            "existing_intratumoral_immune_pixels": existing_intratumoral_immune_pixels,
            "intratumoral_immune_cap_policy": "per_edit_new_pixels_only",
            "existing_immune_pixels_not_subtracted_from_cap": True,
            "max_intratumoral_immune_fraction_of_tumor": max_fraction,
            "max_intratumoral_immune_pixels": max_immune_pixels,
            "remaining_allowed_intratumoral_immune_pixels": max_immune_pixels,
            "tumor_boundary_margin_radius_px": boundary_radius,
            "tumor_boundary_score_policy": "prefer_inner_tumor_invasive_front_not_geometric_center",
            "immune_neighbor_radius_px": immune_radius,
            "used_existing_immune_neighborhood": used_existing_immune,
            "score_prefers": [
                "tumor_pixels_just_inside_tumor_stroma_boundary",
                "tumor_pixels_near_existing_immune_infiltrate",
                "patchy_spots_over_concentric_central_sheet",
            ],
            "spot_policy_min_spot_area_px": min_spot_area_px,
            "spot_policy_max_spot_area_px": _spot_policy_int(
                spot_policy.get("max_spot_area_px"), default=0
            ),
            "spot_policy_max_spots_per_patch": _spot_policy_int(
                spot_policy.get("max_spots_per_patch"), default=0
            ),
        },
        legal_domain=legal,
    )


def _stromal_desmoplasia_policy(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    legal_domain: np.ndarray,
) -> OrganicProjectionPolicy:
    ranges = primitive_config.get("parameter_ranges", {})
    tumor = np.isin(mask, schema.tumor_fine_ids)
    stroma = _safe_label_mask(mask, schema, "Stroma")
    immune = _safe_label_mask(mask, schema, "Immune infiltrate")

    max_distance = _positive_float(
        None, ranges.get("max_distance_from_tumor_px", 64.0)
    )
    dist_to_tumor = ndimage.distance_transform_edt(~tumor)
    peritumoral = (dist_to_tumor <= max_distance) & ~tumor

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

    legal = legal_domain & peritumoral
    if require_immune_stroma_adjacency and np.any(immune):
        stroma_neighbors = ndimage.binary_dilation(
            stroma, structure=np.ones((3, 3), dtype=bool)
        )
        legal &= (~immune) | stroma_neighbors

    tumor_falloff = _positive_float(
        None, ranges.get("desmoplasia_tumor_falloff_radius_px", 48.0)
    )
    stroma_radius = _positive_float(
        None, ranges.get("desmoplasia_stroma_neighbor_radius_px", 24.0)
    )
    score = np.zeros(mask.shape, dtype=float)
    if np.any(tumor):
        score += 0.50 * np.exp(-dist_to_tumor / tumor_falloff)
    if np.any(stroma):
        dist_to_stroma = ndimage.distance_transform_edt(~stroma)
        score += 0.50 * np.exp(-dist_to_stroma / stroma_radius)
    score[~legal] = 0.0
    return OrganicProjectionPolicy(
        spatial_score=score,
        policy_name="stromal_desmoplasia_peritumoral_stroma_expansion",
        policy_params={
            "legal_domain_policy": (
                "primary_or_secondary_sources_outside_tumor_with_peritumoral_limit"
            ),
            "max_distance_from_tumor_px": max_distance,
            "desmoplasia_tumor_falloff_radius_px": tumor_falloff,
            "desmoplasia_stroma_neighbor_radius_px": stroma_radius,
            "require_direct_stroma_adjacency_for_immune": (
                require_immune_stroma_adjacency
            ),
            "max_immune_fraction_of_delta": float(
                constraints.get("max_fraction_of_total_desmoplasia_delta", 0.30)
            ),
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
    min_component_pixels: tuple[int, str] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Apply one cleanup pass and at most one score-ordered refill pass."""

    pre_cleanup_pixels = int(np.count_nonzero(selected))
    if min_component_pixels is None:
        min_component_px = max(
            1, int(round(float(target_pixels) * min_component_fraction))
        )
        min_component_policy = "fraction_of_target_pixels"
    else:
        min_component_px, min_component_policy = min_component_pixels
    cleaned, removed_pixels = _remove_small_components(selected, min_component_px)
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
        "cleanup_min_component_pixels": int(min_component_px),
        "cleanup_min_component_policy": min_component_policy,
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


def _apply_intratumoral_immune_spot_policy(
    selected: np.ndarray,
    *,
    legal_domain: np.ndarray,
    final_score: np.ndarray,
    target_pixels: int,
    primitive_config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    spatial_pattern = primitive_config.get("spatial_pattern", {})
    spot_policy = (
        spatial_pattern.get("spot_policy", {})
        if isinstance(spatial_pattern, Mapping)
        else {}
    )
    if not isinstance(spot_policy, Mapping):
        spot_policy = {}

    max_spot_area_px = _spot_policy_int(spot_policy.get("max_spot_area_px"), default=0)
    max_spots = _spot_policy_int(spot_policy.get("max_spots_per_patch"), default=0)
    if max_spot_area_px <= 0 and max_spots <= 0:
        return selected, {"enabled": False}

    min_spot_area_px = _spot_policy_min_area_px(primitive_config) or 1
    if max_spot_area_px <= 0:
        max_spot_area_px = max(int(target_pixels), min_spot_area_px)
    if max_spots <= 0:
        max_spots = max(1, int(np.ceil(max(int(target_pixels), 1) / max_spot_area_px)))

    target_pixels = min(int(target_pixels), int(np.count_nonzero(legal_domain)))
    if target_pixels <= 0:
        return np.zeros_like(selected, dtype=bool), {
            "enabled": True,
            "max_spot_area_px": int(max_spot_area_px),
            "max_spots_per_patch": int(max_spots),
            "selected_spots": 0,
            "selected_pixels_before": int(np.count_nonzero(selected)),
            "selected_pixels_after": 0,
        }

    seed_score = np.asarray(final_score, dtype=float).copy()
    seed_domain = np.asarray(selected, dtype=bool) & legal_domain
    if not np.any(seed_domain):
        seed_domain = np.asarray(legal_domain, dtype=bool)
    seed_score[~seed_domain] = -np.inf

    result = np.zeros_like(selected, dtype=bool)
    suppressed = np.zeros_like(selected, dtype=bool)
    spot_sizes: list[int] = []
    grow_structure = np.ones((3, 3), dtype=bool)
    spot_gap_radius = max(2, int(round(np.sqrt(float(max_spot_area_px)) / 2.0)))
    remaining = target_pixels

    while remaining > 0 and len(spot_sizes) < max_spots:
        buffered_result = (
            ndimage.binary_dilation(
                result,
                iterations=spot_gap_radius,
                structure=grow_structure,
            )
            if np.any(result)
            else result
        )
        available_seed = seed_domain & ~suppressed & ~buffered_result & ~result
        if not np.any(available_seed):
            break
        seed_scores = seed_score.copy()
        seed_scores[~available_seed] = -np.inf
        seed_mask = _top_k_mask(seed_scores, 1)
        if not np.any(seed_mask):
            break

        spot_budget = min(max_spot_area_px, remaining)
        spot = _grow_score_ordered_spot(
            seed_mask,
            legal_domain=legal_domain & ~result & ~buffered_result,
            final_score=final_score,
            max_pixels=spot_budget,
        )
        pixels = int(np.count_nonzero(spot))
        if pixels == 0:
            suppressed |= seed_mask
            continue
        result |= spot
        spot_sizes.append(pixels)
        remaining = target_pixels - int(np.count_nonzero(result))
        suppressed |= ndimage.binary_dilation(
            spot,
            iterations=spot_gap_radius,
            structure=grow_structure,
        )

    if int(np.count_nonzero(result)) < target_pixels:
        needed = target_pixels - int(np.count_nonzero(result))
        refill, refill_log = _fill_intratumoral_immune_spot_shortfall(
            result,
            legal_domain=legal_domain,
            final_score=final_score,
            needed=needed,
            max_spot_area_px=max_spot_area_px,
            max_extra_spots=max(max_spots - len(spot_sizes), 0),
            spot_gap_radius=spot_gap_radius,
        )
        result |= refill
    else:
        refill_log = {"refill_pixels": 0, "refill_spots": 0}

    final_labeled, final_count = ndimage.label(result, structure=grow_structure)
    final_sizes = [
        int(np.count_nonzero(final_labeled == component_id))
        for component_id in range(1, final_count + 1)
    ]
    oversized = [size for size in final_sizes if size > max_spot_area_px]
    return result, {
        "enabled": True,
        "max_spot_area_px": int(max_spot_area_px),
        "max_spots_per_patch": int(max_spots),
        "min_spot_area_px": int(min_spot_area_px),
        "selected_spots": int(len(spot_sizes)),
        "selected_pixels_before": int(np.count_nonzero(selected)),
        "selected_pixels_after": int(np.count_nonzero(result)),
        "spot_sizes": spot_sizes,
        **refill_log,
        "final_component_count": int(final_count),
        "final_component_sizes": final_sizes,
        "oversized_component_count": int(len(oversized)),
        "spot_gap_radius_px": int(spot_gap_radius),
    }


def _fill_intratumoral_immune_spot_shortfall(
    current: np.ndarray,
    *,
    legal_domain: np.ndarray,
    final_score: np.ndarray,
    needed: int,
    max_spot_area_px: int,
    max_extra_spots: int,
    spot_gap_radius: int,
) -> tuple[np.ndarray, dict[str, int]]:
    if needed <= 0 or max_extra_spots <= 0:
        return np.zeros_like(current, dtype=bool), {
            "refill_pixels": 0,
            "refill_spots": 0,
        }

    refill = np.zeros_like(current, dtype=bool)
    structure = np.ones((3, 3), dtype=bool)
    remaining = int(needed)
    spots = 0
    while remaining > 0 and spots < max_extra_spots:
        occupied = current | refill
        buffered = (
            ndimage.binary_dilation(
                occupied,
                iterations=spot_gap_radius,
                structure=structure,
            )
            if np.any(occupied)
            else occupied
        )
        seed_domain = legal_domain & ~occupied & ~buffered
        seed = _top_k_mask_for_refill(final_score, legal_domain=seed_domain, k=1)
        if not np.any(seed):
            break
        spot = _grow_score_ordered_spot(
            seed,
            legal_domain=seed_domain,
            final_score=final_score,
            max_pixels=min(max_spot_area_px, remaining),
        )
        pixels = int(np.count_nonzero(spot))
        if pixels == 0:
            break
        refill |= spot
        remaining -= pixels
        spots += 1

    return refill, {
        "refill_pixels": int(np.count_nonzero(refill)),
        "refill_spots": int(spots),
    }


def _grow_score_ordered_spot(
    seed: np.ndarray,
    *,
    legal_domain: np.ndarray,
    final_score: np.ndarray,
    max_pixels: int,
) -> np.ndarray:
    spot = np.asarray(seed, dtype=bool) & legal_domain
    if max_pixels <= 0 or not np.any(spot):
        return np.zeros_like(seed, dtype=bool)
    if int(np.count_nonzero(spot)) >= max_pixels:
        return spot

    structure = np.ones((3, 3), dtype=bool)
    while int(np.count_nonzero(spot)) < max_pixels:
        frontier = ndimage.binary_dilation(spot, structure=structure) & legal_domain & ~spot
        if not np.any(frontier):
            break
        needed = int(max_pixels) - int(np.count_nonzero(spot))
        add = _top_k_mask_for_refill(
            final_score,
            legal_domain=frontier,
            k=min(needed, int(np.count_nonzero(frontier))),
        )
        if not np.any(add):
            break
        spot |= add
    return spot & legal_domain


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


def _cleanup_tiny_remaining_tumor_components(
    original_mask: np.ndarray,
    target_mask: np.ndarray,
    selected: np.ndarray,
    *,
    schema: MaskProfileSchema,
    backfill_mask: np.ndarray,
    primitive_config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ranges = primitive_config.get("parameter_ranges", {})
    min_remaining_component_px = _nonnegative_int_value(
        ranges.get("tumor_decrease_min_remaining_component_area_px", 0)
    )
    cleanup_enabled = bool(
        primitive_config.get("spatial_pattern", {}).get("fragment_cleanup", False)
        if isinstance(primitive_config.get("spatial_pattern", {}), Mapping)
        else False
    )
    if not cleanup_enabled or min_remaining_component_px <= 1:
        return selected, target_mask, {
            "enabled": cleanup_enabled,
            "min_remaining_component_area_px": int(min_remaining_component_px),
            "removed_pixels": 0,
            "removed_components": 0,
            "remaining_components_before": 0,
            "remaining_components_after": 0,
        }

    tumor_after = np.isin(target_mask, schema.tumor_fine_ids)
    labeled, count = ndimage.label(tumor_after, structure=np.ones((3, 3), dtype=bool))
    cleanup_mask = np.zeros_like(tumor_after, dtype=bool)
    removed_components = 0
    for component_id in range(1, count + 1):
        component = labeled == component_id
        pixels = int(np.count_nonzero(component))
        if pixels < min_remaining_component_px:
            cleanup_mask |= component
            removed_components += 1

    removed_pixels = int(np.count_nonzero(cleanup_mask))
    if removed_pixels == 0:
        return selected, target_mask, {
            "enabled": True,
            "min_remaining_component_area_px": int(min_remaining_component_px),
            "removed_pixels": 0,
            "removed_components": 0,
            "remaining_components_before": int(count),
            "remaining_components_after": int(count),
        }

    updated_target = np.array(target_mask, copy=True)
    updated_selected = np.asarray(selected, dtype=bool).copy()
    updated_selected |= cleanup_mask
    updated_target[cleanup_mask] = _nearest_backfill_fine_ids(
        original_mask,
        backfill_mask,
        cleanup_mask,
    )
    remaining_after = int(
        ndimage.label(
            np.isin(updated_target, schema.tumor_fine_ids),
            structure=np.ones((3, 3), dtype=bool),
        )[1]
    )
    return updated_selected, updated_target, {
        "enabled": True,
        "min_remaining_component_area_px": int(min_remaining_component_px),
        "removed_pixels": removed_pixels,
        "removed_components": int(removed_components),
        "remaining_components_before": int(count),
        "remaining_components_after": remaining_after,
    }


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
    strength: str = "mild",
) -> int:
    name = primitive_config.get("name")
    ranges = primitive_config.get("parameter_ranges", {})
    if name == "stromal_immune_infiltration":
        bucket = _first_interval(ranges.get("immune_area_delta_fraction"))
        stroma = np.isin(mask, schema.resolve_fine_ids("Stroma"))
        immune = np.isin(mask, schema.resolve_fine_ids("Immune infiltrate"))
        reference_pixels = int(np.count_nonzero(stroma | immune))
    elif name == "tumor_burden_increase":
        bucket = _first_interval(
            ranges.get("target_area_delta_fraction"), strength=strength
        )
        reference_pixels = int(mask.size)
    elif name == "necrosis_appearance":
        bucket = _first_interval(
            ranges.get("target_changed_area_fraction"), strength=strength
        )
        reference_pixels = int(np.count_nonzero(np.isin(mask, schema.tumor_fine_ids)))
    elif name == "intratumoral_immune_infiltration":
        bucket = _first_interval(
            ranges.get("target_changed_area_fraction"), strength=strength
        )
        reference_pixels = int(np.count_nonzero(np.isin(mask, schema.tumor_fine_ids)))
    elif name == "stromal_desmoplasia":
        bucket = _first_interval(ranges.get("stroma_area_delta_fraction"))
        reference_pixels = int(
            np.count_nonzero(np.isin(mask, schema.resolve_fine_ids("Stroma")))
        )
    else:
        bucket = None
        reference_pixels = legal_pixels

    if bucket is None:
        return legal_pixels
    lower, upper = bucket
    midpoint = (lower + upper) / 2.0
    target = int(np.ceil(reference_pixels * midpoint))
    if name == "stromal_desmoplasia":
        target = max(
            target,
            _pixel_floor_from_config(
                ranges.get("min_stroma_area_delta_pixels"),
                strength=strength,
            ),
        )
    return min(target, legal_pixels)


def _stroma_decrease_target_pixels(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    legal_pixels: int,
    strength: str,
) -> int:
    ranges = primitive_config.get("parameter_ranges", {})
    bucket = _first_interval(
        ranges.get("stroma_area_decrease_fraction"), strength=strength
    )
    if bucket is None:
        return legal_pixels
    lower, upper = bucket
    midpoint = (lower + upper) / 2.0
    stroma_pixels = int(
        np.count_nonzero(_safe_label_mask(mask, schema, "Stroma"))
    )
    target = int(np.ceil(stroma_pixels * midpoint))
    return min(target, legal_pixels)


def _necrosis_resolution_target_pixels(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    legal_pixels: int,
    strength: str,
) -> int:
    ranges = primitive_config.get("parameter_ranges", {})
    bucket = _first_interval(
        ranges.get("necrosis_area_decrease_fraction"), strength=strength
    )
    if bucket is None:
        return legal_pixels
    lower, upper = bucket
    midpoint = (lower + upper) / 2.0
    necrosis_pixels = int(
        np.count_nonzero(_safe_label_mask(mask, schema, "Necrosis"))
    )
    target = int(np.ceil(necrosis_pixels * midpoint))
    return min(target, legal_pixels)


def _stroma_decrease_max_removable_pixels(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    stroma_pixels: int,
) -> int:
    ranges = primitive_config.get("parameter_ranges", {})
    value = ranges.get("min_remaining_stroma_fraction", 0.02)
    if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
        raise ValueError("invalid min_remaining_stroma_fraction.")
    total_stroma = int(
        np.count_nonzero(_safe_label_mask(mask, schema, "Stroma"))
    )
    min_remaining = int(np.ceil(total_stroma * float(value)))
    return max(int(stroma_pixels) - min_remaining, 0)


def _tumor_decrease_target_pixels(
    mask: np.ndarray,
    *,
    primitive_config: Mapping[str, Any],
    legal_pixels: int,
    strength: str,
) -> int:
    ranges = primitive_config.get("parameter_ranges", {})
    bucket = _first_interval(ranges.get("target_area_decrease_fraction"), strength=strength)
    if bucket is None:
        return legal_pixels
    lower, upper = bucket
    midpoint = (lower + upper) / 2.0
    target = int(np.ceil(mask.size * midpoint))
    return min(target, legal_pixels)


def _tumor_decrease_max_removable_pixels(
    mask: np.ndarray,
    *,
    primitive_config: Mapping[str, Any],
    tumor_pixels: int,
    strength: str,
) -> int:
    ranges = primitive_config.get("parameter_ranges", {})
    min_remaining = ranges.get("min_remaining_tumor_fraction", {})
    if isinstance(min_remaining, Mapping):
        key = "xlarge_deid" if strength == "xlarge_deid" else "default"
        value = min_remaining.get(key, min_remaining.get("default", 0.02))
    else:
        value = min_remaining
    if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
        raise ValueError("invalid min_remaining_tumor_fraction.")
    min_remaining_pixels = int(np.ceil(mask.size * float(value)))
    return max(int(tumor_pixels) - min_remaining_pixels, 0)


def _immune_decrease_target_pixels(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    legal_pixels: int,
    strength: str,
) -> int:
    ranges = primitive_config.get("parameter_ranges", {})
    bucket = _first_interval(ranges.get("immune_area_decrease_fraction"), strength=strength)
    if bucket is None:
        return legal_pixels
    lower, upper = bucket
    midpoint = (lower + upper) / 2.0
    immune_pixels = int(
        np.count_nonzero(_safe_label_mask(mask, schema, "Immune infiltrate"))
    )
    target = int(np.ceil(immune_pixels * midpoint))
    return min(target, legal_pixels)


def _immune_decrease_max_removable_pixels(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    immune_pixels: int,
) -> int:
    ranges = primitive_config.get("parameter_ranges", {})
    value = ranges.get("min_remaining_immune_fraction", 0.005)
    if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
        raise ValueError("invalid min_remaining_immune_fraction.")
    total_immune = int(
        np.count_nonzero(_safe_label_mask(mask, schema, "Immune infiltrate"))
    )
    min_remaining = int(np.ceil(total_immune * float(value)))
    return max(int(immune_pixels) - min_remaining, 0)


def _pixel_floor_from_config(value: Any, *, strength: str) -> int:
    if isinstance(value, Mapping):
        raw = value.get(strength)
    else:
        raw = value
    if isinstance(raw, (int, float)) and int(raw) > 0:
        return int(raw)
    return 0


def _max_necrosis_fraction_of_tumor(primitive_config: Mapping[str, Any]) -> float:
    value = primitive_config.get("parameter_ranges", {}).get(
        "max_necrosis_fraction_of_tumor", 0.60
    )
    if not isinstance(value, (int, float)) or not 0 < float(value) <= 1:
        raise ValueError("invalid max_necrosis_fraction_of_tumor.")
    return float(value)


def _max_intratumoral_immune_fraction_of_tumor(
    primitive_config: Mapping[str, Any]
) -> float:
    spatial_pattern = primitive_config.get("spatial_pattern", {})
    spot_policy = (
        spatial_pattern.get("spot_policy", {})
        if isinstance(spatial_pattern, Mapping)
        else {}
    )
    if not isinstance(spot_policy, Mapping):
        spot_policy = {}
    ranges = primitive_config.get("parameter_ranges", {})
    value = spot_policy.get(
        "max_total_area_fraction_of_tumor",
        ranges.get("max_changed_area_fraction", 0.30),
    )
    if not isinstance(value, (int, float)) or not 0 < float(value) <= 1:
        raise ValueError("invalid max intratumoral immune fraction of tumor.")
    return float(value)


def _min_component_pixels_for_cleanup(
    primitive_config: Mapping[str, Any],
    *,
    target_pixels: int,
    min_component_fraction: float,
) -> tuple[int, str]:
    spot_min = _spot_policy_min_area_px(primitive_config)
    if spot_min is not None:
        return max(1, spot_min), "spot_policy.min_spot_area_px"
    return (
        max(1, int(round(float(target_pixels) * min_component_fraction))),
        "organic_min_component_fraction",
    )


def _min_necrosis_resolution_component_pixels(
    primitive_config: Mapping[str, Any],
    *,
    target_pixels: int,
) -> tuple[int, str]:
    ranges = primitive_config.get("parameter_ranges", {})
    value = ranges.get("min_necrosis_resolution_component_area_px", 1)
    if isinstance(value, (int, float)) and int(value) > 0:
        return (
            min(int(value), max(1, int(target_pixels))),
            "min_necrosis_resolution_component_area_px",
        )
    return 1, "necrosis_resolution_preserve_small_components"


def _spot_policy_min_area_px(primitive_config: Mapping[str, Any]) -> int | None:
    spatial_pattern = primitive_config.get("spatial_pattern", {})
    if not isinstance(spatial_pattern, Mapping):
        return None
    spot_policy = spatial_pattern.get("spot_policy", {})
    if not isinstance(spot_policy, Mapping):
        return None
    value = spot_policy.get("min_spot_area_px")
    if value is None:
        return None
    return _spot_policy_int(value, default=1)


def _spot_policy_int(value: Any, *, default: int) -> int:
    if value is None:
        return int(default)
    if not isinstance(value, (int, float)) or int(value) < 0:
        raise ValueError("spot_policy pixel values must be non-negative.")
    return int(value)


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


def _immune_decrease_backfill_domain(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    preserve_labels: Sequence[str],
    forbidden_labels: Sequence[str],
) -> tuple[tuple[str, ...], np.ndarray]:
    forbidden = set(preserve_labels) | set(forbidden_labels) | {"Immune infiltrate"}
    labels: list[str] = []
    domain = np.zeros(mask.shape, dtype=bool)
    for label in _backfill_priority_from_config(primitive_config):
        if label in forbidden or label not in schema.readable_labels:
            continue
        label_mask = _safe_label_mask(mask, schema, label)
        if not np.any(label_mask):
            continue
        domain |= label_mask
        labels.append(label)
    domain &= ~np.isin(mask, tuple(schema.skip_fine_ids))
    return tuple(labels), domain


def _necrosis_resolution_backfill_domain(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    preserve_labels: Sequence[str],
    forbidden_labels: Sequence[str],
) -> tuple[tuple[str, ...], np.ndarray]:
    forbidden = set(preserve_labels) | set(forbidden_labels) | {"Necrosis"}
    labels: list[str] = []
    domain = np.zeros(mask.shape, dtype=bool)
    for label in _backfill_priority_from_config(primitive_config):
        if label in forbidden or label not in schema.readable_labels:
            continue
        label_mask = _safe_label_mask(mask, schema, label)
        if not np.any(label_mask):
            continue
        domain |= label_mask
        labels.append(label)
    domain &= ~np.isin(mask, tuple(schema.skip_fine_ids))
    return tuple(labels), domain


def _stroma_decrease_backfill_domain(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    preserve_labels: Sequence[str],
    forbidden_labels: Sequence[str],
) -> tuple[tuple[str, ...], np.ndarray]:
    forbidden = set(preserve_labels) | set(forbidden_labels) | {"Stroma"}
    labels: list[str] = []
    domain = np.zeros(mask.shape, dtype=bool)
    for label in _backfill_priority_from_config(primitive_config):
        if label in forbidden or label not in schema.readable_labels:
            continue
        label_mask = _safe_label_mask(mask, schema, label)
        if not np.any(label_mask):
            continue
        domain |= label_mask
        labels.append(label)
    domain &= ~np.isin(mask, tuple(schema.skip_fine_ids))
    return tuple(labels), domain


def _tumor_decrease_backfill_domain(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    preserve_labels: Sequence[str],
    forbidden_labels: Sequence[str],
) -> tuple[tuple[str, ...], np.ndarray]:
    forbidden = set(preserve_labels) | set(forbidden_labels) | {"Tumor", "Necrosis"}
    labels: list[str] = []
    domain = np.zeros(mask.shape, dtype=bool)
    for label in _backfill_priority_from_config(primitive_config):
        if label in forbidden or label not in schema.readable_labels:
            continue
        label_mask = _safe_label_mask(mask, schema, label)
        if not np.any(label_mask):
            continue
        domain |= label_mask
        labels.append(label)
    domain &= ~np.isin(mask, tuple(schema.skip_fine_ids))
    return tuple(labels), domain


def _canonical_stroma_decrease_name(value: Any) -> str:
    if value == "stromal_reduction":
        return "stromal_reduction"
    return "stroma_decrease"


def _tumor_decrease_protected_boundary(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    tumor: np.ndarray,
) -> np.ndarray:
    protected_context = _external_background_mask(mask, schema.skip_fine_ids)
    protected_context |= _safe_label_mask(mask, schema, "Necrosis")
    return tumor & ndimage.binary_dilation(
        protected_context,
        structure=np.ones((3, 3), dtype=bool),
    )


def _external_background_mask(
    mask: np.ndarray,
    skip_fine_ids: frozenset[int],
) -> np.ndarray:
    background = np.isin(mask, tuple(skip_fine_ids))
    if not np.any(background):
        return np.zeros(mask.shape, dtype=bool)
    labeled, component_count = ndimage.label(
        background,
        structure=np.ones((3, 3), dtype=bool),
    )
    if component_count == 0:
        return np.zeros(mask.shape, dtype=bool)
    border_labels = set(int(label) for label in labeled[0, :] if label)
    border_labels.update(int(label) for label in labeled[-1, :] if label)
    border_labels.update(int(label) for label in labeled[:, 0] if label)
    border_labels.update(int(label) for label in labeled[:, -1] if label)
    if not border_labels:
        return np.zeros(mask.shape, dtype=bool)
    return np.isin(labeled, tuple(border_labels))


def _backfill_priority_from_config(primitive_config: Mapping[str, Any]) -> tuple[str, ...]:
    operation = primitive_config.get("mask_operation", {})
    priority = operation.get("backfill_priority", ()) if isinstance(operation, Mapping) else ()
    if isinstance(priority, list):
        labels = tuple(label for label in priority if isinstance(label, str))
        if labels:
            return labels
    return ("Stroma", "Other tissue", "Normal epithelium", "Tumor")


def _labels_present_in_mask(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    labels: Sequence[str],
) -> list[str]:
    present: list[str] = []
    for label in labels:
        if label in schema.readable_labels and np.any(_safe_label_mask(mask, schema, label)):
            present.append(label)
    return present


def _source_label_contributions(
    mask: np.ndarray,
    selected: np.ndarray,
    *,
    schema: MaskProfileSchema,
    source_labels: Sequence[str],
) -> dict[str, int]:
    contributions: dict[str, int] = {}
    for label in source_labels:
        if label not in schema.readable_labels:
            continue
        contributions[str(label)] = int(
            np.count_nonzero(selected & _safe_label_mask(mask, schema, label))
        )
    return contributions


def _positive_config_float(
    ranges: Mapping[str, Any],
    key: str,
    default: float,
) -> float:
    value = ranges.get(key, default)
    if not isinstance(value, (int, float)) or float(value) <= 0:
        raise ValueError(f"invalid {key}.")
    return float(value)


def _nearest_backfill_fine_ids(
    mask: np.ndarray,
    backfill_mask: np.ndarray,
    change_region: np.ndarray,
) -> np.ndarray:
    return _nearest_source_fine_ids(mask, backfill_mask, change_region)


def _nearest_source_fine_ids(
    mask: np.ndarray,
    source_mask: np.ndarray,
    change_region: np.ndarray,
) -> np.ndarray:
    _, nearest_indices = ndimage.distance_transform_edt(
        ~source_mask,
        return_indices=True,
    )
    row_indices, col_indices = nearest_indices
    nearest_ids = mask[row_indices, col_indices]
    return nearest_ids[change_region]


def _first_interval(value: Any, *, strength: str = "mild") -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        for key in (strength, "mild", "moderate", "significant", "xlarge_deid"):
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


def _empty_immune_decrease_result(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    preserve_labels: Sequence[str],
    forbidden_labels: Sequence[str],
    raw_candidate_pixels: int,
    legal_pixels: int,
    target_pixels: int,
    seed: int,
    raw_legal_overlap: int,
    backfill_labels: Sequence[str],
    policy: OrganicProjectionPolicy,
    warning: str,
) -> PrimitiveEditResult:
    del primitive_config
    ops_log = {
        "backend": ORGANIC_PROJECTION_BACKEND,
        "method": "organic_score_projection_and_deterministic_backfill",
        "primitive": "immune_infiltration_decrease",
        "reference_profile": schema.reference_profile,
        "source_labels": ["Immune infiltrate"],
        "target_label": "nearest_backfill_tissue",
        "backfill_labels": list(backfill_labels),
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
        "area_shortfall": int(target_pixels),
        "projection_retained_fraction": 0.0,
        "changed_area_fraction": 0.0,
        "projection_backend": ORGANIC_PROJECTION_BACKEND,
        "noise_seed": int(seed),
        "component_policy": {
            "policy_name": policy.policy_name,
            "params": policy.policy_params,
            "spatial_score_stats": _score_stats(policy.spatial_score, policy.legal_domain),
            "template_score_stats": {"pixels": 0, "min": 0.0, "max": 0.0, "mean": 0.0},
        },
        "cleanup_removed_pixels": 0,
        "cleanup_refill_pixels": 0,
        "cleanup_single_pass": True,
        "cleanup_iteration_limit": 1,
        "decrease_semantics": "select_immune_pixels_then_backfill_from_nearest_legal_tissue",
        "top_failed_reason": warning,
    }
    return PrimitiveEditResult(
        target_mask=np.array(mask, copy=True),
        change_region=np.zeros(mask.shape, dtype=bool),
        changed_area_fraction=0.0,
        selected_pixels=0,
        warnings=tuple(dict.fromkeys(("proposal_projected_region_empty", warning, *policy.warnings))),
        ops_log=ops_log,
    )


def _empty_necrosis_resolution_result(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    preserve_labels: Sequence[str],
    forbidden_labels: Sequence[str],
    raw_candidate_pixels: int,
    legal_pixels: int,
    target_pixels: int,
    seed: int,
    raw_legal_overlap: int,
    backfill_labels: Sequence[str],
    policy: OrganicProjectionPolicy,
    warning: str,
) -> PrimitiveEditResult:
    ops_log = {
        "backend": ORGANIC_PROJECTION_BACKEND,
        "method": "organic_score_projection_and_deterministic_backfill",
        "primitive": "necrosis_resolution",
        "reference_profile": schema.reference_profile,
        "source_labels": ["Necrosis"],
        "target_label": "nearest_backfill_tissue",
        "backfill_labels": list(backfill_labels),
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
        "area_shortfall": int(target_pixels),
        "projection_retained_fraction": 0.0,
        "changed_area_fraction": 0.0,
        "projection_backend": ORGANIC_PROJECTION_BACKEND,
        "noise_seed": int(seed),
        "component_policy": {
            "policy_name": policy.policy_name,
            "params": policy.policy_params,
            "spatial_score_stats": _score_stats(policy.spatial_score, policy.legal_domain),
            "template_score_stats": {"pixels": 0, "min": 0.0, "max": 0.0, "mean": 0.0},
        },
        "cleanup_removed_pixels": 0,
        "cleanup_refill_pixels": 0,
        "cleanup_single_pass": True,
        "cleanup_iteration_limit": 1,
        "resolution_semantics": "select_necrosis_pixels_then_backfill_from_nearest_legal_tissue",
        "top_failed_reason": warning,
    }
    return PrimitiveEditResult(
        target_mask=np.array(mask, copy=True),
        change_region=np.zeros(mask.shape, dtype=bool),
        changed_area_fraction=0.0,
        selected_pixels=0,
        warnings=tuple(dict.fromkeys(("proposal_projected_region_empty", warning, *policy.warnings))),
        ops_log=ops_log,
    )


def _empty_stroma_decrease_result(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive_name: str,
    preserve_labels: Sequence[str],
    forbidden_labels: Sequence[str],
    raw_candidate_pixels: int,
    legal_pixels: int,
    target_pixels: int,
    seed: int,
    raw_legal_overlap: int,
    backfill_labels: Sequence[str],
    policy: OrganicProjectionPolicy,
    warning: str,
) -> PrimitiveEditResult:
    ops_log = {
        "backend": ORGANIC_PROJECTION_BACKEND,
        "method": "organic_score_projection_and_deterministic_backfill",
        "primitive": primitive_name,
        "reference_profile": schema.reference_profile,
        "source_labels": ["Stroma"],
        "target_label": "nearest_backfill_tissue",
        "backfill_labels": list(backfill_labels),
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
        "area_shortfall": int(target_pixels),
        "projection_retained_fraction": 0.0,
        "changed_area_fraction": 0.0,
        "projection_backend": ORGANIC_PROJECTION_BACKEND,
        "noise_seed": int(seed),
        "component_policy": {
            "policy_name": policy.policy_name,
            "params": policy.policy_params,
            "spatial_score_stats": _score_stats(policy.spatial_score, policy.legal_domain),
            "template_score_stats": {"pixels": 0, "min": 0.0, "max": 0.0, "mean": 0.0},
        },
        "cleanup_removed_pixels": 0,
        "cleanup_refill_pixels": 0,
        "cleanup_single_pass": True,
        "cleanup_iteration_limit": 1,
        "decrease_semantics": "select_stroma_pixels_then_backfill_from_nearest_legal_tissue",
        "top_failed_reason": warning,
    }
    return PrimitiveEditResult(
        target_mask=np.array(mask, copy=True),
        change_region=np.zeros(mask.shape, dtype=bool),
        changed_area_fraction=0.0,
        selected_pixels=0,
        warnings=tuple(dict.fromkeys(("proposal_projected_region_empty", warning, *policy.warnings))),
        ops_log=ops_log,
    )


def _empty_tumor_decrease_result(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    preserve_labels: Sequence[str],
    forbidden_labels: Sequence[str],
    raw_candidate_pixels: int,
    legal_pixels: int,
    target_pixels: int,
    seed: int,
    raw_legal_overlap: int,
    backfill_labels: Sequence[str],
    policy: OrganicProjectionPolicy,
    warning: str,
) -> PrimitiveEditResult:
    ops_log = {
        "backend": ORGANIC_PROJECTION_BACKEND,
        "method": "organic_score_projection_and_deterministic_backfill",
        "primitive": "tumor_burden_decrease",
        "reference_profile": schema.reference_profile,
        "source_labels": ["Tumor"],
        "target_label": "nearest_backfill_tissue",
        "backfill_labels": list(backfill_labels),
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
        "area_shortfall": int(target_pixels),
        "projection_retained_fraction": 0.0,
        "changed_area_fraction": 0.0,
        "projection_backend": ORGANIC_PROJECTION_BACKEND,
        "noise_seed": int(seed),
        "component_policy": {
            "policy_name": policy.policy_name,
            "params": policy.policy_params,
            "spatial_score_stats": _score_stats(policy.spatial_score, policy.legal_domain),
            "template_score_stats": {"pixels": 0, "min": 0.0, "max": 0.0, "mean": 0.0},
        },
        "cleanup_removed_pixels": 0,
        "cleanup_refill_pixels": 0,
        "cleanup_single_pass": True,
        "cleanup_iteration_limit": 1,
        "decrease_semantics": "select_tumor_pixels_then_backfill_from_nearest_legal_tissue",
        "top_failed_reason": warning,
    }
    return PrimitiveEditResult(
        target_mask=np.array(mask, copy=True),
        change_region=np.zeros(mask.shape, dtype=bool),
        changed_area_fraction=0.0,
        selected_pixels=0,
        warnings=tuple(dict.fromkeys(("proposal_projected_region_empty", warning, *policy.warnings))),
        ops_log=ops_log,
    )

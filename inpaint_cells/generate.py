#!/usr/bin/env python3
"""
Unified ProbNet-centered nuclei mask generation.

This is the Phase 4 inference entry point. The frozen ProbNet checkpoint's
scalar P(nucleus) = 1 - P(background) field weights spatial landing positions.
Total counts remain controlled by tissue densities measured from the unedited
source patch. ProbNet's feasible joint class intensity controls spatial landing
positions. Its local categorical nucleus posterior is log-pooled with the
target-tissue empirical type prior so OOD edits retain local evidence without
discarding the expected tissue composition. Tissue-local cumulative posterior
balancing removes the high variance of independent categorical draws in small
edited regions.

Nucleus shapes are sampled from the current reference patch first, preserving
the reference morphology domain. The global nuclei library only fills a
same-class shortage and remains a conservative density/area fallback. It no
longer decides type distribution or places cells by rule-only statistics.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_config import get_config
from inpaint_cells.data.density_targets import (
    expand_edit_mask_to_complete_instances,
    iter_class_components,
)
from inpaint_cells.models.prob_unet import ProbUNet
from inpaint_cells.nuclei_library.library import (
    NucleiLibrary,
    ReferenceFirstNucleiSampler,
    ReferenceNucleiInstancePool,
    place_nucleus_layered,
)
from inpaint_cells.sampling_policy import (
    retry_pool_target,
    retry_transform_specs,
    valid_biological_tissue_mask,
    widen_locally_thin_mask,
)
from inpaint_cells.utils.mask_utils import (
    NUCLEI_CLASSES,
    NUCLEI_RGB,
    NUM_NUCLEI,
    load_nuclei_mask,
    load_tissue_mask,
    overlay,
    save_nuclei_mask,
)

GLAS_GLAND_TISSUE_IDS = frozenset({5, 11, 12, 13})
COUNT_POLICY_NAME = (
    "pre_edit_source_tissue_density_or_target_prior_calibrated_by_"
    "pre_edit_source_times_post_edit_target_area"
)
TYPE_QUOTA_ROUTING_POLICY_NAME = (
    "prior_total_count_then_probnet_local_type_log_pool_with_"
    "cumulative_posterior_balancing"
)
COMPONENT_SHAPE_POLICY_NAME = (
    "component_local_same_class_reference_then_component_calibrated_library"
)
SPATIAL_PRIOR_POLICY_NAME = (
    "expanded_support_context_cleared_probnet"
)
DEFAULT_PROBNET_ODDS_GAMMA = 3.0
DEFAULT_LOCAL_TYPE_PRIOR_WEIGHT = 2.0 / 3.0
SAMPLING_AUDIT_POLICY_NAME = "probnet_patch_relative_count_type_spatial_v3"
SAMPLING_FEEDBACK_POLICY_NAME = "reason_directed_gamma_then_seed_v1"
DEFAULT_SAMPLING_FEEDBACK_ATTEMPTS = 3
DEFAULT_SAMPLING_FEEDBACK_GAMMA_DOWN_FACTOR = 0.75
DEFAULT_SAMPLING_FEEDBACK_GAMMA_UP_FACTOR = 4.0 / 3.0
DEFAULT_SAMPLING_FEEDBACK_GAMMA_MIN = 1.5
DEFAULT_SAMPLING_FEEDBACK_GAMMA_MAX = 5.0
DEFAULT_SAMPLING_CONCENTRATION_Z_THRESHOLD = 1.96


def parse_float_list(value):
    """Parse '1,2,3' or repeated-looking strings into a list of floats."""
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def safe_name_float(value):
    return str(value).replace(".", "p").replace("-", "m")


def load_density_scale(path):
    """Load optional tissue-specific semantic density scale JSON."""
    if not path:
        return {}
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): float(v) for k, v in raw.items()}


def load_checkpoint_model(ckpt_path, device, base_ch):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    with_density_head = bool(
        ckpt.get("center_density_head")
        or any(key.startswith("density_head.") for key in state)
    )
    model = ProbUNet(
        out_ch=NUM_NUCLEI,
        base_ch=int(ckpt.get("base_ch", base_ch)),
        with_density_head=with_density_head,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_fields(model, tissue_map, input_nuclei, edit_mask, cancer_id, device):
    tissue_t = torch.from_numpy(tissue_map.astype(np.int64))[None].to(device)
    nuclei_t = torch.from_numpy(input_nuclei.astype(np.int64))[None].to(device)
    mask_t = torch.from_numpy(edit_mask.astype(np.float32))[None, None].to(device)
    cancer_t = torch.tensor([cancer_id], dtype=torch.int64, device=device)

    with torch.no_grad():
        logits, density = model(
            tissue_t,
            nuclei_t,
            mask_t,
            cancer_t,
            return_density=True,
        )
        prob = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
        density_np = None if density is None else density[0].detach().cpu().numpy()
    return prob, density_np


def predict_prob(model, tissue_map, input_nuclei, edit_mask, cancer_id, device):
    """Backward-compatible semantic-only prediction helper."""
    prob, _ = predict_fields(
        model,
        tissue_map,
        input_nuclei,
        edit_mask,
        cancer_id,
        device,
    )
    return prob


def weighted_mean_area(library, tissue_id, fallback):
    stats = library.stats.get(str(tissue_id), {})
    type_stats = stats.get("nuclei_types", {})
    weighted = []
    weights = []
    for info in type_stats.values():
        mean_area = float(info.get("mean_area", 0.0))
        frac = float(info.get("fraction", 0.0))
        if mean_area > 0 and frac > 0:
            weighted.append(mean_area)
            weights.append(frac)
    if weighted and sum(weights) > 0:
        return float(np.average(weighted, weights=weights))
    return float(fallback)


def adaptive_min_distance(expected_area, args, oversample_factor):
    if args.min_distance_mode == "fixed":
        base = args.min_distance
    else:
        diameter = np.sqrt(max(expected_area, 1.0) / np.pi) * 2.0
        base = diameter * args.min_distance_scale
        base = float(np.clip(base, args.min_distance_min, args.min_distance_max))

    if args.shrink_distance_for_oversample:
        base = base / np.sqrt(max(oversample_factor, 1.0))
    return max(base, args.min_distance_floor)


def poisson_candidates(region_mask, min_distance, max_attempts=30):
    """Poisson disk candidates, intentionally local to keep this entry configurable."""
    h, w = region_mask.shape
    valid_ys, valid_xs = np.where(region_mask)
    if len(valid_ys) == 0:
        return []

    cell_size = max(min_distance / np.sqrt(2.0), 1e-3)
    grid_h = int(np.ceil(h / cell_size))
    grid_w = int(np.ceil(w / cell_size))
    grid = -np.ones((grid_h, grid_w), dtype=np.int64)

    points = []
    active = []
    idx = random.randint(0, len(valid_ys) - 1)
    start = (int(valid_ys[idx]), int(valid_xs[idx]))
    points.append(start)
    active.append(0)
    grid[int(start[0] / cell_size), int(start[1] / cell_size)] = 0

    while active:
        active_idx = random.randint(0, len(active) - 1)
        point_idx = active[active_idx]
        py, px = points[point_idx]
        found = False

        for _ in range(max_attempts):
            angle = random.uniform(0, 2 * np.pi)
            dist = random.uniform(min_distance, 2 * min_distance)
            ny = int(py + dist * np.sin(angle))
            nx = int(px + dist * np.cos(angle))

            if ny < 0 or ny >= h or nx < 0 or nx >= w or not region_mask[ny, nx]:
                continue

            ngy, ngx = int(ny / cell_size), int(nx / cell_size)
            too_close = False
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    gy, gx = ngy + dy, ngx + dx
                    if 0 <= gy < grid_h and 0 <= gx < grid_w and grid[gy, gx] >= 0:
                        ey, ex = points[grid[gy, gx]]
                        if (ny - ey) ** 2 + (nx - ex) ** 2 < min_distance ** 2:
                            too_close = True
                            break
                if too_close:
                    break

            if not too_close:
                new_idx = len(points)
                points.append((ny, nx))
                active.append(new_idx)
                grid[ngy, ngx] = new_idx
                found = True
                break

        if not found:
            active.pop(active_idx)

    return points


def supplement_retry_candidates(candidates, region_mask, minimum_candidates):
    """Add alternate valid centers when Poisson yields too few retry options."""
    points = list(candidates)
    minimum_candidates = int(max(minimum_candidates, 0))
    if len(points) >= minimum_candidates:
        return points
    existing = set(points)
    valid_y, valid_x = np.where(region_mask)
    available = [
        (int(y), int(x))
        for y, x in zip(valid_y, valid_x)
        if (int(y), int(x)) not in existing
    ]
    needed = min(minimum_candidates - len(points), len(available))
    if needed <= 0:
        return points
    chosen = np.random.choice(len(available), size=needed, replace=False)
    points.extend(available[int(index)] for index in chosen)
    return points


def compute_target_count(nuc_prob, tissue_region, tissue_id, library, expected_area, args, scale):
    region_area = int(tissue_region.sum())
    prob_count = float(nuc_prob[tissue_region].sum() / max(expected_area, 1.0))

    library_density = float(library.get_density(tissue_id))
    library_count = library_density * region_area / 10000.0

    if library_count <= 0:
        blended = prob_count
    else:
        blended = args.prob_count_weight * prob_count + (1.0 - args.prob_count_weight) * library_count

    scaled = blended * scale
    max_by_density = args.max_density_per_10k * region_area / 10000.0
    max_allowed = max_by_density
    if library_count > 0 and args.max_count_factor > 0:
        max_allowed = min(max_allowed, library_count * args.max_count_factor)

    clipped = float(np.clip(scaled, args.min_count, max_allowed))
    return round(clipped), {
        "region_area": region_area,
        "prob_count": prob_count,
        "library_density_per_10k": library_density,
        "library_count": library_count,
        "semantic_scale": scale,
        "blended_count": blended,
        "clipped_count": clipped,
    }


def quota_coverage_radius(
    region_area,
    quota,
    candidate_min_distance,
    spacing_scale=0.75,
    maximum=48.0,
):
    """Return a generic quota-aware spacing for the primary placement prefix."""

    if quota <= 0 or region_area <= 0:
        return float(max(candidate_min_distance, 0.0))
    area_spacing = np.sqrt(float(region_area) / float(quota))
    return float(
        np.clip(
            area_spacing * float(spacing_scale),
            float(candidate_min_distance),
            float(max(maximum, candidate_min_distance)),
        )
    )


def adaptive_quota_coverage_count(
    candidates,
    nuc_prob,
    quota,
    *,
    minimum_fraction=0.2,
):
    """Reduce coverage when ProbNet has a genuinely sharp high-score tail.

    A flat or broad field needs quality-diversity coverage. A sharp field
    should retain more of the stable score-descending order. The ratio is
    computed from each component's own score distribution, without any
    dataset, tissue, or organ branch.
    """

    quota = int(max(quota, 0))
    minimum_fraction = float(np.clip(minimum_fraction, 0.0, 1.0))
    if quota <= 0 or not candidates:
        return 0, {
            "policy": "adaptive_probnet_tail_sharpness",
            "quota": quota,
            "coverage_count": 0,
            "coverage_fraction": 0.0,
            "tail_sharpness": 0.0,
            "probability_quantiles": {},
        }

    ys = np.asarray([point[0] for point in candidates], dtype=np.int64)
    xs = np.asarray([point[1] for point in candidates], dtype=np.int64)
    values = np.asarray(nuc_prob, dtype=np.float64)[ys, xs]
    q10, q50, q90, q99 = (
        float(value)
        for value in np.quantile(values, (0.10, 0.50, 0.90, 0.99))
    )
    bulk_span = max(q90 - q10, 1e-8)
    tail_sharpness = float(np.clip((q99 - q90) / bulk_span, 0.0, 1.0))
    coverage_fraction = float(
        1.0 - (1.0 - minimum_fraction) * tail_sharpness
    )
    coverage_count = int(
        np.clip(np.ceil(quota * coverage_fraction), 1, quota)
    )
    return coverage_count, {
        "policy": "adaptive_probnet_tail_sharpness",
        "quota": quota,
        "coverage_count": coverage_count,
        "coverage_fraction": coverage_fraction,
        "minimum_coverage_fraction": minimum_fraction,
        "tail_sharpness": tail_sharpness,
        "probability_quantiles": {
            "q10": q10,
            "q50": q50,
            "q90": q90,
            "q99": q99,
        },
    }


def compile_quota_coverage_contract(
    candidates,
    nuc_prob,
    quota,
    *,
    region_area,
    candidate_min_distance,
    args,
):
    """Compile one generic, auditable coverage contract for a quota stratum."""

    adaptive = bool(getattr(args, "adaptive_quota_coverage", True))
    minimum_fraction = float(
        getattr(args, "quota_coverage_min_fraction", 0.2)
    )
    if adaptive:
        coverage_count, audit = adaptive_quota_coverage_count(
            candidates,
            nuc_prob,
            quota,
            minimum_fraction=minimum_fraction,
        )
    else:
        coverage_count = min(max(int(quota), 0), len(candidates))
        audit = {
            "policy": "fixed_full_quota_coverage",
            "quota": int(max(quota, 0)),
            "coverage_count": int(coverage_count),
            "coverage_fraction": (
                float(coverage_count) / float(max(int(quota), 1))
            ),
            "minimum_coverage_fraction": minimum_fraction,
            "tail_sharpness": None,
            "probability_quantiles": {},
        }
    radius = quota_coverage_radius(
        region_area=region_area,
        quota=quota,
        candidate_min_distance=candidate_min_distance,
        spacing_scale=float(
            getattr(args, "quota_coverage_spacing_scale", 0.75)
        ),
        maximum=float(getattr(args, "quota_coverage_max_radius", 48.0)),
    )
    audit.update(
        {
            "adaptive": adaptive,
            "region_area": int(region_area),
            "candidate_count": len(candidates),
            "coverage_radius": float(radius),
            "candidate_min_distance": float(candidate_min_distance),
        }
    )
    return int(coverage_count), float(radius), audit


def choose_weighted_centers(
    candidates,
    nuc_prob,
    target_count,
    gamma,
    *,
    coverage_count=None,
    coverage_radius=0.0,
):
    """Sample a ProbNet-weighted retry queue with a coverage prefix.

    Gumbel ranking gives a weighted sample without replacement, so candidates
    remain proportional to ProbNet mass instead of becoming deterministic
    Top-N.  When a quota coverage contract is supplied, only its prefix is
    diversified: the first available ProbNet-weighted point at the requested
    separation is selected, falling back to the farthest point when the region
    cannot support that separation.  The untouched weighted order remains the
    retry tail.  This prevents a small quota from collapsing into one local
    peak without turning a weak ProbNet preference into a geometric ring.

    The generic calibration mass is ``odds(probability) ** gamma``. It preserves
    flat fields while expressing sharp ProbNet structure without a dataset,
    tissue, or organ threshold.
    """

    if target_count <= 0 or not candidates:
        return []
    n = min(target_count, len(candidates))
    ys = np.array([p[0] for p in candidates], dtype=np.int64)
    xs = np.array([p[1] for p in candidates], dtype=np.int64)
    log_mass = probability_sampling_log_mass(
        np.asarray(nuc_prob, dtype=np.float64)[ys, xs],
        gamma,
    )
    gumbel = np.random.gumbel(size=log_mass.shape[0])
    weighted_order = np.argsort(
        -(log_mass + gumbel),
        kind="stable",
    )
    prefix_target = min(
        n,
        int(coverage_count) if coverage_count is not None else 0,
    )
    radius = max(float(coverage_radius), 0.0)
    if prefix_target <= 1 or radius <= 0:
        return [candidates[int(index)] for index in weighted_order[:n]]

    coordinates = np.column_stack((ys, xs)).astype(np.float64)
    relative_mass = np.exp(log_mass - float(np.max(log_mass)))
    selected = np.zeros(len(candidates), dtype=bool)
    prefix_indices = [int(weighted_order[0])]
    selected[prefix_indices[0]] = True
    minimum_distance_sq = np.sum(
        (coordinates - coordinates[prefix_indices[0]]) ** 2,
        axis=1,
    )
    # Seed weighted spatial partitions.  Reusing the already sampled Gumbel
    # perturbation keeps this reproducible under the caller's deterministic
    # seed while probability mass and uncovered distance jointly decide which
    # part of a long/curved region receives the next partition.
    while len(prefix_indices) < prefix_target:
        eligible = (~selected) & (minimum_distance_sq >= radius * radius)
        utility = (
            np.log(np.clip(relative_mass, 1e-300, None))
            + np.log(np.maximum(minimum_distance_sq, 1e-12))
            + gumbel
        )
        utility[~eligible] = -np.inf
        if np.any(eligible):
            candidate_index = int(np.argmax(utility))
        else:
            # The finite candidate raster cannot satisfy another radius-
            # separated center.  Continue with the farthest remaining point;
            # the downstream continuity gate still decides whether the
            # achieved configuration is sufficient.
            fallback = np.where(
                selected,
                -np.inf,
                minimum_distance_sq,
            )
            candidate_index = int(np.argmax(fallback))
            if not np.isfinite(fallback[candidate_index]):
                break
        prefix_indices.append(candidate_index)
        selected[candidate_index] = True
        minimum_distance_sq = np.minimum(
            minimum_distance_sq,
            np.sum(
                (coordinates - coordinates[candidate_index]) ** 2,
                axis=1,
            ),
        )

    selected[:] = False
    selected[np.asarray(prefix_indices, dtype=np.int64)] = True

    retry_tail = [
        int(index)
        for index in weighted_order
        if not selected[int(index)]
    ]
    queue = prefix_indices + retry_tail
    return [candidates[index] for index in queue[:n]]


def probability_sampling_log_mass(probability, gamma):
    """Convert ProbNet occupancy probability to calibrated log intensity."""

    values = np.clip(
        np.asarray(probability, dtype=np.float64),
        1e-6,
        1.0 - 1e-6,
    )
    return float(gamma) * (
        np.log(values) - np.log1p(-values)
    )


def probability_sampling_mass(probability, gamma):
    """Return finite relative odds mass for allocation and diagnostics."""

    log_mass = probability_sampling_log_mass(probability, gamma)
    log_mass = log_mass - float(np.max(log_mass))
    return np.exp(log_mass)


def probability_mass_region_centers(region_mask, nuc_prob, gamma):
    """Return supported pixels in a ProbNet-mass weighted retry order."""

    region = np.asarray(region_mask, dtype=bool)
    probability_map = np.asarray(nuc_prob)
    if region.shape != probability_map.shape:
        raise ValueError("region_mask and nuc_prob must share one shape")
    flat_indices = np.flatnonzero(region.ravel())
    if flat_indices.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    log_mass = probability_sampling_log_mass(
        probability_map.ravel()[flat_indices],
        gamma,
    )
    gumbel = np.random.gumbel(size=log_mass.shape[0])
    order = np.argsort(
        -(log_mass + gumbel),
        kind="stable",
    )
    ranked = flat_indices[order]
    width = int(region.shape[1])
    return np.column_stack((ranked // width, ranked % width)).astype(
        np.int64,
        copy=False,
    )


def exact_backfill_candidate_budget(shortfall, available, args):
    """Bound one deterministic exact-count search before the next seed.

    Exact-count placement is still mandatory. This budget only prevents an
    infeasible geometry from trying every pixel in a large tissue component;
    the outer sampling loop then retries with the next deterministic seed.
    """

    missing = max(0, int(shortfall))
    available = max(0, int(available))
    per_missing = max(
        1,
        int(getattr(args, "exact_backfill_candidates_per_missing", 128)),
    )
    floor = max(
        1,
        int(getattr(args, "exact_backfill_candidate_floor", 512)),
    )
    ceiling = max(
        floor,
        int(getattr(args, "exact_backfill_candidate_ceiling", 4096)),
    )
    requested = min(ceiling, max(floor, missing * per_missing))
    return min(available, requested), {
        "policy": "quota_scaled_bounded_search_then_next_deterministic_seed",
        "shortfall": missing,
        "available_candidates": available,
        "candidates_per_missing": per_missing,
        "floor": floor,
        "ceiling": ceiling,
        "budget": min(available, requested),
        "truncated": bool(available > requested),
    }


def same_tissue_quota_reassignment_centers(
    tissue_region,
    output,
    nuc_prob,
    gamma,
    separation,
):
    """Rank currently available centers across one target tissue."""

    occupied = np.asarray(output) > 0
    margin = max(0, int(separation))
    if margin > 0:
        occupied = ndimage.binary_dilation(
            occupied,
            structure=np.ones((2 * margin + 1, 2 * margin + 1), dtype=bool),
        )
    return probability_mass_region_centers(
        np.asarray(tissue_region, dtype=bool) & ~occupied,
        nuc_prob,
        gamma,
    )


def initialize_component_sampling_diagnostics(
    component_areas,
    component_limits,
    component_mass_by_id,
):
    """Register every component before quota placement or reassignment."""

    total_mass = float(sum(component_mass_by_id.values()))
    diagnostics = {}
    for component_id, area in component_areas:
        mass = float(component_mass_by_id.get(component_id, 0.0))
        diagnostics[str(component_id)] = {
            "area": int(area),
            "quota": int(component_limits.get(component_id, 0)),
            "dense_retry": False,
            "expected_occupancy_fraction": 0.0,
            "retry_pool_target": 0,
            "num_candidates": 0,
            "selected_centers": 0,
            "attempted_centers": 0,
            "placed": 0,
            "integrated_probnet_mass": mass,
            "probnet_mass_fraction": (
                float(mass / total_mass) if total_mass > 0 else 0.0
            ),
            "component_count_policy": (
                "integrated_probnet_mass_largest_remainder"
            ),
        }
    return diagnostics


def shape_sampling_diagnostics(
    shape_sampler,
    component_shape_sampling,
    placed_by_shape_source,
    *,
    component_policy_active,
):
    """Report the active product policy even when this stage places no shape."""

    diagnostics = shape_sampler.diagnostics()
    if component_policy_active:
        effective_calibration = _aggregate_component_size_calibration(
            component_shape_sampling
        )
        diagnostics.update(
            {
                "policy": COMPONENT_SHAPE_POLICY_NAME,
                "selected_by_source": dict(placed_by_shape_source),
                "component_local": dict(component_shape_sampling),
                "library_size_calibration": effective_calibration,
            }
        )
    return diagnostics


def _aggregate_component_size_calibration(component_shape_sampling):
    """Summarize the samplers that actually realized component-local shapes."""

    totals = {
        "calibrated_by_type": {},
        "uncalibrated_no_reference_by_type": {},
        "scale_clamped_by_type": {},
    }
    reference_basis = {}
    enabled = True
    min_scale = None
    max_scale = None
    log_area_jitter = None
    for tissue_components in (component_shape_sampling or {}).values():
        for sampler in (tissue_components or {}).values():
            calibration = (sampler or {}).get("library_size_calibration") or {}
            enabled = enabled and bool(calibration.get("enabled", False))
            min_scale = calibration.get("min_scale", min_scale)
            max_scale = calibration.get("max_scale", max_scale)
            log_area_jitter = calibration.get(
                "log_area_jitter", log_area_jitter
            )
            for field, field_totals in totals.items():
                for key, value in (calibration.get(field) or {}).items():
                    field_totals[str(key)] = int(
                        field_totals.get(str(key), 0)
                    ) + int(value)
            for key, value in (
                calibration.get("reference_basis_by_type") or {}
            ).items():
                reference_basis.setdefault(str(key), set()).add(str(value))
    return {
        "enabled": bool(enabled),
        "policy": "target_tissue_then_patch_same_class_empirical_area",
        "scope": "active_component_samplers",
        "min_scale": min_scale,
        "max_scale": max_scale,
        "log_area_jitter": log_area_jitter,
        **totals,
        "reference_basis_by_type": {
            key: sorted(values) for key, values in sorted(reference_basis.items())
        },
    }


def allocate_component_counts(component_areas, target_count, minimum_area):
    """Allocate a tissue-level target count across disconnected components.

    Components large enough to hold a nucleus receive one guaranteed slot when
    the total target count permits it. Remaining slots are distributed by area
    with the largest-remainder method, so probability weighting only controls
    placement *within* a component rather than starving whole components.
    """
    items = [
        (int(component_id), int(area))
        for component_id, area in component_areas
        if int(area) >= int(minimum_area)
    ]
    if target_count <= 0 or not items:
        return {}
    if target_count < len(items):
        largest = sorted(items, key=lambda item: (-item[1], item[0]))[:target_count]
        return {component_id: 1 for component_id, _ in largest}

    quotas = {component_id: 1 for component_id, _ in items}
    remaining = int(target_count) - len(items)
    if remaining <= 0:
        return quotas
    total_area = float(sum(area for _, area in items))
    raw = {
        component_id: remaining * area / total_area
        for component_id, area in items
    }
    floors = {component_id: int(np.floor(value)) for component_id, value in raw.items()}
    for component_id, count in floors.items():
        quotas[component_id] += count
    leftover = remaining - sum(floors.values())
    order = sorted(
        items,
        key=lambda item: (-(raw[item[0]] - floors[item[0]]), -item[1], item[0]),
    )
    for component_id, _ in order[:leftover]:
        quotas[component_id] += 1
    return quotas


def allocate_area_proportional_counts(component_areas, target_count, minimum_area):
    """Allocate counts by component area with the largest-remainder method.

    Unlike :func:`allocate_component_counts`, this policy does not reserve one
    cell for every eligible component.  A component whose expected count is
    below one may therefore receive zero, which preserves the requested
    tissue-level density in the larger components.
    """
    items = [
        (int(component_id), int(area))
        for component_id, area in component_areas
        if int(area) >= int(minimum_area)
    ]
    if target_count <= 0 or not items:
        return {}
    total_area = float(sum(area for _, area in items))
    raw = {
        component_id: int(target_count) * area / total_area
        for component_id, area in items
    }
    quotas = {component_id: int(np.floor(value)) for component_id, value in raw.items()}
    leftover = int(target_count) - sum(quotas.values())
    order = sorted(
        items,
        key=lambda item: (-(raw[item[0]] - quotas[item[0]]), -item[1], item[0]),
    )
    for component_id, _ in order[:leftover]:
        quotas[component_id] += 1
    return {
        component_id: count
        for component_id, count in quotas.items()
        if count > 0
    }


def allocate_weight_proportional_counts(component_weights, target_count):
    """Allocate a fixed total according to integrated ProbNet mass."""

    items = [
        (int(component_id), max(float(weight), 0.0))
        for component_id, weight in component_weights
    ]
    if target_count <= 0 or not items:
        return {}
    total_weight = float(sum(weight for _, weight in items))
    if total_weight <= 0:
        items = [(component_id, 1.0) for component_id, _ in items]
        total_weight = float(len(items))
    raw = {
        component_id: int(target_count) * weight / total_weight
        for component_id, weight in items
    }
    quotas = {
        component_id: int(np.floor(value))
        for component_id, value in raw.items()
    }
    leftover = int(target_count) - sum(quotas.values())
    order = sorted(
        items,
        key=lambda item: (
            -(raw[item[0]] - quotas[item[0]]),
            -item[1],
            item[0],
        ),
    )
    for component_id, _ in order[:leftover]:
        quotas[component_id] += 1
    return {
        component_id: count
        for component_id, count in quotas.items()
        if count > 0
    }


def allocate_type_counts(type_proportions, target_count):
    """Turn a tissue-local type distribution into an exact integer quota."""
    items = [
        (int(nuc_type), float(weight))
        for nuc_type, weight in type_proportions.items()
        if float(weight) > 0
    ]
    if target_count <= 0 or not items:
        return {}
    total_weight = float(sum(weight for _, weight in items))
    raw = {
        nuc_type: int(target_count) * weight / total_weight
        for nuc_type, weight in items
    }
    quotas = {nuc_type: int(np.floor(value)) for nuc_type, value in raw.items()}
    leftover = int(target_count) - sum(quotas.values())
    order = sorted(
        items,
        key=lambda item: (-(raw[item[0]] - quotas[item[0]]), item[0]),
    )
    for nuc_type, _ in order[:leftover]:
        quotas[nuc_type] += 1
    return {nuc_type: count for nuc_type, count in quotas.items() if count > 0}


def supported_nucleus_shape_types(
    library,
    reference_pool,
    tissue_id,
    *,
    force_tissue_library,
):
    """Return classes with a real same-class shape available to this tissue."""

    supported = set()
    if not force_tissue_library and reference_pool is not None:
        supported.update(
            int(nuc_type)
            for nuc_type, count in reference_pool.counts().items()
            if int(count) > 0
        )
    library_instances = getattr(library, "instances", {})
    if force_tissue_library or int(tissue_id) == 3:
        buckets = [library_instances.get(int(tissue_id), ())]
    else:
        buckets = library_instances.values()
    for bucket in buckets:
        supported.update(
            int(instance["type"])
            for instance in bucket
            if int(instance.get("type", 0)) in NUCLEI_CLASSES
        )
    return supported


def constrain_type_quota_to_shape_support(
    requested_counts,
    type_evidence,
    supported_types,
):
    """Redistribute only unsupported quota while preserving the total count."""

    requested = {
        int(nuc_type): int(count)
        for nuc_type, count in (requested_counts or {}).items()
        if int(count) > 0
    }
    supported = {
        int(nuc_type)
        for nuc_type in supported_types
        if int(nuc_type) in NUCLEI_CLASSES
    }
    unsupported = {
        nuc_type: count
        for nuc_type, count in requested.items()
        if nuc_type not in supported
    }
    feasible = {
        nuc_type: count
        for nuc_type, count in requested.items()
        if nuc_type in supported
    }
    redistributed_count = int(sum(unsupported.values()))
    if redistributed_count > 0:
        weights = {
            int(nuc_type): float(weight)
            for nuc_type, weight in (type_evidence or {}).items()
            if int(nuc_type) in supported and float(weight) > 0
        }
        if not weights:
            weights = {int(nuc_type): 1.0 for nuc_type in sorted(supported)}
        if not weights:
            raise RuntimeError(
                "Nucleus type quota has no same-class reference or library "
                "shape support."
            )
        redistributed = allocate_type_counts(weights, redistributed_count)
        for nuc_type, count in redistributed.items():
            feasible[nuc_type] = feasible.get(nuc_type, 0) + int(count)
    if sum(feasible.values()) != sum(requested.values()):
        raise AssertionError("shape-support quota must preserve total count")
    audit = {
        "policy": (
            "preserve_supported_quota_then_redistribute_unsupported_by_"
            "target_type_evidence"
        ),
        "supported_types": sorted(int(value) for value in supported),
        "requested_by_type": {
            str(key): int(value) for key, value in requested.items()
        },
        "unsupported_requested_by_type": {
            str(key): int(value) for key, value in unsupported.items()
        },
        "feasible_by_type": {
            str(key): int(value) for key, value in feasible.items()
        },
        "redistributed_count": redistributed_count,
    }
    return feasible, audit


def fuse_density_head_with_tissue_prior(
    density_by_class,
    tissue_prior,
    *,
    density_weight=0.5,
    adaptive=False,
):
    """Fuse normalized target-conditioned type evidence and tissue prior.

    When ``adaptive`` is enabled, ``density_weight`` is an upper bound. The
    effective head weight is reduced by normalized head entropy and
    Jensen-Shannon agreement with the target-tissue prior. This prevents an
    uncertain density head from copying source-context cell types into a newly
    introduced target tissue.
    """

    requested_weight = float(density_weight)
    if not 0.0 <= requested_weight <= 1.0:
        raise ValueError("density_weight must be between 0 and 1")
    weight = requested_weight

    density_values = np.asarray(density_by_class, dtype=np.float64).reshape(-1)
    density_values = np.clip(density_values, 0.0, None)
    density_distribution = {
        int(nuc_type): float(density_values[index])
        for index, nuc_type in enumerate(NUCLEI_CLASSES)
        if index < density_values.size and density_values[index] > 0
    }
    prior_distribution = {
        int(nuc_type): max(float(value), 0.0)
        for nuc_type, value in (tissue_prior or {}).items()
        if float(value) > 0
    }

    density_total = float(sum(density_distribution.values()))
    prior_total = float(sum(prior_distribution.values()))
    if density_total <= 0 and prior_total <= 0:
        return {}, {
            "adaptive_weighting": bool(adaptive),
            "requested_density_head_weight": requested_weight,
            "density_head_weight": weight,
            "density_head_certainty": 0.0,
            "density_prior_agreement": 0.0,
            "jensen_shannon_divergence": None,
            "density_head_distribution": {},
            "tissue_prior_distribution": {},
            "fused_distribution": {},
            "fallback": "no_type_evidence",
        }
    if density_total <= 0:
        weight = 0.0
        fallback = "tissue_prior_only_no_density_evidence"
    elif prior_total <= 0:
        weight = 1.0
        fallback = "density_head_only_no_tissue_prior"
    else:
        fallback = None

    density_normalized = {
        int(nuc_type): float(value / density_total)
        for nuc_type, value in density_distribution.items()
    }
    prior_normalized = {
        int(nuc_type): float(value / prior_total)
        for nuc_type, value in prior_distribution.items()
    }
    density_head_certainty = 0.0
    density_prior_agreement = 0.0
    jensen_shannon_divergence = None
    if density_total > 0:
        density_vector = np.asarray(
            [
                density_normalized.get(int(nuc_type), 0.0)
                for nuc_type in NUCLEI_CLASSES
            ],
            dtype=np.float64,
        )
        nonzero = density_vector[density_vector > 0]
        entropy = float(-np.sum(nonzero * np.log(nonzero)))
        density_head_certainty = float(
            np.clip(
                1.0 - entropy / np.log(max(len(NUCLEI_CLASSES), 2)),
                0.0,
                1.0,
            )
        )
    if density_total > 0 and prior_total > 0:
        prior_vector = np.asarray(
            [
                prior_normalized.get(int(nuc_type), 0.0)
                for nuc_type in NUCLEI_CLASSES
            ],
            dtype=np.float64,
        )
        midpoint = 0.5 * (density_vector + prior_vector)

        def _kl_divergence(left, right):
            support = left > 0
            return float(
                np.sum(left[support] * np.log2(left[support] / right[support]))
            )

        jensen_shannon_divergence = float(
            np.clip(
                0.5 * _kl_divergence(density_vector, midpoint)
                + 0.5 * _kl_divergence(prior_vector, midpoint),
                0.0,
                1.0,
            )
        )
        density_prior_agreement = 1.0 - jensen_shannon_divergence
        if adaptive:
            weight = float(
                requested_weight
                * density_head_certainty
                * density_prior_agreement
            )
    elif density_total > 0:
        density_prior_agreement = 1.0

    fused = {
        int(nuc_type): (
            weight * density_normalized.get(int(nuc_type), 0.0)
            + (1.0 - weight) * prior_normalized.get(int(nuc_type), 0.0)
        )
        for nuc_type in NUCLEI_CLASSES
    }
    fused = {
        int(nuc_type): float(value)
        for nuc_type, value in fused.items()
        if value > 0
    }
    return fused, {
        "adaptive_weighting": bool(adaptive),
        "requested_density_head_weight": requested_weight,
        "density_head_weight": float(weight),
        "density_head_certainty": density_head_certainty,
        "density_prior_agreement": density_prior_agreement,
        "jensen_shannon_divergence": jensen_shannon_divergence,
        "density_head_distribution": {
            str(key): float(value)
            for key, value in density_normalized.items()
        },
        "tissue_prior_distribution": {
            str(key): float(value)
            for key, value in prior_normalized.items()
        },
        "fused_distribution": {
            str(key): float(value)
            for key, value in fused.items()
        },
        "fallback": fallback,
    }


def spatial_context_halo_radius(
    expected_nucleus_area,
    *,
    diameter_scale=1.25,
    minimum=4,
    maximum=24,
):
    """Return a generic one-nucleus-scale context clearance radius."""

    diameter = 2.0 * np.sqrt(max(float(expected_nucleus_area), 1.0) / np.pi)
    return int(
        np.clip(
            np.ceil(diameter * float(diameter_scale)),
            int(minimum),
            int(maximum),
        )
    )


def blend_context_stabilized_probability(
    context_probability,
    halo_cleared_probability,
    *,
    halo_weight=0.25,
):
    """Geometrically blend raw context and halo-cleared spatial priors."""

    weight = float(halo_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("halo_weight must be between 0 and 1")
    context = np.clip(
        np.asarray(context_probability, dtype=np.float64),
        1e-6,
        1.0,
    )
    stabilized = np.clip(
        np.asarray(halo_cleared_probability, dtype=np.float64),
        1e-6,
        1.0,
    )
    if context.shape != stabilized.shape:
        raise ValueError("spatial probability maps must share one shape")
    return np.exp(
        (1.0 - weight) * np.log(context)
        + weight * np.log(stabilized)
    ).astype(np.float32)


def quota_conditioned_spatial_probability(
    nucleus_probability,
    type_probability,
    type_quota,
    region_mask,
):
    """Condition spatial scoring on the exact target type composition.

    ProbNet emits one spatial channel per nucleus class. Sharing only the
    summed P(nucleus) field lets a strong but minority class pull every target
    type toward its preferred locations. This mixture uses the already
    determined exact quota as generic weights, then preserves the mean scale
    of P(nucleus) so downstream quality/diversity behavior remains calibrated.
    """

    base = np.asarray(nucleus_probability, dtype=np.float64)
    typed = np.asarray(type_probability, dtype=np.float64)
    region = np.asarray(region_mask, dtype=bool)
    if typed.shape != (len(NUCLEI_CLASSES), *base.shape):
        raise ValueError("type_probability must be [num_types, height, width]")
    if region.shape != base.shape:
        raise ValueError("region_mask and nucleus_probability must share shape")

    quota = {
        int(nuc_type): int(count)
        for nuc_type, count in (type_quota or {}).items()
        if int(count) > 0
    }
    total = int(sum(quota.values()))
    if total <= 0 or not np.any(region):
        return base.astype(np.float32), {
            "policy": "unconditioned_no_type_quota",
            "quota_weights": {},
            "mean_scale": 1.0,
        }

    weights = np.asarray(
        [
            float(quota.get(int(nuc_type), 0)) / float(total)
            for nuc_type in NUCLEI_CLASSES
        ],
        dtype=np.float64,
    )
    mixed = np.tensordot(weights, np.clip(typed, 0.0, None), axes=(0, 0))
    base_mean = float(np.mean(base[region]))
    mixed_mean = float(np.mean(mixed[region]))
    mean_scale = base_mean / mixed_mean if mixed_mean > 0 else 1.0
    conditioned = np.clip(mixed * mean_scale, 1e-6, 1.0)
    return conditioned.astype(np.float32), {
        "policy": "exact_type_quota_weighted_probnet_semantic_channels",
        "quota_weights": {
            str(nuc_type): float(weights[index])
            for index, nuc_type in enumerate(NUCLEI_CLASSES)
            if weights[index] > 0
        },
        "base_mean_probability": base_mean,
        "mixed_mean_probability_before_rescale": mixed_mean,
        "mean_scale": float(mean_scale),
    }


def sample_type_with_remaining_quota(type_limits, placed_by_type):
    """Sample from the remaining empirical quota, independent of ProbNet."""
    available = [
        nuc_type
        for nuc_type, limit in type_limits.items()
        if placed_by_type.get(nuc_type, 0) < limit
    ]
    if not available:
        return None
    remaining = np.asarray(
        [type_limits[nuc_type] - placed_by_type.get(nuc_type, 0) for nuc_type in available],
        dtype=np.float64,
    )
    weights = remaining
    weights /= weights.sum()
    return int(np.random.choice(available, p=weights))


def choose_type_with_remaining_quota_at_center(
    type_limits,
    placed_by_type,
    prob,
    center_y,
    center_x,
):
    """Greedily assign the locally strongest type with quota remaining."""

    available = [
        int(nuc_type)
        for nuc_type, limit in type_limits.items()
        if placed_by_type.get(int(nuc_type), 0) < int(limit)
    ]
    if not available:
        return None
    index_by_type = {
        int(nuc_type): index
        for index, nuc_type in enumerate(NUCLEI_CLASSES)
    }
    return max(
        available,
        key=lambda nuc_type: (
            float(
                prob[
                    index_by_type[int(nuc_type)] + 1,
                    int(center_y),
                    int(center_x),
                ]
            ),
            int(type_limits[int(nuc_type)])
            - int(placed_by_type.get(int(nuc_type), 0)),
            -int(nuc_type),
        ),
    )


def count_retained_centers_by_type(nuclei_map, region_mask):
    """Count retained complete nuclei by centroid inside one target region."""

    region = np.asarray(region_mask, dtype=bool)
    counts = {int(nuc_type): 0 for nuc_type in NUCLEI_CLASSES}
    for class_id, _, (center_y, center_x) in iter_class_components(
        np.asarray(nuclei_map),
        num_classes=len(NUCLEI_CLASSES),
    ):
        row = int(np.clip(round(center_y), 0, region.shape[0] - 1))
        col = int(np.clip(round(center_x), 0, region.shape[1] - 1))
        if region[row, col]:
            raw_type = int(NUCLEI_CLASSES[int(class_id) - 1])
            counts[raw_type] += 1
    return counts


def density_calibration_family_ids(dataset_name, target_tissue_id):
    """Return source tissues usable for a target-prior calibration factor."""

    normalized_dataset = str(dataset_name or "").strip().lower()
    target_id = int(target_tissue_id)
    if normalized_dataset == "glas" and target_id in GLAS_GLAND_TISSUE_IDS:
        return GLAS_GLAND_TISSUE_IDS
    return frozenset()


def compute_patch_adaptive_priors(
    *,
    reference_nuclei_raw,
    reference_tissue,
    density_exclusion_region,
    target_tissue,
    generation_region,
    library,
    global_density_scale=1.0,
    local_density_direct_min_area=20000,
    local_density_direct_min_count=10,
    dataset_name=None,
    source_instance_authority=None,
):
    """Estimate count and fallback type priors from the unedited source patch.

    Patch density and calibration evidence are always measured on
    ``reference_tissue`` before the edit. The deletion/generation masks must
    never remove source observations from that estimate: they only define which
    target area will be populated. Exact target-tissue observations take
    precedence. If a GLaS target grade is absent, its grade-specific dataset
    prior is scaled by a cellularity factor measured across the pre-edit gland
    family; the grade priors are never pooled away.

    A sufficiently large source-patch observation is used directly. Sparse
    observations shrink toward the dataset tissue-density prior. Nucleus type
    quotas use the reliable patch-local distribution only as a fallback when
    density-head evidence is unavailable.
    """
    shape = reference_tissue.shape
    if not (
        reference_nuclei_raw.shape
        == density_exclusion_region.shape
        == target_tissue.shape
        == generation_region.shape
        == shape
    ):
        raise ValueError("patch-adaptive prior inputs must share one shape")

    patch_area = int(reference_tissue.size)
    source_centers = []
    authority_audit = {
        "policy": "legacy_semantic_connected_components",
        "observation_quality": "semantic_connected_component",
        "authority_sha256": None,
    }
    if source_instance_authority is not None:
        for item in source_instance_authority.get("instances") or []:
            raw_type = int(item["raw_class_id"])
            row = int(np.clip(round(float(item["row"])), 0, shape[0] - 1))
            col = int(np.clip(round(float(item["col"])), 0, shape[1] - 1))
            source_centers.append((raw_type, row, col))
        authority_audit = {
            "policy": "joint_scene_instance_authority",
            "observation_quality": source_instance_authority.get(
                "observation_quality"
            ),
            "authority_sha256": source_instance_authority.get(
                "authority_sha256"
            ),
        }
    else:
        for class_value in np.unique(reference_nuclei_raw):
            raw_type = int(class_value)
            if raw_type == 0:
                continue
            labeled, count = ndimage.label(
                reference_nuclei_raw == class_value,
                structure=np.ones((3, 3), dtype=np.uint8),
            )
            if count == 0:
                continue
            centers = ndimage.center_of_mass(
                reference_nuclei_raw == class_value,
                labeled,
                range(1, count + 1),
            )
            for center_y, center_x in centers:
                row = int(np.clip(round(center_y), 0, shape[0] - 1))
                col = int(np.clip(round(center_x), 0, shape[1] - 1))
                source_centers.append((raw_type, row, col))

    density_scales = {}
    type_proportions = {}
    tissue_audit = {}
    for tissue_id_value in np.unique(target_tissue[generation_region]):
        tissue_id = int(tissue_id_value)
        if tissue_id == 0:
            continue
        reference_ids = frozenset({tissue_id})
        observed_region = reference_tissue == tissue_id
        observed_area = int(np.count_nonzero(observed_region))
        local_type_counts = {}
        for raw_type, row, col in source_centers:
            if not observed_region[row, col]:
                continue
            local_type_counts[raw_type] = (
                local_type_counts.get(raw_type, 0) + 1
            )
        local_count = int(sum(local_type_counts.values()))
        dataset_density = float(library.get_density(tissue_id))
        local_density = (
            10000.0 * local_count / observed_area
            if observed_area > 0
            else dataset_density
        )
        raw_area_confidence = float(observed_area / patch_area)
        local_is_reliable = bool(
            observed_area >= int(local_density_direct_min_area)
            and local_count >= int(local_density_direct_min_count)
        )
        calibration_family_ids = density_calibration_family_ids(
            dataset_name,
            tissue_id,
        )
        family_calibration = None
        family_audit = None
        if not local_is_reliable and calibration_family_ids:
            family_region = np.isin(
                reference_tissue,
                tuple(calibration_family_ids),
            )
            family_area = int(np.count_nonzero(family_region))
            family_count = int(
                sum(
                    1
                    for _, row, col in source_centers
                    if family_region[row, col]
                )
            )
            family_expected_dataset_count = float(
                sum(
                    np.count_nonzero(reference_tissue == family_tissue_id)
                    * float(library.get_density(int(family_tissue_id)))
                    / 10000.0
                    for family_tissue_id in calibration_family_ids
                )
            )
            family_is_reliable = bool(
                family_area >= int(local_density_direct_min_area)
                and family_count >= int(local_density_direct_min_count)
                and family_expected_dataset_count > 0
            )
            if family_is_reliable:
                family_calibration = float(
                    family_count / family_expected_dataset_count
                )
            family_audit = {
                "source_tissue_ids": sorted(calibration_family_ids),
                "source_area_px": family_area,
                "source_centroid_count": family_count,
                "dataset_expected_count_on_source_area": (
                    family_expected_dataset_count
                ),
                "reliable": family_is_reliable,
                "scale": family_calibration,
            }

        if local_is_reliable:
            effective_confidence = 1.0
            target_density = local_density
            density_mode = "pre_edit_patch_local_direct_reliable"
        elif family_calibration is not None:
            effective_confidence = 0.0
            target_density = dataset_density * family_calibration
            density_mode = (
                "target_dataset_prior_times_pre_edit_family_calibration"
            )
        else:
            effective_confidence = raw_area_confidence
            target_density = (
                effective_confidence * local_density
                + (1.0 - effective_confidence) * dataset_density
            )
            density_mode = (
                "pre_edit_sparse_area_confidence_dataset_shrinkage"
            )
        density_scale = (
            float(global_density_scale) * target_density / dataset_density
            if dataset_density > 0
            else float(global_density_scale)
        )
        density_scales[tissue_id] = float(density_scale)

        if local_is_reliable and local_count > 0:
            selected_type_counts = local_type_counts
            type_source = "patch_local_reliable"
        else:
            dataset_type_distribution = library.get_type_distribution(tissue_id)
            if dataset_type_distribution:
                type_proportions[tissue_id] = {
                    int(key): float(value)
                    for key, value in dataset_type_distribution.items()
                    if float(value) > 0
                }
                selected_type_counts = None
                type_source = "dataset_tissue_prior"
            elif local_type_counts:
                selected_type_counts = local_type_counts
                type_source = "patch_local_sparse_fallback"
            else:
                selected_type_counts = {int(NUCLEI_CLASSES[0]): 1}
                type_source = "default_neoplastic_fallback"
        if selected_type_counts is not None:
            selected_total = float(sum(selected_type_counts.values()))
            type_proportions[tissue_id] = {
                int(key): float(value / selected_total)
                for key, value in selected_type_counts.items()
                if value > 0
            }

        tissue_audit[str(tissue_id)] = {
            "reference_area_px": observed_area,
            "unedited_reference_area_px": observed_area,
            "density_reference_image": "pre_edit_source_patch",
            "density_reference_tissue_ids": sorted(reference_ids),
            "density_reference_deletion_exclusion_applied": False,
            "dataset_prior_calibration_from_pre_edit_source": family_audit,
            "patch_area_px": patch_area,
            "raw_area_confidence": raw_area_confidence,
            "effective_local_confidence": effective_confidence,
            "density_mode": density_mode,
            "local_reliability_min_area_px": int(local_density_direct_min_area),
            "local_reliability_min_count": int(local_density_direct_min_count),
            "local_centroid_count": local_count,
            "local_centroid_count_by_type": {
                str(key): int(value) for key, value in local_type_counts.items()
            },
            "local_density_per_10k_px": local_density,
            "dataset_density_per_10k_px": dataset_density,
            "target_density_per_10k_px": target_density,
            "effective_density_scale_vs_dataset": float(density_scale),
            "type_prior_source": type_source,
            "type_proportions": {
                str(key): float(value)
                for key, value in type_proportions[tissue_id].items()
            },
            "target_generation_area_px": int(
                np.count_nonzero(generation_region & (target_tissue == tissue_id))
            ),
        }

    audit = {
        "instance_authority": authority_audit,
        "checkpoint_role": "spatial_placement_probability_only",
        "count_policy": COUNT_POLICY_NAME,
        "type_policy": "reliable patch-local quota else dataset tissue prior",
        "nucleus_count_rule": (
            "instance_authority_centroid_in_pre_edit_source_tissue_family"
        ),
        "density_exclusion_region_role": (
            "cell_erasure_only_not_source_density_estimation"
        ),
        "tissues": tissue_audit,
    }
    return density_scales, type_proportions, audit


def calibrated_local_type_distribution(
    type_probs,
    *,
    supported_types=None,
    tissue_type_prior=None,
    prior_weight=0.5,
    prior_floor=1e-4,
):
    """Log-pool local ProbNet evidence with a generic target-tissue prior."""

    type_probs = np.asarray(type_probs, dtype=np.float64).copy()
    if type_probs.shape != (len(NUCLEI_CLASSES),):
        raise ValueError("type_probs must contain one value per nucleus class")
    supported_mask = np.ones(len(NUCLEI_CLASSES), dtype=bool)
    if supported_types is not None:
        supported = {int(value) for value in supported_types}
        supported_mask = np.asarray(
            [
                int(nucleus_type) in supported
                for nucleus_type in NUCLEI_CLASSES
            ],
            dtype=bool,
        )
        type_probs[~supported_mask] = 0.0
    type_probs = np.clip(type_probs, 0.0, None)
    total = type_probs.sum()
    if not np.isfinite(total) or total <= 0:
        return None
    type_probs = type_probs / total

    weight = float(np.clip(prior_weight, 0.0, 1.0))
    if not tissue_type_prior or weight <= 0.0:
        return type_probs
    prior = np.asarray(
        [
            float(
                tissue_type_prior.get(
                    int(nucleus_type),
                    tissue_type_prior.get(str(int(nucleus_type)), 0.0),
                )
            )
            for nucleus_type in NUCLEI_CLASSES
        ],
        dtype=np.float64,
    )
    prior = np.clip(prior, 0.0, None)
    prior[~supported_mask] = 0.0
    if not np.isfinite(prior).all() or prior.sum() <= 0:
        return type_probs
    floor = max(float(prior_floor), 0.0)
    prior[supported_mask] = np.maximum(prior[supported_mask], floor)
    prior = prior / prior.sum()

    if weight >= 1.0:
        return prior
    log_pool = (
        (1.0 - weight) * np.log(np.maximum(type_probs, 1e-12))
        + weight * np.log(np.maximum(prior, 1e-12))
    )
    log_pool[~supported_mask] = -np.inf
    log_pool = log_pool - float(np.max(log_pool[supported_mask]))
    pooled = np.exp(log_pool)
    pooled[~supported_mask] = 0.0
    pooled_total = pooled.sum()
    if not np.isfinite(pooled_total) or pooled_total <= 0:
        return type_probs
    return pooled / pooled_total


def confidence_adaptive_type_prior_weights(
    prior_audit,
    *,
    maximum_weight=DEFAULT_LOCAL_TYPE_PRIOR_WEIGHT,
):
    """Trust the tissue prior only where source-patch type support is weak."""

    maximum = float(np.clip(maximum_weight, 0.0, 1.0))
    weights = {}
    for tissue_id, info in (prior_audit.get("tissues") or {}).items():
        confidence = float(
            np.clip(info.get("effective_local_confidence", 0.0), 0.0, 1.0)
        )
        weights[int(tissue_id)] = maximum * (1.0 - confidence)
    return weights


def sample_type_at_center(
    type_prob,
    cy,
    cx,
    args,
    *,
    supported_types=None,
    tissue_type_prior=None,
):
    """Sample a locally informed type with target-tissue prior calibration."""

    probabilities = np.asarray(type_prob, dtype=np.float64)
    if probabilities.shape[0] == len(NUCLEI_CLASSES) + 1:
        probabilities = probabilities[1:]
    if probabilities.shape[0] != len(NUCLEI_CLASSES):
        raise ValueError("type_prob must contain one channel per nucleus class")
    type_probs = calibrated_local_type_distribution(
        probabilities[:, cy, cx],
        supported_types=supported_types,
        tissue_type_prior=tissue_type_prior,
        prior_weight=float(
            getattr(
                args,
                "local_type_prior_weight",
                DEFAULT_LOCAL_TYPE_PRIOR_WEIGHT,
            )
        ),
        prior_floor=float(
            getattr(args, "local_type_prior_floor", 1e-4)
        ),
    )
    if type_probs is None:
        return None
    idx = int(np.random.choice(len(type_probs), p=type_probs))
    return NUCLEI_CLASSES[idx]


def select_low_variance_type(
    type_probs,
    *,
    placed_by_type,
    expected_type_mass,
):
    """Track cumulative posterior mass while keeping integer counts stable."""

    probabilities = np.asarray(type_probs, dtype=np.float64)
    if probabilities.shape != (len(NUCLEI_CLASSES),):
        raise ValueError("type_probs must contain one value per nucleus class")
    total = probabilities.sum()
    if not np.isfinite(probabilities).all() or total <= 0:
        return None
    probabilities = probabilities / total
    expected = np.asarray(expected_type_mass, dtype=np.float64)
    if expected.shape != probabilities.shape:
        raise ValueError(
            "expected_type_mass must contain one value per nucleus class"
        )
    assigned = np.asarray(
        [
            int(placed_by_type.get(int(nucleus_type), 0))
            for nucleus_type in NUCLEI_CLASSES
        ],
        dtype=np.float64,
    )
    projected_deficit = expected + probabilities - assigned
    # Probability is a deterministic tie-breaker; the cumulative deficit is
    # the primary signal and keeps realized small-sample counts near the
    # aggregate local posterior rather than relying on a lucky random draw.
    score = projected_deficit + probabilities * 1e-9
    return int(NUCLEI_CLASSES[int(np.argmax(score))])


def balanced_type_at_center(
    type_prob,
    cy,
    cx,
    args,
    *,
    placed_by_type,
    expected_type_mass,
    supported_types=None,
    tissue_type_prior=None,
    prior_weight=None,
):
    """Choose a local type and return the posterior mass to commit on success."""

    ordered, type_probs = balanced_type_order_at_center(
        type_prob,
        cy,
        cx,
        args,
        placed_by_type=placed_by_type,
        expected_type_mass=expected_type_mass,
        supported_types=supported_types,
        tissue_type_prior=tissue_type_prior,
        prior_weight=prior_weight,
    )
    return (ordered[0] if ordered else None), type_probs


def balanced_type_order_at_center(
    type_prob,
    cy,
    cx,
    args,
    *,
    placed_by_type,
    expected_type_mass,
    supported_types=None,
    tissue_type_prior=None,
    prior_weight=None,
):
    """Rank legal types by the same cumulative-posterior balancing policy.

    The first item is identical to :func:`balanced_type_at_center`.  Remaining
    items are used only by the exact-count backfill when every complete shape
    of the preferred type is geometrically unplaceable at a center.  This
    preserves ProbNet's total cellularity target without shrinking shapes,
    allowing overlap, or inventing an organ-specific type quota.
    """

    probabilities = np.asarray(type_prob, dtype=np.float64)
    if probabilities.shape[0] == len(NUCLEI_CLASSES) + 1:
        probabilities = probabilities[1:]
    if probabilities.shape[0] != len(NUCLEI_CLASSES):
        raise ValueError("type_prob must contain one channel per nucleus class")
    type_probs = calibrated_local_type_distribution(
        probabilities[:, cy, cx],
        supported_types=supported_types,
        tissue_type_prior=tissue_type_prior,
        prior_weight=(
            float(prior_weight)
            if prior_weight is not None
            else float(
                getattr(
                    args,
                    "local_type_prior_weight",
                    DEFAULT_LOCAL_TYPE_PRIOR_WEIGHT,
                )
            )
        ),
        prior_floor=float(
            getattr(args, "local_type_prior_floor", 1e-4)
        ),
    )
    if type_probs is None:
        return (), None
    assigned = np.asarray(
        [
            int(placed_by_type.get(int(nucleus_type), 0))
            for nucleus_type in NUCLEI_CLASSES
        ],
        dtype=np.float64,
    )
    expected = np.asarray(expected_type_mass, dtype=np.float64)
    score = expected + type_probs - assigned + type_probs * 1e-9
    supported = (
        {int(value) for value in supported_types}
        if supported_types is not None
        else {int(value) for value in NUCLEI_CLASSES}
    )
    order = tuple(
        int(NUCLEI_CLASSES[index])
        for index in sorted(
            range(len(NUCLEI_CLASSES)),
            key=lambda index: (-float(score[index]), -float(type_probs[index]), index),
        )
        if int(NUCLEI_CLASSES[index]) in supported and type_probs[index] > 0
    )
    return order, type_probs


def supported_joint_nucleus_probability(type_prob, supported_types):
    """Marginalize ProbNet joint intensity over realizable nucleus classes."""

    probabilities = np.asarray(type_prob, dtype=np.float64)
    if probabilities.shape[0] != len(NUCLEI_CLASSES):
        raise ValueError("type_prob must contain one channel per nucleus class")
    supported = {int(value) for value in supported_types}
    indices = [
        index
        for index, nucleus_type in enumerate(NUCLEI_CLASSES)
        if int(nucleus_type) in supported
    ]
    if not indices:
        return np.zeros(probabilities.shape[1:], dtype=np.float32)
    return np.clip(
        np.sum(probabilities[indices], axis=0),
        1e-12,
        1.0,
    ).astype(np.float32)


def build_reference_pool(nuclei_mask, args):
    if nuclei_mask is None:
        return None
    return ReferenceNucleiInstancePool.from_mask(
        nuclei_mask,
        min_area=args.reference_shape_min_area,
        exclude_border=not args.include_border_reference_shapes,
        max_area_ratio_to_median=args.reference_shape_max_area_ratio,
    )


def sample_instance_for_center(
    sampler,
    tissue_id,
    nuc_type,
    *,
    force_tissue_library=False,
):
    if force_tissue_library:
        return sampler.sample_library_instance(
            tissue_id,
            nuc_type,
            allow_cross_tissue=False,
            requested_type=nuc_type,
            # A newly introduced tissue compartment must use the matching
            # tissue library for morphology, but its scale still belongs to
            # the observed patch.  Disabling calibration here made melanoma
            # library nuclei several times smaller than adjacent native tumor
            # nuclei even though complete same-class references were present.
            calibrate_size=True,
        )
    instance, source = sampler.sample_instance(
        tissue_id,
        nuc_type,
        allow_cross_tissue=(tissue_id != 3),
    )
    if instance is None and tissue_id == 3:
        instance, source = sampler.sample_library_instance(
            tissue_id,
            104,
            allow_cross_tissue=True,
            requested_type=nuc_type,
        )
    if instance is None and tissue_id != 3:
        instance, source = sampler.sample_library_instance(
            tissue_id,
            allow_cross_tissue=False,
            requested_type=nuc_type,
        )
    return instance, source


def place_candidate_with_retries(
    *,
    output,
    candidate_y,
    candidate_x,
    nucleus_type,
    tissue_id,
    shape_sampler,
    center_region,
    valid_tissue_mask,
    dense_retry,
    force_tissue_library,
    args,
    placement_audit=None,
):
    """Try alternate same-class shapes and transforms at one candidate center."""

    shape_trials = int(
        args.dense_placement_shape_trials
        if dense_retry
        else args.placement_shape_trials
    )
    transform_trials = int(
        args.dense_placement_transform_trials
        if dense_retry
        else args.placement_transform_trials
    )
    attempts = 0
    quarantined_reference_shapes = []
    try:
        for _ in range(max(1, shape_trials)):
            instance, shape_source = sample_instance_for_center(
                shape_sampler,
                tissue_id,
                nucleus_type,
                force_tissue_library=force_tissue_library,
            )
            if instance is None:
                break
            for spec in retry_transform_specs(
                args,
                trial_count=transform_trials,
            ):
                offset_y, offset_x = spec["offset_yx"]
                center_y = int(candidate_y + offset_y)
                center_x = int(candidate_x + offset_x)
                attempts += 1
                if (
                    center_y < 0
                    or center_y >= center_region.shape[0]
                    or center_x < 0
                    or center_x >= center_region.shape[1]
                    or not bool(center_region[center_y, center_x])
                ):
                    continue
                current_placement = {}
                placed = place_nucleus_layered(
                    output,
                    center_y,
                    center_x,
                    instance,
                    augment=not args.no_augment_instances,
                    max_overlap_fraction=float(
                        args.max_nucleus_overlap_fraction
                    ),
                    valid_tissue_mask=valid_tissue_mask,
                    require_full_tissue_containment=bool(
                        args.require_full_tissue_containment
                    ),
                    rotation_quarters=int(spec["rotation_quarters"]),
                    flip_horizontal=bool(spec["flip_horizontal"]),
                    flip_vertical=bool(spec["flip_vertical"]),
                    scale=float(spec["scale"]),
                    minimum_separation_px=int(
                        getattr(args, "nucleus_spacing_margin_px", 1)
                    ),
                    placement_metadata=current_placement,
                )
                if placed:
                    if placement_audit is not None:
                        placement_audit.update(current_placement)
                    return (
                        True,
                        str(shape_source),
                        attempts,
                        (center_y, center_x),
                    )
            if str(shape_source) == "reference":
                # Do not immediately resample the same unplaceable reference
                # shape at this center. Restore it after alternate shapes and
                # the library fallback have had a chance.
                quarantined_reference_shapes.append(instance)
    finally:
        for instance in quarantined_reference_shapes:
            shape_sampler.release_failed_instance(instance, "reference")
    return False, None, attempts, None


def center_region_for_nucleus_type(
    center_region,
    nucleus_type,
    center_region_exclusions_by_type=None,
):
    """Return the executable center domain for one typed population."""

    region = np.asarray(center_region, dtype=bool)
    exclusions = center_region_exclusions_by_type or {}
    excluded = exclusions.get(int(nucleus_type))
    if excluded is None:
        excluded = exclusions.get(str(int(nucleus_type)))
    if excluded is None:
        return region
    excluded = np.asarray(excluded, dtype=bool)
    if excluded.shape != region.shape:
        raise ValueError("typed center exclusion and center region must share one shape")
    return region & ~excluded


def place_candidate_with_type_fallback(
    *,
    nucleus_types,
    placement_audit,
    center_region_exclusions_by_type=None,
    **placement_kwargs,
):
    """Try posterior-ordered legal types without relaxing shape geometry."""

    total_trials = 0
    for rank, nucleus_type in enumerate(nucleus_types):
        current_audit = {}
        typed_kwargs = dict(placement_kwargs)
        if "center_region" in placement_kwargs:
            typed_kwargs["center_region"] = center_region_for_nucleus_type(
                placement_kwargs["center_region"],
                nucleus_type,
                center_region_exclusions_by_type,
            )
        placed, shape_source, trials, accepted_center = (
            place_candidate_with_retries(
                nucleus_type=int(nucleus_type),
                placement_audit=current_audit,
                **typed_kwargs,
            )
        )
        total_trials += int(trials)
        if not placed:
            continue
        placement_audit.update(current_audit)
        placement_audit["type_fallback_rank"] = int(rank)
        placement_audit["preferred_nucleus_type"] = int(nucleus_types[0])
        placement_audit["accepted_nucleus_type"] = int(nucleus_type)
        return (
            True,
            int(nucleus_type),
            shape_source,
            total_trials,
            accepted_center,
        )
    return False, None, None, total_trials, None


def realize_compiled_packing_witness(
    *,
    base_output,
    packing_witness,
    center_region,
    valid_tissue_mask,
    tissue_id,
    target_count,
    nucleus_probability,
    minimum_separation_px,
    allowed_nucleus_types=None,
    center_region_exclusions_by_type=None,
):
    """Execute a certified source-shape packing when ranked search exhausts.

    The feasibility compiler has already proven that the complete witness set
    fits E/P/V with the configured separation.  The mature executor normally
    searches independently in ProbNet order; this fallback consumes the same
    immutable witness, ordering eligible placements by ProbNet probability,
    so a certified candidate cannot later fail merely because the stochastic
    search chose a blocking prefix.
    """

    requested = max(0, int(target_count))
    if not packing_witness or requested <= 0:
        return None
    output = np.asarray(base_output).copy()
    centers = np.asarray(center_region, dtype=bool)
    valid = np.asarray(valid_tissue_mask, dtype=bool)
    probability = np.asarray(nucleus_probability, dtype=np.float64)
    allowed_types = {
        int(value)
        for value in (allowed_nucleus_types or NUCLEI_CLASSES)
    }
    height, width = output.shape
    eligible = []
    for item in packing_witness.get("placements") or []:
        row, col = int(item["row"]), int(item["col"])
        if int(item["nucleus_type"]) not in allowed_types:
            continue
        typed_region = center_region_for_nucleus_type(
            centers,
            int(item["nucleus_type"]),
            center_region_exclusions_by_type,
        )
        if (
            row < 0
            or row >= height
            or col < 0
            or col >= width
            or not typed_region[row, col]
        ):
            continue
        eligible.append(item)
    eligible.sort(
        key=lambda item: (
            -float(probability[int(item["row"]), int(item["col"])]),
            int(item["row"]),
            int(item["col"]),
            str(item.get("reference_instance_id", "")),
        )
    )
    accepted = []
    separation = max(0, int(minimum_separation_px))
    for item in eligible:
        if len(accepted) >= requested:
            break
        row, col = int(item["row"]), int(item["col"])
        nucleus_type = int(item["nucleus_type"])
        class_index = nucleus_type - 100
        if class_index not in range(1, len(NUCLEI_CLASSES) + 1):
            continue
        offsets = np.asarray(item.get("offsets_yx") or [], dtype=np.int64)
        if offsets.ndim != 2 or offsets.shape[1:] != (2,) or not len(offsets):
            continue
        rows = row + offsets[:, 0]
        cols = col + offsets[:, 1]
        if (
            np.any(rows < 0)
            or np.any(rows >= height)
            or np.any(cols < 0)
            or np.any(cols >= width)
            or not np.all(valid[rows, cols])
        ):
            continue
        footprint = np.zeros_like(centers, dtype=bool)
        footprint[rows, cols] = True
        guard = (
            ndimage.binary_dilation(
                footprint,
                structure=np.ones((3, 3), dtype=bool),
                iterations=separation,
            )
            if separation > 0
            else footprint
        )
        if np.any(guard & (output > 0)):
            continue
        output[rows, cols] = class_index
        accepted.append(
            {
                "row": row,
                "col": col,
                "nucleus_type": nucleus_type,
                "tissue_id": int(tissue_id),
                "area_px": len(rows),
                "shape_source": "compiled_reference_witness",
                "reference_instance_id": str(
                    item.get("reference_instance_id", "")
                ),
            }
        )
    if len(accepted) != requested:
        return None
    return output, accepted


def packing_witness_shape_distribution_audit(
    accepted_centers,
    packing_witness,
):
    """Check realized class medians against the contract's patch references."""

    medians = {
        int(key): float(value)
        for key, value in (
            (packing_witness or {}).get(
                "class_reference_median_area_px", {}
            )
            or {}
        ).items()
        if float(value) > 0
    }
    interval = (packing_witness or {}).get(
        "local_median_area_ratio_interval", [0.60, 1.67]
    )
    if len(interval) != 2:
        interval = [0.60, 1.67]
    minimum, maximum = float(interval[0]), float(interval[1])
    areas_by_type = {}
    for item in accepted_centers or ():
        nucleus_type = int(item.get("nucleus_type", 0))
        area = float(item.get("area_px", 0))
        if nucleus_type in medians and area > 0:
            areas_by_type.setdefault(nucleus_type, []).append(area)
    metrics = {}
    passed = True
    for nucleus_type, areas in sorted(areas_by_type.items()):
        ratio = float(np.median(areas)) / medians[nucleus_type]
        current = minimum <= ratio <= maximum
        passed = passed and current
        metrics[str(nucleus_type)] = {
            "accepted_count": len(areas),
            "accepted_median_area_px": float(np.median(areas)),
            "reference_median_area_px": medians[nucleus_type],
            "median_area_ratio": ratio,
            "passed": current,
        }
    return {
        "passed": bool(passed and (not accepted_centers or bool(metrics))),
        "ratio_interval": [minimum, maximum],
        "class_metrics": metrics,
    }


def generate_for_gamma(
    prob,
    tissue,
    input_nuclei,
    edit_mask,
    library,
    reference_pool,
    gamma,
    args,
    density_scales,
    density=None,
    type_density=None,
    library_only_tissue_ids=None,
    clear_edit_mask=True,
    type_proportions_by_tissue=None,
    type_prior_weights_by_tissue=None,
    placement_nuc_prob=None,
    placement_type_prob=None,
    retained_by_type_overrides=None,
    population_mask=None,
    new_target_count_overrides=None,
    size_reference_pools_by_tissue=None,
    fallback_size_reference_pool=None,
    allowed_nucleus_types_override=None,
    packing_witness=None,
    center_region_exclusions_by_type=None,
):
    nuc_prob = (
        np.asarray(placement_nuc_prob, dtype=np.float32)
        if placement_nuc_prob is not None
        else 1.0 - prob[0]
    )
    if nuc_prob.shape != tissue.shape:
        raise ValueError("placement probability and tissue must share one shape")
    stabilized_type_prob = (
        np.asarray(placement_type_prob, dtype=np.float32)
        if placement_type_prob is not None
        else np.asarray(prob[1:], dtype=np.float32)
    )
    if stabilized_type_prob.shape != (
        len(NUCLEI_CLASSES),
        *tissue.shape,
    ):
        raise ValueError(
            "placement type probability must be [num_types, height, width]"
        )
    output = input_nuclei.copy()
    if clear_edit_mask:
        output[edit_mask] = 0
    valid_tissue_mask = valid_biological_tissue_mask(
        tissue,
        args.skip_tissue_ids,
    )
    shape_sampler = ReferenceFirstNucleiSampler(
        library,
        reference_pool,
        calibrate_library_size=not args.disable_library_size_calibration,
        library_size_min_scale=args.library_size_min_scale,
        library_size_max_scale=args.library_size_max_scale,
        library_size_log_area_jitter=args.library_size_log_area_jitter,
    )
    library_only_tissue_ids = {
        int(tissue_id)
        for tissue_id in (library_only_tissue_ids or ())
    }
    size_reference_pools_by_tissue = dict(
        size_reference_pools_by_tissue or {}
    )

    diagnostics = {
        "gamma": gamma,
        "max_nucleus_overlap_fraction": float(
            getattr(args, "max_nucleus_overlap_fraction", 0.0)
        ),
        "require_full_tissue_containment": bool(
            getattr(args, "require_full_tissue_containment", True)
        ),
        "full_shape_tissue_policy": (
            "hard_reject_outside_current_target_tissue_then_retry"
        ),
        "nucleus_spacing_margin_px": int(
            getattr(args, "nucleus_spacing_margin_px", 1)
        ),
        "instance_connectivity_policy": (
            "largest_8_connected_component_after_transform"
        ),
        "type_quota_routing_policy": TYPE_QUOTA_ROUTING_POLICY_NAME,
        "placed": 0,
        "placed_by_shape_source": {"reference": 0, "library": 0},
        "reference_pool": reference_pool.describe() if reference_pool is not None else None,
        "tissues": {},
    }
    component_shape_sampling = {}
    component_policy_active = bool(
        density is None and getattr(args, "component_aware_sampling", False)
    )
    population_mask = (
        np.asarray(edit_mask, dtype=bool)
        if population_mask is None
        else np.asarray(population_mask, dtype=bool)
    )
    if population_mask.shape != tissue.shape:
        raise ValueError("population and placement regions must share one shape")
    typed_center_exclusions = {
        int(key): np.asarray(value, dtype=bool)
        for key, value in (center_region_exclusions_by_type or {}).items()
    }
    if any(value.shape != tissue.shape for value in typed_center_exclusions.values()):
        raise ValueError("typed center exclusions and tissue must share one shape")

    for tissue_id in np.unique(tissue[population_mask]):
        tissue_id = int(tissue_id)
        if tissue_id in args.skip_tissue_ids:
            continue

        population_region = population_mask & (tissue == tissue_id)
        tissue_region = edit_mask & (tissue == tissue_id)
        output_before_tissue = output.copy()
        same_tissue_footprint_mask = valid_tissue_mask & (tissue == tissue_id)
        if (
            population_region.sum() < args.min_region_area
            or not np.any(tissue_region)
        ):
            continue

        expected_area = weighted_mean_area(library, tissue_id, args.expected_nucleus_area)
        scale = density_scales.get(tissue_id, args.density_scale)
        class_counts = None
        if density is None:
            target_count, count_info = compute_target_count(
                nuc_prob,
                population_region,
                tissue_id,
                library,
                expected_area,
                args,
                scale,
            )
        else:
            expected_by_class = density[:, population_region].sum(axis=1) * scale
            expected_total = float(expected_by_class.sum())
            max_allowed = (
                args.max_density_per_10k
                * population_region.sum()
                / 10000.0
            )
            target_count = round(
                float(np.clip(expected_total, args.min_count, max_allowed))
            )
            if expected_total > 0 and target_count > 0:
                quotas = expected_by_class / expected_total * target_count
                class_counts = np.floor(quotas).astype(np.int64)
                remainder = target_count - int(class_counts.sum())
                if remainder > 0:
                    order = np.argsort(-(quotas - class_counts))
                    class_counts[order[:remainder]] += 1
            else:
                class_counts = np.zeros(density.shape[0], dtype=np.int64)
            count_info = {
                "region_area": int(population_region.sum()),
                "count_source": "center_density_integral",
                "density_scale": float(scale),
                "expected_count": expected_total,
                "expected_by_class": expected_by_class.tolist(),
                "target_by_class": class_counts.tolist(),
                "clipped_count": target_count,
            }

        retained_override = (retained_by_type_overrides or {}).get(
            str(tissue_id)
        )
        if retained_override is None:
            retained_override = (retained_by_type_overrides or {}).get(
                tissue_id
            )
        if retained_override is None:
            retained_by_type = count_retained_centers_by_type(
                input_nuclei,
                population_region,
            )
            retained_count_source = "raster_connected_components"
        else:
            retained_by_type = {
                int(nuc_type): int(count)
                for nuc_type, count in retained_override.items()
                if int(count) > 0
            }
            retained_count_source = (
                "original_retained_plus_exact_core_placement_ledger"
            )
        retained_count = int(sum(retained_by_type.values()))
        expected_target_count = int(target_count)
        if class_counts is not None:
            retained_internal = np.asarray(
                [
                    retained_by_type.get(int(nuc_type), 0)
                    for nuc_type in NUCLEI_CLASSES
                ],
                dtype=np.int64,
            )
            class_counts = np.maximum(class_counts - retained_internal, 0)
            target_count = int(class_counts.sum())
        else:
            target_count = max(0, expected_target_count - retained_count)
        explicit_new_target = (new_target_count_overrides or {}).get(
            str(tissue_id)
        )
        if explicit_new_target is None:
            explicit_new_target = (new_target_count_overrides or {}).get(
                tissue_id
            )
        if explicit_new_target is not None:
            if density is not None:
                raise ValueError(
                    "explicit new-target count overrides require the mature "
                    "count-prior path, not a supplied density field"
                )
            target_count = max(0, int(explicit_new_target))
            expected_target_count = retained_count + target_count
        count_info.update(
            {
                "expected_total_count_in_generation_region": expected_target_count,
                "retained_centroid_count_in_generation_region": retained_count,
                "retained_centroid_count_source": retained_count_source,
                "retained_centroid_count_by_type": {
                    str(key): int(value)
                    for key, value in retained_by_type.items()
                    if int(value) > 0
                },
                "new_target_count": int(target_count),
                "new_count_policy": (
                    "target_tissue_expected_total_minus_retained_centroids"
                ),
                "population_target_area_px": int(
                    np.count_nonzero(population_region)
                ),
                "placement_center_domain_pixels": int(
                    np.count_nonzero(tissue_region)
                ),
                "population_placement_role_separation": (
                    "count_from_population_target_area_place_centers_only_in_P"
                ),
                "explicit_new_target_count_override": (
                    int(target_count)
                    if explicit_new_target is not None
                    else None
                ),
            }
        )
        supported_types = supported_nucleus_shape_types(
            library,
            reference_pool,
            tissue_id,
            force_tissue_library=(
                tissue_id in library_only_tissue_ids
            ),
        )
        allowed_nucleus_types = {
            int(value)
            for value in (
                allowed_nucleus_types_override
                or getattr(args, "allowed_nucleus_types", None)
                or NUCLEI_CLASSES
            )
        }
        unsupported_requested = allowed_nucleus_types - set(NUCLEI_CLASSES)
        if unsupported_requested:
            raise ValueError(
                "allowed nucleus types are outside the configured CellViT schema: "
                + ", ".join(map(str, sorted(unsupported_requested)))
            )
        supported_types &= allowed_nucleus_types
        if target_count > 0 and not supported_types:
            raise RuntimeError(
                "ProbNet requested nuclei for a tissue with no supported "
                f"same-class instance shapes: tissue_id={tissue_id}."
            )
        type_shape_support = {
            "policy": (
                "probnet_local_posterior_renormalized_to_available_"
                "same_class_shapes"
            ),
            "supported_types": sorted(int(value) for value in supported_types),
        }
        tissue_nuc_prob = supported_joint_nucleus_probability(
            stabilized_type_prob,
            supported_types,
        )
        tissue_sampling_mass = probability_sampling_mass(
            tissue_nuc_prob,
            gamma,
        )
        tissue_type_prior = (type_proportions_by_tissue or {}).get(tissue_id)
        if tissue_type_prior is None:
            tissue_type_prior = (type_proportions_by_tissue or {}).get(
                str(tissue_id)
            )
        local_type_prior_weight = (
            (type_prior_weights_by_tissue or {}).get(tissue_id)
        )
        if local_type_prior_weight is None:
            local_type_prior_weight = (
                (type_prior_weights_by_tissue or {}).get(str(tissue_id))
            )
        if local_type_prior_weight is None:
            local_type_prior_weight = float(
                getattr(
                    args,
                    "local_type_prior_weight",
                    DEFAULT_LOCAL_TYPE_PRIOR_WEIGHT,
                )
            )
        local_type_prior_weight = float(local_type_prior_weight)
        type_routing_audit = {
            "policy": TYPE_QUOTA_ROUTING_POLICY_NAME,
            "prior_role": "total_count_only",
            "density_head_role": "none",
            "tissue_type_prior_role": (
                "target_tissue_empirical_composition_log_pool"
                if tissue_type_prior
                else "unavailable"
            ),
            "tissue_type_prior_weight": local_type_prior_weight,
            "tissue_type_prior": (
                {
                    str(key): float(value)
                    for key, value in tissue_type_prior.items()
                }
                if tissue_type_prior
                else None
            ),
            "local_probnet_role": "position_and_local_type_evidence",
            "position_marginalization": (
                "sum_joint_intensity_over_supported_types"
            ),
            "legacy_type_proportions_ignored": False,
            "legacy_type_density_ignored": bool(type_density is not None),
        }
        oversample_factor = args.oversample_base * (1.0 + args.oversample_gamma_scale * max(gamma - 1.0, 0.0))
        oversample_factor = float(np.clip(oversample_factor, args.oversample_min, args.oversample_max))
        min_distance = adaptive_min_distance(expected_area, args, oversample_factor)
        component_mode = bool(
            density is None and getattr(args, "component_aware_sampling", False)
        )
        candidates = (
            []
            if component_mode
            else poisson_candidates(tissue_region, min_distance, args.poisson_attempts)
        )
        typed_centers = None
        center_component_ids = None
        component_labels = None
        component_limits = {0: target_count}
        component_dense_retry = {0: False}
        component_sampling = None
        global_sampling = None
        quota_coverage_audit = None
        component_shape_samplers = {}
        if density is None:
            if component_mode:
                component_labels, component_count = ndimage.label(
                    tissue_region,
                    structure=np.ones((3, 3), dtype=np.uint8),
                )
                component_areas = [
                    (component_id, int(np.count_nonzero(component_labels == component_id)))
                    for component_id in range(1, component_count + 1)
                ]
                # Tissue-level filtering already removes biologically empty
                # regions. Keep every non-empty disconnected component in the
                # ProbNet-mass allocation so prior controls only the total.
                component_weights = [
                    (
                        component_id,
                        float(
                            np.sum(
                                tissue_sampling_mass[
                                    component_labels == component_id
                                ]
                            )
                        ),
                    )
                    for component_id, _ in component_areas
                ]
                component_limits = allocate_weight_proportional_counts(
                    component_weights,
                    target_count,
                )
                component_mass_by_id = dict(component_weights)
                component_sampling = initialize_component_sampling_diagnostics(
                    component_areas,
                    component_limits,
                    component_mass_by_id,
                )
                centers = []
                center_component_ids = []
                candidates = []
                for component_id, area in component_areas:
                    quota = int(component_limits.get(component_id, 0))
                    component_region = component_labels == component_id
                    local_reference_pool = (
                        reference_pool.subset_by_center_region(component_region)
                        if reference_pool is not None
                        else None
                    )
                    tissue_size_reference_pool = (
                        size_reference_pools_by_tissue.get(tissue_id)
                    )
                    component_shape_samplers[component_id] = (
                        ReferenceFirstNucleiSampler(
                            library,
                            local_reference_pool,
                            size_reference_pool=tissue_size_reference_pool,
                            fallback_size_reference_pool=(
                                fallback_size_reference_pool
                                if fallback_size_reference_pool is not None
                                else reference_pool
                            ),
                            calibrate_library_size=(
                                not args.disable_library_size_calibration
                            ),
                            library_size_min_scale=args.library_size_min_scale,
                            library_size_max_scale=args.library_size_max_scale,
                            library_size_log_area_jitter=(
                                args.library_size_log_area_jitter
                            ),
                        )
                    )
                    if quota <= 0:
                        continue
                    component_candidates = poisson_candidates(
                        component_region,
                        min_distance,
                        args.poisson_attempts,
                    )
                    requested = quota
                    (
                        retry_pool_size,
                        dense_retry,
                        expected_occupancy,
                    ) = retry_pool_target(
                        quota=quota,
                        component_area=area,
                        expected_nucleus_area=expected_area,
                        args=args,
                    )
                    component_dense_retry[component_id] = dense_retry
                    if getattr(args, "backfill_failed_placements", False):
                        component_candidates = supplement_retry_candidates(
                            component_candidates,
                            component_region,
                            retry_pool_size,
                        )
                        requested = len(component_candidates)
                    (
                        coverage_count,
                        coverage_radius,
                        coverage_audit,
                    ) = compile_quota_coverage_contract(
                        component_candidates,
                        tissue_nuc_prob,
                        quota,
                        region_area=area,
                        candidate_min_distance=min_distance,
                        args=args,
                    )
                    selected = choose_weighted_centers(
                        component_candidates,
                        tissue_nuc_prob,
                        requested,
                        gamma,
                        coverage_count=coverage_count,
                        coverage_radius=coverage_radius,
                    )
                    candidates.extend(component_candidates)
                    centers.extend(selected)
                    center_component_ids.extend([component_id] * len(selected))
                    component_sampling[str(component_id)].update({
                        "dense_retry": dense_retry,
                        "expected_occupancy_fraction": expected_occupancy,
                        "retry_pool_target": retry_pool_size,
                        "num_candidates": len(component_candidates),
                        "selected_centers": len(selected),
                        "quota_coverage": coverage_audit,
                    })
            else:
                requested_centers = target_count
                (
                    retry_pool_size,
                    dense_retry,
                    expected_occupancy,
                ) = retry_pool_target(
                    quota=target_count,
                    component_area=int(np.count_nonzero(tissue_region)),
                    expected_nucleus_area=expected_area,
                    args=args,
                )
                component_dense_retry[0] = dense_retry
                if getattr(args, "backfill_failed_placements", False):
                    candidates = supplement_retry_candidates(
                        candidates,
                        tissue_region,
                        retry_pool_size,
                    )
                    requested_centers = len(candidates)
                (
                    coverage_count,
                    coverage_radius,
                    quota_coverage_audit,
                ) = compile_quota_coverage_contract(
                    candidates,
                    tissue_nuc_prob,
                    target_count,
                    region_area=int(np.count_nonzero(tissue_region)),
                    candidate_min_distance=min_distance,
                    args=args,
                )
                centers = choose_weighted_centers(
                    candidates,
                    tissue_nuc_prob,
                    requested_centers,
                    gamma,
                    coverage_count=coverage_count,
                    coverage_radius=coverage_radius,
                )
                center_component_ids = [0] * len(centers)
                global_sampling = {
                    "area": int(np.count_nonzero(tissue_region)),
                    "quota": int(target_count),
                    "integrated_probnet_mass": float(
                        np.sum(tissue_sampling_mass[tissue_region])
                    ),
                    "component_count_policy": (
                        "single_component_total_count"
                    ),
                    "quota_coverage": quota_coverage_audit,
                }
        else:
            available = list(candidates)
            typed_centers = []
            for class_index in np.argsort(-class_counts):
                requested = int(class_counts[class_index])
                if requested <= 0 or not available:
                    continue
                center_score = density[class_index] * (prob[class_index + 1] + 0.05)
                selected = choose_weighted_centers(
                    available,
                    center_score,
                    requested,
                    gamma,
                )
                selected_set = set(selected)
                available = [center for center in available if center not in selected_set]
                typed_centers.extend(
                    (center_y, center_x, NUCLEI_CLASSES[int(class_index)])
                    for center_y, center_x in selected
                )
            centers = [(center_y, center_x) for center_y, center_x, _ in typed_centers]
            center_component_ids = [0] * len(centers)

        placed = 0
        placed_by_shape_source = {"reference": 0, "library": 0}
        center_records = (
            [
                (
                    center_y,
                    center_x,
                    density_type,
                    component_id,
                    bool(component_dense_retry.get(component_id, False)),
                )
                for (center_y, center_x, density_type), component_id in zip(
                    typed_centers,
                    center_component_ids,
                )
            ]
            if typed_centers is not None
            else [
                (
                    cy,
                    cx,
                    None,
                    component_id,
                    bool(component_dense_retry.get(component_id, False)),
                )
                for (cy, cx), component_id in zip(centers, center_component_ids)
            ]
        )
        attempted = 0
        placement_trials = 0
        accepted_center_probabilities = []
        accepted_centers = []
        placed_by_component = {component_id: 0 for component_id in component_limits}
        placed_by_type = {
            int(nuc_type): 0
            for nuc_type in NUCLEI_CLASSES
        }
        expected_type_mass = np.zeros(
            len(NUCLEI_CLASSES),
            dtype=np.float64,
        )
        for cy, cx, density_type, component_id, dense_retry in center_records:
            if placed_by_component.get(component_id, 0) >= component_limits.get(
                component_id, target_count
            ):
                continue
            attempted += 1
            if component_sampling is not None:
                component_sampling[str(component_id)]["attempted_centers"] += 1
            if density_type is not None:
                nuc_type = density_type
                proposed_type_mass = None
            else:
                nuc_type, proposed_type_mass = balanced_type_at_center(
                    stabilized_type_prob,
                    cy,
                    cx,
                    args,
                    placed_by_type=placed_by_type,
                    expected_type_mass=expected_type_mass,
                    supported_types=supported_types,
                    tissue_type_prior=tissue_type_prior,
                    prior_weight=local_type_prior_weight,
                )
            if nuc_type is None:
                continue
            placement_audit = {}
            (
                placed_ok,
                shape_source,
                local_trials,
                accepted_center,
            ) = place_candidate_with_retries(
                output=output,
                candidate_y=cy,
                candidate_x=cx,
                nucleus_type=nuc_type,
                tissue_id=tissue_id,
                shape_sampler=component_shape_samplers.get(
                    component_id,
                    shape_sampler,
                ),
                center_region=center_region_for_nucleus_type(
                    tissue_region,
                    nuc_type,
                    typed_center_exclusions,
                ),
                valid_tissue_mask=same_tissue_footprint_mask,
                dense_retry=dense_retry,
                force_tissue_library=tissue_id in library_only_tissue_ids,
                args=args,
                placement_audit=placement_audit,
            )
            placement_trials += int(local_trials)
            if placed_ok:
                placed += 1
                accepted_y, accepted_x = accepted_center
                accepted_center_probabilities.append(
                    float(tissue_nuc_prob[accepted_y, accepted_x])
                )
                accepted_centers.append(
                    {
                        "row": int(accepted_y),
                        "col": int(accepted_x),
                        "nucleus_type": int(nuc_type),
                        "tissue_id": int(tissue_id),
                        "area_px": int(placement_audit["area_px"]),
                        "shape_source": str(shape_source),
                    }
                )
                placed_by_component[component_id] = (
                    placed_by_component.get(component_id, 0) + 1
                )
                if component_sampling is not None:
                    component_sampling[str(component_id)]["placed"] += 1
                if proposed_type_mass is not None:
                    expected_type_mass += proposed_type_mass
                placed_by_type[nuc_type] = placed_by_type.get(nuc_type, 0) + 1
                placed_by_shape_source[str(shape_source)] += 1

        exact_backfill = {
            "policy": "component_quota_probnet_mass_weighted_pixel_retry",
            "quota_reassignment_policy": (
                "unplaceable_component_quota_to_same_tissue_probnet_mass_tail"
            ),
            "candidate_budget_policy": (
                "quota_scaled_bounded_search_then_next_deterministic_seed"
            ),
            "triggered": bool(placed < target_count),
            "candidate_centers": 0,
            "attempted_centers": 0,
            "placement_trials": 0,
            "placed": 0,
            "placed_by_component": {},
            "component_candidate_budgets": {},
            "unfilled_component_quota_before_reassignment": {},
            "reassignment_candidate_centers": 0,
            "reassignment_candidate_budget": None,
            "reassignment_attempted_centers": 0,
            "reassignment_placement_trials": 0,
            "reassigned_placed": 0,
            "reassigned_placed_by_component": {},
            "type_fallback_placed": 0,
            "type_fallback_placed_by_type": {},
        }
        if (
            placed < target_count
            and getattr(args, "backfill_failed_placements", False)
        ):
            occupied = output > 0
            separation = max(
                0,
                int(getattr(args, "nucleus_spacing_margin_px", 1)),
            )
            if separation > 0:
                occupied = ndimage.binary_dilation(
                    occupied,
                    structure=np.ones(
                        (2 * separation + 1, 2 * separation + 1),
                        dtype=bool,
                    ),
                )
            for component_id, component_limit in sorted(
                component_limits.items()
            ):
                component_remaining = int(component_limit) - int(
                    placed_by_component.get(component_id, 0)
                )
                if component_remaining <= 0:
                    continue
                component_region = (
                    component_labels == component_id
                    if component_labels is not None
                    else tissue_region
                )
                fallback_region = component_region & ~occupied
                ranked_centers = probability_mass_region_centers(
                    fallback_region,
                    tissue_nuc_prob,
                    gamma,
                )
                exact_backfill["candidate_centers"] += int(
                    ranked_centers.shape[0]
                )
                component_budget, component_budget_audit = (
                    exact_backfill_candidate_budget(
                        component_remaining,
                        ranked_centers.shape[0],
                        args,
                    )
                )
                exact_backfill["component_candidate_budgets"][
                    str(component_id)
                ] = component_budget_audit
                component_backfill_placed = 0
                for cy, cx in ranked_centers[:component_budget]:
                    if component_backfill_placed >= component_remaining:
                        break
                    cy = int(cy)
                    cx = int(cx)
                    exact_backfill["attempted_centers"] += 1
                    attempted += 1
                    if component_sampling is not None:
                        component_sampling[str(component_id)][
                            "attempted_centers"
                        ] += 1
                    type_order, proposed_type_mass = balanced_type_order_at_center(
                        stabilized_type_prob,
                        cy,
                        cx,
                        args,
                        placed_by_type=placed_by_type,
                        expected_type_mass=expected_type_mass,
                        supported_types=supported_types,
                        tissue_type_prior=tissue_type_prior,
                        prior_weight=local_type_prior_weight,
                    )
                    if not type_order:
                        continue
                    placement_audit = {}
                    (
                        placed_ok,
                        nuc_type,
                        shape_source,
                        local_trials,
                        accepted_center,
                    ) = place_candidate_with_type_fallback(
                        nucleus_types=type_order,
                        output=output,
                        candidate_y=cy,
                        candidate_x=cx,
                        tissue_id=tissue_id,
                        shape_sampler=component_shape_samplers.get(
                            component_id,
                            shape_sampler,
                        ),
                        center_region=component_region,
                        valid_tissue_mask=same_tissue_footprint_mask,
                        dense_retry=True,
                        force_tissue_library=(
                            tissue_id in library_only_tissue_ids
                        ),
                        args=args,
                        placement_audit=placement_audit,
                        center_region_exclusions_by_type=(
                            typed_center_exclusions
                        ),
                    )
                    placement_trials += int(local_trials)
                    exact_backfill["placement_trials"] += int(local_trials)
                    if not placed_ok:
                        continue
                    placed += 1
                    component_backfill_placed += 1
                    exact_backfill["placed"] += 1
                    if int(placement_audit["type_fallback_rank"]) > 0:
                        exact_backfill["type_fallback_placed"] += 1
                        fallback_by_type = exact_backfill[
                            "type_fallback_placed_by_type"
                        ]
                        fallback_by_type[str(nuc_type)] = int(
                            fallback_by_type.get(str(nuc_type), 0)
                        ) + 1
                    accepted_y, accepted_x = accepted_center
                    accepted_center_probabilities.append(
                        float(tissue_nuc_prob[accepted_y, accepted_x])
                    )
                    accepted_centers.append(
                        {
                            "row": int(accepted_y),
                            "col": int(accepted_x),
                            "nucleus_type": int(nuc_type),
                            "tissue_id": int(tissue_id),
                            "area_px": int(placement_audit["area_px"]),
                            "shape_source": str(shape_source),
                        }
                    )
                    placed_by_component[component_id] = (
                        placed_by_component.get(component_id, 0) + 1
                    )
                    if component_sampling is not None:
                        component_sampling[str(component_id)]["placed"] += 1
                    if proposed_type_mass is not None:
                        expected_type_mass += proposed_type_mass
                    placed_by_type[nuc_type] = (
                        placed_by_type.get(nuc_type, 0) + 1
                    )
                    placed_by_shape_source[str(shape_source)] += 1
                exact_backfill["placed_by_component"][str(component_id)] = (
                    int(component_backfill_placed)
                )

            component_shortfalls = {
                str(component_id): max(
                    0,
                    int(component_limit)
                    - int(placed_by_component.get(component_id, 0)),
                )
                for component_id, component_limit in component_limits.items()
                if int(placed_by_component.get(component_id, 0))
                < int(component_limit)
            }
            exact_backfill[
                "unfilled_component_quota_before_reassignment"
            ] = component_shortfalls

            remaining = int(target_count) - int(placed)
            if remaining > 0:
                ranked_centers = same_tissue_quota_reassignment_centers(
                    tissue_region,
                    output,
                    tissue_nuc_prob,
                    gamma,
                    separation,
                )
                exact_backfill["reassignment_candidate_centers"] = int(
                    ranked_centers.shape[0]
                )
                reassignment_budget, reassignment_budget_audit = (
                    exact_backfill_candidate_budget(
                        remaining,
                        ranked_centers.shape[0],
                        args,
                    )
                )
                exact_backfill["reassignment_candidate_budget"] = (
                    reassignment_budget_audit
                )
                for cy, cx in ranked_centers[:reassignment_budget]:
                    if int(placed) >= int(target_count):
                        break
                    cy = int(cy)
                    cx = int(cx)
                    component_id = (
                        int(component_labels[cy, cx])
                        if component_labels is not None
                        else 0
                    )
                    if component_id <= 0:
                        continue
                    component_region = (
                        component_labels == component_id
                        if component_labels is not None
                        else tissue_region
                    )
                    exact_backfill["reassignment_attempted_centers"] += 1
                    attempted += 1
                    if component_sampling is not None:
                        component_sampling[str(component_id)][
                            "attempted_centers"
                        ] += 1
                    type_order, proposed_type_mass = balanced_type_order_at_center(
                        stabilized_type_prob,
                        cy,
                        cx,
                        args,
                        placed_by_type=placed_by_type,
                        expected_type_mass=expected_type_mass,
                        supported_types=supported_types,
                        tissue_type_prior=tissue_type_prior,
                        prior_weight=local_type_prior_weight,
                    )
                    if not type_order:
                        continue
                    placement_audit = {}
                    (
                        placed_ok,
                        nuc_type,
                        shape_source,
                        local_trials,
                        accepted_center,
                    ) = place_candidate_with_type_fallback(
                        nucleus_types=type_order,
                        output=output,
                        candidate_y=cy,
                        candidate_x=cx,
                        tissue_id=tissue_id,
                        shape_sampler=component_shape_samplers.get(
                            component_id,
                            shape_sampler,
                        ),
                        center_region=component_region,
                        valid_tissue_mask=same_tissue_footprint_mask,
                        dense_retry=True,
                        force_tissue_library=(
                            tissue_id in library_only_tissue_ids
                        ),
                        args=args,
                        placement_audit=placement_audit,
                        center_region_exclusions_by_type=(
                            typed_center_exclusions
                        ),
                    )
                    placement_trials += int(local_trials)
                    exact_backfill["placement_trials"] += int(local_trials)
                    exact_backfill["reassignment_placement_trials"] += int(
                        local_trials
                    )
                    if not placed_ok:
                        continue
                    placed += 1
                    exact_backfill["placed"] += 1
                    exact_backfill["reassigned_placed"] += 1
                    if int(placement_audit["type_fallback_rank"]) > 0:
                        exact_backfill["type_fallback_placed"] += 1
                        fallback_by_type = exact_backfill[
                            "type_fallback_placed_by_type"
                        ]
                        fallback_by_type[str(nuc_type)] = int(
                            fallback_by_type.get(str(nuc_type), 0)
                        ) + 1
                    reassigned_by_component = exact_backfill[
                        "reassigned_placed_by_component"
                    ]
                    reassigned_by_component[str(component_id)] = int(
                        reassigned_by_component.get(str(component_id), 0)
                    ) + 1
                    accepted_y, accepted_x = accepted_center
                    accepted_center_probabilities.append(
                        float(tissue_nuc_prob[accepted_y, accepted_x])
                    )
                    accepted_centers.append(
                        {
                            "row": int(accepted_y),
                            "col": int(accepted_x),
                            "nucleus_type": int(nuc_type),
                            "tissue_id": int(tissue_id),
                            "area_px": int(placement_audit["area_px"]),
                            "shape_source": str(shape_source),
                        }
                    )
                    placed_by_component[component_id] = (
                        placed_by_component.get(component_id, 0) + 1
                    )
                    if component_sampling is not None:
                        component_sampling[str(component_id)]["placed"] += 1
                    if proposed_type_mass is not None:
                        expected_type_mass += proposed_type_mass
                    placed_by_type[nuc_type] = (
                        placed_by_type.get(nuc_type, 0) + 1
                    )
                    placed_by_shape_source[str(shape_source)] += 1

        witness_fallback = None
        witness_shape_audit = packing_witness_shape_distribution_audit(
            accepted_centers,
            packing_witness,
        )
        witness_fallback_reason = (
            "exact_count_shortfall"
            if placed < target_count
            else (
                "local_shape_distribution_mismatch"
                if packing_witness and not witness_shape_audit["passed"]
                else None
            )
        )
        if witness_fallback_reason and packing_witness:
            witness_fallback = realize_compiled_packing_witness(
                base_output=output_before_tissue,
                packing_witness=packing_witness,
                center_region=tissue_region,
                valid_tissue_mask=same_tissue_footprint_mask,
                tissue_id=tissue_id,
                target_count=target_count,
                nucleus_probability=tissue_nuc_prob,
                minimum_separation_px=getattr(
                    args, "nucleus_spacing_margin_px", 1
                ),
                allowed_nucleus_types=supported_types,
                center_region_exclusions_by_type=typed_center_exclusions,
            )
            if witness_fallback is not None:
                output, accepted_centers = witness_fallback
                placed = len(accepted_centers)
                placed_by_shape_source = {
                    "reference": placed,
                    "library": 0,
                }
                accepted_center_probabilities = [
                    float(tissue_nuc_prob[item["row"], item["col"]])
                    for item in accepted_centers
                ]
                placed_by_type = {
                    int(nucleus_type): sum(
                        int(item["nucleus_type"]) == int(nucleus_type)
                        for item in accepted_centers
                    )
                    for nucleus_type in NUCLEI_CLASSES
                }
                expected_type_mass = np.zeros(
                    len(NUCLEI_CLASSES), dtype=np.float64
                )
                for item in accepted_centers:
                    _, proposed = balanced_type_order_at_center(
                        stabilized_type_prob,
                        int(item["row"]),
                        int(item["col"]),
                        args,
                        placed_by_type={},
                        expected_type_mass=np.zeros(
                            len(NUCLEI_CLASSES), dtype=np.float64
                        ),
                        supported_types=supported_types,
                        tissue_type_prior=tissue_type_prior,
                        prior_weight=local_type_prior_weight,
                    )
                    if proposed is not None:
                        expected_type_mass += proposed
                placed_by_component = {
                    component_id: sum(
                        (
                            int(component_labels[item["row"], item["col"]])
                            if component_labels is not None
                            else 0
                        )
                        == component_id
                        for item in accepted_centers
                    )
                    for component_id in component_limits
                }
                exact_backfill["packing_witness_fallback"] = {
                    "used": True,
                    "version": packing_witness.get("version"),
                    "contract_id": packing_witness.get("contract_id"),
                    "placed": placed,
                    "reason": witness_fallback_reason,
                    "pre_fallback_shape_distribution": witness_shape_audit,
                    "selection_policy": (
                        "certified_complete_source_shapes_probnet_ranked"
                    ),
                }

        diagnostics["placed"] += placed
        for source, count in placed_by_shape_source.items():
            diagnostics["placed_by_shape_source"][source] += count
        diagnostics["tissues"][str(tissue_id)] = {
            **count_info,
            "expected_nucleus_area": expected_area,
            "oversample_factor": oversample_factor,
            "min_distance": min_distance,
            "num_candidates": len(candidates),
            "target_count": target_count,
            "selected_centers": len(centers),
            "attempted_centers": attempted,
            "placement_trials": placement_trials,
            "placed": placed,
            "placed_by_shape_source": placed_by_shape_source,
            "candidate_queue_policy": (
                "adaptive_coverage_prefix_then_probnet_odds_mass_retry_tail"
            ),
            "candidate_quality_score": (
                "gamma_times_logit_probnet_probability_plus_seeded_gumbel"
            ),
            "candidate_probability_mass_exponent": float(gamma),
            "candidate_diversity_score": (
                "quota_prefix_minimum_distance_then_seeded_probnet_mass_rank"
            ),
            "candidate_diversity_weight": None,
            "quota_coverage_spacing_scale": float(
                getattr(args, "quota_coverage_spacing_scale", 0.75)
            ),
            "quota_coverage_max_radius": float(
                getattr(args, "quota_coverage_max_radius", 48.0)
            ),
            "quota_coverage_min_fraction": float(
                getattr(args, "quota_coverage_min_fraction", 0.2)
            ),
            "adaptive_quota_coverage": bool(
                getattr(args, "adaptive_quota_coverage", True)
            ),
            "retry_tail_policy": (
                "same_probnet_mass_permutation_then_component_pixel_backfill_"
                "then_same_tissue_quota_reassignment"
            ),
            "accepted_center_probability": (
                {
                    "minimum": float(np.min(accepted_center_probabilities)),
                    "median": float(np.median(accepted_center_probabilities)),
                    "mean": float(np.mean(accepted_center_probabilities)),
                    "maximum": float(np.max(accepted_center_probabilities)),
                }
                if accepted_center_probabilities
                else None
            ),
            "accepted_centers": accepted_centers,
            "type_quota_policy": "none_prior_controls_total_count_only",
            "type_quota_fusion": None,
            "type_shape_support": type_shape_support,
            "quota_conditioned_spatial_prior": None,
            "type_routing": type_routing_audit,
            "center_type_assignment_policy": (
                "probnet_local_log_pool_then_tissue_cumulative_"
                "posterior_balancing"
            ),
            "posterior_expected_by_type": {
                str(nucleus_type): float(expected_type_mass[index])
                for index, nucleus_type in enumerate(NUCLEI_CLASSES)
                if expected_type_mass[index] > 0
            },
            "shape_source_policy": (
                "exact_target_tissue_and_type_library_with_target_tissue_size_calibration"
                if tissue_id in library_only_tissue_ids
                else "reference_first_same_type_then_library"
            ),
            "target_by_type": None,
            "placed_by_type": {
                str(key): int(value)
                for key, value in placed_by_type.items()
                if int(value) > 0
            },
            "component_sampling": component_sampling,
            "global_sampling": global_sampling,
            "exact_count_backfill": exact_backfill,
        }
        if component_shape_samplers:
            component_shape_sampling[str(tissue_id)] = {
                str(component_id): sampler.diagnostics()
                for component_id, sampler in component_shape_samplers.items()
            }
        _require_complete_target_count(
            tissue_id=tissue_id,
            target_count=target_count,
            placed=placed,
            strict=bool(getattr(args, "require_exact_target_count", True)),
        )

    diagnostics["shape_sampling"] = shape_sampling_diagnostics(
        shape_sampler,
        component_shape_sampling,
        diagnostics["placed_by_shape_source"],
        component_policy_active=component_policy_active,
    )
    return output, diagnostics


class PlacementQuotaError(RuntimeError):
    """A deterministic placement attempt could not satisfy an exact quota."""


def _require_complete_target_count(
    *,
    tissue_id,
    target_count,
    placed,
    strict=True,
):
    """Reject silent cell-count loss after destructive boundary regeneration."""

    shortfall = int(target_count) - int(placed)
    if strict and shortfall > 0:
        raise PlacementQuotaError(
            "ProbNet placement did not satisfy the target-tissue count quota: "
            f"tissue_id={int(tissue_id)}, target={int(target_count)}, "
            f"placed={int(placed)}, shortfall={shortfall}."
        )
    return max(shortfall, 0)


def _sum_count_dicts(first, second):
    keys = set(first or {}) | set(second or {})
    return {
        str(key): int((first or {}).get(str(key), 0))
        + int((second or {}).get(str(key), 0))
        for key in keys
        if (
            int((first or {}).get(str(key), 0))
            + int((second or {}).get(str(key), 0))
        )
        > 0
    }


def _sum_float_dicts(first, second):
    keys = set(first or {}) | set(second or {})
    return {
        str(key): float((first or {}).get(str(key), 0.0))
        + float((second or {}).get(str(key), 0.0))
        for key in keys
        if (
            float((first or {}).get(str(key), 0.0))
            + float((second or {}).get(str(key), 0.0))
        )
        > 0.0
    }


def build_buffer_retained_by_type_overrides(
    tissue,
    input_nuclei,
    generation_mask,
    core_tissues,
):
    """Build an exact retained-cell ledger for the second regeneration stage."""

    overrides = {}
    for tissue_id in np.unique(tissue[generation_mask]):
        tissue_id = int(tissue_id)
        core_info = (core_tissues or {}).get(str(tissue_id), {})
        core_placed = int(core_info.get("placed", 0))
        core_by_type = {
            int(nuc_type): int(count)
            for nuc_type, count in (
                core_info.get("placed_by_type") or {}
            ).items()
            if int(count) > 0
        }
        if sum(core_by_type.values()) != core_placed:
            continue

        full_tissue_region = generation_mask & (tissue == tissue_id)
        original_by_type = count_retained_centers_by_type(
            input_nuclei,
            full_tissue_region,
        )
        merged = {
            int(nuc_type): int(count)
            for nuc_type, count in original_by_type.items()
            if int(count) > 0
        }
        for nuc_type, count in core_by_type.items():
            merged[nuc_type] = int(merged.get(nuc_type, 0)) + int(count)
        overrides[str(tissue_id)] = merged
    return overrides


def generate_two_stage_for_gamma(
    prob,
    tissue,
    input_nuclei,
    deletion_mask,
    generation_mask,
    library,
    reference_pool,
    gamma,
    args,
    density_scales,
    *,
    type_density=None,
    library_only_tissue_ids=None,
    type_proportions_by_tissue=None,
    type_prior_weights_by_tissue=None,
    placement_nuc_prob=None,
    placement_type_prob=None,
    population_mask=None,
    required_center_mask=None,
    minimum_required_centers=0,
    maximum_required_centers=None,
    required_nucleus_type=None,
    size_reference_pools_by_tissue=None,
    fallback_size_reference_pool=None,
    packing_witness=None,
):
    """Fill the legal destructive core, then the remaining placement domain.

    ``deletion_mask`` is an erasure footprint and may contain the tails of
    complete source instances outside the legal center region. It must never
    become an implicit placement region. The first stage is therefore the
    explicit intersection E∩P; the second stage uses P and preserves the exact
    count ledger.
    """

    # Joint callers provide a distinct biological population region T_pop.
    # Its complete target quota must be computed once and realized over P;
    # splitting T_pop into E-intersection-P and buffer stages would round and
    # calibrate two partial quotas independently. Legacy callers that omit the
    # mask retain the established two-stage behavior below.
    if population_mask is not None:
        compiled_total_new_count = (
            int(packing_witness.get("requested_count", 0))
            if packing_witness
            else None
        )
        required_center_mask = (
            np.zeros_like(generation_mask, dtype=bool)
            if required_center_mask is None
            else (
                np.asarray(required_center_mask, dtype=bool)
                & np.asarray(generation_mask, dtype=bool)
            )
        )
        minimum_required_centers = max(0, int(minimum_required_centers))
        if packing_witness is not None:
            certified_required = max(
                0,
                int(packing_witness.get("required_seam_count", 0)),
            )
            if minimum_required_centers != certified_required:
                raise PlacementQuotaError(
                    "runtime seam quota differs from the immutable packing "
                    "certificate: "
                    f"runtime={minimum_required_centers}, "
                    f"certificate={certified_required}"
                )
        maximum_required_centers = (
            None
            if maximum_required_centers is None
            else max(0, int(maximum_required_centers))
        )
        if (
            maximum_required_centers is not None
            and maximum_required_centers < minimum_required_centers
        ):
            raise PlacementQuotaError(
                "maximum required centers cannot be below the minimum"
            )
        if minimum_required_centers and not np.any(required_center_mask):
            raise PlacementQuotaError(
                "required center quota has an empty legal placement region"
            )
        if minimum_required_centers:
            required_tissue_ids = [
                int(value)
                for value in np.unique(tissue[required_center_mask])
                if int(value) not in set(args.skip_tissue_ids)
            ]
            if len(required_tissue_ids) != 1:
                raise PlacementQuotaError(
                    "required center region must resolve to exactly one legal "
                    "target tissue"
                )
            required_tissue_id = required_tissue_ids[0]
            required_output, required_diagnostics = generate_for_gamma(
                prob,
                tissue,
                input_nuclei,
                required_center_mask,
                library,
                reference_pool,
                gamma,
                args,
                density_scales,
                type_density=type_density,
                library_only_tissue_ids=library_only_tissue_ids,
                clear_edit_mask=False,
                type_proportions_by_tissue=type_proportions_by_tissue,
                type_prior_weights_by_tissue=type_prior_weights_by_tissue,
                placement_nuc_prob=placement_nuc_prob,
                placement_type_prob=placement_type_prob,
                population_mask=population_mask,
                new_target_count_overrides={
                    required_tissue_id: minimum_required_centers
                },
                size_reference_pools_by_tissue=(
                    size_reference_pools_by_tissue
                ),
                fallback_size_reference_pool=fallback_size_reference_pool,
                allowed_nucleus_types_override=(
                    (int(required_nucleus_type),)
                    if required_nucleus_type is not None
                    else None
                ),
                packing_witness=packing_witness,
            )
            required_info = (required_diagnostics.get("tissues") or {}).get(
                str(required_tissue_id),
                {},
            )
            required_placed = int(required_info.get("placed", 0))
            if compiled_total_new_count is not None:
                if compiled_total_new_count < required_placed:
                    raise PlacementQuotaError(
                        "compiled packing total is below the realized seam quota"
                    )
                remainder_target_overrides = {
                    required_tissue_id: (
                        compiled_total_new_count - required_placed
                    )
                }
            else:
                remainder_target_overrides = None
            # The exact seam quota applies only to the required target class.
            # Other compatible populations may legitimately occupy the same
            # anatomical band (for example inflammatory cells interspersed at
            # a melanoma--stroma boundary).  Keep the full compiled P for the
            # remainder; the typed packing witness and the final continuity
            # gate still cap target-class centers in the seam.
            remainder_generation_mask = np.asarray(
                generation_mask,
                dtype=bool,
            )
            output, diagnostics = generate_for_gamma(
                prob,
                tissue,
                required_output,
                remainder_generation_mask,
                library,
                reference_pool,
                gamma,
                args,
                density_scales,
                type_density=type_density,
                library_only_tissue_ids=library_only_tissue_ids,
                clear_edit_mask=False,
                type_proportions_by_tissue=type_proportions_by_tissue,
                type_prior_weights_by_tissue=type_prior_weights_by_tissue,
                placement_nuc_prob=placement_nuc_prob,
                placement_type_prob=placement_type_prob,
                population_mask=population_mask,
                new_target_count_overrides=remainder_target_overrides,
                size_reference_pools_by_tissue=(
                    size_reference_pools_by_tissue
                ),
                fallback_size_reference_pool=fallback_size_reference_pool,
                packing_witness=packing_witness,
                center_region_exclusions_by_type=(
                    {
                        int(required_nucleus_type): required_center_mask
                    }
                    if required_nucleus_type is not None
                    else None
                ),
            )
            _merge_required_center_stage_diagnostics(
                diagnostics,
                required_diagnostics,
                required_tissue_id=required_tissue_id,
                required_center_pixels=int(
                    np.count_nonzero(required_center_mask)
                ),
                minimum_required_centers=minimum_required_centers,
                maximum_required_centers=maximum_required_centers,
                required_nucleus_type=required_nucleus_type,
            )
            return output, diagnostics
        output, diagnostics = generate_for_gamma(
            prob,
            tissue,
            input_nuclei,
            generation_mask,
            library,
            reference_pool,
            gamma,
            args,
            density_scales,
            type_density=type_density,
            library_only_tissue_ids=library_only_tissue_ids,
            clear_edit_mask=False,
            type_proportions_by_tissue=type_proportions_by_tissue,
            type_prior_weights_by_tissue=type_prior_weights_by_tissue,
            placement_nuc_prob=placement_nuc_prob,
            placement_type_prob=placement_type_prob,
            population_mask=population_mask,
            new_target_count_overrides=_single_tissue_compiled_count_override(
                tissue=tissue,
                population_mask=population_mask,
                compiled_total_new_count=compiled_total_new_count,
                skipped_tissue_ids=set(args.skip_tissue_ids),
            ),
            size_reference_pools_by_tissue=size_reference_pools_by_tissue,
            fallback_size_reference_pool=fallback_size_reference_pool,
            packing_witness=packing_witness,
        )
        diagnostics["regeneration_stages"] = {
            "policy": "single_contract_quota_from_T_pop_with_centers_constrained_to_P_v2",
            "population_target_pixels": int(
                np.count_nonzero(population_mask)
            ),
            "placement_center_pixels": int(
                np.count_nonzero(generation_mask)
            ),
        }
        return output, diagnostics

    core_placement_mask = (
        np.asarray(deletion_mask, dtype=bool)
        & np.asarray(generation_mask, dtype=bool)
    )

    core_output, core_diagnostics = generate_for_gamma(
        prob,
        tissue,
        input_nuclei,
        core_placement_mask,
        library,
        reference_pool,
        gamma,
        args,
        density_scales,
        type_density=type_density,
        library_only_tissue_ids=library_only_tissue_ids,
        clear_edit_mask=False,
        type_proportions_by_tissue=type_proportions_by_tissue,
        type_prior_weights_by_tissue=type_prior_weights_by_tissue,
        placement_nuc_prob=placement_nuc_prob,
        placement_type_prob=placement_type_prob,
        size_reference_pools_by_tissue=size_reference_pools_by_tissue,
        fallback_size_reference_pool=fallback_size_reference_pool,
        packing_witness=packing_witness,
    )
    if np.array_equal(
        core_placement_mask,
        np.asarray(generation_mask, dtype=bool),
    ):
        core_diagnostics["regeneration_stages"] = {
            "policy": "single_stage_E_intersection_P_equals_P",
            "core": core_diagnostics["tissues"],
            "buffer_increment": {},
        }
        return core_output, core_diagnostics

    buffer_retained_by_type = build_buffer_retained_by_type_overrides(
        tissue,
        input_nuclei,
        generation_mask,
        core_diagnostics["tissues"],
    )
    output, buffer_diagnostics = generate_for_gamma(
        prob,
        tissue,
        core_output,
        generation_mask,
        library,
        reference_pool,
        gamma,
        args,
        density_scales,
        type_density=type_density,
        library_only_tissue_ids=library_only_tissue_ids,
        clear_edit_mask=False,
        type_proportions_by_tissue=type_proportions_by_tissue,
        type_prior_weights_by_tissue=type_prior_weights_by_tissue,
        placement_nuc_prob=placement_nuc_prob,
        placement_type_prob=placement_type_prob,
        retained_by_type_overrides=buffer_retained_by_type,
        size_reference_pools_by_tissue=size_reference_pools_by_tissue,
        fallback_size_reference_pool=fallback_size_reference_pool,
        packing_witness=packing_witness,
    )
    buffer_tissues = buffer_diagnostics["tissues"]
    core_tissues = core_diagnostics["tissues"]
    for tissue_id in set(core_tissues) | set(buffer_tissues):
        core_info = core_tissues.get(tissue_id, {})
        buffer_info = buffer_tissues.setdefault(tissue_id, {})
        core_placed = int(core_info.get("placed", 0))
        buffer_placed = int(buffer_info.get("placed", 0))
        buffer_info["core_target_count"] = int(
            core_info.get("target_count", 0)
        )
        buffer_info["core_placed"] = core_placed
        buffer_info["buffer_increment_target_count"] = int(
            buffer_info.get("target_count", 0)
        )
        buffer_info["buffer_increment_placed"] = buffer_placed
        buffer_info["target_count"] = (
            int(core_info.get("target_count", 0))
            + int(buffer_info.get("target_count", 0))
        )
        buffer_info["placed"] = core_placed + buffer_placed
        buffer_info["target_by_type"] = _sum_count_dicts(
            core_info.get("target_by_type"),
            buffer_info.get("target_by_type"),
        )
        buffer_info["placed_by_type"] = _sum_count_dicts(
            core_info.get("placed_by_type"),
            buffer_info.get("placed_by_type"),
        )
        buffer_info["posterior_expected_by_type"] = _sum_float_dicts(
            core_info.get("posterior_expected_by_type"),
            buffer_info.get("posterior_expected_by_type"),
        )
        buffer_info["accepted_centers"] = [
            *(core_info.get("accepted_centers") or []),
            *(buffer_info.get("accepted_centers") or []),
        ]
        buffer_info["two_stage_count_policy"] = (
            "fill_core_then_buffer_deficit_from_exact_placement_ledger"
        )

    core_sources = core_diagnostics["placed_by_shape_source"]
    buffer_sources = buffer_diagnostics["placed_by_shape_source"]
    buffer_diagnostics["placed"] = (
        int(core_diagnostics["placed"])
        + int(buffer_diagnostics["placed"])
    )
    buffer_diagnostics["placed_by_shape_source"] = {
        source: int(core_sources.get(source, 0))
        + int(buffer_sources.get(source, 0))
        for source in ("reference", "library")
    }
    buffer_shape_sampling = buffer_diagnostics.get("shape_sampling") or {}
    core_shape_sampling = core_diagnostics.get("shape_sampling") or {}
    buffer_shape_sampling["selected_by_source"] = dict(
        buffer_diagnostics["placed_by_shape_source"]
    )
    buffer_shape_sampling["component_local_by_regeneration_stage"] = {
        "core": core_shape_sampling.get("component_local") or {},
        "buffer_increment": buffer_shape_sampling.get("component_local") or {},
    }
    buffer_diagnostics["shape_sampling"] = buffer_shape_sampling
    buffer_diagnostics["regeneration_stages"] = {
        "policy": "core_first_then_exact_placement_ledger_buffer_deficit_v2",
        "core": core_tissues,
        "buffer_increment": {
            tissue_id: {
                key: value
                for key, value in info.items()
                if key
                in {
                    "buffer_increment_target_count",
                    "buffer_increment_placed",
                    "target_by_type",
                    "placed_by_type",
                }
            }
            for tissue_id, info in buffer_tissues.items()
        },
    }
    return output, buffer_diagnostics


def _single_tissue_compiled_count_override(
    *,
    tissue,
    population_mask,
    compiled_total_new_count,
    skipped_tissue_ids,
):
    """Bind a packing-certificate count to its single target tissue."""

    if compiled_total_new_count is None:
        return None
    tissue_ids = [
        int(value)
        for value in np.unique(
            np.asarray(tissue)[np.asarray(population_mask, dtype=bool)]
        )
        if int(value) not in skipped_tissue_ids
    ]
    if len(tissue_ids) != 1:
        raise PlacementQuotaError(
            "compiled packing count requires exactly one target tissue"
        )
    return {tissue_ids[0]: max(0, int(compiled_total_new_count))}


def _merge_required_center_stage_diagnostics(
    diagnostics,
    required_diagnostics,
    *,
    required_tissue_id,
    required_center_pixels,
    minimum_required_centers,
    maximum_required_centers=None,
    required_nucleus_type=None,
):
    """Merge a seam-first placement into the one exact population ledger.

    The first stage reserves a small, skill-compiled subset of the final new
    population at the edited interface.  The second stage recomputes the
    remaining quota from the unchanged T_pop denominator after observing those
    retained seam placements.  Adding the two ledgers therefore recovers one
    total quota without using the seam mask as an abundance denominator.
    """

    required_info = (required_diagnostics.get("tissues") or {}).get(
        str(required_tissue_id),
        {},
    )
    final_info = (diagnostics.get("tissues") or {}).get(
        str(required_tissue_id),
    )
    if final_info is None:
        # The required seam stage can legitimately realize the complete T_pop
        # quota.  The remainder sampler then has zero new cells and omits the
        # tissue entry entirely.  Materialize an explicit zero-remainder
        # ledger so the atomic seam+remainder accounting stays total.
        final_info = {
            "target_count": 0,
            "placed": 0,
            "placed_by_type": {},
            "target_by_type": {},
            "posterior_expected_by_type": {},
            "accepted_centers": [],
            "type_shape_support": required_info.get(
                "type_shape_support", {}
            ),
            "zero_remainder_materialized": True,
        }
        diagnostics.setdefault("tissues", {})[
            str(required_tissue_id)
        ] = final_info
    required_placed = int(required_info.get("placed", 0))
    if required_placed < int(minimum_required_centers):
        raise PlacementQuotaError(
            "required center placement quota was not completed: "
            f"required={int(minimum_required_centers)}, placed={required_placed}"
        )
    final_info["target_count"] = int(final_info.get("target_count", 0)) + int(
        required_info.get("target_count", 0)
    )
    final_info["placed"] = int(final_info.get("placed", 0)) + required_placed
    final_info["placed_by_type"] = _sum_count_dicts(
        required_info.get("placed_by_type"),
        final_info.get("placed_by_type"),
    )
    final_info["target_by_type"] = _sum_count_dicts(
        required_info.get("target_by_type"),
        final_info.get("target_by_type"),
    )
    final_info["posterior_expected_by_type"] = _sum_float_dicts(
        required_info.get("posterior_expected_by_type"),
        final_info.get("posterior_expected_by_type"),
    )
    final_info["accepted_centers"] = [
        *(required_info.get("accepted_centers") or []),
        *(final_info.get("accepted_centers") or []),
    ]
    diagnostics["placed"] = int(diagnostics.get("placed", 0)) + int(
        required_diagnostics.get("placed", 0)
    )
    required_sources = required_diagnostics.get("placed_by_shape_source") or {}
    final_sources = diagnostics.setdefault(
        "placed_by_shape_source",
        {"reference": 0, "library": 0},
    )
    for source in ("reference", "library"):
        final_sources[source] = int(final_sources.get(source, 0)) + int(
            required_sources.get(source, 0)
        )
    shape_sampling = diagnostics.get("shape_sampling") or {}
    shape_sampling["selected_by_source"] = dict(final_sources)
    diagnostics["shape_sampling"] = shape_sampling
    diagnostics["regeneration_stages"] = {
        "policy": "typed_seam_quota_then_full_P_population_remainder_v3",
        "required_tissue_id": int(required_tissue_id),
        "required_center_region_pixels": int(required_center_pixels),
        "minimum_required_centers": int(minimum_required_centers),
        "maximum_required_centers": (
            int(maximum_required_centers)
            if maximum_required_centers is not None
            else None
        ),
        "required_nucleus_type": (
            int(required_nucleus_type)
            if required_nucleus_type is not None
            else None
        ),
        "required_placed": required_placed,
        "required_stage": required_diagnostics.get("tissues") or {},
        "population_remainder_stage": diagnostics.get("tissues") or {},
    }


def heatmap_rgb(values, mask=None):
    values = np.clip(values, 0.0, 1.0)
    img = (values * 255).astype(np.uint8)
    colored = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    if mask is not None:
        dim = np.zeros_like(colored)
        dim[:] = [30, 30, 30]
        colored = np.where(mask[..., None], colored, dim)
    return colored


def draw_edit_contour(rgb, edit_mask):
    out = rgb.copy()
    contours, _ = cv2.findContours((edit_mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 2)
    return out


def make_comparison(tissue, input_nuclei, outputs_by_gamma, nuc_prob, edit_mask):
    panels = [
        draw_edit_contour(overlay(tissue, input_nuclei), edit_mask),
        draw_edit_contour(heatmap_rgb(nuc_prob, edit_mask), edit_mask),
    ]
    for gamma, nuclei in outputs_by_gamma:
        panels.append(draw_edit_contour(overlay(tissue, nuclei), edit_mask))

    h, w = tissue.shape
    row = np.concatenate(panels, axis=1)
    labeled = np.zeros((h + 34, row.shape[1], 3), dtype=np.uint8)
    labeled[:34] = 35
    labeled[34:] = row

    labels = ["input", "P(nucleus)"] + [f"gamma={gamma:g}" for gamma, _ in outputs_by_gamma]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, label in enumerate(labels):
        cv2.putText(labeled, label, (i * w + 6, 23), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return labeled


def make_accepted_centers_overlay(
    nuc_prob,
    input_nuclei,
    output_nuclei,
    edit_mask,
):
    """Render final new-instance centroids over the raw ProbNet heatmap."""

    rgb = heatmap_rgb(nuc_prob, edit_mask)
    new_nuclei = np.asarray(output_nuclei).copy()
    new_nuclei[np.asarray(input_nuclei) > 0] = 0
    structure = np.ones((3, 3), dtype=np.uint8)
    for class_index in range(1, NUM_NUCLEI):
        labels, count = ndimage.label(
            new_nuclei == class_index,
            structure=structure,
        )
        if not count:
            continue
        color = tuple(int(value) for value in NUCLEI_RGB[class_index])
        for center_y, center_x in ndimage.center_of_mass(
            new_nuclei == class_index,
            labels,
            range(1, count + 1),
        ):
            cv2.circle(
                rgb,
                (round(center_x), round(center_y)),
                4,
                color,
                -1,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(
                rgb,
                (round(center_x), round(center_y)),
                5,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )
    return draw_edit_contour(rgb, edit_mask)


def _new_nucleus_centers(input_nuclei, output_nuclei, region_mask, tissue):
    """Return every generated instance without counting retained nuclei.

    Placement centers are constrained to ``region_mask``, but an asymmetric
    nucleus placed near its boundary can have a geometric centroid just
    outside that region.  The component is still a valid generated instance;
    dropping it here makes the raster audit under-count an otherwise exact
    placement ledger.  Use a component pixel inside the generation region for
    tissue attribution while keeping every independent output component in
    the global count.
    """

    new_nuclei = np.asarray(output_nuclei).copy()
    new_nuclei[np.asarray(input_nuclei) > 0] = 0
    region = np.asarray(region_mask, dtype=bool)
    records = []
    structure = np.ones((3, 3), dtype=np.uint8)
    for class_index, raw_type in enumerate(NUCLEI_CLASSES, start=1):
        labels, count = ndimage.label(
            new_nuclei == class_index,
            structure=structure,
        )
        if not count:
            continue
        centers = ndimage.center_of_mass(
            new_nuclei == class_index,
            labels,
            range(1, count + 1),
        )
        for component_id, (center_y, center_x) in enumerate(centers, start=1):
            component_in_region = (labels == component_id) & region
            if np.any(component_in_region):
                component_rows, component_cols = np.nonzero(component_in_region)
                nearest = int(
                    np.argmin(
                        (component_rows - float(center_y)) ** 2
                        + (component_cols - float(center_x)) ** 2
                    )
                )
                row = int(component_rows[nearest])
                col = int(component_cols[nearest])
            else:
                row = int(np.clip(round(center_y), 0, region.shape[0] - 1))
                col = int(np.clip(round(center_x), 0, region.shape[1] - 1))
            records.append(
                {
                    "row": row,
                    "col": col,
                    "nucleus_type": int(raw_type),
                    "tissue_id": int(tissue[row, col]),
                }
            )
    return records


def _weighted_quantiles(values, weights, quantiles):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.full(quantiles.shape, np.nan, dtype=np.float64)
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values, kind="stable")
    values = values[order]
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return np.interp(quantiles, cumulative, values)


def _weighted_midrank_cdf(values, weights, observations):
    """Map observations to tie-aware CDF positions under weighted mass."""

    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid) or observations.size == 0:
        return np.asarray([], dtype=np.float64), {
            "unique_values": 0,
            "largest_tied_mass_fraction": None,
        }
    values = values[valid]
    weights = weights[valid]
    unique_values, inverse = np.unique(values, return_inverse=True)
    grouped_mass = np.bincount(
        inverse,
        weights=weights,
        minlength=unique_values.size,
    ).astype(np.float64)
    total_mass = float(grouped_mass.sum())
    if total_mass <= 0:
        return np.asarray([], dtype=np.float64), {
            "unique_values": int(unique_values.size),
            "largest_tied_mass_fraction": None,
        }
    cumulative = np.cumsum(grouped_mass)
    midrank = (cumulative - 0.5 * grouped_mass) / total_mass
    mapped = np.interp(
        observations,
        unique_values,
        midrank,
        left=float(midrank[0]),
        right=float(midrank[-1]),
    )
    return mapped, {
        "unique_values": int(unique_values.size),
        "largest_tied_mass_fraction": float(grouped_mass.max() / total_mass),
    }


def probability_concentration_diagnostics(
    region_probability,
    region_mass,
    accepted_probability,
    *,
    z_threshold=DEFAULT_SAMPLING_CONCENTRATION_Z_THRESHOLD,
):
    """Return a signed, sample-size-aware ProbNet concentration diagnostic.

    Under the frozen evaluation mass, tie-aware CDF positions should be
    centered on one half. Positive z means accepted centers over-concentrate
    on the sharpest ProbNet peaks; negative z means they under-follow them.
    This diagnostic directs a bounded gamma update but is not itself a new
    biological pass threshold.
    """

    accepted_probability = np.asarray(accepted_probability, dtype=np.float64)
    mass_quantiles, mass_info = _weighted_midrank_cdf(
        region_probability,
        region_mass,
        accepted_probability,
    )
    probability_span = (
        float(np.ptp(np.asarray(region_probability, dtype=np.float64)))
        if np.asarray(region_probability).size
        else 0.0
    )
    applicable = bool(
        mass_quantiles.size >= 4
        and probability_span > 1e-6
        and int(mass_info["unique_values"]) >= 16
        and float(mass_info["largest_tied_mass_fraction"] or 1.0) < 0.25
    )
    if not applicable:
        return {
            "applicable": False,
            "mass_quantile_mean": None,
            "bias": None,
            "standard_error": None,
            "z_score": None,
            "z_threshold": float(z_threshold),
            "direction": "not_applicable",
            **mass_info,
        }
    bias = float(np.mean(mass_quantiles) - 0.5)
    standard_error = float(np.sqrt(1.0 / (12.0 * mass_quantiles.size)))
    z_score = float(bias / max(standard_error, 1e-12))
    threshold = max(float(z_threshold), 0.0)
    if z_score > threshold:
        direction = "overconcentrated"
    elif z_score < -threshold:
        direction = "underfollow"
    else:
        direction = "aligned"
    return {
        "applicable": True,
        "mass_quantile_mean": float(np.mean(mass_quantiles)),
        "bias": bias,
        "standard_error": standard_error,
        "z_score": z_score,
        "z_threshold": threshold,
        "direction": direction,
        **mass_info,
    }


def _distribution_tv(expected, observed):
    keys = sorted(set(expected) | set(observed))
    expected_total = float(sum(max(float(expected.get(key, 0.0)), 0.0) for key in keys))
    observed_total = float(sum(max(float(observed.get(key, 0.0)), 0.0) for key in keys))
    if expected_total <= 0 or observed_total <= 0:
        return None
    return 0.5 * float(
        sum(
            abs(
                max(float(expected.get(key, 0.0)), 0.0) / expected_total
                - max(float(observed.get(key, 0.0)), 0.0) / observed_total
            )
            for key in keys
        )
    )


def _spatial_geometry_metrics(
    *,
    region,
    joint_probability,
    evaluation_gamma,
    accepted_rows,
    accepted_cols,
    target_count,
):
    """Audit ProbNet geometry inside one executable placement stratum."""

    region = np.asarray(region, dtype=bool)
    inside = region[accepted_rows, accepted_cols]
    rows = accepted_rows[inside]
    cols = accepted_cols[inside]
    target_count = int(target_count)
    if target_count <= 0:
        return None
    if not np.any(region) or len(rows) != target_count:
        return {
            "target_count": target_count,
            "observed_count": len(rows),
            "region_pixels": int(np.count_nonzero(region)),
            "boundary_quantile_error": float("inf"),
            "probability_mass_coverage_ratio": float("inf"),
            "count_passed": False,
        }
    region_probability = joint_probability[region]
    region_mass = probability_sampling_mass(
        region_probability,
        evaluation_gamma,
    )
    mass_total = float(np.sum(region_mass))
    boundary_distance = ndimage.distance_transform_edt(
        np.pad(region, 1, mode="constant", constant_values=False)
    )[1:-1, 1:-1]
    expected_boundary_quantiles = _weighted_quantiles(
        boundary_distance[region],
        region_mass,
        (0.25, 0.50, 0.75),
    )
    accepted_boundary_quantiles = np.quantile(
        boundary_distance[rows, cols],
        (0.25, 0.50, 0.75),
    )
    spacing = np.sqrt(
        float(np.count_nonzero(region)) / max(float(target_count), 1.0)
    )
    boundary_scale = max(
        float(
            expected_boundary_quantiles[2]
            - expected_boundary_quantiles[0]
        ),
        0.5 * spacing,
        1.0,
    )
    boundary_error = float(
        np.mean(
            np.abs(
                accepted_boundary_quantiles
                - expected_boundary_quantiles
            )
        )
        / boundary_scale
    )
    center_mask = np.zeros(region.shape, dtype=bool)
    center_mask[rows, cols] = True
    nearest_center = ndimage.distance_transform_edt(~center_mask)
    coverage_ratio = float(
        np.sum(nearest_center[region] * region_mass)
        / max(mass_total, 1e-12)
        / max(spacing, 1.0)
    )
    return {
        "target_count": target_count,
        "observed_count": len(rows),
        "region_pixels": int(np.count_nonzero(region)),
        "boundary_quantile_error": boundary_error,
        "probability_mass_coverage_ratio": coverage_ratio,
        "count_passed": True,
    }


def probnet_sampling_alignment_audit(
    *,
    input_nuclei,
    output_nuclei,
    tissue,
    generation_region,
    placement_type_prob,
    gamma,
    generation_diagnostics,
    evaluation_gamma=None,
    concentration_z_threshold=DEFAULT_SAMPLING_CONCENTRATION_Z_THRESHOLD,
    exact_required_region=None,
    exact_required_count=0,
    exact_required_nucleus_type=None,
):
    """Audit count, type, and spatial fidelity against this patch's ProbNet.

    The audit contains no dataset, organ, or tissue-specific branch. A sharp
    boundary prior is expected to produce boundary-aligned centers; a broad
    prior is expected to cover the corresponding broad probability mass.
    """

    tissue = np.asarray(tissue)
    generation_region = np.asarray(generation_region, dtype=bool)
    mask_instance_records = _new_nucleus_centers(
        input_nuclei,
        output_nuclei,
        generation_region,
        tissue,
    )
    tissue_results = {}
    weighted_scores = []
    all_passed = True
    failure_reasons = []
    total_target = 0
    audited_target = 0
    fixed_evaluation_gamma = float(
        gamma if evaluation_gamma is None else evaluation_gamma
    )
    for tissue_key, info in (generation_diagnostics.get("tissues") or {}).items():
        tissue_id = int(tissue_key)
        region = generation_region & (tissue == tissue_id)
        target_count = int(info.get("target_count", 0))
        total_target += target_count
        mask_attributed_records = [
            record
            for record in mask_instance_records
            if record["tissue_id"] == tissue_id
        ]
        accepted_center_field_present = "accepted_centers" in info
        accepted_centers = []
        for raw_record in info.get("accepted_centers") or []:
            row = int(raw_record.get("row", -1))
            col = int(raw_record.get("col", -1))
            if (
                row < 0
                or row >= tissue.shape[0]
                or col < 0
                or col >= tissue.shape[1]
                or not generation_region[row, col]
                or int(tissue[row, col]) != tissue_id
            ):
                continue
            accepted_centers.append(
                {
                    "row": row,
                    "col": col,
                    "nucleus_type": int(raw_record["nucleus_type"]),
                    "tissue_id": tissue_id,
                }
            )
        if (
            accepted_center_field_present
            and len(accepted_centers) == int(info.get("placed", 0))
        ):
            tissue_records = accepted_centers
            center_record_policy = "accepted_placement_center_ledger"
        else:
            tissue_records = mask_attributed_records
            center_record_policy = "output_component_centroid_fallback"
        observed_count = len(tissue_records)
        count_passed = observed_count == target_count
        placed_by_type = {
            int(key): int(value)
            for key, value in (info.get("placed_by_type") or {}).items()
        }
        observed_by_type = {}
        for record in tissue_records:
            nuc_type = int(record["nucleus_type"])
            observed_by_type[nuc_type] = observed_by_type.get(nuc_type, 0) + 1
        expected_by_type = {
            int(key): float(value)
            for key, value in (info.get("posterior_expected_by_type") or {}).items()
        }
        type_tv = _distribution_tv(expected_by_type, observed_by_type)
        type_threshold = max(
            0.25,
            1.0 / np.sqrt(max(target_count, 1)),
        )
        type_applicable = bool(target_count >= 4 and type_tv is not None)
        type_passed = bool(not type_applicable or type_tv <= type_threshold)

        spatial_applicable = bool(target_count >= 4 and observed_count >= 4 and np.any(region))
        probability_bin_tv = None
        probability_threshold = None
        largest_tied_mass_fraction = None
        probability_concentration = {
            "applicable": False,
            "direction": "not_applicable",
        }
        boundary_quantile_error = None
        probability_mass_coverage_ratio = None
        spatial_conditioning_policy = "single_unpartitioned_placement_region"
        spatial_strata = []
        spatial_score = None
        spatial_passed = True
        if spatial_applicable:
            supported_types = [
                int(value)
                for value in (
                    (info.get("type_shape_support") or {}).get("supported_types")
                    or NUCLEI_CLASSES
                )
            ]
            joint_probability = supported_joint_nucleus_probability(
                np.asarray(placement_type_prob, dtype=np.float32),
                supported_types,
            )
            region_probability = joint_probability[region]
            region_mass = probability_sampling_mass(
                region_probability,
                fixed_evaluation_gamma,
            )
            mass_total = float(np.sum(region_mass))
            accepted_rows = np.asarray(
                [record["row"] for record in tissue_records], dtype=np.int64
            )
            accepted_cols = np.asarray(
                [record["col"] for record in tissue_records], dtype=np.int64
            )
            accepted_probability = joint_probability[
                accepted_rows, accepted_cols
            ]

            accepted_mass_quantiles, mass_info = _weighted_midrank_cdf(
                region_probability,
                region_mass,
                accepted_probability,
            )
            largest_tied_mass_fraction = mass_info[
                "largest_tied_mass_fraction"
            ]
            probability_concentration = probability_concentration_diagnostics(
                region_probability,
                region_mass,
                accepted_probability,
                z_threshold=concentration_z_threshold,
            )
            if probability_concentration["applicable"]:
                observed_bins, _ = np.histogram(
                    accepted_mass_quantiles,
                    bins=np.linspace(0.0, 1.0, 5),
                )
                observed_fraction = observed_bins / max(observed_bins.sum(), 1)
                probability_bin_tv = 0.5 * float(
                    np.sum(np.abs(observed_fraction - 0.25))
                )

            boundary_distance = ndimage.distance_transform_edt(
                np.pad(region, 1, mode="constant", constant_values=False)
            )[1:-1, 1:-1]
            expected_boundary_quantiles = _weighted_quantiles(
                boundary_distance[region],
                region_mass,
                (0.25, 0.50, 0.75),
            )
            accepted_boundary_quantiles = np.quantile(
                boundary_distance[accepted_rows, accepted_cols],
                (0.25, 0.50, 0.75),
            )
            spacing = np.sqrt(
                float(np.count_nonzero(region)) / max(float(target_count), 1.0)
            )
            boundary_scale = max(
                float(expected_boundary_quantiles[2] - expected_boundary_quantiles[0]),
                0.5 * spacing,
                1.0,
            )
            boundary_quantile_error = float(
                np.mean(
                    np.abs(
                        accepted_boundary_quantiles
                        - expected_boundary_quantiles
                    )
                )
                / boundary_scale
            )

            center_mask = np.zeros(region.shape, dtype=bool)
            center_mask[accepted_rows, accepted_cols] = True
            nearest_center = ndimage.distance_transform_edt(~center_mask)
            weighted_mean_distance = float(
                np.sum(nearest_center[region] * region_mass)
                / max(mass_total, 1e-12)
            )
            probability_mass_coverage_ratio = float(
                weighted_mean_distance / max(spacing, 1.0)
            )
            if (
                exact_required_region is not None
                and int(exact_required_count) > 0
                and np.any(
                    region
                    & np.asarray(exact_required_region, dtype=bool)
                )
            ):
                required_region = (
                    region
                    & np.asarray(exact_required_region, dtype=bool)
                )
                remainder_region = region & ~required_region
                # A compiled seam may be an exact quota or a minimum quota.
                # In both cases the spatial audit must condition on the
                # *realized* number of accepted centers in that stratum.  Using
                # the declared minimum as if it were exact incorrectly audits
                # seam-constrained centers against the whole placement field.
                # Quota satisfaction itself is enforced by the placement
                # contract; this block audits the geometry of what was placed.
                accepted_types = np.asarray(
                    [record["nucleus_type"] for record in tissue_records],
                    dtype=np.int64,
                )
                required_selector = required_region[
                    accepted_rows, accepted_cols
                ]
                if exact_required_nucleus_type is not None:
                    required_selector &= (
                        accepted_types == int(exact_required_nucleus_type)
                    )
                required_count = int(np.count_nonzero(required_selector))
                remainder_selector = ~required_selector
                remainder_count = int(np.count_nonzero(remainder_selector))
                required_probability = joint_probability
                if exact_required_nucleus_type is not None:
                    required_type = int(exact_required_nucleus_type)
                    if required_type not in set(NUCLEI_CLASSES):
                        raise ValueError(
                            "exact required nucleus type is outside the "
                            "configured observation schema"
                        )
                    required_probability = np.asarray(
                        placement_type_prob,
                        dtype=np.float32,
                    )[NUCLEI_CLASSES.index(required_type)]
                spatial_strata = [
                    _spatial_geometry_metrics(
                        region=required_region,
                        joint_probability=required_probability,
                        evaluation_gamma=fixed_evaluation_gamma,
                        accepted_rows=accepted_rows[required_selector],
                        accepted_cols=accepted_cols[required_selector],
                        target_count=required_count,
                    )
                ]
                if remainder_count > 0:
                    # A typed seam is a quota stratum, not an exclusion zone.
                    # Compatible non-target populations inside that band are
                    # audited with the full placement population rather than
                    # being dropped from both strata.
                    remainder_region = (
                        region
                        if exact_required_nucleus_type is not None
                        else remainder_region
                    )
                    spatial_strata.append(
                        _spatial_geometry_metrics(
                            region=remainder_region,
                            joint_probability=joint_probability,
                            evaluation_gamma=fixed_evaluation_gamma,
                            accepted_rows=accepted_rows[remainder_selector],
                            accepted_cols=accepted_cols[remainder_selector],
                            target_count=remainder_count,
                        )
                    )
                spatial_strata = [
                    item for item in spatial_strata if item is not None
                ]
                if spatial_strata:
                    stratum_weight = float(
                        sum(item["target_count"] for item in spatial_strata)
                    )
                    boundary_quantile_error = float(
                        sum(
                            item["target_count"]
                            * item["boundary_quantile_error"]
                            for item in spatial_strata
                        )
                        / max(stratum_weight, 1.0)
                    )
                    probability_mass_coverage_ratio = float(
                        sum(
                            item["target_count"]
                            * item["probability_mass_coverage_ratio"]
                            for item in spatial_strata
                        )
                        / max(stratum_weight, 1.0)
                    )
                    spatial_conditioning_policy = (
                        "compiled_typed_seam_and_exterior_observed_strata_v3"
                        if exact_required_nucleus_type is not None
                        else "compiled_seam_and_exterior_observed_strata_v2"
                    )
            score_terms = [
                float(np.exp(-boundary_quantile_error)),
                float(np.exp(-probability_mass_coverage_ratio)),
            ]
            if probability_bin_tv is not None:
                score_terms.append(max(0.0, 1.0 - probability_bin_tv))
            spatial_score = float(np.mean(score_terms))
            probability_threshold = max(
                0.40,
                1.0 / np.sqrt(max(target_count, 1)),
            )
            spatial_passed = bool(
                boundary_quantile_error <= 1.25
                and probability_mass_coverage_ratio <= 1.10
                and spatial_score >= 0.45
            )

        tissue_failure_reasons = []
        if not count_passed:
            tissue_failure_reasons.append("COUNT_QUOTA_SHORTFALL")
        if not type_passed:
            tissue_failure_reasons.append("TYPE_POSTERIOR_MISMATCH")
        if not spatial_passed:
            direction = probability_concentration.get("direction")
            if direction == "overconcentrated":
                tissue_failure_reasons.append("PROBNET_OVERCONCENTRATED")
            elif direction == "underfollow":
                tissue_failure_reasons.append("PROBNET_UNDERFOLLOW")
            else:
                tissue_failure_reasons.append("PROBNET_COVERAGE_GAP")
            if (
                boundary_quantile_error is not None
                and boundary_quantile_error > 1.25
            ):
                tissue_failure_reasons.append("PROBNET_BOUNDARY_MISMATCH")
        failure_reasons.extend(tissue_failure_reasons)

        count_score = 1.0 if count_passed else max(
            0.0,
            1.0 - abs(observed_count - target_count) / max(target_count, 1),
        )
        type_score = 1.0 if type_tv is None else max(0.0, 1.0 - type_tv)
        tissue_score_terms = [(0.25, count_score)]
        if type_applicable:
            tissue_score_terms.append((0.25, type_score))
        if spatial_applicable and spatial_score is not None:
            tissue_score_terms.append((0.50, spatial_score))
        score_weight = sum(weight for weight, _ in tissue_score_terms)
        tissue_score = sum(
            weight * score for weight, score in tissue_score_terms
        ) / max(score_weight, 1e-12)
        tissue_passed = bool(count_passed and type_passed and spatial_passed)
        all_passed &= tissue_passed
        if spatial_applicable or type_applicable:
            audited_target += target_count
        weighted_scores.append((max(target_count, 1), tissue_score))
        tissue_results[str(tissue_id)] = {
            "target_count": target_count,
            "observed_new_instance_count": observed_count,
            "mask_component_attributed_count": len(mask_attributed_records),
            "center_record_policy": center_record_policy,
            "count_passed": count_passed,
            "diagnostic_placed_by_type": {
                str(key): int(value) for key, value in placed_by_type.items()
            },
            "observed_by_type": {
                str(key): int(value) for key, value in observed_by_type.items()
            },
            "expected_by_type": {
                str(key): float(value) for key, value in expected_by_type.items()
            },
            "type_tv": type_tv,
            "type_threshold": float(type_threshold),
            "type_applicable": type_applicable,
            "type_passed": type_passed,
            "spatial_applicable": spatial_applicable,
            "probability_bin_tv": probability_bin_tv,
            "probability_bin_tv_role": (
                "diagnostic_score_term_not_hard_gate_under_constrained_"
                "without_replacement_sampling"
            ),
            "probability_bin_tv_reference_threshold": probability_threshold,
            "largest_tied_probability_mass_fraction": (
                largest_tied_mass_fraction
            ),
            "probability_concentration": probability_concentration,
            "boundary_quantile_error": boundary_quantile_error,
            "probability_mass_coverage_ratio": probability_mass_coverage_ratio,
            "spatial_conditioning_policy": spatial_conditioning_policy,
            "spatial_strata": spatial_strata,
            "spatial_score": spatial_score,
            "spatial_passed": spatial_passed,
            "score": tissue_score,
            "passed": tissue_passed,
            "failure_reasons": tissue_failure_reasons,
        }

    total_weight = float(sum(weight for weight, _ in weighted_scores))
    score = (
        float(sum(weight * value for weight, value in weighted_scores) / total_weight)
        if total_weight > 0
        else 1.0
    )
    global_instance_count_passed = len(mask_instance_records) == total_target
    all_passed &= global_instance_count_passed
    if not global_instance_count_passed:
        failure_reasons.append("RASTER_INSTANCE_COUNT_MISMATCH")
    failure_reasons = list(dict.fromkeys(failure_reasons))
    return {
        "policy": SAMPLING_AUDIT_POLICY_NAME,
        "organ_specific_constraints": False,
        "gamma": float(gamma),
        "sampling_gamma": float(gamma),
        "evaluation_gamma": fixed_evaluation_gamma,
        "passed": bool(all_passed),
        "score": score,
        "evidence_coverage": float(audited_target / max(total_target, 1)),
        "new_instance_count": len(mask_instance_records),
        "accepted_center_count": int(
            sum(
                len(info.get("accepted_centers") or [])
                for info in (generation_diagnostics.get("tissues") or {}).values()
            )
        ),
        "global_instance_count_passed": bool(global_instance_count_passed),
        "failure_reasons": failure_reasons,
        "primary_failure_reason": failure_reasons[0] if failure_reasons else None,
        "tissues": tissue_results,
    }


def next_sampling_feedback_parameters(
    *,
    initial_gamma,
    current_gamma,
    base_seed,
    attempt_index,
    previous_failure_reasons,
    gamma_already_adjusted,
    gamma_down_factor=DEFAULT_SAMPLING_FEEDBACK_GAMMA_DOWN_FACTOR,
    gamma_up_factor=DEFAULT_SAMPLING_FEEDBACK_GAMMA_UP_FACTOR,
    gamma_min=DEFAULT_SAMPLING_FEEDBACK_GAMMA_MIN,
    gamma_max=DEFAULT_SAMPLING_FEEDBACK_GAMMA_MAX,
):
    """Return one bounded, reason-directed sampling update.

    Count, tissue, deletion, shape-source and spacing contracts are immutable.
    The controller changes gamma at most once; all other retries only change
    the deterministic seed so the feedback loop remains interpretable.
    """

    reasons = tuple(str(value) for value in (previous_failure_reasons or ()))
    selected_gamma = float(current_gamma)
    action = "initial_sample" if int(attempt_index) == 0 else "resample_seed"
    adjusted = bool(gamma_already_adjusted)
    if int(attempt_index) > 0 and not adjusted:
        overconcentrated = "PROBNET_OVERCONCENTRATED" in reasons
        underfollowing = "PROBNET_UNDERFOLLOW" in reasons
        if overconcentrated and not underfollowing:
            selected_gamma = float(
                np.clip(
                    selected_gamma * float(gamma_down_factor),
                    float(gamma_min),
                    float(gamma_max),
                )
            )
            action = "decrease_gamma"
            adjusted = not np.isclose(selected_gamma, current_gamma)
        elif underfollowing and not overconcentrated:
            selected_gamma = float(
                np.clip(
                    selected_gamma * float(gamma_up_factor),
                    float(gamma_min),
                    float(gamma_max),
                )
            )
            action = "increase_gamma"
            adjusted = not np.isclose(selected_gamma, current_gamma)
    return {
        "attempt_index": int(attempt_index),
        "seed": int(base_seed) + int(attempt_index),
        "initial_gamma": float(initial_gamma),
        "previous_gamma": float(current_gamma),
        "sampling_gamma": selected_gamma,
        "action": action,
        "trigger_reasons": list(reasons),
        "gamma_adjusted": adjusted,
    }


def predict_context_stabilized_spatial_probability(
    model,
    tissue,
    input_nuclei,
    edit_mask,
    cancer_id,
    device,
    context_prob,
    args,
):
    """Remove the artificial changed-region boundary from the placement prior.

    ProbNet receives the edit mask as an input channel. Reusing the exact
    changed-region mask can therefore turn that artificial contour into a
    spatial cue. The production forward expands the prediction support by one
    generic nucleus scale and clears nuclei throughout that support. Placement
    remains restricted to the original changed region.
    """

    context_nuc_prob = np.asarray(1.0 - context_prob[0], dtype=np.float32)
    weight = float(getattr(args, "spatial_context_halo_weight", 1.0))
    radius = spatial_context_halo_radius(
        getattr(args, "expected_nucleus_area", 80.0),
        diameter_scale=float(
            getattr(args, "spatial_context_halo_diameter_scale", 1.25)
        ),
        minimum=int(getattr(args, "spatial_context_halo_min_px", 4)),
        maximum=int(getattr(args, "spatial_context_halo_max_px", 24)),
    )
    if weight <= 0.0:
        return context_nuc_prob, np.asarray(context_prob[1:], dtype=np.float32), {
            "policy": "raw_context_probnet_only",
            "halo_weight": 0.0,
            "halo_radius_px": radius,
            "cleared_retained_nucleus_pixels": 0,
        }

    support_mask = ndimage.binary_dilation(
        np.asarray(edit_mask, dtype=bool),
        iterations=radius,
    )
    support_input = np.asarray(input_nuclei).copy()
    cleared_pixels = int(
        np.count_nonzero((support_input > 0) & support_mask)
    )
    support_input[support_mask] = 0

    support_prob, _ = predict_fields(
        model,
        tissue,
        support_input,
        support_mask,
        cancer_id,
        device,
    )
    support_nuc_prob = np.asarray(1.0 - support_prob[0], dtype=np.float32)
    placement_type_prob = np.stack(
        [
            blend_context_stabilized_probability(
                context_prob[class_index],
                support_prob[class_index],
                halo_weight=weight,
            )
            for class_index in range(1, len(NUCLEI_CLASSES) + 1)
        ],
        axis=0,
    )
    placement_nuc_prob = np.clip(
        np.sum(placement_type_prob, axis=0),
        1e-6,
        1.0,
    ).astype(np.float32)
    support = np.asarray(edit_mask, dtype=bool)

    def _quantiles(values):
        if not np.any(support):
            return {}
        return {
            f"q{int(quantile * 100):02d}": float(
                np.quantile(values[support], quantile)
            )
            for quantile in (0.10, 0.50, 0.90, 0.99)
        }

    return placement_nuc_prob, placement_type_prob, {
        "policy": SPATIAL_PRIOR_POLICY_NAME,
        "halo_weight": weight,
        "halo_radius_px": radius,
        "cleared_retained_nucleus_pixels": cleared_pixels,
        "second_forward_skipped": False,
        "edit_mask_area": int(np.count_nonzero(edit_mask)),
        "prediction_support_area": int(np.count_nonzero(support_mask)),
        "prediction_support_policy": (
            "binary_dilation_one_nucleus_scale_placement_cropped_to_edit"
        ),
        "raw_probability_quantiles": _quantiles(context_nuc_prob),
        "support_cleared_probability_quantiles": _quantiles(
            support_nuc_prob
        ),
        "placement_probability_quantiles": _quantiles(
            placement_nuc_prob
        ),
    }


def run_single(args, model, library, config, density_scales, device):
    tissue = load_tissue_mask(args.input_tissue)
    if args.reference_tissue:
        reference_tissue = load_tissue_mask(args.reference_tissue)
        if reference_tissue.shape != tissue.shape:
            raise ValueError(
                "reference and target tissue masks must have the same size: "
                f"{reference_tissue.shape} vs {tissue.shape}"
            )
    else:
        reference_tissue = tissue
    edit_mask = cv2.imread(args.edit_region, cv2.IMREAD_GRAYSCALE)
    if edit_mask is None:
        raise FileNotFoundError(f"Cannot load edit region mask: {args.edit_region}")
    edit_mask = edit_mask > 128
    if args.placement_region:
        placement_mask = cv2.imread(
            args.placement_region,
            cv2.IMREAD_GRAYSCALE,
        )
        if placement_mask is None:
            raise FileNotFoundError(
                f"Cannot load placement region mask: {args.placement_region}"
            )
        placement_mask = placement_mask > 128
    else:
        placement_mask = edit_mask.copy()
    if args.population_region:
        population_mask = cv2.imread(
            args.population_region,
            cv2.IMREAD_GRAYSCALE,
        )
        if population_mask is None:
            raise FileNotFoundError(
                f"Cannot load population region mask: {args.population_region}"
            )
        population_mask = population_mask > 128
    else:
        population_mask = placement_mask.copy()
    if args.required_placement_region:
        required_placement_mask = cv2.imread(
            args.required_placement_region,
            cv2.IMREAD_GRAYSCALE,
        )
        if required_placement_mask is None:
            raise FileNotFoundError(
                "Cannot load required placement region mask: "
                f"{args.required_placement_region}"
            )
        required_placement_mask = required_placement_mask > 128
    else:
        required_placement_mask = np.zeros_like(placement_mask, dtype=bool)
    if args.deletion_region:
        deletion_mask = cv2.imread(
            args.deletion_region,
            cv2.IMREAD_GRAYSCALE,
        )
        if deletion_mask is None:
            raise FileNotFoundError(
                f"Cannot load deletion region mask: {args.deletion_region}"
            )
        deletion_mask = deletion_mask > 128
    else:
        deletion_mask = edit_mask.copy()
    if deletion_mask.shape != edit_mask.shape:
        raise ValueError("deletion and generation regions must share one shape")
    if placement_mask.shape != edit_mask.shape:
        raise ValueError("placement and generation regions must share one shape")
    if population_mask.shape != edit_mask.shape:
        raise ValueError("population and generation regions must share one shape")
    if required_placement_mask.shape != edit_mask.shape:
        raise ValueError(
            "required placement and generation regions must share one shape"
        )
    if np.any(placement_mask & ~edit_mask):
        raise ValueError("generation region must contain every placement pixel")
    if np.any(deletion_mask & ~edit_mask):
        raise ValueError("generation region must contain every deletion pixel")
    if np.any(population_mask & ~edit_mask):
        raise ValueError("generation region must contain every population pixel")
    if np.any(required_placement_mask & ~placement_mask):
        raise ValueError(
            "placement region must contain every required placement pixel"
        )
    if int(args.minimum_required_placements) < 0:
        raise ValueError("minimum required placements cannot be negative")
    if int(args.maximum_required_placements) < -1:
        raise ValueError(
            "maximum required placements must be -1 or non-negative"
        )
    if (
        int(args.maximum_required_placements) >= 0
        and int(args.maximum_required_placements)
        < int(args.minimum_required_placements)
    ):
        raise ValueError(
            "maximum required placements cannot be below the minimum"
        )
    if (
        args.required_nucleus_type is not None
        and int(args.required_nucleus_type) not in set(NUCLEI_CLASSES)
    ):
        raise ValueError("required nucleus type is outside the CellViT schema")
    if int(args.minimum_required_placements) > 0 and not np.any(
        required_placement_mask
    ):
        raise ValueError(
            "minimum required placements need a non-empty required placement region"
        )
    packing_witness = None
    if args.packing_witness:
        packing_witness = json.loads(
            Path(args.packing_witness).read_text(encoding="utf-8")
        )
        if (
            packing_witness.get("version") != "compiled-packing-witness-v4"
            or int(packing_witness.get("requested_count", 0)) <= 0
            or len(packing_witness.get("placements") or [])
            != int(packing_witness.get("requested_count", 0))
        ):
            raise ValueError("packing witness is malformed or incomplete")
    semantic_edit_pixels = int(np.count_nonzero(deletion_mask))
    if args.widen_edit_region:
        edit_mask = widen_locally_thin_mask(
            edit_mask,
            (reference_tissue != 0) | (tissue != 0),
            minimum_width=args.minimum_mask_width,
        )

    reference_nuclei_path = args.reference_nuclei_shapes or args.input_nuclei
    if reference_nuclei_path:
        reference_nuclei_raw = load_nuclei_mask(reference_nuclei_path, remap=False)
        if reference_nuclei_raw.shape != tissue.shape:
            raise ValueError(
                "reference nuclei shape mask and tissue mask must have the same size: "
                f"{reference_nuclei_raw.shape} vs {tissue.shape}"
            )
        reference_pool = build_reference_pool(reference_nuclei_raw, args)
    else:
        reference_nuclei_raw = np.zeros_like(tissue, dtype=np.uint8)
        reference_pool = None

    if args.input_nuclei:
        # Population abundance/type priors must use the complete observed
        # source mask. ``--reference-nuclei-shapes`` may be a deliberately
        # filtered pool that removes censored or protected shapes and must not
        # silently reduce the patch density estimate.
        population_reference_nuclei_raw = load_nuclei_mask(
            args.input_nuclei,
            remap=False,
        )
        input_nuclei = load_nuclei_mask(args.input_nuclei, remap=True)
    else:
        input_nuclei = np.zeros_like(tissue, dtype=np.int64)
        population_reference_nuclei_raw = reference_nuclei_raw
    size_reference_pools_by_tissue = {}
    complete_size_reference_pool = (
        reference_pool.filtered_for_size_calibration()
        if reference_pool is not None
        else None
    )
    if reference_pool is not None:
        for tissue_id in np.unique(reference_tissue):
            tissue_id = int(tissue_id)
            if tissue_id == 0:
                continue
            current_pool = reference_pool.subset_by_center_region(
                reference_tissue == tissue_id
            ).filtered_for_size_calibration()
            if sum(current_pool.counts().values()) > 0:
                size_reference_pools_by_tissue[tissue_id] = current_pool
    input_nuclei = input_nuclei.copy()
    erasure_mask = (
        deletion_mask.copy()
        if args.trust_complete_deletion_region
        else expand_edit_mask_to_complete_instances(
            input_nuclei,
            deletion_mask,
        )
    )
    edit_mask |= erasure_mask
    input_nuclei[erasure_mask] = 0

    source_instance_authority = None
    if args.source_instance_authority:
        from inpaint_cells.instance_authority import load_instance_authority

        source_instance_authority = load_instance_authority(
            args.source_instance_authority,
            expected_shape=tissue.shape,
            source_nuclei_raw=population_reference_nuclei_raw,
        )
    calibrated_scales, type_proportions, prior_audit = compute_patch_adaptive_priors(
        reference_nuclei_raw=population_reference_nuclei_raw,
        reference_tissue=reference_tissue,
        density_exclusion_region=deletion_mask,
        target_tissue=tissue,
        generation_region=population_mask,
        library=library,
        global_density_scale=args.density_scale,
        local_density_direct_min_area=args.local_density_direct_min_area,
        local_density_direct_min_count=args.local_density_direct_min_count,
        dataset_name=args.dataset,
        source_instance_authority=source_instance_authority,
    )
    type_prior_weights = confidence_adaptive_type_prior_weights(
        prior_audit,
        maximum_weight=float(args.local_type_prior_weight),
    )
    prior_audit["generation_support"] = {
        "semantic_pixels": semantic_edit_pixels,
        "generation_pixels": int(np.count_nonzero(edit_mask)),
        "placement_pixels": int(np.count_nonzero(placement_mask)),
        "population_target_pixels": int(np.count_nonzero(population_mask)),
        "minimum_width_px": int(args.minimum_mask_width),
        "widening_enabled": bool(args.widen_edit_region),
        "source_nucleus_erasure_policy": (
            "externally_certified_complete_instance_union"
            if args.trust_complete_deletion_region
            else "complete_component_on_any_deletion_region_intersection"
        ),
        "buffer_nucleus_policy": (
            "retain_generation_buffer_only_nuclei_as_placement_obstacles"
        ),
        "population_reference_policy": (
            "complete_source_nuclei_independent_of_filtered_shape_pool"
        ),
        "shape_reference_policy": (
            "separately_filtered_reference_nuclei_shapes"
        ),
    }
    for tissue_id, override in density_scales.items():
        calibrated_scales[tissue_id] = calibrated_scales.get(tissue_id, 1.0) * override

    library_only_tissue_ids = {
        int(value)
        for value in np.unique(
            tissue[
                deletion_mask
                & (reference_tissue != tissue)
            ]
        )
        if int(value) != 0
    }
    prob, type_density = predict_fields(
        model, tissue, input_nuclei, edit_mask, config.cancer_type_index, device
    )
    placement_nuc_prob, placement_type_prob, spatial_prior_audit = (
        predict_context_stabilized_spatial_probability(
            model,
            tissue,
            input_nuclei,
            edit_mask,
            config.cancer_type_index,
            device,
            prob,
            args,
        )
    )
    outputs = []
    diagnostics = []

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gamma_values = parse_float_list(args.gamma_values)
    if not gamma_values or any(value <= 0 for value in gamma_values):
        raise ValueError("gamma-values must contain positive values")
    if int(args.sampling_audit_attempts) < 1:
        raise ValueError("sampling-audit-attempts must be at least one")
    if float(args.sampling_feedback_gamma_min) <= 0:
        raise ValueError("sampling-feedback-gamma-min must be positive")
    if float(args.sampling_feedback_gamma_max) < float(
        args.sampling_feedback_gamma_min
    ):
        raise ValueError("sampling feedback gamma max must be >= gamma min")
    if float(args.sampling_feedback_gamma_down_factor) <= 0 or float(
        args.sampling_feedback_gamma_up_factor
    ) <= 0:
        raise ValueError("sampling feedback gamma factors must be positive")
    if float(args.sampling_feedback_concentration_z_threshold) <= 0:
        raise ValueError("sampling feedback concentration z threshold must be positive")
    for idx, gamma in enumerate(gamma_values):
        initial_gamma = float(gamma)
        attempts = []
        attempt_records = []
        maximum_attempts = max(1, int(args.sampling_audit_attempts))
        current_gamma = initial_gamma
        gamma_already_adjusted = False
        previous_failure_reasons = []
        for attempt_index in range(maximum_attempts):
            feedback_parameters = next_sampling_feedback_parameters(
                initial_gamma=initial_gamma,
                current_gamma=current_gamma,
                base_seed=int(args.seed),
                attempt_index=attempt_index,
                previous_failure_reasons=previous_failure_reasons,
                gamma_already_adjusted=gamma_already_adjusted,
                gamma_down_factor=float(args.sampling_feedback_gamma_down_factor),
                gamma_up_factor=float(args.sampling_feedback_gamma_up_factor),
                gamma_min=float(args.sampling_feedback_gamma_min),
                gamma_max=float(args.sampling_feedback_gamma_max),
            )
            attempt_seed = int(feedback_parameters["seed"])
            attempt_gamma = float(feedback_parameters["sampling_gamma"])
            current_gamma = attempt_gamma
            gamma_already_adjusted = bool(
                feedback_parameters["gamma_adjusted"]
            )
            random.seed(attempt_seed)
            np.random.seed(attempt_seed)
            torch.manual_seed(attempt_seed)
            try:
                nuclei, diag = generate_two_stage_for_gamma(
                    prob,
                    tissue,
                    input_nuclei,
                    deletion_mask,
                    placement_mask,
                    library,
                    reference_pool,
                    attempt_gamma,
                    args,
                    calibrated_scales,
                    type_density=type_density,
                    library_only_tissue_ids=library_only_tissue_ids,
                    type_proportions_by_tissue=type_proportions,
                    type_prior_weights_by_tissue=type_prior_weights,
                    placement_nuc_prob=placement_nuc_prob,
                    placement_type_prob=placement_type_prob,
                    population_mask=population_mask,
                    required_center_mask=required_placement_mask,
                    minimum_required_centers=int(
                        args.minimum_required_placements
                    ),
                    maximum_required_centers=(
                        int(args.maximum_required_placements)
                        if int(args.maximum_required_placements) >= 0
                        else None
                    ),
                    required_nucleus_type=args.required_nucleus_type,
                    size_reference_pools_by_tissue=(
                        size_reference_pools_by_tissue
                    ),
                    fallback_size_reference_pool=complete_size_reference_pool,
                    packing_witness=packing_witness,
                )
            except PlacementQuotaError as exc:
                attempt_records.append(
                    {
                        **feedback_parameters,
                        "attempt_index": int(attempt_index),
                        "seed": int(attempt_seed),
                        "stage": "exact_count_placement",
                        "passed": False,
                        "score": 0.0,
                        "evidence_coverage": 0.0,
                        "failure_reasons": ["COUNT_QUOTA_SHORTFALL"],
                        "error": str(exc),
                    }
                )
                previous_failure_reasons = ["COUNT_QUOTA_SHORTFALL"]
                continue
            diag["patch_adaptive_priors"] = prior_audit
            diag["spatial_prior"] = spatial_prior_audit
            audit = probnet_sampling_alignment_audit(
                input_nuclei=input_nuclei,
                output_nuclei=nuclei,
                tissue=tissue,
                generation_region=placement_mask,
                placement_type_prob=placement_type_prob,
                gamma=attempt_gamma,
                generation_diagnostics=diag,
                evaluation_gamma=initial_gamma,
                concentration_z_threshold=float(
                    args.sampling_feedback_concentration_z_threshold
                ),
                exact_required_region=(
                    required_placement_mask
                    if int(args.minimum_required_placements) > 0
                    else None
                ),
                exact_required_count=(
                    max(0, int(args.minimum_required_placements))
                ),
                exact_required_nucleus_type=(
                    int(args.required_nucleus_type)
                    if args.required_nucleus_type is not None
                    and int(args.minimum_required_placements) > 0
                    else None
                ),
            )
            audit["attempt_index"] = attempt_index
            audit["seed"] = attempt_seed
            diag["sampling_audit"] = audit
            attempts.append((attempt_index, nuclei, diag))
            attempt_records.append(
                {
                    **feedback_parameters,
                    "attempt_index": int(attempt_index),
                    "seed": int(attempt_seed),
                    "stage": "sampling_audit",
                    "passed": bool(audit["passed"]),
                    "score": float(audit["score"]),
                    "evidence_coverage": float(audit["evidence_coverage"]),
                    "failure_reasons": list(audit["failure_reasons"]),
                    "primary_failure_reason": audit[
                        "primary_failure_reason"
                    ],
                    "error": None,
                }
            )
            if audit["passed"]:
                break
            previous_failure_reasons = list(audit["failure_reasons"])

        if not attempts:
            placement_errors = "; ".join(
                str(record["error"])
                for record in attempt_records
                if record["stage"] == "exact_count_placement"
            )
            raise RuntimeError(
                "ProbNet exact-count placement failed for every deterministic "
                f"attempt ({maximum_attempts}): {placement_errors}"
            )

        selected_attempt_position = max(
            range(len(attempts)),
            key=lambda position: (
                bool(attempts[position][2]["sampling_audit"]["passed"]),
                float(attempts[position][2]["sampling_audit"]["score"]),
                -attempts[position][0],
            ),
        )
        selected_attempt, nuclei, diag = attempts[selected_attempt_position]
        selected_record = next(
            record
            for record in attempt_records
            if record["stage"] == "sampling_audit"
            and int(record["attempt_index"]) == int(selected_attempt)
        )
        diag["sampling_audit_attempts"] = attempt_records
        diag["sampling_audit_max_attempts"] = maximum_attempts
        diag["sampling_audit_selected_attempt"] = int(selected_attempt)
        diag["sampling_audit_resampled"] = bool(selected_attempt > 0)
        diag["sampling_feedback"] = {
            "policy": SAMPLING_FEEDBACK_POLICY_NAME,
            "initial_gamma": initial_gamma,
            "selected_gamma": float(selected_record["sampling_gamma"]),
            "selected_seed": int(selected_record["seed"]),
            "max_attempts": maximum_attempts,
            "gamma_down_factor": float(
                args.sampling_feedback_gamma_down_factor
            ),
            "gamma_up_factor": float(args.sampling_feedback_gamma_up_factor),
            "gamma_min": float(args.sampling_feedback_gamma_min),
            "gamma_max": float(args.sampling_feedback_gamma_max),
            "concentration_z_threshold": float(
                args.sampling_feedback_concentration_z_threshold
            ),
            "immutable_parameters": [
                "target_count",
                "tissue_and_component_allocation",
                "deletion_and_generation_regions",
                "shape_source_policy",
                "nucleus_spacing_margin_px",
            ],
            "attempts": attempt_records,
            "selected_attempt": int(selected_attempt),
            "resampled": bool(selected_attempt > 0),
            "parameter_adjusted": any(
                record.get("action") in {"decrease_gamma", "increase_gamma"}
                for record in attempt_records
            ),
        }
        diagnostics.append(diag)

        if idx == 0:
            save_path = output_path
        else:
            save_path = output_path.with_name(f"{output_path.stem}_gamma_{safe_name_float(gamma)}{output_path.suffix}")
        save_nuclei_mask(nuclei, str(save_path))
        selected_gamma = float(diag["sampling_feedback"]["selected_gamma"])
        outputs.append((selected_gamma, nuclei))
        source_counts = diag["placed_by_shape_source"]
        print(
            f"gamma={initial_gamma:g}->{selected_gamma:g}: "
            f"placed {diag['placed']} nuclei "
            f"(reference={source_counts['reference']}, library={source_counts['library']}) "
            f"-> {save_path}"
        )

    diagnostics_path = output_path.with_suffix(".diagnostics.json")
    with open(diagnostics_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"Shape sampling diagnostics -> {diagnostics_path}")

    if args.vis_dir:
        vis_dir = Path(args.vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)
        comparison = make_comparison(
            tissue,
            input_nuclei,
            outputs,
            placement_nuc_prob,
            edit_mask,
        )
        cv2.imwrite(str(vis_dir / "gamma_comparison.png"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        raw_heatmap = draw_edit_contour(
            heatmap_rgb(1.0 - prob[0], edit_mask),
            edit_mask,
        )
        cv2.imwrite(
            str(vis_dir / "probnet_heatmap.png"),
            cv2.cvtColor(raw_heatmap, cv2.COLOR_RGB2BGR),
        )
        stabilized_heatmap = draw_edit_contour(
            heatmap_rgb(placement_nuc_prob, edit_mask),
            edit_mask,
        )
        cv2.imwrite(
            str(vis_dir / "stabilized_probnet_heatmap.png"),
            cv2.cvtColor(stabilized_heatmap, cv2.COLOR_RGB2BGR),
        )
        accepted_overlay = make_accepted_centers_overlay(
            placement_nuc_prob,
            input_nuclei,
            outputs[0][1],
            edit_mask,
        )
        cv2.imwrite(
            str(vis_dir / "accepted_centers_overlay.png"),
            cv2.cvtColor(accepted_overlay, cv2.COLOR_RGB2BGR),
        )
        with open(vis_dir / "diagnostics.json", "w") as f:
            json.dump(diagnostics, f, indent=2)

    failed_audits = [
        diag["sampling_audit"]
        for diag in diagnostics
        if not bool((diag.get("sampling_audit") or {}).get("passed"))
    ]
    if bool(args.require_sampling_audit) and failed_audits:
        raise RuntimeError(
            "ProbNet count/type/spatial sampling audit failed after "
            f"{int(args.sampling_audit_attempts)} deterministic attempts; "
            f"best_score={max(float(item['score']) for item in failed_audits):.4f}. "
            f"Diagnostics: {diagnostics_path}"
        )


def discover_batch_samples(data_dir):
    root = Path(data_dir)
    val_dir = root / "val" if (root / "val").is_dir() else root
    gt_tissue = val_dir / "gt_tissue"
    gt_nuclei = val_dir / "gt_nuclei"
    masks = val_dir / "masks"
    if gt_tissue.is_dir() and masks.is_dir():
        samples = []
        for tissue_path in sorted(gt_tissue.glob("*.png")):
            name = tissue_path.name
            nuclei_path = gt_nuclei / name
            mask_path = masks / name
            if nuclei_path.exists() and mask_path.exists():
                samples.append((tissue_path.stem, tissue_path, nuclei_path, mask_path))
        return samples

    samples = []
    for tissue_path in sorted(val_dir.glob("*/tissue_mask.png")):
        sample_dir = tissue_path.parent
        nuclei_path = sample_dir / "nuclei_mask.png"
        mask_path = sample_dir / "edit_mask.png"
        if nuclei_path.exists() and mask_path.exists():
            samples.append((sample_dir.name, tissue_path, nuclei_path, mask_path))
    return samples


def run_batch(args, model, library, config, density_scales, device):
    samples = discover_batch_samples(args.test_dir)
    if args.n > 0:
        samples = samples[:args.n]
    if not samples:
        raise RuntimeError(f"No layered validation samples found in {args.test_dir}")

    output_dir = Path(args.output_dir)
    nuclei_dir = output_dir / "nuclei"
    vis_dir = output_dir / "vis"
    nuclei_dir.mkdir(parents=True, exist_ok=True)
    if args.vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    all_diag = {}
    gamma_values = parse_float_list(args.gamma_values)
    for idx, (name, tissue_path, nuclei_path, mask_path) in enumerate(samples):
        tissue = load_tissue_mask(str(tissue_path))
        gt_nuclei = load_nuclei_mask(str(nuclei_path), remap=True)
        reference_nuclei_raw = load_nuclei_mask(str(nuclei_path), remap=False)
        reference_pool = build_reference_pool(reference_nuclei_raw, args)
        edit_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 128
        deletion_mask = edit_mask.copy()
        semantic_edit_pixels = int(np.count_nonzero(deletion_mask))
        if args.widen_edit_region:
            edit_mask = widen_locally_thin_mask(
                edit_mask,
                tissue != 0,
                minimum_width=args.minimum_mask_width,
            )
        input_nuclei = gt_nuclei.copy()
        erasure_mask = expand_edit_mask_to_complete_instances(
            input_nuclei,
            deletion_mask,
        )
        edit_mask |= erasure_mask
        input_nuclei[erasure_mask] = 0

        calibrated_scales, type_proportions, prior_audit = compute_patch_adaptive_priors(
            reference_nuclei_raw=reference_nuclei_raw,
            reference_tissue=tissue,
            density_exclusion_region=deletion_mask,
            target_tissue=tissue,
            generation_region=edit_mask,
            library=library,
            global_density_scale=args.density_scale,
            local_density_direct_min_area=args.local_density_direct_min_area,
            local_density_direct_min_count=args.local_density_direct_min_count,
            dataset_name=args.dataset,
        )
        type_prior_weights = confidence_adaptive_type_prior_weights(
            prior_audit,
            maximum_weight=float(args.local_type_prior_weight),
        )
        prior_audit["generation_support"] = {
            "semantic_pixels": semantic_edit_pixels,
            "generation_pixels": int(np.count_nonzero(edit_mask)),
            "minimum_width_px": int(args.minimum_mask_width),
            "widening_enabled": bool(args.widen_edit_region),
            "source_nucleus_erasure_policy": (
                "complete_component_on_any_deletion_region_intersection"
            ),
            "buffer_nucleus_policy": (
                "retain_generation_buffer_only_nuclei_as_placement_obstacles"
            ),
        }
        for tissue_id, override in density_scales.items():
            calibrated_scales[tissue_id] = calibrated_scales.get(tissue_id, 1.0) * override

        prob, type_density = predict_fields(
            model, tissue, input_nuclei, edit_mask, config.cancer_type_index, device
        )
        placement_nuc_prob, placement_type_prob, spatial_prior_audit = (
            predict_context_stabilized_spatial_probability(
                model,
                tissue,
                input_nuclei,
                edit_mask,
                config.cancer_type_index,
                device,
                prob,
                args,
            )
        )
        outputs = []
        sample_diag = []
        for gamma in gamma_values:
            nuclei, diag = generate_two_stage_for_gamma(
                prob,
                tissue,
                input_nuclei,
                deletion_mask,
                edit_mask,
                library,
                reference_pool,
                gamma,
                args,
                calibrated_scales,
                type_density=type_density,
                library_only_tissue_ids=(),
                type_proportions_by_tissue=type_proportions,
                type_prior_weights_by_tissue=type_prior_weights,
                placement_nuc_prob=placement_nuc_prob,
                placement_type_prob=placement_type_prob,
            )
            diag["patch_adaptive_priors"] = prior_audit
            diag["spatial_prior"] = spatial_prior_audit
            suffix = "" if len(gamma_values) == 1 else f"_gamma_{safe_name_float(gamma)}"
            out_path = nuclei_dir / f"{name}{suffix}_nuclei.png"
            save_nuclei_mask(nuclei, str(out_path))
            outputs.append((gamma, nuclei))
            sample_diag.append(diag)

        if args.vis_dir:
            comparison = make_comparison(
                tissue,
                input_nuclei,
                outputs,
                placement_nuc_prob,
                edit_mask,
            )
            cv2.imwrite(str(vis_dir / f"{idx:03d}_{name}.png"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

        all_diag[name] = sample_diag
        print(f"[{idx + 1}/{len(samples)}] {name}: " + ", ".join(f"gamma={d['gamma']:g} placed={d['placed']}" for d in sample_diag))

    with open(output_dir / "diagnostics.json", "w") as f:
        json.dump(all_diag, f, indent=2)


def build_parser():
    parser = argparse.ArgumentParser(description="ProbNet-centered Phase 4 nuclei generation")
    parser.add_argument("--dataset", required=True, help="Dataset name: BCSS, PANDA, GlaS, IGNITE, PUMA, ORCA")
    parser.add_argument("--ckpt", required=True, help="ProbNet checkpoint")
    parser.add_argument("--library", required=True, help="Nuclei instance library directory")
    parser.add_argument("--base-ch", type=int, default=64, help="ProbUNet base channels used during training")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test-dir", help="Layered validation dataset directory for batch inference")
    mode.add_argument("--input-tissue", help="Single edited tissue mask PNG")

    parser.add_argument("--input-nuclei", default=None, help="Optional existing nuclei mask for single inference")
    parser.add_argument(
        "--reference-tissue",
        default=None,
        help=(
            "Optional source tissue mask for patch-local density estimation. "
            "Defaults to --input-tissue for legacy callers."
        ),
    )
    parser.add_argument(
        "--reference-nuclei-shapes",
        default=None,
        help=(
            "Full reference nuclei mask used as the first-choice instance-shape pool. "
            "Defaults to --input-nuclei when omitted."
        ),
    )
    parser.add_argument(
        "--source-instance-authority",
        default=None,
        help=(
            "Optional digest-bound source nucleus instance ledger. Joint execution "
            "uses this as the only abundance authority instead of recounting touching "
            "semantic components."
        ),
    )
    parser.add_argument("--edit-region", default=None, help="Single edit region mask PNG")
    parser.add_argument(
        "--population-region",
        default=None,
        help=(
            "Optional target-tissue population accounting region T_pop. "
            "Counts and density are computed from this mask while new "
            "nucleus centers remain restricted to --placement-region. "
            "Defaults to the placement region for legacy callers."
        ),
    )
    parser.add_argument(
        "--placement-region",
        default=None,
        help=(
            "Optional legal nucleus-center region inside --edit-region. "
            "Defaults to --edit-region for backward compatibility."
        ),
    )
    parser.add_argument(
        "--required-placement-region",
        default=None,
        help=(
            "Optional skill-compiled subset of --placement-region that must "
            "receive the first new nucleus centers. It changes placement "
            "allocation only; population abundance is still computed from "
            "--population-region."
        ),
    )
    parser.add_argument(
        "--minimum-required-placements",
        type=int,
        default=0,
        help=(
            "Minimum new centers reserved in --required-placement-region "
            "before the remaining exact T_pop quota is sampled over P."
        ),
    )
    parser.add_argument(
        "--maximum-required-placements",
        type=int,
        default=-1,
        help=(
            "Maximum new centers admitted to --required-placement-region; "
            "equal to the minimum for an exact compiled seam quota."
        ),
    )
    parser.add_argument(
        "--required-nucleus-type",
        type=int,
        default=None,
        help=(
            "Raw CellViT class required for the compiled seam quota."
        ),
    )
    parser.add_argument(
        "--packing-witness",
        default=None,
        help=(
            "Optional executable-contract packing witness used only when the "
            "normal ProbNet-ranked exact placement search exhausts."
        ),
    )
    parser.add_argument(
        "--deletion-region",
        default=None,
        help=(
            "Semantic change support used for destructive instance erasure. "
            "Defaults to --edit-region for legacy callers."
        ),
    )
    parser.add_argument(
        "--trust-complete-deletion-region",
        action="store_true",
        help=(
            "Treat --deletion-region as an externally certified union of "
            "complete instances instead of expanding semantic 8-connect groups."
        ),
    )
    parser.add_argument("--output", default="nuclei_mask.png", help="Single output nuclei mask path")
    parser.add_argument("--output-dir", default="phase4_probnet_generate", help="Batch output directory")
    parser.add_argument("--n", type=int, default=10, help="Batch sample limit; <=0 means all")
    parser.add_argument("--vis-dir", default=None, help="Write gamma comparison PNGs and diagnostics")

    parser.add_argument(
        "--gamma-values",
        default=str(DEFAULT_PROBNET_ODDS_GAMMA),
        help=(
            "Comma-separated exponents for odds-calibrated ProbNet center "
            "sampling"
        ),
    )
    parser.add_argument(
        "--sampling-audit-attempts",
        type=int,
        default=DEFAULT_SAMPLING_FEEDBACK_ATTEMPTS,
        help=(
            "Maximum reason-directed gamma/seed attempts evaluated by the "
            "generic ProbNet count/type/spatial audit."
        ),
    )
    parser.add_argument(
        "--sampling-feedback-gamma-down-factor",
        type=float,
        default=DEFAULT_SAMPLING_FEEDBACK_GAMMA_DOWN_FACTOR,
    )
    parser.add_argument(
        "--sampling-feedback-gamma-up-factor",
        type=float,
        default=DEFAULT_SAMPLING_FEEDBACK_GAMMA_UP_FACTOR,
    )
    parser.add_argument(
        "--sampling-feedback-gamma-min",
        type=float,
        default=DEFAULT_SAMPLING_FEEDBACK_GAMMA_MIN,
    )
    parser.add_argument(
        "--sampling-feedback-gamma-max",
        type=float,
        default=DEFAULT_SAMPLING_FEEDBACK_GAMMA_MAX,
    )
    parser.add_argument(
        "--sampling-feedback-concentration-z-threshold",
        type=float,
        default=DEFAULT_SAMPLING_CONCENTRATION_Z_THRESHOLD,
        help=(
            "Two-sided standard-normal threshold used only to direct a "
            "bounded gamma update after the frozen spatial audit fails."
        ),
    )
    parser.add_argument(
        "--require-sampling-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fail before image generation when no deterministic sampling "
            "attempt passes the patch-relative count/type/spatial audit."
        ),
    )
    parser.add_argument("--prob-count-weight", type=float, default=0.0,
                        help="Frozen default is zero: checkpoint never determines count")
    parser.add_argument("--density-scale", type=float, default=1.0,
                        help="Global semantic density multiplier")
    parser.add_argument("--density-scale-json", default=None,
                        help="Optional JSON mapping tissue_id -> semantic density multiplier")
    parser.add_argument("--expected-nucleus-area", type=float, default=80.0,
                        help="Fallback expected nucleus area in pixels")
    parser.add_argument("--min-count", type=float, default=0.0)
    parser.add_argument("--max-density-per-10k", type=float, default=900.0,
                        help="Absolute count clip: max nuclei per 10k px")
    parser.add_argument("--max-count-factor", type=float, default=2.5,
                        help="If library density exists, cap count at this multiple of library count")
    parser.add_argument("--min-region-area", type=int, default=50)
    parser.add_argument("--type-prob-floor", type=float, default=0.03)
    parser.add_argument(
        "--local-type-prior-weight",
        type=float,
        default=DEFAULT_LOCAL_TYPE_PRIOR_WEIGHT,
        help=(
            "Log-pooling weight for the target-tissue empirical type prior; "
            "the complementary weight retains local ProbNet evidence."
        ),
    )
    parser.add_argument(
        "--local-type-prior-floor",
        type=float,
        default=1e-4,
        help=(
            "Generic smoothing floor applied to supported empirical type "
            "prior entries before log pooling."
        ),
    )
    parser.add_argument(
        "--type-density-head-weight",
        type=float,
        default=1.0,
        help=(
            "Maximum weight of normalized ProbNet density-head type evidence "
            "in the type quota; the remainder uses the target-tissue prior."
        ),
    )
    parser.add_argument(
        "--adaptive-type-density-weight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reduce density-head type weight using head certainty and "
            "agreement with the target-tissue prior."
        ),
    )
    parser.add_argument(
        "--spatial-context-halo-weight",
        type=float,
        default=1.0,
        help=(
            "Weight for a ProbNet pass with retained nuclei cleared within "
            "one nucleus diameter of the edit; 1.0 is the production default."
        ),
    )
    parser.add_argument(
        "--spatial-context-halo-diameter-scale",
        type=float,
        default=1.25,
    )
    parser.add_argument("--spatial-context-halo-min-px", type=int, default=4)
    parser.add_argument("--spatial-context-halo-max-px", type=int, default=24)
    parser.add_argument("--local-density-direct-min-area", type=int, default=20000)
    parser.add_argument("--local-density-direct-min-count", type=int, default=10)
    parser.add_argument("--minimum-mask-width", type=int, default=33)
    parser.add_argument(
        "--widen-edit-region",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Widen locally thin edit branches before erasure and placement.",
    )
    parser.add_argument(
        "--component-aware-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Poisson candidate sampling independently in disconnected components.",
    )
    parser.add_argument(
        "--component-quota-policy",
        choices=("area_largest_remainder", "minimum_one_then_area_largest_remainder"),
        default="area_largest_remainder",
    )
    parser.add_argument(
        "--backfill-failed-placements",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--retry-candidate-multiplier", type=float, default=12.0)
    parser.add_argument("--retry-candidate-floor", type=int, default=64)
    parser.add_argument("--dense-retry-quota-threshold", type=int, default=20)
    parser.add_argument(
        "--dense-retry-occupancy-threshold",
        type=float,
        default=0.12,
    )
    parser.add_argument(
        "--dense-retry-candidate-multiplier",
        type=float,
        default=24.0,
    )
    parser.add_argument("--dense-retry-candidate-floor", type=int, default=128)
    parser.add_argument(
        "--exact-backfill-candidates-per-missing",
        type=int,
        default=128,
        help=(
            "Candidate-center budget multiplier for one exact-count attempt; "
            "the next deterministic seed is used when this budget is exhausted."
        ),
    )
    parser.add_argument(
        "--exact-backfill-candidate-floor",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--exact-backfill-candidate-ceiling",
        type=int,
        default=4096,
    )
    parser.add_argument("--placement-shape-trials", type=int, default=4)
    parser.add_argument("--placement-transform-trials", type=int, default=12)
    parser.add_argument("--dense-placement-shape-trials", type=int, default=6)
    parser.add_argument(
        "--dense-placement-transform-trials",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--placement-retry-scales",
        type=parse_float_list,
        default=(
            1.0,
            1.0,
            1.0,
            0.95,
            0.95,
            0.95,
            0.9,
            0.9,
            0.9,
            0.85,
            0.85,
            0.85,
            0.8,
            0.8,
            0.8,
            0.8,
        ),
    )
    parser.add_argument("--placement-center-jitter-max", type=int, default=8)
    parser.add_argument(
        "--require-full-tissue-containment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require every generated nucleus pixel to remain in valid "
            "biological tissue; rejected shapes continue through retries."
        ),
    )
    parser.add_argument(
        "--max-nucleus-overlap-fraction",
        type=float,
        default=0.0,
        help=(
            "Maximum proposed-nucleus area allowed to overlap an existing "
            "nucleus. Frozen production sampling uses 0.0 to preserve retained "
            "source nuclei bitwise."
        ),
    )
    parser.add_argument(
        "--nucleus-spacing-margin-px",
        type=int,
        default=1,
        help=(
            "Minimum empty-pixel margin between retained/generated nuclei. "
            "One pixel prevents same-class instances from merging."
        ),
    )
    parser.add_argument(
        "--require-exact-target-count",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fail instead of silently returning fewer generated nuclei than "
            "the target-tissue density quota."
        ),
    )

    parser.add_argument("--min-distance-mode", choices=["adaptive", "fixed"], default="adaptive")
    parser.add_argument("--min-distance", type=float, default=8.0,
                        help="Fixed Poisson distance when --min-distance-mode=fixed")
    parser.add_argument("--min-distance-scale", type=float, default=0.75,
                        help="Adaptive distance = nucleus_diameter * scale before oversample shrinking")
    parser.add_argument("--min-distance-min", type=float, default=4.0)
    parser.add_argument("--min-distance-max", type=float, default=18.0)
    parser.add_argument("--min-distance-floor", type=float, default=3.0)
    parser.add_argument("--shrink-distance-for-oversample", action="store_true", default=True)
    parser.add_argument("--no-shrink-distance-for-oversample", dest="shrink_distance_for_oversample",
                        action="store_false")
    parser.add_argument("--oversample-base", type=float, default=3.0)
    parser.add_argument("--oversample-gamma-scale", type=float, default=0.35)
    parser.add_argument("--oversample-min", type=float, default=1.5)
    parser.add_argument("--oversample-max", type=float, default=8.0)
    parser.add_argument("--poisson-attempts", type=int, default=30)
    parser.add_argument(
        "--quota-coverage-spacing-scale",
        type=float,
        default=0.75,
        help=(
            "Generic primary-prefix spacing as a fraction of "
            "sqrt(tissue area / target count)."
        ),
    )
    parser.add_argument(
        "--quota-coverage-max-radius",
        type=float,
        default=48.0,
        help="Maximum generic spacing radius for the primary quota prefix.",
    )
    parser.add_argument(
        "--adaptive-quota-coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use component-local ProbNet tail sharpness to choose how much "
            "of the exact quota receives quality-diversity coverage."
        ),
    )
    parser.add_argument(
        "--quota-coverage-min-fraction",
        type=float,
        default=0.2,
        help=(
            "Minimum quota fraction assigned to the coverage prefix when the "
            "ProbNet high-score tail is maximally sharp."
        ),
    )
    parser.add_argument("--skip-tissue-ids", type=int, nargs="*", default=[],
                        help="Additional tissue IDs to skip")
    parser.add_argument(
        "--allowed-nucleus-types",
        type=int,
        nargs="+",
        default=list(NUCLEI_CLASSES),
        help=(
            "Raw CellViT nucleus IDs permitted for new placements. Existing "
            "retained nuclei are unaffected."
        ),
    )
    parser.add_argument("--no-augment-instances", action="store_true")
    parser.add_argument("--reference-shape-min-area", type=int, default=8)
    parser.add_argument(
        "--reference-shape-max-area-ratio",
        type=float,
        default=0.0,
        help="Reject same-class components above this multiple of median area; <=0 disables.",
    )
    parser.add_argument(
        "--include-border-reference-shapes",
        action="store_true",
        help="Allow clipped nuclei touching the reference patch boundary.",
    )
    parser.add_argument(
        "--disable-library-size-calibration",
        action="store_true",
        help=(
            "Do not resize library fallback nuclei to the current patch's "
            "same-class reference area distribution."
        ),
    )
    parser.add_argument("--library-size-min-scale", type=float, default=0.5)
    parser.add_argument("--library-size-max-scale", type=float, default=2.0)
    parser.add_argument(
        "--library-size-log-area-jitter",
        type=float,
        default=0.05,
        help="Log-space area jitter after empirical same-class area sampling.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.test_dir and args.source_instance_authority:
        raise ValueError(
            "--source-instance-authority is single-case provenance and cannot be "
            "reused across --test-dir; provide one authority ledger per sample"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = get_config(args.dataset)
    args.skip_tissue_ids = set(args.skip_tissue_ids) | set(config.skip_tissues)
    density_scales = load_density_scale(args.density_scale_json)

    if args.input_tissue and not args.edit_region:
        raise ValueError("--edit-region is required with --input-tissue")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Dataset: {config.name} ({config.cancer_type}), cancer_id={config.cancer_type_index}")
    print(f"Device: {device}")
    print(f"Gamma values: {parse_float_list(args.gamma_values)}")
    print("Loading ProbNet...")
    model = load_checkpoint_model(args.ckpt, device, args.base_ch)
    print("Loading nuclei instance library...")
    library = NucleiLibrary(args.library, dataset=config.name)

    if args.test_dir:
        run_batch(args, model, library, config, density_scales, device)
    else:
        run_single(args, model, library, config, density_scales, device)


if __name__ == "__main__":
    main()

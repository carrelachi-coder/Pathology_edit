#!/usr/bin/env python3
"""
Unified ProbNet-centered nuclei mask generation.

This is the Phase 4 inference entry point. The frozen ProbNet checkpoint has
one inference role: its scalar P(nucleus) = 1 - P(background) field weights
spatial landing positions. Counts/densities and exact type quotas are derived
from the current patch and dataset tissue priors, independently of checkpoint
density/count/type heads.

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
from scipy import ndimage
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_config import get_config
from inpaint_cells.data.density_targets import select_instances_by_centroid
from inpaint_cells.models.prob_unet import ProbUNet
from inpaint_cells.nuclei_library.library import (
    NucleiLibrary,
    ReferenceFirstNucleiSampler,
    ReferenceNucleiInstancePool,
    place_nucleus_layered,
)
from inpaint_cells.utils.mask_utils import (
    NUM_NUCLEI,
    NUCLEI_CLASSES,
    load_tissue_mask,
    load_nuclei_mask,
    overlay,
    save_nuclei_mask,
)
from inpaint_cells.sampling_policy import (
    retry_pool_target,
    retry_transform_specs,
    valid_biological_tissue_mask,
    widen_locally_thin_mask,
)


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
    return int(round(clipped)), {
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


def choose_weighted_centers(
    candidates,
    nuc_prob,
    target_count,
    gamma,
    *,
    coverage_count=None,
    coverage_radius=0.0,
):
    """Build a ProbNet-ranked queue with an optional spatially covered prefix.

    The complete retry tail remains in stable descending ProbNet-score order.
    Only the primary quota prefix defers candidates that are too close to an
    already selected higher-score point. This prevents a narrow high-score band
    from consuming an entire tissue quota while preserving ProbNet as the
    spatial ranking source.
    """

    if target_count <= 0 or not candidates:
        return []
    n = min(target_count, len(candidates))
    ys = np.array([p[0] for p in candidates], dtype=np.int64)
    xs = np.array([p[1] for p in candidates], dtype=np.int64)
    scores = np.power(np.clip(nuc_prob[ys, xs], 0.0, 1.0), gamma)
    score_order = np.argsort(-scores, kind="stable")

    prefix_target = min(
        n,
        int(coverage_count) if coverage_count is not None else 0,
    )
    radius = float(coverage_radius)
    if prefix_target <= 1 or radius <= 0:
        return [candidates[int(i)] for i in score_order[:n]]

    radius_sq = radius * radius
    prefix_indices = []
    for candidate_index in score_order:
        candidate_index = int(candidate_index)
        y = int(ys[candidate_index])
        x = int(xs[candidate_index])
        if any(
            (y - int(ys[selected_index])) ** 2
            + (x - int(xs[selected_index])) ** 2
            < radius_sq
            for selected_index in prefix_indices
        ):
            continue
        prefix_indices.append(candidate_index)
        if len(prefix_indices) >= prefix_target:
            break

    prefix_set = set(prefix_indices)
    retry_tail = [
        int(candidate_index)
        for candidate_index in score_order
        if int(candidate_index) not in prefix_set
    ]
    queue = prefix_indices + retry_tail
    return [candidates[candidate_index] for candidate_index in queue[:n]]


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
):
    """Freeze count and type priors independently of the ProbNet checkpoint.

    A sufficiently large patch-local observation is used directly.  Sparse
    observations shrink toward the dataset tissue-density prior.  Nucleus type
    quotas use the reliable patch-local distribution, otherwise the dataset
    library distribution.  The checkpoint is therefore responsible only for
    the scalar spatial placement field ``P(nucleus)``.
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
    local_counts = {}
    local_counts_by_type = {}
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
            if density_exclusion_region[row, col]:
                continue
            tissue_id = int(reference_tissue[row, col])
            if tissue_id == 0:
                continue
            local_counts[tissue_id] = local_counts.get(tissue_id, 0) + 1
            tissue_counts = local_counts_by_type.setdefault(tissue_id, {})
            tissue_counts[raw_type] = tissue_counts.get(raw_type, 0) + 1

    density_scales = {}
    type_proportions = {}
    tissue_audit = {}
    for tissue_id_value in np.unique(target_tissue[generation_region]):
        tissue_id = int(tissue_id_value)
        if tissue_id == 0:
            continue
        observed_region = (
            (reference_tissue == tissue_id) & ~density_exclusion_region
        )
        observed_area = int(np.count_nonzero(observed_region))
        local_count = int(local_counts.get(tissue_id, 0))
        local_type_counts = local_counts_by_type.get(tissue_id, {})
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
        effective_confidence = 1.0 if local_is_reliable else raw_area_confidence
        target_density = (
            effective_confidence * local_density
            + (1.0 - effective_confidence) * dataset_density
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
            "unedited_reference_area_px": observed_area,
            "patch_area_px": patch_area,
            "raw_area_confidence": raw_area_confidence,
            "effective_local_confidence": effective_confidence,
            "density_mode": (
                "patch_local_direct_reliable"
                if local_is_reliable
                else "area_confidence_patch_local_dataset_shrinkage"
            ),
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
        "checkpoint_role": "spatial_placement_probability_only",
        "count_policy": "reliable patch-local density else area-weighted dataset shrinkage",
        "type_policy": "reliable patch-local quota else dataset tissue prior",
        "nucleus_count_rule": "class-component centroid outside density exclusion region",
        "tissues": tissue_audit,
    }
    return density_scales, type_proportions, audit


def sample_type_at_center(prob, cy, cx, args):
    type_probs = prob[1:, cy, cx].astype(np.float64)
    total = type_probs.sum()
    if total < args.type_prob_floor:
        return None
    type_probs = type_probs / total
    idx = int(np.random.choice(len(type_probs), p=type_probs))
    return NUCLEI_CLASSES[idx]


def build_reference_pool(nuclei_mask, args):
    if nuclei_mask is None:
        return None
    return ReferenceNucleiInstancePool.from_mask(
        nuclei_mask,
        min_area=args.reference_shape_min_area,
        exclude_border=not args.include_border_reference_shapes,
        max_area_ratio_to_median=args.reference_shape_max_area_ratio,
    )


def sample_instance_for_center(sampler, tissue_id, nuc_type):
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
    args,
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
    for _ in range(max(1, shape_trials)):
        instance, shape_source = sample_instance_for_center(
            shape_sampler,
            tissue_id,
            nucleus_type,
        )
        if instance is None:
            break
        for spec in retry_transform_specs(args, trial_count=transform_trials):
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
            placed = place_nucleus_layered(
                output,
                center_y,
                center_x,
                instance,
                augment=not args.no_augment_instances,
                max_overlap_fraction=float(args.max_nucleus_overlap_fraction),
                valid_tissue_mask=valid_tissue_mask,
                require_full_tissue_containment=bool(
                    args.require_full_tissue_containment
                ),
                rotation_quarters=int(spec["rotation_quarters"]),
                flip_horizontal=bool(spec["flip_horizontal"]),
                flip_vertical=bool(spec["flip_vertical"]),
                scale=float(spec["scale"]),
            )
            if placed:
                return True, str(shape_source), attempts, (center_y, center_x)
    return False, None, attempts, None


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
    clear_edit_mask=True,
    type_proportions_by_tissue=None,
):
    nuc_prob = 1.0 - prob[0]
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

    diagnostics = {
        "gamma": gamma,
        "max_nucleus_overlap_fraction": float(
            getattr(args, "max_nucleus_overlap_fraction", 0.0)
        ),
        "require_full_tissue_containment": bool(
            getattr(args, "require_full_tissue_containment", True)
        ),
        "full_shape_tissue_policy": (
            "hard_reject_outside_valid_biological_tissue_then_retry"
        ),
        "placed": 0,
        "placed_by_shape_source": {"reference": 0, "library": 0},
        "reference_pool": reference_pool.describe() if reference_pool is not None else None,
        "tissues": {},
    }

    for tissue_id in np.unique(tissue[edit_mask]):
        tissue_id = int(tissue_id)
        if tissue_id in args.skip_tissue_ids:
            continue

        tissue_region = edit_mask & (tissue == tissue_id)
        if tissue_region.sum() < args.min_region_area:
            continue

        expected_area = weighted_mean_area(library, tissue_id, args.expected_nucleus_area)
        scale = density_scales.get(tissue_id, args.density_scale)
        class_counts = None
        if density is None:
            target_count, count_info = compute_target_count(
                nuc_prob, tissue_region, tissue_id, library, expected_area, args, scale
            )
        else:
            expected_by_class = density[:, tissue_region].sum(axis=1) * scale
            expected_total = float(expected_by_class.sum())
            max_allowed = args.max_density_per_10k * tissue_region.sum() / 10000.0
            target_count = int(round(float(np.clip(expected_total, args.min_count, max_allowed))))
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
                "region_area": int(tissue_region.sum()),
                "count_source": "center_density_integral",
                "density_scale": float(scale),
                "expected_count": expected_total,
                "expected_by_class": expected_by_class.tolist(),
                "target_by_class": class_counts.tolist(),
                "clipped_count": target_count,
            }

        type_limits = None
        if density is None and type_proportions_by_tissue:
            local_type_proportions = type_proportions_by_tissue.get(tissue_id)
            if local_type_proportions:
                type_limits = allocate_type_counts(
                    local_type_proportions,
                    target_count,
                )

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
        component_limits = {0: target_count}
        component_dense_retry = {0: False}
        component_sampling = None
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
                # largest-remainder allocation so the requested tissue count
                # is never silently dropped by a component-size threshold.
                minimum_component_area = 1
                component_quota_policy = getattr(
                    args,
                    "component_quota_policy",
                    "minimum_one_then_area_largest_remainder",
                )
                if component_quota_policy == "area_largest_remainder":
                    component_limits = allocate_area_proportional_counts(
                        component_areas,
                        target_count,
                        minimum_component_area,
                    )
                else:
                    component_limits = allocate_component_counts(
                        component_areas,
                        target_count,
                        minimum_component_area,
                    )
                centers = []
                center_component_ids = []
                candidates = []
                component_sampling = {}
                for component_id, area in component_areas:
                    quota = int(component_limits.get(component_id, 0))
                    if quota <= 0:
                        continue
                    component_region = component_labels == component_id
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
                    coverage_radius = quota_coverage_radius(
                        region_area=area,
                        quota=quota,
                        candidate_min_distance=min_distance,
                        spacing_scale=float(
                            getattr(args, "quota_coverage_spacing_scale", 0.75)
                        ),
                        maximum=float(
                            getattr(args, "quota_coverage_max_radius", 48.0)
                        ),
                    )
                    selected = choose_weighted_centers(
                        component_candidates,
                        nuc_prob,
                        requested,
                        gamma,
                        coverage_count=quota,
                        coverage_radius=coverage_radius,
                    )
                    candidates.extend(component_candidates)
                    centers.extend(selected)
                    center_component_ids.extend([component_id] * len(selected))
                    component_sampling[str(component_id)] = {
                        "area": int(area),
                        "quota": quota,
                        "dense_retry": dense_retry,
                        "expected_occupancy_fraction": expected_occupancy,
                        "retry_pool_target": retry_pool_size,
                        "num_candidates": len(component_candidates),
                        "selected_centers": len(selected),
                        "attempted_centers": 0,
                        "placed": 0,
                        "coverage_prefix_target": quota,
                        "coverage_radius": coverage_radius,
                    }
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
                coverage_radius = quota_coverage_radius(
                    region_area=int(np.count_nonzero(tissue_region)),
                    quota=target_count,
                    candidate_min_distance=min_distance,
                    spacing_scale=float(
                        getattr(args, "quota_coverage_spacing_scale", 0.75)
                    ),
                    maximum=float(
                        getattr(args, "quota_coverage_max_radius", 48.0)
                    ),
                )
                centers = choose_weighted_centers(
                    candidates,
                    nuc_prob,
                    requested_centers,
                    gamma,
                    coverage_count=target_count,
                    coverage_radius=coverage_radius,
                )
                center_component_ids = [0] * len(centers)
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
        placed_by_component = {component_id: 0 for component_id in component_limits}
        placed_by_type = {
            int(nuc_type): 0
            for nuc_type in (type_limits or {})
        }
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
            elif type_limits is not None:
                nuc_type = sample_type_with_remaining_quota(
                    type_limits,
                    placed_by_type,
                )
            else:
                nuc_type = sample_type_at_center(prob, cy, cx, args)
            if nuc_type is None:
                continue
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
                shape_sampler=shape_sampler,
                center_region=tissue_region,
                valid_tissue_mask=valid_tissue_mask,
                dense_retry=dense_retry,
                args=args,
            )
            placement_trials += int(local_trials)
            if placed_ok:
                placed += 1
                accepted_y, accepted_x = accepted_center
                accepted_center_probabilities.append(
                    float(nuc_prob[accepted_y, accepted_x])
                )
                placed_by_component[component_id] = (
                    placed_by_component.get(component_id, 0) + 1
                )
                if component_sampling is not None:
                    component_sampling[str(component_id)]["placed"] += 1
                if type_limits is not None:
                    placed_by_type[nuc_type] = placed_by_type.get(nuc_type, 0) + 1
                placed_by_shape_source[str(shape_source)] += 1

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
                "probnet_score_descending_with_quota_coverage_prefix"
            ),
            "quota_coverage_spacing_scale": float(
                getattr(args, "quota_coverage_spacing_scale", 0.75)
            ),
            "quota_coverage_max_radius": float(
                getattr(args, "quota_coverage_max_radius", 48.0)
            ),
            "retry_tail_policy": "stable_descending_probnet_score",
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
            "type_quota_policy": (
                "empirical_exact_quota_checkpoint_independent"
                if type_limits is not None
                else "probnet_per_center_sampling"
            ),
            "target_by_type": (
                {str(key): int(value) for key, value in type_limits.items()}
                if type_limits is not None
                else None
            ),
            "placed_by_type": (
                {str(key): int(value) for key, value in placed_by_type.items()}
                if type_limits is not None
                else None
            ),
            "component_sampling": component_sampling,
        }

    diagnostics["shape_sampling"] = shape_sampler.diagnostics()
    return output, diagnostics


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
    semantic_edit_pixels = int(np.count_nonzero(edit_mask))
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
        input_nuclei = load_nuclei_mask(args.input_nuclei, remap=True)
    else:
        input_nuclei = np.zeros_like(tissue, dtype=np.int64)
    input_nuclei = input_nuclei.copy()
    erasure_mask = select_instances_by_centroid(input_nuclei, edit_mask)
    input_nuclei[erasure_mask] = 0

    calibrated_scales, type_proportions, prior_audit = compute_patch_adaptive_priors(
        reference_nuclei_raw=reference_nuclei_raw,
        reference_tissue=reference_tissue,
        density_exclusion_region=edit_mask,
        target_tissue=tissue,
        generation_region=edit_mask,
        library=library,
        global_density_scale=args.density_scale,
        local_density_direct_min_area=args.local_density_direct_min_area,
        local_density_direct_min_count=args.local_density_direct_min_count,
    )
    prior_audit["generation_support"] = {
        "semantic_pixels": semantic_edit_pixels,
        "generation_pixels": int(np.count_nonzero(edit_mask)),
        "minimum_width_px": int(args.minimum_mask_width),
        "widening_enabled": bool(args.widen_edit_region),
        "source_nucleus_erasure_policy": (
            "complete_component_if_centroid_in_support"
        ),
    }
    for tissue_id, override in density_scales.items():
        calibrated_scales[tissue_id] = calibrated_scales.get(tissue_id, 1.0) * override

    prob = predict_prob(
        model, tissue, input_nuclei, edit_mask, config.cancer_type_index, device
    )
    outputs = []
    diagnostics = []

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gamma_values = parse_float_list(args.gamma_values)
    for idx, gamma in enumerate(gamma_values):
        nuclei, diag = generate_for_gamma(
            prob,
            tissue,
            input_nuclei,
            edit_mask,
            library,
            reference_pool,
            gamma,
            args,
            calibrated_scales,
            type_proportions_by_tissue=type_proportions,
        )
        diag["patch_adaptive_priors"] = prior_audit
        diagnostics.append(diag)

        if idx == 0:
            save_path = output_path
        else:
            save_path = output_path.with_name(f"{output_path.stem}_gamma_{safe_name_float(gamma)}{output_path.suffix}")
        save_nuclei_mask(nuclei, str(save_path))
        outputs.append((gamma, nuclei))
        source_counts = diag["placed_by_shape_source"]
        print(
            f"gamma={gamma:g}: placed {diag['placed']} nuclei "
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
        comparison = make_comparison(tissue, input_nuclei, outputs, 1.0 - prob[0], edit_mask)
        cv2.imwrite(str(vis_dir / "gamma_comparison.png"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        with open(vis_dir / "diagnostics.json", "w") as f:
            json.dump(diagnostics, f, indent=2)


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
        semantic_edit_pixels = int(np.count_nonzero(edit_mask))
        if args.widen_edit_region:
            edit_mask = widen_locally_thin_mask(
                edit_mask,
                tissue != 0,
                minimum_width=args.minimum_mask_width,
            )
        input_nuclei = gt_nuclei.copy()
        erasure_mask = select_instances_by_centroid(input_nuclei, edit_mask)
        input_nuclei[erasure_mask] = 0

        calibrated_scales, type_proportions, prior_audit = compute_patch_adaptive_priors(
            reference_nuclei_raw=reference_nuclei_raw,
            reference_tissue=tissue,
            density_exclusion_region=edit_mask,
            target_tissue=tissue,
            generation_region=edit_mask,
            library=library,
            global_density_scale=args.density_scale,
            local_density_direct_min_area=args.local_density_direct_min_area,
            local_density_direct_min_count=args.local_density_direct_min_count,
        )
        prior_audit["generation_support"] = {
            "semantic_pixels": semantic_edit_pixels,
            "generation_pixels": int(np.count_nonzero(edit_mask)),
            "minimum_width_px": int(args.minimum_mask_width),
            "widening_enabled": bool(args.widen_edit_region),
            "source_nucleus_erasure_policy": (
                "complete_component_if_centroid_in_support"
            ),
        }
        for tissue_id, override in density_scales.items():
            calibrated_scales[tissue_id] = calibrated_scales.get(tissue_id, 1.0) * override

        prob = predict_prob(
            model, tissue, input_nuclei, edit_mask, config.cancer_type_index, device
        )
        outputs = []
        sample_diag = []
        for gamma in gamma_values:
            nuclei, diag = generate_for_gamma(
                prob,
                tissue,
                input_nuclei,
                edit_mask,
                library,
                reference_pool,
                gamma,
                args,
                calibrated_scales,
                type_proportions_by_tissue=type_proportions,
            )
            diag["patch_adaptive_priors"] = prior_audit
            suffix = "" if len(gamma_values) == 1 else f"_gamma_{safe_name_float(gamma)}"
            out_path = nuclei_dir / f"{name}{suffix}_nuclei.png"
            save_nuclei_mask(nuclei, str(out_path))
            outputs.append((gamma, nuclei))
            sample_diag.append(diag)

        if args.vis_dir:
            comparison = make_comparison(tissue, input_nuclei, outputs, 1.0 - prob[0], edit_mask)
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
    parser.add_argument("--edit-region", default=None, help="Single edit region mask PNG")
    parser.add_argument("--output", default="nuclei_mask.png", help="Single output nuclei mask path")
    parser.add_argument("--output-dir", default="phase4_probnet_generate", help="Batch output directory")
    parser.add_argument("--n", type=int, default=10, help="Batch sample limit; <=0 means all")
    parser.add_argument("--vis-dir", default=None, help="Write gamma comparison PNGs and diagnostics")

    parser.add_argument("--gamma-values", default="1.5",
                        help="Comma-separated gamma values for weighted center sampling")
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
    parser.add_argument("--skip-tissue-ids", type=int, nargs="*", default=[],
                        help="Additional tissue IDs to skip")
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

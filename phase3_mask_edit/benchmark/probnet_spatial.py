"""Frozen ProbNet spatial-only benchmark primitives.

The checkpoint contributes only ``P(nucleus)``. Counts, nucleus types,
component quotas, candidate pools, shape plans, and retry transforms are
constructed independently and shared by every spatial sampler.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import random
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

from inpaint_cells.generate import (
    adaptive_min_distance,
    allocate_area_proportional_counts,
    allocate_type_counts,
    compute_target_count,
    poisson_candidates,
    supplement_retry_candidates,
    weighted_mean_area,
)
from inpaint_cells.nuclei_library.library import ReferenceFirstNucleiSampler
from inpaint_cells.sampling_policy import (
    retry_pool_target,
    valid_biological_tissue_mask,
)
from phase3_mask_edit.benchmark.probnet_compact import (
    CanonicalInstance,
    _retry_transform_specs,
    place_instance_recorded,
    stable_seed,
)


SAMPLERS = ("probnet", "poisson_only", "uniform", "boundary_distance")
RAW_TYPES = (101, 102, 103, 104, 105)
MPP = 0.25


@dataclass(frozen=True)
class PlannedSlot:
    """One exact requested nucleus independent of candidate ordering."""

    slot_id: str
    tissue_id: int
    component_id: int
    raw_type: int


@dataclass
class ShapeTrial:
    """One frozen shape and its frozen retry-transform sequence."""

    instance: Mapping[str, Any]
    source: str
    transforms: list[dict[str, Any]]


def frozen_sampler_args(
    *, skip_tissue_ids: Sequence[int] = ()
) -> SimpleNamespace:
    """Return the production placement constants frozen in the benchmark plan."""

    return SimpleNamespace(
        expected_nucleus_area=80.0,
        prob_count_weight=0.0,
        density_scale=1.0,
        min_count=0.0,
        max_density_per_10k=900.0,
        max_count_factor=2.5,
        min_region_area=50,
        min_distance_mode="adaptive",
        min_distance=8.0,
        min_distance_scale=0.75,
        min_distance_min=4.0,
        min_distance_max=18.0,
        min_distance_floor=3.0,
        shrink_distance_for_oversample=True,
        oversample_base=3.0,
        oversample_gamma_scale=0.35,
        oversample_min=1.5,
        oversample_max=8.0,
        poisson_attempts=30,
        skip_tissue_ids=set(int(value) for value in skip_tissue_ids),
        disable_library_size_calibration=False,
        library_size_min_scale=0.5,
        library_size_max_scale=2.0,
        library_size_log_area_jitter=0.05,
        placement_shape_trials=4,
        placement_transform_trials=12,
        dense_placement_shape_trials=6,
        dense_placement_transform_trials=24,
        placement_retry_scales=(
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
        placement_center_jitter_max=8,
        retry_candidate_multiplier=12.0,
        retry_candidate_floor=64,
        dense_retry_quota_threshold=20,
        dense_retry_occupancy_threshold=0.12,
        dense_retry_candidate_multiplier=24.0,
        dense_retry_candidate_floor=128,
        max_nucleus_overlap_fraction=0.0,
        backfill_failed_placements=True,
        require_full_tissue_containment=True,
        component_quota_policy="area_largest_remainder",
        no_augment_instances=False,
        local_density_direct_min_area=20000,
        local_density_direct_min_count=10,
    )


def raw_to_internal(values: np.ndarray) -> np.ndarray:
    """Convert persisted 101--105 labels to the model's internal 1--5 labels."""

    result = np.zeros(values.shape, dtype=np.int64)
    if int(values.max(initial=0)) <= 5:
        return values.astype(np.int64, copy=True)
    for class_index, raw_type in enumerate(RAW_TYPES, start=1):
        result[values == raw_type] = class_index
    return result


def instances_with_centres_in_region(
    instances: Sequence[CanonicalInstance],
    region: np.ndarray,
    tissue_map: np.ndarray,
    skip_tissue_ids: Sequence[int],
) -> list[CanonicalInstance]:
    """Select complete instances by centroid, retaining boundary crossers whole."""

    skipped = set(int(value) for value in skip_tissue_ids)
    selected: list[CanonicalInstance] = []
    height, width = region.shape
    for instance in instances:
        center_x, center_y = instance.centroid_xy
        row = int(np.clip(round(center_y), 0, height - 1))
        col = int(np.clip(round(center_x), 0, width - 1))
        if region[row, col] and int(tissue_map[row, col]) not in skipped:
            selected.append(instance)
    return selected


def erase_complete_instances(
    nuclei_internal: np.ndarray,
    instances: Sequence[CanonicalInstance],
) -> np.ndarray:
    """Erase selected full components without clearing unrelated support pixels."""

    result = nuclei_internal.copy()
    for instance in instances:
        result[instance.mask] = 0
    return result


def _exact_type_slots(
    quotas: Mapping[int, int], *, seed: int
) -> list[int]:
    values = [
        int(raw_type)
        for raw_type, count in sorted(quotas.items())
        for _ in range(int(count))
    ]
    rng = np.random.default_rng(seed)
    if values:
        values = [values[int(index)] for index in rng.permutation(len(values))]
    return values


def build_oracle_plan(
    *,
    tissue_map: np.ndarray,
    generation_region: np.ndarray,
    hidden_instances: Sequence[CanonicalInstance],
    library: Any,
    args: SimpleNamespace,
    seed: int,
) -> tuple[dict[str, Any], list[PlannedSlot], dict[int, np.ndarray]]:
    """Freeze P1 oracle count/type quotas and area-based component quotas."""

    instances_by_tissue: dict[int, list[CanonicalInstance]] = defaultdict(list)
    height, width = tissue_map.shape
    for instance in hidden_instances:
        center_x, center_y = instance.centroid_xy
        row = int(np.clip(round(center_y), 0, height - 1))
        col = int(np.clip(round(center_x), 0, width - 1))
        tissue_id = int(tissue_map[row, col])
        if tissue_id not in args.skip_tissue_ids and tissue_id != 0:
            instances_by_tissue[tissue_id].append(instance)

    plan: dict[str, Any] = {
        "count_source": "oracle_hidden_instances",
        "type_source": "oracle_hidden_instance_types",
        "component_quota_policy": "area_largest_remainder",
        "target_count": 0,
        "tissues": {},
    }
    slots: list[PlannedSlot] = []
    component_labels_by_tissue: dict[int, np.ndarray] = {}
    for tissue_id, tissue_instances in sorted(instances_by_tissue.items()):
        target_count = len(tissue_instances)
        if target_count <= 0:
            continue
        tissue_region = generation_region & (tissue_map == tissue_id)
        labels, component_count = ndimage.label(
            tissue_region, structure=np.ones((3, 3), dtype=np.uint8)
        )
        expected_area = weighted_mean_area(
            library, tissue_id, args.expected_nucleus_area
        )
        minimum_component_area = 1
        component_areas = [
            (component_id, int(np.count_nonzero(labels == component_id)))
            for component_id in range(1, component_count + 1)
        ]
        component_quotas = allocate_area_proportional_counts(
            component_areas, target_count, minimum_component_area
        )
        allocated_count = int(sum(component_quotas.values()))
        if allocated_count != target_count:
            raise RuntimeError(
                f"Component allocation lost quota for tissue {tissue_id}: "
                f"{allocated_count}/{target_count}"
            )
        type_quotas = dict(
            Counter(int(instance.raw_type) for instance in tissue_instances)
        )
        type_slots = _exact_type_slots(
            type_quotas, seed=stable_seed(seed, tissue_id, "type_slots")
        )
        cursor = 0
        for component_id, component_quota in sorted(component_quotas.items()):
            for component_slot in range(int(component_quota)):
                raw_type = int(type_slots[cursor])
                slots.append(
                    PlannedSlot(
                        slot_id=(
                            f"t{tissue_id:02d}_c{component_id:03d}_"
                            f"s{component_slot:04d}"
                        ),
                        tissue_id=tissue_id,
                        component_id=int(component_id),
                        raw_type=raw_type,
                    )
                )
                cursor += 1
        component_labels_by_tissue[tissue_id] = labels
        plan["target_count"] += target_count
        plan["tissues"][str(tissue_id)] = {
            "target_count": target_count,
            "target_by_type": {
                str(key): int(value) for key, value in sorted(type_quotas.items())
            },
            "target_by_component": {
                str(key): int(value)
                for key, value in sorted(component_quotas.items())
            },
            "expected_nucleus_area": float(expected_area),
            "minimum_component_area": minimum_component_area,
        }
    return plan, slots, component_labels_by_tissue


def build_statistical_plan(
    *,
    tissue_map: np.ndarray,
    generation_region: np.ndarray,
    density_scales: Mapping[int, float],
    type_proportions_by_tissue: Mapping[int, Mapping[int, float]],
    library: Any,
    args: SimpleNamespace,
    seed: int,
) -> tuple[dict[str, Any], list[PlannedSlot], dict[int, np.ndarray]]:
    """Freeze production statistical counts and exact type/component quotas."""

    plan: dict[str, Any] = {
        "count_source": (
            "reliable_patch_local_density_else_area_weighted_profile_shrinkage"
        ),
        "type_source": (
            "reliable_patch_local_quota_else_profile_tissue_distribution"
        ),
        "component_quota_policy": "area_largest_remainder",
        "target_count": 0,
        "tissues": {},
    }
    slots: list[PlannedSlot] = []
    component_labels_by_tissue: dict[int, np.ndarray] = {}
    nucleus_probability_placeholder = np.zeros(tissue_map.shape, dtype=np.float32)
    for tissue_id_value in np.unique(tissue_map[generation_region]):
        tissue_id = int(tissue_id_value)
        if tissue_id == 0 or tissue_id in args.skip_tissue_ids:
            continue
        tissue_region = generation_region & (tissue_map == tissue_id)
        if int(tissue_region.sum()) < int(args.min_region_area):
            continue
        expected_area = weighted_mean_area(
            library, tissue_id, args.expected_nucleus_area
        )
        scale = float(density_scales.get(tissue_id, args.density_scale))
        target_count, count_info = compute_target_count(
            nucleus_probability_placeholder,
            tissue_region,
            tissue_id,
            library,
            expected_area,
            args,
            scale,
        )
        if target_count <= 0:
            continue
        type_proportions = type_proportions_by_tissue.get(tissue_id, {})
        type_quotas = allocate_type_counts(type_proportions, target_count)
        if sum(type_quotas.values()) != target_count:
            raise RuntimeError(
                f"Type allocation lost quota for tissue {tissue_id}: "
                f"{sum(type_quotas.values())}/{target_count}"
            )
        labels, component_count = ndimage.label(
            tissue_region, structure=np.ones((3, 3), dtype=np.uint8)
        )
        minimum_component_area = 1
        component_areas = [
            (component_id, int(np.count_nonzero(labels == component_id)))
            for component_id in range(1, component_count + 1)
        ]
        component_quotas = allocate_area_proportional_counts(
            component_areas, target_count, minimum_component_area
        )
        if sum(component_quotas.values()) != target_count:
            raise RuntimeError(
                f"Component allocation lost quota for tissue {tissue_id}: "
                f"{sum(component_quotas.values())}/{target_count}"
            )
        type_slots = _exact_type_slots(
            type_quotas, seed=stable_seed(seed, tissue_id, "type_slots")
        )
        cursor = 0
        for component_id, component_quota in sorted(component_quotas.items()):
            for component_slot in range(int(component_quota)):
                slots.append(
                    PlannedSlot(
                        slot_id=(
                            f"t{tissue_id:02d}_c{component_id:03d}_"
                            f"s{component_slot:04d}"
                        ),
                        tissue_id=tissue_id,
                        component_id=int(component_id),
                        raw_type=int(type_slots[cursor]),
                    )
                )
                cursor += 1
        component_labels_by_tissue[tissue_id] = labels
        plan["target_count"] += target_count
        plan["tissues"][str(tissue_id)] = {
            **count_info,
            "target_count": target_count,
            "target_by_type": {
                str(key): int(value) for key, value in sorted(type_quotas.items())
            },
            "target_by_component": {
                str(key): int(value)
                for key, value in sorted(component_quotas.items())
            },
            "expected_nucleus_area": float(expected_area),
            "minimum_component_area": minimum_component_area,
        }
    return plan, slots, component_labels_by_tissue


def _freeze_shape_trials(
    *,
    slots: Sequence[PlannedSlot],
    library: Any,
    reference_pool: Any,
    args: SimpleNamespace,
    dense_components: set[tuple[int, int]],
    seed: int,
) -> tuple[dict[str, list[ShapeTrial]], dict[str, Any]]:
    random.seed(stable_seed(seed, "shared_shape_plan"))
    np.random.seed(stable_seed(seed, "shared_shape_plan_numpy"))
    sampler = ReferenceFirstNucleiSampler(
        library,
        reference_pool,
        calibrate_library_size=not args.disable_library_size_calibration,
        library_size_min_scale=args.library_size_min_scale,
        library_size_max_scale=args.library_size_max_scale,
        library_size_log_area_jitter=args.library_size_log_area_jitter,
    )
    result: dict[str, list[ShapeTrial]] = {}
    unavailable = 0
    dense_slots = 0
    for slot in slots:
        is_dense = (slot.tissue_id, slot.component_id) in dense_components
        if is_dense:
            dense_slots += 1
        shape_trial_count = int(
            args.dense_placement_shape_trials
            if is_dense
            else args.placement_shape_trials
        )
        transform_trial_count = int(
            args.dense_placement_transform_trials
            if is_dense
            else args.placement_transform_trials
        )
        trials: list[ShapeTrial] = []
        for _ in range(max(1, shape_trial_count)):
            instance, source = sampler.sample_instance(
                slot.tissue_id,
                slot.raw_type,
                allow_cross_tissue=True,
            )
            if instance is None:
                unavailable += 1
                break
            if int(instance.get("type", 0)) != slot.raw_type:
                raise RuntimeError("Shape sampler violated the exact-class contract")
            trials.append(
                ShapeTrial(
                    instance=instance,
                    source=str(source),
                    transforms=_retry_transform_specs(
                        args,
                        trial_count=transform_trial_count,
                    ),
                )
            )
        result[slot.slot_id] = trials
    diagnostics = sampler.diagnostics()
    diagnostics["unavailable_shape_trials"] = unavailable
    diagnostics["dense_slot_count"] = dense_slots
    diagnostics["ordinary_slot_count"] = len(slots) - dense_slots
    return result, diagnostics


def _candidate_pools(
    *,
    plan: Mapping[str, Any],
    component_labels_by_tissue: Mapping[int, np.ndarray],
    library: Any,
    args: SimpleNamespace,
    gamma: float,
    seed: int,
) -> tuple[
    dict[tuple[int, int], list[tuple[int, int]]],
    dict[tuple[int, int], dict[str, Any]],
]:
    result: dict[tuple[int, int], list[tuple[int, int]]] = {}
    audit: dict[tuple[int, int], dict[str, Any]] = {}
    for tissue_id_text, tissue_plan in sorted(plan["tissues"].items()):
        tissue_id = int(tissue_id_text)
        expected_area = float(tissue_plan["expected_nucleus_area"])
        oversample = args.oversample_base * (
            1.0 + args.oversample_gamma_scale * max(float(gamma) - 1.0, 0.0)
        )
        oversample = float(
            np.clip(oversample, args.oversample_min, args.oversample_max)
        )
        min_distance = adaptive_min_distance(expected_area, args, oversample)
        labels = component_labels_by_tissue[tissue_id]
        for component_id_text, quota_value in sorted(
            tissue_plan["target_by_component"].items(),
            key=lambda item: int(item[0]),
        ):
            component_id = int(component_id_text)
            quota = int(quota_value)
            component_seed = stable_seed(
                seed, tissue_id, component_id, "shared_candidate_pool"
            )
            random.seed(component_seed)
            np.random.seed(component_seed)
            component_region = labels == component_id
            candidates = poisson_candidates(
                component_region, min_distance, args.poisson_attempts
            )
            minimum_candidates, dense, occupancy = retry_pool_target(
                quota=quota,
                component_area=int(np.count_nonzero(component_region)),
                expected_nucleus_area=expected_area,
                args=args,
            )
            candidates = supplement_retry_candidates(
                candidates, component_region, minimum_candidates
            )
            result[(tissue_id, component_id)] = [
                (int(row), int(col)) for row, col in candidates
            ]
            audit[(tissue_id, component_id)] = {
                "quota": quota,
                "component_area": int(np.count_nonzero(component_region)),
                "expected_nucleus_area": expected_area,
                "expected_occupancy_fraction": occupancy,
                "dense_retry": dense,
                "retry_pool_target": minimum_candidates,
                "candidate_pool_size": len(candidates),
            }
    return result, audit


def _ordered_candidates(
    candidates: Sequence[tuple[int, int]],
    *,
    sampler: str,
    nucleus_probability: np.ndarray,
    component_region: np.ndarray,
    gamma: float,
    seed: int,
) -> list[tuple[int, int]]:
    if sampler not in SAMPLERS:
        raise ValueError(f"Unknown spatial sampler: {sampler}")
    values = list(candidates)
    if not values:
        return []
    rng = np.random.default_rng(seed)
    if sampler == "poisson_only":
        return values
    if sampler == "uniform":
        return [values[int(index)] for index in rng.permutation(len(values))]
    if sampler == "boundary_distance":
        distances = ndimage.distance_transform_edt(component_region)
        tie = rng.random(len(values))
        return [
            value
            for _, _, value in sorted(
                (
                    -float(distances[row, col]),
                    float(tie[index]),
                    (row, col),
                )
                for index, (row, col) in enumerate(values)
            )
        ]
    rows = np.asarray([value[0] for value in values], dtype=np.int64)
    cols = np.asarray([value[1] for value in values], dtype=np.int64)
    weights = np.power(
        np.clip(nucleus_probability[rows, cols], 0.0, 1.0), float(gamma)
    ) + 1e-12
    gumbels = rng.gumbel(size=len(values))
    keys = np.log(weights) + gumbels
    order = np.argsort(-keys, kind="stable")
    return [values[int(index)] for index in order]


def _try_slot(
    nuclei_map: np.ndarray,
    *,
    slot: PlannedSlot,
    candidate_y: int,
    candidate_x: int,
    shape_trials: Sequence[ShapeTrial],
    center_region: np.ndarray,
    generation_region: np.ndarray,
    valid_tissue_mask: np.ndarray,
    require_full_tissue_containment: bool,
) -> tuple[dict[str, Any] | None, Counter[str], int]:
    rejections: Counter[str] = Counter()
    placement_trials = 0
    for shape_index, shape_trial in enumerate(shape_trials):
        for transform_index, spec in enumerate(shape_trial.transforms):
            offset_y, offset_x = spec["offset_yx"]
            center_y = int(candidate_y + offset_y)
            center_x = int(candidate_x + offset_x)
            placement_trials += 1
            if (
                center_y < 0
                or center_y >= center_region.shape[0]
                or center_x < 0
                or center_x >= center_region.shape[1]
                or not bool(center_region[center_y, center_x])
            ):
                rejections["jitter_center_outside_component"] += 1
                continue
            record = place_instance_recorded(
                nuclei_map,
                center_y=center_y,
                center_x=center_x,
                instance=shape_trial.instance,
                shape_source=shape_trial.source,
                edit_mask=generation_region,
                valid_tissue_mask=valid_tissue_mask,
                augment=True,
                instance_id=slot.slot_id,
                rotation_quarters=int(spec["rotation_quarters"]),
                flip_horizontal=bool(spec["flip_horizontal"]),
                flip_vertical=bool(spec["flip_vertical"]),
                scale=float(spec["scale"]),
                require_full_tissue_containment=require_full_tissue_containment,
            )
            if record["placement_status"] == "accepted":
                record.update(
                    {
                        "candidate_center_xy": [
                            int(candidate_x),
                            int(candidate_y),
                        ],
                        "quota_tissue_id": slot.tissue_id,
                        "quota_component_id": slot.component_id,
                        "quota_raw_type": slot.raw_type,
                        "shape_trial_index": shape_index,
                        "transform_trial_index": transform_index,
                        "placement_trial_count": placement_trials,
                        "trial_rejection_counts": dict(sorted(rejections.items())),
                    }
                )
                return record, rejections, placement_trials
            rejections[str(record.get("rejection_reason") or "unknown")] += 1
    return None, rejections, placement_trials


def generate_fixed_plan_layout(
    *,
    probability: np.ndarray,
    tissue_map: np.ndarray,
    input_nuclei: np.ndarray,
    generation_region: np.ndarray,
    plan: Mapping[str, Any],
    slots: Sequence[PlannedSlot],
    component_labels_by_tissue: Mapping[int, np.ndarray],
    library: Any,
    reference_pool: Any,
    sampler: str,
    gamma: float,
    args: SimpleNamespace,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Place one layout while changing only candidate ordering/selection."""

    if probability.shape[0] < 1:
        raise ValueError("Probability tensor must contain a background channel")
    max_overlap_fraction = float(
        getattr(args, "max_nucleus_overlap_fraction", 0.0)
    )
    if max_overlap_fraction != 0.0:
        raise ValueError(
            "Frozen spatial layout requires max_nucleus_overlap_fraction=0.0"
        )
    if not bool(getattr(args, "backfill_failed_placements", True)):
        raise ValueError(
            "Frozen spatial layout requires retry-pool backfilling"
        )
    nucleus_probability = 1.0 - probability[0]
    output = input_nuclei.copy()
    retained = input_nuclei > 0
    shared_pools, retry_pool_audit = _candidate_pools(
        plan=plan,
        component_labels_by_tissue=component_labels_by_tissue,
        library=library,
        args=args,
        gamma=gamma,
        seed=seed,
    )
    dense_components = {
        key
        for key, value in retry_pool_audit.items()
        if bool(value["dense_retry"])
    }
    shape_trials, shape_diagnostics = _freeze_shape_trials(
        slots=slots,
        library=library,
        reference_pool=reference_pool,
        args=args,
        dense_components=dense_components,
        seed=seed,
    )
    valid_tissue = valid_biological_tissue_mask(
        tissue_map,
        args.skip_tissue_ids,
    )
    ordered: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for key, candidates in shared_pools.items():
        tissue_id, component_id = key
        ordered[key] = _ordered_candidates(
            candidates,
            sampler=sampler,
            nucleus_probability=nucleus_probability,
            component_region=component_labels_by_tissue[tissue_id] == component_id,
            gamma=gamma,
            seed=stable_seed(seed, sampler, tissue_id, component_id),
        )
    cursor: dict[tuple[int, int], int] = defaultdict(int)
    records: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    placement_trials = 0
    for slot in slots:
        key = (slot.tissue_id, slot.component_id)
        candidates = ordered.get(key, [])
        accepted: dict[str, Any] | None = None
        while cursor[key] < len(candidates) and accepted is None:
            candidate_y, candidate_x = candidates[cursor[key]]
            cursor[key] += 1
            accepted, local_rejections, local_trials = _try_slot(
                output,
                slot=slot,
                candidate_y=candidate_y,
                candidate_x=candidate_x,
                shape_trials=shape_trials.get(slot.slot_id, []),
                center_region=(
                    component_labels_by_tissue[slot.tissue_id]
                    == slot.component_id
                ),
                generation_region=generation_region,
                valid_tissue_mask=valid_tissue,
                require_full_tissue_containment=bool(
                    args.require_full_tissue_containment
                ),
            )
            rejection_counts.update(local_rejections)
            placement_trials += local_trials
        if accepted is None:
            records.append(
                {
                    "instance_id": slot.slot_id,
                    "cell_type": slot.raw_type,
                    "quota_tissue_id": slot.tissue_id,
                    "quota_component_id": slot.component_id,
                    "quota_raw_type": slot.raw_type,
                    "placement_status": "rejected",
                    "rejection_reason": (
                        "shape_unavailable"
                        if not shape_trials.get(slot.slot_id)
                        else "candidate_pool_exhausted"
                    ),
                }
            )
        else:
            records.append(accepted)

    accepted = [
        record for record in records if record["placement_status"] == "accepted"
    ]
    requested_by_type = Counter(slot.raw_type for slot in slots)
    placed_by_type = Counter(int(record["cell_type"]) for record in accepted)
    requested_by_component = Counter(
        (slot.tissue_id, slot.component_id) for slot in slots
    )
    placed_by_component = Counter(
        (int(record["quota_tissue_id"]), int(record["quota_component_id"]))
        for record in accepted
    )
    retained_unchanged = bool(
        np.array_equal(output[retained], input_nuclei[retained])
    )
    diagnostics = {
        "checkpoint_role": "spatial_placement_probability_only",
        "sampler": sampler,
        "gamma": float(gamma),
        "max_nucleus_overlap_fraction": max_overlap_fraction,
        "overlap_rejection_policy": "hard_reject_then_retry_next_candidate",
        "full_shape_tissue_policy": (
            "hard_reject_outside_valid_biological_tissue_then_retry"
        ),
        "require_full_tissue_containment": bool(
            args.require_full_tissue_containment
        ),
        "retry_pool_backfill": True,
        "requested": len(slots),
        "placed": len(accepted),
        "unfilled": len(slots) - len(accepted),
        "placement_completion": (
            len(accepted) / len(slots) if slots else 1.0
        ),
        "exact_type_quota": requested_by_type == placed_by_type,
        "exact_component_quota": requested_by_component == placed_by_component,
        "requested_by_type": {
            str(key): int(value) for key, value in sorted(requested_by_type.items())
        },
        "placed_by_type": {
            str(key): int(value) for key, value in sorted(placed_by_type.items())
        },
        "placement_trials": placement_trials,
        "candidate_pool_size": int(sum(map(len, shared_pools.values()))),
        "used_candidate_count": int(sum(cursor.values())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "outside_tissue_pixels": int(
            sum(int(record.get("outside_tissue_pixels", 0)) for record in accepted)
        ),
        "overlap_pixels": int(
            sum(int(record.get("overlap_pixels", 0)) for record in accepted)
        ),
        "retained_input_nuclei_unchanged": retained_unchanged,
        "shape_sampling": shape_diagnostics,
        "retry_pool_by_component": {
            f"{tissue_id}:{component_id}": value
            for (tissue_id, component_id), value in sorted(
                retry_pool_audit.items()
            )
        },
    }
    if not retained_unchanged:
        raise RuntimeError("Spatial benchmark overwrote retained input nuclei")
    return output, records, diagnostics


def _accepted_points(
    records: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    accepted = [
        record for record in records if record.get("placement_status") == "accepted"
    ]
    points = np.asarray(
        [record["center_xy"] for record in accepted], dtype=np.float64
    ).reshape((-1, 2))
    types = np.asarray(
        [int(record["cell_type"]) for record in accepted], dtype=np.int64
    )
    return points, types


def _target_points(
    instances: Sequence[CanonicalInstance],
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        [instance.centroid_xy for instance in instances], dtype=np.float64
    ).reshape((-1, 2))
    types = np.asarray(
        [int(instance.raw_type) for instance in instances], dtype=np.int64
    )
    return points, types


def _nnd(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.asarray([], dtype=np.float64)
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1) * MPP


def _point_f1(
    target_points: np.ndarray,
    target_types: np.ndarray,
    predicted_points: np.ndarray,
    predicted_types: np.ndarray,
    *,
    tolerance_um: float,
    class_aware: bool,
) -> float:
    if len(target_points) == 0 and len(predicted_points) == 0:
        return 1.0
    if len(target_points) == 0 or len(predicted_points) == 0:
        return 0.0
    distances = cdist(target_points, predicted_points) * MPP
    if class_aware:
        distances[target_types[:, None] != predicted_types[None, :]] = np.inf
    finite = np.isfinite(distances)
    cost = np.where(finite, distances, tolerance_um + 1e6)
    rows, cols = linear_sum_assignment(cost)
    true_positive = int(np.count_nonzero(distances[rows, cols] <= tolerance_um))
    return float(
        2 * true_positive / (len(target_points) + len(predicted_points))
    )


def _ripley_k(points: np.ndarray, area_pixels: int, radii_um: np.ndarray) -> np.ndarray:
    if len(points) < 2 or area_pixels <= 0:
        return np.full(radii_um.shape, np.nan, dtype=np.float64)
    distances_um = cdist(points, points) * MPP
    np.fill_diagonal(distances_um, np.inf)
    denominator = float(len(points) * (len(points) - 1))
    area_um2 = float(area_pixels) * MPP * MPP
    return np.asarray(
        [
            area_um2
            * float(np.count_nonzero(distances_um <= radius))
            / denominator
            for radius in radii_um
        ],
        dtype=np.float64,
    )


def spatial_metrics(
    *,
    hidden_instances: Sequence[CanonicalInstance],
    placement_records: Sequence[Mapping[str, Any]],
    generation_region: np.ndarray,
    tissue_map: np.ndarray,
    component_labels_by_tissue: Mapping[int, np.ndarray],
) -> dict[str, Any]:
    """Compute the predeclared P1 spatial and descriptive recovery endpoints."""

    target_points, target_types = _target_points(hidden_instances)
    predicted_points, predicted_types = _accepted_points(placement_records)
    target_nnd = _nnd(target_points)
    predicted_nnd = _nnd(predicted_points)
    nnd_w1 = (
        float(wasserstein_distance(target_nnd, predicted_nnd))
        if target_nnd.size and predicted_nnd.size
        else float("nan")
    )

    boundary_distance = ndimage.distance_transform_edt(generation_region) * MPP

    def sample_distance(points: np.ndarray) -> np.ndarray:
        if not len(points):
            return np.asarray([], dtype=np.float64)
        cols = np.clip(np.rint(points[:, 0]).astype(int), 0, tissue_map.shape[1] - 1)
        rows = np.clip(np.rint(points[:, 1]).astype(int), 0, tissue_map.shape[0] - 1)
        return boundary_distance[rows, cols]

    target_boundary = sample_distance(target_points)
    predicted_boundary = sample_distance(predicted_points)
    boundary_w1 = (
        float(wasserstein_distance(target_boundary, predicted_boundary))
        if target_boundary.size and predicted_boundary.size
        else float("nan")
    )

    radii_um = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
    target_k = _ripley_k(
        target_points, int(np.count_nonzero(generation_region)), radii_um
    )
    predicted_k = _ripley_k(
        predicted_points, int(np.count_nonzero(generation_region)), radii_um
    )
    normalization = np.pi * np.square(radii_um)
    ripley_error = (
        float(np.nanmean(np.abs(target_k - predicted_k) / normalization))
        if np.any(np.isfinite(target_k)) and np.any(np.isfinite(predicted_k))
        else float("nan")
    )

    def component_counts(points: np.ndarray) -> Counter[tuple[int, int]]:
        counts: Counter[tuple[int, int]] = Counter()
        for center_x, center_y in points:
            row = int(np.clip(round(center_y), 0, tissue_map.shape[0] - 1))
            col = int(np.clip(round(center_x), 0, tissue_map.shape[1] - 1))
            tissue_id = int(tissue_map[row, col])
            labels = component_labels_by_tissue.get(tissue_id)
            component_id = int(labels[row, col]) if labels is not None else 0
            counts[(tissue_id, component_id)] += 1
        return counts

    target_components = component_counts(target_points)
    predicted_components = component_counts(predicted_points)
    keys = set(target_components) | set(predicted_components)
    occupancy_error = float(
        sum(
            abs(target_components.get(key, 0) - predicted_components.get(key, 0))
            for key in keys
        )
        / max(len(target_points), 1)
    )
    return {
        "target_count": int(len(target_points)),
        "generated_count": int(len(predicted_points)),
        "nnd_w1_um": nnd_w1,
        "ripley_k_normalized_l1": ripley_error,
        "boundary_distance_w1_um": boundary_w1,
        "component_occupancy_l1_per_target": occupancy_error,
        "point_f1_4um": _point_f1(
            target_points,
            target_types,
            predicted_points,
            predicted_types,
            tolerance_um=4.0,
            class_aware=False,
        ),
        "class_aware_point_f1_4um": _point_f1(
            target_points,
            target_types,
            predicted_points,
            predicted_types,
            tolerance_um=4.0,
            class_aware=True,
        ),
    }

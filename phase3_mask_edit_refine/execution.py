"""Compile Planner intent into a topology-safe executable pixel contract."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy import ndimage

from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.candidates import compile_depth_profile_map
from phase3_mask_edit_refine.models import (
    EditPlan,
    PlannedInterface,
    RefineContractError,
    ResolvedAreaContract,
)
from phase3_mask_edit_refine.scene import SceneAnalysis
from phase3_mask_edit_refine.topology import (
    protected_narrow_necks,
    source_deletion_limit,
    topology_safe_priority_grow,
)

EXECUTION_SOLVER_VERSION = "mask-edit-refine-topology-solver-v5"


@dataclass(frozen=True)
class _CompilerWork:
    planned: PlannedInterface
    anchor_masks: tuple[np.ndarray, ...]
    anchor_mask: np.ndarray
    legal_source: np.ndarray
    priority: np.ndarray
    source_component: np.ndarray
    item_capacity_px: int
    source_deletion_limit_px: int
    protected_source_necks: np.ndarray


@dataclass(frozen=True)
class CompiledReplayPart:
    """One interface part reproduced by the authoritative topology solver."""

    planned: PlannedInterface
    anchor_masks: tuple[np.ndarray, ...]
    anchor_mask: np.ndarray
    change_region: np.ndarray
    legal_capacity_px: int
    source_deletion_limit_px: int
    protected_source_necks: np.ndarray
    topology_audit: dict[str, Any]


def compile_edit_plan(
    plan: EditPlan,
    *,
    source_mask: np.ndarray,
    schema: MaskProfileSchema,
    scene: SceneAnalysis,
) -> tuple[EditPlan, dict[str, Any]]:
    compiled, audit, _parts, _replay = compile_edit_plan_with_witness(
        plan,
        source_mask=source_mask,
        schema=schema,
        scene=scene,
    )
    return compiled, audit


def compile_edit_plan_with_witness(
    plan: EditPlan,
    *,
    source_mask: np.ndarray,
    schema: MaskProfileSchema,
    scene: SceneAnalysis,
) -> tuple[
    EditPlan,
    dict[str, Any],
    tuple[CompiledReplayPart, ...],
    dict[str, Any],
]:
    """Resolve area, allocations, and depth with topology constraints up front.

    The desired area remains immutable.  For a ranged task whose fallback
    policy explicitly permits it, the solver returns the largest topology-safe
    realization it can construct at or below the desired value.  It never goes
    below the task's hard minimum and never expands to an unselected interface.
    """

    mask = np.asarray(source_mask)
    source_ids = tuple(
        int(value)
        for value in plan.tool_program.parameter_ranges.get(
            "editable_source_fine_ids", ()
        )
    ) or tuple(
        fine_id
        for label in plan.source_labels
        for fine_id in schema.resolve_fine_ids(label)
    )
    target_ids = tuple(schema.resolve_fine_ids(plan.target_label))
    source_region = np.isin(mask, source_ids)
    target_region = np.isin(mask, target_ids)
    topology_policy = _topology_policy(plan)
    desired_pixels = plan.area_budget.target_pixels(mask, source_region)
    hard_min_pixels, hard_max_pixels = plan.area_budget.hard_pixel_interval(
        mask, source_region
    )
    desired_pixels = min(desired_pixels, hard_max_pixels)

    works = _prepare_compiler_work(
        plan,
        source_mask=mask,
        source_region=source_region,
        scene=scene,
    )
    if not works:
        raise RefineContractError(
            "execution solver found no legal pixels on the selected interfaces"
        )

    requested_weights = np.asarray(
        [item.planned.execution_contract.area_allocation_fraction for item in works],
        dtype=float,
    )
    (
        _allocations,
        selected_by_work,
        grow_audits,
        topology_audit,
        search_audit,
    ) = _resolve_topology_safe_area(
        works,
        desired_pixels=desired_pixels,
        hard_min_pixels=hard_min_pixels,
        weights=requested_weights,
        source_region=source_region,
        target_region=target_region,
        scene=scene,
        fallback_policy=plan.area_budget.fallback_policy,
        **topology_policy,
    )
    realized_allocations = np.asarray(
        [int(np.count_nonzero(item)) for item in selected_by_work], dtype=int
    )
    resolved_pixels = int(realized_allocations.sum())
    if resolved_pixels < desired_pixels:
        if plan.area_budget.fallback_policy != "max_feasible_below_target":
            raise RefineContractError(
                "exact area is not topology-safe on the selected interfaces: "
                f"desired={desired_pixels}, maximum_safe={resolved_pixels}"
            )
        if resolved_pixels < hard_min_pixels:
            raise RefineContractError(
                "maximum topology-safe area is below the task hard minimum: "
                f"minimum={hard_min_pixels}, maximum_safe={resolved_pixels}"
            )

    active_indices = [
        index for index, value in enumerate(realized_allocations.tolist()) if value > 0
    ]
    if not active_indices:
        raise RefineContractError("execution solver produced no topology-safe edit pixels")

    binding_constraint = _binding_constraint(
        desired_pixels=desired_pixels,
        resolved_pixels=resolved_pixels,
        works=works,
        audits=grow_audits,
    )
    resolved_area = ResolvedAreaContract(
        desired_pixels=int(desired_pixels),
        hard_min_pixels=int(hard_min_pixels),
        hard_max_pixels=int(hard_max_pixels),
        resolved_pixels=int(resolved_pixels),
        fallback_policy=plan.area_budget.fallback_policy,
        used_fallback=resolved_pixels < desired_pixels,
        binding_constraint=binding_constraint,
        solver_version=EXECUTION_SOLVER_VERSION,
    )

    compiled_interfaces: list[PlannedInterface] = []
    audit_interfaces: list[dict[str, Any]] = []
    for index, work in enumerate(works):
        allocated_pixels = int(realized_allocations[index])
        requested = work.planned.execution_contract.depth_profile
        selected = selected_by_work[index]
        if allocated_pixels <= 0:
            audit_interfaces.append(
                {
                    "interface_id": work.planned.interface_id,
                    "status": "dropped_zero_safe_allocation",
                    "requested_allocation_fraction": float(requested_weights[index]),
                    "legal_capacity_pixels": int(work.item_capacity_px),
                    "topology_grow": grow_audits[index],
                }
            )
            continue

        required_scale = work.priority[selected]
        resolved_peak = float(max(1.0, np.max(required_scale)))
        band_max = float(work.planned.allowed_edit_band_px[1])
        resolved_peak = min(band_max, resolved_peak * 1.001)
        edge_ratio = requested.edge_depth_px / max(requested.peak_depth_px, 1e-6)
        noise_ratio = requested.noise_amplitude_px / max(
            requested.peak_depth_px, 1e-6
        )
        compiled_profile = replace(
            requested,
            peak_depth_px=resolved_peak,
            edge_depth_px=float(np.clip(edge_ratio, 0.0, 1.0)) * resolved_peak,
            noise_amplitude_px=min(noise_ratio * resolved_peak, resolved_peak),
        )
        compiled_execution = replace(
            work.planned.execution_contract,
            area_allocation_fraction=allocated_pixels / max(resolved_pixels, 1),
            depth_profile=compiled_profile,
        )
        compiled_interfaces.append(
            replace(work.planned, execution_contract=compiled_execution)
        )
        audit_interfaces.append(
            {
                "interface_id": work.planned.interface_id,
                "status": "compiled",
                "anchor_segment_ids": list(
                    work.planned.execution_contract.anchor_segment_ids
                ),
                "requested_allocation_fraction": float(requested_weights[index]),
                "compiled_allocation_fraction": allocated_pixels
                / max(resolved_pixels, 1),
                "allocated_pixels": allocated_pixels,
                "legal_capacity_pixels": int(work.item_capacity_px),
                "source_deletion_limit_pixels": int(
                    work.source_deletion_limit_px
                ),
                "requested_peak_depth_px": requested.peak_depth_px,
                "compiled_peak_depth_px": resolved_peak,
                "edge_to_peak_ratio": edge_ratio,
                "noise_to_peak_ratio": noise_ratio,
                "allowed_band_px": [
                    float(work.planned.allowed_edit_band_px[0]),
                    band_max,
                ],
                "topology_grow": grow_audits[index],
            }
        )

    compiled_plan = replace(
        plan,
        candidate_interfaces=tuple(compiled_interfaces),
        resolved_area=resolved_area,
    )
    compiled_by_id = {
        item.interface_id: item for item in compiled_interfaces
    }
    witness_parts = tuple(
        CompiledReplayPart(
            planned=compiled_by_id[work.planned.interface_id],
            anchor_masks=work.anchor_masks,
            anchor_mask=work.anchor_mask,
            change_region=np.asarray(selected, dtype=bool),
            legal_capacity_px=work.item_capacity_px,
            source_deletion_limit_px=work.source_deletion_limit_px,
            protected_source_necks=work.protected_source_necks,
            topology_audit=dict(grow_audit),
        )
        for work, selected, grow_audit in zip(
            works, selected_by_work, grow_audits
        )
        if work.planned.interface_id in compiled_by_id and np.any(selected)
    )
    witness_realized = sum(
        int(np.count_nonzero(item.change_region)) for item in witness_parts
    )
    if witness_realized != resolved_pixels or not topology_audit["passed"]:
        raise RefineContractError(
            "execution solver produced an invalid reusable topology witness"
        )
    replay_audit = {
        "replay_version": EXECUTION_SOLVER_VERSION,
        "resolved_pixels": int(resolved_pixels),
        "realized_pixels": int(witness_realized),
        "whole_mask_topology": topology_audit,
        "reused_from_compiler": True,
    }
    return compiled_plan, {
        "compiler_version": EXECUTION_SOLVER_VERSION,
        "desired_pixels": int(desired_pixels),
        "hard_allowed_pixels": [int(hard_min_pixels), int(hard_max_pixels)],
        "resolved_pixels": int(resolved_pixels),
        "used_fallback": bool(resolved_area.used_fallback),
        "fallback_policy": plan.area_budget.fallback_policy,
        "binding_constraint": binding_constraint,
        "whole_mask_topology": topology_audit,
        "area_search": search_audit,
        # Backward-compatible audit key used by existing evaluation scripts.
        "target_pixels": int(resolved_pixels),
        "interfaces": audit_interfaces,
        "reusable_topology_witness": True,
    }, witness_parts, replay_audit


def replay_compiled_edit_plan(
    plan: EditPlan,
    *,
    source_mask: np.ndarray,
    schema: MaskProfileSchema,
    scene: SceneAnalysis,
) -> tuple[tuple[CompiledReplayPart, ...], dict[str, Any]]:
    """Replay a compiled plan with the same owner and topology state machine.

    Candidate generation used to contain a second approximation of compiler
    ownership and growth.  Multi-interface plans could therefore compile but
    fail to draw.  This replay is the executable baseline candidate: it uses
    the compiled per-interface peaks, allocation fractions and the exact
    topology solver that certified the plan.
    """

    if plan.resolved_area is None:
        raise RefineContractError("only a compiled EditPlan can be replayed")
    mask = np.asarray(source_mask)
    source_ids = tuple(
        int(value)
        for value in plan.tool_program.parameter_ranges.get(
            "editable_source_fine_ids", ()
        )
    ) or tuple(
        fine_id
        for label in plan.source_labels
        for fine_id in schema.resolve_fine_ids(label)
    )
    target_ids = tuple(schema.resolve_fine_ids(plan.target_label))
    source_region = np.isin(mask, source_ids)
    target_region = np.isin(mask, target_ids)
    prepared = _prepare_compiler_work(
        plan,
        source_mask=mask,
        source_region=source_region,
        scene=scene,
    )
    # The initial compiler explores the full allowed band and then records the
    # maximum priority actually used as compiled peak. Replay must honor that
    # recorded envelope rather than reopening the full band.
    works = tuple(
        replace(
            work,
            legal_source=(
                work.legal_source
                & (
                    work.priority
                    <= work.planned.execution_contract.depth_profile.peak_depth_px
                    * 1.001
                    + 1e-9
                )
            ),
            item_capacity_px=int(
                np.count_nonzero(
                    work.legal_source
                    & (
                        work.priority
                        <= work.planned.execution_contract.depth_profile.peak_depth_px
                        * 1.001
                        + 1e-9
                    )
                )
            ),
        )
        for work in prepared
    )
    works = tuple(work for work in works if work.item_capacity_px > 0)
    if not works:
        raise RefineContractError("compiled plan replay has no legal pixels")
    weights = np.asarray(
        [work.planned.execution_contract.area_allocation_fraction for work in works],
        dtype=float,
    )
    resolved = int(plan.resolved_area.resolved_pixels)
    allocations = _bounded_allocations(
        works,
        target_pixels=resolved,
        weights=weights,
    )
    topology_policy = _topology_policy(plan)
    selected, audits = _simulate_topology_safe_execution(
        works,
        allocations=allocations,
        desired_pixels=resolved,
        target_region=target_region,
        scene=scene,
        seed=0,
        **topology_policy,
    )
    realized = sum(int(np.count_nonzero(item)) for item in selected)
    topology = _whole_mask_topology_audit(
        source_region=source_region,
        target_region=target_region,
        selected_by_work=selected,
        works=works,
        allow_source_component_resolution=bool(
            topology_policy["allow_source_component_resolution"]
        ),
        allow_target_hole_resolution=bool(
            topology_policy["allow_target_hole_resolution"]
        ),
        allow_source_component_split=bool(
            topology_policy["allow_source_component_split"]
        ),
        minimum_residual_components=int(
            topology_policy["minimum_residual_components"]
        ),
        maximum_residual_components=int(
            topology_policy["maximum_residual_components"]
        ),
        minimum_residual_component_area_px=int(
            topology_policy["minimum_residual_component_area_px"]
        ),
        minimum_residual_spacing_px=int(
            topology_policy["minimum_residual_spacing_px"]
        ),
        residual_area_floor_fraction=float(
            topology_policy["residual_area_floor_fraction"]
        ),
    )
    if realized != resolved or not topology["passed"]:
        raise RefineContractError(
            "compiled topology program is not replayable: "
            f"resolved={resolved}, realized={realized}, topology={topology}"
        )
    parts = tuple(
        CompiledReplayPart(
            planned=work.planned,
            anchor_masks=work.anchor_masks,
            anchor_mask=work.anchor_mask,
            change_region=change,
            legal_capacity_px=work.item_capacity_px,
            source_deletion_limit_px=work.source_deletion_limit_px,
            protected_source_necks=work.protected_source_necks,
            topology_audit=audit,
        )
        for work, change, audit in zip(works, selected, audits)
        if np.any(change)
    )
    return parts, {
        "replay_version": EXECUTION_SOLVER_VERSION,
        "resolved_pixels": resolved,
        "realized_pixels": realized,
        "whole_mask_topology": topology,
    }


def _prepare_compiler_work(
    plan: EditPlan,
    *,
    source_mask: np.ndarray,
    source_region: np.ndarray,
    scene: SceneAnalysis,
) -> tuple[_CompilerWork, ...]:
    prohibited = np.zeros_like(source_mask, dtype=bool)
    for region in scene.prohibited_region_masks.values():
        prohibited |= np.asarray(region, dtype=bool)
    anchor_groups = tuple(
        tuple(
            scene.anchor_masks[anchor_id]
            for anchor_id in planned.execution_contract.anchor_segment_ids
            if anchor_id in scene.anchor_masks
        )
        for planned in plan.candidate_interfaces
    )
    if any(
        len(group) != len(planned.execution_contract.anchor_segment_ids)
        for group, planned in zip(anchor_groups, plan.candidate_interfaces)
    ):
        raise RefineContractError("execution solver cannot resolve all selected anchors")
    anchor_unions = tuple(np.logical_or.reduce(group) for group in anchor_groups)
    params = plan.tool_program.parameter_ranges
    allow_source_resolution = bool(
        params.get("allow_source_component_resolution", False)
    )
    maximum_changed_fraction = min(
        1.0 if allow_source_resolution else 0.55,
        float(params.get("max_source_component_changed_fraction", 0.55)),
    )
    minimum_remaining = max(
        0 if allow_source_resolution else 64,
        int(params.get("min_source_component_remaining_px", 64)),
    )
    # Build each interface's executable envelope independently before assigning
    # overlapping pixels to an owner.  The former nearest-anchor-first
    # partition could hand a pixel to an interface whose tapered profile could
    # not actually reach it; that pixel was then discarded even when another
    # selected interface could execute it.  As a result, adding a legal front
    # could *reduce* combined capacity.  Eligibility-first ownership makes the
    # union monotone: a new interface may win overlapping pixels, but it can no
    # longer steal and invalidate capacity from an existing one.
    provisional: list[dict[str, Any]] = []
    for index, (planned, anchor_masks, anchor) in enumerate(
        zip(plan.candidate_interfaces, anchor_groups, anchor_unions)
    ):
        interface = scene.interface_masks.get(planned.interface_id)
        source_component = scene.component_masks.get(planned.source_component_id)
        if interface is None or source_component is None:
            continue
        _, nearest_interface = ndimage.distance_transform_edt(
            ~interface, return_indices=True
        )
        anchor_influence = anchor[
            nearest_interface[0], nearest_interface[1]
        ]
        distance = ndimage.distance_transform_edt(~anchor)
        requested = planned.execution_contract.depth_profile
        peak = max(requested.peak_depth_px, 1e-6)
        unit_profile = replace(
            requested,
            peak_depth_px=1.0,
            edge_depth_px=float(
                np.clip(requested.edge_depth_px / peak, 0.0, 1.0)
            ),
            noise_amplitude_px=0.0,
        )
        unit_depth = compile_depth_profile_map(
            anchor_masks, profile=unit_profile, shape=source_mask.shape
        )
        required_scale = distance / np.maximum(unit_depth, 1e-3)
        band_min, band_max = planned.allowed_edit_band_px
        legal_envelope = (
            source_component
            & source_region
            & ~prohibited
            & anchor_influence
            & (distance >= max(0.0, band_min))
            & (distance <= band_max)
            & (required_scale <= band_max + 1e-6)
        )
        deletion_limit = source_deletion_limit(
            int(np.count_nonzero(source_component)),
            maximum_changed_fraction=maximum_changed_fraction,
            minimum_remaining_pixels=minimum_remaining,
        )
        if params.get("tissue_geometry_mode") == "residual_fragmentation":
            required_scale = _residual_fragmentation_priority(
                source_component=np.asarray(source_component, dtype=bool),
                legal_envelope=legal_envelope,
                default_priority=required_scale,
                minimum_residual_component_area_px=int(
                    params.get("minimum_residual_component_area_px", 1)
                ),
                minimum_residual_spacing_px=int(
                    params.get("minimum_residual_spacing_px", 0)
                ),
            )
        provisional.append(
            {
                "planned": planned,
                "anchor_masks": anchor_masks,
                "anchor_mask": anchor,
                "legal_envelope": legal_envelope,
                "priority": required_scale,
                "source_component": np.asarray(source_component, dtype=bool),
                "source_deletion_limit_px": deletion_limit,
                "protected_source_necks": protected_narrow_necks(source_component),
            }
        )
    if not provisional:
        return ()

    owner_cost = np.stack(
        [
            np.where(item["legal_envelope"], item["priority"], np.inf)
            for item in provisional
        ]
    )
    assignment = np.argmin(owner_cost, axis=0)
    has_owner = np.any(np.isfinite(owner_cost), axis=0)
    works: list[_CompilerWork] = []
    for index, item in enumerate(provisional):
        legal = item["legal_envelope"] & has_owner & (assignment == index)
        works.append(
            _CompilerWork(
                planned=item["planned"],
                anchor_masks=item["anchor_masks"],
                anchor_mask=item["anchor_mask"],
                legal_source=legal,
                priority=item["priority"],
                source_component=item["source_component"],
                item_capacity_px=int(np.count_nonzero(legal)),
                source_deletion_limit_px=item["source_deletion_limit_px"],
                protected_source_necks=item["protected_source_necks"],
            )
        )
    return tuple(item for item in works if item.item_capacity_px > 0)


def _residual_fragmentation_priority(
    *,
    source_component: np.ndarray,
    legal_envelope: np.ndarray,
    default_priority: np.ndarray,
    minimum_residual_component_area_px: int,
    minimum_residual_spacing_px: int,
) -> np.ndarray:
    """Prioritize a source-owned neck corridor before peripheral turnover.

    A generic distance-from-interface erosion peels an entire tumor perimeter
    before it ever reaches an internal neck. Residual fragmentation instead
    needs a narrow stromal corridor between robust interior lobes. This helper
    discovers those lobes by deterministic erosion, constructs their Voronoi
    separator, and assigns that separator the lowest execution cost. The
    ordinary topology solver and whole-mask audit remain authoritative: a
    convex source with no stable multi-lobe witness still fails closed.
    """

    source = np.asarray(source_component, dtype=bool)
    legal = np.asarray(legal_envelope, dtype=bool)
    if not np.any(source & legal):
        return np.asarray(default_priority, dtype=float)
    structure = np.ones((3, 3), dtype=bool)
    distance = ndimage.distance_transform_edt(source)
    max_iterations = max(1, int(np.floor(distance.max(initial=0.0))) - 1)
    seed_labels = None
    seed_count = 0
    for iterations in range(1, max_iterations + 1):
        eroded = distance > float(iterations)
        labeled, count = ndimage.label(eroded, structure=structure)
        sizes = sorted(
            (
                int(np.count_nonzero(labeled == index)),
                index,
            )
            for index in range(1, count + 1)
            if int(np.count_nonzero(labeled == index))
            >= minimum_residual_component_area_px
        )
        if len(sizes) < 2:
            continue
        # Start from the two largest stable lobes. Additional residual foci
        # may be introduced by a future mechanism-specific program, but using
        # every erosion fragment as a seed creates tiny diagonal raster
        # remnants rather than biologically meaningful residual foci.
        retained = sorted(sizes, reverse=True)[:2]
        seed_labels = np.zeros_like(labeled, dtype=np.int32)
        for new_id, (_size, old_id) in enumerate(retained, start=1):
            seed_labels[labeled == old_id] = new_id
        seed_count = len(retained)
        break
    if seed_labels is None or seed_count < 2:
        return np.asarray(default_priority, dtype=float)

    seeds = seed_labels > 0
    _distance_to_seed, nearest = ndimage.distance_transform_edt(
        ~seeds,
        return_indices=True,
    )
    partition = seed_labels[nearest[0], nearest[1]]
    separator = np.zeros_like(source, dtype=bool)
    for row_offset, col_offset in ((1, 0), (0, 1), (1, 1), (1, -1)):
        shifted = np.roll(partition, (row_offset, col_offset), axis=(0, 1))
        valid = source & (partition > 0) & (shifted > 0) & (partition != shifted)
        if row_offset > 0:
            valid[:row_offset, :] = False
        if col_offset > 0:
            valid[:, :col_offset] = False
        elif col_offset < 0:
            valid[:, col_offset:] = False
        separator |= valid
    if not np.any(separator & legal):
        return np.asarray(default_priority, dtype=float)
    corridor_radius = max(
        1,
        int(np.ceil(max(1, minimum_residual_spacing_px + 1) / 2.0)),
    )
    corridor = ndimage.binary_dilation(
        separator,
        structure=structure,
        iterations=corridor_radius,
    ) & source & legal
    # Fill tiny source remnants enclosed by the proposed corridor. They are
    # raster artifacts of a diagonal cut, not residual foci, and leaving them
    # would violate the same minimum-focus-area contract used by the gate.
    provisional_after = source & ~corridor
    provisional_labels, provisional_count = ndimage.label(
        provisional_after, structure=structure
    )
    tiny_remnants = np.zeros_like(source, dtype=bool)
    for index in range(1, provisional_count + 1):
        component = provisional_labels == index
        if int(np.count_nonzero(component)) < minimum_residual_component_area_px:
            tiny_remnants |= component
    corridor |= tiny_remnants & legal
    corridor_distance = ndimage.distance_transform_edt(~corridor)
    tie_break = np.asarray(default_priority, dtype=float)
    tie_break /= max(float(np.max(tie_break[legal], initial=1.0)), 1.0)
    return corridor_distance + 1e-3 * tie_break


def _bounded_allocations(
    works: tuple[_CompilerWork, ...],
    *,
    target_pixels: int,
    weights: np.ndarray,
) -> tuple[int, ...]:
    """Allocate pixels with item capacities and shared source-component caps."""

    if not works or target_pixels <= 0:
        return tuple(0 for _ in works)
    normalized = np.asarray(weights, dtype=float)
    normalized = normalized / max(float(normalized.sum()), 1e-12)
    capacities = np.asarray([item.item_capacity_px for item in works], dtype=int)
    allocations = np.minimum(
        np.floor(normalized * target_pixels).astype(int), capacities
    )
    group_indices: dict[str, list[int]] = {}
    for index, work in enumerate(works):
        group_indices.setdefault(work.planned.source_component_id, []).append(index)
    for indices in group_indices.values():
        limit = min(works[index].source_deletion_limit_px for index in indices)
        observed = int(allocations[indices].sum())
        if observed <= limit:
            continue
        scaled = allocations[indices].astype(float) * (limit / max(observed, 1))
        reduced = np.floor(scaled).astype(int)
        remainder = limit - int(reduced.sum())
        order = sorted(
            range(len(indices)),
            key=lambda local: (-(scaled[local] - reduced[local]), indices[local]),
        )
        for local in order[:remainder]:
            reduced[local] += 1
        allocations[indices] = reduced

    remaining = target_pixels - int(allocations.sum())
    while remaining > 0:
        group_used = {
            group: int(allocations[indices].sum())
            for group, indices in group_indices.items()
        }
        spare = np.zeros(len(works), dtype=int)
        for index, work in enumerate(works):
            group = work.planned.source_component_id
            group_remaining = max(
                0, work.source_deletion_limit_px - group_used[group]
            )
            spare[index] = min(
                int(capacities[index] - allocations[index]), group_remaining
            )
        active = np.flatnonzero(spare > 0)
        if active.size == 0:
            break
        active_weights = normalized[active]
        active_weights /= max(float(active_weights.sum()), 1e-12)
        proposal = np.floor(active_weights * remaining).astype(int)
        proposal = np.minimum(proposal, spare[active])
        if int(proposal.sum()) == 0:
            chosen = int(
                max(active.tolist(), key=lambda idx: (normalized[idx], -idx))
            )
            proposal[np.where(active == chosen)[0][0]] = 1
        for local, index in enumerate(active.tolist()):
            allocations[index] += int(proposal[local])
        remaining = target_pixels - int(allocations.sum())
    return tuple(int(value) for value in allocations)


def _topology_policy(plan: EditPlan) -> dict[str, Any]:
    """Return deterministic primitive-owned topology permissions.

    These flags are compiled from reviewed primitive skills by the joint
    tissue planner. They are not inferred from image pixels and default to the
    conservative generic burden-edit policy for legacy callers.
    """

    params = plan.tool_program.parameter_ranges
    return {
        "allow_source_component_resolution": bool(
            params.get("allow_source_component_resolution", False)
        ),
        "allow_target_hole_resolution": bool(
            params.get("allow_target_hole_resolution", False)
        ),
        "allow_source_component_split": bool(
            params.get("allow_source_component_split", False)
        ),
        "minimum_residual_components": max(
            1, int(params.get("minimum_residual_components", 1))
        ),
        "maximum_residual_components": max(
            1, int(params.get("maximum_residual_components", 1))
        ),
        "minimum_residual_component_area_px": max(
            1, int(params.get("minimum_residual_component_area_px", 1))
        ),
        "minimum_residual_spacing_px": max(
            0, int(params.get("minimum_residual_spacing_px", 0))
        ),
        "residual_area_floor_fraction": float(
            params.get("residual_area_floor_fraction", 0.0)
        ),
        "minimum_changed_component_area_px": max(
            1, int(params.get("min_component_area_px", 16))
        ),
    }


def _resolve_topology_safe_area(
    works: tuple[_CompilerWork, ...],
    *,
    desired_pixels: int,
    hard_min_pixels: int,
    weights: np.ndarray,
    source_region: np.ndarray,
    target_region: np.ndarray,
    scene: SceneAnalysis,
    fallback_policy: str,
    allow_source_component_resolution: bool = False,
    allow_target_hole_resolution: bool = False,
    allow_source_component_split: bool = False,
    minimum_residual_components: int = 1,
    maximum_residual_components: int = 1,
    minimum_residual_component_area_px: int = 1,
    minimum_residual_spacing_px: int = 0,
    residual_area_floor_fraction: float = 0.0,
    minimum_changed_component_area_px: int = 16,
) -> tuple[
    tuple[int, ...],
    tuple[np.ndarray, ...],
    tuple[dict[str, Any], ...],
    dict[str, Any],
    dict[str, Any],
]:
    attempts: list[dict[str, Any]] = []

    def attempt(total: int):
        allocations = _bounded_allocations(
            works, target_pixels=total, weights=weights
        )
        selected, audits = _simulate_topology_safe_execution(
            works,
            allocations=allocations,
            desired_pixels=total,
            target_region=target_region,
            scene=scene,
            seed=0,
            allow_source_component_resolution=(
                allow_source_component_resolution
            ),
            allow_target_hole_resolution=allow_target_hole_resolution,
            allow_source_component_split=allow_source_component_split,
            minimum_changed_component_area_px=(
                minimum_changed_component_area_px
            ),
        )
        realized = sum(int(np.count_nonzero(item)) for item in selected)
        topology = _whole_mask_topology_audit(
            source_region=source_region,
            target_region=target_region,
            selected_by_work=selected,
            works=works,
            allow_source_component_resolution=(
                allow_source_component_resolution
            ),
            allow_target_hole_resolution=allow_target_hole_resolution,
            allow_source_component_split=allow_source_component_split,
            minimum_residual_components=minimum_residual_components,
            maximum_residual_components=maximum_residual_components,
            minimum_residual_component_area_px=(
                minimum_residual_component_area_px
            ),
            minimum_residual_spacing_px=minimum_residual_spacing_px,
            residual_area_floor_fraction=residual_area_floor_fraction,
        )
        valid = realized == total and bool(topology["passed"])
        attempts.append(
            {
                "requested_pixels": int(total),
                "realized_pixels": int(realized),
                "topology_passed": bool(topology["passed"]),
                "valid": bool(valid),
            }
        )
        return allocations, selected, audits, topology, realized, valid

    first = attempt(desired_pixels)
    if first[-1]:
        return (*first[:4], {"attempts": attempts, "selection": "desired"})

    first_realized = int(first[-2])
    if (
        first_realized < desired_pixels
        and bool(first[3]["passed"])
        and first_realized >= hard_min_pixels
    ):
        # The full desired request exhausted every reachable safe front. Its
        # realized prefix is therefore the maximum found by this solver.
        realized_allocations = tuple(
            int(np.count_nonzero(item)) for item in first[1]
        )
        return (
            realized_allocations,
            first[1],
            first[2],
            first[3],
            {"attempts": attempts, "selection": "maximum_reachable_prefix"},
        )

    if fallback_policy != "max_feasible_below_target":
        raise RefineContractError(
            "exact area violates whole-mask topology or reachable capacity: "
            f"desired={desired_pixels}, realized={first_realized}, "
            f"topology={first[3]}"
        )

    upper = min(desired_pixels - 1, max(hard_min_pixels, first_realized - 1))
    if upper < hard_min_pixels:
        raise RefineContractError(
            "no topology-safe area remains inside the hard allowed interval"
        )
    # The solver already treats reachable growth as a monotone prefix. Probe
    # the immutable task floor first: if the floor itself is unsafe, testing
    # 16 larger areas cannot make the task executable and previously made one
    # failed replan take minutes on multi-interface patches. Once the floor is
    # known safe, binary search the largest safe prefix below the failed upper
    # bound.
    floor_result = attempt(hard_min_pixels)
    if not floor_result[-1]:
        raise RefineContractError(
            "no whole-mask topology-safe edit reaches the task hard minimum: "
            f"minimum={hard_min_pixels}, realized={int(floor_result[-2])}, "
            f"topology={floor_result[3]}, desired_probe_topology={first[3]}"
        )
    low = hard_min_pixels
    high = upper + 1
    best = floor_result
    while high - low > 1:
        middle = (low + high) // 2
        result = attempt(middle)
        if result[-1]:
            low = middle
            best = result
        else:
            high = middle
    return (
        *best[:4],
        {
            "attempts": attempts,
            "selection": "largest_verified_safe_below_desired",
            "monotone_prefix_assumption": True,
            "floor_first_search": True,
            "selected_pixels": int(low),
            "first_known_invalid_above": int(high),
        },
    )


def _whole_mask_topology_audit(
    *,
    source_region: np.ndarray,
    target_region: np.ndarray,
    selected_by_work: tuple[np.ndarray, ...],
    works: tuple[_CompilerWork, ...],
    allow_source_component_resolution: bool = False,
    allow_target_hole_resolution: bool = False,
    allow_source_component_split: bool = False,
    minimum_residual_components: int = 1,
    maximum_residual_components: int = 1,
    minimum_residual_component_area_px: int = 1,
    minimum_residual_spacing_px: int = 0,
    residual_area_floor_fraction: float = 0.0,
) -> dict[str, Any]:
    change = np.logical_or.reduce(selected_by_work)
    source_after = source_region & ~change
    target_after = target_region | change
    structure = np.ones((3, 3), dtype=bool)
    source_components_before = int(ndimage.label(source_region, structure=structure)[1])
    source_components_after = int(ndimage.label(source_after, structure=structure)[1])
    target_components_before = int(ndimage.label(target_region, structure=structure)[1])
    target_components_after = int(ndimage.label(target_after, structure=structure)[1])
    source_holes_before = _hole_count(source_region)
    source_holes_after = _hole_count(source_after)
    target_holes_before = _hole_count(target_region)
    target_holes_after = _hole_count(target_after)
    target_merge = target_components_after < target_components_before
    source_hole_change_allowed = target_merge and (
        len({item.planned.target_component_id for item in works}) > 1
    )
    source_holes_valid = (
        source_holes_after <= source_holes_before
        if allow_source_component_resolution
        else (
            source_holes_after == source_holes_before
            or source_hole_change_allowed
        )
    )
    residual_source_before = source_region
    residual_source_after = source_after
    if allow_source_component_split:
        selected_source = np.logical_or.reduce(
            tuple(work.source_component for work in works)
        )
        residual_source_before = source_region & selected_source
        residual_source_after = source_after & selected_source
    source_labeled_after, source_components_after_selected = ndimage.label(
        residual_source_after, structure=structure
    )
    residual_sizes = sorted(
        int(np.count_nonzero(source_labeled_after == index))
        for index in range(1, source_components_after_selected + 1)
    )
    source_area_before = int(np.count_nonzero(residual_source_before))
    source_area_after = int(np.count_nonzero(residual_source_after))
    residual_fraction = source_area_after / max(source_area_before, 1)
    residual_spacing_px = _minimum_component_spacing_px(
        source_labeled_after,
        source_components_after_selected,
    )
    if allow_source_component_split:
        source_components_valid = bool(
            minimum_residual_components <= source_components_after_selected
            <= maximum_residual_components
            and residual_sizes
            and min(residual_sizes) >= minimum_residual_component_area_px
            and residual_spacing_px + 1e-9 >= minimum_residual_spacing_px
            and residual_fraction + 1e-9 >= residual_area_floor_fraction
        )
    else:
        source_components_valid = (
            source_components_after <= source_components_before
            if allow_source_component_resolution
            else source_components_after == source_components_before
        )
    if allow_source_component_split:
        # Each additional residual source focus is represented as one
        # additional hole in the already-adjacent target compartment. This is
        # the intended dual topology of a bounded fragmentation corridor, not
        # an arbitrary target-ring artifact.
        allowed_added_target_holes = max(
            0,
            source_components_after - source_components_before,
        )
        target_holes_valid = bool(
            target_holes_before
            <= target_holes_after
            <= target_holes_before + allowed_added_target_holes
        )
    else:
        target_holes_valid = (
            target_holes_after <= target_holes_before
            if allow_target_hole_resolution
            else target_holes_after == target_holes_before
        )
    passed = (
        source_components_valid
        and target_components_after <= target_components_before
        and source_holes_valid
        and target_holes_valid
    )
    return {
        "passed": bool(passed),
        "source_components_before": source_components_before,
        "source_components_after": source_components_after,
        "selected_source_components_after": int(
            source_components_after_selected
        ),
        "target_components_before": target_components_before,
        "target_components_after": target_components_after,
        "source_holes_before": source_holes_before,
        "source_holes_after": source_holes_after,
        "target_holes_before": target_holes_before,
        "target_holes_after": target_holes_after,
        "selected_target_component_ids": sorted(
            {item.planned.target_component_id for item in works}
        ),
        "target_merge": bool(target_merge),
        "source_hole_change_allowed_by_selected_target_merge": bool(
            source_hole_change_allowed
        ),
        "allow_source_component_resolution": bool(
            allow_source_component_resolution
        ),
        "allow_target_hole_resolution": bool(
            allow_target_hole_resolution
        ),
        "allow_source_component_split": bool(allow_source_component_split),
        "minimum_residual_components": int(minimum_residual_components),
        "maximum_residual_components": int(maximum_residual_components),
        "minimum_residual_component_area_px": int(
            minimum_residual_component_area_px
        ),
        "minimum_residual_spacing_px": int(minimum_residual_spacing_px),
        "observed_minimum_residual_spacing_px": float(
            residual_spacing_px
        ),
        "residual_component_sizes_px": residual_sizes,
        "residual_area_fraction": float(residual_fraction),
        "residual_area_floor_fraction": float(residual_area_floor_fraction),
    }


def _minimum_component_spacing_px(
    labeled: np.ndarray,
    component_count: int,
) -> float:
    """Return the closest Euclidean gap between distinct residual foci."""

    if component_count < 2:
        return float("inf")
    minimum = float("inf")
    for left in range(1, component_count):
        distance = ndimage.distance_transform_edt(labeled != left)
        for right in range(left + 1, component_count + 1):
            minimum = min(
                minimum,
                float(np.min(distance[labeled == right], initial=np.inf)),
            )
    return minimum


def _hole_count(mask: np.ndarray) -> int:
    holes = ndimage.binary_fill_holes(mask) & ~mask
    return int(ndimage.label(holes, structure=np.ones((3, 3), dtype=bool))[1])


def _simulate_topology_safe_execution(
    works: tuple[_CompilerWork, ...],
    *,
    allocations: tuple[int, ...],
    desired_pixels: int,
    target_region: np.ndarray,
    scene: SceneAnalysis,
    seed: int,
    allow_source_component_resolution: bool = False,
    allow_target_hole_resolution: bool = False,
    allow_source_component_split: bool = False,
    minimum_residual_components: int = 1,
    maximum_residual_components: int = 1,
    minimum_residual_component_area_px: int = 1,
    minimum_residual_spacing_px: int = 0,
    residual_area_floor_fraction: float = 0.0,
    minimum_changed_component_area_px: int = 16,
) -> tuple[tuple[np.ndarray, ...], tuple[dict[str, Any], ...]]:
    # Whole-mask residual constraints are audited after all fronts have been
    # jointly simulated. Accept them here so the shared topology-policy object
    # can be passed intact without duplicating partial per-front semantics.
    del (
        minimum_residual_components,
        maximum_residual_components,
        minimum_residual_component_area_px,
        minimum_residual_spacing_px,
        residual_area_floor_fraction,
    )
    source_states = {
        work.planned.source_component_id: np.array(
            scene.component_masks[work.planned.source_component_id], copy=True
        )
        for work in works
    }
    target_state = np.array(target_region, copy=True)
    selected_target = np.zeros_like(target_region, dtype=bool)
    for work in works:
        selected_target |= scene.component_masks[work.planned.target_component_id]
    unselected_target = target_region & ~selected_target
    deleted_by_source = {
        component_id: 0 for component_id in source_states
    }
    selected_by_work = [
        np.zeros_like(target_region, dtype=bool) for _ in works
    ]
    audit_lists: list[list[dict[str, int]]] = [[] for _ in works]

    def grow(
        index: int,
        requested: int,
        pass_index: int,
        *,
        continue_existing_front_only: bool = False,
    ) -> int:
        if requested <= 0:
            return 0
        work = works[index]
        component_id = work.planned.source_component_id
        continued_frontier = ndimage.binary_dilation(
            selected_by_work[index], structure=np.ones((3, 3), dtype=bool)
        )
        if continue_existing_front_only and not np.any(
            selected_by_work[index]
        ):
            return 0
        frontier = (
            continued_frontier
            if continue_existing_front_only
            else work.anchor_mask | continued_frontier
        )
        selected, audit = topology_safe_priority_grow(
            work.legal_source & ~selected_by_work[index],
            interface_mask=frontier,
            target_pixels=requested,
            priority=work.priority,
            source_component_state=source_states[component_id],
            target_state=target_state,
            unselected_target=unselected_target,
            maximum_source_deletions=work.source_deletion_limit_px,
            already_deleted_from_source=deleted_by_source[component_id],
            protected_source_necks=(
                None
                if (
                    allow_source_component_resolution
                    or allow_source_component_split
                )
                else work.protected_source_necks
            ),
            seed=seed + index * 13007 + pass_index * 104729,
            allow_source_component_resolution=(
                allow_source_component_resolution
            ),
            allow_target_hole_resolution=allow_target_hole_resolution,
            allow_source_component_split=allow_source_component_split,
        )
        realized = int(np.count_nonzero(selected))
        selected_by_work[index] |= selected
        deleted_by_source[component_id] += realized
        audit_lists[index].append(audit.to_metadata())
        return realized

    for index, allocation in enumerate(allocations):
        grow(index, int(allocation), 0)

    # Reallocate any topology-caused deficit to the remaining selected fronts.
    # This is the joint solver step: interface ratios are preferred, but hard
    # topology and the total desired area take precedence over a brittle exact
    # per-interface quota.
    for pass_index in range(1, len(works) + 2):
        realized_total = sum(int(np.count_nonzero(item)) for item in selected_by_work)
        deficit = desired_pixels - realized_total
        if deficit <= 0:
            break
        progress = 0
        order = sorted(
            range(len(works)),
            key=lambda index: (
                -(works[index].item_capacity_px - int(np.count_nonzero(selected_by_work[index]))),
                index,
            ),
        )
        for index in order:
            if deficit <= 0:
                break
            work = works[index]
            component_id = work.planned.source_component_id
            item_spare = work.item_capacity_px - int(
                np.count_nonzero(selected_by_work[index])
            )
            group_spare = (
                work.source_deletion_limit_px - deleted_by_source[component_id]
            )
            request = min(deficit, item_spare, group_spare)
            obtained = grow(index, int(request), pass_index)
            progress += obtained
            deficit -= obtained
        if progress <= 0:
            break

    # Multiple addressable anchor chunks may seed the same biological front.
    # Near an exact area cutoff, a low-priority chunk used to receive only a
    # handful of pixels (case 534 produced a 4 px satellite).  Consolidate
    # such raster fragments inside the authoritative solver, restore their
    # source/target states atomically, and spend the reclaimed budget only by
    # continuing already-established fronts.  The downstream gate therefore
    # verifies the same minimum that candidate generation already guarantees.
    minimum_component = max(1, int(minimum_changed_component_area_px))
    for cleanup_pass in range(len(works) + 2):
        combined = np.logical_or.reduce(selected_by_work)
        labels, component_count = ndimage.label(
            combined, structure=np.ones((3, 3), dtype=bool)
        )
        tiny = np.zeros_like(combined, dtype=bool)
        for component_id in range(1, component_count + 1):
            component = labels == component_id
            if int(np.count_nonzero(component)) < minimum_component:
                tiny |= component
        reclaimed = int(np.count_nonzero(tiny))
        if reclaimed <= 0:
            break
        for index, selected in enumerate(selected_by_work):
            removed = selected & tiny
            removed_count = int(np.count_nonzero(removed))
            if removed_count <= 0:
                continue
            component_id = works[index].planned.source_component_id
            selected_by_work[index][removed] = False
            source_states[component_id][removed] = True
            target_state[removed] = target_region[removed]
            deleted_by_source[component_id] -= removed_count
            audit_lists[index].append(
                {
                    "tiny_component_pixels_reclaimed": removed_count,
                    "tiny_component_cleanup_calls": 1,
                }
            )
        deficit = desired_pixels - sum(
            int(np.count_nonzero(item)) for item in selected_by_work
        )
        progress = 0
        order = sorted(
            range(len(works)),
            key=lambda index: (
                -int(np.count_nonzero(selected_by_work[index])),
                index,
            ),
        )
        for index in order:
            if deficit <= 0:
                break
            work = works[index]
            component_id = work.planned.source_component_id
            request = min(
                deficit,
                work.item_capacity_px
                - int(np.count_nonzero(selected_by_work[index])),
                work.source_deletion_limit_px
                - deleted_by_source[component_id],
            )
            obtained = grow(
                index,
                max(0, int(request)),
                len(works) + 2 + cleanup_pass,
                continue_existing_front_only=True,
            )
            progress += obtained
            deficit -= obtained
        if progress <= 0:
            break

    summarized: list[dict[str, Any]] = []
    for calls in audit_lists:
        totals: dict[str, int] = {}
        for call in calls:
            for key, value in call.items():
                totals[key] = totals.get(key, 0) + int(value)
        totals["call_count"] = len(calls)
        summarized.append(totals)
    return tuple(selected_by_work), tuple(summarized)


def _binding_constraint(
    *,
    desired_pixels: int,
    resolved_pixels: int,
    works: tuple[_CompilerWork, ...],
    audits: tuple[dict[str, Any], ...],
) -> str:
    if resolved_pixels >= desired_pixels:
        return "desired_area_realized"
    sums = {
        "source_topology": sum(
            int(item.get("rejected_source_connectivity", 0))
            + int(item.get("rejected_source_hole_change", 0))
            for item in audits
        ),
        "target_topology": sum(
            int(item.get("rejected_target_hole_change", 0))
            + int(item.get("rejected_target_island", 0))
            + int(item.get("rejected_unselected_target_contact", 0))
            for item in audits
        ),
        "source_retention": sum(
            int(item.get("rejected_source_retention", 0)) for item in audits
        ),
    }
    if max(sums.values(), default=0) > 0:
        return max(sums, key=lambda key: (sums[key], key))
    total_legal = sum(item.item_capacity_px for item in works)
    return (
        "legal_interface_capacity"
        if total_legal < desired_pixels
        else "reachable_interface_capacity"
    )

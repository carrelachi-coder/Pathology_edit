"""Compile Planner intent into a topology-safe executable pixel contract."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy import ndimage

from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.candidates import (
    compile_depth_profile_map,
    compile_directional_tapered_projection_field,
)
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


class TopologySafeAreaUnderfillError(RefineContractError):
    """Carry a machine-readable failed area probe back to the Planner.

    The message remains suitable for existing audit logs, while ``feedback``
    prevents the joint candidate compiler from reducing an underfill to an
    opaque string and forgetting which interfaces actually contributed.
    """

    def __init__(self, message: str, *, feedback: dict[str, Any]) -> None:
        super().__init__(message)
        self.feedback = dict(feedback)


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
        source_region=source_region,
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
        maximum_residual_area_fraction=float(
            topology_policy["maximum_residual_area_fraction"]
        ),
        minimum_residual_component_fraction=float(
            topology_policy["minimum_residual_component_fraction"]
        ),
        maximum_dominant_residual_component_fraction=float(
            topology_policy[
                "maximum_dominant_residual_component_fraction"
            ]
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
    target_change_pixels = (
        int(plan.resolved_area.resolved_pixels)
        if plan.resolved_area is not None
        else int(plan.area_budget.target_pixels(source_mask, source_region))
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
        residual_fragmentation = (
            params.get("tissue_geometry_mode") == "residual_fragmentation"
        )
        legal_envelope = (
            source_component
            & source_region
            & ~prohibited
            & anchor_influence
            & (distance >= max(0.0, band_min))
            & (distance <= band_max)
        )
        if not residual_fragmentation:
            legal_envelope &= required_scale <= band_max + 1e-6
        if params.get("tissue_geometry_mode") == "annotation_anchored_narrow_connected_extension":
            legal_envelope, required_scale = (
                compile_directional_tapered_projection_field(
                    legal_envelope,
                    anchor_mask=anchor,
                    parent_mask=scene.component_masks.get(
                        planned.target_component_id,
                        np.zeros_like(source_mask, dtype=bool),
                    ),
                    maximum_depth_px=band_max,
                    maximum_width_px=float(
                        params.get("directional_maximum_width_px", 24.0)
                    ),
                    tip_width_px=float(
                        params.get("directional_tip_width_px", 2.0)
                    ),
                )
            )
        deletion_limit = source_deletion_limit(
            int(np.count_nonzero(source_component)),
            maximum_changed_fraction=maximum_changed_fraction,
            minimum_remaining_pixels=minimum_remaining,
        )
        if residual_fragmentation:
            required_scale = _residual_fragmentation_priority(
                source_component=np.asarray(source_component, dtype=bool),
                legal_envelope=legal_envelope,
                default_priority=required_scale,
                minimum_residual_components=int(
                    params.get("minimum_residual_components", 2)
                ),
                maximum_residual_components=int(
                    params.get("maximum_residual_components", 6)
                ),
                minimum_residual_component_area_px=int(
                    params.get("minimum_residual_component_area_px", 1)
                ),
                minimum_residual_spacing_px=int(
                    params.get("minimum_residual_spacing_px", 0)
                ),
                minimum_residual_component_fraction=float(
                    params.get("minimum_residual_component_fraction", 0.0)
                ),
                maximum_dominant_residual_component_fraction=float(
                    params.get(
                        "maximum_dominant_residual_component_fraction", 1.0
                    )
                ),
                target_change_pixels=max(
                    1,
                    round(
                        target_change_pixels
                        * planned.execution_contract.area_allocation_fraction
                    ),
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
    minimum_residual_components: int,
    maximum_residual_components: int,
    minimum_residual_component_area_px: int,
    minimum_residual_spacing_px: int,
    minimum_residual_component_fraction: float,
    maximum_dominant_residual_component_fraction: float,
    target_change_pixels: int = 0,
) -> np.ndarray:
    """Prioritize balanced, traversing corridors before peripheral turnover.

    Fragmentation must distribute residual mass among several meaningful foci.
    A natural-neck-only heuristic can split off a tiny satellite while leaving
    almost the whole tumor connected.  Build deterministic quantile partitions
    along several source-owned axes, select the best legal separator set, and
    give those interior corridors the lowest execution cost.  Final topology
    and balance remain authoritative in the whole-mask audit.
    """

    source = np.asarray(source_component, dtype=bool)
    legal = np.asarray(legal_envelope, dtype=bool)
    if not np.any(source & legal):
        return np.asarray(default_priority, dtype=float)
    structure = np.ones((3, 3), dtype=bool)
    focus_count = int(
        np.clip(minimum_residual_components, 2, maximum_residual_components)
    )
    coordinates = np.argwhere(source).astype(float)
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(coordinates) - 1, 1)
    _values, vectors = np.linalg.eigh(covariance)
    major = vectors[:, -1]
    minor = vectors[:, 0]
    raw_axes = (
        major,
        minor,
        np.asarray((1.0, 0.0)),
        np.asarray((0.0, 1.0)),
        np.asarray((1.0, 1.0)),
        np.asarray((1.0, -1.0)),
    )
    axes: list[np.ndarray] = []
    for axis in raw_axes:
        normalized = np.asarray(axis, dtype=float) / max(
            float(np.linalg.norm(axis)), 1e-12
        )
        if any(abs(float(np.dot(normalized, seen))) > 0.995 for seen in axes):
            continue
        axes.append(normalized)

    corridor_radius = max(
        1,
        int(np.ceil(max(1, minimum_residual_spacing_px + 1) / 2.0)),
    )
    best_corridor = None
    best_score = None
    for axis_index, axis in enumerate(axes):
        projection = centered @ axis
        boundaries = np.quantile(
            projection,
            np.arange(1, focus_count, dtype=float) / focus_count,
        )
        partition = np.zeros_like(source, dtype=np.int16)
        partition_values = np.searchsorted(
            boundaries, projection, side="right"
        ) + 1
        rows = coordinates[:, 0].astype(int)
        cols = coordinates[:, 1].astype(int)
        partition[rows, cols] = partition_values
        separator = np.zeros_like(source, dtype=bool)
        vertical = (
            source[1:, :]
            & source[:-1, :]
            & (partition[1:, :] != partition[:-1, :])
        )
        horizontal = (
            source[:, 1:]
            & source[:, :-1]
            & (partition[:, 1:] != partition[:, :-1])
        )
        separator[1:, :] |= vertical
        separator[:-1, :] |= vertical
        separator[:, 1:] |= horizontal
        separator[:, :-1] |= horizontal
        corridor = ndimage.binary_dilation(
            separator,
            structure=structure,
            iterations=corridor_radius,
        ) & source & legal
        if not np.any(corridor):
            continue
        corridor_distance = ndimage.distance_transform_edt(~corridor)
        tie_break = np.asarray(default_priority, dtype=float)
        tie_break /= max(float(np.max(tie_break[legal], initial=1.0)), 1.0)
        approximate_priority = corridor_distance + 1e-3 * tie_break
        eligible = source & legal
        eligible_ids = np.flatnonzero(eligible)
        approximate_change = np.zeros_like(source, dtype=bool)
        selected_count = min(
            max(int(target_change_pixels), int(np.count_nonzero(corridor))),
            len(eligible_ids),
        )
        if selected_count > 0:
            order = np.argsort(
                approximate_priority.ravel()[eligible_ids], kind="stable"
            )
            approximate_change.ravel()[eligible_ids[order[:selected_count]]] = True
        provisional_after = source & ~approximate_change
        provisional_labels, provisional_count = ndimage.label(
            provisional_after, structure=structure
        )
        provisional_sizes = sorted(
            int(np.count_nonzero(provisional_labels == index))
            for index in range(1, provisional_count + 1)
        )
        total = max(sum(provisional_sizes), 1)
        fractions = [size / total for size in provisional_sizes]
        minimum_fraction = min(fractions, default=0.0)
        dominant_fraction = max(fractions, default=1.0)
        contract_valid = bool(
            focus_count <= provisional_count <= maximum_residual_components
            and provisional_sizes
            and min(provisional_sizes)
            >= minimum_residual_component_area_px
            and minimum_fraction + 1e-9
            >= minimum_residual_component_fraction
            and dominant_fraction
            <= maximum_dominant_residual_component_fraction + 1e-9
        )
        subminimum_pixels = sum(
            size
            for size in provisional_sizes
            if size < minimum_residual_component_area_px
        )
        score = (
            int(contract_valid),
            -subminimum_pixels,
            min(provisional_count, maximum_residual_components),
            minimum_fraction,
            -dominant_fraction,
            -int(np.count_nonzero(corridor)),
            -axis_index,
        )
        if best_score is None or score > best_score:
            best_corridor = corridor
            best_score = score
    if best_corridor is None:
        return np.asarray(default_priority, dtype=float)
    corridor = best_corridor
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
        "maximum_residual_area_fraction": float(
            params.get("maximum_residual_area_fraction", 1.0)
        ),
        "minimum_residual_component_fraction": float(
            params.get("minimum_residual_component_fraction", 0.0)
        ),
        "maximum_dominant_residual_component_fraction": float(
            params.get(
                "maximum_dominant_residual_component_fraction", 1.0
            )
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
    maximum_residual_area_fraction: float = 1.0,
    minimum_residual_component_fraction: float = 0.0,
    maximum_dominant_residual_component_fraction: float = 1.0,
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
            source_region=source_region,
            target_region=target_region,
            scene=scene,
            seed=0,
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
            maximum_residual_area_fraction=maximum_residual_area_fraction,
            minimum_residual_component_fraction=(
                minimum_residual_component_fraction
            ),
            maximum_dominant_residual_component_fraction=(
                maximum_dominant_residual_component_fraction
            ),
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
            maximum_residual_area_fraction=maximum_residual_area_fraction,
            minimum_residual_component_fraction=(
                minimum_residual_component_fraction
            ),
            maximum_dominant_residual_component_fraction=(
                maximum_dominant_residual_component_fraction
            ),
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
        realized_pixels = int(floor_result[-2])
        topology_passed = bool(floor_result[3]["passed"])
        safe_realized_pixels = realized_pixels if topology_passed else 0
        interface_contributions = [
            {
                "interface_id": work.planned.interface_id,
                "source_component_id": work.planned.source_component_id,
                "target_component_id": work.planned.target_component_id,
                "anchor_segment_ids": list(
                    work.planned.execution_contract.anchor_segment_ids
                ),
                "requested_allocation_pixels": int(allocation),
                "actual_contribution_pixels": int(np.count_nonzero(selected)),
                "legal_capacity_pixels": int(work.item_capacity_px),
                "source_deletion_limit_pixels": int(
                    work.source_deletion_limit_px
                ),
            }
            for work, allocation, selected in zip(
                works, floor_result[0], floor_result[1]
            )
        ]
        checker_id = (
            "whole_mask_topology_safe_area_capacity"
            if topology_passed
            else "whole_mask_topology_audit"
        )
        message = (
            "no whole-mask topology-safe edit reaches the task hard minimum: "
            f"minimum={hard_min_pixels}, realized={realized_pixels}, "
            f"topology={floor_result[3]}, desired_probe_topology={first[3]}"
        )
        raise TopologySafeAreaUnderfillError(
            message,
            feedback={
                "stage": "tissue_area_underfill",
                "checker_id": checker_id,
                "requested_pixels": int(desired_pixels),
                "policy_floor_pixels": int(hard_min_pixels),
                "realized_pixels": realized_pixels,
                "topology_safe_realized_pixels": safe_realized_pixels,
                "deficit_to_target_pixels": max(
                    0, int(desired_pixels) - safe_realized_pixels
                ),
                "deficit_to_floor_pixels": max(
                    0, int(hard_min_pixels) - safe_realized_pixels
                ),
                "topology_passed": topology_passed,
                "topology": floor_result[3],
                "desired_probe_realized_pixels": first_realized,
                "desired_probe_topology": first[3],
                "interface_contributions": interface_contributions,
                "required_action": (
                    "expand_interface_set_and_redistribute"
                    if topology_passed
                    else "redistribute_across_alternate_interfaces"
                ),
            },
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
    maximum_residual_area_fraction: float = 1.0,
    minimum_residual_component_fraction: float = 0.0,
    maximum_dominant_residual_component_fraction: float = 1.0,
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
    residual_total = max(sum(residual_sizes), 1)
    residual_component_fractions = [
        size / residual_total for size in residual_sizes
    ]
    minimum_observed_component_fraction = min(
        residual_component_fractions, default=0.0
    )
    dominant_component_fraction = max(
        residual_component_fractions, default=1.0
    )
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
            and residual_fraction <= maximum_residual_area_fraction + 1e-9
            and minimum_observed_component_fraction + 1e-9
            >= minimum_residual_component_fraction
            and dominant_component_fraction
            <= maximum_dominant_residual_component_fraction + 1e-9
        )
    else:
        source_components_valid = (
            source_components_after <= source_components_before
            if allow_source_component_resolution
            else source_components_after == source_components_before
        )
    if allow_source_component_split:
        # Bind newly enclosed target holes to the residual foci themselves,
        # rather than to the raw target-hole count.  At a three-label
        # junction, extending Stroma can expose an unchanged third-tissue
        # island that was previously connected to Tumor; that changes the raw
        # hole count without creating an extra tumor focus or a target-ring
        # artifact.  Every target hole that actually contains selected
        # residual source must still map to one of the certified foci.
        residual_target_holes_after = _hole_intersection_count(
            target_after, residual_source_after
        )
        target_holes_valid = bool(
            target_holes_before
            <= target_holes_after
            and residual_target_holes_after
            <= source_components_after_selected
        )
    else:
        residual_target_holes_after = 0
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
        "fragmentation_residual_target_holes_after": int(
            residual_target_holes_after
        ),
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
        "maximum_residual_area_fraction": float(
            maximum_residual_area_fraction
        ),
        "residual_component_fractions": residual_component_fractions,
        "minimum_observed_residual_component_fraction": float(
            minimum_observed_component_fraction
        ),
        "minimum_required_residual_component_fraction": float(
            minimum_residual_component_fraction
        ),
        "dominant_residual_component_fraction": float(
            dominant_component_fraction
        ),
        "maximum_dominant_residual_component_fraction": float(
            maximum_dominant_residual_component_fraction
        ),
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


def _hole_intersection_count(mask: np.ndarray, region: np.ndarray) -> int:
    """Count enclosed complement components that contain ``region`` pixels."""

    holes = ndimage.binary_fill_holes(mask) & ~mask
    labels, count = ndimage.label(
        holes, structure=np.ones((3, 3), dtype=bool)
    )
    return sum(
        bool(np.any(region & (labels == index)))
        for index in range(1, count + 1)
    )


def _simulate_topology_safe_execution(
    works: tuple[_CompilerWork, ...],
    *,
    allocations: tuple[int, ...],
    desired_pixels: int,
    source_region: np.ndarray,
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
    maximum_residual_area_fraction: float = 1.0,
    minimum_residual_component_fraction: float = 0.0,
    maximum_dominant_residual_component_fraction: float = 1.0,
    minimum_changed_component_area_px: int = 16,
) -> tuple[tuple[np.ndarray, ...], tuple[dict[str, Any], ...]]:
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

    if allow_source_component_split:
        selected_by_work, cleanup = _rebalance_fragmentation_residual_islands(
            tuple(selected_by_work),
            works=works,
            source_region=source_region,
            target_region=target_region,
            minimum_residual_components=minimum_residual_components,
            maximum_residual_components=maximum_residual_components,
            minimum_residual_component_area_px=(
                minimum_residual_component_area_px
            ),
            minimum_residual_spacing_px=minimum_residual_spacing_px,
            residual_area_floor_fraction=residual_area_floor_fraction,
            maximum_residual_area_fraction=maximum_residual_area_fraction,
            minimum_residual_component_fraction=(
                minimum_residual_component_fraction
            ),
            maximum_dominant_residual_component_fraction=(
                maximum_dominant_residual_component_fraction
            ),
        )
        selected_by_work = list(selected_by_work)
        if cleanup["applied"]:
            audit_lists[cleanup["assigned_work_index"]].append(
                {
                    "fragmentation_residual_island_pixels_added": cleanup[
                        "tiny_pixels_added"
                    ],
                    "fragmentation_spacing_pixels_added": cleanup[
                        "spacing_pixels_added"
                    ],
                    "fragmentation_balance_pixels_added": cleanup[
                        "balance_pixels_added"
                    ],
                    "fragmentation_noncritical_edge_pixels_reclaimed": cleanup[
                        "pixels_reclaimed"
                    ],
                    "fragmentation_residual_cleanup_calls": 1,
                }
            )

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

    # A traversing corridor can leave a sub-component-sized raster tail when
    # the simple-point grower exhausts its local queue a few pixels before the
    # exact requested area.  Do not invoke the global fallback protocol for a
    # tail that is too small to constitute a valid changed component.  Extend
    # an already-established, compiler-owned front transactionally and accept
    # each pixel only while the complete fragmentation topology still passes.
    if allow_source_component_split:
        deficit = desired_pixels - sum(
            int(np.count_nonzero(item)) for item in selected_by_work
        )
        if 0 < deficit < minimum_component:
            selected_by_work, added = _fragmentation_exact_area_backfill(
                tuple(selected_by_work),
                works=works,
                source_region=source_region,
                target_region=target_region,
                requested_pixels=deficit,
                minimum_residual_components=minimum_residual_components,
                maximum_residual_components=maximum_residual_components,
                minimum_residual_component_area_px=(
                    minimum_residual_component_area_px
                ),
                minimum_residual_spacing_px=minimum_residual_spacing_px,
                residual_area_floor_fraction=residual_area_floor_fraction,
                maximum_residual_area_fraction=(
                    maximum_residual_area_fraction
                ),
                minimum_residual_component_fraction=(
                    minimum_residual_component_fraction
                ),
                maximum_dominant_residual_component_fraction=(
                    maximum_dominant_residual_component_fraction
                ),
            )
            selected_by_work = list(selected_by_work)
            if added:
                audit_lists[0].append(
                    {
                        "fragmentation_exact_tail_pixels_added": added,
                        "fragmentation_exact_tail_repair_calls": 1,
                    }
                )

    summarized: list[dict[str, Any]] = []
    for calls in audit_lists:
        totals: dict[str, int] = {}
        for call in calls:
            for key, value in call.items():
                totals[key] = totals.get(key, 0) + int(value)
        totals["call_count"] = len(calls)
        summarized.append(totals)
    return tuple(selected_by_work), tuple(summarized)


def _fragmentation_exact_area_backfill(
    selected_by_work: tuple[np.ndarray, ...],
    *,
    works: tuple[_CompilerWork, ...],
    source_region: np.ndarray,
    target_region: np.ndarray,
    requested_pixels: int,
    minimum_residual_components: int,
    maximum_residual_components: int,
    minimum_residual_component_area_px: int,
    minimum_residual_spacing_px: int,
    residual_area_floor_fraction: float,
    maximum_residual_area_fraction: float,
    minimum_residual_component_fraction: float,
    maximum_dominant_residual_component_fraction: float,
) -> tuple[tuple[np.ndarray, ...], int]:
    """Fill a tiny exact-area tail along existing owned fragmentation fronts."""

    requested = max(0, int(requested_pixels))
    if requested <= 0 or not selected_by_work:
        return selected_by_work, 0
    original = tuple(np.asarray(item, dtype=bool) for item in selected_by_work)
    updated = [np.array(item, copy=True) for item in original]
    structure = np.ones((3, 3), dtype=bool)
    for _ in range(requested):
        combined = np.logical_or.reduce(tuple(updated))
        options: list[tuple[float, int, int, int]] = []
        for index, (work, selected) in enumerate(zip(works, updated)):
            established_front = work.anchor_mask | ndimage.binary_dilation(
                selected, structure=structure
            )
            eligible = work.legal_source & ~combined & established_front
            for row, col in np.argwhere(eligible):
                options.append(
                    (float(work.priority[row, col]), index, int(row), int(col))
                )
        options.sort()
        accepted = False
        for _priority, index, row, col in options[:8192]:
            updated[index][row, col] = True
            audit = _whole_mask_topology_audit(
                source_region=source_region,
                target_region=target_region,
                selected_by_work=tuple(updated),
                works=works,
                allow_source_component_split=True,
                minimum_residual_components=minimum_residual_components,
                maximum_residual_components=maximum_residual_components,
                minimum_residual_component_area_px=(
                    minimum_residual_component_area_px
                ),
                minimum_residual_spacing_px=minimum_residual_spacing_px,
                residual_area_floor_fraction=residual_area_floor_fraction,
                maximum_residual_area_fraction=(
                    maximum_residual_area_fraction
                ),
                minimum_residual_component_fraction=(
                    minimum_residual_component_fraction
                ),
                maximum_dominant_residual_component_fraction=(
                    maximum_dominant_residual_component_fraction
                ),
            )
            if audit["passed"]:
                accepted = True
                break
            updated[index][row, col] = False
        if not accepted:
            return selected_by_work, 0
    return tuple(updated), requested


def _rebalance_fragmentation_residual_islands(
    selected_by_work: tuple[np.ndarray, ...],
    *,
    works: tuple[_CompilerWork, ...],
    source_region: np.ndarray,
    target_region: np.ndarray,
    minimum_residual_components: int,
    maximum_residual_components: int,
    minimum_residual_component_area_px: int,
    minimum_residual_spacing_px: int,
    residual_area_floor_fraction: float,
    maximum_residual_area_fraction: float = 1.0,
    minimum_residual_component_fraction: float = 0.0,
    maximum_dominant_residual_component_fraction: float = 1.0,
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    """Remove raster micro-islands without changing the resolved area.

    A generic priority front can complete a legal stromal corridor while
    leaving one or two source pixels surrounded by that corridor.  Those
    pixels are neither a biological residual focus nor evidence that the
    requested split is impossible, but the strict residual topology gate must
    still reject them.  This cleanup is therefore deliberately transactional:
    it fills every sub-minimum residual island, reclaims the same number of
    pixels from noncritical change edges, and commits only if the unchanged
    authoritative topology audit passes after every reclaimed pixel.
    """

    unchanged = {
        "applied": False,
        "pixels_added": 0,
        "pixels_reclaimed": 0,
        "tiny_pixels_added": 0,
        "spacing_pixels_added": 0,
        "balance_pixels_added": 0,
        "assigned_work_index": 0,
    }
    if not selected_by_work or minimum_residual_component_area_px <= 1:
        return selected_by_work, unchanged

    selected_source = np.logical_or.reduce(
        tuple(work.source_component for work in works)
    )
    combined = np.logical_or.reduce(selected_by_work)
    target_after = np.asarray(target_region, dtype=bool) | combined
    residual = selected_source & ~combined
    labels, _count = ndimage.label(
        residual, structure=np.ones((3, 3), dtype=bool)
    )
    sizes = np.bincount(labels.ravel())[1:]
    cleanup_ids = []
    tiny_ids = []
    balance_ids = []
    residual_total = max(int(sizes.sum()), 1)
    for index, size in enumerate(sizes.tolist(), start=1):
        size = int(size)
        if size <= 0:
            continue
        below_absolute_floor = size < int(minimum_residual_component_area_px)
        below_relative_floor = (
            size / residual_total + 1e-9
            < float(minimum_residual_component_fraction)
        )
        if not (below_absolute_floor or below_relative_floor):
            continue
        component = labels == index
        surrounding_ring = ndimage.binary_dilation(
            component, structure=np.ones((3, 3), dtype=bool)
        ) & ~component
        # A sub-minimum cap is repairable when converting it extends the
        # already-connected target compartment.  Requiring every ring pixel
        # to be target was too strict at three-label junctions: a one-pixel
        # tumor cap can be bounded by the new stromal corridor on one side and
        # an unedited third tissue class on the other.  The cap is still not a
        # biological residual focus, and filling it cannot create a target
        # island as long as at least one 8-neighbour already belongs to the
        # final target.  The transactional whole-mask audit below remains the
        # authority for component counts, holes, spacing and residual balance.
        if np.any(surrounding_ring) and np.any(target_after[surrounding_ring]):
            cleanup_ids.append(index)
            if below_absolute_floor:
                tiny_ids.append(index)
            else:
                balance_ids.append(index)
    cleanup_ids = tuple(cleanup_ids)
    tiny_ids = tuple(tiny_ids)
    balance_ids = tuple(balance_ids)
    focus_cleanup = np.isin(labels, cleanup_ids)
    tiny = np.isin(labels, tiny_ids)
    balance = np.isin(labels, balance_ids)
    tiny_pixels = int(np.count_nonzero(tiny))
    # Large accidental islands are a genuinely different topology and must
    # remain an abstention.  The 512-pixel cap is still far below a valid
    # residual focus in production-scale masks, but accommodates the aggregate
    # one- and two-pixel caps produced along several long raster corridors.
    if tiny_pixels > 512:
        return selected_by_work, unchanged
    balance_pixels = int(np.count_nonzero(balance))
    maximum_focus_cleanup_pixels = min(
        32768,
        max(4096, round(np.count_nonzero(selected_source) * 0.06)),
    )
    if int(np.count_nonzero(focus_cleanup)) > maximum_focus_cleanup_pixels:
        return selected_by_work, unchanged

    spacing = _fragmentation_spacing_repair(
        selected_source=selected_source,
        target_after=target_after | focus_cleanup,
        minimum_residual_spacing_px=minimum_residual_spacing_px,
        maximum_added_pixels=(
            maximum_focus_cleanup_pixels
            - int(np.count_nonzero(focus_cleanup))
        ),
    )
    if spacing is None:
        return selected_by_work, unchanged
    repair = focus_cleanup | spacing
    reclaimed = int(np.count_nonzero(repair))
    if reclaimed <= 0 or reclaimed > maximum_focus_cleanup_pixels:
        return selected_by_work, unchanged

    updated = [np.array(item, copy=True) for item in selected_by_work]
    assigned_indices: list[int] = []
    for row, col in np.argwhere(repair):
        assigned = next(
            (
                index
                for index, work in enumerate(works)
                if work.legal_source[row, col]
            ),
            next(
                (
                    index
                    for index, work in enumerate(works)
                    if work.source_component[row, col]
                ),
                None,
            ),
        )
        if assigned is None:
            return selected_by_work, unchanged
        updated[assigned][row, col] = True
        assigned_indices.append(assigned)

    filled_audit = _whole_mask_topology_audit(
        source_region=source_region,
        target_region=target_region,
        selected_by_work=tuple(updated),
        works=works,
        allow_source_component_split=True,
        minimum_residual_components=minimum_residual_components,
        maximum_residual_components=maximum_residual_components,
        minimum_residual_component_area_px=minimum_residual_component_area_px,
        minimum_residual_spacing_px=minimum_residual_spacing_px,
        residual_area_floor_fraction=residual_area_floor_fraction,
        maximum_residual_area_fraction=maximum_residual_area_fraction,
        minimum_residual_component_fraction=(
            minimum_residual_component_fraction
        ),
        maximum_dominant_residual_component_fraction=(
            maximum_dominant_residual_component_fraction
        ),
    )
    if not filled_audit["passed"]:
        return selected_by_work, unchanged

    removed = 0
    maximum_rounds = min(64, reclaimed + 1)
    for _reclaim_round in range(maximum_rounds):
        combined_after_fill = np.logical_or.reduce(tuple(updated))
        residual_after_fill = selected_source & ~combined_after_fill
        residual_edge = ndimage.binary_dilation(
            residual_after_fill, structure=np.ones((3, 3), dtype=bool)
        )
        candidates: list[tuple[float, int, int, int]] = []
        for index, (work, selected) in enumerate(zip(works, updated)):
            # Reclaim only along an existing residual-focus boundary.
            # Recompute this frontier after each successful layer so a broad
            # external retreat can donate more than one pixel of depth without
            # ever sampling an interior corridor pixel speculatively.
            removable = selected & ~repair & residual_edge
            for row, col in np.argwhere(removable):
                candidates.append(
                    (
                        float(work.priority[row, col]),
                        index,
                        int(row),
                        int(col),
                    )
                )
        candidates.sort(reverse=True)
        maximum_trials = min(8192, max(512, reclaimed * 8))
        progress = 0
        offset = 0
        while offset < min(len(candidates), maximum_trials):
            remaining = reclaimed - removed
            if remaining <= 0:
                break
            batch_size = min(512, remaining, len(candidates) - offset)
            accepted = False
            while batch_size >= 1:
                batch = candidates[offset : offset + batch_size]
                for _priority, index, row, col in batch:
                    updated[index][row, col] = False
                audit = _whole_mask_topology_audit(
                    source_region=source_region,
                    target_region=target_region,
                    selected_by_work=tuple(updated),
                    works=works,
                    allow_source_component_split=True,
                    minimum_residual_components=minimum_residual_components,
                    maximum_residual_components=maximum_residual_components,
                    minimum_residual_component_area_px=(
                        minimum_residual_component_area_px
                    ),
                    minimum_residual_spacing_px=minimum_residual_spacing_px,
                    residual_area_floor_fraction=residual_area_floor_fraction,
                    maximum_residual_area_fraction=(
                        maximum_residual_area_fraction
                    ),
                    minimum_residual_component_fraction=(
                        minimum_residual_component_fraction
                    ),
                    maximum_dominant_residual_component_fraction=(
                        maximum_dominant_residual_component_fraction
                    ),
                )
                if audit["passed"]:
                    removed += batch_size
                    progress += batch_size
                    offset += batch_size
                    accepted = True
                    break
                for _priority, index, row, col in batch:
                    updated[index][row, col] = True
                batch_size //= 2
            if not accepted:
                offset += 1
            else:
                # A successful batch changes the residual frontier. Rebuild
                # candidates in the outer loop before donating another layer;
                # continuing on the stale frontier needlessly probes interior
                # corridor pixels and can turn a linear repair into thousands
                # of full-raster audits.
                break
        if removed >= reclaimed:
            break
        if progress <= 0:
            break
    if removed != reclaimed:
        return selected_by_work, unchanged
    if sum(int(np.count_nonzero(item)) for item in updated) != sum(
        int(np.count_nonzero(item)) for item in selected_by_work
    ):
        return selected_by_work, unchanged
    return tuple(updated), {
        "applied": True,
        "pixels_added": reclaimed,
        "pixels_reclaimed": removed,
        "tiny_pixels_added": tiny_pixels,
        "spacing_pixels_added": int(np.count_nonzero(spacing)),
        "balance_pixels_added": balance_pixels,
        "assigned_work_index": min(assigned_indices),
    }


def _fragmentation_spacing_repair(
    *,
    selected_source: np.ndarray,
    target_after: np.ndarray,
    minimum_residual_spacing_px: int,
    maximum_added_pixels: int,
) -> np.ndarray | None:
    """Extend target-connected corridor edges until residual foci are spaced."""

    repair = np.zeros_like(selected_source, dtype=bool)
    required = max(0, int(minimum_residual_spacing_px))
    if required <= 0:
        return repair
    structure = np.ones((3, 3), dtype=bool)
    # Each iteration advances one target-connected boundary layer from one
    # side of the closest violating focus pair.  The bounded full topology
    # audit in the caller remains authoritative after all pairs are repaired.
    maximum_iterations = max(8, required * 16)
    for _ in range(maximum_iterations):
        current_target = np.asarray(target_after, dtype=bool) | repair
        residual = np.asarray(selected_source, dtype=bool) & ~current_target
        labels, count = ndimage.label(residual, structure=structure)
        violating_options: list[
            tuple[float, int, int, np.ndarray]
        ] = []
        target_frontier = ndimage.binary_dilation(
            current_target, structure=structure
        )
        for left in range(1, count):
            left_component = labels == left
            distance_to_left = ndimage.distance_transform_edt(
                ~left_component
            )
            for right in range(left + 1, count + 1):
                right_component = labels == right
                gap = float(
                    np.min(distance_to_left[right_component], initial=np.inf)
                )
                if gap + 1e-9 >= required:
                    continue
                distance_to_right = ndimage.distance_transform_edt(
                    ~right_component
                )
                left_edge = (
                    left_component
                    & target_frontier
                    & (distance_to_right < required)
                )
                right_edge = (
                    right_component
                    & target_frontier
                    & (distance_to_left < required)
                )
                for side, edge in enumerate((left_edge, right_edge)):
                    pixels = int(np.count_nonzero(edge))
                    if pixels > 0:
                        violating_options.append(
                            (gap, pixels, side, edge)
                        )
        if not violating_options:
            return repair
        _gap, _pixels, _side, chosen = min(
            violating_options,
            key=lambda item: (item[0], item[1], item[2]),
        )
        repair |= chosen
        if int(np.count_nonzero(repair)) > max(0, maximum_added_pixels):
            return None
    return None


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

"""Deterministic interface-bound candidate generation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy import ndimage

from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.models import (
    CandidateMask,
    DepthProfile,
    EditPlan,
    InterfaceExecutionContract,
    PlannedInterface,
    RefineContractError,
)
from phase3_mask_edit_refine.scene import SceneAnalysis
from phase3_mask_edit_refine.skills import ActiveKnowledgeBundle
from phase3_mask_edit_refine.tool_adapters import (
    TOOL_ADAPTER_VERSION,
    organic_v2_projection,
    smooth_noise,
)
from phase3_mask_edit_refine.topology import (
    protected_narrow_necks,
    source_deletion_limit,
    topology_safe_priority_grow,
)

SUPPORTED_TOOLS = frozenset({"interface_sdf", "connected_morphology", "organic_v2"})


@dataclass(frozen=True)
class _InterfaceWork:
    planned: PlannedInterface
    interface_mask: np.ndarray
    anchor_mask: np.ndarray
    anchor_masks: tuple[np.ndarray, ...]
    legal_source: np.ndarray
    target_fine_id: int
    capacity_px: int
    contact_px: int
    source_component: np.ndarray
    source_deletion_limit_px: int
    protected_source_necks: np.ndarray


def generate_candidates(
    source_mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    scene: SceneAnalysis,
    plan: EditPlan,
    bundle: ActiveKnowledgeBundle,
    seed: int,
    candidate_limit: int | None = None,
    compiled_replay_parts: tuple[Any, ...] | None = None,
    compiled_replay_audit: dict[str, Any] | None = None,
) -> tuple[CandidateMask, ...]:
    """Generate diverse candidates without accepting LLM-provided pixels.

    A candidate can use more than one planned interface.  The immutable area
    budget is distributed across broad legal interfaces, then each allocation
    is drawn independently and composed.  This prevents a large edit from
    being forced through one short boundary segment.
    """

    mask = np.asarray(source_mask)
    allowed_tools = tuple(
        tool
        for tool in plan.tool_program.allowed_tools
        if tool in bundle.edit_contract.allowed_tools and tool in SUPPORTED_TOOLS
    )
    if not allowed_tools:
        raise RefineContractError("EditPlan contains no supported deterministic tools")

    source_ids = tuple(
        int(value)
        for value in plan.tool_program.parameter_ranges.get(
            "editable_source_fine_ids", ()
        )
    ) or tuple(
        sorted(
            {
                fine_id
                for label in plan.source_labels
                for fine_id in schema.resolve_fine_ids(label)
            }
        )
    )
    source_region = np.isin(mask, source_ids)
    desired_pixels = plan.area_budget.target_pixels(mask, source_region)
    target_pixels = (
        plan.resolved_area.resolved_pixels
        if plan.resolved_area is not None
        else desired_pixels
    )
    work_items = _prepare_interfaces(
        mask,
        schema=schema,
        scene=scene,
        plan=plan,
        source_region=source_region,
    )
    if not work_items:
        raise RefineContractError("EditPlan contains no interface with legal pixels")

    count = plan.tool_program.candidate_count
    if candidate_limit is not None:
        count = min(count, max(1, int(candidate_limit)))
    candidates: list[CandidateMask] = []
    seen_hashes: set[bytes] = set()
    if plan.resolved_area is not None:
        # Local import avoids a module cycle: the compiler imports the shared
        # depth-profile compiler from this module during initialization.
        from phase3_mask_edit_refine.execution import replay_compiled_edit_plan

        if compiled_replay_parts is None or compiled_replay_audit is None:
            replay_parts, replay_audit = replay_compiled_edit_plan(
                plan,
                source_mask=mask,
                schema=schema,
                scene=scene,
            )
        else:
            replay_parts = compiled_replay_parts
            replay_audit = compiled_replay_audit
        replay_change = np.zeros_like(mask, dtype=bool)
        replay_target = np.array(mask, copy=True)
        replay_traces = []
        replay_target_fine_ids = []
        for part_index, part in enumerate(replay_parts):
            target_fine_id = _target_fine_id(
                mask,
                schema=schema,
                scene=scene,
                target_component_id=part.planned.target_component_id,
                target_label=plan.target_label,
            )
            change = np.asarray(part.change_region, dtype=bool)
            replay_change |= change
            replay_target[change] = int(target_fine_id)
            replay_target_fine_ids.append(int(target_fine_id))
            profile = part.planned.execution_contract.depth_profile
            replay_traces.append(
                {
                    "interface_id": part.planned.interface_id,
                    "source_component_id": part.planned.source_component_id,
                    "target_component_id": part.planned.target_component_id,
                    "target_fine_id": int(target_fine_id),
                    "allocated_pixels": int(np.count_nonzero(change)),
                    "realized_pixels": int(np.count_nonzero(change)),
                    "requested_anchor_segment_ids": list(
                        part.planned.execution_contract.anchor_segment_ids
                    ),
                    "requested_area_allocation_fraction": float(
                        part.planned.execution_contract.area_allocation_fraction
                    ),
                    "allowed_band_px": list(part.planned.allowed_edit_band_px),
                    "noise_amplitude_px": 0.0,
                    "depth_profile": {
                        "mode": profile.mode,
                        "peak_depth_px": profile.peak_depth_px,
                        "edge_depth_px": profile.edge_depth_px,
                        "taper_fraction": profile.taper_fraction,
                        "lobe_count": profile.lobe_count,
                        "noise_correlation_px": profile.noise_correlation_px,
                    },
                    "shape_variant": 0,
                    "available_legal_pixels": int(part.legal_capacity_px),
                    "anchor": {
                        "mode": "planner_selected_executable_anchors",
                        "anchor_segment_ids": list(
                            part.planned.execution_contract.anchor_segment_ids
                        ),
                        "anchor_pixels": int(np.count_nonzero(part.anchor_mask)),
                    },
                    "topology_safe_growth": dict(part.topology_audit),
                    "seed": int(part_index * 13007),
                }
            )
        replay_pixels = int(np.count_nonzero(replay_change))
        if replay_pixels != target_pixels:
            raise RefineContractError(
                "compiled replay changed an unexpected number of pixels: "
                f"expected={target_pixels}, observed={replay_pixels}"
            )
        interface_ids = [part.planned.interface_id for part in replay_parts]
        source_component_ids = sorted(
            {part.planned.source_component_id for part in replay_parts}
        )
        target_component_ids = sorted(
            {part.planned.target_component_id for part in replay_parts}
        )
        target_fine_ids = sorted(set(replay_target_fine_ids))
        replay_trace = {
            "seed": int(seed),
            "target_fine_id": int(target_fine_ids[0]),
            "target_fine_ids": target_fine_ids,
            "tool_adapter_version": TOOL_ADAPTER_VERSION,
            "requested_target_pixels": int(target_pixels),
            "desired_target_pixels": int(desired_pixels),
            "resolved_target_pixels": int(target_pixels),
            "area_fallback_used": bool(plan.resolved_area.used_fallback),
            "area_binding_constraint": plan.resolved_area.binding_constraint,
            "interface_ids": interface_ids,
            "source_component_id": source_component_ids[0],
            "source_component_ids": source_component_ids,
            "target_component_id": target_component_ids[0],
            "target_component_ids": target_component_ids,
            "shape_variant": 0,
            "parts": replay_traces,
            "compiled_topology_replay": replay_audit,
        }
        candidates.append(
            CandidateMask(
                candidate_id="cand:001",
                interface_id=interface_ids[0],
                tool_name="interface_sdf",
                target_mask=replay_target,
                change_region=replay_change,
                tool_trace=replay_trace,
            )
        )
        seen_hashes.add(
            np.packbits(replay_change, axis=None).tobytes()
            + replay_target[replay_change].tobytes()
        )
    # Organic-v2 remains an available deterministic family, but its legacy
    # projection recomputes full-patch nearest-label fields.  Repeating it for
    # every parameter round produced minute-scale tails without adding a new
    # interface hypothesis.  Exercise each allowed family once, then spend the
    # remaining diversity budget on the fast SDF/morphology families.  If a
    # contract allows only organic-v2 it is necessarily repeated.
    fast_tools = tuple(tool for tool in allowed_tools if tool != "organic_v2")
    initial_tool_round = allowed_tools
    repeat_tool_round = fast_tools or allowed_tools
    max_attempts = max(count * 8, 48)
    for variation in range(max_attempts):
        if len(candidates) >= count:
            break
        if variation < len(initial_tool_round):
            tool_name = initial_tool_round[variation]
            shape_variant = 0
        else:
            repeat_index = variation - len(initial_tool_round)
            tool_name = repeat_tool_round[
                repeat_index % len(repeat_tool_round)
            ]
            shape_variant = 1 + repeat_index // len(repeat_tool_round)
        active = work_items
        allocations = _allocate_pixels(active, target_pixels=target_pixels)
        if sum(allocations) < target_pixels:
            continue

        combined_change = np.zeros_like(mask, dtype=bool)
        target = np.array(mask, copy=True)
        source_states = {
            work.planned.source_component_id: np.array(
                work.source_component, copy=True
            )
            for work in active
        }
        deleted_by_source = {
            component_id: 0 for component_id in source_states
        }
        target_ids = tuple(schema.resolve_fine_ids(plan.target_label))
        target_state = np.isin(mask, target_ids)
        selected_target = np.zeros_like(mask, dtype=bool)
        for work in active:
            selected_target |= scene.component_masks[
                work.planned.target_component_id
            ]
        unselected_target = target_state & ~selected_target
        part_traces: list[dict[str, Any]] = []
        for part_index, (work, allocation) in enumerate(zip(active, allocations)):
            if allocation <= 0:
                continue
            legal_source = work.legal_source & ~combined_change
            candidate_seed = int(seed + variation * 104729 + part_index * 13007)
            change, _, trace = _generate_one(
                mask,
                schema=schema,
                legal_source=legal_source,
                anchor_mask=work.anchor_mask,
                anchor_masks=work.anchor_masks,
                planned_band=work.planned.allowed_edit_band_px,
                execution=work.planned.execution_contract,
                source_labels=plan.source_labels,
                target_label=plan.target_label,
                target_fine_id=work.target_fine_id,
                primitive_id=plan.primitive_id,
                target_pixels=allocation,
                tool_name=tool_name,
                seed=candidate_seed,
                shape_variant=shape_variant,
                parameter_ranges=plan.tool_program.parameter_ranges,
                source_component_state=source_states[
                    work.planned.source_component_id
                ],
                target_state=target_state,
                unselected_target=unselected_target,
                maximum_source_deletions=work.source_deletion_limit_px,
                already_deleted_from_source=deleted_by_source[
                    work.planned.source_component_id
                ],
                protected_source_necks=work.protected_source_necks,
                topology_seed=(
                    part_index * 13007
                    if variation == 0 and tool_name == "interface_sdf"
                    else candidate_seed
                ),
            )
            combined_change |= change
            deleted_by_source[work.planned.source_component_id] += int(
                np.count_nonzero(change)
            )
            target[change] = int(work.target_fine_id)
            part_traces.append(
                {
                    **trace,
                    "interface_id": work.planned.interface_id,
                    "source_component_id": work.planned.source_component_id,
                    "target_component_id": work.planned.target_component_id,
                    "target_fine_id": int(work.target_fine_id),
                    "allocated_pixels": int(allocation),
                    "realized_pixels": int(np.count_nonzero(change)),
                    "requested_anchor_segment_ids": list(
                        work.planned.execution_contract.anchor_segment_ids
                    ),
                    "requested_area_allocation_fraction": float(
                        work.planned.execution_contract.area_allocation_fraction
                    ),
                    "seed": candidate_seed,
                }
            )

        if int(np.count_nonzero(combined_change)) != target_pixels:
            continue
        digest = np.packbits(combined_change, axis=None).tobytes() + target[combined_change].tobytes()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        interface_ids = [item.planned.interface_id for item in active]
        source_component_ids = sorted({item.planned.source_component_id for item in active})
        target_component_ids = sorted({item.planned.target_component_id for item in active})
        target_fine_ids = sorted({int(item.target_fine_id) for item in active})
        trace = {
            "seed": int(seed + variation * 104729),
            "target_fine_id": int(target_fine_ids[0]),
            "target_fine_ids": target_fine_ids,
            "tool_adapter_version": TOOL_ADAPTER_VERSION,
            "requested_target_pixels": int(target_pixels),
            "desired_target_pixels": int(desired_pixels),
            "resolved_target_pixels": int(target_pixels),
            "area_fallback_used": bool(
                plan.resolved_area.used_fallback
                if plan.resolved_area is not None
                else False
            ),
            "area_binding_constraint": (
                plan.resolved_area.binding_constraint
                if plan.resolved_area is not None
                else "uncompiled_exact_target"
            ),
            "interface_ids": interface_ids,
            "source_component_id": source_component_ids[0],
            "source_component_ids": source_component_ids,
            "target_component_id": target_component_ids[0],
            "target_component_ids": target_component_ids,
            "shape_variant": int(shape_variant),
            "parts": part_traces,
        }
        candidates.append(
            CandidateMask(
                candidate_id=f"cand:{len(candidates) + 1:03d}",
                interface_id=interface_ids[0],
                tool_name=tool_name,
                target_mask=target,
                change_region=combined_change,
                tool_trace=trace,
            )
        )
    if not candidates:
        raise RefineContractError(
            "deterministic generators could not realize the area budget on legal interfaces"
        )
    return tuple(candidates)


def _prepare_interfaces(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    scene: SceneAnalysis,
    plan: EditPlan,
    source_region: np.ndarray,
) -> tuple[_InterfaceWork, ...]:
    prohibited = np.zeros_like(mask, dtype=bool)
    for region in scene.prohibited_region_masks.values():
        prohibited |= np.asarray(region, dtype=bool)
    result: list[_InterfaceWork] = []
    graph_interfaces = {item.interface_id: item for item in scene.graph.interfaces}
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
    resolved_anchor_groups = tuple(
        tuple(
            scene.anchor_masks[anchor_id]
            for anchor_id in planned.execution_contract.anchor_segment_ids
            if anchor_id in scene.anchor_masks
        )
        for planned in plan.candidate_interfaces
    )
    if any(
        len(group) != len(planned.execution_contract.anchor_segment_ids)
        for group, planned in zip(resolved_anchor_groups, plan.candidate_interfaces)
    ):
        return ()
    anchor_unions = tuple(
        np.logical_or.reduce(group) for group in resolved_anchor_groups
    )
    provisional: list[dict[str, Any]] = []
    for planned_index, planned in enumerate(plan.candidate_interfaces):
        interface_mask = scene.interface_masks.get(planned.interface_id)
        source_component = scene.component_masks.get(planned.source_component_id)
        observed = graph_interfaces.get(planned.interface_id)
        if interface_mask is None or source_component is None or observed is None:
            continue
        selected_anchor_masks = resolved_anchor_groups[planned_index]
        anchor_mask = anchor_unions[planned_index]
        _distance, nearest = ndimage.distance_transform_edt(
            ~interface_mask, return_indices=True
        )
        nearest_belongs_to_anchor = anchor_mask[nearest[0], nearest[1]]
        anchor_distance = ndimage.distance_transform_edt(~anchor_mask)
        band_min, band_max = planned.allowed_edit_band_px
        requested = planned.execution_contract.depth_profile
        peak_depth = max(requested.peak_depth_px, 1e-6)
        unit_profile = replace(
            requested,
            peak_depth_px=1.0,
            edge_depth_px=float(
                np.clip(requested.edge_depth_px / peak_depth, 0.0, 1.0)
            ),
            noise_amplitude_px=0.0,
        )
        unit_depth = compile_depth_profile_map(
            selected_anchor_masks,
            profile=unit_profile,
            shape=mask.shape,
        )
        required_scale = anchor_distance / np.maximum(unit_depth, 1e-3)
        ownership_envelope = (
            source_component
            & source_region
            & ~prohibited
            & nearest_belongs_to_anchor
            & (anchor_distance >= max(0.0, band_min))
            & (anchor_distance <= band_max)
            & (required_scale <= band_max + 1e-6)
        )
        legal_envelope = ownership_envelope & (
            required_scale <= peak_depth * 1.001 + 1e-9
        )
        provisional.append(
            {
                "planned": planned,
                "interface_mask": interface_mask,
                "anchor_mask": anchor_mask,
                "anchor_masks": selected_anchor_masks,
                "ownership_envelope": ownership_envelope,
                "legal_envelope": legal_envelope,
                "owner_cost": required_scale,
                "target_fine_id": _target_fine_id(
                    mask,
                    schema=schema,
                    scene=scene,
                    target_component_id=planned.target_component_id,
                    target_label=plan.target_label,
                ),
                "contact_px": max(1, int(observed.contact_pixels)),
                "source_component": np.asarray(source_component, dtype=bool),
                "source_deletion_limit_px": source_deletion_limit(
                    int(np.count_nonzero(source_component)),
                    maximum_changed_fraction=maximum_changed_fraction,
                    minimum_remaining_pixels=minimum_remaining,
                ),
                "protected_source_necks": protected_narrow_necks(
                    source_component
                ),
            }
        )
    if not provisional:
        return ()

    # Match the compiler's eligibility-first ownership rule on the immutable
    # full allowed band.  The compiled peak then only narrows each fixed
    # owner's executable pixels; it must not repartition ownership after the
    # compiler has already certified per-interface allocations.
    owner_cost = np.stack(
        [
            np.where(item["ownership_envelope"], item["owner_cost"], np.inf)
            for item in provisional
        ]
    )
    assignment = np.argmin(owner_cost, axis=0)
    has_owner = np.any(np.isfinite(owner_cost), axis=0)
    for planned_index, item in enumerate(provisional):
        legal = item["legal_envelope"] & has_owner & (assignment == planned_index)
        capacity = int(np.count_nonzero(legal))
        if capacity <= 0:
            continue
        result.append(
            _InterfaceWork(
                planned=item["planned"],
                interface_mask=item["interface_mask"],
                anchor_mask=item["anchor_mask"],
                anchor_masks=item["anchor_masks"],
                legal_source=legal,
                target_fine_id=item["target_fine_id"],
                capacity_px=capacity,
                contact_px=item["contact_px"],
                source_component=item["source_component"],
                source_deletion_limit_px=item["source_deletion_limit_px"],
                protected_source_necks=item["protected_source_necks"],
            )
        )
    return tuple(result)


def _allocate_pixels(
    items: tuple[_InterfaceWork, ...], *, target_pixels: int
) -> tuple[int, ...]:
    if not items:
        return ()
    weights = np.asarray(
        [item.planned.execution_contract.area_allocation_fraction for item in items],
        dtype=float,
    )
    raw = weights * target_pixels
    allocations = np.floor(raw).astype(int)
    remainder = target_pixels - int(allocations.sum())
    order = sorted(
        range(len(items)), key=lambda index: (-(raw[index] - allocations[index]), index)
    )
    for index in order[:remainder]:
        allocations[index] += 1
    capacities = np.asarray([item.capacity_px for item in items])
    if np.any(allocations > capacities):
        return tuple(0 for _ in items)
    return tuple(int(value) for value in allocations)


def _generate_one(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    legal_source: np.ndarray,
    anchor_mask: np.ndarray,
    anchor_masks: tuple[np.ndarray, ...],
    planned_band: tuple[float, float],
    execution: InterfaceExecutionContract,
    source_labels: tuple[str, ...],
    target_label: str,
    target_fine_id: int,
    primitive_id: str,
    target_pixels: int,
    tool_name: str,
    seed: int,
    shape_variant: int,
    parameter_ranges: dict[str, Any],
    source_component_state: np.ndarray,
    target_state: np.ndarray,
    unselected_target: np.ndarray,
    maximum_source_deletions: int,
    already_deleted_from_source: int,
    protected_source_necks: np.ndarray,
    topology_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    allow_source_resolution = bool(
        parameter_ranges.get("allow_source_component_resolution", False)
    )
    allow_target_hole_resolution = bool(
        parameter_ranges.get("allow_target_hole_resolution", False)
    )
    allow_source_component_split = bool(
        parameter_ranges.get("allow_source_component_split", False)
    )
    effective_protected_necks = (
        None
        if (allow_source_resolution or allow_source_component_split)
        else protected_source_necks
    )
    distance = ndimage.distance_transform_edt(~anchor_mask)
    band_min, band_max = planned_band
    profile = execution.depth_profile
    geometric_allowed = (
        legal_source
        & (distance >= max(0.0, band_min))
        & (distance <= min(band_max, profile.peak_depth_px))
    )
    desired_depth = compile_depth_profile_map(
        anchor_masks, profile=profile, shape=mask.shape
    )
    normalized_distance = distance / np.maximum(desired_depth, 1e-3)
    # The compiled profile is a pixel-binding envelope, not merely a priority
    # hint. Keeping the same normalization as the compiler and fidelity gate
    # eliminates silent execution drift at strongly tapered sub-pixel ends.
    allowed = geometric_allowed & (normalized_distance <= 1.001 + 1e-9)
    # The zero-noise member exactly reproduces the compiler's feasibility
    # ordering; later members explore controlled organic variation.
    amplitude_factors = (0.0, 0.75, 1.0, 1.25, 0.90)
    noise_amplitude = profile.noise_amplitude_px * amplitude_factors[
        shape_variant % len(amplitude_factors)
    ]
    noise = smooth_noise(
        mask.shape,
        seed=seed,
        amplitude=noise_amplitude / max(profile.peak_depth_px, 1.0),
        correlation_px=profile.noise_correlation_px,
    )
    anchor_trace = {
        "mode": "planner_selected_executable_anchors",
        "anchor_segment_ids": list(execution.anchor_segment_ids),
        "anchor_pixels": int(np.count_nonzero(anchor_mask)),
    }

    if tool_name == "organic_v2":
        template_depth = 1.0
        raw_template = allowed & ((normalized_distance + noise) <= template_depth)
        _, template_change, trace = organic_v2_projection(
            mask,
            raw_template,
            schema=schema,
            source_labels=source_labels,
            target_label=target_label,
            primitive_name=_legacy_primitive_name(primitive_id),
            target_pixels=target_pixels,
            seed=seed,
            primitive_config={
                "organic_projection": {
                    "template_spillover_fraction": 0.03 + 0.02 * (shape_variant % 4),
                    "min_component_fraction": 0.02,
                }
            },
        )
        template_change = np.asarray(template_change, dtype=bool) & allowed
        priority = normalized_distance + noise - 0.35 * template_change.astype(float)
        change, topology_audit = topology_safe_priority_grow(
            allowed,
            interface_mask=anchor_mask,
            target_pixels=target_pixels,
            priority=priority,
            source_component_state=source_component_state,
            target_state=target_state,
            unselected_target=unselected_target,
            maximum_source_deletions=maximum_source_deletions,
            already_deleted_from_source=already_deleted_from_source,
            protected_source_necks=effective_protected_necks,
            seed=topology_seed + 17,
            allow_source_component_resolution=allow_source_resolution,
            allow_target_hole_resolution=allow_target_hole_resolution,
            allow_source_component_split=allow_source_component_split,
        )
        target = np.array(mask, copy=True)
        target[change] = int(target_fine_id)
        return change, target, {
            "organic_v2": trace,
            "allowed_band_px": list(planned_band),
            "noise_amplitude_px": noise_amplitude,
            "template_depth_normalized": template_depth,
            "depth_profile": {
                "mode": profile.mode,
                "peak_depth_px": profile.peak_depth_px,
                "edge_depth_px": profile.edge_depth_px,
                "taper_fraction": profile.taper_fraction,
                "lobe_count": profile.lobe_count,
                "noise_correlation_px": profile.noise_correlation_px,
            },
            "anchor": anchor_trace,
            "topology_safe_growth": topology_audit.to_metadata(),
        }

    if tool_name == "connected_morphology":
        # A low-frequency basin favours one broad lobe rather than an equal-width
        # ribbon. The growth itself remains connected to the original interface.
        noise = 0.55 * noise + 0.45 * smooth_noise(
            mask.shape,
            seed=seed + 7919,
            amplitude=(noise_amplitude / max(profile.peak_depth_px, 1.0)) * 1.4,
            correlation_px=profile.noise_correlation_px,
        )
    selected, topology_audit = topology_safe_priority_grow(
        allowed,
        interface_mask=anchor_mask,
        target_pixels=target_pixels,
        priority=normalized_distance + noise,
        source_component_state=source_component_state,
        target_state=target_state,
        unselected_target=unselected_target,
        maximum_source_deletions=maximum_source_deletions,
        already_deleted_from_source=already_deleted_from_source,
        protected_source_necks=effective_protected_necks,
        seed=topology_seed,
        allow_source_component_resolution=allow_source_resolution,
        allow_target_hole_resolution=allow_target_hole_resolution,
        allow_source_component_split=allow_source_component_split,
    )
    target = np.array(mask, copy=True)
    target[selected] = int(target_fine_id)
    return selected, target, {
        "allowed_band_px": list(planned_band),
        "noise_amplitude_px": noise_amplitude,
        "depth_profile": {
            "mode": profile.mode,
            "peak_depth_px": profile.peak_depth_px,
            "edge_depth_px": profile.edge_depth_px,
            "taper_fraction": profile.taper_fraction,
            "lobe_count": profile.lobe_count,
            "noise_correlation_px": profile.noise_correlation_px,
        },
        "shape_variant": shape_variant,
        "available_legal_pixels": int(np.count_nonzero(allowed)),
        "anchor": anchor_trace,
        "topology_safe_growth": topology_audit.to_metadata(),
    }


def compile_depth_profile_map(
    anchor_masks: tuple[np.ndarray, ...], *, profile: DepthProfile, shape: tuple[int, int]
) -> np.ndarray:
    """Compile the Planner depth profile into a per-pixel desired depth field."""

    anchor_union = np.logical_or.reduce(anchor_masks)
    _, nearest = ndimage.distance_transform_edt(~anchor_union, return_indices=True)
    depth_at_anchor = np.zeros(shape, dtype=float)
    # Consecutive selected anchor IDs are only addressable chunks, not separate
    # biological lobes. Merge touching chunks and taper at the true outer ends.
    grouped, group_count = ndimage.label(
        anchor_union, structure=np.ones((3, 3), dtype=bool)
    )
    for group_index in range(1, group_count + 1):
        anchor = grouped == group_index
        coordinates = np.argwhere(anchor)
        if coordinates.size == 0:
            continue
        if coordinates.shape[0] == 1 or profile.mode == "uniform_front":
            values = np.full(coordinates.shape[0], profile.peak_depth_px, dtype=float)
        else:
            centered = coordinates.astype(float) - np.mean(
                coordinates, axis=0, keepdims=True
            )
            covariance = np.cov(centered, rowvar=False)
            axis = np.linalg.eigh(covariance)[1][:, -1]
            projection = centered @ axis
            low, high = float(np.min(projection)), float(np.max(projection))
            position = (projection - low) / max(high - low, 1e-6)
            taper = max(profile.taper_fraction, 1e-6)
            ramp = np.minimum(
                1.0, np.minimum(position / taper, (1.0 - position) / taper)
            )
            ramp = np.clip(ramp, 0.0, 1.0)
            smooth_ramp = ramp * ramp * (3.0 - 2.0 * ramp)
            if profile.mode == "multi_lobe" and profile.lobe_count > 1:
                modulation = 0.72 + 0.28 * np.sin(
                    np.pi * profile.lobe_count * position
                ) ** 2
                smooth_ramp *= modulation
            values = profile.edge_depth_px + (
                profile.peak_depth_px - profile.edge_depth_px
            ) * smooth_ramp
        depth_at_anchor[coordinates[:, 0], coordinates[:, 1]] = values
    return depth_at_anchor[nearest[0], nearest[1]]


def _target_fine_id(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    scene: SceneAnalysis,
    target_component_id: str,
    target_label: str,
) -> int:
    fine_ids = tuple(schema.resolve_fine_ids(target_label))
    component = scene.component_masks[target_component_id]
    values = mask[component]
    values = values[np.isin(values, fine_ids)]
    if values.size:
        unique, counts = np.unique(values, return_counts=True)
        return int(unique[int(np.argmax(counts))])
    return int(fine_ids[0])


def _legacy_primitive_name(primitive_id: str) -> str:
    return primitive_id.removesuffix("-v1").replace("-", "_")

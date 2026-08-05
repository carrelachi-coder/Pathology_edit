"""Fail-closed deterministic candidate gates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.candidates import compile_depth_profile_map
from phase3_mask_edit_refine.models import (
    CandidateMask,
    CaseContext,
    EditPlan,
    GateCheck,
    GateReport,
)
from phase3_mask_edit_refine.scene import SceneAnalysis, profile_signature_metrics
from phase3_mask_edit_refine.skills import ActiveKnowledgeBundle

GateFunction = Callable[["GateContext"], GateCheck]


BASE_REQUIRED_CHECKS = (
    "profile_signature_consistency",
    "profile_required_provenance",
    "semantic_capability_guard",
    "background_seed_protection",
    "label_transition",
    "unrequested_labels_preserved",
    "changed_area",
    "interface_contact",
    "execution_contract_fidelity",
    "prohibited_region",
    "component_topology",
    "edited_label_topology",
    "source_component_retention",
    "depth_span_ratio",
    "boundary_naturalness",
    "parallel_boundary_artifact",
    "provenance_complete",
    "tumor_stroma_interface",
)


@dataclass(frozen=True)
class GateContext:
    case: CaseContext
    source_mask: np.ndarray
    schema: MaskProfileSchema
    scene: SceneAnalysis
    bundle: ActiveKnowledgeBundle
    plan: EditPlan
    candidate: CandidateMask


class GateRegistry:
    """Registry that treats missing required checks as a startup error."""

    def __init__(self) -> None:
        self._checks: dict[str, GateFunction] = {
            "profile_signature_consistency": _check_profile_signature,
            "profile_required_provenance": _check_profile_required_provenance,
            "semantic_capability_guard": _check_semantic_capability,
            "background_seed_protection": _check_background_seed_protection,
            "label_transition": _check_label_transition,
            "unrequested_labels_preserved": _check_unrequested_preserved,
            "changed_area": _check_changed_area,
            "interface_contact": _check_interface_contact,
            "execution_contract_fidelity": _check_execution_contract_fidelity,
            "prohibited_region": _check_prohibited_regions,
            "component_topology": _check_component_topology,
            "edited_label_topology": _check_edited_label_topology,
            "source_component_retention": _check_source_component_retention,
            "depth_span_ratio": _check_depth_span_ratio,
            "boundary_naturalness": _check_boundary_naturalness,
            "parallel_boundary_artifact": _check_parallel_boundary_artifact,
            "provenance_complete": _check_provenance,
            "tumor_stroma_interface": _check_tumor_stroma_interface,
        }
        missing = sorted(set(BASE_REQUIRED_CHECKS) - set(self._checks))
        if missing:
            raise RuntimeError("required gate implementations missing: " + ", ".join(missing))

    @property
    def available_checker_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._checks))

    def run(self, context: GateContext) -> GateReport:
        requested = list(BASE_REQUIRED_CHECKS)
        for check_id in context.bundle.edit_contract.required_check_ids:
            if check_id not in requested:
                requested.append(check_id)
        missing = [check_id for check_id in requested if check_id not in self._checks]
        if missing:
            checks = tuple(
                GateCheck(
                    check_id=check_id,
                    passed=False,
                    severity="hard",
                    detail="required checker is not registered",
                )
                for check_id in missing
            )
            return GateReport(context.candidate.candidate_id, False, checks)
        checks = tuple(self._checks[check_id](context) for check_id in requested)
        return GateReport(
            candidate_id=context.candidate.candidate_id,
            passed=all(check.passed for check in checks if check.severity == "hard"),
            checks=checks,
        )


def _check_profile_signature(context: GateContext) -> GateCheck:
    metrics = profile_signature_metrics(context.source_mask, schema=context.schema)
    unknown = metrics["unknown_ids"]
    empirical = context.bundle.annotation_profile.capabilities.get("empirical_statistics")
    outliers: dict[str, dict[str, float]] = {}
    if isinstance(empirical, dict):
        for key in (
            "background_fraction",
            "background_components_per_mpx",
            "background_border_connected_fraction",
            "internal_background_fraction",
        ):
            envelope = empirical.get(key)
            observed = metrics.get(key)
            if not isinstance(envelope, dict) or not isinstance(observed, (int, float)):
                continue
            lower, upper = envelope.get("p01"), envelope.get("p99")
            if (
                isinstance(lower, (int, float))
                and isinstance(upper, (int, float))
                and not float(lower) <= float(observed) <= float(upper)
            ):
                outliers[key] = {
                    "observed": float(observed),
                    "p01": float(lower),
                    "p99": float(upper),
                }
    passed = not unknown and not outliers
    metrics["empirical_signature_outliers"] = outliers
    return _result(
        "profile_signature_consistency",
        passed,
        "mask IDs match declared annotation profile"
        if passed
        else f"unknown fine IDs={unknown}; empirical signature outliers={outliers}",
        metrics=metrics,
    )


def _check_profile_required_provenance(context: GateContext) -> GateCheck:
    required = context.bundle.annotation_profile.capabilities.get(
        "required_provenance_fields", []
    )
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        return _result(
            "profile_required_provenance",
            False,
            "annotation profile required_provenance_fields is malformed",
        )
    missing = [field for field in required if not context.case.provenance.get(field)]
    passed = not missing
    return _result(
        "profile_required_provenance",
        passed,
        "all annotation-profile provenance fields are present"
        if passed
        else "missing annotation-profile provenance: " + ", ".join(missing),
        metrics={"required_fields": required, "missing_fields": missing},
    )


def _check_semantic_capability(context: GateContext) -> GateCheck:
    contract = context.bundle.edit_contract
    source_set = set(context.plan.source_labels)
    passed = (
        context.plan.target_label == contract.target_label
        and bool(source_set)
        and source_set.issubset(contract.source_label_options)
        and context.plan.target_label in context.schema.readable_labels
        and source_set.issubset(context.schema.readable_labels)
        and context.plan.target_label not in source_set
    )
    return _result(
        "semantic_capability_guard",
        passed,
        "plan semantics are within the composed skill/profile capability"
        if passed
        else "plan source/target semantics exceed the composed capability",
        metrics={
            "plan_source_labels": sorted(source_set),
            "allowed_source_labels": list(contract.source_label_options),
            "plan_target_label": context.plan.target_label,
            "allowed_target_label": contract.target_label,
        },
    )


def _check_background_seed_protection(context: GateContext) -> GateCheck:
    source = np.asarray(context.source_mask)
    target = np.asarray(context.candidate.target_mask)
    source_background = np.isin(source, tuple(context.schema.skip_fine_ids))
    target_background = np.isin(target, tuple(context.schema.skip_fine_ids))
    background_changed = int(np.count_nonzero(source_background != target_background))
    background_pixels_edited = int(
        np.count_nonzero(source_background & context.candidate.change_region)
    )
    source_component_ids = _trace_ids(
        context.candidate, plural="source_component_ids", singular="source_component_id"
    )
    target_component_ids = _trace_ids(
        context.candidate, plural="target_component_ids", singular="target_component_id"
    )
    source_component_valid = bool(source_component_ids) and all(
        item in context.scene.component_masks for item in source_component_ids
    )
    target_component_valid = bool(target_component_ids) and all(
        item in context.scene.component_masks for item in target_component_ids
    )
    policy = context.bundle.annotation_profile.capabilities.get("background_policy", {})
    policy_valid = (
        isinstance(policy, dict)
        and policy.get("editable") is False
        and policy.get("may_seed_edit") is False
        and policy.get("preserve_exactly") is True
    )
    passed = (
        policy_valid
        and background_changed == 0
        and background_pixels_edited == 0
        and source_component_valid
        and target_component_valid
    )
    return _result(
        "background_seed_protection",
        passed,
        "background is pixel-exact and neither selected component is background"
        if passed
        else "background changed, entered the edit, seeded a component, or policy is incomplete",
        metrics={
            "background_changed_pixels": background_changed,
            "background_pixels_in_change_region": background_pixels_edited,
            "source_component_valid": source_component_valid,
            "target_component_valid": target_component_valid,
            "background_policy": policy if isinstance(policy, dict) else None,
        },
    )
def _check_label_transition(context: GateContext) -> GateCheck:
    source = np.asarray(context.source_mask)
    target = np.asarray(context.candidate.target_mask)
    change = np.asarray(context.candidate.change_region, dtype=bool)
    actual_diff = source != target
    source_ids = tuple(
        sorted(
            {
                fine_id
                for label in context.plan.source_labels
                for fine_id in context.schema.resolve_fine_ids(label)
            }
        )
    )
    target_ids_raw = context.candidate.tool_trace.get("target_fine_ids")
    if isinstance(target_ids_raw, list) and all(isinstance(item, int) for item in target_ids_raw):
        expected_target_ids = tuple(target_ids_raw)
    else:
        expected_target_id = context.candidate.tool_trace.get("target_fine_id")
        expected_target_ids = (expected_target_id,) if isinstance(expected_target_id, int) else ()
    diff_matches = np.array_equal(actual_diff, change)
    changed_source_legal = bool(np.all(np.isin(source[change], source_ids))) if np.any(change) else False
    target_legal = (
        bool(expected_target_ids)
        and bool(np.all(np.isin(target[change], expected_target_ids)))
        if np.any(change)
        else False
    )
    passed = diff_matches and changed_source_legal and target_legal
    return _result(
        "label_transition",
        passed,
        "all diff pixels follow the declared source-to-target transition"
        if passed
        else "candidate diff contains an undeclared source, target, or change-region mismatch",
        metrics={
            "diff_pixels": int(np.count_nonzero(actual_diff)),
            "declared_change_pixels": int(np.count_nonzero(change)),
            "source_ids": list(source_ids),
            "target_fine_ids": list(expected_target_ids),
        },
    )


def _check_unrequested_preserved(context: GateContext) -> GateCheck:
    source = np.asarray(context.source_mask)
    target = np.asarray(context.candidate.target_mask)
    change = np.asarray(context.candidate.change_region, dtype=bool)
    violations = int(np.count_nonzero((source != target) & ~change))
    passed = violations == 0
    return _result(
        "unrequested_labels_preserved",
        passed,
        "all pixels outside the declared change region are identical"
        if passed
        else f"{violations} pixels changed outside the declared region",
        metrics={"violation_pixels": violations},
    )


def _check_changed_area(context: GateContext) -> GateCheck:
    source_region = np.isin(
        context.source_mask,
        tuple(
            fine_id
            for label in context.plan.source_labels
            for fine_id in context.schema.resolve_fine_ids(label)
        ),
    )
    hard_lower, hard_upper = context.plan.area_budget.hard_pixel_interval(
        context.source_mask, source_region
    )
    changed = int(np.count_nonzero(context.candidate.change_region))
    resolved = context.plan.resolved_area
    resolved_pixels = (
        resolved.resolved_pixels
        if resolved is not None
        else context.plan.area_budget.target_pixels(context.source_mask, source_region)
    )
    tolerance = max(
        1,
        int(
            np.ceil(
                resolved_pixels * context.plan.area_budget.relative_tolerance
            )
        ),
    )
    resolved_lower = max(hard_lower, resolved_pixels - tolerance)
    resolved_upper = min(hard_upper, resolved_pixels + tolerance)
    passed = resolved_lower <= changed <= resolved_upper and changed > 0
    return _result(
        "changed_area",
        passed,
        f"changed pixels {changed} within [{resolved_lower}, {resolved_upper}]"
        if passed
        else f"changed pixels {changed} outside [{resolved_lower}, {resolved_upper}]",
        metrics={
            "changed_pixels": changed,
            "desired_pixels": context.plan.area_budget.target_pixels(
                context.source_mask, source_region
            ),
            "resolved_pixels": resolved_pixels,
            "resolved_tolerance_pixels": tolerance,
            "allowed_min_pixels": resolved_lower,
            "allowed_max_pixels": resolved_upper,
            "hard_min_pixels": hard_lower,
            "hard_max_pixels": hard_upper,
            "fallback_policy": context.plan.area_budget.fallback_policy,
            "fallback_used": bool(resolved.used_fallback if resolved else False),
            "binding_constraint": (
                resolved.binding_constraint if resolved else "uncompiled_exact_target"
            ),
            "basis": context.plan.area_budget.basis,
        },
    )


def _check_interface_contact(context: GateContext) -> GateCheck:
    interface_ids = _candidate_interface_ids(context.candidate)
    interface_masks = [
        context.scene.interface_masks[item]
        for item in interface_ids
        if item in context.scene.interface_masks
    ]
    if not interface_masks or len(interface_masks) != len(interface_ids):
        return _result("interface_contact", False, "selected interface does not exist")
    interface = np.logical_or.reduce(interface_masks)
    change = np.asarray(context.candidate.change_region, dtype=bool)
    labeled, count = ndimage.label(change, structure=np.ones((3, 3), dtype=bool))
    touching_components = 0
    for component_id in range(1, count + 1):
        component = labeled == component_id
        if np.any(ndimage.binary_dilation(component) & interface):
            touching_components += 1
    passed = count > 0 and touching_components == count
    return _result(
        "interface_contact",
        passed,
        "every changed component touches at least one selected pre-edit interface"
        if passed
        else f"{touching_components}/{count} changed components touch the interface",
        metrics={"component_count": count, "touching_component_count": touching_components},
    )


def _check_execution_contract_fidelity(context: GateContext) -> GateCheck:
    """Re-derive whether pixels realize the Planner's executable contract."""

    planned = _candidate_planned_interfaces(context)
    trace_interface_ids = _candidate_interface_ids(context.candidate)
    planned_ids = tuple(item.interface_id for item in context.plan.candidate_interfaces)
    change = np.asarray(context.candidate.change_region, dtype=bool)
    if tuple(trace_interface_ids) != planned_ids or len(planned) != len(planned_ids):
        return _result(
            "execution_contract_fidelity",
            False,
            "candidate did not execute exactly the planned interface set and order",
            metrics={
                "planned_interface_ids": list(planned_ids),
                "trace_interface_ids": list(trace_interface_ids),
            },
        )

    anchor_masks: list[np.ndarray] = []
    influence_masks: list[np.ndarray] = []
    anchor_mask_groups: list[tuple[np.ndarray, ...]] = []
    distances: list[np.ndarray] = []
    for item in planned:
        masks = tuple(
            context.scene.anchor_masks[anchor_id]
            for anchor_id in item.execution_contract.anchor_segment_ids
            if anchor_id in context.scene.anchor_masks
        )
        if len(masks) != len(item.execution_contract.anchor_segment_ids):
            return _result(
                "execution_contract_fidelity",
                False,
                f"planned anchor mask is missing for {item.interface_id}",
            )
        anchor = np.logical_or.reduce(masks)
        interface = context.scene.interface_masks[item.interface_id]
        _, nearest = ndimage.distance_transform_edt(~interface, return_indices=True)
        influence = anchor[nearest[0], nearest[1]]
        anchor_masks.append(anchor)
        anchor_mask_groups.append(masks)
        influence_masks.append(influence)
        distances.append(ndimage.distance_transform_edt(~anchor))

    distance_stack = np.stack(distances, axis=0)
    assignment = np.argmin(distance_stack, axis=0)
    target_pixels = int(np.count_nonzero(change))
    raw_allocations = np.asarray(
        [item.execution_contract.area_allocation_fraction for item in planned]
    ) * target_pixels
    expected_allocations = np.floor(raw_allocations).astype(int)
    remainder = target_pixels - int(expected_allocations.sum())
    allocation_order = sorted(
        range(len(planned)),
        key=lambda index: (
            -(raw_allocations[index] - expected_allocations[index]),
            index,
        ),
    )
    for index in allocation_order[:remainder]:
        expected_allocations[index] += 1

    per_interface: dict[str, object] = {}
    passed = True
    for index, item in enumerate(planned):
        execution = item.execution_contract
        effective_min_coverage = max(
            0.50, execution.min_anchor_coverage_fraction
        )
        effective_max_off_anchor = min(
            0.02, execution.max_off_anchor_contact_fraction
        )
        effective_allocation_tolerance = min(
            0.02, execution.allocation_tolerance_fraction
        )
        assigned = change & (assignment == index)
        realized = int(np.count_nonzero(assigned))
        expected = int(expected_allocations[index])
        allocation_tolerance = max(
            1, int(np.ceil(target_pixels * effective_allocation_tolerance))
        )
        allocation_error = abs(realized - expected)
        anchor = anchor_masks[index]
        touched_anchor = anchor & ndimage.binary_dilation(
            assigned, structure=np.ones((3, 3), dtype=bool)
        )
        coverage = int(np.count_nonzero(touched_anchor)) / max(
            int(np.count_nonzero(anchor)), 1
        )
        interface = context.scene.interface_masks[item.interface_id]
        touched_interface = interface & ndimage.binary_dilation(
            assigned, structure=np.ones((3, 3), dtype=bool)
        )
        off_anchor_contact = int(np.count_nonzero(touched_interface & ~anchor))
        off_anchor_fraction = off_anchor_contact / max(
            int(np.count_nonzero(touched_interface)), 1
        )
        outside_influence = int(
            np.count_nonzero(assigned & ~influence_masks[index])
        )
        remaining_assigned_source = np.isin(
            context.candidate.target_mask,
            tuple(
                fine_id
                for label in context.plan.source_labels
                for fine_id in context.schema.resolve_fine_ids(label)
            ),
        )
        front = assigned & ndimage.binary_dilation(
            remaining_assigned_source, structure=np.ones((3, 3), dtype=bool)
        )
        desired_depth = compile_depth_profile_map(
            anchor_mask_groups[index],
            profile=execution.depth_profile,
            shape=change.shape,
        )
        normalized_front_depth = distances[index][front] / np.maximum(
            desired_depth[front], 1e-3
        )
        depth_violation_fraction = (
            float(np.mean(normalized_front_depth > 1.10))
            if normalized_front_depth.size
            else 0.0
        )
        component_count = int(
            ndimage.label(assigned, structure=np.ones((3, 3), dtype=bool))[1]
        )
        allowed_components = max(
            1,
            execution.depth_profile.lobe_count
            * len(execution.anchor_segment_ids),
        )
        item_passed = (
            allocation_error <= allocation_tolerance
            and coverage >= effective_min_coverage
            and off_anchor_fraction <= effective_max_off_anchor
            and outside_influence == 0
            and depth_violation_fraction <= 0.02
            and 0 < component_count <= allowed_components
        )
        passed = passed and item_passed
        per_interface[item.interface_id] = {
            "anchor_segment_ids": list(execution.anchor_segment_ids),
            "expected_pixels": expected,
            "realized_pixels": realized,
            "allocation_error_pixels": allocation_error,
            "allocation_tolerance_pixels": allocation_tolerance,
            "anchor_coverage_fraction": coverage,
            "requested_minimum_anchor_coverage_fraction": execution.min_anchor_coverage_fraction,
            "effective_minimum_anchor_coverage_fraction": effective_min_coverage,
            "off_anchor_contact_pixels": off_anchor_contact,
            "off_anchor_contact_fraction": off_anchor_fraction,
            "requested_maximum_off_anchor_contact_fraction": execution.max_off_anchor_contact_fraction,
            "effective_maximum_off_anchor_contact_fraction": effective_max_off_anchor,
            "outside_anchor_influence_pixels": outside_influence,
            "depth_profile_violation_fraction": depth_violation_fraction,
            "component_count": component_count,
            "allowed_component_count": allowed_components,
            "passed": item_passed,
        }
    return _result(
        "execution_contract_fidelity",
        passed,
        "candidate pixels realize the selected anchors, allocation, and depth profile"
        if passed
        else "candidate pixels diverge from the executable Planner contract",
        metrics={"interfaces": per_interface},
    )


def _check_prohibited_regions(context: GateContext) -> GateCheck:
    planned = _candidate_planned_interfaces(context)
    if not planned:
        return _result("prohibited_region", False, "candidate interface is absent from plan")
    missing: list[str] = []
    overlap = 0
    declared_ids = {
        region_id for item in planned for region_id in item.prohibited_region_ids
    }
    for region_id in sorted(declared_ids):
        region = context.scene.prohibited_region_masks.get(region_id)
        if region is None:
            missing.append(region_id)
            continue
        overlap += int(np.count_nonzero(region & context.candidate.change_region))
    # Scene-derived hard prohibited masks are never optional Planner hints.
    undeclared_scene_overlap = sum(
        int(np.count_nonzero(region & context.candidate.change_region))
        for region_id, region in context.scene.prohibited_region_masks.items()
        if region_id not in declared_ids
    )
    overlap += undeclared_scene_overlap
    passed = not missing and overlap == 0
    return _result(
        "prohibited_region",
        passed,
        "candidate avoids all declared prohibited regions"
        if passed
        else f"missing regions={missing}, overlap_pixels={overlap}",
        metrics={
            "missing_region_ids": missing,
            "overlap_pixels": overlap,
            "scene_derived_overlap_pixels": undeclared_scene_overlap,
        },
    )


def _check_component_topology(context: GateContext) -> GateCheck:
    change = np.asarray(context.candidate.change_region, dtype=bool)
    labeled, count = ndimage.label(change, structure=np.ones((3, 3), dtype=bool))
    sizes = [int(np.count_nonzero(labeled == idx)) for idx in range(1, count + 1)]
    params = context.plan.tool_program.parameter_ranges
    # Multiple explicitly planned fronts may legitimately produce multiple
    # changed components. Keep a small hard ceiling, but do not collapse a
    # three/four-interface plan back to two and then reject faithful execution.
    max_components = min(4, max(1, int(params.get("max_changed_components", 2))))
    min_area = max(16, int(params.get("min_component_area_px", 16)))
    tiny = [size for size in sizes if size < min_area]
    passed = 0 < count <= max_components and not tiny
    return _result(
        "component_topology",
        passed,
        "changed-region component topology is within the plan contract"
        if passed
        else f"component_count={count}, max={max_components}, tiny_components={tiny}",
        metrics={
            "component_count": count,
            "component_sizes": sizes,
            "max_changed_components": max_components,
            "min_component_area_px": min_area,
        },
    )


def _check_edited_label_topology(context: GateContext) -> GateCheck:
    source = np.asarray(context.source_mask)
    target = np.asarray(context.candidate.target_mask)
    source_ids = tuple(
        sorted(
            {
                fine_id
                for label in context.plan.source_labels
                for fine_id in context.schema.resolve_fine_ids(label)
            }
        )
    )
    target_ids = tuple(context.schema.resolve_fine_ids(context.plan.target_label))
    source_before = np.isin(source, source_ids)
    source_after = np.isin(target, source_ids)
    target_before = np.isin(source, target_ids)
    target_after = np.isin(target, target_ids)
    source_components_before = _component_count(source_before)
    source_components_after = _component_count(source_after)
    target_labeled_before, target_components_before = ndimage.label(
        target_before, structure=np.ones((3, 3), dtype=bool)
    )
    target_labeled_after, target_components_after = ndimage.label(
        target_after, structure=np.ones((3, 3), dtype=bool)
    )
    source_holes_before = _hole_count(source_before)
    source_holes_after = _hole_count(source_after)
    target_holes_before = _hole_count(target_before)
    target_holes_after = _hole_count(target_after)
    source_split = source_components_after > source_components_before
    target_split_or_island = target_components_after > target_components_before
    target_merge = target_components_after < target_components_before
    selected_target_component_ids = {
        item.target_component_id for item in context.plan.candidate_interfaces
    }
    selected_target_before_ids = {
        int(value)
        for component_id in selected_target_component_ids
        for value in np.unique(
            target_labeled_before[
                context.scene.component_masks.get(
                    component_id, np.zeros_like(target_before, dtype=bool)
                )
            ]
        ).tolist()
        if int(value) > 0
    }
    merged_before_component_groups: list[list[int]] = []
    unallowed_merge_groups: list[list[int]] = []
    for component_index in range(1, target_components_after + 1):
        before_ids = sorted(
            int(value)
            for value in np.unique(
                target_labeled_before[target_labeled_after == component_index]
            ).tolist()
            if int(value) > 0
        )
        if len(before_ids) <= 1:
            continue
        merged_before_component_groups.append(before_ids)
        if not set(before_ids).issubset(selected_target_before_ids):
            unallowed_merge_groups.append(before_ids)
    # Selecting multiple pre-edit target components is an auditable implicit
    # request to let those named components coalesce. Any unselected component
    # entering the merger remains a hard failure.
    unallowed_target_merge = bool(unallowed_merge_groups)
    source_hole_changed = source_holes_after != source_holes_before
    target_hole_changed = target_holes_after != target_holes_before
    disallowed_source_hole_change = source_hole_changed and not (
        target_merge and not unallowed_target_merge
    )
    passed = not any(
        (
            source_split,
            target_split_or_island,
            unallowed_target_merge,
            disallowed_source_hole_change,
            target_hole_changed,
        )
    )
    return _result(
        "edited_label_topology",
        passed,
        "source/target component and hole topology is preserved"
        if passed
        else (
            "candidate causes a source split, target island/split, unselected target "
            "component merge, or changes protected source/target holes"
        ),
        metrics={
            "source_components_before": source_components_before,
            "source_components_after": source_components_after,
            "target_components_before": target_components_before,
            "target_components_after": target_components_after,
            "source_holes_before": source_holes_before,
            "source_holes_after": source_holes_after,
            "target_holes_before": target_holes_before,
            "target_holes_after": target_holes_after,
            "source_split": source_split,
            "target_split_or_island": target_split_or_island,
            "target_merge": target_merge,
            "selected_target_component_ids": sorted(selected_target_component_ids),
            "selected_target_before_component_indices": sorted(
                selected_target_before_ids
            ),
            "merged_before_component_groups": merged_before_component_groups,
            "unallowed_target_merge_groups": unallowed_merge_groups,
            "unallowed_target_merge": unallowed_target_merge,
            "source_hole_changed": source_hole_changed,
            "disallowed_source_hole_change": disallowed_source_hole_change,
            "target_hole_changed": target_hole_changed,
            # Backward-compatible keys retained for older audit consumers.
            "new_source_hole": source_holes_after > source_holes_before,
            "new_target_hole": target_holes_after > target_holes_before,
        },
    )


def _check_source_component_retention(context: GateContext) -> GateCheck:
    change = np.asarray(context.candidate.change_region, dtype=bool)
    component_ids = _trace_ids(
        context.candidate, plural="source_component_ids", singular="source_component_id"
    )
    max_fraction = min(
        0.55,
        float(
            context.plan.tool_program.parameter_ranges.get(
                "max_source_component_changed_fraction", 0.55
            )
        ),
    )
    min_remaining = max(
        64,
        int(
            context.plan.tool_program.parameter_ranges.get(
                "min_source_component_remaining_px", 64
            )
        ),
    )
    metrics: dict[str, dict[str, float | int]] = {}
    missing: list[str] = []
    passed = bool(component_ids)
    for component_id in component_ids:
        component = context.scene.component_masks.get(component_id)
        if component is None:
            missing.append(component_id)
            passed = False
            continue
        area = int(np.count_nonzero(component))
        changed = int(np.count_nonzero(component & change))
        remaining = area - changed
        fraction = changed / max(area, 1)
        metrics[component_id] = {
            "area_px": area,
            "changed_px": changed,
            "remaining_px": remaining,
            "changed_fraction": fraction,
        }
        if fraction > max_fraction or (changed > 0 and remaining < min_remaining):
            passed = False
    return _result(
        "source_component_retention",
        passed and not missing,
        "every edited source component retains a plausible residual structure"
        if passed and not missing
        else (
            "an edited component is missing or is consumed beyond the allowed fraction; "
            f"missing={missing}"
        ),
        metrics={
            "components": metrics,
            "max_changed_fraction": max_fraction,
            "min_remaining_px": min_remaining,
            "missing_component_ids": missing,
        },
    )
def _check_depth_span_ratio(context: GateContext) -> GateCheck:
    planned = _candidate_planned_interfaces(context)
    interface_masks = [
        context.scene.interface_masks[item.interface_id]
        for item in planned
        if item.interface_id in context.scene.interface_masks
    ]
    if not interface_masks or len(interface_masks) != len(planned):
        return _result("depth_span_ratio", False, "selected interface mask is empty")
    interface = np.logical_or.reduce(interface_masks)
    change = np.asarray(context.candidate.change_region, dtype=bool)
    distances = ndimage.distance_transform_edt(~interface)[change]
    if distances.size == 0:
        return _result("depth_span_ratio", False, "change region is empty")
    interface_contact = interface & ndimage.binary_dilation(change)
    span = max(1, int(np.count_nonzero(interface_contact)))
    max_depth = float(np.max(distances))
    p95_depth = float(np.percentile(distances, 95))
    ratio = max_depth / span
    band_max = max(float(item.allowed_edit_band_px[1]) for item in planned)
    max_ratio = min(
        1.25,
        float(context.plan.tool_program.parameter_ranges.get("max_depth_span_ratio", 1.25)),
    )
    passed = max_depth <= band_max + 1e-6 and ratio <= max_ratio
    return _result(
        "depth_span_ratio",
        passed,
        "candidate depth is supported by a sufficiently broad interface"
        if passed
        else (
            f"max_depth={max_depth:.2f}, band_max={band_max:.2f}, "
            f"depth/span={ratio:.3f}, allowed={max_ratio:.3f}"
        ),
        metrics={
            "max_depth_px": max_depth,
            "p95_depth_px": p95_depth,
            "interface_contact_pixels": span,
            "max_depth_span_ratio": ratio,
            "allowed_depth_span_ratio": max_ratio,
            "allowed_band_max_px": band_max,
        },
    )


def _check_boundary_naturalness(context: GateContext) -> GateCheck:
    change = np.asarray(context.candidate.change_region, dtype=bool)
    area = int(np.count_nonzero(change))
    if area == 0:
        return _result("boundary_naturalness", False, "change region is empty")
    rows, cols = np.where(change)
    bbox_area = int((rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1))
    fill = area / max(bbox_area, 1)
    boundary = change & ~ndimage.binary_erosion(change, structure=np.ones((3, 3)))
    perimeter = int(np.count_nonzero(boundary))
    compactness = (perimeter * perimeter) / max(4.0 * np.pi * area, 1.0)
    max_fill = min(
        0.985,
        float(
            context.plan.tool_program.parameter_ranges.get(
                "max_bbox_fill_fraction", 0.985
            )
        ),
    )
    max_compactness = min(
        40.0,
        float(
            context.plan.tool_program.parameter_ranges.get(
                "max_boundary_compactness", 40.0
            )
        ),
    )
    rectangle_like = area >= 64 and fill > max_fill
    passed = not rectangle_like and compactness <= max_compactness
    return _result(
        "boundary_naturalness",
        passed,
        "candidate avoids rectangle-like fill and extreme boundary roughness"
        if passed
        else f"bbox_fill={fill:.4f}, compactness={compactness:.3f}",
        metrics={
            "bbox_fill_fraction": fill,
            "boundary_compactness": compactness,
            "rectangle_like": rectangle_like,
        },
    )


def _check_parallel_boundary_artifact(context: GateContext) -> GateCheck:
    """Reject a long equal-depth offset ribbon, including curved rings.

    A curved gland/tumor boundary can still acquire an obviously synthetic
    constant-width annulus. Linearity is retained as an audit metric, but is
    not required for the veto. Pathology/H&E agreement remains a visual critic
    responsibility.
    """

    change = np.asarray(context.candidate.change_region, dtype=bool)
    planned = _candidate_planned_interfaces(context)
    if not np.any(change) or not planned:
        return _result("parallel_boundary_artifact", False, "change or interface is empty")
    source_ids = tuple(
        fine_id
        for label in context.plan.source_labels
        for fine_id in context.schema.resolve_fine_ids(label)
    )
    remaining_source = np.isin(context.candidate.target_mask, source_ids)
    minimum_depth_cv = max(
        0.25,
        float(
            context.plan.tool_program.parameter_ranges.get(
                "min_parallel_front_depth_cv", 0.15
            )
        ),
    )
    min_linearity = min(
        20.0,
        float(
            context.plan.tool_program.parameter_ranges.get(
                "parallel_front_linearity_ratio", 20.0
            )
        ),
    )
    min_depth = min(
        5.0,
        float(
            context.plan.tool_program.parameter_ranges.get(
                "parallel_front_min_depth_px", 5.0
            )
        ),
    )
    min_front = min(
        64,
        int(
            context.plan.tool_program.parameter_ranges.get(
                "parallel_front_min_pixels", 64
            )
        ),
    )
    anchor_unions = []
    for item in planned:
        masks = [
            context.scene.anchor_masks[anchor_id]
            for anchor_id in item.execution_contract.anchor_segment_ids
            if anchor_id in context.scene.anchor_masks
        ]
        if len(masks) != len(item.execution_contract.anchor_segment_ids):
            return _result(
                "parallel_boundary_artifact",
                False,
                f"selected anchor is missing for {item.interface_id}",
            )
        anchor_unions.append(np.logical_or.reduce(masks))
    assignment = np.argmin(
        np.stack(
            [ndimage.distance_transform_edt(~anchor) for anchor in anchor_unions],
            axis=0,
        ),
        axis=0,
    )
    per_interface: dict[str, dict[str, float | int | bool]] = {}
    artifact = False
    for item_index, item in enumerate(planned):
        interface = context.scene.interface_masks.get(item.interface_id)
        if interface is None:
            return _result(
                "parallel_boundary_artifact", False, "selected interface is missing"
            )
        assigned = change & (assignment == item_index)
        front = assigned & ndimage.binary_dilation(
            remaining_source, structure=np.ones((3, 3))
        )
        distances = ndimage.distance_transform_edt(~interface)[front]
        if distances.size < 24:
            per_interface[item.interface_id] = {
                "front_pixels": int(distances.size),
                "artifact": False,
            }
            continue
        mean_depth = float(np.mean(distances))
        depth_cv = float(np.std(distances) / max(mean_depth, 1e-6))
        rows, cols = np.where(front)
        coords = np.column_stack((rows, cols)).astype(float)
        covariance = np.cov(coords, rowvar=False)
        eigenvalues = np.sort(np.linalg.eigvalsh(covariance))
        linearity_ratio = float(eigenvalues[-1] / max(eigenvalues[0], 1e-6))
        item_artifact = (
            distances.size >= min_front
            and mean_depth >= min_depth
            and depth_cv < minimum_depth_cv
        )
        artifact = artifact or item_artifact
        per_interface[item.interface_id] = {
            "front_pixels": int(distances.size),
            "mean_front_depth_px": mean_depth,
            "front_depth_cv": depth_cv,
            "front_linearity_ratio": linearity_ratio,
            "artifact": item_artifact,
        }
    return _result(
        "parallel_boundary_artifact",
        not artifact,
        "new boundary has sufficiently non-uniform organic depth variation"
        if not artifact
        else (
            "at least one planned interface forms a near-parallel ribbon/annulus"
        ),
        metrics={
            "interfaces": per_interface,
            "minimum_required_front_depth_cv": minimum_depth_cv,
            "linearity_audit_threshold": min_linearity,
            "artifact": artifact,
        },
    )
def _check_provenance(context: GateContext) -> GateCheck:
    required = ("source_image_sha256", "source_mask_sha256")
    missing = [key for key in required if not context.case.provenance.get(key)]
    trace_required = ("seed", "target_fine_id", "tool_adapter_version")
    trace_missing = [key for key in trace_required if key not in context.candidate.tool_trace]
    passed = not missing and not trace_missing
    return _result(
        "provenance_complete",
        passed,
        "source and tool provenance are complete"
        if passed
        else f"missing case provenance={missing}, tool trace={trace_missing}",
        metrics={"missing_case_fields": missing, "missing_tool_fields": trace_missing},
    )


def _check_tumor_stroma_interface(context: GateContext) -> GateCheck:
    graph_by_id = {item.interface_id: item for item in context.scene.graph.interfaces}
    interface_ids = _candidate_interface_ids(context.candidate)
    interfaces = [graph_by_id[item] for item in interface_ids if item in graph_by_id]
    if not interfaces or len(interfaces) != len(interface_ids):
        return _result("tumor_stroma_interface", False, "selected interface is unknown")
    passed = all(
        interface.source_label in context.plan.source_labels
        and interface.target_label == context.plan.target_label
        for interface in interfaces
    )
    return _result(
        "tumor_stroma_interface",
        passed,
        "selected directed interface matches the plan source and target"
        if passed
        else (
            f"one or more interfaces disagree with plan "
            f"{list(context.plan.source_labels)}->{context.plan.target_label}"
        ),
        metrics={
            "interfaces": [
                {
                    "interface_id": interface.interface_id,
                    "source_label": interface.source_label,
                    "target_label": interface.target_label,
                }
                for interface in interfaces
            ],
        },
    )


def _candidate_interface_ids(candidate: CandidateMask) -> tuple[str, ...]:
    return _trace_ids(candidate, plural="interface_ids", singular="interface_id")


def _trace_ids(
    candidate: CandidateMask, *, plural: str, singular: str
) -> tuple[str, ...]:
    raw = candidate.tool_trace.get(plural)
    if isinstance(raw, list) and raw and all(isinstance(item, str) for item in raw):
        return tuple(dict.fromkeys(raw))
    fallback = (
        candidate.interface_id
        if singular == "interface_id"
        else candidate.tool_trace.get(singular)
    )
    return (fallback,) if isinstance(fallback, str) and fallback else ()


def _candidate_planned_interfaces(context: GateContext):
    interface_ids = set(_candidate_interface_ids(context.candidate))
    return tuple(
        item for item in context.plan.candidate_interfaces if item.interface_id in interface_ids
    )


def _result(
    check_id: str,
    passed: bool,
    detail: str,
    *,
    metrics: dict[str, object] | None = None,
) -> GateCheck:
    return GateCheck(
        check_id=check_id,
        passed=bool(passed),
        severity="hard",
        detail=detail,
        metrics=dict(metrics or {}),
    )


def _component_count(mask: np.ndarray) -> int:
    return int(ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))[1])


def _hole_count(mask: np.ndarray) -> int:
    holes = ndimage.binary_fill_holes(mask) & ~mask
    return _component_count(holes)

"""Independent tissue Planner adapter that can bind multiple legal components."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from phase3_mask_edit_refine.agents import (
    EDIT_PLAN_JSON_SCHEMA,
    EDIT_PLAN_SCHEMA_VERSION,
    OpenAIResponsesJSONClient,
    validate_edit_plan,
)
from phase3_mask_edit_refine.models import (
    CaseContext,
    DepthProfile,
    EditPlan,
    InterfaceExecutionContract,
    PlannedInterface,
    RefineContractError,
    ToolProgram,
)
from phase3_mask_edit_refine.scene import SceneAnalysis
from phase3_mask_edit_refine.skills import ActiveKnowledgeBundle

from .feasibility import JointNucleiPreflight
from .planner_inputs import validate_mask_planner_image_paths
from .skills.repository import JointSkillBundle

JOINT_TISSUE_DECISION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["abstain", "abstain_reason", "plan"],
    "properties": {
        "abstain": {"type": "boolean"},
        "abstain_reason": {"type": ["string", "null"]},
        "plan": {"anyOf": [EDIT_PLAN_JSON_SCHEMA, {"type": "null"}]},
    },
}


def _mask_planner_case_metadata(case: CaseContext) -> dict[str, Any]:
    """Remove the raw histology locator from tissue-planning metadata."""

    metadata = dict(case.to_metadata())
    metadata.pop("source_image_uri", None)
    return metadata


def _normalize_integer_allocations(
    allocations: Sequence[int],
) -> tuple[float, ...]:
    """Convert realized integer pixel allocations into unit-sum weights."""

    total = sum(int(item) for item in allocations)
    if total <= 0:
        raise RefineContractError(
            "interface allocation produced no executable tissue pixels"
        )
    return tuple(int(item) / total for item in allocations)


def _effective_tissue_topology(
    joint_bundle: JointSkillBundle,
    *,
    primitive_id: str,
    retry_index: int,
    feedback_stage: str | None,
) -> dict[str, Any]:
    """Resolve primitive defaults plus a reviewed mechanism fallback.

    A mechanism fallback is activated only after the ordinary interface-front
    plan has failed compilation. This prevents a high-area task from silently
    changing every organ into component deletion while still allowing a
    complete, pathology-recognized structural unit to resolve when shallow
    fronts cannot reach the hard area floor.
    """

    primitive = joint_bundle.primitive
    result = {
        "geometry_mode": primitive.tissue_geometry_mode,
        "allow_source_component_resolution": (
            primitive.allow_source_component_resolution
        ),
        "allow_target_hole_resolution": primitive.allow_target_hole_resolution,
        "maximum_source_component_changed_fraction": (
            primitive.maximum_source_component_changed_fraction
        ),
        "minimum_source_component_remaining_px": (
            primitive.minimum_source_component_remaining_px
        ),
        "allow_source_component_split": primitive.allow_source_component_split,
        "minimum_residual_components": primitive.minimum_residual_components,
        "maximum_residual_components": primitive.maximum_residual_components,
        "minimum_residual_component_area_px": (
            primitive.minimum_residual_component_area_px
        ),
        "minimum_residual_spacing_px": primitive.minimum_residual_spacing_px,
        "residual_area_floor_fraction": primitive.residual_area_floor_fraction,
        "fallback_activated": False,
    }
    fallback = joint_bundle.mechanism.tissue_program.topology_fallback_for(
        primitive_id
    )
    if (
        fallback is not None
        and retry_index > 0
        and feedback_stage == "planning_or_compilation"
    ):
        result.update(
            {
                "geometry_mode": fallback.geometry_mode,
                "allow_source_component_resolution": (
                    fallback.allow_source_component_resolution
                ),
                "allow_target_hole_resolution": (
                    fallback.allow_target_hole_resolution
                ),
                "maximum_source_component_changed_fraction": (
                    fallback.maximum_source_component_changed_fraction
                ),
                "minimum_source_component_remaining_px": (
                    fallback.minimum_source_component_remaining_px
                ),
                "fallback_activated": True,
            }
        )
    return result


@dataclass(frozen=True)
class OpenAIJointAwareTissuePlanner:
    """Mask-graph tissue Planner that selects certified interfaces and anchors."""

    client: OpenAIResponsesJSONClient
    escalation_client: OpenAIResponsesJSONClient | None = None
    max_contract_attempts: int = 2
    name: str = "openai_certified_mask_tissue_planner"

    def create_joint_tissue_plan(
        self,
        *,
        case: CaseContext,
        scene: SceneAnalysis,
        bundle: ActiveKnowledgeBundle,
        joint_bundle: JointSkillBundle,
        image_paths: Sequence[str | Path],
        nuclei_preflight: JointNucleiPreflight | None = None,
        execution_feedback: Mapping[str, Any] | None = None,
    ) -> tuple[EditPlan, dict[str, Any]]:
        image_paths = validate_mask_planner_image_paths(image_paths)
        if nuclei_preflight is None:
            raise RefineContractError(
                "joint-aware tissue Planner requires nuclei preflight"
            )
        payload = {
            "case": _mask_planner_case_metadata(case),
            "tissue_scene": scene.graph.to_metadata(),
            "tissue_skill_bundle": bundle.to_metadata(),
            "joint_skill_bundle": joint_bundle.to_metadata(),
            "joint_mechanism_contract": {
                "recognition": joint_bundle.mechanism.recognition.__dict__,
                "representability": joint_bundle.mechanism.representability.__dict__,
                "tissue_program": joint_bundle.mechanism.tissue_program.__dict__,
                "coupling": joint_bundle.mechanism.coupling.__dict__,
                "planner_policy": joint_bundle.mechanism.planner_policy.__dict__,
            },
            "nuclei_preflight": nuclei_preflight.to_metadata(),
            "previous_execution_feedback": dict(execution_feedback or {}),
            "requirements": {
                "select_only_nuclei_feasible_interfaces": True,
                "select_real_anchor_segment_ids": True,
                "respect_immutable_area_budget": True,
                "prefer_broad_shallow_interfaces": True,
                "do_not_output_pixels_or_polygons": True,
                "abstain_when_certificate_capacity_is_insufficient": True,
                "use_skill_selection_preferences": True,
                "source_H&E_is_prohibited": True,
                "do_not_infer_unannotated_histology": True,
            },
        }
        errors: list[str] = []
        clients = [self.client] * self.max_contract_attempts
        if self.escalation_client is not None:
            clients.append(self.escalation_client)
        for attempt, client in enumerate(clients, start=1):
            raw, usage = client.call(
                system_prompt=(
                    "You are the certified mask-graph tissue planning stage. Annotation "
                    "semantics, tissue topology, complete-nucleus capacity, candidate "
                    "certificates, and the selected skill mechanism are mandatory. Apply "
                    "the skill's hard constraints and selection preferences, then select "
                    "interface and anchor IDs only. Raw H&E is prohibited and cannot be "
                    "used to invent an unannotated structure. Return an explicit abstention "
                    "when the mask-owned requirements cannot be jointly satisfied."
                ),
                user_prompt=json.dumps(
                    {**payload, "previous_contract_errors": errors},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                image_paths=image_paths,
                schema_name="joint_aware_tissue_plan",
                json_schema=JOINT_TISSUE_DECISION_SCHEMA,
            )
            if raw.get("abstain") is True:
                raise RefineContractError(
                    "joint-aware tissue Planner abstained: "
                    + str(raw.get("abstain_reason") or "insufficient evidence")
                )
            try:
                raw_plan = raw.get("plan")
                if not isinstance(raw_plan, dict):
                    raise RefineContractError("joint tissue plan is missing")
                plan = EditPlan.from_mapping(raw_plan)
                if plan.normalized_intent != case.instruction:
                    raise RefineContractError(
                        "joint tissue Planner modified the parser-owned intent"
                    )
                validate_edit_plan(
                    plan,
                    case=case,
                    scene=scene,
                    bundle=bundle,
                )
                self._validate_joint_binding(
                    plan=plan,
                    joint_bundle=joint_bundle,
                    nuclei_preflight=nuclei_preflight,
                    scene=scene,
                )
            except (TypeError, ValueError) as exc:
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                continue
            return plan, {
                "provider": self.name,
                "contract_attempt": attempt,
                "escalated": client is self.escalation_client,
                **usage,
            }
        raise RefineContractError(
            "joint-aware tissue Planner exhausted contract attempts: "
            + "; ".join(errors)
        )

    @staticmethod
    def _validate_joint_binding(
        *,
        plan: EditPlan,
        joint_bundle: JointSkillBundle,
        nuclei_preflight: JointNucleiPreflight,
        scene: SceneAnalysis,
    ) -> None:
        feasible = set(nuclei_preflight.feasible_interface_ids)
        selected = {item.interface_id for item in plan.candidate_interfaces}
        if not selected or not selected.issubset(feasible):
            raise RefineContractError(
                "tissue Planner selected an interface without certified nuclei capacity"
            )
        label_contract = (
            joint_bundle.mechanism.tissue_program.primitive_label_contracts.get(
                plan.primitive_id
            )
        )
        if label_contract is None:
            raise RefineContractError("joint mechanism has no primitive contract")
        if not set(plan.source_labels).issubset(label_contract["source_labels"]):
            raise RefineContractError("tissue source labels violate joint mechanism")
        if plan.target_label not in label_contract["target_labels"]:
            raise RefineContractError("tissue target label violates joint mechanism")
        anchor_to_interface = {
            item.anchor_segment_id: item.interface_id
            for item in scene.graph.anchor_segments
        }
        for item in plan.candidate_interfaces:
            anchors = item.execution_contract.anchor_segment_ids
            if not anchors or any(
                anchor_to_interface.get(anchor_id) != item.interface_id
                for anchor_id in anchors
            ):
                raise RefineContractError(
                    "tissue plan contains an unknown or detached anchor"
                )
        if plan.planner_confidence < 0.70:
            raise RefineContractError(
                "joint-aware tissue plan confidence is below 0.70"
            )


@dataclass(frozen=True)
class MultiInterfaceResearchTissuePlanner:
    """Use all needed legal source components; no H&E authority is claimed."""

    name: str = "multi_interface_research_tissue_planner_v2"

    def create_joint_tissue_plan(
        self,
        *,
        case: CaseContext,
        scene: SceneAnalysis,
        bundle: ActiveKnowledgeBundle,
        joint_bundle: JointSkillBundle,
        image_paths: Sequence[str | Path],
        nuclei_preflight: JointNucleiPreflight | None = None,
        execution_feedback: Mapping[str, Any] | None = None,
    ) -> tuple[EditPlan, dict[str, Any]]:
        del image_paths
        mechanism_contract = joint_bundle.mechanism.tissue_program.primitive_label_contracts.get(case.primitive_id)
        if mechanism_contract is None:
            raise RefineContractError("joint mechanism has no primitive label contract")
        allowed_sources = set(bundle.edit_contract.source_label_options).intersection(mechanism_contract["source_labels"])
        if bundle.edit_contract.target_label not in mechanism_contract["target_labels"]:
            raise RefineContractError("annotation-resolved target is illegal for the joint mechanism")
        legal = [
            item for item in scene.graph.interfaces
            if item.source_label in allowed_sources and item.target_label == bundle.edit_contract.target_label
        ]
        if nuclei_preflight is not None:
            feasible_ids = set(nuclei_preflight.feasible_interface_ids)
            legal = [item for item in legal if item.interface_id in feasible_ids]
        if not legal:
            raise RefineContractError(
                "no directed interface satisfies both the tissue and preflight nuclei contracts"
            )
        feedback = dict(execution_feedback or {})
        retry_index = max(0, int(feedback.get("retry_index", 0)))
        failed_interface_ids = set(feedback.get("failed_interface_ids", ()))
        topology = _effective_tissue_topology(
            joint_bundle,
            primitive_id=case.primitive_id,
            retry_index=retry_index,
            feedback_stage=feedback.get("stage"),
        )
        component_turnover = (
            topology["geometry_mode"] == "component_boundary_turnover"
        )
        residual_fragmentation = (
            topology["geometry_mode"] == "residual_fragmentation"
        )

        # One connected source component can border several independent target
        # components.  Collapsing those contacts to its single longest edge was
        # the reason a large editable stroma/tumor component could yield a tiny
        # 0.8--1.5% edit.  Keep every directed component-pair interface and let
        # the pixel owner assignment downstream make their influence zones
        # disjoint.
        capacity_by_id: dict[str, int] = {}
        for item in legal:
            preflight_item = (
                nuclei_preflight.interface(item.interface_id)
                if nuclei_preflight is not None
                else None
            )
            capacity_by_id[item.interface_id] = int(
                preflight_item.editable_tissue_capacity_pixels
                if preflight_item is not None
                else 0
            )
        if component_turnover:
            # Rasterization can split one biological component boundary into
            # several directed segments. Independent quotas on those segments
            # create wedge seams and concentric cut lines, so retain one
            # representative per source/target component pair. A retry can
            # choose an alternative segment when the prior one failed.
            by_component_pair = {}
            for item in legal:
                by_component_pair.setdefault(
                    (item.source_component_id, item.target_component_id), []
                ).append(item)
            legal = [
                min(
                    items,
                    key=lambda item: (
                        item.interface_id in failed_interface_ids,
                        -item.contact_pixels,
                        -capacity_by_id[item.interface_id],
                        item.interface_id,
                    ),
                )
                for _, items in sorted(by_component_pair.items())
            ]
        labels = sorted({item.source_label for item in legal})
        source_label = max(
            labels,
            key=lambda label: (
                sum(
                    capacity_by_id[item.interface_id]
                    for item in legal
                    if item.source_label == label
                ),
                sum(
                    item.contact_pixels
                    for item in legal
                    if item.source_label == label
                ),
                label,
            ),
        )
        ranked = sorted(
            (item for item in legal if item.source_label == source_label),
            key=lambda item: (
                item.interface_id in failed_interface_ids,
                -capacity_by_id[item.interface_id],
                -item.contact_pixels,
                item.interface_id,
            ),
        )
        if residual_fragmentation and ranked:
            # Fragmentation is defined within one pre-existing invasive-tumor
            # component. Selecting unrelated components could satisfy an
            # island-count gate without actually fragmenting anything.
            capacity_by_source: dict[str, int] = {}
            for item in ranked:
                capacity_by_source[item.source_component_id] = (
                    capacity_by_source.get(item.source_component_id, 0)
                    + capacity_by_id[item.interface_id]
                )
            selected_source_component_id = max(
                capacity_by_source,
                key=lambda component_id: (
                    capacity_by_source[component_id],
                    sum(
                        item.contact_pixels
                        for item in ranked
                        if item.source_component_id == component_id
                    ),
                    component_id,
                ),
            )
            ranked = [
                item
                for item in ranked
                if item.source_component_id == selected_source_component_id
            ]
        source_region = np.zeros((scene.graph.height, scene.graph.width), dtype=bool)
        for item in ranked:
            source_region |= scene.component_masks[item.source_component_id]
        target_pixels = case.area_budget.target_pixels(source_region, source_region)
        hard_min_pixels, _hard_max_pixels = case.area_budget.hard_pixel_interval(
            source_region, source_region
        )
        selected = []
        capacities = []
        cumulative = 0
        # A planning/compilation failure means the raw preflight capacities
        # did not survive the shared topology compiler.  Adding only two more
        # interfaces per retry can still stop at another optimistic capacity
        # threshold and report a sub-maximal fallback.  On that specific
        # feedback stage, expose every legal, cell-feasible interface to the
        # compiler; its disjoint pixel ownership, source-retention ceiling and
        # whole-mask topology audit then determine the actual maximum.  Gate or
        # cell-feasibility retries retain the gradual diversification policy.
        extra_after_capacity = (
            len(ranked)
            if component_turnover
            else min(8, retry_index * 4)
        )
        capacity_reached_at: int | None = None
        for item in ranked[:32]:
            preflight_item = (
                nuclei_preflight.interface(item.interface_id)
                if nuclei_preflight is not None
                else None
            )
            if preflight_item is not None:
                capacity = int(preflight_item.editable_tissue_capacity_pixels)
            else:
                interface_mask = scene.interface_masks[item.interface_id]
                source_component = scene.component_masks[item.source_component_id]
                distance = ndimage.distance_transform_edt(~interface_mask)
                depth_cap = max(1, min(128, int(item.contact_pixels)))
                capacity = int(
                    np.count_nonzero(source_component & (distance <= depth_cap))
                )
            if capacity <= 0:
                continue
            selected.append(item)
            capacities.append(capacity)
            cumulative += capacity
            # One long, cell-feasible interface is preferable to forcing a
            # small request through several disconnected fronts.  Additional
            # interfaces are selected only when their capacity is needed.
            if cumulative >= target_pixels and capacity_reached_at is None:
                capacity_reached_at = len(selected)
            if (
                capacity_reached_at is not None
                and len(selected) >= capacity_reached_at + extra_after_capacity
            ):
                break
        if not selected:
            raise RefineContractError("preflight left no executable tissue capacity")
        total_capacity = max(1, sum(capacities))
        if topology["allow_source_component_resolution"]:
            # Prefer completing whole biological compartments instead of
            # shaving the same proportion from every component. Proportional
            # erosion leaves multiple equal-depth residual ribbons; greedy
            # completion leaves at most one partially resolved compartment.
            remaining = target_pixels
            component_allocations = []
            retained_selected = []
            retained_capacities = []
            for item, capacity in zip(selected, capacities):
                requested = min(capacity, max(0, remaining))
                if requested <= 0:
                    continue
                retained_selected.append(item)
                retained_capacities.append(capacity)
                component_allocations.append(requested)
                remaining -= requested
            realized_component_capacity = sum(component_allocations)
            if realized_component_capacity < hard_min_pixels:
                raise RefineContractError(
                    "component resolution capacity cannot reach the tissue hard "
                    f"minimum: capacity={realized_component_capacity}, "
                    f"minimum={hard_min_pixels}"
                )
            # A ranged task explicitly permits the authoritative topology
            # compiler to resolve the largest safe component prefix below the
            # desired target.  The planner therefore retains a complete
            # component witness that clears the hard floor instead of rejecting
            # it merely because it cannot hit the preferred 19% exactly.
            selected = retained_selected
            capacities = retained_capacities
        else:
            component_allocations = [
                min(
                    capacity,
                    max(1, round(target_pixels * capacity / total_capacity)),
                )
                for capacity in capacities
            ]
        # ``component_allocations`` are integer pixel requests.  In the
        # proportional branch, independent rounding can make their sum differ
        # from ``target_pixels`` by one or more pixels.  The execution contract
        # stores relative weights, so normalize by the realized integer sum
        # instead of the nominal target.  Otherwise a valid multi-interface
        # plan can fail closed only because its weights add up to e.g.
        # 1.00002008.
        allocation_fractions = _normalize_integer_allocations(
            component_allocations
        )
        rule_ids = tuple(rule.rule_id for rule in bundle.active_rules) + tuple(item.constraint_id for item in bundle.active_mask_constraints)
        front_contract = joint_bundle.mechanism.tissue_program.front
        planned = []
        for interface, capacity, requested_allocation, fraction in zip(
            selected,
            capacities,
            component_allocations,
            allocation_fractions,
        ):
            preflight_item = (
                nuclei_preflight.interface(interface.interface_id)
                if nuclei_preflight is not None
                else None
            )
            initial_depth_cap = float(
                preflight_item.gate_bounded_depth_px
                if preflight_item is not None
                else max(1, min(128, int(interface.contact_pixels)))
            )
            if component_turnover or residual_fragmentation:
                # A closed compartment is one biological object even when the
                # raster graph splits its boundary into several directed
                # segments. Residual fragmentation likewise needs the complete
                # outside boundary so the deterministic corridor compiler can
                # enter and leave an editable neck. Candidate-local cell
                # feasibility remains authoritative in both cases.
                anchor_ids = tuple(interface.anchor_segment_ids)
            elif (
                retry_index > 0
                and not front_contract.directional_sector_required
            ):
                anchor_ids = tuple(
                    preflight_item.cell_feasible_anchor_segment_ids
                    if preflight_item is not None
                    and preflight_item.cell_feasible_anchor_segment_ids
                    else interface.anchor_segment_ids
                )
            else:
                anchor_ids = _select_executable_anchor_ids(
                    scene,
                    interface=interface,
                    required_pixels=requested_allocation,
                    maximum_depth_px=initial_depth_cap,
                    allowed_anchor_ids=(
                        preflight_item.cell_feasible_anchor_segment_ids
                        if preflight_item is not None
                        else ()
                    ),
                    maximum_selected_anchor_fraction=(
                        front_contract.maximum_selected_anchor_fraction
                    ),
                    minimum_unselected_anchor_count=(
                        front_contract.minimum_unselected_anchor_count
                    ),
                )
            if not anchor_ids:
                raise RefineContractError(
                    "mechanism requires a directional boundary sector but the "
                    "interface has no executable sector after leaving its protected "
                    "unedited boundary"
                )
            anchor_contact = max(
                1,
                sum(
                    int(np.count_nonzero(scene.anchor_masks[item]))
                    for item in anchor_ids
                ),
            )
            # The allowed band is a hard executable envelope, not the desired
            # realized depth. Keep it aligned with the mechanism-owned
            # depth/span contract; the tapered depth profile below remains the
            # mechanism-specific shape control.  Using 0.80 here duplicated an
            # obsolete preflight heuristic and made valid multi-interface
            # capacity disappear between planning and compilation.
            depth_cap = float(
                initial_depth_cap
                if component_turnover
                else min(
                    initial_depth_cap,
                    max(
                        2,
                        int(
                            np.floor(
                                anchor_contact
                                * front_contract.maximum_depth_span_ratio
                            )
                        ),
                    ),
                )
            )
            estimated_depth = requested_allocation / anchor_contact
            # A tapered/multi-lobe envelope needs more peak depth than the
            # simple area/contact average.  It is nevertheless clamped by the
            # same preflight depth cap that the downstream gate audits.
            peak = float(np.clip(np.ceil(estimated_depth * 2.0), 2, depth_cap))
            requested_mode = (
                "uniform_front"
                if topology["allow_source_component_resolution"]
                else front_contract.profile_mode
            )
            if peak < 5 and requested_mode == "multi_lobe":
                requested_mode = "tapered_lobe"
            lobe_count = (
                front_contract.lobe_count
                if requested_mode == "multi_lobe"
                else 1
            )
            edge_ratio = front_contract.edge_depth_ratio
            noise_ratio = front_contract.noise_depth_ratio
            planned.append(
                PlannedInterface(
                    interface_id=interface.interface_id,
                    source_component_id=interface.source_component_id,
                    target_component_id=interface.target_component_id,
                    anchor_segment=(
                        "directional_contiguous_sector"
                        if front_contract.directional_sector_required
                        else "full_directed_interface"
                    ),
                    allowed_edit_band_px=(0.0, depth_cap),
                    execution_contract=InterfaceExecutionContract(
                        anchor_segment_ids=anchor_ids,
                        area_allocation_fraction=float(fraction),
                        depth_profile=DepthProfile(
                            mode=requested_mode,
                            peak_depth_px=peak,
                            edge_depth_px=max(0.5, peak * edge_ratio),
                            taper_fraction=front_contract.taper_fraction,
                            lobe_count=lobe_count,
                            noise_amplitude_px=min(14.0, peak * noise_ratio),
                            noise_correlation_px=float(
                                np.clip(interface.contact_pixels / 6.0, 6.0, 20.0)
                            ),
                        ),
                        min_anchor_coverage_fraction=0.50,
                        # Rasterized curved interfaces can expose one boundary
                        # pixel just outside a selected segment at each end.
                        # A 2% fractional limit is brittle on short segments
                        # (3/145 fails although it is the same connected
                        # front); 3% remains strict while covering that finite
                        # endpoint effect.
                        max_off_anchor_contact_fraction=0.03,
                        allocation_tolerance_fraction=0.02,
                    ),
                    prohibited_region_ids=(),
                    supporting_rule_ids=rule_ids,
                    expected_morphology=(
                        "continuous component-boundary turnover without remote islands"
                        if component_turnover
                        else "distributed shallow-to-moderate lobes over independent legal source components"
                    ),
                    confidence=0.45,
                )
            )
        plan = EditPlan(
            schema_version=EDIT_PLAN_SCHEMA_VERSION,
            case_id=case.case_id,
            normalized_intent=case.instruction,
            primitive_id=case.primitive_id,
            source_labels=(source_label,),
            target_label=bundle.edit_contract.target_label,
            area_budget=case.area_budget,
            candidate_interfaces=tuple(planned),
            tool_program=ToolProgram(
                allowed_tools=bundle.edit_contract.allowed_tools,
                parameter_ranges={
                    "max_changed_components": min(
                        32,
                        sum(
                            max(
                                1,
                                item.execution_contract.depth_profile.lobe_count
                                * len(item.execution_contract.anchor_segment_ids),
                            )
                            for item in planned
                        ),
                    ),
                    "min_component_area_px": 16,
                    "max_depth_span_ratio": (
                        front_contract.maximum_depth_span_ratio
                    ),
                    "max_bbox_fill_fraction": 0.985,
                    "max_boundary_compactness": (
                        front_contract.maximum_boundary_compactness
                    ),
                    "max_source_component_changed_fraction": (
                        topology["maximum_source_component_changed_fraction"]
                    ),
                    "min_source_component_remaining_px": (
                        topology["minimum_source_component_remaining_px"]
                    ),
                    "allow_source_component_resolution": (
                        topology["allow_source_component_resolution"]
                    ),
                    "allow_target_hole_resolution": (
                        topology["allow_target_hole_resolution"]
                    ),
                    "allow_source_component_split": (
                        topology["allow_source_component_split"]
                    ),
                    "minimum_residual_components": (
                        topology["minimum_residual_components"]
                    ),
                    "maximum_residual_components": (
                        topology["maximum_residual_components"]
                    ),
                    "minimum_residual_component_area_px": (
                        topology["minimum_residual_component_area_px"]
                    ),
                    "minimum_residual_spacing_px": (
                        topology["minimum_residual_spacing_px"]
                    ),
                    "residual_area_floor_fraction": (
                        topology["residual_area_floor_fraction"]
                    ),
                    "target_component_merge_policy": (
                        joint_bundle.mechanism.tissue_program.target_component_merge_policy
                    ),
                    "tissue_geometry_mode": (
                        topology["geometry_mode"]
                    ),
                    "editable_source_fine_ids": list(
                        joint_bundle.annotation_profile.mechanism_editable_source_fine_ids.get(
                            f"{joint_bundle.mechanism.mechanism_id}::{case.primitive_id}",
                            joint_bundle.annotation_profile.mechanism_editable_source_fine_ids.get(
                                joint_bundle.mechanism.mechanism_id, ()
                            ),
                        )
                    ),
                    "editable_target_fine_ids": list(
                        joint_bundle.annotation_profile.mechanism_editable_target_fine_ids.get(
                            f"{joint_bundle.mechanism.mechanism_id}::{case.primitive_id}",
                            joint_bundle.annotation_profile.mechanism_editable_target_fine_ids.get(
                                joint_bundle.mechanism.mechanism_id, ()
                            ),
                        )
                    ),
                    "mechanism_topology_fallback_activated": bool(
                        topology["fallback_activated"]
                    ),
                    "min_parallel_front_depth_cv": (
                        0.10 if component_turnover else (
                            0.18 if residual_fragmentation else 0.25
                        )
                    ),
                    "parallel_front_linearity_ratio": 20.0,
                    "parallel_front_min_depth_px": 5.0,
                    "parallel_front_min_pixels": 64,
                    "directional_sector_required": (
                        front_contract.directional_sector_required
                    ),
                    "maximum_selected_anchor_fraction": (
                        front_contract.maximum_selected_anchor_fraction
                    ),
                    "minimum_unselected_anchor_count": (
                        front_contract.minimum_unselected_anchor_count
                    ),
                },
                # Four independently parameterized tissue fronts are enough
                # because the joint stage realizes three cell layouts per
                # passing tissue candidate, preserving the 12-pair joint
                # portfolio without paying for eight tissue masks that can
                # never survive ``maximum_tissue_candidates=4``.
                candidate_count=4,
            ),
            hard_invariants=tuple(sorted(set(bundle.edit_contract.required_check_ids))),
            uncertainties=("current Codex session supplied the mechanism; this deterministic adapter only compiled certified mask geometry",),
            planner_confidence=0.45,
            escalation_reason="requires_independent_mask_condition_critic",
        )
        return plan, {
            "provider": self.name,
            "selected_interface_count": len(planned),
            "estimated_capacity_pixels": cumulative,
            "requested_pixels": target_pixels,
            "nuclei_preflight_version": (
                nuclei_preflight.version if nuclei_preflight is not None else None
            ),
            "nuclei_feasible_interface_count": (
                len(nuclei_preflight.feasible_interface_ids)
                if nuclei_preflight is not None
                else None
            ),
            "supports_pathology_vision": False,
            "execution_retry_index": retry_index,
            "previous_execution_feedback": feedback,
            "input_tokens": 0,
            "output_tokens": 0,
        }


def _select_executable_anchor_ids(
    scene: SceneAnalysis,
    *,
    interface,
    required_pixels: int,
    maximum_depth_px: float,
    allowed_anchor_ids: tuple[str, ...] = (),
    maximum_selected_anchor_fraction: float = 1.0,
    minimum_unselected_anchor_count: int = 0,
) -> tuple[str, ...]:
    """Choose the shortest broad anchor group with enough legal capacity.

    Using every addressable anchor on a long winding boundary turns a modest
    area request into a very thin high-perimeter ribbon.  This deterministic
    compiler keeps adding spatially adjacent anchors until both capacity and a
    preferred shallow depth/span envelope are available.
    """

    source_component = scene.component_masks[interface.source_component_id]
    prohibited = np.zeros_like(source_component, dtype=bool)
    for region in scene.prohibited_region_masks.values():
        prohibited |= np.asarray(region, dtype=bool)
    anchor_metadata = {
        item.anchor_segment_id: item
        for item in scene.graph.anchor_segments
        if item.interface_id == interface.interface_id
    }
    records = []
    allowed = set(allowed_anchor_ids)
    for anchor_id in interface.anchor_segment_ids:
        if allowed and anchor_id not in allowed:
            continue
        anchor = scene.anchor_masks[anchor_id]
        contact = max(1, int(np.count_nonzero(anchor)))
        local_depth = min(float(maximum_depth_px), max(2.0, contact * 0.80))
        distance = ndimage.distance_transform_edt(~anchor)
        capacity = int(
            np.count_nonzero(
                source_component
                & ~prohibited
                & (distance <= local_depth)
            )
        )
        metadata = anchor_metadata.get(anchor_id)
        centroid = (
            metadata.centroid_xy
            if metadata is not None
            else tuple(reversed(ndimage.center_of_mass(anchor)))
        )
        records.append((anchor_id, capacity, contact, centroid))
    if not records:
        return ()
    total_anchor_count = len(interface.anchor_segment_ids)
    selection_limit = min(
        len(records),
        int(np.floor(total_anchor_count * maximum_selected_anchor_fraction)),
        total_anchor_count - minimum_unselected_anchor_count,
    )
    if selection_limit < 1:
        return ()
    records.sort(key=lambda item: (-item[1], item[0]))
    selected = [records.pop(0)]
    preferred_minimum_span = int(
        np.ceil(np.sqrt(max(1.0, 2.0 * required_pixels / 0.45)))
    )
    while records and len(selected) < selection_limit:
        union = np.logical_or.reduce(
            [scene.anchor_masks[item[0]] for item in selected]
        )
        contact = int(np.count_nonzero(union))
        group_depth = min(float(maximum_depth_px), max(2.0, contact * 0.80))
        capacity = int(
            np.count_nonzero(
                source_component
                & ~prohibited
                & (ndimage.distance_transform_edt(~union) <= group_depth)
            )
        )
        if capacity >= required_pixels and contact >= preferred_minimum_span:
            break
        selected_centroids = np.asarray([item[3] for item in selected], dtype=float)
        next_index = min(
            range(len(records)),
            key=lambda index: (
                float(
                    np.min(
                        np.linalg.norm(
                            selected_centroids
                            - np.asarray(records[index][3], dtype=float),
                            axis=1,
                        )
                    )
                ),
                -records[index][1],
                records[index][0],
            ),
        )
        selected.append(records.pop(next_index))
    return tuple(sorted(item[0] for item in selected))

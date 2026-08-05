"""Independent tissue Planner adapter that can bind multiple legal components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import ndimage

from phase3_mask_edit_refine.agents import EDIT_PLAN_SCHEMA_VERSION
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

from .skills.repository import JointSkillBundle
from .feasibility import JointNucleiPreflight


@dataclass(frozen=True)
class MultiInterfaceResearchTissuePlanner:
    """Use all needed legal source components; no H&E authority is claimed."""

    name: str = "multi_interface_research_tissue_planner"

    def create_joint_tissue_plan(
        self,
        *,
        case: CaseContext,
        scene: SceneAnalysis,
        bundle: ActiveKnowledgeBundle,
        joint_bundle: JointSkillBundle,
        image_paths: Sequence[str | Path],
        nuclei_preflight: JointNucleiPreflight | None = None,
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
        # Avoid allocating the same source component through several target
        # contacts. Keep its longest compatible interface, then include enough
        # independent source components to express the immutable area request.
        best_by_source = {}
        for item in legal:
            current = best_by_source.get(item.source_component_id)
            if current is None or (item.contact_pixels, item.interface_id) > (current.contact_pixels, current.interface_id):
                best_by_source[item.source_component_id] = item
        ranked = sorted(best_by_source.values(), key=lambda item: (-item.contact_pixels, item.interface_id))
        source_label = ranked[0].source_label
        ranked = [item for item in ranked if item.source_label == source_label]
        source_region = np.zeros((scene.graph.height, scene.graph.width), dtype=bool)
        for item in ranked:
            source_region |= scene.component_masks[item.source_component_id]
        target_pixels = case.area_budget.target_pixels(source_region, source_region)
        selected = []
        capacities = []
        cumulative = 0
        for item in ranked[:16]:
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
            if cumulative >= target_pixels:
                break
        if not selected:
            raise RefineContractError("preflight left no executable tissue capacity")
        total_capacity = max(1, sum(capacities))
        rule_ids = tuple(rule.rule_id for rule in bundle.active_rules) + tuple(item.constraint_id for item in bundle.active_mask_constraints)
        planned = []
        for interface, capacity in zip(selected, capacities):
            fraction = capacity / total_capacity
            requested_allocation = min(
                capacity,
                max(1, int(round(target_pixels * fraction))),
            )
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
            anchor_ids = _select_executable_anchor_ids(
                scene,
                interface=interface,
                required_pixels=requested_allocation,
                maximum_depth_px=initial_depth_cap,
            )
            anchor_contact = max(
                1,
                sum(
                    int(np.count_nonzero(scene.anchor_masks[item]))
                    for item in anchor_ids
                ),
            )
            depth_cap = float(
                min(initial_depth_cap, max(2, int(np.floor(anchor_contact * 0.80))))
            )
            estimated_depth = requested_allocation / anchor_contact
            # A tapered/multi-lobe envelope needs more peak depth than the
            # simple area/contact average.  It is nevertheless clamped by the
            # same preflight depth cap that the downstream gate audits.
            peak = float(np.clip(np.ceil(estimated_depth * 2.0), 2, depth_cap))
            multi_lobe = interface.contact_pixels >= 24 and peak >= 5
            lobe_count = 3 if interface.contact_pixels >= 72 else 2
            edge_ratio = 0.10 if multi_lobe else 0.28
            noise_ratio = 0.30 if multi_lobe else 0.20
            planned.append(
                PlannedInterface(
                    interface_id=interface.interface_id,
                    source_component_id=interface.source_component_id,
                    target_component_id=interface.target_component_id,
                    anchor_segment="full_directed_interface",
                    allowed_edit_band_px=(0.0, depth_cap),
                    execution_contract=InterfaceExecutionContract(
                        anchor_segment_ids=anchor_ids,
                        area_allocation_fraction=float(fraction),
                        depth_profile=DepthProfile(
                            mode=("multi_lobe" if multi_lobe else "tapered_lobe"),
                            peak_depth_px=peak,
                            edge_depth_px=max(0.5, peak * edge_ratio),
                            taper_fraction=(0.42 if multi_lobe else 0.34),
                            lobe_count=(lobe_count if multi_lobe else 1),
                            noise_amplitude_px=min(14.0, peak * noise_ratio),
                            noise_correlation_px=float(
                                np.clip(interface.contact_pixels / 6.0, 6.0, 20.0)
                            ),
                        ),
                        min_anchor_coverage_fraction=0.35,
                        max_off_anchor_contact_fraction=0.02,
                        allocation_tolerance_fraction=0.02,
                    ),
                    prohibited_region_ids=(),
                    supporting_rule_ids=rule_ids,
                    expected_morphology="distributed shallow-to-moderate lobes over independent legal source components",
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
                        4,
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
                    "max_depth_span_ratio": 1.25,
                    "max_bbox_fill_fraction": 0.985,
                    "max_boundary_compactness": 40.0,
                    "max_source_component_changed_fraction": 0.55,
                    "min_source_component_remaining_px": 64,
                    "min_parallel_front_depth_cv": 0.25,
                    "parallel_front_linearity_ratio": 20.0,
                    "parallel_front_min_depth_px": 5.0,
                    "parallel_front_min_pixels": 64,
                },
                candidate_count=12,
            ),
            hard_invariants=tuple(sorted(set(bundle.edit_contract.required_check_ids))),
            uncertainties=("current Codex session supplied mechanism; this tissue adapter did not inspect H&E",),
            planner_confidence=0.45,
            escalation_reason="requires_independent_multimodal_joint_critic",
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
            "input_tokens": 0,
            "output_tokens": 0,
        }


def _select_executable_anchor_ids(
    scene: SceneAnalysis,
    *,
    interface,
    required_pixels: int,
    maximum_depth_px: float,
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
    for anchor_id in interface.anchor_segment_ids:
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
        return tuple(interface.anchor_segment_ids)
    records.sort(key=lambda item: (-item[1], item[0]))
    selected = [records.pop(0)]
    preferred_minimum_span = int(
        np.ceil(np.sqrt(max(1.0, 2.0 * required_pixels / 0.45)))
    )
    while records:
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

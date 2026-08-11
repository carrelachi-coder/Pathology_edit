"""Joint Planner contracts and an explicitly non-visual offline implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from phase3_mask_edit_refine.models import EditPlan

from .models import (
    CellEditPlan,
    CouplingPlan,
    JointCaseContext,
    JointContractError,
    JointEditPlan,
)
from .scene import JointSceneAnalysis
from .skills.repository import JointSkillBundle
from .skills.schema import JointMechanismSkill

JOINT_PLAN_SCHEMA_VERSION = "joint-pathology-edit-plan-v2"

LOCAL_POPULATION_PRIMITIVES = frozenset(
    {
        "cell-type-abundance-decrease-v1",
        "cell-type-abundance-increase-v1",
        "cellularity-decrease-v1",
        "cellularity-increase-v1",
    }
)


@dataclass(frozen=True)
class JointInterpretationOption:
    primitive_id: str
    semantic_fit: str
    semantic_priority: int
    semantic_rationale: str
    mechanism: JointMechanismSkill
    feasibility: Mapping[str, Any]

    @property
    def option_id(self) -> str:
        return f"{self.primitive_id}::{self.mechanism.mechanism_id}"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id,
            "primitive_id": self.primitive_id,
            "semantic_fit": self.semantic_fit,
            "semantic_priority": self.semantic_priority,
            "semantic_rationale": self.semantic_rationale,
            "mechanism_id": self.mechanism.mechanism_id,
            "mechanism_summary": self.mechanism.summary,
            "required_observations": list(
                self.mechanism.recognition.required_observations
            ),
            "contraindications": list(
                self.mechanism.recognition.contraindications
            ),
            "minimum_confidence": self.mechanism.recognition.minimum_confidence,
            "feasibility": dict(self.feasibility),
        }


class JointPlanner(Protocol):
    name: str
    supports_pathology_vision: bool

    def select_interpretation(
        self,
        *,
        case: JointCaseContext,
        scene: JointSceneAnalysis,
        options: Sequence[JointInterpretationOption],
        image_paths: Sequence[str | Path],
    ) -> tuple[str, str, dict[str, Any]]: ...

    def create_plan(
        self,
        *,
        case: JointCaseContext,
        scene: JointSceneAnalysis,
        bundle: JointSkillBundle,
        tissue_plan: EditPlan | None,
        image_paths: Sequence[str | Path],
    ) -> tuple[JointEditPlan, dict[str, Any]]: ...


@dataclass(frozen=True)
class HeuristicJointPlanner:
    """Compile a skill-selected plan offline; never production-authoritative."""

    name: str = "heuristic_joint_planner"
    supports_pathology_vision: bool = False

    def select_interpretation(self, *, case, scene, options, image_paths):
        del scene, image_paths
        requested = case.provenance.get("joint_mechanism_id")
        if requested == "__abstain__":
            reason = case.provenance.get(
                "joint_mechanism_abstain_reason",
                "current-session visual review found no safely representable mechanism",
            )
            raise JointContractError(f"offline visual Planner abstained: {reason}")
        requested_primitive = case.provenance.get("joint_primitive_id")
        matching = [
            item
            for item in options
            if item.mechanism.mechanism_id == requested
            and (
                requested_primitive is None
                or item.primitive_id == requested_primitive
            )
        ]
        if not isinstance(requested, str) or len(matching) != 1:
            raise JointContractError(
                "offline joint planner requires an unambiguous provenance "
                "joint_mechanism_id/joint_primitive_id decision; it will not "
                "resolve natural-language ambiguity without H&E vision"
            )
        selected = matching[0]
        return selected.primitive_id, requested, {
            "provider": self.name,
            "selection_mode": "explicit_research_metadata",
            "supports_pathology_vision": False,
            "selection": {
                "option_id": selected.option_id,
                "primitive_id": selected.primitive_id,
                "mechanism_id": requested,
                "semantic_fit": selected.semantic_fit,
                "interpretation_explanation": (
                    "explicit offline research metadata selected this auditable interpretation"
                ),
            },
        }

    def create_plan(
        self,
        *,
        case: JointCaseContext,
        scene: JointSceneAnalysis,
        bundle: JointSkillBundle,
        tissue_plan: EditPlan | None,
        image_paths: Sequence[str | Path],
    ) -> tuple[JointEditPlan, dict[str, Any]]:
        del image_paths
        mechanism = bundle.mechanism
        label_contract = mechanism.tissue_program.primitive_label_contracts.get(
            case.primitive_id
        )
        if label_contract is None:
            raise JointContractError(
                "joint mechanism has no label contract for the primitive"
            )
        if bundle.primitive.scope == "tissue_and_cell":
            if tissue_plan is None:
                raise JointContractError("tissue primitive requires a tissue plan")
            if not set(tissue_plan.source_labels).issubset(
                label_contract["source_labels"]
            ):
                raise JointContractError(
                    "compiled tissue plan source label is illegal for the joint mechanism"
                )
            if tissue_plan.target_label not in label_contract["target_labels"]:
                raise JointContractError(
                    "compiled tissue plan target label is illegal for the joint mechanism"
                )
        elif tissue_plan is not None:
            raise JointContractError("cell-only primitive forbids a tissue plan")
        if mechanism.representability.required_auxiliary_structures:
            available = set(scene.auxiliary_structure_masks)
            missing = sorted(
                set(mechanism.representability.required_auxiliary_structures)
                - available
            )
            if missing:
                raise JointContractError(
                    "joint mechanism lacks required auxiliary observations: "
                    + ", ".join(missing)
                )
        if (
            not mechanism.representability.allow_semantic_instance_fallback
            and scene.cells.observation_quality != "native_instance"
        ):
            raise JointContractError(
                "mechanism requires native nucleus instances; semantic fallback is forbidden"
            )
        protected = tuple(
            item.instance_id for item in scene.cells.instances if item.touches_border
        )
        if bundle.primitive.scope == "cell_only":
            if case.primitive_id in LOCAL_POPULATION_PRIMITIVES:
                return self._create_local_population_plan(
                    case=case,
                    scene=scene,
                    bundle=bundle,
                    tissue_plan=tissue_plan,
                    protected=protected,
                )
            requested_interfaces = case.provenance.get("joint_interface_ids", ())
            if isinstance(requested_interfaces, str):
                requested_interfaces = (requested_interfaces,)
            known = {item.interface_id for item in scene.tissue.graph.interfaces}
            interface_ids = tuple(
                value for value in requested_interfaces if value in known
            )
            if not interface_ids:
                host = set(bundle.primitive.host_tissue_labels)
                compatible = [
                    item
                    for item in scene.tissue.graph.interfaces
                    if (item.source_label in host and item.target_label == "Tumor")
                    or (item.target_label in host and item.source_label == "Tumor")
                ]
                compatible.sort(
                    key=lambda item: (-item.contact_pixels, item.interface_id)
                )
                interface_ids = tuple(item.interface_id for item in compatible[:3])
            if not interface_ids:
                raise JointContractError(
                    "cell-only heuristic planner found no tumor/host interface"
                )
            anchor_ids = tuple(
                anchor
                for item in scene.tissue.graph.interfaces
                if item.interface_id in interface_ids
                for anchor in item.anchor_segment_ids[:2]
            )
            baseline_mode = "structured_add"
            quota_role = "explicit_increment"
            layout = mechanism.cell_program.layout_for(case.primitive_id)
            actions = ("retain", "add")
            classes = bundle.primitive.target_cell_classes
            core_zone = "selected_interface_receiving_side"
            halo_zone = "skill_bounded_interface_halo"
        else:
            interface_ids = tuple(
                item.interface_id for item in tissue_plan.candidate_interfaces
            )
            anchor_ids = tuple(
                anchor_id
                for item in tissue_plan.candidate_interfaces
                for anchor_id in item.execution_contract.anchor_segment_ids
            )
            render_owned_clearance = (
                case.primitive_id
                in mechanism.cell_program.render_owned_clearance_primitives
            )
            baseline_mode = (
                "render_owned_clearance"
                if render_owned_clearance
                else "regenerate_target_population"
            )
            quota_role = "within_total_quota"
            if render_owned_clearance:
                layout = "preserve_only"
                actions = ("retain", "remove_whole")
                classes = bundle.primitive.target_cell_classes
            else:
                layout = mechanism.cell_program.layout_for(case.primitive_id)
                actions = mechanism.cell_program.actions
                # The primitive owns the direction-specific target population
                # (for example necrosis appearance 2/4 versus resolution 1).
                # The mechanism union is only a capability envelope across all
                # primitives and must not leak irrelevant classes into one run.
                classes = (
                    bundle.primitive.target_cell_classes
                    or mechanism.cell_program.allowed_cell_classes
                )
            core_zone = "tissue_change"
            halo_zone = (
                "skill_bounded_interface_halo"
                if mechanism.coupling.cell_only_target_fraction > 0
                else None
            )
        plan = JointEditPlan(
            schema_version=JOINT_PLAN_SCHEMA_VERSION,
            case_id=case.case_id,
            normalized_intent=case.compiled_normalized_intent(),
            selected_mechanism_id=mechanism.mechanism_id,
            supporting_observations=(
                "skill contract selected from explicit four-axis case metadata",
                "deterministic tissue and nucleus scene graph available",
            ),
            supporting_rule_ids=bundle.active_rule_ids,
            representability_confidence=0.40,
            tissue_plan=tissue_plan,
            cell_plan=CellEditPlan(
                core_zone=core_zone,
                halo_zone=halo_zone,
                actions=actions,
                allowed_cell_classes=classes,
                layout_program_id=layout,
                protected_instance_ids=protected,
                supporting_rule_ids=mechanism.coupling.compatibility_rule_ids,
                expected_morphology="; ".join(
                    mechanism.render.required_for(case.primitive_id)
                ),
                baseline_mode=baseline_mode,
                interface_ids=interface_ids,
                anchor_ids=anchor_ids,
                mechanism_program_id=layout,
                mechanism_quota_role=quota_role,
            ),
            coupling_plan=CouplingPlan(
                compatibility_rule_ids=mechanism.coupling.compatibility_rule_ids,
                area_contract_id=(
                    "cell-count-extent-v1"
                    if bundle.primitive.scope == "cell_only"
                    else "joint-union-g2-v1"
                ),
                render_support_policy_id=mechanism.coupling.render_support_policy_id,
                allow_neoplastic_in_non_tumor_tissue=(
                    mechanism.coupling.allow_neoplastic_in_non_tumor_tissue
                ),
                maximum_halo_px=mechanism.cell_program.halo_distance_px[1],
            ),
            uncertainties=(
                "offline heuristic did not inspect H&E; visual pathology review is required",
            ),
            escalation_reason="requires_multimodal_joint_planner_and_critic",
            structural_unit_ids=_structural_units_for_interfaces(
                scene, interface_ids
            ),
        )
        return plan, {
            "provider": self.name,
            "supports_pathology_vision": False,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    def _create_local_population_plan(
        self, *, case, scene, bundle, tissue_plan, protected
    ):
        if tissue_plan is not None:
            raise JointContractError(
                "local population primitive forbids tissue changes"
            )
        requested_zone = case.provenance.get("joint_population_zone_id")
        zones = {
            item.zone_id: item
            for item in scene.population.zones
            if item.zone_kind == "component"
        }
        component_labels = {
            item.component_id: item.label for item in scene.tissue.graph.components
        }
        host = set(bundle.primitive.host_tissue_labels)
        eligible_zones = [
            item
            for item in zones.values()
            if component_labels.get(item.tissue_component_id) in host
            and item.area_px > 0
        ]
        eligible_zones.sort(
            key=lambda item: (-item.nucleus_count, -item.area_px, item.zone_id)
        )
        if isinstance(requested_zone, str) and requested_zone in zones:
            zone = zones[requested_zone]
            if zone not in eligible_zones:
                raise JointContractError(
                    "requested population zone is not a legal host component"
                )
        elif eligible_zones:
            zone = eligible_zones[0]
        else:
            raise JointContractError("no component population zone can host the edit")

        component_label = component_labels.get(zone.tissue_component_id)
        compatible_classes = set(
            bundle.cell_observation_profile.tissue_compatible_classes.get(
                component_label, ()
            )
        )

        raw_classes = case.provenance.get("target_cell_class_ids", ())
        if isinstance(raw_classes, int):
            raw_classes = (raw_classes,)
        if not isinstance(raw_classes, (list, tuple)):
            raise JointContractError("target_cell_class_ids must be a sequence")
        classes = tuple(
            sorted(
                {
                    int(value)
                    for value in raw_classes
                    if int(value) in bundle.mechanism.cell_program.allowed_cell_classes
                    and int(value) in compatible_classes
                }
            )
        )
        abundance = case.primitive_id.startswith("cell-type-abundance-")
        if abundance and len(classes) != 1:
            raise JointContractError(
                "offline cell abundance planning requires exactly one target_cell_class_id"
            )
        if not classes:
            classes = tuple(
                class_id
                for class_id, count in sorted(zone.class_counts.items())
                if count > 0
                and class_id in bundle.mechanism.cell_program.allowed_cell_classes
                and class_id in compatible_classes
            )
        if not classes:
            raise JointContractError(
                "selected population zone has no observable compatible cell class"
            )
        increase = case.primitive_id.endswith("increase-v1")
        baseline = "structured_add" if increase else "selective_remove"
        action = ("retain", "add") if increase else ("retain", "remove_whole")
        interface_ids: tuple[str, ...] = ()
        anchor_ids: tuple[str, ...] = ()
        spatial_anchor_type = "not_applicable"
        spatial_anchor_observation = None
        layout = bundle.mechanism.cell_program.layout_for(case.primitive_id)
        if layout == "localized_density_gradient":
            depletion = bundle.mechanism.cell_program.cellularity_depletion
            raw_anchor = case.provenance.get("cellularity_depletion_anchor")
            if depletion is None or not isinstance(raw_anchor, Mapping):
                raise JointContractError(
                    "cellularity decrease requires an explicit visual depletion anchor"
                )
            spatial_anchor_type = str(raw_anchor.get("type", ""))
            if spatial_anchor_type not in depletion.allowed_anchor_types:
                raise JointContractError(
                    "depletion anchor type is not allowed by the mechanism skill"
                )
            raw_interfaces = raw_anchor.get("interface_ids", ())
            raw_anchors = raw_anchor.get("anchor_ids", ())
            if not isinstance(raw_interfaces, (list, tuple)) or not isinstance(
                raw_anchors, (list, tuple)
            ):
                raise JointContractError(
                    "depletion interface_ids and anchor_ids must be sequences"
                )
            interface_ids = tuple(str(value) for value in raw_interfaces)
            anchor_ids = tuple(str(value) for value in raw_anchors)
            spatial_anchor_observation = str(
                raw_anchor.get("observation", "")
            ).strip()
            confidence = float(raw_anchor.get("confidence", 0.0))
            interfaces = {
                item.interface_id: item for item in scene.tissue.graph.interfaces
            }
            anchors = {
                item.anchor_segment_id: item
                for item in scene.tissue.graph.anchor_segments
            }
            if not interface_ids or set(interface_ids) - set(interfaces):
                raise JointContractError(
                    "depletion Planner selected an empty or unknown interface"
                )
            if not anchor_ids or set(anchor_ids) - set(anchors):
                raise JointContractError(
                    "depletion Planner selected an empty or unknown anchor"
                )
            detached = [
                value
                for value in anchor_ids
                if anchors[value].interface_id not in interface_ids
            ]
            if detached:
                raise JointContractError(
                    "depletion anchors are detached from the selected interface"
                )
            selected_component = zone.tissue_component_id
            neighbor_labels = set()
            for interface_id in interface_ids:
                interface = interfaces[interface_id]
                if interface.source_component_id == selected_component:
                    neighbor_labels.add(interface.target_label)
                elif interface.target_component_id == selected_component:
                    neighbor_labels.add(interface.source_label)
                else:
                    raise JointContractError(
                        "depletion interface does not touch the selected population component"
                    )
            if neighbor_labels - set(depletion.allowed_neighbor_labels):
                raise JointContractError(
                    "depletion interface neighbor is not allowed by the mechanism skill"
                )
            if (
                not spatial_anchor_observation
                or confidence < bundle.mechanism.recognition.minimum_confidence
            ):
                raise JointContractError(
                    "depletion anchor lacks a confident visible pathology observation"
                )
        plan = JointEditPlan(
            schema_version=JOINT_PLAN_SCHEMA_VERSION,
            case_id=case.case_id,
            normalized_intent=case.compiled_normalized_intent(),
            selected_mechanism_id=bundle.mechanism.mechanism_id,
            supporting_observations=(
                "explicit component population zone selected",
                "local class counts and complete source instances available",
            ),
            supporting_rule_ids=bundle.active_rule_ids,
            representability_confidence=0.40,
            tissue_plan=None,
            cell_plan=CellEditPlan(
                core_zone=zone.zone_id,
                halo_zone=None,
                actions=action,
                allowed_cell_classes=classes,
                layout_program_id=layout,
                protected_instance_ids=protected,
                supporting_rule_ids=bundle.mechanism.coupling.compatibility_rule_ids,
                expected_morphology="; ".join(
                    bundle.mechanism.render.required_for(case.primitive_id)
                ),
                baseline_mode=baseline,
                interface_ids=interface_ids,
                anchor_ids=anchor_ids,
                spatial_anchor_type=spatial_anchor_type,
                spatial_anchor_observation=spatial_anchor_observation,
                mechanism_program_id=layout,
                mechanism_quota_role=(
                    "explicit_increment" if increase else "explicit_decrement"
                ),
            ),
            coupling_plan=CouplingPlan(
                compatibility_rule_ids=(
                    bundle.mechanism.coupling.compatibility_rule_ids
                ),
                area_contract_id="cell-count-extent-v1",
                render_support_policy_id=(
                    bundle.mechanism.coupling.render_support_policy_id
                ),
                allow_neoplastic_in_non_tumor_tissue=False,
                maximum_halo_px=bundle.mechanism.cell_program.halo_distance_px[1],
            ),
            uncertainties=(
                "offline heuristic did not inspect H&E; visual pathology review is required",
            ),
            escalation_reason="requires_multimodal_joint_planner_and_critic",
            structural_unit_ids=_structural_units_for_components(
                scene, (zone.tissue_component_id,)
            ),
        )
        return plan, {
            "provider": self.name,
            "supports_pathology_vision": False,
            "planning_mode": (
                "explicit_interface_anchored_density_gradient_contract"
                if case.primitive_id == "cellularity-decrease-v1"
                else "explicit_population_zone_contract"
            ),
            "input_tokens": 0,
            "output_tokens": 0,
        }


def _structural_units_for_interfaces(
    scene: JointSceneAnalysis,
    interface_ids: tuple[str, ...],
) -> tuple[str, ...]:
    interfaces = {
        item.interface_id: item for item in scene.tissue.graph.interfaces
    }
    component_ids: set[str] = set()
    for interface_id in interface_ids:
        interface = interfaces.get(interface_id)
        if interface is None:
            continue
        component_ids.update(
            (interface.source_component_id, interface.target_component_id)
        )
    return _structural_units_for_components(scene, tuple(component_ids))


def _structural_units_for_components(
    scene: JointSceneAnalysis,
    component_ids: tuple[str, ...],
) -> tuple[str, ...]:
    selected = set(component_ids)
    return tuple(
        sorted(
            str(item["unit_id"])
            for item in scene.structural_hierarchy.get("structure_units", ())
            if isinstance(item, Mapping)
            and item.get("unit_id")
            and item.get("parent_tissue_component_id") in selected
        )
    )

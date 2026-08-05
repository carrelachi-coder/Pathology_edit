"""Joint Planner contracts and an explicitly non-visual offline implementation."""

from __future__ import annotations

from collections.abc import Sequence
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


class JointPlanner(Protocol):
    name: str
    supports_pathology_vision: bool

    def select_mechanism(
        self,
        *,
        case: JointCaseContext,
        scene: JointSceneAnalysis,
        mechanisms: Sequence[JointMechanismSkill],
        image_paths: Sequence[str | Path],
    ) -> tuple[str, dict[str, Any]]: ...

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

    def select_mechanism(self, *, case, scene, mechanisms, image_paths):
        del scene, image_paths
        requested = case.provenance.get("joint_mechanism_id")
        if requested == "__abstain__":
            reason = case.provenance.get(
                "joint_mechanism_abstain_reason",
                "current-session visual review found no safely representable mechanism",
            )
            raise JointContractError(f"offline visual Planner abstained: {reason}")
        available = {item.mechanism_id for item in mechanisms}
        if not isinstance(requested, str) or requested not in available:
            raise JointContractError(
                "offline joint planner requires provenance.joint_mechanism_id; "
                "it will not guess a pathology mechanism without H&E vision"
            )
        return requested, {
            "provider": self.name,
            "selection_mode": "explicit_research_metadata",
            "supports_pathology_vision": False,
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
        label_contract = mechanism.tissue_program.primitive_label_contracts.get(case.primitive_id)
        if label_contract is None:
            raise JointContractError("joint mechanism has no label contract for the primitive")
        if bundle.primitive.scope == "tissue_and_cell":
            if tissue_plan is None:
                raise JointContractError("tissue primitive requires a tissue plan")
            if not set(tissue_plan.source_labels).issubset(label_contract["source_labels"]):
                raise JointContractError("compiled tissue plan source label is illegal for the joint mechanism")
            if tissue_plan.target_label not in label_contract["target_labels"]:
                raise JointContractError("compiled tissue plan target label is illegal for the joint mechanism")
        elif tissue_plan is not None:
            raise JointContractError("cell-only primitive forbids a tissue plan")
        if mechanism.representability.required_auxiliary_structures:
            available = set(scene.auxiliary_structure_masks)
            missing = sorted(
                set(mechanism.representability.required_auxiliary_structures) - available
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
            item.instance_id
            for item in scene.cells.instances
            if item.touches_border or bundle.primitive.scope == "cell_only"
        )
        if bundle.primitive.scope == "cell_only":
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
                    if (
                        item.source_label in host and item.target_label == "Tumor"
                    )
                    or (
                        item.target_label in host and item.source_label == "Tumor"
                    )
                ]
                compatible.sort(
                    key=lambda item: (-item.contact_pixels, item.interface_id)
                )
                interface_ids = tuple(
                    item.interface_id for item in compatible[:3]
                )
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
            layout = mechanism.cell_program.layout_programs[0]
            actions = ("retain", "add")
            classes = bundle.primitive.target_cell_classes
            core_zone = "selected_interface_receiving_side"
            halo_zone = "skill_bounded_interface_halo"
        else:
            interface_ids = tuple(
                item.interface_id for item in tissue_plan.candidate_interfaces
            )
            anchor_ids = tuple(
                item.anchor_segment for item in tissue_plan.candidate_interfaces
            )
            baseline_mode = "regenerate_target_population"
            quota_role = "within_total_quota"
            layout = (
                "population_replacement"
                if "population_replacement" in mechanism.cell_program.layout_programs
                else mechanism.cell_program.layout_programs[0]
            )
            actions = mechanism.cell_program.actions
            classes = mechanism.cell_program.allowed_cell_classes
            core_zone = "tissue_change"
            halo_zone = (
                "skill_bounded_interface_halo"
                if mechanism.coupling.cell_only_target_fraction > 0
                else None
            )
        plan = JointEditPlan(
            schema_version=JOINT_PLAN_SCHEMA_VERSION,
            case_id=case.case_id,
            normalized_intent=case.instruction,
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
                expected_morphology="; ".join(mechanism.render.required_findings),
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
        )
        return plan, {
            "provider": self.name,
            "supports_pathology_vision": False,
            "input_tokens": 0,
            "output_tokens": 0,
        }

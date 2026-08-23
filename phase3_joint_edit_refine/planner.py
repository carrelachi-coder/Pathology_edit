"""Joint Planner contracts and an explicitly non-visual offline implementation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from phase3_mask_edit_refine.models import EditPlan

from .authority import validate_mechanism_nucleus_authority
from .clarification import PlannerClarificationRequired
from .models import (
    CellEditPlan,
    CouplingPlan,
    JointCaseContext,
    JointContractError,
    JointEditPlan,
)
from .planner_inputs import MaskPlannerArtifactRegistry
from .planner_policy import preference_metadata
from .scene import JointSceneAnalysis
from .skills.repository import JointSkillBundle
from .skills.schema import JointMechanismSkill

JOINT_PLAN_SCHEMA_VERSION = "joint-pathology-edit-plan-v2"
_CELL_PORTFOLIO_ISSUER = object()
_DEPLETION_ANCHOR_ISSUER = object()
_ISSUED_CELL_PORTFOLIOS: dict[int, tuple[tuple[int, str], ...]] = {}

LOCAL_POPULATION_PRIMITIVES = frozenset(
    {
        "cell-type-abundance-decrease-v1",
        "cell-type-abundance-increase-v1",
        "cellularity-decrease-v1",
        "cellularity-increase-v1",
        "neoplastic-cell-abundance-decrease-v1",
        "neoplastic-cell-abundance-increase-v1",
    }
)


def _depletion_anchor_binding_sha256(
    *,
    case: JointCaseContext,
    zone_id: str,
    anchor_type: str,
    interface_ids: tuple[str, ...],
    anchor_ids: tuple[str, ...],
) -> str:
    provenance = case.provenance
    payload = {
        "case_id": case.case_id,
        "pathology_domain_id": case.pathology_domain_id,
        "annotation_profile_id": case.annotation_profile_id,
        "primitive_id": case.primitive_id,
        "mechanism_id": provenance.get("joint_mechanism_id"),
        "source_tissue_mask_sha256": provenance.get(
            "source_tissue_mask_sha256"
        ),
        "source_nuclei_mask_sha256": provenance.get(
            "source_nuclei_mask_sha256"
        ),
        "zone_id": zone_id,
        "anchor_type": anchor_type,
        "interface_ids": list(interface_ids),
        "anchor_ids": list(anchor_ids),
    }
    if not payload["source_tissue_mask_sha256"] or not payload[
        "source_nuclei_mask_sha256"
    ]:
        raise JointContractError(
            "compiler-owned depletion anchor lacks source mask digests"
        )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


@dataclass(frozen=True)
class CompilerOwnedDepletionAnchor:
    """Typed mask-graph anchor capability unavailable through case provenance."""

    zone_id: str
    anchor_type: str
    interface_ids: tuple[str, ...]
    anchor_ids: tuple[str, ...]
    binding_sha256: str
    _issuer: object | None = None

    @classmethod
    def issue(
        cls,
        *,
        case: JointCaseContext,
        zone_id: str,
        interface_ids: Sequence[str],
        anchor_ids: Sequence[str],
        anchor_type: str = "interface",
    ) -> CompilerOwnedDepletionAnchor:
        interfaces = tuple(str(value) for value in interface_ids)
        anchors = tuple(str(value) for value in anchor_ids)
        return cls(
            zone_id=str(zone_id),
            anchor_type=str(anchor_type),
            interface_ids=interfaces,
            anchor_ids=anchors,
            binding_sha256=_depletion_anchor_binding_sha256(
                case=case,
                zone_id=str(zone_id),
                anchor_type=str(anchor_type),
                interface_ids=interfaces,
                anchor_ids=anchors,
            ),
            _issuer=_DEPLETION_ANCHOR_ISSUER,
        )

    def validate(self, *, case: JointCaseContext) -> None:
        if self._issuer is not _DEPLETION_ANCHOR_ISSUER:
            raise JointContractError(
                "depletion anchor was not issued by the portfolio compiler"
            )
        expected = _depletion_anchor_binding_sha256(
            case=case,
            zone_id=self.zone_id,
            anchor_type=self.anchor_type,
            interface_ids=self.interface_ids,
            anchor_ids=self.anchor_ids,
        )
        if expected != self.binding_sha256:
            raise JointContractError(
                "compiler-owned depletion anchor is digest-detached"
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
        feasibility = dict(self.feasibility)
        # Semantic selection exposes only quantities actually measured during
        # capability preflight. Spatial/naturalness preference metrics belong
        # to the later immutable tissue/cell candidate portfolios.
        feasibility_metrics = {
            key: value
            for key, value in {
                "semantic_priority": float(self.semantic_priority),
                "feasible_interface_count": feasibility.get(
                    "feasible_interface_count"
                ),
                "aggregate_tissue_capacity_pixels": feasibility.get(
                    "aggregate_tissue_capacity_pixels"
                ),
                "meaningful_tissue_floor_pixels": feasibility.get(
                    "meaningful_tissue_floor_pixels"
                ),
                "complete_reference_instances": feasibility.get(
                    "complete_reference_instances"
                ),
                "candidate_infiltration_interface_count": feasibility.get(
                    "candidate_infiltration_interface_count"
                ),
            }.items()
            if isinstance(value, (int, float))
        }
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
            "planner_policy": {
                "allowed_observation_sources": list(
                    self.mechanism.planner_policy.allowed_observation_sources
                ),
                "prohibited_observation_sources": list(
                    self.mechanism.planner_policy.prohibited_observation_sources
                ),
                "hard_constraint_checker_ids": list(
                    self.mechanism.planner_policy.hard_constraint_checker_ids
                ),
                "selection_preferences": list(
                    self.mechanism.planner_policy.selection_preferences
                ),
                "preference_metric_bindings": list(
                    preference_metadata(self.mechanism.planner_policy)
                ),
                "clarification_triggers": list(
                    self.mechanism.planner_policy.clarification_triggers
                ),
                "allowed_decisions": list(
                    self.mechanism.planner_policy.allowed_decisions
                ),
                "requires_explicit_primitive_intent": (
                    self.mechanism.planner_policy.requires_explicit_primitive_intent
                ),
            },
            "feasibility": feasibility,
            "deterministic_candidate_metrics": feasibility_metrics,
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
        artifact_registry: MaskPlannerArtifactRegistry | None = None,
    ) -> tuple[str, str, dict[str, Any]]: ...

    def create_plan(
        self,
        *,
        case: JointCaseContext,
        scene: JointSceneAnalysis,
        bundle: JointSkillBundle,
        tissue_plan: EditPlan | None,
        image_paths: Sequence[str | Path],
        artifact_registry: MaskPlannerArtifactRegistry | None = None,
        candidate_portfolio: Sequence[Any] = (),
        compiler_owned_depletion_anchor: CompilerOwnedDepletionAnchor | None = None,
    ) -> tuple[JointEditPlan, dict[str, Any]]: ...


@dataclass(frozen=True)
class CertifiedCellPlanCandidate:
    """Immutable pre-LLM cell plan with an exact executable-capacity witness."""

    candidate_id: str
    plan: JointEditPlan
    deterministic_candidate_metrics: dict[str, float]
    allowed_tool_program_ids: tuple[str, ...]
    compiler_certificate_sha256: str
    executable_contract_id: str
    authority_binding_sha256: str
    _issuer: object | None = None

    def to_metadata(self) -> dict[str, Any]:
        if self._issuer is not _CELL_PORTFOLIO_ISSUER:
            raise JointContractError(
                "cell portfolio was not issued by the workflow compiler"
            )
        return {
            "candidate_id": self.candidate_id,
            "interface_ids": list(self.plan.cell_plan.interface_ids),
            "anchor_ids": list(self.plan.cell_plan.anchor_ids),
            "zone_id": self.plan.cell_plan.core_zone,
            "allowed_tool_program_ids": list(self.allowed_tool_program_ids),
            "deterministic_candidate_metrics": dict(
                self.deterministic_candidate_metrics
            ),
            "compiler_certificate_sha256": self.compiler_certificate_sha256,
            "executable_contract_id": self.executable_contract_id,
            "authority_binding_sha256": self.authority_binding_sha256,
            "veto_reasons": [],
        }


@dataclass(frozen=True)
class CellPlanSelectionHandle:
    """Exact compiler certificate selected by the Planner."""

    candidate_id: str
    compiler_certificate_sha256: str
    selected_tool_program_id: str
    executable_contract_id: str
    plan_sha256: str
    authority_binding_sha256: str

    @classmethod
    def from_metadata(cls, value: Any) -> CellPlanSelectionHandle:
        if not isinstance(value, Mapping):
            raise JointContractError(
                "cell Planner omitted the typed certificate selection handle"
            )
        required = (
            "candidate_id",
            "compiler_certificate_sha256",
            "selected_tool_program_id",
            "executable_contract_id",
            "plan_sha256",
            "authority_binding_sha256",
        )
        if set(value) != set(required) or not all(
            isinstance(value.get(key), str) and value[key]
            for key in required
        ):
            raise JointContractError("cell selection handle is malformed")
        return cls(**{key: str(value[key]) for key in required})

    @classmethod
    def from_candidate(
        cls,
        candidate: CertifiedCellPlanCandidate,
        *,
        selected_tool_program_id: str,
    ) -> CellPlanSelectionHandle:
        return cls(
            candidate_id=candidate.candidate_id,
            compiler_certificate_sha256=(
                candidate.compiler_certificate_sha256
            ),
            selected_tool_program_id=selected_tool_program_id,
            executable_contract_id=candidate.executable_contract_id,
            plan_sha256=hashlib.sha256(
                json.dumps(
                    candidate.plan.to_metadata(),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            authority_binding_sha256=candidate.authority_binding_sha256,
        )

    def to_metadata(self) -> dict[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "compiler_certificate_sha256": self.compiler_certificate_sha256,
            "selected_tool_program_id": self.selected_tool_program_id,
            "executable_contract_id": self.executable_contract_id,
            "plan_sha256": self.plan_sha256,
            "authority_binding_sha256": self.authority_binding_sha256,
        }

    def validate_candidate(
        self,
        candidate: CertifiedCellPlanCandidate,
        *,
        plan: JointEditPlan,
    ) -> None:
        expected = self.from_candidate(
            candidate,
            selected_tool_program_id=self.selected_tool_program_id,
        )
        if self != expected or plan != candidate.plan:
            raise JointContractError(
                "cell selection handle is detached from the exact compiler certificate"
            )


@dataclass(frozen=True)
class CellPlanCandidateVeto:
    """Audited, non-selectable cell-plan variant rejected before the LLM."""

    candidate_id: str
    bound_interface_ids: tuple[str, ...]
    bound_anchor_ids: tuple[str, ...]
    bound_zone_id: str | None
    veto_reasons: tuple[str, ...]
    compiler_certificate_sha256: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "interface_ids": list(self.bound_interface_ids),
            "anchor_ids": list(self.bound_anchor_ids),
            "zone_id": self.bound_zone_id,
            "selectable": False,
            "veto_reasons": list(self.veto_reasons),
            "compiler_certificate_sha256": (
                self.compiler_certificate_sha256
            ),
        }


@dataclass(frozen=True)
class CertifiedCellPlanPortfolio:
    """Immutable pre-LLM survivors plus audited non-selectable variants."""

    survivors: tuple[CertifiedCellPlanCandidate, ...]
    vetoed: tuple[CellPlanCandidateVeto, ...]
    authority_binding: dict[str, Any]
    authority_binding_sha256: str
    _issuer: object | None = None

    def __iter__(self):
        return iter(self.survivors)

    def __len__(self) -> int:
        return len(self.survivors)

    def __getitem__(self, index: int) -> CertifiedCellPlanCandidate:
        return self.survivors[index]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "surviving_candidates": [
                item.to_metadata() for item in self.survivors
            ],
            "vetoed_candidates": [
                item.to_metadata() for item in self.vetoed
            ],
            "authority_binding_sha256": self.authority_binding_sha256,
        }

    def validate_authority(self, *, expected_binding_sha256: str) -> None:
        issued = _ISSUED_CELL_PORTFOLIOS.get(id(self))
        observed = tuple(
            (id(item), item.compiler_certificate_sha256)
            for item in self.survivors
        )
        binding_sha = hashlib.sha256(
            json.dumps(
                self.authority_binding,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        if (
            self._issuer is not _CELL_PORTFOLIO_ISSUER
            or issued is None
            or issued != observed
            or binding_sha != self.authority_binding_sha256
            or binding_sha != expected_binding_sha256
        ):
            raise JointContractError(
                "cell portfolio is not the compiler-issued capability for this case/budget"
            )
        for item in self.survivors:
            validate_cell_plan_candidate(item)


def certify_cell_plan_candidate(
    *,
    plan: JointEditPlan,
    deterministic_candidate_metrics: Mapping[str, float],
    allowed_tool_program_ids: Sequence[str],
    executable_contract_id: str,
    authority_binding_sha256: str,
    _issuer: object | None = None,
) -> CertifiedCellPlanCandidate:
    if _issuer is not _CELL_PORTFOLIO_ISSUER:
        raise JointContractError(
            "cell certificates can only be issued by the workflow compiler"
        )
    payload = {
        "plan": plan.to_metadata(),
        "deterministic_candidate_metrics": dict(
            sorted(deterministic_candidate_metrics.items())
        ),
        "allowed_tool_program_ids": list(allowed_tool_program_ids),
        "executable_contract_id": executable_contract_id,
        "authority_binding_sha256": authority_binding_sha256,
    }
    certificate_sha = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return CertifiedCellPlanCandidate(
        candidate_id="cell-plan:" + certificate_sha[:20],
        plan=plan,
        deterministic_candidate_metrics=dict(deterministic_candidate_metrics),
        allowed_tool_program_ids=tuple(allowed_tool_program_ids),
        compiler_certificate_sha256=certificate_sha,
        executable_contract_id=executable_contract_id,
        authority_binding_sha256=authority_binding_sha256,
        _issuer=_CELL_PORTFOLIO_ISSUER,
    )


def validate_cell_plan_candidate(candidate: CertifiedCellPlanCandidate) -> None:
    if candidate._issuer is not _CELL_PORTFOLIO_ISSUER:
        raise JointContractError(
            "cell candidate was not issued by the workflow compiler"
        )
    expected = certify_cell_plan_candidate(
        plan=candidate.plan,
        deterministic_candidate_metrics=(
            candidate.deterministic_candidate_metrics
        ),
        allowed_tool_program_ids=candidate.allowed_tool_program_ids,
        executable_contract_id=candidate.executable_contract_id,
        authority_binding_sha256=candidate.authority_binding_sha256,
        _issuer=_CELL_PORTFOLIO_ISSUER,
    )
    if (
        expected.candidate_id != candidate.candidate_id
        or expected.compiler_certificate_sha256
        != candidate.compiler_certificate_sha256
    ):
        raise JointContractError("cell candidate certificate SHA is detached")


def _issue_cell_plan_portfolio(
    *,
    candidates: Sequence[dict[str, Any]],
    vetoed: Sequence[CellPlanCandidateVeto],
    authority_binding: Mapping[str, Any],
) -> CertifiedCellPlanPortfolio:
    binding = dict(authority_binding)
    binding_sha = hashlib.sha256(
        json.dumps(
            binding,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    survivors = tuple(
        certify_cell_plan_candidate(
            **payload,
            authority_binding_sha256=binding_sha,
            _issuer=_CELL_PORTFOLIO_ISSUER,
        )
        for payload in candidates
    )
    portfolio = CertifiedCellPlanPortfolio(
        survivors=survivors,
        vetoed=tuple(vetoed),
        authority_binding=binding,
        authority_binding_sha256=binding_sha,
        _issuer=_CELL_PORTFOLIO_ISSUER,
    )
    _ISSUED_CELL_PORTFOLIOS[id(portfolio)] = tuple(
        (id(item), item.compiler_certificate_sha256)
        for item in survivors
    )
    return portfolio


@dataclass(frozen=True)
class HeuristicJointPlanner:
    """Compile a skill-selected plan offline; never production-authoritative."""

    name: str = "heuristic_joint_planner"
    supports_pathology_vision: bool = False

    def select_interpretation(
        self, *, case, scene, options, image_paths, artifact_registry=None
    ):
        del scene, image_paths, artifact_registry
        requested = case.provenance.get("joint_mechanism_id")
        if requested == "__clarify__":
            primitive_ids = tuple(
                dict.fromkeys(
                    item.primitive_id
                    for item in sorted(
                        options,
                        key=lambda value: (
                            value.semantic_priority,
                            value.primitive_id,
                            value.mechanism.mechanism_id,
                        ),
                    )
                )
            )[:3]
            raise PlannerClarificationRequired(
                str(
                    case.provenance.get(
                        "joint_mechanism_clarification_reason",
                        "current-session review found multiple executable pathological meanings that the instruction does not distinguish",
                    )
                ),
                primitive_ids=primitive_ids,
            )
        if requested == "__abstain__":
            reason = case.provenance.get(
                "joint_mechanism_abstain_reason",
                "current-session mask-graph review found no safely representable mechanism",
            )
            raise JointContractError(f"offline mask-graph Planner abstained: {reason}")
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
                "resolve natural-language ambiguity without the mask-graph LLM Planner"
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
        artifact_registry: MaskPlannerArtifactRegistry | None = None,
        candidate_portfolio: Sequence[Any] = (),
        compiler_owned_depletion_anchor: CompilerOwnedDepletionAnchor | None = None,
    ) -> tuple[JointEditPlan, dict[str, Any]]:
        del image_paths, artifact_registry
        if candidate_portfolio:
            candidates = tuple(candidate_portfolio)
            selected_index = int(
                case.provenance.get("cell_portfolio_candidate_index", 0)
            )
            if not 0 <= selected_index < len(candidates):
                raise JointContractError(
                    "requested cell portfolio candidate index is out of range: "
                    f"index={selected_index}, candidates={len(candidates)}"
                )
            selected = candidates[selected_index]
            tool_program_id = selected.allowed_tool_program_ids[0]
            return selected.plan, {
                "provider": self.name,
                "supports_pathology_vision": False,
                "planning_mode": "indexed_pre_llm_certified_cell_candidate",
                "selected_candidate_index": selected_index,
                "selected_candidate_id": selected.candidate_id,
                "selected_tool_program_id": tool_program_id,
                "selection_handle": CellPlanSelectionHandle.from_candidate(
                    selected,
                    selected_tool_program_id=tool_program_id,
                ).to_metadata(),
                "portfolio_candidate_count": len(candidates),
                "input_tokens": 0,
                "output_tokens": 0,
            }
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
        nucleus_authority = validate_mechanism_nucleus_authority(
            scene.cells.instances,
            allow_semantic_instance_fallback=(
                mechanism.representability.allow_semantic_instance_fallback
            ),
            required_cell_classes=(
                mechanism.representability.required_cell_classes
                or bundle.primitive.target_cell_classes
            ),
            actions=mechanism.cell_program.actions,
        )
        if not nucleus_authority["passed"]:
            raise JointContractError(
                "mechanism nucleus authority is insufficient: "
                + ", ".join(nucleus_authority["reasons"])
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
                    compiler_owned_depletion_anchor=(
                        compiler_owned_depletion_anchor
                    ),
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
                if (
                    "external_boundary_binding"
                    in mechanism.planner_policy.hard_constraint_checker_ids
                ):
                    from .feasibility import classify_tumor_stroma_boundary

                    compatible = [
                        item
                        for item in compatible
                        if classify_tumor_stroma_boundary(
                            scene=scene,
                            interface=item,
                            allowed_host_labels=tuple(sorted(host)),
                        )["external_tumor_stroma_boundary"]
                    ]
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
            requested_anchors = case.provenance.get("joint_anchor_ids", ())
            if isinstance(requested_anchors, str):
                requested_anchors = (requested_anchors,)
            if requested_anchors:
                known_anchors = {
                    anchor
                    for item in scene.tissue.graph.interfaces
                    if item.interface_id in interface_ids
                    for anchor in item.anchor_segment_ids
                }
                anchor_ids = tuple(
                    value for value in requested_anchors if value in known_anchors
                )
                if not anchor_ids:
                    raise JointContractError(
                        "requested cell anchors are detached from the certified interfaces"
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
                "offline heuristic does not resolve natural-language ambiguity or rank certified options semantically",
            ),
            escalation_reason="requires_mask_graph_llm_planner_and_independent_condition_critic",
            structural_unit_ids=_structural_units_for_interfaces(
                scene, interface_ids
            ),
            supporting_preference_rule_ids=(
                mechanism.planner_policy.selection_preferences
            ),
        )
        return plan, {
            "provider": self.name,
            "supports_pathology_vision": False,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    def _create_local_population_plan(
        self,
        *,
        case,
        scene,
        bundle,
        tissue_plan,
        protected,
        compiler_owned_depletion_anchor=None,
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
        raw_classes = case.provenance.get("target_cell_class_ids", ())
        if isinstance(raw_classes, int):
            raw_classes = (raw_classes,)
        if not isinstance(raw_classes, (list, tuple)):
            raise JointContractError("target_cell_class_ids must be a sequence")
        requested_class_ids = tuple(sorted({int(value) for value in raw_classes}))
        abundance = case.primitive_id.startswith(
            (
                "cell-type-abundance-",
                "neoplastic-cell-abundance-",
            )
        )
        host = set(bundle.primitive.host_tissue_labels)
        eligible_zones = [
            item
            for item in zones.values()
            if component_labels.get(item.tissue_component_id) in host
            and item.area_px > 0
        ]
        if (
            abundance
            and requested_class_ids
            and case.annotation_profile_id
            in {"glas-gland-v1", "panda-gleason-v1"}
        ):
            # Gland-bearing profiles need an explicit compartment distinction:
            # neoplastic abundance is intratumoral, while immune/connective
            # abundance is interglandular stromal.  This prevents a global
            # class count from selecting a lumen-bearing gland component.
            required_label = (
                "Tumor" if set(requested_class_ids) == {1} else "Stroma"
            )
            eligible_zones = [
                item
                for item in eligible_zones
                if component_labels.get(item.tissue_component_id)
                == required_label
            ]
        eligible_zones.sort(
            key=lambda item: (
                -sum(
                    int(item.class_counts.get(class_id, 0))
                    for class_id in requested_class_ids
                )
                if abundance and requested_class_ids
                else -item.nucleus_count,
                -item.nucleus_count,
                -item.area_px,
                item.zone_id,
            )
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
        increase = case.primitive_id.endswith("increase-v1")
        if (
            not classes
            and increase
            and case.annotation_profile_id == "panda-gleason-v1"
        ):
            # A legal PANDA host component may contain no nuclei even though
            # the same tissue class elsewhere supplies verified source-shape
            # authority.  Addition is allowed to populate that empty zone;
            # depletion continues to require observable local instances.
            classes = tuple(
                sorted(
                    {
                        int(item.class_id)
                        for item in scene.cells.instances
                        if item.completeness_status == "complete"
                        and not item.quality_flags
                        and item.class_id
                        in bundle.mechanism.cell_program.allowed_cell_classes
                        and item.class_id in compatible_classes
                    }
                )
            )
        if not classes:
            raise JointContractError(
                "selected population zone has no observable compatible cell class"
            )
        baseline = "structured_add" if increase else "selective_remove"
        action = ("retain", "add") if increase else ("retain", "remove_whole")
        interface_ids: tuple[str, ...] = ()
        anchor_ids: tuple[str, ...] = ()
        spatial_anchor_type = "not_applicable"
        spatial_anchor_observation = None
        layout = bundle.mechanism.cell_program.layout_for(case.primitive_id)
        if layout == "localized_density_gradient":
            depletion = bundle.mechanism.cell_program.cellularity_depletion
            if case.pathology_domain_id == "breast-invasive-carcinoma-v1":
                raw_anchor = case.provenance.get(
                    "cellularity_depletion_anchor"
                )
            else:
                if not isinstance(
                    compiler_owned_depletion_anchor,
                    CompilerOwnedDepletionAnchor,
                ):
                    raise JointContractError(
                        "non-Breast cellularity decrease requires a compiler-owned "
                        "depletion anchor capability"
                    )
                compiler_owned_depletion_anchor.validate(case=case)
                if compiler_owned_depletion_anchor.zone_id != zone.zone_id:
                    raise JointContractError(
                        "compiler-owned depletion anchor belongs to another population zone"
                    )
                raw_anchor = {
                    "type": compiler_owned_depletion_anchor.anchor_type,
                    "interface_ids": list(
                        compiler_owned_depletion_anchor.interface_ids
                    ),
                    "anchor_ids": list(
                        compiler_owned_depletion_anchor.anchor_ids
                    ),
                    "observation": (
                        "compiler-issued deterministic mask-only population peak"
                        if compiler_owned_depletion_anchor.anchor_type
                        == "population_peak"
                        else "compiler-issued deterministic mask-graph adjacency"
                    ),
                    "confidence": 1.0,
                }
            if depletion is None or not isinstance(raw_anchor, Mapping):
                raise JointContractError(
                    "cellularity decrease requires an explicit mask-graph depletion anchor"
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
            if spatial_anchor_type == "interface":
                interfaces = {
                    item.interface_id: item
                    for item in scene.tissue.graph.interfaces
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
            elif interface_ids or anchor_ids:
                raise JointContractError(
                    "population-peak depletion cannot claim interface anchors"
                )
            if (
                not spatial_anchor_observation
                or confidence < bundle.mechanism.recognition.minimum_confidence
            ):
                raise JointContractError(
                    "depletion anchor lacks a confident mask-graph certificate"
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
                "offline heuristic does not resolve natural-language ambiguity or rank certified options semantically",
            ),
            escalation_reason="requires_mask_graph_llm_planner_and_independent_condition_critic",
            structural_unit_ids=_structural_units_for_components(
                scene, (zone.tissue_component_id,)
            ),
            supporting_preference_rule_ids=(
                bundle.mechanism.planner_policy.selection_preferences
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
            {
                str(item["unit_id"])
                for item in scene.structural_hierarchy.get(
                    "structure_units", ()
                )
                if isinstance(item, Mapping)
                and item.get("unit_id")
                and item.get("parent_tissue_component_id") in selected
            }
        )
    )

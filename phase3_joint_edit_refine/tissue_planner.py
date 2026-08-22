"""Independent tissue Planner adapter that can bind multiple legal components."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from skimage.morphology import convex_hull_image

from phase3_mask_edit_refine.agents import (
    EDIT_PLAN_SCHEMA_VERSION,
    OpenAIResponsesJSONClient,
    validate_edit_plan,
)
from phase3_mask_edit_refine.evidence import load_id_mask
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
from .llm_audit_tokens import (
    TISSUE_ABSTAIN_TOKEN,
    TISSUE_SELECTION_TOKEN,
    isolate_provider_usage,
    require_optional_token,
    require_token,
)
from .models import JointContractError
from .planner_inputs import (
    MaskPlannerArtifactRegistry,
    validate_mask_planner_image_paths,
)
from .planner_policy import PREFERENCE_METRIC_CATALOG
from .portfolio_authority import (
    build_tissue_portfolio_authority_binding,
    canonical_metadata_sha256,
)
from .skills.repository import JointSkillBundle
from .tissue_tools import (
    JOINT_TOOL_FAMILY_TO_EXECUTOR,
    compile_tissue_tool_program,
    validate_tissue_plan_tool_binding,
)


def _expected_tissue_authority_binding_sha256(
    *,
    case: CaseContext,
    joint_case: Any,
    joint_bundle: JointSkillBundle,
    tissue_bundle: ActiveKnowledgeBundle,
    allocation: Any,
    nuclei_preflight: JointNucleiPreflight,
    candidate_portfolio: Any,
) -> str:
    binding = getattr(candidate_portfolio, "authority_binding", None)
    if not isinstance(binding, Mapping):
        raise RefineContractError(
            "tissue portfolio lacks its compiler-owned authority binding"
        )
    if joint_case is None or allocation is None:
        raise RefineContractError(
            "online tissue Planner requires current joint case and allocation authority"
        )
    expected = build_tissue_portfolio_authority_binding(
        joint_case=joint_case,
        tissue_case=case,
        source_tissue=load_id_mask(case.source_mask_uri),
        joint_bundle=joint_bundle,
        tissue_bundle=tissue_bundle,
        allocation=allocation,
        nuclei_preflight=nuclei_preflight,
    )
    if dict(binding) != expected:
        raise RefineContractError(
            "tissue portfolio authority is detached from current runtime inputs or skills"
        )
    return canonical_metadata_sha256(expected)


JOINT_TISSUE_DECISION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "abstain",
        "abstain_reason",
        "decision_id",
        "selected_candidate_id",
        "selected_tool_family",
        "supporting_preference_rule_ids",
        "selection_explanation",
        "confidence",
    ],
    "properties": {
        "abstain": {"type": "boolean"},
        "abstain_reason": {
            "type": ["string", "null"],
            "enum": [TISSUE_ABSTAIN_TOKEN, None],
        },
        "decision_id": {
            "type": ["string", "null"],
            "enum": ["select_certified_tissue_plan_candidate", None],
        },
        "selected_candidate_id": {"type": ["string", "null"]},
        "selected_tool_family": {"type": ["string", "null"]},
        "supporting_preference_rule_ids": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "uniqueItems": True,
        },
        "selection_explanation": {
            "type": ["string", "null"],
            "enum": [TISSUE_SELECTION_TOKEN, None],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
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


def _component_turnover_profile_mode(
    *,
    preferred_mode: str,
    allow_source_component_resolution: bool,
    requested_allocation_px: int,
    source_component_area_px: int,
) -> str:
    """Use an equal-depth front only when the source is actually resolved.

    ``allow_source_component_resolution`` is a topology permission, not a
    request to erase every selected component with a constant-width annulus.
    A partial boundary turnover must retain the mechanism-owned organic depth
    profile; otherwise a small necrosis-resolution budget becomes the exact
    parallel ring rejected by the downstream artifact gate.
    """

    resolves_component = (
        allow_source_component_resolution
        and source_component_area_px > 0
        and requested_allocation_px >= source_component_area_px
    )
    return "uniform_front" if resolves_component else preferred_mode


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
        "minimum_source_component_changed_fraction": (
            primitive.minimum_source_component_changed_fraction
        ),
        "allow_target_component_creation": (
            primitive.allow_target_component_creation
        ),
        "maximum_new_target_components": (
            primitive.maximum_new_target_components
        ),
        "maximum_source_component_changed_fraction": (
            primitive.maximum_source_component_changed_fraction
        ),
        "minimum_source_component_remaining_px": (
            primitive.minimum_source_component_remaining_px
        ),
        "maximum_selected_source_components": (
            primitive.maximum_selected_source_components
        ),
        "minimum_dominant_change_component_fraction": (
            primitive.minimum_dominant_change_component_fraction
        ),
        "allow_source_component_split": primitive.allow_source_component_split,
        "minimum_residual_components": primitive.minimum_residual_components,
        "maximum_residual_components": primitive.maximum_residual_components,
        "minimum_residual_component_area_px": (
            primitive.minimum_residual_component_area_px
        ),
        "minimum_residual_spacing_px": primitive.minimum_residual_spacing_px,
        "residual_area_floor_fraction": primitive.residual_area_floor_fraction,
        "maximum_residual_area_fraction": (
            primitive.maximum_residual_area_fraction
        ),
        "minimum_residual_component_fraction": (
            primitive.minimum_residual_component_fraction
        ),
        "maximum_dominant_residual_component_fraction": (
            primitive.maximum_dominant_residual_component_fraction
        ),
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
        joint_case: Any | None = None,
        allocation: Any | None = None,
        execution_feedback: Mapping[str, Any] | None = None,
        artifact_registry: MaskPlannerArtifactRegistry | None = None,
        candidate_portfolio: Sequence[Any] = (),
    ) -> tuple[EditPlan, dict[str, Any]]:
        image_paths = validate_mask_planner_image_paths(
            image_paths, case=case, artifact_registry=artifact_registry
        )
        if nuclei_preflight is None:
            raise RefineContractError(
                "joint-aware tissue Planner requires nuclei preflight"
            )
        portfolio_object = candidate_portfolio
        portfolio = tuple(portfolio_object)
        vetoed_portfolio = tuple(
            getattr(candidate_portfolio, "vetoed", ())
        )
        if not portfolio:
            raise RefineContractError(
                "online tissue Planner requires a pre-LLM compiler portfolio"
            )
        if not hasattr(portfolio_object, "validate_authority"):
            raise RefineContractError(
                "online tissue Planner requires a compiler-issued portfolio capability"
            )
        expected_binding = _expected_tissue_authority_binding_sha256(
            case=case,
            joint_case=joint_case,
            joint_bundle=joint_bundle,
            tissue_bundle=bundle,
            allocation=allocation,
            nuclei_preflight=nuclei_preflight,
            candidate_portfolio=portfolio_object,
        )
        portfolio_object.validate_authority(
            expected_binding_sha256=expected_binding
        )
        candidates_by_id = {item.candidate_id: item for item in portfolio}
        if len(candidates_by_id) != len(portfolio):
            raise RefineContractError("tissue portfolio candidate IDs are not unique")
        compiled_tools = compile_tissue_tool_program(
            primitive_id=portfolio[0].compiled_plan.primitive_id,
            mechanism_id=joint_bundle.mechanism.mechanism_id,
            mechanism_allowed_families=(
                joint_bundle.mechanism.tissue_program.allowed_tools
            ),
            primitive_allowed_executors=bundle.edit_contract.allowed_tools,
        )
        portfolio_metadata = []
        for item in portfolio:
            item.validate_identity()
            if (
                not item.allowed_tool_families
                or not set(item.allowed_tool_families).issubset(
                    compiled_tools.allowed_joint_families
                )
            ):
                raise RefineContractError(
                    "tissue candidate tool families are detached from the current mechanism"
                )
            if item.tool_program_sha256 != compiled_tools.program_sha256:
                raise RefineContractError(
                    "tissue candidate concrete program SHA is detached"
                )
            portfolio_metadata.append(item.to_metadata())
        payload = {
            "case": _mask_planner_case_metadata(case),
            "tissue_scene": scene.graph.to_metadata(),
            "tissue_skill_bundle": bundle.to_metadata(),
            "joint_skill_bundle": joint_bundle.to_metadata(),
            "joint_mechanism_contract": {
                "recognition": asdict(joint_bundle.mechanism.recognition),
                "representability": asdict(
                    joint_bundle.mechanism.representability
                ),
                "tissue_program": asdict(
                    joint_bundle.mechanism.tissue_program
                ),
                "coupling": asdict(joint_bundle.mechanism.coupling),
                "planner_policy": asdict(
                    joint_bundle.mechanism.planner_policy
                ),
            },
            "nuclei_preflight": nuclei_preflight.to_metadata(),
            "previous_execution_feedback": dict(execution_feedback or {}),
            "certified_tissue_plan_candidates": portfolio_metadata,
            "vetoed_tissue_plan_candidates": [
                item.to_metadata() for item in vetoed_portfolio
            ],
            "requirements": {
                "select_only_nuclei_feasible_interfaces": True,
                "select_real_anchor_segment_ids": True,
                "respect_immutable_area_budget": True,
                "select_only_listed_candidate_ids": True,
                "select_only_listed_tool_families": True,
                "cite_only_skill_preference_rule_ids": True,
                "do_not_output_pixels_polygons_coordinates_counts_density_or_area": True,
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
                    "the skill's selection preferences, then select only one listed "
                    "certified candidate ID and one listed tool family. The compiler, not "
                    "you, owns interfaces, anchors, pixels, depth, area, cell counts and "
                    "all numeric parameters. Raw H&E is prohibited and cannot be "
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
            provider_usage = isolate_provider_usage(usage)
            if raw.get("abstain") is True:
                require_optional_token(
                    raw.get("abstain_reason"),
                    expected=TISSUE_ABSTAIN_TOKEN,
                    field="abstain_reason",
                )
                raise RefineContractError(
                    "joint-aware tissue Planner abstained: "
                    + TISSUE_ABSTAIN_TOKEN
                )
            try:
                if raw.get("decision_id") != "select_certified_tissue_plan_candidate":
                    raise RefineContractError(
                        "LLM returned an illegal tissue planning decision"
                    )
                if raw["decision_id"] not in set(
                    joint_bundle.mechanism.planner_policy.allowed_decisions
                ):
                    raise RefineContractError(
                        "tissue planning decision is outside the skill policy"
                    )
                candidate_id = raw.get("selected_candidate_id")
                if candidate_id not in candidates_by_id:
                    raise RefineContractError(
                        "LLM selected an unknown or vetoed tissue plan candidate"
                    )
                selected_witness = candidates_by_id[candidate_id]
                compiled_plan = selected_witness.compiled_plan
                candidate_metrics = selected_witness.deterministic_candidate_metrics
                selected_family = raw.get("selected_tool_family")
                if selected_family not in selected_witness.allowed_tool_families:
                    raise RefineContractError(
                        "LLM selected a tissue tool family outside the compiled mechanism program"
                    )
                preference_ids = raw.get("supporting_preference_rule_ids")
                if (
                    not isinstance(preference_ids, list)
                    or not preference_ids
                    or set(preference_ids)
                    - set(joint_bundle.mechanism.planner_policy.selection_preferences)
                ):
                    raise RefineContractError(
                        "LLM cited an unknown tissue-selection preference rule"
                    )
                missing_metrics = {
                    PREFERENCE_METRIC_CATALOG[rule_id][0]
                    for rule_id in preference_ids
                } - set(candidate_metrics)
                if missing_metrics:
                    raise RefineContractError(
                        "LLM cited a preference without a certified metric: "
                        + ", ".join(sorted(missing_metrics))
                    )
                selected_executor = JOINT_TOOL_FAMILY_TO_EXECUTOR[selected_family]
                if selected_executor not in compiled_plan.tool_program.allowed_tools:
                    raise RefineContractError(
                        "selected tool family is detached from the certified plan"
                    )
                plan = replace(
                    compiled_plan,
                    tool_program=replace(
                        compiled_plan.tool_program,
                        allowed_tools=(selected_executor,),
                        parameter_ranges={
                            **compiled_plan.tool_program.parameter_ranges,
                            "joint_tissue_tool_program": (
                                compiled_tools.to_metadata()
                            ),
                            "planner_selection_certificate": {
                                "decision_id": raw["decision_id"],
                                "selected_candidate_id": candidate_id,
                                "selected_tool_family": selected_family,
                                "supporting_preference_rule_ids": list(
                                    preference_ids
                                ),
                                "selection_explanation": raw.get(
                                    "selection_explanation"
                                ),
                                "confidence_audit_only": raw.get("confidence"),
                                "compiler_certificate_sha256": (
                                    selected_witness.compiler_certificate_sha256
                                ),
                                "authority_binding_sha256": (
                                    selected_witness.authority_binding_sha256
                                ),
                                "execution_raster_sha256": (
                                    selected_witness.execution_raster_sha256
                                ),
                                "tissue_gate_report_sha256": (
                                    selected_witness.tissue_gate_report_sha256
                                ),
                            },
                        },
                    ),
                )
                if plan.normalized_intent != case.instruction:
                    raise RefineContractError(
                        "joint tissue Planner modified the parser-owned intent"
                    )
                require_token(
                    raw.get("selection_explanation"),
                    expected=TISSUE_SELECTION_TOKEN,
                    field="selection_explanation",
                )
                validate_edit_plan(
                    plan,
                    case=case,
                    scene=scene,
                    bundle=bundle,
                )
                self._validate_joint_binding(
                    plan=plan,
                    tissue_bundle=bundle,
                    joint_bundle=joint_bundle,
                    nuclei_preflight=nuclei_preflight,
                    scene=scene,
                )
            except (JointContractError, RefineContractError, TypeError, ValueError) as exc:
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                continue
            return plan, {
                "provider": self.name,
                "contract_attempt": attempt,
                "escalated": client is self.escalation_client,
                    "compiler": selected_witness.to_metadata(),
                    "portfolio_candidate_count": len(portfolio),
                    "ranking_mode": (
                        "rank_surviving_candidates"
                        if len(portfolio) > 1
                        else "single_candidate_accept_or_abstain"
                    ),
                "selected_candidate_id": candidate_id,
                "selected_tool_family": selected_family,
                "supporting_preference_rule_ids": list(preference_ids),
                "provider_usage": provider_usage,
            }
        raise RefineContractError(
            "joint-aware tissue Planner exhausted contract attempts: "
            + "; ".join(errors)
        )

    @staticmethod
    def _validate_joint_binding(
        *,
        plan: EditPlan,
        tissue_bundle: ActiveKnowledgeBundle,
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
        compiled_tools = compile_tissue_tool_program(
            primitive_id=plan.primitive_id,
            mechanism_id=joint_bundle.mechanism.mechanism_id,
            mechanism_allowed_families=(
                joint_bundle.mechanism.tissue_program.allowed_tools
            ),
            primitive_allowed_executors=(
                tissue_bundle.edit_contract.allowed_tools
            ),
        )
        # The actual primitive contract is carried by the caller's mask bundle;
        # a detached or fabricated compiled program still fails here.
        validate_tissue_plan_tool_binding(plan, compiled=compiled_tools)
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
        # Planner confidence is retained for audit/calibration only. Interface,
        # anchor, tool and capacity legality are established above from
        # deterministic certificates and cannot be granted or revoked by the
        # model's self-reported confidence.


def _source_component_capacity_limits(
    interfaces: Sequence[Any],
    *,
    capacity_by_id: Mapping[str, int],
    nuclei_preflight: JointNucleiPreflight | None,
) -> dict[str, int]:
    """Return one certified capacity per source component, never per edge."""

    by_source: dict[str, list[Any]] = {}
    for item in interfaces:
        by_source.setdefault(item.source_component_id, []).append(item)
    certified = (
        nuclei_preflight.feasible_tissue_capacity_by_source_component
        if nuclei_preflight is not None
        else {}
    )
    limits: dict[str, int] = {}
    for component_id, items in by_source.items():
        raw_sum = sum(capacity_by_id[item.interface_id] for item in items)
        if component_id in certified:
            limits[component_id] = min(
                raw_sum, int(certified[component_id])
            )
            continue
        reports = [
            nuclei_preflight.interface(item.interface_id)
            for item in items
        ] if nuclei_preflight is not None else []
        reported_limits = [
            int(report.source_component_capacity_pixels)
            for report in reports
            if report is not None
        ]
        limits[component_id] = min(
            raw_sum,
            min(reported_limits) if reported_limits else raw_sum,
        )
    return limits


def _rank_interfaces_by_marginal_capacity(
    interfaces: Sequence[Any],
    *,
    capacity_by_id: Mapping[str, int],
    component_capacity_limits: Mapping[str, int],
    locked_interface_ids: Sequence[str] = (),
    failed_interface_ids: set[str] | frozenset[str] = frozenset(),
    previous_actual_by_interface: Mapping[str, int] | None = None,
    previous_actual_by_source: Mapping[str, int] | None = None,
) -> tuple[list[Any], dict[str, int]]:
    """Bounded greedy order using source-component-capped marginal capacity."""

    by_id = {item.interface_id: item for item in interfaces}
    ordered: list[Any] = []
    credited: dict[str, int] = {}
    used_by_source: dict[str, int] = {}
    actual_by_interface = dict(previous_actual_by_interface or {})
    for interface_id in dict.fromkeys(str(value) for value in locked_interface_ids):
        item = by_id.get(interface_id)
        if item is None:
            continue
        component_id = item.source_component_id
        remaining = max(
            0,
            int(component_capacity_limits.get(component_id, 0))
            - used_by_source.get(component_id, 0),
        )
        credit = min(
            int(capacity_by_id.get(interface_id, 0)),
            max(0, int(actual_by_interface.get(interface_id, remaining))),
            remaining,
        )
        credited[interface_id] = credit
        used_by_source[component_id] = (
            used_by_source.get(component_id, 0) + credit
        )
        ordered.append(item)
    for component_id, actual in dict(previous_actual_by_source or {}).items():
        used_by_source[str(component_id)] = min(
            int(component_capacity_limits.get(str(component_id), 0)),
            max(used_by_source.get(str(component_id), 0), int(actual)),
        )

    remaining_items = [
        item for item in interfaces if item.interface_id not in credited
    ]
    while remaining_items:
        def marginal(item: Any) -> int:
            component_id = item.source_component_id
            return min(
                int(capacity_by_id.get(item.interface_id, 0)),
                max(
                    0,
                    int(component_capacity_limits.get(component_id, 0))
                    - used_by_source.get(component_id, 0),
                ),
            )

        item = min(
            remaining_items,
            key=lambda candidate: (
                candidate.interface_id in failed_interface_ids,
                marginal(candidate) <= 0,
                -marginal(candidate),
                -int(capacity_by_id.get(candidate.interface_id, 0)),
                -candidate.contact_pixels,
                candidate.interface_id,
            ),
        )
        credit = marginal(item)
        credited[item.interface_id] = credit
        used_by_source[item.source_component_id] = (
            used_by_source.get(item.source_component_id, 0) + credit
        )
        ordered.append(item)
        remaining_items.remove(item)
    return ordered, credited


def _component_capped_allocation_capacities(
    interfaces: Sequence[Any],
    *,
    capacity_by_id: Mapping[str, int],
    component_capacity_limits: Mapping[str, int],
) -> list[int]:
    """Distribute each source component's unique capacity across its fronts."""

    credits = [0 for _ in interfaces]
    indices_by_source: dict[str, list[int]] = {}
    for index, item in enumerate(interfaces):
        indices_by_source.setdefault(item.source_component_id, []).append(index)
    for component_id, indices in indices_by_source.items():
        raw = np.asarray(
            [capacity_by_id[interfaces[index].interface_id] for index in indices],
            dtype=float,
        )
        total = min(
            int(raw.sum()), int(component_capacity_limits[component_id])
        )
        if total <= 0 or float(raw.sum()) <= 0:
            continue
        exact = raw * (total / float(raw.sum()))
        allocated = np.floor(exact).astype(int)
        remainder = total - int(allocated.sum())
        order = sorted(
            range(len(indices)),
            key=lambda local: (
                -(exact[local] - allocated[local]),
                interfaces[indices[local]].interface_id,
            ),
        )
        for local in order[:remainder]:
            allocated[local] += 1
        for local, index in enumerate(indices):
            credits[index] = int(allocated[local])
    return credits


@dataclass(frozen=True)
class MultiInterfaceResearchTissuePlanner:
    """Use all needed legal source components; no H&E authority is claimed."""

    name: str = "multi_interface_research_tissue_planner_v3"

    def create_joint_tissue_plan(
        self,
        *,
        case: CaseContext,
        scene: SceneAnalysis,
        bundle: ActiveKnowledgeBundle,
        joint_bundle: JointSkillBundle,
        image_paths: Sequence[str | Path],
        nuclei_preflight: JointNucleiPreflight | None = None,
        joint_case: Any | None = None,
        allocation: Any | None = None,
        execution_feedback: Mapping[str, Any] | None = None,
        artifact_registry: MaskPlannerArtifactRegistry | None = None,
        candidate_portfolio: Sequence[Any] = (),
    ) -> tuple[EditPlan, dict[str, Any]]:
        del image_paths, artifact_registry
        portfolio = tuple(candidate_portfolio)
        if portfolio:
            if not hasattr(candidate_portfolio, "validate_authority"):
                raise RefineContractError(
                    "research tissue Planner requires a compiler-issued portfolio"
                )
            expected_binding = _expected_tissue_authority_binding_sha256(
                case=case,
                joint_case=joint_case,
                joint_bundle=joint_bundle,
                tissue_bundle=bundle,
                allocation=allocation,
                nuclei_preflight=nuclei_preflight,
                candidate_portfolio=candidate_portfolio,
            )
            candidate_portfolio.validate_authority(
                expected_binding_sha256=expected_binding
            )

            def preference_key(witness):
                values = []
                metrics = witness.deterministic_candidate_metrics
                for rule_id in joint_bundle.mechanism.planner_policy.selection_preferences:
                    metric_id, direction = PREFERENCE_METRIC_CATALOG[rule_id]
                    value = float(metrics.get(metric_id, 0.0))
                    values.append(value if direction == "max" else -value)
                return (*values, witness.candidate_id)

            selected_witness = max(portfolio, key=preference_key)
            selected_witness.validate_identity()
            selected_family = selected_witness.allowed_tool_families[0]
            selected_executor = JOINT_TOOL_FAMILY_TO_EXECUTOR[selected_family]
            compiled_plan = selected_witness.compiled_plan
            plan = replace(
                compiled_plan,
                tool_program=replace(
                    compiled_plan.tool_program,
                    allowed_tools=(selected_executor,),
                    parameter_ranges={
                        **compiled_plan.tool_program.parameter_ranges,
                        "planner_selection_certificate": {
                            "decision_id": "select_certified_tissue_plan_candidate",
                            "selected_candidate_id": selected_witness.candidate_id,
                            "selected_tool_family": selected_family,
                            "supporting_preference_rule_ids": list(
                                joint_bundle.mechanism.planner_policy.selection_preferences
                            ),
                            "compiler_certificate_sha256": (
                                selected_witness.compiler_certificate_sha256
                            ),
                            "authority_binding_sha256": (
                                selected_witness.authority_binding_sha256
                            ),
                            "execution_raster_sha256": (
                                selected_witness.execution_raster_sha256
                            ),
                            "tissue_gate_report_sha256": (
                                selected_witness.tissue_gate_report_sha256
                            ),
                        },
                    },
                ),
            )
            validate_edit_plan(plan, case=case, scene=scene, bundle=bundle)
            return plan, {
                "provider": self.name,
                "selection_mode": "compiler_portfolio_preference_ranking",
                "selected_candidate_id": selected_witness.candidate_id,
                "selected_tool_family": selected_family,
                "portfolio_candidate_count": len(portfolio),
                "previous_execution_feedback": dict(execution_feedback or {}),
            }
        del joint_case, allocation
        mechanism_contract = joint_bundle.mechanism.tissue_program.primitive_label_contracts.get(case.primitive_id)
        if mechanism_contract is None:
            raise RefineContractError("joint mechanism has no primitive label contract")
        allowed_sources = set(bundle.edit_contract.source_label_options).intersection(mechanism_contract["source_labels"])
        if bundle.edit_contract.target_label not in mechanism_contract["target_labels"]:
            raise RefineContractError("annotation-resolved target is illegal for the joint mechanism")
        compiled_tools = compile_tissue_tool_program(
            primitive_id=case.primitive_id,
            mechanism_id=joint_bundle.mechanism.mechanism_id,
            mechanism_allowed_families=(
                joint_bundle.mechanism.tissue_program.allowed_tools
            ),
            primitive_allowed_executors=bundle.edit_contract.allowed_tools,
        )
        directional_projection = (
            compiled_tools.allowed_concrete_executors
            == ("directional_tapered_projection",)
        )
        front_contract = joint_bundle.mechanism.tissue_program.front
        reference_equivalent_diameter = float(
            np.sqrt(
                4.0
                * max(1.0, float(nuclei_preflight.reference_area_p95))
                / np.pi
            )
        )
        if case.annotation_profile_id == "panda-gleason-v1":
            # A Pattern-5 cord must remain narrow, but it must still admit a
            # complete source-calibrated neoplastic nucleus.  PANDA nuclei are
            # substantially larger in raster pixels than the historical
            # Breast fixtures, so a fixed 24-pixel cap made valid cell
            # execution impossible.
            directional_maximum_width_px = (
                28.0
                if reference_equivalent_diameter >= 16.0
                else float(
                    np.clip(
                        0.75 * reference_equivalent_diameter,
                        8.0,
                        28.0,
                    )
                )
            )
            directional_tip_width_px = float(
                np.clip(0.12 * reference_equivalent_diameter, 2.0, 5.0)
            )
        elif (
            case.annotation_profile_id == "ignite-semantic-v1"
            and case.primitive_id == "infiltrative-nest-cord-extension-v1"
        ):
            # The IGNITE cord must be wide enough for one complete native
            # neoplastic nucleus at the seam.  The historical 24 px cap was
            # narrower than the observed source-calibrated footprints and
            # produced a tissue-only cord that cell execution could not fill.
            directional_maximum_width_px = float(
                np.clip(1.5 * reference_equivalent_diameter, 12.0, 40.0)
            )
            directional_tip_width_px = float(
                np.clip(0.12 * reference_equivalent_diameter, 2.0, 5.0)
            )
        else:
            directional_maximum_width_px = float(
                np.clip(4.0 * reference_equivalent_diameter, 8.0, 24.0)
            )
            directional_tip_width_px = 2.0
        legal = [
            item for item in scene.graph.interfaces
            if item.source_label in allowed_sources and item.target_label == bundle.edit_contract.target_label
        ]
        if nuclei_preflight is not None:
            feasible_ids = set(nuclei_preflight.feasible_interface_ids)
            legal = [item for item in legal if item.interface_id in feasible_ids]
        if front_contract.directional_sector_required:
            legal = [
                item
                for item in legal
                if _directional_sector_selection_limit(
                    interface_anchor_ids=item.anchor_segment_ids,
                    allowed_anchor_ids=(
                        nuclei_preflight.interface(
                            item.interface_id
                        ).cell_feasible_anchor_segment_ids
                        if nuclei_preflight is not None
                        and nuclei_preflight.interface(item.interface_id)
                        is not None
                        else ()
                    ),
                    maximum_selected_anchor_fraction=(
                        front_contract.maximum_selected_anchor_fraction
                    ),
                    minimum_unselected_anchor_count=(
                        front_contract.minimum_unselected_anchor_count
                    ),
                )
                >= 1
            ]
        if not legal:
            raise RefineContractError(
                "no directed interface satisfies both the tissue and preflight nuclei contracts"
            )
        feedback = dict(execution_feedback or {})
        retry_index = max(0, int(feedback.get("retry_index", 0)))
        failed_interface_ids = set(feedback.get("failed_interface_ids", ()))
        locked_interface_ids = tuple(
            str(value) for value in feedback.get("selected_interface_ids", ())
        )
        preferred_anchor_ids = tuple(
            str(value) for value in feedback.get("preferred_anchor_ids", ())
        )
        feedback_stage = str(feedback.get("stage", ""))
        area_underfill_replan = feedback_stage == "tissue_area_underfill"
        topology = _effective_tissue_topology(
            joint_bundle,
            primitive_id=case.primitive_id,
            retry_index=retry_index,
            feedback_stage=feedback_stage,
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
            if preflight_item is not None:
                capacity = int(
                    preflight_item.editable_tissue_capacity_pixels
                )
            else:
                interface_mask = scene.interface_masks[item.interface_id]
                source_component = scene.component_masks[
                    item.source_component_id
                ]
                distance = ndimage.distance_transform_edt(~interface_mask)
                depth_cap = max(1, min(128, int(item.contact_pixels)))
                capacity = int(
                    np.count_nonzero(
                        source_component & (distance <= depth_cap)
                    )
                )
            capacity_by_id[item.interface_id] = capacity
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
        component_capacity_limits = _source_component_capacity_limits(
            legal,
            capacity_by_id=capacity_by_id,
            nuclei_preflight=nuclei_preflight,
        )
        labels = sorted({item.source_label for item in legal})
        source_label = max(
            labels,
            key=lambda label: (
                any(
                    item.interface_id in locked_interface_ids
                    for item in legal
                    if item.source_label == label
                ),
                sum(
                    capacity
                    for component_id, capacity in component_capacity_limits.items()
                    if any(
                        item.source_component_id == component_id
                        and item.source_label == label
                        for item in legal
                    )
                ),
                sum(
                    item.contact_pixels
                    for item in legal
                    if item.source_label == label
                ),
                label,
            ),
        )
        source_legal = [
            item for item in legal if item.source_label == source_label
        ]
        ranked, marginal_capacity_by_id = (
            _rank_interfaces_by_marginal_capacity(
                source_legal,
                capacity_by_id=capacity_by_id,
                component_capacity_limits=component_capacity_limits,
                locked_interface_ids=locked_interface_ids,
                failed_interface_ids=failed_interface_ids,
                previous_actual_by_interface=feedback.get(
                    "interface_actual_contribution_pixels", {}
                ),
                previous_actual_by_source=feedback.get(
                    "actual_contribution_by_source_component", {}
                ),
            )
        )
        if topology["maximum_selected_source_components"] == 1 and ranked:
            # Coherent retreat and fragmentation are both defined within one
            # pre-existing invasive-tumor component. Selecting unrelated
            # components makes footprint regression look like scattered edge
            # nibbles and can satisfy a fragmentation count without actually
            # fragmenting one biological focus.
            capacity_by_source: dict[str, int] = {}
            for item in ranked:
                capacity_by_source[item.source_component_id] = int(
                    component_capacity_limits[item.source_component_id]
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
        # An area-underfill retry must expose every legal, cell-feasible front
        # to the compiler. Its disjoint pixel ownership, source-retention cap,
        # and whole-mask topology audit then decide the global safe fallback.
        # Other retries retain gradual diversification.
        extra_after_capacity = (
            len(ranked)
            if component_turnover or area_underfill_replan
            else min(8, retry_index * 4)
        )
        capacity_reached_at: int | None = None
        for item in ranked[:32]:
            capacity = capacity_by_id[item.interface_id]
            if capacity <= 0:
                continue
            selected.append(item)
            capacities.append(capacity)
            capacity_credit = int(
                marginal_capacity_by_id.get(item.interface_id, 0)
            )
            cumulative += capacity_credit
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
        allocation_capacities = _component_capped_allocation_capacities(
            selected,
            capacity_by_id=capacity_by_id,
            component_capacity_limits=component_capacity_limits,
        )
        total_capacity = max(1, sum(allocation_capacities))
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
            planning_target_pixels = min(target_pixels, total_capacity)
            component_allocations = [
                min(
                    capacity,
                    max(
                        1,
                        round(
                            planning_target_pixels
                            * allocation_capacity
                            / total_capacity
                        ),
                    ),
                )
                for capacity, allocation_capacity in zip(
                    capacities, allocation_capacities
                )
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
                    minimum_selected_anchor_count=(
                        front_contract.minimum_selected_anchor_count
                    ),
                    prefer_shallow_front=(
                        case.primitive_id
                        == "invasive-tumor-footprint-decrease-v1"
                    ),
                    preferred_anchor_ids=preferred_anchor_ids,
                    expand_to_selection_limit=area_underfill_replan,
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
            if directional_projection:
                # The special executor binds one compact attachment within the
                # certified anchor and therefore cannot use full-anchor contact
                # as its area denominator.  Solve the trapezoidal projection
                # depth from the requested area and the source-calibrated width.
                estimated_directional_depth = (
                    2.0
                    * requested_allocation
                    / max(
                        1.0,
                        directional_maximum_width_px
                        + directional_tip_width_px,
                    )
                )
                peak = float(
                    np.clip(
                        np.ceil(estimated_directional_depth * 1.18),
                        4,
                        depth_cap,
                    )
                )
            else:
                peak = float(
                    np.clip(np.ceil(estimated_depth * 2.0), 2, depth_cap)
                )
            requested_mode = _component_turnover_profile_mode(
                preferred_mode=front_contract.profile_mode,
                allow_source_component_resolution=bool(
                    topology["allow_source_component_resolution"]
                ),
                requested_allocation_px=int(requested_allocation),
                source_component_area_px=int(
                    np.count_nonzero(
                        scene.component_masks[interface.source_component_id]
                    )
                ),
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
                        min_anchor_coverage_fraction=(
                            0.03 if directional_projection else 0.50
                        ),
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
                allowed_tools=compiled_tools.allowed_concrete_executors,
                parameter_ranges={
                    "joint_tissue_tool_program": compiled_tools.to_metadata(),
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
                    "min_source_component_changed_fraction": (
                        topology["minimum_source_component_changed_fraction"]
                    ),
                    "min_source_component_remaining_px": (
                        topology["minimum_source_component_remaining_px"]
                    ),
                    "maximum_selected_source_components": (
                        topology["maximum_selected_source_components"]
                    ),
                    "minimum_dominant_change_component_fraction": (
                        topology[
                            "minimum_dominant_change_component_fraction"
                        ]
                    ),
                    "allow_source_component_resolution": (
                        topology["allow_source_component_resolution"]
                    ),
                    "allow_target_hole_resolution": (
                        topology["allow_target_hole_resolution"]
                    ),
                    "allow_target_component_creation": (
                        topology["allow_target_component_creation"]
                    ),
                    "maximum_new_target_components": (
                        topology["maximum_new_target_components"]
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
                    "maximum_residual_area_fraction": (
                        topology["maximum_residual_area_fraction"]
                    ),
                    "minimum_residual_component_fraction": (
                        topology["minimum_residual_component_fraction"]
                    ),
                    "maximum_dominant_residual_component_fraction": (
                        topology[
                            "maximum_dominant_residual_component_fraction"
                        ]
                    ),
                    "target_component_merge_policy": (
                        joint_bundle.mechanism.tissue_program.target_component_merge_policy
                    ),
                    "tissue_geometry_mode": (
                        "annotation_anchored_narrow_connected_extension"
                        if directional_projection
                        else topology["geometry_mode"]
                    ),
                    **(
                        {
                            "fragmentation_full_selected_component_support": True
                        }
                        if case.annotation_profile_id == "panda-gleason-v1"
                        and case.primitive_id
                        == "residual-tumor-fragmentation-v1"
                        else {}
                    ),
                    "directional_maximum_width_px": (
                        directional_maximum_width_px
                        if directional_projection
                        else None
                    ),
                    "directional_tip_width_px": (
                        directional_tip_width_px
                        if directional_projection
                        else None
                    ),
                    "directional_centerline_first": bool(
                        directional_projection
                        and case.annotation_profile_id == "ignite-semantic-v1"
                        and case.primitive_id
                        == "infiltrative-nest-cord-extension-v1"
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
                # Ordinary primitives retain four tissue variants. Residual
                # fragmentation already performs its multi-axis geometry
                # search inside the executor, so it emits one tissue witness;
                # the joint stage still realizes three mature cell layouts.
                candidate_count=_tissue_geometry_candidate_count(
                    residual_fragmentation=residual_fragmentation
                ),
            ),
            hard_invariants=tuple(sorted(set(bundle.edit_contract.required_check_ids))),
            uncertainties=("current Codex session supplied the mechanism; this deterministic adapter only compiled certified mask geometry",),
            planner_confidence=0.45,
            escalation_reason="requires_independent_mask_condition_critic",
        )
        return plan, {
            "provider": self.name,
            "selected_interface_count": len(planned),
            "selected_interface_ids": [
                item.interface_id for item in selected
            ],
            "estimated_capacity_pixels": total_capacity,
            "source_component_capped_capacity_pixels": total_capacity,
            "selection_marginal_capacity_pixels": {
                item.interface_id: int(
                    marginal_capacity_by_id.get(item.interface_id, 0)
                )
                for item in selected
            },
            "source_component_capacity_limits": {
                component_id: int(capacity)
                for component_id, capacity in sorted(
                    component_capacity_limits.items()
                )
                if any(
                    item.source_component_id == component_id
                    for item in selected
                )
            },
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
    minimum_selected_anchor_count: int = 1,
    prefer_shallow_front: bool = False,
    preferred_anchor_ids: tuple[str, ...] = (),
    expand_to_selection_limit: bool = False,
) -> tuple[str, ...]:
    """Choose the shortest broad anchor group with enough legal capacity.

    Using every addressable anchor on a long winding boundary turns a modest
    area request into a very thin high-perimeter ribbon.  This deterministic
    compiler keeps adding spatially adjacent anchors until both capacity and a
    preferred shallow depth/span envelope are available.
    """

    source_component = scene.component_masks[interface.source_component_id]
    convex_hull_depth = (
        ndimage.distance_transform_edt(convex_hull_image(source_component))
        if prefer_shallow_front
        else np.zeros_like(source_component, dtype=float)
    )
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
        neighborhood_radius = float(
            np.clip(np.sqrt(contact) * 2.5, 18.0, 48.0)
        )
        neighborhood = distance <= neighborhood_radius
        source_enclosure_fraction = float(
            np.count_nonzero(source_component & neighborhood)
            / max(1, np.count_nonzero(neighborhood))
        )
        # A locally half-exposed boundary can still be the bottom of a deep
        # re-entrant cleft between two tumor lobes.  Growing there widens the
        # cleft into an apparent internal excavation, and the topology guard
        # may then leave a conspicuous straight tumor seam across the change.
        # Distance to the component's convex-hull boundary separates those
        # deep concavities from a genuinely exposed footprint front.
        hull_concavity_depth = float(
            np.mean(convex_hull_depth[np.asarray(anchor, dtype=bool)])
        )
        metadata = anchor_metadata.get(anchor_id)
        centroid = (
            metadata.centroid_xy
            if metadata is not None
            else tuple(reversed(ndimage.center_of_mass(anchor)))
        )
        records.append(
            (
                anchor_id,
                capacity,
                contact,
                centroid,
                source_enclosure_fraction,
                hull_concavity_depth,
            )
        )
    if not records:
        return ()
    selection_limit = _directional_sector_selection_limit(
        interface_anchor_ids=interface.anchor_segment_ids,
        allowed_anchor_ids=allowed_anchor_ids,
        maximum_selected_anchor_fraction=maximum_selected_anchor_fraction,
        minimum_unselected_anchor_count=minimum_unselected_anchor_count,
    )
    required_anchor_count = max(1, int(minimum_selected_anchor_count))
    if selection_limit < required_anchor_count:
        return ()
    preferred = set(preferred_anchor_ids)
    records.sort(
        key=lambda item: (
            (
                item[5],
                item[4],
                item[1] / max(item[2], 1),
                item[0] not in preferred,
                -item[2],
                item[0],
            )
            if prefer_shallow_front
            else (
                item[0] not in preferred,
                -item[1],
                -item[2],
                item[0],
            )
        )
    )
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
        if (
            len(selected) >= required_anchor_count
            and not expand_to_selection_limit
            and capacity >= required_pixels
            and contact >= preferred_minimum_span
        ):
            break
        path_distance = _interface_geodesic_distance(
            scene.interface_masks[interface.interface_id],
            union,
        )
        selected_centroids = np.asarray([item[3] for item in selected], dtype=float)
        next_index = min(
            range(len(records)),
            key=lambda index: (
                _minimum_mask_distance(
                    path_distance,
                    scene.anchor_masks[records[index][0]],
                ),
                records[index][5] if prefer_shallow_front else 0.0,
                records[index][4] if prefer_shallow_front else 0.0,
                (
                    records[index][1] / max(records[index][2], 1)
                    if prefer_shallow_front
                    else -records[index][1]
                ),
                records[index][0] not in preferred,
                float(
                    np.min(
                        np.linalg.norm(
                            selected_centroids
                            - np.asarray(records[index][3], dtype=float),
                            axis=1,
                        )
                    )
                ),
                records[index][0],
            ),
        )
        selected.append(records.pop(next_index))
    return tuple(sorted(item[0] for item in selected))


def _interface_geodesic_distance(
    interface_mask: np.ndarray,
    seed_mask: np.ndarray,
) -> np.ndarray:
    """Return 8-connected path steps along one directed interface raster.

    Euclidean centroid distance can jump across a thin stromal cleft and pick
    the opposing tumor bank even though it is far away along the biological
    interface.  Restricting the expansion path to the interface keeps a broad
    retreat on one continuous boundary sector.
    """

    interface = np.asarray(interface_mask, dtype=bool)
    seeds = interface & np.asarray(seed_mask, dtype=bool)
    distance = np.full(interface.shape, np.inf, dtype=float)
    queue: deque[tuple[int, int]] = deque()
    for row, col in np.argwhere(seeds):
        distance[int(row), int(col)] = 0.0
        queue.append((int(row), int(col)))
    offsets = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    height, width = interface.shape
    while queue:
        row, col = queue.popleft()
        next_distance = distance[row, col] + 1.0
        for row_offset, col_offset in offsets:
            next_row, next_col = row + row_offset, col + col_offset
            if not (
                0 <= next_row < height
                and 0 <= next_col < width
                and interface[next_row, next_col]
                and not np.isfinite(distance[next_row, next_col])
            ):
                continue
            distance[next_row, next_col] = next_distance
            queue.append((next_row, next_col))
    return distance


def _minimum_mask_distance(distance: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(distance, dtype=float)[np.asarray(mask, dtype=bool)]
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if finite.size else float("inf")


def _directional_sector_selection_limit(
    *,
    interface_anchor_ids: tuple[str, ...],
    allowed_anchor_ids: tuple[str, ...] = (),
    maximum_selected_anchor_fraction: float,
    minimum_unselected_anchor_count: int,
) -> int:
    """Return the executable anchor count before admitting an interface.

    Global underfill replans must not add an interface whose complete anchor
    inventory cannot leave the mechanism-owned unedited boundary witness.
    Filtering it after allocation aborts an otherwise valid maximum-safe
    fallback and incorrectly turns a local zero-capacity front into a global
    failure.
    """

    interface_ids = tuple(interface_anchor_ids)
    total_anchor_count = len(interface_ids)
    allowed = set(allowed_anchor_ids)
    selectable_count = sum(
        not allowed or anchor_id in allowed for anchor_id in interface_ids
    )
    return max(
        0,
        min(
            selectable_count,
            int(
                np.floor(
                    total_anchor_count * maximum_selected_anchor_fraction
                )
            ),
            total_anchor_count - minimum_unselected_anchor_count,
        ),
    )


def _tissue_geometry_candidate_count(*, residual_fragmentation: bool) -> int:
    """Return independent outer tissue variants after mechanism search."""

    # The fragmentation executor already scores PCA major/minor, cardinal and
    # diagonal traversing axes inside each candidate and retains the best legal
    # corridor set.  Additional outer depth-profile variants repeat that
    # expensive whole-mask topology solve and, on large tumors, preferentially
    # leave raster microfoci.  One best tissue witness still produces three
    # independent mature cell-layout candidates at the joint stage.
    return 1 if residual_fragmentation else 4

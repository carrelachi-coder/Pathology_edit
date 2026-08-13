"""Strict mask-graph Planner/Critic adapters for the joint pipeline."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from phase3_mask_edit_refine.agents import OpenAIResponsesJSONClient
from phase3_mask_edit_refine.evidence import load_id_mask
from phase3_mask_edit_refine.models import EditPlan

from .clarification import PlannerClarificationRequired
from .feasibility import classify_tumor_stroma_boundary
from .llm_audit_tokens import (
    CELL_ABSTAIN_TOKEN,
    CELL_SELECTION_TOKEN,
    CRITIC_ABSTAIN_SUMMARY_TOKEN,
    CRITIC_SUMMARY_TOKEN,
    CRITIC_VETO_TOKEN,
    JOINT_ABSTAIN_TOKEN,
    JOINT_ANCHOR_TOKEN,
    JOINT_EXPECTATION_TOKEN,
    JOINT_OBSERVATION_TOKEN,
    SEMANTIC_ABSTAIN_TOKEN,
    SEMANTIC_CLARIFICATION_TOKEN,
    SEMANTIC_OBSERVATION_TOKEN,
    SEMANTIC_SELECTION_TOKEN,
    isolate_provider_usage,
    require_exact_tokens,
    require_optional_token,
    require_token,
    require_token_subset,
)
from .models import (
    CellEditPlan,
    CouplingPlan,
    JointCaseContext,
    JointContractError,
    JointCriticRanking,
    JointCriticResult,
    JointEditPlan,
)
from .nuclei import load_nuclei_mask
from .planner import (
    JOINT_PLAN_SCHEMA_VERSION,
    LOCAL_POPULATION_PRIMITIVES,
    CellPlanSelectionHandle,
    CertifiedCellPlanCandidate,
    JointInterpretationOption,
    validate_cell_plan_candidate,
)
from .planner_inputs import (
    MaskPlannerArtifactRegistry,
    validate_mask_planner_image_paths,
)
from .planner_policy import PREFERENCE_METRIC_CATALOG
from .portfolio_authority import (
    build_cell_portfolio_authority_binding,
    canonical_metadata_sha256,
)
from .scene import JointSceneAnalysis
from .skills.repository import JointSkillBundle


def _expected_cell_authority_binding_sha256(
    *,
    case: JointCaseContext,
    bundle: JointSkillBundle,
    portfolio: Any,
) -> str:
    binding = getattr(portfolio, "authority_binding", None)
    if not isinstance(binding, Mapping):
        raise JointContractError(
            "cell portfolio lacks compiler-owned authority binding"
        )
    source_tissue = np.ascontiguousarray(
        load_id_mask(case.source_tissue_mask_uri)
    )
    source_nuclei = np.ascontiguousarray(
        load_nuclei_mask(case.source_nuclei_mask_uri)
    )
    expected = build_cell_portfolio_authority_binding(
        case=case,
        source_tissue=source_tissue,
        source_nuclei=source_nuclei,
        joint_bundle=bundle,
    )
    if dict(binding) != expected:
        raise JointContractError(
            "cell portfolio authority is detached from current runtime inputs or skills"
        )
    return canonical_metadata_sha256(expected)


def _mask_planner_case_metadata(case: JointCaseContext) -> dict[str, Any]:
    """Return case metadata without a raw histology locator.

    The source-image digest remains part of the audit bundle, but the execution
    Planner receives neither H&E pixels nor a path from which they could be
    loaded. Tissue and nuclei locators remain because they are the declared
    observation authority for this planning stage.
    """

    metadata = dict(case.to_metadata())
    metadata.pop("source_image_uri", None)
    return metadata


@dataclass(frozen=True)
class OpenAIMultimodalJointPlanner:
    client: OpenAIResponsesJSONClient
    escalation_client: OpenAIResponsesJSONClient | None = None
    max_contract_attempts: int = 2
    name: str = "openai_mask_graph_joint_planner"
    supports_pathology_vision: bool = False

    def select_interpretation(
        self,
        *,
        case: JointCaseContext,
        scene: JointSceneAnalysis,
        options: Sequence[JointInterpretationOption],
        image_paths: Sequence[str | Path],
        artifact_registry: MaskPlannerArtifactRegistry | None = None,
    ) -> tuple[str, str, dict[str, Any]]:
        image_paths = validate_mask_planner_image_paths(
            image_paths, case=case, artifact_registry=artifact_registry
        )
        payload = {
            "case": _mask_planner_case_metadata(case),
            "scene": scene.to_metadata(),
            "available_interpretations": [
                item.to_metadata() for item in options
            ],
            "requirements": {
                "choose_only_listed_primitive_mechanism_pair": True,
                "use_only_mask_graph_and_candidate_certificate_inputs": True,
                "prefer_best_semantic_fit_that_is_deterministically_supported": True,
                "semantic_fit_is_a_prior_not_a_veto": True,
                "contextual_option_may_win_when_mask_graph_support_is_substantially_stronger": True,
                "contextual_fit_requires_explicit_explanation": True,
                "do_not_abstain_when_a_listed_option_is_supported": True,
                "abstain_only_if_no_listed_option_has_required_certificate_evidence": True,
                "request_user_clarification_only_when_two_or_three_executable_primitives_remain_and_they_encode_materially_different_user_intent": True,
                "do_not_request_clarification_for_tool_parameters_owned_by_skills": True,
                "do_not_infer_annotation_or_population_profile": True,
                "source_H&E_is_prohibited_for_execution_planning": True,
                "do_not_infer_unannotated_histology": True,
                "return_only_certified_candidate_ids": True,
                "cite_only_measured_capability_metric_ids": True,
            },
        }
        available = {item.option_id: item for item in options}
        errors = []
        for attempt, client in enumerate(self._contract_clients(), start=1):
            raw, usage = client.call(
                system_prompt=(
                    "You are the mask-graph semantic-resolution and planning stage of a joint "
                    "pathology editor. The user's natural language may intentionally leave tumor "
                    "increase underspecified. Select the most semantically faithful listed "
                    "primitive-mechanism pair supported by the tissue mask, nuclei mask, scene "
                    "graph, skill rules, and deterministic candidate certificates. Semantic fit is "
                    "a prior, not an absolute "
                    "ordering: a contextual interpretation may outrank a direct one when the patch "
                    "has substantially stronger certified mask support. If two or three executable "
                    "primitive meanings remain equally plausible but encode materially different "
                    "user intent that masks cannot recover, request user clarification and list only "
                    "those primitive IDs. Do not ask about numeric parameters, interfaces or a "
                    "mechanism already resolved by skill rules and certificates. Abstain only when "
                    "none of the listed options is supportable. Do not output pixels, coordinates, "
                    "counts or density multipliers. Never infer invasive morphology, lumen, "
                    "secretion, fibrosis, tumor bed, or another unannotated structure from raw H&E; "
                    "raw H&E is not an execution-planning input."
                ),
                user_prompt=json.dumps(
                    {**payload, "previous_contract_errors": errors},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                image_paths=image_paths,
                schema_name="joint_pathology_interpretation_selection",
                json_schema=MECHANISM_SELECTION_SCHEMA,
            )
            provider_usage = isolate_provider_usage(usage)
            if raw.get("clarification_required") is True:
                require_optional_token(
                    raw.get("clarification_reason"),
                    expected=SEMANTIC_CLARIFICATION_TOKEN,
                    field="clarification_reason",
                )
                primitive_ids = _strings(
                    raw.get("clarification_primitive_ids"),
                    "clarification_primitive_ids",
                )
                available_primitives = {
                    item.primitive_id for item in options
                }
                if (
                    not 2 <= len(primitive_ids) <= 3
                    or len(set(primitive_ids)) != len(primitive_ids)
                    or set(primitive_ids) - available_primitives
                ):
                    raise JointContractError(
                        "joint Planner returned an invalid clarification option set"
                    )
                raise PlannerClarificationRequired(
                    SEMANTIC_CLARIFICATION_TOKEN,
                    primitive_ids=primitive_ids,
                )
            if raw.get("abstain") is True:
                require_optional_token(
                    raw.get("abstain_reason"),
                    expected=SEMANTIC_ABSTAIN_TOKEN,
                    field="abstain_reason",
                )
                raise JointContractError(
                    "joint mechanism Planner abstained: "
                    + SEMANTIC_ABSTAIN_TOKEN
                )
            try:
                if _required_string(raw, "decision_id") != "select_primitive_mechanism_pair":
                    raise JointContractError(
                        "joint Planner returned an illegal interpretation decision"
                    )
                primitive_id = _required_string(raw, "primitive_id")
                mechanism_id = _required_string(raw, "mechanism_id")
                option_id = f"{primitive_id}::{mechanism_id}"
                if option_id not in available:
                    raise JointContractError(
                        "joint Planner selected an unavailable primitive-mechanism pair"
                    )
                selected = available[option_id]
                if (
                    "select_primitive_mechanism_pair"
                    not in selected.mechanism.planner_policy.allowed_decisions
                ):
                    raise JointContractError(
                        "skill policy forbids primitive-mechanism selection"
                    )
                confidence = _unit(raw, "confidence")
                explanation = require_token(
                    raw.get("interpretation_explanation"),
                    expected=SEMANTIC_SELECTION_TOKEN,
                    field="interpretation_explanation",
                )
                observations = require_exact_tokens(
                    raw.get("supporting_observations"),
                    expected=(SEMANTIC_OBSERVATION_TOKEN,),
                    field="supporting_observations",
                )
                capability_metric_ids = _strings(
                    raw.get("supporting_capability_metric_ids"),
                    "supporting_capability_metric_ids",
                )
                available_metrics = set(
                    selected.to_metadata()["deterministic_candidate_metrics"]
                )
                unknown_metrics = set(capability_metric_ids) - available_metrics
                if unknown_metrics:
                    raise JointContractError(
                        "joint Planner cited unmeasured capability metrics: "
                        + ", ".join(sorted(unknown_metrics))
                    )
                require_exact_tokens(
                    raw.get("observed_contraindications", []),
                    expected=(),
                    field="observed_contraindications",
                )
                # Confidence, prose observations, and self-reported
                # contraindications are audit-only LLM outputs. They never
                # authorize or invalidate an option: the listed option already
                # carries deterministic preflight and hard-gate certificates.
            except JointContractError as exc:
                errors.append(f"attempt {attempt}: {exc}")
                continue
            return primitive_id, mechanism_id, {
                "provider": self.name,
                "stage": "semantic_and_mechanism_selection",
                "contract_attempt": attempt,
                "escalated": client is self.escalation_client,
                "selection": {
                    "option_id": option_id,
                    "primitive_id": primitive_id,
                    "mechanism_id": mechanism_id,
                    "semantic_fit": selected.semantic_fit,
                    "semantic_priority": selected.semantic_priority,
                    "interpretation_explanation": explanation,
                    "supporting_observations": list(observations),
                    "supporting_capability_metric_ids": list(
                        capability_metric_ids
                    ),
                    "confidence": confidence,
                },
                "provider_usage": provider_usage,
            }
        raise JointContractError(
            "joint mechanism Planner exhausted contract attempts: " + "; ".join(errors)
        )

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
    ) -> tuple[JointEditPlan, dict[str, Any]]:
        image_paths = validate_mask_planner_image_paths(
            image_paths, case=case, artifact_registry=artifact_registry
        )
        if bundle.primitive.scope == "cell_only":
            return self._select_certified_cell_plan(
                case=case,
                bundle=bundle,
                portfolio=candidate_portfolio,
                image_paths=image_paths,
            )
        payload = {
            "case": _mask_planner_case_metadata(case),
            "scene": scene.to_metadata(),
            "selected_mechanism": bundle.to_metadata(),
            "mechanism_contract": {
                "recognition": asdict(bundle.mechanism.recognition),
                "representability": asdict(bundle.mechanism.representability),
                "tissue_program": asdict(bundle.mechanism.tissue_program),
                "cell_program": asdict(bundle.mechanism.cell_program),
                "coupling": asdict(bundle.mechanism.coupling),
                "render": asdict(bundle.mechanism.render),
                "planner_policy": asdict(bundle.mechanism.planner_policy),
            },
            "compiled_tissue_plan": (
                tissue_plan.to_metadata() if tissue_plan is not None else None
            ),
            "primitive_contract": asdict(bundle.primitive),
            "requirements": {
                "accept_only_if_tissue_plan_matches_skill_and_candidate_certificate": True,
                "cell_plan_is_required_even_when_policy_is_retain": True,
                "cite_only_active_rule_ids": True,
                "select_only_skill_allowed_layout": True,
                "compiled_layout_program_is_immutable": (
                    bundle.mechanism.cell_program.layout_for(case.primitive_id)
                ),
                "do_not_output_polygons_pixels_coordinates_counts_or_density_multipliers": True,
                "area_budget_is_immutable_and_compiler_owned": True,
                "semantic_intent_is_immutable_and_parser_owned": (
                    case.compiled_normalized_intent()
                ),
                "cell_only_primitive_must_preserve_tissue": (
                    bundle.primitive.scope == "cell_only"
                ),
                "select_interface_and_anchor_ids_not_coordinates": True,
                "local_population_primitives_select_component_population_zone": True,
                "cellularity_decrease_requires_mask_graph_interface_anchor_and_density_gradient": (
                    case.primitive_id == "cellularity-decrease-v1"
                ),
                "choose_baseline_and_mechanism_program_separately": True,
                "source_H&E_is_prohibited_for_execution_planning": True,
                "do_not_infer_unannotated_histology": True,
            },
        }
        errors = []
        for attempt, client in enumerate(self._contract_clients(), start=1):
            raw, usage = client.call(
                system_prompt=(
                    "You are a mask-graph joint pathology edit Planner. Review the already compiled "
                    "deterministic tissue interface plan (or an explicit preserve-tissue contract), "
                    "the tissue/nuclei masks, scene graph, candidate certificate, and skill policy. "
                    "Output a tissue binding, cell intent and coupling "
                    "intent. The Semantic Parser already owns the immutable user intent; do not "
                    "reinterpret it. Deterministic tools own every pixel, coordinate, count and "
                    "numeric spatial parameter. Do not infer unannotated pathology from H&E. Use "
                    "the explicit abstain field instead of guessing."
                ),
                user_prompt=json.dumps(
                    {**payload, "previous_contract_errors": errors},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                image_paths=image_paths,
                schema_name="joint_pathology_edit_plan",
                json_schema=JOINT_PLAN_JSON_SCHEMA,
            )
            provider_usage = isolate_provider_usage(usage)
            if raw.get("abstain") is True:
                require_optional_token(
                    raw.get("abstain_reason"),
                    expected=JOINT_ABSTAIN_TOKEN,
                    field="abstain_reason",
                )
                raise JointContractError(
                    "joint edit Planner abstained: " + JOINT_ABSTAIN_TOKEN
                )
            try:
                plan = self._parse_plan(
                    raw=raw,
                    case=case,
                    scene=scene,
                    bundle=bundle,
                    tissue_plan=tissue_plan,
                )
            except JointContractError as exc:
                errors.append(f"attempt {attempt}: {exc}")
                continue
            return plan, {
                "provider": self.name,
                "stage": "joint_plan",
                "contract_attempt": attempt,
                "escalated": client is self.escalation_client,
                "provider_usage": provider_usage,
            }
        raise JointContractError(
            "joint edit Planner exhausted contract attempts: " + "; ".join(errors)
        )

    def _select_certified_cell_plan(
        self,
        *,
        case: JointCaseContext,
        bundle: JointSkillBundle,
        portfolio: Sequence[Any],
        image_paths: Sequence[str | Path],
    ) -> tuple[JointEditPlan, dict[str, Any]]:
        portfolio_object = portfolio
        vetoed = tuple(getattr(portfolio_object, "vetoed", ()))
        portfolio = tuple(portfolio_object)
        if not portfolio or any(
            not isinstance(item, CertifiedCellPlanCandidate)
            for item in portfolio
        ):
            raise JointContractError(
                "cell-only LLM planning requires a pre-LLM certified portfolio"
            )
        if not hasattr(portfolio_object, "validate_authority"):
            raise JointContractError(
                "cell-only LLM planning requires a compiler-issued portfolio capability"
            )
        expected_binding = _expected_cell_authority_binding_sha256(
            case=case,
            bundle=bundle,
            portfolio=portfolio_object,
        )
        portfolio_object.validate_authority(
            expected_binding_sha256=expected_binding
        )
        for item in portfolio:
            validate_cell_plan_candidate(item)
            if item.plan.selected_mechanism_id != bundle.mechanism.mechanism_id:
                raise JointContractError(
                    "cell candidate belongs to another mechanism"
                )
        available = {item.candidate_id: item for item in portfolio}
        if len(available) != len(portfolio):
            raise JointContractError("cell candidate portfolio IDs are not unique")
        payload = {
            "case": _mask_planner_case_metadata(case),
            "selected_mechanism": bundle.mechanism.mechanism_id,
            "planner_policy": asdict(bundle.mechanism.planner_policy),
            "certified_cell_plan_candidates": [
                item.to_metadata() for item in portfolio
            ],
            "vetoed_cell_plan_candidates": [
                item.to_metadata() for item in vetoed
            ],
            "requirements": {
                "select_only_surviving_candidate_id": True,
                "select_only_candidate_tool_program": True,
                "do_not_output_geometry_counts_or_density": True,
                "source_H&E_is_prohibited": True,
            },
        }
        errors = []
        for attempt, client in enumerate(self._contract_clients(), start=1):
            raw, usage = client.call(
                system_prompt=(
                    "Select one immutable cell-plan certificate using only mask-graph "
                    "metrics and skill preference IDs. The compiler owns zones, "
                    "interfaces, anchors, counts, shapes and all numeric parameters."
                ),
                user_prompt=json.dumps(
                    {**payload, "previous_contract_errors": errors},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                image_paths=image_paths,
                schema_name="certified_cell_plan_selection",
                json_schema=CELL_PLAN_SELECTION_SCHEMA,
            )
            provider_usage = isolate_provider_usage(usage)
            if raw.get("abstain") is True:
                require_optional_token(
                    raw.get("abstain_reason"),
                    expected=CELL_ABSTAIN_TOKEN,
                    field="abstain_reason",
                )
                raise JointContractError(
                    "cell-plan Planner abstained: " + CELL_ABSTAIN_TOKEN
                )
            try:
                decision = _required_string(raw, "decision_id")
                if decision != "select_certified_cell_plan_candidate" or decision not in set(
                    bundle.mechanism.planner_policy.allowed_decisions
                ):
                    raise JointContractError(
                        "cell-plan decision is outside the skill policy"
                    )
                candidate_id = _required_string(raw, "selected_candidate_id")
                selected = available.get(candidate_id)
                if selected is None:
                    raise JointContractError(
                        "LLM selected an unknown or vetoed cell candidate"
                    )
                program_id = _required_string(raw, "selected_tool_program_id")
                if program_id not in selected.allowed_tool_program_ids:
                    raise JointContractError(
                        "LLM selected a cell tool program outside the certificate"
                    )
                preference_ids = _strings(
                    raw.get("supporting_preference_rule_ids"),
                    "supporting_preference_rule_ids",
                )
                if set(preference_ids) - set(
                    bundle.mechanism.planner_policy.selection_preferences
                ):
                    raise JointContractError(
                        "LLM cited an unknown cell-selection preference"
                    )
                missing_metrics = {
                    PREFERENCE_METRIC_CATALOG[rule_id][0]
                    for rule_id in preference_ids
                } - set(selected.deterministic_candidate_metrics)
                if missing_metrics:
                    raise JointContractError(
                        "cell candidate omits a cited measured metric: "
                        + ", ".join(sorted(missing_metrics))
                    )
                validate_cell_plan_candidate(selected)
                require_token(
                    raw.get("selection_explanation"),
                    expected=CELL_SELECTION_TOKEN,
                    field="selection_explanation",
                )
            except JointContractError as exc:
                errors.append(f"attempt {attempt}: {exc}")
                continue
            selection_handle = CellPlanSelectionHandle.from_candidate(
                selected,
                selected_tool_program_id=program_id,
            )
            return selected.plan, {
                "provider": self.name,
                "stage": "certified_cell_plan_selection",
                "contract_attempt": attempt,
                "selected_candidate_id": selected.candidate_id,
                "selected_tool_program_id": program_id,
                "compiler_certificate_sha256": (
                    selected.compiler_certificate_sha256
                ),
                "executable_contract_id": selected.executable_contract_id,
                "selection_handle": selection_handle.to_metadata(),
                "portfolio_candidate_count": len(portfolio),
                "ranking_mode": (
                    "rank_surviving_candidates"
                    if len(portfolio) > 1
                    else "single_candidate_accept_or_abstain"
                ),
                "provider_usage": provider_usage,
            }
        raise JointContractError(
            "cell-plan Planner exhausted contract attempts: " + "; ".join(errors)
        )

    def _parse_plan(self, *, raw, case, scene, bundle, tissue_plan):
        _reject_prohibited_geometry_payload(raw)
        decision_ids = set(
            _strings(raw.get("decision_ids"), "decision_ids")
        )
        if decision_ids - set(
            bundle.mechanism.planner_policy.allowed_decisions
        ):
            raise JointContractError(
                "joint Planner returned a decision outside the skill policy"
            )
        required_decisions = {"select_allowed_tool_program"}
        if bundle.primitive.scope == "tissue_and_cell" or case.primitive_id not in LOCAL_POPULATION_PRIMITIVES:
            required_decisions.add("select_certified_interface_anchor_ids")
        if not required_decisions.issubset(
            decision_ids
        ):
            raise JointContractError(
                "joint Planner omitted required deterministic decision bindings"
            )
        if raw.get("tissue_plan_accepted") is not True:
            raise JointContractError(
                "joint Planner rejected the tissue execution contract"
            )
        local_population = case.primitive_id in LOCAL_POPULATION_PRIMITIVES
        anchored_depletion = case.primitive_id == "cellularity-decrease-v1"
        bound_interfaces = set(
            _strings(
                raw.get("bound_interface_ids"),
                "bound_interface_ids",
                allow_empty=local_population and not anchored_depletion,
            )
        )
        if bundle.primitive.scope == "tissue_and_cell":
            if tissue_plan is None:
                raise JointContractError("tissue primitive lacks compiled tissue plan")
            expected_interfaces = {
                item.interface_id for item in tissue_plan.candidate_interfaces
            }
            if bound_interfaces != expected_interfaces:
                raise JointContractError(
                    "joint Planner did not bind every compiled tissue interface"
                )
        else:
            if tissue_plan is not None:
                raise JointContractError("cell-only primitive received a tissue plan")
            known_interfaces = {
                item.interface_id: item for item in scene.tissue.graph.interfaces
            }
            unknown = bound_interfaces - set(known_interfaces)
            if unknown:
                raise JointContractError(
                    "joint Planner selected unknown cell interfaces: "
                    + ", ".join(sorted(unknown))
                )
            if (
                "external_boundary_binding"
                in bundle.mechanism.planner_policy.hard_constraint_checker_ids
            ):
                non_external = [
                    interface_id
                    for interface_id in bound_interfaces
                    if not classify_tumor_stroma_boundary(
                        scene=scene,
                        interface=known_interfaces[interface_id],
                    )["external_tumor_stroma_boundary"]
                ]
                if non_external:
                    raise JointContractError(
                        "joint Planner selected an internal or non-Tumor--Stroma interface: "
                        + ", ".join(sorted(non_external))
                    )
            if not local_population:
                host = set(bundle.primitive.host_tissue_labels)
                incompatible = [
                    interface_id
                    for interface_id in bound_interfaces
                    if not (
                        (
                            known_interfaces[interface_id].source_label in host
                            and known_interfaces[interface_id].target_label == "Tumor"
                        )
                        or (
                            known_interfaces[interface_id].target_label in host
                            and known_interfaces[interface_id].source_label == "Tumor"
                        )
                    )
                ]
                if incompatible:
                    raise JointContractError(
                        "cell-only Planner selected a non tumor/host interface: "
                        + ", ".join(incompatible)
                    )
        if (
            _required_string(raw, "selected_mechanism_id")
            != bundle.mechanism.mechanism_id
        ):
            raise JointContractError("joint Planner changed the selected mechanism")
        supporting_rules = _strings(
            raw.get("supporting_rule_ids"), "supporting_rule_ids"
        )
        unknown_rules = set(supporting_rules) - set(bundle.active_rule_ids)
        if unknown_rules:
            raise JointContractError(
                "joint Planner cited unknown rules: " + ", ".join(sorted(unknown_rules))
            )
        required_rules = set(
            bundle.mechanism.coupling.compatibility_rule_ids
        )
        missing_rules = required_rules - set(supporting_rules)
        if missing_rules:
            raise JointContractError(
                "joint Planner omitted required mechanism rules: "
                + ", ".join(sorted(missing_rules))
            )
        preference_ids = _strings(
            raw.get("supporting_preference_rule_ids"),
            "supporting_preference_rule_ids",
        )
        unknown_preferences = set(preference_ids) - set(
            bundle.mechanism.planner_policy.selection_preferences
        )
        if unknown_preferences:
            raise JointContractError(
                "joint Planner cited unknown preference rules: "
                + ", ".join(sorted(unknown_preferences))
            )
        mechanism = bundle.mechanism
        raw_cell_plan = raw.get("cell_plan")
        if not isinstance(raw_cell_plan, Mapping):
            raise JointContractError("cell_plan is required")
        raw_cell_plan = dict(raw_cell_plan)
        require_token(
            raw_cell_plan.get("expected_morphology"),
            expected=JOINT_EXPECTATION_TOKEN,
            field="cell_plan.expected_morphology",
        )
        raw_cell_plan["expected_morphology"] = "; ".join(
            mechanism.render.required_for(case.primitive_id)
        )
        if raw_cell_plan.get("spatial_anchor_observation") is not None:
            require_token(
                raw_cell_plan.get("spatial_anchor_observation"),
                expected=JOINT_ANCHOR_TOKEN,
                field="cell_plan.spatial_anchor_observation",
            )
        mandatory_protected = tuple(
            item.instance_id for item in scene.cells.instances if item.touches_border
        )
        raw_cell_plan["protected_instance_ids"] = list(mandatory_protected)
        raw_cell_plan["interface_ids"] = sorted(bound_interfaces)
        cell_plan = CellEditPlan.from_mapping(raw_cell_plan)
        unknown_cell_rules = set(cell_plan.supporting_rule_ids) - set(
            bundle.active_rule_ids
        )
        if unknown_cell_rules:
            raise JointContractError(
                "joint Planner cell plan cited unknown rules: "
                + ", ".join(sorted(unknown_cell_rules))
            )
        missing_cell_rules = required_rules - set(cell_plan.supporting_rule_ids)
        if missing_cell_rules:
            raise JointContractError(
                "joint Planner cell plan omitted required mechanism rules: "
                + ", ".join(sorted(missing_cell_rules))
            )
        known_anchors = {
            anchor.anchor_segment_id: anchor
            for anchor in scene.tissue.graph.anchor_segments
            if anchor.interface_id in bound_interfaces
        }
        if (not local_population or anchored_depletion) and (
            not cell_plan.anchor_ids or set(cell_plan.anchor_ids) - set(known_anchors)
        ):
            raise JointContractError(
                "joint Planner cell anchor IDs are empty or outside bound interfaces"
            )
        if local_population:
            zone = next(
                (
                    item
                    for item in scene.population.zones
                    if item.zone_id == cell_plan.core_zone
                ),
                None,
            )
            component_labels = {
                item.component_id: item.label for item in scene.tissue.graph.components
            }
            if (
                zone is None
                or zone.zone_kind != "component"
                or component_labels.get(zone.tissue_component_id)
                not in bundle.primitive.host_tissue_labels
            ):
                raise JointContractError(
                    "local population primitive must bind one legal component population zone"
                )
            if anchored_depletion:
                depletion = bundle.mechanism.cell_program.cellularity_depletion
                interfaces = {
                    item.interface_id: item
                    for item in scene.tissue.graph.interfaces
                }
                if (
                    depletion is None
                    or cell_plan.spatial_anchor_type
                    not in depletion.allowed_anchor_types
                    or cell_plan.layout_program_id
                    != "localized_density_gradient"
                    or cell_plan.mechanism_program_id
                    != "localized_density_gradient"
                ):
                    raise JointContractError(
                        "cellularity decrease lacks the skill-owned anchored gradient program"
                    )
                neighbor_labels = set()
                for interface_id in cell_plan.interface_ids:
                    interface = interfaces[interface_id]
                    if interface.source_component_id == zone.tissue_component_id:
                        neighbor_labels.add(interface.target_label)
                    elif interface.target_component_id == zone.tissue_component_id:
                        neighbor_labels.add(interface.source_label)
                    else:
                        raise JointContractError(
                            "depletion interface does not touch the selected population component"
                        )
                if neighbor_labels - set(depletion.allowed_neighbor_labels):
                    raise JointContractError(
                        "depletion interface neighbor is not allowed by the mechanism skill"
                    )
            elif (
                cell_plan.interface_ids
                or cell_plan.anchor_ids
                or cell_plan.spatial_anchor_type != "not_applicable"
                or cell_plan.spatial_anchor_observation is not None
            ):
                raise JointContractError(
                    "non-depletion local population primitive must not bind interface anchors"
                )
            component_label = component_labels.get(zone.tissue_component_id)
            compatible_classes = set(
                bundle.cell_observation_profile.tissue_compatible_classes.get(
                    component_label, ()
                )
            )
            if not set(cell_plan.allowed_cell_classes).issubset(compatible_classes):
                raise JointContractError(
                    "local population Planner selected a cell class incompatible with the bound tissue component"
                )
            if (
                case.primitive_id.startswith("cell-type-abundance-")
                and len(cell_plan.allowed_cell_classes) != 1
            ):
                raise JointContractError(
                    "cell abundance primitive requires exactly one observable cell class"
                )
        missing_auxiliary = sorted(
            set(mechanism.representability.required_auxiliary_structures)
            - set(scene.auxiliary_structure_masks)
        )
        if missing_auxiliary:
            raise JointContractError(
                "joint mechanism lacks required auxiliary maps: "
                + ", ".join(missing_auxiliary)
            )
        if (
            not mechanism.representability.allow_semantic_instance_fallback
            and scene.cells.observation_quality != "native_instance"
        ):
            raise JointContractError(
                "mechanism requires native nucleus instances; semantic fallback is forbidden"
            )
        label_contract = mechanism.tissue_program.primitive_label_contracts.get(
            case.primitive_id
        )
        if label_contract is None:
            raise JointContractError("joint mechanism has no primitive label contract")
        if tissue_plan is not None:
            if not set(tissue_plan.source_labels).issubset(
                label_contract["source_labels"]
            ):
                raise JointContractError(
                    "compiled tissue source labels violate the joint mechanism"
                )
            if tissue_plan.target_label not in label_contract["target_labels"]:
                raise JointContractError(
                    "compiled tissue target label violates the joint mechanism"
                )
        if set(cell_plan.actions) - set(mechanism.cell_program.actions):
            raise JointContractError(
                "joint Planner cell actions exceed the mechanism contract"
            )
        if set(cell_plan.allowed_cell_classes) - set(
            mechanism.cell_program.allowed_cell_classes
        ):
            raise JointContractError(
                "joint Planner cell classes exceed the mechanism contract"
            )
        expected_layout = mechanism.cell_program.layout_for(case.primitive_id)
        if cell_plan.layout_program_id != expected_layout:
            raise JointContractError(
                "joint Planner changed the skill-compiled primitive layout"
            )
        if cell_plan.baseline_mode not in bundle.primitive.allowed_baseline_modes:
            raise JointContractError("joint Planner selected an illegal baseline mode")
        if cell_plan.baseline_mode == "render_owned_clearance":
            if tuple(cell_plan.actions) != ("retain", "remove_whole"):
                raise JointContractError(
                    "render-owned clearance permits retain/remove_whole only"
                )
            if cell_plan.layout_program_id != "preserve_only":
                raise JointContractError(
                    "render-owned clearance forbids a nucleus placement layout"
                )
            if (
                case.primitive_id
                not in mechanism.cell_program.render_owned_clearance_primitives
            ):
                raise JointContractError(
                    "mechanism skill does not expose render-owned clearance"
                )
        if cell_plan.mechanism_quota_role not in bundle.primitive.allowed_quota_roles:
            raise JointContractError(
                "joint Planner selected an illegal mechanism quota role"
            )
        if cell_plan.mechanism_program_id != expected_layout:
            raise JointContractError(
                "joint Planner changed the skill-compiled mechanism program"
            )
        if bundle.primitive.target_cell_classes and set(
            cell_plan.allowed_cell_classes
        ) - set(bundle.primitive.target_cell_classes):
            raise JointContractError(
                "joint Planner selected a class outside the primitive contract"
            )
        raw_coupling = raw.get("coupling_plan")
        if not isinstance(raw_coupling, Mapping):
            raise JointContractError("coupling_plan is required")
        coupling_rules = _strings(
            raw_coupling.get("compatibility_rule_ids"), "compatibility_rule_ids"
        )
        if set(coupling_rules) != set(
            mechanism.coupling.compatibility_rule_ids
        ):
            raise JointContractError(
                "joint Planner omitted or changed required coupling rules"
            )
        coupling = CouplingPlan(
            compatibility_rule_ids=coupling_rules,
            area_contract_id=(
                "cell-count-extent-v1"
                if bundle.primitive.scope == "cell_only"
                else "joint-union-g2-v1"
            ),
            render_support_policy_id=mechanism.coupling.render_support_policy_id,
            allow_neoplastic_in_non_tumor_tissue=mechanism.coupling.allow_neoplastic_in_non_tumor_tissue,
            maximum_halo_px=mechanism.cell_program.halo_distance_px[1],
        )
        structural_unit_ids = _strings(
            raw.get("structural_unit_ids", []),
            "structural_unit_ids",
            allow_empty=True,
        )
        known_structural_units = set(scene.structural_unit_masks)
        unknown_structural_units = set(structural_unit_ids) - known_structural_units
        if unknown_structural_units:
            raise JointContractError(
                "joint Planner selected unknown structural units: "
                + ", ".join(sorted(unknown_structural_units))
            )
        if tissue_plan is not None and known_structural_units:
            bound_components = set()
            interfaces_by_id = {
                item.interface_id: item for item in scene.tissue.graph.interfaces
            }
            for interface_id in cell_plan.interface_ids:
                interface = interfaces_by_id.get(interface_id)
                if interface is not None:
                    bound_components.update(
                        (
                            interface.source_component_id,
                            interface.target_component_id,
                        )
                    )
            unit_parent = {
                str(item.get("unit_id")): item.get("parent_tissue_component_id")
                for item in scene.structural_hierarchy.get("structure_units", ())
                if isinstance(item, Mapping)
            }
            eligible_units = {
                unit_id
                for unit_id, parent_id in unit_parent.items()
                if parent_id in bound_components
            }
            if eligible_units and not structural_unit_ids:
                raise JointContractError(
                    "joint Planner omitted structural units on a structure-aware interface"
                )
            if set(structural_unit_ids) - eligible_units:
                raise JointContractError(
                    "joint Planner structural units are not bound to selected interfaces"
                )
        plan = JointEditPlan(
            schema_version=JOINT_PLAN_SCHEMA_VERSION,
            case_id=case.case_id,
            normalized_intent=case.compiled_normalized_intent(),
            selected_mechanism_id=mechanism.mechanism_id,
            supporting_observations=require_exact_tokens(
                raw.get("supporting_observations"),
                expected=(JOINT_OBSERVATION_TOKEN,),
                field="supporting_observations",
            ),
            supporting_rule_ids=supporting_rules,
            representability_confidence=_unit(raw, "representability_confidence"),
            tissue_plan=tissue_plan,
            cell_plan=cell_plan,
            coupling_plan=coupling,
            uncertainties=require_token_subset(
                raw.get("uncertainties", []),
                allowed=frozenset({"mask_annotation_limits_apply"}),
                field="uncertainties",
            ),
            escalation_reason=require_optional_token(
                raw.get("escalation_reason"),
                expected="requires_human_semantic_review",
                field="escalation_reason",
            ),
            structural_unit_ids=structural_unit_ids,
            supporting_preference_rule_ids=preference_ids,
        )
        return plan

    def _contract_clients(self) -> tuple[OpenAIResponsesJSONClient, ...]:
        if self.max_contract_attempts not in {1, 2}:
            raise JointContractError("joint Planner contract attempts must be 1 or 2")
        clients = [self.client] * self.max_contract_attempts
        if self.escalation_client is not None:
            clients.append(self.escalation_client)
        return tuple(clients)


@dataclass(frozen=True)
class OpenAIMultimodalJointCritic:
    client: OpenAIResponsesJSONClient
    name: str = "openai_mask_condition_critic"
    supports_pathology_vision: bool = False

    def review(
        self,
        *,
        case,
        bundle,
        candidates,
        gate_reports,
        image_paths,
        artifact_registry: MaskPlannerArtifactRegistry | None = None,
    ):
        image_paths = validate_mask_planner_image_paths(
            image_paths,
            case=case,
            artifact_registry=artifact_registry,
            candidate_portfolio=candidates,
        )
        passed_ids = [item.candidate_id for item in gate_reports if item.passed]
        payload = {
            "case": {
                "case_id": case.case_id,
                "instruction": case.instruction,
                "pathology_domain_id": case.pathology_domain_id,
                "annotation_profile_id": case.annotation_profile_id,
            },
            "mechanism_id": bundle.mechanism.mechanism_id,
            "gate_passing_candidate_ids": passed_ids,
            "gate_reports": [
                item.to_metadata() for item in gate_reports if item.passed
            ],
            "active_rule_ids": list(bundle.active_rule_ids),
            "required_findings": list(
                bundle.mechanism.render.required_for(case.primitive_id)
            ),
            "veto_findings": list(
                bundle.mechanism.render.vetoes_for(case.primitive_id)
            ),
            "render_only_claims": list(bundle.mechanism.render.render_only_claims),
            "annotation_operational_stroma_policy": (
                bundle.annotation_profile.operational_stroma_policy
            ),
            "planner_policy": asdict(bundle.mechanism.planner_policy),
            "requirements": {
                "rank_only_gate_passing_candidates": True,
                "review_tissue_and_nuclei_as_one_condition": True,
                "do_not_restore_gate_failures": True,
                "veto_if_condition_violates_skill_or_certificate": True,
                "source_H&E_is_prohibited": True,
            },
        }
        raw, usage = self.client.call(
            system_prompt=(
                "You are an independent mask-condition critic. You receive no Planner "
                "free-form reasoning. Rank only deterministic-gate-passing joint candidates, "
                "considering tissue geometry, complete nuclei layouts, their coupling, skill "
                "preferences, and deterministic certificates. A hard-gate failure can never be "
                "waived. Raw H&E is prohibited and you must not diagnose an unannotated structure."
            ),
            user_prompt=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            image_paths=image_paths,
            schema_name="joint_pathology_critic",
            json_schema=JOINT_CRITIC_JSON_SCHEMA,
        )
        provider_usage = isolate_provider_usage(usage)
        rankings = []
        critic_text: list[str] = []
        for item in raw.get("rankings", []):
            if not isinstance(item, Mapping):
                raise JointContractError("joint critic ranking must be an object")
            candidate_id = _required_string(item, "candidate_id")
            if candidate_id not in passed_ids:
                raise JointContractError(
                    "joint critic ranked a gate-failing or unknown candidate"
                )
            rule_ids = _strings(item.get("supporting_rule_ids"), "supporting_rule_ids")
            if set(rule_ids) - set(bundle.active_rule_ids):
                raise JointContractError("joint critic cited unknown rules")
            veto_reasons = require_token_subset(
                item.get("veto_reasons", []),
                allowed=frozenset({CRITIC_VETO_TOKEN}),
                field="veto_reasons",
            )
            critic_text.extend(veto_reasons)
            rankings.append(
                JointCriticRanking(
                    candidate_id=candidate_id,
                    score=_unit(item, "score"),
                    confidence=_unit(item, "confidence"),
                    supporting_rule_ids=rule_ids,
                    veto_reasons=veto_reasons,
                )
            )
        summary = _required_string(raw, "summary")
        require_token_subset(
            [summary],
            allowed=frozenset(
                {CRITIC_SUMMARY_TOKEN, CRITIC_ABSTAIN_SUMMARY_TOKEN}
            ),
            field="summary",
        )
        return JointCriticResult(
            rankings=tuple(rankings),
            abstain=_required_bool(raw, "abstain"),
            summary=summary,
            usage={"provider": self.name, "provider_usage": provider_usage},
        )


MECHANISM_SELECTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "abstain",
        "abstain_reason",
        "clarification_required",
        "clarification_reason",
        "clarification_primitive_ids",
        "primitive_id",
        "mechanism_id",
        "decision_id",
        "interpretation_explanation",
        "supporting_observations",
        "supporting_capability_metric_ids",
        "observed_contraindications",
        "confidence",
    ],
    "properties": {
        "abstain": {"type": "boolean"},
        "abstain_reason": {
            "type": ["string", "null"],
            "enum": [SEMANTIC_ABSTAIN_TOKEN, None],
        },
        "clarification_required": {"type": "boolean"},
        "clarification_reason": {
            "type": ["string", "null"],
            "enum": [SEMANTIC_CLARIFICATION_TOKEN, None],
        },
        "clarification_primitive_ids": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 3,
        },
        "primitive_id": {"type": ["string", "null"]},
        "mechanism_id": {"type": ["string", "null"]},
        "decision_id": {
            "type": ["string", "null"],
            "enum": ["select_primitive_mechanism_pair", None],
        },
        "interpretation_explanation": {
            "type": ["string", "null"],
            "enum": [SEMANTIC_SELECTION_TOKEN, None],
        },
        "supporting_observations": {
            "type": "array",
            "items": {"type": "string", "enum": [SEMANTIC_OBSERVATION_TOKEN]},
        },
        "supporting_capability_metric_ids": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "observed_contraindications": {
            "type": "array",
            "maxItems": 0,
            "items": {"type": "string"},
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

CELL_PLAN_SELECTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "abstain",
        "abstain_reason",
        "decision_id",
        "selected_candidate_id",
        "selected_tool_program_id",
        "supporting_preference_rule_ids",
        "selection_explanation",
        "confidence",
    ],
    "properties": {
        "abstain": {"type": "boolean"},
        "abstain_reason": {
            "type": ["string", "null"],
            "enum": [CELL_ABSTAIN_TOKEN, None],
        },
        "decision_id": {
            "type": ["string", "null"],
            "enum": ["select_certified_cell_plan_candidate", None],
        },
        "selected_candidate_id": {"type": ["string", "null"]},
        "selected_tool_program_id": {"type": ["string", "null"]},
        "supporting_preference_rule_ids": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string"},
        },
        "selection_explanation": {
            "type": ["string", "null"],
            "enum": [CELL_SELECTION_TOKEN, None],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

JOINT_PLAN_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "abstain",
        "abstain_reason",
        "selected_mechanism_id",
        "decision_ids",
        "supporting_observations",
        "supporting_rule_ids",
        "supporting_preference_rule_ids",
        "representability_confidence",
        "tissue_plan_accepted",
        "bound_interface_ids",
        "structural_unit_ids",
        "cell_plan",
        "coupling_plan",
        "uncertainties",
        "escalation_reason",
    ],
    "properties": {
        "abstain": {"type": "boolean"},
        "abstain_reason": {
            "type": ["string", "null"],
            "enum": [JOINT_ABSTAIN_TOKEN, None],
        },
        "selected_mechanism_id": {"type": ["string", "null"]},
        "decision_ids": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {
                "type": "string",
                "enum": [
                    "select_certified_interface_anchor_ids",
                    "select_allowed_tool_program",
                    "request_clarification",
                    "abstain"
                ]
            }
        },
        "supporting_observations": {
            "type": "array",
            "items": {"type": "string", "enum": [JOINT_OBSERVATION_TOKEN]},
        },
        "supporting_rule_ids": {"type": "array", "items": {"type": "string"}},
        "supporting_preference_rule_ids": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "representability_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "tissue_plan_accepted": {"type": "boolean"},
        "bound_interface_ids": {"type": "array", "items": {"type": "string"}},
        "structural_unit_ids": {
            "type": "array",
            "items": {"type": "string"},
            "uniqueItems": True,
        },
        "cell_plan": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": [
                "core_zone",
                "halo_zone",
                "actions",
                "allowed_cell_classes",
                "layout_program_id",
                "anchor_ids",
                "spatial_anchor_type",
                "spatial_anchor_observation",
                "baseline_mode",
                "mechanism_program_id",
                "mechanism_quota_role",
                "supporting_rule_ids",
                "expected_morphology",
            ],
            "properties": {
                "core_zone": {"type": "string"},
                "halo_zone": {"type": ["string", "null"]},
                "actions": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "string",
                        "enum": ["retain", "remove_whole", "add"],
                    },
                },
                "allowed_cell_classes": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "layout_program_id": {"type": "string"},
                "anchor_ids": {"type": "array", "items": {"type": "string"}},
                "spatial_anchor_type": {
                    "type": "string",
                    "enum": ["not_applicable", "interface"],
                },
                "spatial_anchor_observation": {
                    "type": ["string", "null"],
                    "enum": [JOINT_ANCHOR_TOKEN, None],
                },
                "baseline_mode": {
                    "type": "string",
                    "enum": [
                        "preserve",
                        "regenerate_target_population",
                        "selective_remove",
                        "structured_add",
                        "render_owned_clearance",
                    ],
                },
                "mechanism_program_id": {"type": "string"},
                "mechanism_quota_role": {
                    "type": "string",
                    "enum": [
                        "within_total_quota",
                        "explicit_increment",
                        "explicit_decrement",
                    ],
                },
                "supporting_rule_ids": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string"},
                },
                "expected_morphology": {
                    "type": "string",
                    "enum": [JOINT_EXPECTATION_TOKEN],
                },
            },
        },
        "coupling_plan": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": ["compatibility_rule_ids"],
            "properties": {
                "compatibility_rule_ids": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string"},
                }
            },
        },
        "uncertainties": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["mask_annotation_limits_apply"],
            },
        },
        "escalation_reason": {
            "type": ["string", "null"],
            "enum": ["requires_human_semantic_review", None],
        },
    },
}

JOINT_CRITIC_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["rankings", "abstain", "summary"],
    "properties": {
        "rankings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "candidate_id",
                    "score",
                    "confidence",
                    "supporting_rule_ids",
                    "veto_reasons",
                ],
                "properties": {
                    "candidate_id": {"type": "string"},
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "supporting_rule_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "veto_reasons": {
                        "type": "array",
                        "items": {"type": "string", "enum": [CRITIC_VETO_TOKEN]},
                    },
                },
            },
        },
        "abstain": {"type": "boolean"},
        "summary": {
            "type": "string",
            "enum": [CRITIC_SUMMARY_TOKEN, CRITIC_ABSTAIN_SUMMARY_TOKEN],
        },
    },
}


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise JointContractError(f"{key} must be a non-empty string")
    return value.strip()


def _reject_prohibited_geometry_payload(payload: Mapping[str, Any]) -> None:
    """Reject direct LLM control over compiler-owned numeric geometry."""

    prohibited_keys = {
        "pixels",
        "polygon",
        "polygons",
        "coordinates",
        "nucleus_coordinates",
        "nucleus_count",
        "cell_count",
        "density_multiplier",
        "area_budget",
        "shape_mask",
        "shape_masks",
    }

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key)
                if key_text in prohibited_keys:
                    raise JointContractError(
                        "joint Planner attempted compiler-owned numeric geometry: "
                        + f"{path}.{key_text}"
                    )
                walk(child, f"{path}.{key_text}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(payload, "plan")


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise JointContractError("optional string must be non-empty")
    return value.strip()


def _strings(value: Any, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise JointContractError(f"{label} must be a list of non-empty strings")
    if not value and not allow_empty:
        raise JointContractError(f"{label} cannot be empty")
    return tuple(item.strip() for item in value)


def _unit(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
        raise JointContractError(f"{key} must be in [0, 1]")
    return float(value)


def _required_bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise JointContractError(f"{key} must be boolean")
    return value

"""Multimodal Planner and independent Critic interfaces."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from phase3_mask_edit_refine.models import (
    CandidateMask,
    CaseContext,
    CriticRanking,
    CriticResult,
    DepthProfile,
    EditPlan,
    GateReport,
    InterfaceExecutionContract,
    PlannedInterface,
    RefineContractError,
    ToolProgram,
)
from phase3_mask_edit_refine.scene import SceneAnalysis
from phase3_mask_edit_refine.skills import ActiveKnowledgeBundle

EDIT_PLAN_SCHEMA_VERSION = "mask-edit-refine-plan-v2"


class AgentProviderError(RuntimeError):
    """Raised when a remote Planner or Critic cannot return strict JSON."""


class Planner(Protocol):
    name: str

    def create_plan(
        self,
        *,
        case: CaseContext,
        scene: SceneAnalysis,
        bundle: ActiveKnowledgeBundle,
        image_paths: Sequence[str | Path],
    ) -> tuple[EditPlan, dict[str, Any]]:
        """Return a validated plan candidate and provider usage metadata."""


class Critic(Protocol):
    name: str
    supports_pathology_vision: bool

    def review(
        self,
        *,
        case: CaseContext,
        bundle: ActiveKnowledgeBundle,
        candidates: Sequence[CandidateMask],
        gate_reports: Sequence[GateReport],
        image_paths: Sequence[str | Path],
    ) -> CriticResult:
        """Rank only candidates that already passed deterministic gates."""


@dataclass(frozen=True)
class HeuristicInterfacePlanner:
    """Offline research planner; deterministic and never a production default."""

    name: str = "heuristic_interface_planner"

    def create_plan(
        self,
        *,
        case: CaseContext,
        scene: SceneAnalysis,
        bundle: ActiveKnowledgeBundle,
        image_paths: Sequence[str | Path],
    ) -> tuple[EditPlan, dict[str, Any]]:
        del image_paths
        contract = bundle.edit_contract
        legal = scene.interfaces_for(
            source_labels=contract.source_label_options,
            target_label=contract.target_label,
        )
        if not legal:
            raise RefineContractError(
                "no directed component interface satisfies the composed edit contract"
            )
        ranked = sorted(
            legal,
            key=lambda item: (-item.contact_pixels, item.interface_id),
        )[:2]
        source_label = ranked[0].source_label
        ranked = [item for item in ranked if item.source_label == source_label]
        source_component = scene.component_masks[ranked[0].source_component_id]
        target_pixels = case.area_budget.target_pixels(scene.component_masks[ranked[0].source_component_id], source_component)
        rule_ids = tuple(rule.rule_id for rule in bundle.active_rules) + tuple(
            item.constraint_id for item in bundle.active_mask_constraints
        )
        planned: list[PlannedInterface] = []
        total_contact = max(1, sum(item.contact_pixels for item in ranked))
        for interface in ranked:
            allocation_fraction = interface.contact_pixels / total_contact
            allocated_pixels = target_pixels * allocation_fraction
            estimated_depth = allocated_pixels / max(interface.contact_pixels, 1)
            # The band is a legal envelope, not the intended constant depth.
            # Leave room for a broad tapered lobe to vary naturally while the
            # area gate continues to enforce the immutable task budget.
            peak_depth = float(np.clip(np.ceil(estimated_depth * 1.40), 4, 120))
            # The compiler resolves the exact peak required by the immutable
            # area allocation. This is the maximum legal search envelope.
            band_max = 128.0
            planned.append(
                PlannedInterface(
                    interface_id=interface.interface_id,
                    source_component_id=interface.source_component_id,
                    target_component_id=interface.target_component_id,
                    anchor_segment="full_directed_interface",
                    allowed_edit_band_px=(0.0, band_max),
                    execution_contract=InterfaceExecutionContract(
                        anchor_segment_ids=interface.anchor_segment_ids,
                        area_allocation_fraction=float(allocation_fraction),
                        depth_profile=DepthProfile(
                            mode="tapered_lobe",
                            peak_depth_px=peak_depth,
                            edge_depth_px=max(1.0, peak_depth * 0.55),
                            taper_fraction=0.18,
                            lobe_count=1,
                            noise_amplitude_px=min(8.0, peak_depth * 0.12),
                            noise_correlation_px=24.0,
                        ),
                        min_anchor_coverage_fraction=0.50,
                        max_off_anchor_contact_fraction=0.02,
                        allocation_tolerance_fraction=0.02,
                    ),
                    prohibited_region_ids=(),
                    supporting_rule_ids=rule_ids,
                    expected_morphology=(
                        "broad interface-bound change with continuous, smoothly varying depth"
                    ),
                    confidence=min(0.95, 0.65 + interface.contact_pixels / 4096.0),
                )
            )
        plan = EditPlan(
            schema_version=EDIT_PLAN_SCHEMA_VERSION,
            case_id=case.case_id,
            normalized_intent=case.instruction,
            primitive_id=case.primitive_id,
            source_labels=(source_label,),
            target_label=contract.target_label,
            area_budget=case.area_budget,
            candidate_interfaces=tuple(planned),
            tool_program=ToolProgram(
                allowed_tools=contract.allowed_tools,
                parameter_ranges={
                    "max_changed_components": 2,
                    "min_component_area_px": 16,
                    "max_depth_span_ratio": 1.25,
                    "max_bbox_fill_fraction": 0.985,
                    "max_boundary_compactness": 40.0,
                    "max_source_component_changed_fraction": 0.55,
                    "min_source_component_remaining_px": 64,
                    "min_parallel_front_depth_cv": 0.15,
                    "parallel_front_linearity_ratio": 20.0,
                    "parallel_front_min_depth_px": 5.0,
                    "parallel_front_min_pixels": 64,
                },
                candidate_count=12,
            ),
            hard_invariants=tuple(
                sorted(
                    set(contract.required_check_ids).union(
                        {
                            "label_transition",
                            "unrequested_labels_preserved",
                            "changed_area",
                            "interface_contact",
                            "execution_contract_fidelity",
                            "component_topology",
                            "depth_span_ratio",
                            "provenance_complete",
                        }
                    )
                )
            ),
            uncertainties=("heuristic planner did not inspect H&E",),
            planner_confidence=min(item.confidence for item in planned),
            escalation_reason="heuristic_research_mode_requires_multimodal_review",
        )
        return plan, {"provider": self.name, "input_tokens": 0, "output_tokens": 0}


@dataclass(frozen=True)
class DeterministicResearchCritic:
    """Metric ranker that abstains when any hard rule needs visual pathology review."""

    name: str = "deterministic_research_critic"
    supports_pathology_vision: bool = False

    def review(
        self,
        *,
        case: CaseContext,
        bundle: ActiveKnowledgeBundle,
        candidates: Sequence[CandidateMask],
        gate_reports: Sequence[GateReport],
        image_paths: Sequence[str | Path],
    ) -> CriticResult:
        del case, image_paths
        reports = {report.candidate_id: report for report in gate_reports if report.passed}
        rankings: list[CriticRanking] = []
        for candidate in candidates:
            report = reports.get(candidate.candidate_id)
            if report is None:
                continue
            metrics = {
                check.check_id: check.metrics
                for check in report.checks
            }
            depth = float(metrics.get("depth_span_ratio", {}).get("max_depth_span_ratio", 99.0))
            compactness = float(
                metrics.get("boundary_naturalness", {}).get("boundary_compactness", 99.0)
            )
            score = 1.0 / (1.0 + depth + 0.05 * compactness)
            rankings.append(
                CriticRanking(
                    candidate_id=candidate.candidate_id,
                    score=score,
                    confidence=0.40,
                    supporting_rule_ids=tuple(
                        rule.rule_id
                        for rule in bundle.active_rules
                        if not rule.critic_requirement
                    )
                    + tuple(
                        item.constraint_id
                        for item in bundle.active_mask_constraints
                        if not item.critic_requirement
                    ),
                    veto_reasons=(),
                )
            )
        rankings.sort(key=lambda item: (-item.score, item.candidate_id))
        unassessed = sorted(
            [
                item.constraint_id
                for item in bundle.active_mask_constraints
                if item.critic_requirement
            ]
        )
        return CriticResult(
            rankings=tuple(rankings),
            abstain=bool(unassessed) or not rankings,
            summary=(
                "research metric ranking only; unassessed visual hard rules: "
                + ", ".join(unassessed)
                if unassessed
                else "research metric ranking completed"
            ),
            usage={"provider": self.name, "input_tokens": 0, "output_tokens": 0},
        )


@dataclass(frozen=True)
class FixtureCritic:
    """Test/evaluation critic with an explicit pre-reviewed payload."""

    result: CriticResult
    name: str = "fixture_critic"
    supports_pathology_vision: bool = True

    def review(self, **_: Any) -> CriticResult:
        return self.result


@dataclass(frozen=True)
class OpenAIResponsesJSONClient:
    """Minimal Responses API client with strict JSON schema output."""

    model: str
    reasoning_effort: str = "medium"
    api_base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_sec: float = 180.0
    max_retries: int = 2
    image_detail: str = "auto"

    def call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_paths: Sequence[str | Path],
        schema_name: str,
        json_schema: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], dict[str, Any]]:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise AgentProviderError(
                f"missing API key environment variable: {self.api_key_env}"
            )
        content: list[dict[str, Any]] = [{"type": "input_text", "text": user_prompt}]
        for image_path in image_paths:
            content.append(
                {
                    "type": "input_image",
                    "image_url": _image_path_to_data_url(image_path),
                    "detail": self.image_detail,
                }
            )
        payload = {
            "model": self.model,
            "reasoning": {"effort": self.reasoning_effort},
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": content},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": dict(json_schema),
                }
            },
        }
        response = self._post(payload, api_key=api_key)
        text = _responses_output_text(response)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AgentProviderError("Responses API output was not valid JSON") from exc
        if not isinstance(parsed, Mapping):
            raise AgentProviderError("Responses API JSON output root must be an object")
        usage = response.get("usage", {})
        usage_metadata = dict(usage) if isinstance(usage, Mapping) else {}
        usage_metadata.update(
            {
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "prompt_sha256": hashlib.sha256(
                    (system_prompt + "\n" + user_prompt + "\n" + json.dumps(json_schema, sort_keys=True)).encode("utf-8")
                ).hexdigest(),
                "image_sha256": [
                    hashlib.sha256(Path(path).read_bytes()).hexdigest()
                    for path in image_paths
                ],
            }
        )
        return parsed, usage_metadata

    def _post(self, payload: Mapping[str, Any], *, api_key: str) -> dict[str, Any]:
        endpoint = self.api_base_url.rstrip("/") + "/responses"
        data = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                endpoint,
                data=data,
                method="POST",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                    decoded = json.loads(response.read().decode("utf-8"))
                if not isinstance(decoded, dict):
                    raise AgentProviderError("Responses API root must be an object")
                return decoded
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = AgentProviderError(f"Responses API HTTP {exc.code}: {body}")
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
            if attempt < self.max_retries:
                time.sleep(min(30.0, 2.0**attempt))
        raise AgentProviderError(f"Responses API request failed: {last_error}")


@dataclass(frozen=True)
class OpenAIMultimodalPlanner:
    client: OpenAIResponsesJSONClient
    name: str = "openai_multimodal_planner"
    max_schema_attempts: int = 2

    def create_plan(
        self,
        *,
        case: CaseContext,
        scene: SceneAnalysis,
        bundle: ActiveKnowledgeBundle,
        image_paths: Sequence[str | Path],
    ) -> tuple[EditPlan, dict[str, Any]]:
        if self.max_schema_attempts not in {1, 2}:
            raise RefineContractError("Planner max_schema_attempts must be 1 or 2")
        base_prompt = _planner_prompt(case=case, scene=scene, bundle=bundle)
        system_prompt = (
            "You are a pathology mask-edit planner. Select legal existing interfaces and "
            "deterministic tools. Never output polygons, raster pixels, or a new area budget. "
            "Treat every active mask constraint as mandatory and cite its constraint_id on "
            "every planned interface. Do not claim that a semantic mask guarantees microscopic "
            "H&E morphology; preserve generation_handoff requirements for downstream rendering. "
            "Return only the strict JSON schema. State uncertainty instead of guessing."
        )
        attempt_audit: list[dict[str, Any]] = []
        validation_feedback = ""
        for attempt_index in range(self.max_schema_attempts):
            prompt = base_prompt + validation_feedback
            raw, usage = self.client.call(
                system_prompt=system_prompt,
                user_prompt=prompt,
                image_paths=image_paths,
                schema_name="mask_edit_refine_plan",
                json_schema=EDIT_PLAN_JSON_SCHEMA,
            )
            try:
                plan = EditPlan.from_mapping(raw)
                validate_edit_plan(plan, case=case, scene=scene, bundle=bundle)
            except (TypeError, ValueError) as exc:
                attempt_audit.append(
                    {
                        "attempt": attempt_index + 1,
                        "status": "rejected_by_contract",
                        "error": f"{type(exc).__name__}: {exc}",
                        "raw_response": dict(raw),
                        "usage": usage,
                    }
                )
                if attempt_index + 1 >= self.max_schema_attempts:
                    raise AgentProviderError(
                        "Planner failed strict schema/contract validation after "
                        f"{self.max_schema_attempts} attempts: {exc}"
                    ) from exc
                validation_feedback = (
                    "\nThe previous response was rejected by deterministic contract validation. "
                    "Correct the error without changing the immutable case contract. Error: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            attempt_audit.append(
                {
                    "attempt": attempt_index + 1,
                    "status": "accepted",
                    "usage": usage,
                }
            )
            aggregate = _aggregate_attempt_usage(attempt_audit)
            return plan, {
                "provider": self.name,
                "model": self.client.model,
                **aggregate,
            }
        raise AgentProviderError("Planner exhausted schema attempts without a result")


@dataclass(frozen=True)
class OpenAIMultimodalCritic:
    client: OpenAIResponsesJSONClient
    name: str = "openai_multimodal_critic"
    supports_pathology_vision: bool = True

    def review(
        self,
        *,
        case: CaseContext,
        bundle: ActiveKnowledgeBundle,
        candidates: Sequence[CandidateMask],
        gate_reports: Sequence[GateReport],
        image_paths: Sequence[str | Path],
    ) -> CriticResult:
        passed_ids = [report.candidate_id for report in gate_reports if report.passed]
        prompt = _critic_prompt(
            case=case,
            bundle=bundle,
            gate_reports=gate_reports,
            passed_ids=passed_ids,
        )
        raw, usage = self.client.call(
            system_prompt=(
                "You are an independent pathology morphology critic. You do not see the "
                "Planner's free-form reasoning. Review only deterministic-gate-passing "
                "candidates. A deterministic failure can never be restored. Return strict JSON."
            ),
            user_prompt=prompt,
            image_paths=image_paths,
            schema_name="mask_edit_refine_critic",
            json_schema=CRITIC_JSON_SCHEMA,
        )
        result = _critic_result_from_mapping(raw, usage={"model": self.client.model, **usage})
        unknown = {item.candidate_id for item in result.rankings} - set(passed_ids)
        if unknown:
            raise RefineContractError(
                "critic ranked candidates that did not pass deterministic gates: "
                + ", ".join(sorted(unknown))
            )
        return result


def validate_edit_plan(
    plan: EditPlan,
    *,
    case: CaseContext,
    scene: SceneAnalysis,
    bundle: ActiveKnowledgeBundle,
) -> None:
    errors: list[str] = []
    if plan.schema_version != EDIT_PLAN_SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if plan.case_id != case.case_id:
        errors.append("case_id mismatch")
    if plan.primitive_id != case.primitive_id:
        errors.append("primitive_id mismatch")
    if plan.area_budget != case.area_budget:
        errors.append("Planner modified the immutable area budget")
    if plan.resolved_area is not None:
        # Pixel denominators are verified by the compiler and changed-area gate;
        # here we validate the immutable policy fields that do not require the
        # source raster.
        if plan.resolved_area.fallback_policy != case.area_budget.fallback_policy:
            errors.append("resolved area changed the immutable fallback policy")
        if plan.resolved_area.solver_version == "":
            errors.append("resolved area is missing solver provenance")
    contract = bundle.edit_contract
    if plan.target_label != contract.target_label:
        errors.append("target label exceeds composed contract")
    if not set(plan.source_labels).issubset(contract.source_label_options):
        errors.append("source labels exceed composed contract")
    if not set(plan.tool_program.allowed_tools).issubset(contract.allowed_tools):
        errors.append("tool program exceeds composed contract")
    scene_interfaces = {item.interface_id: item for item in scene.graph.interfaces}
    known_rules = {rule.rule_id for rule in bundle.active_rules}.union(
        item.constraint_id for item in bundle.active_mask_constraints
    )
    required_mask_constraints = {
        item.constraint_id for item in bundle.active_mask_constraints
    }
    allocation_total = sum(
        item.execution_contract.area_allocation_fraction
        for item in plan.candidate_interfaces
    )
    if not np.isclose(allocation_total, 1.0, rtol=0.0, atol=1e-6):
        errors.append(
            f"execution area allocations must sum to 1.0, observed {allocation_total:.8f}"
        )
    planned_interface_ids = [item.interface_id for item in plan.candidate_interfaces]
    if len(set(planned_interface_ids)) != len(planned_interface_ids):
        errors.append("each interface_id may appear only once; combine its anchor segments")
    used_anchor_ids: set[str] = set()
    for interface in plan.candidate_interfaces:
        observed = scene_interfaces.get(interface.interface_id)
        if observed is None:
            errors.append(f"unknown interface_id {interface.interface_id}")
            continue
        if (
            observed.source_component_id != interface.source_component_id
            or observed.target_component_id != interface.target_component_id
            or observed.source_label not in plan.source_labels
            or observed.target_label != plan.target_label
        ):
            errors.append(f"interface contract mismatch for {interface.interface_id}")
        band_min, band_max = interface.allowed_edit_band_px
        if band_min < 0 or band_max <= band_min:
            errors.append(f"invalid edit band for {interface.interface_id}")
        execution = interface.execution_contract
        unknown_anchors = set(execution.anchor_segment_ids) - set(
            observed.anchor_segment_ids
        )
        if unknown_anchors:
            errors.append(
                f"interface {interface.interface_id} cites anchors outside the interface "
                f"{sorted(unknown_anchors)}"
            )
        duplicated_anchors = used_anchor_ids.intersection(execution.anchor_segment_ids)
        if duplicated_anchors:
            errors.append(
                f"anchor segments are allocated more than once {sorted(duplicated_anchors)}"
            )
        used_anchor_ids.update(execution.anchor_segment_ids)
        if execution.depth_profile.peak_depth_px > band_max + 1e-6:
            errors.append(
                f"interface {interface.interface_id} peak depth exceeds allowed band"
            )
        unknown_rules = set(interface.supporting_rule_ids) - known_rules
        if unknown_rules:
            errors.append(
                f"interface {interface.interface_id} cites unknown rules {sorted(unknown_rules)}"
            )
        missing_constraints = required_mask_constraints - set(
            interface.supporting_rule_ids
        )
        if missing_constraints:
            errors.append(
                f"interface {interface.interface_id} omits active mask constraints "
                f"{sorted(missing_constraints)}"
            )
    if errors:
        raise RefineContractError("invalid EditPlan: " + "; ".join(errors))


def critic_satisfies_hard_rules(
    result: CriticResult,
    *,
    bundle: ActiveKnowledgeBundle,
    minimum_confidence: float = 0.70,
) -> tuple[bool, tuple[str, ...]]:
    if result.abstain or not result.rankings:
        return False, (result.summary,)
    top = result.rankings[0]
    required = {
        item.constraint_id
        for item in bundle.active_mask_constraints
        if item.critic_requirement
    }
    missing = sorted(required - set(top.supporting_rule_ids))
    reasons = list(top.veto_reasons)
    if missing:
        reasons.append("critic did not affirm hard visual rules: " + ", ".join(missing))
    if top.confidence < minimum_confidence:
        reasons.append(
            f"critic confidence {top.confidence:.3f} below {minimum_confidence:.3f}"
        )
    return not reasons, tuple(reasons)


def _planner_prompt(
    *, case: CaseContext, scene: SceneAnalysis, bundle: ActiveKnowledgeBundle
) -> str:
    payload = {
        "case": case.to_metadata(),
        "scene_graph": scene.graph.to_metadata(),
        "composed_contract": bundle.to_metadata(),
        "requirements": {
            "select_only_existing_interface_ids": True,
            "select_only_anchor_segment_ids_listed_for_that_interface": True,
            "execution_contract_is_pixel_binding_not_advisory": True,
            "area_allocation_fractions_must_sum_to_one": True,
            "depth_profile_peak_must_not_exceed_allowed_band": True,
            "depth_profile_ratios_are_preserved_while_exact_peak_is_compiled": True,
            "tool_program_parameter_ranges_are_gate_calibration_not_spatial_intent": True,
            "do_not_output_pixels_or_polygons": True,
            "area_budget_is_immutable": True,
            "area_fallback_is_deterministic_not_planner_controlled": True,
            "candidate_count": 12,
            "prefer_broad_interface_before_depth": True,
            "cite_active_rule_ids": True,
            "cite_every_active_mask_constraint_id": True,
            "do_not_claim_generation_handoff_is_mask_guaranteed": True,
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _critic_prompt(
    *,
    case: CaseContext,
    bundle: ActiveKnowledgeBundle,
    gate_reports: Sequence[GateReport],
    passed_ids: Sequence[str],
) -> str:
    payload = {
        "case_id": case.case_id,
        "instruction": case.instruction,
        "pathology_domain_id": case.pathology_domain_id,
        "annotation_profile_id": case.annotation_profile_id,
        "primitive_id": case.primitive_id,
        "gate_passing_candidate_ids": list(passed_ids),
        "gate_reports": [report.to_metadata() for report in gate_reports if report.passed],
        "pathology_reference_rules": [
            {
                "rule_id": rule.rule_id,
                "claim": rule.claim,
                "required_observation": rule.required_observation,
                "severity": rule.severity,
                "critic_requirement": rule.critic_requirement,
                "exceptions": list(rule.exceptions),
                "expected_morphology": list(rule.expected_morphology),
                "forbidden_morphology": list(rule.forbidden_morphology),
                "counterexamples": list(rule.counterexamples),
            }
            for rule in bundle.active_rules
        ],
        "active_mask_constraints": [
            {
                "constraint_id": item.constraint_id,
                "mask_statement": item.mask_statement,
                "observability": list(item.observability),
                "enforcement": item.enforcement,
                "checker_ids": list(item.checker_ids),
                "critic_requirement": item.critic_requirement,
                "generation_handoff": list(item.generation_handoff),
                "known_limitations": list(item.known_limitations),
            }
            for item in bundle.active_mask_constraints
        ],
        "requirements": {
            "rank_only_gate_passing_candidates": True,
            "veto_morphologically_illegal_candidates": True,
            "use_pathology_rules_as_visual_reference_not_mask_guarantees": True,
            "mask_constraints_are_mandatory_and_cannot_be_waived": True,
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _critic_result_from_mapping(
    payload: Mapping[str, Any], *, usage: dict[str, Any]
) -> CriticResult:
    rankings_raw = payload.get("rankings")
    if not isinstance(rankings_raw, list):
        raise RefineContractError("critic rankings must be a list")
    rankings: list[CriticRanking] = []
    for raw in rankings_raw:
        if not isinstance(raw, Mapping):
            raise RefineContractError("critic ranking items must be objects")
        rankings.append(
            CriticRanking(
                candidate_id=_required_string(raw, "candidate_id"),
                score=_unit_number(raw, "score"),
                confidence=_unit_number(raw, "confidence"),
                supporting_rule_ids=_string_tuple(raw.get("supporting_rule_ids", [])),
                veto_reasons=_string_tuple(raw.get("veto_reasons", [])),
            )
        )
    abstain = payload.get("abstain")
    if not isinstance(abstain, bool):
        raise RefineContractError("critic abstain must be boolean")
    return CriticResult(
        rankings=tuple(rankings),
        abstain=abstain,
        summary=_required_string(payload, "summary"),
        usage=usage,
    )


def _responses_output_text(payload: Mapping[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str):
        return direct
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, Mapping) and part.get("type") == "output_text":
                    text = part.get("text")
                    if isinstance(text, str):
                        return text
    raise AgentProviderError("Responses API payload contains no output_text")


def _aggregate_attempt_usage(attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Preserve rejected responses while accounting for every attempted call."""

    input_tokens = 0
    output_tokens = 0
    cached_tokens = 0
    reasoning_effort: str | None = None
    for attempt in attempts:
        usage = attempt.get("usage")
        if not isinstance(usage, Mapping):
            continue
        input_value = usage.get("input_tokens")
        output_value = usage.get("output_tokens")
        input_tokens += int(input_value) if isinstance(input_value, (int, float)) else 0
        output_tokens += int(output_value) if isinstance(output_value, (int, float)) else 0
        details = usage.get("input_tokens_details")
        if isinstance(details, Mapping):
            cached_value = details.get("cached_tokens")
            cached_tokens += (
                int(cached_value) if isinstance(cached_value, (int, float)) else 0
            )
        effort = usage.get("reasoning_effort")
        if isinstance(effort, str):
            reasoning_effort = effort
    result: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens},
        "schema_attempt_count": len(attempts),
        "schema_attempts": list(attempts),
    }
    if reasoning_effort is not None:
        result["reasoning_effort"] = reasoning_effort
    return result


def _image_path_to_data_url(path: str | Path) -> str:
    value = Path(path)
    try:
        encoded = base64.b64encode(value.read_bytes()).decode("ascii")
    except OSError as exc:
        raise AgentProviderError(f"could not read image: {value}") from exc
    suffix = value.suffix.lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(
        suffix, "image/png"
    )
    return f"data:{mime};base64,{encoded}"


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RefineContractError(f"{key} must be a non-empty string")
    return value.strip()


def _unit_number(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
        raise RefineContractError(f"{key} must be numeric in [0, 1]")
    return float(value)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RefineContractError("expected a list of strings")
    return tuple(value)


EDIT_PLAN_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version", "case_id", "normalized_intent", "primitive_id",
        "source_labels", "target_label", "area_budget", "candidate_interfaces",
        "tool_program", "hard_invariants", "uncertainties", "planner_confidence",
        "escalation_reason"
    ],
    "properties": {
        "schema_version": {"type": "string"},
        "case_id": {"type": "string"},
        "normalized_intent": {"type": "string"},
        "primitive_id": {"type": "string"},
        "source_labels": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "target_label": {"type": "string"},
        "area_budget": {
            "type": "object", "additionalProperties": False,
            "required": ["target_fraction", "min_fraction", "max_fraction", "basis", "relative_tolerance", "fallback_policy"],
            "properties": {
                "target_fraction": {"type": "number"}, "min_fraction": {"type": "number"},
                "max_fraction": {"type": "number"}, "basis": {"type": "string"},
                "relative_tolerance": {"type": "number"},
                "fallback_policy": {
                    "type": "string",
                    "enum": ["exact", "max_feasible_below_target"]
                }
            }
        },
        "candidate_interfaces": {
            "type": "array", "minItems": 1,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["interface_id", "source_component_id", "target_component_id", "anchor_segment", "allowed_edit_band_px", "execution_contract", "prohibited_region_ids", "supporting_rule_ids", "expected_morphology", "confidence"],
                "properties": {
                    "interface_id": {"type": "string"}, "source_component_id": {"type": "string"},
                    "target_component_id": {"type": "string"}, "anchor_segment": {"type": "string"},
                    "allowed_edit_band_px": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                    "execution_contract": {
                        "type": "object", "additionalProperties": False,
                        "required": ["anchor_segment_ids", "area_allocation_fraction", "depth_profile", "min_anchor_coverage_fraction", "max_off_anchor_contact_fraction", "allocation_tolerance_fraction"],
                        "properties": {
                            "anchor_segment_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                            "area_allocation_fraction": {"type": "number"},
                            "depth_profile": {
                                "type": "object", "additionalProperties": False,
                                "required": ["mode", "peak_depth_px", "edge_depth_px", "taper_fraction", "lobe_count", "noise_amplitude_px", "noise_correlation_px"],
                                "properties": {
                                    "mode": {"type": "string", "enum": ["tapered_lobe", "uniform_front", "multi_lobe"]},
                                    "peak_depth_px": {"type": "number"},
                                    "edge_depth_px": {"type": "number"},
                                    "taper_fraction": {"type": "number"},
                                    "lobe_count": {"type": "integer"},
                                    "noise_amplitude_px": {"type": "number"},
                                    "noise_correlation_px": {"type": "number"}
                                }
                            },
                            "min_anchor_coverage_fraction": {"type": "number"},
                            "max_off_anchor_contact_fraction": {"type": "number"},
                            "allocation_tolerance_fraction": {"type": "number"}
                        }
                    },
                    "prohibited_region_ids": {"type": "array", "items": {"type": "string"}},
                    "supporting_rule_ids": {"type": "array", "items": {"type": "string"}},
                    "expected_morphology": {"type": "string"}, "confidence": {"type": "number"}
                }
            }
        },
        "tool_program": {
            "type": "object", "additionalProperties": False,
            "required": ["allowed_tools", "parameter_ranges", "candidate_count"],
            "properties": {
                "allowed_tools": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "parameter_ranges": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "max_changed_components",
                        "min_component_area_px",
                        "max_depth_span_ratio",
                        "max_bbox_fill_fraction",
                        "max_boundary_compactness",
                        "max_source_component_changed_fraction",
                        "min_source_component_remaining_px",
                        "min_parallel_front_depth_cv",
                        "parallel_front_linearity_ratio",
                        "parallel_front_min_depth_px",
                        "parallel_front_min_pixels"
                    ],
                    "properties": {
                        "max_changed_components": {"type": "integer"},
                        "min_component_area_px": {"type": "integer"},
                        "max_depth_span_ratio": {"type": "number"},
                        "max_bbox_fill_fraction": {"type": "number"},
                        "max_boundary_compactness": {"type": "number"},
                        "max_source_component_changed_fraction": {"type": "number"},
                        "min_source_component_remaining_px": {"type": "integer"},
                        "min_parallel_front_depth_cv": {"type": "number"},
                        "parallel_front_linearity_ratio": {"type": "number"},
                        "parallel_front_min_depth_px": {"type": "number"},
                        "parallel_front_min_pixels": {"type": "integer"}
                    }
                },
                "candidate_count": {"type": "integer"}
            }
        },
        "hard_invariants": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
        "planner_confidence": {"type": "number"},
        "escalation_reason": {"type": ["string", "null"]}
    }
}


CRITIC_JSON_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["rankings", "abstain", "summary"],
    "properties": {
        "rankings": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["candidate_id", "score", "confidence", "supporting_rule_ids", "veto_reasons"],
                "properties": {
                    "candidate_id": {"type": "string"}, "score": {"type": "number"},
                    "confidence": {"type": "number"},
                    "supporting_rule_ids": {"type": "array", "items": {"type": "string"}},
                    "veto_reasons": {"type": "array", "items": {"type": "string"}}
                }
            }
        },
        "abstain": {"type": "boolean"}, "summary": {"type": "string"}
    }
}

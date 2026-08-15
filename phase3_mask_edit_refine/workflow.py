"""Architecture B orchestration: plan, generate, gate, criticize, or abstain."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from phase3_mask_edit_refine.agents import (
    BREAST_EXECUTION_DOMAIN_ID,
    BREAST_EXECUTION_PROFILE_ID,
    Critic,
    Planner,
    critic_satisfies_hard_rules,
    validate_edit_plan,
)
from phase3_mask_edit_refine.audit import AuditWriter
from phase3_mask_edit_refine.candidates import generate_candidates
from phase3_mask_edit_refine.cost import summarize_cost
from phase3_mask_edit_refine.evidence import load_id_mask, sha256_file
from phase3_mask_edit_refine.execution import compile_edit_plan
from phase3_mask_edit_refine.gates import GateContext, GateRegistry
from phase3_mask_edit_refine.models import (
    CandidateMask,
    CaseContext,
    CriticResult,
    EditPlan,
    GateReport,
    RefineContractError,
    SceneGraph,
    WorkflowResult,
)
from phase3_mask_edit_refine.scene import SceneAnalysis, build_scene_analysis
from phase3_mask_edit_refine.skills import ActiveKnowledgeBundle, SkillRepository
from phase3_mask_edit_refine.visualization import (
    save_critic_contact_sheet,
    save_mask_planner_panels,
    save_planner_panels,
)


@dataclass
class EscalationBudget:
    """Batch-shared Sol escalation budget with an initial one-case allowance."""

    max_fraction: float = 0.10
    cases_seen: int = 0
    cases_escalated: int = 0
    escalated_case_ids: set[str] = field(default_factory=set, repr=False)

    def register_case(self) -> None:
        self.cases_seen += 1

    def can_escalate(self) -> bool:
        allowed = max(1, int(np.floor(self.cases_seen * self.max_fraction)))
        return self.cases_escalated < allowed

    def consume(self, *, case_id: str) -> bool:
        if case_id in self.escalated_case_ids:
            return False
        if not self.can_escalate():
            return False
        self.cases_escalated += 1
        self.escalated_case_ids.add(case_id)
        return True

    def to_metadata(self) -> dict[str, Any]:
        return {
            "max_fraction": self.max_fraction,
            "cases_seen": self.cases_seen,
            "cases_escalated": self.cases_escalated,
            "escalated_case_ids": sorted(self.escalated_case_ids),
        }


@dataclass(frozen=True)
class WorkflowConfig:
    production: bool = True
    planner_confidence_threshold: float = 0.70
    critic_confidence_threshold: float = 0.70
    critic_min_score_margin: float = 0.10


@dataclass(frozen=True)
class _AttemptArtifacts:
    plan: EditPlan
    candidates: tuple[CandidateMask, ...]
    gate_reports: tuple[GateReport, ...]
    planner_usage: dict[str, Any]


class MaskEditRefineWorkflow:
    def __init__(
        self,
        *,
        planner: Planner,
        critic: Critic,
        skill_repository: SkillRepository | None = None,
        gate_registry: GateRegistry | None = None,
        escalation_planner: Planner | None = None,
        escalation_critic: Critic | None = None,
        escalation_budget: EscalationBudget | None = None,
        config: WorkflowConfig | None = None,
    ) -> None:
        self.planner = planner
        self.critic = critic
        self.escalation_planner = escalation_planner
        self.escalation_critic = escalation_critic or critic
        self.skills = skill_repository or SkillRepository()
        self.gates = gate_registry or GateRegistry()
        self.escalation_budget = escalation_budget or EscalationBudget()
        self.config = config or WorkflowConfig()

    def run(self, case: CaseContext, *, output_root: str | Path) -> WorkflowResult:
        self.escalation_budget.register_case()
        audit = AuditWriter(output_root, case_id=case.case_id)
        source_mask: np.ndarray | None = None
        scene: SceneAnalysis | None = None
        bundle: ActiveKnowledgeBundle | None = None
        final_plan: EditPlan | None = None
        final_reports: tuple[GateReport, ...] = ()
        final_critic: CriticResult | None = None
        candidates: tuple[CandidateMask, ...] = ()
        usage: dict[str, Any] = {
            "escalation": self.escalation_budget.to_metadata(),
            "planner_calls": [],
            "critic_calls": [],
        }
        reasons: list[str] = []

        try:
            case.validate_local_inputs()
            source_mask = load_id_mask(case.source_mask_uri)
            _validate_source_digests(case)
            _validate_image_dimensions(case.source_image_uri, source_mask.shape)
            schema = self.skills.annotation_schema(case.annotation_profile_id)
            bundle = self.skills.compose(
                pathology_domain_id=case.pathology_domain_id,
                annotation_profile_id=case.annotation_profile_id,
                primitive_id=case.primitive_id,
                production=self.config.production,
                available_checker_ids=self.gates.available_checker_ids,
            )
            scene = build_scene_analysis(
                source_mask,
                schema=schema,
                pixel_size_um=case.pixel_size_um,
            )
            audit.write_inputs(
                case=case,
                source_mask=source_mask,
                scene_graph=scene.graph,
                bundle=bundle,
            )
            non_breast_mask_only = (
                case.pathology_domain_id != BREAST_EXECUTION_DOMAIN_ID
                or case.annotation_profile_id != BREAST_EXECUTION_PROFILE_ID
            )
            planner_panels = (
                save_mask_planner_panels(
                    mask=source_mask,
                    scene=scene,
                    output_dir=audit.case_dir / "planner_panels",
                )
                if non_breast_mask_only
                else save_planner_panels(
                    image_path=case.source_image_uri,
                    mask=source_mask,
                    scene=scene,
                    output_dir=audit.case_dir / "planner_panels",
                )
            )
            execution_planner_images = (
                () if non_breast_mask_only else planner_panels
            )
            audit.paths["planner_panels"] = str(audit.case_dir / "planner_panels")

            attempt = self._plan_generate_gate(
                planner=self.planner,
                case=case,
                source_mask=source_mask,
                scene=scene,
                bundle=bundle,
                image_paths=execution_planner_images,
            )
            usage["planner_calls"].append(attempt.planner_usage)
            escalation_reason = self._attempt_escalation_reason(attempt)
            if escalation_reason and self.escalation_planner is not None:
                escalated = self._try_escalation(
                    reason=escalation_reason,
                    case=case,
                    source_mask=source_mask,
                    scene=scene,
                    bundle=bundle,
                    image_paths=execution_planner_images,
                    usage=usage,
                )
                if escalated is not None:
                    attempt = escalated

            final_plan = attempt.plan
            candidates = attempt.candidates
            final_reports = attempt.gate_reports
            audit.write_plan(final_plan, usage=attempt.planner_usage)
            audit.write_candidates(candidates)
            audit.write_gate_reports(final_reports)
            passing = [
                candidate
                for candidate in candidates
                if next(
                    report.passed
                    for report in final_reports
                    if report.candidate_id == candidate.candidate_id
                )
            ]
            if not passing:
                reasons.append("no_candidate_passed_deterministic_gates")
                return self._finish_abstain(
                    audit=audit,
                    case=case,
                    source_mask=source_mask,
                    scene=scene.graph,
                    plan=final_plan,
                    reports=final_reports,
                    critic=None,
                    reasons=reasons,
                    usage=usage,
                )

            contact_sheet = save_critic_contact_sheet(
                image_path=(
                    None if non_breast_mask_only else case.source_image_uri
                ),
                source_mask=source_mask,
                candidates=candidates,
                gate_reports=final_reports,
                scene=scene,
                output_path=audit.case_dir / "critic_contact_sheet.png",
            )
            audit.paths["critic_contact_sheet"] = contact_sheet
            critic_images = (
                ()
                if non_breast_mask_only
                else (*planner_panels[:2], contact_sheet)
            )
            final_critic = self.critic.review(
                case=case,
                bundle=bundle,
                candidates=passing,
                gate_reports=final_reports,
                image_paths=critic_images,
            )
            usage["critic_calls"].append(final_critic.usage)
            audit.write_critic(final_critic)
            critic_ok, critic_reasons = critic_satisfies_hard_rules(
                final_critic,
                bundle=bundle,
                minimum_confidence=self.config.critic_confidence_threshold,
            )
            score_margin_ok = _critic_margin_ok(
                final_critic, minimum=self.config.critic_min_score_margin
            )
            if not critic_ok or not score_margin_ok:
                reasons.extend(critic_reasons)
                if not score_margin_ok:
                    reasons.append("critic_top_score_margin_too_small")
                escalated = None
                if self.escalation_planner is not None:
                    escalated = self._try_escalation(
                        reason="critic_disagreement_or_low_confidence",
                        case=case,
                        source_mask=source_mask,
                        scene=scene,
                        bundle=bundle,
                        image_paths=execution_planner_images,
                        usage=usage,
                    )
                if escalated is not None:
                    final_plan = escalated.plan
                    candidates = escalated.candidates
                    final_reports = escalated.gate_reports
                    audit.write_plan(final_plan, usage=escalated.planner_usage)
                    audit.write_candidates(candidates)
                    audit.write_gate_reports(final_reports)
                    passing = [
                        item
                        for item in candidates
                        if next(
                            report.passed
                            for report in final_reports
                            if report.candidate_id == item.candidate_id
                        )
                    ]
                    if passing:
                        contact_sheet = save_critic_contact_sheet(
                            image_path=(
                                None
                                if non_breast_mask_only
                                else case.source_image_uri
                            ),
                            source_mask=source_mask,
                            candidates=candidates,
                            gate_reports=final_reports,
                            scene=scene,
                            output_path=audit.case_dir / "critic_contact_sheet_escalated.png",
                        )
                        final_critic = self.escalation_critic.review(
                            case=case,
                            bundle=bundle,
                            candidates=passing,
                            gate_reports=final_reports,
                            image_paths=(
                                ()
                                if non_breast_mask_only
                                else (*planner_panels[:2], contact_sheet)
                            ),
                        )
                        usage["critic_calls"].append(final_critic.usage)
                        audit.write_critic(final_critic)
                        critic_ok, critic_reasons = critic_satisfies_hard_rules(
                            final_critic,
                            bundle=bundle,
                            minimum_confidence=self.config.critic_confidence_threshold,
                        )
                        score_margin_ok = _critic_margin_ok(
                            final_critic,
                            minimum=self.config.critic_min_score_margin,
                        )
                        reasons = [] if critic_ok and score_margin_ok else list(critic_reasons)

            if final_critic is None or not final_critic.rankings:
                reasons.append("critic_returned_no_rankings")
            elif (not critic_ok or not score_margin_ok) and not reasons:
                reasons.append("critic_did_not_approve_a_candidate")
            if reasons:
                return self._finish_abstain(
                    audit=audit,
                    case=case,
                    source_mask=source_mask,
                    scene=scene.graph,
                    plan=final_plan,
                    reports=final_reports,
                    critic=final_critic,
                    reasons=reasons,
                    usage=usage,
                )

            selected_id = final_critic.rankings[0].candidate_id
            selected = next(item for item in candidates if item.candidate_id == selected_id)
            status = "selected" if self.config.production else "selected_research"
            usage["escalation"] = self.escalation_budget.to_metadata()
            usage["cost"] = summarize_cost(
                [*usage["planner_calls"], *usage["critic_calls"]]
            )
            audit.write_selection(
                status=status,
                selected_candidate_id=selected_id,
                abstain_reasons=(),
                target_mask=selected.target_mask,
                source_mask=source_mask,
                usage=usage,
            )
            return WorkflowResult(
                status=status,
                case_context=case,
                scene_graph=scene.graph,
                edit_plan=final_plan,
                gate_reports=final_reports,
                critic_result=final_critic,
                selected_candidate_id=selected_id,
                target_mask=selected.target_mask,
                abstain_reasons=(),
                artifact_paths=dict(audit.paths),
                usage=usage,
            )
        # Fail closed at the orchestration boundary: every unexpected tool,
        # model, evidence, or artifact error must produce an audited abstention.
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"{type(exc).__name__}: {exc}")
            if source_mask is None:
                # Preserve a machine-readable failure even when the input mask cannot load.
                source_mask = np.zeros((1, 1), dtype=np.uint8)
            return self._finish_abstain(
                audit=audit,
                case=case,
                source_mask=source_mask,
                scene=scene.graph if scene else None,
                plan=final_plan,
                reports=final_reports,
                critic=final_critic,
                reasons=reasons,
                usage=usage,
            )

    def _plan_generate_gate(
        self,
        *,
        planner: Planner,
        case: CaseContext,
        source_mask: np.ndarray,
        scene: SceneAnalysis,
        bundle: ActiveKnowledgeBundle,
        image_paths: Sequence[str | Path],
    ) -> _AttemptArtifacts:
        raw_plan, planner_usage = planner.create_plan(
            case=case,
            scene=scene,
            bundle=bundle,
            image_paths=image_paths,
        )
        validate_edit_plan(raw_plan, case=case, scene=scene, bundle=bundle)
        schema = self.skills.annotation_schema(case.annotation_profile_id)
        plan, compiler_usage = compile_edit_plan(
            raw_plan,
            source_mask=source_mask,
            schema=schema,
            scene=scene,
        )
        validate_edit_plan(plan, case=case, scene=scene, bundle=bundle)
        planner_usage = {
            **planner_usage,
            "execution_compiler": compiler_usage,
            "raw_edit_plan": raw_plan.to_metadata(),
        }
        candidates = generate_candidates(
            source_mask,
            schema=schema,
            scene=scene,
            plan=plan,
            bundle=bundle,
            seed=case.seed,
        )
        reports = tuple(
            self.gates.run(
                GateContext(
                    case=case,
                    source_mask=source_mask,
                    schema=schema,
                    scene=scene,
                    bundle=bundle,
                    plan=plan,
                    candidate=candidate,
                )
            )
            for candidate in candidates
        )
        return _AttemptArtifacts(plan, candidates, reports, planner_usage)

    def _attempt_escalation_reason(self, attempt: _AttemptArtifacts) -> str | None:
        if attempt.plan.planner_confidence < self.config.planner_confidence_threshold:
            return "planner_low_confidence"
        if not any(report.passed for report in attempt.gate_reports):
            return "all_candidates_failed_deterministic_gates"
        return None

    def _try_escalation(
        self,
        *,
        reason: str,
        case: CaseContext,
        source_mask: np.ndarray,
        scene: SceneAnalysis,
        bundle: ActiveKnowledgeBundle,
        image_paths: Sequence[str | Path],
        usage: dict[str, Any],
    ) -> _AttemptArtifacts | None:
        if self.escalation_planner is None:
            return None
        if not self.escalation_budget.consume(case_id=case.case_id):
            usage.setdefault("escalation_denied", []).append(reason)
            return None
        attempt = self._plan_generate_gate(
            planner=self.escalation_planner,
            case=case,
            source_mask=source_mask,
            scene=scene,
            bundle=bundle,
            image_paths=image_paths,
        )
        usage["planner_calls"].append({"escalation_reason": reason, **attempt.planner_usage})
        usage["escalation"] = self.escalation_budget.to_metadata()
        usage["cost"] = summarize_cost(
            [*usage.get("planner_calls", []), *usage.get("critic_calls", [])]
        )
        return attempt

    def _finish_abstain(
        self,
        *,
        audit: AuditWriter,
        case: CaseContext,
        source_mask: np.ndarray,
        scene: SceneGraph | None,
        plan: EditPlan | None,
        reports: tuple[GateReport, ...],
        critic: CriticResult | None,
        reasons: Sequence[str],
        usage: dict[str, Any],
    ) -> WorkflowResult:
        usage["escalation"] = self.escalation_budget.to_metadata()
        audit.write_selection(
            status="abstained",
            selected_candidate_id=None,
            abstain_reasons=reasons,
            target_mask=None,
            source_mask=source_mask,
            usage=usage,
        )
        return WorkflowResult(
            status="abstained",
            case_context=case,
            scene_graph=scene,
            edit_plan=plan,
            gate_reports=reports,
            critic_result=critic,
            selected_candidate_id=None,
            target_mask=None,
            abstain_reasons=tuple(reasons),
            artifact_paths=dict(audit.paths),
            usage=usage,
        )


def _validate_source_digests(case: CaseContext) -> None:
    required = {
        "source_image_sha256": case.source_image_uri,
        "source_mask_sha256": case.source_mask_uri,
    }
    missing = [key for key in required if not case.provenance.get(key)]
    if missing:
        raise RefineContractError("source provenance missing digests: " + ", ".join(missing))
    mismatches: list[str] = []
    for key, path in required.items():
        if "://" in path and not path.startswith("file://"):
            continue
        observed = sha256_file(path.removeprefix("file://"))
        expected = str(case.provenance[key]).lower()
        if observed.lower() != expected:
            mismatches.append(key)
    if mismatches:
        raise RefineContractError("source digest mismatch: " + ", ".join(mismatches))


def _validate_image_dimensions(image_path: str | Path, mask_shape: tuple[int, int]) -> None:
    with Image.open(image_path) as image:
        size = image.size
    expected = (mask_shape[1], mask_shape[0])
    if size != expected:
        raise RefineContractError(f"image/mask size mismatch: image={size}, mask={expected}")


def _critic_margin_ok(result: CriticResult, *, minimum: float) -> bool:
    if result.abstain or not result.rankings:
        return False
    if len(result.rankings) == 1:
        return True
    ordered = sorted(result.rankings, key=lambda item: (-item.score, item.candidate_id))
    return ordered[0].score - ordered[1].score >= minimum

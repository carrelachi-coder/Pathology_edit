"""Transactional orchestration for ordered pathology mask-edit programs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from phase3_mask_edit.core.mask_io import save_id_mask
from phase3_mask_edit_refine.evidence import sha256_file

from .models import (
    JointCaseContext,
    JointContractError,
    JointCriticRanking,
    JointCriticResult,
    JointWorkflowResult,
)
from .nuclei import to_raw_nuclei_mask
from .program_planner import (
    EditProgram,
    EditProgramStep,
    SemanticProgramPlanner,
    bind_program_step_selection,
    legacy_semantic_intent_for_step,
)
from .semantic_request import (
    SEMANTIC_CELL_CLASS_IDS,
    SemanticRequest,
    SemanticRequestParser,
)
from .workflow import JointPathologyEditWorkflow


PROGRAM_RUN_SCHEMA_VERSION = "joint-edit-program-run-v1"


@dataclass(frozen=True)
class ProgramStepAudit:
    step_id: str
    intent_id: str
    status: str
    primitive_id: str | None
    mechanism_id: str | None
    input_tissue_sha256: str
    input_nuclei_sha256: str
    output_tissue_sha256: str | None
    output_nuclei_sha256: str | None
    selected_candidate_id: str | None
    workflow_status: str
    workflow_artifact_paths: dict[str, str]
    reasons: tuple[str, ...] = ()
    clarification_request: dict[str, Any] | None = None

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProgramRunResult:
    status: str
    semantic_request: SemanticRequest
    edit_program: EditProgram
    step_audits: tuple[ProgramStepAudit, ...]
    final_case_context: JointCaseContext
    artifact_paths: dict[str, str]
    evaluation: dict[str, Any]
    schema_version: str = PROGRAM_RUN_SCHEMA_VERSION

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "semantic_request_sha256": self.semantic_request.request_sha256,
            "edit_program_sha256": self.edit_program.program_sha256,
            "steps": [item.to_metadata() for item in self.step_audits],
            "final_case_id": self.final_case_context.case_id,
            "artifact_paths": dict(self.artifact_paths),
            "evaluation": dict(self.evaluation),
        }


@dataclass(frozen=True)
class DeterministicMaskProgramEvaluator:
    """Approve only compiler candidates whose required code gates all pass.

    This evaluator does not inspect free-form pathology semantics and makes no
    clinical realism claim.  It is the deterministic commit authority for a
    mask program.  Human or multimodal visual review remains a separate,
    optional downstream audit and can never override a failed hard gate.
    """

    name: str = "deterministic_mask_program_evaluator_v1"
    supports_pathology_vision: bool = False

    def review(
        self,
        *,
        case,
        bundle,
        candidates,
        gate_reports,
        image_paths,
        artifact_registry=None,
    ) -> JointCriticResult:
        del image_paths, artifact_registry
        reports = {item.candidate_id: item for item in gate_reports}
        rankings: list[JointCriticRanking] = []
        for candidate in candidates:
            report = reports.get(candidate.candidate_id)
            if report is None or not report.passed:
                continue
            if bundle.primitive.budget_mode == "count_extent":
                budget = case.cell_count_extent_budget
                desired = int(budget.target_delta_count if budget is not None else 0)
                actual = int(candidate.tool_trace.get("placed_count", desired))
                error = abs(actual - desired) / max(1, desired)
            else:
                budget = case.joint_area_budget
                desired = float(budget.target_fraction if budget is not None else 0.0)
                actual = float(candidate.ledger.joint_fraction)
                error = abs(actual - desired) / max(desired, 1e-6)
            rankings.append(
                JointCriticRanking(
                    candidate_id=candidate.candidate_id,
                    score=max(0.0, 1.0 - error),
                    confidence=1.0,
                    supporting_rule_ids=bundle.active_rule_ids,
                    veto_reasons=(),
                )
            )
        rankings.sort(key=lambda item: (-item.score, item.candidate_id))
        return JointCriticResult(
            rankings=tuple(rankings),
            abstain=not rankings,
            summary=(
                "candidate selection is owned by deterministic hard gates and "
                "budget error; no LLM evaluator participates in mask acceptance"
            ),
            usage={
                "provider": self.name,
                "deterministic": True,
                "input_tokens": 0,
                "output_tokens": 0,
            },
        )

    def evaluate_program(
        self,
        *,
        program: EditProgram,
        step_audits: Sequence[ProgramStepAudit],
    ) -> dict[str, Any]:
        errors: list[str] = []
        audit_by_step = {item.step_id: item for item in step_audits}
        previous: ProgramStepAudit | None = None
        for step in program.steps:
            audit = audit_by_step.get(step.step_id)
            if audit is None:
                if step.status == "validated":
                    errors.append(f"{step.step_id}: validated step has no audit")
                continue
            if audit.status == "validated":
                if audit.output_tissue_sha256 is None or audit.output_nuclei_sha256 is None:
                    errors.append(f"{step.step_id}: validated step has no output digest")
                if previous is not None and previous.status == "validated":
                    if audit.input_tissue_sha256 != previous.output_tissue_sha256:
                        errors.append(f"{step.step_id}: tissue digest chain is detached")
                    if audit.input_nuclei_sha256 != previous.output_nuclei_sha256:
                        errors.append(f"{step.step_id}: nuclei digest chain is detached")
                previous = audit
            elif any(
                later.status == "validated"
                for later in step_audits[step.order_index :]
            ):
                errors.append(
                    f"{step.step_id}: a later step ran after this step failed or paused"
                )
        completed = sum(item.status == "validated" for item in step_audits)
        return {
            "evaluator": self.name,
            "deterministic": True,
            "passed": not errors and completed == len(program.steps),
            "completed_steps": completed,
            "required_steps": len(program.steps),
            "errors": errors,
            "claim_scope": "mask-contract execution only",
            "visual_pathology_approval": False,
        }


class SequentialEditProgramWorkflow:
    """Execute one validated primitive at a time on the latest accepted state."""

    def __init__(
        self,
        *,
        step_workflow: JointPathologyEditWorkflow,
        program_planner: SemanticProgramPlanner | None = None,
        evaluator: DeterministicMaskProgramEvaluator | None = None,
    ) -> None:
        self.step_workflow = step_workflow
        self.program_planner = program_planner or SemanticProgramPlanner()
        self.evaluator = evaluator or DeterministicMaskProgramEvaluator()

    def run(
        self,
        raw_case: Mapping[str, Any],
        *,
        semantic_parser: SemanticRequestParser,
        output_root: str | Path,
        production: bool = False,
        clarification_decisions: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> ProgramRunResult:
        instruction = str(raw_case.get("instruction") or "").strip()
        if not instruction:
            raise JointContractError("program input requires an instruction")
        request = semantic_parser.parse(instruction)
        template = _program_case_template(raw_case)
        program = self.program_planner.plan(
            request,
            case_template=template,
            production=production,
        )
        root = Path(output_root) / template.case_id
        root.mkdir(parents=True, exist_ok=True)
        paths = {
            "semantic_request": str(root / "semantic_request.json"),
            "initial_program": str(root / "edit_program.initial.json"),
            "final_program": str(root / "edit_program.final.json"),
            "program_result": str(root / "program_result.json"),
        }
        _write_json(Path(paths["semantic_request"]), request.to_metadata())
        _write_json(Path(paths["initial_program"]), program.to_metadata())

        if program.status == "clarification_required":
            evaluation = self.evaluator.evaluate_program(
                program=program, step_audits=()
            )
            return self._finish(
                root=root,
                paths=paths,
                request=request,
                program=program,
                audits=(),
                final_case=template,
                status="clarification_required",
                evaluation=evaluation,
            )

        current = template
        decisions = clarification_decisions or {}
        audits: list[ProgramStepAudit] = []
        updated_steps = list(program.steps)
        stopped_status: str | None = None
        for index, step in enumerate(program.steps):
            intent = next(
                item for item in request.intents if item.intent_id == step.intent_id
            )
            step_case = _bind_step_case(
                current,
                step=step,
                intent=intent,
                request=request,
                clarification_decision=decisions.get(step.intent_id),
            )
            step_root = root / "steps" / step.step_id
            result = self.step_workflow.run(step_case, output_root=step_root)
            audit = self._consume_step_result(
                root=root,
                step=step,
                result=result,
            )
            audits.append(audit)
            if audit.status != "validated":
                stopped_status = audit.status
                updated_steps[index] = replace(step, status=audit.status)
                for later_index in range(index + 1, len(updated_steps)):
                    updated_steps[later_index] = replace(
                        updated_steps[later_index], status="not_run"
                    )
                break
            selected_primitive = str(result.case_context.primitive_id)
            mechanism = str(result.condition.pathology_mechanism)
            updated_steps[index] = bind_program_step_selection(
                step,
                primitive_id=selected_primitive,
                mechanism_id=mechanism,
                validated=True,
            )
            current = _advance_case_state(
                result.case_context,
                audit=audit,
                source_image_uri=template.source_image_uri,
            )

        if stopped_status == "clarification_required":
            final_status = "clarification_required"
        elif stopped_status == "review_required":
            final_status = "review_required"
        elif stopped_status is not None:
            final_status = "partially_validated" if any(
                item.status == "validated" for item in audits
            ) else "failed"
        else:
            final_status = "validated"
        final_program = replace(
            program,
            steps=tuple(updated_steps),
            status=final_status,
        )
        evaluation = self.evaluator.evaluate_program(
            program=final_program,
            step_audits=audits,
        )
        if final_status == "validated" and not evaluation["passed"]:
            final_status = "failed"
            final_program = replace(final_program, status="failed")
        return self._finish(
            root=root,
            paths=paths,
            request=request,
            program=final_program,
            audits=tuple(audits),
            final_case=current,
            status=final_status,
            evaluation=evaluation,
        )

    def _consume_step_result(
        self,
        *,
        root: Path,
        step: EditProgramStep,
        result: JointWorkflowResult,
    ) -> ProgramStepAudit:
        input_tissue = result.case_context.source_tissue_mask_uri
        input_nuclei = result.case_context.source_nuclei_mask_uri
        if result.status not in {"selected", "selected_research"}:
            status = (
                "clarification_required"
                if result.status == "clarification_required"
                else "review_required"
                if result.status == "review_required"
                else "failed"
            )
            return ProgramStepAudit(
                step_id=step.step_id,
                intent_id=step.intent_id,
                status=status,
                primitive_id=result.case_context.primitive_id,
                mechanism_id=(
                    result.joint_plan.selected_mechanism_id
                    if result.joint_plan is not None
                    else None
                ),
                input_tissue_sha256=sha256_file(input_tissue),
                input_nuclei_sha256=sha256_file(input_nuclei),
                output_tissue_sha256=None,
                output_nuclei_sha256=None,
                selected_candidate_id=result.selected_candidate_id,
                workflow_status=result.status,
                workflow_artifact_paths=dict(result.artifact_paths),
                reasons=tuple(result.abstain_reasons),
                clarification_request=result.clarification_request,
            )
        if result.condition is None:
            raise JointContractError("selected workflow result has no condition")
        selected_reports = tuple(
            item
            for item in result.gate_reports
            if item.candidate_id == result.selected_candidate_id
        )
        if not selected_reports or not all(item.passed for item in selected_reports):
            raise JointContractError(
                "program step cannot commit a candidate without passing code gates"
            )
        state_dir = root / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        tissue_path = save_id_mask(
            result.condition.target_tissue_mask,
            state_dir / f"{step.step_id}.tissue.png",
        )
        nuclei_path = save_id_mask(
            to_raw_nuclei_mask(result.condition.target_nuclei_mask),
            state_dir / f"{step.step_id}.nuclei.png",
        )
        return ProgramStepAudit(
            step_id=step.step_id,
            intent_id=step.intent_id,
            status="validated",
            primitive_id=result.case_context.primitive_id,
            mechanism_id=result.condition.pathology_mechanism,
            input_tissue_sha256=sha256_file(input_tissue),
            input_nuclei_sha256=sha256_file(input_nuclei),
            output_tissue_sha256=sha256_file(tissue_path),
            output_nuclei_sha256=sha256_file(nuclei_path),
            selected_candidate_id=result.selected_candidate_id,
            workflow_status=result.status,
            workflow_artifact_paths={
                **dict(result.artifact_paths),
                "committed_tissue_mask": str(tissue_path),
                "committed_nuclei_mask": str(nuclei_path),
            },
        )

    def _finish(
        self,
        *,
        root: Path,
        paths: dict[str, str],
        request: SemanticRequest,
        program: EditProgram,
        audits: tuple[ProgramStepAudit, ...],
        final_case: JointCaseContext,
        status: str,
        evaluation: dict[str, Any],
    ) -> ProgramRunResult:
        _write_json(Path(paths["final_program"]), program.to_metadata())
        result = ProgramRunResult(
            status=status,
            semantic_request=request,
            edit_program=program,
            step_audits=audits,
            final_case_context=final_case,
            artifact_paths=paths,
            evaluation=evaluation,
        )
        _write_json(Path(paths["program_result"]), result.to_metadata())
        return result


def _program_case_template(raw_case: Mapping[str, Any]) -> JointCaseContext:
    payload = dict(raw_case)
    payload.setdefault("primitive_id", "cohesive-boundary-expansion-v1")
    payload.pop("semantic_intent", None)
    payload.pop("clarification_decision", None)
    return JointCaseContext.from_mapping(payload)


def _bind_step_case(
    current: JointCaseContext,
    *,
    step: EditProgramStep,
    intent,
    request: SemanticRequest,
    clarification_decision: Mapping[str, Any] | None,
) -> JointCaseContext:
    if not step.candidates:
        raise JointContractError("program step has no primitive candidate")
    semantic_intent = legacy_semantic_intent_for_step(
        intent=intent,
        candidates=step.candidates,
    )
    provenance = dict(current.provenance)
    for key in (
        "joint_mechanism_id",
        "joint_primitive_id",
        "joint_interface_ids",
        "joint_anchor_ids",
        "joint_population_zone_id",
        "target_cell_class_ids",
        "target_cell_class_resolution",
    ):
        provenance.pop(key, None)
    provenance.update(
        {
            "semantic_request_sha256": request.request_sha256,
            "edit_program_step_id": step.step_id,
            "edit_program_intent_id": step.intent_id,
        }
    )
    if intent.cell_class is not None:
        resolved = SEMANTIC_CELL_CLASS_IDS.get(
            current.cell_observation_profile_id, {}
        ).get(intent.cell_class)
        if resolved is None:
            raise JointContractError(
                f"{current.cell_observation_profile_id} cannot distinguish "
                f"semantic cell class {intent.cell_class!r}"
            )
        provenance["target_cell_class_ids"] = list(resolved)
        provenance["target_cell_class_resolution"] = {
            "semantic_cell_class": intent.cell_class,
            "observation_profile_id": current.cell_observation_profile_id,
            "resolved_class_ids": list(resolved),
            "authority": "versioned_v4_observation_profile",
        }
    return replace(
        current,
        case_id=f"{current.case_id}--{step.step_id}",
        instruction=intent.source_text,
        primitive_id=step.candidates[0].primitive_id,
        provenance=provenance,
        semantic_intent=semantic_intent,
        clarification_decision=(
            dict(clarification_decision)
            if clarification_decision is not None
            else {}
        ),
        auxiliary_structure_uris=dict(current.auxiliary_structure_uris),
    )


def _advance_case_state(
    case: JointCaseContext,
    *,
    audit: ProgramStepAudit,
    source_image_uri: str,
) -> JointCaseContext:
    tissue_path = audit.workflow_artifact_paths["committed_tissue_mask"]
    nuclei_path = audit.workflow_artifact_paths["committed_nuclei_mask"]
    provenance = dict(case.provenance)
    provenance.update(
        {
            "source_tissue_mask_sha256": audit.output_tissue_sha256,
            "source_nuclei_mask_sha256": audit.output_nuclei_sha256,
            "original_label_map_digest": audit.output_tissue_sha256,
            "original_instance_mask_digest": audit.output_nuclei_sha256,
            "preprocessing_revision": (
                str(provenance.get("preprocessing_revision") or "unknown")
                + f"+{audit.step_id}"
            ),
            "available_auxiliary_structures": [],
        }
    )
    for key in (
        "source_nuclei_instances_sha256",
        "joint_mechanism_id",
        "joint_primitive_id",
        "joint_interface_ids",
        "joint_anchor_ids",
        "joint_population_zone_id",
        "auxiliary_structure_sha256",
        "auxiliary_structure_provenance",
    ):
        provenance.pop(key, None)
    return replace(
        case,
        source_image_uri=source_image_uri,
        source_tissue_mask_uri=tissue_path,
        source_nuclei_mask_uri=nuclei_path,
        source_nuclei_instances_uri=None,
        provenance=provenance,
        semantic_intent={},
        clarification_decision={},
        auxiliary_structure_uris={},
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

"""CLI for primitive-free single- and multi-intent mask-edit programs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase3_mask_edit_refine.agents import OpenAIResponsesJSONClient

from .agents import OpenAIMultimodalJointPlanner
from .mature_probnet_adapter import MatureProbNetCellExecutor, MatureProbNetConfig
from .probnet_adapter import FrozenProbNetSpatialRanker
from .program_planner import DeterministicProgramJointPlanner, SemanticProgramPlanner
from .program_workflow import (
    DeterministicMaskProgramEvaluator,
    SequentialEditProgramWorkflow,
)
from .semantic_request import (
    OpenAISemanticRequestParser,
    PreboundSemanticRequestParser,
    RuleBasedSemanticRequestParser,
)
from .skills.repository import JointSkillRepository
from .tissue_planner import MultiInterfaceResearchTissuePlanner, OpenAIJointAwareTissuePlanner
from .workflow import JointPathologyEditWorkflow, JointWorkflowConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse one natural-language request into ordered intents and execute "
            "one validated mask-edit primitive per intent"
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--semantic-parser",
        choices=("auto", "rule-based", "prebound", "api"),
        default="auto",
    )
    parser.add_argument("--agent-mode", choices=("offline", "api"), default="offline")
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--semantic-model", default="gpt-5.6-luna")
    parser.add_argument("--escalation-model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--semantic-reasoning-effort", default="low")
    parser.add_argument("--api-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--cell-executor", choices=("research", "mature"), default="research"
    )
    parser.add_argument("--probnet-checkpoint")
    parser.add_argument("--nuclei-instance-library")
    parser.add_argument("--probnet-dataset")
    parser.add_argument("--probnet-base-ch", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--meta-eval", action="store_true")
    parser.add_argument(
        "--clarification-decisions",
        help=(
            "Optional JSON mapping case_id -> intent_id -> digest-bound "
            "clarification decision"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.production and args.cell_executor != "mature":
        raise ValueError("production edit programs require --cell-executor mature")
    if args.meta_eval and args.cell_executor != "mature":
        raise ValueError("--meta-eval requires --cell-executor mature")
    if args.semantic_parser == "api" and args.agent_mode != "api":
        raise ValueError("--semantic-parser api requires --agent-mode api")
    if args.production and not (
        args.agent_mode == "api" and args.semantic_parser in {"auto", "api"}
    ):
        raise ValueError("production edit programs require the API v4 Parser")
    if args.cell_executor == "mature" and not all(
        (args.probnet_checkpoint, args.nuclei_instance_library, args.probnet_dataset)
    ):
        raise ValueError(
            "mature execution requires checkpoint, instance library and dataset"
        )

    records = _load_records(Path(args.manifest))
    decisions = (
        _load_decisions(Path(args.clarification_decisions))
        if args.clarification_decisions
        else {}
    )
    planner_client = escalation_client = semantic_client = None
    if args.agent_mode == "api":
        planner_client = OpenAIResponsesJSONClient(
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            api_base_url=args.api_base_url,
            api_key_env=args.api_key_env,
        )
        escalation_client = OpenAIResponsesJSONClient(
            model=args.escalation_model,
            reasoning_effort=args.reasoning_effort,
            api_base_url=args.api_base_url,
            api_key_env=args.api_key_env,
        )
        semantic_client = OpenAIResponsesJSONClient(
            model=args.semantic_model,
            reasoning_effort=args.semantic_reasoning_effort,
            api_base_url=args.api_base_url,
            api_key_env=args.api_key_env,
        )

    repository = JointSkillRepository()
    summaries = []
    for raw in records:
        case_id = str(raw.get("case_id") or "")
        if not case_id:
            raise ValueError("every program record requires case_id")
        if args.semantic_parser == "prebound":
            payload = raw.get("prebound_semantic_request")
            if not isinstance(payload, dict):
                raise ValueError(
                    "prebound Parser requires prebound_semantic_request per case"
                )
            semantic_parser = PreboundSemanticRequestParser(payload)
        elif args.semantic_parser == "api" or (
            args.semantic_parser == "auto" and semantic_client is not None
        ):
            semantic_parser = OpenAISemanticRequestParser(semantic_client)
        else:
            semantic_parser = RuleBasedSemanticRequestParser()

        population_id = str(raw.get("cell_population_profile_id") or "")
        population = repository.cell_population_profiles[population_id]
        ranker = None
        cell_executor = None
        if args.cell_executor == "mature":
            if args.probnet_dataset.casefold() != population.probnet_dataset_name.casefold():
                raise ValueError(
                    "--probnet-dataset does not match the case population profile"
                )
            cell_executor = MatureProbNetCellExecutor(
                MatureProbNetConfig(
                    dataset_name=args.probnet_dataset,
                    checkpoint=args.probnet_checkpoint,
                    instance_library=args.nuclei_instance_library,
                    device=args.device,
                    base_channels=args.probnet_base_ch,
                )
            )
            ranker = FrozenProbNetSpatialRanker.from_checkpoint(
                args.probnet_checkpoint,
                cancer_id=population.probnet_cancer_id,
                pathology_domain_id=str(raw["pathology_domain_id"]),
                device=args.device,
                base_channels=args.probnet_base_ch,
            )
        elif args.probnet_checkpoint:
            ranker = FrozenProbNetSpatialRanker.from_checkpoint(
                args.probnet_checkpoint,
                cancer_id=population.probnet_cancer_id,
                pathology_domain_id=str(raw["pathology_domain_id"]),
                device=args.device,
                base_channels=args.probnet_base_ch,
            )

        evaluator = DeterministicMaskProgramEvaluator()
        step_workflow = JointPathologyEditWorkflow(
            tissue_planner=(
                OpenAIJointAwareTissuePlanner(planner_client, escalation_client)
                if planner_client is not None
                else MultiInterfaceResearchTissuePlanner()
            ),
            joint_planner=(
                OpenAIMultimodalJointPlanner(planner_client, escalation_client)
                if planner_client is not None
                else DeterministicProgramJointPlanner()
            ),
            critic=evaluator,
            joint_skills=repository,
            ranker=ranker,
            cell_executor=cell_executor,
            config=JointWorkflowConfig(
                production=args.production,
                require_mature_probnet_for_target_population_regeneration=(
                    args.meta_eval
                ),
                require_probnet_ranker_for_cell_addition=args.meta_eval,
            ),
        )
        runner = SequentialEditProgramWorkflow(
            step_workflow=step_workflow,
            program_planner=SemanticProgramPlanner(),
            evaluator=evaluator,
        )
        result = runner.run(
            raw,
            semantic_parser=semantic_parser,
            output_root=args.output_root,
            production=args.production,
            clarification_decisions=decisions.get(case_id),
        )
        summaries.append(
            {
                "case_id": case_id,
                "status": result.status,
                "semantic_request_sha256": result.semantic_request.request_sha256,
                "edit_program_sha256": result.edit_program.program_sha256,
                "completed_steps": result.evaluation["completed_steps"],
                "required_steps": result.evaluation["required_steps"],
                "program_result": result.artifact_paths["program_result"],
            }
        )
    summary_path = Path(args.output_root) / "edit_program_run_summary.json"
    _write_json(summary_path, summaries)
    print(json.dumps({"cases": len(summaries), "summary": str(summary_path)}))
    if any(item["status"] == "failed" for item in summaries):
        return 2
    if any(
        item["status"] in {"clarification_required", "review_required"}
        for item in summaries
    ):
        return 3
    if any(item["status"] == "partially_validated" for item in summaries):
        return 4
    return 0


def _load_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.casefold() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        records = payload if isinstance(payload, list) else [payload]
    if not records or not all(isinstance(item, dict) for item in records):
        raise ValueError("program manifest must contain JSON objects")
    return records


def _load_decisions(path: Path) -> dict[str, dict[str, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("program clarification decisions must be a JSON object")
    result: dict[str, dict[str, dict]] = {}
    for case_id, intent_decisions in payload.items():
        if not isinstance(intent_decisions, dict):
            raise ValueError("each case decision must map intent IDs to decisions")
        result[str(case_id)] = {
            str(intent_id): dict(decision)
            for intent_id, decision in intent_decisions.items()
            if isinstance(decision, dict)
        }
    return result


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

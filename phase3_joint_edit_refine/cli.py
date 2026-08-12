"""Research CLI for generating auditable joint condition candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase3_mask_edit_refine.agents import (
    OpenAIResponsesJSONClient,
)

from .agents import OpenAIMultimodalJointCritic, OpenAIMultimodalJointPlanner
from .critic import DeterministicJointResearchCritic
from .mature_probnet_adapter import (
    MatureProbNetCellExecutor,
    MatureProbNetConfig,
)
from .planner import HeuristicJointPlanner
from .probnet_adapter import FrozenProbNetSpatialRanker
from .semantic_parser import (
    OpenAIClinicalScenarioParser,
    PreboundSemanticParser,
    RuleBasedSemanticParser,
    ScenarioClarificationRequired,
    bind_semantic_intent,
)
from .skills.repository import JointSkillRepository
from .tissue_planner import (
    MultiInterfaceResearchTissuePlanner,
    OpenAIJointAwareTissuePlanner,
)
from .workflow import JointPathologyEditWorkflow, JointWorkflowConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate joint tissue+nuclei candidates without changing the legacy pipeline")
    parser.add_argument("--manifest", required=True, help="JSON object/list or JSONL of JointCaseContext records")
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--clarification-decisions",
        help=(
            "Optional JSON object/list of digest-bound clarification decisions; "
            "each record must include case_id and clarification_decision"
        ),
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run only the named case; repeat to select several cases",
    )
    parser.add_argument("--production", action="store_true", help="Require internally reviewed skills; draft catalog will fail closed")
    parser.add_argument("--agent-mode", choices=("offline", "api"), default="offline")
    parser.add_argument(
        "--semantic-parser",
        choices=("auto", "rule-based", "prebound", "api"),
        default="auto",
        help=(
            "Parse the simple user instruction before visual mechanism planning; "
            "auto uses the API parser in api mode and the test-only deterministic "
            "parser offline. Codex-session shadows must use prebound."
        ),
    )
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--semantic-model", default="gpt-5.6-luna")
    parser.add_argument("--escalation-model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--semantic-reasoning-effort", default="low")
    parser.add_argument("--api-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--cell-executor",
        choices=("research", "mature"),
        default="research",
        help=(
            "Use the deterministic research layout executor or the unchanged "
            "mature online ProbNet regeneration pipeline"
        ),
    )
    parser.add_argument(
        "--probnet-checkpoint",
        help="Frozen ProbNet checkpoint for ranking or mature regeneration",
    )
    parser.add_argument(
        "--nuclei-instance-library",
        help="Mature source-first nucleus instance library directory",
    )
    parser.add_argument(
        "--probnet-dataset",
        help=(
            "Explicit mature-pipeline dataset configuration name; it is never "
            "inferred from pathology domain or annotation profile"
        ),
    )
    parser.add_argument("--probnet-base-ch", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--meta-eval",
        action="store_true",
        help=(
            "Fail closed unless target-population regeneration uses mature "
            "ProbNet and cell-only additions use its frozen spatial ranker"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.production and args.cell_executor != "mature":
        raise ValueError("production joint execution requires --cell-executor mature")
    if args.meta_eval and args.cell_executor != "mature":
        raise ValueError("Meta evaluation requires --cell-executor mature")
    if args.production and not (
        args.agent_mode == "api" and args.semantic_parser in {"auto", "api"}
    ):
        raise ValueError(
            "production natural-language execution requires the API semantic "
            "parser; use offline+prebound only for a Codex-session shadow"
        )
    if args.cell_executor == "mature" and not all(
        (
            args.probnet_checkpoint,
            args.nuclei_instance_library,
            args.probnet_dataset,
        )
    ):
        raise ValueError(
            "mature cell execution requires --probnet-checkpoint, "
            "--nuclei-instance-library and explicit --probnet-dataset"
        )
    records = _load_records(Path(args.manifest))
    decisions = (
        _load_clarification_decisions(Path(args.clarification_decisions))
        if args.clarification_decisions
        else {}
    )
    unknown_decisions = sorted(set(decisions) - {str(item.get("case_id")) for item in records})
    if unknown_decisions:
        raise ValueError(
            "clarification decisions name cases outside the manifest: "
            + ", ".join(unknown_decisions)
        )
    records = [
        {
            **item,
            **(
                {"clarification_decision": decisions[str(item.get("case_id"))]}
                if str(item.get("case_id")) in decisions
                else {}
            ),
        }
        for item in records
    ]
    if args.case_id:
        selected = set(args.case_id)
        records = [item for item in records if item.get("case_id") in selected]
        missing = selected.difference(item.get("case_id") for item in records)
        if missing:
            raise ValueError(f"case IDs not present in manifest: {sorted(missing)}")
    client = None
    escalation_client = None
    semantic_client = None
    if args.agent_mode == "api":
        client = OpenAIResponsesJSONClient(
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
    if args.semantic_parser == "api" and client is None:
        raise ValueError("--semantic-parser api requires --agent-mode api")
    summaries = []
    for raw in records:
        if args.semantic_parser == "prebound":
            prebound = raw.get("prebound_semantic_intent")
            if not isinstance(prebound, dict):
                raise ValueError(
                    "--semantic-parser prebound requires a frozen "
                    "prebound_semantic_intent on every case"
                )
            semantic_parser = PreboundSemanticParser(prebound)
        else:
            semantic_parser = (
                OpenAIClinicalScenarioParser(semantic_client)
                if args.semantic_parser == "api"
                or (args.semantic_parser == "auto" and client is not None)
                else RuleBasedSemanticParser()
            )
        try:
            case, semantic_intent = bind_semantic_intent(raw, semantic_parser)
        except ScenarioClarificationRequired as exc:
            case_id = str(raw.get("case_id") or "unknown")
            request_path = (
                Path(args.output_root)
                / case_id
                / "scenario_clarification_request.json"
            )
            request_path.parent.mkdir(parents=True, exist_ok=True)
            request_path.write_text(
                json.dumps(exc.request, indent=2, ensure_ascii=False, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            summaries.append(
                {
                    "case_id": case_id,
                    "status": "clarification_required",
                    "clarification_request": str(request_path.resolve()),
                    "clarification_request_sha256": exc.request["request_sha256"],
                }
            )
            continue
        population = repository.cell_population_profiles[
            case.cell_population_profile_id
        ]
        ranker = None
        cell_executor = None
        if args.cell_executor == "mature":
            if (
                args.probnet_dataset.lower()
                != population.probnet_dataset_name.lower()
            ):
                raise ValueError(
                    "--probnet-dataset does not match the case cell population profile: "
                    f"expected {population.probnet_dataset_name}, got {args.probnet_dataset}"
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
                pathology_domain_id=case.pathology_domain_id,
                device=args.device,
                base_channels=args.probnet_base_ch,
            )
        elif args.probnet_checkpoint:
            ranker = FrozenProbNetSpatialRanker.from_checkpoint(
                args.probnet_checkpoint,
                cancer_id=population.probnet_cancer_id,
                pathology_domain_id=case.pathology_domain_id,
                device=args.device,
                base_channels=args.probnet_base_ch,
            )
        workflow = JointPathologyEditWorkflow(
            tissue_planner=(
                OpenAIJointAwareTissuePlanner(client, escalation_client)
                if client
                else MultiInterfaceResearchTissuePlanner()
            ),
            joint_planner=(
                OpenAIMultimodalJointPlanner(client, escalation_client)
                if client
                else HeuristicJointPlanner()
            ),
            critic=(OpenAIMultimodalJointCritic(client) if client else DeterministicJointResearchCritic()),
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
        result = workflow.run(case, output_root=args.output_root)
        resolution_path = result.artifact_paths.get(
            "semantic_resolution.json"
        )
        semantic_resolution = (
            json.loads(Path(resolution_path).read_text(encoding="utf-8"))
            if resolution_path and Path(resolution_path).is_file()
            else None
        )
        summaries.append(
            {
                "case_id": case.case_id,
                "status": result.status,
                "semantic_intent": semantic_intent.to_metadata(),
                "semantic_request": semantic_intent.to_metadata(),
                "semantic_resolution": semantic_resolution,
                "selected_candidate_id": result.selected_candidate_id,
                "clarification_request": result.clarification_request,
                "abstain_reasons": list(result.abstain_reasons),
                "artifact_paths": result.artifact_paths,
            }
        )
    output = Path(args.output_root) / "joint_run_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"cases": len(summaries), "summary": str(output)}, sort_keys=True))
    if any(item["status"] == "abstained" for item in summaries):
        return 2
    if any(item["status"] == "clarification_required" for item in summaries):
        return 3
    return 0


def _load_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        records = payload if isinstance(payload, list) else [payload]
    if not records or not all(isinstance(item, dict) for item in records):
        raise ValueError("joint manifest must contain one or more JSON objects")
    return records


def _load_clarification_decisions(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "case_id" not in payload:
        records = [
            {"case_id": case_id, "clarification_decision": decision}
            for case_id, decision in payload.items()
        ]
    else:
        records = payload if isinstance(payload, list) else [payload]
    decisions: dict[str, dict] = {}
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("clarification decision record must be an object")
        case_id = str(record.get("case_id") or "").strip()
        decision = record.get("clarification_decision")
        if not case_id or not isinstance(decision, dict):
            raise ValueError(
                "clarification decision record requires case_id and clarification_decision"
            )
        if case_id in decisions:
            raise ValueError(f"duplicate clarification decision for {case_id}")
        decisions[case_id] = decision
    return decisions


if __name__ == "__main__":
    raise SystemExit(main())

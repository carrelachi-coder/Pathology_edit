"""Research CLI for generating auditable joint condition candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase3_mask_edit_refine.agents import (
    OpenAIMultimodalPlanner,
    OpenAIResponsesJSONClient,
)

from .agents import OpenAIMultimodalJointCritic, OpenAIMultimodalJointPlanner
from .critic import DeterministicJointResearchCritic
from .models import JointCaseContext
from .mature_probnet_adapter import (
    MatureProbNetCellExecutor,
    MatureProbNetConfig,
)
from .planner import HeuristicJointPlanner
from .probnet_adapter import FrozenProbNetSpatialRanker
from .skills.repository import JointSkillRepository
from .tissue_planner import MultiInterfaceResearchTissuePlanner
from .workflow import JointPathologyEditWorkflow, JointWorkflowConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate joint tissue+nuclei candidates without changing the legacy pipeline")
    parser.add_argument("--manifest", required=True, help="JSON object/list or JSONL of JointCaseContext records")
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run only the named case; repeat to select several cases",
    )
    parser.add_argument("--production", action="store_true", help="Require internally reviewed skills; draft catalog will fail closed")
    parser.add_argument("--agent-mode", choices=("offline", "api"), default="offline")
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--reasoning-effort", default="medium")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.production and args.cell_executor != "mature":
        raise ValueError("production joint execution requires --cell-executor mature")
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
    if args.case_id:
        selected = set(args.case_id)
        records = [item for item in records if item.get("case_id") in selected]
        missing = selected.difference(item.get("case_id") for item in records)
        if missing:
            raise ValueError(f"case IDs not present in manifest: {sorted(missing)}")
    client = None
    if args.agent_mode == "api":
        client = OpenAIResponsesJSONClient(
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            api_base_url=args.api_base_url,
            api_key_env=args.api_key_env,
        )
    repository = JointSkillRepository()
    summaries = []
    for raw in records:
        case = JointCaseContext.from_mapping(raw)
        ranker = None
        cell_executor = None
        if args.cell_executor == "mature":
            cell_executor = MatureProbNetCellExecutor(
                MatureProbNetConfig(
                    dataset_name=args.probnet_dataset,
                    checkpoint=args.probnet_checkpoint,
                    instance_library=args.nuclei_instance_library,
                    device=args.device,
                    base_channels=args.probnet_base_ch,
                )
            )
        elif args.probnet_checkpoint:
            population = repository.cell_population_profiles[case.cell_population_profile_id]
            ranker = FrozenProbNetSpatialRanker.from_checkpoint(
                args.probnet_checkpoint,
                cancer_id=population.probnet_cancer_id,
                pathology_domain_id=case.pathology_domain_id,
                device=args.device,
            )
        workflow = JointPathologyEditWorkflow(
            tissue_planner=(OpenAIMultimodalPlanner(client) if client else MultiInterfaceResearchTissuePlanner()),
            joint_planner=(OpenAIMultimodalJointPlanner(client) if client else HeuristicJointPlanner()),
            critic=(OpenAIMultimodalJointCritic(client) if client else DeterministicJointResearchCritic()),
            joint_skills=repository,
            ranker=ranker,
            cell_executor=cell_executor,
            config=JointWorkflowConfig(production=args.production),
        )
        result = workflow.run(case, output_root=args.output_root)
        summaries.append(
            {
                "case_id": case.case_id,
                "status": result.status,
                "selected_candidate_id": result.selected_candidate_id,
                "abstain_reasons": list(result.abstain_reasons),
                "artifact_paths": result.artifact_paths,
            }
        )
    output = Path(args.output_root) / "joint_run_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"cases": len(summaries), "summary": str(output)}, sort_keys=True))
    return 0 if all(item["status"] != "abstained" for item in summaries) else 2


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


if __name__ == "__main__":
    raise SystemExit(main())

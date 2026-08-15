"""Independent command line entrypoint for mask-edit-refine."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from phase3_mask_edit_refine.agents import (
    DeterministicResearchCritic,
    HeuristicInterfacePlanner,
    OpenAIMultimodalCritic,
    OpenAIMultimodalPlanner,
    OpenAIResponsesJSONClient,
    validate_legacy_online_agent_scope,
)
from phase3_mask_edit_refine.evaluation import load_evaluation_jsonl, score_evaluation
from phase3_mask_edit_refine.evidence import (
    EvidenceManifest,
    build_annotation_profile_statistics,
    verify_case_run_bundle,
    verify_evidence_files,
)
from phase3_mask_edit_refine.gates import GateRegistry
from phase3_mask_edit_refine.models import CaseContext
from phase3_mask_edit_refine.skills import SkillRepository
from phase3_mask_edit_refine.workflow import (
    EscalationBudget,
    MaskEditRefineWorkflow,
    WorkflowConfig,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list-skills":
            return _list_skills(args)
        if args.command == "validate-skills":
            return _validate_skills(args)
        if args.command == "verify-evidence":
            return _verify_evidence(args)
        if args.command == "verify-run-bundle":
            return _verify_run_bundle(args)
        if args.command == "profile-stats":
            return _profile_stats(args)
        if args.command == "score-eval":
            return _score_eval(args)
        if args.command == "run":
            return _run(args)
    # The CLI boundary converts every workflow/configuration failure into a
    # stable machine-readable exit instead of leaking a traceback to callers.
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}))
        return 2
    parser.error("command is required")
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mask-edit-refine",
        description="Independent, auditable Architecture-B pathology mask editor.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-skills")
    list_parser.add_argument("--kind", choices=("pathology_domain", "annotation_profile", "edit_primitive"))

    subparsers.add_parser("validate-skills")

    verify = subparsers.add_parser("verify-evidence")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--allow-missing-digests", action="store_true")

    run_bundle = subparsers.add_parser("verify-run-bundle")
    run_bundle.add_argument("--manifest", required=True)

    stats = subparsers.add_parser("profile-stats")
    stats.add_argument("--manifest", required=True)
    stats.add_argument("--output", required=True)
    stats.add_argument("--split", default="train", choices=("train", "validation", "test"))

    evaluate = subparsers.add_parser("score-eval")
    evaluate.add_argument("--records", required=True, help="Blind-evaluation JSONL")
    evaluate.add_argument("--output")

    run = subparsers.add_parser("run")
    run.add_argument("--case", required=True, help="CaseContext JSON path")
    run.add_argument("--output-root", required=True)
    run.add_argument(
        "--config", default="configs/mask_edit_refine/default.json"
    )
    run.add_argument("--provider", choices=("openai", "heuristic"), default="openai")
    run.add_argument("--research", action="store_true", help="Allow draft skills; output is research-only")
    run.add_argument("--planner-model")
    run.add_argument("--critic-model")
    run.add_argument("--planner-reasoning")
    run.add_argument("--critic-reasoning")
    run.add_argument("--api-base-url")
    run.add_argument("--api-key-env", default="OPENAI_API_KEY")
    run.add_argument("--enable-sol-escalation", action="store_true")
    run.add_argument("--sol-model")
    run.add_argument("--sol-reasoning")
    run.add_argument("--max-sol-fraction", type=float)
    return parser


def _list_skills(args: argparse.Namespace) -> int:
    repository = SkillRepository()
    for skill_id in repository.list(kind=args.kind):
        print(skill_id)
    return 0


def _validate_skills(args: argparse.Namespace) -> int:
    del args
    repository = SkillRepository()
    gates = GateRegistry()
    payload = {
        "status": "valid",
        "skill_count": len(repository.list()),
        "skills": list(repository.list()),
        "available_checkers": list(gates.available_checker_ids),
        "production_ready_skills": [
            skill_id
            for skill_id in repository.list()
            if repository.get(skill_id).review_status == "internally_reviewed"
        ],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _verify_evidence(args: argparse.Namespace) -> int:
    manifest = EvidenceManifest.load(args.manifest)
    report = verify_evidence_files(
        manifest, require_digests=not args.allow_missing_digests
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


def _verify_run_bundle(args: argparse.Namespace) -> int:
    report = verify_case_run_bundle(args.manifest)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


def _profile_stats(args: argparse.Namespace) -> int:
    repository = SkillRepository()
    manifest = EvidenceManifest.load(args.manifest)
    schema = repository.annotation_schema(manifest.annotation_profile_id)
    stats = build_annotation_profile_statistics(manifest, schema=schema, split=args.split)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": "written", "output": str(output), "record_count": stats["record_count"]}))
    return 0


def _score_eval(args: argparse.Namespace) -> int:
    report = score_evaluation(load_evaluation_jsonl(args.records))
    rendered = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0 if all(
        model_report["release_passed"] for model_report in report["models"].values()
    ) else 1


def _run(args: argparse.Namespace) -> int:
    runtime = _load_runtime_config(args.config)
    model_config = runtime["models"]
    workflow_config = runtime["workflow"]
    planner_model = args.planner_model or model_config["planner"]["model"]
    critic_model = args.critic_model or model_config["critic"]["model"]
    planner_reasoning = (
        args.planner_reasoning or model_config["planner"]["reasoning_effort"]
    )
    critic_reasoning = (
        args.critic_reasoning or model_config["critic"]["reasoning_effort"]
    )
    sol_model = args.sol_model or model_config["escalation"]["model"]
    sol_reasoning = (
        args.sol_reasoning or model_config["escalation"]["reasoning_effort"]
    )
    max_sol_fraction = (
        args.max_sol_fraction
        if args.max_sol_fraction is not None
        else float(model_config["escalation"]["max_fraction"])
    )
    payload = json.loads(Path(args.case).read_text(encoding="utf-8"))
    case = CaseContext.from_mapping(payload)
    if args.provider == "heuristic":
        if not args.research:
            raise ValueError("heuristic provider is research-only; pass --research")
        planner = HeuristicInterfacePlanner()
        critic = DeterministicResearchCritic()
        escalation_planner = None
        escalation_critic = None
    else:
        validate_legacy_online_agent_scope(case)
        client_options = {
            "api_base_url": args.api_base_url
            or os.environ.get("OPENAI_API_BASE_URL", "https://api.openai.com/v1"),
            "api_key_env": args.api_key_env,
        }
        planner = OpenAIMultimodalPlanner(
            OpenAIResponsesJSONClient(
                model=planner_model,
                reasoning_effort=planner_reasoning,
                **client_options,
            )
        )
        critic = OpenAIMultimodalCritic(
            OpenAIResponsesJSONClient(
                model=critic_model,
                reasoning_effort=critic_reasoning,
                **client_options,
            )
        )
        if args.enable_sol_escalation:
            escalation_planner = OpenAIMultimodalPlanner(
                OpenAIResponsesJSONClient(
                    model=sol_model,
                    reasoning_effort=sol_reasoning,
                    **client_options,
                )
            )
            escalation_critic = OpenAIMultimodalCritic(
                OpenAIResponsesJSONClient(
                    model=sol_model,
                    reasoning_effort=sol_reasoning,
                    **client_options,
                )
            )
        else:
            escalation_planner = None
            escalation_critic = None
    workflow = MaskEditRefineWorkflow(
        planner=planner,
        critic=critic,
        escalation_planner=escalation_planner,
        escalation_critic=escalation_critic,
        escalation_budget=EscalationBudget(max_fraction=max_sol_fraction),
        config=WorkflowConfig(
            production=not args.research,
            planner_confidence_threshold=float(
                workflow_config["planner_confidence_threshold"]
            ),
            critic_confidence_threshold=float(
                workflow_config["critic_confidence_threshold"]
            ),
            critic_min_score_margin=float(workflow_config["critic_min_score_margin"]),
        ),
    )
    result = workflow.run(case, output_root=args.output_root)
    print(json.dumps(result.to_metadata(), indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if result.status.startswith("selected") else 1


def _load_runtime_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("runtime config root must be an object")
    if payload.get("schema_version") != "mask-edit-refine-config-v1":
        raise ValueError("unsupported mask-edit-refine config schema_version")
    if payload.get("workflow", {}).get("legacy_fallback") is not False:
        raise ValueError("mask-edit-refine forbids legacy fallback")
    return payload


if __name__ == "__main__":
    sys.exit(main())

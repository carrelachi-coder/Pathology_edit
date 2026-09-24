"""Classify terminal D/E failures without changing frozen-case outcomes."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def classify(reasons: list[str], outcome: str) -> str:
    joined = "\n".join(reasons)
    if outcome == "validated":
        return "validated"
    if outcome == "timeout":
        return "wall_clock_timeout"
    if "CodexCLIQueueError" in joined:
        if "ValueError:" in joined:
            return "planner_output_format"
        return "planner_transport"
    if "no natural-language interpretation survives skill and deterministic feasibility" in joined:
        return "source_or_skill_infeasible"
    if "cell-only pre-LLM portfolio has no exact-capacity survivor" in joined:
        return "source_exact_capacity_infeasible"
    if "no cell-only candidate passed its joint condition gates" in joined:
        return "cell_condition_gate_rejection"
    if "no paired tissue--cell candidate passed all joint gates" in joined:
        return "joint_gate_rejection"
    if "all candidate-local cell executions failed" in joined:
        return "cell_execution_failure"
    if "deterministic_replan_stalled" in joined:
        return "replanning_stalled"
    if "joint edit Planner" in joined or "joint Planner" in joined:
        return "planner_contract"
    return "other_failure"


def audit(root: Path) -> dict:
    records = []
    for path in sorted((root / "results").glob("*.json")):
        result = json.loads(path.read_text())
        name = result["case_id"]
        program = root / "runs" / name / name / "program_result.json"
        reasons = []
        if program.exists():
            for step in json.loads(program.read_text()).get("steps", []):
                reasons.extend(str(value) for value in step.get("reasons", []))
        if result.get("reason"):
            reasons.append(str(result["reason"]))
        category = classify(reasons, result["outcome"])
        gates = Counter()
        for report in (root / "runs" / name).rglob("joint_gate_reports*.json"):
            for candidate in json.loads(report.read_text()):
                for check in candidate.get("checks", []):
                    if check.get("passed") is False:
                        gates[str(check.get("check_id"))] += 1
        feedback = []
        for item in (root / "runs" / name).rglob("execution_feedback_pass_*.json"):
            feedback.extend(json.loads(item.read_text()).get("errors", []))
        records.append({
            "case_id": name,
            "dataset": result["dataset"],
            "primitive_id": result["primitive_id"],
            "raw_outcome": result["outcome"],
            "failure_category": category,
            "reasons": reasons,
            "gate_failures": dict(gates),
            "execution_feedback_errors": feedback,
        })
    by_dataset = defaultdict(Counter)
    for item in records:
        by_dataset[item["dataset"]][item["failure_category"]] += 1
    return {
        "completed": len(records),
        "category_counts": dict(Counter(item["failure_category"] for item in records)),
        "by_dataset": {key: dict(value) for key, value in sorted(by_dataset.items())},
        "gate_failures": dict(sum((Counter(item["gate_failures"]) for item in records), Counter())),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.root)
    if args.output:
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

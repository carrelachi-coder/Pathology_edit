"""Classify unresolved v4 cases after all numbered recovery attempts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from r2_de_six_v4_failure_audit import audit, classify
from r2_de_six_v4_reconcile import reconcile


def summarize(root: Path) -> dict:
    initial = {item["case_id"]: item for item in audit(root)["records"]}
    reconciled = reconcile(root)
    unresolved = []
    for row in reconciled["records"]:
        if row["validated_after_retries"]:
            continue
        case_id = row["case_id"]
        first = initial[case_id]
        attempts = row["retry_attempts"]
        if attempts:
            latest = attempts[-1]
            metadata = json.loads(Path(latest["metadata_path"]).read_text())
            program_path = metadata.get("program_result_path")
            reasons = []
            if program_path:
                program = json.loads(Path(program_path).read_text())
                reasons = [
                    str(reason)
                    for step in program.get("steps", [])
                    for reason in step.get("reasons", [])
                ]
            category = classify(reasons, metadata["status"])
            attempt = latest["attempt"]
        else:
            reasons = first["reasons"]
            category = first["failure_category"]
            attempt = 1
        unresolved.append({
            "case_id": case_id,
            "dataset": row["dataset"],
            "primitive_id": first["primitive_id"],
            "initial_category": first["failure_category"],
            "final_category": category,
            "latest_attempt": attempt,
            "reasons": reasons,
        })
    by_dataset = defaultdict(Counter)
    for item in unresolved:
        by_dataset[item["dataset"]][item["final_category"]] += 1
    return {
        "cohort_sha256": reconciled["cohort_sha256"],
        "planned": reconciled["total_planned"],
        "validated_after_retries": reconciled["validated_after_retries"],
        "unresolved": len(unresolved),
        "category_counts": dict(Counter(item["final_category"] for item in unresolved)),
        "by_dataset": {key: dict(value) for key, value in sorted(by_dataset.items())},
        "records": unresolved,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    if args.output:
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}))


if __name__ == "__main__":
    main()

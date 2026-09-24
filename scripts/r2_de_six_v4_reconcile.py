"""Reconcile frozen initial D/E outcomes with auditable same-case retries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconcile(root: Path) -> dict:
    protocol = json.loads((root / "protocol.json").read_text())
    cohort_path = root / "frozen_cohort.json"
    if sha256(cohort_path) != protocol["cohort_sha256"]:
        raise RuntimeError("frozen cohort digest changed")
    cohort = json.loads(cohort_path.read_text())
    case_dataset = {row["record"]["case_id"]: row["dataset"] for row in cohort}
    if len(case_dataset) != len(cohort):
        raise RuntimeError("duplicate frozen case ID")
    with (root / "D_request_outcomes.csv").open(newline="") as stream:
        initial = {row["case_id"]: row for row in csv.DictReader(stream)}
    if set(initial) - set(case_dataset):
        raise RuntimeError("initial summary contains an unfrozen case")

    records = []
    for case_id, dataset in case_dataset.items():
        first = initial.get(case_id)
        first_result = root / "results" / f"{case_id}.json"
        if bool(first) != first_result.exists():
            raise RuntimeError(f"initial result/summary mismatch: {case_id}")
        attempts = []
        retry_parent = root / "retry_attempts" / case_id
        for directory in sorted(
            retry_parent.glob("attempt-*"),
            key=lambda path: int(path.name.split("-")[-1]),
        ):
            metadata_path = directory / "retry_metadata.json"
            if not metadata_path.exists():
                raise RuntimeError(f"retry lacks metadata: {directory}")
            metadata = json.loads(metadata_path.read_text())
            if metadata["case_id"] != case_id or metadata["dataset"] != dataset:
                raise RuntimeError(f"retry identity mismatch: {directory}")
            if metadata["attempt"] != int(directory.name.split("-")[-1]):
                raise RuntimeError(f"retry attempt number mismatch: {directory}")
            if metadata["frozen_cohort_sha256"] != protocol["cohort_sha256"]:
                raise RuntimeError(f"retry cohort mismatch: {directory}")
            if metadata["first_result_sha256"] != sha256(first_result):
                raise RuntimeError(f"retry initial-result link mismatch: {directory}")
            e_path = directory / "independent_E.json"
            passed = metadata.get("independent_E_passed") is True
            if passed:
                if not e_path.exists():
                    raise RuntimeError(f"validated retry lacks independent E: {directory}")
                e = json.loads(e_path.read_text())
                if not (metadata.get("program_status") == "validated"
                        and metadata.get("program_evaluation_passed") is True
                        and e.get("case_id") == case_id
                        and e.get("independent_raster_contract_checks_passed") is True):
                    raise RuntimeError(f"retry validation/E disagreement: {directory}")
            attempts.append({
                "attempt": metadata["attempt"],
                "status": metadata.get("status"),
                "program_status": metadata.get("program_status"),
                "independent_E_passed": passed,
                "code_commit": metadata["code_commit"],
                "metadata_path": str(metadata_path),
                "independent_E_path": str(e_path) if e_path.exists() else None,
                "independent_E_sha256": sha256(e_path) if e_path.exists() else None,
            })
        accepted = next((a for a in reversed(attempts) if a["independent_E_passed"]), None)
        initial_outcome = first["outcome"] if first else "pending"
        if initial_outcome == "validated":
            metric = root / "metrics" / f"{case_id}.json"
            if not metric.exists() or not json.loads(metric.read_text()).get(
                "independent_raster_contract_checks_passed"
            ):
                raise RuntimeError(f"initial validated case lacks passing E: {case_id}")
        final_validated = initial_outcome == "validated" or accepted is not None
        records.append({
            "case_id": case_id, "dataset": dataset,
            "initial_outcome": initial_outcome,
            "initial_result_path": str(first_result) if first else None,
            "accepted_retry_attempt": accepted["attempt"] if accepted else None,
            "validated_after_retries": final_validated,
            "retry_attempts": attempts,
        })

    by_dataset = defaultdict(lambda: {"planned": 0, "initial_completed": 0,
                                     "initial_validated": 0, "retry_recovered": 0,
                                     "validated_after_retries": 0})
    for row in records:
        counts = by_dataset[row["dataset"]]
        counts["planned"] += 1
        counts["initial_completed"] += row["initial_outcome"] != "pending"
        counts["initial_validated"] += row["initial_outcome"] == "validated"
        counts["retry_recovered"] += (row["initial_outcome"] != "validated"
                                      and row["accepted_retry_attempt"] is not None)
        counts["validated_after_retries"] += row["validated_after_retries"]
    return {
        "cohort_sha256": protocol["cohort_sha256"],
        "total_planned": len(cohort),
        "initial_completed": len(initial),
        "initial_outcomes": dict(Counter(row["outcome"] for row in initial.values())),
        "initial_validated": sum(row["initial_outcome"] == "validated" for row in records),
        "retry_recovered": sum(row["initial_outcome"] != "validated" and
                               row["accepted_retry_attempt"] is not None for row in records),
        "validated_after_retries": sum(row["validated_after_retries"] for row in records),
        "active_retry_attempts": sum(a["status"] == "running" for row in records
                                     for a in row["retry_attempts"]),
        "by_dataset": dict(sorted(by_dataset.items())),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = reconcile(args.root)
    if args.output:
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "records"},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()

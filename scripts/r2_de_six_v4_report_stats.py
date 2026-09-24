"""Summarize independently checked D/E results for the frozen six-dataset cohort."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from r2_de_six_v4_reconcile import reconcile


def summarize(root: Path) -> dict:
    report = reconcile(root)
    cohort = json.loads((root / "frozen_cohort.json").read_text())
    groups = defaultdict(set)
    for row in cohort:
        groups[row["dataset"]].add(row["source_group"])
    metrics = defaultdict(list)
    for row in report["records"]:
        if not row["validated_after_retries"]:
            continue
        if row["accepted_retry_attempt"] is not None:
            attempt = next(
                item for item in row["retry_attempts"]
                if item["attempt"] == row["accepted_retry_attempt"]
            )
            path = Path(attempt["independent_E_path"])
        else:
            path = root / "metrics" / f"{row['case_id']}.json"
        result = json.loads(path.read_text())
        if result["case_id"] != row["case_id"] or not result[
            "independent_raster_contract_checks_passed"
        ]:
            raise RuntimeError(f"independent E mismatch: {row['case_id']}")
        metrics[row["dataset"]].append(result)
    by_dataset = {}
    for dataset, counts in report["by_dataset"].items():
        values = metrics[dataset]
        if len(values) != counts["validated_after_retries"]:
            raise RuntimeError(f"E count mismatch: {dataset}")
        by_dataset[dataset] = {
            **counts,
            "source_groups": len(groups[dataset]),
            "E_measured": len(values),
            "median_joint_change_fraction": median(
                item["joint_fraction"] for item in values
            ) if values else None,
            "median_generation_region_fraction": median(
                item["G_fraction"] for item in values
            ) if values else None,
            "max_change_outside_G_pixels": max(
                (item["change_outside_G_pixels"] for item in values), default=None
            ),
            "max_protected_tissue_changed_pixels": max(
                (item["protected_tissue_changed_pixels"] for item in values),
                default=None,
            ),
            "max_unauthorized_tissue_transition_pixels": max(
                (item["unauthorized_tissue_transition_pixels"] for item in values),
                default=None,
            ),
            "max_nuclear_label_changes_outside_declared_cell_region": max(
                (item["nuclear_label_changes_outside_declared_cell_region"]
                 for item in values), default=None,
            ),
        }
    return {
        "cohort_sha256": report["cohort_sha256"],
        "total_planned": report["total_planned"],
        "initial_completed": report["initial_completed"],
        "initial_outcomes": report["initial_outcomes"],
        "initial_validated": report["initial_validated"],
        "retry_recovered": report["retry_recovered"],
        "validated_after_retries": report["validated_after_retries"],
        "active_retry_attempts": report["active_retry_attempts"],
        "E_measured_unique_validated_cases": sum(len(v) for v in metrics.values()),
        "by_dataset": by_dataset,
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
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

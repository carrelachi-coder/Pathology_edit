#!/usr/bin/env python3
"""Freeze patient-level four-target approvals from explicit mask-review decisions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


TARGET_LABELS = (
    "u1_moderate",
    "u1_significant",
    "u2_moderate",
    "u2_significant",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_identity(record: dict[str, Any]) -> None:
    path = Path(str(record.get("path", "")))
    if not path.is_file():
        raise ValueError(f"review artifact is missing: {path}")
    if path.stat().st_size != int(record.get("bytes", -1)):
        raise ValueError(f"review artifact size changed: {path}")
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"review artifact content changed: {path}")


def freeze_approvals(queue: dict, decisions: dict) -> list[dict]:
    if queue.get("schema_version") != "reserved-four-target-mask-review-queue-v1":
        raise ValueError("unsupported review queue schema")
    if decisions.get("schema_version") != "reserved-four-target-mask-review-decisions-v1":
        raise ValueError("unsupported review decisions schema")

    rows = decisions.get("decisions")
    if not isinstance(rows, list):
        raise ValueError("decisions must be a list")
    by_key: dict[tuple[str, str], dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each decision must be an object")
        key = (str(row.get("case_id", "")), str(row.get("candidate_id", "")))
        if not all(key) or key in by_key:
            raise ValueError("decision identity is empty or duplicated")
        if row.get("decision") not in {"approved", "rejected"}:
            raise ValueError("decision must be approved or rejected")
        if not str(row.get("reviewer", "")).strip() or not str(row.get("reason", "")).strip():
            raise ValueError("every decision needs a reviewer and reason")
        by_key[key] = row

    consumed: set[tuple[str, str]] = set()
    frozen_patients = []
    for patient in queue.get("patients", []):
        cases = patient.get("cases", [])
        if [row.get("target") for row in cases] != list(TARGET_LABELS):
            raise ValueError("patient target order is incomplete or changed")
        frozen_cases = []
        patient_approved = patient.get("status") == "ready_for_independent_mask_review"
        for case in cases:
            approved = []
            candidate_rows = case.get("candidates", [])
            if case.get("status") != "review_required":
                patient_approved = False
            else:
                verify_identity(case["review_board"])
            for candidate in candidate_rows:
                verify_identity(candidate["evidence"])
                key = (str(case["case_id"]), str(candidate["candidate_id"]))
                decision = by_key.get(key)
                if decision is None:
                    raise ValueError(f"missing explicit candidate decision: {key}")
                consumed.add(key)
                if decision["decision"] == "approved":
                    approved.append(
                        {
                            "candidate_id": candidate["candidate_id"],
                            "evidence": candidate["evidence"],
                            "reviewer": decision["reviewer"],
                            "reason": decision["reason"],
                        }
                    )
            if len(approved) > 1:
                raise ValueError(f"multiple candidates approved for one case: {case['case_id']}")
            if len(approved) != 1:
                patient_approved = False
            frozen_cases.append(
                {
                    "case_index": case["case_index"],
                    "case_id": case["case_id"],
                    "target": case["target"],
                    "status": "approved" if len(approved) == 1 else "not_approved",
                    "approved_candidate": approved[0] if approved else None,
                }
            )
        frozen_patients.append(
            {
                "patient_order": patient["patient_order"],
                "patient_id": patient["patient_id"],
                "reference_id": patient["reference_id"],
                "status": "four_targets_approved" if patient_approved else "not_approved",
                "cases": frozen_cases,
            }
        )
    unknown = set(by_key) - consumed
    if unknown:
        raise ValueError(f"decisions contain candidates outside the review queue: {sorted(unknown)}")
    return frozen_patients


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists; approval freezes are append-only")
    queue = json.loads(args.review_queue.read_text())
    decisions = json.loads(args.decisions.read_text())
    queue_binding = decisions.get("review_queue", {})
    if queue_binding.get("sha256") != sha256_file(args.review_queue):
        raise ValueError("decisions are not bound to this review queue")
    patients = freeze_approvals(queue, decisions)
    args.output.mkdir(parents=True)
    manifest = {
        "schema_version": "reserved-four-target-approval-freeze-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "mask-only independent review; no H&E-based selection",
        "review_queue": identity(args.review_queue),
        "decisions": identity(args.decisions),
        "patients": patients,
        "counts": {
            "patients": len(patients),
            "four_targets_approved": sum(
                row["status"] == "four_targets_approved" for row in patients
            ),
            "not_approved": sum(row["status"] != "four_targets_approved" for row in patients),
        },
        "h_and_e_selection_used": False,
        "qualification_updated": False,
    }
    (args.output / "four_target_approval_freeze.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest["counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

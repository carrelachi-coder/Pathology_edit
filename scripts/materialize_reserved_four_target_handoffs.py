#!/usr/bin/env python3
"""Materialize independently reviewed handoffs from the strict approval freeze."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase3_joint_edit_refine.approval_handoff import (
    promote_independently_reviewed_joint_candidate,
)


REVIEW_TEXT_FIELDS = (
    "tissue_geometry",
    "nucleus_layout",
    "coupling",
    "preserved_conditions",
    "limitations",
    "verdict",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_identity(record: dict[str, Any]) -> Path:
    path = Path(str(record.get("path", "")))
    if (
        not path.is_file()
        or path.stat().st_size != int(record.get("bytes", -1))
        or sha256_file(path) != record.get("sha256")
    ):
        raise ValueError(f"review evidence identity changed: {path}")
    return path


def materialize(
    freeze: dict,
    *,
    output: Path,
    promoter: Callable[..., dict[str, str]] = promote_independently_reviewed_joint_candidate,
) -> list[dict]:
    if freeze.get("schema_version") != "reserved-four-target-approval-freeze-v1":
        raise ValueError("unsupported approval freeze schema")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for patient in freeze.get("patients", []):
        if patient.get("status") != "four_targets_approved":
            continue
        cases = patient.get("cases", [])
        if len(cases) != 4 or any(case.get("status") != "approved" for case in cases):
            raise ValueError("approved patient lacks four approved targets")
        patient_rows = []
        for case in cases:
            approved = case.get("approved_candidate") or {}
            evidence_path = verify_identity(approved.get("evidence") or {})
            evidence = json.loads(evidence_path.read_text())
            if (
                evidence.get("case_id") != case.get("case_id")
                or evidence.get("candidate_id") != approved.get("candidate_id")
            ):
                raise ValueError("review evidence is bound to another case or candidate")
            audit_dir = Path(str(case.get("audit_case_dir") or ""))
            if not audit_dir.is_dir():
                raise ValueError("approved case lacks its immutable audit directory")
            case_output = output / f"case_{int(case['case_index']):04d}"
            case_output.mkdir()
            review = {
                "schema_version": "joint-mask-condition-independent-review-v1",
                "case_id": case["case_id"],
                "candidate_id": approved["candidate_id"],
                "reviewer": "codex_current_chat",
                "decision": "approved",
                "scope": "research_mask_condition_only",
                "source_he_provided_to_review": False,
                "hard_gate_waiver": False,
                "confidence": approved["confidence"],
                "reviewed_utc": approved["reviewed_utc"],
                **{field: approved[field] for field in REVIEW_TEXT_FIELDS},
                "evidence": evidence,
            }
            review_path = case_output / "independent_review.json"
            review_path.write_text(json.dumps(review, indent=2) + "\n")
            paths = promoter(
                audit_dir,
                candidate_id=approved["candidate_id"],
                review_path=review_path,
                output_dir=case_output,
            )
            manifest_path = Path(paths["manifest"])
            patient_rows.append(
                {
                    "case_index": case["case_index"],
                    "case_id": case["case_id"],
                    "target": case["target"],
                    "candidate_id": approved["candidate_id"],
                    "independent_review": identity(review_path),
                    "handoff_manifest": identity(manifest_path),
                }
            )
        rows.append(
            {
                "patient_order": patient["patient_order"],
                "patient_id": patient["patient_id"],
                "reference_id": patient["reference_id"],
                "cases": patient_rows,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.approval_freeze.read_text())
    patients = materialize(freeze, output=args.output)
    manifest = {
        "schema_version": "reserved-four-target-handoff-materialization-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "approval_freeze": identity(args.approval_freeze),
        "patients": patients,
        "counts": {
            "patients": len(patients),
            "handoffs": sum(len(patient["cases"]) for patient in patients),
        },
        "h_and_e_generation_started": False,
        "qualification_updated": False,
    }
    path = args.output / "handoff_materialization_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

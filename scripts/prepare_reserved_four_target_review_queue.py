#!/usr/bin/env python3
"""Prepare exact independent mask-review evidence for compiled reserved cases."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase3_joint_edit_refine.independent_review import review_evidence


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


def load_inputs(
    cases_path: Path, candidate_manifest_path: Path, receipts_path: Path
) -> tuple[list[dict], dict, dict[int, dict]]:
    cases = json.loads(cases_path.read_text())
    candidate_manifest = json.loads(candidate_manifest_path.read_text())
    receipts = json.loads(receipts_path.read_text())
    if candidate_manifest.get("schema_version") != "reserved-four-target-candidate-freeze-v1":
        raise ValueError("unsupported candidate freeze schema")
    if candidate_manifest.get("cases", {}).get("sha256") != sha256_file(cases_path):
        raise ValueError("cases differ from the candidate freeze")
    if not isinstance(receipts, list):
        raise ValueError("compilation receipts must be a list")
    by_index: dict[int, dict] = {}
    for receipt in receipts:
        index = int(receipt.get("case_index", -1))
        if index in by_index or not 0 <= index < len(cases):
            raise ValueError("compilation receipt index is duplicate or invalid")
        if receipt.get("case_id") != cases[index].get("case_id"):
            raise ValueError("compilation receipt case identity mismatch")
        by_index[index] = receipt
    return cases, candidate_manifest, by_index


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def prepare_queue(
    *,
    cases: list[dict],
    candidate_manifest: dict,
    receipts_by_index: dict[int, dict],
    output: Path,
    evidence_builder: Callable[..., dict] = review_evidence,
) -> list[dict]:
    output.mkdir(parents=True, exist_ok=False)
    evidence_root = output / "evidence"
    evidence_root.mkdir()
    patients = []
    for patient in candidate_manifest.get("selected_candidates", []):
        indices = [int(value) for value in patient.get("case_indices", [])]
        if len(indices) != 4:
            raise ValueError("reserved patient does not have four frozen target indices")
        case_rows = []
        for expected_label, index in zip(TARGET_LABELS, indices):
            case = cases[index]
            case_id = str(case["case_id"])
            if not case_id.endswith("__" + expected_label):
                raise ValueError("four-target order or identity changed")
            receipt = receipts_by_index.get(index)
            if receipt is None:
                case_rows.append(
                    {"case_index": index, "case_id": case_id, "target": expected_label,
                     "status": "not_compiled"}
                )
                continue
            terminal = str(receipt.get("terminal_status"))
            case_row = {
                "case_index": index,
                "case_id": case_id,
                "target": expected_label,
                "status": terminal,
            }
            if terminal == "review_required":
                candidate_ids = receipt.get("passing_candidate_ids")
                audit_dir = Path(str(receipt.get("audit_case_dir")))
                board = audit_dir / "joint_condition_mask_review.png"
                if (
                    not isinstance(candidate_ids, list)
                    or not candidate_ids
                    or len(set(candidate_ids)) != len(candidate_ids)
                    or not audit_dir.is_dir()
                    or not board.is_file()
                ):
                    raise ValueError(f"review-required receipt is incomplete: {case_id}")
                candidate_rows = []
                case_output = evidence_root / f"case_{index:04d}"
                case_output.mkdir()
                for candidate_id in candidate_ids:
                    candidate_id = str(candidate_id)
                    evidence = evidence_builder(audit_dir, candidate_id=candidate_id)
                    suffix = hashlib.sha256(candidate_id.encode()).hexdigest()[:12]
                    evidence_path = case_output / f"{safe_name(candidate_id)}__{suffix}.json"
                    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
                    candidate_rows.append(
                        {
                            "candidate_id": candidate_id,
                            "evidence": identity(evidence_path),
                        }
                    )
                case_row.update(
                    audit_case_dir=str(audit_dir.resolve()),
                    review_board=identity(board),
                    candidates=candidate_rows,
                )
            case_rows.append(case_row)
        ready = all(row["status"] == "review_required" for row in case_rows)
        patients.append(
            {
                "patient_order": patient["patient_order"],
                "patient_id": patient["patient_id"],
                "reference_id": patient["reference_id"],
                "status": (
                    "ready_for_independent_mask_review"
                    if ready
                    else "not_ready_for_independent_mask_review"
                ),
                "cases": case_rows,
            }
        )
    return patients


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--compilation-receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists; review queues are append-only")
    cases, candidate_manifest, receipts = load_inputs(
        args.cases, args.candidate_manifest, args.compilation_receipts
    )
    patients = prepare_queue(
        cases=cases,
        candidate_manifest=candidate_manifest,
        receipts_by_index=receipts,
        output=args.output,
    )
    manifest = {
        "schema_version": "reserved-four-target-mask-review-queue-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "exact mask-only evidence preparation; no approval, H&E generation, or formal admission",
        "cases": identity(args.cases),
        "candidate_manifest": identity(args.candidate_manifest),
        "compilation_receipts": identity(args.compilation_receipts),
        "patients": patients,
        "counts": {
            "patients": len(patients),
            "ready_for_independent_mask_review": sum(
                row["status"] == "ready_for_independent_mask_review" for row in patients
            ),
            "not_ready_for_independent_mask_review": sum(
                row["status"] != "ready_for_independent_mask_review" for row in patients
            ),
        },
        "decisions_created": 0,
        "handoffs_created": 0,
    }
    (args.output / "review_queue.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

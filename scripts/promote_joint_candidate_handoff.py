#!/usr/bin/env python3
"""Promote explicit user-approved audited joint candidates to Generate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase3_joint_edit_refine.approval_handoff import (
    promote_audited_joint_candidate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hash-validate persisted passing candidates and emit exact Online "
            "joint-generation-handoff-v3 manifests."
        )
    )
    parser.add_argument(
        "--approval-manifest",
        required=True,
        type=Path,
        help=(
            "JSON object/list. Each record requires audit_case_dir, candidate_id, "
            "and the explicit joint-candidate-user-approval-v1 fields."
        ),
    )
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(args.approval_manifest.read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else [payload]
    if not records or not all(isinstance(item, dict) for item in records):
        raise ValueError("approval manifest must contain one or more JSON objects")
    summaries = []
    for record in records:
        audit_case_dir = record.get("audit_case_dir")
        candidate_id = record.get("candidate_id")
        if not isinstance(audit_case_dir, str) or not audit_case_dir:
            raise ValueError("approval record requires audit_case_dir")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("approval record requires candidate_id")
        approval = {
            key: value
            for key, value in record.items()
            if key not in {"audit_case_dir"}
        }
        case_id = str(record.get("case_id") or "")
        output_dir = args.output_root / case_id
        paths = promote_audited_joint_candidate(
            audit_case_dir,
            candidate_id=candidate_id,
            approval=approval,
            output_dir=output_dir,
        )
        manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
        summaries.append(
            {
                "case_id": case_id,
                "candidate_id": candidate_id,
                "manifest": paths["manifest"],
                "joint_pixels": manifest["ledger"]["joint_pixels"],
                "generation_support_pixels": manifest["ledger"][
                    "generation_support_pixels"
                ],
                "adjacent_context_pixels": manifest["generation_context"][
                    "adjacent_context_pixels"
                ],
            }
        )
    summary_path = args.output_root / "promotion_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"cases": len(summaries), "summary": str(summary_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

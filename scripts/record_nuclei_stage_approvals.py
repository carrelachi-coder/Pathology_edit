#!/usr/bin/env python3
"""Record hash-verified human decisions for a staged nuclei review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase3_mask_edit.audit.staged_review import record_nuclei_stage_decisions


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = record_nuclei_stage_decisions(
        args.manifest,
        approved_case_ids=tuple(args.approve_case_id or ()),
        revision_required_case_ids=tuple(
            args.revision_required_case_id or ()
        ),
    )
    print(json.dumps(manifest["approval"], indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record nuclei-stage approvals after verifying all image inputs."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--approve-case-id", action="append")
    parser.add_argument("--revision-required-case-id", action="append")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())

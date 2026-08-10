#!/usr/bin/env python3
"""Freeze a reviewed G2-v2 image--instruction--mechanism manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase3_joint_edit_refine.g2_v2_manifest import freeze_g2_v2_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-manifest", required=True)
    parser.add_argument("--qualification-jsonl", required=True)
    parser.add_argument("--he-decisions-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-cases", type=int, default=600)
    parser.add_argument(
        "--execution-qualification-jsonl",
        help=(
            "Optional source-only executable preflight ledger; failed pairs "
            "are frozen as abstentions"
        ),
    )
    args = parser.parse_args()
    result = freeze_g2_v2_manifest(
        args.legacy_manifest,
        args.qualification_jsonl,
        args.he_decisions_jsonl,
        output_dir=args.output_dir,
        expected_cases=args.expected_cases,
        execution_qualification_jsonl=args.execution_qualification_jsonl,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

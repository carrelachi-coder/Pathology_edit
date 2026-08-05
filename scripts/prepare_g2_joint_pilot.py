#!/usr/bin/env python3
"""Prepare selection/fetch plan or a local joint manifest for the G2 pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase3_joint_edit_refine.g2_pilot import (  # noqa: E402
    build_local_joint_records,
    select_stratified_cases,
    write_fetch_plan,
)
from phase3_joint_edit_refine.models import JointCaseContext  # noqa: E402
from phase3_joint_edit_refine.visualization import build_source_review_boards  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g2-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--mechanism-decisions", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.g2_manifest.read_text(encoding="utf-8"))
    rows = select_stratified_cases(
        payload,
        per_organ={"breast": 8, "colorectal": 9, "prostate": 8, "lung": 8, "oral": 8, "skin": 9},
        mandatory_case_ids=(
            "g2_152_colorectal_tumor_decrease_2824880108",
            "g2_175_colorectal_stroma_increase_28509939a5",
        ),
    )
    paths = write_fetch_plan(rows, output_dir=args.output_dir)
    if args.asset_root:
        decisions = json.loads(args.mechanism_decisions.read_text(encoding="utf-8")) if args.mechanism_decisions else {}
        records = build_local_joint_records(rows, asset_root=args.asset_root, mechanism_decisions=decisions)
        manifest = args.output_dir / "g2_joint_pilot_manifest.json"
        manifest.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
        paths["joint_manifest"] = str(manifest)
        boards = build_source_review_boards(
            [JointCaseContext.from_mapping(item) for item in records],
            output_dir=args.output_dir / "source_review_boards",
        )
        paths["source_review_boards"] = boards
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

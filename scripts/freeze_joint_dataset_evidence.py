#!/usr/bin/env python3
"""Freeze the six local annotation datasets used by joint edit refine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase3_joint_edit_refine.evidence_freeze import (
    freeze_dataset_evidence,
    verify_frozen_evidence_index,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grouped-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    result = freeze_dataset_evidence(
        args.grouped_manifest,
        output_root=args.output_root,
        code_revision=args.code_revision,
        workers=args.workers,
    )
    verification = verify_frozen_evidence_index(result["index_path"])
    print(
        json.dumps(
            {"freeze": result, "verification": verification},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if verification["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

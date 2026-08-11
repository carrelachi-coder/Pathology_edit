#!/usr/bin/env python3
"""Freeze the three-arm generator ablation for one approved joint handoff."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase3_joint_edit_refine.generator_ablation import (
    prepare_generator_paired_ablation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--generator-snapshot", required=True)
    args = parser.parse_args()
    result = prepare_generator_paired_ablation(
        args.handoff_manifest,
        output_root=args.output_root,
        generator_snapshot=args.generator_snapshot,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

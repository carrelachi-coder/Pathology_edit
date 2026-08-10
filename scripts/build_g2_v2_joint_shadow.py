#!/usr/bin/env python3
"""Select and materialize a stratified G2-v2 joint shadow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase3_joint_edit_refine.g2_v2_shadow import build_g2_v2_shadow


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-organ", type=int, default=8)
    parser.add_argument("--abstain-controls", type=int, default=2)
    args = parser.parse_args()
    result = build_g2_v2_shadow(
        args.frozen_manifest,
        output_dir=args.output_dir,
        per_organ=args.per_organ,
        abstain_controls=args.abstain_controls,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

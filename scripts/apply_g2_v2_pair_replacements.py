#!/usr/bin/env python3
"""Create a new frozen G2-v2 manifest from digest-bound pair replacements."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase3_joint_edit_refine.g2_pair_replacements import (
    apply_g2_v2_pair_replacements,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--replacement-ledger", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = apply_g2_v2_pair_replacements(
        args.base_manifest,
        args.replacement_ledger,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

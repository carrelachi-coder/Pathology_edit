#!/usr/bin/env python3
"""Apply audited current-Codex visual plan decisions to a shadow manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase3_joint_edit_refine.g2_plan_overrides import apply_plan_overrides


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--overrides", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = apply_plan_overrides(
        args.manifest,
        args.overrides,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

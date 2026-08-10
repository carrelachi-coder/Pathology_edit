#!/usr/bin/env python3
"""Freeze the current-Codex source H&E review into a per-case ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase3_joint_edit_refine.g2_he_review import write_g2_he_review


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = write_g2_he_review(
        args.qualification_jsonl,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

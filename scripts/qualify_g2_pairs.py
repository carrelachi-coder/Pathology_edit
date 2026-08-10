#!/usr/bin/env python3
"""Build a read-only G2 pair qualification ledger and source review boards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase3_joint_edit_refine.g2_qualification import qualify_g2_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--board-page-size", type=int, default=12)
    parser.add_argument("--tile-size", type=int, default=176)
    args = parser.parse_args()
    if args.board_page_size <= 0 or args.tile_size < 96:
        raise ValueError("board page size must be positive and tile size >= 96")
    result = qualify_g2_manifest(
        args.manifest,
        output_dir=args.output_dir,
        board_page_size=args.board_page_size,
        tile_size=args.tile_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

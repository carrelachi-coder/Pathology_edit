#!/usr/bin/env python3
"""Build or verify the committed official mask-skill catalog manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase3_mask_edit_refine.skills.catalog_manifest import (
    OFFICIAL_CATALOG_MANIFEST_PATH,
    build_catalog_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = build_catalog_manifest()
    rendered = json.dumps(expected, indent=2, ensure_ascii=False) + "\n"
    path = OFFICIAL_CATALOG_MANIFEST_PATH
    if args.check:
        if not path.is_file() or path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("official mask-skill catalog manifest is stale")
        return 0
    path.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

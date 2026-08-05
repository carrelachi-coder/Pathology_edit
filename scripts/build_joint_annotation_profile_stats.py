#!/usr/bin/env python3
"""Build draft annotation-profile geometry statistics from a frozen manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase3_joint_edit_refine.profile_statistics import (  # noqa: E402
    build_annotation_profile_statistics,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotation-profile-id", required=True)
    parser.add_argument("--data-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    payload = json.loads(raw)
    records = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("statistics manifest must be a list or contain cases[]")
    result = build_annotation_profile_statistics(
        records,
        annotation_profile_id=args.annotation_profile_id,
        data_revision=args.data_revision,
        evidence_manifest_sha256=hashlib.sha256(raw).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"patches": len(records), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

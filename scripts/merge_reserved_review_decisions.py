#!/usr/bin/env python3
"""Merge append-only per-case reserved mask reviews into one queue-bound record."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "reserved-four-target-mask-review-decisions-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def candidate_keys(queue: dict) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for patient in queue.get("patients", []):
        for case in patient.get("cases", []):
            for candidate in case.get("candidates", []):
                keys.add((case["case_id"], candidate["candidate_id"]))
    return keys


def merge(queue: dict, documents: list[dict], *, require_complete: bool) -> list[dict]:
    allowed = candidate_keys(queue)
    merged: dict[tuple[str, str], dict] = {}
    for document in documents:
        if document.get("schema_version") != SCHEMA:
            raise ValueError("unsupported review decisions schema")
        if document.get("h_and_e_selection_used") is not False:
            raise ValueError("H&E-based selection is prohibited")
        for row in document.get("decisions", []):
            key = (row.get("case_id"), row.get("candidate_id"))
            if key not in allowed:
                raise ValueError(f"decision candidate is outside target queue: {key}")
            if key in merged:
                if merged[key] != row:
                    raise ValueError(f"conflicting duplicate decision: {key}")
                continue
            merged[key] = row
    if require_complete:
        missing = allowed - set(merged)
        if missing:
            raise ValueError(f"target queue has candidates without decisions: {sorted(missing)}")
    return [merged[key] for key in sorted(merged)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--decision", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists; merged decisions are append-only")
    queue = json.loads(args.review_queue.read_text())
    documents = [json.loads(path.read_text()) for path in args.decision]
    rows = merge(queue, documents, require_complete=args.require_complete)
    output = {
        "schema_version": SCHEMA,
        "review_queue": identity(args.review_queue),
        "scope": "mask-only independent review; no generated H&E was available or used",
        "decisions": rows,
        "decision_sources": [identity(path) for path in args.decision],
        "source_he_provided_to_review": False,
        "h_and_e_selection_used": False,
        "qualification_updated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"decisions": len(rows), "candidates": len(candidate_keys(queue))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

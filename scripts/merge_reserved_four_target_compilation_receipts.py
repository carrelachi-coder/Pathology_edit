#!/usr/bin/env python3
"""Merge append-only reserved compilation runs into one review-bound receipt set."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def merge_runs(run_directories: list[Path]) -> tuple[list[dict], list[dict]]:
    if not run_directories:
        raise ValueError("at least one compilation run is required")
    bindings: tuple[str, str] | None = None
    merged: dict[int, dict] = {}
    sources: list[dict] = []
    for run_directory in run_directories:
        identity_path = run_directory / "run_identity.json"
        receipts_path = run_directory / "compilation_receipts.json"
        if not identity_path.is_file() or not receipts_path.is_file():
            raise ValueError(f"compilation run is incomplete: {run_directory}")
        run_identity = json.loads(identity_path.read_text())
        receipts = json.loads(receipts_path.read_text())
        if run_identity.get("schema_version") != "reserved-four-target-compilation-run-v1":
            raise ValueError(f"unsupported run identity: {run_directory}")
        current_bindings = (
            str(run_identity.get("cases", {}).get("sha256", "")),
            str(run_identity.get("candidate_manifest", {}).get("sha256", "")),
        )
        if not all(current_bindings):
            raise ValueError(f"run identity is missing frozen input hashes: {run_directory}")
        if bindings is None:
            bindings = current_bindings
        elif bindings != current_bindings:
            raise ValueError("compilation runs use different frozen cases or manifests")
        if not isinstance(receipts, list):
            raise ValueError(f"compilation receipts are not a list: {run_directory}")
        declared = [int(value) for value in run_identity.get("selected_case_indices", [])]
        observed = [int(row.get("case_index", -1)) for row in receipts]
        if observed != sorted(observed) or observed != sorted(declared):
            raise ValueError(f"receipt indices do not match the run identity: {run_directory}")
        for row in receipts:
            index = int(row.get("case_index", -1))
            if index < 0 or index in merged:
                raise ValueError(f"duplicate or invalid case index across runs: {index}")
            if not str(row.get("case_id", "")) or not str(row.get("terminal_status", "")):
                raise ValueError(f"receipt lacks terminal identity: {run_directory} index {index}")
            merged[index] = row
        sources.append(
            {
                "run_directory": str(run_directory.resolve()),
                "run_identity": identity(identity_path),
                "compilation_receipts": identity(receipts_path),
            }
        )
    return [merged[index] for index in sorted(merged)], sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--expected-case-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists; merged receipts are append-only")
    receipts, sources = merge_runs(args.run)
    if args.expected_case_count <= 0 or len(receipts) != args.expected_case_count:
        raise ValueError(
            f"merged case count {len(receipts)} does not equal expected {args.expected_case_count}"
        )
    if [int(row["case_index"]) for row in receipts] != list(range(args.expected_case_count)):
        raise ValueError("merged receipts do not cover every expected case exactly once")
    args.output.mkdir(parents=True)
    receipts_path = args.output / "compilation_receipts.json"
    receipts_path.write_text(json.dumps(receipts, indent=2) + "\n")
    manifest = {
        "schema_version": "reserved-four-target-compilation-receipt-merge-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "receipt aggregation only; no retry, approval, H&E generation, or qualification update",
        "expected_case_count": args.expected_case_count,
        "merged_case_count": len(receipts),
        "source_runs": sources,
        "compilation_receipts": identity(receipts_path),
        "automatic_retries": 0,
        "qualification_updated": False,
    }
    (args.output / "merge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"merged_case_count": len(receipts)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

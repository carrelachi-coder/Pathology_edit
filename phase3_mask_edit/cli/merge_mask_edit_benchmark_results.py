"""Merge targeted mask-edit reruns into a base evaluation without overwriting inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from phase3_mask_edit.benchmark.models import write_eval_csv
from phase3_mask_edit.benchmark.reporting import (
    summarize_semantic_rows,
    write_semantic_report,
)
from phase3_mask_edit.cli.run_mask_edit_benchmark import (
    summarize_rows,
    write_report_csv,
)
from phase3_mask_edit.core.mask_io import save_metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--rerun", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args(argv)

    base_rows = _read_rows(args.base)
    rerun_rows = _read_rows(args.rerun)
    merged = {_key(row): dict(row) for row in base_rows}
    if len(merged) != len(base_rows):
        raise ValueError("base evaluation contains duplicate sample_id/mode rows")
    replacements = 0
    additions = 0
    for row in rerun_rows:
        key = _key(row)
        if key in merged:
            replacements += 1
        else:
            additions += 1
        merged[key] = dict(row)
    rows = sorted(merged.values(), key=_key)

    args.output.mkdir(parents=True, exist_ok=True)
    eval_path = write_eval_csv(rows, args.output / "benchmark_eval_results.csv")
    legacy_report = summarize_rows(rows)
    save_metadata(legacy_report, args.output / "benchmark_report.json")
    write_report_csv(legacy_report, args.output / "benchmark_report.csv")
    semantic_report = summarize_semantic_rows(
        rows,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    write_semantic_report(semantic_report, args.output)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base": {
            "path": str(args.base.resolve()),
            "sha256": _sha256(args.base),
            "rows": len(base_rows),
        },
        "rerun": {
            "path": str(args.rerun.resolve()),
            "sha256": _sha256(args.rerun),
            "rows": len(rerun_rows),
        },
        "output": str(eval_path),
        "merged_rows": len(rows),
        "replacements": replacements,
        "additions": additions,
    }
    save_metadata(manifest, args.output / "merge_manifest.json")
    if args.print_summary:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _key(row: Mapping[str, Any]) -> tuple[str, str]:
    sample_id = str(row.get("sample_id") or "")
    mode = str(row.get("mode") or "")
    if not sample_id or not mode:
        raise ValueError("every evaluation row must contain sample_id and mode")
    return sample_id, mode


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

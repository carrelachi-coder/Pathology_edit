"""Select an exact mask-edit benchmark subset for targeted reruns."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--failed-only", action="store_true")
    parser.add_argument("--include-not-all-ok", action="store_true")
    parser.add_argument("--primitives", nargs="+")
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args(argv)

    if not args.failed_only and not args.include_not_all_ok and not args.primitives:
        parser.error(
            "select at least one of --failed-only, --include-not-all-ok, or --primitives"
        )

    with args.eval_results.open("r", encoding="utf-8", newline="") as stream:
        rows = [dict(row) for row in csv.DictReader(stream)]

    primitive_filter = set(args.primitives or ())
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        reasons: list[str] = []
        if args.failed_only and row.get("status") != "completed":
            reasons.append("execution_failed")
        if args.include_not_all_ok and not _truthy(row.get("all_ok")):
            reasons.append("all_ok_false")
        if primitive_filter and row.get("primitive") in primitive_filter:
            reasons.append("primitive_selected")
        if not reasons:
            continue
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            continue
        selected[sample_id] = {
            "sample_id": sample_id,
            "organ": row.get("organ", ""),
            "profile": row.get("profile", ""),
            "primitive": row.get("primitive", ""),
            "strength": row.get("strength", ""),
            "previous_status": row.get("status", ""),
            "previous_error": row.get("error", ""),
            "selection_reasons": reasons,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for sample_id in sorted(selected):
            stream.write(
                json.dumps(selected[sample_id], ensure_ascii=False, sort_keys=True)
                + "\n"
            )
    if args.print_summary:
        print(
            json.dumps(
                {"selected": len(selected), "output": str(args.output)}, indent=2
            )
        )
    return 0


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


if __name__ == "__main__":
    raise SystemExit(main())

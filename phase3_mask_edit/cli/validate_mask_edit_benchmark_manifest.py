"""Validate mask-edit intent manifests before expensive benchmark execution."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from phase3_mask_edit.benchmark.models import BenchmarkIntent, read_intents_jsonl
from phase3_mask_edit.specialized.catalog import specialized_primitive_names
from phase3_mask_edit.core.mask_io import save_metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intents", required=True, type=Path)
    parser.add_argument("--shortfalls", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-per-cell", type=int, default=100)
    parser.add_argument("--max-per-wsi-per-cell", type=int, default=10)
    parser.add_argument("--require-all-specialized", action="store_true")
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args(argv)

    intents = read_intents_jsonl(args.intents)
    shortfalls = _read_shortfalls(args.shortfalls)
    report = validate_manifest(
        intents,
        shortfalls=shortfalls,
        expected_per_cell=args.expected_per_cell,
        max_per_wsi_per_cell=args.max_per_wsi_per_cell,
        require_all_specialized=args.require_all_specialized,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_metadata(report, args.output)
    if args.print_summary:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["valid"] else 2


def validate_manifest(
    intents: list[BenchmarkIntent],
    *,
    shortfalls: dict[tuple[str, str, str], dict[str, Any]],
    expected_per_cell: int,
    max_per_wsi_per_cell: int,
    require_all_specialized: bool,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    sample_counts = Counter(item.sample_id for item in intents)
    duplicate_ids = sorted(
        sample_id for sample_id, count in sample_counts.items() if count > 1
    )
    if duplicate_ids:
        errors.append({"check": "duplicate_sample_ids", "sample_ids": duplicate_ids})

    missing_fields: dict[str, list[str]] = defaultdict(list)
    for item in intents:
        for field, value in (
            ("image_path", item.image_path),
            ("wsi_id", item.wsi_id),
            ("patient_id", item.patient_id),
            ("source_dataset", item.source_dataset),
            ("ordinal_group_id", item.ordinal_group_id),
        ):
            if not value:
                missing_fields[field].append(item.sample_id)
        if item.qc_status not in {"accepted", "manual_review"}:
            errors.append(
                {
                    "check": "invalid_qc_status",
                    "sample_id": item.sample_id,
                    "qc_status": item.qc_status,
                }
            )
    for field, sample_ids in missing_fields.items():
        errors.append(
            {
                "check": f"missing_{field}",
                "count": len(sample_ids),
                "sample_ids": sample_ids[:50],
            }
        )

    cells: dict[tuple[str, str, str], list[BenchmarkIntent]] = defaultdict(list)
    for item in intents:
        cells[(item.organ, item.primitive, item.strength)].append(item)
    all_cells = set(cells) | set(shortfalls)
    for cell in sorted(all_cells):
        actual = len(cells.get(cell, ()))
        shortfall = shortfalls.get(cell)
        if actual == expected_per_cell and shortfall is None:
            continue
        if shortfall is None:
            errors.append(
                {
                    "check": "undocumented_cell_count",
                    "cell": list(cell),
                    "actual": actual,
                }
            )
            continue
        if (
            int(shortfall.get("available", -1)) != actual
            or int(shortfall.get("requested", -1)) != expected_per_cell
        ):
            errors.append(
                {
                    "check": "shortfall_mismatch",
                    "cell": list(cell),
                    "actual": actual,
                    "shortfall": shortfall,
                }
            )

    for cell, items in sorted(cells.items()):
        per_wsi = Counter(item.wsi_id for item in items)
        violations = {
            wsi_id: count
            for wsi_id, count in per_wsi.items()
            if count > max_per_wsi_per_cell
        }
        if violations:
            errors.append(
                {
                    "check": "wsi_cap_exceeded",
                    "cell": list(cell),
                    "violations": violations,
                }
            )

    ordinal: dict[str, list[BenchmarkIntent]] = defaultdict(list)
    for item in intents:
        ordinal[item.ordinal_group_id].append(item)
    for group_id, items in ordinal.items():
        references = {
            (
                item.mask_path,
                item.primitive,
                json.dumps(item.region_hint, sort_keys=True),
            )
            for item in items
        }
        if len(references) != 1:
            errors.append(
                {"check": "ordinal_reference_mismatch", "ordinal_group_id": group_id}
            )
        if len({item.strength for item in items}) < 3:
            warnings.append(
                {
                    "check": "ordinal_group_has_fewer_than_three_strengths",
                    "ordinal_group_id": group_id,
                }
            )

    present_specialized = {item.primitive for item in intents if item.specialized}
    missing_specialized = sorted(
        set(specialized_primitive_names()) - present_specialized
    )
    if require_all_specialized and missing_specialized:
        errors.append(
            {
                "check": "missing_specialized_primitives",
                "primitives": missing_specialized,
            }
        )

    return {
        "valid": not errors,
        "num_intents": len(intents),
        "num_cells": len(all_cells),
        "num_wsi": len({item.wsi_id for item in intents}),
        "num_patients": len({item.patient_id for item in intents}),
        "num_ordinal_groups": len(ordinal),
        "present_specialized_primitives": sorted(present_specialized),
        "missing_specialized_primitives": missing_specialized,
        "errors": errors,
        "warnings": warnings,
    }


def _read_shortfalls(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {
            (str(row["organ"]), str(row["primitive"]), str(row["strength"])): dict(row)
            for row in csv.DictReader(stream)
        }


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit CellViT supervision referenced by a Segmentator manifest."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
from PIL import Image


ALLOWED_VALUES = {0, 101, 102, 103, 104, 105}


def _nuclei_path(row: dict[str, object]) -> Path:
    dataset_root = Path(str(row["dataset_root"]))
    nuclei_dir = Path(str(row.get("nuclei_dir", "nuclei_masks")))
    if not nuclei_dir.is_absolute():
        nuclei_dir = dataset_root / nuclei_dir
    return nuclei_dir / str(row.get("nuclei", Path(str(row["image"])).name))


def _audit_row(payload: tuple[str, dict[str, object]]) -> dict[str, object]:
    split, row = payload
    path = _nuclei_path(row)
    result: dict[str, object] = {
        "split": split,
        "dataset_id": str(row["dataset_id"]),
        "sample_id": str(row.get("sample_id", row["image"])),
        "path": str(path),
        "exists": path.exists(),
        "nonempty": False,
        "values": [],
        "unexpected_values": [],
        "error": None,
    }
    if not path.exists():
        return result
    try:
        values = {int(value) for value in np.unique(np.asarray(Image.open(path).convert("L")))}
        result["values"] = sorted(values)
        result["unexpected_values"] = sorted(values - ALLOWED_VALUES)
        result["nonempty"] = bool(values & (ALLOWED_VALUES - {0}))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tasks = [
        (split, row)
        for split in args.splits
        for row in manifest.get(split, [])
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(_audit_row, tasks, chunksize=64))

    summary: dict[str, object] = {}
    failures = []
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row["dataset_id"]))].append(row)
        if not row["exists"] or row["error"] or row["unexpected_values"]:
            failures.append(row)
    for (split, dataset_id), group in sorted(grouped.items()):
        value_presence = Counter(
            value
            for row in group
            for value in row["values"]
        )
        summary.setdefault(split, {})[dataset_id] = {
            "records": len(group),
            "existing": sum(bool(row["exists"]) for row in group),
            "nonempty": sum(bool(row["nonempty"]) for row in group),
            "errors": sum(row["error"] is not None for row in group),
            "unexpected_label_files": sum(bool(row["unexpected_values"]) for row in group),
            "value_file_presence": {str(key): value for key, value in sorted(value_presence.items())},
        }
    report = {
        "manifest": str(args.manifest.resolve()),
        "allowed_values": sorted(ALLOWED_VALUES),
        "total_records": len(rows),
        "summary": summary,
        "failure_count": len(failures),
        "failures": failures[:100],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "total_records": len(rows), "failure_count": len(failures)}))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

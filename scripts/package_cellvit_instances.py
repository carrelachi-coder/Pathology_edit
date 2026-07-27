#!/usr/bin/env python3
"""Recover canonical instance-level CellViT JSONs for packaged benchmark masks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_cellvit_single_patch import rasterize_cells_json


STAGED_PREFIX = re.compile(r"^patch_\d+_")


def safe_source_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-items", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_index(raw_root: Path) -> dict[str, list[Path]]:
    candidates: dict[str, list[Path]] = {}
    for filelist in raw_root.rglob("cellvit_filelist.csv"):
        result_root = filelist.parent / "cellvit_results"
        with filelist.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                staged_stem = Path(row["path"]).stem
                source_stem = STAGED_PREFIX.sub("", staged_stem)
                path = result_root / f"{staged_stem}_cells.json"
                if path.is_file():
                    candidates.setdefault(source_stem, []).append(path)
    for paths in candidates.values():
        paths.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    return candidates


def main() -> int:
    args = parse_args()
    with args.patch_manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if args.max_items is not None:
        rows = rows[: args.max_items]
    index = candidate_index(args.raw_root)
    checkpoint_hash = sha256(args.checkpoint)
    args.output_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    skipped = 0
    failed = []
    for row in rows:
        annotation_id = row["annotation_id"]
        output = args.output_root / f"{annotation_id}.json"
        if output.is_file() and not args.overwrite:
            skipped += 1
            continue
        source_stem = safe_source_stem(row["stem"])
        image_path = Path(row["package_image_path"])
        mask_path = Path(row["package_cellvit_mask_path"])
        with Image.open(mask_path) as image:
            expected = np.asarray(image)
        selected = None
        diagnostics = []
        for candidate in index.get(source_stem, []):
            try:
                actual = rasterize_cells_json(candidate, image_path)
                difference = int(np.count_nonzero(actual != expected))
                diagnostics.append({"path": str(candidate), "different_pixels": difference})
                if difference == 0:
                    selected = candidate
                    break
            except Exception as exc:
                diagnostics.append({"path": str(candidate), "error": str(exc)})
        if selected is None:
            failed.append(
                {
                    "annotation_id": annotation_id,
                    "source_stem": source_stem,
                    "candidate_count": len(index.get(source_stem, [])),
                    "diagnostics": diagnostics[:10],
                }
            )
            continue
        payload = json.loads(selected.read_text(encoding="utf-8"))
        cells = payload.get("cells")
        if not isinstance(cells, list):
            failed.append({"annotation_id": annotation_id, "error": "missing cells list"})
            continue
        payload["benchmark_provenance"] = {
            "status": "completed",
            "annotation_id": annotation_id,
            "source_image": str(image_path),
            "source_semantic_mask": str(mask_path),
            "raw_cells_json": str(selected),
            "semantic_mask_exact_match": True,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "mpp": 0.25,
            "magnification": 40.0,
            "mask_type_map": {"1": 101, "2": 102, "3": 103, "4": 104, "5": 105},
        }
        output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        completed += 1
        if completed % 100 == 0 or completed + skipped == len(rows):
            print(f"[{completed + skipped}/{len(rows)}] packaged={completed} skipped={skipped}", flush=True)

    summary = {
        "requested": len(rows),
        "completed_this_run": completed,
        "skipped": skipped,
        "available": sum((args.output_root / f"{row['annotation_id']}.json").is_file() for row in rows),
        "failed": failed,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "exact_semantic_mask_match_required": True,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

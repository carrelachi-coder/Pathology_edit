#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, PngImagePlugin

PngImagePlugin.MAX_TEXT_CHUNK = 256 * 1024 * 1024
PngImagePlugin.MAX_TEXT_MEMORY = 1024 * 1024 * 1024

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from segmentator.patch_selection import compute_mask_features


_MASK_DIR: Path | None = None


def _initialize(mask_dir: str) -> None:
    global _MASK_DIR
    _MASK_DIR = Path(mask_dir)


def _recompute(row: dict[str, str]) -> dict[str, object]:
    if _MASK_DIR is None:
        raise RuntimeError("worker mask directory was not initialized")
    with Image.open(_MASK_DIR / row["filename"]) as image:
        mask = np.asarray(image, dtype=np.uint8)
    features = compute_mask_features(mask, row["organ"])
    output: dict[str, object] = dict(row)
    output.update(features.__dict__)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute selection features from saved epoch-7 masks.")
    parser.add_argument("--metrics-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--log-every", type=int, default=10000)
    args = parser.parse_args()

    rows_by_filename: dict[str, dict[str, str]] = {}
    for path in args.metrics_csv:
        for row in csv.DictReader(path.open()):
            rows_by_filename[row["filename"]] = row
    rows = [rows_by_filename[name] for name in sorted(rows_by_filename)]
    missing_masks = [row["filename"] for row in rows if not (args.mask_dir / row["filename"]).exists()]
    if missing_masks:
        raise RuntimeError(f"missing masks before feature recompute: {missing_masks[:10]} total={len(missing_masks)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    completed: list[dict[str, object]] = []
    with Pool(args.workers, initializer=_initialize, initargs=(str(args.mask_dir),)) as pool:
        for index, row in enumerate(pool.imap_unordered(_recompute, rows, chunksize=32), start=1):
            completed.append(row)
            if index % args.log_every == 0 or index == len(rows):
                rate = index / max(time.time() - start, 1e-6)
                print(f"features {index}/{len(rows)} rate={rate:.1f}/s", flush=True)
    completed.sort(key=lambda row: str(row["filename"]))
    fields: list[str] = []
    for row in completed:
        for key in row:
            if key not in fields:
                fields.append(key)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(completed)
    summary = {
        "rows": len(completed),
        "workers": args.workers,
        "elapsed_seconds": time.time() - start,
        "output": str(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

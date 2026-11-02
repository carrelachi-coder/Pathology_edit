#!/usr/bin/env python3
"""Batch CellViT++ inference using the same path as phase3_end_to_end_ui.py.

phase3_end_to_end_ui.py invokes scripts/run_cellvit_single_patch.py, which:
1. converts a PNG patch to an OpenSlide-readable pyramidal TIFF;
2. runs CellViT++ cellvit/detect_cells.py process_wsi;
3. rasterizes the resulting *_cells.json into a uint8 mask with IDs 101-105.

This script keeps those semantics but uses CellViT++ process_dataset so the model
loads once for all images.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid

from phase3_mask_edit.benchmark.experiment_identity import sha256_file
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

DEFAULT_REPO = Path("/home/lyw/wqx-DL/flow-edit/FlowEdit-main")
DEFAULT_CELLVIT_ROOT = DEFAULT_REPO / "CellViT-plus-plus-main" / "CellViT-plus-plus-main"
DEFAULT_MODEL = DEFAULT_CELLVIT_ROOT / "checkpoints" / "CellViT-SAM-H-x40-AMP-001.pth"
DEFAULT_RUNNER = DEFAULT_REPO / "scripts" / "run_cellvit_single_patch.py"
CELL_TYPE_TO_MASK_ID = {1: 101, 2: 102, 3: 103, 4: 104, 5: 105}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def safe_staged_stem(index: int, image_path: Path) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", image_path.stem).strip("_")
    return f"patch_{index:05d}_{safe}"


def write_openslide_tiff(image_path: Path, staged: Path) -> None:
    try:
        import pyvips  # type: ignore

        image = pyvips.Image.new_from_file(str(image_path), access="sequential")
        image.tiffsave(
            str(staged),
            tile=True,
            pyramid=True,
            compression="jpeg",
            tile_width=256,
            tile_height=256,
            bigtiff=True,
        )
        return
    except Exception:
        pass

    cmd = [
        "vips",
        "tiffsave",
        str(image_path),
        str(staged),
        "--tile",
        "--pyramid",
        "--compression",
        "jpeg",
        "--tile-width",
        "256",
        "--tile-height",
        "256",
        "--bigtiff",
    ]
    subprocess.run(cmd, check=True)


def rasterize_cells_json(cells_json: Path, image_path: Path) -> np.ndarray:
    sys.path.insert(0, str(DEFAULT_REPO))
    from scripts.run_cellvit_single_patch import rasterize_cells_json as rasterize_ui  # noqa: E402

    return rasterize_ui(cells_json, image_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch CellViT++ UI-equivalent segmentation.")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--json-output-dir",
        help="Optional canonical per-image CellViT JSON output directory.",
    )
    parser.add_argument("--raw-outdir", default="/data1/zhao/wqx/patch_selected_local/cellvit_raw_ui_equiv")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--cellvit-root", default=str(DEFAULT_CELLVIT_ROOT))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mpp", type=float, default=0.25)
    parser.add_argument("--magnification", type=float, default=40.0)
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shards", type=int, default=1, help="Total number of shards for parallel runs.")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index in [0, shards).")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--clean-staged", action="store_true")
    args = parser.parse_args()

    images_dir = Path(args.images_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    json_output_dir = Path(args.json_output_dir).resolve() if args.json_output_dir else None
    raw_outdir = Path(args.raw_outdir).resolve()
    staged_dir = raw_outdir / "openslide_inputs"
    result_dir = raw_outdir / "cellvit_results" / uuid.uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=True)
    if json_output_dir is not None:
        json_output_dir.mkdir(parents=True, exist_ok=True)
    staged_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    if args.shards < 1:
        raise ValueError("--shards must be >= 1")
    if not 0 <= args.shard_index < args.shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < shards")

    all_images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        all_images = all_images[: args.limit]
    images = [p for idx, p in enumerate(all_images) if idx % args.shards == args.shard_index]
    requested = len(images)
    model_sha256 = sha256_file(Path(args.model))
    reused = 0
    valid_zero = 0
    pending = []
    for p in images:
        output = output_dir / f"{p.stem}.png"
        canonical = json_output_dir / f"{p.stem}.json" if json_output_dir else None
        if output.exists() or (canonical and canonical.exists()):
            if not args.skip_existing or canonical is None or not canonical.is_file() or not output.is_file():
                raise ValueError(f"existing CellViT artifacts require validated reuse or new output directories: {p.name}")
            receipt = canonical.with_suffix(".completion.json")
            if not receipt.is_file() or json.loads(receipt.read_text()).get("json_sha256") != sha256_file(canonical):
                raise ValueError("CellViT detection JSON lacks matching independent completion receipt")
            payload = json.loads(canonical.read_text())
            provenance = payload.get("benchmark_provenance", {})
            if (provenance.get("status") != "completed"
                or provenance.get("image_sha256") != sha256_file(p)
                or provenance.get("model_sha256") != model_sha256
                or provenance.get("mask_sha256") != sha256_file(output)
                or any(provenance.get(k) != float(getattr(args, k)) for k in ("mpp", "magnification", "resolution"))):
                raise ValueError(f"CellViT existing run lacks matching identity: {p.name}; use new output directories")
            if payload.get("cells") == []:
                valid_zero += 1
            reused += 1
        else:
            pending.append(p)
    images = pending

    print(
        f"Staging {len(images)} images as OpenSlide TIFFs "
        f"(shard {args.shard_index + 1}/{args.shards})...",
        flush=True,
    )
    csv_path = raw_outdir / "cellvit_filelist.csv"
    mappings: list[tuple[Path, Path]] = []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "slide_mpp", "magnification"])
        writer.writeheader()
        for idx, image_path in enumerate(images):
            staged_path = staged_dir / f"{safe_staged_stem(idx, image_path)}_{sha256_file(image_path)[:16]}.tiff"
            if not staged_path.exists():
                write_openslide_tiff(image_path, staged_path)
            writer.writerow(
                {
                    "path": str(staged_path),
                    "slide_mpp": float(args.mpp),
                    "magnification": float(args.magnification),
                }
            )
            mappings.append((image_path, staged_path))
            if (idx + 1) % 100 == 0 or (idx + 1) == len(images):
                print(f"  staged {idx + 1}/{len(images)}", flush=True)

    detect_script = Path(args.cellvit_root) / "cellvit" / "detect_cells.py"
    cmd = [
        sys.executable,
        str(detect_script),
        "--model",
        str(Path(args.model).resolve()),
        "--outdir",
        str(result_dir),
        "--gpu",
        str(args.gpu),
        "--batch_size",
        str(args.batch_size),
        "--resolution",
        str(args.resolution),
        "--geojson",
        "process_dataset",
        "--filelist",
        str(csv_path),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(args.cellvit_root).resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    print("Running CellViT++ process_dataset...", flush=True)
    print(" ".join(cmd), flush=True)
    if mappings:
        subprocess.run(cmd, cwd=str(args.cellvit_root), env=env, check=True)

    print("Rasterizing CellViT JSON outputs...", flush=True)
    ok = 0
    failed = 0
    log_lines: list[str] = []
    for image_path, staged_path in mappings:
        cells_json = result_dir / f"{staged_path.stem}_cells.json"
        output_path = output_dir / f"{image_path.stem}.png"
        try:
            if not cells_json.exists():
                if json_output_dir is not None:
                    (json_output_dir / f"{image_path.stem}.json").write_text(
                        json.dumps({"benchmark_provenance": {
                            "status": "missing", "image": str(image_path),
                            "raw_cells_json": str(cells_json),
                        }}), encoding="utf-8",
                    )
                raise FileNotFoundError(f"missing CellViT raw output: {cells_json}")
            payload = json.loads(cells_json.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("cells"), list):
                raise ValueError(f"invalid CellViT raw cells list: {cells_json}")
            mask = rasterize_cells_json(cells_json, image_path)
            Image.fromarray(mask, mode="L").save(output_path)
            if json_output_dir is not None:
                payload["benchmark_provenance"] = {
                    "status": "completed",
                    "image_sha256": sha256_file(image_path),
                    "model_sha256": model_sha256,
                    "mask_sha256": sha256_file(output_path),
                    "mpp": float(args.mpp),
                    "image": str(image_path),
                    "raw_cells_json": str(cells_json),
                    "model": str(Path(args.model).resolve()),
                    "slide_mpp": float(args.mpp),
                    "magnification": float(args.magnification),
                    "resolution": float(args.resolution),
                }
                (json_output_dir / f"{image_path.stem}.json").write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
            if json_output_dir is not None:
                canonical = json_output_dir / f"{image_path.stem}.json"
                canonical.with_suffix(".completion.json").write_text(json.dumps({"json_sha256": sha256_file(canonical), "mask_sha256": sha256_file(output_path), "input_sha256": sha256_file(image_path), "model_sha256": model_sha256}), encoding="utf-8")
            cells = payload.get("cells", []) if isinstance(payload, dict) else []
            if not cells:
                valid_zero += 1
            ids = sorted(int(v) for v in np.unique(mask).tolist())
            log_lines.append(f"OK\t{image_path.name}\t{len(cells)} cells\tids={ids}")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            log_lines.append(f"ERR\t{image_path.name}\t{exc}")
    (output_dir / "batch_segment_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    if args.clean_staged:
        shutil.rmtree(staged_dir, ignore_errors=True)

    (output_dir / "measurement_summary.json").write_text(json.dumps({
        "requested": requested, "completed": ok, "reused_validated": reused, "valid_zero": valid_zero,
        "failed_or_missing": failed, "status": "incomplete" if failed else "complete",
    }, indent=2), encoding="utf-8")
    print(f"Done. ok={ok} failed={failed}. Output: {output_dir}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run CellViT++ on one patch and write a 0/101-105 nuclei mask PNG.

The script is intentionally shaped for the local UI command template:

    python scripts/run_cellvit_single_patch.py --image {image} --output-mask {output} --model path/to/CellViT.pth

It uses the local CellViT++ source checkout by default, then rasterizes the
resulting ``*_cells.json`` into the Phase 4/5 nuclei-mask convention:

    0   background
    101 neoplastic
    102 inflammatory
    103 connective
    104 dead
    105 epithelial
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_CELLVIT_ROOT = Path(
    r"D:\WQX\datasets\CellViT-plus-plus-main\CellViT-plus-plus-main"
)
CELL_TYPE_TO_MASK_ID = {
    1: 101,
    2: 102,
    3: 103,
    4: 104,
    5: 105,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image_path = Path(args.image).resolve()
    output_mask = Path(args.output_mask).resolve()
    cellvit_root = Path(args.cellvit_root).resolve()
    model_path = Path(args.model).resolve()

    _validate_inputs(image_path, output_mask, cellvit_root, model_path)
    run_dir = _resolve_run_dir(args, image_path, output_mask)
    cells_json = Path(args.cells_json).resolve() if args.cells_json else None

    if cells_json is None:
        cells_json = _run_cellvit(
            image_path=image_path,
            run_dir=run_dir,
            cellvit_root=cellvit_root,
            model_path=model_path,
            mpp=args.mpp,
            magnification=args.magnification,
            gpu=args.gpu,
            batch_size=args.batch_size,
            resolution=args.resolution,
            classifier_path=Path(args.classifier).resolve() if args.classifier else None,
            geojson=args.geojson,
            enforce_amp=args.enforce_amp,
        )

    mask = rasterize_cells_json(cells_json, image_path)
    output_mask.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(output_mask)

    summary = {
        "status": "completed",
        "image": str(image_path),
        "output_mask": str(output_mask),
        "cells_json": str(cells_json),
        "cellvit_root": str(cellvit_root),
        "model": str(model_path),
        "mpp": args.mpp,
        "magnification": args.magnification,
        "unique_mask_ids": sorted(int(v) for v in np.unique(mask).tolist()),
    }
    summary_path = output_mask.with_suffix(".cellvit_single_patch.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.clean_raw:
        shutil.rmtree(run_dir, ignore_errors=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local CellViT++ inference on one patch and write a nuclei mask."
    )
    parser.add_argument("--image", required=True, help="Input RGB patch image.")
    parser.add_argument("--output-mask", required=True, help="Output uint8 nuclei mask PNG.")
    parser.add_argument("--model", required=True, help="CellViT segmentation checkpoint .pth/.pt.")
    parser.add_argument(
        "--cellvit-root",
        default=str(DEFAULT_CELLVIT_ROOT),
        help="Local CellViT++ source root containing cellvit/detect_cells.py.",
    )
    parser.add_argument(
        "--cells-json",
        help="Optional existing *_cells.json to rasterize without running inference.",
    )
    parser.add_argument(
        "--raw-outdir",
        help="Optional directory for raw CellViT JSON/GeoJSON outputs.",
    )
    parser.add_argument("--mpp", type=float, default=0.25, help="Patch slide_mpp metadata.")
    parser.add_argument("--magnification", type=float, default=40.0, help="Patch magnification metadata.")
    parser.add_argument("--resolution", type=float, choices=(0.25, 0.5), default=0.25)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--classifier", help="Optional classifier checkpoint path.")
    parser.add_argument("--geojson", action="store_true", help="Also write raw CellViT GeoJSON outputs.")
    parser.add_argument("--enforce-amp", action="store_true")
    parser.add_argument("--clean-raw", action="store_true", help="Delete raw CellViT output dir after mask save.")
    return parser


def _validate_inputs(
    image_path: Path,
    output_mask: Path,
    cellvit_root: Path,
    model_path: Path,
) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"CellViT model checkpoint not found: {model_path}")
    detect_script = cellvit_root / "cellvit" / "detect_cells.py"
    if not detect_script.exists():
        raise FileNotFoundError(f"CellViT detect_cells.py not found: {detect_script}")
    output_mask.parent.mkdir(parents=True, exist_ok=True)


def _resolve_run_dir(args: argparse.Namespace, image_path: Path, output_mask: Path) -> Path:
    if args.raw_outdir:
        run_dir = Path(args.raw_outdir).resolve()
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = output_mask.parent / f"{image_path.stem}_cellvit_raw_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _run_cellvit(
    *,
    image_path: Path,
    run_dir: Path,
    cellvit_root: Path,
    model_path: Path,
    mpp: float,
    magnification: float,
    gpu: int,
    batch_size: int,
    resolution: float,
    classifier_path: Path | None,
    geojson: bool,
    enforce_amp: bool,
) -> Path:
    detect_script = cellvit_root / "cellvit" / "detect_cells.py"
    wsi_properties = json.dumps(
        {"slide_mpp": float(mpp), "magnification": float(magnification)}
    )
    cmd = [
        sys.executable,
        str(detect_script),
        "--model",
        str(model_path),
        "--outdir",
        str(run_dir),
        "--gpu",
        str(gpu),
        "--batch_size",
        str(batch_size),
        "--resolution",
        str(resolution),
    ]
    if classifier_path is not None:
        cmd.extend(["--classifier_path", str(classifier_path)])
    if geojson:
        cmd.append("--geojson")
    if enforce_amp:
        cmd.append("--enforce_amp")
    cmd.extend(
        [
            "process_wsi",
            "--wsi_path",
            str(image_path),
            "--wsi_properties",
            wsi_properties,
        ]
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(cellvit_root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=cellvit_root, env=env, check=True)

    expected = run_dir / f"{image_path.stem}_cells.json"
    if expected.exists():
        return expected
    candidates = sorted(run_dir.glob("*_cells.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"CellViT finished but no *_cells.json was found in {run_dir}")
    return candidates[0]


def rasterize_cells_json(cells_json: str | Path, image_path: str | Path) -> np.ndarray:
    cells_path = Path(cells_json)
    with Image.open(image_path) as image:
        width, height = image.size
    payload = json.loads(cells_path.read_text(encoding="utf-8"))
    cells = payload.get("cells", [])
    if not isinstance(cells, list):
        raise ValueError(f"Invalid cells JSON, expected list at key 'cells': {cells_path}")
    return rasterize_cells(cells, width=width, height=height)


def rasterize_cells(cells: list[dict[str, Any]], *, width: int, height: int) -> np.ndarray:
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for cell in cells:
        mask_id = CELL_TYPE_TO_MASK_ID.get(int(cell.get("type", 0)))
        contour = cell.get("contour", [])
        if mask_id is None or not isinstance(contour, list) or len(contour) < 3:
            continue
        points = _clip_contour(contour, width=width, height=height)
        if len(points) >= 3:
            draw.polygon(points, fill=int(mask_id))
    return np.asarray(canvas, dtype=np.uint8)


def _clip_contour(contour: list[Any], *, width: int, height: int) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for point in contour:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        x = min(max(x, 0), width - 1)
        y = min(max(y, 0), height - 1)
        points.append((x, y))
    return points


if __name__ == "__main__":
    raise SystemExit(main())

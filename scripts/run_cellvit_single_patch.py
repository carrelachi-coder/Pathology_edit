#!/usr/bin/env python3
"""Run source-checkout CellViT++ on one patch and write a nuclei mask PNG."""

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
    model_path = Path(args.model).resolve()
    cellvit_root = Path(args.cellvit_root).resolve()

    _validate_inputs(image_path, output_mask, model_path)
    run_dir = _resolve_run_dir(args, image_path, output_mask)
    cells_json = Path(args.cells_json).resolve() if args.cells_json else None

    if cells_json is None:
        cells_json = _run_cellvit_source_flow(
            image_path=image_path,
            run_dir=run_dir,
            cellvit_root=cellvit_root,
            model_path=model_path,
            mpp=args.mpp,
            magnification=args.magnification,
            gpu=args.gpu,
            batch_size=args.batch_size,
            cellvit_python=Path(args.cellvit_python).resolve(),
            resolution=args.resolution,
            classifier_path=Path(args.classifier).resolve() if args.classifier else None,
            geojson=args.geojson,
            enforce_amp=args.enforce_amp,
        )

    mask = rasterize_cells_json(cells_json, image_path)
    if not np.any(mask > 0):
        raise RuntimeError(
            "CellViT produced an empty nuclei mask. "
            f"Selected JSON: {cells_json}\n"
            f"Candidate JSON diagnostics:\n{_cellvit_json_diagnostics(run_dir)}"
        )

    output_mask.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(output_mask)
    summary = {
        "status": "completed",
        "image": str(image_path),
        "output_mask": str(output_mask),
        "cells_json": str(cells_json),
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
        description="Run CellViT++ inference on one patch and write a nuclei mask."
    )
    parser.add_argument("--image", required=True, help="Input RGB patch image.")
    parser.add_argument("--output-mask", required=True, help="Output uint8 nuclei mask PNG.")
    parser.add_argument("--model", required=True, help="CellViT segmentation checkpoint .pth/.pt.")
    parser.add_argument(
        "--cellvit-root",
        default=str(DEFAULT_CELLVIT_ROOT),
        help="CellViT++ source root containing cellvit/detect_cells.py.",
    )
    parser.add_argument("--cells-json", help="Optional existing *_cells.json to rasterize.")
    parser.add_argument("--raw-outdir", help="Optional directory for raw CellViT outputs.")
    parser.add_argument("--mpp", type=float, default=0.25, help="Patch slide_mpp metadata.")
    parser.add_argument("--magnification", type=float, default=40.0)
    parser.add_argument("--resolution", type=float, choices=(0.25, 0.5), default=0.25)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--cellvit-python",
        default=sys.executable,
        help="Python executable used to run the CellViT++ source script.",
    )
    parser.add_argument("--classifier", help="Optional classifier checkpoint path.")
    parser.add_argument("--geojson", action="store_true", default=True)
    parser.add_argument("--enforce-amp", action="store_true")
    parser.add_argument("--clean-raw", action="store_true")
    return parser


def _validate_inputs(image_path: Path, output_mask: Path, model_path: Path) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"CellViT model checkpoint not found: {model_path}")
    output_mask.parent.mkdir(parents=True, exist_ok=True)


def _resolve_run_dir(args: argparse.Namespace, image_path: Path, output_mask: Path) -> Path:
    if args.raw_outdir:
        run_dir = Path(args.raw_outdir).resolve()
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = output_mask.parent / f"{image_path.stem}_cellvit_raw_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resolve_cellvit_root(root: Path) -> Path:
    candidates = [
        root,
        root / "CellViT-plus-plus-main",
        root / root.name,
    ]
    for candidate in candidates:
        if (candidate / "cellvit" / "detect_cells.py").exists():
            return candidate
    matches = sorted(root.glob("*/cellvit/detect_cells.py")) if root.exists() else []
    if matches:
        return matches[0].parents[1]
    return root


def _run_cellvit_source_flow(
    *,
    image_path: Path,
    run_dir: Path,
    cellvit_root: Path,
    model_path: Path,
    mpp: float,
    magnification: float,
    gpu: int,
    batch_size: int,
    cellvit_python: Path,
    resolution: float,
    classifier_path: Path | None,
    geojson: bool,
    enforce_amp: bool,
) -> Path:
    cellvit_root = _resolve_cellvit_root(cellvit_root)
    detect_script = cellvit_root / "cellvit" / "detect_cells.py"
    if not detect_script.exists():
        raise FileNotFoundError(f"CellViT source script not found: {detect_script}")
    result_dir = run_dir / "cellvit_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    cellvit_image_path = _stage_openslide_image(
        image_path=image_path,
        run_dir=run_dir,
        cellvit_python=cellvit_python,
    )
    wsi_properties = json.dumps(
        {
            "slide_mpp": float(mpp),
            "magnification": float(magnification),
        }
    )
    cmd = [
        str(cellvit_python),
        str(detect_script),
        "--model",
        str(model_path),
        "--outdir",
        str(result_dir),
        "--gpu",
        str(gpu),
        "--batch_size",
        str(batch_size),
        "--resolution",
        str(resolution),
    ]
    if geojson:
        cmd.append("--geojson")
    if classifier_path is not None:
        cmd.extend(["--classifier_path", str(classifier_path)])
    if enforce_amp:
        cmd.append("--enforce_amp")
    cmd.extend(
        [
            "process_wsi",
            "--wsi_path",
            str(cellvit_image_path),
            "--wsi_properties",
            wsi_properties,
        ]
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(cellvit_root) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        subprocess.run(
            cmd,
            cwd=cellvit_root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        parts = [f"CellViT source process_wsi failed with exit code {exc.returncode}."]
        parts.append(f"Command: {exc.cmd!r}")
        stdout = (exc.stdout or "").strip()
        stderr = (exc.stderr or "").strip()
        if stdout:
            parts.append(f"stdout:\n{stdout}")
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        raise RuntimeError("\n".join(parts)) from exc

    cells_json = _find_cellvit_cells_json(run_dir=run_dir, result_dir=result_dir)
    if cells_json is None:
        raise FileNotFoundError(
            "CellViT finished but no JSON with a top-level 'cells' list was found.\n"
            f"Candidate JSON diagnostics:\n{_cellvit_json_diagnostics(run_dir)}"
        )
    if _json_cell_count(cells_json) == 0:
        raise RuntimeError(
            "CellViT finished but all detected-cell JSON files are empty.\n"
            f"Candidate JSON diagnostics:\n{_cellvit_json_diagnostics(run_dir)}"
        )
    return cells_json


def _stage_openslide_image(*, image_path: Path, run_dir: Path, cellvit_python: Path) -> Path:
    if image_path.suffix.lower() in {".tif", ".tiff"}:
        _verify_openslide_can_open(image_path, cellvit_python=cellvit_python)
        return image_path
    staged_dir = run_dir / "openslide_inputs"
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged = staged_dir / f"{image_path.stem}.tiff"
    _write_openslide_tiff(image_path=image_path, staged=staged)
    _verify_openslide_can_open(staged, cellvit_python=cellvit_python)
    return staged


def _write_openslide_tiff(*, image_path: Path, staged: Path) -> None:
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

    vips_cmd = [
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
    try:
        subprocess.run(vips_cmd, check=True, capture_output=True, text=True)
        return
    except Exception:
        pass

    with Image.open(image_path).convert("RGB") as image:
        image.save(staged, format="TIFF", compression="LZW")


def _verify_openslide_can_open(path: Path, *, cellvit_python: Path) -> None:
    code = (
        "from openslide import OpenSlide; "
        "import sys; "
        "slide = OpenSlide(sys.argv[1]); "
        "print(slide.dimensions)"
    )
    try:
        subprocess.run(
            [str(cellvit_python), "-c", code, str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(
            "OpenSlide cannot open the staged CellViT input image. "
            f"Path: {path}\n{details}"
        ) from exc

def _find_cellvit_cells_json(*, run_dir: Path, result_dir: Path) -> Path | None:
    candidates = sorted(
        {path for root in (result_dir, run_dir) if root.exists() for path in root.rglob("*.json")},
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        if path.name.endswith("_cells.json") and _json_cell_count(path) > 0:
            return path
    for path in candidates:
        if _json_cell_count(path) > 0:
            return path
    for path in candidates:
        if _json_cell_count(path) == 0:
            return path
    return None


def _json_cell_count(path: Path) -> int | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("cells"), list):
        return None
    return len(payload["cells"])


def _cellvit_json_diagnostics(root: Path) -> str:
    lines = []
    for path in sorted(root.rglob("*.json")):
        count = _json_cell_count(path)
        label = "no top-level cells list" if count is None else f"{count} cells"
        lines.append(f"- {path}: {label}")
    return "\n".join(lines[:50]) if lines else f"No JSON files found under {root}"


def rasterize_cells_json(cells_json: str | Path, image_path: str | Path) -> np.ndarray:
    cells_path = Path(cells_json)
    with Image.open(image_path) as image:
        width, height = image.size
    payload = json.loads(cells_path.read_text(encoding="utf-8"))
    cells = payload.get("cells", [])
    if not isinstance(cells, list):
        raise ValueError(f"Invalid cells JSON, expected list at key 'cells': {cells_path}")
    metadata = payload.get("wsi_metadata", {})
    return rasterize_cells(cells, width=width, height=height, metadata=metadata)


def rasterize_cells(
    cells: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    metadata: dict[str, Any] | None = None,
) -> np.ndarray:
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for cell in cells:
        mask_id = CELL_TYPE_TO_MASK_ID.get(int(cell.get("type", 0)))
        contour = _cellvit_contour_for_single_patch(
            cell, width=width, height=height, metadata=metadata
        )
        if mask_id is None or not isinstance(contour, list) or len(contour) < 3:
            continue
        points = _clip_contour(contour, width=width, height=height)
        if len(points) >= 3:
            draw.polygon(points, fill=int(mask_id))
    return np.asarray(canvas, dtype=np.uint8)


def _cellvit_contour_for_single_patch(
    cell: dict[str, Any],
    *,
    width: int,
    height: int,
    metadata: dict[str, Any] | None = None,
) -> list[Any]:
    """Convert CellViT's WSI-global contour back to this patch's pixel frame.

    A small single image is embedded into CellViT's 1024x1024 tile frame.  The
    JSON contour is in that tile/global frame, so translate it back to the
    input image frame before rasterizing a 512x512-style mask.
    """
    contour = cell.get("contour", [])
    offset = cell.get("offset_global")
    if not isinstance(contour, list) or not _is_xy_pair(offset):
        return contour

    patch_size = _metadata_number(metadata, "patch_size")
    if patch_size is not None:
        x_offset = max(float(patch_size) - width, 0.0) + float(offset[1])
        y_offset = max(float(patch_size) - height, 0.0) + float(offset[0])
        return _shift_contour(contour, x_offset=x_offset, y_offset=y_offset)

    source_order = _shift_contour(contour, x_offset=float(offset[1]), y_offset=float(offset[0]))
    flipped_order = _shift_contour(contour, x_offset=float(offset[0]), y_offset=float(offset[1]))
    if _contour_in_bounds_score(flipped_order, width=width, height=height) > _contour_in_bounds_score(
        source_order, width=width, height=height
    ):
        return flipped_order
    return source_order


def _metadata_number(metadata: dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(metadata, dict) or key not in metadata:
        return None
    try:
        return float(metadata[key])
    except (TypeError, ValueError):
        return None


def _shift_contour(contour: list[Any], *, x_offset: float, y_offset: float) -> list[Any]:
    local_contour = []
    for point in contour:
        if not _is_xy_pair(point):
            local_contour.append(point)
            continue
        local_contour.append([float(point[0]) - x_offset, float(point[1]) - y_offset])
    return local_contour


def _contour_in_bounds_score(contour: list[Any], *, width: int, height: int) -> int:
    score = 0
    for point in contour:
        if not _is_xy_pair(point):
            continue
        x = float(point[0])
        y = float(point[1])
        if 0 <= x < width and 0 <= y < height:
            score += 1
    return score


def _is_xy_pair(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return False
    try:
        float(value[0])
        float(value[1])
    except (TypeError, ValueError):
        return False
    return True


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

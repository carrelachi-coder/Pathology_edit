#!/usr/bin/env python3
"""Batch CellViT++ cell segmentation for patch images.

Loads the CellViT-SAM-H model ONCE and processes all images in a directory,
producing per-image nuclei masks with the Phase 3/4/5 convention:

    0   background
    101 neoplastic
    102 inflammatory
    103 connective
    104 dead
    105 epithelial

The staging / rasterization logic mirrors scripts/run_cellvit_single_patch.py
(the version shipped inside CellViT-plus-plus-main), which is what
phase3_end_to_end_ui.py invokes for single-patch CellViT inference.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

DEFAULT_CELLVIT_ROOT = Path(
    "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/CellViT-plus-plus-main/CellViT-plus-plus-main"
)
DEFAULT_MODEL = DEFAULT_CELLVIT_ROOT / "checkpoints" / "CellViT-SAM-H-x40-AMP-001.pth"
CELL_TYPE_TO_MASK_ID = {1: 101, 2: 102, 3: 103, 4: 104, 5: 105}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def stage_single_patch(image_path: Path, run_dir: Path, mpp: float) -> tuple[Path, int, int]:
    """Stage a single image as a CellViT patched-slide dataset (1024x1024 padded)."""
    patched_slide_path = run_dir / "patched_wsi"
    patches_dir = patched_slide_path / "patches"
    metadata_dir = patched_slide_path / "metadata"
    patches_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    patch_name = f"{image_path.stem}_row0_col0.png"
    staged_patch = patches_dir / patch_name
    with Image.open(image_path).convert("RGB") as image:
        width, height = image.size
        if (width, height) == (1024, 1024):
            image.save(staged_patch)
        else:
            canvas = Image.new("RGB", (1024, 1024), (255, 255, 255))
            canvas.paste(image, (0, 0))
            canvas.save(staged_patch)

    (metadata_dir / f"{patch_name}.yaml").write_text(
        "\n".join(
            ["row: 0", "col: 0", "x: 0", "y: 0", f"width: {width}", f"height: {height}"]
        )
        + "\n",
        encoding="utf-8",
    )
    (patched_slide_path / "metadata.yaml").write_text(
        "\n".join(
            [
                "patch_size: 1024",
                "patch_overlap: 64",
                f"target_patch_mpp: {float(mpp)}",
                f"base_mpp: {float(mpp)}",
                "downsampling: 1",
                "label_map:",
                "  0: Background",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    patch_metadata = [{patch_name: {"metadata_path": f"metadata/{patch_name}.yaml"}}]
    (patched_slide_path / "patch_metadata.json").write_text(
        json.dumps(patch_metadata, indent=2), encoding="utf-8"
    )
    return patched_slide_path, width, height


def rasterize_cells(cells: list[dict[str, Any]], width: int, height: int) -> np.ndarray:
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for cell in cells:
        mask_id = CELL_TYPE_TO_MASK_ID.get(int(cell.get("type", 0)))
        contour = cell.get("contour", [])
        if mask_id is None or not isinstance(contour, list) or len(contour) < 3:
            continue
        points: list[tuple[int, int]] = []
        for point in contour:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            x = min(max(int(round(float(point[0]))), 0), width - 1)
            y = min(max(int(round(float(point[1]))), 0), height - 1)
            points.append((x, y))
        if len(points) >= 3:
            draw.polygon(points, fill=int(mask_id))
    return np.asarray(canvas, dtype=np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch CellViT++ cell segmentation producing 0/101-105 nuclei masks."
    )
    parser.add_argument("--images-dir", required=True, help="Directory of input patch images.")
    parser.add_argument("--output-dir", required=True, help="Directory for output nuclei mask PNGs.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="CellViT checkpoint .pth.")
    parser.add_argument("--cellvit-root", default=str(DEFAULT_CELLVIT_ROOT))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mpp", type=float, default=0.25)
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--geojson", action="store_true")
    parser.add_argument("--enforce-amp", action="store_true")
    parser.add_argument("--tmp-dir", default="/tmp/cellvit_batch_tmp")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N images.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip images that already have an output mask.")
    args = parser.parse_args()

    cellvit_root = Path(args.cellvit_root).resolve()
    sys.path.insert(0, str(cellvit_root))
    from cellvit.data.dataclass.wsi import WSI  # noqa: E402
    from cellvit.inference.inference_disk import CellViTInference  # noqa: E402

    images_dir = Path(args.images_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        images = images[: args.limit]

    print(f"Loading CellViT model: {args.model}", flush=True)
    t0 = time.time()
    celldetector = CellViTInference(
        model_path=args.model,
        gpu=args.gpu,
        batch_size=args.batch_size,
        geojson=args.geojson,
        enforce_mixed_precision=args.enforce_amp,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    # Monkey-patch: CellViT++ crashes (IndexError) in OverlapCellCleaner when
    # zero cells are detected.  Handle empty cell lists gracefully so those
    # patches simply produce an all-background mask.
    _orig_post_process = celldetector._post_process_edge_cells

    def _safe_post_process_edge_cells(cell_list):
        if not cell_list:
            return []
        return _orig_post_process(cell_list=cell_list)

    celldetector._post_process_edge_cells = _safe_post_process_edge_cells

    print(f"Processing {len(images)} images -> {output_dir}", flush=True)

    success = 0
    skipped = 0
    failed = 0
    log_path = output_dir / "batch_segment_log.txt"
    log_lines: list[str] = []

    for i, image_path in enumerate(images):
        output_path = output_dir / f"{image_path.stem}.png"
        if args.skip_existing and output_path.exists():
            skipped += 1
            continue

        run_dir = tmp_dir / image_path.stem
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)

        width: int | None = None
        height: int | None = None
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            patched_slide_path, width, height = stage_single_patch(image_path, run_dir, args.mpp)

            wsi_file = WSI(
                name=image_path.stem,
                patient=image_path.stem,
                slide_path=str(image_path),
                patched_slide_path=str(patched_slide_path),
            )
            celldetector.process_wsi(wsi_file, resolution=args.resolution)

            cells_json = patched_slide_path / "cell_detection" / "cells.json"
            payload = json.loads(cells_json.read_text(encoding="utf-8"))
            cells = payload.get("cells", [])

            mask = rasterize_cells(cells, width, height)
            Image.fromarray(mask, mode="L").save(output_path)
            success += 1
            n_cells = len(cells)
            unique_ids = sorted(int(v) for v in np.unique(mask).tolist())
            log_lines.append(f"OK\t{image_path.name}\t{n_cells} cells\tids={unique_ids}")
        except Exception as exc:  # noqa: BLE001
            # CellViT++ can throw IndexError/KeyError('type') for valid patches
            # where no nuclei are detected.  Treat these as an empty cell mask.
            if width is not None and height is not None and str(exc) in {"list index out of range", "'type'"}:
                mask = np.zeros((height, width), dtype=np.uint8)
                Image.fromarray(mask, mode="L").save(output_path)
                success += 1
                log_lines.append(f"OK_EMPTY\t{image_path.name}\t0 cells\tids=[0]\t{exc}")
            else:
                failed += 1
                log_lines.append(f"ERR\t{image_path.name}\t{exc}")

        shutil.rmtree(run_dir, ignore_errors=True)

        if (i + 1) % 20 == 0 or (i + 1) == len(images):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(images) - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i + 1}/{len(images)}] ok={success} skip={skipped} err={failed}"
                f"  {rate:.1f} img/s  ETA {eta / 60:.1f} min",
                flush=True,
            )
            log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(
        f"\nDone in {(time.time() - t0) / 60:.1f} min. "
        f"success={success} skipped={skipped} failed={failed}",
        flush=True,
    )
    print(f"Log: {log_path}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

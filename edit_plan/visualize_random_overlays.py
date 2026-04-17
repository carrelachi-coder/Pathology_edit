#!/usr/bin/env python
"""Randomly visualize image, tissue mask, and cell mask overlays."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_config.unified_labels import CELL_COLOR_MAP, UNIFIED_COLOR_MAP  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "edit_datasets",
        help="Root containing dataset folders and manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "edit_plan" / "overlay_visualizations",
        help="Directory for saved overlay images.",
    )
    parser.add_argument("--samples-per-dataset", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260417)
    parser.add_argument("--tissue-alpha", type=float, default=0.35)
    parser.add_argument("--cell-alpha", type=float, default=0.80)
    return parser.parse_args()


def read_metadata(dataset_dir: Path, source_dir: Path | None = None) -> list[dict]:
    search_dirs = [dataset_dir]
    if source_dir and source_dir != dataset_dir:
        search_dirs.append(source_dir)

    last_error: Exception | None = None
    for base_dir in search_dirs:
        try:
            return read_metadata_from_dir(base_dir)
        except (FileNotFoundError, PermissionError) as exc:
            last_error = exc
    raise last_error or FileNotFoundError(f"No metadata found for {dataset_dir}")


def read_metadata_from_dir(base_dir: Path) -> list[dict]:
    json_path = base_dir / "metadata.json"
    jsonl_path = base_dir / "metadata.jsonl"
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in ("samples", "items", "data"):
                if isinstance(data.get(key), list):
                    return data[key]
        if isinstance(data, list):
            return data
        raise ValueError(f"Unsupported metadata.json format: {json_path}")
    if jsonl_path.exists():
        rows = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    raise FileNotFoundError(f"No metadata.json or metadata.jsonl in {base_dir}")


def build_source_map(dataset_root: Path) -> dict[str, Path]:
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    return {
        item["dataset"]: Path(item["source"])
        for item in manifest.get("datasets", [])
        if item.get("status") == "ready" and item.get("source")
    }


def resolve_image_path(record: dict, dataset_dir: Path, source_dir: Path | None) -> Path:
    image_rel = Path(record["image"])
    candidates = [
        dataset_dir / image_rel,
        source_dir / image_rel if source_dir else None,
        dataset_dir / "images" / image_rel.name,
        source_dir / "images" / image_rel.name if source_dir else None,
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found for {record.get('image')} in {dataset_dir.name}")


def mask_path(dataset_dir: Path, source_dir: Path | None, subdir: str, record: dict) -> Path:
    name = Path(record.get("conditioning_image", record["image"])).name
    candidates = [dataset_dir / subdir / name]
    if source_dir:
        candidates.append(source_dir / subdir / name)
    for path in candidates:
        if path.exists():
            try:
                with path.open("rb"):
                    pass
                return path
            except PermissionError:
                continue
    raise FileNotFoundError(f"Readable mask not found: {dataset_dir / subdir / name}")


def color_mask(mask_image: Image.Image, color_map: dict[int, list[int]]) -> np.ndarray:
    raw_mask = np.asarray(mask_image)
    if raw_mask.ndim == 2:
        colored = np.zeros((*raw_mask.shape, 3), dtype=np.uint8)
        for label_id, color in color_map.items():
            colored[raw_mask == label_id] = np.asarray(color, dtype=np.uint8)
        return colored

    mask = np.asarray(mask_image.convert("RGB"), dtype=np.uint8)
    colored = np.zeros_like(mask, dtype=np.uint8)
    for label_id, color in color_map.items():
        color_arr = np.asarray(color, dtype=np.uint8)
        matches = (raw_mask[..., 0] == label_id) | np.all(mask == color_arr, axis=-1)
        colored[matches] = color_arr
    return colored


def overlay(base: Image.Image, tissue: Image.Image, cells: Image.Image, tissue_alpha: float, cell_alpha: float) -> Image.Image:
    base_np = np.asarray(base.convert("RGB"), dtype=np.float32)
    tissue_np = color_mask(tissue, UNIFIED_COLOR_MAP).astype(np.float32)
    cell_np = color_mask(cells, CELL_COLOR_MAP).astype(np.float32)

    out = base_np.copy()
    tissue_pixels = tissue_np.sum(axis=-1) > 0
    cell_pixels = cell_np.sum(axis=-1) > 0

    out[tissue_pixels] = (1.0 - tissue_alpha) * out[tissue_pixels] + tissue_alpha * tissue_np[tissue_pixels]
    out[cell_pixels] = (1.0 - cell_alpha) * out[cell_pixels] + cell_alpha * cell_np[cell_pixels]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def add_label(image: Image.Image, label: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 28), "white")
    canvas.paste(image, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), label, fill=(0, 0, 0), font=ImageFont.load_default())
    return canvas


def make_triptych(base: Image.Image, tissue: Image.Image, cells: Image.Image, combined: Image.Image, label: str) -> Image.Image:
    panels = [
        add_label(base.convert("RGB"), "image"),
        add_label(overlay(base, tissue, Image.new("RGB", base.size), 0.45, 0.0), "image + tissuemask"),
        add_label(overlay(base, Image.new("RGB", base.size), cells, 0.0, 0.85), "image + cellmask"),
        add_label(combined, "image + tissue + cell"),
    ]
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels) + 28
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), label, fill=(0, 0, 0), font=ImageFont.load_default())
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 28))
        x += panel.width
    return canvas


def make_contact_sheet(images: list[Image.Image], cols: int = 2) -> Image.Image:
    if not images:
        raise ValueError("No images for contact sheet")
    rows = (len(images) + cols - 1) // cols
    cell_w = max(image.width for image in images)
    cell_h = max(image.height for image in images)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    for idx, image in enumerate(images):
        row, col = divmod(idx, cols)
        sheet.paste(image, (col * cell_w, row * cell_h))
    return sheet


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    source_map = build_source_map(args.dataset_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = sorted(
        path for path in args.dataset_root.iterdir()
        if path.is_dir() and ((path / "metadata.json").exists() or (path / "metadata.jsonl").exists())
    )
    if not dataset_dirs:
        raise FileNotFoundError(f"No dataset metadata found under {args.dataset_root}")

    summary = {}
    for dataset_dir in dataset_dirs:
        source_dir = source_map.get(dataset_dir.name)
        records = read_metadata(dataset_dir, source_dir)
        picked = rng.sample(records, min(args.samples_per_dataset, len(records)))
        dataset_out = args.output_dir / dataset_dir.name
        dataset_out.mkdir(parents=True, exist_ok=True)
        triptychs = []

        for idx, record in enumerate(picked, 1):
            image_path = resolve_image_path(record, dataset_dir, source_dir)
            tissue_path = mask_path(dataset_dir, source_dir, "tissue_masks", record)
            cell_path = mask_path(dataset_dir, source_dir, "nuclei_masks", record)

            base = Image.open(image_path).convert("RGB")
            tissue = Image.open(tissue_path).convert("RGB").resize(base.size, Image.Resampling.NEAREST)
            cells = Image.open(cell_path).convert("RGB").resize(base.size, Image.Resampling.NEAREST)
            combined = overlay(base, tissue, cells, args.tissue_alpha, args.cell_alpha)

            stem = Path(record["image"]).stem
            combined.save(dataset_out / f"{idx:02d}_{stem}_overlay.png")
            triptych = make_triptych(base, tissue, cells, combined, f"{dataset_dir.name} #{idx}: {stem}")
            triptych.save(dataset_out / f"{idx:02d}_{stem}_panels.png")
            triptychs.append(triptych)

        make_contact_sheet(triptychs, cols=2).save(args.output_dir / f"{dataset_dir.name}_contact_sheet.png")
        summary[dataset_dir.name] = len(picked)
        print(f"{dataset_dir.name}: saved {len(picked)} samples to {dataset_out}")

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "samples": summary}, f, indent=2)
    print(f"Done. Output: {args.output_dir}")


if __name__ == "__main__":
    main()

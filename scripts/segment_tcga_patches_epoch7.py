#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, PngImagePlugin
import torch
import torchvision.transforms.functional as TF

PngImagePlugin.MAX_TEXT_CHUNK = 256 * 1024 * 1024
PngImagePlugin.MAX_TEXT_MEMORY = 1024 * 1024 * 1024

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from segmentator.data import normalize_image_tensor
from segmentator.inference import load_checkpoint
from segmentator.patch_selection import compute_image_quality, compute_mask_features


NUM_CLASSES = 8
CLASS_NAMES = (
    "background",
    "tumor",
    "stroma",
    "necrosis",
    "immune_infiltrate",
    "normal_epithelium",
    "blood_vessel",
    "other_tissue",
)


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    boundary = np.zeros(mask.shape, dtype=bool)
    horizontal = mask[:, 1:] != mask[:, :-1]
    vertical = mask[1:, :] != mask[:-1, :]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary


def _read_manifest(path: Path, shard_index: int, num_shards: int, max_images: int) -> list[dict[str, str]]:
    rows = sorted(csv.DictReader(path.open()), key=lambda row: row["filename"])
    selected = [row for index, row in enumerate(rows) if index % num_shards == shard_index]
    return selected[:max_images] if max_images > 0 else selected


def _existing_rows(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    return {row["filename"] for row in csv.DictReader(path.open()) if row.get("filename")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run epoch-7 segmentator and emit selection metrics.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mask-out-dir", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--metric-workers", type=int, default=16)
    parser.add_argument("--save-workers", type=int, default=8)
    parser.add_argument("--compress-level", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()

    args.mask_out_dir.mkdir(parents=True, exist_ok=True)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    rows = _read_manifest(args.manifest, args.shard_index, args.num_shards, args.max_images)
    completed = _existing_rows(args.csv_out)
    todo = [row for row in rows if row["filename"] not in completed or not (args.mask_out_dir / row["filename"]).exists()]
    print(
        f"[shard {args.shard_index}/{args.num_shards}] requested={len(rows)} completed={len(rows)-len(todo)} todo={len(todo)}",
        flush=True,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    model = load_checkpoint(
        args.checkpoint,
        num_classes=NUM_CLASSES,
        freeze_encoder=True,
        decoder="mask2former",
        mask2former_queries=100,
        mask2former_ignore_index=255,
    ).to(device).eval()
    print(f"model loaded checkpoint={args.checkpoint} device={device}", flush=True)

    fields = [
        "filename",
        "stem",
        "case_id",
        "wsi",
        "x",
        "y",
        "project_id",
        "organ",
        "image_path",
        "text_path",
        "width",
        "height",
        "mean_confidence",
        "mean_entropy",
        "boundary_entropy",
        "tissue_fraction",
        "laplacian_variance",
        "tenengrad",
        "dynamic_range",
        "near_black_fraction",
        "near_white_tissue_fraction",
        "mean_saturation",
        "tissue_pixels",
        "other_fraction",
        "valid_class_count",
        "interface_density",
        "class_entropy",
        "shape_irregularity",
        "speckle_fraction",
        "positive_complexity",
        "pre_normalized_score",
    ] + [f"pix_{name}" for name in CLASS_NAMES]
    write_header = not args.csv_out.exists() or args.csv_out.stat().st_size == 0
    csv_handle = args.csv_out.open("a", newline="")
    writer = csv.DictWriter(csv_handle, fieldnames=fields)
    if write_header:
        writer.writeheader()
        csv_handle.flush()

    load_pool = ThreadPoolExecutor(max_workers=args.num_workers)
    metric_pool = ThreadPoolExecutor(max_workers=args.metric_workers)
    save_pool = ThreadPoolExecutor(max_workers=args.save_workers)
    failures: list[dict[str, str]] = []

    def load_image(row: dict[str, str]) -> tuple[dict[str, str], Image.Image | None]:
        try:
            with Image.open(row["image_path"]) as image:
                return row, image.convert("RGB")
        except Exception as exc:
            failures.append({"filename": row["filename"], "error": f"{type(exc).__name__}: {exc}"})
            return row, None

    def save_mask(filename: str, mask: np.ndarray) -> None:
        Image.fromarray(mask, mode="L").save(
            args.mask_out_dir / filename,
            compress_level=args.compress_level,
        )

    def compute_output_row(
        manifest_row: dict[str, str],
        image: Image.Image,
        mask: np.ndarray,
        probs: np.ndarray,
        entropy: np.ndarray,
    ) -> dict[str, object]:
        image_array = np.asarray(image, dtype=np.uint8)
        quality = compute_image_quality(image_array, tissue_mask=mask != 0)
        features = compute_mask_features(mask, manifest_row["organ"])
        counts = np.bincount(mask.reshape(-1), minlength=NUM_CLASSES)
        boundary = _boundary_mask(mask)
        confidence = probs.max(axis=0)
        output_row: dict[str, object] = {
            key: manifest_row[key]
            for key in ("filename", "stem", "case_id", "wsi", "x", "y", "project_id", "organ", "image_path", "text_path")
        }
        output_row.update(
            {
                "width": image.width,
                "height": image.height,
                "mean_confidence": float(confidence.mean()),
                "mean_entropy": float(entropy.mean()),
                "boundary_entropy": float(entropy[boundary].mean()) if np.any(boundary) else 0.0,
                **quality.__dict__,
                **features.__dict__,
            }
        )
        output_row.pop("organ", None)
        output_row["organ"] = manifest_row["organ"]
        for class_id, class_name in enumerate(CLASS_NAMES):
            output_row[f"pix_{class_name}"] = int(counts[class_id])
        return output_row

    start = time.time()
    processed = 0
    for offset in range(0, len(todo), args.batch_size):
        loaded = list(load_pool.map(load_image, todo[offset : offset + args.batch_size]))
        valid = [(row, image) for row, image in loaded if image is not None]
        if not valid:
            continue
        tensors = [normalize_image_tensor(TF.to_tensor(image)) for _, image in valid]
        batch = torch.stack(tensors).to(device, non_blocking=True)
        with torch.inference_mode():
            if args.amp and device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    outputs = model(batch)
            else:
                outputs = model(batch)
        predictions = outputs["pred"].cpu().numpy().astype(np.uint8)
        probabilities = outputs["probs"].float().cpu().numpy()
        entropies = outputs["entropy"].float().cpu().numpy()

        metric_futures = []
        for (manifest_row, image), mask, probs, entropy in zip(valid, predictions, probabilities, entropies):
            save_pool.submit(save_mask, manifest_row["filename"], mask.copy())
            metric_futures.append(metric_pool.submit(compute_output_row, manifest_row, image, mask, probs, entropy))
        for future in metric_futures:
            output_row = future.result()
            writer.writerow(output_row)
        csv_handle.flush()
        processed += len(valid)
        if processed % args.log_every < len(valid) or offset + args.batch_size >= len(todo):
            elapsed = max(time.time() - start, 1e-6)
            rate = processed / elapsed
            eta = (len(todo) - processed) / max(rate, 1e-6)
            print(
                f"[shard {args.shard_index}] {processed}/{len(todo)} rate={rate:.1f} img/s ETA={eta/60:.1f} min failures={len(failures)}",
                flush=True,
            )

    save_pool.shutdown(wait=True)
    metric_pool.shutdown(wait=True)
    load_pool.shutdown(wait=True)
    csv_handle.close()
    summary = {
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "requested": len(rows),
        "previously_completed": len(rows) - len(todo),
        "processed": processed,
        "failures": failures,
        "elapsed_seconds": time.time() - start,
    }
    args.csv_out.with_suffix(args.csv_out.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

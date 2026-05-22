from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segmentator.config import DatasetManifest
from segmentator.data import build_manifest


def _count_records(records) -> tuple[Counter[int], Counter[int]]:
    pixel_counts: Counter[int] = Counter()
    patch_counts: Counter[int] = Counter()
    for record in records:
        mask = np.array(Image.open(record.mask_path).convert("L"), dtype=np.int64)
        values, value_counts = np.unique(mask, return_counts=True)
        pixel_counts.update({int(v): int(c) for v, c in zip(values, value_counts)})
        patch_counts.update({int(v): 1 for v in values})
    return pixel_counts, patch_counts


def _print_split(name: str, manifest: DatasetManifest, records, num_classes: int) -> None:
    pixel_counts, patch_counts = _count_records(records)
    total_pixels = sum(pixel_counts.values())
    print(f"\n{name}: patches={len(records)}, pixels={total_pixels}")
    for idx in range(num_classes):
        class_name = manifest.classes[idx] if idx < len(manifest.classes) else f"class_{idx}"
        pixels = pixel_counts.get(idx, 0)
        print(
            f"{idx:>2d} {class_name:18s} "
            f"pixels={pixels:>12d} "
            f"ratio={pixels / max(total_pixels, 1):.6f} "
            f"patches={patch_counts.get(idx, 0):>6d} "
            f"patch_ratio={patch_counts.get(idx, 0) / max(len(records), 1):.4f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Check mask value range for the Stage 4 split.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--train-split", type=int, default=1000)
    parser.add_argument("--val-split", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=8)
    args = parser.parse_args()

    manifest = build_manifest(Path(args.dataset_root), args.train_split, args.val_split, seed=args.seed)
    counts: Counter[int] = Counter()
    bad_files = []
    for record in [*manifest.train, *manifest.val]:
        mask = np.array(Image.open(record.mask_path).convert("L"), dtype=np.int64)
        values, value_counts = np.unique(mask, return_counts=True)
        counts.update({int(v): int(c) for v, c in zip(values, value_counts)})
        bad = [int(v) for v in values if v < 0 or v >= args.num_classes]
        if bad:
            bad_files.append((record.mask_path.name, bad))

    print("values:", dict(sorted(counts.items())))
    print("bad file count:", len(bad_files))
    for name, values in bad_files[:20]:
        print(name, values)
    _print_split("train", manifest, manifest.train, args.num_classes)
    _print_split("val", manifest, manifest.val, args.num_classes)
    _print_split("train+val", manifest, [*manifest.train, *manifest.val], args.num_classes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

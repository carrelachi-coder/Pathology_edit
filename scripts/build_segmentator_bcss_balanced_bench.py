from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
from typing import Iterable

import numpy as np
from PIL import Image

CLASS_NAMES = {
    0: "background",
    1: "tumor",
    2: "stroma",
    3: "necrosis",
    4: "immune_infiltrate",
    5: "normal_epithelium",
    6: "blood_vessel",
    7: "other_tissue",
}

RARE_CLASSES = (3, 6, 5, 4)
CORE_CLASSES = (1, 2, 3, 4, 5, 6)


def _sorted_pngs(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() == ".png")


def _resolve_dirs(root: Path) -> tuple[Path, Path]:
    if (root / "patches" / "images").exists():
        return root / "patches" / "images", root / "patches" / "tissue_masks"
    return root / "images", root / "tissue_masks"


def _mask_counts(mask_path: Path, num_classes: int) -> list[int]:
    mask = np.array(Image.open(mask_path).convert("L"), dtype=np.int64)
    valid = (mask >= 0) & (mask < num_classes)
    mask = np.where(valid, mask, num_classes - 1)
    return np.bincount(mask.reshape(-1), minlength=num_classes).astype(np.int64).tolist()


def _verify_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


def _summarize(records: Iterable[dict[str, object]], num_classes: int) -> dict[str, object]:
    pixel_counts = [0] * num_classes
    patch_counts = [0] * num_classes
    records = list(records)
    for record in records:
        counts = record["class_pixels"]
        assert isinstance(counts, list)
        for idx, count in enumerate(counts):
            pixel_counts[idx] += int(count)
            if int(count) > 0:
                patch_counts[idx] += 1
    total_pixels = sum(pixel_counts)
    return {
        "patches": len(records),
        "total_pixels": total_pixels,
        "classes": [
            {
                "id": idx,
                "name": CLASS_NAMES.get(idx, f"class_{idx}"),
                "pixels": pixel_counts[idx],
                "pixel_ratio": pixel_counts[idx] / max(total_pixels, 1),
                "patches": patch_counts[idx],
                "patch_ratio": patch_counts[idx] / max(len(records), 1),
            }
            for idx in range(num_classes)
        ],
    }


def _greedy_select(
    pool: list[dict[str, object]],
    count: int,
    rng: random.Random,
    rare_classes: tuple[int, ...],
    min_rare_patches: int,
    num_classes: int,
) -> list[dict[str, object]]:
    remaining = pool[:]
    rng.shuffle(remaining)
    selected: list[dict[str, object]] = []
    patch_counts = [0] * num_classes
    pixel_counts = [0] * num_classes

    def add(record: dict[str, object]) -> None:
        selected.append(record)
        counts = record["class_pixels"]
        assert isinstance(counts, list)
        for idx, value in enumerate(counts):
            value = int(value)
            pixel_counts[idx] += value
            if value > 0:
                patch_counts[idx] += 1

    while remaining and len(selected) < count:
        deficits = {
            cls: max(0, min_rare_patches - patch_counts[cls])
            for cls in rare_classes
        }
        if not any(deficits.values()):
            break
        best_idx = 0
        best_score = -1.0
        for idx, record in enumerate(remaining):
            counts = record["class_pixels"]
            assert isinstance(counts, list)
            score = 0.0
            for cls, deficit in deficits.items():
                if deficit <= 0:
                    continue
                cls_pixels = int(counts[cls])
                if cls_pixels > 0:
                    score += deficit * (1.0 + np.log1p(cls_pixels))
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_score <= 0:
            break
        add(remaining.pop(best_idx))

    if len(selected) < count:
        remaining.sort(
            key=lambda record: (
                -sum(int(record["class_pixels"][cls]) > 0 for cls in CORE_CLASSES),
                str(record["image"]),
            )
        )
        selected_names = {str(record["image"]) for record in selected}
        for record in remaining:
            if len(selected) >= count:
                break
            if str(record["image"]) not in selected_names:
                add(record)

    return selected


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build full BCSS class stats and a class-balanced fixed segmentator bench."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--train-split", type=int, default=5000)
    parser.add_argument("--val-split", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--min-rare-train-patches", type=int, default=250)
    parser.add_argument("--min-rare-val-patches", type=int, default=40)
    parser.add_argument("--output", default="segmentator_runs/stage4_bcss_5000_500_balanced_manifest.json")
    parser.add_argument("--stats-output", default="segmentator_runs/stage4_bcss_full_class_stats.json")
    parser.add_argument(
        "--no-verify-images",
        action="store_true",
        help="Skip image integrity checks before writing the bench manifest.",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root)
    images_dir, masks_dir = _resolve_dirs(root)
    image_paths = _sorted_pngs(images_dir)
    if len(image_paths) < args.train_split + args.val_split:
        raise SystemExit(f"need {args.train_split + args.val_split} images, found {len(image_paths)}")

    records = []
    missing_masks = []
    skipped_corrupt = []
    for image_path in image_paths:
        mask_path = masks_dir / image_path.name
        if not mask_path.exists():
            missing_masks.append(image_path.name)
            continue
        if not args.no_verify_images:
            try:
                _verify_image(image_path)
                _verify_image(mask_path)
            except Exception as exc:
                skipped_corrupt.append(
                    {
                        "image": image_path.name,
                        "mask": mask_path.name,
                        "error": str(exc),
                    }
                )
                continue
        records.append(
            {
                "image": image_path.name,
                "mask": mask_path.name,
                "class_pixels": _mask_counts(mask_path, args.num_classes),
            }
        )
    if missing_masks:
        raise SystemExit(f"missing masks for {len(missing_masks)} images; first: {missing_masks[:5]}")
    if len(records) < args.train_split + args.val_split:
        raise SystemExit(
            f"need {args.train_split + args.val_split} valid image/mask pairs after verification, "
            f"found {len(records)}; skipped corrupt={len(skipped_corrupt)}"
        )

    full_summary = _summarize(records, args.num_classes)
    stats_path = Path(args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(
            {
                "dataset_root": str(root),
                "images_dir": str(images_dir),
                "masks_dir": str(masks_dir),
                "skipped_corrupt": skipped_corrupt,
                "summary": full_summary,
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rng = random.Random(args.seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    val = _greedy_select(
        shuffled,
        args.val_split,
        rng,
        rare_classes=RARE_CLASSES,
        min_rare_patches=args.min_rare_val_patches,
        num_classes=args.num_classes,
    )
    val_names = {str(record["image"]) for record in val}
    train_pool = [record for record in shuffled if str(record["image"]) not in val_names]
    train = _greedy_select(
        train_pool,
        args.train_split,
        rng,
        rare_classes=RARE_CLASSES,
        min_rare_patches=args.min_rare_train_patches,
        num_classes=args.num_classes,
    )

    manifest = {
        "dataset_root": str(root),
        "seed": args.seed,
        "strategy": "rare-class greedy coverage, then core-class diversity fill",
        "train_split": args.train_split,
        "val_split": args.val_split,
        "rare_classes": [CLASS_NAMES[idx] for idx in RARE_CLASSES],
        "min_rare_train_patches": args.min_rare_train_patches,
        "min_rare_val_patches": args.min_rare_val_patches,
        "skipped_corrupt_count": len(skipped_corrupt),
        "skipped_corrupt": skipped_corrupt,
        "train": [str(record["image"]) for record in train],
        "val": [str(record["image"]) for record in val],
        "summary": {
            "full": full_summary,
            "train": _summarize(train, args.num_classes),
            "val": _summarize(val, args.num_classes),
            "train_val": _summarize([*train, *val], args.num_classes),
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"stats": str(stats_path), "manifest": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

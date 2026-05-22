from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


def _sorted_pngs(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() == ".png")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a fixed BCSS Stage 4 split manifest.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--train-split", type=int, default=1000)
    parser.add_argument("--val-split", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="segmentator_runs/stage4_bcss_split_manifest.json")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    images_dir = root / "images" if (root / "images").exists() else root / "patches" / "images"
    masks_dir = root / "tissue_masks" if (root / "tissue_masks").exists() else root / "patches" / "tissue_masks"
    image_paths = _sorted_pngs(images_dir)
    if len(image_paths) < args.train_split + args.val_split:
        raise SystemExit(f"need {args.train_split + args.val_split} images, found {len(image_paths)}")

    rng = random.Random(args.seed)
    shuffled = image_paths[:]
    rng.shuffle(shuffled)
    selected = shuffled[: args.train_split + args.val_split]
    train = selected[: args.train_split]
    val = selected[args.train_split :]

    payload = {
        "dataset_root": str(root),
        "seed": args.seed,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "train": [p.name for p in train],
        "val": [p.name for p in val],
        "check": {
            "images": len(image_paths),
            "masks": len(_sorted_pngs(masks_dir)),
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

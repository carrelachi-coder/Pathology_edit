from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random


DATASETS = {
    "bcss": (("BCSS_PATCHES", "BCSS/BCSS_PATCHES"), "images", "tissue_masks"),
    "ignite": (("IGNITE_PATCHES",), "images", "tissue_masks"),
    "orca": (("ORCA_PATCHES", "ORCA/ORCA_PATCHES"), "images", "tissue_masks"),
    "panda": (("PANDA_PATCHES", "PANDA/PANDA_PATCHES"), "images", "tissue_masks"),
    "glas": (("GlaS_PATCHES", "GLAS/GlaS_PATCHES", "GLAS_PATCHES"), "images", "tissue_masks"),
    "puma": (("PUMA_PATCHES", "PUMA/PUMA_PATCHES"), "images", "tissue_masks"),
}


def _sorted_pngs(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() == ".png")


def _resolve_dataset_root(datasets_root: Path, dataset_id: str) -> Path:
    root_candidates, _, _ = DATASETS[dataset_id]
    for root_rel in root_candidates:
        dataset_root = datasets_root / root_rel
        if dataset_root.exists():
            return dataset_root
    searched = [str(datasets_root / root_rel) for root_rel in root_candidates]
    raise FileNotFoundError(f"could not find dataset '{dataset_id}'. Searched: {searched}")


def _records_for_dataset(datasets_root: Path, dataset_id: str, limit: int | None) -> list[dict[str, str]]:
    _, images_rel, masks_rel = DATASETS[dataset_id]
    dataset_root = _resolve_dataset_root(datasets_root, dataset_id)
    images_dir = dataset_root / images_rel
    masks_dir = dataset_root / masks_rel
    if not images_dir.exists():
        raise FileNotFoundError(images_dir)
    if not masks_dir.exists():
        raise FileNotFoundError(masks_dir)

    records = []
    for image_path in _sorted_pngs(images_dir):
        mask_path = masks_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        records.append(
            {
                "dataset_id": dataset_id,
                "dataset_root": str(dataset_root),
                "images_dir": images_rel,
                "masks_dir": masks_rel,
                "image": image_path.name,
                "mask": mask_path.name,
                "sample_id": f"{dataset_id}:{image_path.stem}",
            }
        )
    return records[:limit] if limit is not None else records


def _split_records(records: list[dict[str, str]], val_fraction: float, rng: random.Random) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    records = records[:]
    rng.shuffle(records)
    val_count = max(1, int(round(len(records) * val_fraction))) if records else 0
    return records[val_count:], records[:val_count]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a multi-dataset segmentator manifest.")
    parser.add_argument(
        "--datasets-root",
        default="/data/wqx/flowedit/data",
        help="Directory containing BCSS_PATCHES, GlaS_PATCHES, IGNITE_PATCHES, ORCA_PATCHES, PANDA_PATCHES, and PUMA_PATCHES.",
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-per-dataset", type=int, default=None)
    parser.add_argument("--output", default="segmentator_runs/stage4_multidataset_manifest.json")
    args = parser.parse_args()

    unknown = sorted(set(args.datasets) - set(DATASETS))
    if unknown:
        raise SystemExit(f"unknown datasets: {unknown}; available: {sorted(DATASETS)}")

    rng = random.Random(args.seed)
    train: list[dict[str, str]] = []
    val: list[dict[str, str]] = []
    summary: dict[str, dict[str, int]] = {}
    for dataset_id in args.datasets:
        records = _records_for_dataset(Path(args.datasets_root), dataset_id, args.max_per_dataset)
        ds_train, ds_val = _split_records(records, args.val_fraction, rng)
        train.extend(ds_train)
        val.extend(ds_val)
        summary[dataset_id] = {"total": len(records), "train": len(ds_train), "val": len(ds_val)}

    rng.shuffle(train)
    rng.shuffle(val)
    train_counts = defaultdict(int)
    val_counts = defaultdict(int)
    for record in train:
        train_counts[record["dataset_id"]] += 1
    for record in val:
        val_counts[record["dataset_id"]] += 1

    payload = {
        "dataset_root": str(Path(args.datasets_root)),
        "seed": args.seed,
        "strategy": "per-dataset split; use --balanced-datasets during training for equal dataset sampling",
        "datasets": args.datasets,
        "val_fraction": args.val_fraction,
        "max_per_dataset": args.max_per_dataset,
        "summary": summary,
        "train_counts": dict(sorted(train_counts.items())),
        "val_counts": dict(sorted(val_counts.items())),
        "train": train,
        "val": val,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(out), "train": len(train), "val": len(val), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re


DATASETS = {
    "bcss": (("BCSS_PATCHES", "BCSS/BCSS_PATCHES"), "images", "tissue_masks"),
    "ignite": (("IGNITE_PATCHES",), "images", "tissue_masks"),
    "orca": (("ORCA_PATCHES", "ORCA/ORCA_PATCHES"), "images", "tissue_masks"),
    "panda": (("PANDA_PATCHES", "PANDA/PANDA_PATCHES"), "images", "tissue_masks"),
    "glas": (("GlaS_PATCHES", "GLAS/GlaS_PATCHES", "GLAS_PATCHES"), "images", "tissue_masks"),
    "puma": (("PUMA_PATCHES", "PUMA/PUMA_PATCHES"), "images", "tissue_masks"),
}


def _group_id(dataset_id: str, filename: str) -> str:
    stem = Path(filename).stem
    if dataset_id == "bcss":
        return stem.split("_x", 1)[0]
    if dataset_id == "ignite":
        return stem.split("_he_", 1)[0]
    if dataset_id == "orca":
        return re.sub(r"_\d+$", "", stem.split("_py", 1)[0])
    if dataset_id == "panda":
        return stem.split("_y", 1)[0]
    return stem.split("_py", 1)[0]


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
                "nuclei_dir": "nuclei_masks",
                "image": image_path.name,
                "mask": mask_path.name,
                "nuclei": image_path.name,
                "sample_id": f"{dataset_id}:{image_path.stem}",
                "group_id": _group_id(dataset_id, image_path.name),
            }
        )
    return records[:limit] if limit is not None else records


def _split_records(
    records: list[dict[str, str]],
    val_fraction: float,
    test_fraction: float,
    rng: random.Random,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped[record["group_id"]].append(record)
    groups = list(grouped.items())
    rng.shuffle(groups)
    if len(groups) < 3 and test_fraction > 0:
        raise ValueError("group-disjoint train/val/test split requires at least three groups")

    total = len(records)
    test_target = int(round(total * test_fraction))
    val_target = int(round(total * val_fraction))
    test_groups: set[str] = set()
    val_groups: set[str] = set()
    test_count = 0
    val_count = 0
    for group_id, items in groups:
        test_deficit = test_target - test_count
        val_deficit = val_target - val_count
        if test_deficit > 0 and test_deficit >= val_deficit:
            test_groups.add(group_id)
            test_count += len(items)
        elif val_deficit > 0:
            val_groups.add(group_id)
            val_count += len(items)

    train = [record for record in records if record["group_id"] not in test_groups | val_groups]
    val = [record for record in records if record["group_id"] in val_groups]
    test = [record for record in records if record["group_id"] in test_groups]
    if not train or not val or (test_fraction > 0 and not test):
        raise ValueError("group split produced an empty partition")
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _assert_group_disjoint(*splits: list[dict[str, str]]) -> None:
    group_sets = [{record["group_id"] for record in split} for split in splits]
    for left in range(len(group_sets)):
        for right in range(left + 1, len(group_sets)):
            overlap = group_sets[left] & group_sets[right]
            if overlap:
                raise RuntimeError(f"group leakage across splits: {sorted(overlap)[:5]}")


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
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--max-per-dataset", type=int, default=None)
    parser.add_argument("--output", default="segmentator_runs/stage4_multidataset_manifest.json")
    args = parser.parse_args()
    if args.val_fraction <= 0 or args.test_fraction < 0 or args.val_fraction + args.test_fraction >= 1:
        raise SystemExit("val/test fractions must be non-negative and sum to less than one")

    unknown = sorted(set(args.datasets) - set(DATASETS))
    if unknown:
        raise SystemExit(f"unknown datasets: {unknown}; available: {sorted(DATASETS)}")

    rng = random.Random(args.seed)
    train: list[dict[str, str]] = []
    val: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
    summary: dict[str, dict[str, int]] = {}
    for dataset_id in args.datasets:
        records = _records_for_dataset(Path(args.datasets_root), dataset_id, args.max_per_dataset)
        ds_train, ds_val, ds_test = _split_records(records, args.val_fraction, args.test_fraction, rng)
        _assert_group_disjoint(ds_train, ds_val, ds_test)
        train.extend(ds_train)
        val.extend(ds_val)
        test.extend(ds_test)
        summary[dataset_id] = {
            "total": len(records),
            "train": len(ds_train),
            "val": len(ds_val),
            "test": len(ds_test),
            "train_groups": len({item["group_id"] for item in ds_train}),
            "val_groups": len({item["group_id"] for item in ds_val}),
            "test_groups": len({item["group_id"] for item in ds_test}),
        }

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    _assert_group_disjoint(train, val, test)
    train_counts = defaultdict(int)
    val_counts = defaultdict(int)
    test_counts = defaultdict(int)
    for record in train:
        train_counts[record["dataset_id"]] += 1
    for record in val:
        val_counts[record["dataset_id"]] += 1
    for record in test:
        test_counts[record["dataset_id"]] += 1

    payload = {
        "dataset_root": str(Path(args.datasets_root)),
        "seed": args.seed,
        "strategy": "per-dataset group-disjoint split by WSI/patient/ROI",
        "datasets": args.datasets,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "max_per_dataset": args.max_per_dataset,
        "summary": summary,
        "train_counts": dict(sorted(train_counts.items())),
        "val_counts": dict(sorted(val_counts.items())),
        "test_counts": dict(sorted(test_counts.items())),
        "train": train,
        "val": val,
        "test": test,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(out), "train": len(train), "val": len(val), "test": len(test), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

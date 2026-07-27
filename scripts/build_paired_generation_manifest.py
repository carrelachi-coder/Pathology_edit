#!/usr/bin/env python3
"""Build both A->B and B->A generation rows from the clinician package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-manifest", type=Path, required=True)
    parser.add_argument("--pair-review", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cellvit-instance-root", type=Path)
    parser.add_argument("--conic-root", type=Path)
    parser.add_argument("--require-accepted", action="store_true")
    parser.add_argument("--seed-start", type=int, default=42000)
    parser.add_argument("--patch-size", type=int, default=512)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def boxes_overlap(left: dict, right: dict, patch_size: int) -> bool:
    left_x, left_y = int(left["x"]), int(left["y"])
    right_x, right_y = int(right["x"]), int(right["y"])
    return abs(left_x - right_x) < patch_size and abs(left_y - right_y) < patch_size


def required_path(row: dict, field: str) -> str:
    value = row.get(field)
    if not value:
        raise ValueError(f"{row.get('annotation_id')}: missing {field}")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def main() -> int:
    args = parse_args()
    patches = read_csv(args.patch_manifest)
    review = {}
    if args.pair_review:
        review = {row["pair_id"]: row for row in read_csv(args.pair_review)}
    pairs: dict[str, dict[str, dict]] = {}
    for row in patches:
        pair = pairs.setdefault(row["pair_id"], {})
        side = row["side"].lower()
        if side in pair:
            raise ValueError(f"duplicate {row['pair_id']} side {side}")
        pair[side] = row

    records = []
    rejected = []
    for pair_id in sorted(pairs):
        sides = pairs[pair_id]
        if set(sides) != {"a", "b"}:
            raise ValueError(f"{pair_id}: expected sides a/b, got {sorted(sides)}")
        a, b = sides["a"], sides["b"]
        if a["organ"] != b["organ"] or a["wsi"] != b["wsi"]:
            raise ValueError(f"{pair_id}: organ/WSI mismatch")
        if boxes_overlap(a, b, args.patch_size):
            raise ValueError(f"{pair_id}: coordinate boxes overlap")
        pair_review = review.get(pair_id, {})
        pair_keep = str(pair_review.get("pair_keep", "pending")).strip().lower()
        if args.require_accepted and pair_keep not in {"1", "true", "yes", "keep", "accepted"}:
            rejected.append(pair_id)
            continue
        for direction, reference, target in (("a_to_b", a, b), ("b_to_a", b, a)):
            sample_id = f"{pair_id}-{direction}"
            prompt_path = Path(required_path(target, "caption_en_path"))
            record = {
                "sample_id": sample_id,
                "pair_id": pair_id,
                "direction": direction,
                "organ": target["organ"],
                "wsi_id": target["wsi"],
                "case_id": target.get("case_id"),
                "pair_score": float(target["pair_score"]),
                "pair_review_status": pair_keep or "pending",
                "reference_annotation_id": reference["annotation_id"],
                "target_annotation_id": target["annotation_id"],
                "reference_image": required_path(reference, "package_image_path"),
                "reference_tissue_mask": required_path(reference, "editable_tissue_mask_path"),
                "reference_nuclei_mask": required_path(reference, "package_cellvit_mask_path"),
                "target_image": required_path(target, "package_image_path"),
                "target_tissue_mask": required_path(target, "editable_tissue_mask_path"),
                "target_nuclei_mask": required_path(target, "package_cellvit_mask_path"),
                "target_tissue_review_status": target.get("tissue_mask_review_status") or "pending",
                "prompt": prompt_path.read_text(encoding="utf-8-sig").strip(),
                "prompt_path": str(prompt_path),
                "seed": args.seed_start + len(records),
                "source_resolution": 512,
                "source_mpp": 0.25,
                "source_fov_um": 128.0,
            }
            if args.cellvit_instance_root:
                record["target_cellvit_instances"] = str(
                    args.cellvit_instance_root / f"{target['annotation_id']}.json"
                )
            if args.conic_root:
                record["target_conic_instance_type_mask"] = str(
                    args.conic_root / target["annotation_id"] / "conic.npy"
                )
            records.append(record)

    target_images = [row["target_image"] for row in records]
    if len(target_images) != len(set(target_images)):
        raise ValueError("direction manifest must use each target RGB exactly once")
    counts = {}
    for row in records:
        counts[row["organ"]] = counts.get(row["organ"], 0) + 1
    payload = {
        "schema_version": 1,
        "records": records,
        "provenance": {
            "patch_manifest": str(args.patch_manifest.resolve()),
            "patch_manifest_sha256": sha256(args.patch_manifest),
            "pair_review": str(args.pair_review.resolve()) if args.pair_review else None,
            "pair_review_sha256": sha256(args.pair_review) if args.pair_review else None,
            "require_accepted": args.require_accepted,
            "rejected_pair_count": len(rejected),
            "rejected_pairs": rejected,
            "coordinate_overlap_policy": f"axis-aligned {args.patch_size}x{args.patch_size} boxes",
        },
        "summary": {
            "directions": len(records),
            "pairs": len(records) // 2,
            "unique_target_images": len(set(target_images)),
            "unique_wsis": len({row["wsi_id"] for row in records}),
            "organ_counts": counts,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

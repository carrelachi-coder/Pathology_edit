#!/usr/bin/env python3
"""Build the frozen nuclei/generation manifest for the paired U1/U2 cohort."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from PIL import Image

from phase3_mask_edit.benchmark.pathokid import sha256_file, stable_digest


PRIMITIVES = (
    ("tumor_burden_increase", "u1"),
    ("stromal_immune_infiltration", "u2"),
)
STRENGTHS = ("moderate", "significant")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-cohort-manifest", type=Path, required=True)
    parser.add_argument("--mask-run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-reference-count", type=int, default=300)
    parser.add_argument("--generation-seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def validate_mask_pair(
    *,
    reference_tissue: Path,
    target_tissue: Path,
    change_region: Path,
) -> tuple[int, float]:
    for path in (reference_tissue, target_tissue, change_region):
        if not path.is_file():
            raise FileNotFoundError(path)
    reference = load_mask(reference_tissue)
    target = load_mask(target_tissue)
    changed = load_mask(change_region) > 128
    if reference.shape != target.shape or reference.shape != changed.shape:
        raise ValueError(
            f"mask shape mismatch: {reference.shape}, {target.shape}, {changed.shape}"
        )
    actual = reference != target
    if not np.array_equal(actual, changed):
        raise ValueError(f"change region does not equal source/target difference: {target_tissue}")
    pixels = int(np.count_nonzero(changed))
    return pixels, float(pixels / changed.size)


def main() -> int:
    args = parse_args()
    references = read_jsonl(args.final_cohort_manifest)
    if len(references) != args.expected_reference_count:
        raise ValueError(
            f"expected {args.expected_reference_count} references, found {len(references)}"
        )
    reference_ids = [str(row.get("sample_id") or "") for row in references]
    if not all(reference_ids) or len(set(reference_ids)) != len(reference_ids):
        raise ValueError("reference sample IDs must be non-empty and unique")

    rows: list[dict] = []
    for reference in references:
        reference_id = str(reference["sample_id"])
        reference_tissue = Path(reference["reference_tissue_mask"])
        for primitive, short_name in PRIMITIVES:
            pair_id = f"{reference_id}__{short_name}"
            for strength in STRENGTHS:
                mask_dir = (
                    args.mask_run_root
                    / "cases"
                    / reference_id
                    / "final"
                    / primitive
                )
                target_tissue = mask_dir / f"{strength}_target_mask.png"
                change_region = mask_dir / f"{strength}_change_region.png"
                changed_pixels, changed_fraction = validate_mask_pair(
                    reference_tissue=reference_tissue,
                    target_tissue=target_tissue,
                    change_region=change_region,
                )
                sample_id = f"{pair_id}__{strength}"
                row = {
                    "schema_version": 1,
                    "status": "frozen_mask_target",
                    "sample_id": sample_id,
                    "reference_id": reference_id,
                    "ordinal_group_id": pair_id,
                    "pair_id": pair_id,
                    "moderate_sample_id": f"{pair_id}__moderate",
                    "wsi_id": reference["wsi_id"],
                    "patient_id": reference.get(
                        "patient_id", reference["wsi_id"]
                    ),
                    "profile": "BCSS",
                    "primitive": primitive,
                    "strength": strength,
                    "reference_image": reference["reference_image"],
                    "reference_tissue_mask": str(reference_tissue),
                    "reference_nuclei_mask": reference[
                        "reference_nuclei_mask"
                    ],
                    "target_tissue_mask": str(target_tissue),
                    "change_region": str(change_region),
                    "changed_pixels": changed_pixels,
                    "changed_area_fraction": changed_fraction,
                    "generation_seed": args.generation_seed,
                    "nuclei_strength_policy": (
                        "independent_direct_from_original_reference"
                    ),
                    "mask_selection_key": reference["selection_key"],
                    "sha256": {
                        "reference_image": sha256_file(
                            Path(reference["reference_image"])
                        ),
                        "reference_tissue_mask": sha256_file(reference_tissue),
                        "reference_nuclei_mask": sha256_file(
                            Path(reference["reference_nuclei_mask"])
                        ),
                        "target_tissue_mask": sha256_file(target_tissue),
                        "change_region": sha256_file(change_region),
                    },
                }
                rows.append(row)

    expected_rows = args.expected_reference_count * len(PRIMITIVES) * len(STRENGTHS)
    if len(rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} target rows, built {len(rows)}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError("target sample IDs are not unique")

    args.output_root.mkdir(parents=True, exist_ok=True)
    combined_path = args.output_root / "combined_source_manifest.jsonl"
    write_jsonl(combined_path, rows)
    smoke_path = args.output_root / "smoke_source_manifest.jsonl"
    write_jsonl(smoke_path, rows[: len(PRIMITIVES) * len(STRENGTHS)])
    group_paths: dict[str, str] = {}
    for primitive, short_name in PRIMITIVES:
        for strength in STRENGTHS:
            group_rows = [
                row
                for row in rows
                if row["primitive"] == primitive and row["strength"] == strength
            ]
            path = args.output_root / "source_manifests" / f"{short_name}_{strength}.jsonl"
            write_jsonl(path, group_rows)
            group_paths[f"{primitive}/{strength}"] = str(path)

    counts = Counter((row["primitive"], row["strength"]) for row in rows)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_count": len(references),
        "wsi_count": len({row["wsi_id"] for row in references}),
        "target_count": len(rows),
        "group_counts": {
            f"{primitive}/{strength}": counts[(primitive, strength)]
            for primitive, _ in PRIMITIVES
            for strength in STRENGTHS
        },
        "generation_seed": args.generation_seed,
        "nuclei_strength_policy": "independent_direct_from_original_reference",
        "final_cohort_manifest": str(args.final_cohort_manifest),
        "mask_run_root": str(args.mask_run_root),
        "combined_source_manifest": str(combined_path),
        "smoke_source_manifest": str(smoke_path),
        "group_source_manifests": group_paths,
        "combined_manifest_digest": stable_digest(
            [
                row["sample_id"]
                + ":"
                + row["sha256"]["target_tissue_mask"]
                + ":"
                + row["sha256"]["change_region"]
                for row in rows
            ]
        ),
    }
    (args.output_root / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

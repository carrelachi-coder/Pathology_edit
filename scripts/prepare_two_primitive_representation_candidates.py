#!/usr/bin/env python3
"""Freeze an immune-rich BCSS candidate pool for two-primitive analysis."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from phase3_mask_edit.benchmark.pathokid import sha256_file, stable_digest
from phase3_mask_edit.core.labels import MaskProfileSchema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=900)
    parser.add_argument("--minimum-immune-fraction", type=float, default=0.10)
    parser.add_argument("--minimum-tumor-fraction", type=float, default=0.10)
    parser.add_argument("--minimum-stroma-fraction", type=float, default=0.30)
    parser.add_argument("--maximum-background-fraction", type=float, default=0.0)
    parser.add_argument("--maximum-other-fraction", type=float, default=0.0)
    parser.add_argument("--neighborhood-radius-px", type=float, default=96.0)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def wsi_id(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def evaluate_mask(
    path: Path,
    *,
    dataset_root: Path,
    schema: MaskProfileSchema,
    args: argparse.Namespace,
) -> dict:
    mask = load_mask(path)
    tumor = np.isin(mask, schema.tumor_fine_ids)
    stroma = np.isin(mask, schema.resolve_fine_ids("Stroma"))
    immune = np.isin(mask, schema.resolve_fine_ids("Immune infiltrate"))
    background = np.isin(mask, tuple(schema.skip_fine_ids))
    other = np.isin(mask, schema.resolve_fine_ids("Other tissue"))
    total = int(mask.size)
    fractions = {
        "tumor": float(np.count_nonzero(tumor) / total),
        "stroma": float(np.count_nonzero(stroma) / total),
        "immune": float(np.count_nonzero(immune) / total),
        "background": float(np.count_nonzero(background) / total),
        "other": float(np.count_nonzero(other) / total),
    }
    reasons = []
    if fractions["immune"] < args.minimum_immune_fraction:
        reasons.append("immune_fraction_below_minimum")
    if fractions["tumor"] < args.minimum_tumor_fraction:
        reasons.append("tumor_fraction_below_minimum")
    if fractions["stroma"] < args.minimum_stroma_fraction:
        reasons.append("stroma_fraction_below_minimum")
    if fractions["background"] > args.maximum_background_fraction:
        reasons.append("background_fraction_above_maximum")
    if fractions["other"] > args.maximum_other_fraction:
        reasons.append("other_fraction_above_maximum")

    image_path = dataset_root / "images" / path.name
    nuclei_path = dataset_root / "nuclei_masks" / path.name
    if not image_path.is_file():
        reasons.append("reference_image_missing")
    if not nuclei_path.is_file():
        reasons.append("reference_nuclei_mask_missing")

    peritumoral_stroma_pixels = None
    immune_supported_stroma_pixels = None
    if not reasons:
        distance_to_tumor = ndimage.distance_transform_edt(~tumor)
        peritumoral_stroma_pixels = int(
            np.count_nonzero(
                stroma
                & (distance_to_tumor <= float(args.neighborhood_radius_px))
            )
        )
        distance_to_immune = ndimage.distance_transform_edt(~immune)
        immune_supported_stroma_pixels = int(
            np.count_nonzero(
                stroma
                & (distance_to_immune <= float(args.neighborhood_radius_px))
            )
        )
    row = {
        "schema_version": 1,
        "status": "eligible" if not reasons else "rejected",
        "reasons": reasons,
        "sample_id": (
            f"BCSS_two_primitive_{stable_digest([path.name, args.seed])[:12]}"
        ),
        "wsi_id": wsi_id(path),
        "patient_id": wsi_id(path),
        "profile": "BCSS",
        "reference_image": str(image_path),
        "reference_tissue_mask": str(path),
        "reference_nuclei_mask": str(nuclei_path),
        "source_fractions": fractions,
        "source_pixels": {
            "tumor": int(np.count_nonzero(tumor)),
            "stroma": int(np.count_nonzero(stroma)),
            "immune": int(np.count_nonzero(immune)),
            "background": int(np.count_nonzero(background)),
            "other": int(np.count_nonzero(other)),
            "peritumoral_stroma_within_radius": peritumoral_stroma_pixels,
            "immune_supported_stroma_within_radius": (
                immune_supported_stroma_pixels
            ),
        },
        "neighborhood_radius_px": float(args.neighborhood_radius_px),
        "selection_key": stable_digest(
            [path.name, args.seed, "candidate_order"]
        ),
    }
    if not reasons:
        row["sha256"] = {
            "reference_image": sha256_file(image_path),
            "reference_tissue_mask": sha256_file(path),
            "reference_nuclei_mask": sha256_file(nuclei_path),
        }
    return row


def round_robin(rows: list[dict], count: int) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["wsi_id"]), []).append(row)
    for items in grouped.values():
        items.sort(key=lambda item: str(item["selection_key"]))
    names = sorted(
        grouped,
        key=lambda name: stable_digest([name, "wsi_order"]),
    )
    selected = []
    depth = 0
    while len(selected) < count:
        added = 0
        for name in names:
            items = grouped[name]
            if depth >= len(items):
                continue
            selected.append(items[depth])
            added += 1
            if len(selected) == count:
                break
        if added == 0:
            raise RuntimeError("round-robin selection exhausted")
        depth += 1
    return sorted(selected, key=lambda row: str(row["sample_id"]))


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def main() -> int:
    args = parse_args()
    tissue_root = args.dataset_root / "tissue_masks"
    paths = sorted(tissue_root.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"no tissue masks under {tissue_root}")
    schema = MaskProfileSchema.from_reference_profile("BCSS")
    audit = [
        evaluate_mask(
            path,
            dataset_root=args.dataset_root,
            schema=schema,
            args=args,
        )
        for path in paths
    ]
    eligible = [row for row in audit if row["status"] == "eligible"]
    if len(eligible) < args.candidate_count:
        raise RuntimeError(
            f"only {len(eligible)} eligible rows for {args.candidate_count} candidates"
        )
    selected = round_robin(eligible, args.candidate_count)
    for row in selected:
        row["status"] = "selected_candidate"

    args.output_root.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_root / "static_prefilter_audit.jsonl"
    manifest_path = args.output_root / "candidate_pool_manifest.jsonl"
    write_jsonl(audit_path, audit)
    write_jsonl(manifest_path, selected)

    reason_counts = Counter(
        reason
        for row in audit
        if row["status"] == "rejected"
        for reason in row["reasons"]
    )
    wsi_counts = Counter(str(row["wsi_id"]) for row in selected)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "analysis": "immune_rich_two_primitive_static_candidate_pool",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(args.dataset_root),
        "source_mask_count": len(paths),
        "eligible_count": len(eligible),
        "eligible_wsi_count": len({row["wsi_id"] for row in eligible}),
        "selected_candidate_count": len(selected),
        "selected_wsi_count": len(wsi_counts),
        "selected_max_patches_per_wsi": max(wsi_counts.values()),
        "selected_min_patches_per_wsi": min(wsi_counts.values()),
        "thresholds": {
            "minimum_immune_fraction": args.minimum_immune_fraction,
            "minimum_tumor_fraction": args.minimum_tumor_fraction,
            "minimum_stroma_fraction": args.minimum_stroma_fraction,
            "maximum_background_fraction": args.maximum_background_fraction,
            "maximum_other_fraction": args.maximum_other_fraction,
        },
        "selection": {
            "policy": "deterministic_wsi_round_robin_after_mask_only_thresholds",
            "seed": args.seed,
            "no_rgb_embedding_or_generation_selection": True,
        },
        "selected_fraction_distributions": {
            label: distribution(
                [float(row["source_fractions"][label]) for row in selected]
            )
            for label in ("immune", "tumor", "stroma", "background", "other")
        },
        "rejected_reason_counts": dict(sorted(reason_counts.items())),
        "candidate_pool_manifest": str(manifest_path),
        "static_prefilter_audit": str(audit_path),
        "candidate_pool_digest": stable_digest(selected),
    }
    summary_path = args.output_root / "candidate_pool_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

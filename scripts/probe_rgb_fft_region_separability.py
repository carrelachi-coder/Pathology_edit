#!/usr/bin/env python3
"""Probe whether the RGB+FFT region descriptor separates same-label instances."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controlnet_train.data.common import load_image_tensor, load_tissue_mask, resolve_path
from controlnet_train.training.cross_v1_losses import (
    RegionalRgbFftLossConfig,
    _region_rgb_fft_descriptor,
    _resize_mask_to_image,
)
from dataset_config import COARSE_LABELS, FINE_LABELS, FINE_TO_PARENT


@dataclass(frozen=True)
class SampleEntry:
    index: int
    dataset: str
    sample_id: str
    image_path: Path
    tissue_mask_path: Path


@dataclass(frozen=True)
class DescriptorItem:
    label_name: str
    label_id: int
    sample: SampleEntry
    region_pixels: int
    region_fraction: float
    mean: torch.Tensor
    std: torch.Tensor
    fft: torch.Tensor

    @property
    def concat(self) -> torch.Tensor:
        return torch.cat([self.mean, self.std, self.fft], dim=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure separability of the Cross V1 RGB+FFT region descriptor."
    )
    parser.add_argument("--metadata", required=True, help="Metadata json/jsonl with image paths.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--metadata-base-dir",
        default=None,
        help="Base directory for relative metadata paths. Defaults to metadata parent.",
    )
    parser.add_argument(
        "--image-field",
        default="auto",
        help=(
            "Metadata image field. Use 'auto' to accept either flat 'image' rows "
            "or cross-meta pair rows with target_image/reference_image."
        ),
    )
    parser.add_argument(
        "--mask-field",
        default=None,
        help="Optional metadata field for the tissue mask. If absent, derive from --mask-dir.",
    )
    parser.add_argument("--mask-dir", default="tissue_masks")
    parser.add_argument(
        "--sample-id-field",
        default="auto",
        help="Metadata sample id field. In auto mode, pair metadata uses sample_id/reference_sample_id.",
    )
    parser.add_argument(
        "--label-mode",
        choices=("coarse_tissue", "coarse", "fine", "tissue"),
        default="coarse_tissue",
        help="Remap fine labels to coarse tissue IDs when requested.",
    )
    parser.add_argument("--label-a", default="tumor")
    parser.add_argument("--label-b", default="stroma")
    parser.add_argument("--samples-per-label", type=int, default=64)
    parser.add_argument("--candidate-pool-size", type=int, default=5000)
    parser.add_argument("--min-region-pixels", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--allow-same-image-cross-pairs",
        action="store_true",
        help="Include cross-label pairs drawn from the same source image.",
    )
    parser.add_argument("--reference-region-mean-weight", type=float, default=1.0)
    parser.add_argument("--reference-region-std-weight", type=float, default=0.5)
    parser.add_argument("--reference-region-fft-weight", type=float, default=0.25)
    parser.add_argument("--reference-region-fft-bins", type=int, default=6)
    parser.add_argument("--reference-region-fft-size", type=int, default=64)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.samples_per_label <= 0:
        raise ValueError("--samples-per-label must be positive")
    if args.candidate_pool_size <= 0:
        raise ValueError("--candidate-pool-size must be positive")
    if args.min_region_pixels <= 0:
        raise ValueError("--min-region-pixels must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.reference_region_fft_bins <= 0:
        raise ValueError("--reference-region-fft-bins must be positive")
    if args.reference_region_fft_size <= 0:
        raise ValueError("--reference-region-fft-size must be positive")
    if not 0.0 <= args.reference_region_mean_weight:
        raise ValueError("--reference-region-mean-weight must be non-negative")
    if not 0.0 <= args.reference_region_std_weight:
        raise ValueError("--reference-region-std-weight must be non-negative")
    if not 0.0 <= args.reference_region_fft_weight:
        raise ValueError("--reference-region-fft-weight must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    metadata_path = Path(args.metadata)
    base_dir = Path(args.metadata_base_dir) if args.metadata_base_dir else metadata_path.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_metadata(metadata_path)
    label_mode = normalize_label_mode(args.label_mode)
    label_names = build_label_name_lookup(label_mode, COARSE_LABELS, FINE_LABELS)
    label_a = parse_label(args.label_a, label_names)
    label_b = parse_label(args.label_b, label_names)
    if label_a == label_b:
        raise ValueError("--label-a and --label-b must resolve to different labels")

    entries = build_entries(
        records,
        base_dir=base_dir,
        image_field=args.image_field,
        mask_field=args.mask_field,
        mask_dir=args.mask_dir,
        sample_id_field=args.sample_id_field,
    )
    rng = random.Random(args.seed)
    rng.shuffle(entries)
    entries = entries[: int(args.candidate_pool_size)]

    device = resolve_device(args.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.torch_dtype]
    descriptor_config = RegionalRgbFftLossConfig(
        mean_weight=float(args.reference_region_mean_weight),
        std_weight=float(args.reference_region_std_weight),
        fft_weight=float(args.reference_region_fft_weight),
        fft_bins=int(args.reference_region_fft_bins),
        fft_size=int(args.reference_region_fft_size),
    )

    items_by_label, skipped, scanned_entries = collect_descriptor_items(
        entries,
        label_a=label_a,
        label_b=label_b,
        label_names=label_names,
        label_mode=label_mode,
        descriptor_config=descriptor_config,
        samples_per_label=int(args.samples_per_label),
        min_region_pixels=int(args.min_region_pixels),
        batch_size=int(args.batch_size),
        device=device,
        dtype=dtype,
        progress_every=int(args.progress_every),
    )

    if len(items_by_label[label_a]) < 2 or len(items_by_label[label_b]) < 2:
        summary = {
            "status": "insufficient_valid_descriptors",
            "metadata": str(metadata_path),
            "label_mode": label_mode,
            "label_a": label_a,
            "label_b": label_b,
            "counts": {str(label): len(items_by_label[label]) for label in (label_a, label_b)},
            "candidate_entries": len(entries),
            "scanned_entries": scanned_entries,
            "skipped_preview": skipped[:100],
        }
        write_json(output_dir / "rgb_fft_region_descriptor_separability_summary.json", summary)
        raise RuntimeError("Not enough descriptors; see summary json")

    descriptor_rows = descriptor_table(items_by_label[label_a] + items_by_label[label_b])
    write_csv(output_dir / "rgb_fft_region_descriptor_items.csv", descriptor_rows)

    pair_rows, summary = build_pair_outputs(
        items_a=items_by_label[label_a],
        items_b=items_by_label[label_b],
        allow_same_image_cross_pairs=bool(args.allow_same_image_cross_pairs),
        mean_weight=float(args.reference_region_mean_weight),
        std_weight=float(args.reference_region_std_weight),
        fft_weight=float(args.reference_region_fft_weight),
    )
    write_csv(output_dir / "rgb_fft_region_descriptor_pairs.csv", pair_rows)

    summary.update(
        {
            "metadata": str(metadata_path),
            "image_field": args.image_field,
            "mask_field": args.mask_field,
            "mask_dir": args.mask_dir,
            "label_mode": label_mode,
            "label_a": {
                "id": int(label_a),
                "name": canonical_label_name(label_a, label_names, fallback=str(label_a)),
            },
            "label_b": {
                "id": int(label_b),
                "name": canonical_label_name(label_b, label_names, fallback=str(label_b)),
            },
            "samples_per_label_requested": int(args.samples_per_label),
            "candidate_entries": len(entries),
            "scanned_entries": scanned_entries,
            "skipped_count": len(skipped),
            "skipped_preview": skipped[:100],
            "outputs": {
                "items_csv": "rgb_fft_region_descriptor_items.csv",
                "pairs_csv": "rgb_fft_region_descriptor_pairs.csv",
                "summary_json": "rgb_fft_region_descriptor_separability_summary.json",
            },
        }
    )
    write_json(output_dir / "rgb_fft_region_descriptor_separability_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))
    return 0


def read_metadata(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    payload = json.loads(path.read_text(encoding="utf8"))
    if isinstance(payload, dict):
        rows = payload.get("pairs") or payload.get("records")
        if isinstance(rows, list):
            return rows
        raise ValueError("metadata dict must contain a 'pairs' or 'records' list")
    if isinstance(payload, list):
        return payload
    raise TypeError(f"unsupported metadata payload type: {type(payload)!r}")


def build_entries(
    records: list[dict[str, Any]],
    *,
    base_dir: Path,
    image_field: str,
    mask_field: str | None,
    mask_dir: str,
    sample_id_field: str,
) -> list[SampleEntry]:
    entries: list[SampleEntry] = []
    seen: set[tuple[str, str]] = set()

    def add_entry(
        *,
        record: dict[str, Any],
        record_index: int,
        image_value: Any,
        mask_value: Any | None,
        sample_id_value: Any | None,
    ) -> None:
        if not image_value:
            return
        image_path = resolve_path(image_value, base_dir)
        sample_id = str(sample_id_value or image_path.stem)
        if mask_value:
            tissue_mask_path = resolve_path(mask_value, base_dir)
        else:
            tissue_mask_path = base_dir / mask_dir / f"{Path(sample_id).stem}.png"
        key = (str(image_path), str(tissue_mask_path))
        if key in seen:
            return
        seen.add(key)
        entries.append(
            SampleEntry(
                index=record_index,
                dataset=str(record.get("dataset") or "unknown"),
                sample_id=sample_id,
                image_path=image_path,
                tissue_mask_path=tissue_mask_path,
            )
        )

    for index, record in enumerate(records):
        if image_field == "auto":
            flat_image = record.get("image")
            if flat_image:
                flat_sample_field = None if sample_id_field == "auto" else sample_id_field
                add_entry(
                    record=record,
                    record_index=index,
                    image_value=flat_image,
                    mask_value=record.get(mask_field) if mask_field else None,
                    sample_id_value=record.get(flat_sample_field or "sample_id"),
                )
                continue

            add_entry(
                record=record,
                record_index=index,
                image_value=record.get("target_image"),
                mask_value=record.get("target_tissue_mask"),
                sample_id_value=record.get("sample_id"),
            )
            add_entry(
                record=record,
                record_index=index,
                image_value=record.get("reference_image"),
                mask_value=record.get("reference_tissue_mask"),
                sample_id_value=record.get("reference_sample_id"),
            )
            continue

        explicit_sample_field = None if sample_id_field == "auto" else sample_id_field
        add_entry(
            record=record,
            record_index=index,
            image_value=record.get(image_field),
            mask_value=record.get(mask_field) if mask_field else None,
            sample_id_value=record.get(explicit_sample_field or "sample_id"),
        )
    return entries


def collect_descriptor_items(
    entries: list[SampleEntry],
    *,
    label_a: int,
    label_b: int,
    label_names: dict[str, int],
    label_mode: str,
    descriptor_config: RegionalRgbFftLossConfig,
    samples_per_label: int,
    min_region_pixels: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    progress_every: int,
) -> tuple[dict[int, list[DescriptorItem]], list[dict[str, Any]], int]:
    remap_lookup = build_coarse_lookup(FINE_TO_PARENT, device=torch.device("cpu"))
    label_name_a = canonical_label_name(label_a, label_names, fallback=str(label_a))
    label_name_b = canonical_label_name(label_b, label_names, fallback=str(label_b))
    items_by_label: dict[int, list[DescriptorItem]] = {label_a: [], label_b: []}
    skipped: list[dict[str, Any]] = []
    pending_entries: list[SampleEntry] = []
    pending_images: list[torch.Tensor] = []
    pending_masks: list[torch.Tensor] = []
    seen_for_label: dict[int, set[str]] = {label_a: set(), label_b: set()}
    scanned = 0

    def enough() -> bool:
        return all(len(items_by_label[label]) >= int(samples_per_label) for label in (label_a, label_b))

    def flush() -> None:
        nonlocal scanned
        if not pending_entries:
            return
        images = torch.stack(pending_images).to(device=device, dtype=dtype)
        masks = torch.stack(pending_masks).to(device=device)
        for batch_index, entry in enumerate(pending_entries):
            image = images[batch_index]
            mask = masks[batch_index]
            if tuple(int(v) for v in mask.shape[-2:]) != tuple(int(v) for v in image.shape[-2:]):
                mask = _resize_mask_to_image(mask.unsqueeze(0), tuple(int(v) for v in image.shape[-2:]))[0]
            if label_mode == "coarse_tissue":
                mask = remap_fine_to_coarse(mask, remap_lookup)
            for label_id, label_name in ((label_a, label_name_a), (label_b, label_name_b)):
                if len(items_by_label[label_id]) >= int(samples_per_label):
                    continue
                if str(entry.image_path) in seen_for_label[label_id]:
                    continue
                region = mask == int(label_id)
                region_pixels = int(region.sum().item())
                if region_pixels < min_region_pixels:
                    continue
                desc = _region_rgb_fft_descriptor(image, region, config=descriptor_config)
                items_by_label[label_id].append(
                    DescriptorItem(
                        label_name=label_name,
                        label_id=int(label_id),
                        sample=entry,
                        region_pixels=region_pixels,
                        region_fraction=float(region_pixels / max(1, int(region.numel()))),
                        mean=desc["mean"].detach().float().cpu(),
                        std=desc["std"].detach().float().cpu(),
                        fft=desc["fft"].detach().float().cpu(),
                    )
                )
                seen_for_label[label_id].add(str(entry.image_path))
        scanned += len(pending_entries)
        if progress_every > 0 and scanned % int(progress_every) < len(pending_entries):
            print(
                f"[rgb-fft-separability] scanned={scanned}/{len(entries)} "
                f"{label_name_a}={len(items_by_label[label_a])} "
                f"{label_name_b}={len(items_by_label[label_b])}",
                flush=True,
            )
        pending_entries.clear()
        pending_images.clear()
        pending_masks.clear()

    for entry in entries:
        if enough():
            break
        try:
            image = load_image_tensor(entry.image_path)
            mask = load_tissue_mask(entry.tissue_mask_path)
        except Exception as exc:  # noqa: BLE001 - diagnostic script should keep going.
            skipped.append(
                {
                    "index": entry.index,
                    "sample_id": entry.sample_id,
                    "image_path": str(entry.image_path),
                    "mask_path": str(entry.tissue_mask_path),
                    "reason": f"load_failed:{type(exc).__name__}",
                    "detail": str(exc),
                }
            )
            continue
        pending_entries.append(entry)
        pending_images.append(image)
        pending_masks.append(mask)
        if len(pending_entries) >= batch_size:
            flush()
    flush()
    return items_by_label, skipped, scanned


def build_pair_outputs(
    *,
    items_a: list[DescriptorItem],
    items_b: list[DescriptorItem],
    allow_same_image_cross_pairs: bool,
    mean_weight: float,
    std_weight: float,
    fft_weight: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pair_rows: list[dict[str, Any]] = []
    # Per-dimension standardization scale computed over ALL items (a+b together).
    # Without this, mean/std/fft have wildly different absolute scales and the
    # largest-magnitude component dominates total_distance, masking the true
    # separability of the other components.
    norm_scale = compute_norm_scale(items_a + items_b)
    within_a = pairwise_rows(
        items_a,
        "within_a",
        mean_weight=mean_weight,
        std_weight=std_weight,
        fft_weight=fft_weight,
        norm_scale=norm_scale,
    )
    within_b = pairwise_rows(
        items_b,
        "within_b",
        mean_weight=mean_weight,
        std_weight=std_weight,
        fft_weight=fft_weight,
        norm_scale=norm_scale,
    )
    cross = cross_pair_rows(
        items_a,
        items_b,
        allow_same_image_cross_pairs=allow_same_image_cross_pairs,
        mean_weight=mean_weight,
        std_weight=std_weight,
        fft_weight=fft_weight,
        norm_scale=norm_scale,
    )
    pair_rows.extend(within_a)
    pair_rows.extend(within_b)
    pair_rows.extend(cross)

    summary = {
        "counts": {
            "label_a": len(items_a),
            "label_b": len(items_b),
            "within_a_pairs": len(within_a),
            "within_b_pairs": len(within_b),
            "cross_pairs": len(cross),
        },
        "distance_stats": {},
        "variance": {
            "label_a": variance_summary(items_a),
            "label_b": variance_summary(items_b),
        },
    }
    for metric in (
        "mean_distance",
        "std_distance",
        "fft_distance",
        "total_distance",
        "norm_mean_distance",
        "norm_std_distance",
        "norm_fft_distance",
        "norm_total_distance",
        "concat_cosine_distance",
    ):
        stats_a = describe_values(torch.tensor([row[metric] for row in within_a], dtype=torch.float32))
        stats_b = describe_values(torch.tensor([row[metric] for row in within_b], dtype=torch.float32))
        same_values = [row[metric] for row in within_a + within_b]
        cross_values = [row[metric] for row in cross]
        stats_same = describe_values(torch.tensor(same_values, dtype=torch.float32))
        stats_cross = describe_values(torch.tensor(cross_values, dtype=torch.float32))
        margin = None
        if stats_same.get("count", 0) and stats_cross.get("count", 0):
            margin = float(stats_cross["mean"] - stats_same["mean"])
        summary["distance_stats"][metric] = {
            "within_a": stats_a,
            "within_b": stats_b,
            "within_all": stats_same,
            "cross": stats_cross,
            "cross_minus_within_mean": margin,
            "cross_greater_than_within_probability": greater_than_probability(cross_values, same_values),
        }

    # ---- Instance-separability verdict (the actual question this probe answers) ----
    # Uses the SCALE-NORMALIZED total distance, the only number that is not
    # dominated by one component's raw magnitude. Interprets two things:
    #   1. cross / within ratio  -> can the descriptor separate TYPES at all?
    #   2. within absolute spread -> are same-type INSTANCES distinguishable, or
    #      collapsed (region loss satisfiable by a type-average fill)?
    norm = summary["distance_stats"].get("norm_total_distance", {})
    within_stats = norm.get("within_all", {})
    cross_stats = norm.get("cross", {})
    verdict: dict[str, Any] = {"metric": "norm_total_distance"}
    w_mean = within_stats.get("mean")
    c_mean = cross_stats.get("mean")
    w_median = within_stats.get("median")
    if w_mean and c_mean and w_mean > 0:
        ratio = float(c_mean / w_mean)
        sep_prob = norm.get("cross_greater_than_within_probability")
        verdict.update(
            {
                "within_instance_distance_mean": w_mean,
                "within_instance_distance_median": w_median,
                "cross_type_distance_mean": c_mean,
                "cross_over_within_ratio": ratio,
                "cross_greater_than_within_probability": sep_prob,
            }
        )
        # Heuristic reading. ratio≈1 and high overlap => descriptor cannot tell
        # same-type instances apart any better than it tells types apart, i.e.
        # it mostly encodes TYPE; region loss then bottoms out at type-average.
        if sep_prob is not None and sep_prob >= 0.80 and ratio >= 1.5:
            verdict["reading"] = (
                "types_clearly_separable; within-instance spread is substantial "
                "relative to nothing-collapsed — descriptor DOES carry instance-level "
                "variation. region-loss plateau is likely NOT a descriptor limit; "
                "look downstream (injection/projection)."
            )
        elif ratio < 1.2:
            verdict["reading"] = (
                "cross≈within: descriptor barely separates types better than "
                "same-type instances. It mostly encodes a coarse signal; region "
                "loss can be satisfied by a type-average fill. DESCRIPTOR IS A "
                "BOTTLENECK — change the descriptor (finer stats / instance-aware "
                "features)."
            )
        else:
            verdict["reading"] = (
                "intermediate: types separable but instance spread modest. "
                "Inspect within distribution directly; descriptor may only weakly "
                "encode instance identity."
            )
    else:
        verdict["reading"] = "insufficient_data"
    summary["instance_separability_verdict"] = verdict
    return pair_rows, summary


def pairwise_rows(
    items: list[DescriptorItem],
    group: str,
    *,
    mean_weight: float,
    std_weight: float,
    fft_weight: float,
    norm_scale: dict[str, torch.Tensor] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            rows.append(
                format_pair(
                    items[i],
                    items[j],
                    pair_group=group,
                    mean_weight=mean_weight,
                    std_weight=std_weight,
                    fft_weight=fft_weight,
                    norm_scale=norm_scale,
                )
            )
    return rows


def cross_pair_rows(
    items_a: list[DescriptorItem],
    items_b: list[DescriptorItem],
    *,
    allow_same_image_cross_pairs: bool,
    mean_weight: float,
    std_weight: float,
    fft_weight: float,
    norm_scale: dict[str, torch.Tensor] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for item_a in items_a:
        for item_b in items_b:
            if not allow_same_image_cross_pairs and item_a.sample.image_path == item_b.sample.image_path:
                continue
            rows.append(
                format_pair(
                    item_a,
                    item_b,
                    pair_group="cross",
                    mean_weight=mean_weight,
                    std_weight=std_weight,
                    fft_weight=fft_weight,
                    norm_scale=norm_scale,
                )
            )
    return rows


def format_pair(
    item_i: DescriptorItem,
    item_j: DescriptorItem,
    *,
    pair_group: str,
    mean_weight: float,
    std_weight: float,
    fft_weight: float,
    norm_scale: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    distances = descriptor_distances(
        item_i,
        item_j,
        mean_weight=mean_weight,
        std_weight=std_weight,
        fft_weight=fft_weight,
        norm_scale=norm_scale,
    )
    return {
        "pair_group": pair_group,
        "label_i": item_i.label_name,
        "label_j": item_j.label_name,
        "sample_i": item_i.sample.sample_id,
        "sample_j": item_j.sample.sample_id,
        "image_i": str(item_i.sample.image_path),
        "image_j": str(item_j.sample.image_path),
        "same_image": item_i.sample.image_path == item_j.sample.image_path,
        "mean_distance": distances["mean_distance"],
        "std_distance": distances["std_distance"],
        "fft_distance": distances["fft_distance"],
        "total_distance": distances["total_distance"],
        "norm_mean_distance": distances["norm_mean_distance"],
        "norm_std_distance": distances["norm_std_distance"],
        "norm_fft_distance": distances["norm_fft_distance"],
        "norm_total_distance": distances["norm_total_distance"],
        "concat_cosine_distance": distances["concat_cosine_distance"],
        "pixels_i": item_i.region_pixels,
        "pixels_j": item_j.region_pixels,
    }


def compute_norm_scale(items: list[DescriptorItem]) -> dict[str, torch.Tensor]:
    """Per-dimension std over all items, used to z-score each descriptor component.

    Returns a dict with 'mean','std','fft' scale tensors. A floor of 1e-6 avoids
    division by zero on constant dimensions.
    """
    if not items:
        return {}
    mean_stack = torch.stack([item.mean for item in items]).float()
    std_stack = torch.stack([item.std for item in items]).float()
    fft_stack = torch.stack([item.fft for item in items]).float()
    return {
        "mean": mean_stack.std(dim=0, unbiased=False).clamp_min(1e-6),
        "std": std_stack.std(dim=0, unbiased=False).clamp_min(1e-6),
        "fft": fft_stack.std(dim=0, unbiased=False).clamp_min(1e-6),
    }


def descriptor_distances(
    item_i: DescriptorItem,
    item_j: DescriptorItem,
    *,
    mean_weight: float,
    std_weight: float,
    fft_weight: float,
    norm_scale: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    mean_distance = float(F.l1_loss(item_i.mean, item_j.mean).item())
    std_distance = float(F.l1_loss(item_i.std, item_j.std).item())
    fft_distance = float(F.l1_loss(item_i.fft, item_j.fft).item())
    total_weight = float(mean_weight) + float(std_weight) + float(fft_weight)
    weighted = float(mean_weight) * mean_distance
    weighted += float(std_weight) * std_distance
    weighted += float(fft_weight) * fft_distance
    total_distance = weighted / total_weight if total_weight > 0.0 else weighted

    # Scale-normalized distances: z-score each component by its per-dimension
    # cross-sample std before measuring L1. This is the trustworthy number for
    # comparing within vs cross, because it stops the largest-magnitude component
    # (often fft) from dominating and hiding whether mean/std separate instances.
    norm_total = float("nan")
    norm_mean_distance = mean_distance
    norm_std_distance = std_distance
    norm_fft_distance = fft_distance
    if norm_scale:
        nm = float(F.l1_loss(item_i.mean / norm_scale["mean"], item_j.mean / norm_scale["mean"]).item())
        ns = float(F.l1_loss(item_i.std / norm_scale["std"], item_j.std / norm_scale["std"]).item())
        nf = float(F.l1_loss(item_i.fft / norm_scale["fft"], item_j.fft / norm_scale["fft"]).item())
        norm_mean_distance, norm_std_distance, norm_fft_distance = nm, ns, nf
        norm_weighted = float(mean_weight) * nm + float(std_weight) * ns + float(fft_weight) * nf
        norm_total = norm_weighted / total_weight if total_weight > 0.0 else norm_weighted

    return {
        "mean_distance": mean_distance,
        "std_distance": std_distance,
        "fft_distance": fft_distance,
        "total_distance": total_distance,
        "norm_mean_distance": norm_mean_distance,
        "norm_std_distance": norm_std_distance,
        "norm_fft_distance": norm_fft_distance,
        "norm_total_distance": norm_total,
        "concat_cosine_distance": cosine_distance(item_i.concat, item_j.concat),
    }


def cosine_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    left_n = F.normalize(left.float(), dim=0, eps=1e-6)
    right_n = F.normalize(right.float(), dim=0, eps=1e-6)
    return float((1.0 - torch.dot(left_n, right_n)).item())


def variance_summary(items: list[DescriptorItem]) -> dict[str, Any]:
    if not items:
        return {"count": 0}
    mean_stack = torch.stack([item.mean for item in items]).float()
    std_stack = torch.stack([item.std for item in items]).float()
    fft_stack = torch.stack([item.fft for item in items]).float()
    concat_stack = torch.stack([item.concat for item in items]).float()
    return {
        "count": len(items),
        "mean_token_centered_l2": describe_values(torch.linalg.vector_norm(mean_stack - mean_stack.mean(dim=0), dim=1)),
        "std_token_centered_l2": describe_values(torch.linalg.vector_norm(std_stack - std_stack.mean(dim=0), dim=1)),
        "fft_token_centered_l2": describe_values(torch.linalg.vector_norm(fft_stack - fft_stack.mean(dim=0), dim=1)),
        "concat_centered_l2": describe_values(torch.linalg.vector_norm(concat_stack - concat_stack.mean(dim=0), dim=1)),
    }


def build_coarse_lookup(fine_to_parent: dict[int, int], *, device: torch.device) -> torch.Tensor:
    lookup = torch.full((max(fine_to_parent) + 1,), -1, dtype=torch.long, device=device)
    for fine_id, parent_id in fine_to_parent.items():
        lookup[int(fine_id)] = int(parent_id)
    return lookup


def remap_fine_to_coarse(labels: torch.Tensor, lookup: torch.Tensor) -> torch.Tensor:
    lookup = lookup.to(device=labels.device)
    clamped = labels.clamp(min=0, max=lookup.shape[0] - 1)
    coarse = lookup[clamped]
    return torch.where(labels > 0, coarse, torch.full_like(coarse, -1))


def normalize_label_mode(value: str) -> str:
    raw = str(value).strip().lower().replace("-", "_")
    if raw in {"coarse", "coarse_tissue", "parent", "parent_tissue"}:
        return "coarse_tissue"
    if raw in {"fine", "tissue"}:
        return "fine"
    raise ValueError(f"unsupported label mode: {value!r}")


def build_label_name_lookup(
    label_mode: str,
    coarse_labels: dict[int, str],
    fine_labels: dict[int, str],
) -> dict[str, int]:
    labels = coarse_labels if label_mode == "coarse_tissue" else fine_labels
    lookup: dict[str, int] = {}
    for label_id, label_name in labels.items():
        variants = {
            str(label_id),
            label_name,
            label_name.lower(),
            label_name.lower().replace(" ", "_"),
            label_name.lower().replace(" ", "-"),
        }
        if label_name.lower() == "tumor":
            variants.update({"tumour", "cancer"})
        if label_name.lower() == "immune infiltrate":
            variants.update({"immune", "lymphocyte", "inflammatory"})
        if label_name.lower() == "normal epithelium":
            variants.update({"normal", "epithelium", "normal_epi"})
        for variant in variants:
            lookup[normalize_label_name_key(variant)] = int(label_id)
    return lookup


def parse_label(value: str, lookup: dict[str, int]) -> int:
    key = normalize_label_name_key(value)
    if key in lookup:
        return int(lookup[key])
    try:
        return int(value)
    except ValueError as exc:
        known = ", ".join(sorted(set(lookup))[:40])
        raise ValueError(f"Unknown label {value!r}; known examples: {known}") from exc


def canonical_label_name(label_id: int, lookup: dict[str, int], *, fallback: str) -> str:
    for key, value in lookup.items():
        if value == int(label_id) and not key.isdigit():
            return key
    return fallback


def normalize_label_name_key(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def describe_values(values: torch.Tensor) -> dict[str, Any]:
    values = values.float().flatten()
    if values.numel() == 0:
        return {"count": 0}
    quantiles = torch.quantile(
        values,
        torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=values.dtype),
    )
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "q05": float(quantiles[0].item()),
        "q25": float(quantiles[1].item()),
        "median": float(quantiles[2].item()),
        "q75": float(quantiles[3].item()),
        "q95": float(quantiles[4].item()),
        "max": float(values.max().item()),
    }


def greater_than_probability(left_values: list[float], right_values: list[float]) -> float | None:
    if not left_values or not right_values:
        return None
    left = torch.tensor(left_values, dtype=torch.float32)
    right = torch.tensor(right_values, dtype=torch.float32)
    return float((left[:, None] > right[None, :]).float().mean().item())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf8")
        return
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def descriptor_table(items: list[DescriptorItem]) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(items):
        rows.append(
            {
                "descriptor_index": index,
                "label_id": item.label_id,
                "label_name": item.label_name,
                "dataset": item.sample.dataset,
                "sample_id": item.sample.sample_id,
                "image_path": str(item.sample.image_path),
                "tissue_mask_path": str(item.sample.tissue_mask_path),
                "region_pixels": item.region_pixels,
                "region_fraction": item.region_fraction,
                "mean_norm": float(torch.linalg.vector_norm(item.mean).item()),
                "std_norm": float(torch.linalg.vector_norm(item.std).item()),
                "fft_norm": float(torch.linalg.vector_norm(item.fft).item()),
                "concat_norm": float(torch.linalg.vector_norm(item.concat).item()),
            }
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())

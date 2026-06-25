#!/usr/bin/env python3
"""Probe real-target -> UNI vs reference -> UNI region descriptor separability.

This diagnostic simulates the reference-region loss path used by the UNI backend:

    target/generated RGB -> optional VAE encode/decode -> frozen UNI tokens -> region mean/std
    reference RGB        -> frozen UNI tokens                         -> region mean/std

It uses real target images as the best-case generated image proxy. The output
checks whether same-label target/reference regions stay closer than cross-label
regions, and whether the paired reference is closer than unpaired same-label
references.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass(frozen=True)
class PairEntry:
    index: int
    dataset: str
    sample_id: str
    reference_sample_id: str
    target_image_path: Path
    target_tissue_mask_path: Path
    reference_image_path: Path
    reference_tissue_mask_path: Path

    @property
    def target_key(self) -> str:
        return descriptor_key("target", self.target_image_path, self.target_tissue_mask_path)

    @property
    def reference_key(self) -> str:
        return descriptor_key("reference", self.reference_image_path, self.reference_tissue_mask_path)


@dataclass(frozen=True)
class ImageEntry:
    key: str
    role: str
    dataset: str
    sample_id: str
    image_path: Path
    tissue_mask_path: Path


@dataclass(frozen=True)
class DescriptorItem:
    label_name: str
    label_id: int
    image: ImageEntry
    token_count: int
    token_fraction: float
    mean: torch.Tensor
    std: torch.Tensor

    @property
    def concat(self) -> torch.Tensor:
        return torch.cat([self.mean, self.std], dim=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate target->UNI vs reference->UNI region descriptor separability."
    )
    parser.add_argument("--metadata", required=True, help="Cross metadata JSON/JSONL with pairs.")
    parser.add_argument("--uni-checkpoint-path", required=True, help="UNI-2h pytorch_model.bin path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--target-vae-roundtrip",
        action="store_true",
        help="Encode/decode target images through the FLUX VAE before extracting UNI features.",
    )
    parser.add_argument(
        "--pretrained-model-name-or-path",
        default=None,
        help="FLUX model path/name used to load subfolder='vae' when --target-vae-roundtrip is set.",
    )
    parser.add_argument(
        "--vae-torch-dtype",
        choices=("same", "fp32", "bf16", "fp16"),
        default="same",
        help="VAE dtype for target roundtrip. Defaults to --torch-dtype.",
    )
    parser.add_argument(
        "--metadata-base-dir",
        default=None,
        help="Base directory for relative metadata paths. Defaults to metadata parent.",
    )
    parser.add_argument(
        "--label-mode",
        choices=("coarse_tissue", "coarse", "fine", "tissue"),
        default="coarse_tissue",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Label name or id to probe. Repeatable. Defaults to tumor and stroma.",
    )
    parser.add_argument("--candidate-pool-size", type=int, default=5000)
    parser.add_argument("--samples-per-label", type=int, default=64)
    parser.add_argument("--min-region-tokens", type=int, default=2)
    parser.add_argument("--min-region-fraction", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--mean-weight", type=float, default=1.0)
    parser.add_argument("--std-weight", type=float, default=0.5)
    parser.add_argument("--pooled-cosine-weight", type=float, default=0.25)
    parser.add_argument(
        "--unpaired-references-per-target",
        type=int,
        default=8,
        help="Sample this many unpaired same-label references per target descriptor. 0 disables.",
    )
    parser.add_argument(
        "--max-unpaired-pairs",
        type=int,
        default=20000,
        help="Global cap for unpaired_same_label pair rows.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.label is None:
        args.label = ["tumor", "stroma"]
    if len(args.label) < 2:
        raise ValueError("Provide at least two --label values for cross-label comparison.")
    if args.candidate_pool_size <= 0:
        raise ValueError("--candidate-pool-size must be positive")
    if args.samples_per_label <= 0:
        raise ValueError("--samples-per-label must be positive")
    if args.min_region_tokens <= 0:
        raise ValueError("--min-region-tokens must be positive")
    if not 0.0 <= args.min_region_fraction <= 1.0:
        raise ValueError("--min-region-fraction must be in [0, 1]")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.unpaired_references_per_target < 0:
        raise ValueError("--unpaired-references-per-target must be non-negative")
    if args.max_unpaired_pairs < 0:
        raise ValueError("--max-unpaired-pairs must be non-negative")
    if args.target_vae_roundtrip and not args.pretrained_model_name_or_path:
        raise ValueError("--pretrained-model-name-or-path is required with --target-vae-roundtrip")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from controlnet_train.data.common import load_image_tensor, load_tissue_mask
    from controlnet_train.modules.reference_image_encoder import (
        ReferenceImageEncoder,
        resize_mask_to_token_labels,
    )
    from dataset_config import COARSE_LABELS, FINE_LABELS, FINE_TO_PARENT

    metadata_path = Path(args.metadata)
    base_dir = Path(args.metadata_base_dir) if args.metadata_base_dir else metadata_path.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_mode = normalize_label_mode(args.label_mode)
    label_lookup = build_label_name_lookup(label_mode, COARSE_LABELS, FINE_LABELS)
    label_ids = [parse_label(value, label_lookup) for value in args.label]
    if len(set(label_ids)) != len(label_ids):
        raise ValueError(f"--label values must resolve to unique ids, got {label_ids}")

    records = read_metadata(metadata_path)
    pair_entries = build_pair_entries(records, base_dir=base_dir)
    rng = random.Random(args.seed)
    rng.shuffle(pair_entries)
    pair_entries = pair_entries[: int(args.candidate_pool_size)]

    image_entries = build_image_entries(pair_entries)
    device = resolve_device(args.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.torch_dtype]
    encoder = ReferenceImageEncoder(args.uni_checkpoint_path, skip_perceiver=True)
    encoder.to(device=device, dtype=dtype)
    encoder.eval()
    vae = None
    vae_dtype = dtype
    if args.target_vae_roundtrip:
        from diffusers import AutoencoderKL

        vae_dtype = dtype if args.vae_torch_dtype == "same" else parse_torch_dtype(args.vae_torch_dtype)
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            torch_dtype=vae_dtype,
        )
        vae.to(device=device, dtype=vae_dtype)
        vae.eval()
        vae.requires_grad_(False)

    descriptors, skipped = collect_descriptors(
        image_entries,
        label_ids=label_ids,
        label_lookup=label_lookup,
        label_mode=label_mode,
        fine_to_parent=FINE_TO_PARENT,
        encoder=encoder,
        target_vae_roundtrip=bool(args.target_vae_roundtrip),
        vae=vae,
        vae_dtype=vae_dtype,
        load_image_tensor=load_image_tensor,
        load_tissue_mask=load_tissue_mask,
        resize_mask_to_token_labels=resize_mask_to_token_labels,
        samples_per_label=int(args.samples_per_label),
        min_region_tokens=int(args.min_region_tokens),
        min_region_fraction=float(args.min_region_fraction),
        batch_size=int(args.batch_size),
        device=device,
        dtype=dtype,
        progress_every=int(args.progress_every),
    )

    pair_rows, summary = build_pair_outputs(
        pair_entries=pair_entries,
        descriptors=descriptors,
        label_ids=label_ids,
        rng=random.Random(args.seed + 1009),
        unpaired_references_per_target=int(args.unpaired_references_per_target),
        max_unpaired_pairs=int(args.max_unpaired_pairs),
        mean_weight=float(args.mean_weight),
        std_weight=float(args.std_weight),
        pooled_cosine_weight=float(args.pooled_cosine_weight),
    )

    descriptor_rows = descriptor_table(descriptors)
    write_csv(output_dir / "uni_target_reference_region_descriptors.csv", descriptor_rows)
    write_csv(output_dir / "uni_target_reference_region_pairs.csv", pair_rows)
    summary.update(
        {
            "metadata": str(metadata_path),
            "target_input_mode": "vae_roundtrip" if args.target_vae_roundtrip else "real_rgb",
            "pretrained_model_name_or_path": str(args.pretrained_model_name_or_path)
            if args.target_vae_roundtrip
            else None,
            "label_mode": label_mode,
            "labels": [
                {"id": int(label_id), "name": canonical_label_name(label_id, label_lookup, fallback=str(label_id))}
                for label_id in label_ids
            ],
            "candidate_pair_entries": len(pair_entries),
            "unique_image_entries": len(image_entries),
            "descriptor_count": len(descriptors),
            "descriptor_role_label_counts": role_label_counts(descriptors),
            "skipped_count": len(skipped),
            "skipped_preview": skipped[:100],
            "outputs": {
                "descriptors_csv": "uni_target_reference_region_descriptors.csv",
                "pairs_csv": "uni_target_reference_region_pairs.csv",
                "summary_json": "uni_target_reference_region_separability_summary.json",
            },
        }
    )
    write_json(output_dir / "uni_target_reference_region_separability_summary.json", summary)
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


def build_pair_entries(records: list[dict[str, Any]], *, base_dir: Path) -> list[PairEntry]:
    entries: list[PairEntry] = []
    seen: set[tuple[str, str, str, str]] = set()
    required = ("target_image", "target_tissue_mask", "reference_image", "reference_tissue_mask")
    for index, record in enumerate(records):
        if not all(record.get(key) for key in required):
            continue
        target_image = resolve_metadata_path(record["target_image"], base_dir)
        target_mask = resolve_metadata_path(record["target_tissue_mask"], base_dir)
        reference_image = resolve_metadata_path(record["reference_image"], base_dir)
        reference_mask = resolve_metadata_path(record["reference_tissue_mask"], base_dir)
        key = (str(target_image), str(target_mask), str(reference_image), str(reference_mask))
        if key in seen:
            continue
        seen.add(key)
        entries.append(
            PairEntry(
                index=index,
                dataset=str(record.get("dataset") or "unknown"),
                sample_id=str(record.get("sample_id") or target_image.stem),
                reference_sample_id=str(record.get("reference_sample_id") or reference_image.stem),
                target_image_path=target_image,
                target_tissue_mask_path=target_mask,
                reference_image_path=reference_image,
                reference_tissue_mask_path=reference_mask,
            )
        )
    return entries


def build_image_entries(pair_entries: list[PairEntry]) -> list[ImageEntry]:
    entries: dict[str, ImageEntry] = {}
    for pair in pair_entries:
        entries.setdefault(
            pair.target_key,
            ImageEntry(
                key=pair.target_key,
                role="target",
                dataset=pair.dataset,
                sample_id=pair.sample_id,
                image_path=pair.target_image_path,
                tissue_mask_path=pair.target_tissue_mask_path,
            ),
        )
        entries.setdefault(
            pair.reference_key,
            ImageEntry(
                key=pair.reference_key,
                role="reference",
                dataset=pair.dataset,
                sample_id=pair.reference_sample_id,
                image_path=pair.reference_image_path,
                tissue_mask_path=pair.reference_tissue_mask_path,
            ),
        )
    return list(entries.values())


def collect_descriptors(
    image_entries: list[ImageEntry],
    *,
    label_ids: list[int],
    label_lookup: dict[str, int],
    label_mode: str,
    fine_to_parent: dict[int, int],
    encoder,
    target_vae_roundtrip: bool,
    vae,
    vae_dtype: torch.dtype,
    load_image_tensor,
    load_tissue_mask,
    resize_mask_to_token_labels,
    samples_per_label: int,
    min_region_tokens: int,
    min_region_fraction: float,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    progress_every: int,
) -> tuple[dict[tuple[str, int], DescriptorItem], list[dict[str, Any]]]:
    descriptors: dict[tuple[str, int], DescriptorItem] = {}
    skipped: list[dict[str, Any]] = []
    pending_entries: list[ImageEntry] = []
    pending_images: list[torch.Tensor] = []
    pending_masks: list[torch.Tensor] = []
    roles = ("target", "reference")
    per_role_label_count = {
        role: {int(label_id): 0 for label_id in label_ids}
        for role in roles
    }
    remap_lookup = build_coarse_lookup(fine_to_parent, device=torch.device("cpu"))
    scanned = 0

    def enough() -> bool:
        return all(
            per_role_label_count[role][int(label_id)] >= samples_per_label
            for role in roles
            for label_id in label_ids
        )

    def flush() -> None:
        nonlocal scanned
        if not pending_entries:
            return
        images = torch.stack(pending_images).to(device=device, dtype=dtype)
        masks = torch.stack(pending_masks)
        with torch.no_grad():
            if target_vae_roundtrip:
                if vae is None:
                    raise ValueError("vae is required when target_vae_roundtrip=True")
                target_indices = [
                    batch_index for batch_index, entry in enumerate(pending_entries) if entry.role == "target"
                ]
                if target_indices:
                    index_tensor = torch.tensor(target_indices, device=images.device, dtype=torch.long)
                    roundtripped = vae_roundtrip_images(
                        vae,
                        images.index_select(0, index_tensor),
                        torch_dtype=vae_dtype,
                    )
                    images = images.clone()
                    images.index_copy_(0, index_tensor, roundtripped.to(device=images.device, dtype=dtype))
            features = encoder.extract_uni_features(images).float().cpu()
        labels = resize_mask_to_token_labels(masks, int(features.shape[1]))
        if label_mode == "coarse_tissue":
            labels = remap_fine_to_coarse(labels, remap_lookup)
        for batch_index, entry in enumerate(pending_entries):
            for label_id in label_ids:
                if per_role_label_count[entry.role][int(label_id)] >= samples_per_label:
                    continue
                region = labels[batch_index] == int(label_id)
                token_count = int(region.sum().item())
                token_fraction = float(token_count / max(1, int(region.numel())))
                if token_count < min_region_tokens or token_fraction < min_region_fraction:
                    continue
                tokens = features[batch_index, region]
                descriptors[(entry.key, int(label_id))] = DescriptorItem(
                    label_name=canonical_label_name(label_id, label_lookup, fallback=str(label_id)),
                    label_id=int(label_id),
                    image=entry,
                    token_count=token_count,
                    token_fraction=token_fraction,
                    mean=tokens.mean(dim=0),
                    std=torch.sqrt(tokens.var(dim=0, unbiased=False) + 1e-6),
                )
                per_role_label_count[entry.role][int(label_id)] += 1
        scanned += len(pending_entries)
        if progress_every > 0 and scanned % int(progress_every) < len(pending_entries):
            counts = " ".join(
                f"{role}:{label}={per_role_label_count[role][int(label)]}"
                for role in roles
                for label in label_ids
            )
            print(f"[uni-target-ref] scanned={scanned}/{len(image_entries)} {counts}", flush=True)
        pending_entries.clear()
        pending_images.clear()
        pending_masks.clear()

    for entry in image_entries:
        if enough():
            break
        try:
            pending_images.append(load_image_tensor(entry.image_path))
            pending_masks.append(load_tissue_mask(entry.tissue_mask_path))
            pending_entries.append(entry)
        except Exception as exc:  # noqa: BLE001 - diagnostic script should keep going.
            skipped.append(
                {
                    "sample_id": entry.sample_id,
                    "role": entry.role,
                    "image_path": str(entry.image_path),
                    "mask_path": str(entry.tissue_mask_path),
                    "reason": f"load_failed:{type(exc).__name__}",
                    "detail": str(exc),
                }
            )
            continue
        if len(pending_entries) >= batch_size:
            flush()
    flush()
    return descriptors, skipped


def build_pair_outputs(
    *,
    pair_entries: list[PairEntry],
    descriptors: dict[tuple[str, int], DescriptorItem],
    label_ids: list[int],
    rng: random.Random,
    unpaired_references_per_target: int,
    max_unpaired_pairs: int,
    mean_weight: float,
    std_weight: float,
    pooled_cosine_weight: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pair_rows: list[dict[str, Any]] = []
    target_items_by_label: dict[int, list[tuple[PairEntry, DescriptorItem]]] = {label: [] for label in label_ids}
    reference_items_by_label: dict[int, list[DescriptorItem]] = {label: [] for label in label_ids}

    seen_reference_items: set[tuple[str, int]] = set()
    for pair in pair_entries:
        for label_id in label_ids:
            target_item = descriptors.get((pair.target_key, int(label_id)))
            reference_item = descriptors.get((pair.reference_key, int(label_id)))
            if target_item is not None:
                target_items_by_label[int(label_id)].append((pair, target_item))
            if reference_item is not None and (reference_item.image.key, int(label_id)) not in seen_reference_items:
                reference_items_by_label[int(label_id)].append(reference_item)
                seen_reference_items.add((reference_item.image.key, int(label_id)))
            if target_item is None or reference_item is None:
                continue
            pair_rows.append(
                format_pair(
                    target_item,
                    reference_item,
                    pair_group="paired_same_label",
                    pair=pair,
                    ref_label_id=int(label_id),
                    mean_weight=mean_weight,
                    std_weight=std_weight,
                    pooled_cosine_weight=pooled_cosine_weight,
                )
            )
            for other_label_id in label_ids:
                if int(other_label_id) == int(label_id):
                    continue
                cross_ref = descriptors.get((pair.reference_key, int(other_label_id)))
                if cross_ref is None:
                    continue
                pair_rows.append(
                    format_pair(
                        target_item,
                        cross_ref,
                        pair_group="paired_cross_label",
                        pair=pair,
                        ref_label_id=int(other_label_id),
                        mean_weight=mean_weight,
                        std_weight=std_weight,
                        pooled_cosine_weight=pooled_cosine_weight,
                    )
                )

    unpaired_count = 0
    if unpaired_references_per_target > 0 and max_unpaired_pairs > 0:
        for label_id in label_ids:
            ref_pool = reference_items_by_label[int(label_id)]
            for pair, target_item in target_items_by_label[int(label_id)]:
                target_path_key = (pair.target_image_path, pair.target_tissue_mask_path)
                paired_reference_path_key = (pair.reference_image_path, pair.reference_tissue_mask_path)
                candidates = [
                    item
                    for item in ref_pool
                    if (item.image.image_path, item.image.tissue_mask_path)
                    not in {target_path_key, paired_reference_path_key}
                ]
                if not candidates:
                    continue
                rng.shuffle(candidates)
                for reference_item in candidates[:unpaired_references_per_target]:
                    if unpaired_count >= max_unpaired_pairs:
                        break
                    pair_rows.append(
                        format_pair(
                            target_item,
                            reference_item,
                            pair_group="unpaired_same_label",
                            pair=pair,
                            ref_label_id=int(label_id),
                            mean_weight=mean_weight,
                            std_weight=std_weight,
                            pooled_cosine_weight=pooled_cosine_weight,
                        )
                    )
                    unpaired_count += 1
                if unpaired_count >= max_unpaired_pairs:
                    break
            if unpaired_count >= max_unpaired_pairs:
                break

    summary = {
        "counts": {
            "pair_entries": len(pair_entries),
            "paired_same_label_pairs": count_group(pair_rows, "paired_same_label"),
            "paired_cross_label_pairs": count_group(pair_rows, "paired_cross_label"),
            "unpaired_same_label_pairs": count_group(pair_rows, "unpaired_same_label"),
        },
        "distance_stats": summarize_pair_rows(pair_rows),
    }
    summary["comparisons"] = build_comparison_summary(pair_rows)
    summary["target_reference_uni_verdict"] = build_verdict(summary)
    return pair_rows, summary


def format_pair(
    target_item: DescriptorItem,
    reference_item: DescriptorItem,
    *,
    pair_group: str,
    pair: PairEntry,
    ref_label_id: int,
    mean_weight: float,
    std_weight: float,
    pooled_cosine_weight: float,
) -> dict[str, Any]:
    distances = descriptor_distances(
        target_item,
        reference_item,
        mean_weight=mean_weight,
        std_weight=std_weight,
        pooled_cosine_weight=pooled_cosine_weight,
    )
    return {
        "pair_group": pair_group,
        "metadata_index": pair.index,
        "dataset": pair.dataset,
        "target_sample_id": pair.sample_id,
        "paired_reference_sample_id": pair.reference_sample_id,
        "target_label_id": target_item.label_id,
        "target_label_name": target_item.label_name,
        "reference_label_id": int(ref_label_id),
        "reference_label_name": reference_item.label_name,
        "reference_sample_id": reference_item.image.sample_id,
        "target_role": target_item.image.role,
        "reference_role": reference_item.image.role,
        "target_image": str(target_item.image.image_path),
        "reference_image": str(reference_item.image.image_path),
        "target_tokens": target_item.token_count,
        "reference_tokens": reference_item.token_count,
        **distances,
    }


def descriptor_distances(
    target_item: DescriptorItem,
    reference_item: DescriptorItem,
    *,
    mean_weight: float,
    std_weight: float,
    pooled_cosine_weight: float,
) -> dict[str, float]:
    mean_l1 = float(F.l1_loss(target_item.mean, reference_item.mean).item())
    std_l1 = float(F.l1_loss(target_item.std, reference_item.std).item())
    mean_cosine_distance = cosine_distance(target_item.mean, reference_item.mean)
    std_cosine_distance = cosine_distance(target_item.std, reference_item.std)
    concat_cosine_distance = cosine_distance(target_item.concat, reference_item.concat)
    total_weight = float(mean_weight) + float(std_weight) + float(pooled_cosine_weight)
    weighted = float(mean_weight) * mean_l1
    weighted += float(std_weight) * std_l1
    weighted += float(pooled_cosine_weight) * mean_cosine_distance
    total_distance = weighted / total_weight if total_weight > 0.0 else weighted
    return {
        "mean_l1_distance": mean_l1,
        "std_l1_distance": std_l1,
        "mean_cosine_distance": mean_cosine_distance,
        "std_cosine_distance": std_cosine_distance,
        "concat_cosine_distance": concat_cosine_distance,
        "two_token_average_cosine_distance": float((mean_cosine_distance + std_cosine_distance) * 0.5),
        "region_loss_style_distance": total_distance,
    }


def summarize_pair_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = sorted({str(row["pair_group"]) for row in rows})
    metrics = (
        "mean_l1_distance",
        "std_l1_distance",
        "mean_cosine_distance",
        "std_cosine_distance",
        "concat_cosine_distance",
        "two_token_average_cosine_distance",
        "region_loss_style_distance",
    )
    summary: dict[str, Any] = {}
    for metric in metrics:
        metric_summary = {}
        for group in groups:
            values = [float(row[metric]) for row in rows if row["pair_group"] == group]
            metric_summary[group] = describe_values(torch.tensor(values, dtype=torch.float32))
        summary[metric] = metric_summary
    return summary


def build_comparison_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "region_loss_style_distance",
        "concat_cosine_distance",
        "two_token_average_cosine_distance",
    )
    summary: dict[str, Any] = {}
    for metric in metrics:
        paired_same = [float(row[metric]) for row in rows if row["pair_group"] == "paired_same_label"]
        paired_cross = [float(row[metric]) for row in rows if row["pair_group"] == "paired_cross_label"]
        unpaired_same = [float(row[metric]) for row in rows if row["pair_group"] == "unpaired_same_label"]
        metric_summary = {
            "paired_cross_minus_paired_same_mean": mean_difference(paired_cross, paired_same),
            "paired_cross_greater_than_paired_same_probability": greater_than_probability(paired_cross, paired_same),
            "unpaired_same_minus_paired_same_mean": mean_difference(unpaired_same, paired_same),
            "unpaired_same_greater_than_paired_same_probability": greater_than_probability(unpaired_same, paired_same),
        }
        summary[metric] = metric_summary
    return summary


def build_verdict(summary: dict[str, Any]) -> dict[str, Any]:
    comparison = summary.get("comparisons", {}).get("region_loss_style_distance", {})
    cross_prob = comparison.get("paired_cross_greater_than_paired_same_probability")
    cross_margin = comparison.get("paired_cross_minus_paired_same_mean")
    unpaired_prob = comparison.get("unpaired_same_greater_than_paired_same_probability")
    unpaired_margin = comparison.get("unpaired_same_minus_paired_same_mean")
    verdict = {
        "primary_metric": "region_loss_style_distance",
        "paired_cross_greater_than_paired_same_probability": cross_prob,
        "paired_cross_minus_paired_same_mean": cross_margin,
        "unpaired_same_greater_than_paired_same_probability": unpaired_prob,
        "unpaired_same_minus_paired_same_mean": unpaired_margin,
    }
    if cross_prob is None:
        verdict["reading"] = "insufficient paired same/cross rows"
    elif cross_prob >= 0.80 and (cross_margin is not None and cross_margin > 0.0):
        verdict["reading"] = (
            "target->UNI vs ref->UNI preserves same-label-vs-cross-label separability; "
            "the decode/UNI path is viable for a UNI region loss sanity check."
        )
    else:
        verdict["reading"] = (
            "target->UNI vs ref->UNI separability is weak; validate masks/labels and "
            "do not enable UNI region loss until this is understood."
        )
    return verdict


def descriptor_table(descriptors: dict[tuple[str, int], DescriptorItem]) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(descriptors.values()):
        rows.append(
            {
                "descriptor_index": index,
                "label_id": item.label_id,
                "label_name": item.label_name,
                "role": item.image.role,
                "dataset": item.image.dataset,
                "sample_id": item.image.sample_id,
                "image_path": str(item.image.image_path),
                "tissue_mask_path": str(item.image.tissue_mask_path),
                "token_count": item.token_count,
                "token_fraction": item.token_fraction,
                "mean_norm": float(torch.linalg.vector_norm(item.mean).item()),
                "std_norm": float(torch.linalg.vector_norm(item.std).item()),
                "concat_norm": float(torch.linalg.vector_norm(item.concat).item()),
            }
        )
    return rows


def role_label_counts(descriptors: dict[tuple[str, int], DescriptorItem]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for item in descriptors.values():
        role_counts = counts.setdefault(item.image.role, {})
        label_key = f"{item.label_id}:{item.label_name}"
        role_counts[label_key] = role_counts.get(label_key, 0) + 1
    return counts


def count_group(rows: list[dict[str, Any]], group: str) -> int:
    return sum(1 for row in rows if row["pair_group"] == group)


def mean_difference(left_values: list[float], right_values: list[float]) -> float | None:
    if not left_values or not right_values:
        return None
    left = torch.tensor(left_values, dtype=torch.float32)
    right = torch.tensor(right_values, dtype=torch.float32)
    return float(left.mean().item() - right.mean().item())


def greater_than_probability(left_values: list[float], right_values: list[float]) -> float | None:
    if not left_values or not right_values:
        return None
    left = torch.tensor(left_values, dtype=torch.float32)
    right = torch.tensor(right_values, dtype=torch.float32)
    return float((left[:, None] > right[None, :]).float().mean().item())


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


def cosine_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    left_n = F.normalize(left.float(), dim=0, eps=1e-6)
    right_n = F.normalize(right.float(), dim=0, eps=1e-6)
    return float((1.0 - torch.dot(left_n, right_n)).item())


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


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def parse_torch_dtype(value: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[value]


def vae_roundtrip_images(vae, images: torch.Tensor, *, torch_dtype: torch.dtype) -> torch.Tensor:
    """Deterministically encode/decode images through the FLUX VAE."""
    device = next(vae.parameters()).device
    images = images.to(device=device, dtype=torch_dtype)
    posterior = vae.encode(images * 2.0 - 1.0).latent_dist
    if hasattr(posterior, "mode"):
        latents = posterior.mode()
    elif hasattr(posterior, "mean"):
        latents = posterior.mean
    else:
        latents = posterior.sample()

    scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0) or 1.0)
    shift_factor = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
    latents = (latents - shift_factor) * scaling_factor
    latent_input = (latents / scaling_factor) + shift_factor
    decoded = vae.decode(latent_input.to(device=device, dtype=torch_dtype), return_dict=False)[0]
    return ((decoded.float() / 2.0) + 0.5).clamp(0.0, 1.0)


def resolve_metadata_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(str(path_value).replace("\\", "/"))
    if path.is_absolute():
        return path
    return base_dir / path


def descriptor_key(role: str, image_path: Path, mask_path: Path) -> str:
    return f"{role}::{image_path}::{mask_path}"


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


if __name__ == "__main__":
    raise SystemExit(main())

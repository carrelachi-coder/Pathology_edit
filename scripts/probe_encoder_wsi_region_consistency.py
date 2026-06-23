#!/usr/bin/env python3
"""Probe same-WSI target/reference regional consistency for encoder layers.

For a texture/style layer to be useful as a reference-following signal, it is
not enough that it reacts to blur or stain perturbations. It should also place
same-WSI target/reference regions of the same tissue closer than same-tissue
regions from different WSIs.

This script compares:

    paired same-label target/ref regions from the metadata pair
    vs.
    unpaired same-label reference regions from different WSIs

for UNI final tokens, optional UNI intermediate layers, and/or CONCH tokens.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probe_encoder_texture_sensitivity import load_encoders  # noqa: E402
from probe_uni_target_reference_region_separability import (  # noqa: E402
    build_coarse_lookup,
    build_label_name_lookup,
    canonical_label_name,
    cosine_distance,
    describe_values,
    greater_than_probability,
    mean_difference,
    normalize_label_mode,
    parse_label,
    parse_torch_dtype,
    read_metadata,
    remap_fine_to_coarse,
    resolve_device,
    resolve_metadata_path,
    write_json,
)


@dataclass(frozen=True)
class PairEntry:
    index: int
    dataset: str
    target_sample_id: str
    reference_sample_id: str
    target_wsi_id: str
    reference_wsi_id: str
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
    wsi_id: str
    image_path: Path
    tissue_mask_path: Path


@dataclass
class DescriptorItem:
    backend: str
    label_id: int
    label_name: str
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
        description="Measure same-WSI target/ref same-label consistency for UNI layers or CONCH."
    )
    parser.add_argument("--metadata", required=True, help="Cross metadata JSON/JSONL with pairs.")
    parser.add_argument("--metadata-base-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", choices=("uni", "conch", "both"), default="uni")
    parser.add_argument("--uni-checkpoint-path", default=None)
    parser.add_argument(
        "--uni-layer",
        type=int,
        action="append",
        default=None,
        help="1-based UNI transformer block to probe. Repeatable.",
    )
    parser.add_argument(
        "--uni-include-final",
        action="store_true",
        help="When --uni-layer is set, also include final UNI tokens.",
    )
    parser.add_argument("--conch-checkpoint-path", default=None)
    parser.add_argument(
        "--conch-layer",
        type=int,
        action="append",
        default=None,
        help="1-based CONCH visual transformer block to probe. Repeatable.",
    )
    parser.add_argument(
        "--conch-include-final",
        action="store_true",
        help="When --conch-layer is set, also include final CONCH tokens.",
    )
    parser.add_argument("--conch-root", default=None)
    parser.add_argument("--conch-model-cfg", default="conch_ViT-B-16")
    parser.add_argument("--case-id-field", default="case_id")
    parser.add_argument("--target-case-id-field", default="target_case_id")
    parser.add_argument("--reference-case-id-field", default="reference_case_id")
    parser.add_argument(
        "--label-mode",
        choices=("coarse_tissue", "coarse", "fine", "tissue"),
        default="coarse_tissue",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Label name/id to probe. Repeatable. Defaults to tumor and stroma.",
    )
    parser.add_argument("--candidate-pool-size", type=int, default=5000)
    parser.add_argument("--samples-per-label", type=int, default=96)
    parser.add_argument("--min-region-tokens", type=int, default=2)
    parser.add_argument("--min-region-fraction", type=float, default=0.0)
    parser.add_argument("--unpaired-references-per-target", type=int, default=8)
    parser.add_argument("--max-unpaired-pairs", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--mean-weight", type=float, default=1.0)
    parser.add_argument("--std-weight", type=float, default=0.5)
    parser.add_argument("--pooled-cosine-weight", type=float, default=0.25)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.label is None:
        args.label = ["tumor", "stroma"]
    if args.backend in {"uni", "both"} and not args.uni_checkpoint_path:
        raise ValueError("--uni-checkpoint-path is required for --backend uni/both")
    if args.backend in {"conch", "both"} and not args.conch_checkpoint_path:
        raise ValueError("--conch-checkpoint-path is required for --backend conch/both")
    if args.candidate_pool_size <= 0:
        raise ValueError("--candidate-pool-size must be positive")
    if args.samples_per_label <= 0:
        raise ValueError("--samples-per-label must be positive")
    if args.min_region_tokens <= 0:
        raise ValueError("--min-region-tokens must be positive")
    if not 0.0 <= args.min_region_fraction <= 1.0:
        raise ValueError("--min-region-fraction must be in [0, 1]")
    if args.unpaired_references_per_target < 0:
        raise ValueError("--unpaired-references-per-target must be non-negative")
    if args.max_unpaired_pairs < 0:
        raise ValueError("--max-unpaired-pairs must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    for layer in args.uni_layer or []:
        if layer <= 0:
            raise ValueError("--uni-layer is 1-based and must be positive")
    for layer in args.conch_layer or []:
        if layer <= 0:
            raise ValueError("--conch-layer is 1-based and must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from controlnet_train.data.common import load_image_tensor, load_tissue_mask
    from controlnet_train.modules.reference_image_encoder import resize_mask_to_token_labels
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
    pair_entries = build_pair_entries(
        records,
        base_dir=base_dir,
        case_id_field=str(args.case_id_field),
        target_case_id_field=str(args.target_case_id_field),
        reference_case_id_field=str(args.reference_case_id_field),
    )
    rng = random.Random(args.seed)
    rng.shuffle(pair_entries)
    pair_entries = pair_entries[: int(args.candidate_pool_size)]
    image_entries = build_image_entries(pair_entries)

    device = resolve_device(args.device)
    dtype = parse_torch_dtype(args.torch_dtype)
    encoders = load_encoders(args, device=device, dtype=dtype)

    descriptors, skipped = collect_descriptors(
        image_entries,
        encoders=encoders,
        label_ids=label_ids,
        label_lookup=label_lookup,
        label_mode=label_mode,
        fine_to_parent=FINE_TO_PARENT,
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
        backends=[encoder.name for encoder in encoders],
        label_ids=label_ids,
        rng=random.Random(args.seed + 1009),
        unpaired_references_per_target=int(args.unpaired_references_per_target),
        max_unpaired_pairs=int(args.max_unpaired_pairs),
        mean_weight=float(args.mean_weight),
        std_weight=float(args.std_weight),
        pooled_cosine_weight=float(args.pooled_cosine_weight),
    )

    descriptor_rows = descriptor_table(descriptors)
    write_csv_rows(output_dir / "encoder_wsi_region_consistency_descriptors.csv", descriptor_rows)
    write_csv_rows(output_dir / "encoder_wsi_region_consistency_pairs.csv", pair_rows)
    summary.update(
        {
            "metadata": str(metadata_path),
            "metadata_base_dir": str(base_dir),
            "backend": str(args.backend),
            "uni_checkpoint_path": str(args.uni_checkpoint_path) if args.uni_checkpoint_path else None,
            "uni_layers": [int(layer) for layer in args.uni_layer or []],
            "uni_include_final": bool(args.uni_include_final),
            "conch_checkpoint_path": str(args.conch_checkpoint_path) if args.conch_checkpoint_path else None,
            "conch_root": str(args.conch_root) if args.conch_root else None,
            "conch_model_cfg": str(args.conch_model_cfg),
            "conch_layers": [int(layer) for layer in args.conch_layer or []],
            "conch_include_final": bool(args.conch_include_final),
            "label_mode": label_mode,
            "labels": [
                {"id": int(label_id), "name": canonical_label_name(label_id, label_lookup, fallback=str(label_id))}
                for label_id in label_ids
            ],
            "candidate_pair_entries": len(pair_entries),
            "unique_image_entries": len(image_entries),
            "descriptor_count": len(descriptors),
            "skipped_count": len(skipped),
            "skipped_preview": skipped[:100],
            "outputs": {
                "descriptors_csv": "encoder_wsi_region_consistency_descriptors.csv",
                "pairs_csv": "encoder_wsi_region_consistency_pairs.csv",
                "summary_json": "encoder_wsi_region_consistency_summary.json",
            },
        }
    )
    write_json(output_dir / "encoder_wsi_region_consistency_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))
    return 0


def build_pair_entries(
    records: list[dict[str, Any]],
    *,
    base_dir: Path,
    case_id_field: str,
    target_case_id_field: str,
    reference_case_id_field: str,
) -> list[PairEntry]:
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

        target_sample_id = str(record.get("sample_id") or target_image.stem)
        reference_sample_id = str(record.get("reference_sample_id") or reference_image.stem)
        shared_case = str(record.get(case_id_field) or "")
        target_wsi = str(record.get(target_case_id_field) or shared_case or infer_wsi_id(target_sample_id, target_image))
        reference_wsi = str(
            record.get(reference_case_id_field) or shared_case or infer_wsi_id(reference_sample_id, reference_image)
        )
        entries.append(
            PairEntry(
                index=index,
                dataset=str(record.get("dataset") or "unknown"),
                target_sample_id=target_sample_id,
                reference_sample_id=reference_sample_id,
                target_wsi_id=target_wsi,
                reference_wsi_id=reference_wsi,
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
                sample_id=pair.target_sample_id,
                wsi_id=pair.target_wsi_id,
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
                wsi_id=pair.reference_wsi_id,
                image_path=pair.reference_image_path,
                tissue_mask_path=pair.reference_tissue_mask_path,
            ),
        )
    return list(entries.values())


def collect_descriptors(
    image_entries: list[ImageEntry],
    *,
    encoders,
    label_ids: list[int],
    label_lookup: dict[str, int],
    label_mode: str,
    fine_to_parent: dict[int, int],
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
) -> tuple[dict[tuple[str, str, int], DescriptorItem], list[dict[str, Any]]]:
    descriptors: dict[tuple[str, str, int], DescriptorItem] = {}
    skipped: list[dict[str, Any]] = []
    pending_entries: list[ImageEntry] = []
    pending_images: list[torch.Tensor] = []
    pending_masks: list[torch.Tensor] = []
    counts = {
        encoder.name: {
            role: {int(label_id): 0 for label_id in label_ids}
            for role in ("target", "reference")
        }
        for encoder in encoders
    }
    remap_lookup = build_coarse_lookup(fine_to_parent, device=torch.device("cpu"))
    scanned = 0

    def enough() -> bool:
        return all(
            counts[encoder.name][role][int(label_id)] >= samples_per_label
            for encoder in encoders
            for role in ("target", "reference")
            for label_id in label_ids
        )

    def flush() -> None:
        nonlocal scanned
        if not pending_entries:
            return
        images = torch.stack(pending_images).to(device=device, dtype=dtype)
        masks = torch.stack(pending_masks)
        for encoder in encoders:
            with torch.no_grad():
                features = encoder.extract_features(images).float().cpu()
            labels = resize_mask_to_token_labels(masks, int(features.shape[1]))
            if label_mode == "coarse_tissue":
                labels = remap_fine_to_coarse(labels, remap_lookup)
            for batch_index, entry in enumerate(pending_entries):
                for label_id in label_ids:
                    label_id = int(label_id)
                    if counts[encoder.name][entry.role][label_id] >= samples_per_label:
                        continue
                    region = labels[batch_index] == label_id
                    token_count = int(region.sum().item())
                    token_fraction = float(token_count / max(1, int(region.numel())))
                    if token_count < min_region_tokens or token_fraction < min_region_fraction:
                        continue
                    tokens = features[batch_index, region]
                    descriptors[(encoder.name, entry.key, label_id)] = DescriptorItem(
                        backend=encoder.name,
                        label_id=label_id,
                        label_name=canonical_label_name(label_id, label_lookup, fallback=str(label_id)),
                        image=entry,
                        token_count=token_count,
                        token_fraction=token_fraction,
                        mean=tokens.mean(dim=0),
                        std=torch.sqrt(tokens.var(dim=0, unbiased=False) + 1e-6),
                    )
                    counts[encoder.name][entry.role][label_id] += 1
        scanned += len(pending_entries)
        if progress_every > 0 and scanned % int(progress_every) < len(pending_entries):
            print(f"[wsi-consistency] scanned={scanned}/{len(image_entries)}", flush=True)
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
        except Exception as exc:  # noqa: BLE001
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
    descriptors: dict[tuple[str, str, int], DescriptorItem],
    backends: list[str],
    label_ids: list[int],
    rng: random.Random,
    unpaired_references_per_target: int,
    max_unpaired_pairs: int,
    mean_weight: float,
    std_weight: float,
    pooled_cosine_weight: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pair_rows: list[dict[str, Any]] = []
    ref_pool: dict[tuple[str, int], list[DescriptorItem]] = {
        (backend, int(label_id)): []
        for backend in backends
        for label_id in label_ids
    }
    seen_refs: set[tuple[str, str, int]] = set()
    for item in descriptors.values():
        if item.image.role != "reference":
            continue
        key = (item.backend, item.image.key, int(item.label_id))
        if key in seen_refs:
            continue
        seen_refs.add(key)
        ref_pool[(item.backend, int(item.label_id))].append(item)

    unpaired_count = 0
    for backend in backends:
        for pair in pair_entries:
            for label_id in label_ids:
                label_id = int(label_id)
                target_item = descriptors.get((backend, pair.target_key, label_id))
                reference_item = descriptors.get((backend, pair.reference_key, label_id))
                if target_item is not None and reference_item is not None:
                    pair_rows.append(
                        format_pair(
                            target_item,
                            reference_item,
                            pair_group="paired_same_label",
                            pair=pair,
                            reference_label_id=label_id,
                            mean_weight=mean_weight,
                            std_weight=std_weight,
                            pooled_cosine_weight=pooled_cosine_weight,
                        )
                    )
                if target_item is None:
                    continue
                if unpaired_references_per_target <= 0 or unpaired_count >= max_unpaired_pairs:
                    continue
                candidates = [
                    item
                    for item in ref_pool[(backend, label_id)]
                    if item.image.wsi_id != target_item.image.wsi_id
                    and item.image.image_path not in {pair.target_image_path, pair.reference_image_path}
                ]
                if not candidates:
                    continue
                rng.shuffle(candidates)
                for item in candidates[:unpaired_references_per_target]:
                    if unpaired_count >= max_unpaired_pairs:
                        break
                    pair_rows.append(
                        format_pair(
                            target_item,
                            item,
                            pair_group="unpaired_same_label_different_wsi",
                            pair=pair,
                            reference_label_id=label_id,
                            mean_weight=mean_weight,
                            std_weight=std_weight,
                            pooled_cosine_weight=pooled_cosine_weight,
                        )
                    )
                    unpaired_count += 1

    summary = {
        "counts": {
            "pair_entries": len(pair_entries),
            "paired_same_label_pairs": count_group(pair_rows, "paired_same_label"),
            "unpaired_same_label_different_wsi_pairs": count_group(
                pair_rows,
                "unpaired_same_label_different_wsi",
            ),
        },
        "distance_stats": summarize_pair_rows(pair_rows, label_ids=label_ids, backends=backends),
        "comparisons": build_comparison_summary(pair_rows, label_ids=label_ids, backends=backends),
    }
    summary["same_wsi_texture_layer_verdict"] = build_verdict(summary)
    return pair_rows, summary


def format_pair(
    target_item: DescriptorItem,
    reference_item: DescriptorItem,
    *,
    pair_group: str,
    pair: PairEntry,
    reference_label_id: int,
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
        "backend": target_item.backend,
        "pair_group": pair_group,
        "metadata_index": pair.index,
        "dataset": pair.dataset,
        "target_sample_id": target_item.image.sample_id,
        "reference_sample_id": reference_item.image.sample_id,
        "paired_reference_sample_id": pair.reference_sample_id,
        "target_wsi_id": target_item.image.wsi_id,
        "reference_wsi_id": reference_item.image.wsi_id,
        "same_wsi": target_item.image.wsi_id == reference_item.image.wsi_id,
        "target_label_id": target_item.label_id,
        "target_label_name": target_item.label_name,
        "reference_label_id": int(reference_label_id),
        "reference_label_name": reference_item.label_name,
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
    mean_cos = cosine_distance(target_item.mean, reference_item.mean)
    std_cos = cosine_distance(target_item.std, reference_item.std)
    concat_cos = cosine_distance(target_item.concat, reference_item.concat)
    total_weight = float(mean_weight) + float(std_weight) + float(pooled_cosine_weight)
    weighted = float(mean_weight) * mean_l1
    weighted += float(std_weight) * std_l1
    weighted += float(pooled_cosine_weight) * mean_cos
    region_loss_style = weighted / total_weight if total_weight > 0.0 else weighted
    return {
        "mean_l1_distance": mean_l1,
        "std_l1_distance": std_l1,
        "mean_cosine_distance": mean_cos,
        "std_cosine_distance": std_cos,
        "concat_cosine_distance": concat_cos,
        "two_token_average_cosine_distance": float((mean_cos + std_cos) * 0.5),
        "region_loss_style_distance": float(region_loss_style),
    }


def summarize_pair_rows(
    rows: list[dict[str, Any]],
    *,
    label_ids: list[int],
    backends: list[str],
) -> dict[str, Any]:
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
    for backend in backends:
        backend_summary: dict[str, Any] = {}
        for label_id in label_ids:
            label_rows = [
                row for row in rows
                if row["backend"] == backend and int(row["target_label_id"]) == int(label_id)
            ]
            label_summary: dict[str, Any] = {}
            groups = sorted({str(row["pair_group"]) for row in label_rows})
            for metric in metrics:
                label_summary[metric] = {
                    group: describe_values(
                        torch.tensor([float(row[metric]) for row in label_rows if row["pair_group"] == group])
                    )
                    for group in groups
                }
            backend_summary[str(int(label_id))] = label_summary
        summary[backend] = backend_summary
    return summary


def build_comparison_summary(
    rows: list[dict[str, Any]],
    *,
    label_ids: list[int],
    backends: list[str],
) -> dict[str, Any]:
    metrics = (
        "region_loss_style_distance",
        "concat_cosine_distance",
        "two_token_average_cosine_distance",
    )
    summary: dict[str, Any] = {}
    for backend in backends:
        backend_summary: dict[str, Any] = {}
        for label_id in label_ids:
            label_rows = [
                row for row in rows
                if row["backend"] == backend and int(row["target_label_id"]) == int(label_id)
            ]
            metric_summary: dict[str, Any] = {}
            for metric in metrics:
                paired = [float(row[metric]) for row in label_rows if row["pair_group"] == "paired_same_label"]
                unpaired = [
                    float(row[metric])
                    for row in label_rows
                    if row["pair_group"] == "unpaired_same_label_different_wsi"
                ]
                metric_summary[metric] = {
                    "paired_same_label_mean": mean_or_none(paired),
                    "different_wsi_same_label_mean": mean_or_none(unpaired),
                    "different_wsi_minus_paired_mean": mean_difference(unpaired, paired),
                    "different_wsi_greater_than_paired_probability": greater_than_probability(unpaired, paired),
                    "paired_count": len(paired),
                    "different_wsi_count": len(unpaired),
                }
            backend_summary[str(int(label_id))] = metric_summary
        summary[backend] = backend_summary
    return summary


def build_verdict(summary: dict[str, Any]) -> dict[str, Any]:
    verdict: dict[str, Any] = {}
    comparisons = summary.get("comparisons", {})
    for backend, backend_summary in comparisons.items():
        verdict[backend] = {}
        for label_id, label_summary in backend_summary.items():
            primary = label_summary.get("region_loss_style_distance", {})
            prob = primary.get("different_wsi_greater_than_paired_probability")
            margin = primary.get("different_wsi_minus_paired_mean")
            if prob is None:
                reading = "insufficient paired or different-WSI rows"
            elif prob >= 0.65 and margin is not None and margin > 0.0:
                reading = (
                    "same-WSI same-label target/ref regions are closer than different-WSI "
                    "same-label refs; this is a plausible reference texture/style layer."
                )
            else:
                reading = (
                    "same-WSI consistency is weak; this layer may not be useful for "
                    "reference texture/style following even if it reacts to perturbations."
                )
            verdict[backend][label_id] = {
                "primary_metric": "region_loss_style_distance",
                "different_wsi_greater_than_paired_probability": prob,
                "different_wsi_minus_paired_mean": margin,
                "reading": reading,
            }
    return verdict


def descriptor_table(descriptors: dict[tuple[str, str, int], DescriptorItem]) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(descriptors.values()):
        rows.append(
            {
                "descriptor_index": index,
                "backend": item.backend,
                "role": item.image.role,
                "dataset": item.image.dataset,
                "sample_id": item.image.sample_id,
                "wsi_id": item.image.wsi_id,
                "image_path": str(item.image.image_path),
                "tissue_mask_path": str(item.image.tissue_mask_path),
                "label_id": item.label_id,
                "label_name": item.label_name,
                "token_count": item.token_count,
                "token_fraction": item.token_fraction,
                "mean_norm": float(torch.linalg.vector_norm(item.mean).item()),
                "std_norm": float(torch.linalg.vector_norm(item.std).item()),
                "concat_norm": float(torch.linalg.vector_norm(item.concat).item()),
            }
        )
    return rows


def infer_wsi_id(sample_id: str, image_path: Path | None = None) -> str:
    value = str(sample_id or (image_path.stem if image_path is not None else "")).strip()
    marker_index = value.find("_py")
    if marker_index > 0:
        return value[:marker_index]
    return value or (image_path.stem if image_path is not None else "")


def descriptor_key(role: str, image_path: Path, mask_path: Path) -> str:
    return f"{role}::{image_path}::{mask_path}"


def count_group(rows: list[dict[str, Any]], group: str) -> int:
    return sum(1 for row in rows if row["pair_group"] == group)


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(torch.tensor(values, dtype=torch.float32).mean().item())


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())

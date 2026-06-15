#!/usr/bin/env python3
"""Probe real-target -> CONCH vs reference -> CONCH region separability.

This mirrors probe_uni_target_reference_region_separability.py, but swaps the
frozen encoder to CONCH. It is meant to validate the scorer now used by the
Cross V1 CONCH reference-region loss:

    target/generated RGB -> optional VAE encode/decode -> frozen CONCH tokens -> region mean/std
    reference RGB        -> frozen CONCH tokens                         -> region mean/std

The key instance check is whether an intended paired same-label reference is
closer than unpaired same-label references from different WSIs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probe_uni_target_reference_region_separability import (  # noqa: E402
    DescriptorItem,
    build_coarse_lookup,
    build_comparison_summary,
    build_label_name_lookup,
    canonical_label_name,
    count_group,
    descriptor_distances,
    descriptor_key,
    descriptor_table,
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
    role_label_counts,
    summarize_pair_rows,
    vae_roundtrip_images,
    write_csv,
    write_json,
)


@dataclass(frozen=True)
class PairEntry:
    index: int
    dataset: str
    sample_id: str
    reference_sample_id: str
    target_case_id: str
    reference_case_id: str
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
    case_id: str
    wsi_id: str
    image_path: Path
    tissue_mask_path: Path


class ConchProbeEncoder:
    """Adapter so the UNI probe collector can call extract_uni_features()."""

    def __init__(self, encoder) -> None:
        self.encoder = encoder

    def extract_uni_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder.extract_features(images)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate target->CONCH vs reference->CONCH region descriptor separability."
    )
    parser.add_argument("--metadata", required=True, help="Cross metadata JSON/JSONL with pairs.")
    parser.add_argument("--conch-checkpoint-path", required=True, help="CONCH pytorch_model.bin path.")
    parser.add_argument("--conch-root", default=None, help="Local CONCH repo root.")
    parser.add_argument("--conch-model-cfg", default="conch_ViT-B-16")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--target-vae-roundtrip",
        action="store_true",
        help="Encode/decode target images through the FLUX VAE before extracting CONCH features.",
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
    parser.add_argument("--case-id-field", default="case_id")
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
    parser.add_argument(
        "--allow-same-wsi-unpaired",
        action="store_true",
        help="Allow unpaired same-label references from the target WSI. Default is different WSI only.",
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
    from controlnet_train.modules.conch_feature_encoder import ConchFeatureEncoder
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
    )
    rng = random.Random(args.seed)
    rng.shuffle(pair_entries)
    pair_entries = pair_entries[: int(args.candidate_pool_size)]
    image_entries = build_image_entries(pair_entries)

    device = resolve_device(args.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.torch_dtype]
    conch_encoder = ConchFeatureEncoder(
        args.conch_checkpoint_path,
        conch_root=args.conch_root,
        model_cfg=args.conch_model_cfg,
    )
    conch_encoder.to(device=device, dtype=dtype)
    conch_encoder.eval()
    conch_encoder.requires_grad_(False)
    encoder = ConchProbeEncoder(conch_encoder)

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
        require_different_wsi_unpaired=not bool(args.allow_same_wsi_unpaired),
        mean_weight=float(args.mean_weight),
        std_weight=float(args.std_weight),
        pooled_cosine_weight=float(args.pooled_cosine_weight),
    )

    descriptor_rows = descriptor_table(descriptors)
    write_csv(output_dir / "conch_target_reference_region_descriptors.csv", descriptor_rows)
    write_csv(output_dir / "conch_target_reference_region_pairs.csv", pair_rows)
    summary.update(
        {
            "encoder_backend": "conch",
            "conch_checkpoint_path": str(args.conch_checkpoint_path),
            "conch_root": str(args.conch_root) if args.conch_root else None,
            "conch_model_cfg": str(args.conch_model_cfg),
            "metadata": str(metadata_path),
            "target_input_mode": "vae_roundtrip" if args.target_vae_roundtrip else "real_rgb",
            "pretrained_model_name_or_path": str(args.pretrained_model_name_or_path)
            if args.target_vae_roundtrip
            else None,
            "unpaired_same_label_constraint": (
                "different_wsi" if not bool(args.allow_same_wsi_unpaired) else "any_wsi"
            ),
            "label_mode": label_mode,
            "labels": [
                {"id": int(label_id), "name": canonical_label_name(label_id, label_lookup, fallback=str(label_id))}
                for label_id in label_ids
            ],
            "candidate_pair_entries": len(pair_entries),
            "unique_image_entries": len(image_entries),
            "unique_wsi_ids": len({entry.wsi_id for entry in image_entries}),
            "descriptor_count": len(descriptors),
            "descriptor_role_label_counts": role_label_counts(descriptors),
            "skipped_count": len(skipped),
            "skipped_preview": skipped[:100],
            "outputs": {
                "descriptors_csv": "conch_target_reference_region_descriptors.csv",
                "pairs_csv": "conch_target_reference_region_pairs.csv",
                "summary_json": "conch_target_reference_region_separability_summary.json",
            },
        }
    )
    write_json(output_dir / "conch_target_reference_region_separability_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))
    return 0


def build_pair_entries(
    records: list[dict[str, Any]],
    *,
    base_dir: Path,
    case_id_field: str,
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
        dataset = str(record.get("dataset") or "unknown")
        sample_id = str(record.get("sample_id") or target_image.stem)
        reference_sample_id = str(record.get("reference_sample_id") or reference_image.stem)
        target_case_id = str(record.get(case_id_field) or infer_case_id(sample_id, target_image))
        reference_case_id = str(record.get("reference_case_id") or infer_case_id(reference_sample_id, reference_image))
        entries.append(
            PairEntry(
                index=index,
                dataset=dataset,
                sample_id=sample_id,
                reference_sample_id=reference_sample_id,
                target_case_id=target_case_id,
                reference_case_id=reference_case_id,
                target_wsi_id=f"{dataset}::{target_case_id}",
                reference_wsi_id=f"{dataset}::{reference_case_id}",
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
                case_id=pair.target_case_id,
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
                case_id=pair.reference_case_id,
                wsi_id=pair.reference_wsi_id,
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
            print(f"[conch-target-ref] scanned={scanned}/{len(image_entries)} {counts}", flush=True)
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
    require_different_wsi_unpaired: bool,
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
                candidates = []
                for item in ref_pool:
                    item_path_key = (item.image.image_path, item.image.tissue_mask_path)
                    if item_path_key in {target_path_key, paired_reference_path_key}:
                        continue
                    if require_different_wsi_unpaired and getattr(item.image, "wsi_id", None) == target_item.image.wsi_id:
                        continue
                    candidates.append(item)
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
            "unpaired_same_label_different_wsi_pairs": sum(
                1 for row in pair_rows if row["pair_group"] == "unpaired_same_label" and not row["same_wsi"]
            ),
        },
        "distance_stats": summarize_pair_rows(pair_rows),
    }
    summary["wsi_same_label_distance_stats"] = {
        metric: summarize_wsi_distances(pair_rows, metric)
        for metric in (
            "region_loss_style_distance",
            "concat_cosine_distance",
            "two_token_average_cosine_distance",
        )
    }
    summary["comparisons"] = build_comparison_summary(pair_rows)
    summary["target_reference_conch_verdict"] = build_verdict(summary)
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
    target_wsi_id = getattr(target_item.image, "wsi_id", infer_wsi_id(target_item.image.dataset, target_item.image.sample_id))
    reference_wsi_id = getattr(
        reference_item.image,
        "wsi_id",
        infer_wsi_id(reference_item.image.dataset, reference_item.image.sample_id),
    )
    return {
        "pair_group": pair_group,
        "metadata_index": pair.index,
        "dataset": pair.dataset,
        "target_sample_id": pair.sample_id,
        "paired_reference_sample_id": pair.reference_sample_id,
        "target_case_id": getattr(target_item.image, "case_id", pair.target_case_id),
        "reference_case_id": getattr(reference_item.image, "case_id", pair.reference_case_id),
        "target_wsi_id": target_wsi_id,
        "reference_wsi_id": reference_wsi_id,
        "same_wsi": target_wsi_id == reference_wsi_id,
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
    if unpaired_prob is None:
        verdict["reading"] = "insufficient different-WSI unpaired same-label rows"
    elif unpaired_prob >= 0.65 and (unpaired_margin is not None and unpaired_margin > 0.0):
        verdict["reading"] = (
            "CONCH region statistics preserve same-label cross-WSI instance signal: "
            "paired same-label refs are closer than unpaired same-label refs from other WSIs."
        )
    else:
        verdict["reading"] = (
            "CONCH same-label cross-WSI instance separability is weak in this probe; "
            "the scorer may still collapse toward tissue-type averages."
        )
    if cross_prob is not None:
        verdict["label_separability_note"] = (
            "paired cross-label is farther than paired same-label"
            if cross_prob >= 0.80 and (cross_margin is not None and cross_margin > 0.0)
            else "paired cross-label separation is weak; inspect labels/masks"
        )
    return verdict


def summarize_wsi_distances(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    same_values = [
        float(row[metric])
        for row in rows
        if row["target_label_id"] == row["reference_label_id"] and bool(row.get("same_wsi"))
    ]
    different_values = [
        float(row[metric])
        for row in rows
        if row["target_label_id"] == row["reference_label_id"] and not bool(row.get("same_wsi"))
    ]
    return {
        "same_wsi_same_label": describe_values(torch.tensor(same_values, dtype=torch.float32)),
        "different_wsi_same_label": describe_values(torch.tensor(different_values, dtype=torch.float32)),
        "different_minus_same_mean": mean_difference(different_values, same_values),
        "different_greater_than_same_probability": greater_than_probability(different_values, same_values),
    }


def infer_case_id(sample_id: str, image_path: Path | None = None) -> str:
    value = str(sample_id or (image_path.stem if image_path is not None else "")).strip()
    if not value and image_path is not None:
        value = image_path.stem
    marker_index = value.find("_py")
    if marker_index > 0:
        return value[:marker_index]
    return value or "unknown"


def infer_wsi_id(dataset: str, sample_id: str) -> str:
    return f"{dataset}::{infer_case_id(sample_id)}"


if __name__ == "__main__":
    raise SystemExit(main())

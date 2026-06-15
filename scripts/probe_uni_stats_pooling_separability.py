"""Probe whether UNI mean/std region pooling preserves reference-instance separation.

This diagnostic intentionally bypasses Cross V1 proj_mlp/IP modules:

    RGB reference crop -> frozen UNI patch tokens -> label-selected mean/std tokens

It measures whether the fixed pooling operation collapses different reference
instances of the same tissue label into a tight blob, and whether cross-label
descriptors remain farther apart.
"""

from __future__ import annotations

import argparse
import csv
import json
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
    token_count: int
    token_fraction: float
    mean: torch.Tensor
    std: torch.Tensor

    @property
    def concat(self) -> torch.Tensor:
        return torch.cat([self.mean, self.std], dim=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure separability of frozen UNI mean/std pooled region descriptors "
            "without using Cross V1 proj_mlp or IP-Adapter modules."
        )
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Raw Cross metadata json/jsonl with image and mask fields.",
    )
    parser.add_argument(
        "--selection-manifest",
        default=os.environ.get("PROBE_SELECTION_MANIFEST"),
        help="Optional frozen probe manifest from diagnose_cross_v1_generation_gate.py.",
    )
    parser.add_argument("--uni-checkpoint-path", required=True, help="UNI-2h pytorch_model.bin path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--metadata-base-dir",
        default=None,
        help="Base dir for relative metadata paths. Defaults to metadata parent.",
    )
    parser.add_argument("--image-field", default="reference_image")
    parser.add_argument("--mask-field", default="reference_tissue_mask")
    parser.add_argument("--sample-id-field", default="reference_sample_id")
    parser.add_argument(
        "--label-mode",
        choices=("coarse_tissue", "coarse", "fine", "tissue"),
        default="coarse_tissue",
        help="Pool labels on the UNI token grid after optional fine->coarse remap.",
    )
    parser.add_argument("--label-a", default="tumor", help="First label name or id.")
    parser.add_argument("--label-b", default="stroma", help="Second label name or id.")
    parser.add_argument("--samples-per-label", type=int, default=50)
    parser.add_argument("--candidate-pool-size", type=int, default=5000)
    parser.add_argument("--min-region-tokens", type=int, default=2)
    parser.add_argument("--min-region-fraction", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--allow-same-image-cross-pairs",
        action="store_true",
        help="Include A-vs-B pairs from the same image if one crop contains both labels.",
    )
    parser.add_argument(
        "--vae-roundtrip",
        action="store_true",
        help=(
            "Pass each image through VAE encode->decode before UNI to approximate "
            "the information loss of the loss-path (latent->decode->UNI). This is a "
            "LOWER-BOUND check: it captures VAE degradation but NOT model generation "
            "error, so passing here means VAE is not the bottleneck, while failing "
            "here rules out the UNI region-loss outright."
        ),
    )
    parser.add_argument(
        "--vae-checkpoint-path",
        default=None,
        help="Path/repo for the VAE (required when --vae-roundtrip is set).",
    )
    parser.add_argument(
        "--vae-subfolder",
        default="vae",
        help="Subfolder for from_pretrained when loading the VAE (default: vae).",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.samples_per_label <= 0:
        raise ValueError("--samples-per-label must be positive")
    if args.candidate_pool_size <= 0:
        raise ValueError("--candidate-pool-size must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.min_region_tokens <= 0:
        raise ValueError("--min-region-tokens must be positive")
    if not 0.0 <= args.min_region_fraction <= 1.0:
        raise ValueError("--min-region-fraction must be in [0, 1]")
    if args.vae_roundtrip and not args.vae_checkpoint_path:
        raise ValueError("--vae-roundtrip requires --vae-checkpoint-path")
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
    label_names = build_label_name_lookup(label_mode, COARSE_LABELS, FINE_LABELS)
    label_a = parse_label(args.label_a, label_names)
    label_b = parse_label(args.label_b, label_names)
    if label_a == label_b:
        raise ValueError("--label-a and --label-b must resolve to different labels")

    records = read_metadata(metadata_path)
    if looks_like_selection_manifest(records):
        raise ValueError(
            "`--metadata` looks like a selection_manifest.json, not raw cross metadata. "
            "Pass raw cross metadata via --metadata and the frozen probe list via --selection-manifest."
        )
    if args.selection_manifest:
        records = select_records_from_manifest(Path(args.selection_manifest), records)
    entries = build_entries(
        records,
        base_dir=base_dir,
        image_field=args.image_field,
        mask_field=args.mask_field,
        sample_id_field=args.sample_id_field,
    )
    rng = random.Random(args.seed)
    rng.shuffle(entries)
    entries = entries[: int(args.candidate_pool_size)]

    device = resolve_device(args.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.torch_dtype]
    encoder = ReferenceImageEncoder(args.uni_checkpoint_path, skip_perceiver=True)
    encoder.to(device=device, dtype=dtype)
    encoder.eval()

    vae = None
    if args.vae_roundtrip:
        from diffusers import AutoencoderKL

        vae = AutoencoderKL.from_pretrained(
            args.vae_checkpoint_path,
            subfolder=args.vae_subfolder,
        )
        vae.to(device=device, dtype=dtype)
        vae.eval()
        print(
            f"[uni-stats-pooling] VAE round-trip ENABLED from {args.vae_checkpoint_path} "
            f"(subfolder={args.vae_subfolder}); measuring decoded-image separability.",
            flush=True,
        )

    remap_lookup = build_coarse_lookup(FINE_TO_PARENT, device=torch.device("cpu"))
    items_by_label: dict[int, list[DescriptorItem]] = {label_a: [], label_b: []}
    skipped: list[dict[str, Any]] = []
    pending_entries: list[SampleEntry] = []
    pending_images: list[torch.Tensor] = []
    pending_masks: list[torch.Tensor] = []
    seen_for_label: dict[int, set[str]] = {label_a: set(), label_b: set()}
    scanned = 0

    def enough() -> bool:
        return all(len(items_by_label[label]) >= args.samples_per_label for label in (label_a, label_b))

    def flush_batch() -> None:
        nonlocal scanned
        if not pending_entries:
            return
        images = torch.stack(pending_images).to(device=device, dtype=dtype)
        masks = torch.stack(pending_masks)
        if vae is not None:
            images = vae_roundtrip(vae, images, dtype=dtype)
        with torch.no_grad():
            features = encoder.extract_uni_features(images).float().cpu()
        token_labels = resize_mask_to_token_labels(masks, encoder.num_spatial_tokens)
        if label_mode == "coarse_tissue":
            token_labels = remap_fine_to_coarse(token_labels, remap_lookup)

        for batch_index, entry in enumerate(pending_entries):
            for label_id, label_name in ((label_a, str(args.label_a)), (label_b, str(args.label_b))):
                image_key = str(entry.image_path)
                if image_key in seen_for_label[label_id]:
                    continue
                if len(items_by_label[label_id]) >= args.samples_per_label:
                    continue
                region = token_labels[batch_index] == int(label_id)
                token_count = int(region.sum().item())
                token_fraction = float(token_count / max(1, int(region.numel())))
                if token_count < args.min_region_tokens or token_fraction < args.min_region_fraction:
                    continue
                region_tokens = features[batch_index, region]
                mean = region_tokens.mean(dim=0)
                std = torch.sqrt(region_tokens.var(dim=0, unbiased=False) + 1e-6)
                items_by_label[label_id].append(
                    DescriptorItem(
                        label_name=canonical_label_name(label_id, label_names, fallback=label_name),
                        label_id=int(label_id),
                        sample=entry,
                        token_count=token_count,
                        token_fraction=token_fraction,
                        mean=mean,
                        std=std,
                    )
                )
                seen_for_label[label_id].add(image_key)
        scanned += len(pending_entries)
        if args.progress_every > 0 and scanned % int(args.progress_every) < len(pending_entries):
            print(
                f"[uni-stats-pooling] scanned={scanned}/{len(entries)} "
                f"{canonical_label_name(label_a, label_names, fallback=str(label_a))}={len(items_by_label[label_a])} "
                f"{canonical_label_name(label_b, label_names, fallback=str(label_b))}={len(items_by_label[label_b])}",
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
                    "reason": f"load_failed:{type(exc).__name__}",
                    "detail": str(exc),
                }
            )
            continue
        pending_entries.append(entry)
        pending_images.append(image)
        pending_masks.append(mask)
        if len(pending_entries) >= args.batch_size:
            flush_batch()
    flush_batch()

    if len(items_by_label[label_a]) < 2 or len(items_by_label[label_b]) < 2:
        summary = {
            "status": "insufficient_valid_descriptors",
            "label_mode": label_mode,
            "label_a": label_a,
            "label_b": label_b,
            "counts": {str(label): len(items_by_label[label]) for label in (label_a, label_b)},
            "candidate_entries": len(entries),
            "skipped_preview": skipped[:100],
        }
        write_json(output_dir / "uni_stats_pooling_separability_summary.json", summary)
        raise RuntimeError("Not enough descriptors; see summary json")

    descriptor_rows = descriptor_table(items_by_label[label_a] + items_by_label[label_b])
    write_csv(output_dir / "uni_stats_pooling_descriptors.csv", descriptor_rows)

    pair_rows, summary = build_pair_outputs(
        items_a=items_by_label[label_a],
        items_b=items_by_label[label_b],
        allow_same_image_cross_pairs=bool(args.allow_same_image_cross_pairs),
    )
    write_csv(output_dir / "uni_stats_pooling_pairs.csv", pair_rows)

    summary.update(
        {
            "metadata": str(metadata_path),
            "image_field": args.image_field,
            "mask_field": args.mask_field,
            "label_mode": label_mode,
            "vae_roundtrip": bool(args.vae_roundtrip),
            "vae_checkpoint_path": args.vae_checkpoint_path if args.vae_roundtrip else None,
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
            "scanned_entries": scanned,
            "skipped_count": len(skipped),
            "skipped_preview": skipped[:100],
            "outputs": {
                "descriptors_csv": "uni_stats_pooling_descriptors.csv",
                "pairs_csv": "uni_stats_pooling_pairs.csv",
                "summary_json": "uni_stats_pooling_separability_summary.json",
            },
        }
    )
    write_json(output_dir / "uni_stats_pooling_separability_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def looks_like_selection_manifest(records: list[dict[str, Any]]) -> bool:
    if not records:
        return False
    has_manifest_keys = any("paired_reference_image" in row for row in records)
    has_raw_reference_masks = any("reference_tissue_mask" in row for row in records)
    return has_manifest_keys and not has_raw_reference_masks


def select_records_from_manifest(
    manifest_path: Path,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    if not isinstance(manifest, list):
        raise ValueError(f"selection manifest must be a list, got {type(manifest)!r}")

    by_target_ref = {
        (path_key(row.get("target_image")), path_key(row.get("reference_image"))): row
        for row in records
        if row.get("target_image") and row.get("reference_image")
    }
    by_sample_ref = {
        (str(row.get("sample_id") or ""), str(row.get("reference_sample_id") or "")): row
        for row in records
        if row.get("target_image") and row.get("reference_image")
    }
    by_target = {}
    for row in records:
        target_key = path_key(row.get("target_image"))
        if target_key:
            by_target.setdefault(target_key, []).append(row)

    selected: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for item in manifest:
        target_key = path_key(item.get("target_image"))
        paired_ref_key = path_key(item.get("paired_reference_image"))
        record = by_target_ref.get((target_key, paired_ref_key))
        if record is None:
            sample_key = (
                str(item.get("sample_id") or ""),
                str(item.get("paired_reference_sample_id") or ""),
            )
            record = by_sample_ref.get(sample_key)
        if record is None:
            candidates = by_target.get(target_key) or []
            if len(candidates) == 1:
                record = candidates[0]
        if record is None:
            missing.append(
                {
                    "target_image": item.get("target_image"),
                    "paired_reference_image": item.get("paired_reference_image"),
                    "sample_id": item.get("sample_id"),
                    "paired_reference_sample_id": item.get("paired_reference_sample_id"),
                }
            )
            continue
        selected.append(dict(record))

    if missing:
        raise ValueError(
            f"selection manifest could not be matched to raw metadata for {len(missing)} rows; "
            f"example={missing[0]}"
        )
    return selected


def path_key(value: Any) -> str:
    if value is None:
        return ""
    return str(Path(str(value).replace("\\", "/")).expanduser())


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


def read_metadata(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    payload = json.loads(path.read_text(encoding="utf8"))
    if isinstance(payload, dict):
        records = payload.get("pairs") or payload.get("records")
        if not isinstance(records, list):
            raise ValueError("JSON metadata dict must contain a 'pairs' or 'records' list")
        return records
    if isinstance(payload, list):
        return payload
    raise TypeError(f"unsupported metadata payload type: {type(payload)!r}")


def build_entries(
    records: list[dict[str, Any]],
    *,
    base_dir: Path,
    image_field: str,
    mask_field: str,
    sample_id_field: str,
) -> list[SampleEntry]:
    entries: list[SampleEntry] = []
    seen: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        if image_field not in record or mask_field not in record:
            continue
        image_path = resolve_metadata_path(record[image_field], base_dir)
        mask_path = resolve_metadata_path(record[mask_field], base_dir)
        key = (str(image_path), str(mask_path))
        if key in seen:
            continue
        seen.add(key)
        entries.append(
            SampleEntry(
                index=index,
                dataset=str(record.get("dataset") or "unknown"),
                sample_id=str(record.get(sample_id_field) or record.get("sample_id") or image_path.stem),
                image_path=image_path,
                tissue_mask_path=mask_path,
            )
        )
    return entries


def resolve_metadata_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(str(path_value).replace("\\", "/"))
    if path.is_absolute():
        return path
    return base_dir / path


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


@torch.no_grad()
def vae_roundtrip(vae: Any, images: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    """Encode then decode images through the VAE to mimic the loss-path degradation.

    Images come in as [0,1] RGB (B,3,H,W). The VAE expects [-1,1]. We take the
    posterior mode (deterministic) and decode immediately. Since the latent never
    leaves the VAE here, no scaling_factor juggling is needed. Output is mapped
    back to [0,1] and clamped, matching the training decode convention.

    This reproduces VAE information loss but NOT model generation error, so it is
    a LOWER-BOUND check: passing means VAE is not the bottleneck; failing rules
    out the UNI region-loss outright.
    """
    x = images.to(dtype=dtype) * 2.0 - 1.0
    latents = vae.encode(x).latent_dist.mode()
    decoded = vae.decode(latents.to(dtype=dtype), return_dict=False)[0]
    decoded = (decoded.float() / 2.0) + 0.5
    return decoded.clamp(0.0, 1.0).to(dtype=dtype)


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
                "token_count": item.token_count,
                "token_fraction": item.token_fraction,
                "mean_norm": float(torch.linalg.vector_norm(item.mean).item()),
                "std_norm": float(torch.linalg.vector_norm(item.std).item()),
                "concat_norm": float(torch.linalg.vector_norm(item.concat).item()),
            }
        )
    return rows


def build_pair_outputs(
    *,
    items_a: list[DescriptorItem],
    items_b: list[DescriptorItem],
    allow_same_image_cross_pairs: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pair_rows: list[dict[str, Any]] = []
    within_a = pairwise_rows(items_a, "within_a")
    within_b = pairwise_rows(items_b, "within_b")
    cross = cross_pair_rows(
        items_a,
        items_b,
        allow_same_image_cross_pairs=allow_same_image_cross_pairs,
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
    for metric in ("mean_distance", "std_distance", "concat_distance", "two_token_average_distance"):
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
    return pair_rows, summary


def pairwise_rows(items: list[DescriptorItem], group: str) -> list[dict[str, Any]]:
    rows = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            rows.append(format_pair(items[i], items[j], pair_group=group))
    return rows


def cross_pair_rows(
    items_a: list[DescriptorItem],
    items_b: list[DescriptorItem],
    *,
    allow_same_image_cross_pairs: bool,
) -> list[dict[str, Any]]:
    rows = []
    for item_a in items_a:
        for item_b in items_b:
            if not allow_same_image_cross_pairs and item_a.sample.image_path == item_b.sample.image_path:
                continue
            rows.append(format_pair(item_a, item_b, pair_group="cross"))
    return rows


def format_pair(item_i: DescriptorItem, item_j: DescriptorItem, *, pair_group: str) -> dict[str, Any]:
    mean_distance = cosine_distance(item_i.mean, item_j.mean)
    std_distance = cosine_distance(item_i.std, item_j.std)
    concat_distance = cosine_distance(item_i.concat, item_j.concat)
    return {
        "pair_group": pair_group,
        "label_i": item_i.label_name,
        "label_j": item_j.label_name,
        "sample_i": item_i.sample.sample_id,
        "sample_j": item_j.sample.sample_id,
        "image_i": str(item_i.sample.image_path),
        "image_j": str(item_j.sample.image_path),
        "same_image": item_i.sample.image_path == item_j.sample.image_path,
        "mean_distance": mean_distance,
        "std_distance": std_distance,
        "concat_distance": concat_distance,
        "two_token_average_distance": float((mean_distance + std_distance) * 0.5),
        "tokens_i": item_i.token_count,
        "tokens_j": item_j.token_count,
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
    concat_stack = torch.stack([item.concat for item in items]).float()
    return {
        "count": len(items),
        "mean_token_centered_l2": describe_values(torch.linalg.vector_norm(mean_stack - mean_stack.mean(dim=0), dim=1)),
        "std_token_centered_l2": describe_values(torch.linalg.vector_norm(std_stack - std_stack.mean(dim=0), dim=1)),
        "concat_centered_l2": describe_values(torch.linalg.vector_norm(concat_stack - concat_stack.mean(dim=0), dim=1)),
    }


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")


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

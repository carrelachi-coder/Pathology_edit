"""Analyze UNI2-h tumor feature similarity across WSIs.

This script uses the current ReferenceImageEncoder UNI preprocessing and
extracts final UNI patch tokens. Tumor tokens are selected on the 16x16 UNI
token grid from a tissue mask, pooled per image, then compared across WSI/case
groups.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
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
    case_id: str
    wsi_id: str
    image_path: Path
    tissue_mask_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure current UNI2-h tumor feature similarity across WSIs."
    )
    parser.add_argument("--metadata", required=True, help="Cross JSON or dataset JSONL metadata.")
    parser.add_argument("--uni-checkpoint", required=True, help="UNI-2h pytorch_model.bin path.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Optional Cross V1 checkpoint dir containing phase5_conditioning.pt and "
            "phase5_ip_adapter.pt. Required for --feature-stage projected/encoder_hid_proj."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--metadata-base-dir",
        default=None,
        help="Base directory for relative metadata paths. Defaults to metadata parent.",
    )
    parser.add_argument(
        "--image-field",
        default="target_image",
        help="Image path field. Use reference_image for reference-side analysis.",
    )
    parser.add_argument(
        "--mask-field",
        default="target_tissue_mask",
        help="Tissue mask path field. Use reference_tissue_mask for reference-side analysis.",
    )
    parser.add_argument(
        "--sample-id-field",
        default="sample_id",
        help="Sample id field. Use reference_sample_id with --image-field reference_image.",
    )
    parser.add_argument("--case-id-field", default="case_id")
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Fallback dataset name when metadata rows do not contain a dataset field.",
    )
    parser.add_argument(
        "--tumor-label",
        type=int,
        action="append",
        default=None,
        help="Tissue id treated as tumor. May be repeated. Defaults to 1.",
    )
    parser.add_argument("--min-tumor-tokens", type=int, default=1)
    parser.add_argument("--min-tumor-fraction", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument(
        "--samples-per-wsi",
        type=int,
        default=0,
        help="Limit patches per WSI before global max-samples. 0 means no per-WSI cap.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument(
        "--feature-stage",
        choices=["uni", "projected", "encoder_hid_proj"],
        default="uni",
        help="Feature stage to pool over tumor tokens.",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--max-pair-rows",
        type=int,
        default=200000,
        help="Maximum rows written to tumor_similarity_pairs.csv.",
    )
    parser.add_argument(
        "--full-matrix-max-samples",
        type=int,
        default=5000,
        help="Use full sample cosine matrix up to this N; above it, sample pair stats.",
    )
    parser.add_argument("--nearest-k", type=int, default=10)
    parser.add_argument("--nearest-chunk-size", type=int, default=512)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.tumor_label is None:
        args.tumor_label = [1]
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.min_tumor_tokens < 1:
        raise ValueError("--min-tumor-tokens must be >= 1")
    if args.min_tumor_fraction < 0.0 or args.min_tumor_fraction > 1.0:
        raise ValueError("--min-tumor-fraction must be within [0, 1]")
    if args.feature_stage != "uni" and not args.checkpoint:
        raise ValueError("--checkpoint is required for --feature-stage projected/encoder_hid_proj")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from controlnet_train.data.common import (
        load_image_tensor,
        load_tissue_mask,
        parse_sample_identity,
    )
    from controlnet_train.modules.reference_image_encoder import (
        ReferenceImageEncoder,
        resize_mask_to_token_labels,
    )

    metadata_path = Path(args.metadata)
    base_dir = Path(args.metadata_base_dir) if args.metadata_base_dir else metadata_path.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    records = read_metadata(metadata_path)
    entries = build_entries(
        records,
        base_dir=base_dir,
        image_field=args.image_field,
        mask_field=args.mask_field,
        sample_id_field=args.sample_id_field,
        case_id_field=args.case_id_field,
        dataset_name=args.dataset_name,
        parse_sample_identity=parse_sample_identity,
    )
    selected_entries = select_entries(
        entries,
        rng=rng,
        max_samples=args.max_samples,
        samples_per_wsi=args.samples_per_wsi,
    )

    device = resolve_device(args.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.torch_dtype]
    encoder_hid_proj = None
    if args.checkpoint:
        from controlnet_train.cli.diagnose_ref_signal import (
            load_encoder_hid_proj_from_checkpoint,
            load_ref_encoder_from_checkpoint,
        )

        encoder, encoder_config = load_ref_encoder_from_checkpoint(
            args.checkpoint,
            args.uni_checkpoint,
            str(device),
            dtype,
            skip_perceiver=True,
        )
        if args.feature_stage == "encoder_hid_proj":
            encoder_hid_proj = load_encoder_hid_proj_from_checkpoint(
                args.checkpoint,
                hidden_dim=int(encoder_config["hidden_dim"]),
                device=str(device),
                dtype=dtype,
            )
            if encoder_hid_proj is None:
                raise FileNotFoundError(
                    f"{args.checkpoint} does not contain phase5_ip_adapter.pt/encoder_hid_proj"
                )
    else:
        encoder = ReferenceImageEncoder(args.uni_checkpoint, skip_perceiver=True)
        encoder.to(device=device, dtype=dtype)
        encoder.eval()

    embeddings: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    pending_entries: list[SampleEntry] = []
    pending_images: list[torch.Tensor] = []
    pending_masks: list[torch.Tensor] = []
    encoded_count = 0
    next_progress = max(1, int(args.progress_every or 0))

    def flush_batch() -> None:
        nonlocal encoded_count, next_progress
        if not pending_entries:
            return
        images = torch.stack(pending_images).to(device=device)
        masks = torch.stack(pending_masks)
        with torch.no_grad():
            features = encode_feature_tokens(
                encoder,
                encoder_hid_proj,
                images,
                feature_stage=args.feature_stage,
            ).float().cpu()
        labels = resize_mask_to_token_labels(masks, encoder.num_spatial_tokens)
        tumor_mask = build_label_mask(labels, args.tumor_label)
        for local_idx, entry in enumerate(pending_entries):
            sample_tumor = tumor_mask[local_idx]
            tumor_count = int(sample_tumor.sum().item())
            tumor_fraction = float(tumor_count / max(1, int(sample_tumor.numel())))
            if tumor_count < args.min_tumor_tokens or tumor_fraction < args.min_tumor_fraction:
                skipped.append(
                    {
                        "index": entry.index,
                        "sample_id": entry.sample_id,
                        "wsi_id": entry.wsi_id,
                        "reason": "insufficient_tumor_tokens",
                        "tumor_token_count": tumor_count,
                        "tumor_fraction": tumor_fraction,
                    }
                )
                continue
            pooled = features[local_idx, sample_tumor].mean(dim=0)
            embeddings.append(pooled.cpu())
            rows.append(
                {
                    "embedding_index": len(rows),
                    "source_index": entry.index,
                    "dataset": entry.dataset,
                    "case_id": entry.case_id,
                    "wsi_id": entry.wsi_id,
                    "sample_id": entry.sample_id,
                    "image_path": str(entry.image_path),
                    "tissue_mask_path": str(entry.tissue_mask_path),
                    "tumor_token_count": tumor_count,
                    "tumor_fraction": tumor_fraction,
                }
            )
        encoded_count += len(pending_entries)
        if args.progress_every > 0 and encoded_count >= next_progress:
            print(
                f"[uni-sim] encoded={encoded_count}/{len(selected_entries)} "
                f"valid={len(rows)} skipped={len(skipped)} stage={args.feature_stage}",
                flush=True,
            )
            while next_progress <= encoded_count:
                next_progress += int(args.progress_every)
        pending_entries.clear()
        pending_images.clear()
        pending_masks.clear()

    for entry in selected_entries:
        try:
            image = load_image_tensor(entry.image_path)
            tissue_mask = load_tissue_mask(entry.tissue_mask_path)
        except Exception as exc:  # noqa: BLE001 - diagnostic script should keep going.
            skipped.append(
                {
                    "index": entry.index,
                    "sample_id": entry.sample_id,
                    "wsi_id": entry.wsi_id,
                    "reason": f"load_failed:{type(exc).__name__}",
                    "detail": str(exc),
                }
            )
            continue
        pending_entries.append(entry)
        pending_images.append(image)
        pending_masks.append(tissue_mask)
        if len(pending_entries) >= args.batch_size:
            flush_batch()
    flush_batch()

    if not embeddings:
        summary = {
            "status": "no_valid_embeddings",
            "metadata": str(metadata_path),
            "selected_entries": len(selected_entries),
            "skipped_count": len(skipped),
            "skipped": skipped[:100],
        }
        write_json(output_dir / "tumor_similarity_summary.json", summary)
        raise RuntimeError("No valid tumor embeddings were produced; see tumor_similarity_summary.json")

    raw_embedding_tensor = torch.stack(embeddings).float()
    global_feature_mean = raw_embedding_tensor.mean(dim=0, keepdim=True)
    raw_norm = torch.linalg.vector_norm(raw_embedding_tensor, dim=1)
    centered_l2 = torch.linalg.vector_norm(raw_embedding_tensor - global_feature_mean, dim=1)
    for row, norm_value, centered_value in zip(rows, raw_norm.tolist(), centered_l2.tolist()):
        row["raw_feature_norm"] = float(norm_value)
        row["raw_centered_l2"] = float(centered_value)
    embedding_tensor = F.normalize(raw_embedding_tensor, dim=1)
    embedding_pt_name = f"tumor_{args.feature_stage}_embeddings.pt"
    embedding_csv_name = f"tumor_{args.feature_stage}_embeddings.csv"
    torch.save(
        {
            "embeddings": embedding_tensor,
            "raw_embeddings": raw_embedding_tensor,
            "global_feature_mean": global_feature_mean.squeeze(0),
            "rows": rows,
            "tumor_labels": args.tumor_label,
            "feature_stage": args.feature_stage,
            "feature_source": feature_stage_description(args.feature_stage),
        },
        output_dir / embedding_pt_name,
    )
    write_csv(output_dir / embedding_csv_name, rows)

    summary = build_similarity_outputs(
        embeddings=embedding_tensor,
        rows=rows,
        output_dir=output_dir,
        rng=rng,
        max_pair_rows=args.max_pair_rows,
        full_matrix_max_samples=args.full_matrix_max_samples,
        nearest_k=args.nearest_k,
        nearest_chunk_size=args.nearest_chunk_size,
    )
    summary["raw_feature_norm"] = describe_values(raw_norm)
    summary["raw_centered_l2"] = describe_values(centered_l2)
    summary.update(
        {
            "metadata": str(metadata_path),
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "feature_stage": args.feature_stage,
            "feature_source": feature_stage_description(args.feature_stage),
            "image_field": args.image_field,
            "mask_field": args.mask_field,
            "tumor_labels": args.tumor_label,
            "raw_records": len(records),
            "candidate_entries_after_dedup": len(entries),
            "selected_entries": len(selected_entries),
            "valid_embeddings": len(rows),
            "skipped_count": len(skipped),
            "skipped_preview": skipped[:100],
        }
    )
    summary["outputs"]["embeddings_pt"] = embedding_pt_name
    summary["outputs"]["embeddings_csv"] = embedding_csv_name
    write_json(output_dir / "tumor_similarity_summary.json", summary)
    write_json(output_dir / "skipped_samples.json", {"skipped": skipped})

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


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
        records = payload.get("pairs")
        if records is None:
            records = payload.get("records")
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
    case_id_field: str,
    dataset_name: str | None,
    parse_sample_identity,
) -> list[SampleEntry]:
    entries: list[SampleEntry] = []
    seen_paths: set[str] = set()
    for index, record in enumerate(records):
        image_key = image_field if image_field in record else "image"
        if image_key not in record:
            continue
        image_path = resolve_metadata_path(record[image_key], base_dir)
        mask_path = resolve_mask_path(record, mask_field, image_path, base_dir)
        sample_id = str(record.get(sample_id_field) or record.get("sample_id") or image_path.stem)
        case_id = str(record.get(case_id_field) or parse_sample_identity(sample_id)[0])
        dataset = str(record.get("dataset") or dataset_name or base_dir.name or "unknown")
        wsi_id = f"{dataset}::{case_id}"
        dedup_key = str(image_path)
        if dedup_key in seen_paths:
            continue
        seen_paths.add(dedup_key)
        entries.append(
            SampleEntry(
                index=index,
                dataset=dataset,
                sample_id=sample_id,
                case_id=case_id,
                wsi_id=wsi_id,
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


def resolve_mask_path(record: dict[str, Any], mask_field: str, image_path: Path, base_dir: Path) -> Path:
    if mask_field in record:
        return resolve_metadata_path(record[mask_field], base_dir)
    for fallback in ("tissue_mask", "tissue_mask_path"):
        if fallback in record:
            return resolve_metadata_path(record[fallback], base_dir)
    if image_path.parent.name == "images":
        return image_path.parent.parent / "tissue_masks" / f"{image_path.stem}.png"
    return image_path.parent / "tissue_masks" / f"{image_path.stem}.png"


def select_entries(
    entries: list[SampleEntry],
    *,
    rng: random.Random,
    max_samples: int,
    samples_per_wsi: int,
) -> list[SampleEntry]:
    selected = list(entries)
    if samples_per_wsi and samples_per_wsi > 0:
        grouped: dict[str, list[SampleEntry]] = defaultdict(list)
        for entry in selected:
            grouped[entry.wsi_id].append(entry)
        selected = []
        for group_entries in grouped.values():
            group_entries = list(group_entries)
            rng.shuffle(group_entries)
            selected.extend(group_entries[:samples_per_wsi])
    rng.shuffle(selected)
    if max_samples and max_samples > 0:
        selected = selected[:max_samples]
    return selected


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def build_label_mask(labels: torch.Tensor, allowed_labels: list[int]) -> torch.Tensor:
    allowed = torch.tensor(allowed_labels, dtype=labels.dtype, device=labels.device)
    return (labels.unsqueeze(-1) == allowed.view(1, 1, -1)).any(dim=-1)


def encode_feature_tokens(
    encoder,
    encoder_hid_proj,
    images: torch.Tensor,
    *,
    feature_stage: str,
) -> torch.Tensor:
    if feature_stage == "uni":
        return encoder.extract_uni_features(images)
    projected = encoder.encode_projected_patch_tokens(images)
    if feature_stage == "projected":
        return projected
    if feature_stage == "encoder_hid_proj":
        if encoder_hid_proj is None:
            raise ValueError("encoder_hid_proj is required for feature_stage='encoder_hid_proj'")
        gate = encoder.reference_presence_gate(
            images,
            device=projected.device,
            dtype=projected.dtype,
        )
        projected = projected * gate
        return encoder_hid_proj([projected])[0] * gate.to(
            device=projected.device,
            dtype=projected.dtype,
        )
    raise ValueError(f"unsupported feature_stage: {feature_stage!r}")


def feature_stage_description(feature_stage: str) -> str:
    if feature_stage == "uni":
        return "ReferenceImageEncoder.extract_uni_features final UNI patch tokens"
    if feature_stage == "projected":
        return "ReferenceImageEncoder.encode_projected_patch_tokens after checkpoint proj_mlp"
    if feature_stage == "encoder_hid_proj":
        return "checkpoint encoder_hid_proj(projected spatial tokens) before IP K/V"
    return str(feature_stage)


def build_similarity_outputs(
    *,
    embeddings: torch.Tensor,
    rows: list[dict[str, Any]],
    output_dir: Path,
    rng: random.Random,
    max_pair_rows: int,
    full_matrix_max_samples: int,
    nearest_k: int,
    nearest_chunk_size: int,
) -> dict[str, Any]:
    n = int(embeddings.shape[0])
    wsi_ids = [str(row["wsi_id"]) for row in rows]
    datasets = [str(row["dataset"]) for row in rows]
    sample_ids = [str(row["sample_id"]) for row in rows]
    sample_summary: dict[str, Any]

    if n <= full_matrix_max_samples:
        sim = embeddings @ embeddings.T
        sample_summary = summarize_full_matrix(sim, wsi_ids, datasets)
        pair_rows = pair_rows_from_matrix(
            sim=sim,
            rows=rows,
            max_pair_rows=max_pair_rows,
            rng=rng,
        )
    else:
        sample_summary, pair_rows = summarize_sampled_pairs(
            embeddings=embeddings,
            rows=rows,
            max_pair_rows=max_pair_rows,
            rng=rng,
        )
    write_csv(output_dir / "tumor_similarity_pairs.csv", pair_rows)

    centroid_embeddings, centroid_rows = build_wsi_centroids(embeddings, rows)
    centroid_summary: dict[str, Any]
    if len(centroid_rows) >= 2:
        centroid_sim = centroid_embeddings @ centroid_embeddings.T
        centroid_summary = summarize_full_matrix(
            centroid_sim,
            [str(row["wsi_id"]) for row in centroid_rows],
            [str(row["dataset"]) for row in centroid_rows],
        )
        write_csv(
            output_dir / "tumor_wsi_centroid_similarity.csv",
            pair_rows_from_matrix(
                sim=centroid_sim,
                rows=centroid_rows,
                max_pair_rows=max_pair_rows,
                rng=rng,
            ),
        )
    else:
        centroid_summary = {"status": "need_at_least_two_wsi_centroids"}
        write_csv(output_dir / "tumor_wsi_centroid_similarity.csv", [])

    nn_rows = nearest_neighbor_rows(
        embeddings=embeddings,
        wsi_ids=wsi_ids,
        sample_ids=sample_ids,
        rows=rows,
        k=nearest_k,
        chunk_size=nearest_chunk_size,
    )
    write_csv(output_dir / "tumor_nearest_neighbors.csv", nn_rows)

    return {
        "num_embeddings": n,
        "num_wsi": len(set(wsi_ids)),
        "num_datasets": len(set(datasets)),
        "sample_level": sample_summary,
        "wsi_centroid_level": {
            **centroid_summary,
            "num_centroids": len(centroid_rows),
        },
        "outputs": {
            "embeddings_pt": "tumor_uni_embeddings.pt",
            "embeddings_csv": "tumor_uni_embeddings.csv",
            "sample_pairs_csv": "tumor_similarity_pairs.csv",
            "wsi_centroid_pairs_csv": "tumor_wsi_centroid_similarity.csv",
            "nearest_neighbors_csv": "tumor_nearest_neighbors.csv",
            "skipped_json": "skipped_samples.json",
        },
    }


def summarize_full_matrix(
    sim: torch.Tensor,
    wsi_ids: list[str],
    datasets: list[str],
) -> dict[str, Any]:
    n = int(sim.shape[0])
    if n < 2:
        return {"status": "need_at_least_two_embeddings"}
    tri_i, tri_j = torch.triu_indices(n, n, offset=1)
    values = sim[tri_i, tri_j].float().cpu()
    wsi_group = encode_groups(wsi_ids)
    dataset_group = encode_groups(datasets)
    same_wsi = wsi_group[tri_i] == wsi_group[tri_j]
    same_dataset = dataset_group[tri_i] == dataset_group[tri_j]
    return {
        "all_pairs": describe_values(values),
        "same_wsi": describe_values(values[same_wsi]),
        "different_wsi": describe_values(values[~same_wsi]),
        "same_dataset": describe_values(values[same_dataset]),
        "different_dataset": describe_values(values[~same_dataset]),
    }


def encode_groups(values: list[str]) -> torch.Tensor:
    lookup: dict[str, int] = {}
    encoded = []
    for value in values:
        if value not in lookup:
            lookup[value] = len(lookup)
        encoded.append(lookup[value])
    return torch.tensor(encoded, dtype=torch.long)


def summarize_sampled_pairs(
    *,
    embeddings: torch.Tensor,
    rows: list[dict[str, Any]],
    max_pair_rows: int,
    rng: random.Random,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n = int(embeddings.shape[0])
    pairs = random_pair_indices(n, max_pair_rows, rng)
    pair_rows = []
    values = []
    same_wsi_flags = []
    same_dataset_flags = []
    for i, j in pairs:
        cosine = float(torch.dot(embeddings[i], embeddings[j]).item())
        same_wsi = rows[i]["wsi_id"] == rows[j]["wsi_id"]
        same_dataset = rows[i]["dataset"] == rows[j]["dataset"]
        values.append(cosine)
        same_wsi_flags.append(same_wsi)
        same_dataset_flags.append(same_dataset)
        pair_rows.append(format_pair_row(rows, i, j, cosine))
    value_tensor = torch.tensor(values, dtype=torch.float32)
    same_wsi_tensor = torch.tensor(same_wsi_flags, dtype=torch.bool)
    same_dataset_tensor = torch.tensor(same_dataset_flags, dtype=torch.bool)
    summary = {
        "sampled_pairs": len(pair_rows),
        "all_pairs_sample": describe_values(value_tensor),
        "same_wsi_sample": describe_values(value_tensor[same_wsi_tensor]),
        "different_wsi_sample": describe_values(value_tensor[~same_wsi_tensor]),
        "same_dataset_sample": describe_values(value_tensor[same_dataset_tensor]),
        "different_dataset_sample": describe_values(value_tensor[~same_dataset_tensor]),
    }
    return summary, pair_rows


def describe_values(values: torch.Tensor) -> dict[str, Any]:
    values = values.float().flatten()
    if values.numel() == 0:
        return {"count": 0}
    quantiles = torch.quantile(
        values,
        torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], dtype=values.dtype),
    )
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "q01": float(quantiles[0].item()),
        "q05": float(quantiles[1].item()),
        "q25": float(quantiles[2].item()),
        "median": float(quantiles[3].item()),
        "q75": float(quantiles[4].item()),
        "q95": float(quantiles[5].item()),
        "q99": float(quantiles[6].item()),
        "max": float(values.max().item()),
    }


def pair_rows_from_matrix(
    *,
    sim: torch.Tensor,
    rows: list[dict[str, Any]],
    max_pair_rows: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    n = int(sim.shape[0])
    all_pairs_count = n * (n - 1) // 2
    if max_pair_rows <= 0:
        selected_pairs: list[tuple[int, int]] = []
    elif all_pairs_count <= max_pair_rows:
        selected_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        selected_pairs = random_pair_indices(n, max_pair_rows, rng)
    return [
        format_pair_row(rows, i, j, float(sim[i, j].item()))
        for i, j in selected_pairs
    ]


def random_pair_indices(n: int, count: int, rng: random.Random) -> list[tuple[int, int]]:
    if n < 2 or count <= 0:
        return []
    total = n * (n - 1) // 2
    count = min(count, total)
    if count == total:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < count:
        i = rng.randrange(n)
        j = rng.randrange(n - 1)
        if j >= i:
            j += 1
        if i > j:
            i, j = j, i
        pairs.add((i, j))
    return sorted(pairs)


def format_pair_row(rows: list[dict[str, Any]], i: int, j: int, cosine: float) -> dict[str, Any]:
    row_i = rows[i]
    row_j = rows[j]
    return {
        "i": i,
        "j": j,
        "sample_i": row_i["sample_id"],
        "sample_j": row_j["sample_id"],
        "wsi_i": row_i["wsi_id"],
        "wsi_j": row_j["wsi_id"],
        "dataset_i": row_i["dataset"],
        "dataset_j": row_j["dataset"],
        "same_wsi": row_i["wsi_id"] == row_j["wsi_id"],
        "same_dataset": row_i["dataset"] == row_j["dataset"],
        "cosine": cosine,
    }


def build_wsi_centroids(
    embeddings: torch.Tensor,
    rows: list[dict[str, Any]],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[str(row["wsi_id"])].append(idx)
    centroid_rows = []
    centroids = []
    for wsi_id in sorted(groups):
        indices = groups[wsi_id]
        centroid = embeddings[indices].mean(dim=0)
        centroid = F.normalize(centroid, dim=0)
        first = rows[indices[0]]
        centroids.append(centroid)
        centroid_rows.append(
            {
                "sample_id": f"{wsi_id}__centroid",
                "dataset": first["dataset"],
                "case_id": first["case_id"],
                "wsi_id": wsi_id,
                "tumor_patch_count": len(indices),
                "mean_tumor_token_count": float(
                    sum(float(rows[i]["tumor_token_count"]) for i in indices) / len(indices)
                ),
            }
        )
    if not centroids:
        return torch.empty(0, embeddings.shape[1]), []
    return torch.stack(centroids), centroid_rows


def nearest_neighbor_rows(
    *,
    embeddings: torch.Tensor,
    wsi_ids: list[str],
    sample_ids: list[str],
    rows: list[dict[str, Any]],
    k: int,
    chunk_size: int,
) -> list[dict[str, Any]]:
    if k <= 0 or embeddings.shape[0] < 2:
        return []
    n = int(embeddings.shape[0])
    output_rows: list[dict[str, Any]] = []
    chunk_size = max(1, int(chunk_size))
    wsi_group = encode_groups(wsi_ids)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        scores = embeddings[start:end] @ embeddings.T
        for local, global_i in enumerate(range(start, end)):
            scores[local, global_i] = -math.inf
            same_wsi = wsi_group == wsi_group[global_i]
            scores[local, same_wsi] = -math.inf
            valid_count = int(torch.isfinite(scores[local]).sum().item())
            if valid_count <= 0:
                continue
            top_k = min(k, valid_count)
            values, indices = torch.topk(scores[local], k=top_k)
            for rank, (value, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
                output_rows.append(
                    {
                        "query_index": global_i,
                        "neighbor_rank": rank,
                        "query_sample": sample_ids[global_i],
                        "neighbor_sample": sample_ids[idx],
                        "query_wsi": wsi_ids[global_i],
                        "neighbor_wsi": wsi_ids[idx],
                        "query_dataset": rows[global_i]["dataset"],
                        "neighbor_dataset": rows[idx]["dataset"],
                        "same_dataset": rows[global_i]["dataset"] == rows[idx]["dataset"],
                        "cosine": float(value),
                    }
                )
    return output_rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())

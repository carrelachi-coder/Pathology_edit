"""Calibrate Cross V1 directionality descriptors on real target images.

This script answers two instrumentation questions before generation-side
directionality is allowed to vote:

1. True-target calibration:
   Treat the real target crop as the generated image and check whether the
   descriptor says it is closer to the paired reference than to an alternate
   reference, using the same probe identities as the generation gate.
2. Descriptor-space margin:
   Measure whether the selected descriptor space has same-WSI vs same-dataset
   and different-dataset separation on real tumor crops.
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


ALTERNATE_MODES = ("same_dataset", "different_dataset")


@dataclass(frozen=True)
class EncodeItem:
    key: str
    dataset: str
    sample_id: str
    case_id: str
    wsi_id: str
    image_path: Path
    tissue_mask_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate Cross V1 descriptor directionality on real target crops."
    )
    parser.add_argument("--metadata", required=True)
    parser.add_argument(
        "--selection-manifest",
        default=None,
        help="selection_manifest.json from diagnose_cross_v1_generation_gate.py.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--uni-checkpoint-path", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Required for --feature-stage projected/encoder_hid_proj.",
    )
    parser.add_argument(
        "--feature-stage",
        choices=("uni", "projected", "encoder_hid_proj"),
        default="uni",
    )
    parser.add_argument(
        "--metadata-base-dir",
        default=None,
        help="Base directory for relative metadata paths. Defaults to metadata parent.",
    )
    parser.add_argument("--alternate-mode", choices=("same_dataset", "different_dataset", "both"), default="both")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--selection-seed", type=int, default=20260611)
    parser.add_argument("--tumor-label", type=int, action="append", default=None)
    parser.add_argument("--min-tumor-fraction", type=float, default=0.02)
    parser.add_argument("--min-tumor-tokens", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260611)
    parser.add_argument("--permutation-iters", type=int, default=2000)
    parser.add_argument("--permutation-seed", type=int, default=20260612)
    parser.add_argument(
        "--knn-max-samples",
        type=int,
        default=1000,
        help="Real-crop pool size for descriptor margin. 0 disables pool margin.",
    )
    parser.add_argument("--knn-samples-per-wsi", type=int, default=30)
    parser.add_argument("--knn-image-field", default="target_image")
    parser.add_argument("--knn-mask-field", default="target_tissue_mask")
    parser.add_argument("--knn-sample-id-field", default="sample_id")
    parser.add_argument("--nearest-k", type=int, default=5)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.tumor_label is None:
        args.tumor_label = [1]
    if args.feature_stage != "uni" and not args.checkpoint:
        raise ValueError("--checkpoint is required for projected/encoder_hid_proj calibration")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.min_tumor_tokens < 1:
        raise ValueError("--min-tumor-tokens must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from controlnet_train.data.common import load_image_tensor, load_tissue_mask, parse_sample_identity
    from controlnet_train.modules.reference_image_encoder import ReferenceImageEncoder, resize_mask_to_token_labels

    metadata_path = Path(args.metadata)
    base_dir = Path(args.metadata_base_dir) if args.metadata_base_dir else metadata_path.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = [normalize_record_paths(row, base_dir) for row in read_metadata(metadata_path)]
    alternate_modes = parse_alternate_modes(args.alternate_mode)
    probes = load_or_select_probes(
        records,
        selection_manifest=Path(args.selection_manifest) if args.selection_manifest else None,
        alternate_modes=alternate_modes,
        num_samples=args.num_samples,
        seed=args.selection_seed,
        tumor_labels=args.tumor_label,
        min_tumor_fraction=args.min_tumor_fraction,
        load_tissue_mask=load_tissue_mask,
        parse_sample_identity=parse_sample_identity,
    )

    device = resolve_device(args.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.torch_dtype]
    encoder, encoder_hid_proj = load_descriptor_encoder(
        checkpoint=args.checkpoint,
        uni_checkpoint_path=args.uni_checkpoint_path,
        feature_stage=args.feature_stage,
        device=device,
        dtype=dtype,
        ReferenceImageEncoder=ReferenceImageEncoder,
    )

    item_map = build_probe_encode_items(probes, parse_sample_identity=parse_sample_identity)
    embeddings, item_rows, skipped = encode_items(
        list(item_map.values()),
        encoder=encoder,
        encoder_hid_proj=encoder_hid_proj,
        load_image_tensor=load_image_tensor,
        load_tissue_mask=load_tissue_mask,
        resize_mask_to_token_labels=resize_mask_to_token_labels,
        feature_stage=args.feature_stage,
        tumor_labels=args.tumor_label,
        min_tumor_tokens=args.min_tumor_tokens,
        batch_size=args.batch_size,
        device=device,
    )
    calibration_rows = build_calibration_rows(probes, embeddings)
    write_csv(output_dir / "descriptor_true_target_calibration.csv", calibration_rows)
    write_csv(output_dir / "descriptor_probe_embeddings.csv", item_rows)

    summary: dict[str, Any] = {
        "metadata": str(metadata_path),
        "selection_manifest": str(args.selection_manifest) if args.selection_manifest else None,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "feature_stage": args.feature_stage,
        "tumor_labels": args.tumor_label,
        "num_probes": len(probes),
        "encoded_probe_items": len(item_rows),
        "skipped_probe_items": skipped[:100],
        "true_target_calibration": summarize_calibration_rows(
            calibration_rows,
            bootstrap_iters=args.bootstrap_iters,
            bootstrap_seed=args.bootstrap_seed,
            permutation_iters=args.permutation_iters,
            permutation_seed=args.permutation_seed,
        ),
    }

    if args.knn_max_samples > 0:
        pool_items = build_knn_pool_items(
            records,
            parse_sample_identity=parse_sample_identity,
            image_field=args.knn_image_field,
            mask_field=args.knn_mask_field,
            sample_id_field=args.knn_sample_id_field,
            max_samples=args.knn_max_samples,
            samples_per_wsi=args.knn_samples_per_wsi,
            seed=args.selection_seed,
        )
        pool_embeddings, pool_rows, pool_skipped = encode_items(
            pool_items,
            encoder=encoder,
            encoder_hid_proj=encoder_hid_proj,
            load_image_tensor=load_image_tensor,
            load_tissue_mask=load_tissue_mask,
            resize_mask_to_token_labels=resize_mask_to_token_labels,
            feature_stage=args.feature_stage,
            tumor_labels=args.tumor_label,
            min_tumor_tokens=args.min_tumor_tokens,
            batch_size=args.batch_size,
            device=device,
        )
        knn_summary, nearest_rows = summarize_descriptor_pool(
            pool_embeddings,
            pool_rows,
            nearest_k=args.nearest_k,
        )
        summary["descriptor_space_margin"] = knn_summary
        summary["knn_pool"] = {
            "image_field": args.knn_image_field,
            "mask_field": args.knn_mask_field,
            "selected_items": len(pool_items),
            "valid_embeddings": len(pool_rows),
            "skipped_count": len(pool_skipped),
            "skipped_preview": pool_skipped[:100],
        }
        write_csv(output_dir / "descriptor_knn_pool_embeddings.csv", pool_rows)
        write_csv(output_dir / "descriptor_nearest_neighbors.csv", nearest_rows)
    else:
        summary["descriptor_space_margin"] = {"status": "disabled"}

    write_json(output_dir / "descriptor_gate_calibration_summary.json", summary)
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
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        rows = payload.get("pairs") or payload.get("records")
        if isinstance(rows, list):
            return rows
    raise ValueError("metadata must be a list, JSONL, or dict with pairs/records")


def normalize_record_paths(record: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    row = dict(record)
    for key, value in list(row.items()):
        if not isinstance(value, str):
            continue
        if key.endswith("_image") or key.endswith("_mask") or key.endswith("_path"):
            row[key] = str(resolve_path(value, base_dir))
    return row


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else base_dir / path


def parse_alternate_modes(value: str) -> list[str]:
    if value == "both":
        return list(ALTERNATE_MODES)
    return [value]


def load_or_select_probes(
    records: list[dict[str, Any]],
    *,
    selection_manifest: Path | None,
    alternate_modes: list[str],
    num_samples: int,
    seed: int,
    tumor_labels: list[int],
    min_tumor_fraction: float,
    load_tissue_mask,
    parse_sample_identity,
) -> list[dict[str, Any]]:
    if selection_manifest is not None:
        return probes_from_manifest(
            selection_manifest,
            records,
            alternate_modes=alternate_modes,
            parse_sample_identity=parse_sample_identity,
        )
    return select_probes_from_records(
        records,
        alternate_modes=alternate_modes,
        num_samples=num_samples,
        seed=seed,
        tumor_labels=tumor_labels,
        min_tumor_fraction=min_tumor_fraction,
        load_tissue_mask=load_tissue_mask,
        parse_sample_identity=parse_sample_identity,
    )


def probes_from_manifest(
    manifest_path: Path,
    records: list[dict[str, Any]],
    *,
    alternate_modes: list[str],
    parse_sample_identity,
) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    by_target_ref = {
        (path_key(row.get("target_image")), path_key(row.get("reference_image"))): row
        for row in records
        if row.get("target_image") and row.get("reference_image")
    }
    by_ref = defaultdict(list)
    for row in records:
        if row.get("reference_image"):
            by_ref[path_key(row.get("reference_image"))].append(row)

    probes = []
    for probe_index, item in enumerate(manifest):
        target_key = path_key(item.get("target_image"))
        paired_ref_key = path_key(item.get("paired_reference_image"))
        paired = by_target_ref.get((target_key, paired_ref_key))
        if paired is None:
            paired = find_first_record_with_target(records, target_key)
        if paired is None:
            raise ValueError(f"manifest target not found in metadata: {item.get('target_image')}")

        paired = dict(paired)
        paired["target_image"] = str(item.get("target_image") or paired.get("target_image"))
        paired["reference_image"] = str(item.get("paired_reference_image") or paired.get("reference_image"))
        alternates: dict[str, dict[str, Any]] = {}
        for mode in alternate_modes:
            alt_payload = (item.get("alternates") or {}).get(mode)
            if not alt_payload:
                continue
            alt_key = path_key(alt_payload.get("reference_image"))
            candidates = by_ref.get(alt_key) or []
            if not candidates:
                raise ValueError(f"manifest alternate ref not found in metadata: {alt_payload}")
            alternates[mode] = dict(candidates[0])
        probes.append(
            {
                "probe_index": probe_index,
                "metadata_index": item.get("metadata_index"),
                "paired": enrich_record_identity(paired, parse_sample_identity=parse_sample_identity),
                "alternates": {
                    mode: enrich_record_identity(row, parse_sample_identity=parse_sample_identity)
                    for mode, row in alternates.items()
                },
            }
        )
    return probes


def find_first_record_with_target(records: list[dict[str, Any]], target_key: str) -> dict[str, Any] | None:
    for row in records:
        if path_key(row.get("target_image")) == target_key:
            return row
    return None


def select_probes_from_records(
    records: list[dict[str, Any]],
    *,
    alternate_modes: list[str],
    num_samples: int,
    seed: int,
    tumor_labels: list[int],
    min_tumor_fraction: float,
    load_tissue_mask,
    parse_sample_identity,
) -> list[dict[str, Any]]:
    valid = []
    for index, row in enumerate(records):
        if not all(key in row for key in ("target_image", "target_tissue_mask", "reference_image", "reference_tissue_mask")):
            continue
        if not record_has_tumor(row, "target_tissue_mask", tumor_labels, min_tumor_fraction, load_tissue_mask):
            continue
        if not record_has_tumor(row, "reference_tissue_mask", tumor_labels, min_tumor_fraction, load_tissue_mask):
            continue
        valid.append((index, enrich_record_identity(row, parse_sample_identity=parse_sample_identity)))
    rng = random.Random(seed)
    rng.shuffle(valid)
    selected = valid[: max(1, int(num_samples))]
    probes = []
    for metadata_index, paired in selected:
        alternates = {
            mode: choose_alternate_reference(paired, valid, mode=mode, seed=seed + metadata_index)
            for mode in alternate_modes
        }
        probes.append(
            {
                "probe_index": len(probes),
                "metadata_index": metadata_index,
                "paired": paired,
                "alternates": alternates,
            }
        )
    return probes


def record_has_tumor(
    record: dict[str, Any],
    mask_field: str,
    tumor_labels: list[int],
    min_fraction: float,
    load_tissue_mask,
) -> bool:
    try:
        mask = load_tissue_mask(record[mask_field])
    except Exception:
        return False
    allowed = torch.tensor(tumor_labels, dtype=mask.dtype)
    tumor = (mask.unsqueeze(-1) == allowed.view(1, 1, -1)).any(dim=-1)
    return float(tumor.float().mean().item()) >= float(min_fraction)


def choose_alternate_reference(
    paired: dict[str, Any],
    candidates: list[tuple[int, dict[str, Any]]],
    *,
    mode: str,
    seed: int,
) -> dict[str, Any]:
    paired_dataset = str(paired.get("dataset") or "")
    paired_case = str(paired.get("reference_case_id") or "")
    eligible = []
    for _, row in candidates:
        if row.get("reference_case_id") == paired_case:
            continue
        if path_key(row.get("reference_image")) == path_key(paired.get("reference_image")):
            continue
        dataset = str(row.get("dataset") or "")
        if mode == "same_dataset" and paired_dataset and dataset != paired_dataset:
            continue
        if mode == "different_dataset" and paired_dataset and dataset == paired_dataset:
            continue
        eligible.append(row)
    if not eligible:
        raise ValueError(f"no {mode} alternate found for {paired.get('sample_id')}")
    return random.Random(seed + (17 if mode == "same_dataset" else 29)).choice(eligible)


def enrich_record_identity(record: dict[str, Any], *, parse_sample_identity) -> dict[str, Any]:
    row = dict(record)
    sample_id = str(row.get("sample_id") or Path(str(row.get("target_image", "target"))).stem)
    ref_sample_id = str(row.get("reference_sample_id") or Path(str(row.get("reference_image", "ref"))).stem)
    target_case = first_present(row, ("target_case_id", "case_id", "target_wsi_id", "slide_id"))
    ref_case = first_present(row, ("reference_case_id", "reference_wsi_id", "reference_slide_id"))
    if target_case is None:
        target_case = parse_sample_identity(sample_id)[0]
    if ref_case is None:
        ref_case = parse_sample_identity(ref_sample_id)[0]
    dataset = str(row.get("dataset") or "unknown")
    row.update(
        {
            "sample_id": sample_id,
            "reference_sample_id": ref_sample_id,
            "target_case_id": str(target_case),
            "reference_case_id": str(ref_case),
            "target_wsi_id": f"{dataset}::{target_case}",
            "reference_wsi_id": f"{dataset}::{ref_case}",
        }
    )
    return row


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def path_key(value: Any) -> str:
    if value is None:
        return ""
    return str(Path(str(value).replace("\\", "/")).expanduser())


def build_probe_encode_items(probes: list[dict[str, Any]], *, parse_sample_identity) -> dict[str, EncodeItem]:
    items: dict[str, EncodeItem] = {}
    for probe in probes:
        paired = probe["paired"]
        target_key = f"target::{path_key(paired['target_image'])}"
        items[target_key] = make_encode_item(
            target_key,
            paired,
            image_field="target_image",
            mask_field="target_tissue_mask",
            sample_id_field="sample_id",
            case_id_field="target_case_id",
            wsi_id_field="target_wsi_id",
            parse_sample_identity=parse_sample_identity,
        )
        paired_key = f"ref::{path_key(paired['reference_image'])}"
        items[paired_key] = make_encode_item(
            paired_key,
            paired,
            image_field="reference_image",
            mask_field="reference_tissue_mask",
            sample_id_field="reference_sample_id",
            case_id_field="reference_case_id",
            wsi_id_field="reference_wsi_id",
            parse_sample_identity=parse_sample_identity,
        )
        for alternate in probe["alternates"].values():
            alternate_key = f"ref::{path_key(alternate['reference_image'])}"
            items[alternate_key] = make_encode_item(
                alternate_key,
                alternate,
                image_field="reference_image",
                mask_field="reference_tissue_mask",
                sample_id_field="reference_sample_id",
                case_id_field="reference_case_id",
                wsi_id_field="reference_wsi_id",
                parse_sample_identity=parse_sample_identity,
            )
    return items


def make_encode_item(
    key: str,
    record: dict[str, Any],
    *,
    image_field: str,
    mask_field: str,
    sample_id_field: str,
    case_id_field: str,
    wsi_id_field: str,
    parse_sample_identity,
) -> EncodeItem:
    sample_id = str(record.get(sample_id_field) or Path(str(record[image_field])).stem)
    case_id = str(record.get(case_id_field) or parse_sample_identity(sample_id)[0])
    dataset = str(record.get("dataset") or "unknown")
    wsi_id = str(record.get(wsi_id_field) or f"{dataset}::{case_id}")
    return EncodeItem(
        key=key,
        dataset=dataset,
        sample_id=sample_id,
        case_id=case_id,
        wsi_id=wsi_id,
        image_path=Path(str(record[image_field])),
        tissue_mask_path=Path(str(record[mask_field])),
    )


def load_descriptor_encoder(
    *,
    checkpoint: str | None,
    uni_checkpoint_path: str,
    feature_stage: str,
    device: torch.device,
    dtype: torch.dtype,
    ReferenceImageEncoder,
):
    encoder_hid_proj = None
    if checkpoint:
        from controlnet_train.cli.diagnose_ref_signal import (
            load_encoder_hid_proj_from_checkpoint,
            load_ref_encoder_from_checkpoint,
        )

        encoder, config = load_ref_encoder_from_checkpoint(
            checkpoint,
            uni_checkpoint_path,
            str(device),
            dtype,
            skip_perceiver=True,
        )
        if feature_stage == "encoder_hid_proj":
            encoder_hid_proj = load_encoder_hid_proj_from_checkpoint(
                checkpoint,
                hidden_dim=int(config["hidden_dim"]),
                device=str(device),
                dtype=dtype,
            )
            if encoder_hid_proj is None:
                raise FileNotFoundError(f"{checkpoint} does not contain encoder_hid_proj")
    else:
        encoder = ReferenceImageEncoder(uni_checkpoint_path, skip_perceiver=True)
        encoder.to(device=device, dtype=dtype)
        encoder.eval()
    return encoder, encoder_hid_proj


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def encode_items(
    items: list[EncodeItem],
    *,
    encoder,
    encoder_hid_proj,
    load_image_tensor,
    load_tissue_mask,
    resize_mask_to_token_labels,
    feature_stage: str,
    tumor_labels: list[int],
    min_tumor_tokens: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], list[dict[str, Any]]]:
    embeddings: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    pending_items: list[EncodeItem] = []
    pending_images: list[torch.Tensor] = []
    pending_masks: list[torch.Tensor] = []

    def flush() -> None:
        if not pending_items:
            return
        images = torch.stack(pending_images).to(device=device)
        masks = torch.stack(pending_masks)
        with torch.no_grad():
            features = encode_feature_tokens(
                encoder,
                encoder_hid_proj,
                images,
                feature_stage=feature_stage,
            ).float().cpu()
        labels = resize_mask_to_token_labels(masks, int(features.shape[1]))
        tumor_mask = build_label_mask(labels, tumor_labels)
        for local_idx, item in enumerate(pending_items):
            sample_tumor = tumor_mask[local_idx]
            tumor_count = int(sample_tumor.sum().item())
            if tumor_count < int(min_tumor_tokens):
                skipped.append(
                    {
                        "key": item.key,
                        "sample_id": item.sample_id,
                        "reason": "insufficient_tumor_tokens",
                        "tumor_token_count": tumor_count,
                    }
                )
                continue
            pooled = features[local_idx, sample_tumor].mean(dim=0)
            raw_norm = float(torch.linalg.vector_norm(pooled).item())
            pooled = F.normalize(pooled.float(), dim=0).cpu()
            embeddings[item.key] = pooled
            rows.append(
                {
                    "key": item.key,
                    "dataset": item.dataset,
                    "sample_id": item.sample_id,
                    "case_id": item.case_id,
                    "wsi_id": item.wsi_id,
                    "image_path": str(item.image_path),
                    "tissue_mask_path": str(item.tissue_mask_path),
                    "tumor_token_count": tumor_count,
                    "raw_feature_norm": raw_norm,
                }
            )
        pending_items.clear()
        pending_images.clear()
        pending_masks.clear()

    for item in items:
        try:
            pending_images.append(load_image_tensor(item.image_path))
            pending_masks.append(load_tissue_mask(item.tissue_mask_path))
            pending_items.append(item)
        except Exception as exc:  # noqa: BLE001 - diagnostic should keep going.
            skipped.append(
                {
                    "key": item.key,
                    "sample_id": item.sample_id,
                    "reason": f"load_failed:{type(exc).__name__}",
                    "detail": str(exc),
                }
            )
            continue
        if len(pending_items) >= batch_size:
            flush()
    flush()
    return embeddings, rows, skipped


def encode_feature_tokens(encoder, encoder_hid_proj, images: torch.Tensor, *, feature_stage: str) -> torch.Tensor:
    if feature_stage == "uni":
        return encoder.extract_uni_features(images)
    projected = encoder.encode_projected_patch_tokens(images)
    if feature_stage == "projected":
        return projected
    if feature_stage == "encoder_hid_proj":
        gate = encoder.reference_presence_gate(images, device=projected.device, dtype=projected.dtype)
        projected = projected * gate
        return encoder_hid_proj([projected])[0] * gate.to(device=projected.device, dtype=projected.dtype)
    raise ValueError(f"unsupported feature_stage: {feature_stage}")


def build_label_mask(labels: torch.Tensor, tumor_labels: list[int]) -> torch.Tensor:
    allowed = torch.tensor(tumor_labels, dtype=labels.dtype, device=labels.device)
    return (labels.unsqueeze(-1) == allowed.view(1, 1, -1)).any(dim=-1)


def build_calibration_rows(
    probes: list[dict[str, Any]],
    embeddings: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    rows = []
    for probe in probes:
        paired = probe["paired"]
        target_key = f"target::{path_key(paired['target_image'])}"
        paired_key = f"ref::{path_key(paired['reference_image'])}"
        target_embedding = embeddings.get(target_key)
        paired_embedding = embeddings.get(paired_key)
        if target_embedding is None or paired_embedding is None:
            continue
        for mode, alternate in sorted(probe["alternates"].items()):
            alternate_key = f"ref::{path_key(alternate['reference_image'])}"
            alternate_embedding = embeddings.get(alternate_key)
            if alternate_embedding is None:
                continue
            paired_cos = float(torch.dot(target_embedding, paired_embedding).item())
            alternate_cos = float(torch.dot(target_embedding, alternate_embedding).item())
            margin = paired_cos - alternate_cos
            rows.append(
                {
                    "probe_index": probe["probe_index"],
                    "metadata_index": probe.get("metadata_index"),
                    "alternate_mode": mode,
                    "dataset": paired.get("dataset", ""),
                    "target_sample_id": paired.get("sample_id", ""),
                    "paired_reference_sample_id": paired.get("reference_sample_id", ""),
                    "alternate_reference_sample_id": alternate.get("reference_sample_id", ""),
                    "target_case_id": paired.get("target_case_id", ""),
                    "paired_reference_case_id": paired.get("reference_case_id", ""),
                    "alternate_reference_case_id": alternate.get("reference_case_id", ""),
                    "target_paired_cosine": paired_cos,
                    "target_alternate_cosine": alternate_cos,
                    "cosine_margin": margin,
                    "paired_win": margin > 0.0,
                }
            )
    return rows


def summarize_calibration_rows(
    rows: list[dict[str, Any]],
    *,
    bootstrap_iters: int,
    bootstrap_seed: int,
    permutation_iters: int,
    permutation_seed: int,
) -> dict[str, Any]:
    if not rows:
        return {"status": "missing", "note": "no calibration rows"}
    by_mode = {}
    for mode in sorted({str(row["alternate_mode"]) for row in rows}):
        mode_rows = [row for row in rows if str(row["alternate_mode"]) == mode]
        by_mode[mode] = summarize_binary_margin_rows(
            mode_rows,
            bootstrap_iters=bootstrap_iters,
            bootstrap_seed=bootstrap_seed + stable_int_hash(mode),
            permutation_iters=permutation_iters,
            permutation_seed=permutation_seed + stable_int_hash(mode),
        )
    return {
        "status": "ok",
        "overall": summarize_binary_margin_rows(
            rows,
            bootstrap_iters=bootstrap_iters,
            bootstrap_seed=bootstrap_seed,
            permutation_iters=permutation_iters,
            permutation_seed=permutation_seed,
        ),
        "by_alternate_mode": by_mode,
    }


def summarize_binary_margin_rows(
    rows: list[dict[str, Any]],
    *,
    bootstrap_iters: int,
    bootstrap_seed: int,
    permutation_iters: int,
    permutation_seed: int,
) -> dict[str, Any]:
    wins = [1.0 if row["paired_win"] in (True, "True", "true", "1", 1) else 0.0 for row in rows]
    margins = [float(row["cosine_margin"]) for row in rows if math.isfinite(float(row["cosine_margin"]))]
    win_rate = finite_mean(wins)
    mean_margin = finite_mean(margins)
    n = len(rows)
    null_se = binomial_null_se(0.5, n)
    return {
        "n": n,
        "win_rate": win_rate,
        "win_rate_boot_se": bootstrap_stderr(wins, iters=bootstrap_iters, seed=bootstrap_seed),
        "win_rate_null_se": null_se,
        "win_rate_z_vs_0.5": (win_rate - 0.5) / null_se if null_se > 0 else math.nan,
        "mean_cosine_margin": mean_margin,
        "margin_boot_se": bootstrap_stderr(margins, iters=bootstrap_iters, seed=bootstrap_seed + 1009),
        "mean_target_paired_cosine": finite_mean([float(row["target_paired_cosine"]) for row in rows]),
        "mean_target_alternate_cosine": finite_mean([float(row["target_alternate_cosine"]) for row in rows]),
        "permutation": permutation_margin_test(
            wins,
            margins,
            iters=permutation_iters,
            seed=permutation_seed,
        ),
    }


def permutation_margin_test(
    wins: list[float],
    margins: list[float],
    *,
    iters: int,
    seed: int,
) -> dict[str, Any]:
    pairs = [(float(win), float(margin)) for win, margin in zip(wins, margins) if math.isfinite(margin)]
    if len(pairs) <= 1 or iters <= 0:
        return {"status": "missing"}
    rng = random.Random(seed)
    null_wins = []
    null_margins = []
    for _ in range(int(iters)):
        win_total = 0.0
        margin_total = 0.0
        for win, margin in pairs:
            if rng.randrange(2):
                win_total += 1.0 - win
                margin_total -= margin
            else:
                win_total += win
                margin_total += margin
        null_wins.append(win_total / len(pairs))
        null_margins.append(margin_total / len(pairs))
    return {
        "status": "ok",
        "null_win_rate_mean": finite_mean(null_wins),
        "null_win_rate_se": sample_std(null_wins),
        "null_margin_mean": finite_mean(null_margins),
        "null_margin_se": sample_std(null_margins),
    }


def build_knn_pool_items(
    records: list[dict[str, Any]],
    *,
    parse_sample_identity,
    image_field: str,
    mask_field: str,
    sample_id_field: str,
    max_samples: int,
    samples_per_wsi: int,
    seed: int,
) -> list[EncodeItem]:
    items = []
    seen: set[str] = set()
    for row in records:
        if image_field not in row or mask_field not in row:
            continue
        key = path_key(row[image_field])
        if key in seen:
            continue
        seen.add(key)
        row = enrich_record_identity(row, parse_sample_identity=parse_sample_identity)
        if image_field.startswith("reference"):
            case_field = "reference_case_id"
            wsi_field = "reference_wsi_id"
            fallback_sample_field = "reference_sample_id"
        else:
            case_field = "target_case_id"
            wsi_field = "target_wsi_id"
            fallback_sample_field = "sample_id"
        items.append(
            make_encode_item(
                f"pool::{key}",
                row,
                image_field=image_field,
                mask_field=mask_field,
                sample_id_field=sample_id_field if sample_id_field in row else fallback_sample_field,
                case_id_field=case_field,
                wsi_id_field=wsi_field,
                parse_sample_identity=parse_sample_identity,
            )
        )
    rng = random.Random(seed)
    if samples_per_wsi > 0:
        grouped: dict[str, list[EncodeItem]] = defaultdict(list)
        for item in items:
            grouped[item.wsi_id].append(item)
        items = []
        for group_items in grouped.values():
            group_items = list(group_items)
            rng.shuffle(group_items)
            items.extend(group_items[:samples_per_wsi])
    rng.shuffle(items)
    if max_samples > 0:
        items = items[:max_samples]
    return items


def summarize_descriptor_pool(
    embeddings: dict[str, torch.Tensor],
    rows: list[dict[str, Any]],
    *,
    nearest_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(rows) < 2:
        return {"status": "need_at_least_two_embeddings"}, []
    matrix = torch.stack([embeddings[str(row["key"])] for row in rows]).float()
    sim = matrix @ matrix.T
    n = int(sim.shape[0])
    dataset_ids = encode_groups([str(row["dataset"]) for row in rows])
    wsi_ids = encode_groups([str(row["wsi_id"]) for row in rows])
    tri_i, tri_j = torch.triu_indices(n, n, offset=1)
    values = sim[tri_i, tri_j]
    same_wsi = wsi_ids[tri_i] == wsi_ids[tri_j]
    same_dataset = dataset_ids[tri_i] == dataset_ids[tri_j]
    same_dataset_diff_wsi = same_dataset & ~same_wsi
    different_dataset = ~same_dataset
    nearest_rows = nearest_neighbor_rows(sim, rows, same_wsi_ids=wsi_ids, k=nearest_k)
    same_wsi_stats = describe_values(values[same_wsi])
    same_dataset_diff_stats = describe_values(values[same_dataset_diff_wsi])
    different_dataset_stats = describe_values(values[different_dataset])
    return (
        {
            "status": "ok",
            "num_embeddings": n,
            "pair_stats": {
                "all_pairs": describe_values(values),
                "same_wsi": same_wsi_stats,
                "same_dataset_different_wsi": same_dataset_diff_stats,
                "different_dataset": different_dataset_stats,
            },
            "margins": {
                "same_wsi_minus_same_dataset_different_wsi": (
                    safe_mean(same_wsi_stats) - safe_mean(same_dataset_diff_stats)
                ),
                "same_dataset_different_wsi_minus_different_dataset": (
                    safe_mean(same_dataset_diff_stats) - safe_mean(different_dataset_stats)
                ),
            },
            "nearest_neighbor": summarize_nearest_rows(nearest_rows),
        },
        nearest_rows,
    )


def nearest_neighbor_rows(
    sim: torch.Tensor,
    rows: list[dict[str, Any]],
    *,
    same_wsi_ids: torch.Tensor,
    k: int,
) -> list[dict[str, Any]]:
    output = []
    n = int(sim.shape[0])
    k = max(1, int(k))
    for i in range(n):
        scores = sim[i].clone()
        scores[i] = -math.inf
        top_k = min(k, max(0, n - 1))
        if top_k <= 0:
            continue
        values, indices = torch.topk(scores, k=top_k)
        has_same_wsi_candidate = bool(((same_wsi_ids == same_wsi_ids[i]) & (torch.arange(n) != i)).any().item())
        for rank, (value, j) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
            output.append(
                {
                    "query_index": i,
                    "neighbor_rank": rank,
                    "query_sample": rows[i]["sample_id"],
                    "neighbor_sample": rows[j]["sample_id"],
                    "query_wsi": rows[i]["wsi_id"],
                    "neighbor_wsi": rows[j]["wsi_id"],
                    "query_dataset": rows[i]["dataset"],
                    "neighbor_dataset": rows[j]["dataset"],
                    "same_wsi": rows[i]["wsi_id"] == rows[j]["wsi_id"],
                    "same_dataset": rows[i]["dataset"] == rows[j]["dataset"],
                    "query_has_same_wsi_candidate": has_same_wsi_candidate,
                    "cosine": float(value),
                }
            )
    return output


def summarize_nearest_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    top1 = [row for row in rows if int(row["neighbor_rank"]) == 1]
    eligible = [row for row in top1 if row["query_has_same_wsi_candidate"] in (True, "True", "true", "1", 1)]
    return {
        "top1_count": len(top1),
        "top1_same_wsi_rate_all": finite_mean([1.0 if row["same_wsi"] else 0.0 for row in top1]),
        "top1_same_dataset_rate_all": finite_mean([1.0 if row["same_dataset"] else 0.0 for row in top1]),
        "top1_same_wsi_rate_eligible": finite_mean([1.0 if row["same_wsi"] else 0.0 for row in eligible]),
        "top1_same_wsi_eligible_count": len(eligible),
    }


def encode_groups(values: list[str]) -> torch.Tensor:
    lookup: dict[str, int] = {}
    encoded = []
    for value in values:
        if value not in lookup:
            lookup[value] = len(lookup)
        encoded.append(lookup[value])
    return torch.tensor(encoded, dtype=torch.long)


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


def safe_mean(stats: dict[str, Any]) -> float:
    value = stats.get("mean", math.nan)
    return float(value) if math.isfinite(float(value)) else math.nan


def finite_mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else math.nan


def sample_std(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) <= 1:
        return math.nan
    mean = finite_mean(finite)
    return math.sqrt(sum((value - mean) ** 2 for value in finite) / (len(finite) - 1))


def bootstrap_stderr(values: list[float], *, iters: int, seed: int) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) <= 1:
        return math.nan
    if iters <= 0:
        return sample_std(finite) / math.sqrt(len(finite))
    rng = random.Random(seed)
    means = []
    n = len(finite)
    for _ in range(int(iters)):
        total = 0.0
        for _ in range(n):
            total += finite[rng.randrange(n)]
        means.append(total / n)
    return sample_std(means)


def binomial_null_se(null_rate: float, n: int) -> float:
    if n <= 0:
        return math.nan
    return math.sqrt(max(null_rate * (1.0 - null_rate), 0.0) / n)


def stable_int_hash(value: str) -> int:
    result = 0
    for character in value:
        result = (result * 131 + ord(character)) % 1_000_000_007
    return result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())

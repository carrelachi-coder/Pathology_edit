"""Inference-only Cross V1 reference projection diagnostics.

This script does not import or run the training entrypoint. It loads an
eval-ready Cross V1 checkpoint through the inference bundle, then probes:

* A: whether reference tokens remain separated after each IP to_v projection.
* B: whether IP attention residual channel means change across blocks when the
  reference image is swapped while target inputs are held fixed.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_cross_v1_generation_gate import (  # noqa: E402
    ALTERNATE_MODES,
    build_manifest_row,
    choose_alternate_reference,
    parse_indices,
    parse_mode_selection,
    read_records,
    record_sample_id,
    reference_case_id,
    reference_sample_id,
    resolve_prompt,
    select_gate_records,
    select_records_from_manifest,
)


DEFAULT_PROMPT = "H&E stained cancer histopathology at 40x magnification"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross V1 inference-only W_v and IP attention output diagnostics."
    )
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--checkpoint", required=True, help="Eval-ready Cross V1 checkpoint dir.")
    parser.add_argument("--uni-checkpoint-path", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selection-manifest", default=os.environ.get("PROBE_SELECTION_MANIFEST"))
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument(
        "--record-indices",
        default="",
        help="Optional comma-separated metadata indices. Overrides --num-samples.",
    )
    parser.add_argument("--selection-seed", type=int, default=20260611)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument(
        "--alternate-mode",
        choices=("same_dataset", "different_dataset", "both"),
        default="same_dataset",
        help="Different-WSI alternate reference source.",
    )
    parser.add_argument(
        "--alternate-mask-policy",
        choices=("paired", "alternate"),
        default="paired",
        help=(
            "Use paired reference masks for the alternate image to isolate image/stain "
            "signal, or use the alternate record's own masks."
        ),
    )
    parser.add_argument("--tumor-label", type=int, default=1)
    parser.add_argument("--min-tumor-fraction", type=float, default=0.02)
    parser.add_argument(
        "--require-tumor-filter",
        action="store_true",
        help=(
            "Open every metadata mask and keep only records meeting the tumor-fraction "
            "filter. Off by default because this diagnostic only needs paired vs "
            "different-WSI reference images."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=1,
        help="Used only for B attention-output probing. A does not sample.",
    )
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--ip-scale", type=float, default=1.0)
    parser.add_argument(
        "--regional-ip-soft-bias",
        type=float,
        default=None,
        help=(
            "Optional inference-time override for global-soft-bias IP routing. "
            "Use e.g. 0.5 to test whether the trained/default b≈4 is compressing "
            "paired-vs-alternate attention outputs."
        ),
    )
    parser.add_argument("--prompt-source", choices=("metadata", "dataset"), default="dataset")
    parser.add_argument("--prompt", default=None)
    parser.add_argument(
        "--skip-attention-output-probe",
        action="store_true",
        help="Run only A (token -> W_v projection) and skip generation for B.",
    )
    parser.add_argument(
        "--skip-v-projection-pairwise",
        action="store_true",
        help="Skip all-pairs V_IP projection cosine diagnostics across encoded reference arms.",
    )
    parser.add_argument(
        "--v-pairwise-max-references",
        type=int,
        default=64,
        help="Maximum encoded reference arms to keep for all-pairs V_IP diagnostics; <=0 keeps all.",
    )
    parser.add_argument(
        "--v-pairwise-max-pairs-per-block",
        type=int,
        default=0,
        help="Optional cap on sampled reference pairs per block/branch for V_IP pairwise diagnostics.",
    )
    parser.add_argument(
        "--save-channel-means",
        action="store_true",
        help="Save full first-IP-output channel mean vectors as .pt files.",
    )
    parser.add_argument("--token-separated-max-cosine", type=float, default=0.60)
    parser.add_argument("--v-collapse-min-cosine", type=float, default=0.98)
    parser.add_argument("--mean-unchanged-min-cosine", type=float, default=0.995)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata_path = Path(args.metadata)
    records = read_records(metadata_path)
    alternate_modes = parse_mode_selection(args.alternate_mode, ALTERNATE_MODES)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.selection_manifest:
        selected = select_records_from_manifest(
            Path(args.selection_manifest),
            records,
            alternate_modes=alternate_modes,
        )
    elif args.require_tumor_filter:
        selected = select_gate_records(
            records,
            record_indices=parse_indices(args.record_indices),
            num_samples=args.num_samples,
            seed=args.selection_seed,
            tumor_label=args.tumor_label,
            min_tumor_fraction=args.min_tumor_fraction,
            alternate_modes=alternate_modes,
        )
    else:
        selected = select_fast_projection_records(
            records,
            record_indices=parse_indices(args.record_indices),
            num_samples=args.num_samples,
            seed=args.selection_seed,
            alternate_modes=alternate_modes,
        )

    manifest = [
        build_manifest_row(index, paired, alternates)
        for index, paired, alternates in selected
    ]
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf8",
    )
    print(f"selected {len(selected)} inference projection probes")

    from controlnet_train.inference.pipeline_cross_v1 import (
        load_cross_v1_bundle,
        run_cross_v1_bundle,
        set_ip_adapter_scale,
        set_ip_soft_bias,
    )

    dtype_by_name = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    bundle = load_cross_v1_bundle(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        checkpoint_path=args.checkpoint,
        uni_checkpoint_path=args.uni_checkpoint_path,
        device=args.device,
        torch_dtype=dtype_by_name[args.torch_dtype],
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        ip_adapter_scale=args.ip_scale,
    )
    set_ip_adapter_scale(bundle.flux_pipeline.transformer, args.ip_scale)
    soft_bias_override = None
    if args.regional_ip_soft_bias is not None:
        soft_bias_override = set_ip_soft_bias(
            bundle.flux_pipeline.transformer,
            args.regional_ip_soft_bias,
        )
        print(
            "regional_ip_soft_bias override "
            f"requested={soft_bias_override['requested']:g} "
            f"applied={soft_bias_override['applied']} "
            f"params={soft_bias_override['parameter_count']}"
        )

    projection_rows: list[dict[str, Any]] = []
    v_projection_pairwise_rows: list[dict[str, Any]] = []
    v_pairwise_reference_arms: dict[str, dict[str, Any]] = {}
    attention_rows: list[dict[str, Any]] = []
    attention_block_rows: list[dict[str, Any]] = []
    probe_summaries: list[dict[str, Any]] = []

    for probe_index, (metadata_index, paired, alternates) in enumerate(selected):
        sample_id = record_sample_id(paired)
        sample_dir = output_dir / f"{probe_index:03d}_{safe_name(sample_id)}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        target_tissue, target_nuclei = load_target_masks(paired)

        for alternate_mode in alternate_modes:
            alternate = alternates[alternate_mode]
            arms = build_reference_arms(
                paired=paired,
                alternate=alternate,
                alternate_mask_policy=args.alternate_mask_policy,
            )
            encoded = {
                name: encode_reference_arm(bundle, payload)
                for name, payload in arms.items()
            }
            if not args.skip_v_projection_pairwise:
                add_v_pairwise_reference_arm(
                    v_pairwise_reference_arms,
                    encoded=encoded["paired"],
                    metadata_index=metadata_index,
                    probe_index=probe_index,
                    parent_sample_id=sample_id,
                    role="paired",
                    alternate_mode="",
                    reference_record=paired,
                    alternate_mask_policy=args.alternate_mask_policy,
                )
                add_v_pairwise_reference_arm(
                    v_pairwise_reference_arms,
                    encoded=encoded["alternate_feature"],
                    metadata_index=metadata_index,
                    probe_index=probe_index,
                    parent_sample_id=sample_id,
                    role="alternate_feature",
                    alternate_mode=alternate_mode,
                    reference_record=alternate,
                    alternate_mask_policy=args.alternate_mask_policy,
                )
            pair_projection_rows = collect_v_projection_pair_rows(
                bundle=bundle,
                paired=encoded["paired"],
                alternate=encoded["alternate_feature"],
                metadata_index=metadata_index,
                sample_id=sample_id,
                paired_reference_sample_id=reference_sample_id(paired),
                alternate_reference_sample_id=reference_sample_id(alternate),
                alternate_mode=alternate_mode,
            )
            projection_rows.extend(pair_projection_rows)

            token_stats = tensor_pair_stats(
                encoded["paired"]["ref_features"],
                encoded["alternate_feature"]["ref_features"],
            )
            ip_hidden_stats = tensor_pair_stats(
                flatten_tensors(encoded["paired"]["ip_hidden_states"]),
                flatten_tensors(encoded["alternate_feature"]["ip_hidden_states"]),
            )

            attention_summary = None
            if not args.skip_attention_output_probe:
                prompt = resolve_prompt(args, paired) or DEFAULT_PROMPT
                arm_collectors: dict[str, dict[str, Any]] = {}
                probe_seed = int(args.generation_seed) + int(metadata_index) * 1009
                for arm_name in ("paired", "alternate_feature"):
                    collector: dict[str, Any] = {
                        "store_first_ip_output": True,
                        "store_attention_pooling": True,
                    }
                    with temporary_ip_debug_collector(bundle.flux_pipeline.transformer, collector):
                        run_cross_v1_bundle(
                            bundle,
                            reference_image=arms[arm_name]["image"],
                            reference_tissue_mask=arms[arm_name]["tissue_mask"],
                            reference_nuclei_mask=arms[arm_name]["nuclei_mask"],
                            target_tissue_mask=target_tissue,
                            target_nuclei_mask=target_nuclei,
                            prompt=prompt,
                            seed=probe_seed,
                        )
                    arm_collectors[arm_name] = collector

                attention_summary, block_rows = summarize_attention_pair(
                    paired_collector=arm_collectors["paired"],
                    alternate_collector=arm_collectors["alternate_feature"],
                    metadata_index=metadata_index,
                    sample_id=sample_id,
                    paired_reference_sample_id=reference_sample_id(paired),
                    alternate_reference_sample_id=reference_sample_id(alternate),
                    alternate_mode=alternate_mode,
                    regional_ip_soft_bias=(
                        float(args.regional_ip_soft_bias)
                        if args.regional_ip_soft_bias is not None
                        else None
                    ),
                    regional_ip_soft_bias_applied=bool(
                        soft_bias_override and soft_bias_override.get("applied", False)
                    ),
                    save_dir=sample_dir if args.save_channel_means else None,
                )
                attention_rows.append(attention_summary)
                attention_block_rows.extend(block_rows)

            projection_summary = summarize_projection_rows(pair_projection_rows, args)
            probe_summary = {
                "metadata_index": int(metadata_index),
                "sample_id": sample_id,
                "paired_reference_sample_id": reference_sample_id(paired),
                "alternate_reference_sample_id": reference_sample_id(alternate),
                "paired_reference_case_id": reference_case_id(paired),
                "alternate_reference_case_id": reference_case_id(alternate),
                "alternate_mode": alternate_mode,
                "alternate_mask_policy": args.alternate_mask_policy,
                "regional_ip_soft_bias": (
                    float(args.regional_ip_soft_bias)
                    if args.regional_ip_soft_bias is not None
                    else None
                ),
                "regional_ip_soft_bias_applied": bool(
                    soft_bias_override and soft_bias_override.get("applied", False)
                ),
                "ref_token_pair": token_stats,
                "ip_hidden_pair": ip_hidden_stats,
                "v_projection": projection_summary,
                "attention_output": attention_summary,
            }
            probe_summaries.append(probe_summary)
            (sample_dir / f"{alternate_mode}_diagnostics.json").write_text(
                json.dumps(probe_summary, ensure_ascii=False, indent=2, allow_nan=True),
                encoding="utf8",
            )
            print(format_probe_line(probe_summary))

    if not args.skip_v_projection_pairwise:
        v_projection_pairwise_rows = collect_v_projection_pairwise_rows(
            bundle=bundle,
            reference_arms=list(v_pairwise_reference_arms.values()),
            max_references=args.v_pairwise_max_references,
            max_pairs_per_block=args.v_pairwise_max_pairs_per_block,
            seed=args.selection_seed,
        )
        print(
            "computed V_IP pairwise diagnostics "
            f"references={len(v_pairwise_reference_arms)} "
            f"rows={len(v_projection_pairwise_rows)}"
        )

    write_csv(output_dir / "v_projection_rows.csv", projection_rows)
    write_csv(output_dir / "v_projection_pairwise_rows.csv", v_projection_pairwise_rows)
    write_csv(output_dir / "attention_output_rows.csv", attention_rows)
    write_csv(output_dir / "attention_block_rows.csv", attention_block_rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "metadata": str(args.metadata),
        "num_probes": len(probe_summaries),
        "alternate_modes": alternate_modes,
        "alternate_mask_policy": args.alternate_mask_policy,
        "ip_scale": float(args.ip_scale),
        "regional_ip_soft_bias": (
            float(args.regional_ip_soft_bias)
            if args.regional_ip_soft_bias is not None
            else None
        ),
        "regional_ip_soft_bias_override": soft_bias_override,
        "thresholds": {
            "token_separated_max_cosine": float(args.token_separated_max_cosine),
            "v_collapse_min_cosine": float(args.v_collapse_min_cosine),
            "mean_unchanged_min_cosine": float(args.mean_unchanged_min_cosine),
        },
        "v_projection": summarize_all_projection_rows(projection_rows, args),
        "v_projection_pairwise": summarize_all_v_projection_pairwise_rows(v_projection_pairwise_rows),
        "attention_output": summarize_all_attention_rows(attention_rows, args),
        "attention_blocks": summarize_all_attention_block_rows(attention_block_rows, args),
        "probe_summaries": probe_summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf8",
    )
    print(f"wrote inference projection diagnostics to {output_dir}")
    return 0


def select_fast_projection_records(
    records: list[dict[str, Any]],
    *,
    record_indices: list[int],
    num_samples: int,
    seed: int,
    alternate_modes: list[str],
) -> list[tuple[int, dict[str, Any], dict[str, dict[str, Any]]]]:
    """Select probes without scanning mask images across the full train metadata."""
    indexed = [
        (index, record)
        for index, record in enumerate(records)
        if record.get("target_image")
        and record.get("reference_image")
        and record.get("reference_tissue_mask")
        and record.get("reference_nuclei_mask")
        and record.get("target_tissue_mask")
        and record.get("target_nuclei_mask")
    ]
    if record_indices:
        by_index = dict(indexed)
        missing = [index for index in record_indices if index not in by_index]
        if missing:
            raise ValueError(f"requested indices are missing required fields: {missing}")
        paired_candidates = [(index, by_index[index]) for index in record_indices]
    else:
        paired_candidates = list(indexed)
        random.Random(seed).shuffle(paired_candidates)
        paired_candidates = paired_candidates[: max(1, int(num_samples))]

    selected = []
    for metadata_index, paired in paired_candidates:
        alternates: dict[str, dict[str, Any]] = {}
        for mode in alternate_modes:
            alternates[mode] = choose_alternate_reference(
                paired,
                indexed,
                seed=seed + metadata_index,
                mode=mode,
            )
        selected.append((metadata_index, paired, alternates))
    return selected


def load_target_masks(record: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    from controlnet_train.data.common import load_nuclei_mask, load_tissue_mask

    return load_tissue_mask(record["target_tissue_mask"]), load_nuclei_mask(record["target_nuclei_mask"])


def build_reference_arms(
    *,
    paired: dict[str, Any],
    alternate: dict[str, Any],
    alternate_mask_policy: str,
) -> dict[str, dict[str, torch.Tensor]]:
    from controlnet_train.data.common import load_image_tensor, load_nuclei_mask, load_tissue_mask

    paired_image = load_image_tensor(paired["reference_image"])
    paired_tissue = load_tissue_mask(paired["reference_tissue_mask"])
    paired_nuclei = load_nuclei_mask(paired["reference_nuclei_mask"])
    alternate_image = resize_image_tensor(load_image_tensor(alternate["reference_image"]), paired_image.shape[1:])
    if alternate_mask_policy == "alternate":
        alternate_tissue = resize_mask_tensor(load_tissue_mask(alternate["reference_tissue_mask"]), paired_tissue.shape)
        alternate_nuclei = resize_mask_tensor(load_nuclei_mask(alternate["reference_nuclei_mask"]), paired_nuclei.shape)
    else:
        alternate_tissue = paired_tissue
        alternate_nuclei = paired_nuclei
    return {
        "paired": {
            "image": paired_image,
            "tissue_mask": paired_tissue,
            "nuclei_mask": paired_nuclei,
        },
        "alternate_feature": {
            "image": alternate_image,
            "tissue_mask": alternate_tissue,
            "nuclei_mask": alternate_nuclei,
        },
    }


@torch.inference_mode()
def encode_reference_arm(bundle, payload: dict[str, torch.Tensor]) -> dict[str, Any]:
    ref_encoder = bundle.ref_encoder
    transformer = bundle.flux_pipeline.transformer
    device = bundle.device
    image = payload["image"]
    reference_batch = image.unsqueeze(0).to(
        device=device,
        dtype=next(ref_encoder.uni.parameters()).dtype,
    )
    if bundle.regional_ip_adapter:
        tissue_batch = payload["tissue_mask"].unsqueeze(0).to(device=device)
        nuclei_batch = payload["nuclei_mask"].unsqueeze(0).to(device=device)
        ref_features, token_labels = ref_encoder.encode_region_ip_tokens(
            reference_batch,
            tissue_batch,
            nuclei_mask=nuclei_batch,
            token_mode=bundle.regional_ip_token_mode,
            label_mode=bundle.regional_ip_label_mode,
        )
    else:
        ref_features = ref_encoder(reference_batch)
        token_labels = None
    ref_features = ref_features.to(device=device)
    ref_gate = ref_encoder.reference_presence_gate(
        reference_batch,
        device=device,
        dtype=next(transformer.encoder_hid_proj.parameters()).dtype,
    )
    ip_hidden_states = transformer.encoder_hid_proj([ref_features])
    ip_hidden_states = [
        hidden.to(device=device) * ref_gate.to(device=device, dtype=hidden.dtype)
        for hidden in ip_hidden_states
    ]
    return {
        "ref_features": ref_features.detach(),
        "ip_hidden_states": [hidden.detach() for hidden in ip_hidden_states],
        "token_labels": token_labels.detach() if torch.is_tensor(token_labels) else None,
    }


@torch.inference_mode()
def collect_v_projection_pair_rows(
    *,
    bundle,
    paired: dict[str, Any],
    alternate: dict[str, Any],
    metadata_index: int,
    sample_id: str,
    paired_reference_sample_id: str,
    alternate_reference_sample_id: str,
    alternate_mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block_type, block_index, processor in iter_ip_processors(bundle.flux_pipeline.transformer):
        for branch_index, to_v_ip in enumerate(getattr(processor, "to_v_ip", [])):
            paired_input = paired["ip_hidden_states"][branch_index]
            alternate_input = alternate["ip_hidden_states"][branch_index]
            paired_projected = to_v_ip(
                paired_input.detach().clone().to(device=to_v_ip.weight.device, dtype=to_v_ip.weight.dtype)
            )
            alternate_projected = to_v_ip(
                alternate_input.detach().clone().to(device=to_v_ip.weight.device, dtype=to_v_ip.weight.dtype)
            )
            input_stats = tensor_pair_stats(paired_input, alternate_input)
            value_stats = tensor_pair_stats(paired_projected, alternate_projected)
            mean_stats = tensor_pair_stats(
                channel_mean(paired_projected),
                channel_mean(alternate_projected),
            )
            rows.append(
                {
                    "metadata_index": int(metadata_index),
                    "sample_id": sample_id,
                    "paired_reference_sample_id": paired_reference_sample_id,
                    "alternate_reference_sample_id": alternate_reference_sample_id,
                    "alternate_mode": alternate_mode,
                    "block_type": block_type,
                    "block_index": int(block_index),
                    "block": f"{block_type}_{block_index}",
                    "branch_index": int(branch_index),
                    "input_numel_match": bool(input_stats["numel_match"]),
                    "input_cosine": input_stats["cosine"],
                    "input_l1": input_stats["l1"],
                    "input_rmse": input_stats["rmse"],
                    "value_numel_match": bool(value_stats["numel_match"]),
                    "value_cosine": value_stats["cosine"],
                    "value_l1": value_stats["l1"],
                    "value_rmse": value_stats["rmse"],
                    "value_channel_mean_cosine": mean_stats["cosine"],
                    "value_channel_mean_l1": mean_stats["l1"],
                    "value_channel_mean_rmse": mean_stats["rmse"],
                    "value_l2_norm_paired": tensor_l2_norm(paired_projected),
                    "value_l2_norm_alternate": tensor_l2_norm(alternate_projected),
                }
            )
    return rows


def add_v_pairwise_reference_arm(
    reference_arms: dict[str, dict[str, Any]],
    *,
    encoded: dict[str, Any],
    metadata_index: int,
    probe_index: int,
    parent_sample_id: str,
    role: str,
    alternate_mode: str,
    reference_record: dict[str, Any],
    alternate_mask_policy: str,
) -> None:
    reference_id = reference_sample_id(reference_record)
    case_id = reference_case_id(reference_record)
    if role == "paired":
        arm_key = f"paired|{metadata_index}|{reference_id}"
    elif alternate_mask_policy == "paired":
        arm_key = f"{role}|{metadata_index}|{alternate_mode}|{reference_id}|paired_mask"
    else:
        arm_key = f"{role}|{alternate_mode}|{reference_id}|alternate_mask"
    if arm_key in reference_arms:
        return
    reference_arms[arm_key] = {
        "arm_id": arm_key,
        "metadata_index": int(metadata_index),
        "probe_index": int(probe_index),
        "parent_sample_id": parent_sample_id,
        "role": role,
        "alternate_mode": alternate_mode,
        "alternate_mask_policy": alternate_mask_policy,
        "reference_sample_id": reference_id,
        "reference_case_id": case_id,
        "reference_image": str(reference_record.get("reference_image", "")),
        "ip_hidden_states": [hidden.detach().cpu() for hidden in encoded["ip_hidden_states"]],
    }


@torch.inference_mode()
def collect_v_projection_pairwise_rows(
    *,
    bundle,
    reference_arms: list[dict[str, Any]],
    max_references: int,
    max_pairs_per_block: int,
    seed: int,
) -> list[dict[str, Any]]:
    if max_references > 0:
        reference_arms = reference_arms[: int(max_references)]
    if len(reference_arms) < 2:
        return []
    pair_indices = [
        (left_index, right_index)
        for left_index in range(len(reference_arms))
        for right_index in range(left_index + 1, len(reference_arms))
    ]
    if max_pairs_per_block > 0 and len(pair_indices) > max_pairs_per_block:
        rng = random.Random(seed)
        rng.shuffle(pair_indices)
        pair_indices = pair_indices[: int(max_pairs_per_block)]

    rows: list[dict[str, Any]] = []
    for block_type, block_index, processor in iter_ip_processors(bundle.flux_pipeline.transformer):
        for branch_index, to_v_ip in enumerate(getattr(processor, "to_v_ip", [])):
            projected_values: list[torch.Tensor | None] = []
            for arm in reference_arms:
                hidden_states = arm["ip_hidden_states"]
                if branch_index >= len(hidden_states):
                    projected_values.append(None)
                    continue
                hidden = hidden_states[branch_index].to(
                    device=to_v_ip.weight.device,
                    dtype=to_v_ip.weight.dtype,
                )
                projected_values.append(to_v_ip(hidden).detach().cpu())
            for pairwise_index, (left_index, right_index) in enumerate(pair_indices):
                left_arm = reference_arms[left_index]
                right_arm = reference_arms[right_index]
                left_hidden_states = left_arm["ip_hidden_states"]
                right_hidden_states = right_arm["ip_hidden_states"]
                left_hidden = (
                    left_hidden_states[branch_index]
                    if branch_index < len(left_hidden_states)
                    else None
                )
                right_hidden = (
                    right_hidden_states[branch_index]
                    if branch_index < len(right_hidden_states)
                    else None
                )
                left_value = projected_values[left_index]
                right_value = projected_values[right_index]
                input_stats = tensor_pair_stats(left_hidden, right_hidden)
                input_token_stats = token_pair_stats(left_hidden, right_hidden)
                value_stats = tensor_pair_stats(left_value, right_value)
                value_token_stats = token_pair_stats(left_value, right_value)
                mean_stats = tensor_pair_stats(
                    channel_mean(left_value),
                    channel_mean(right_value),
                )
                role_pair = role_pair_label(str(left_arm["role"]), str(right_arm["role"]))
                same_parent = left_arm["metadata_index"] == right_arm["metadata_index"]
                same_reference_sample = (
                    str(left_arm["reference_sample_id"]) == str(right_arm["reference_sample_id"])
                )
                same_reference_case = (
                    str(left_arm["reference_case_id"]) == str(right_arm["reference_case_id"])
                )
                rows.append(
                    {
                        "pairwise_index": int(pairwise_index),
                        "block_type": block_type,
                        "block_index": int(block_index),
                        "block": f"{block_type}_{block_index}",
                        "branch_index": int(branch_index),
                        "left_arm_id": left_arm["arm_id"],
                        "right_arm_id": right_arm["arm_id"],
                        "left_metadata_index": int(left_arm["metadata_index"]),
                        "right_metadata_index": int(right_arm["metadata_index"]),
                        "left_probe_index": int(left_arm["probe_index"]),
                        "right_probe_index": int(right_arm["probe_index"]),
                        "same_parent_probe": bool(same_parent),
                        "left_role": left_arm["role"],
                        "right_role": right_arm["role"],
                        "role_pair": role_pair,
                        "left_alternate_mode": left_arm["alternate_mode"],
                        "right_alternate_mode": right_arm["alternate_mode"],
                        "left_parent_sample_id": left_arm["parent_sample_id"],
                        "right_parent_sample_id": right_arm["parent_sample_id"],
                        "left_reference_sample_id": left_arm["reference_sample_id"],
                        "right_reference_sample_id": right_arm["reference_sample_id"],
                        "same_reference_sample": bool(same_reference_sample),
                        "sample_relation": (
                            "same_reference_sample"
                            if same_reference_sample
                            else "different_reference_sample"
                        ),
                        "left_reference_case_id": left_arm["reference_case_id"],
                        "right_reference_case_id": right_arm["reference_case_id"],
                        "same_reference_case": bool(same_reference_case),
                        "case_relation": (
                            "same_reference_case"
                            if same_reference_case
                            else "different_reference_case"
                        ),
                        "input_numel_match": bool(input_stats["numel_match"]),
                        "input_cosine": input_stats["cosine"],
                        "input_l1": input_stats["l1"],
                        "input_rmse": input_stats["rmse"],
                        "input_token_num_vectors": int(input_token_stats["num_vectors"]),
                        "input_token_flat_cosine": input_token_stats["flat_cosine"],
                        "input_token_mean_cosine": input_token_stats["mean_cosine"],
                        "input_token_min_cosine": input_token_stats["min_cosine"],
                        "input_token_max_cosine": input_token_stats["max_cosine"],
                        "value_numel_match": bool(value_stats["numel_match"]),
                        "value_cosine": value_stats["cosine"],
                        "value_l1": value_stats["l1"],
                        "value_rmse": value_stats["rmse"],
                        "value_token_num_vectors": int(value_token_stats["num_vectors"]),
                        "value_token_flat_cosine": value_token_stats["flat_cosine"],
                        "value_token_mean_cosine": value_token_stats["mean_cosine"],
                        "value_token_min_cosine": value_token_stats["min_cosine"],
                        "value_token_max_cosine": value_token_stats["max_cosine"],
                        "value_channel_mean_cosine": mean_stats["cosine"],
                        "value_channel_mean_l1": mean_stats["l1"],
                        "value_channel_mean_rmse": mean_stats["rmse"],
                        "value_l2_norm_left": tensor_l2_norm(left_value),
                        "value_l2_norm_right": tensor_l2_norm(right_value),
                    }
                )
    return rows


def role_pair_label(left_role: str, right_role: str) -> str:
    role_rank = {"paired": 0, "alternate_feature": 1}
    left_key = (role_rank.get(left_role, 99), left_role)
    right_key = (role_rank.get(right_role, 99), right_role)
    if left_key <= right_key:
        return f"{left_role}_vs_{right_role}"
    return f"{right_role}_vs_{left_role}"


def iter_ip_processors(transformer) -> Iterator[tuple[str, int, Any]]:
    for block_type, blocks in (
        ("double", getattr(transformer, "transformer_blocks", [])),
        ("single", getattr(transformer, "single_transformer_blocks", [])),
    ):
        for index, block in enumerate(blocks):
            processor = getattr(getattr(block, "attn", None), "processor", None)
            if processor is not None and hasattr(processor, "to_v_ip"):
                yield block_type, index, processor


@contextlib.contextmanager
def temporary_ip_debug_collector(transformer, collector: dict[str, Any] | None):
    processors: list[tuple[Any, bool, Any]] = []
    patched_module = None
    original_recorder = None
    if collector is not None:
        from controlnet_train.training import flux_phase5_cross_v1 as cross_v1_training

        patched_module = cross_v1_training
        original_recorder = cross_v1_training._record_ip_attention_debug
        cross_v1_training._record_ip_attention_debug = _record_ip_attention_debug_with_channel_means
        for _, _, processor in iter_ip_processors(transformer):
            processors.append(
                (
                    processor,
                    hasattr(processor, "_ip_debug_collector"),
                    getattr(processor, "_ip_debug_collector", None),
                )
            )
            setattr(processor, "_ip_debug_collector", collector)
    try:
        yield
    finally:
        for processor, had_value, old_value in processors:
            if had_value:
                setattr(processor, "_ip_debug_collector", old_value)
            elif hasattr(processor, "_ip_debug_collector"):
                delattr(processor, "_ip_debug_collector")
        if patched_module is not None and original_recorder is not None:
            patched_module._record_ip_attention_debug = original_recorder


def _record_ip_attention_debug_with_channel_means(
    collector: dict | None,
    block_name: str,
    hidden_states: torch.Tensor,
    scaled_ip_output: torch.Tensor,
    debug_payload: dict[str, object] | None = None,
) -> None:
    """Inference-only recorder that adds per-block channel means to the existing debug payload."""
    if collector is None:
        return
    with torch.no_grad():
        hidden = hidden_states.detach().float()
        ip_output = scaled_ip_output.detach().float()
        hidden_plus_ip = hidden + ip_output if hidden.shape == ip_output.shape else None
        hidden_norm = float(torch.linalg.vector_norm(hidden).item()) if hidden.numel() else 0.0
        ip_norm = float(torch.linalg.vector_norm(ip_output).item()) if ip_output.numel() else 0.0
        hidden_plus_ip_norm = tensor_l2_norm(hidden_plus_ip)
        ratio = ip_norm / max(hidden_norm, 1e-12)
        records = collector.setdefault("records", [])
        call_index = len(records)
        scalar_debug = {
            key: value
            for key, value in (debug_payload or {}).items()
            if not torch.is_tensor(value)
        }
        records.append(
            {
                "block": str(block_name),
                "call_index": int(call_index),
                "hidden_norm": hidden_norm,
                "ip_norm": ip_norm,
                "ratio": ratio,
                **scalar_debug,
            }
        )
        block_record = {
                "block": str(block_name),
                "call_index": int(call_index),
                "shape": [int(value) for value in ip_output.shape],
                "hidden_shape": [int(value) for value in hidden.shape],
                "ip_output": ip_output.cpu(),
                "hidden": hidden.cpu(),
                "hidden_plus_ip": hidden_plus_ip.cpu() if torch.is_tensor(hidden_plus_ip) else None,
                "channel_mean": channel_mean(ip_output),
                "channel_std": channel_std(ip_output),
                "hidden_channel_mean": channel_mean(hidden),
                "hidden_channel_std": channel_std(hidden),
                "hidden_plus_ip_channel_mean": channel_mean(hidden_plus_ip),
                "hidden_plus_ip_channel_std": channel_std(hidden_plus_ip),
                "scalar_mean": tensor_scalar_mean(ip_output),
                "scalar_abs_mean": tensor_scalar_abs_mean(ip_output),
                "hidden_scalar_mean": tensor_scalar_mean(hidden),
                "hidden_scalar_abs_mean": tensor_scalar_abs_mean(hidden),
                "hidden_plus_ip_scalar_mean": tensor_scalar_mean(hidden_plus_ip),
                "hidden_plus_ip_scalar_abs_mean": tensor_scalar_abs_mean(hidden_plus_ip),
                "hidden_norm": hidden_norm,
                "ip_norm": ip_norm,
                "hidden_plus_ip_norm": hidden_plus_ip_norm,
                "ratio": ratio,
                **scalar_debug,
            }
        for key in ("uniform_ip_output", "label_uniform_ip_output", "attention_key_mass"):
            value = (debug_payload or {}).get(key)
            if torch.is_tensor(value):
                block_record[key] = value.detach().float().cpu()
        collector.setdefault("block_outputs", []).append(block_record)
        should_store = bool(collector.get("store_first_ip_output", False))
        is_double_block = str(block_name).startswith("block_") or str(block_name) == "block"
        if should_store and is_double_block and "first_ip_output" not in collector:
            collector["first_ip_output"] = ip_output.cpu()
            collector["first_ip_block"] = str(block_name)


def summarize_attention_pair(
    *,
    paired_collector: dict[str, Any],
    alternate_collector: dict[str, Any],
    metadata_index: int,
    sample_id: str,
    paired_reference_sample_id: str,
    alternate_reference_sample_id: str,
    alternate_mode: str,
    regional_ip_soft_bias: float | None,
    regional_ip_soft_bias_applied: bool,
    save_dir: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paired_output = paired_collector.get("first_ip_output")
    alternate_output = alternate_collector.get("first_ip_output")
    output_stats = tensor_pair_stats(paired_output, alternate_output)
    token_stats = token_pair_stats(paired_output, alternate_output)
    paired_mean = channel_mean(paired_output) if torch.is_tensor(paired_output) else None
    alternate_mean = channel_mean(alternate_output) if torch.is_tensor(alternate_output) else None
    mean_stats = tensor_pair_stats(paired_mean, alternate_mean)
    block_rows = collect_attention_block_pair_rows(
        paired_collector=paired_collector,
        alternate_collector=alternate_collector,
        metadata_index=metadata_index,
        sample_id=sample_id,
        paired_reference_sample_id=paired_reference_sample_id,
        alternate_reference_sample_id=alternate_reference_sample_id,
        alternate_mode=alternate_mode,
        regional_ip_soft_bias=regional_ip_soft_bias,
        regional_ip_soft_bias_applied=regional_ip_soft_bias_applied,
    )
    block_cosines = finite_values(block_rows, "channel_mean_cosine")
    block_l1_values = finite_values(block_rows, "channel_mean_l1")
    if save_dir is not None and torch.is_tensor(paired_mean) and torch.is_tensor(alternate_mean):
        torch.save(
            {
                "paired_channel_mean": paired_mean.detach().cpu(),
                "alternate_channel_mean": alternate_mean.detach().cpu(),
                "paired_reference_sample_id": paired_reference_sample_id,
                "alternate_reference_sample_id": alternate_reference_sample_id,
            },
            save_dir / f"{alternate_mode}_attention_channel_means.pt",
        )
    if save_dir is not None:
        save_attention_block_channel_means(
            save_dir / f"{alternate_mode}_attention_block_channel_means.pt",
            paired_collector=paired_collector,
            alternate_collector=alternate_collector,
            paired_reference_sample_id=paired_reference_sample_id,
            alternate_reference_sample_id=alternate_reference_sample_id,
        )
    summary = {
        "metadata_index": int(metadata_index),
        "sample_id": sample_id,
        "paired_reference_sample_id": paired_reference_sample_id,
        "alternate_reference_sample_id": alternate_reference_sample_id,
        "alternate_mode": alternate_mode,
        "regional_ip_soft_bias": regional_ip_soft_bias,
        "regional_ip_soft_bias_applied": bool(regional_ip_soft_bias_applied),
        "first_ip_block_paired": paired_collector.get("first_ip_block"),
        "first_ip_block_alternate": alternate_collector.get("first_ip_block"),
        "ip_attention_record_count_paired": len(paired_collector.get("records", [])),
        "ip_attention_record_count_alternate": len(alternate_collector.get("records", [])),
        "output_numel_match": bool(output_stats["numel_match"]),
        "output_cosine": output_stats["cosine"],
        "output_l1": output_stats["l1"],
        "output_rmse": output_stats["rmse"],
        "token_num_vectors": int(token_stats["num_vectors"]),
        "token_flat_cosine": token_stats["flat_cosine"],
        "token_mean_cosine": token_stats["mean_cosine"],
        "token_min_cosine": token_stats["min_cosine"],
        "token_max_cosine": token_stats["max_cosine"],
        "token_l1": token_stats["l1"],
        "token_rmse": token_stats["rmse"],
        "channel_mean_numel_match": bool(mean_stats["numel_match"]),
        "channel_mean_cosine": mean_stats["cosine"],
        "channel_mean_l1": mean_stats["l1"],
        "channel_mean_rmse": mean_stats["rmse"],
        "token_minus_channel_mean_cosine_gap": (
            token_stats["mean_cosine"] - mean_stats["cosine"]
            if math.isfinite(float(token_stats["mean_cosine"]))
            and math.isfinite(float(mean_stats["cosine"]))
            else math.nan
        ),
        "paired_output_channel_mean_scalar": tensor_scalar_mean(paired_mean),
        "alternate_output_channel_mean_scalar": tensor_scalar_mean(alternate_mean),
        "paired_output_l2": tensor_l2_norm(paired_output),
        "alternate_output_l2": tensor_l2_norm(alternate_output),
        "block_row_count": len(block_rows),
        "block_channel_mean_cosine_mean": safe_mean(block_cosines),
        "block_channel_mean_cosine_min": safe_min(block_cosines),
        "block_channel_mean_cosine_max": safe_max(block_cosines),
        "block_channel_mean_l1_mean": safe_mean(block_l1_values),
        "block_channel_mean_l1_max": safe_max(block_l1_values),
    }
    return summary, block_rows


def collect_attention_block_pair_rows(
    *,
    paired_collector: dict[str, Any],
    alternate_collector: dict[str, Any],
    metadata_index: int,
    sample_id: str,
    paired_reference_sample_id: str,
    alternate_reference_sample_id: str,
    alternate_mode: str,
    regional_ip_soft_bias: float | None,
    regional_ip_soft_bias_applied: bool,
) -> list[dict[str, Any]]:
    paired_outputs = list(paired_collector.get("block_outputs", []))
    alternate_outputs = list(alternate_collector.get("block_outputs", []))
    rows: list[dict[str, Any]] = []
    paired_visit_counts: dict[str, int] = defaultdict(int)
    alternate_visit_counts: dict[str, int] = defaultdict(int)
    for pair_index, (paired_record, alternate_record) in enumerate(zip(paired_outputs, alternate_outputs)):
        paired_block = str(paired_record.get("block"))
        alternate_block = str(alternate_record.get("block"))
        paired_visit_index = paired_visit_counts[paired_block]
        alternate_visit_index = alternate_visit_counts[alternate_block]
        paired_visit_counts[paired_block] += 1
        alternate_visit_counts[alternate_block] += 1
        block_type, block_index = parse_ip_block_name(paired_block)
        mean_stats = tensor_pair_stats(
            paired_record.get("channel_mean"),
            alternate_record.get("channel_mean"),
        )
        std_stats = tensor_pair_stats(
            paired_record.get("channel_std"),
            alternate_record.get("channel_std"),
        )
        hidden_mean_stats = tensor_pair_stats(
            paired_record.get("hidden_channel_mean"),
            alternate_record.get("hidden_channel_mean"),
        )
        hidden_plus_ip_mean_stats = tensor_pair_stats(
            paired_record.get("hidden_plus_ip_channel_mean"),
            alternate_record.get("hidden_plus_ip_channel_mean"),
        )
        hidden_plus_ip_std_stats = tensor_pair_stats(
            paired_record.get("hidden_plus_ip_channel_std"),
            alternate_record.get("hidden_plus_ip_channel_std"),
        )
        token_stats = token_pair_stats(
            paired_record.get("ip_output"),
            alternate_record.get("ip_output"),
        )
        uniform_token_stats = token_pair_stats(
            paired_record.get("uniform_ip_output"),
            alternate_record.get("uniform_ip_output"),
        )
        label_uniform_token_stats = token_pair_stats(
            paired_record.get("label_uniform_ip_output"),
            alternate_record.get("label_uniform_ip_output"),
        )
        paired_ip_vs_uniform_stats = token_pair_stats(
            paired_record.get("ip_output"),
            paired_record.get("uniform_ip_output"),
        )
        alternate_ip_vs_uniform_stats = token_pair_stats(
            alternate_record.get("ip_output"),
            alternate_record.get("uniform_ip_output"),
        )
        paired_ip_vs_label_uniform_stats = token_pair_stats(
            paired_record.get("ip_output"),
            paired_record.get("label_uniform_ip_output"),
        )
        alternate_ip_vs_label_uniform_stats = token_pair_stats(
            alternate_record.get("ip_output"),
            alternate_record.get("label_uniform_ip_output"),
        )
        key_mass_stats = tensor_pair_stats(
            paired_record.get("attention_key_mass"),
            alternate_record.get("attention_key_mass"),
        )
        hidden_token_stats = token_pair_stats(
            paired_record.get("hidden"),
            alternate_record.get("hidden"),
        )
        hidden_plus_ip_token_stats = token_pair_stats(
            paired_record.get("hidden_plus_ip"),
            alternate_record.get("hidden_plus_ip"),
        )
        row = {
                "metadata_index": int(metadata_index),
                "sample_id": sample_id,
                "paired_reference_sample_id": paired_reference_sample_id,
                "alternate_reference_sample_id": alternate_reference_sample_id,
                "alternate_mode": alternate_mode,
                "regional_ip_soft_bias": regional_ip_soft_bias,
                "regional_ip_soft_bias_applied": bool(regional_ip_soft_bias_applied),
                "pair_index": int(pair_index),
                "paired_call_index": int(paired_record.get("call_index", pair_index)),
                "alternate_call_index": int(alternate_record.get("call_index", pair_index)),
                "paired_block": paired_block,
                "alternate_block": alternate_block,
                "block_name_match": paired_block == alternate_block,
                "block_type": block_type,
                "block_index": block_index,
                "block_visit_index": int(paired_visit_index),
                "alternate_block_visit_index": int(alternate_visit_index),
                "paired_shape": str(paired_record.get("shape")),
                "alternate_shape": str(alternate_record.get("shape")),
                "token_num_vectors": int(token_stats["num_vectors"]),
                "token_flat_cosine": token_stats["flat_cosine"],
                "token_mean_cosine": token_stats["mean_cosine"],
                "token_min_cosine": token_stats["min_cosine"],
                "token_max_cosine": token_stats["max_cosine"],
                "token_l1": token_stats["l1"],
                "token_rmse": token_stats["rmse"],
                "uniform_token_num_vectors": int(uniform_token_stats["num_vectors"]),
                "uniform_token_flat_cosine": uniform_token_stats["flat_cosine"],
                "uniform_token_mean_cosine": uniform_token_stats["mean_cosine"],
                "uniform_token_l1": uniform_token_stats["l1"],
                "uniform_token_rmse": uniform_token_stats["rmse"],
                "label_uniform_token_num_vectors": int(label_uniform_token_stats["num_vectors"]),
                "label_uniform_token_flat_cosine": label_uniform_token_stats["flat_cosine"],
                "label_uniform_token_mean_cosine": label_uniform_token_stats["mean_cosine"],
                "label_uniform_token_l1": label_uniform_token_stats["l1"],
                "label_uniform_token_rmse": label_uniform_token_stats["rmse"],
                "paired_ip_vs_uniform_flat_cosine": paired_ip_vs_uniform_stats["flat_cosine"],
                "alternate_ip_vs_uniform_flat_cosine": alternate_ip_vs_uniform_stats["flat_cosine"],
                "paired_ip_vs_label_uniform_flat_cosine": paired_ip_vs_label_uniform_stats["flat_cosine"],
                "alternate_ip_vs_label_uniform_flat_cosine": alternate_ip_vs_label_uniform_stats["flat_cosine"],
                "attention_key_mass_numel_match": bool(key_mass_stats["numel_match"]),
                "attention_key_mass_cosine": key_mass_stats["cosine"],
                "attention_key_mass_l1": key_mass_stats["l1"],
                "hidden_token_num_vectors": int(hidden_token_stats["num_vectors"]),
                "hidden_token_flat_cosine": hidden_token_stats["flat_cosine"],
                "hidden_token_mean_cosine": hidden_token_stats["mean_cosine"],
                "hidden_token_min_cosine": hidden_token_stats["min_cosine"],
                "hidden_token_max_cosine": hidden_token_stats["max_cosine"],
                "hidden_token_l1": hidden_token_stats["l1"],
                "hidden_token_rmse": hidden_token_stats["rmse"],
                "hidden_plus_ip_token_num_vectors": int(hidden_plus_ip_token_stats["num_vectors"]),
                "hidden_plus_ip_token_flat_cosine": hidden_plus_ip_token_stats["flat_cosine"],
                "hidden_plus_ip_token_mean_cosine": hidden_plus_ip_token_stats["mean_cosine"],
                "hidden_plus_ip_token_min_cosine": hidden_plus_ip_token_stats["min_cosine"],
                "hidden_plus_ip_token_max_cosine": hidden_plus_ip_token_stats["max_cosine"],
                "hidden_plus_ip_token_l1": hidden_plus_ip_token_stats["l1"],
                "hidden_plus_ip_token_rmse": hidden_plus_ip_token_stats["rmse"],
                "channel_mean_numel_match": bool(mean_stats["numel_match"]),
                "channel_mean_cosine": mean_stats["cosine"],
                "channel_mean_l1": mean_stats["l1"],
                "channel_mean_rmse": mean_stats["rmse"],
                "channel_std_cosine": std_stats["cosine"],
                "channel_std_l1": std_stats["l1"],
                "hidden_channel_mean_cosine": hidden_mean_stats["cosine"],
                "hidden_channel_mean_l1": hidden_mean_stats["l1"],
                "hidden_plus_ip_channel_mean_cosine": hidden_plus_ip_mean_stats["cosine"],
                "hidden_plus_ip_channel_mean_l1": hidden_plus_ip_mean_stats["l1"],
                "hidden_plus_ip_channel_mean_rmse": hidden_plus_ip_mean_stats["rmse"],
                "hidden_plus_ip_channel_std_cosine": hidden_plus_ip_std_stats["cosine"],
                "hidden_plus_ip_channel_std_l1": hidden_plus_ip_std_stats["l1"],
                "paired_channel_mean_scalar": tensor_scalar_mean(paired_record.get("channel_mean")),
                "alternate_channel_mean_scalar": tensor_scalar_mean(alternate_record.get("channel_mean")),
                "paired_hidden_channel_mean_scalar": tensor_scalar_mean(paired_record.get("hidden_channel_mean")),
                "alternate_hidden_channel_mean_scalar": tensor_scalar_mean(alternate_record.get("hidden_channel_mean")),
                "paired_hidden_plus_ip_channel_mean_scalar": tensor_scalar_mean(
                    paired_record.get("hidden_plus_ip_channel_mean")
                ),
                "alternate_hidden_plus_ip_channel_mean_scalar": tensor_scalar_mean(
                    alternate_record.get("hidden_plus_ip_channel_mean")
                ),
                "paired_scalar_mean": float(paired_record.get("scalar_mean", math.nan)),
                "alternate_scalar_mean": float(alternate_record.get("scalar_mean", math.nan)),
                "paired_scalar_abs_mean": float(paired_record.get("scalar_abs_mean", math.nan)),
                "alternate_scalar_abs_mean": float(alternate_record.get("scalar_abs_mean", math.nan)),
                "paired_hidden_scalar_mean": float(paired_record.get("hidden_scalar_mean", math.nan)),
                "alternate_hidden_scalar_mean": float(alternate_record.get("hidden_scalar_mean", math.nan)),
                "paired_hidden_plus_ip_scalar_mean": float(
                    paired_record.get("hidden_plus_ip_scalar_mean", math.nan)
                ),
                "alternate_hidden_plus_ip_scalar_mean": float(
                    alternate_record.get("hidden_plus_ip_scalar_mean", math.nan)
                ),
                "paired_ip_norm": float(paired_record.get("ip_norm", math.nan)),
                "alternate_ip_norm": float(alternate_record.get("ip_norm", math.nan)),
                "paired_hidden_norm": float(paired_record.get("hidden_norm", math.nan)),
                "alternate_hidden_norm": float(alternate_record.get("hidden_norm", math.nan)),
                "paired_hidden_plus_ip_norm": float(paired_record.get("hidden_plus_ip_norm", math.nan)),
                "alternate_hidden_plus_ip_norm": float(alternate_record.get("hidden_plus_ip_norm", math.nan)),
                "paired_ratio": float(paired_record.get("ratio", math.nan)),
                "alternate_ratio": float(alternate_record.get("ratio", math.nan)),
            }
        for key in (
            "attention_branch_count",
            "attention_allowed_tokens_per_query_mean",
            "attention_allowed_tokens_per_query_min",
            "attention_allowed_tokens_per_query_max",
            "attention_entropy_mean",
            "attention_entropy_normalized_mean",
            "attention_effective_tokens_mean",
            "attention_max_weight_mean",
            "attention_max_weight_max",
            "attention_tv_from_uniform_mean",
            "attention_prob_uniform_cosine",
            "attention_output_uniform_flat_cosine",
            "attention_output_label_uniform_flat_cosine",
            "attention_label_uniform_token_fraction",
            "attention_key_mass_entropy",
            "attention_key_mass_effective_tokens",
            "attention_key_mass_top1",
        ):
            row[f"paired_{key}"] = float(paired_record.get(key, math.nan))
            row[f"alternate_{key}"] = float(alternate_record.get(key, math.nan))
        rows.append(row)
    if len(paired_outputs) != len(alternate_outputs):
        rows.append(
            {
                "metadata_index": int(metadata_index),
                "sample_id": sample_id,
                "paired_reference_sample_id": paired_reference_sample_id,
                "alternate_reference_sample_id": alternate_reference_sample_id,
                "alternate_mode": alternate_mode,
                "regional_ip_soft_bias": regional_ip_soft_bias,
                "regional_ip_soft_bias_applied": bool(regional_ip_soft_bias_applied),
                "pair_index": len(rows),
                "paired_block": "__unmatched_count__",
                "alternate_block": "__unmatched_count__",
                "block_name_match": False,
                "block_type": "unmatched",
                "block_index": -1,
                "block_visit_index": -1,
                "paired_record_count": len(paired_outputs),
                "alternate_record_count": len(alternate_outputs),
            }
        )
    return rows


def save_attention_block_channel_means(
    path: Path,
    *,
    paired_collector: dict[str, Any],
    alternate_collector: dict[str, Any],
    paired_reference_sample_id: str,
    alternate_reference_sample_id: str,
) -> None:
    def pack(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        packed = []
        for record in records:
            packed.append(
                {
                    "block": record.get("block"),
                    "call_index": record.get("call_index"),
                    "shape": record.get("shape"),
                    "channel_mean": (
                        record["channel_mean"].detach().cpu()
                        if torch.is_tensor(record.get("channel_mean"))
                        else None
                    ),
                    "channel_std": (
                        record["channel_std"].detach().cpu()
                        if torch.is_tensor(record.get("channel_std"))
                        else None
                    ),
                }
            )
        return packed

    torch.save(
        {
            "paired_reference_sample_id": paired_reference_sample_id,
            "alternate_reference_sample_id": alternate_reference_sample_id,
            "paired": pack(list(paired_collector.get("block_outputs", []))),
            "alternate": pack(list(alternate_collector.get("block_outputs", []))),
        },
        path,
    )


def parse_ip_block_name(block_name: str) -> tuple[str, int]:
    if block_name.startswith("single_block_"):
        suffix = block_name.removeprefix("single_block_")
        return "single", int(suffix) if suffix.isdigit() else -1
    if block_name.startswith("block_"):
        suffix = block_name.removeprefix("block_")
        return "double", int(suffix) if suffix.isdigit() else -1
    if block_name == "block":
        return "double", -1
    if block_name == "single_block":
        return "single", -1
    return "unknown", -1


def summarize_projection_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    value_cosines = finite_values(rows, "value_cosine")
    input_cosines = finite_values(rows, "input_cosine")
    collapse_rows = [
        row for row in rows
        if math.isfinite(float(row["input_cosine"]))
        and math.isfinite(float(row["value_cosine"]))
        and float(row["input_cosine"]) <= float(args.token_separated_max_cosine)
        and float(row["value_cosine"]) >= float(args.v_collapse_min_cosine)
    ]
    return {
        "num_rows": len(rows),
        "input_cosine_mean": safe_mean(input_cosines),
        "input_cosine_min": safe_min(input_cosines),
        "input_cosine_max": safe_max(input_cosines),
        "value_cosine_mean": safe_mean(value_cosines),
        "value_cosine_min": safe_min(value_cosines),
        "value_cosine_max": safe_max(value_cosines),
        "collapse_candidate_rows": len(collapse_rows),
    }


def summarize_all_projection_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    summary = summarize_projection_rows(rows, args)
    by_block_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_block_type[str(row["block_type"])].append(row)
    summary["by_block_type"] = {
        key: summarize_projection_rows(value, args)
        for key, value in sorted(by_block_type.items())
    }
    return summary


def summarize_v_projection_pairwise_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    input_cosines = finite_values(rows, "input_cosine")
    input_token_flat_cosines = finite_values(rows, "input_token_flat_cosine")
    input_token_mean_cosines = finite_values(rows, "input_token_mean_cosine")
    value_cosines = finite_values(rows, "value_cosine")
    value_token_flat_cosines = finite_values(rows, "value_token_flat_cosine")
    value_token_mean_cosines = finite_values(rows, "value_token_mean_cosine")
    value_channel_mean_cosines = finite_values(rows, "value_channel_mean_cosine")
    return {
        "num_rows": len(rows),
        "same_parent_probe_rows": sum(1 for row in rows if bool(row.get("same_parent_probe", False))),
        "same_reference_sample_rows": sum(1 for row in rows if bool(row.get("same_reference_sample", False))),
        "same_reference_case_rows": sum(1 for row in rows if bool(row.get("same_reference_case", False))),
        "input_cosine_mean": safe_mean(input_cosines),
        "input_cosine_min": safe_min(input_cosines),
        "input_cosine_max": safe_max(input_cosines),
        "input_token_flat_cosine_mean": safe_mean(input_token_flat_cosines),
        "input_token_flat_cosine_min": safe_min(input_token_flat_cosines),
        "input_token_flat_cosine_max": safe_max(input_token_flat_cosines),
        "input_token_mean_cosine_mean": safe_mean(input_token_mean_cosines),
        "input_token_mean_cosine_min": safe_min(input_token_mean_cosines),
        "input_token_mean_cosine_max": safe_max(input_token_mean_cosines),
        "value_cosine_mean": safe_mean(value_cosines),
        "value_cosine_min": safe_min(value_cosines),
        "value_cosine_max": safe_max(value_cosines),
        "value_token_flat_cosine_mean": safe_mean(value_token_flat_cosines),
        "value_token_flat_cosine_min": safe_min(value_token_flat_cosines),
        "value_token_flat_cosine_max": safe_max(value_token_flat_cosines),
        "value_token_mean_cosine_mean": safe_mean(value_token_mean_cosines),
        "value_token_mean_cosine_min": safe_min(value_token_mean_cosines),
        "value_token_mean_cosine_max": safe_max(value_token_mean_cosines),
        "value_channel_mean_cosine_mean": safe_mean(value_channel_mean_cosines),
        "value_channel_mean_cosine_min": safe_min(value_channel_mean_cosines),
        "value_channel_mean_cosine_max": safe_max(value_channel_mean_cosines),
    }


def summarize_all_v_projection_pairwise_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_v_projection_pairwise_rows(rows)
    group_specs = {
        "by_role_pair": "role_pair",
        "by_sample_relation": "sample_relation",
        "by_case_relation": "case_relation",
        "by_block_type": "block_type",
    }
    for output_key, row_key in group_specs.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row.get(row_key, "unknown"))].append(row)
        summary[output_key] = {
            key: summarize_v_projection_pairwise_rows(value)
            for key, value in sorted(groups.items())
        }

    by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_block[str(row.get("block", "unknown"))].append(row)
    summary["by_block"] = {
        key: summarize_v_projection_pairwise_rows(value)
        for key, value in sorted(by_block.items(), key=lambda item: block_sort_key(item[0]))
    }
    return summary


def summarize_all_attention_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    token_cosines = finite_values(rows, "token_mean_cosine")
    channel_cosines = finite_values(rows, "channel_mean_cosine")
    unchanged = [
        row for row in rows
        if math.isfinite(float(row.get("channel_mean_cosine", math.nan)))
        and float(row["channel_mean_cosine"]) >= float(args.mean_unchanged_min_cosine)
    ]
    return {
        "num_rows": len(rows),
        "token_mean_cosine_mean": safe_mean(token_cosines),
        "token_mean_cosine_min": safe_min(token_cosines),
        "token_mean_cosine_max": safe_max(token_cosines),
        "channel_mean_cosine_mean": safe_mean(channel_cosines),
        "channel_mean_cosine_min": safe_min(channel_cosines),
        "channel_mean_cosine_max": safe_max(channel_cosines),
        "token_minus_channel_mean_cosine_gap_mean": safe_mean(
            finite_values(rows, "token_minus_channel_mean_cosine_gap")
        ),
        "mean_unchanged_candidate_rows": len(unchanged),
    }


def summarize_all_attention_block_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    channel_cosines = finite_values(rows, "channel_mean_cosine")
    channel_l1_values = finite_values(rows, "channel_mean_l1")
    token_flat_cosines = finite_values(rows, "token_flat_cosine")
    token_cosines = finite_values(rows, "token_mean_cosine")
    hidden_token_flat_cosines = finite_values(rows, "hidden_token_flat_cosine")
    hidden_plus_ip_token_flat_cosines = finite_values(rows, "hidden_plus_ip_token_flat_cosine")
    hidden_plus_ip_cosines = finite_values(rows, "hidden_plus_ip_channel_mean_cosine")
    hidden_plus_ip_l1_values = finite_values(rows, "hidden_plus_ip_channel_mean_l1")
    unchanged = [
        row for row in rows
        if math.isfinite(float(row.get("channel_mean_cosine", math.nan)))
        and float(row["channel_mean_cosine"]) >= float(args.mean_unchanged_min_cosine)
    ]
    by_block_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        block_type = str(row.get("block_type", "unknown"))
        block_name = str(row.get("paired_block", row.get("block", "unknown")))
        by_block_type[block_type].append(row)
        by_block[block_name].append(row)
    return {
        "num_rows": len(rows),
        "token_flat_cosine_mean": safe_mean(token_flat_cosines),
        "token_flat_cosine_min": safe_min(token_flat_cosines),
        "token_flat_cosine_max": safe_max(token_flat_cosines),
        "token_mean_cosine_mean": safe_mean(token_cosines),
        "token_mean_cosine_min": safe_min(token_cosines),
        "token_mean_cosine_max": safe_max(token_cosines),
        "hidden_token_flat_cosine_mean": safe_mean(hidden_token_flat_cosines),
        "hidden_token_flat_cosine_min": safe_min(hidden_token_flat_cosines),
        "hidden_token_flat_cosine_max": safe_max(hidden_token_flat_cosines),
        "hidden_plus_ip_token_flat_cosine_mean": safe_mean(hidden_plus_ip_token_flat_cosines),
        "hidden_plus_ip_token_flat_cosine_min": safe_min(hidden_plus_ip_token_flat_cosines),
        "hidden_plus_ip_token_flat_cosine_max": safe_max(hidden_plus_ip_token_flat_cosines),
        "channel_mean_cosine_mean": safe_mean(channel_cosines),
        "channel_mean_cosine_min": safe_min(channel_cosines),
        "channel_mean_cosine_max": safe_max(channel_cosines),
        "channel_mean_l1_mean": safe_mean(channel_l1_values),
        "channel_mean_l1_max": safe_max(channel_l1_values),
        "hidden_plus_ip_channel_mean_cosine_mean": safe_mean(hidden_plus_ip_cosines),
        "hidden_plus_ip_channel_mean_cosine_min": safe_min(hidden_plus_ip_cosines),
        "hidden_plus_ip_channel_mean_cosine_max": safe_max(hidden_plus_ip_cosines),
        "hidden_plus_ip_channel_mean_l1_mean": safe_mean(hidden_plus_ip_l1_values),
        "hidden_plus_ip_channel_mean_l1_max": safe_max(hidden_plus_ip_l1_values),
        "mean_unchanged_candidate_rows": len(unchanged),
        "by_block_type": {
            key: summarize_attention_block_group(value, args)
            for key, value in sorted(by_block_type.items())
        },
        "by_block": {
            key: summarize_attention_block_group(value, args)
            for key, value in sorted(by_block.items(), key=lambda item: block_sort_key(item[0]))
        },
    }


def summarize_attention_block_group(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    channel_cosines = finite_values(rows, "channel_mean_cosine")
    channel_l1_values = finite_values(rows, "channel_mean_l1")
    token_flat_cosines = finite_values(rows, "token_flat_cosine")
    token_cosines = finite_values(rows, "token_mean_cosine")
    hidden_token_flat_cosines = finite_values(rows, "hidden_token_flat_cosine")
    hidden_plus_ip_token_flat_cosines = finite_values(rows, "hidden_plus_ip_token_flat_cosine")
    hidden_plus_ip_cosines = finite_values(rows, "hidden_plus_ip_channel_mean_cosine")
    hidden_plus_ip_l1_values = finite_values(rows, "hidden_plus_ip_channel_mean_l1")
    unchanged = [
        row for row in rows
        if math.isfinite(float(row.get("channel_mean_cosine", math.nan)))
        and float(row["channel_mean_cosine"]) >= float(args.mean_unchanged_min_cosine)
    ]
    return {
        "num_rows": len(rows),
        "token_flat_cosine_mean": safe_mean(token_flat_cosines),
        "token_flat_cosine_min": safe_min(token_flat_cosines),
        "token_flat_cosine_max": safe_max(token_flat_cosines),
        "token_mean_cosine_mean": safe_mean(token_cosines),
        "token_mean_cosine_min": safe_min(token_cosines),
        "token_mean_cosine_max": safe_max(token_cosines),
        "hidden_token_flat_cosine_mean": safe_mean(hidden_token_flat_cosines),
        "hidden_token_flat_cosine_min": safe_min(hidden_token_flat_cosines),
        "hidden_token_flat_cosine_max": safe_max(hidden_token_flat_cosines),
        "hidden_plus_ip_token_flat_cosine_mean": safe_mean(hidden_plus_ip_token_flat_cosines),
        "hidden_plus_ip_token_flat_cosine_min": safe_min(hidden_plus_ip_token_flat_cosines),
        "hidden_plus_ip_token_flat_cosine_max": safe_max(hidden_plus_ip_token_flat_cosines),
        "channel_mean_cosine_mean": safe_mean(channel_cosines),
        "channel_mean_cosine_min": safe_min(channel_cosines),
        "channel_mean_cosine_max": safe_max(channel_cosines),
        "channel_mean_l1_mean": safe_mean(channel_l1_values),
        "channel_mean_l1_max": safe_max(channel_l1_values),
        "hidden_plus_ip_channel_mean_cosine_mean": safe_mean(hidden_plus_ip_cosines),
        "hidden_plus_ip_channel_mean_cosine_min": safe_min(hidden_plus_ip_cosines),
        "hidden_plus_ip_channel_mean_cosine_max": safe_max(hidden_plus_ip_cosines),
        "hidden_plus_ip_channel_mean_l1_mean": safe_mean(hidden_plus_ip_l1_values),
        "hidden_plus_ip_channel_mean_l1_max": safe_max(hidden_plus_ip_l1_values),
        "mean_unchanged_candidate_rows": len(unchanged),
    }


def block_sort_key(block_name: str) -> tuple[int, int, str]:
    block_type, block_index = parse_ip_block_name(block_name)
    type_rank = {"double": 0, "single": 1}.get(block_type, 2)
    return type_rank, int(block_index), block_name


def tensor_pair_stats(left: Any, right: Any) -> dict[str, Any]:
    left_tensor = flatten_tensor(left)
    right_tensor = flatten_tensor(right)
    if left_tensor is None or right_tensor is None:
        return empty_pair_stats()
    if left_tensor.numel() == 0 or right_tensor.numel() == 0 or left_tensor.numel() != right_tensor.numel():
        stats = empty_pair_stats()
        stats["left_numel"] = int(left_tensor.numel())
        stats["right_numel"] = int(right_tensor.numel())
        return stats
    left_tensor = left_tensor.float()
    right_tensor = right_tensor.float()
    diff = left_tensor - right_tensor
    return {
        "left_numel": int(left_tensor.numel()),
        "right_numel": int(right_tensor.numel()),
        "numel_match": True,
        "cosine": float(torch.nn.functional.cosine_similarity(left_tensor[None], right_tensor[None]).item()),
        "l1": float(diff.abs().mean().item()),
        "rmse": float(torch.sqrt(torch.mean(diff * diff)).item()),
        "l2": float(torch.linalg.vector_norm(diff).item()),
    }


def token_pair_stats(left: Any, right: Any) -> dict[str, Any]:
    left_tensor = _coerce_tensor(left)
    right_tensor = _coerce_tensor(right)
    if left_tensor is None or right_tensor is None:
        return empty_token_pair_stats()
    if left_tensor.shape != right_tensor.shape or left_tensor.ndim < 2:
        stats = empty_token_pair_stats()
        stats["left_shape"] = list(left_tensor.shape)
        stats["right_shape"] = list(right_tensor.shape)
        return stats
    left_tokens = left_tensor.detach().float().reshape(-1, left_tensor.shape[-1]).cpu()
    right_tokens = right_tensor.detach().float().reshape(-1, right_tensor.shape[-1]).cpu()
    if left_tokens.numel() == 0 or right_tokens.numel() == 0 or left_tokens.shape != right_tokens.shape:
        stats = empty_token_pair_stats()
        stats["left_shape"] = list(left_tensor.shape)
        stats["right_shape"] = list(right_tensor.shape)
        return stats
    token_cosines = torch.nn.functional.cosine_similarity(left_tokens, right_tokens, dim=-1)
    diff = left_tokens - right_tokens
    flat_cosine = float(
        torch.nn.functional.cosine_similarity(left_tokens.reshape(1, -1), right_tokens.reshape(1, -1)).item()
    )
    return {
        "num_vectors": int(token_cosines.numel()),
        "flat_cosine": flat_cosine,
        "mean_cosine": float(token_cosines.mean().item()),
        "min_cosine": float(token_cosines.min().item()),
        "max_cosine": float(token_cosines.max().item()),
        "l1": float(diff.abs().mean().item()),
        "rmse": float(torch.sqrt(torch.mean(diff * diff)).item()),
    }


def empty_pair_stats() -> dict[str, Any]:
    return {
        "left_numel": 0,
        "right_numel": 0,
        "numel_match": False,
        "cosine": math.nan,
        "l1": math.nan,
        "rmse": math.nan,
        "l2": math.nan,
    }


def empty_token_pair_stats() -> dict[str, Any]:
    return {
        "num_vectors": 0,
        "flat_cosine": math.nan,
        "mean_cosine": math.nan,
        "min_cosine": math.nan,
        "max_cosine": math.nan,
        "l1": math.nan,
        "rmse": math.nan,
    }


def flatten_tensors(values: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
    if not values:
        return torch.empty(0)
    return torch.cat([value.detach().float().reshape(-1).cpu() for value in values], dim=0)


def flatten_tensor(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().float().reshape(-1).cpu()
    return torch.as_tensor(value, dtype=torch.float32).reshape(-1).cpu()


def _coerce_tensor(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value
    return torch.as_tensor(value)


def channel_mean(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None or not torch.is_tensor(value) or value.numel() == 0:
        return None
    if value.ndim == 1:
        return value.detach().float().cpu()
    return value.detach().float().mean(dim=tuple(range(value.ndim - 1))).cpu()


def channel_std(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None or not torch.is_tensor(value) or value.numel() == 0:
        return None
    if value.ndim == 1:
        return torch.zeros_like(value.detach().float().cpu())
    return value.detach().float().std(dim=tuple(range(value.ndim - 1)), unbiased=False).cpu()


def tensor_l2_norm(value: Any) -> float:
    tensor = flatten_tensor(value)
    if tensor is None or tensor.numel() == 0:
        return math.nan
    return float(torch.linalg.vector_norm(tensor.float()).item())


def tensor_scalar_mean(value: Any) -> float:
    tensor = flatten_tensor(value)
    if tensor is None or tensor.numel() == 0:
        return math.nan
    return float(tensor.float().mean().item())


def tensor_scalar_abs_mean(value: Any) -> float:
    tensor = flatten_tensor(value)
    if tensor is None or tensor.numel() == 0:
        return math.nan
    return float(tensor.float().abs().mean().item())


def resize_image_tensor(image: torch.Tensor, hw: torch.Size | tuple[int, int]) -> torch.Tensor:
    if tuple(image.shape[1:]) == tuple(hw):
        return image
    resized = torch.nn.functional.interpolate(
        image.unsqueeze(0),
        size=tuple(int(v) for v in hw),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return resized.clamp(0.0, 1.0).contiguous()


def resize_mask_tensor(mask: torch.Tensor, hw: torch.Size | tuple[int, int]) -> torch.Tensor:
    if tuple(mask.shape) == tuple(hw):
        return mask
    resized = torch.nn.functional.interpolate(
        mask.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        size=tuple(int(v) for v in hw),
        mode="nearest",
    ).squeeze(0).squeeze(0)
    return resized.to(dtype=torch.long).contiguous()


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def safe_min(values: list[float]) -> float:
    return float(min(values)) if values else math.nan


def safe_max(values: list[float]) -> float:
    return float(max(values)) if values else math.nan


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_probe_line(summary: dict[str, Any]) -> str:
    token_cos = float(summary["ref_token_pair"]["cosine"])
    v_summary = summary["v_projection"]
    line = (
        f"[A] {summary['sample_id']} alt={summary['alternate_mode']} "
        f"token_cos={token_cos:.4f} "
        f"V_cos_mean={float(v_summary['value_cosine_mean']):.4f} "
        f"V_cos_min={float(v_summary['value_cosine_min']):.4f}"
    )
    attention = summary.get("attention_output")
    if attention is not None:
        line += (
            f" | [B] first={attention.get('first_ip_block_paired')} "
            f"flat_cos={float(attention['token_flat_cosine']):.4f} "
            f"token_cos={float(attention['token_mean_cosine']):.4f} "
            f"cent_cos={float(attention['channel_mean_cosine']):.4f} "
            f"mean_l1={float(attention['channel_mean_l1']):.6g}"
        )
    return line


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)[:120]


if __name__ == "__main__":
    raise SystemExit(main())

"""Diagnose whether Cross V3 cross-attention retrieves the correct reference texture.

This assumes z_ref itself has already been validated by the VAE reconstruction
diagnostic. For each anchor reference image, it compares one-step denoising with:

    correct_ref tokens, mismatched_ref tokens, and zero tokens

The noisy latent is always from the anchor image. If cross-attention learned to
retrieve reference texture, correct_ref should reduce denoising loss compared
with mismatched_ref and zero tokens, especially at higher timesteps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


ZERO_TOKENS_VARIANT = "zero_tokens"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross V3 correct-vs-mismatch reference retrieval diagnostic.")
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--checkpoint", required=True, help="Cross V3 checkpoint dir.")
    parser.add_argument("--metadata", required=True, help="metadata_cross_{train,val}.json path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-anchors", type=int, default=8)
    parser.add_argument("--num-mismatches", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anchor-reference-sample-id", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--timesteps", default="500,700,900")
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--thumbnail-size", type=int, default=192)
    parser.add_argument("--overview-max-samples", type=int, default=32)
    return parser


def parse_args(args=None) -> argparse.Namespace:
    return build_parser().parse_args(args)


def main(argv=None) -> int:
    args = parse_args(argv)

    from controlnet_train.cli.eval_controlnet_flux_cross import (
        _safe_name,
        read_cross_metadata,
    )
    from controlnet_train.data.common import load_image_tensor
    from scripts.diagnose_cross_v3_z_ref_reconstruction import (
        fixed_timestep_reconstructions_reference_tokens_only,
        load_cross_v3_z_ref_only_bundle,
        parse_fixed_t_eval_timesteps,
    )
    from controlnet_train.inference.pipeline_cross_v3 import CROSS_V3_PROMPT

    import torch

    dtype_by_name = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    records = unique_reference_records(read_cross_metadata(args.metadata))
    anchors = select_anchor_records(
        records,
        anchor_reference_sample_ids=args.anchor_reference_sample_id,
        num_anchors=args.num_anchors,
        seed=args.seed,
    )
    timesteps = parse_fixed_t_eval_timesteps(args.timesteps)
    rng = random.Random(args.seed)

    bundle = load_cross_v3_z_ref_only_bundle(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        checkpoint_path=args.checkpoint,
        device=args.device,
        torch_dtype=dtype_by_name[args.torch_dtype],
        num_inference_steps=1,
        guidance_scale=args.guidance_scale,
    )

    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    panel_paths: list[Path] = []

    token_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, Image.Image]] = {}
    with torch.no_grad():
        for anchor_index, anchor in enumerate(anchors):
            anchor_id = reference_record_id(anchor)
            anchor_image_tensor = load_image_tensor(anchor["reference_image"])
            anchor_image_pil = Image.open(anchor["reference_image"]).convert("RGB")
            correct_tokens, _, _, _ = get_cached_tokens(
                bundle=bundle,
                record=anchor,
                token_cache=token_cache,
                load_image_tensor=load_image_tensor,
            )
            zero_tokens = torch.zeros_like(correct_tokens)
            correct_results = fixed_timestep_reconstructions_reference_tokens_only(
                bundle=bundle,
                reference_image=anchor_image_tensor,
                reference_tokens=correct_tokens,
                prompt=CROSS_V3_PROMPT,
                timesteps=timesteps,
                seed=int(args.noise_seed) + anchor_index,
            )
            zero_results = fixed_timestep_reconstructions_reference_tokens_only(
                bundle=bundle,
                reference_image=anchor_image_tensor,
                reference_tokens=zero_tokens,
                prompt=CROSS_V3_PROMPT,
                timesteps=timesteps,
                seed=int(args.noise_seed) + anchor_index,
            )
            preview_key = select_preview_key_for_retrieval(correct_results)

            mismatches = select_mismatch_records(
                records,
                anchor=anchor,
                num_mismatches=args.num_mismatches,
                rng=rng,
            )
            for mismatch_index, mismatch in enumerate(mismatches):
                mismatch_id = reference_record_id(mismatch)
                mismatch_tokens, _, _, mismatch_image_pil = get_cached_tokens(
                    bundle=bundle,
                    record=mismatch,
                    token_cache=token_cache,
                    load_image_tensor=load_image_tensor,
                )
                mismatch_results = fixed_timestep_reconstructions_reference_tokens_only(
                    bundle=bundle,
                    reference_image=anchor_image_tensor,
                    reference_tokens=mismatch_tokens,
                    prompt=CROSS_V3_PROMPT,
                    timesteps=timesteps,
                    seed=int(args.noise_seed) + anchor_index,
                )
                row = build_retrieval_row(
                    anchor_index=anchor_index,
                    mismatch_index=mismatch_index,
                    anchor=anchor,
                    mismatch=mismatch,
                    correct_results=correct_results,
                    mismatch_results=mismatch_results,
                    zero_results=zero_results,
                    preview_key=preview_key,
                )
                rows.append(row)

                sample_dir = samples_dir / (
                    f"{anchor_index:04d}_{mismatch_index:02d}_anchor_{_safe_name(anchor_id)}"
                    f"__mismatch_{_safe_name(mismatch_id)}"
                )
                sample_dir.mkdir(parents=True, exist_ok=True)
                panel = make_retrieval_panel(
                    anchor=anchor_image_pil,
                    mismatch=mismatch_image_pil,
                    correct=correct_results[preview_key]["image"],
                    mismatch_prediction=mismatch_results[preview_key]["image"],
                    zero=zero_results[preview_key]["image"],
                    thumbnail_size=args.thumbnail_size,
                    title=f"anchor={anchor_id} | mismatch={mismatch_id} | {preview_key}",
                )
                panel_path = sample_dir / "panel.png"
                panel.save(panel_path)
                if len(panel_paths) < args.overview_max_samples:
                    panel_paths.append(panel_path)
                (sample_dir / "metrics.json").write_text(
                    json.dumps(row, indent=2, ensure_ascii=False, allow_nan=True),
                    encoding="utf8",
                )
                print(
                    f"[{len(rows)}] anchor={anchor_id} mismatch={mismatch_id} "
                    f"{preview_key}_mismatch_minus_correct={row[f'mismatch_minus_correct_loss_{preview_key}']:.4f} "
                    f"{preview_key}_zero_minus_correct={row[f'zero_minus_correct_loss_{preview_key}']:.4f} "
                    f"correct_vs_mismatch_pred_rel={row[f'noise_pred_correct_vs_mismatch_relative_l2_{preview_key}']:.4f}"
                )

    write_rows(output_dir, rows)
    summary = build_retrieval_summary(rows)
    summary["timesteps"] = timesteps
    summary["interpretation"] = interpret_retrieval_summary(summary)
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf8",
    )
    if panel_paths:
        make_overview(panel_paths).save(output_dir / "overview_grid.png")
    print(f"wrote Cross V3 reference retrieval diagnostic outputs to {output_dir}")
    return 0


def get_cached_tokens(
    *,
    bundle,
    record: dict[str, Any],
    token_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, Image.Image]],
    load_image_tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Image.Image]:
    from scripts.diagnose_cross_v3_z_ref_reconstruction import build_z_ref_only_reference_tokens

    ref_id = reference_record_id(record)
    if ref_id in token_cache:
        return token_cache[ref_id]
    image_tensor = load_image_tensor(record["reference_image"])
    image_pil = Image.open(record["reference_image"]).convert("RGB")
    tokens, z_ref = build_z_ref_only_reference_tokens(bundle, reference_image=image_tensor)
    token_cache[ref_id] = (tokens, z_ref, image_tensor, image_pil)
    return token_cache[ref_id]


def build_retrieval_row(
    *,
    anchor_index: int,
    mismatch_index: int,
    anchor: dict[str, Any],
    mismatch: dict[str, Any],
    correct_results: dict[str, dict[str, Any]],
    mismatch_results: dict[str, dict[str, Any]],
    zero_results: dict[str, dict[str, Any]],
    preview_key: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "anchor_index": anchor_index,
        "mismatch_index": mismatch_index,
        "anchor_reference_sample_id": reference_record_id(anchor),
        "mismatch_reference_sample_id": reference_record_id(mismatch),
        "anchor_dataset": anchor.get("dataset", ""),
        "mismatch_dataset": mismatch.get("dataset", ""),
        "preview_timestep_key": preview_key,
    }
    for timestep_key in sorted(set(correct_results) & set(mismatch_results) & set(zero_results)):
        correct = correct_results[timestep_key]
        mismatch_result = mismatch_results[timestep_key]
        zero = zero_results[timestep_key]
        correct_loss = float(correct["loss"])
        mismatch_loss = float(mismatch_result["loss"])
        zero_loss = float(zero["loss"])
        row[f"correct_loss_{timestep_key}"] = correct_loss
        row[f"mismatch_loss_{timestep_key}"] = mismatch_loss
        row[f"zero_loss_{timestep_key}"] = zero_loss
        row[f"mismatch_minus_correct_loss_{timestep_key}"] = mismatch_loss - correct_loss
        row[f"zero_minus_correct_loss_{timestep_key}"] = zero_loss - correct_loss
        for prefix, left, right in (
            ("correct_vs_mismatch", correct, mismatch_result),
            ("correct_vs_zero", correct, zero),
        ):
            stats = compare_flat_tensors(left["noise_pred_flat"], right["noise_pred_flat"])
            for name, value in stats.items():
                row[f"noise_pred_{prefix}_{name}_{timestep_key}"] = value
    return row


def compare_flat_tensors(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left_f = left.detach().float().reshape(-1)
    right_f = right.detach().float().reshape(-1)
    if left_f.numel() != right_f.numel():
        raise ValueError(f"Tensor sizes differ: {left_f.numel()} vs {right_f.numel()}")
    diff = left_f - right_f
    left_norm = float(torch.linalg.vector_norm(left_f).item())
    right_norm = float(torch.linalg.vector_norm(right_f).item())
    diff_l2 = float(torch.linalg.vector_norm(diff).item())
    denom = max(left_norm * right_norm, 1e-12)
    cosine = float(torch.dot(left_f, right_f).item() / denom)
    return {
        "l1": float(diff.abs().mean().item()),
        "l2": diff_l2,
        "relative_l2": float(diff_l2 / max(right_norm, 1e-12)),
        "cosine": cosine,
    }


def unique_reference_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        raise ValueError("metadata contains no records")
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for record in records:
        ref_id = reference_record_id(record)
        if ref_id in seen:
            continue
        seen.add(ref_id)
        output.append({**record, "reference_sample_id": ref_id})
    return output


def reference_record_id(record: dict[str, Any]) -> str:
    return str(record.get("reference_sample_id") or Path(record["reference_image"]).stem)


def select_anchor_records(
    records: list[dict[str, Any]],
    *,
    anchor_reference_sample_ids: list[str],
    num_anchors: int,
    seed: int,
) -> list[dict[str, Any]]:
    if anchor_reference_sample_ids:
        by_id = {reference_record_id(record): record for record in records}
        missing = [sample_id for sample_id in anchor_reference_sample_ids if sample_id not in by_id]
        if missing:
            raise ValueError(f"anchor reference sample_id(s) not found: {missing}")
        return [by_id[sample_id] for sample_id in anchor_reference_sample_ids]
    if num_anchors <= 0 or num_anchors >= len(records):
        return list(records)
    selected = list(records)
    random.Random(seed).shuffle(selected)
    return selected[:num_anchors]


def select_mismatch_records(
    records: list[dict[str, Any]],
    *,
    anchor: dict[str, Any],
    num_mismatches: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    anchor_id = reference_record_id(anchor)
    candidates = [record for record in records if reference_record_id(record) != anchor_id]
    if not candidates:
        raise ValueError("Need at least two unique references for mismatch retrieval diagnostics.")
    rng.shuffle(candidates)
    return candidates[: max(1, min(num_mismatches, len(candidates)))]


def select_preview_key_for_retrieval(results: dict[str, dict[str, Any]]) -> str:
    if not results:
        raise ValueError("No timestep results available.")
    selected = min(results.values(), key=lambda result: abs(float(result["timestep"]) - 700.0))
    return str(selected["timestep_key"])


def build_retrieval_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"num_pairs": 0.0}
    output: dict[str, Any] = {"num_pairs": float(len(rows))}
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (float, int))
        }
    )
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
        if values:
            output[f"{key}_mean"] = float(np.mean(values))
            output[f"{key}_std"] = float(np.std(values))
    delta_keys = [key for key in numeric_keys if key.startswith("mismatch_minus_correct_loss_")]
    if delta_keys:
        deltas = [
            float(row[key])
            for row in rows
            for key in delta_keys
            if key in row and math.isfinite(float(row[key]))
        ]
        output["mismatch_minus_correct_loss_all_mean"] = float(np.mean(deltas))
        output["mismatch_minus_correct_loss_all_std"] = float(np.std(deltas))
    zero_delta_keys = [key for key in numeric_keys if key.startswith("zero_minus_correct_loss_")]
    if zero_delta_keys:
        zero_deltas = [
            float(row[key])
            for row in rows
            for key in zero_delta_keys
            if key in row and math.isfinite(float(row[key]))
        ]
        output["zero_minus_correct_loss_all_mean"] = float(np.mean(zero_deltas))
        output["zero_minus_correct_loss_all_std"] = float(np.std(zero_deltas))
    return output


def interpret_retrieval_summary(summary: dict[str, Any]) -> str:
    mismatch_delta = float(summary.get("mismatch_minus_correct_loss_all_mean", math.nan))
    zero_delta = float(summary.get("zero_minus_correct_loss_all_mean", math.nan))
    if math.isfinite(mismatch_delta) and math.isfinite(zero_delta) and mismatch_delta > 0.01 and zero_delta > 0.01:
        return "cross_attention_prefers_correct_reference"
    if math.isfinite(mismatch_delta) and abs(mismatch_delta) < 0.003:
        return "no_clear_correct_reference_retrieval"
    return "mixed_or_weak_correct_reference_retrieval"


def make_retrieval_panel(
    *,
    anchor: Image.Image,
    mismatch: Image.Image,
    correct: Image.Image,
    mismatch_prediction: Image.Image,
    zero: Image.Image,
    thumbnail_size: int,
    title: str,
) -> Image.Image:
    images = [
        ("anchor_ref", anchor.convert("RGB")),
        ("correct_ref_pred", correct.convert("RGB")),
        ("mismatch_ref", mismatch.convert("RGB")),
        ("mismatch_pred", mismatch_prediction.convert("RGB")),
        (ZERO_TOKENS_VARIANT, zero.convert("RGB")),
    ]
    thumbs = [(label, thumbnail(image, thumbnail_size)) for label, image in images]
    label_h = 34
    title_h = 28
    panel = Image.new("RGB", (thumbnail_size * len(thumbs), thumbnail_size + label_h + title_h), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((6, 6), title[:160], fill=(0, 0, 0))
    for idx, (label, image) in enumerate(thumbs):
        x = idx * thumbnail_size
        panel.paste(image, (x, title_h))
        draw.text((x + 6, title_h + thumbnail_size + 8), label, fill=(0, 0, 0))
    return panel


def thumbnail(image: Image.Image, size: int) -> Image.Image:
    thumb = image.copy()
    thumb.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(thumb, ((size - thumb.width) // 2, (size - thumb.height) // 2))
    return canvas


def make_overview(panel_paths: list[Path]) -> Image.Image:
    panels = [Image.open(path).convert("RGB") for path in panel_paths]
    width = max(panel.width for panel in panels)
    height = sum(panel.height for panel in panels)
    overview = Image.new("RGB", (width, height), "white")
    y = 0
    for panel in panels:
        overview.paste(panel, (0, y))
        y += panel.height
    return overview


def write_rows(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.jsonl").open("w", encoding="utf8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "metrics.csv").open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())

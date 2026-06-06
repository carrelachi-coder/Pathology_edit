"""Cross V4 same-class reference-swap inference diagnostic.

Runs the same target masks with:
  1) the normal metadata reference,
  2) another reference for the same target and coverage bucket,
  3) zeroed reference tokens.

This is an inference-only diagnostic for deciding whether a 4k checkpoint is
using reference appearance before changing the training recipe.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose Cross V4 reference usage with same-class swap inference.")
    parser.add_argument("--pretrained-model-name-or-path")
    parser.add_argument("--checkpoint", help="Cross V4 checkpoint directory, e.g. checkpoint-4000.")
    parser.add_argument("--metadata", required=True, help="metadata_cross_train.json or metadata_cross_val.json.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument(
        "--prompt-source",
        choices=["fixed", "metadata", "dataset"],
        default="fixed",
        help="fixed matches Cross V4 training prompt; metadata/dataset are optional stress tests.",
    )
    parser.add_argument("--prompt", default=None)
    parser.add_argument(
        "--swap-scope",
        choices=["cross_case", "dataset", "case"],
        default="cross_case",
        help=(
            "Reference-swap search scope. cross_case strictly picks same covered tissue "
            "classes from a different WSI/case by default."
        ),
    )
    parser.add_argument(
        "--allow-same-case-fallback",
        action="store_true",
        help="For --swap-scope cross_case, fall back to same-WSI refs if no cross-case ref exists.",
    )
    parser.add_argument(
        "--dry-run-swap-selection",
        action="store_true",
        help="Only write/print selected normal-ref to swap-ref pairs; do not load the model.",
    )
    parser.add_argument("--thumbnail-size", type=int, default=192)
    parser.add_argument("--overview-max-samples", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _read_cross_metadata(args.metadata)
    swap_index = _build_swap_index(records)
    selected = _select_records_with_swap(
        records,
        swap_index,
        num_samples=args.num_samples,
        seed=args.seed,
        swap_scope=args.swap_scope,
        allow_same_case_fallback=args.allow_same_case_fallback,
    )
    if args.dry_run_swap_selection:
        rows = _dry_run_swap_selection(
            selected,
            swap_index,
            seed=args.seed,
            swap_scope=args.swap_scope,
            allow_same_case_fallback=args.allow_same_case_fallback,
        )
        (output_dir / "swap_selection.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=True),
            encoding="utf8",
        )
        cross_case_count = sum(1 for row in rows if row.get("case_id") != row.get("swap_case_id"))
        print(
            f"wrote {len(rows)} swap selections to {output_dir / 'swap_selection.json'} "
            f"(cross_case={cross_case_count}/{len(rows)})"
        )
        return 0

    if not selected:
        print(
            "No records have an alternate reference for the requested swap scope. "
            "Try --swap-scope dataset or --allow-same-case-fallback to inspect weaker swaps."
        )
        return 1

    if not args.pretrained_model_name_or_path or not args.checkpoint:
        raise SystemExit("--pretrained-model-name-or-path and --checkpoint are required unless --dry-run-swap-selection is set.")

    import torch

    from controlnet_train.cli.eval_controlnet_flux_cross import (
        compute_cross_metrics,
        _make_overview,
        _pil_to_chw_float,
        _safe_name,
        _save_error_image,
        _save_mask_image,
    )
    from controlnet_train.data.common import (
        default_prompt_for_dataset,
        load_image_tensor,
        load_nuclei_mask,
        load_tissue_mask,
    )
    from controlnet_train.inference.pipeline_cross_v4 import (
        CROSS_V4_REFERENCE_WITH_REF,
        CROSS_V4_REFERENCE_ZERO_REF,
        CROSS_V4_PROMPT,
        load_cross_v4_bundle,
        run_cross_v4_bundle,
    )

    dtype_by_name = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_cross_v4_bundle(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        checkpoint_path=args.checkpoint,
        device=args.device,
        torch_dtype=dtype_by_name[args.torch_dtype],
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
    )
    run_config = _build_run_config(args, bundle)
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf8",
    )
    print(
        "Cross V4 inference config: "
        f"checkpoint={run_config['checkpoint']} "
        f"prompt_source={run_config['prompt_source']} "
        f"steps={run_config['num_inference_steps']} "
        f"guidance={run_config['guidance_scale']} "
        f"control_scale={run_config['controlnet_conditioning_scale']}"
    )

    summary_rows: list[dict[str, Any]] = []
    panel_paths: list[Path] = []
    rng = random.Random(args.seed)
    for index, record in enumerate(selected):
        swap_record = _choose_swap_record(
            record,
            swap_index,
            rng,
            swap_scope=args.swap_scope,
            allow_same_case_fallback=args.allow_same_case_fallback,
        )
        sample_id = str(record.get("sample_id") or Path(record["target_image"]).stem)
        ref_id = str(record.get("reference_sample_id") or Path(record["reference_image"]).stem)
        sample_dir = samples_dir / f"{index:04d}_{_safe_name(sample_id)}__ref_{_safe_name(ref_id)}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        if swap_record is None:
            row = _base_row(index, record)
            row["status"] = "missing_swap_reference"
            summary_rows.append(row)
            print(f"[{index + 1}/{len(selected)}] skip {sample_id}: no alternate reference")
            continue

        prompt = _resolve_prompt(
            record=record,
            prompt_override=args.prompt,
            prompt_source=args.prompt_source,
            fixed_prompt=CROSS_V4_PROMPT,
            default_prompt_for_dataset=default_prompt_for_dataset,
        )
        target_tissue_mask = load_tissue_mask(record["target_tissue_mask"])
        target_nuclei_mask = load_nuclei_mask(record["target_nuclei_mask"])
        target_pil = Image.open(record["target_image"]).convert("RGB")
        target_array = _pil_to_chw_float(target_pil)
        _save_mask_image(np.asarray(Image.open(record["target_tissue_mask"])), sample_dir / "target_tissue_mask.png")
        _save_mask_image(np.asarray(Image.open(record["target_nuclei_mask"])), sample_dir / "target_nuclei_mask.png")
        target_pil.save(sample_dir / "target.png")

        variants = [
            ("normal", record, CROSS_V4_REFERENCE_WITH_REF),
            ("same_class_swap", swap_record, CROSS_V4_REFERENCE_WITH_REF),
            ("zero_ref", record, CROSS_V4_REFERENCE_ZERO_REF),
        ]
        variant_results: list[dict[str, Any]] = []
        for variant_name, ref_record, ref_mode in variants:
            reference_pil = Image.open(ref_record["reference_image"]).convert("RGB")
            prediction = run_cross_v4_bundle(
                bundle,
                reference_image=load_image_tensor(ref_record["reference_image"]),
                reference_tissue_mask=load_tissue_mask(ref_record["reference_tissue_mask"]),
                reference_nuclei_mask=load_nuclei_mask(ref_record["reference_nuclei_mask"]),
                target_tissue_mask=target_tissue_mask,
                target_nuclei_mask=target_nuclei_mask,
                prompt=prompt,
                reference_condition_mode=ref_mode,
            ).convert("RGB")
            pred_array = _pil_to_chw_float(prediction)
            metrics = compute_cross_metrics(pred_array, target_array)
            abs_error = np.abs(pred_array - target_array).mean(axis=0)

            reference_pil.save(sample_dir / f"reference_{variant_name}.png")
            prediction.save(sample_dir / f"prediction_{variant_name}.png")
            _save_mask_image(
                np.asarray(Image.open(ref_record["reference_tissue_mask"])),
                sample_dir / f"reference_tissue_mask_{variant_name}.png",
            )
            _save_error_image(abs_error, sample_dir / f"abs_error_{variant_name}.png")
            variant_results.append(
                {
                    "name": variant_name,
                    "record": ref_record,
                    "reference": reference_pil,
                    "prediction": prediction,
                    "pred_array": pred_array,
                    "metrics": metrics,
                }
            )

        comparison = _compare_variants(variant_results)
        row = {
            **_base_row(index, record),
            "status": "ok",
            "prompt": prompt,
            "checkpoint": str(bundle.checkpoint_path),
            "pretrained_model_name_or_path": str(bundle.pretrained_model_name_or_path),
            "num_inference_steps": int(bundle.num_inference_steps),
            "guidance_scale": float(bundle.guidance_scale),
            "controlnet_conditioning_scale": float(bundle.controlnet_conditioning_scale),
            "swap_target_sample_id": str(swap_record.get("sample_id", "")),
            "swap_reference_sample_id": str(swap_record.get("reference_sample_id", "")),
            "swap_case_id": str(swap_record.get("case_id", "")),
            "swap_scope": args.swap_scope,
            "swap_is_cross_case": str(swap_record.get("case_id", "")) != str(record.get("case_id", "")),
            **comparison,
        }
        summary_rows.append(row)
        (sample_dir / "diagnostic.json").write_text(
            json.dumps(row, indent=2, ensure_ascii=False, allow_nan=True),
            encoding="utf8",
        )
        panel = _make_swap_panel(variant_results, target_pil, args.thumbnail_size, title=sample_id)
        panel_path = sample_dir / "panel_same_class_swap.png"
        panel.save(panel_path)
        if len(panel_paths) < args.overview_max_samples:
            panel_paths.append(panel_path)
        print(
            f"[{index + 1}/{len(selected)}] {sample_id} ref={ref_id} "
            f"case={row['case_id']} swap_case={row['swap_case_id']} "
            f"swap_ref={row['swap_reference_sample_id']} "
            f"normal_vs_swap_l1={row['normal_vs_same_class_swap_l1']:.4f} "
            f"normal_vs_zero_l1={row['normal_vs_zero_ref_l1']:.4f}"
        )

    (output_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf8",
    )
    if panel_paths:
        _make_overview(panel_paths).save(output_dir / "overview_grid.png")
    print(f"wrote Cross V4 same-class swap diagnostics to {output_dir}")
    return 0


def _read_cross_metadata(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf8"))
    if isinstance(payload, dict):
        records = payload.get("pairs")
        if not isinstance(records, list):
            raise ValueError("cross metadata dict must contain a 'pairs' list")
        return records
    if isinstance(payload, list):
        return payload
    raise TypeError(f"unsupported cross metadata payload type: {type(payload)!r}")


def _build_swap_index(records: list[dict[str, Any]]) -> dict[tuple[str, str, tuple[int, ...], str], list[dict[str, Any]]]:
    index: dict[tuple[str, str, tuple[int, ...], str], list[dict[str, Any]]] = {}
    for record in records:
        for scope in ("case", "dataset"):
            for include_difficulty in (True, False):
                index.setdefault(_swap_key(record, scope=scope, include_difficulty=include_difficulty), []).append(
                    record
                )
    return index


def _swap_key(
    record: dict[str, Any],
    *,
    scope: str,
    include_difficulty: bool,
) -> tuple[str, str, tuple[int, ...], str]:
    scope_value = str(record.get("case_id", "")) if scope == "case" else ""
    difficulty = str(record.get("pair_difficulty", "") if include_difficulty else "")
    return (
        str(record.get("dataset", "")),
        scope_value,
        _covered_tissue_key(record),
        difficulty,
    )


def _choose_swap_record(
    record: dict[str, Any],
    swap_index: dict[tuple[str, str, tuple[int, ...], str], list[dict[str, Any]]],
    rng: random.Random,
    *,
    swap_scope: str,
    allow_same_case_fallback: bool,
) -> dict[str, Any] | None:
    ref_id = str(record.get("reference_sample_id", ""))
    original_case_id = str(record.get("case_id", ""))
    scopes = ["case"] if swap_scope == "case" else ["dataset"]
    for scope in scopes:
        for include_difficulty in (True, False):
            candidates = [
                candidate
                for candidate in swap_index.get(_swap_key(record, scope=scope, include_difficulty=include_difficulty), [])
                if str(candidate.get("reference_sample_id", "")) != ref_id
                and (swap_scope != "cross_case" or str(candidate.get("case_id", "")) != original_case_id)
            ]
            if candidates:
                return rng.choice(candidates)
    if swap_scope == "cross_case" and allow_same_case_fallback:
        # Optional weaker fallback: useful for metadata coverage checks, not for the main diagnosis.
        candidates = [
            candidate
            for candidate in swap_index.get(_swap_key(record, scope="case", include_difficulty=False), [])
            if str(candidate.get("reference_sample_id", "")) != ref_id
        ]
        if candidates:
            return rng.choice(candidates)
    return None


def _select_records_with_swap(
    records: list[dict[str, Any]],
    swap_index: dict[tuple[str, str, tuple[int, ...], str], list[dict[str, Any]]],
    *,
    num_samples: int,
    seed: int,
    swap_scope: str,
    allow_same_case_fallback: bool,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    selected: list[dict[str, Any]] = []
    probe_rng = random.Random(seed)
    for record in shuffled:
        if _choose_swap_record(
            record,
            swap_index,
            probe_rng,
            swap_scope=swap_scope,
            allow_same_case_fallback=allow_same_case_fallback,
        ):
            selected.append(record)
            if len(selected) >= num_samples:
                break
    return selected


def _dry_run_swap_selection(
    selected: list[dict[str, Any]],
    swap_index: dict[tuple[str, str, tuple[int, ...], str], list[dict[str, Any]]],
    *,
    seed: int,
    swap_scope: str,
    allow_same_case_fallback: bool,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(selected):
        swap_record = _choose_swap_record(
            record,
            swap_index,
            rng,
            swap_scope=swap_scope,
            allow_same_case_fallback=allow_same_case_fallback,
        )
        if swap_record is None:
            continue
        row = {
            **_base_row(index, record),
            "swap_target_sample_id": str(swap_record.get("sample_id", "")),
            "swap_reference_sample_id": str(swap_record.get("reference_sample_id", "")),
            "swap_case_id": str(swap_record.get("case_id", "")),
            "swap_scope": swap_scope,
            "swap_is_cross_case": str(swap_record.get("case_id", "")) != str(record.get("case_id", "")),
            "covered_tissue_key": list(_covered_tissue_key(record)),
            "swap_covered_tissue_key": list(_covered_tissue_key(swap_record)),
        }
        rows.append(row)
        print(
            f"[{index + 1}/{len(selected)}] {row['sample_id']} "
            f"case={row['case_id']} ref={row['reference_sample_id']} -> "
            f"swap_case={row['swap_case_id']} swap_ref={row['swap_reference_sample_id']} "
            f"cross_case={row['swap_is_cross_case']}"
        )
    return rows


def _covered_tissue_key(record: dict[str, Any]) -> tuple[int, ...]:
    raw = record.get("covered_target_tissue_ids")
    if isinstance(raw, list) and raw:
        return tuple(sorted(int(value) for value in raw))
    raw_missing = record.get("missing_target_tissue_ids")
    if isinstance(raw_missing, list) and raw_missing:
        return tuple([-int(value) for value in sorted(raw_missing)])
    return ()


def _resolve_prompt(
    *,
    record: dict[str, Any],
    prompt_override: str | None,
    prompt_source: str,
    fixed_prompt: str,
    default_prompt_for_dataset,
) -> str:
    if prompt_override:
        return prompt_override
    if prompt_source == "fixed":
        return fixed_prompt
    if prompt_source == "metadata" and record.get("prompt"):
        return str(record["prompt"])
    if prompt_source == "dataset" and record.get("dataset"):
        return default_prompt_for_dataset(str(record["dataset"]))
    return str(record.get("prompt") or "H&E stained cancer histopathology at 40x magnification")


def _build_run_config(args: argparse.Namespace, bundle) -> dict[str, Any]:
    control_spec = bundle.control_spec
    reference_spec = bundle.reference_spec
    return {
        "pretrained_model_name_or_path": str(bundle.pretrained_model_name_or_path),
        "checkpoint": str(bundle.checkpoint_path),
        "metadata": str(args.metadata),
        "output_dir": str(args.output_dir),
        "seed": int(args.seed),
        "device": str(bundle.device),
        "torch_dtype": str(bundle.torch_dtype).replace("torch.", ""),
        "num_inference_steps": int(bundle.num_inference_steps),
        "guidance_scale": float(bundle.guidance_scale),
        "controlnet_conditioning_scale": float(bundle.controlnet_conditioning_scale),
        "prompt_source": str(args.prompt_source),
        "prompt_override": args.prompt,
        "swap_scope": str(args.swap_scope),
        "allow_same_case_fallback": bool(args.allow_same_case_fallback),
        "control_spec": {
            "raw_channels": int(control_spec.raw_channels),
            "packed_channels": int(control_spec.packed_channels),
            "tissue_channels": int(control_spec.tissue_channels),
            "nuclei_channels": int(control_spec.nuclei_channels),
        },
        "reference_spec": {
            "reference_latent_channels": int(reference_spec.reference_latent_channels),
            "tissue_channels": int(reference_spec.tissue_channels),
            "nuclei_channels": int(reference_spec.nuclei_channels),
            "token_dim": int(reference_spec.token_dim),
            "route_anchor_mode": str(reference_spec.normalized_route_anchor_mode),
            "tissue_prior_tokens_per_class": int(reference_spec.tissue_prior_tokens_per_class),
            "cell_prior_tokens_per_class": int(reference_spec.cell_prior_tokens_per_class),
            "global_style_tokens": int(reference_spec.global_style_tokens),
        },
        "attention_bias_config": dict(bundle.attention_bias_config),
    }


def _base_row(index: int, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": int(index),
        "sample_id": str(record.get("sample_id", "")),
        "reference_sample_id": str(record.get("reference_sample_id", "")),
        "dataset": str(record.get("dataset", "")),
        "case_id": str(record.get("case_id", "")),
        "pair_difficulty": str(record.get("pair_difficulty", "")),
        "tissue_coverage_ratio": float(record.get("tissue_coverage_ratio", math.nan)),
        "area_coverage_ratio": float(record.get("area_coverage_ratio", math.nan)),
    }


def _compare_variants(variant_results: list[dict[str, Any]]) -> dict[str, float]:
    by_name = {result["name"]: result for result in variant_results}
    normal = by_name["normal"]["pred_array"]
    same_class = by_name["same_class_swap"]["pred_array"]
    zero_ref = by_name["zero_ref"]["pred_array"]
    return {
        "normal_vs_same_class_swap_l1": float(np.abs(normal - same_class).mean()),
        "normal_vs_same_class_swap_mse": float(np.square(normal - same_class).mean()),
        "normal_vs_zero_ref_l1": float(np.abs(normal - zero_ref).mean()),
        "normal_vs_zero_ref_mse": float(np.square(normal - zero_ref).mean()),
        "same_class_swap_vs_zero_ref_l1": float(np.abs(same_class - zero_ref).mean()),
        "same_class_swap_vs_zero_ref_mse": float(np.square(same_class - zero_ref).mean()),
    }


def _make_swap_panel(
    variant_results: list[dict[str, Any]],
    target: Image.Image,
    thumbnail_size: int,
    *,
    title: str,
) -> Image.Image:
    thumbs: list[tuple[str, Image.Image]] = [("target", target)]
    for result in variant_results:
        thumbs.append((f"{result['name']} ref", result["reference"]))
        thumbs.append((f"{result['name']} pred", result["prediction"]))
    width = thumbnail_size * len(thumbs)
    title_h = 30
    label_h = 24
    panel = Image.new("RGB", (width, title_h + thumbnail_size + label_h), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((6, 6), title, fill=(0, 0, 0))
    for col, (label, image) in enumerate(thumbs):
        thumb = image.resize((thumbnail_size, thumbnail_size), Image.Resampling.BILINEAR)
        x = col * thumbnail_size
        panel.paste(thumb, (x, title_h))
        draw.text((x + 4, title_h + thumbnail_size + 4), label, fill=(0, 0, 0))
    return panel


if __name__ == "__main__":
    raise SystemExit(main())

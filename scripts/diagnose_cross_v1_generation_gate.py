"""Run the fixed-seed Cross V1 generation gate.

Each selected target is generated from three reference feature arms while the
paired reference labels are held fixed:

* paired: paired reference image
* alternate_feature: same-dataset, different-WSI reference image
* alternate_feature_cross_dataset: different-dataset, different-WSI reference
  image
* zero: all-zero reference image

This isolates reference-image usage from regional label routing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


VARIANTS = ("paired", "alternate_feature", "zero")
ALTERNATE_MODES = ("same_dataset", "different_dataset")
PROMPT_MODES = ("dataset", "empty")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross V1 fixed-seed generation gate.")
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--checkpoint", required=True, help="Eval-ready Cross V1 checkpoint dir.")
    parser.add_argument("--uni-checkpoint-path", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument(
        "--record-indices",
        default="",
        help="Optional comma-separated metadata indices. Overrides --num-samples.",
    )
    parser.add_argument("--selection-seed", type=int, default=20260611)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--scales", default="0.5,0.75,1.0")
    parser.add_argument(
        "--alternate-mode",
        choices=("same_dataset", "different_dataset", "both"),
        default="same_dataset",
        help="Reference-image alternate to probe.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("dataset", "empty", "both"),
        default="dataset",
        help="Prompt probe to run alongside the reference-image probe.",
    )
    parser.add_argument("--tumor-label", type=int, default=1)
    parser.add_argument("--min-tumor-fraction", type=float, default=0.02)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--prompt-source", choices=("metadata", "dataset"), default="dataset")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--source-latent-init-strength", type=float, default=0.0)
    parser.add_argument("--thumbnail-size", type=int, default=192)
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Write the paired/alternate manifest without loading the model.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata_path = Path(args.metadata)
    records = read_records(metadata_path)
    scales = parse_scales(args.scales)
    alternate_modes = parse_mode_selection(args.alternate_mode, ALTERNATE_MODES)
    prompt_modes = parse_mode_selection(args.prompt_mode, PROMPT_MODES)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = select_gate_records(
        records,
        record_indices=parse_indices(args.record_indices),
        num_samples=args.num_samples,
        seed=args.selection_seed,
        tumor_label=args.tumor_label,
        min_tumor_fraction=args.min_tumor_fraction,
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
    print(f"selected {len(selected)} fixed generation probes")
    for row in manifest:
        alternates = row["alternates"]
        alternate_text = ", ".join(
            f"{mode}={alternates[mode]['reference_sample_id']}"
            for mode in alternate_modes
            if mode in alternates
        )
        print(
            f"  index={row['metadata_index']} target={row['sample_id']} "
            f"paired={row['paired_reference_sample_id']} "
            f"alternates={alternate_text}"
        )
    if args.selection_only:
        return 0

    import torch

    from controlnet_train.data.common import (
        load_image_tensor,
        load_nuclei_mask,
        load_tissue_mask,
    )
    from controlnet_train.inference.pipeline_cross_v1 import (
        _packed_flux_image_token_count,
        _tissue_fallback_region_labels,
        load_cross_v1_bundle,
        run_cross_v1_bundle,
        set_ip_adapter_scale,
    )
    from controlnet_train.modules.reference_image_encoder import build_region_ip_token_labels
    from controlnet_train.training.flux_phase5_cross_v1 import (
        _build_region_attention_mask_and_query_gate,
    )
    from controlnet_train.modules.reference_image_encoder import resize_mask_to_token_labels

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
        ip_adapter_scale=scales[0],
    )

    metric_rows: list[dict[str, Any]] = []
    panel_paths: list[Path] = []
    for probe_index, (metadata_index, paired, alternates) in enumerate(selected):
        sample_id = record_sample_id(paired)
        sample_dir = output_dir / f"{probe_index:03d}_{safe_name(sample_id)}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        target_pil = Image.open(paired["target_image"]).convert("RGB")
        paired_ref_pil = Image.open(paired["reference_image"]).convert("RGB")
        target_tissue = load_tissue_mask(paired["target_tissue_mask"])
        target_nuclei = load_nuclei_mask(paired["target_nuclei_mask"])
        paired_ref_tissue = load_tissue_mask(paired["reference_tissue_mask"])
        paired_ref_nuclei = load_nuclei_mask(paired["reference_nuclei_mask"])
        paired_ref_tensor = load_image_tensor(paired["reference_image"])
        zero_ref_tensor = paired_ref_tensor.new_zeros(paired_ref_tensor.shape)
        region_stats = compute_generation_region_stats(
            bundle=bundle,
            reference_image=paired_ref_tensor,
            reference_tissue_mask=paired_ref_tissue,
            reference_nuclei_mask=paired_ref_nuclei,
            target_tissue_mask=target_tissue,
            target_nuclei_mask=target_nuclei,
            build_region_ip_token_labels=build_region_ip_token_labels,
            build_region_attention_mask_and_query_gate=_build_region_attention_mask_and_query_gate,
            packed_flux_image_token_count=_packed_flux_image_token_count,
            tissue_fallback_region_labels=_tissue_fallback_region_labels,
            resize_mask_to_token_labels=resize_mask_to_token_labels,
        )

        target_pil.save(sample_dir / "target.png")
        paired_ref_pil.save(sample_dir / "reference_paired.png")
        Image.new("RGB", paired_ref_pil.size, "black").save(sample_dir / "reference_zero.png")

        for alternate_mode in alternate_modes:
            alternate = alternates[alternate_mode]
            alternate_ref_pil = Image.open(alternate["reference_image"]).convert("RGB")
            alternate_ref_tensor = load_image_tensor(alternate["reference_image"])
            alternate_ref_pil.save(sample_dir / f"reference_{alternate_mode}.png")
            for prompt_mode in prompt_modes:
                prompt = "" if prompt_mode == "empty" else resolve_prompt(args, paired)
                combo_dir = sample_dir / f"alternate_{alternate_mode}" / f"prompt_{prompt_mode}"
                combo_dir.mkdir(parents=True, exist_ok=True)

                variant_tensors = {
                    "paired": paired_ref_tensor,
                    "alternate_feature": alternate_ref_tensor,
                    "zero": zero_ref_tensor,
                }
                predictions: dict[tuple[float, str], Image.Image] = {}
                for scale in scales:
                    set_ip_adapter_scale(bundle.flux_pipeline.transformer, scale)
                    for variant in VARIANTS:
                        print(
                            f"[{probe_index + 1}/{len(selected)}] {sample_id} "
                            f"alt={alternate_mode} prompt={prompt_mode} "
                            f"scale={scale:g} variant={variant}"
                        )
                        prediction = run_cross_v1_bundle(
                            bundle,
                            reference_image=variant_tensors[variant],
                            # All arms intentionally retain paired labels.
                            reference_tissue_mask=paired_ref_tissue,
                            reference_nuclei_mask=paired_ref_nuclei,
                            target_tissue_mask=target_tissue,
                            target_nuclei_mask=target_nuclei,
                            prompt=prompt,
                            source_latent_init_strength=args.source_latent_init_strength,
                            seed=args.generation_seed,
                        )
                        predictions[(scale, variant)] = prediction
                        prediction.save(combo_dir / f"prediction_scale_{scale:g}_{variant}.png")

                    rows = compare_scale_outputs(
                        metadata_index=metadata_index,
                        sample_id=sample_id,
                        paired=paired,
                        alternate=alternate,
                        alternate_mode=alternate_mode,
                        prompt_mode=prompt_mode,
                        scale=scale,
                        region_stats=region_stats,
                        target=target_pil,
                        target_tissue=np.asarray(target_tissue),
                        tumor_label=args.tumor_label,
                        predictions=predictions,
                    )
                    metric_rows.extend(rows)

                panel = make_probe_panel(
                    sample_id=sample_id,
                    target=target_pil,
                    paired_reference=paired_ref_pil,
                    alternate_reference=alternate_ref_pil,
                    predictions=predictions,
                    scales=scales,
                    thumbnail_size=args.thumbnail_size,
                )
                panel_path = combo_dir / "generation_gate.png"
                panel.save(panel_path)
                panel_paths.append(panel_path)

    write_metrics(output_dir / "generation_gate_metrics.csv", metric_rows)
    summary = summarize_metrics(metric_rows)
    summary.update(
        {
            "checkpoint": str(args.checkpoint),
            "generation_seed": int(args.generation_seed),
            "selection_seed": int(args.selection_seed),
            "scales": scales,
            "alternate_modes": alternate_modes,
            "prompt_modes": prompt_modes,
            "num_samples": len(selected),
            "variant_semantics": {
                "paired": "paired image + paired labels",
                "alternate_feature": "same-dataset different-WSI image + paired labels",
                "zero": "zero image + paired labels",
            },
            "region_stats": summarize_region_stats(metric_rows),
            "combo_region_stats": summarize_combo_region_stats(metric_rows),
        }
    )
    (output_dir / "generation_gate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf8",
    )
    if panel_paths and len(panel_paths) <= 48:
        make_overview(panel_paths).save(output_dir / "generation_gate_overview.png")
    elif panel_paths:
        print(
            f"Skipping overview panel because {len(panel_paths)} combo panels were generated."
        )
    print(f"wrote generation gate to {output_dir}")
    return 0


def read_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf8"))
    if isinstance(payload, dict):
        payload = payload.get("pairs")
    if not isinstance(payload, list):
        raise ValueError("metadata must be a list or a dict containing a 'pairs' list")
    return [dict(row) for row in payload]


def parse_indices(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_scales(value: str) -> list[float]:
    scales = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not scales or any(scale < 0 for scale in scales):
        raise ValueError("--scales must contain nonnegative comma-separated values")
    return scales


def parse_mode_selection(value: str, allowed: tuple[str, ...]) -> list[str]:
    selected: list[str] = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if item == "both":
            selected.extend(list(allowed))
            continue
        if item not in allowed:
            raise ValueError(f"invalid mode {item!r}; expected one of {allowed!r} or 'both'")
        selected.append(item)
    if not selected:
        raise ValueError(f"mode selection {value!r} produced no modes")
    deduped: list[str] = []
    seen: set[str] = set()
    for item in selected:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def select_gate_records(
    records: list[dict[str, Any]],
    *,
    record_indices: list[int],
    num_samples: int,
    seed: int,
    tumor_label: int,
    min_tumor_fraction: float,
    alternate_modes: list[str],
) -> list[tuple[int, dict[str, Any], dict[str, dict[str, Any]]]]:
    valid = [
        (index, record)
        for index, record in enumerate(records)
        if record_has_tumor(record, "target_tissue_mask", tumor_label, min_tumor_fraction)
        and record_has_tumor(record, "reference_tissue_mask", tumor_label, min_tumor_fraction)
    ]
    if record_indices:
        by_index = dict(valid)
        missing = [index for index in record_indices if index not in by_index]
        if missing:
            raise ValueError(f"requested indices are missing or fail tumor filters: {missing}")
        paired_candidates = [(index, by_index[index]) for index in record_indices]
    else:
        paired_candidates = list(valid)
        random.Random(seed).shuffle(paired_candidates)
        paired_candidates = paired_candidates[: max(1, int(num_samples))]

    selected = []
    for metadata_index, paired in paired_candidates:
        alternates: dict[str, dict[str, Any]] = {}
        for mode in alternate_modes:
            alternates[mode] = choose_alternate_reference(
                paired,
                valid,
                seed=seed + metadata_index,
                mode=mode,
            )
        selected.append((metadata_index, paired, alternates))
    return selected


def choose_alternate_reference(
    paired: dict[str, Any],
    candidates: list[tuple[int, dict[str, Any]]],
    *,
    seed: int,
    mode: str,
) -> dict[str, Any]:
    paired_dataset = str(paired.get("dataset") or "")
    paired_case = reference_case_id(paired)
    if mode not in ALTERNATE_MODES:
        raise ValueError(f"unknown alternate mode: {mode}")
    eligible = []
    for _, record in candidates:
        if reference_case_id(record) == paired_case:
            continue
        if Path(record["reference_image"]) == Path(paired["reference_image"]):
            continue
        record_dataset = str(record.get("dataset") or "")
        if mode == "same_dataset" and paired_dataset and record_dataset != paired_dataset:
            continue
        if mode == "different_dataset":
            if paired_dataset and record_dataset == paired_dataset:
                continue
        eligible.append(record)
    if not eligible:
        raise ValueError(
            f"no {mode} alternate found for {record_sample_id(paired)}"
        )
    mode_seed = seed + (17 if mode == "same_dataset" else 29)
    return random.Random(mode_seed).choice(eligible)


def record_has_tumor(
    record: dict[str, Any],
    field: str,
    tumor_label: int,
    min_fraction: float,
) -> bool:
    path = Path(record[field])
    if not path.exists():
        return False
    mask = np.asarray(Image.open(path))
    return float(np.mean(mask == tumor_label)) >= float(min_fraction)


def record_sample_id(record: dict[str, Any]) -> str:
    return str(record.get("sample_id") or Path(record["target_image"]).stem)


def reference_sample_id(record: dict[str, Any]) -> str:
    return str(record.get("reference_sample_id") or Path(record["reference_image"]).stem)


def reference_case_id(record: dict[str, Any]) -> str:
    for key in ("reference_case_id", "reference_wsi_id", "reference_slide_id"):
        value = record.get(key)
        if value:
            return str(value)
    sample_id = reference_sample_id(record)
    marker = sample_id.rfind("_py")
    return sample_id[:marker] if marker >= 0 else sample_id


def build_manifest_row(
    metadata_index: int,
    paired: dict[str, Any],
    alternates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "metadata_index": metadata_index,
        "dataset": paired.get("dataset"),
        "sample_id": record_sample_id(paired),
        "paired_reference_sample_id": reference_sample_id(paired),
        "paired_reference_case_id": reference_case_id(paired),
        "alternates": {
            mode: {
                "reference_sample_id": reference_sample_id(record),
                "reference_case_id": reference_case_id(record),
                "dataset": record.get("dataset"),
                "reference_image": record["reference_image"],
            }
            for mode, record in alternates.items()
        },
        "target_image": paired["target_image"],
        "paired_reference_image": paired["reference_image"],
    }


def resolve_prompt(args: argparse.Namespace, record: dict[str, Any]) -> str:
    if args.prompt:
        return str(args.prompt)
    if args.prompt_source == "metadata" and record.get("prompt"):
        return str(record["prompt"])
    try:
        from controlnet_train.data.common import default_prompt_for_dataset

        if record.get("dataset"):
            return default_prompt_for_dataset(str(record["dataset"]))
    except (KeyError, ValueError):
        pass
    return str(record.get("prompt") or "H&E stained cancer histopathology at 40x magnification")


def compare_scale_outputs(
    *,
    metadata_index: int,
    sample_id: str,
    paired: dict[str, Any],
    alternate: dict[str, Any],
    alternate_mode: str,
    prompt_mode: str,
    scale: float,
    region_stats: dict[str, float | int | bool],
    target: Image.Image,
    target_tissue: np.ndarray,
    tumor_label: int,
    predictions: dict[tuple[float, str], Image.Image],
) -> list[dict[str, Any]]:
    arrays = {
        variant: image_array(predictions[(scale, variant)])
        for variant in VARIANTS
    }
    target_array = image_array(target)
    tumor_mask = np.asarray(target_tissue) == int(tumor_label)
    rows = []
    for variant in VARIANTS:
        rows.append(
            {
                "metadata_index": metadata_index,
                "sample_id": sample_id,
                "dataset": paired.get("dataset", ""),
                "paired_reference_sample_id": reference_sample_id(paired),
                "alternate_reference_sample_id": reference_sample_id(alternate),
                "alternate_mode": alternate_mode,
                "prompt_mode": prompt_mode,
                "scale": scale,
                "variant": variant,
                "region_active_q": float(region_stats["active_query_fraction"]),
                "region_missing_q": float(region_stats["missing_query_fraction"]),
                "region_fallback_q": float(region_stats["fallback_query_fraction"]),
                "region_null_q": float(region_stats["null_query_fraction"]),
                "region_allowed_pairs": float(region_stats["allowed_valid_pair_fraction"]),
                "prompt_is_empty": prompt_mode == "empty",
                "target_full_l1": masked_l1(arrays[variant], target_array, None),
                "target_tumor_l1": masked_l1(arrays[variant], target_array, tumor_mask),
                "paired_output_full_l1": masked_l1(
                    arrays[variant], arrays["paired"], None
                ),
                "paired_output_tumor_l1": masked_l1(
                    arrays[variant], arrays["paired"], tumor_mask
                ),
                "output_edge_energy": edge_energy(arrays[variant], tumor_mask),
            }
        )
    return rows


def compute_generation_region_stats(
    *,
    bundle,
    reference_image,
    reference_tissue_mask,
    reference_nuclei_mask,
    target_tissue_mask,
    target_nuclei_mask,
    build_region_ip_token_labels,
    build_region_attention_mask_and_query_gate,
    packed_flux_image_token_count,
    tissue_fallback_region_labels,
    resize_mask_to_token_labels,
) -> dict[str, float | int | bool]:
    if not bool(getattr(bundle, "regional_ip_adapter", False)):
        return {
            "active_query_fraction": math.nan,
            "missing_query_fraction": math.nan,
            "fallback_query_fraction": math.nan,
            "null_query_fraction": math.nan,
            "allowed_valid_pair_fraction": math.nan,
        }
    device = bundle.device
    reference_tissue = reference_tissue_mask.unsqueeze(0).to(device=device)
    reference_nuclei = reference_nuclei_mask.unsqueeze(0).to(device=device)
    target_tissue = target_tissue_mask.unsqueeze(0).to(device=device)
    target_nuclei = target_nuclei_mask.unsqueeze(0).to(device=device)
    key_len = int(getattr(bundle.ref_encoder, "num_spatial_tokens", 256))
    query_len = packed_flux_image_token_count(reference_image, bundle.flux_pipeline)
    key_labels = build_region_ip_token_labels(
        tissue_mask=reference_tissue,
        num_tokens=key_len,
        nuclei_mask=reference_nuclei,
        label_mode=bundle.regional_ip_label_mode,
    ).to(device=device)
    query_labels = build_region_ip_token_labels(
        tissue_mask=target_tissue,
        num_tokens=query_len,
        nuclei_mask=target_nuclei,
        label_mode=bundle.regional_ip_label_mode,
    ).to(device=device)
    key_fallback = tissue_fallback_region_labels(
        key_labels,
        label_mode=bundle.regional_ip_label_mode,
    ).to(device=device)
    query_fallback = resize_mask_to_token_labels(
        target_tissue,
        query_len,
    ).to(device=device)
    _, _, stats = build_region_attention_mask_and_query_gate(
        query_region_labels=query_labels,
        key_region_labels=key_labels,
        batch_size=1,
        query_len=query_len,
        key_len=key_len,
        device=query_labels.device,
        dtype=bundle.torch_dtype,
        strict=bool(getattr(bundle, "regional_ip_strict", True)),
        query_fallback_labels=query_fallback,
        key_fallback_labels=key_fallback,
    )
    return stats


def image_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def masked_l1(left: np.ndarray, right: np.ndarray, mask: np.ndarray | None) -> float:
    difference = np.abs(left - right)
    if mask is None:
        return float(difference.mean())
    valid = np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return math.nan
    return float(difference[valid].mean())


def edge_energy(image: np.ndarray, mask: np.ndarray | None) -> float:
    gray = image.mean(axis=-1)
    grad_y, grad_x = np.gradient(gray)
    magnitude = np.sqrt(np.square(grad_x) + np.square(grad_y))
    if mask is not None and np.any(mask):
        magnitude = magnitude[np.asarray(mask, dtype=bool)]
    return float(magnitude.mean())


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "by_scale": summarize_rows_by_scale(rows),
        "by_combo": {},
    }
    combos = sorted(
        {
            (
                str(row.get("alternate_mode") or "same_dataset"),
                str(row.get("prompt_mode") or "dataset"),
            )
            for row in rows
        }
    )
    for alternate_mode, prompt_mode in combos:
        combo_rows = [
            row
            for row in rows
            if str(row.get("alternate_mode") or "same_dataset") == alternate_mode
            and str(row.get("prompt_mode") or "dataset") == prompt_mode
        ]
        summary["by_combo"][f"{alternate_mode}/{prompt_mode}"] = {
            "by_scale": summarize_rows_by_scale(combo_rows),
            "region_stats": summarize_region_stats(combo_rows),
        }
    return summary


def summarize_rows_by_scale(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_scale: dict[str, Any] = {}
    for scale in sorted({float(row["scale"]) for row in rows}):
        scale_rows = [row for row in rows if float(row["scale"]) == scale]
        by_variant = {}
        for variant in VARIANTS:
            variant_rows = [row for row in scale_rows if row["variant"] == variant]
            by_variant[variant] = {
                key: finite_mean([float(row[key]) for row in variant_rows])
                for key in (
                    "target_full_l1",
                    "target_tumor_l1",
                    "paired_output_full_l1",
                    "paired_output_tumor_l1",
                    "output_edge_energy",
                )
            }
        by_scale[str(scale)] = by_variant
    return by_scale


def summarize_region_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "region_active_q",
        "region_missing_q",
        "region_fallback_q",
        "region_null_q",
        "region_allowed_pairs",
    )
    return {
        key: {
            "mean": finite_mean([float(row[key]) for row in rows if key in row]),
            "min": finite_min([float(row[key]) for row in rows if key in row]),
            "max": finite_max([float(row[key]) for row in rows if key in row]),
        }
        for key in keys
    }


def summarize_combo_region_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    combos = sorted(
        {
            (
                str(row.get("alternate_mode") or "same_dataset"),
                str(row.get("prompt_mode") or "dataset"),
            )
            for row in rows
        }
    )
    return {
        f"{alternate_mode}/{prompt_mode}": summarize_region_stats(
            [
                row
                for row in rows
                if str(row.get("alternate_mode") or "same_dataset") == alternate_mode
                and str(row.get("prompt_mode") or "dataset") == prompt_mode
            ]
        )
        for alternate_mode, prompt_mode in combos
    }


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else math.nan


def finite_min(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.min(finite)) if finite else math.nan


def finite_max(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.max(finite)) if finite else math.nan


def make_probe_panel(
    *,
    sample_id: str,
    target: Image.Image,
    paired_reference: Image.Image,
    alternate_reference: Image.Image,
    predictions: dict[tuple[float, str], Image.Image],
    scales: list[float],
    thumbnail_size: int,
) -> Image.Image:
    columns = 5
    rows = 1 + len(scales)
    label_height = 28
    width = columns * thumbnail_size
    height = rows * (thumbnail_size + label_height)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    header = [
        ("Target", target),
        ("Paired ref", paired_reference),
        ("Alternate ref", alternate_reference),
        ("Zero ref", Image.new("RGB", paired_reference.size, "black")),
        (safe_name(sample_id)[:24], target),
    ]
    for column, (label, image) in enumerate(header):
        paste_cell(canvas, draw, font, image, label, column, 0, thumbnail_size, label_height)

    for row_index, scale in enumerate(scales, start=1):
        paired = predictions[(scale, "paired")]
        alternate = predictions[(scale, "alternate_feature")]
        zero = predictions[(scale, "zero")]
        difference = difference_image(paired, alternate)
        cells = [
            (f"scale {scale:g}", target),
            ("paired", paired),
            ("alternate feature", alternate),
            ("zero", zero),
            ("|paired-alt|", difference),
        ]
        for column, (label, image) in enumerate(cells):
            paste_cell(
                canvas,
                draw,
                font,
                image,
                label,
                column,
                row_index,
                thumbnail_size,
                label_height,
            )
    return canvas


def paste_cell(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    image: Image.Image,
    label: str,
    column: int,
    row: int,
    size: int,
    label_height: int,
) -> None:
    x = column * size
    y = row * (size + label_height)
    thumb = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    canvas.paste(thumb, (x, y))
    draw.text((x + 4, y + size + 7), label, fill="black", font=font)


def difference_image(left: Image.Image, right: Image.Image) -> Image.Image:
    left_array = image_array(left)
    right_array = image_array(right)
    difference = np.abs(left_array - right_array)
    scale = max(float(np.quantile(difference, 0.99)), 1e-6)
    visual = np.clip(difference / scale, 0.0, 1.0)
    return Image.fromarray((visual * 255.0).round().astype(np.uint8), mode="RGB")


def make_overview(paths: list[Path]) -> Image.Image:
    panels = [Image.open(path).convert("RGB") for path in paths]
    width = max(panel.width for panel in panels)
    height = sum(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for panel in panels:
        canvas.paste(panel, (0, y))
        y += panel.height
    return canvas


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


if __name__ == "__main__":
    raise SystemExit(main())

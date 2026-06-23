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
import os
import random
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageFont

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_config import FINE_TO_PARENT


VARIANTS = ("paired", "alternate_feature", "zero")
ALTERNATE_MODES = ("same_dataset", "different_dataset")
PROMPT_MODES = ("dataset", "empty")
NUCLEI_STAIN_LABEL_OFFSET = 256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross V1 fixed-seed generation gate.")
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
    parser.add_argument(
        "--regional-ip-soft-bias",
        type=float,
        default=None,
        help=(
            "Optional inference-time override for global_soft_bias IP routing. "
            "Same-label pairs get +b and other-label pairs get -b. Omit to use checkpoint weights."
        ),
    )
    parser.add_argument(
        "--color-match",
        choices=("none", "lab", "macenko", "hed", "hd"),
        default="lab",
        help=(
            "Postprocess predictions with stain normalization. "
            "'lab' matches mean/std in Lab space; 'macenko' matches H&E stain "
            "statistics to the paired reference. 'hed'/'hd' are aliases for "
            "macenko."
        ),
    )
    parser.add_argument(
        "--color-match-scope",
        choices=("region", "global"),
        default="region",
        help=(
            "For Macenko stain matching, match by target/reference "
            "tissue+nuclei composite masks or over all non-background tissue."
        ),
    )
    parser.add_argument("--color-match-strength", type=float, default=1.0)
    parser.add_argument(
        "--color-match-concentration-stat",
        choices=("p99", "mean-std"),
        default="p99",
        help=(
            "For Macenko, match stain concentrations by p99 scaling or by "
            "mean/std affine statistics. mean-std usually tracks global stain "
            "tone and contrast more aggressively."
        ),
    )
    parser.add_argument("--color-match-background-label", type=int, default=0)
    parser.add_argument(
        "--color-match-fallback",
        choices=("pooled", "skip"),
        default="pooled",
        help="Fallback for regional stain labels missing from the paired reference mask.",
    )
    parser.add_argument("--color-match-macenko-io", type=float, default=240.0)
    parser.add_argument("--color-match-macenko-beta", type=float, default=0.15)
    parser.add_argument("--color-match-macenko-alpha", type=float, default=1.0)
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

    if args.selection_manifest:
        selected = select_records_from_manifest(
            Path(args.selection_manifest),
            records,
            alternate_modes=alternate_modes,
        )
    else:
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
        set_ip_soft_bias,
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
    if args.color_match != "none":
        print(
            "color_match "
            f"method={normalize_color_match_method(args.color_match)} "
            f"requested={args.color_match} "
            f"scope={args.color_match_scope} "
            "reference=paired_reference"
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
                        raw_path = combo_dir / f"prediction_scale_{scale:g}_{variant}_raw.png"
                        prediction.save(raw_path)
                        prediction = apply_prediction_color_match(
                            args=args,
                            source=prediction,
                            reference=paired_ref_pil,
                            target_tissue_mask=target_tissue,
                            target_nuclei_mask=target_nuclei,
                            reference_tissue_mask=paired_ref_tissue,
                            reference_nuclei_mask=paired_ref_nuclei,
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
                        reference_tissue=np.asarray(paired_ref_tissue),
                        reference_image=paired_ref_pil,
                        tumor_label=args.tumor_label,
                        predictions=predictions,
                    )
                    for row in rows:
                        row["regional_ip_soft_bias"] = (
                            float(args.regional_ip_soft_bias)
                            if args.regional_ip_soft_bias is not None
                            else math.nan
                        )
                        row["regional_ip_soft_bias_applied"] = bool(
                            soft_bias_override and soft_bias_override.get("applied", False)
                        )
                        row["color_match_method"] = args.color_match
                        row["color_match_normalized_method"] = normalize_color_match_method(
                            args.color_match
                        )
                        row["color_match_applied"] = bool(args.color_match != "none")
                        row["color_match_reference_mode"] = "paired_reference"
                        row["color_match_scope"] = args.color_match_scope
                        row["color_match_strength"] = float(args.color_match_strength)
                        row["color_match_concentration_stat"] = args.color_match_concentration_stat
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
            "color_match": {
                "method": args.color_match,
                "normalized_method": normalize_color_match_method(args.color_match),
                "applied": args.color_match != "none",
                "reference_mode": "paired_reference",
                "scope": args.color_match_scope,
                "strength": float(args.color_match_strength),
                "concentration_stat": args.color_match_concentration_stat,
                "background_label": int(args.color_match_background_label),
                "fallback": args.color_match_fallback,
                "macenko_io": float(args.color_match_macenko_io),
                "macenko_beta": float(args.color_match_macenko_beta),
                "macenko_alpha": float(args.color_match_macenko_alpha),
            },
            "regional_ip_soft_bias": (
                float(args.regional_ip_soft_bias) if args.regional_ip_soft_bias is not None else None
            ),
            "regional_ip_soft_bias_override": soft_bias_override,
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


def select_records_from_manifest(
    manifest_path: Path,
    records: list[dict[str, Any]],
    *,
    alternate_modes: list[str],
) -> list[tuple[int, dict[str, Any], dict[str, dict[str, Any]]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    by_target_ref = {
        (path_key(row.get("target_image")), path_key(row.get("reference_image"))): (index, row)
        for index, row in enumerate(records)
        if row.get("target_image") and row.get("reference_image")
    }
    by_sample_ref = {
        (record_sample_id(row), reference_sample_id(row)): (index, row)
        for index, row in enumerate(records)
        if row.get("target_image") and row.get("reference_image")
    }
    by_ref: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_ref_sample_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row.get("reference_image"):
            by_ref[path_key(row.get("reference_image"))].append(row)
            by_ref_sample_id[reference_sample_id(row)].append(row)
    selected = []
    for item in manifest:
        target_key = path_key(item.get("target_image"))
        paired_ref_key = path_key(item.get("paired_reference_image"))
        found = by_target_ref.get((target_key, paired_ref_key))
        if found is None:
            sample_key = (
                str(item.get("sample_id") or ""),
                str(item.get("paired_reference_sample_id") or ""),
            )
            found = by_sample_ref.get(sample_key)
        if found is None:
            raise ValueError(
                f"manifest probe not found in metadata: target={item.get('target_image')} "
                f"paired_ref={item.get('paired_reference_image')}"
            )
        metadata_index, paired = found
        alternates: dict[str, dict[str, Any]] = {}
        for mode in alternate_modes:
            payload = (item.get("alternates") or {}).get(mode)
            if not payload:
                raise ValueError(f"manifest missing alternate mode {mode!r}")
            candidates = by_ref.get(path_key(payload.get("reference_image"))) or []
            if not candidates:
                candidates = by_ref_sample_id.get(str(payload.get("reference_sample_id") or "")) or []
            if not candidates:
                raise ValueError(f"manifest alternate not found in metadata: {payload}")
            alternates[mode] = candidates[0]
        selected.append((int(item.get("metadata_index", metadata_index)), paired, alternates))
    return selected


def path_key(value: Any) -> str:
    if value is None:
        return ""
    return str(Path(str(value).replace("\\", "/")).expanduser())


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
    reference_tissue: np.ndarray,
    reference_image: Image.Image,
    tumor_label: int,
    predictions: dict[tuple[float, str], Image.Image],
) -> list[dict[str, Any]]:
    arrays = {
        variant: image_array(predictions[(scale, variant)])
        for variant in VARIANTS
    }
    target_array = image_array(target)
    reference_array = image_array(reference_image)
    tumor_mask = np.asarray(target_tissue) == int(tumor_label)
    target_stroma_mask = coarse_label_mask(np.asarray(target_tissue), 2)
    reference_tumor_mask = coarse_label_mask(np.asarray(reference_tissue), int(tumor_label))
    reference_stroma_mask = coarse_label_mask(np.asarray(reference_tissue), 2)
    rows = []
    for variant in VARIANTS:
        stroma_to_ref_tumor = descriptor_distance(
            arrays[variant],
            target_stroma_mask,
            reference_array,
            reference_tumor_mask,
        )
        stroma_to_ref_stroma = descriptor_distance(
            arrays[variant],
            target_stroma_mask,
            reference_array,
            reference_stroma_mask,
        )
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
                "texture_probe_stroma_to_ref_tumor_l2": stroma_to_ref_tumor,
                "texture_probe_stroma_to_ref_stroma_l2": stroma_to_ref_stroma,
                "texture_probe_tumor_minus_stroma_margin": (
                    stroma_to_ref_tumor - stroma_to_ref_stroma
                    if math.isfinite(stroma_to_ref_tumor) and math.isfinite(stroma_to_ref_stroma)
                    else math.nan
                ),
            }
        )
    return rows


def _match_image_color_to_reference(
    *,
    source: Image.Image,
    reference: Image.Image,
    method: str,
) -> Image.Image:
    method = normalize_color_match_method(method)
    if method == "lab":
        return _mean_std_transfer_pil_lab(source=source, reference=reference)
    raise ValueError(f"Unsupported color match method: {method}")


def normalize_color_match_method(method: str) -> str:
    normalized = str(method or "none").strip().lower()
    if normalized in {"hed", "hd"}:
        return "macenko"
    return normalized


def apply_prediction_color_match(
    *,
    args: argparse.Namespace,
    source: Image.Image,
    reference: Image.Image,
    target_tissue_mask: Any,
    target_nuclei_mask: Any,
    reference_tissue_mask: Any,
    reference_nuclei_mask: Any,
) -> Image.Image:
    method = normalize_color_match_method(args.color_match)
    source_rgb = source.convert("RGB")
    reference_rgb = reference.convert("RGB")
    if method == "none":
        return source_rgb
    if method == "lab":
        matched = _mean_std_transfer_pil_lab(source=source_rgb, reference=reference_rgb)
    elif method == "macenko":
        matched = macenko_match_prediction_to_reference(
            source=source_rgb,
            reference=reference_rgb,
            target_tissue_mask=target_tissue_mask,
            target_nuclei_mask=target_nuclei_mask,
            reference_tissue_mask=reference_tissue_mask,
            reference_nuclei_mask=reference_nuclei_mask,
            scope=args.color_match_scope,
            background_label=args.color_match_background_label,
            fallback=args.color_match_fallback,
            concentration_stat=args.color_match_concentration_stat,
            io=args.color_match_macenko_io,
            beta=args.color_match_macenko_beta,
            alpha=args.color_match_macenko_alpha,
        )
    else:
        raise ValueError(f"Unsupported color match method: {args.color_match}")
    return blend_images(
        base=source_rgb,
        edited=matched,
        strength=float(args.color_match_strength),
    )


def macenko_match_prediction_to_reference(
    *,
    source: Image.Image,
    reference: Image.Image,
    target_tissue_mask: Any,
    target_nuclei_mask: Any,
    reference_tissue_mask: Any,
    reference_nuclei_mask: Any,
    scope: str,
    background_label: int,
    fallback: str,
    concentration_stat: str,
    io: float,
    beta: float,
    alpha: float,
) -> Image.Image:
    source_rgb = source.convert("RGB")
    reference_rgb = reference.convert("RGB")
    source_array = np.asarray(source_rgb, dtype=np.uint8)
    reference_array = np.asarray(reference_rgb, dtype=np.uint8)
    normalized_scope = str(scope or "region").strip().lower()
    if normalized_scope == "region":
        target_mask = composite_stain_mask(
            target_tissue_mask,
            target_nuclei_mask,
            size=source_rgb.size,
        )
        reference_mask = composite_stain_mask(
            reference_tissue_mask,
            reference_nuclei_mask,
            size=reference_rgb.size,
        )
        matched_array = macenko_stain_transfer_by_mask(
            source_array,
            reference_array,
            target_mask,
            reference_mask,
            background_label=int(background_label),
            fallback=fallback,
            concentration_stat=concentration_stat,
            io=float(io),
            beta=float(beta),
            alpha=float(alpha),
        )
    elif normalized_scope == "global":
        target_tissue = mask_to_numpy(target_tissue_mask, size=source_rgb.size)
        reference_tissue = mask_to_numpy(reference_tissue_mask, size=reference_rgb.size)
        bg = int(background_label)
        matched_array = macenko_stain_transfer(
            source_array,
            reference_array,
            source_mask=target_tissue != bg,
            reference_mask=reference_tissue != bg,
            concentration_stat=concentration_stat,
            io=float(io),
            beta=float(beta),
            alpha=float(alpha),
        )
    else:
        raise ValueError(f"Unsupported color-match scope: {scope}")
    return Image.fromarray(matched_array, mode="RGB")


def blend_images(*, base: Image.Image, edited: Image.Image, strength: float) -> Image.Image:
    alpha = float(np.clip(strength, 0.0, 1.0))
    if alpha >= 1.0:
        return edited.convert("RGB")
    if alpha <= 0.0:
        return base.convert("RGB")
    base_array = np.asarray(base.convert("RGB"), dtype=np.float32)
    edited_array = np.asarray(edited.convert("RGB"), dtype=np.float32)
    if base_array.shape != edited_array.shape:
        edited_array = np.asarray(
            edited.convert("RGB").resize(base.size, Image.Resampling.BICUBIC),
            dtype=np.float32,
        )
    output = base_array * (1.0 - alpha) + edited_array * alpha
    return Image.fromarray(np.clip(output.round(), 0, 255).astype(np.uint8), mode="RGB")


def composite_stain_mask(
    tissue_mask: Any,
    nuclei_mask: Any,
    *,
    size: tuple[int, int],
) -> np.ndarray:
    tissue = mask_to_numpy(tissue_mask, size=size).copy()
    nuclei = mask_to_numpy(nuclei_mask, size=size)
    nuclei_pixels = nuclei != 0
    tissue[nuclei_pixels] = nuclei[nuclei_pixels] + NUCLEI_STAIN_LABEL_OFFSET
    return tissue


def mask_to_numpy(mask: Any, *, size: tuple[int, int]) -> np.ndarray:
    import torch

    if torch.is_tensor(mask):
        array = mask.detach().cpu().numpy()
    else:
        array = np.asarray(mask)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3:
        array = array[..., 0]
    array = np.asarray(array)
    target_width, target_height = tuple(size)
    if array.shape != (target_height, target_width):
        if array.size == 0 or int(np.nanmax(array)) <= 255:
            image_array = array.astype(np.uint8, copy=False)
        else:
            image_array = array.astype(np.uint16, copy=False)
        resized = Image.fromarray(image_array).resize(size, Image.Resampling.NEAREST)
        array = np.asarray(resized)
    return array.astype(np.int64, copy=False)


def macenko_stain_transfer_by_mask(
    source: np.ndarray,
    reference: np.ndarray,
    target_mask: np.ndarray,
    reference_mask: np.ndarray,
    *,
    background_label: int = 0,
    fallback: str = "pooled",
    concentration_stat: str = "p99",
    io: float = 240.0,
    beta: float = 0.15,
    alpha: float = 1.0,
    min_region_pixels: int = 10,
) -> np.ndarray:
    source = np.asarray(source, dtype=np.uint8)
    reference = np.asarray(reference, dtype=np.uint8)
    output = np.asarray(source, dtype=np.uint8).copy()
    target_mask = np.asarray(target_mask)
    reference_mask = np.asarray(reference_mask)
    pooled_source = target_mask != int(background_label)
    pooled_reference = reference_mask != int(background_label)
    he_source = estimate_macenko_stain_matrix(
        source,
        mask=pooled_source,
        io=io,
        beta=beta,
        alpha=alpha,
    )
    he_reference = estimate_macenko_stain_matrix(
        reference,
        mask=pooled_reference,
        io=io,
        beta=beta,
        alpha=alpha,
    )
    conc_source = macenko_concentrations(source, he_source, io=io)
    conc_reference = macenko_concentrations(reference, he_reference, io=io)
    target_labels = [
        int(label)
        for label in np.unique(target_mask)
        if int(label) != int(background_label)
    ]
    reference_labels = {
        int(label)
        for label in np.unique(reference_mask)
        if int(label) != int(background_label)
    }
    pooled_reference = reference_mask != int(background_label)
    fallback_mode = str(fallback or "pooled").strip().lower()
    for label in sorted(target_labels):
        source_region = target_mask == int(label)
        if int(source_region.sum()) < int(min_region_pixels):
            continue
        if label in reference_labels and int((reference_mask == label).sum()) >= int(min_region_pixels):
            reference_region = reference_mask == label
        elif fallback_mode == "pooled" and int(pooled_reference.sum()) >= int(min_region_pixels):
            reference_region = pooled_reference
        else:
            continue
        transferred = macenko_apply_concentration_match(
            source,
            conc_source,
            conc_reference,
            he_reference,
            source_mask=source_region,
            reference_mask=reference_region,
            concentration_stat=concentration_stat,
            io=io,
        )
        output[source_region] = transferred[source_region]
    return output


def macenko_apply_concentration_match(
    source: np.ndarray,
    conc_source: np.ndarray,
    conc_reference: np.ndarray,
    reference_stain_matrix: np.ndarray,
    *,
    source_mask: np.ndarray,
    reference_mask: np.ndarray,
    concentration_stat: str = "p99",
    io: float = 240.0,
) -> np.ndarray:
    source = np.asarray(source, dtype=np.uint8)
    h, w, _ = source.shape
    source_select = valid_bool_mask(source_mask, (h, w))
    if source_select is None:
        return source.copy()
    reference_select = np.asarray(reference_mask, dtype=bool)
    if not np.any(reference_select):
        return source.copy()
    source_flat_mask = source_select.reshape(-1)
    reference_flat_mask = reference_select.reshape(-1)
    if int(source_flat_mask.sum()) < 1 or int(reference_flat_mask.sum()) < 1:
        return source.copy()
    region_conc = match_stain_concentrations(
        conc_source[source_flat_mask],
        conc_reference[reference_flat_mask],
        concentration_stat=concentration_stat,
    )
    region_rgb = od_to_rgb(region_conc @ reference_stain_matrix, io=io)
    output = source.copy().reshape(-1, 3)
    output[source_flat_mask] = region_rgb
    return output.reshape(h, w, 3)


def macenko_stain_transfer(
    source: np.ndarray,
    reference: np.ndarray,
    *,
    source_mask: np.ndarray | None = None,
    reference_mask: np.ndarray | None = None,
    concentration_stat: str = "p99",
    io: float = 240.0,
    beta: float = 0.15,
    alpha: float = 1.0,
) -> np.ndarray:
    source = np.asarray(source, dtype=np.uint8)
    reference = np.asarray(reference, dtype=np.uint8)
    h, w, _ = source.shape
    source_select = valid_bool_mask(source_mask, (h, w))
    reference_select = valid_bool_mask(reference_mask, reference.shape[:2])
    he_source = estimate_macenko_stain_matrix(
        source,
        mask=source_select,
        io=io,
        beta=beta,
        alpha=alpha,
    )
    he_reference = estimate_macenko_stain_matrix(
        reference,
        mask=reference_select,
        io=io,
        beta=beta,
        alpha=alpha,
    )
    conc_source = macenko_concentrations(source, he_source, io=io)
    conc_reference = macenko_concentrations(reference, he_reference, io=io)
    source_flat_mask = (
        source_select.reshape(-1)
        if source_select is not None
        else np.ones((h * w,), dtype=bool)
    )
    reference_flat_mask = (
        reference_select.reshape(-1)
        if reference_select is not None
        else np.ones((reference.shape[0] * reference.shape[1],), dtype=bool)
    )
    if int(source_flat_mask.sum()) < 1 or int(reference_flat_mask.sum()) < 1:
        return source.copy()
    conc_matched = conc_source.copy()
    conc_matched[source_flat_mask] = match_stain_concentrations(
        conc_source[source_flat_mask],
        conc_reference[reference_flat_mask],
        concentration_stat=concentration_stat,
    )
    od_new = conc_matched @ he_reference
    rgb_new = od_to_rgb(od_new.reshape(h, w, 3), io=io)
    output = source.copy()
    if source_select is None:
        output = rgb_new
    else:
        output[source_select] = rgb_new[source_select]
    return output


def match_stain_concentrations(
    source_conc: np.ndarray,
    reference_conc: np.ndarray,
    *,
    concentration_stat: str = "p99",
) -> np.ndarray:
    source = np.asarray(source_conc, dtype=np.float64)
    reference = np.asarray(reference_conc, dtype=np.float64)
    if source.size == 0 or reference.size == 0:
        return source.copy()
    mode = str(concentration_stat or "p99").strip().lower()
    if mode == "p99":
        max_source = np.percentile(source, 99, axis=0)
        max_reference = np.percentile(reference, 99, axis=0)
        max_source = np.where(max_source < 1e-6, 1e-6, max_source)
        return np.clip(source * (max_reference / max_source)[None, :], 0.0, None)
    if mode == "mean-std":
        source_mean = source.mean(axis=0)
        source_std = source.std(axis=0)
        reference_mean = reference.mean(axis=0)
        reference_std = reference.std(axis=0)
        source_std = np.where(source_std < 1e-6, 1.0, source_std)
        matched = (source - source_mean[None, :]) * (reference_std / source_std)[None, :]
        matched = matched + reference_mean[None, :]
        reference_cap = np.percentile(reference, 99.5, axis=0)
        reference_cap = np.maximum(reference_cap, reference_mean + 3.0 * reference_std)
        return np.clip(matched, 0.0, reference_cap[None, :])
    raise ValueError(
        "--color-match-concentration-stat must be one of: p99, mean-std"
    )


def rgb_to_od(image: np.ndarray, io: float = 240.0) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    return -np.log((image + 1.0) / float(io))


def od_to_rgb(od: np.ndarray, io: float = 240.0) -> np.ndarray:
    rgb = float(io) * np.exp(-np.asarray(od, dtype=np.float64))
    return np.clip(rgb, 0, 255).astype(np.uint8)


def estimate_macenko_stain_matrix(
    image: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    io: float = 240.0,
    beta: float = 0.15,
    alpha: float = 1.0,
) -> np.ndarray:
    od = rgb_to_od(image, io=io).reshape(-1, 3)
    if mask is not None:
        od = od[np.asarray(mask, dtype=bool).reshape(-1)]
    od = od[np.all(np.isfinite(od), axis=1)]
    if od.shape[0] < 3:
        return default_he_matrix()
    od = np.clip(od, 0.0, None)
    stain_strength = np.linalg.norm(od, axis=1)
    od_hat = od[stain_strength > float(beta)]
    if od_hat.shape[0] < 10:
        relaxed_threshold = max(float(beta) * 0.25, 1e-6)
        od_hat = od[stain_strength > relaxed_threshold]
    if od_hat.shape[0] < 3:
        return default_he_matrix()
    try:
        cov = np.cov(od_hat.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        if not np.all(np.isfinite(eigvals)) or not np.all(np.isfinite(eigvecs)):
            return default_he_matrix()
        order = np.argsort(eigvals)[::-1][:2]
        v = eigvecs[:, order]
        if v[0, 0] < 0:
            v[:, 0] *= -1
        if v[0, 1] < 0:
            v[:, 1] *= -1
        projection = od_hat @ v
        phi = np.arctan2(projection[:, 1], projection[:, 0])
        min_phi = np.percentile(phi, float(alpha))
        max_phi = np.percentile(phi, 100.0 - float(alpha))
        v1 = v @ np.array([np.cos(min_phi), np.sin(min_phi)])
        v2 = v @ np.array([np.cos(max_phi), np.sin(max_phi)])
        he = np.array([v1, v2]) if v1[0] > v2[0] else np.array([v2, v1])
        he = np.clip(he, 1e-6, None)
        norms = np.linalg.norm(he, axis=1, keepdims=True)
        if np.any(norms < 1e-8) or not np.all(np.isfinite(he)):
            return default_he_matrix()
        return he / norms
    except np.linalg.LinAlgError:
        return default_he_matrix()


def macenko_concentrations(
    image: np.ndarray,
    stain_matrix: np.ndarray,
    *,
    io: float = 240.0,
) -> np.ndarray:
    od = rgb_to_od(image, io=io).reshape(-1, 3)
    concentrations = np.linalg.lstsq(stain_matrix.T, od.T, rcond=None)[0].T
    return np.clip(concentrations, 0.0, None)


def default_he_matrix() -> np.ndarray:
    he = np.array(
        [
            [0.65, 0.70, 0.29],
            [0.07, 0.99, 0.11],
        ],
        dtype=np.float64,
    )
    return he / np.linalg.norm(he, axis=1, keepdims=True)


def valid_bool_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    value = np.asarray(mask, dtype=bool)
    if value.shape != tuple(shape):
        return None
    if not np.any(value):
        return None
    return value


@lru_cache(maxsize=1)
def _lab_color_transforms() -> tuple[Any, Any]:
    srgb_profile = ImageCms.createProfile("sRGB")
    lab_profile = ImageCms.createProfile("LAB")
    rgb_to_lab = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")
    lab_to_rgb = ImageCms.buildTransformFromOpenProfiles(lab_profile, srgb_profile, "LAB", "RGB")
    return rgb_to_lab, lab_to_rgb


def _mean_std_transfer_pil_lab(*, source: Image.Image, reference: Image.Image) -> Image.Image:
    rgb_to_lab, lab_to_rgb = _lab_color_transforms()
    source_rgb_image = source.convert("RGB")
    reference_rgb_image = reference.convert("RGB")
    source_rgb = np.asarray(source_rgb_image, dtype=np.uint8)
    reference_rgb = np.asarray(reference_rgb_image, dtype=np.uint8)
    source_lab = np.asarray(ImageCms.applyTransform(source_rgb_image, rgb_to_lab), dtype=np.float32)
    reference_lab = np.asarray(ImageCms.applyTransform(reference_rgb_image, rgb_to_lab), dtype=np.float32)
    source_mask = _tissue_mask_from_rgb(source_rgb.astype(np.float32) / 255.0)
    reference_mask = _tissue_mask_from_rgb(reference_rgb.astype(np.float32) / 255.0)

    if not np.any(source_mask) or not np.any(reference_mask):
        return source_rgb_image

    matched_lab = source_lab.copy()
    for channel in range(3):
        source_values = source_lab[..., channel][source_mask]
        reference_values = reference_lab[..., channel][reference_mask]
        source_std = float(source_values.std())
        reference_std = float(reference_values.std())
        matched_lab[..., channel][source_mask] = (
            (source_values - float(source_values.mean()))
            * (reference_std / max(source_std, 1e-6))
            + float(reference_values.mean())
        )

    matched_lab = np.clip(matched_lab, 0.0, 255.0).round().astype(np.uint8)
    matched_rgb = np.asarray(
        ImageCms.applyTransform(Image.fromarray(matched_lab, mode="LAB"), lab_to_rgb),
        dtype=np.uint8,
    )
    output = source_rgb.copy()
    output[source_mask] = matched_rgb[source_mask]
    return Image.fromarray(output, mode="RGB")


def _tissue_mask_from_rgb(rgb_float: np.ndarray, threshold: float = 0.85) -> np.ndarray:
    return rgb_float.mean(axis=-1) < threshold


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
    query_len = packed_flux_image_token_count(reference_image, bundle.flux_pipeline)
    _, key_labels = bundle.ref_encoder.encode_region_ip_tokens(
        reference_image.unsqueeze(0).to(device=device, dtype=bundle.torch_dtype),
        reference_tissue,
        nuclei_mask=reference_nuclei,
        token_mode=bundle.regional_ip_token_mode,
        label_mode=bundle.regional_ip_label_mode,
    )
    key_labels = key_labels.to(device=device)
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
        key_len=int(key_labels.shape[1]),
        device=query_labels.device,
        dtype=bundle.torch_dtype,
        strict=bool(getattr(bundle, "regional_ip_strict", True)),
        soft_bias=None,
        use_soft_bias=bool(getattr(bundle, "regional_ip_use_soft_bias", False)),
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


def coarse_label_mask(mask: np.ndarray, coarse_label: int) -> np.ndarray:
    values = np.asarray(mask)
    result = np.zeros(values.shape, dtype=bool)
    for fine_label in np.unique(values):
        fine_int = int(fine_label)
        parent = int(FINE_TO_PARENT.get(fine_int, fine_int))
        if parent == int(coarse_label):
            result |= values == fine_label
    return result


def rgb_descriptor(image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    valid = np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return None
    pixels = np.asarray(image, dtype=np.float32)[valid]
    return np.concatenate([pixels.mean(axis=0), pixels.std(axis=0)])


def descriptor_distance(
    left_image: np.ndarray,
    left_mask: np.ndarray,
    right_image: np.ndarray,
    right_mask: np.ndarray,
) -> float:
    left = rgb_descriptor(left_image, left_mask)
    right = rgb_descriptor(right_image, right_mask)
    if left is None or right is None:
        return math.nan
    return float(np.linalg.norm(left - right))


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
                    "texture_probe_stroma_to_ref_tumor_l2",
                    "texture_probe_stroma_to_ref_stroma_l2",
                    "texture_probe_tumor_minus_stroma_margin",
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

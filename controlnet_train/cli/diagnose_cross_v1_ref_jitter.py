"""Diagnose Cross V1 reference-image jitter sensitivity.

This diagnostic keeps the same target/task and the same reference identity, then
runs several random perturbations of the reference image. Reference masks are
kept fixed by default, so output differences mostly measure whether the
reference image/IP-Adapter branch is controlling appearance.
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
from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controlnet_train.cli.diagnose_cross_v1_ip_sensitivity import (
    collect_reference_feature_diagnostics,
    format_scale,
    image_l1,
    image_mse,
    make_overview,
    make_sample_panel,
    normalize_cross_records,
    parse_scale_values,
    pil_to_chw_float,
    read_cross_metadata,
    resolve_eval_prompt,
    safe_name,
    select_eval_records,
    set_ip_adapter_scale,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 1A diagnostic: fixed target/task/ref identity, random jittered "
            "reference images."
        )
    )
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--checkpoint", required=True, help="Cross V1 checkpoint dir.")
    parser.add_argument("--uni-checkpoint-path", required=True, help="UNI2-h pytorch_model.bin path.")
    parser.add_argument("--metadata", required=True, help="metadata_cross_{train,val}.json path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument(
        "--controlnet-conditioning-scales",
        default=None,
        help=(
            "Optional comma-separated ControlNet scale sweep. If omitted, uses "
            "--controlnet-conditioning-scale."
        ),
    )
    parser.add_argument("--ip-scale", type=float, default=1.0)
    parser.add_argument(
        "--ip-scales",
        default=None,
        help="Optional comma-separated IP scale sweep. If omitted, uses --ip-scale.",
    )
    parser.add_argument(
        "--regional-ip-soft-bias",
        type=float,
        default=None,
        help=(
            "Optional inference-time override for global_soft_bias IP routing. "
            "Same-label pairs get +b and other-label pairs get -b. Omit to use checkpoint weights."
        ),
    )
    parser.add_argument("--prompt-source", choices=["metadata", "dataset"], default="dataset")
    parser.add_argument("--prompt", default=None, help="Override every sample with one prompt.")
    parser.add_argument(
        "--jitter-types",
        default="hed,rgb,noise",
        help="Comma-separated jitter families from: hed, rgb, noise.",
    )
    parser.add_argument("--jitters-per-type", type=int, default=3)
    parser.add_argument("--rgb-jitter-strength", type=float, default=0.25)
    parser.add_argument("--noise-std", type=float, default=0.04)
    parser.add_argument("--hed-sigma", type=float, default=0.4)
    parser.add_argument("--hed-beta", type=float, default=0.08)
    parser.add_argument("--hed-strong-alpha-sampling", action="store_true", default=True)
    parser.add_argument("--no-hed-strong-alpha-sampling", dest="hed_strong_alpha_sampling", action="store_false")
    parser.add_argument("--hed-alpha-min", type=float, default=0.4)
    parser.add_argument("--hed-alpha-low", type=float, default=0.75)
    parser.add_argument("--hed-alpha-high", type=float, default=1.25)
    parser.add_argument("--hed-alpha-max", type=float, default=1.8)
    parser.add_argument(
        "--replace-reference-masks",
        action="store_true",
        help=(
            "Also perturb/replace reference masks if a future jitter type supports it. "
            "Current jitter types keep masks fixed either way."
        ),
    )
    parser.add_argument("--thumbnail-size", type=int, default=160)
    parser.add_argument("--overview-max-samples", type=int, default=12)
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def parse_jitter_types(value: str) -> list[str]:
    allowed = {"hed", "rgb", "noise"}
    result: list[str] = []
    for raw_part in value.split(","):
        part = raw_part.strip().lower()
        if not part:
            continue
        if part not in allowed:
            raise ValueError(f"Unsupported jitter type {part!r}; choose from {sorted(allowed)}.")
        if part not in result:
            result.append(part)
    if not result:
        raise ValueError("At least one jitter type is required.")
    return result


def main(argv=None) -> int:
    args = parse_args(argv)
    jitter_types = parse_jitter_types(args.jitter_types)
    ip_scales = parse_scale_values(args.ip_scales) if args.ip_scales else [float(args.ip_scale)]
    controlnet_scales = (
        parse_scale_values(args.controlnet_conditioning_scales)
        if args.controlnet_conditioning_scales
        else [float(args.controlnet_conditioning_scale)]
    )
    all_records = read_cross_metadata(args.metadata)
    records = select_eval_records(all_records, num_samples=args.num_samples, seed=args.seed)

    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    import torch

    from controlnet_train.inference.pipeline_cross_v1 import load_cross_v1_bundle, set_ip_soft_bias

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
        controlnet_conditioning_scale=controlnet_scales[0],
        ip_adapter_scale=ip_scales[0],
    )
    set_ip_adapter_scale(bundle.flux_pipeline.transformer, ip_scales[0])
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

    rows: list[dict[str, Any]] = []
    panel_paths: list[Path] = []
    rng = random.Random(args.seed)
    for index, record in enumerate(records):
        sample_rows, panel_path = run_sample_jitter_grid(
            bundle=bundle,
            record=record,
            sample_index=index,
            output_root=samples_dir,
            jitter_types=jitter_types,
            jitters_per_type=args.jitters_per_type,
            rng=rng,
            prompt_override=args.prompt,
            prompt_source=args.prompt_source,
            ip_scales=ip_scales,
            controlnet_scales=controlnet_scales,
            rgb_jitter_strength=args.rgb_jitter_strength,
            noise_std=args.noise_std,
            hed_config={
                "sigma": args.hed_sigma,
                "beta": args.hed_beta,
                "strong_alpha_sampling": args.hed_strong_alpha_sampling,
                "alpha_min": args.hed_alpha_min,
                "alpha_low": args.hed_alpha_low,
                "alpha_high": args.hed_alpha_high,
                "alpha_max": args.hed_alpha_max,
            },
            thumbnail_size=args.thumbnail_size,
        )
        for row in sample_rows:
            row["regional_ip_soft_bias"] = (
                float(args.regional_ip_soft_bias) if args.regional_ip_soft_bias is not None else math.nan
            )
            row["regional_ip_soft_bias_applied"] = bool(
                soft_bias_override and soft_bias_override.get("applied", False)
            )
        rows.extend(sample_rows)
        if panel_path is not None and len(panel_paths) < args.overview_max_samples:
            panel_paths.append(panel_path)
        print(f"[{index + 1}/{len(records)}] {sample_rows[0]['sample_id']} wrote {len(sample_rows)} outputs")

    write_rows(output_dir, rows)
    summary = aggregate_jitter_rows(rows)
    summary["ip_scale"] = float(ip_scales[0]) if len(ip_scales) == 1 else None
    summary["ip_scales"] = ip_scales
    summary["regional_ip_soft_bias"] = (
        float(args.regional_ip_soft_bias) if args.regional_ip_soft_bias is not None else None
    )
    summary["regional_ip_soft_bias_override"] = soft_bias_override
    summary["controlnet_conditioning_scales"] = controlnet_scales
    summary["jitter_types"] = jitter_types
    summary["jitters_per_type"] = int(args.jitters_per_type)
    summary["fixed_reference_masks"] = not bool(args.replace_reference_masks)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf8",
    )
    if panel_paths:
        make_overview(panel_paths).save(output_dir / "overview_grid.png")
    print(f"wrote ref-jitter diagnostic outputs to {output_dir}")
    return 0


def run_sample_jitter_grid(
    *,
    bundle,
    record: dict[str, Any],
    sample_index: int,
    output_root: Path,
    jitter_types: list[str],
    jitters_per_type: int,
    rng: random.Random,
    prompt_override: str | None,
    prompt_source: str,
    ip_scales: list[float],
    controlnet_scales: list[float],
    rgb_jitter_strength: float,
    noise_std: float,
    hed_config: dict[str, Any],
    thumbnail_size: int,
) -> tuple[list[dict[str, Any]], Path | None]:
    sample_id = str(record.get("sample_id") or Path(record["target_image"]).stem)
    ref_id = str(record.get("reference_sample_id") or Path(record["reference_image"]).stem)
    sample_dir = output_root / f"{sample_index:04d}_{safe_name(sample_id)}__ref_{safe_name(ref_id)}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    import torch

    from controlnet_train.data.common import (
        default_prompt_for_dataset,
        load_image_tensor,
        load_nuclei_mask,
        load_tissue_mask,
    )
    from controlnet_train.inference.pipeline_cross_v1 import run_cross_v1_bundle

    reference_image_path = Path(record["reference_image"])
    target_image_path = Path(record["target_image"])
    reference_image = load_image_tensor(reference_image_path)
    reference_tissue_mask = load_tissue_mask(record["reference_tissue_mask"])
    reference_nuclei_mask = load_nuclei_mask(record["reference_nuclei_mask"])
    target_tissue_mask = load_tissue_mask(record["target_tissue_mask"])
    target_nuclei_mask = load_nuclei_mask(record["target_nuclei_mask"])
    target_pil = Image.open(target_image_path).convert("RGB")
    reference_pil = Image.open(reference_image_path).convert("RGB")
    reference_pil.save(sample_dir / "reference_original.png")
    target_pil.save(sample_dir / "target.png")

    variant_inputs = build_jittered_reference_variants(
        reference_image=reference_image,
        reference_tissue_mask=reference_tissue_mask,
        reference_nuclei_mask=reference_nuclei_mask,
        jitter_types=jitter_types,
        jitters_per_type=jitters_per_type,
        rng=rng,
        rgb_jitter_strength=rgb_jitter_strength,
        noise_std=noise_std,
        hed_config=hed_config,
    )
    variants = list(variant_inputs.keys())

    prompt = resolve_eval_prompt(
        record=record,
        prompt_override=prompt_override,
        prompt_source=prompt_source,
        default_prompt_for_dataset=default_prompt_for_dataset,
    )

    rows: list[dict[str, Any]] = []
    outputs: dict[tuple[str, float, float], Image.Image] = {}
    output_arrays: dict[tuple[str, float, float], np.ndarray] = {}
    target_array = pil_to_chw_float(target_pil)

    with torch.no_grad():
        feature_diagnostics = collect_reference_feature_diagnostics(
            bundle=bundle,
            variant_inputs=variant_inputs,
            variants=variants,
        )
        (sample_dir / "reference_feature_diagnostics.json").write_text(
            json.dumps(feature_diagnostics, indent=2, ensure_ascii=False, allow_nan=True),
            encoding="utf8",
        )
        for variant_name, variant in variant_inputs.items():
            jitter_ref_pil = tensor_to_pil(variant["image"])
            jitter_ref_pil.save(sample_dir / f"reference_{variant_name}.png")
            for controlnet_scale in controlnet_scales:
                bundle.controlnet_conditioning_scale = float(controlnet_scale)
                for ip_scale in ip_scales:
                    set_ip_adapter_scale(bundle.flux_pipeline.transformer, ip_scale)
                    prediction = run_cross_v1_bundle(
                        bundle,
                        reference_image=variant["image"],
                        reference_tissue_mask=variant["tissue_mask"],
                        reference_nuclei_mask=variant["nuclei_mask"],
                        target_tissue_mask=target_tissue_mask,
                        target_nuclei_mask=target_nuclei_mask,
                        prompt=prompt,
                    )
                    key = (variant_name, ip_scale, controlnet_scale)
                    outputs[key] = prediction
                    output_arrays[key] = pil_to_chw_float(prediction)
                    prediction_path = (
                        sample_dir
                        / (
                            f"{variant_name}_ip_scale_{format_scale(ip_scale)}"
                            f"_cn_scale_{format_scale(controlnet_scale)}.png"
                        )
                    )
                    prediction.save(prediction_path)
                    del prediction
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    rows.append(
                        {
                            "sample_index": sample_index,
                            "sample_id": sample_id,
                            "reference_sample_id": ref_id,
                            "variant": variant_name,
                            "jitter_type": variant["jitter_type"],
                            "jitter_seed": variant["jitter_seed"],
                            "ip_scale": float(ip_scale),
                            "controlnet_scale": float(controlnet_scale),
                            "path": str(prediction_path),
                            "reference_jitter_path": str(sample_dir / f"reference_{variant_name}.png"),
                            "l1_to_target": image_l1(output_arrays[key], target_array),
                            "mse_to_target": image_mse(output_arrays[key], target_array),
                            "reference_l1_vs_normal": image_l1(
                                pil_to_chw_float(jitter_ref_pil),
                                pil_to_chw_float(reference_pil),
                            ),
                            **feature_diagnostics.get(variant_name, {}),
                        }
                    )

    normal_outputs = {
        (ip_scale, controlnet_scale): output_arrays[("normal", ip_scale, controlnet_scale)]
        for controlnet_scale in controlnet_scales
        for ip_scale in ip_scales
        if ("normal", ip_scale, controlnet_scale) in output_arrays
    }
    for row in rows:
        ip_scale = float(row["ip_scale"])
        controlnet_scale = float(row["controlnet_scale"])
        current = output_arrays[(str(row["variant"]), ip_scale, controlnet_scale)]
        normal_output = normal_outputs.get((ip_scale, controlnet_scale))
        if normal_output is not None:
            row["l1_vs_normal_output"] = image_l1(current, normal_output)
            row["mse_vs_normal_output"] = image_mse(current, normal_output)

    (sample_dir / "diagnostics.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf8",
    )
    panel = make_sample_panel(
        reference=reference_pil,
        target=target_pil,
        outputs=outputs,
        variants=variants,
        scales=ip_scales,
        controlnet_scales=controlnet_scales,
        thumbnail_size=thumbnail_size,
        title=f"{sample_id} | ref={ref_id} | ref jitter only",
    )
    panel_path = sample_dir / "ref_jitter_grid.png"
    panel.save(panel_path)
    return rows, panel_path


def build_jittered_reference_variants(
    *,
    reference_image,
    reference_tissue_mask,
    reference_nuclei_mask,
    jitter_types: list[str],
    jitters_per_type: int,
    rng: random.Random,
    rgb_jitter_strength: float,
    noise_std: float,
    hed_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {
        "normal": {
            "image": reference_image,
            "tissue_mask": reference_tissue_mask,
            "nuclei_mask": reference_nuclei_mask,
            "jitter_type": "normal",
            "jitter_seed": None,
        }
    }
    count = max(0, int(jitters_per_type))
    for jitter_type in jitter_types:
        for jitter_index in range(count):
            jitter_seed = rng.randrange(0, 2**31 - 1)
            variant_name = f"{jitter_type}_{jitter_index:02d}_seed_{jitter_seed}"
            result[variant_name] = {
                "image": perturb_reference_image(
                    reference_image,
                    jitter_type=jitter_type,
                    seed=jitter_seed,
                    rgb_jitter_strength=rgb_jitter_strength,
                    noise_std=noise_std,
                    hed_config=hed_config,
                ),
                "tissue_mask": reference_tissue_mask,
                "nuclei_mask": reference_nuclei_mask,
                "jitter_type": jitter_type,
                "jitter_seed": int(jitter_seed),
            }
    return result


def perturb_reference_image(
    image,
    *,
    jitter_type: str,
    seed: int,
    rgb_jitter_strength: float,
    noise_std: float,
    hed_config: dict[str, Any],
):
    import torch

    if jitter_type == "hed":
        from controlnet_train.data.hed_stain_augment import HEDStainAugment

        previous_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        try:
            augment = HEDStainAugment(**hed_config)
            return augment(image, augment.sample(device=image.device)).clamp(0.0, 1.0).contiguous()
        finally:
            torch.random.set_rng_state(previous_state)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    image_f = image.detach().float().cpu()
    if jitter_type == "rgb":
        strength = max(0.0, float(rgb_jitter_strength))
        brightness = 1.0 + _uniform(generator, -strength, strength)
        contrast = 1.0 + _uniform(generator, -strength, strength)
        saturation = 1.0 + _uniform(generator, -strength, strength)
        channel_scale = torch.empty(3, 1, 1).uniform_(1.0 - strength, 1.0 + strength, generator=generator)
        mean = image_f.mean(dim=(1, 2), keepdim=True)
        gray = image_f.mean(dim=0, keepdim=True)
        out = (image_f - mean) * contrast + mean
        out = gray + (out - gray) * saturation
        out = out * brightness * channel_scale
        return out.clamp(0.0, 1.0).to(dtype=image.dtype, device=image.device).contiguous()
    if jitter_type == "noise":
        noise = torch.randn(image_f.shape, generator=generator, dtype=image_f.dtype) * max(0.0, float(noise_std))
        return (image_f + noise).clamp(0.0, 1.0).to(dtype=image.dtype, device=image.device).contiguous()
    raise ValueError(f"Unsupported jitter type: {jitter_type}")


def _uniform(generator, low: float, high: float) -> float:
    import torch

    value = torch.rand((), generator=generator)
    return float((value * (float(high) - float(low)) + float(low)).item())


def tensor_to_pil(image) -> Image.Image:
    array = image.detach().float().cpu().clamp(0.0, 1.0).numpy()
    array = np.transpose(array, (1, 2, 0))
    return Image.fromarray((array * 255.0 + 0.5).astype(np.uint8), mode="RGB")


def aggregate_jitter_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, float | None, float | None], list[dict[str, Any]]] = defaultdict(list)
    has_scale_grid = any("ip_scale" in row or "controlnet_scale" in row for row in rows)
    for row in rows:
        ip_scale = float(row["ip_scale"]) if "ip_scale" in row else None
        controlnet_scale = float(row["controlnet_scale"]) if "controlnet_scale" in row else None
        grouped[(str(row["jitter_type"]), ip_scale, controlnet_scale)].append(row)

    metric_keys = [
        "reference_l1_vs_normal",
        "l1_vs_normal_output",
        "mse_vs_normal_output",
        "l1_to_target",
        "mse_to_target",
        "ref_features_vs_normal_cosine",
        "ref_features_vs_normal_l1",
        "ref_features_vs_normal_rmse",
        "ip_hidden_states_vs_normal_cosine",
        "ip_hidden_states_vs_normal_l1",
        "ip_hidden_states_vs_normal_rmse",
    ]
    by_jitter_type: dict[str, dict[str, float]] = {}
    for (jitter_type, ip_scale, controlnet_scale), group_rows in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            -1.0 if item[0][1] is None else item[0][1],
            -1.0 if item[0][2] is None else item[0][2],
        ),
    ):
        key = jitter_type
        if has_scale_grid and ip_scale is not None and controlnet_scale is not None:
            key = f"{jitter_type}@ip{format_scale(ip_scale)}_cn{format_scale(controlnet_scale)}"
        by_jitter_type[key] = {"num_outputs": float(len(group_rows))}
        for metric_key in metric_keys:
            values = [
                float(row[metric_key])
                for row in group_rows
                if metric_key in row and math.isfinite(float(row[metric_key]))
            ]
            if values:
                by_jitter_type[jitter_type][f"{metric_key}_mean"] = float(np.mean(values))
                by_jitter_type[jitter_type][f"{metric_key}_std"] = float(np.std(values))

    return {
        "num_outputs": len(rows),
        "num_samples": len({row["sample_id"] for row in rows}),
        "by_jitter_type": by_jitter_type,
    }


def write_rows(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    jsonl_path = output_dir / "diagnostics.jsonl"
    with jsonl_path.open("w", encoding="utf8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")

    if not rows:
        return
    csv_path = output_dir / "diagnostics.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())

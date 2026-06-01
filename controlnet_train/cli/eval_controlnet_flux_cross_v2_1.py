"""Evaluate a Phase 5.3 Cross V2.1 FLUX ControlNet."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Phase 5.3 Cross V2.1 FLUX ControlNet.")
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--checkpoint", required=True, help="Cross V2.1 checkpoint dir.")
    parser.add_argument("--metadata", required=True, help="metadata_cross_{train,val}.json path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--prompt-source", choices=["metadata", "dataset"], default="dataset")
    parser.add_argument("--prompt", default=None, help="Override every sample with one prompt.")
    parser.add_argument(
        "--color-match",
        choices=("none", "lab"),
        default="lab",
        help="Postprocess predictions to match reference stain/color statistics.",
    )
    parser.add_argument("--overview-max-samples", type=int, default=32)
    parser.add_argument("--thumbnail-size", type=int, default=192)
    parser.add_argument(
        "--run-zero-ref-ablation",
        action="store_true",
        help=(
            "Also evaluate each sample with the whole reference-side condition "
            "(reference latent, tissue features, nuclei features) zeroed."
        ),
    )
    return parser


def parse_args(args=None) -> argparse.Namespace:
    return build_parser().parse_args(args)


def main(argv=None) -> int:
    args = parse_args(argv)

    from controlnet_train.cli.eval_controlnet_flux_cross import (
        _make_overview,
        _make_panel,
        _match_image_color_to_reference,
        _mask_to_rgb,
        _pil_to_chw_float,
        _resolve_eval_prompt,
        _safe_name,
        _save_error_image,
        _save_mask_image,
        _write_metrics,
        aggregate_metrics,
        compute_cross_metrics,
        read_cross_metadata,
        select_eval_records,
    )

    records = select_eval_records(
        read_cross_metadata(args.metadata),
        num_samples=args.num_samples,
        seed=args.seed,
    )
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    import torch

    from controlnet_train.data.common import (
        default_prompt_for_dataset,
        load_image_tensor,
        load_nuclei_mask,
        load_tissue_mask,
    )
    from controlnet_train.inference.pipeline_cross_v2_1 import (
        CROSS_V2_1_REFERENCE_WITH_REF,
        CROSS_V2_1_REFERENCE_ZERO_REF,
        load_cross_v2_1_bundle,
        run_cross_v2_1_bundle,
    )

    dtype_by_name = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    bundle = load_cross_v2_1_bundle(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        checkpoint_path=args.checkpoint,
        device=args.device,
        torch_dtype=dtype_by_name[args.torch_dtype],
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
    )

    metric_rows: list[dict[str, Any]] = []
    panel_paths: list[Path] = []
    for index, record in enumerate(records):
        sample_id = str(record.get("sample_id") or Path(record["target_image"]).stem)
        ref_id = str(record.get("reference_sample_id") or Path(record["reference_image"]).stem)
        sample_dir = samples_dir / f"{index:04d}_{_safe_name(sample_id)}__ref_{_safe_name(ref_id)}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        reference_image_path = Path(record["reference_image"])
        reference_tissue_mask_path = Path(record["reference_tissue_mask"])
        reference_nuclei_mask_path = Path(record["reference_nuclei_mask"])
        target_image_path = Path(record["target_image"])
        target_tissue_mask_path = Path(record["target_tissue_mask"])
        target_nuclei_mask_path = Path(record["target_nuclei_mask"])

        reference_image_tensor = load_image_tensor(reference_image_path)
        reference_tissue_mask = load_tissue_mask(reference_tissue_mask_path)
        reference_nuclei_mask = load_nuclei_mask(reference_nuclei_mask_path)
        target_tissue_mask = load_tissue_mask(target_tissue_mask_path)
        target_nuclei_mask = load_nuclei_mask(target_nuclei_mask_path)
        prompt = _resolve_eval_prompt(
            record=record,
            prompt_override=args.prompt,
            prompt_source=args.prompt_source,
            default_prompt_for_dataset=default_prompt_for_dataset,
        )

        reference_pil = Image.open(reference_image_path).convert("RGB")
        reference_pil.save(sample_dir / "reference.png")
        target_pil = Image.open(target_image_path).convert("RGB")
        target_pil.save(sample_dir / "target.png")
        target_array = _pil_to_chw_float(target_pil)
        reference_tissue_array = np.asarray(Image.open(reference_tissue_mask_path))
        target_tissue_array = np.asarray(Image.open(target_tissue_mask_path))
        _save_mask_image(np.asarray(Image.open(reference_tissue_mask_path)), sample_dir / "reference_tissue_mask.png")
        _save_mask_image(np.asarray(Image.open(target_tissue_mask_path)), sample_dir / "target_tissue_mask.png")
        _save_mask_image(np.asarray(Image.open(reference_nuclei_mask_path)), sample_dir / "reference_nuclei_mask.png")
        _save_mask_image(np.asarray(Image.open(target_nuclei_mask_path)), sample_dir / "target_nuclei_mask.png")

        variant_results = []
        for variant in _reference_variants(args.run_zero_ref_ablation):
            with torch.no_grad():
                prediction = run_cross_v2_1_bundle(
                    bundle,
                    reference_image=reference_image_tensor,
                    reference_tissue_mask=reference_tissue_mask,
                    reference_nuclei_mask=reference_nuclei_mask,
                    target_tissue_mask=target_tissue_mask,
                    target_nuclei_mask=target_nuclei_mask,
                    prompt=prompt,
                    reference_condition_mode=variant,
                )

            result = _save_variant_outputs(
                prediction=prediction,
                variant=variant,
                sample_dir=sample_dir,
                reference_pil=reference_pil,
                target_pil=target_pil,
                target_array=target_array,
                color_match=args.color_match,
                thumbnail_size=args.thumbnail_size,
                reference_tissue=reference_tissue_array,
                target_tissue=target_tissue_array,
                title=f"{sample_id} | ref={ref_id} | {variant}",
                match_image_color_to_reference=_match_image_color_to_reference,
                compute_cross_metrics=compute_cross_metrics,
                pil_to_chw_float=_pil_to_chw_float,
                save_error_image=_save_error_image,
                make_panel=_make_panel,
            )
            variant_results.append(result)

            metric_row = {
                "index": index,
                "sample_id": sample_id,
                "reference_sample_id": ref_id,
                "dataset": record.get("dataset", ""),
                "pair_difficulty": record.get("pair_difficulty", ""),
                "tissue_coverage_ratio": float(record.get("tissue_coverage_ratio", math.nan)),
                "area_coverage_ratio": float(record.get("area_coverage_ratio", math.nan)),
                "reference_condition_mode": variant,
                "zero_ref_ablation": variant == CROSS_V2_1_REFERENCE_ZERO_REF,
                "color_match_applied": args.color_match != "none",
                "controlnet_conditioning_scale": float(args.controlnet_conditioning_scale),
                **result["metrics"],
            }
            metric_rows.append(metric_row)
            result["metric_row"] = metric_row
            (sample_dir / f"metrics_{variant}.json").write_text(
                json.dumps(metric_row, indent=2, ensure_ascii=False, allow_nan=True),
                encoding="utf8",
            )

        metrics_payload = {result["variant"]: result["metric_row"] for result in variant_results}
        if args.run_zero_ref_ablation:
            comparison = _build_ref_ablation_comparison(variant_results)
            metrics_payload["comparison"] = comparison
            (sample_dir / "ref_ablation_comparison.json").write_text(
                json.dumps(comparison, indent=2, ensure_ascii=False, allow_nan=True),
                encoding="utf8",
            )

        primary = variant_results[0]
        prediction = primary["prediction"]
        metrics = primary["metrics"]
        if not args.run_zero_ref_ablation:
            prediction.save(sample_dir / "prediction.png")
            primary["raw_prediction"].save(sample_dir / "prediction_raw.png")
            _save_error_image(primary["abs_error"], sample_dir / "abs_error.png")
            primary["panel"].save(sample_dir / "panel.png")
            metrics_payload = primary["metric_row"]
        (sample_dir / "metrics.json").write_text(
            json.dumps(metrics_payload, indent=2, ensure_ascii=False, allow_nan=True),
            encoding="utf8",
        )

        if args.run_zero_ref_ablation:
            panel = _make_ref_ablation_panel(
                reference=reference_pil,
                target=target_pil,
                reference_tissue=reference_tissue_array,
                target_tissue=target_tissue_array,
                variant_results=variant_results,
                thumbnail_size=args.thumbnail_size,
                title=f"{sample_id} | ref={ref_id}",
                mask_to_rgb=_mask_to_rgb,
            )
        else:
            panel = primary["panel"]
        panel_path = sample_dir / "panel.png"
        panel.save(panel_path)
        if len(panel_paths) < args.overview_max_samples:
            panel_paths.append(panel_path)

        message = (
            f"[{index + 1}/{len(records)}] {sample_id} ref={ref_id} "
            f"with_ref_l1={metrics['full_l1']:.4f} with_ref_psnr={metrics['full_psnr']:.2f}"
        )
        if args.run_zero_ref_ablation:
            zero_metrics = next(
                result["metrics"] for result in variant_results if result["variant"] == CROSS_V2_1_REFERENCE_ZERO_REF
            )
            message += f" zero_ref_l1={zero_metrics['full_l1']:.4f} zero_ref_psnr={zero_metrics['full_psnr']:.2f}"
        print(message)

    _write_metrics(output_dir, metric_rows)
    summary = aggregate_metrics(metric_rows)
    summary["controlnet_conditioning_scale"] = float(args.controlnet_conditioning_scale)
    summary["reference_condition_modes"] = _reference_variants(args.run_zero_ref_ablation)
    summary["run_zero_ref_ablation"] = bool(args.run_zero_ref_ablation)
    if args.run_zero_ref_ablation:
        summary["by_reference_condition_mode"] = _aggregate_by_reference_mode(metric_rows)
        summary["ref_ablation_delta"] = _aggregate_ref_ablation_delta(metric_rows)
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf8",
    )
    if panel_paths:
        _make_overview(panel_paths).save(output_dir / "overview_grid.png")
    print(f"wrote eval outputs to {output_dir}")
    return 0


def _reference_variants(run_zero_ref_ablation: bool) -> list[str]:
    from controlnet_train.inference.pipeline_cross_v2_1 import (
        CROSS_V2_1_REFERENCE_WITH_REF,
        CROSS_V2_1_REFERENCE_ZERO_REF,
    )

    variants = [CROSS_V2_1_REFERENCE_WITH_REF]
    if run_zero_ref_ablation:
        variants.append(CROSS_V2_1_REFERENCE_ZERO_REF)
    return variants


def _save_variant_outputs(
    *,
    prediction: Image.Image,
    variant: str,
    sample_dir: Path,
    reference_pil: Image.Image,
    target_pil: Image.Image,
    target_array: np.ndarray,
    color_match: str,
    thumbnail_size: int,
    reference_tissue: np.ndarray,
    target_tissue: np.ndarray,
    title: str,
    match_image_color_to_reference,
    compute_cross_metrics,
    pil_to_chw_float,
    save_error_image,
    make_panel,
) -> dict[str, Any]:
    raw_prediction = prediction.convert("RGB")
    raw_prediction.save(sample_dir / f"prediction_{variant}_raw.png")
    prediction = raw_prediction
    if color_match == "lab":
        prediction = match_image_color_to_reference(
            source=raw_prediction,
            reference=reference_pil,
            method=color_match,
        )
    prediction.save(sample_dir / f"prediction_{variant}.png")

    pred_array = pil_to_chw_float(prediction)
    metrics = compute_cross_metrics(pred_array, target_array)
    abs_error = np.abs(pred_array - target_array).mean(axis=0)
    save_error_image(abs_error, sample_dir / f"abs_error_{variant}.png")
    panel = make_panel(
        reference=reference_pil,
        prediction=prediction,
        target=target_pil,
        reference_tissue=reference_tissue,
        target_tissue=target_tissue,
        abs_error=abs_error,
        thumbnail_size=thumbnail_size,
        title=title,
    )
    panel.save(sample_dir / f"panel_{variant}.png")
    return {
        "variant": variant,
        "raw_prediction": raw_prediction,
        "prediction": prediction,
        "pred_array": pred_array,
        "metrics": metrics,
        "metric_row": {
            "reference_condition_mode": variant,
            "color_match_applied": color_match != "none",
            **metrics,
        },
        "abs_error": abs_error,
        "panel": panel,
    }


def _build_ref_ablation_comparison(variant_results: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant = {result["variant"]: result for result in variant_results}
    with_ref = by_variant.get("with_ref")
    zero_ref = by_variant.get("zero_ref")
    if with_ref is None or zero_ref is None:
        return {}

    output_abs_diff = np.abs(with_ref["pred_array"] - zero_ref["pred_array"])
    comparison = {
        "with_ref": with_ref["metrics"],
        "zero_ref": zero_ref["metrics"],
        "delta_zero_ref_minus_with_ref": {
            key: float(zero_ref["metrics"][key] - with_ref["metrics"][key])
            for key in ("full_l1", "full_mse", "full_psnr")
            if key in with_ref["metrics"] and key in zero_ref["metrics"]
        },
        "prediction_l1_between_modes": float(output_abs_diff.mean()),
        "prediction_mse_between_modes": float(np.square(with_ref["pred_array"] - zero_ref["pred_array"]).mean()),
    }
    return comparison


def _make_ref_ablation_panel(
    *,
    reference: Image.Image,
    target: Image.Image,
    reference_tissue: np.ndarray,
    target_tissue: np.ndarray,
    variant_results: list[dict[str, Any]],
    thumbnail_size: int,
    title: str,
    mask_to_rgb,
) -> Image.Image:
    images: list[tuple[str, Image.Image]] = [
        ("reference", reference.convert("RGB")),
        ("with_ref", variant_results[0]["prediction"].convert("RGB")),
    ]
    zero_ref = next((result for result in variant_results if result["variant"] == "zero_ref"), None)
    if zero_ref is not None:
        images.append(("zero_ref", zero_ref["prediction"].convert("RGB")))
        diff = np.abs(variant_results[0]["pred_array"] - zero_ref["pred_array"]).mean(axis=0)
        images.append(("ref_delta", Image.fromarray((np.clip(diff, 0.0, 1.0) * 255).astype(np.uint8), mode="L").convert("RGB")))
    images.extend(
        [
            ("target", target.convert("RGB")),
            ("ref_tissue", mask_to_rgb(reference_tissue)),
            ("target_tissue", mask_to_rgb(target_tissue)),
        ]
    )

    thumbs = [(label, _thumbnail(image, thumbnail_size)) for label, image in images]
    label_h = 34
    title_h = 28
    width = thumbnail_size * len(thumbs)
    height = thumbnail_size + label_h + title_h
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((6, 6), title[:160], fill=(0, 0, 0))
    for idx, (label, image) in enumerate(thumbs):
        x = idx * thumbnail_size
        panel.paste(image, (x, title_h))
        draw.text((x + 6, title_h + thumbnail_size + 8), label, fill=(0, 0, 0))
    return panel


def _thumbnail(image: Image.Image, size: int) -> Image.Image:
    thumb = image.copy()
    thumb.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - thumb.width) // 2
    y = (size - thumb.height) // 2
    canvas.paste(thumb, (x, y))
    return canvas


def _aggregate_by_reference_mode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("reference_condition_mode", "")), []).append(row)
    return {
        mode: aggregate
        for mode, aggregate in (
            (mode, _aggregate_metric_subset(mode_rows))
            for mode, mode_rows in sorted(grouped.items())
        )
        if mode
    }


def _aggregate_metric_subset(rows: list[dict[str, Any]]) -> dict[str, float]:
    from controlnet_train.cli.eval_controlnet_flux_cross import aggregate_metrics

    return aggregate_metrics(rows)


def _aggregate_ref_ablation_delta(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_index: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_index.setdefault(int(row["index"]), {})[str(row.get("reference_condition_mode", ""))] = row

    deltas: list[dict[str, float]] = []
    for variants in by_index.values():
        with_ref = variants.get("with_ref")
        zero_ref = variants.get("zero_ref")
        if not with_ref or not zero_ref:
            continue
        deltas.append(
            {
                "full_l1_delta": float(zero_ref["full_l1"] - with_ref["full_l1"]),
                "full_mse_delta": float(zero_ref["full_mse"] - with_ref["full_mse"]),
                "full_psnr_delta": float(zero_ref["full_psnr"] - with_ref["full_psnr"]),
            }
        )

    if not deltas:
        return {}
    keys = list(deltas[0].keys())
    summary = {"num_pairs": float(len(deltas))}
    for key in keys:
        values = [row[key] for row in deltas if math.isfinite(row[key])]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
    return summary


if __name__ == "__main__":
    raise SystemExit(main())

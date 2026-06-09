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
from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


CROSS_V1_REFERENCE_WITH_REF = "with_ref"
CROSS_V1_REFERENCE_ZERO_REF = "zero_ref"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Phase 5.3 Cross V1 FLUX ControlNet.")
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--checkpoint", required=True, help="Cross V1 checkpoint dir.")
    parser.add_argument("--uni-checkpoint-path", required=True, help="UNI2-h pytorch_model.bin path.")
    parser.add_argument("--metadata", required=True, help="metadata_cross_{train,val}.json path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--ip-scale", type=float, default=1.0)
    parser.add_argument(
        "--run-zero-ref-ablation",
        action="store_true",
        help=(
            "Also evaluate each sample with the IP-Adapter reference image replaced "
            "by an all-zero image. Reference tissue/nuclei masks are unchanged."
        ),
    )
    parser.add_argument(
        "--source-latent-init-strength",
        type=float,
        default=0.0,
        help=(
            "Img2img-style source/ref latent start strength in [0,1]. "
            "0 keeps random-noise sampling; try 0.25-0.45 for ref-preserving edits."
        ),
    )
    parser.add_argument(
        "--mask-chord-scale",
        type=float,
        default=0.0,
        help=(
            "Enable source-vs-target mask condition guidance. "
            "0 keeps the baseline; try 0.5, 1.0, 1.5."
        ),
    )
    parser.add_argument(
        "--mask-chord-use-gate",
        action="store_true",
        help=(
            "Gate mask-chord guidance to tissue/nuclei label changes, preserving "
            "unchanged regions more strongly."
        ),
    )
    parser.add_argument(
        "--mask-chord-gate-dilate-radius",
        type=int,
        default=0,
        help="Optional dilation radius, in VAE-latent pixels, for the mask-chord change gate.",
    )
    parser.add_argument(
        "--mask-chord-gate-feather-radius",
        type=int,
        default=0,
        help="Optional average-pool feather radius, in VAE-latent pixels, for the mask-chord change gate.",
    )
    parser.add_argument(
        "--mask-chord-gate-outside-scale",
        type=float,
        default=0.0,
        help=(
            "Residual mask-chord scale outside the changed gate in [0,1]. "
            "0 fully suppresses outside changes; 0.1 allows weak global harmonization."
        ),
    )
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
    return parser


def parse_args(args=None) -> argparse.Namespace:
    return build_parser().parse_args(args)


def read_cross_metadata(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf8"))
    return normalize_cross_records(payload)


def normalize_cross_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        records = payload.get("pairs")
        if not isinstance(records, list):
            raise ValueError("cross metadata dict must contain a 'pairs' list")
        return records
    if isinstance(payload, list):
        return payload
    raise TypeError(f"unsupported cross metadata payload type: {type(payload)!r}")


def select_eval_records(
    records: list[dict[str, Any]],
    *,
    num_samples: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if num_samples is None or num_samples <= 0 or num_samples >= len(records):
        return list(records)
    selected = list(records)
    random.Random(seed).shuffle(selected)
    return selected[:num_samples]


def compute_cross_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = _as_chw_float(prediction)
    tgt = _as_chw_float(target)
    if pred.shape != tgt.shape:
        raise ValueError(f"prediction and target shapes differ: {pred.shape} vs {tgt.shape}")

    abs_err = np.abs(pred - tgt)
    sq_err = np.square(pred - tgt)
    mse = float(sq_err.mean())
    return {
        "full_l1": float(abs_err.mean()),
        "full_mse": mse,
        "full_psnr": _psnr(mse),
    }


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    metric_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (float, int)) and key not in {"index"}
    ]
    summary = {"num_samples": float(len(rows))}
    for key in metric_keys:
        values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
    return summary


def main(argv=None) -> int:
    args = parse_args(argv)
    records = select_eval_records(
        read_cross_metadata(args.metadata), num_samples=args.num_samples, seed=args.seed,
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
    from controlnet_train.inference.pipeline_cross_v1 import (
        load_cross_v1_bundle,
        run_cross_v1_bundle,
    )

    dtype_by_name = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_by_name[args.torch_dtype]
    bundle = load_cross_v1_bundle(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        checkpoint_path=args.checkpoint,
        uni_checkpoint_path=args.uni_checkpoint_path,
        device=args.device,
        torch_dtype=torch_dtype,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        ip_adapter_scale=args.ip_scale,
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

        reference_image = load_image_tensor(reference_image_path)
        reference_tissue_mask = load_tissue_mask(reference_tissue_mask_path)
        reference_nuclei_mask = load_nuclei_mask(reference_nuclei_mask_path)
        target_tissue_mask = load_tissue_mask(target_tissue_mask_path)
        target_nuclei_mask = load_nuclei_mask(target_nuclei_mask_path)

        reference_pil = Image.open(reference_image_path).convert("RGB")
        reference_pil.save(sample_dir / "reference.png")
        target_pil = Image.open(target_image_path).convert("RGB")
        target_pil.save(sample_dir / "target.png")
        target_array = _pil_to_chw_float(target_pil)
        reference_tissue_array = np.asarray(Image.open(reference_tissue_mask_path))
        target_tissue_array = np.asarray(Image.open(target_tissue_mask_path))
        _save_mask_image(reference_tissue_array, sample_dir / "reference_tissue_mask.png")
        _save_mask_image(target_tissue_array, sample_dir / "target_tissue_mask.png")
        _save_mask_image(np.asarray(Image.open(reference_nuclei_mask_path)), sample_dir / "reference_nuclei_mask.png")
        _save_mask_image(np.asarray(Image.open(target_nuclei_mask_path)), sample_dir / "target_nuclei_mask.png")

        prompt = _resolve_eval_prompt(
            record=record,
            prompt_override=args.prompt,
            prompt_source=args.prompt_source,
            default_prompt_for_dataset=default_prompt_for_dataset,
        )

        variant_results: list[dict[str, Any]] = []
        for variant in _reference_variants(args.run_zero_ref_ablation):
            reference_image_for_model = _reference_image_for_mode(reference_image, variant)
            if args.run_zero_ref_ablation:
                _reference_pil_for_mode(reference_pil, variant).save(sample_dir / f"reference_used_{variant}.png")

            with torch.no_grad():
                raw_prediction = run_cross_v1_bundle(
                    bundle,
                    reference_image=reference_image_for_model,
                    reference_tissue_mask=reference_tissue_mask,
                    reference_nuclei_mask=reference_nuclei_mask,
                    target_tissue_mask=target_tissue_mask,
                    target_nuclei_mask=target_nuclei_mask,
                    prompt=prompt,
                    source_latent_init_strength=args.source_latent_init_strength,
                    mask_chord_scale=args.mask_chord_scale,
                    mask_chord_use_gate=args.mask_chord_use_gate,
                    mask_chord_gate_dilate_radius=args.mask_chord_gate_dilate_radius,
                    mask_chord_gate_feather_radius=args.mask_chord_gate_feather_radius,
                    mask_chord_gate_outside_scale=args.mask_chord_gate_outside_scale,
                )

            if args.run_zero_ref_ablation:
                raw_prediction.save(sample_dir / f"prediction_{variant}_raw.png")
            else:
                raw_prediction.save(sample_dir / "prediction_raw.png")

            prediction = raw_prediction
            if args.color_match == "lab":
                prediction = _match_image_color_to_reference(
                    source=raw_prediction,
                    reference=reference_pil,
                    method=args.color_match,
                )
            if args.run_zero_ref_ablation:
                prediction.save(sample_dir / f"prediction_{variant}.png")
            else:
                prediction.save(sample_dir / "prediction.png")

            pred_array = _pil_to_chw_float(prediction)
            metrics = compute_cross_metrics(pred_array, target_array)
            abs_error = np.abs(pred_array - target_array).mean(axis=0)
            if args.run_zero_ref_ablation:
                _save_error_image(abs_error, sample_dir / f"abs_error_{variant}.png")
            else:
                _save_error_image(abs_error, sample_dir / "abs_error.png")

            metric_row = {
                "index": index,
                "sample_id": sample_id,
                "reference_sample_id": ref_id,
                "dataset": record.get("dataset", ""),
                "pair_difficulty": record.get("pair_difficulty", ""),
                "tissue_coverage_ratio": float(record.get("tissue_coverage_ratio", math.nan)),
                "area_coverage_ratio": float(record.get("area_coverage_ratio", math.nan)),
                "color_match_applied": args.color_match != "none",
                "ip_scale": float(args.ip_scale),
                "controlnet_conditioning_scale": float(args.controlnet_conditioning_scale),
                "source_latent_init_strength": float(args.source_latent_init_strength),
                "mask_chord_scale": float(args.mask_chord_scale),
                "mask_chord_use_gate": bool(args.mask_chord_use_gate),
                "mask_chord_gate_dilate_radius": int(args.mask_chord_gate_dilate_radius),
                "mask_chord_gate_feather_radius": int(args.mask_chord_gate_feather_radius),
                "mask_chord_gate_outside_scale": float(args.mask_chord_gate_outside_scale),
                **metrics,
            }
            if args.run_zero_ref_ablation:
                metric_row["reference_condition_mode"] = variant
                metric_row["zero_ref_ablation"] = variant == CROSS_V1_REFERENCE_ZERO_REF
            metric_rows.append(metric_row)
            if args.run_zero_ref_ablation:
                (sample_dir / f"metrics_{variant}.json").write_text(
                    json.dumps(metric_row, indent=2, ensure_ascii=False, allow_nan=True),
                    encoding="utf8",
                )

            panel = _make_panel(
                reference=reference_pil,
                prediction=prediction,
                target=target_pil,
                reference_tissue=reference_tissue_array,
                target_tissue=target_tissue_array,
                abs_error=abs_error,
                thumbnail_size=args.thumbnail_size,
                title=f"{sample_id} | ref={ref_id} | {variant}",
            )
            if args.run_zero_ref_ablation:
                panel.save(sample_dir / f"panel_{variant}.png")
            variant_results.append(
                {
                    "variant": variant,
                    "prediction": prediction,
                    "pred_array": pred_array,
                    "metrics": metrics,
                    "metric_row": metric_row,
                    "abs_error": abs_error,
                    "panel": panel,
                }
            )

        primary = variant_results[0]
        if args.run_zero_ref_ablation:
            metrics_payload = {result["variant"]: result["metric_row"] for result in variant_results}
            comparison = _build_ref_ablation_comparison(variant_results)
            metrics_payload["comparison"] = comparison
            (sample_dir / "ref_ablation_comparison.json").write_text(
                json.dumps(comparison, indent=2, ensure_ascii=False, allow_nan=True),
                encoding="utf8",
            )
        else:
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
            )
        else:
            panel = primary["panel"]
        panel_path = sample_dir / "panel.png"
        panel.save(panel_path)
        if len(panel_paths) < args.overview_max_samples:
            panel_paths.append(panel_path)

        if args.run_zero_ref_ablation:
            with_ref_metrics = primary["metrics"]
            zero_ref_metrics = next(
                result["metrics"]
                for result in variant_results
                if result["variant"] == CROSS_V1_REFERENCE_ZERO_REF
            )
            print(
                f"[{index + 1}/{len(records)}] {sample_id} ref={ref_id} "
                f"with_ref_l1={with_ref_metrics['full_l1']:.4f} "
                f"with_ref_psnr={with_ref_metrics['full_psnr']:.2f} "
                f"zero_ref_l1={zero_ref_metrics['full_l1']:.4f} "
                f"zero_ref_psnr={zero_ref_metrics['full_psnr']:.2f}"
            )
        else:
            metrics = primary["metrics"]
            print(
                f"[{index + 1}/{len(records)}] {sample_id} ref={ref_id} "
                f"full_l1={metrics['full_l1']:.4f} full_psnr={metrics['full_psnr']:.2f}"
            )

    _write_metrics(output_dir, metric_rows)
    summary = aggregate_metrics(metric_rows)
    summary["ip_scale"] = float(args.ip_scale)
    summary["controlnet_conditioning_scale"] = float(args.controlnet_conditioning_scale)
    summary["source_latent_init_strength"] = float(args.source_latent_init_strength)
    summary["mask_chord_scale"] = float(args.mask_chord_scale)
    summary["mask_chord_use_gate"] = bool(args.mask_chord_use_gate)
    summary["mask_chord_gate_dilate_radius"] = int(args.mask_chord_gate_dilate_radius)
    summary["mask_chord_gate_feather_radius"] = int(args.mask_chord_gate_feather_radius)
    summary["mask_chord_gate_outside_scale"] = float(args.mask_chord_gate_outside_scale)
    if args.run_zero_ref_ablation:
        summary["reference_condition_modes"] = _reference_variants(args.run_zero_ref_ablation)
        summary["run_zero_ref_ablation"] = True
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


def _resolve_eval_prompt(
    *,
    record: dict[str, Any],
    prompt_override: str | None,
    prompt_source: str,
    default_prompt_for_dataset,
) -> str:
    if prompt_override:
        return prompt_override
    if prompt_source == "metadata":
        prompt = record.get("prompt")
        if prompt:
            return str(prompt)
    if prompt_source == "dataset":
        dataset = record.get("dataset")
        if dataset:
            return default_prompt_for_dataset(str(dataset))
    return str(record.get("prompt") or "H&E stained cancer histopathology at 40x magnification")


def _as_chw_float(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array, dtype=np.float32)
    if result.ndim != 3:
        raise ValueError(f"expected image array with 3 dimensions, got shape {result.shape}")
    if result.shape[-1] in {1, 3}:
        result = np.transpose(result, (2, 0, 1))
    if result.max(initial=0.0) > 1.0:
        result = result / 255.0
    return np.clip(result, 0.0, 1.0)


def _psnr(mse: float) -> float:
    if math.isnan(mse):
        return math.nan
    if mse <= 0.0:
        return math.inf
    return float(-10.0 * math.log10(mse))


def _pil_to_chw_float(image: Image.Image) -> np.ndarray:
    return np.transpose(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0, (2, 0, 1))


def _match_image_color_to_reference(
    *,
    source: Image.Image,
    reference: Image.Image,
    method: str,
) -> Image.Image:
    if method == "lab":
        return _mean_std_transfer_pil_lab(source=source, reference=reference)
    raise ValueError(f"Unsupported color match method: {method}")


def _mean_std_transfer_pil_lab(*, source: Image.Image, reference: Image.Image) -> Image.Image:
    from skimage.color import lab2rgb, rgb2lab

    source_rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    reference_rgb = np.asarray(reference.convert("RGB"), dtype=np.float32) / 255.0
    source_lab = rgb2lab(source_rgb).astype(np.float32)
    reference_lab = rgb2lab(reference_rgb).astype(np.float32)
    source_mask = _tissue_mask_from_rgb(source_rgb)
    reference_mask = _tissue_mask_from_rgb(reference_rgb)

    if not np.any(source_mask) or not np.any(reference_mask):
        return source.convert("RGB")

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

    matched_rgb = np.clip(lab2rgb(matched_lab), 0.0, 1.0)
    output = source_rgb.copy()
    output[source_mask] = matched_rgb[source_mask]
    return Image.fromarray((output * 255.0).round().astype(np.uint8), mode="RGB")


def _tissue_mask_from_rgb(rgb_float: np.ndarray, threshold: float = 0.85) -> np.ndarray:
    return rgb_float.mean(axis=-1) < threshold


def _reference_variants(run_zero_ref_ablation: bool) -> list[str]:
    variants = [CROSS_V1_REFERENCE_WITH_REF]
    if run_zero_ref_ablation:
        variants.append(CROSS_V1_REFERENCE_ZERO_REF)
    return variants


def _reference_image_for_mode(reference_image, mode: str):
    if mode == CROSS_V1_REFERENCE_WITH_REF:
        return reference_image
    if mode == CROSS_V1_REFERENCE_ZERO_REF:
        return reference_image.new_zeros(reference_image.shape)
    raise ValueError(f"Unsupported Cross V1 reference condition mode: {mode!r}")


def _reference_pil_for_mode(reference: Image.Image, mode: str) -> Image.Image:
    if mode == CROSS_V1_REFERENCE_WITH_REF:
        return reference.convert("RGB")
    if mode == CROSS_V1_REFERENCE_ZERO_REF:
        return Image.new("RGB", reference.size, "black")
    raise ValueError(f"Unsupported Cross V1 reference condition mode: {mode!r}")


def _build_ref_ablation_comparison(variant_results: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant = {result["variant"]: result for result in variant_results}
    with_ref = by_variant.get(CROSS_V1_REFERENCE_WITH_REF)
    zero_ref = by_variant.get(CROSS_V1_REFERENCE_ZERO_REF)
    if with_ref is None or zero_ref is None:
        return {}

    output_abs_diff = np.abs(with_ref["pred_array"] - zero_ref["pred_array"])
    return {
        "with_ref": with_ref["metrics"],
        "zero_ref": zero_ref["metrics"],
        "delta_zero_ref_minus_with_ref": {
            key: float(zero_ref["metrics"][key] - with_ref["metrics"][key])
            for key in ("full_l1", "full_mse", "full_psnr")
            if key in with_ref["metrics"] and key in zero_ref["metrics"]
        },
        "prediction_l1_between_modes": float(output_abs_diff.mean()),
        "prediction_mse_between_modes": float(
            np.square(with_ref["pred_array"] - zero_ref["pred_array"]).mean()
        ),
    }


def _make_ref_ablation_panel(
    *,
    reference: Image.Image,
    target: Image.Image,
    reference_tissue: np.ndarray,
    target_tissue: np.ndarray,
    variant_results: list[dict[str, Any]],
    thumbnail_size: int,
    title: str,
) -> Image.Image:
    by_variant = {result["variant"]: result for result in variant_results}
    with_ref = by_variant[CROSS_V1_REFERENCE_WITH_REF]
    zero_ref = by_variant.get(CROSS_V1_REFERENCE_ZERO_REF)

    images: list[tuple[str, Image.Image]] = [
        ("reference", reference.convert("RGB")),
        ("with_ref", with_ref["prediction"].convert("RGB")),
    ]
    if zero_ref is not None:
        images.append(("zero_ref", zero_ref["prediction"].convert("RGB")))
        diff = np.abs(with_ref["pred_array"] - zero_ref["pred_array"]).mean(axis=0)
        images.append(
            (
                "ref_delta",
                Image.fromarray(
                    (np.clip(diff, 0.0, 1.0) * 255).astype(np.uint8),
                    mode="L",
                ).convert("RGB"),
            )
        )
    images.extend(
        [
            ("target", target.convert("RGB")),
            ("ref_tissue", _mask_to_rgb(reference_tissue)),
            ("target_tissue", _mask_to_rgb(target_tissue)),
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


def _aggregate_by_reference_mode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        mode = str(row.get("reference_condition_mode", ""))
        if mode:
            grouped.setdefault(mode, []).append(row)
    return {
        mode: aggregate_metrics(mode_rows)
        for mode, mode_rows in sorted(grouped.items())
    }


def _aggregate_ref_ablation_delta(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_index: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_index.setdefault(int(row["index"]), {})[str(row.get("reference_condition_mode", ""))] = row

    deltas: list[dict[str, float]] = []
    for variants in by_index.values():
        with_ref = variants.get(CROSS_V1_REFERENCE_WITH_REF)
        zero_ref = variants.get(CROSS_V1_REFERENCE_ZERO_REF)
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
    summary = {"num_pairs": float(len(deltas))}
    for key in deltas[0]:
        values = [row[key] for row in deltas if math.isfinite(row[key])]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
    return summary


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)[:120]


def _save_mask_image(mask: np.ndarray, path: Path) -> None:
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    if array.dtype.kind in {"f", "b"}:
        array = (array > 0).astype(np.uint8) * 255
    else:
        max_value = int(array.max(initial=0))
        array = array.astype(np.uint8)
        if max_value > 0 and max_value < 32:
            array = (array.astype(np.float32) * (255.0 / max_value)).astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


def _save_error_image(error: np.ndarray, path: Path) -> None:
    normalized = np.clip(error, 0.0, 1.0)
    Image.fromarray((normalized * 255).astype(np.uint8), mode="L").save(path)


def _make_panel(
    *,
    reference: Image.Image,
    prediction: Image.Image,
    target: Image.Image,
    reference_tissue: np.ndarray,
    target_tissue: np.ndarray,
    abs_error: np.ndarray,
    thumbnail_size: int,
    title: str,
) -> Image.Image:
    images = [
        ("reference", reference.convert("RGB")),
        ("prediction", prediction.convert("RGB")),
        ("target", target.convert("RGB")),
        ("ref_tissue", _mask_to_rgb(reference_tissue)),
        ("target_tissue", _mask_to_rgb(target_tissue)),
        ("abs_error", Image.fromarray((np.clip(abs_error, 0.0, 1.0) * 255).astype(np.uint8), mode="L").convert("RGB")),
    ]
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


def _mask_to_rgb(mask: np.ndarray) -> Image.Image:
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    labels = array.astype(np.int64, copy=False)
    palette = np.array(
        [
            [30, 30, 30],
            [210, 65, 65],
            [74, 145, 84],
            [180, 155, 75],
            [80, 135, 190],
            [190, 180, 70],
            [170, 95, 175],
            [90, 170, 170],
            [215, 125, 60],
            [130, 130, 190],
            [120, 170, 95],
            [210, 170, 100],
            [200, 140, 60],
            [190, 90, 40],
            [170, 70, 120],
            [120, 80, 160],
        ],
        dtype=np.uint8,
    )
    rgb = palette[np.clip(labels, 0, len(palette) - 1)]
    return Image.fromarray(rgb, mode="RGB")


def _thumbnail(image: Image.Image, size: int) -> Image.Image:
    thumb = image.copy()
    thumb.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - thumb.width) // 2
    y = (size - thumb.height) // 2
    canvas.paste(thumb, (x, y))
    return canvas


def _make_overview(panel_paths: list[Path]) -> Image.Image:
    panels = [Image.open(path).convert("RGB") for path in panel_paths]
    width = max(panel.width for panel in panels)
    height = sum(panel.height for panel in panels)
    overview = Image.new("RGB", (width, height), "white")
    y = 0
    for panel in panels:
        overview.paste(panel, (0, y))
        y += panel.height
    return overview


def _write_metrics(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    jsonl_path = output_dir / "metrics.jsonl"
    with jsonl_path.open("w", encoding="utf8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")

    if not rows:
        return
    csv_path = output_dir / "metrics.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())

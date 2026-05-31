"""Evaluate a Phase 5.3 Cross V2.1 FLUX ControlNet."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

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
    return parser


def parse_args(args=None) -> argparse.Namespace:
    return build_parser().parse_args(args)


def main(argv=None) -> int:
    args = parse_args(argv)

    from controlnet_train.cli.eval_controlnet_flux_cross import (
        _make_overview,
        _make_panel,
        _match_image_color_to_reference,
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

        with torch.no_grad():
            prediction = run_cross_v2_1_bundle(
                bundle,
                reference_image=reference_image_tensor,
                reference_tissue_mask=reference_tissue_mask,
                reference_nuclei_mask=reference_nuclei_mask,
                target_tissue_mask=target_tissue_mask,
                target_nuclei_mask=target_nuclei_mask,
                prompt=prompt,
            )

        reference_pil = Image.open(reference_image_path).convert("RGB")
        reference_pil.save(sample_dir / "reference.png")
        target_pil = Image.open(target_image_path).convert("RGB")
        target_pil.save(sample_dir / "target.png")
        prediction.save(sample_dir / "prediction_raw.png")
        if args.color_match == "lab":
            prediction = _match_image_color_to_reference(
                source=prediction,
                reference=reference_pil,
                method=args.color_match,
            )
        prediction.save(sample_dir / "prediction.png")
        _save_mask_image(np.asarray(Image.open(reference_tissue_mask_path)), sample_dir / "reference_tissue_mask.png")
        _save_mask_image(np.asarray(Image.open(target_tissue_mask_path)), sample_dir / "target_tissue_mask.png")
        _save_mask_image(np.asarray(Image.open(reference_nuclei_mask_path)), sample_dir / "reference_nuclei_mask.png")
        _save_mask_image(np.asarray(Image.open(target_nuclei_mask_path)), sample_dir / "target_nuclei_mask.png")

        pred_array = _pil_to_chw_float(prediction)
        target_array = _pil_to_chw_float(target_pil)
        metrics = compute_cross_metrics(pred_array, target_array)
        metric_row = {
            "index": index,
            "sample_id": sample_id,
            "reference_sample_id": ref_id,
            "dataset": record.get("dataset", ""),
            "pair_difficulty": record.get("pair_difficulty", ""),
            "tissue_coverage_ratio": float(record.get("tissue_coverage_ratio", math.nan)),
            "area_coverage_ratio": float(record.get("area_coverage_ratio", math.nan)),
            "color_match_applied": args.color_match != "none",
            "controlnet_conditioning_scale": float(args.controlnet_conditioning_scale),
            **metrics,
        }
        metric_rows.append(metric_row)
        (sample_dir / "metrics.json").write_text(
            json.dumps(metric_row, indent=2, ensure_ascii=False, allow_nan=True),
            encoding="utf8",
        )

        abs_error = np.abs(pred_array - target_array).mean(axis=0)
        _save_error_image(abs_error, sample_dir / "abs_error.png")
        panel = _make_panel(
            reference=reference_pil,
            prediction=prediction,
            target=target_pil,
            reference_tissue=np.asarray(Image.open(reference_tissue_mask_path)),
            target_tissue=np.asarray(Image.open(target_tissue_mask_path)),
            abs_error=abs_error,
            thumbnail_size=args.thumbnail_size,
            title=f"{sample_id} | ref={ref_id}",
        )
        panel_path = sample_dir / "panel.png"
        panel.save(panel_path)
        if len(panel_paths) < args.overview_max_samples:
            panel_paths.append(panel_path)

        print(
            f"[{index + 1}/{len(records)}] {sample_id} ref={ref_id} "
            f"full_l1={metrics['full_l1']:.4f} full_psnr={metrics['full_psnr']:.2f}"
        )

    _write_metrics(output_dir, metric_rows)
    summary = aggregate_metrics(metric_rows)
    summary["controlnet_conditioning_scale"] = float(args.controlnet_conditioning_scale)
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf8",
    )
    if panel_paths:
        _make_overview(panel_paths).save(output_dir / "overview_grid.png")
    print(f"wrote eval outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

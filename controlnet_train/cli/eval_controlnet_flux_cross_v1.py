"""Evaluate Phase 5.3 Cross V1 (IP-Adapter reference attention) on cross-reconstruction metadata."""

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
    parser.add_argument("--prompt-source", choices=["metadata", "dataset"], default="dataset")
    parser.add_argument("--prompt", default=None, help="Override every sample with one prompt.")
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

        prompt = _resolve_eval_prompt(
            record=record,
            prompt_override=args.prompt,
            prompt_source=args.prompt_source,
            default_prompt_for_dataset=default_prompt_for_dataset,
        )

        with torch.no_grad():
            prediction = run_cross_v1_bundle(
                bundle,
                reference_image=reference_image,
                reference_tissue_mask=reference_tissue_mask,
                reference_nuclei_mask=reference_nuclei_mask,
                target_tissue_mask=target_tissue_mask,
                target_nuclei_mask=target_nuclei_mask,
                prompt=prompt,
            )

        prediction.save(sample_dir / "prediction.png")
        reference_pil = Image.open(reference_image_path).convert("RGB")
        reference_pil.save(sample_dir / "reference.png")
        target_pil = Image.open(target_image_path).convert("RGB")
        target_pil.save(sample_dir / "target.png")
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
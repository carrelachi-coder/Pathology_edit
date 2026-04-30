"""Evaluate a Phase 5 FLUX inpaint ControlNet on synthetic inpaint metadata."""

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
    parser = argparse.ArgumentParser(description="Evaluate Phase 5 FLUX inpaint ControlNet.")
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--checkpoint", required=True, help="Inference-ready final output dir.")
    parser.add_argument("--metadata", required=True, help="metadata_inpaint_val.jsonl path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--overview-max-samples", type=int, default=32)
    parser.add_argument("--thumbnail-size", type=int, default=192)
    return parser


def parse_args(args=None) -> argparse.Namespace:
    return build_parser().parse_args(args)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


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


def compute_inpaint_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    change_mask: np.ndarray,
) -> dict[str, float]:
    pred = _as_chw_float(prediction)
    tgt = _as_chw_float(target)
    if pred.shape != tgt.shape:
        raise ValueError(f"prediction and target shapes differ: {pred.shape} vs {tgt.shape}")

    mask = _as_hw_mask(change_mask)
    if mask.shape != pred.shape[1:]:
        raise ValueError(f"change_mask shape {mask.shape} does not match image shape {pred.shape[1:]}")

    abs_err = np.abs(pred - tgt)
    sq_err = np.square(pred - tgt)
    keep_mask = ~mask

    metrics = {
        "change_ratio": float(mask.mean()),
        "full_l1": float(abs_err.mean()),
        "full_mse": float(sq_err.mean()),
    }
    metrics["full_psnr"] = _psnr(metrics["full_mse"])
    metrics.update(_region_metrics(abs_err, sq_err, mask, "change"))
    metrics.update(_region_metrics(abs_err, sq_err, keep_mask, "keep"))
    return metrics


def _as_chw_float(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array, dtype=np.float32)
    if result.ndim != 3:
        raise ValueError(f"expected image array with 3 dimensions, got shape {result.shape}")
    if result.shape[-1] in {1, 3}:
        result = np.transpose(result, (2, 0, 1))
    if result.max(initial=0.0) > 1.0:
        result = result / 255.0
    return np.clip(result, 0.0, 1.0)


def _as_hw_mask(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array)
    if result.ndim == 3 and result.shape[0] == 1:
        result = result[0]
    if result.ndim == 3 and result.shape[-1] == 1:
        result = result[..., 0]
    if result.ndim != 2:
        raise ValueError(f"expected mask array with 2 dimensions, got shape {result.shape}")
    return result > 0


def _region_metrics(
    abs_err: np.ndarray,
    sq_err: np.ndarray,
    mask: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    if not np.any(mask):
        return {
            f"{prefix}_l1": math.nan,
            f"{prefix}_mse": math.nan,
            f"{prefix}_psnr": math.nan,
        }
    l1 = float(abs_err[:, mask].mean())
    mse = float(sq_err[:, mask].mean())
    return {
        f"{prefix}_l1": l1,
        f"{prefix}_mse": mse,
        f"{prefix}_psnr": _psnr(mse),
    }


def _psnr(mse: float) -> float:
    if math.isnan(mse):
        return math.nan
    if mse <= 0.0:
        return math.inf
    return float(-10.0 * math.log10(mse))


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
    records = select_eval_records(read_jsonl(args.metadata), num_samples=args.num_samples, seed=args.seed)
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    import torch

    from controlnet_train.data.common import (
        default_prompt_for_dataset,
        load_binary_mask,
        load_image_tensor,
        load_nuclei_mask,
        load_tissue_mask,
    )
    from controlnet_train.inference import load_inpaint_bundle, run_inpaint_bundle
    from controlnet_train.inference.pipeline import LoadedEditInputs

    dtype_by_name = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_by_name[args.torch_dtype]
    bundle = load_inpaint_bundle(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        checkpoint_path=args.checkpoint,
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
        sample_dir = samples_dir / f"{index:04d}_{_safe_name(sample_id)}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        source_image_path = Path(record.get("source_image") or record["target_image"])
        target_image_path = Path(record["target_image"])
        tissue_mask_path = Path(record["target_tissue_mask"])
        nuclei_mask_path = Path(record["target_nuclei_mask"])
        change_mask_path = Path(record["change_region_mask"])

        change_mask = load_binary_mask(change_mask_path)[0]
        inputs = LoadedEditInputs(
            reference_image_path=source_image_path,
            reference_tissue_mask_path=tissue_mask_path,
            reference_nuclei_mask_path=nuclei_mask_path,
            target_tissue_mask_path=tissue_mask_path,
            target_nuclei_mask_path=nuclei_mask_path,
            output_dir=sample_dir,
            reference_image=load_image_tensor(source_image_path),
            reference_tissue_mask=load_tissue_mask(tissue_mask_path),
            reference_nuclei_mask=load_nuclei_mask(nuclei_mask_path),
            target_tissue_mask=load_tissue_mask(tissue_mask_path),
            target_nuclei_mask=load_nuclei_mask(nuclei_mask_path),
            prompt=record.get("prompt"),
            dataset=record.get("dataset"),
            force_mode="inpaint",
            save_debug_artifacts=True,
        )
        prompt = record.get("prompt") or default_prompt_for_dataset(record.get("dataset", "BCSS"))

        with torch.no_grad():
            prediction = run_inpaint_bundle(bundle, inputs, prompt, change_mask)

        prediction.save(sample_dir / "prediction.png")
        target_image = Image.open(target_image_path).convert("RGB")
        target_image.save(sample_dir / "target.png")
        erased_image_path = record.get("erased_source_image")
        erased_image = Image.open(erased_image_path).convert("RGB") if erased_image_path else target_image
        erased_image.save(sample_dir / "erased_source.png")
        _save_mask_image(change_mask.detach().cpu().numpy(), sample_dir / "change_region_mask.png")

        pred_array = _pil_to_chw_float(prediction)
        target_array = _pil_to_chw_float(target_image)
        mask_array = change_mask.detach().cpu().numpy()
        metrics = compute_inpaint_metrics(pred_array, target_array, mask_array)
        metric_row = {
            "index": index,
            "sample_id": sample_id,
            "dataset": record.get("dataset", ""),
            "mask_mode": record.get("mask_mode", record.get("edit_type", "")),
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
            erased=erased_image,
            prediction=prediction,
            target=target_image,
            change_mask=mask_array,
            abs_error=abs_error,
            thumbnail_size=args.thumbnail_size,
            title=sample_id,
        )
        panel_path = sample_dir / "panel.png"
        panel.save(panel_path)
        if len(panel_paths) < args.overview_max_samples:
            panel_paths.append(panel_path)

        print(
            f"[{index + 1}/{len(records)}] {sample_id} "
            f"change_l1={metrics['change_l1']:.4f} keep_l1={metrics['keep_l1']:.4f}"
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


def _pil_to_chw_float(image: Image.Image) -> np.ndarray:
    return np.transpose(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0, (2, 0, 1))


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)[:120]


def _save_mask_image(mask: np.ndarray, path: Path) -> None:
    array = (_as_hw_mask(mask).astype(np.uint8) * 255)
    Image.fromarray(array, mode="L").save(path)


def _save_error_image(error: np.ndarray, path: Path) -> None:
    normalized = np.clip(error, 0.0, 1.0)
    Image.fromarray((normalized * 255).astype(np.uint8), mode="L").save(path)


def _make_panel(
    *,
    erased: Image.Image,
    prediction: Image.Image,
    target: Image.Image,
    change_mask: np.ndarray,
    abs_error: np.ndarray,
    thumbnail_size: int,
    title: str,
) -> Image.Image:
    images = [
        ("erased", erased.convert("RGB")),
        ("prediction", prediction.convert("RGB")),
        ("target", target.convert("RGB")),
        ("change", Image.fromarray(_as_hw_mask(change_mask).astype(np.uint8) * 255, mode="L").convert("RGB")),
        ("abs_error", Image.fromarray((np.clip(abs_error, 0.0, 1.0) * 255).astype(np.uint8), mode="L").convert("RGB")),
    ]
    thumbs = [(label, _thumbnail(image, thumbnail_size)) for label, image in images]
    label_h = 34
    title_h = 28
    width = thumbnail_size * len(thumbs)
    height = thumbnail_size + label_h + title_h
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((6, 6), title[:120], fill=(0, 0, 0))
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

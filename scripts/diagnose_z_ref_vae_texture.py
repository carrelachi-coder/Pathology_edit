"""Diagnose whether z_ref itself preserves reference texture.

This only tests the FLUX VAE path:

    reference image -> VAE encode -> z_ref -> VAE decode -> reconstruction

No ControlNet, transformer, or reference-token projection is loaded.
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose VAE z_ref texture preservation.")
    parser.add_argument("--pretrained-model-name-or-path", required=True, help="FLUX model dir/path.")
    parser.add_argument("--metadata", default=None, help="Optional metadata_cross_{train,val}.json path.")
    parser.add_argument("--image", action="append", default=[], help="Reference image path. May be repeated.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-sample-id", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--thumbnail-size", type=int, default=192)
    parser.add_argument("--overview-max-samples", type=int, default=32)
    parser.add_argument("--glcm-levels", type=int, default=32)
    parser.add_argument("--glcm-distances", default="1,2,4")
    parser.add_argument("--glcm-angles", default="0,45,90,135")
    return parser


def parse_args(args=None) -> argparse.Namespace:
    return build_parser().parse_args(args)


def main(argv=None) -> int:
    args = parse_args(argv)

    from controlnet_train.cli.eval_controlnet_flux_cross import _safe_name, read_cross_metadata
    from controlnet_train.data.common import load_image_tensor
    from scripts.diagnose_cross_v3_ref_mismatch import (
        _distance_stats,
        _parse_angles,
        _parse_int_list,
        _prefix_stats,
        image_quant_stats,
    )

    records = build_reference_records(
        image_paths=args.image,
        metadata_records=read_cross_metadata(args.metadata) if args.metadata else None,
        reference_sample_ids=args.reference_sample_id,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.torch_dtype, device)

    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=dtype,
    ).to(device)
    vae.eval()

    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    glcm_distances = _parse_int_list(args.glcm_distances)
    glcm_angles = _parse_angles(args.glcm_angles)

    rows: list[dict[str, Any]] = []
    panel_paths: list[Path] = []
    with torch.no_grad():
        for index, record in enumerate(records):
            ref_id = reference_record_id(record)
            sample_dir = samples_dir / f"{index:04d}_ref_{_safe_name(ref_id)}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            reference_path = Path(record["reference_image"])
            reference_tensor = load_image_tensor(reference_path)
            reference_pil = Image.open(reference_path).convert("RGB")
            reference_array = pil_to_chw_float(reference_pil)
            z_ref, reconstruction_pil = encode_decode_z_ref(
                vae=vae,
                image=reference_tensor,
                device=device,
                dtype=dtype,
            )
            reconstruction_array = pil_to_chw_float(reconstruction_pil)
            abs_error = np.abs(reconstruction_array - reference_array).mean(axis=0)

            reference_pil.save(sample_dir / "reference.png")
            reconstruction_pil.save(sample_dir / "z_ref_vae_reconstruction.png")
            save_error_image(abs_error, sample_dir / "abs_error.png")
            panel = make_panel(
                reference=reference_pil,
                reconstruction=reconstruction_pil,
                abs_error=abs_error,
                thumbnail_size=args.thumbnail_size,
                title=f"z_ref VAE texture diagnostic | ref={ref_id}",
            )
            panel_path = sample_dir / "panel.png"
            panel.save(panel_path)
            if len(panel_paths) < args.overview_max_samples:
                panel_paths.append(panel_path)

            reference_stats = image_quant_stats(
                reference_pil,
                levels=args.glcm_levels,
                distances=glcm_distances,
                angles=glcm_angles,
            )
            reconstruction_stats = image_quant_stats(
                reconstruction_pil,
                levels=args.glcm_levels,
                distances=glcm_distances,
                angles=glcm_angles,
            )
            stat_row = {
                **_prefix_stats("reference", reference_stats),
                **_prefix_stats("reconstruction", reconstruction_stats),
            }
            stat_row.update(_distance_stats(stat_row, left="reconstruction", right="reference", prefix="recon_ref"))
            row = {
                "index": index,
                "reference_sample_id": ref_id,
                "reference_image": str(reference_path),
                "dataset": record.get("dataset", ""),
                **compute_array_metrics(reconstruction_array, reference_array),
                **z_ref_stats(z_ref),
                **stat_row,
            }
            rows.append(row)
            (sample_dir / "metrics.json").write_text(
                json.dumps(row, indent=2, ensure_ascii=False, allow_nan=True),
                encoding="utf8",
            )
            print(
                f"[{index + 1}/{len(records)}] ref={ref_id} "
                f"vae_l1={row['full_l1']:.4f} vae_psnr={row['full_psnr']:.2f} "
                f"recon_ref_glcm_l2={row['recon_ref_glcm_l2']:.4f} z_ref_std={row['z_ref_std']:.4f}"
            )

    write_rows(output_dir, rows)
    summary = aggregate_rows(rows)
    summary["diagnostic"] = "vae_z_ref_texture_capacity"
    summary["interpretation"] = interpret_summary(summary)
    summary["glcm_config"] = {
        "levels": args.glcm_levels,
        "distances": glcm_distances,
        "angles_degrees": glcm_angles,
    }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf8",
    )
    if panel_paths:
        make_overview(panel_paths).save(output_dir / "overview_grid.png")
    print(f"wrote z_ref VAE diagnostic outputs to {output_dir}")
    return 0


def build_reference_records(
    *,
    image_paths: list[str],
    metadata_records: list[dict[str, Any]] | None,
    reference_sample_ids: list[str],
    num_samples: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if image_paths:
        return [
            {"reference_sample_id": Path(path).stem, "reference_image": path, "dataset": ""}
            for path in image_paths
        ]
    if metadata_records is None:
        raise ValueError("Provide either --image or --metadata.")
    records = unique_reference_records(metadata_records)
    if reference_sample_ids:
        by_id = {reference_record_id(record): record for record in records}
        missing = [sample_id for sample_id in reference_sample_ids if sample_id not in by_id]
        if missing:
            raise ValueError(f"reference sample_id(s) not found: {missing}")
        return [by_id[sample_id] for sample_id in reference_sample_ids]
    if num_samples is None or num_samples <= 0 or num_samples >= len(records):
        return records
    selected = list(records)
    random.Random(seed).shuffle(selected)
    return selected[:num_samples]


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


def resolve_device(device: str) -> str:
    value = str(device or "cuda").strip().lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"CUDA device {device!r} was requested, but CUDA is not available.")
    return value


def resolve_dtype(name: str, device: str) -> torch.dtype:
    if "cpu" in str(device).lower():
        return torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


@torch.no_grad()
def encode_decode_z_ref(*, vae, image: torch.Tensor, device: str, dtype: torch.dtype) -> tuple[torch.Tensor, Image.Image]:
    image_batch = image.unsqueeze(0).to(device=device, dtype=dtype)
    encoded_input = image_batch * 2.0 - 1.0
    posterior = vae.encode(encoded_input).latent_dist
    latent = posterior.mode() if hasattr(posterior, "mode") else posterior.mean
    z_ref = (latent - vae.config.shift_factor) * vae.config.scaling_factor
    decoded_latent = (z_ref / vae.config.scaling_factor) + vae.config.shift_factor
    decoded = vae.decode(decoded_latent.to(dtype=dtype), return_dict=False)[0]
    decoded = torch.clamp((decoded.float() + 1.0) / 2.0, 0.0, 1.0)
    array = decoded[0].detach().cpu().permute(1, 2, 0).numpy()
    return z_ref, Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB")


def pil_to_chw_float(image: Image.Image) -> np.ndarray:
    return np.transpose(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0, (2, 0, 1))


def compute_array_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    if pred.shape != tgt.shape:
        raise ValueError(f"prediction and target shapes differ: {pred.shape} vs {tgt.shape}")
    diff = pred - tgt
    mse = float(np.square(diff).mean())
    psnr = math.inf if mse <= 0.0 else float(-10.0 * math.log10(mse))
    return {"full_l1": float(np.abs(diff).mean()), "full_mse": mse, "full_psnr": psnr}


def z_ref_stats(z_ref: torch.Tensor) -> dict[str, float]:
    value = z_ref.detach().float()
    return {
        "z_ref_mean": float(value.mean().item()),
        "z_ref_std": float(value.std().item()),
        "z_ref_min": float(value.min().item()),
        "z_ref_max": float(value.max().item()),
        "z_ref_l2_norm": float(torch.linalg.vector_norm(value).item()),
        "z_ref_shape_b": float(value.shape[0]),
        "z_ref_shape_c": float(value.shape[1]),
        "z_ref_shape_h": float(value.shape[2]),
        "z_ref_shape_w": float(value.shape[3]),
    }


def save_error_image(error: np.ndarray, path: Path) -> None:
    Image.fromarray((np.clip(error, 0.0, 1.0) * 255).astype(np.uint8), mode="L").save(path)


def make_panel(
    *,
    reference: Image.Image,
    reconstruction: Image.Image,
    abs_error: np.ndarray,
    thumbnail_size: int,
    title: str,
) -> Image.Image:
    images = [
        ("reference", reference.convert("RGB")),
        ("vae_decode_z_ref", reconstruction.convert("RGB")),
        ("abs_error", Image.fromarray((np.clip(abs_error, 0.0, 1.0) * 255).astype(np.uint8), mode="L").convert("RGB")),
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


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"num_samples": 0.0}
    output: dict[str, float] = {"num_samples": float(len(rows))}
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (float, int)) and key != "index"})
    for key in keys:
        values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
        if values:
            output[f"{key}_mean"] = float(np.mean(values))
            output[f"{key}_std"] = float(np.std(values))
    return output


def interpret_summary(summary: dict[str, float]) -> str:
    l1 = float(summary.get("full_l1_mean", math.nan))
    psnr = float(summary.get("full_psnr_mean", math.nan))
    glcm = float(summary.get("recon_ref_glcm_l2_mean", math.nan))
    if (math.isfinite(l1) and l1 <= 0.06) or (math.isfinite(psnr) and psnr >= 22.0):
        return "z_ref_vae_preserves_reference_texture"
    if (math.isfinite(l1) and l1 >= 0.14) or (math.isfinite(psnr) and psnr <= 16.0):
        return "z_ref_vae_reconstruction_is_poor"
    if math.isfinite(glcm) and glcm <= 0.15:
        return "z_ref_vae_texture_stats_are_close"
    return "z_ref_vae_texture_capacity_mixed"


if __name__ == "__main__":
    raise SystemExit(main())

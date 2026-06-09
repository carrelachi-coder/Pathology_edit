"""Visualize and quantify Cross V2.2 z_ref_bank construction.

This diagnostic builds the same block-bank reference latent used by Cross V2.2,
decodes it through the FLUX VAE, and writes visual + JSON checks for:

* mosaic/block seam severity
* label validity after downsampling to latent resolution
* zero_ref/reference-mask ablation order
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controlnet_train.modules.cross_v2_2_conditioning import (
    apply_cross_v2_2_reference_mode,
    build_cross_v2_2_block_bank_reference_latent,
)


EPS = 1e-12
MASK_PALETTE = np.array(
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode and inspect Cross V2.2 z_ref_bank block-bank latents."
    )
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--metadata", required=True, help="metadata_cross_{train,val}.json path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional Cross V2.2 checkpoint dir. Reads saved bank block/label config.",
    )
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Inspect exactly one metadata index. Overrides --num-samples selection.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bank-seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:N, cpu, or auto.")
    parser.add_argument("--torch-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--reference-bank-block-size",
        type=int,
        default=None,
        help="Override saved/default latent-grid block size.",
    )
    parser.add_argument(
        "--reference-bank-label-mode",
        choices=["tissue", "nuclei", "tissue_nuclei"],
        default=None,
        help="Override saved/default bank label mode.",
    )
    parser.add_argument(
        "--candidate-block-sizes",
        default=None,
        help="Comma-separated latent block sizes to decode, e.g. 4,8,16.",
    )
    parser.add_argument("--thumbnail-size", type=int, default=192)
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    from diffusers import AutoencoderKL

    from controlnet_train.cli.eval_controlnet_flux_cross import read_cross_metadata
    from controlnet_train.data.common import load_image_tensor, load_nuclei_mask, load_tissue_mask

    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    records = read_cross_metadata(args.metadata)
    selected = select_records(
        records,
        sample_index=args.sample_index,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    if not selected:
        raise ValueError("No records selected for z_ref_bank diagnostic.")

    checkpoint_config = load_bank_config_from_checkpoint(args.checkpoint) if args.checkpoint else {}
    active_block_size = max(
        1,
        int(
            args.reference_bank_block_size
            if args.reference_bank_block_size is not None
            else checkpoint_config.get("reference_bank_block_size", 4)
        ),
    )
    label_mode = str(
        args.reference_bank_label_mode
        if args.reference_bank_label_mode is not None
        else checkpoint_config.get("reference_bank_label_mode", "tissue_nuclei")
    )
    candidate_block_sizes = parse_block_sizes(args.candidate_block_sizes, default=[active_block_size])
    candidate_block_sizes = unique_with_active_first(candidate_block_sizes, active=active_block_size)

    device = resolve_device(args.device)
    torch_dtype = resolve_dtype(args.torch_dtype, device)

    loaded = [
        load_record_tensors(
            record_index=record_index,
            record=record,
            metadata_path=Path(args.metadata),
            load_image_tensor=load_image_tensor,
            load_tissue_mask=load_tissue_mask,
            load_nuclei_mask=load_nuclei_mask,
        )
        for record_index, record in selected
    ]
    validate_uniform_batch_shapes(loaded)

    reference_images = torch.stack([item["reference_image"] for item in loaded], dim=0)
    reference_tissue_masks = torch.stack([item["reference_tissue_mask"] for item in loaded], dim=0)
    reference_nuclei_masks = torch.stack([item["reference_nuclei_mask"] for item in loaded], dim=0)
    target_tissue_masks = torch.stack([item["target_tissue_mask"] for item in loaded], dim=0)
    target_nuclei_masks = torch.stack([item["target_nuclei_mask"] for item in loaded], dim=0)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch_dtype,
    ).to(device)
    vae.eval()

    with torch.inference_mode():
        z_ref = encode_images_to_deterministic_latents(vae, reference_images, torch_dtype)
        decoded_z_ref = decode_latents_to_images(vae, z_ref, torch_dtype)

        bank_outputs: dict[int, dict[str, Any]] = {}
        for block_size in candidate_block_sizes:
            generator = torch.Generator(device=z_ref.device).manual_seed(int(args.bank_seed) + block_size)
            z_ref_bank = build_cross_v2_2_block_bank_reference_latent(
                z_ref=z_ref,
                reference_tissue_mask=reference_tissue_masks,
                reference_nuclei_mask=reference_nuclei_masks,
                target_tissue_mask=target_tissue_masks,
                target_nuclei_mask=target_nuclei_masks,
                block_size=block_size,
                label_mode=label_mode,
                generator=generator,
            )
            decoded_bank = decode_latents_to_images(vae, z_ref_bank, torch_dtype)
            bank_outputs[block_size] = {
                "z_ref_bank": z_ref_bank.detach().cpu(),
                "decoded": decoded_bank,
                "summary": summarize_bank(
                    z_ref_bank=z_ref_bank,
                    z_ref=z_ref,
                    decoded_bank=decoded_bank,
                    decoded_z_ref=decoded_z_ref,
                    block_size=block_size,
                    image_shape=tuple(reference_images.shape[-2:]),
                ),
            }

    latent_size = tuple(int(v) for v in z_ref.shape[-2:])
    ref_tissue_latent = downsample_label_mask(reference_tissue_masks, latent_size).cpu()
    ref_nuclei_latent = downsample_label_mask(reference_nuclei_masks, latent_size).cpu()
    tar_tissue_latent = downsample_label_mask(target_tissue_masks, latent_size).cpu()
    tar_nuclei_latent = downsample_label_mask(target_nuclei_masks, latent_size).cpu()

    panel_paths = []
    sample_reports = []
    for batch_index, item in enumerate(loaded):
        sample_dir = samples_dir / f"{batch_index:04d}_{safe_name(item['sample_id'])}__ref_{safe_name(item['reference_sample_id'])}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        reference_pil = Image.open(item["reference_image_path"]).convert("RGB")
        target_pil = Image.open(item["target_image_path"]).convert("RGB")
        reference_pil.save(sample_dir / "reference.png")
        target_pil.save(sample_dir / "target.png")
        decoded_z_ref[batch_index].save(sample_dir / "z_ref_decoded.png")

        original_mask_images = write_original_mask_images(
            sample_dir=sample_dir,
            ref_tissue=item["reference_tissue_mask"],
            ref_nuclei=item["reference_nuclei_mask"],
            tar_tissue=item["target_tissue_mask"],
            tar_nuclei=item["target_nuclei_mask"],
        )
        latent_mask_images = write_latent_mask_images(
            sample_dir=sample_dir,
            image_size=reference_pil.size,
            ref_tissue=ref_tissue_latent[batch_index],
            ref_nuclei=ref_nuclei_latent[batch_index],
            tar_tissue=tar_tissue_latent[batch_index],
            tar_nuclei=tar_nuclei_latent[batch_index],
        )

        bank_panel_images = []
        active_bank_panel = None
        bank_reports = {}
        for block_size in candidate_block_sizes:
            decoded_bank = bank_outputs[block_size]["decoded"][batch_index]
            bank_path = sample_dir / f"z_ref_bank_b{block_size}_decoded.png"
            decoded_bank.save(bank_path)
            gridded = draw_block_grid(
                decoded_bank,
                latent_size=latent_size,
                block_size=block_size,
            )
            gridded.save(sample_dir / f"z_ref_bank_b{block_size}_decoded_grid.png")
            if block_size == active_block_size:
                active_bank_panel = (f"z_ref_bank_b{block_size}", gridded)
            else:
                bank_panel_images.append((f"bank_b{block_size}", gridded))
            save_decoded_abs_diff(
                decoded_bank,
                decoded_z_ref[batch_index],
                sample_dir / f"z_ref_bank_b{block_size}_vs_z_ref_absdiff.png",
            )
            bank_reports[str(block_size)] = bank_outputs[block_size]["summary"]["per_sample"][batch_index]
        if active_bank_panel is None:
            raise RuntimeError(f"Active bank block size {active_block_size} was not decoded.")

        panel = make_panel(
            title=f"{item['sample_id']} | ref={item['reference_sample_id']}",
            columns=[
                ("reference", reference_pil),
                active_bank_panel,
                ("target", target_pil),
                ("target_tissue", original_mask_images["tar_tissue"]),
                ("target_nuclei", original_mask_images["tar_nuclei"]),
                ("z_ref_dec", decoded_z_ref[batch_index]),
                *bank_panel_images,
                ("ref_tissue_lat", latent_mask_images["ref_tissue"]),
                ("tar_tissue_lat", latent_mask_images["tar_tissue"]),
                ("ref_nuclei_lat", latent_mask_images["ref_nuclei"]),
                ("tar_nuclei_lat", latent_mask_images["tar_nuclei"]),
            ],
            thumbnail_size=int(args.thumbnail_size),
        )
        panel_path = sample_dir / "panel.png"
        panel.save(panel_path)
        panel_paths.append(panel_path)

        sample_report = build_sample_report(
            item=item,
            batch_index=batch_index,
            latent_size=latent_size,
            ref_tissue_latent=ref_tissue_latent,
            ref_nuclei_latent=ref_nuclei_latent,
            tar_tissue_latent=tar_tissue_latent,
            tar_nuclei_latent=tar_nuclei_latent,
            bank_reports=bank_reports,
            image_shape=tuple(reference_images.shape[-2:]),
            reference_bank_label_mode=label_mode,
        )
        (sample_dir / "diagnostics.json").write_text(
            json.dumps(sample_report, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
            encoding="utf8",
        )
        sample_reports.append(sample_report)

    overview_path = None
    if panel_paths:
        overview = make_overview(panel_paths)
        overview_path = output_dir / "overview_grid.png"
        overview.save(overview_path)

    active_bank = bank_outputs[active_block_size]["z_ref_bank"].to(device=device, dtype=z_ref.dtype)
    summary = {
        "pretrained_model_name_or_path": str(args.pretrained_model_name_or_path),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "metadata": str(args.metadata),
        "output_dir": str(output_dir),
        "overview_grid": str(overview_path) if overview_path else None,
        "num_samples": len(sample_reports),
        "device": str(device),
        "torch_dtype": str(torch_dtype).replace("torch.", ""),
        "active_reference_bank_block_size": int(active_block_size),
        "candidate_block_sizes": [int(v) for v in candidate_block_sizes],
        "reference_bank_label_mode": label_mode,
        "latent_shape_bchw": list(z_ref.shape),
        "image_shape_bchw": list(reference_images.shape),
        "batch_bank_summary": {
            str(block_size): bank_outputs[block_size]["summary"]["batch"]
            for block_size in candidate_block_sizes
        },
        "label_validity_summary": aggregate_label_validity(sample_reports),
        "zero_reference_order_confirmation": build_zero_reference_order_report(active_bank),
        "call_order_static_confirmation": call_order_static_confirmation(),
        "samples": sample_reports,
    }
    summary["recommendation"] = build_recommendation(summary)
    (output_dir / "diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True))
    return 0


def select_records(
    records: list[dict[str, Any]],
    *,
    sample_index: int | None,
    num_samples: int,
    seed: int,
) -> list[tuple[int, dict[str, Any]]]:
    indexed = list(enumerate(records))
    if sample_index is not None:
        if not indexed:
            return []
        index = sample_index % len(indexed)
        return [indexed[index]]
    if num_samples <= 0 or num_samples >= len(indexed):
        return indexed
    selected = list(indexed)
    random.Random(seed).shuffle(selected)
    return selected[:num_samples]


def load_record_tensors(
    *,
    record_index: int,
    record: dict[str, Any],
    metadata_path: Path,
    load_image_tensor,
    load_tissue_mask,
    load_nuclei_mask,
) -> dict[str, Any]:
    base_dir = metadata_path.parent
    reference_image_path = resolve_record_path(record["reference_image"], base_dir)
    target_image_path = resolve_record_path(record["target_image"], base_dir)
    reference_tissue_mask_path = resolve_record_path(record["reference_tissue_mask"], base_dir)
    reference_nuclei_mask_path = resolve_record_path(record["reference_nuclei_mask"], base_dir)
    target_tissue_mask_path = resolve_record_path(record["target_tissue_mask"], base_dir)
    target_nuclei_mask_path = resolve_record_path(record["target_nuclei_mask"], base_dir)
    return {
        "record_index": int(record_index),
        "sample_id": str(record.get("sample_id") or target_image_path.stem),
        "reference_sample_id": str(record.get("reference_sample_id") or reference_image_path.stem),
        "dataset": record.get("dataset", ""),
        "reference_image_path": reference_image_path,
        "target_image_path": target_image_path,
        "reference_tissue_mask_path": reference_tissue_mask_path,
        "reference_nuclei_mask_path": reference_nuclei_mask_path,
        "target_tissue_mask_path": target_tissue_mask_path,
        "target_nuclei_mask_path": target_nuclei_mask_path,
        "reference_image": load_image_tensor(reference_image_path),
        "reference_tissue_mask": load_tissue_mask(reference_tissue_mask_path),
        "reference_nuclei_mask": load_nuclei_mask(reference_nuclei_mask_path),
        "target_tissue_mask": load_tissue_mask(target_tissue_mask_path),
        "target_nuclei_mask": load_nuclei_mask(target_nuclei_mask_path),
    }


def resolve_record_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(str(path_value).replace("\\", "/"))
    if path.exists() or path.is_absolute():
        return path
    base_candidate = base_dir / path
    if base_candidate.exists():
        return base_candidate
    return path


def validate_uniform_batch_shapes(items: list[dict[str, Any]]) -> None:
    if not items:
        raise ValueError("Cannot build an empty diagnostic batch.")
    shape_keys = [
        "reference_image",
        "reference_tissue_mask",
        "reference_nuclei_mask",
        "target_tissue_mask",
        "target_nuclei_mask",
    ]
    expected = {key: tuple(items[0][key].shape) for key in shape_keys}
    for item in items[1:]:
        for key in shape_keys:
            actual = tuple(item[key].shape)
            if actual != expected[key]:
                raise ValueError(
                    "Selected records must share one batch shape. "
                    f"{key} for {item['sample_id']} is {actual}, expected {expected[key]}."
                )


def load_bank_config_from_checkpoint(checkpoint_path: str | Path) -> dict[str, object]:
    path = Path(checkpoint_path)
    state_path = path / "phase5_conditioning.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing Cross V2.2 conditioning artifact: {state_path}")
    state = torch_load_weights(state_path)
    saved_spec = state.get("cross_v2_2_control_spec") or {}
    return {
        "reference_bank_block_size": max(1, int(saved_spec.get("reference_bank_block_size", 4) or 4)),
        "reference_bank_label_mode": str(saved_spec.get("reference_bank_label_mode", "tissue_nuclei")),
        "zero_reference_mask_features": bool(saved_spec.get("zero_reference_mask_features", True)),
    }


def torch_load_weights(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_block_sizes(value: str | None, *, default: Iterable[int]) -> list[int]:
    if value is None or not str(value).strip():
        return [max(1, int(v)) for v in default]
    sizes = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if not raw:
            continue
        sizes.append(max(1, int(raw)))
    if not sizes:
        raise ValueError("--candidate-block-sizes did not contain any integer block sizes.")
    return sizes


def unique_with_active_first(values: Iterable[int], *, active: int) -> list[int]:
    ordered = [int(active), *[int(value) for value in values]]
    result = []
    seen = set()
    for value in ordered:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def resolve_device(value: str) -> torch.device:
    normalized = str(value or "cuda").lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device(normalized)


def resolve_dtype(value: str, device: torch.device) -> torch.dtype:
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[value]
    if device.type == "cpu" and dtype != torch.float32:
        return torch.float32
    return dtype


def encode_images_to_deterministic_latents(vae, images: torch.Tensor, torch_dtype: torch.dtype) -> torch.Tensor:
    device = next(vae.parameters()).device
    images = images.to(device=device, dtype=torch_dtype)
    images = images * 2.0 - 1.0
    posterior = vae.encode(images).latent_dist
    latents = posterior.mode() if hasattr(posterior, "mode") else posterior.mean
    return (latents - vae.config.shift_factor) * vae.config.scaling_factor


def decode_latents_to_images(vae, latents: torch.Tensor, torch_dtype: torch.dtype) -> list[Image.Image]:
    device = next(vae.parameters()).device
    latent_input = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    decoded = vae.decode(
        latent_input.to(device=device, dtype=torch_dtype),
        return_dict=False,
    )[0]
    decoded = ((decoded.float() / 2.0) + 0.5).clamp(0.0, 1.0).detach().cpu()
    images = []
    for sample in decoded:
        array = sample.permute(1, 2, 0).numpy()
        images.append(Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB"))
    return images


def downsample_label_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"label mask must have shape (B,H,W) or (H,W), got {tuple(mask.shape)}")
    if tuple(mask.shape[-2:]) == tuple(size):
        return mask.long()
    return F.interpolate(mask.unsqueeze(1).float(), size=size, mode="nearest").squeeze(1).long()


def summarize_bank(
    *,
    z_ref_bank: torch.Tensor,
    z_ref: torch.Tensor,
    decoded_bank: list[Image.Image],
    decoded_z_ref: list[Image.Image],
    block_size: int,
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    seam = compute_seam_metrics(z_ref_bank, block_size=block_size)
    latent_delta = tensor_delta_stats(z_ref_bank, z_ref)
    per_sample = []
    for index in range(z_ref_bank.shape[0]):
        per_sample.append(
            {
                "block_size_latent": int(block_size),
                "block_size_pixels_estimate": block_size_pixels_estimate(
                    block_size=block_size,
                    latent_shape=tuple(z_ref_bank.shape[-2:]),
                    image_shape=image_shape,
                ),
                "latent_delta_vs_z_ref": tensor_delta_stats(z_ref_bank[index : index + 1], z_ref[index : index + 1]),
                "decoded_l1_vs_z_ref": image_l1(decoded_bank[index], decoded_z_ref[index]),
                "seam_metrics": compute_seam_metrics(z_ref_bank[index : index + 1], block_size=block_size),
            }
        )
    return {
        "batch": {
            "block_size_latent": int(block_size),
            "block_size_pixels_estimate": block_size_pixels_estimate(
                block_size=block_size,
                latent_shape=tuple(z_ref_bank.shape[-2:]),
                image_shape=image_shape,
            ),
            "latent_grid": list(z_ref_bank.shape[-2:]),
            "latent_block_grid": [
                int(math.ceil(z_ref_bank.shape[-2] / block_size)),
                int(math.ceil(z_ref_bank.shape[-1] / block_size)),
            ],
            "num_blocks_per_sample": int(
                math.ceil(z_ref_bank.shape[-2] / block_size)
                * math.ceil(z_ref_bank.shape[-1] / block_size)
            ),
            "seam_metrics": seam,
            "latent_delta_vs_z_ref": latent_delta,
            "decoded_l1_vs_z_ref_mean": float(np.mean([row["decoded_l1_vs_z_ref"] for row in per_sample])),
        },
        "per_sample": per_sample,
    }


def block_size_pixels_estimate(
    *,
    block_size: int,
    latent_shape: tuple[int, int],
    image_shape: tuple[int, int],
) -> list[float]:
    latent_h, latent_w = latent_shape
    image_h, image_w = image_shape
    return [
        float(block_size * image_h / max(latent_h, 1)),
        float(block_size * image_w / max(latent_w, 1)),
    ]


def compute_seam_metrics(latents: torch.Tensor, *, block_size: int) -> dict[str, float]:
    if latents.ndim != 4:
        raise ValueError(f"latents must have shape (B,C,H,W), got {tuple(latents.shape)}")
    block_size = max(1, int(block_size))
    _, _, height, width = latents.shape
    latents = latents.float()

    horizontal_diff = (latents[:, :, 1:, :] - latents[:, :, :-1, :]).abs().mean(dim=1)
    vertical_diff = (latents[:, :, :, 1:] - latents[:, :, :, :-1]).abs().mean(dim=1)

    horizontal_boundary_indices = list(range(block_size - 1, max(height - 1, 0), block_size))
    vertical_boundary_indices = list(range(block_size - 1, max(width - 1, 0), block_size))

    boundary_values = []
    non_boundary_values = []
    if horizontal_diff.numel():
        boundary_mask = torch.zeros(horizontal_diff.shape[1], dtype=torch.bool, device=horizontal_diff.device)
        if horizontal_boundary_indices:
            boundary_mask[horizontal_boundary_indices] = True
            boundary_values.append(horizontal_diff[:, boundary_mask, :].reshape(-1))
        non_boundary_values.append(horizontal_diff[:, ~boundary_mask, :].reshape(-1))
    if vertical_diff.numel():
        boundary_mask = torch.zeros(vertical_diff.shape[2], dtype=torch.bool, device=vertical_diff.device)
        if vertical_boundary_indices:
            boundary_mask[vertical_boundary_indices] = True
            boundary_values.append(vertical_diff[:, :, boundary_mask].reshape(-1))
        non_boundary_values.append(vertical_diff[:, :, ~boundary_mask].reshape(-1))

    boundary = cat_nonempty(boundary_values)
    non_boundary = cat_nonempty(non_boundary_values)
    boundary_mean = safe_tensor_mean(boundary)
    non_boundary_mean = safe_tensor_mean(non_boundary)
    all_adjacent = []
    if horizontal_diff.numel():
        all_adjacent.append(horizontal_diff.reshape(-1))
    if vertical_diff.numel():
        all_adjacent.append(vertical_diff.reshape(-1))
    all_adjacent_mean = safe_tensor_mean(torch.cat(all_adjacent) if all_adjacent else None)
    return {
        "boundary_mean_abs_adjacent_delta": boundary_mean,
        "non_boundary_mean_abs_adjacent_delta": non_boundary_mean,
        "seam_over_non_boundary_ratio": float(boundary_mean / max(non_boundary_mean, EPS))
        if math.isfinite(boundary_mean) and math.isfinite(non_boundary_mean)
        else math.nan,
        "all_mean_abs_adjacent_delta": all_adjacent_mean,
        "horizontal_boundary_count": float(len(horizontal_boundary_indices)),
        "vertical_boundary_count": float(len(vertical_boundary_indices)),
    }


def safe_tensor_mean(value: torch.Tensor | None) -> float:
    if value is None or value.numel() == 0:
        return math.nan
    return float(value.float().mean().item())


def cat_nonempty(values: list[torch.Tensor]) -> torch.Tensor | None:
    nonempty = [value for value in values if value.numel()]
    if not nonempty:
        return None
    return torch.cat(nonempty)


def tensor_delta_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a.float() - b.float()).reshape(-1)
    if diff.numel() == 0:
        return {"mean_abs": math.nan, "max_abs": math.nan, "rms": math.nan}
    return {
        "mean_abs": float(diff.abs().mean().item()),
        "max_abs": float(diff.abs().max().item()),
        "rms": float(torch.sqrt(diff.pow(2).mean()).item()),
    }


def image_l1(a: Image.Image, b: Image.Image) -> float:
    arr_a = np.asarray(a.convert("RGB"), dtype=np.float32) / 255.0
    arr_b = np.asarray(b.convert("RGB"), dtype=np.float32) / 255.0
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"image shapes differ: {arr_a.shape} vs {arr_b.shape}")
    return float(np.abs(arr_a - arr_b).mean())


def write_latent_mask_images(
    *,
    sample_dir: Path,
    image_size: tuple[int, int],
    ref_tissue: torch.Tensor,
    ref_nuclei: torch.Tensor,
    tar_tissue: torch.Tensor,
    tar_nuclei: torch.Tensor,
) -> dict[str, Image.Image]:
    masks = {
        "ref_tissue": mask_to_rgb(ref_tissue),
        "ref_nuclei": mask_to_rgb(ref_nuclei),
        "tar_tissue": mask_to_rgb(tar_tissue),
        "tar_nuclei": mask_to_rgb(tar_nuclei),
    }
    resized = {}
    for name, image in masks.items():
        upscaled = image.resize(image_size, resample=Image.Resampling.NEAREST)
        upscaled.save(sample_dir / f"{name}_latent_nearest.png")
        resized[name] = upscaled
    return resized


def write_original_mask_images(
    *,
    sample_dir: Path,
    ref_tissue: torch.Tensor,
    ref_nuclei: torch.Tensor,
    tar_tissue: torch.Tensor,
    tar_nuclei: torch.Tensor,
) -> dict[str, Image.Image]:
    masks = {
        "ref_tissue": mask_to_rgb(ref_tissue),
        "ref_nuclei": mask_to_rgb(ref_nuclei),
        "tar_tissue": mask_to_rgb(tar_tissue),
        "tar_nuclei": mask_to_rgb(tar_nuclei),
    }
    for name, image in masks.items():
        image.save(sample_dir / f"{name}_mask.png")
    return masks


def mask_to_rgb(mask: torch.Tensor | np.ndarray) -> Image.Image:
    labels = np.asarray(mask.detach().cpu() if isinstance(mask, torch.Tensor) else mask).astype(np.int64)
    rgb = MASK_PALETTE[np.clip(labels, 0, len(MASK_PALETTE) - 1)]
    return Image.fromarray(rgb, mode="RGB")


def draw_block_grid(image: Image.Image, *, latent_size: tuple[int, int], block_size: int) -> Image.Image:
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    latent_h, latent_w = latent_size
    width, height = result.size
    scale_x = width / max(latent_w, 1)
    scale_y = height / max(latent_h, 1)
    line_color = (255, 230, 40)
    for x_latent in range(block_size, latent_w, block_size):
        x = int(round(x_latent * scale_x))
        draw.line([(x, 0), (x, height)], fill=line_color, width=1)
    for y_latent in range(block_size, latent_h, block_size):
        y = int(round(y_latent * scale_y))
        draw.line([(0, y), (width, y)], fill=line_color, width=1)
    return result


def save_decoded_abs_diff(a: Image.Image, b: Image.Image, path: Path) -> None:
    arr_a = np.asarray(a.convert("RGB"), dtype=np.float32) / 255.0
    arr_b = np.asarray(b.convert("RGB"), dtype=np.float32) / 255.0
    diff = np.abs(arr_a - arr_b).mean(axis=-1)
    Image.fromarray((np.clip(diff, 0.0, 1.0) * 255).round().astype(np.uint8), mode="L").save(path)


def make_panel(*, title: str, columns: list[tuple[str, Image.Image]], thumbnail_size: int) -> Image.Image:
    label_h = 34
    title_h = 28
    thumbs = [(label, thumbnail(image, thumbnail_size)) for label, image in columns]
    width = thumbnail_size * len(thumbs)
    height = thumbnail_size + label_h + title_h
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((6, 6), title[:180], fill=(0, 0, 0))
    for index, (label, image) in enumerate(thumbs):
        x = index * thumbnail_size
        panel.paste(image, (x, title_h))
        draw.text((x + 6, title_h + thumbnail_size + 8), label[:28], fill=(0, 0, 0))
    return panel


def thumbnail(image: Image.Image, size: int) -> Image.Image:
    thumb = image.convert("RGB").copy()
    thumb.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - thumb.width) // 2
    y = (size - thumb.height) // 2
    canvas.paste(thumb, (x, y))
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


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)[:120]


def build_sample_report(
    *,
    item: dict[str, Any],
    batch_index: int,
    latent_size: tuple[int, int],
    ref_tissue_latent: torch.Tensor,
    ref_nuclei_latent: torch.Tensor,
    tar_tissue_latent: torch.Tensor,
    tar_nuclei_latent: torch.Tensor,
    bank_reports: dict[str, Any],
    image_shape: tuple[int, int],
    reference_bank_label_mode: str,
) -> dict[str, Any]:
    image_h, image_w = image_shape
    scale_y = image_h / max(latent_size[0], 1)
    scale_x = image_w / max(latent_size[1], 1)
    enriched_bank_reports = {}
    for block_size_text, bank_report in bank_reports.items():
        block_size = int(block_size_text)
        enriched_bank_reports[block_size_text] = {
            **bank_report,
            "block_label_match_summary": block_label_match_summary(
                reference_tissue_latent=ref_tissue_latent[batch_index],
                reference_nuclei_latent=ref_nuclei_latent[batch_index],
                target_tissue_latent=tar_tissue_latent[batch_index],
                target_nuclei_latent=tar_nuclei_latent[batch_index],
                block_size=block_size,
                label_mode=reference_bank_label_mode,
            ),
        }
    return {
        "record_index": int(item["record_index"]),
        "batch_index": int(batch_index),
        "sample_id": str(item["sample_id"]),
        "reference_sample_id": str(item["reference_sample_id"]),
        "dataset": item.get("dataset", ""),
        "image_shape_hw": [int(image_h), int(image_w)],
        "latent_shape_hw": [int(latent_size[0]), int(latent_size[1])],
        "image_pixels_per_latent_cell": [float(scale_y), float(scale_x)],
        "reference_tissue_label_report": label_grid_report(
            item["reference_tissue_mask"],
            ref_tissue_latent[batch_index],
        ),
        "reference_nuclei_label_report": label_grid_report(
            item["reference_nuclei_mask"],
            ref_nuclei_latent[batch_index],
        ),
        "target_tissue_label_report": label_grid_report(
            item["target_tissue_mask"],
            tar_tissue_latent[batch_index],
        ),
        "target_nuclei_label_report": label_grid_report(
            item["target_nuclei_mask"],
            tar_nuclei_latent[batch_index],
        ),
        "reference_nuclei_component_report": component_scale_report(
            item["reference_nuclei_mask"],
            ref_nuclei_latent[batch_index],
            pixels_per_latent_cell=scale_y * scale_x,
        ),
        "target_nuclei_component_report": component_scale_report(
            item["target_nuclei_mask"],
            tar_nuclei_latent[batch_index],
            pixels_per_latent_cell=scale_y * scale_x,
        ),
        "bank_reports": enriched_bank_reports,
    }


def label_grid_report(original: torch.Tensor | np.ndarray, latent: torch.Tensor | np.ndarray) -> dict[str, Any]:
    original_np = as_int_numpy(original)
    latent_np = as_int_numpy(latent)
    original_counts = unique_counts(original_np)
    latent_counts = unique_counts(latent_np)
    original_nonzero = {label for label in original_counts if label != 0}
    latent_nonzero = {label for label in latent_counts if label != 0}
    retained = sorted(original_nonzero & latent_nonzero)
    missing = sorted(original_nonzero - latent_nonzero)
    return {
        "original_shape": list(original_np.shape),
        "latent_shape": list(latent_np.shape),
        "original_unique_counts": stringify_keys(original_counts),
        "latent_unique_counts": stringify_keys(latent_counts),
        "original_nonzero_label_count": int(len(original_nonzero)),
        "latent_nonzero_label_count": int(len(latent_nonzero)),
        "retained_nonzero_labels": [int(v) for v in retained],
        "missing_nonzero_labels": [int(v) for v in missing],
        "nonzero_label_retention_fraction": float(len(retained) / len(original_nonzero))
        if original_nonzero
        else 1.0,
        "original_positive_fraction": positive_fraction(original_np),
        "latent_positive_fraction": positive_fraction(latent_np),
        "original_transition_density": label_transition_density(original_np),
        "latent_transition_density": label_transition_density(latent_np),
    }


def block_label_match_summary(
    *,
    reference_tissue_latent: torch.Tensor,
    reference_nuclei_latent: torch.Tensor,
    target_tissue_latent: torch.Tensor,
    target_nuclei_latent: torch.Tensor,
    block_size: int,
    label_mode: str,
) -> dict[str, Any]:
    ref_tissue = reference_tissue_latent.long()
    ref_nuclei = reference_nuclei_latent.long()
    tar_tissue = target_tissue_latent.long()
    tar_nuclei = target_nuclei_latent.long()
    height, width = tuple(ref_tissue.shape[-2:])
    block_size = max(1, int(block_size))

    ref_summary = summarize_block_label_grid(ref_tissue, ref_nuclei, block_size=block_size)
    target_summary = summarize_block_label_grid(tar_tissue, tar_nuclei, block_size=block_size)

    pools = build_block_label_pool_counts(ref_tissue, ref_nuclei, block_size=block_size, label_mode=label_mode)
    fallback_counts = {"exact": 0, "tissue": 0, "nuclei": 0, "all": 0}
    for y0, y1, x0, x1 in iter_block_slices(height, width, block_size):
        tissue_label = majority_label_tensor(tar_tissue[y0:y1, x0:x1])
        nuclei_label = majority_label_tensor(tar_nuclei[y0:y1, x0:x1])
        stage = select_pool_stage(
            pools=pools,
            tissue_label=tissue_label,
            nuclei_label=nuclei_label,
            label_mode=label_mode,
        )
        fallback_counts[stage] += 1

    total_blocks = max(1, target_summary["block_count"])
    return {
        "label_mode": str(label_mode),
        "block_size_latent": int(block_size),
        "reference_blocks": ref_summary,
        "target_blocks": target_summary,
        "pool_selection_counts": fallback_counts,
        "pool_selection_fractions": {
            key: float(value / total_blocks) for key, value in fallback_counts.items()
        },
        "interpretation": interpret_nuclei_block_usage(ref_summary, target_summary, fallback_counts),
    }


def summarize_block_label_grid(tissue: torch.Tensor, nuclei: torch.Tensor, *, block_size: int) -> dict[str, Any]:
    height, width = tuple(tissue.shape[-2:])
    block_count = 0
    nuclei_any_nonzero = 0
    nuclei_majority_nonzero = 0
    nuclei_any_but_majority_zero = 0
    tissue_majority_counts: dict[int, int] = {}
    nuclei_majority_counts: dict[int, int] = {}
    for y0, y1, x0, x1 in iter_block_slices(height, width, block_size):
        block_count += 1
        tissue_label = majority_label_tensor(tissue[y0:y1, x0:x1])
        nuclei_block = nuclei[y0:y1, x0:x1]
        nuclei_label = majority_label_tensor(nuclei_block)
        any_nonzero = bool((nuclei_block != 0).any().item())
        if any_nonzero:
            nuclei_any_nonzero += 1
        if nuclei_label != 0:
            nuclei_majority_nonzero += 1
        if any_nonzero and nuclei_label == 0:
            nuclei_any_but_majority_zero += 1
        tissue_majority_counts[tissue_label] = tissue_majority_counts.get(tissue_label, 0) + 1
        nuclei_majority_counts[nuclei_label] = nuclei_majority_counts.get(nuclei_label, 0) + 1
    any_positive = max(1, nuclei_any_nonzero)
    total = max(1, block_count)
    return {
        "block_count": int(block_count),
        "tissue_majority_counts": stringify_keys(tissue_majority_counts),
        "nuclei_majority_counts": stringify_keys(nuclei_majority_counts),
        "nuclei_any_nonzero_block_fraction": float(nuclei_any_nonzero / total),
        "nuclei_majority_nonzero_block_fraction": float(nuclei_majority_nonzero / total),
        "nuclei_any_but_majority_zero_block_fraction": float(nuclei_any_but_majority_zero / total),
        "nuclei_any_but_majority_zero_among_nuclei_blocks_fraction": float(
            nuclei_any_but_majority_zero / any_positive
        ),
    }


def build_block_label_pool_counts(
    tissue: torch.Tensor,
    nuclei: torch.Tensor,
    *,
    block_size: int,
    label_mode: str,
) -> dict[str, dict[Any, int]]:
    height, width = tuple(tissue.shape[-2:])
    exact: dict[Any, int] = {}
    tissue_counts: dict[int, int] = {}
    nuclei_counts: dict[int, int] = {}
    for y0, y1, x0, x1 in iter_block_slices(height, width, block_size):
        tissue_label = majority_label_tensor(tissue[y0:y1, x0:x1])
        nuclei_label = majority_label_tensor(nuclei[y0:y1, x0:x1])
        key = label_key(tissue_label, nuclei_label, label_mode)
        exact[key] = exact.get(key, 0) + 1
        tissue_counts[tissue_label] = tissue_counts.get(tissue_label, 0) + 1
        nuclei_counts[nuclei_label] = nuclei_counts.get(nuclei_label, 0) + 1
    return {"exact": exact, "tissue": tissue_counts, "nuclei": nuclei_counts}


def select_pool_stage(
    *,
    pools: dict[str, dict[Any, int]],
    tissue_label: int,
    nuclei_label: int,
    label_mode: str,
) -> str:
    if pools["exact"].get(label_key(tissue_label, nuclei_label, label_mode), 0) > 0:
        return "exact"
    if label_mode == "tissue_nuclei" and pools["tissue"].get(int(tissue_label), 0) > 0:
        return "tissue"
    if label_mode == "tissue_nuclei" and pools["nuclei"].get(int(nuclei_label), 0) > 0:
        return "nuclei"
    return "all"


def interpret_nuclei_block_usage(
    reference_summary: dict[str, Any],
    target_summary: dict[str, Any],
    fallback_counts: dict[str, int],
) -> str:
    target_any = float(target_summary["nuclei_any_nonzero_block_fraction"])
    target_majority = float(target_summary["nuclei_majority_nonzero_block_fraction"])
    target_lost = float(target_summary["nuclei_any_but_majority_zero_among_nuclei_blocks_fraction"])
    ref_majority = float(reference_summary["nuclei_majority_nonzero_block_fraction"])
    if target_any > 0.05 and target_majority < 0.05:
        return (
            "nuclei pixels exist in target blocks, but block majority is almost always 0; "
            "z_ref_bank selection is effectively not using nuclei layout."
        )
    if target_lost > 0.5:
        return (
            "more than half of target blocks containing nuclei still get nuclei majority 0; "
            "background dominates the nuclei label at this block size."
        )
    if ref_majority < 0.05 and fallback_counts.get("exact", 0) > 0:
        return (
            "reference pools mostly have nuclei majority 0, so exact matches are likely tissue+background-nuclei matches."
        )
    return "nuclei majority labels survive at this block size; inspect pool_selection_fractions for fallback behavior."


def iter_block_slices(height: int, width: int, block_size: int):
    for y0 in range(0, height, block_size):
        y1 = min(y0 + block_size, height)
        for x0 in range(0, width, block_size):
            x1 = min(x0 + block_size, width)
            yield y0, y1, x0, x1


def majority_label_tensor(labels: torch.Tensor) -> int:
    flat = labels.reshape(-1).long()
    if flat.numel() == 0:
        return 0
    values, counts = torch.unique(flat, sorted=True, return_counts=True)
    return int(values[counts.argmax()].item())


def label_key(tissue_label: int, nuclei_label: int, label_mode: str) -> Any:
    if label_mode == "tissue":
        return int(tissue_label)
    if label_mode == "nuclei":
        return int(nuclei_label)
    return int(tissue_label), int(nuclei_label)


def as_int_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value).astype(np.int64, copy=False)


def unique_counts(array: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(array.astype(np.int64, copy=False), return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def stringify_keys(mapping: dict[int, int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(mapping.items())}


def positive_fraction(array: np.ndarray) -> float:
    if array.size == 0:
        return math.nan
    return float(np.count_nonzero(array) / array.size)


def label_transition_density(array: np.ndarray) -> float:
    if array.ndim != 2 or array.size == 0:
        return math.nan
    changes = 0
    total = 0
    if array.shape[0] > 1:
        changes += int(np.count_nonzero(array[1:, :] != array[:-1, :]))
        total += int((array.shape[0] - 1) * array.shape[1])
    if array.shape[1] > 1:
        changes += int(np.count_nonzero(array[:, 1:] != array[:, :-1]))
        total += int(array.shape[0] * (array.shape[1] - 1))
    return float(changes / total) if total else math.nan


def component_scale_report(
    original: torch.Tensor | np.ndarray,
    latent: torch.Tensor | np.ndarray,
    *,
    pixels_per_latent_cell: float,
) -> dict[str, Any]:
    original_areas = connected_component_areas(as_int_numpy(original) != 0)
    latent_areas = connected_component_areas(as_int_numpy(latent) != 0)
    return {
        "original_positive_component_summary_pixels": component_area_summary(original_areas),
        "latent_positive_component_summary_cells": component_area_summary(latent_areas),
        "original_components_smaller_than_one_latent_cell_fraction": fraction_below(
            original_areas,
            pixels_per_latent_cell,
        ),
        "estimated_pixels_per_latent_cell": float(pixels_per_latent_cell),
    }


def connected_component_areas(mask: np.ndarray) -> list[int]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.size == 0:
        return []
    visited = np.zeros(mask.shape, dtype=bool)
    areas = []
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if visited[y, x] or not mask[y, x]:
                continue
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            area = 0
            while queue:
                cy, cx = queue.popleft()
                area += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    queue.append((ny, nx))
            areas.append(area)
    return areas


def component_area_summary(areas: list[int]) -> dict[str, float]:
    if not areas:
        return {
            "count": 0.0,
            "min": math.nan,
            "median": math.nan,
            "mean": math.nan,
            "max": math.nan,
        }
    values = np.asarray(areas, dtype=np.float32)
    return {
        "count": float(values.size),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def fraction_below(values: list[int], threshold: float) -> float:
    if not values:
        return math.nan
    arr = np.asarray(values, dtype=np.float32)
    return float(np.count_nonzero(arr < threshold) / arr.size)


def aggregate_label_validity(sample_reports: list[dict[str, Any]]) -> dict[str, float]:
    if not sample_reports:
        return {}
    keys = [
        "reference_tissue_label_report",
        "reference_nuclei_label_report",
        "target_tissue_label_report",
        "target_nuclei_label_report",
    ]
    output = {}
    for key in keys:
        retentions = [
            float(report[key]["nonzero_label_retention_fraction"])
            for report in sample_reports
            if key in report
        ]
        positives = [
            float(report[key]["latent_positive_fraction"])
            for report in sample_reports
            if key in report
        ]
        output[f"{key}_retention_fraction_mean"] = float(np.mean(retentions)) if retentions else math.nan
        output[f"{key}_latent_positive_fraction_mean"] = float(np.mean(positives)) if positives else math.nan
    return output


def build_zero_reference_order_report(z_ref_bank: torch.Tensor) -> dict[str, Any]:
    mock_ref_tissue = torch.ones(
        z_ref_bank.shape[0],
        2,
        z_ref_bank.shape[2],
        z_ref_bank.shape[3],
        device=z_ref_bank.device,
        dtype=z_ref_bank.dtype,
    )
    mock_ref_nuclei = torch.full_like(mock_ref_tissue, 2.0)
    with_ref = apply_cross_v2_2_reference_mode(
        z_ref=z_ref_bank,
        ref_tissue_feat=mock_ref_tissue,
        ref_nuclei_feat=mock_ref_nuclei,
        mode="with_ref",
    )
    zero_ref = apply_cross_v2_2_reference_mode(
        z_ref=z_ref_bank,
        ref_tissue_feat=mock_ref_tissue,
        ref_nuclei_feat=mock_ref_nuclei,
        mode="zero_ref",
    )
    return {
        "build_happens_before_zero_ref_ablation": True,
        "zero_reference_mask_features_only_zeroes": ["ref_tissue_feat", "ref_nuclei_feat"],
        "zero_ref_ablation_zeroes": ["z_ref_bank", "ref_tissue_feat", "ref_nuclei_feat"],
        "with_ref_preserves_z_ref_bank_max_abs_delta": max_abs_delta(with_ref[0], z_ref_bank),
        "zero_ref_z_ref_bank_max_abs": max_abs(zero_ref[0]),
        "zero_ref_ref_tissue_feat_max_abs": max_abs(zero_ref[1]),
        "zero_ref_ref_nuclei_feat_max_abs": max_abs(zero_ref[2]),
    }


def max_abs(value: torch.Tensor) -> float:
    return float(value.detach().float().abs().max().item()) if value.numel() else math.nan


def max_abs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return max_abs(a - b)


def call_order_static_confirmation() -> dict[str, Any]:
    return {
        "training": [
            "encode target/noising/reference latents",
            "encode reference and target mask features",
            "build_cross_v2_2_block_bank_reference_latent from raw masks",
            "if zero_reference_mask_features: zero ref_tissue_feat/ref_nuclei_feat",
            "build_cross_v2_2_condition with z_ref_bank first",
        ],
        "inference_and_eval": [
            "encode reference latent",
            "encode reference and target mask features",
            "build_cross_v2_2_block_bank_reference_latent from raw masks",
            "if zero_reference_mask_features: zero ref_tissue_feat/ref_nuclei_feat",
            "apply_cross_v2_2_reference_mode; zero_ref zeros z_ref_bank/ref_tissue/ref_nuclei",
            "build_cross_v2_2_condition with z_ref_bank first",
        ],
        "source_locations": {
            "training_build_then_mask_zero": "controlnet_train/training/flux_phase5_cross_v2_2.py:_build_cross_v2_2_control_batch",
            "inference_build_then_mode_ablation": "controlnet_train/inference/pipeline_cross_v2_2.py:run_cross_v2_2_bundle",
            "zero_ref_implementation": "controlnet_train/modules/cross_v2_2_conditioning.py:apply_cross_v2_2_reference_mode",
        },
    }


def build_recommendation(summary: dict[str, Any]) -> str:
    active = str(summary["active_reference_bank_block_size"])
    bank = summary["batch_bank_summary"].get(active, {})
    seam_ratio = float(
        bank.get("seam_metrics", {}).get("seam_over_non_boundary_ratio", math.nan)
    )
    num_blocks = int(bank.get("num_blocks_per_sample", 0) or 0)
    pixel_block = bank.get("block_size_pixels_estimate", [math.nan, math.nan])
    min_pixel_block = min(float(pixel_block[0]), float(pixel_block[1]))
    reasons = []
    if math.isfinite(seam_ratio) and seam_ratio > 1.35:
        reasons.append(f"seam ratio {seam_ratio:.2f} is elevated")
    if num_blocks > 128:
        reasons.append(f"{num_blocks} blocks per sample can look fragmented")
    if math.isfinite(min_pixel_block) and min_pixel_block < 48:
        reasons.append(f"pixel block size is only about {min_pixel_block:.1f}px")
    if reasons:
        return (
            "Inspect overview_grid.png; if decoded bank also looks tiled, try "
            "--reference-bank-block-size 8 or 16, or switch to larger contiguous region sampling. "
            + "Signals: "
            + "; ".join(reasons)
            + "."
        )
    return "Current block size is not numerically flagged; still inspect decoded bank/grid images for visible tiling."


if __name__ == "__main__":
    raise SystemExit(main())

"""Prototype layered Cross V2.2 condition construction.

This script tries the split proposed in discussion:

1. remove nuclei from the reference image and fill the holes with same-tissue
   background, producing a tissue-only reference image;
2. build a large-block tissue-only latent bank from that image;
3. build a separate nuclei RGBA layer by pasting reference nuclei appearance
   onto the target nuclei mask;
4. alpha-composite the nuclei layer over the tissue bank and VAE round-trip the
   result as a candidate z_ref condition image.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controlnet_train.modules.cross_v2_2_conditioning import (
    build_cross_v2_2_block_bank_reference_latent,
)
from scripts.diagnose_cross_v2_2_z_ref_bank import (
    decode_latents_to_images,
    draw_block_grid,
    encode_images_to_deterministic_latents,
    load_record_tensors,
    make_overview,
    make_panel,
    mask_to_rgb,
    resolve_device,
    resolve_dtype,
    safe_name,
    select_records,
    validate_uniform_batch_shapes,
)


@dataclass(frozen=True)
class NucleusPrototype:
    label: int
    area: int
    bbox: tuple[int, int, int, int]
    rgb: np.ndarray
    alpha: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prototype tissue-only bank + nuclei-bank layered Cross V2.2 condition."
    )
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--metadata", required=True, help="metadata_cross_{train,val}.json path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bank-seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:N, cpu, or auto.")
    parser.add_argument("--torch-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--tissue-bank-block-size", type=int, default=8)
    parser.add_argument(
        "--candidate-tissue-block-sizes",
        default=None,
        help="Optional comma-separated tissue bank block sizes to preview, e.g. 8,16.",
    )
    parser.add_argument("--nuclei-hole-dilate-radius", type=int, default=3)
    parser.add_argument("--hole-feather-radius", type=float, default=1.5)
    parser.add_argument("--nucleus-context-radius", type=int, default=4)
    parser.add_argument("--nucleus-alpha-feather", type=float, default=1.0)
    parser.add_argument("--min-nucleus-area", type=int, default=4)
    parser.add_argument(
        "--max-target-nuclei-components",
        type=int,
        default=0,
        help="Limit pasted target components per sample. 0 means no limit.",
    )
    parser.add_argument("--thumbnail-size", type=int, default=176)
    parser.add_argument("--save-latents", action="store_true")
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
        raise ValueError("No records selected for layered condition prototype.")

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

    device = resolve_device(args.device)
    torch_dtype = resolve_dtype(args.torch_dtype, device)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch_dtype,
    ).to(device)
    vae.eval()

    reference_tissue_masks = torch.stack([item["reference_tissue_mask"] for item in loaded], dim=0)
    target_tissue_masks = torch.stack([item["target_tissue_mask"] for item in loaded], dim=0)
    zero_reference_nuclei = torch.zeros_like(reference_tissue_masks)
    zero_target_nuclei = torch.zeros_like(target_tissue_masks)

    tissue_only_arrays = []
    per_sample_precompute = []
    for item in loaded:
        ref_rgb = np.asarray(Image.open(item["reference_image_path"]).convert("RGB"), dtype=np.uint8)
        ref_tissue = tensor_to_int_array(item["reference_tissue_mask"])
        ref_nuclei = tensor_to_int_array(item["reference_nuclei_mask"])
        tar_nuclei = tensor_to_int_array(item["target_nuclei_mask"])
        hole = dilate_binary(ref_nuclei > 0, radius=int(args.nuclei_hole_dilate_radius))
        tissue_only, fill_report = remove_nuclei_with_same_tissue_fill(
            ref_rgb,
            hole,
            ref_tissue,
            feather_radius=float(args.hole_feather_radius),
        )
        prototypes = extract_nucleus_prototypes(
            ref_rgb,
            ref_nuclei,
            context_radius=int(args.nucleus_context_radius),
            alpha_feather=float(args.nucleus_alpha_feather),
            min_area=int(args.min_nucleus_area),
        )
        nuclei_layer, nuclei_report = synthesize_nuclei_layer(
            target_nuclei=tar_nuclei,
            prototypes=prototypes,
            seed=int(args.seed) + int(item["record_index"]),
            alpha_feather=float(args.nucleus_alpha_feather),
            max_components=int(args.max_target_nuclei_components),
        )
        tissue_only_arrays.append(tissue_only)
        per_sample_precompute.append(
            {
                "ref_rgb": ref_rgb,
                "hole": hole,
                "tissue_only": tissue_only,
                "prototypes": prototypes,
                "nuclei_layer": nuclei_layer,
                "fill_report": fill_report,
                "nuclei_report": nuclei_report,
            }
        )

    tissue_only_tensor = torch.stack(
        [image_array_to_tensor(array) for array in tissue_only_arrays],
        dim=0,
    )

    active_block_size = max(1, int(args.tissue_bank_block_size))
    candidate_block_sizes = parse_block_sizes(
        args.candidate_tissue_block_sizes,
        default=[active_block_size],
    )
    candidate_block_sizes = unique_with_active_first(candidate_block_sizes, active=active_block_size)

    with torch.inference_mode():
        tissue_only_latents = encode_images_to_deterministic_latents(vae, tissue_only_tensor, torch_dtype)
        tissue_bank_outputs: dict[int, dict[str, Any]] = {}
        for block_size in candidate_block_sizes:
            generator = torch.Generator(device=tissue_only_latents.device).manual_seed(
                int(args.bank_seed) + block_size
            )
            z_tissue_bank = build_cross_v2_2_block_bank_reference_latent(
                z_ref=tissue_only_latents,
                reference_tissue_mask=reference_tissue_masks,
                reference_nuclei_mask=zero_reference_nuclei,
                target_tissue_mask=target_tissue_masks,
                target_nuclei_mask=zero_target_nuclei,
                block_size=block_size,
                label_mode="tissue",
                generator=generator,
            )
            decoded = decode_latents_to_images(vae, z_tissue_bank, torch_dtype)
            tissue_bank_outputs[block_size] = {
                "latent": z_tissue_bank.detach().cpu(),
                "decoded": decoded,
            }

    active_tissue_bank = tissue_bank_outputs[active_block_size]["decoded"]
    layered_condition_arrays = []
    for batch_index, precomputed in enumerate(per_sample_precompute):
        tissue_bank_rgb = np.asarray(active_tissue_bank[batch_index].convert("RGB"), dtype=np.uint8)
        layered = alpha_composite_rgb(tissue_bank_rgb, precomputed["nuclei_layer"])
        layered_condition_arrays.append(layered)

    layered_condition_tensor = torch.stack(
        [image_array_to_tensor(array) for array in layered_condition_arrays],
        dim=0,
    )
    with torch.inference_mode():
        layered_condition_latents = encode_images_to_deterministic_latents(
            vae,
            layered_condition_tensor,
            torch_dtype,
        )
        layered_condition_roundtrip = decode_latents_to_images(
            vae,
            layered_condition_latents,
            torch_dtype,
        )

    sample_reports = []
    panel_paths = []
    latent_size = tuple(int(v) for v in tissue_only_latents.shape[-2:])
    for batch_index, item in enumerate(loaded):
        sample_dir = samples_dir / f"{batch_index:04d}_{safe_name(item['sample_id'])}__ref_{safe_name(item['reference_sample_id'])}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        precomputed = per_sample_precompute[batch_index]
        reference_pil = Image.open(item["reference_image_path"]).convert("RGB")
        target_pil = Image.open(item["target_image_path"]).convert("RGB")
        reference_tissue_rgb = mask_to_rgb(item["reference_tissue_mask"])
        reference_nuclei_rgb = mask_to_rgb(item["reference_nuclei_mask"])
        target_tissue_rgb = mask_to_rgb(item["target_tissue_mask"])
        target_nuclei_rgb = mask_to_rgb(item["target_nuclei_mask"])
        hole_pil = binary_mask_to_image(precomputed["hole"])
        tissue_only_pil = Image.fromarray(precomputed["tissue_only"], mode="RGB")
        nuclei_layer_white = rgba_on_background(precomputed["nuclei_layer"], background=(255, 255, 255))
        layered_condition_pil = Image.fromarray(layered_condition_arrays[batch_index], mode="RGB")

        reference_pil.save(sample_dir / "reference.png")
        target_pil.save(sample_dir / "target.png")
        reference_tissue_rgb.save(sample_dir / "reference_tissue_mask.png")
        reference_nuclei_rgb.save(sample_dir / "reference_nuclei_mask.png")
        target_tissue_rgb.save(sample_dir / "target_tissue_mask.png")
        target_nuclei_rgb.save(sample_dir / "target_nuclei_mask.png")
        hole_pil.save(sample_dir / "reference_nuclei_hole_dilated.png")
        tissue_only_pil.save(sample_dir / "reference_tissue_only.png")
        nuclei_layer_white.save(sample_dir / "nuclei_layer_on_white.png")
        Image.fromarray(precomputed["nuclei_layer"], mode="RGBA").save(sample_dir / "nuclei_layer_rgba.png")
        layered_condition_pil.save(sample_dir / "layered_condition_rgb.png")
        layered_condition_roundtrip[batch_index].save(sample_dir / "layered_condition_vae_roundtrip.png")

        active_bank_grid = draw_block_grid(
            active_tissue_bank[batch_index],
            latent_size=latent_size,
            block_size=active_block_size,
        )
        active_bank_grid.save(sample_dir / f"tissue_bank_b{active_block_size}_decoded_grid.png")
        for block_size in candidate_block_sizes:
            decoded = tissue_bank_outputs[block_size]["decoded"][batch_index]
            decoded.save(sample_dir / f"tissue_bank_b{block_size}_decoded.png")
            if block_size != active_block_size:
                draw_block_grid(decoded, latent_size=latent_size, block_size=block_size).save(
                    sample_dir / f"tissue_bank_b{block_size}_decoded_grid.png"
                )

        panel = make_panel(
            title=f"{item['sample_id']} | ref={item['reference_sample_id']}",
            columns=[
                ("reference", reference_pil),
                ("ref_nuclei", reference_nuclei_rgb),
                ("nuclei_hole", hole_pil),
                ("ref_tissue_only", tissue_only_pil),
                (f"tissue_bank_b{active_block_size}", active_bank_grid),
                ("target", target_pil),
                ("target_tissue", target_tissue_rgb),
                ("target_nuclei", target_nuclei_rgb),
                ("nuclei_layer", nuclei_layer_white),
                ("layered_condition", layered_condition_pil),
                ("vae_roundtrip", layered_condition_roundtrip[batch_index]),
            ],
            thumbnail_size=int(args.thumbnail_size),
        )
        panel_path = sample_dir / "panel.png"
        panel.save(panel_path)
        panel_paths.append(panel_path)

        report = {
            "record_index": int(item["record_index"]),
            "batch_index": int(batch_index),
            "sample_id": str(item["sample_id"]),
            "reference_sample_id": str(item["reference_sample_id"]),
            "image_shape_hw": list(precomputed["ref_rgb"].shape[:2]),
            "latent_shape_hw": list(latent_size),
            "tissue_bank_block_size": int(active_block_size),
            "fill_report": precomputed["fill_report"],
            "nuclei_report": precomputed["nuclei_report"],
            "output_files": {
                "panel": str(panel_path),
                "reference_tissue_only": str(sample_dir / "reference_tissue_only.png"),
                "tissue_bank_decoded_grid": str(sample_dir / f"tissue_bank_b{active_block_size}_decoded_grid.png"),
                "nuclei_layer_rgba": str(sample_dir / "nuclei_layer_rgba.png"),
                "layered_condition_rgb": str(sample_dir / "layered_condition_rgb.png"),
                "layered_condition_vae_roundtrip": str(sample_dir / "layered_condition_vae_roundtrip.png"),
            },
        }
        (sample_dir / "diagnostics.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
            encoding="utf8",
        )
        sample_reports.append(report)

    overview_path = None
    if panel_paths:
        overview_path = output_dir / "overview_grid.png"
        make_overview(panel_paths).save(overview_path)

    if args.save_latents:
        torch.save(
            {
                "layered_condition_latents": layered_condition_latents.detach().cpu(),
                "tissue_only_latents": tissue_only_latents.detach().cpu(),
                "tissue_bank_latents": tissue_bank_outputs[active_block_size]["latent"],
                "metadata": {
                    "tissue_bank_block_size": int(active_block_size),
                    "latent_shape_hw": list(latent_size),
                    "condition_order_note": "Use layered_condition_latents as the z_ref/layered reference condition prototype.",
                },
            },
            output_dir / "layered_condition_latents.pt",
        )

    summary = {
        "pretrained_model_name_or_path": str(args.pretrained_model_name_or_path),
        "metadata": str(args.metadata),
        "output_dir": str(output_dir),
        "overview_grid": str(overview_path) if overview_path else None,
        "num_samples": len(sample_reports),
        "device": str(device),
        "torch_dtype": str(torch_dtype).replace("torch.", ""),
        "tissue_bank_block_size": int(active_block_size),
        "candidate_tissue_block_sizes": [int(value) for value in candidate_block_sizes],
        "nuclei_hole_dilate_radius": int(args.nuclei_hole_dilate_radius),
        "nucleus_context_radius": int(args.nucleus_context_radius),
        "sample_summary": aggregate_sample_reports(sample_reports),
        "samples": sample_reports,
    }
    (output_dir / "diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True))
    return 0


def remove_nuclei_with_same_tissue_fill(
    rgb: np.ndarray,
    hole_mask: np.ndarray,
    tissue_mask: np.ndarray,
    *,
    feather_radius: float = 1.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    rgb = np.asarray(rgb, dtype=np.uint8)
    hole = np.asarray(hole_mask, dtype=bool)
    tissue = np.asarray(tissue_mask).astype(np.int64, copy=False)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"rgb must have shape (H,W,3), got {rgb.shape}")
    if hole.shape != rgb.shape[:2] or tissue.shape != rgb.shape[:2]:
        raise ValueError(
            f"shape mismatch: rgb={rgb.shape[:2]} hole={hole.shape} tissue={tissue.shape}"
        )

    filled = rgb.copy()
    filled_pixels = np.zeros(hole.shape, dtype=bool)
    per_label: dict[str, int] = {}
    for label in sorted(int(value) for value in np.unique(tissue)):
        domain = tissue == label
        label_holes = hole & domain
        if not np.any(label_holes):
            continue
        source = domain & ~hole
        if not np.any(source):
            per_label[str(label)] = 0
            continue
        nearest_y, nearest_x = nearest_source_indices(source)
        filled[label_holes] = rgb[nearest_y[label_holes], nearest_x[label_holes]]
        filled_pixels |= label_holes
        per_label[str(label)] = int(np.count_nonzero(label_holes))

    fallback_holes = hole & ~filled_pixels
    if np.any(fallback_holes):
        source = ~hole
        if np.any(source):
            nearest_y, nearest_x = nearest_source_indices(source)
            filled[fallback_holes] = rgb[nearest_y[fallback_holes], nearest_x[fallback_holes]]
            filled_pixels |= fallback_holes

    if feather_radius > 0 and np.any(hole):
        filled = blur_inside_mask(filled, hole, sigma=float(feather_radius))

    report = {
        "hole_pixels": int(np.count_nonzero(hole)),
        "hole_fraction": float(np.count_nonzero(hole) / max(hole.size, 1)),
        "same_tissue_filled_pixels": int(np.count_nonzero(filled_pixels & hole)),
        "fallback_filled_pixels": int(np.count_nonzero(fallback_holes)),
        "per_tissue_hole_pixels": per_label,
    }
    return filled.astype(np.uint8), report


def nearest_source_indices(source_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, indices = ndimage.distance_transform_edt(~source_mask.astype(bool), return_indices=True)
    return indices[0], indices[1]


def blur_inside_mask(rgb: np.ndarray, mask: np.ndarray, *, sigma: float) -> np.ndarray:
    rgb_float = rgb.astype(np.float32)
    blurred = ndimage.gaussian_filter(rgb_float, sigma=(sigma, sigma, 0.0))
    soft = ndimage.gaussian_filter(mask.astype(np.float32), sigma=max(sigma, 1e-3))
    soft = np.clip(soft, 0.0, 1.0)[..., None]
    blended = rgb_float * (1.0 - soft) + blurred * soft
    return np.clip(blended, 0, 255).round().astype(np.uint8)


def extract_nucleus_prototypes(
    rgb: np.ndarray,
    nuclei_mask: np.ndarray,
    *,
    context_radius: int,
    alpha_feather: float,
    min_area: int,
) -> list[NucleusPrototype]:
    rgb = np.asarray(rgb, dtype=np.uint8)
    nuclei = np.asarray(nuclei_mask).astype(np.int64, copy=False)
    prototypes: list[NucleusPrototype] = []
    structure = np.ones((3, 3), dtype=bool)
    for label in sorted(int(value) for value in np.unique(nuclei) if int(value) != 0):
        labeled, count = ndimage.label(nuclei == label, structure=structure)
        for component_id in range(1, count + 1):
            component = labeled == component_id
            area = int(np.count_nonzero(component))
            if area < min_area:
                continue
            y0, y1, x0, x1 = padded_bbox(component, context_radius, shape=nuclei.shape)
            local_mask = component[y0:y1, x0:x1]
            alpha = soft_alpha_from_mask(local_mask, sigma=alpha_feather)
            local_rgb = fill_patch_outside_mask_with_nearest(
                rgb[y0:y1, x0:x1],
                local_mask,
            )
            prototypes.append(
                NucleusPrototype(
                    label=label,
                    area=area,
                    bbox=(y0, y1, x0, x1),
                    rgb=local_rgb,
                    alpha=alpha,
                )
            )
    return prototypes


def synthesize_nuclei_layer(
    *,
    target_nuclei: np.ndarray,
    prototypes: list[NucleusPrototype],
    seed: int,
    alpha_feather: float,
    max_components: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    target = np.asarray(target_nuclei).astype(np.int64, copy=False)
    height, width = target.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rng = random.Random(seed)
    by_label: dict[int, list[NucleusPrototype]] = {}
    for prototype in prototypes:
        by_label.setdefault(int(prototype.label), []).append(prototype)
    all_prototypes = list(prototypes)

    target_components = connected_label_components(target, min_area=1)
    if max_components > 0:
        target_components = target_components[:max_components]

    pasted = 0
    missing = 0
    per_label: dict[str, int] = {}
    for component in target_components:
        candidates = by_label.get(component["label"]) or all_prototypes
        if not candidates:
            missing += 1
            continue
        prototype = choose_prototype(candidates, target_area=component["area"], rng=rng)
        paste_prototype_on_component(
            rgba,
            prototype,
            component,
            alpha_feather=alpha_feather,
        )
        pasted += 1
        per_label[str(component["label"])] = per_label.get(str(component["label"]), 0) + 1

    report = {
        "reference_prototype_count": int(len(prototypes)),
        "reference_prototype_counts_by_label": {
            str(label): int(len(items)) for label, items in sorted(by_label.items())
        },
        "target_component_count": int(len(target_components)),
        "pasted_component_count": int(pasted),
        "missing_prototype_component_count": int(missing),
        "pasted_counts_by_label": per_label,
        "alpha_coverage_fraction": float(np.count_nonzero(rgba[..., 3]) / max(height * width, 1)),
    }
    return rgba, report


def connected_label_components(mask: np.ndarray, *, min_area: int) -> list[dict[str, Any]]:
    mask = np.asarray(mask).astype(np.int64, copy=False)
    components: list[dict[str, Any]] = []
    structure = np.ones((3, 3), dtype=bool)
    for label in sorted(int(value) for value in np.unique(mask) if int(value) != 0):
        labeled, count = ndimage.label(mask == label, structure=structure)
        for component_id in range(1, count + 1):
            component = labeled == component_id
            area = int(np.count_nonzero(component))
            if area < min_area:
                continue
            y0, y1, x0, x1 = padded_bbox(component, 0, shape=mask.shape)
            components.append(
                {
                    "label": label,
                    "area": area,
                    "bbox": (y0, y1, x0, x1),
                    "mask": component[y0:y1, x0:x1].copy(),
                }
            )
    components.sort(key=lambda item: (item["bbox"][0], item["bbox"][2], item["label"]))
    return components


def choose_prototype(
    candidates: list[NucleusPrototype],
    *,
    target_area: int,
    rng: random.Random,
) -> NucleusPrototype:
    if not candidates:
        raise ValueError("cannot choose from an empty prototype list")
    closest = sorted(
        candidates,
        key=lambda proto: abs(math.log((proto.area + 1.0) / (target_area + 1.0))),
    )[: min(5, len(candidates))]
    return rng.choice(closest)


def paste_prototype_on_component(
    canvas: np.ndarray,
    prototype: NucleusPrototype,
    component: dict[str, Any],
    *,
    alpha_feather: float,
) -> None:
    y0, y1, x0, x1 = component["bbox"]
    target_h = max(1, y1 - y0)
    target_w = max(1, x1 - x0)
    proto_rgb = Image.fromarray(prototype.rgb, mode="RGB").resize(
        (target_w, target_h),
        resample=Image.Resampling.BICUBIC,
    )
    resized_rgb = np.asarray(proto_rgb, dtype=np.uint8)
    target_alpha = soft_alpha_from_mask(component["mask"], sigma=alpha_feather)
    src_alpha = target_alpha.astype(np.float32) / 255.0
    dst_alpha = canvas[y0:y1, x0:x1, 3].astype(np.float32) / 255.0
    out_alpha = src_alpha + dst_alpha * (1.0 - src_alpha)
    if np.max(out_alpha) <= 0:
        return
    dst_rgb = canvas[y0:y1, x0:x1, :3].astype(np.float32)
    src_rgb = resized_rgb.astype(np.float32)
    out_rgb = (src_rgb * src_alpha[..., None] + dst_rgb * dst_alpha[..., None] * (1.0 - src_alpha[..., None])) / np.maximum(
        out_alpha[..., None],
        1e-6,
    )
    canvas[y0:y1, x0:x1, :3] = np.clip(out_rgb, 0, 255).round().astype(np.uint8)
    canvas[y0:y1, x0:x1, 3] = np.clip(out_alpha * 255.0, 0, 255).round().astype(np.uint8)


def padded_bbox(mask: np.ndarray, pad: int, *, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0, 1, 0, 1
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(shape[1], int(xs.max()) + pad + 1)
    return y0, y1, x0, x1


def soft_alpha_from_mask(mask: np.ndarray, *, sigma: float) -> np.ndarray:
    alpha = np.asarray(mask, dtype=np.float32)
    if sigma > 0:
        alpha = ndimage.gaussian_filter(alpha, sigma=float(sigma))
        max_value = float(alpha.max())
        if max_value > 0:
            alpha = alpha / max_value
    return np.clip(alpha * 255.0, 0, 255).round().astype(np.uint8)


def fill_patch_outside_mask_with_nearest(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    patch = np.asarray(rgb, dtype=np.uint8).copy()
    if not np.any(mask):
        return patch
    nearest_y, nearest_x = nearest_source_indices(mask)
    outside = ~mask
    patch[outside] = patch[nearest_y[outside], nearest_x[outside]]
    return patch


def dilate_binary(mask: np.ndarray, *, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask
    return ndimage.binary_dilation(mask, structure=disk_structure(radius))


def disk_structure(radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (yy * yy + xx * xx) <= radius * radius


def alpha_composite_rgb(background_rgb: np.ndarray, foreground_rgba: np.ndarray) -> np.ndarray:
    bg = np.asarray(background_rgb, dtype=np.float32)
    fg = np.asarray(foreground_rgba, dtype=np.float32)
    alpha = fg[..., 3:4] / 255.0
    out = fg[..., :3] * alpha + bg * (1.0 - alpha)
    return np.clip(out, 0, 255).round().astype(np.uint8)


def rgba_on_background(rgba: np.ndarray, *, background: tuple[int, int, int]) -> Image.Image:
    bg = np.zeros(rgba.shape[:2] + (3,), dtype=np.uint8)
    bg[..., 0] = background[0]
    bg[..., 1] = background[1]
    bg[..., 2] = background[2]
    return Image.fromarray(alpha_composite_rgb(bg, rgba), mode="RGB")


def binary_mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((np.asarray(mask, dtype=bool) * 255).astype(np.uint8), mode="L").convert("RGB")


def tensor_to_int_array(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.int64, copy=False)


def image_array_to_tensor(array: np.ndarray) -> torch.Tensor:
    rgb = np.asarray(array, dtype=np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()


def parse_block_sizes(value: str | None, *, default: list[int]) -> list[int]:
    if value is None or not str(value).strip():
        return [max(1, int(v)) for v in default]
    sizes = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if raw:
            sizes.append(max(1, int(raw)))
    if not sizes:
        raise ValueError("--candidate-tissue-block-sizes did not contain any integer block sizes.")
    return sizes


def unique_with_active_first(values: list[int], *, active: int) -> list[int]:
    ordered = [int(active), *[int(value) for value in values]]
    result = []
    seen = set()
    for value in ordered:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def aggregate_sample_reports(sample_reports: list[dict[str, Any]]) -> dict[str, float]:
    if not sample_reports:
        return {}
    hole_fractions = [float(row["fill_report"]["hole_fraction"]) for row in sample_reports]
    pasted_counts = [float(row["nuclei_report"]["pasted_component_count"]) for row in sample_reports]
    prototype_counts = [float(row["nuclei_report"]["reference_prototype_count"]) for row in sample_reports]
    alpha_coverage = [float(row["nuclei_report"]["alpha_coverage_fraction"]) for row in sample_reports]
    return {
        "hole_fraction_mean": float(np.mean(hole_fractions)),
        "reference_prototype_count_mean": float(np.mean(prototype_counts)),
        "pasted_component_count_mean": float(np.mean(pasted_counts)),
        "nuclei_alpha_coverage_fraction_mean": float(np.mean(alpha_coverage)),
    }


if __name__ == "__main__":
    raise SystemExit(main())

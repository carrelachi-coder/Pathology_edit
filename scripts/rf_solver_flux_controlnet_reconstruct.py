#!/usr/bin/env python
"""Pure FLUX inversion followed by Cross V1 ControlNet denoising.

This is the reconstruction/edit gate for the Inversion+ControlNet direction:

1. Encode the target pathology image with RF-Solver-Edit's FLUX AE.
2. Run pure RF-Solver inversion with the base FLUX model to get z_T.
3. Release the RF-Solver model.
4. Denoise z_T with the trained pathology Cross V1 ControlNet, optionally
   injecting late single-block K/V from a reference RF inversion.

By default the ControlNet reference is the target image itself. When
--kv-inject is set, the K/V reference defaults to metadata reference_image.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

os.environ.setdefault(
    "SKIMAGE_DATADIR", str(Path(tempfile.gettempdir()) / "skimage-data")
)


DEFAULT_PROMPT = "hematoxylin and eosin stained pathology tissue microscopy image"
DEFAULT_PSNR_THRESHOLD = 25.0
DEFAULT_SSIM_THRESHOLD = 0.85
DEFAULT_KV_INJECT_START_STEP = 18
DEFAULT_KV_INJECT_AFTER_LAYER = 20
DEFAULT_KV_INJECT_STRENGTH = 0.2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Invert one pathology image with pure RF-Solver FLUX, then denoise "
            "the inversion noise with the pathology Cross V1 ControlNet. "
            "Optionally enables late RF-Solver reference K/V injection."
        )
    )
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--metadata-index", type=int, default=0)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument(
        "--image-field",
        choices=("target_image", "reference_image"),
        default="target_image",
    )
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--source-prompt", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument("--target-tissue-mask", type=Path, default=None)
    parser.add_argument("--target-nuclei-mask", type=Path, default=None)
    parser.add_argument("--reference-image", type=Path, default=None)
    parser.add_argument("--reference-tissue-mask", type=Path, default=None)
    parser.add_argument("--reference-nuclei-mask", type=Path, default=None)
    parser.add_argument(
        "--kv-reference-image",
        type=Path,
        default=None,
        help=(
            "Reference image whose RF-Solver single-block K/V features are "
            "injected during ControlNet denoise. Defaults to metadata "
            "reference_image when --kv-inject is set."
        ),
    )
    parser.add_argument(
        "--kv-reference-tissue-mask",
        type=Path,
        default=None,
        help=(
            "Tissue mask for --kv-reference-image when --regional-mode is enabled. "
            "Defaults to metadata reference_tissue_mask for metadata K/V refs, or "
            "the target tissue mask when the K/V ref is the target image."
        ),
    )
    parser.add_argument(
        "--kv-reference-nuclei-mask",
        type=Path,
        default=None,
        help=(
            "Nuclei mask for --kv-reference-image when --regional-mode is enabled. "
            "Defaults to metadata reference_nuclei_mask for metadata K/V refs, or "
            "the target nuclei mask when the K/V ref is the target image."
        ),
    )
    parser.add_argument(
        "--kv-reference-prompt",
        default=None,
        help="Prompt used for the K/V reference inversion; defaults to source prompt.",
    )
    parser.add_argument(
        "--controlnet-reference-source",
        choices=("self", "metadata", "explicit"),
        default="self",
        help=(
            "Reference image for the ControlNet/IP path. self uses the target "
            "image itself, metadata uses the record reference_image, explicit "
            "requires --reference-image and matching reference masks."
        ),
    )

    parser.add_argument(
        "--pretrained-model-name-or-path",
        default=os.environ.get("MODEL_DIR") or os.environ.get("FLUX_DIFFUSERS_ROOT"),
        help="Local diffusers FLUX.1-dev directory for the ControlNet pipeline.",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("CONTROLNET_CHECKPOINT")
        or os.environ.get("CONTROLNET_CKPT"),
        help="Cross V1 ControlNet checkpoint directory.",
    )
    parser.add_argument(
        "--uni-checkpoint-path",
        default=os.environ.get("UNI_CHECKPOINT") or os.environ.get("UNI_CKPT"),
        help="UNI-2h pytorch_model.bin path used by the reference encoder.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument(
        "--with-second-order",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use RF-Solver second-order updates for inversion and ControlNet denoise.",
    )
    parser.add_argument("--rf-inversion-guidance", type=float, default=1.0)
    parser.add_argument("--controlnet-guidance-scale", type=float, default=1.0)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=0.6)
    parser.add_argument(
        "--controlnet-start-step",
        type=int,
        default=0,
        help=(
            "Disable ControlNet residuals and Cross V1 IP tokens for the first "
            "N denoise steps. Set this to --num-inference-steps for a pure "
            "diffusers-FLUX denoise compatibility check from RF zT."
        ),
    )
    parser.add_argument(
        "--ip-scale",
        type=float,
        default=1.0,
        help=(
            "Cross V1 reference IP scale. For the strictest no-reference ablation "
            "set this to 0; default self-reference keeps target appearance."
        ),
    )
    parser.add_argument("--regional-ip-soft-bias", type=float, default=None)
    parser.add_argument(
        "--regional-mode",
        choices=("none", "tissue", "nuclei", "tissue_nuclei"),
        default="none",
        help=(
            "Token-level regional attention mask for late K/V injection. "
            "Only affects --kv-inject; default keeps the old global K/V path."
        ),
    )
    parser.add_argument(
        "--kv-protect-target-nuclei",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use high-resolution nuclei occupancy pooling to reduce late K/V "
            "injection on target tokens that contain nuclei."
        ),
    )
    parser.add_argument(
        "--kv-target-nuclei-inject-scale",
        type=float,
        default=0.0,
        help=(
            "Per-query K/V injection scale for target nuclei-present tokens when "
            "--kv-protect-target-nuclei is enabled. 0 preserves native target "
            "attention for those tokens; 1 disables the protection."
        ),
    )
    parser.add_argument(
        "--kv-block-ref-nuclei-to-target-non-nuclei",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use high-resolution nuclei occupancy pooling to prevent target "
            "non-nuclei tokens from attending to reference nuclei-present tokens "
            "during late K/V injection."
        ),
    )
    parser.add_argument(
        "--kv-nuclei-occupancy-dilate-px",
        type=int,
        default=4,
        help="Dilate nuclei masks by this many image pixels before token occupancy pooling.",
    )
    parser.add_argument(
        "--kv-nuclei-occupancy-min-pixels",
        type=int,
        default=3,
        help="Mark a token nuclei-present if its pooled cell contains at least this many nuclei pixels.",
    )
    parser.add_argument(
        "--kv-nuclei-occupancy-min-fraction",
        type=float,
        default=0.01,
        help="Mark a token nuclei-present if this fraction of its pooled cell is nuclei.",
    )
    parser.add_argument(
        "--kv-inject",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable late RF-Solver reference K/V injection inside diffusers "
            "FLUX single blocks during the ControlNet denoise loop."
        ),
    )
    parser.add_argument(
        "--kv-inject-mode",
        choices=("v-only", "kv"),
        default="kv",
        help="Use Q_i0+K_i0+V_ref or Q_i0+K_ref+V_ref for image tokens.",
    )
    parser.add_argument(
        "--kv-inject-start-step",
        type=int,
        default=DEFAULT_KV_INJECT_START_STEP,
        help="First denoise step index where K/V injection is allowed.",
    )
    parser.add_argument(
        "--kv-inject-after-layer",
        type=int,
        default=DEFAULT_KV_INJECT_AFTER_LAYER,
        help="Inject only in single_transformer_blocks with index >= this value.",
    )
    parser.add_argument(
        "--kv-inject-strength",
        type=float,
        default=DEFAULT_KV_INJECT_STRENGTH,
        help="Linear blend strength from current K/V toward reference K/V.",
    )
    parser.add_argument(
        "--kv-inject-after-t",
        type=float,
        default=None,
        help=(
            "Optional RF/FLUX timestep threshold. When set, injection/capture "
            "only happens for t <= this value; otherwise it is derived from "
            "--kv-inject-start-step."
        ),
    )
    parser.add_argument(
        "--kv-save-feature-debug",
        action="store_true",
        help="Write RF K/V capture and diffusers injection event samples to JSON.",
    )

    parser.add_argument("--rf-solver-root", type=Path, default=_env_path("RF_SOLVER_ROOT"))
    parser.add_argument("--name", default="flux-dev")
    parser.add_argument("--rf-offload", action="store_true")
    parser.add_argument(
        "--save-rf-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also save a pure RF-Solver reconstruction from the same inversion zT.",
    )
    parser.add_argument("--rf-baseline-guidance", type=float, default=1.0)
    parser.add_argument("--flux-diffusers-root", type=Path, default=_env_path("FLUX_DIFFUSERS_ROOT"))
    parser.add_argument("--t5-model-path", type=Path, default=_env_path("T5_MODEL_PATH"))
    parser.add_argument("--t5-tokenizer-path", type=Path, default=_env_path("T5_TOKENIZER_PATH"))
    parser.add_argument("--clip-model-path", type=Path, default=_env_path("CLIP_MODEL_PATH"))
    parser.add_argument("--clip-tokenizer-path", type=Path, default=_env_path("CLIP_TOKENIZER_PATH"))
    parser.add_argument("--allow-text-encoder-download", action="store_true")

    parser.add_argument("--psnr-threshold", type=float, default=DEFAULT_PSNR_THRESHOLD)
    parser.add_argument("--ssim-threshold", type=float, default=DEFAULT_SSIM_THRESHOLD)
    parser.add_argument("--fail-on-threshold", action="store_true")
    return parser


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    return run(args)


def validate_args(args: argparse.Namespace) -> None:
    missing = []
    if args.pretrained_model_name_or_path is None:
        missing.append("--pretrained-model-name-or-path or FLUX_DIFFUSERS_ROOT")
    if args.checkpoint is None:
        missing.append("--checkpoint or CONTROLNET_CHECKPOINT/CONTROLNET_CKPT")
    if args.uni_checkpoint_path is None:
        missing.append("--uni-checkpoint-path or UNI_CHECKPOINT/UNI_CKPT")
    if missing:
        raise ValueError("Missing required ControlNet inputs: " + ", ".join(missing))
    if args.num_inference_steps <= 0:
        raise ValueError("--num-inference-steps must be positive.")
    if args.controlnet_start_step < 0:
        raise ValueError("--controlnet-start-step must be >= 0.")
    if args.kv_inject_start_step < 0:
        raise ValueError("--kv-inject-start-step must be >= 0.")
    if args.kv_inject_strength < 0.0 or args.kv_inject_strength > 1.0:
        raise ValueError("--kv-inject-strength must be in [0, 1].")
    if args.kv_inject_after_layer < 0:
        raise ValueError("--kv-inject-after-layer must be >= 0.")
    if args.regional_mode != "none" and not args.kv_inject:
        raise ValueError("--regional-mode requires --kv-inject.")
    if args.kv_protect_target_nuclei and not args.kv_inject:
        raise ValueError("--kv-protect-target-nuclei requires --kv-inject.")
    if args.kv_block_ref_nuclei_to_target_non_nuclei and not args.kv_inject:
        raise ValueError("--kv-block-ref-nuclei-to-target-non-nuclei requires --kv-inject.")
    if args.kv_target_nuclei_inject_scale < 0.0 or args.kv_target_nuclei_inject_scale > 1.0:
        raise ValueError("--kv-target-nuclei-inject-scale must be in [0, 1].")
    if args.kv_nuclei_occupancy_dilate_px < 0:
        raise ValueError("--kv-nuclei-occupancy-dilate-px must be >= 0.")
    if args.kv_nuclei_occupancy_min_pixels < 0:
        raise ValueError("--kv-nuclei-occupancy-min-pixels must be >= 0.")
    if args.kv_nuclei_occupancy_min_fraction < 0.0 or args.kv_nuclei_occupancy_min_fraction > 1.0:
        raise ValueError("--kv-nuclei-occupancy-min-fraction must be in [0, 1].")
    if (
        (args.kv_protect_target_nuclei or args.kv_block_ref_nuclei_to_target_non_nuclei)
        and args.kv_nuclei_occupancy_min_pixels <= 0
        and args.kv_nuclei_occupancy_min_fraction <= 0.0
    ):
        raise ValueError(
            "At least one nuclei occupancy threshold must be positive: "
            "--kv-nuclei-occupancy-min-pixels or --kv-nuclei-occupancy-min-fraction."
        )
    if args.image is None and args.metadata is None:
        raise ValueError("--image or --metadata is required.")
    if args.controlnet_reference_source == "explicit" and args.reference_image is None:
        raise ValueError("--controlnet-reference-source explicit requires --reference-image.")


def run(args: argparse.Namespace) -> int:
    started_at = time.perf_counter()
    record = select_record(args.metadata, args.sample_id, args.metadata_index)
    resolved = resolve_inputs(args, record)
    output_dir: Path = resolved["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    original, crop_info = crop_to_multiple(load_rgb_image(resolved["image_path"]))
    width, height = original.size
    original_path = output_dir / "original_cropped.png"
    original.save(original_path)

    target_tissue_mask = load_mask_image(resolved["target_tissue_mask"], crop_size=original.size)
    target_nuclei_mask = load_mask_image(
        resolved["target_nuclei_mask"],
        crop_size=original.size,
        remap_nuclei=True,
    )

    reference_image, reference_tissue_mask, reference_nuclei_mask = load_reference_condition(
        resolved=resolved,
        original=original,
        target_tissue_mask=target_tissue_mask,
        target_nuclei_mask=target_nuclei_mask,
    )
    reference_path = output_dir / "controlnet_reference.png"
    reference_image.save(reference_path)
    kv_reference_image = None
    kv_reference_path = None
    kv_reference_tissue_mask = None
    kv_reference_nuclei_mask = None
    if args.kv_inject:
        kv_reference_source = resolved["kv_reference_image"]
        if kv_reference_source is None:
            raise ValueError(
                "--kv-inject needs a K/V reference image. Use metadata with "
                "reference_image or pass --kv-reference-image."
            )
        kv_reference_image = crop_to_multiple(load_rgb_image(kv_reference_source))[0]
        if kv_reference_image.size != original.size:
            raise ValueError(
                "K/V reference image must crop to the same size as the target "
                f"for token alignment. target={original.size} reference={kv_reference_image.size}"
            )
        kv_reference_path = output_dir / "kv_reference.png"
        kv_reference_image.save(kv_reference_path)
        if args.regional_mode in {"tissue", "tissue_nuclei"}:
            kv_reference_tissue_mask = load_mask_image(
                resolved["kv_reference_tissue_mask"],
                crop_size=kv_reference_image.size,
            )
        if args.regional_mode in {"nuclei", "tissue_nuclei"} or args.kv_block_ref_nuclei_to_target_non_nuclei:
            kv_reference_nuclei_mask = load_mask_image(
                resolved["kv_reference_nuclei_mask"],
                crop_size=kv_reference_image.size,
                remap_nuclei=True,
            )

    prompt = resolved["source_prompt"]
    print(
        "Stage 1: pure RF-Solver FLUX inversion "
        f"({args.num_inference_steps} steps, second_order={args.with_second_order})"
    )
    inversion = run_rf_solver_inversion(
        args=args,
        image=original,
        prompt=prompt,
        output_dir=output_dir,
        kv_reference_image=kv_reference_image,
        kv_reference_prompt=resolved["kv_reference_prompt"],
    )
    print(
        "RF inversion complete | packed_shape={} | image_tokens={}".format(
            inversion["noise_shape"],
            inversion["image_seq_len"],
        )
    )
    kv_reference_features = inversion.get("kv_reference_features")
    kv_regional_artifacts: dict[str, str] = {}
    if kv_reference_features is not None and needs_kv_regional_payload(args):
        regional_payload, kv_regional_artifacts = build_kv_regional_label_payload(
            args=args,
            resolved=resolved,
            output_dir=output_dir,
            num_image_tokens=int(inversion["image_seq_len"]),
            target_size=original.size,
            reference_size=kv_reference_image.size if kv_reference_image is not None else original.size,
        )
        kv_reference_features = dict(kv_reference_features)
        kv_reference_features["regional_labels"] = regional_payload

    print(
        "Stage 2: ControlNet denoise from RF inversion noise "
        f"({'with' if args.kv_inject else 'no'} K/V injection)"
    )
    reconstruction, controlnet_denoise_debug = run_controlnet_denoise(
        args=args,
        initial_packed_latents=inversion["noise"],
        timesteps=inversion["timesteps"],
        prompt=prompt,
        output_size=(height, width),
        reference_image=pil_to_float_tensor(reference_image),
        reference_tissue_mask=mask_to_long_tensor(reference_tissue_mask),
        reference_nuclei_mask=mask_to_long_tensor(reference_nuclei_mask),
        target_tissue_mask=mask_to_long_tensor(target_tissue_mask),
        target_nuclei_mask=mask_to_long_tensor(target_nuclei_mask),
        kv_reference_features=kv_reference_features,
    )

    reconstruction_path = output_dir / "controlnet_reconstruction.png"
    reconstruction.save(reconstruction_path)
    kv_reconstruction_path = None
    if args.kv_inject:
        kv_reconstruction_path = output_dir / "controlnet_kv_reconstruction.png"
        reconstruction.save(kv_reconstruction_path)
    diff_image = make_diff_image(original, reconstruction)
    diff_path = output_dir / "diff.png"
    diff_image.save(diff_path)
    panels = [
        ("original_cropped", original),
        ("controlnet_reference", reference_image),
    ]
    if kv_reference_image is not None:
        panels.append(("kv_reference", kv_reference_image))
    panels.extend(
        [
            ("cn_denoise_reconstruction", reconstruction),
            ("absolute_diff", diff_image),
        ]
    )
    comparison = make_comparison_image(panels, columns=min(5, len(panels)))
    comparison_path = output_dir / "comparison.png"
    comparison.save(comparison_path)

    metrics = image_metrics(original, reconstruction)
    psnr_value = metric_value(metrics["psnr"])
    ssim_value = float(metrics["ssim"])
    gate_pass = psnr_value > args.psnr_threshold or ssim_value > args.ssim_threshold
    metrics.update(
        {
            "gate": "green_light" if gate_pass else "red_light",
            "mode": (
                "rf_inversion_controlnet_reconstruction_with_late_ref_kv_injection"
                if args.kv_inject
                else "rf_inversion_controlnet_reconstruction_no_kv_injection"
            ),
            "kv_inject": bool(args.kv_inject),
            "kv_inject_mode": args.kv_inject_mode,
            "kv_inject_start_step": args.kv_inject_start_step,
            "kv_inject_after_layer": args.kv_inject_after_layer,
            "kv_inject_strength": args.kv_inject_strength,
            "regional_mode": args.regional_mode,
            "kv_protect_target_nuclei": bool(args.kv_protect_target_nuclei),
            "kv_target_nuclei_inject_scale": args.kv_target_nuclei_inject_scale,
            "kv_block_ref_nuclei_to_target_non_nuclei": bool(
                args.kv_block_ref_nuclei_to_target_non_nuclei
            ),
            "kv_nuclei_occupancy_dilate_px": args.kv_nuclei_occupancy_dilate_px,
            "kv_nuclei_occupancy_min_pixels": args.kv_nuclei_occupancy_min_pixels,
            "kv_nuclei_occupancy_min_fraction": args.kv_nuclei_occupancy_min_fraction,
            "kv_reference_image": str(resolved["kv_reference_image"]) if resolved["kv_reference_image"] is not None else None,
            "kv_reference_tissue_mask": (
                str(resolved["kv_reference_tissue_mask"])
                if resolved.get("kv_reference_tissue_mask") is not None
                else None
            ),
            "kv_reference_nuclei_mask": (
                str(resolved["kv_reference_nuclei_mask"])
                if resolved.get("kv_reference_nuclei_mask") is not None
                else None
            ),
            "kv_reference_prompt": resolved["kv_reference_prompt"],
            "num_inference_steps": args.num_inference_steps,
            "with_second_order": args.with_second_order,
            "rf_inversion_guidance": args.rf_inversion_guidance,
            "controlnet_guidance_scale": args.controlnet_guidance_scale,
            "controlnet_conditioning_scale": args.controlnet_conditioning_scale,
            "controlnet_start_step": args.controlnet_start_step,
            "ip_scale": args.ip_scale,
            "controlnet_reference_source": resolved["reference_source"],
            "source_prompt": prompt,
            "prompt_source": resolved["prompt_source"],
            "input_image": str(resolved["image_path"]),
            "metadata_path": str(args.metadata) if args.metadata is not None else None,
            "metadata_index": args.metadata_index if args.metadata is not None else None,
            "metadata_sample_id": args.sample_id,
            "metadata_image_field": args.image_field if args.metadata is not None else None,
            "metadata_record": metadata_summary(record),
            "crop": crop_info,
            "rf_inversion": {
                "name": args.name,
                "noise_shape": inversion["noise_shape"],
                "image_seq_len": inversion["image_seq_len"],
                "latent_shape": inversion["latent_shape"],
                "noise_summary": inversion["noise_summary"],
                "baseline": inversion["baseline"],
                "timesteps_head": inversion["timesteps"][:5],
                "timesteps_tail": inversion["timesteps"][-5:],
                "local_weight_paths": inversion["local_weight_paths"],
                "text_encoder_paths": inversion["text_encoder_paths"],
                "kv_reference_features": inversion["kv_reference_summary"],
            },
            "controlnet": {
                "pretrained_model_name_or_path": str(args.pretrained_model_name_or_path),
                "checkpoint": str(args.checkpoint),
                "uni_checkpoint_path": str(args.uni_checkpoint_path),
                "torch_dtype": args.torch_dtype,
                "regional_ip_soft_bias": args.regional_ip_soft_bias,
                "start_step": args.controlnet_start_step,
                "denoise_debug": controlnet_denoise_debug,
            },
            "artifacts": {
                "original_cropped": str(original_path),
                "controlnet_reference": str(reference_path),
                "kv_reference": str(kv_reference_path) if kv_reference_path is not None else None,
                "reconstruction": str(reconstruction_path),
                "kv_reconstruction": str(kv_reconstruction_path) if kv_reconstruction_path is not None else None,
                "diff": str(diff_path),
                "comparison": str(comparison_path),
                "kv_regional_token_masks": kv_regional_artifacts,
            },
            "psnr_threshold": args.psnr_threshold,
            "ssim_threshold": args.ssim_threshold,
            "runtime_seconds": round(time.perf_counter() - started_at, 3),
            "notes": (
                "This gate tests whether pure FLUX inversion noise can be "
                "denoised back onto the pathology manifold by the trained Cross "
                "V1 ControlNet. When kv_inject=true, late single-block image "
                "token K/V are blended from the RF-Solver reference inversion."
            ),
        }
    )
    metrics_path = output_dir / "metrics.json"
    write_json(metrics_path, metrics)
    if args.kv_save_feature_debug and args.kv_inject:
        write_json(
            output_dir / "kv_injection_debug.json",
            {
                "rf_reference": inversion["kv_reference_summary"],
                "diffusers_denoise": controlnet_denoise_debug.get("kv_injection"),
            },
        )
    print(
        "Saved ControlNet reconstruction to {} | kv_inject={} | gate={} | PSNR={} | SSIM={:.6f}".format(
            output_dir,
            bool(args.kv_inject),
            metrics["gate"],
            metrics["psnr"],
            metrics["ssim"],
        )
    )
    if not gate_pass and args.fail_on_threshold:
        return 2
    return 0


def run_rf_solver_inversion(
    *,
    args: argparse.Namespace,
    image: Image.Image,
    prompt: str,
    output_dir: Path,
    kv_reference_image: Image.Image | None = None,
    kv_reference_prompt: str | None = None,
) -> dict[str, Any]:
    import torch

    rf_helpers = import_rf_helpers()
    rf_helpers["prepend_rf_solver_src"](args.rf_solver_root)
    rf = rf_helpers["import_rf_solver_modules"]()
    if args.kv_inject:
        rf_helpers["install_cross_image_forward_patch"](rf["SingleStreamBlock"])
    if args.name not in rf["configs"]:
        available = ", ".join(sorted(rf["configs"].keys()))
        raise ValueError(f"Unknown --name {args.name!r}; available names: {available}")

    torch.set_grad_enabled(False)
    torch_device = torch.device(args.device)
    if torch_device.type == "cuda" and torch_device.index is not None:
        torch.cuda.set_device(torch_device.index)

    local_weight_paths = rf_helpers["configure_local_flux_weight_paths"](rf, args)
    print(f"Loading RF-Solver components for {args.name} on {torch_device}")
    print(f"Local FLUX weight paths: {json.dumps(local_weight_paths, sort_keys=True)}")
    ae = rf["load_ae"](args.name, device="cpu" if args.rf_offload else torch_device)
    if args.rf_offload:
        ae.encoder.to(torch_device)
    latents = rf_helpers["encode_image"](image, torch_device, ae)
    reference_latents = None
    if args.kv_inject:
        if kv_reference_image is None:
            raise ValueError("Internal error: --kv-inject was set but kv_reference_image is missing.")
        reference_latents = rf_helpers["encode_image"](kv_reference_image, torch_device, ae)
    latent_shape = list(latents.shape)
    if args.rf_offload:
        ae.cpu()
        maybe_empty_cuda_cache(torch_device)

    max_text_length = 256 if args.name == "flux-schnell" else 512
    text_encoder_paths = rf_helpers["resolve_text_encoder_paths"](args)
    if rf_helpers["validate_text_encoder_paths"](text_encoder_paths):
        t5, clip = rf_helpers["load_local_text_encoders"](
            device=torch_device,
            max_length=max_text_length,
            paths=text_encoder_paths,
        )
    elif args.allow_text_encoder_download:
        print(
            "Local FLUX text encoder paths were not found; falling back to "
            "RF-Solver-Edit's downloader because --allow-text-encoder-download is set."
        )
        t5 = rf["load_t5"](torch_device, max_length=max_text_length)
        clip = rf["load_clip"](torch_device)
    else:
        raise FileNotFoundError(
            "Local text encoder paths were not found and downloads are disabled. "
            "Set FLUX_DIFFUSERS_ROOT to the local FLUX.1-dev diffusers directory, "
            "or pass --t5-model-path --t5-tokenizer-path --clip-model-path "
            "--clip-tokenizer-path."
        )

    prepared = rf["prepare"](t5, clip, latents, prompt=prompt)
    prepared_reference = None
    if args.kv_inject:
        prepared_reference = rf["prepare"](
            t5,
            clip,
            reference_latents,
            prompt=kv_reference_prompt or prompt,
        )
        if int(prepared_reference["img"].shape[1]) != int(prepared["img"].shape[1]):
            raise ValueError(
                "Target/reference packed image token counts differ; K/V injection "
                f"requires equal lengths. target={int(prepared['img'].shape[1])} "
                f"reference={int(prepared_reference['img'].shape[1])}."
            )
    timesteps = rf["get_schedule"](
        args.num_inference_steps,
        prepared["img"].shape[1],
        shift=(args.name != "flux-schnell"),
    )
    image_seq_len = int(prepared["img"].shape[1])

    if args.rf_offload:
        t5.cpu()
        clip.cpu()
        maybe_empty_cuda_cache(torch_device)

    model = rf["load_flow_model"](
        args.name,
        device="cpu" if args.rf_offload else torch_device,
    )
    if args.rf_offload:
        model.to(torch_device)

    denoise_fn = rf["denoise"] if args.with_second_order else rf_helpers["denoise_first_order"]
    feature_bank = None
    kv_reference_summary: dict[str, Any] = {"enabled": bool(args.kv_inject)}
    kv_reference_features = None
    requested_inject_steps = kv_inject_steps_for_capture(args)
    capture_inject_steps = kv_capture_inject_steps(args)
    if args.kv_inject:
        feature_bank = RFReferenceKVFeatureBank(
            after_layer=args.kv_inject_after_layer,
            image_token_count=image_seq_len,
            inject_after_t=resolve_kv_inject_after_t(args, timesteps),
        )
    info = {
        "feature_path": str(output_dir / "rf_inversion_features_unused"),
        "feature": {},
        "inject_step": 0,
    }
    noise, _ = denoise_fn(
        model,
        **prepared,
        timesteps=timesteps,
        guidance=args.rf_inversion_guidance,
        inverse=True,
        info=info,
    )
    noise_cpu = noise.detach().to("cpu")
    noise_shape = list(noise_cpu.shape)
    noise_summary = tensor_debug_summary(noise_cpu)
    print(
        "RF inversion zT summary: "
        f"shape={noise_summary['shape']} mean={noise_summary['mean']:.6f} "
        f"std={noise_summary['std']:.6f}"
    )

    if args.kv_inject:
        print(
            "Running RF-Solver reference inversion to capture single-block K/V "
            f"(after_layer={args.kv_inject_after_layer}, "
            f"capture_steps={capture_inject_steps}, active_late_steps={requested_inject_steps})..."
        )
        if feature_bank is None or prepared_reference is None:
            raise RuntimeError("K/V feature bank was not initialized.")
        feature_bank.phase = "reference_inversion"
        _reference_noise, _ = denoise_fn(
            model,
            **prepared_reference,
            timesteps=timesteps,
            guidance=args.rf_inversion_guidance,
            inverse=True,
            info={
                "feature_path": str(output_dir / "rf_reference_kv_features"),
                "feature": {},
                "inject_step": capture_inject_steps,
                "_cross_bank": feature_bank,
            },
        )
        del _reference_noise
        kv_reference_features = build_diffusers_kv_reference_features(
            feature_bank=feature_bank,
            image_token_count=image_seq_len,
        )
        if int(kv_reference_features.get("feature_count", 0)) <= 0:
            raise RuntimeError(
                "No RF reference K/V features were captured. Try lowering "
                "--kv-inject-start-step, setting --kv-inject-after-t explicitly, "
                "or checking RF-Solver single-block feature hooks."
            )
        kv_reference_summary = {
            **kv_reference_features["summary"],
            "enabled": True,
            "prompt": kv_reference_prompt or prompt,
            "requested_active_steps": requested_inject_steps,
            "capture_inject_steps": capture_inject_steps,
            "inject_after_t": resolve_kv_inject_after_t(args, timesteps),
            "mode": args.kv_inject_mode,
            "strength": args.kv_inject_strength,
        }
        if args.kv_save_feature_debug:
            (output_dir / "rf_reference_kv_feature_events.json").write_text(
                json.dumps(feature_bank.events, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(
            "Captured reference K/V features | count={} | blocks={}".format(
                kv_reference_summary.get("feature_count"),
                kv_reference_summary.get("block_ids"),
            )
        )

    baseline_summary: dict[str, Any] = {"enabled": False}
    if args.save_rf_baseline:
        print("Running pure RF-Solver baseline reconstruction from the same zT...")
        baseline_info = {
            "feature_path": str(output_dir / "rf_baseline_features_unused"),
            "feature": {},
            "inject_step": 0,
        }
        baseline_inputs = dict(prepared)
        baseline_inputs["img"] = noise
        rf_reconstruction, _ = denoise_fn(
            model,
            **baseline_inputs,
            timesteps=timesteps,
            guidance=args.rf_baseline_guidance,
            inverse=False,
            info=baseline_info,
        )
        if args.rf_offload:
            model.cpu()
            maybe_empty_cuda_cache(torch_device)
            ae.decoder.to(torch_device)
        baseline_image = rf_helpers["decode_image"](
            rf_reconstruction,
            height=image.height,
            width=image.width,
            ae=ae,
            unpack=rf["unpack"],
            device=torch_device,
        )
        baseline_path = output_dir / "rf_baseline_reconstruction.png"
        baseline_image.save(baseline_path)
        baseline_diff_path = output_dir / "rf_baseline_diff.png"
        make_diff_image(image, baseline_image).save(baseline_diff_path)
        baseline_metrics = image_metrics(image, baseline_image)
        baseline_summary = {
            "enabled": True,
            "guidance": args.rf_baseline_guidance,
            "artifacts": {
                "reconstruction": str(baseline_path),
                "diff": str(baseline_diff_path),
            },
            "metrics": baseline_metrics,
        }
        print(
            "Pure RF baseline | PSNR={} | SSIM={:.6f}".format(
                baseline_metrics["psnr"],
                baseline_metrics["ssim"],
            )
        )
        del rf_reconstruction, baseline_inputs

    del noise, prepared, prepared_reference, model, t5, clip, ae, latents, reference_latents
    maybe_empty_cuda_cache(torch_device)
    gc.collect()
    return {
        "noise": noise_cpu,
        "noise_shape": noise_shape,
        "noise_summary": noise_summary,
        "baseline": baseline_summary,
        "latent_shape": latent_shape,
        "image_seq_len": image_seq_len,
        "timesteps": [float(t) for t in timesteps],
        "local_weight_paths": local_weight_paths,
        "text_encoder_paths": {
            key: str(value) if value is not None else None
            for key, value in text_encoder_paths.items()
        },
        "kv_reference_features": kv_reference_features,
        "kv_reference_summary": kv_reference_summary,
    }


def run_controlnet_denoise(
    *,
    args: argparse.Namespace,
    initial_packed_latents: Any,
    timesteps: list[float],
    prompt: str,
    output_size: tuple[int, int],
    reference_image: Any,
    reference_tissue_mask: Any,
    reference_nuclei_mask: Any,
    target_tissue_mask: Any,
    target_nuclei_mask: Any,
    kv_reference_features: dict[str, Any] | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    import torch

    from controlnet_train.inference.pipeline_cross_v1 import (
        load_cross_v1_bundle,
        run_cross_v1_controlnet_denoise_from_packed_latents,
        set_ip_soft_bias,
    )

    dtype_by_name = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    bundle = load_cross_v1_bundle(
        pretrained_model_name_or_path=str(args.pretrained_model_name_or_path),
        checkpoint_path=str(args.checkpoint),
        uni_checkpoint_path=str(args.uni_checkpoint_path),
        device=args.device,
        torch_dtype=dtype_by_name[args.torch_dtype],
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.controlnet_guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        ip_adapter_scale=args.ip_scale,
    )
    if args.regional_ip_soft_bias is not None:
        summary = set_ip_soft_bias(bundle.flux_pipeline.transformer, args.regional_ip_soft_bias)
        print(f"Applied regional IP soft-bias override: {json.dumps(summary, sort_keys=True)}")

    result = run_cross_v1_controlnet_denoise_from_packed_latents(
        bundle,
        initial_packed_latents=initial_packed_latents,
        reference_image=reference_image,
        reference_tissue_mask=reference_tissue_mask,
        reference_nuclei_mask=reference_nuclei_mask,
        target_tissue_mask=target_tissue_mask,
        target_nuclei_mask=target_nuclei_mask,
        prompt=prompt,
        output_size=output_size,
        timesteps=timesteps,
        guidance_scale=args.controlnet_guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        with_second_order=args.with_second_order,
        controlnet_start_step=args.controlnet_start_step,
        kv_reference_features=kv_reference_features,
        kv_inject_mode=args.kv_inject_mode,
        kv_inject_strength=args.kv_inject_strength,
        kv_inject_start_step=args.kv_inject_start_step,
        kv_inject_after_layer=args.kv_inject_after_layer,
        kv_inject_after_t=args.kv_inject_after_t,
        seed=args.seed,
        return_debug=True,
    )
    if not isinstance(result, tuple):
        return result, {}
    return result


def build_kv_regional_label_payload(
    *,
    args: argparse.Namespace,
    resolved: dict[str, Any],
    output_dir: Path,
    num_image_tokens: int,
    target_size: tuple[int, int],
    reference_size: tuple[int, int],
) -> tuple[dict[str, Any], dict[str, str]]:
    artifacts: dict[str, str] = {}
    payload: dict[str, Any] = {"mode": str(args.regional_mode)}
    if args.regional_mode != "none":
        rf_helpers = import_rf_helpers()
        regional_labels = rf_helpers["build_regional_token_labels"](
            mode=args.regional_mode,
            target_tissue_mask=resolved["target_tissue_mask"],
            reference_tissue_mask=resolved["kv_reference_tissue_mask"],
            target_nuclei_mask=resolved["target_nuclei_mask"],
            reference_nuclei_mask=resolved["kv_reference_nuclei_mask"],
            num_image_tokens=int(num_image_tokens),
            target_size=target_size,
            reference_size=reference_size,
        )
        artifacts.update(
            rf_helpers["save_regional_token_overlays"](
                regional_labels,
                output_dir / "kv_regional",
            )
        )
        if regional_labels is not None:
            payload.update(
                {
                    "mode": regional_labels.mode,
                    "target_tissue": regional_labels.target_tissue,
                    "reference_tissue": regional_labels.reference_tissue,
                    "target_nuclei": regional_labels.target_nuclei,
                    "reference_nuclei": regional_labels.reference_nuclei,
                    "target_composite": regional_labels.target_composite,
                    "reference_composite": regional_labels.reference_composite,
                }
            )

    if args.kv_protect_target_nuclei or args.kv_block_ref_nuclei_to_target_non_nuclei:
        target_present, target_stats = build_nuclei_occupancy_tokens(
            resolved["target_nuclei_mask"],
            crop_size=target_size,
            num_image_tokens=int(num_image_tokens),
            dilate_px=int(args.kv_nuclei_occupancy_dilate_px),
            min_pixels=int(args.kv_nuclei_occupancy_min_pixels),
            min_fraction=float(args.kv_nuclei_occupancy_min_fraction),
        )
        payload["target_nuclei_present"] = target_present
        occupancy_summary: dict[str, Any] = {
            "target": target_stats,
            "dilate_px": int(args.kv_nuclei_occupancy_dilate_px),
            "min_pixels": int(args.kv_nuclei_occupancy_min_pixels),
            "min_fraction": float(args.kv_nuclei_occupancy_min_fraction),
        }
        artifacts["target_nuclei_present"] = save_bool_token_grid(
            target_present,
            output_dir / "kv_regional" / "token_masks" / "target_nuclei_present_occupancy_32x32.png",
        )
        if args.kv_block_ref_nuclei_to_target_non_nuclei:
            reference_present, reference_stats = build_nuclei_occupancy_tokens(
                resolved["kv_reference_nuclei_mask"],
                crop_size=reference_size,
                num_image_tokens=int(num_image_tokens),
                dilate_px=int(args.kv_nuclei_occupancy_dilate_px),
                min_pixels=int(args.kv_nuclei_occupancy_min_pixels),
                min_fraction=float(args.kv_nuclei_occupancy_min_fraction),
            )
            payload["reference_nuclei_present"] = reference_present
            occupancy_summary["reference"] = reference_stats
            artifacts["reference_nuclei_present"] = save_bool_token_grid(
                reference_present,
                output_dir / "kv_regional" / "token_masks" / "reference_nuclei_present_occupancy_32x32.png",
            )
        payload.update(
            {
                "protect_target_nuclei": bool(args.kv_protect_target_nuclei),
                "target_nuclei_inject_scale": float(args.kv_target_nuclei_inject_scale),
                "block_ref_nuclei_to_target_non_nuclei": bool(
                    args.kv_block_ref_nuclei_to_target_non_nuclei
                ),
                "nuclei_occupancy": occupancy_summary,
            }
        )

    artifacts.update(
        save_kv_mask_correspondence_panel(
            payload=payload,
            resolved=resolved,
            output_dir=output_dir,
            target_size=target_size,
            reference_size=reference_size,
        )
    )
    return payload, artifacts


def needs_kv_regional_payload(args: argparse.Namespace) -> bool:
    return (
        args.regional_mode != "none"
        or bool(args.kv_protect_target_nuclei)
        or bool(args.kv_block_ref_nuclei_to_target_non_nuclei)
    )


def build_nuclei_occupancy_tokens(
    mask_path: Path,
    *,
    crop_size: tuple[int, int],
    num_image_tokens: int,
    dilate_px: int,
    min_pixels: int,
    min_fraction: float,
) -> tuple[Any, dict[str, Any]]:
    import torch

    side = int(round(float(num_image_tokens) ** 0.5))
    if side * side != int(num_image_tokens):
        raise ValueError(f"Expected square token grid, got num_image_tokens={num_image_tokens}.")
    mask = load_mask_image(mask_path, crop_size=crop_size, remap_nuclei=True)
    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if int(dilate_px) > 0:
        radius = int(dilate_px)
        dilated = Image.fromarray(binary * 255, mode="L").filter(
            ImageFilter.MaxFilter(size=radius * 2 + 1)
        )
        binary = (np.asarray(dilated, dtype=np.uint8) > 0).astype(np.uint8)
    height, width = binary.shape
    y_edges = np.linspace(0, height, side + 1, dtype=np.int64)
    x_edges = np.linspace(0, width, side + 1, dtype=np.int64)
    present = np.zeros((side, side), dtype=bool)
    counts = np.zeros((side, side), dtype=np.int64)
    fractions = np.zeros((side, side), dtype=np.float32)
    for y_index in range(side):
        y0 = int(y_edges[y_index])
        y1 = int(y_edges[y_index + 1])
        for x_index in range(side):
            x0 = int(x_edges[x_index])
            x1 = int(x_edges[x_index + 1])
            cell = binary[y0:y1, x0:x1]
            count = int(cell.sum())
            area = max(1, int(cell.size))
            frac = float(count) / float(area)
            counts[y_index, x_index] = count
            fractions[y_index, x_index] = frac
            present[y_index, x_index] = (
                (int(min_pixels) > 0 and count >= int(min_pixels))
                or (float(min_fraction) > 0.0 and frac >= float(min_fraction))
            )
    tensor = torch.from_numpy(present.reshape(1, side * side))
    stats = {
        "mask_path": str(mask_path),
        "token_grid": [side, side],
        "image_size": [int(width), int(height)],
        "present_token_count": int(present.sum()),
        "present_token_fraction": float(present.mean()),
        "max_cell_pixels": int(counts.max()) if counts.size else 0,
        "max_cell_fraction": float(fractions.max()) if fractions.size else 0.0,
        "mean_cell_pixels": float(counts.mean()) if counts.size else 0.0,
    }
    return tensor, stats


def save_bool_token_grid(tokens: Any, path: Path, *, scale: int = 16) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(tokens.detach().cpu().reshape(-1), dtype=np.uint8)
    side = int(round(float(array.size) ** 0.5))
    if side * side != int(array.size):
        raise ValueError(f"Token grid must be square, got {array.size} tokens.")
    image = Image.fromarray((array.reshape(side, side) * 255).astype(np.uint8), mode="L")
    image = image.convert("RGB").resize((side * int(scale), side * int(scale)), Image.Resampling.NEAREST)
    image.save(path)
    return str(path)


def save_kv_mask_correspondence_panel(
    *,
    payload: dict[str, Any],
    resolved: dict[str, Any],
    output_dir: Path,
    target_size: tuple[int, int],
    reference_size: tuple[int, int],
) -> dict[str, str]:
    panels: list[tuple[str, Image.Image]] = []
    panel_size = (256, 256)

    def _add(name: str, image: Image.Image | None) -> None:
        if image is None:
            return
        panels.append(
            (
                name,
                image.resize(panel_size, Image.Resampling.NEAREST),
            )
        )

    _add(
        "target_tissue_full",
        label_mask_preview_image(
            resolved.get("target_tissue_mask"),
            crop_size=target_size,
            remap_nuclei=False,
        ),
    )
    _add(
        "ref_tissue_full",
        label_mask_preview_image(
            resolved.get("kv_reference_tissue_mask"),
            crop_size=reference_size,
            remap_nuclei=False,
        ),
    )
    _add(
        "target_nuclei_full",
        label_mask_preview_image(
            resolved.get("target_nuclei_mask"),
            crop_size=target_size,
            remap_nuclei=True,
        ),
    )
    _add(
        "ref_nuclei_full",
        label_mask_preview_image(
            resolved.get("kv_reference_nuclei_mask"),
            crop_size=reference_size,
            remap_nuclei=True,
        ),
    )

    for name in (
        "target_tissue",
        "reference_tissue",
        "target_nuclei",
        "reference_nuclei",
        "target_composite",
        "reference_composite",
    ):
        _add(name, token_tensor_preview_image(payload.get(name), binary=False))
    _add(
        "target_nuclei_present",
        token_tensor_preview_image(payload.get("target_nuclei_present"), binary=True),
    )
    _add(
        "ref_nuclei_present",
        token_tensor_preview_image(payload.get("reference_nuclei_present"), binary=True),
    )

    if not panels:
        return {}
    path = output_dir / "kv_regional" / "mask_correspondence.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison = make_comparison_image(panels, columns=min(4, len(panels)))
    comparison.save(path)
    return {"mask_correspondence": str(path)}


def label_mask_preview_image(
    path: Path | None,
    *,
    crop_size: tuple[int, int],
    remap_nuclei: bool,
) -> Image.Image | None:
    if path is None:
        return None
    mask = load_mask_image(path, crop_size=crop_size, remap_nuclei=remap_nuclei)
    return label_array_to_preview(np.asarray(mask, dtype=np.int64), binary=False)


def token_tensor_preview_image(value: Any, *, binary: bool) -> Image.Image | None:
    if value is None:
        return None
    tensor = value.detach().cpu() if hasattr(value, "detach") else np.asarray(value)
    array = np.asarray(tensor).reshape(-1)
    side = int(round(float(array.size) ** 0.5))
    if side * side != int(array.size):
        return None
    return label_array_to_preview(array.reshape(side, side), binary=binary)


def label_array_to_preview(array: np.ndarray, *, binary: bool) -> Image.Image:
    values = np.asarray(array, dtype=np.int64)
    if binary:
        rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
        rgb[values > 0] = np.array([255, 64, 64], dtype=np.uint8)
        return Image.fromarray(rgb, mode="RGB")

    palette = np.array(
        [
            [66, 135, 245],
            [245, 130, 49],
            [60, 180, 75],
            [230, 25, 75],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
            [250, 190, 190],
            [0, 128, 128],
            [230, 190, 255],
            [170, 110, 40],
        ],
        dtype=np.uint8,
    )
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    labels = [int(label) for label in np.unique(values) if int(label) >= 0]
    labels = [label for label in labels if label != 0] + ([0] if 0 in labels else [])
    for index, label in enumerate(labels):
        if label < 0:
            continue
        if label == 0:
            color = np.array([32, 32, 32], dtype=np.uint8)
        else:
            color = palette[index % len(palette)]
        rgb[values == label] = color
    return Image.fromarray(rgb, mode="RGB")


def import_rf_helpers() -> dict[str, Any]:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    for path in (script_dir, project_root):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    import rf_solver_flux_reconstruct as rf_runner

    return {
        "prepend_rf_solver_src": rf_runner.prepend_rf_solver_src,
        "import_rf_solver_modules": rf_runner.import_rf_solver_modules,
        "configure_local_flux_weight_paths": rf_runner.configure_local_flux_weight_paths,
        "resolve_text_encoder_paths": rf_runner.resolve_text_encoder_paths,
        "validate_text_encoder_paths": rf_runner.validate_text_encoder_paths,
        "load_local_text_encoders": rf_runner.load_local_text_encoders,
        "encode_image": rf_runner.encode_image,
        "decode_image": rf_runner.decode_image,
        "denoise_first_order": rf_runner.denoise_first_order,
        "install_cross_image_forward_patch": rf_runner.install_cross_image_forward_patch,
        "CrossImageFeatureBank": rf_runner.CrossImageFeatureBank,
        "build_regional_token_labels": rf_runner.build_regional_token_labels,
        "save_regional_token_overlays": rf_runner.save_regional_token_overlays,
    }


def kv_inject_steps_for_capture(args: argparse.Namespace) -> int:
    if not args.kv_inject:
        return 0
    total_predictions = max(0, int(args.num_inference_steps) - 1)
    if total_predictions <= 0:
        return 0
    active = max(0, total_predictions - int(args.kv_inject_start_step))
    return min(total_predictions, active)


def kv_capture_inject_steps(args: argparse.Namespace) -> int:
    if not args.kv_inject:
        return 0
    return max(0, int(args.num_inference_steps) - 1)


class RFReferenceKVFeatureBank:
    def __init__(
        self,
        *,
        after_layer: int,
        image_token_count: int,
        inject_after_t: float,
    ) -> None:
        self.after_layer = int(after_layer)
        self.image_token_count = int(image_token_count)
        self.text_token_count: int | None = None
        self.inject_after_t = float(inject_after_t)
        self.phase = "reference_inversion"
        self.features: dict[str, dict[str, dict[str, Any]]] = {"reference": {}}
        self.events: list[dict[str, Any]] = []
        self.skipped: list[dict[str, Any]] = []

    def should_touch(self, info: dict[str, Any], block_id: int) -> bool:
        if not bool(info.get("inject")):
            return False
        if int(block_id) < self.after_layer:
            return False
        if not bool(info.get("inverse")):
            return False
        return float(info.get("t", 0.0)) <= self.inject_after_t

    def store(
        self,
        role: str,
        info: dict[str, Any],
        block_id: int,
        k: Any,
        v: Any,
    ) -> None:
        if role != "reference":
            return
        key = rf_kv_feature_key(info, block_id)
        self.features.setdefault("reference", {})[key] = {
            "K": k.detach().to("cpu"),
            "V": v.detach().to("cpu"),
        }
        total_tokens = int(k.shape[2])
        self.text_token_count = total_tokens - self.image_token_count
        if len(self.events) < 2048:
            self.events.append(
                {
                    "phase": self.phase,
                    "role": role,
                    "action": "store_kv",
                    "key": key,
                    "block_id": int(block_id),
                    "t": float(info.get("t", 0.0)),
                    "second_order": bool(info.get("second_order", False)),
                    "k_shape": list(k.shape),
                    "v_shape": list(v.shape),
                    "tokens_total": total_tokens,
                    "text_tokens": self.text_token_count,
                    "image_tokens": self.image_token_count,
                }
            )

    def summary(self) -> dict[str, Any]:
        entries = self.features.get("reference", {})
        blocks = sorted(
            {
                int(parsed["block_id"])
                for key in entries
                if (parsed := parse_rf_kv_feature_key(key)) is not None
            }
        )
        example: dict[str, Any] = {}
        if entries:
            first_key = next(iter(entries))
            payload = entries[first_key]
            example = {
                "example_key": first_key,
                "example_k_shape": list(payload["K"].shape),
                "example_v_shape": list(payload["V"].shape),
            }
        return {
            "after_layer": self.after_layer,
            "inject_after_t": self.inject_after_t,
            "image_token_count": self.image_token_count,
            "text_token_count": self.text_token_count,
            "reference": {
                "feature_count": len(entries),
                "block_ids": blocks,
                **example,
            },
            "event_count": len(self.events),
            "events_sample": self.events[:40],
        }


def rf_kv_feature_key(info: dict[str, Any], block_id: int) -> str:
    return (
        f"{info['t']}_{info['second_order']}_{int(block_id)}_"
        f"{info.get('type', 'single')}_KV"
    )


def resolve_kv_inject_after_t(args: argparse.Namespace, timesteps: list[float]) -> float:
    if args.kv_inject_after_t is not None:
        return float(args.kv_inject_after_t)
    if len(timesteps) < 2:
        return 1.0
    index = min(max(int(args.kv_inject_start_step), 0), len(timesteps) - 2)
    return float(timesteps[index])


def build_diffusers_kv_reference_features(
    *,
    feature_bank: Any,
    image_token_count: int,
) -> dict[str, Any]:
    features = feature_bank.features.get("reference", {})
    by_block: dict[int, list[dict[str, Any]]] = {}
    for key, payload in features.items():
        parsed = parse_rf_kv_feature_key(key)
        if parsed is None:
            continue
        by_block.setdefault(parsed["block_id"], []).append(
            {
                "key": key,
                "t": parsed["t"],
                "second_order": parsed["second_order"],
                "payload": payload,
            }
        )
    for entries in by_block.values():
        entries.sort(key=lambda item: (bool(item["second_order"]), float(item["t"])))
    summary = feature_bank.summary()
    summary.update(
        {
            "feature_count": len(features),
            "block_ids": sorted(by_block.keys()),
            "image_token_count": int(image_token_count),
        }
    )
    return {
        "by_block": by_block,
        "feature_count": len(features),
        "image_token_count": int(image_token_count),
        "summary": summary,
    }


def parse_rf_kv_feature_key(key: str) -> dict[str, Any] | None:
    parts = str(key).split("_")
    if len(parts) < 5:
        return None
    try:
        return {
            "t": float(parts[0]),
            "second_order": parts[1] == "True",
            "block_id": int(parts[2]),
            "stream_type": parts[3],
            "kind": parts[4],
        }
    except ValueError:
        return None


def select_record(
    metadata_path: Path | None,
    sample_id: str | None,
    metadata_index: int,
) -> dict[str, Any] | None:
    if metadata_path is None:
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        records = payload.get("pairs") or payload.get("records")
        if not isinstance(records, list):
            raise ValueError(f"Unsupported metadata object in {metadata_path}")
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(f"Unsupported metadata payload in {metadata_path}")
    if sample_id is not None:
        for record in records:
            if str(record.get("sample_id")) == sample_id:
                return record
        raise ValueError(f"sample_id {sample_id!r} not found in {metadata_path}")
    if metadata_index < 0 or metadata_index >= len(records):
        raise IndexError(f"--metadata-index {metadata_index} out of range for {len(records)} records")
    return records[metadata_index]


def resolve_inputs(args: argparse.Namespace, record: dict[str, Any] | None) -> dict[str, Any]:
    image_path = args.image
    prompt = args.source_prompt
    prompt_source = "cli" if prompt is not None else "default"
    target_tissue_mask = args.target_tissue_mask
    target_nuclei_mask = args.target_nuclei_mask
    reference_image = args.reference_image
    reference_tissue_mask = args.reference_tissue_mask
    reference_nuclei_mask = args.reference_nuclei_mask
    kv_reference_image = args.kv_reference_image
    kv_reference_tissue_mask = args.kv_reference_tissue_mask
    kv_reference_nuclei_mask = args.kv_reference_nuclei_mask
    kv_reference_prompt = args.kv_reference_prompt
    reference_source = args.controlnet_reference_source

    if record is not None:
        if image_path is None:
            image_value = record.get(args.image_field)
            if not image_value:
                raise KeyError(f"Selected metadata record has no {args.image_field!r}")
            image_path = Path(str(image_value))
        if prompt is None and record.get(args.prompt_field) is not None:
            prompt = str(record.get(args.prompt_field))
            prompt_source = f"metadata.{args.prompt_field}"

        if args.image_field == "target_image":
            target_tissue_mask = target_tissue_mask or _path_from_record(record, "target_tissue_mask")
            target_nuclei_mask = target_nuclei_mask or _path_from_record(record, "target_nuclei_mask")
        else:
            target_tissue_mask = target_tissue_mask or _path_from_record(record, "reference_tissue_mask")
            target_nuclei_mask = target_nuclei_mask or _path_from_record(record, "reference_nuclei_mask")

        if reference_source == "metadata":
            reference_image = reference_image or _path_from_record(record, "reference_image")
            reference_tissue_mask = reference_tissue_mask or _path_from_record(record, "reference_tissue_mask")
            reference_nuclei_mask = reference_nuclei_mask or _path_from_record(record, "reference_nuclei_mask")
        if kv_reference_image is None:
            kv_reference_image = _path_from_record(record, "reference_image")
            kv_reference_tissue_mask = kv_reference_tissue_mask or _path_from_record(record, "reference_tissue_mask")
            kv_reference_nuclei_mask = kv_reference_nuclei_mask or _path_from_record(record, "reference_nuclei_mask")

    if image_path is None:
        raise ValueError("--image is required unless --metadata supplies an image.")
    if prompt is None:
        prompt = DEFAULT_PROMPT
    if kv_reference_prompt is None:
        kv_reference_prompt = prompt
    if target_tissue_mask is None or target_nuclei_mask is None:
        raise ValueError(
            "Target tissue/nuclei masks are required. Supply --target-tissue-mask "
            "--target-nuclei-mask, or use metadata with mask fields."
        )

    if reference_source == "explicit":
        if reference_image is None or reference_tissue_mask is None or reference_nuclei_mask is None:
            raise ValueError(
                "Explicit ControlNet reference requires --reference-image, "
                "--reference-tissue-mask, and --reference-nuclei-mask."
            )
    elif reference_source == "metadata":
        if reference_image is None or reference_tissue_mask is None or reference_nuclei_mask is None:
            raise ValueError(
                "Metadata ControlNet reference requires reference_image/"
                "reference_tissue_mask/reference_nuclei_mask fields."
            )
    else:
        reference_image = image_path
        reference_tissue_mask = target_tissue_mask
        reference_nuclei_mask = target_nuclei_mask
    if args.regional_mode != "none":
        if kv_reference_image is None:
            raise ValueError("--regional-mode requires a K/V reference image.")
        if kv_reference_tissue_mask is None and Path(kv_reference_image) == Path(image_path):
            kv_reference_tissue_mask = target_tissue_mask
        if kv_reference_nuclei_mask is None and Path(kv_reference_image) == Path(image_path):
            kv_reference_nuclei_mask = target_nuclei_mask
        if args.regional_mode in {"tissue", "tissue_nuclei"} and kv_reference_tissue_mask is None:
            raise ValueError(
                f"--regional-mode {args.regional_mode} requires --kv-reference-tissue-mask "
                "or metadata reference_tissue_mask."
            )
        if args.regional_mode in {"nuclei", "tissue_nuclei"} and kv_reference_nuclei_mask is None:
            raise ValueError(
                f"--regional-mode {args.regional_mode} requires --kv-reference-nuclei-mask "
                "or metadata reference_nuclei_mask."
            )
    if args.kv_block_ref_nuclei_to_target_non_nuclei:
        if kv_reference_image is None:
            raise ValueError("--kv-block-ref-nuclei-to-target-non-nuclei requires a K/V reference image.")
        if kv_reference_nuclei_mask is None and Path(kv_reference_image) == Path(image_path):
            kv_reference_nuclei_mask = target_nuclei_mask
        if kv_reference_nuclei_mask is None:
            raise ValueError(
                "--kv-block-ref-nuclei-to-target-non-nuclei requires "
                "--kv-reference-nuclei-mask or metadata reference_nuclei_mask."
            )

    output_dir = args.output_dir
    if output_dir is None:
        sample_id = None
        if record is not None:
            sample_id = str(record.get("sample_id") or Path(str(image_path)).stem)
        else:
            sample_id = Path(str(image_path)).stem
        safe_sample_id = "".join(
            char if char.isalnum() or char in "._-" else "_"
            for char in sample_id
        )
        output_dir = (
            Path("phase5_runs")
            / "rf_solver_flux_controlnet_reconstruct"
            / f"{safe_sample_id}_{args.num_inference_steps}_{reference_source}"
        )

    return {
        "record": record,
        "image_path": Path(image_path),
        "source_prompt": prompt,
        "prompt_source": prompt_source,
        "target_tissue_mask": Path(target_tissue_mask),
        "target_nuclei_mask": Path(target_nuclei_mask),
        "reference_image": Path(reference_image),
        "reference_tissue_mask": Path(reference_tissue_mask),
        "reference_nuclei_mask": Path(reference_nuclei_mask),
        "reference_source": reference_source,
        "kv_reference_image": Path(kv_reference_image) if kv_reference_image is not None else None,
        "kv_reference_tissue_mask": (
            Path(kv_reference_tissue_mask) if kv_reference_tissue_mask is not None else None
        ),
        "kv_reference_nuclei_mask": (
            Path(kv_reference_nuclei_mask) if kv_reference_nuclei_mask is not None else None
        ),
        "kv_reference_prompt": kv_reference_prompt,
        "output_dir": output_dir,
    }


def _path_from_record(record: dict[str, Any], key: str) -> Path | None:
    value = record.get(key)
    return Path(str(value)) if value else None


def metadata_summary(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    keys = [
        "dataset",
        "sample_id",
        "reference_sample_id",
        "case_id",
        "pair_difficulty",
        "distance",
        "target_image",
        "reference_image",
        "target_tissue_mask",
        "reference_tissue_mask",
        "target_nuclei_mask",
        "reference_nuclei_mask",
        "prompt",
    ]
    return {key: record[key] for key in keys if key in record}


def load_reference_condition(
    *,
    resolved: dict[str, Any],
    original: Image.Image,
    target_tissue_mask: Image.Image,
    target_nuclei_mask: Image.Image,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    if resolved["reference_source"] == "self":
        return original.copy(), target_tissue_mask.copy(), target_nuclei_mask.copy()
    reference_image = crop_to_multiple(load_rgb_image(resolved["reference_image"]))[0]
    reference_tissue = load_mask_image(
        resolved["reference_tissue_mask"],
        crop_size=reference_image.size,
    )
    reference_nuclei = load_mask_image(
        resolved["reference_nuclei_mask"],
        crop_size=reference_image.size,
        remap_nuclei=True,
    )
    return reference_image, reference_tissue, reference_nuclei


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def crop_to_multiple(image: Image.Image, multiple: int = 16) -> tuple[Image.Image, dict[str, int]]:
    width, height = image.size
    cropped_width = width - (width % multiple)
    cropped_height = height - (height % multiple)
    if cropped_width <= 0 or cropped_height <= 0:
        raise ValueError(f"Image too small after crop-to-{multiple}: {width}x{height}")
    return image.crop((0, 0, cropped_width, cropped_height)), {
        "original_width": width,
        "original_height": height,
        "cropped_width": cropped_width,
        "cropped_height": cropped_height,
        "crop_left": 0,
        "crop_top": 0,
    }


def load_mask_image(
    path: Path,
    *,
    crop_size: tuple[int, int],
    remap_nuclei: bool = False,
) -> Image.Image:
    with Image.open(path) as image:
        mask = ImageOps.exif_transpose(image).convert("L")
        width, height = crop_size
        mask = mask.crop((0, 0, int(width), int(height)))
    if not remap_nuclei:
        return mask
    try:
        from controlnet_train.data.common import remap_nuclei_ids_array

        remapped = remap_nuclei_ids_array(np.asarray(mask, dtype=np.int64))
        return Image.fromarray(remapped.astype(np.uint8), mode="L")
    except Exception as exc:
        warnings.warn(
            f"Could not remap nuclei ids for {path}; using raw mask values. Error: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return mask


def pil_to_float_tensor(image: Image.Image) -> Any:
    import torch

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def mask_to_long_tensor(mask: Image.Image) -> Any:
    import torch

    return torch.from_numpy(np.asarray(mask, dtype=np.int64)).contiguous()


def image_metrics(original: Image.Image, reconstruction: Image.Image) -> dict[str, Any]:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    original_arr = np.asarray(original, dtype=np.uint8)
    reconstruction_arr = np.asarray(reconstruction, dtype=np.uint8)
    if original_arr.shape != reconstruction_arr.shape:
        height = min(original_arr.shape[0], reconstruction_arr.shape[0])
        width = min(original_arr.shape[1], reconstruction_arr.shape[1])
        original_arr = original_arr[:height, :width]
        reconstruction_arr = reconstruction_arr[:height, :width]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        psnr = float(peak_signal_noise_ratio(original_arr, reconstruction_arr, data_range=255))
    ssim = float(
        structural_similarity(
            original_arr,
            reconstruction_arr,
            channel_axis=2,
            data_range=255,
        )
    )
    abs_err = np.abs(original_arr.astype(np.int16) - reconstruction_arr.astype(np.int16))
    return {
        "psnr": "inf" if math.isinf(psnr) else psnr,
        "psnr_is_infinite": math.isinf(psnr),
        "ssim": ssim,
        "mean_abs_error": float(abs_err.mean()),
        "max_abs_error": int(abs_err.max()),
        "comparison_shape": list(original_arr.shape),
    }


def metric_value(value: Any) -> float:
    return math.inf if value == "inf" else float(value)


def tensor_debug_summary(tensor: Any) -> dict[str, Any]:
    values = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "mean": float(values.mean().cpu().item()),
        "std": float(values.std(unbiased=False).cpu().item()),
        "min": float(values.min().cpu().item()),
        "max": float(values.max().cpu().item()),
    }


def make_diff_image(original: Image.Image, reconstruction: Image.Image) -> Image.Image:
    original_arr = np.asarray(original, dtype=np.uint8)
    reconstruction_arr = np.asarray(reconstruction, dtype=np.uint8)
    height = min(original_arr.shape[0], reconstruction_arr.shape[0])
    width = min(original_arr.shape[1], reconstruction_arr.shape[1])
    diff = np.abs(
        original_arr[:height, :width].astype(np.int16)
        - reconstruction_arr[:height, :width].astype(np.int16)
    ).astype(np.uint8)
    return Image.fromarray(diff, mode="RGB")


def make_comparison_image(
    panels: list[tuple[str, Image.Image]],
    *,
    columns: int,
) -> Image.Image:
    if not panels:
        raise ValueError("make_comparison_image requires at least one panel.")
    width, height = panels[0][1].size
    label_height = 24
    gap = 8
    rows = int(math.ceil(len(panels) / columns))
    canvas = Image.new(
        "RGB",
        (
            columns * width + (columns - 1) * gap,
            rows * (height + label_height) + (rows - 1) * gap,
        ),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, panel) in enumerate(panels):
        row = index // columns
        col = index % columns
        x = col * (width + gap)
        y = row * (height + label_height + gap)
        draw.text((x + 4, y + 4), label, fill=(0, 0, 0))
        canvas.paste(panel.resize((width, height)), (x, y + label_height))
    return canvas


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def maybe_empty_cuda_cache(device: Any) -> None:
    import torch

    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())

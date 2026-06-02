"""Pure z_ref reference-path diagnostic for Cross V3.

The default diagnostic is teacher-forced one-step reconstruction: encode the
reference image to latent space, add fixed-timestep flow noise, and ask the
transformer to predict the reference velocity with only z_ref-derived reference
tokens. Target/ControlNet structure is bypassed, and reference tissue/nuclei
features are zeroed.

Free sampling from random noise is intentionally opt-in because, without the
target/ControlNet path, the base FLUX prior can dominate and generate arbitrary
semantic images that are not meaningful evidence about texture transfer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


Z_REF_ONLY_VARIANT = "z_ref_only"
ZERO_TOKENS_VARIANT = "zero_tokens"


@dataclass
class ZRefOnlyBundle:
    pretrained_model_name_or_path: str | Path
    checkpoint_path: Path
    device: str
    torch_dtype: torch.dtype
    num_inference_steps: int
    guidance_scale: float
    flux_pipeline: object
    condition_modules: dict[str, torch.nn.Module]
    reference_spec: object


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose Cross V3 reference texture transfer with fixed-t one-step "
            "reference reconstruction using z_ref-derived tokens and no target/ControlNet path."
        )
    )
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--checkpoint", required=True, help="Cross V3 checkpoint dir.")
    parser.add_argument("--metadata", required=True, help="metadata_cross_{train,val}.json path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reference-sample-id",
        action="append",
        default=[],
        help="Specific reference_sample_id to reconstruct. May be repeated.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=28, help="Used only with --run-free-sampling.")
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument(
        "--generation-seed",
        type=int,
        default=42,
        help="Base noise seed. Each sample uses generation_seed + sample index.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Minimal text prompt. Defaults to the fixed Cross V3 prompt.",
    )
    parser.add_argument(
        "--color-match",
        choices=("none", "lab"),
        default="none",
        help="Only applied to the optional --run-free-sampling OOD stress test.",
    )
    parser.add_argument("--thumbnail-size", type=int, default=192)
    parser.add_argument("--overview-max-samples", type=int, default=32)
    parser.add_argument(
        "--run-zero-z-ref-ablation",
        action="store_true",
        help="Also sample with all appended reference tokens zeroed.",
    )
    parser.add_argument(
        "--fixed-t-eval-timesteps",
        default="100,300,500,700",
        help="Comma-separated FLUX training timesteps for one-step reference reconstruction.",
    )
    parser.add_argument("--fixed-t-eval-seed", type=int, default=42)
    parser.add_argument(
        "--run-free-sampling",
        action="store_true",
        help="Also run the old random-noise free sampling path as an explicit OOD stress test.",
    )
    parser.add_argument("--glcm-levels", type=int, default=32)
    parser.add_argument("--glcm-distances", default="1,2,4")
    parser.add_argument("--glcm-angles", default="0,45,90,135")
    return parser


def parse_args(args=None) -> argparse.Namespace:
    return build_parser().parse_args(args)


def main(argv=None) -> int:
    args = parse_args(argv)

    from controlnet_train.cli.eval_controlnet_flux_cross import (
        _match_image_color_to_reference,
        _pil_to_chw_float,
        _safe_name,
        compute_cross_metrics,
        read_cross_metadata,
    )
    from controlnet_train.data.common import load_image_tensor
    from controlnet_train.inference.pipeline_cross_v3 import (
        CROSS_V3_PROMPT,
    )
    from scripts.diagnose_cross_v3_ref_mismatch import (
        _distance_stats,
        _parse_angles,
        _parse_int_list,
        _prefix_stats,
        image_quant_stats,
    )

    dtype_by_name = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    records = select_reference_records(
        read_cross_metadata(args.metadata),
        reference_sample_ids=args.reference_sample_id,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_cross_v3_z_ref_only_bundle(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        checkpoint_path=args.checkpoint,
        device=args.device,
        torch_dtype=dtype_by_name[args.torch_dtype],
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )
    prompt = str(args.prompt or CROSS_V3_PROMPT)
    variants = reference_variants(args.run_zero_z_ref_ablation)
    fixed_t_eval_timesteps = parse_fixed_t_eval_timesteps(args.fixed_t_eval_timesteps)
    glcm_distances = _parse_int_list(args.glcm_distances)
    glcm_angles = _parse_angles(args.glcm_angles)

    rows: list[dict[str, Any]] = []
    free_sampling_rows: list[dict[str, Any]] = []
    panel_paths: list[Path] = []
    for index, record in enumerate(records):
        sample_id = str(record.get("sample_id") or Path(record.get("target_image", "")).stem)
        ref_id = reference_record_id(record)
        sample_dir = samples_dir / f"{index:04d}_ref_{_safe_name(ref_id)}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        reference_image_path = Path(record["reference_image"])
        reference_tensor = load_image_tensor(reference_image_path)
        reference_pil = Image.open(reference_image_path).convert("RGB")
        reference_pil.save(sample_dir / "reference.png")
        reference_array = _pil_to_chw_float(reference_pil)
        reference_stats = image_quant_stats(
            reference_pil,
            levels=args.glcm_levels,
            distances=glcm_distances,
            angles=glcm_angles,
        )

        with torch.no_grad():
            reference_tokens, z_ref = build_z_ref_only_reference_tokens(
                bundle,
                reference_image=reference_tensor,
            )
        token_stats = tensor_stats(reference_tokens)
        z_ref_stats = tensor_stats(z_ref)

        variant_results: list[dict[str, Any]] = []
        sample_seed = int(args.generation_seed) + index
        timestep_seed = int(args.fixed_t_eval_seed) + index
        for variant in variants:
            active_tokens = (
                torch.zeros_like(reference_tokens)
                if variant == ZERO_TOKENS_VARIANT
                else reference_tokens
            )
            with torch.no_grad():
                timestep_results = fixed_timestep_reconstructions_reference_tokens_only(
                    bundle=bundle,
                    reference_image=reference_tensor,
                    reference_tokens=active_tokens,
                    prompt=prompt,
                    timesteps=fixed_t_eval_timesteps,
                    seed=timestep_seed,
                )

            prefix = f"_{variant}"
            for timestep_key, timestep_result in timestep_results.items():
                timestep_result["image"].save(sample_dir / f"one_step{prefix}_{timestep_key}.png")
                Image.fromarray(
                    (np.clip(timestep_result["abs_error"], 0.0, 1.0) * 255).astype(np.uint8),
                    mode="L",
                ).save(sample_dir / f"one_step_abs_error{prefix}_{timestep_key}.png")
            best_preview = select_preview_timestep(timestep_results)

            prediction_stats = image_quant_stats(
                best_preview["image"],
                levels=args.glcm_levels,
                distances=glcm_distances,
                angles=glcm_angles,
            )
            stat_row = {
                **_prefix_stats("reference", reference_stats),
                **_prefix_stats("prediction", prediction_stats),
            }
            stat_row.update(_distance_stats(stat_row, left="prediction", right="reference", prefix="pred_ref"))
            metrics = compute_cross_metrics(best_preview["pred_array"], reference_array)
            row = {
                "index": index,
                "sample_id": sample_id,
                "reference_sample_id": ref_id,
                "variant": variant,
                "diagnostic_mode": "teacher_forced_one_step",
                "dataset": record.get("dataset", ""),
                "controlnet_used": False,
                "target_condition_used": False,
                "ref_mask_features_used": False,
                "prompt": prompt,
                "fixed_t_eval_seed": timestep_seed,
                "preview_timestep": best_preview["timestep"],
                "preview_timestep_key": best_preview["timestep_key"],
                "z_ref_l2_norm": z_ref_stats["l2_norm"],
                "z_ref_std": z_ref_stats["std"],
                "reference_token_l2_norm": token_stats["l2_norm"] if variant == Z_REF_ONLY_VARIANT else 0.0,
                "reference_token_std": token_stats["std"] if variant == Z_REF_ONLY_VARIANT else 0.0,
                "preview_velocity_mse": best_preview["loss"],
                "preview_sigma": best_preview["sigma"],
                **metrics,
                **stat_row,
                **flatten_timestep_metrics(timestep_results),
            }
            variant_results.append(
                {
                    "variant": variant,
                    "row": row,
                    "prediction": best_preview["image"],
                    "pred_array": best_preview["pred_array"],
                    "abs_error": best_preview["abs_error"],
                    "preview_timestep_key": best_preview["timestep_key"],
                    "timestep_results": timestep_results,
                }
            )

            if args.run_free_sampling:
                with torch.no_grad():
                    raw_prediction = sample_with_reference_tokens_only(
                        bundle=bundle,
                        reference_tokens=active_tokens,
                        output_size=tuple(int(v) for v in reference_tensor.shape[1:]),
                        prompt=prompt,
                        seed=sample_seed,
                    ).convert("RGB")
                free_prefix = "" if len(variants) == 1 else f"_{variant}"
                raw_prediction.save(sample_dir / f"free_prediction{free_prefix}_raw.png")
                free_prediction = raw_prediction
                if args.color_match == "lab":
                    free_prediction = _match_image_color_to_reference(
                        source=raw_prediction,
                        reference=reference_pil,
                        method=args.color_match,
                    )
                free_prediction.save(sample_dir / f"free_prediction{free_prefix}.png")
                free_pred_array = _pil_to_chw_float(free_prediction)
                free_metrics = compute_cross_metrics(free_pred_array, reference_array)
                free_sampling_rows.append(
                    {
                        "index": index,
                        "sample_id": sample_id,
                        "reference_sample_id": ref_id,
                        "variant": variant,
                        "diagnostic_mode": "free_sampling_ood",
                        "generation_seed": sample_seed,
                        "color_match_applied": args.color_match != "none",
                        **free_metrics,
                    }
                )

        attach_zero_token_delta(variant_results, sample_dir=sample_dir)
        for result in variant_results:
            rows.append(result["row"])

        panel = make_reference_reconstruction_panel(
            reference=reference_pil,
            variant_results=variant_results,
            thumbnail_size=args.thumbnail_size,
            title=f"z_ref-only one-step reference diagnostic | ref={ref_id}",
        )
        panel_path = sample_dir / "panel.png"
        panel.save(panel_path)
        if len(panel_paths) < args.overview_max_samples:
            panel_paths.append(panel_path)

        (sample_dir / "metrics.json").write_text(
            json.dumps(
                {
                    result["variant"]: result["row"]
                    for result in variant_results
                },
                indent=2,
                ensure_ascii=False,
                allow_nan=True,
            ),
            encoding="utf8",
        )
        primary = next(result for result in variant_results if result["variant"] == Z_REF_ONLY_VARIANT)
        message = (
            f"[{index + 1}/{len(records)}] ref={ref_id} "
            f"preview={primary['row']['preview_timestep_key']} "
            f"z_ref_loss={primary['row']['preview_velocity_mse']:.4f} "
            f"z_ref_l1={primary['row']['full_l1']:.4f} "
            f"z_ref_psnr={primary['row']['full_psnr']:.2f}"
        )
        if "preview_velocity_mse_delta_zero_minus_z_ref" in primary["row"]:
            message += f" zero_minus_z_ref={primary['row']['preview_velocity_mse_delta_zero_minus_z_ref']:.4f}"
        if "preview_noise_pred_relative_l2_vs_zero" in primary["row"]:
            message += f" pred_rel_delta={primary['row']['preview_noise_pred_relative_l2_vs_zero']:.4f}"
        print(message)

    write_rows(output_dir, rows)
    if free_sampling_rows:
        write_rows(output_dir, free_sampling_rows, stem="free_sampling_metrics")
    summary = build_z_ref_reconstruction_summary(rows)
    summary["architecture"] = "cross_v3"
    summary["controlnet_used"] = False
    summary["target_condition_used"] = False
    summary["ref_mask_features_used"] = False
    summary["prompt"] = prompt
    summary["glcm_config"] = {
        "levels": args.glcm_levels,
        "distances": glcm_distances,
        "angles_degrees": glcm_angles,
    }
    summary["fixed_t_eval_timesteps"] = fixed_t_eval_timesteps
    if free_sampling_rows:
        summary["free_sampling"] = aggregate_fixed_t_losses(free_sampling_rows)
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf8",
    )
    if panel_paths:
        make_overview(panel_paths).save(output_dir / "overview_grid.png")
    print(f"wrote z_ref reconstruction diagnostic outputs to {output_dir}")
    return 0


def reference_variants(run_zero_z_ref_ablation: bool) -> list[str]:
    variants = [Z_REF_ONLY_VARIANT]
    if run_zero_z_ref_ablation:
        variants.append(ZERO_TOKENS_VARIANT)
    return variants


def reference_record_id(record: dict[str, Any]) -> str:
    return str(record.get("reference_sample_id") or Path(record["reference_image"]).stem)


def select_reference_records(
    records: list[dict[str, Any]],
    *,
    reference_sample_ids: list[str],
    num_samples: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    candidates = unique_reference_records(records)
    if reference_sample_ids:
        by_id = {reference_record_id(record): record for record in candidates}
        missing = [sample_id for sample_id in reference_sample_ids if sample_id not in by_id]
        if missing:
            raise ValueError(f"reference sample_id(s) not found: {missing}")
        return [by_id[sample_id] for sample_id in reference_sample_ids]
    if num_samples is None or num_samples <= 0 or num_samples >= len(candidates):
        return candidates
    selected = list(candidates)
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


def load_cross_v3_z_ref_only_bundle(
    *,
    pretrained_model_name_or_path: str | Path,
    checkpoint_path: str | Path,
    device: str = "cuda",
    torch_dtype: torch.dtype | None = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
) -> ZRefOnlyBundle:
    from diffusers import FluxPipeline

    from controlnet_train.inference.pipeline_cross_v3 import (
        _load_cross_v3_reference_spec,
        _resolve_device,
        _resolve_torch_dtype,
        _torch_load_weights,
    )
    from controlnet_train.modules.cross_v3_conditioning import CrossV3ReferenceContextEncoder

    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_torch_dtype(torch_dtype, resolved_device)
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint}")
    conditioning_path = checkpoint / "phase5_conditioning.pt"
    if not conditioning_path.exists():
        raise FileNotFoundError(f"Missing phase5_conditioning.pt under checkpoint path: {checkpoint}")

    reference_spec = _load_cross_v3_reference_spec(checkpoint)
    state = _torch_load_weights(conditioning_path)
    reference_state = state["reference_context_encoder"]
    reference_context_encoder = CrossV3ReferenceContextEncoder(
        reference_latent_channels=reference_spec.reference_latent_channels,
        tissue_channels=reference_spec.tissue_channels,
        nuclei_channels=reference_spec.nuclei_channels,
        token_dim=reference_spec.token_dim,
        hidden_dim=reference_state["proj_in.weight"].shape[0],
        output_init_std=reference_spec.output_init_std,
        route_anchor_mode=reference_spec.route_anchor_mode,
        route_embedding_init_std=reference_spec.route_embedding_init_std,
    )
    reference_context_encoder.load_state_dict(reference_state)
    reference_context_encoder.to(device=resolved_device, dtype=resolved_dtype)
    reference_context_encoder.eval()

    pipe = FluxPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=resolved_dtype,
    )
    pipe.to(resolved_device)
    pipe.set_progress_bar_config(disable=True)
    return ZRefOnlyBundle(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        checkpoint_path=checkpoint,
        device=resolved_device,
        torch_dtype=resolved_dtype,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        flux_pipeline=pipe,
        condition_modules={"reference_context_encoder": reference_context_encoder},
        reference_spec=reference_spec,
    )


@torch.inference_mode()
def build_z_ref_only_reference_tokens(bundle, *, reference_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from controlnet_train.inference.pipeline_cross_v3 import _encode_images_to_latents

    z_ref = _encode_images_to_latents(
        bundle.flux_pipeline.vae,
        reference_image.unsqueeze(0),
        bundle.torch_dtype,
    )
    ref_tissue_feat = torch.zeros(
        z_ref.shape[0],
        int(bundle.reference_spec.tissue_channels),
        z_ref.shape[2],
        z_ref.shape[3],
        device=z_ref.device,
        dtype=bundle.torch_dtype,
    )
    ref_nuclei_feat = torch.zeros(
        z_ref.shape[0],
        int(bundle.reference_spec.nuclei_channels),
        z_ref.shape[2],
        z_ref.shape[3],
        device=z_ref.device,
        dtype=bundle.torch_dtype,
    )
    reference_tokens = bundle.condition_modules["reference_context_encoder"](
        z_ref=z_ref,
        ref_tissue_feat=ref_tissue_feat,
        ref_nuclei_feat=ref_nuclei_feat,
        ref_tissue_ids=torch.zeros(
            z_ref.shape[0],
            int(reference_image.shape[1]),
            int(reference_image.shape[2]),
            device=z_ref.device,
            dtype=torch.long,
        ),
    )
    return reference_tokens, z_ref


def select_preview_timestep(timestep_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not timestep_results:
        raise ValueError("No timestep reconstruction results were produced.")
    return min(
        timestep_results.values(),
        key=lambda result: abs(float(result["timestep"]) - 500.0),
    )


def flatten_timestep_metrics(timestep_results: dict[str, dict[str, Any]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for timestep_key, result in sorted(timestep_results.items(), key=lambda item: float(item[1]["timestep"])):
        metrics = result["metrics"]
        output[f"velocity_mse_{timestep_key}"] = float(result["loss"])
        output[f"sigma_{timestep_key}"] = float(result["sigma"])
        output[f"one_step_l1_{timestep_key}"] = float(metrics["full_l1"])
        output[f"one_step_mse_{timestep_key}"] = float(metrics["full_mse"])
        output[f"one_step_psnr_{timestep_key}"] = float(metrics["full_psnr"])
    return output


@torch.inference_mode()
def sample_with_reference_tokens_only(
    *,
    bundle,
    reference_tokens: torch.Tensor,
    output_size: tuple[int, int],
    prompt: str,
    seed: int,
) -> Image.Image:
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

    from controlnet_train.inference.pipeline_cross_v3 import _calculate_shift
    from controlnet_train.modules.cross_v3_conditioning import append_cross_v3_reference_context

    pipe = bundle.flux_pipeline
    torch_device = torch.device(bundle.device)
    height, width = output_size
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=[prompt],
        prompt_2=[prompt],
        device=torch_device,
    )
    if text_ids.dim() == 3:
        text_ids = text_ids[0]
    transformer_embeds, transformer_text_ids = append_cross_v3_reference_context(
        prompt_embeds=prompt_embeds,
        text_ids=text_ids,
        reference_tokens=reference_tokens.to(device=torch_device, dtype=prompt_embeds.dtype),
    )

    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents, latent_image_ids = pipe.prepare_latents(
        1,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        torch_device,
        generator=torch.Generator(device=torch_device).manual_seed(seed),
        latents=None,
    )
    sigmas = np.linspace(1.0, 1 / bundle.num_inference_steps, bundle.num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = _calculate_shift(
        image_seq_len=image_seq_len,
        base_seq_len=pipe.scheduler.config.get("base_image_seq_len", 256),
        max_seq_len=pipe.scheduler.config.get("max_image_seq_len", 4096),
        base_shift=pipe.scheduler.config.get("base_shift", 0.5),
        max_shift=pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, _ = retrieve_timesteps(
        pipe.scheduler,
        bundle.num_inference_steps,
        torch_device,
        sigmas=sigmas,
        mu=mu,
    )

    for timestep in timesteps:
        expanded_timestep = timestep.expand(latents.shape[0]).to(latents.dtype)
        transformer_guidance = None
        if pipe.transformer.config.guidance_embeds:
            transformer_guidance = torch.tensor(
                [bundle.guidance_scale],
                device=torch_device,
            ).expand(latents.shape[0])
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=expanded_timestep / 1000,
            guidance=transformer_guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=transformer_embeds,
            txt_ids=transformer_text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        latents_dtype = latents.dtype
        latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
        if latents.dtype != latents_dtype:
            latents = latents.to(latents_dtype)

    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents.to(dtype=bundle.torch_dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


@torch.inference_mode()
def fixed_timestep_reconstructions_reference_tokens_only(
    *,
    bundle,
    reference_image: torch.Tensor,
    reference_tokens: torch.Tensor,
    prompt: str,
    timesteps: list[float],
    seed: int,
) -> dict[str, dict[str, Any]]:
    from controlnet_train.inference.pipeline_cross_v3 import (
        _encode_images_to_latents,
        _format_timestep_key,
        _per_sample_mse,
        _prepare_packed_latent_image_ids,
        _sigma_for_timestep,
    )
    from controlnet_train.modules.cross_v3_conditioning import append_cross_v3_reference_context

    pipe = bundle.flux_pipeline
    torch_device = torch.device(bundle.device)
    height, width = tuple(int(v) for v in reference_image.shape[1:])
    pixel_latents = _encode_images_to_latents(
        pipe.vae,
        reference_image.unsqueeze(0),
        bundle.torch_dtype,
    )
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=[prompt],
        prompt_2=[prompt],
        device=torch_device,
    )
    if text_ids.dim() == 3:
        text_ids = text_ids[0]
    transformer_embeds, transformer_text_ids = append_cross_v3_reference_context(
        prompt_embeds=prompt_embeds,
        text_ids=text_ids,
        reference_tokens=reference_tokens.to(device=torch_device, dtype=prompt_embeds.dtype),
    )

    bsz = int(pixel_latents.shape[0])
    packed_pixel_latents = pipe._pack_latents(
        pixel_latents,
        bsz,
        pixel_latents.shape[1],
        pixel_latents.shape[2],
        pixel_latents.shape[3],
    )
    latent_image_ids = _prepare_packed_latent_image_ids(
        packed_height=pixel_latents.shape[2] // 2,
        packed_width=pixel_latents.shape[3] // 2,
        device=torch_device,
        dtype=bundle.torch_dtype,
    )
    generator = torch.Generator(device=torch_device).manual_seed(seed)
    noise = torch.randn(
        packed_pixel_latents.shape,
        generator=generator,
        device=packed_pixel_latents.device,
        dtype=packed_pixel_latents.dtype,
    )

    reference_array = reference_image.detach().float().cpu().numpy()
    results: dict[str, dict[str, Any]] = {}
    for timestep_value in timesteps:
        timestep = torch.tensor([timestep_value], device=torch_device, dtype=torch.float32)
        sigma = _sigma_for_timestep(
            pipe.scheduler,
            timestep,
            n_dim=packed_pixel_latents.ndim,
            dtype=packed_pixel_latents.dtype,
        )
        noisy_model_input = (1.0 - sigma) * packed_pixel_latents + sigma * noise
        expanded_timestep = timestep.expand(bsz).to(dtype=packed_pixel_latents.dtype)
        transformer_guidance = None
        if pipe.transformer.config.guidance_embeds:
            transformer_guidance = torch.full((bsz,), bundle.guidance_scale, device=torch_device)
        noise_pred = pipe.transformer(
            hidden_states=noisy_model_input,
            timestep=expanded_timestep / 1000,
            guidance=transformer_guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=transformer_embeds,
            txt_ids=transformer_text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        target_velocity = noise - packed_pixel_latents
        loss = _per_sample_mse(noise_pred, target_velocity).mean()
        reconstructed_packed = noisy_model_input - sigma * noise_pred
        reconstructed_latents = pipe._unpack_latents(
            reconstructed_packed,
            height,
            width,
            pipe.vae_scale_factor,
        )
        reconstructed_latents = (
            reconstructed_latents / pipe.vae.config.scaling_factor
        ) + pipe.vae.config.shift_factor
        decoded = pipe.vae.decode(reconstructed_latents.to(dtype=bundle.torch_dtype), return_dict=False)[0]
        image = pipe.image_processor.postprocess(decoded, output_type="pil")[0].convert("RGB")
        pred_array = pil_to_chw_float(image)
        abs_error = np.abs(pred_array - reference_array).mean(axis=0)
        timestep_key = _format_timestep_key(timestep_value)
        results[timestep_key] = {
            "timestep_key": timestep_key,
            "timestep": float(timestep_value),
            "sigma": float(sigma.detach().float().flatten()[0].cpu().item()),
            "loss": float(loss.detach().cpu().item()),
            "noise_pred_flat": noise_pred.detach().float().reshape(-1).cpu(),
            "image": image,
            "pred_array": pred_array,
            "abs_error": abs_error,
            "metrics": compute_array_metrics(pred_array, reference_array),
        }
    return results


@torch.inference_mode()
def fixed_timestep_losses_reference_tokens_only(
    *,
    bundle,
    reference_image: torch.Tensor,
    reference_tokens: torch.Tensor,
    prompt: str,
    timesteps: list[float],
    seed: int,
) -> dict[str, float]:
    reconstructions = fixed_timestep_reconstructions_reference_tokens_only(
        bundle=bundle,
        reference_image=reference_image,
        reference_tokens=reference_tokens,
        prompt=prompt,
        timesteps=timesteps,
        seed=seed,
    )
    return {key: float(result["loss"]) for key, result in reconstructions.items()}


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
    return {
        "full_l1": float(np.abs(diff).mean()),
        "full_mse": mse,
        "full_psnr": psnr,
    }


def attach_zero_token_delta(variant_results: list[dict[str, Any]], *, sample_dir: Path) -> None:
    by_variant = {result["variant"]: result for result in variant_results}
    z_ref = by_variant.get(Z_REF_ONLY_VARIANT)
    zero = by_variant.get(ZERO_TOKENS_VARIANT)
    if z_ref is None or zero is None:
        return
    diff = np.abs(z_ref["pred_array"] - zero["pred_array"])
    l1 = float(diff.mean())
    mse = float(np.square(z_ref["pred_array"] - zero["pred_array"]).mean())
    z_ref["row"]["prediction_l1_vs_zero_tokens"] = l1
    z_ref["row"]["prediction_mse_vs_zero_tokens"] = mse
    zero["row"]["prediction_l1_vs_z_ref_only"] = l1
    zero["row"]["prediction_mse_vs_z_ref_only"] = mse
    if (
        z_ref["row"].get("preview_timestep_key") == zero["row"].get("preview_timestep_key")
        and "preview_velocity_mse" in z_ref["row"]
        and "preview_velocity_mse" in zero["row"]
    ):
        loss_delta = float(zero["row"]["preview_velocity_mse"] - z_ref["row"]["preview_velocity_mse"])
        z_ref["row"]["preview_velocity_mse_delta_zero_minus_z_ref"] = loss_delta
        zero["row"]["preview_velocity_mse_delta_zero_minus_z_ref"] = loss_delta
    attach_noise_prediction_deltas(z_ref, zero)
    Image.fromarray((np.clip(diff.mean(axis=0), 0.0, 1.0) * 255).astype(np.uint8), mode="L").save(
        sample_dir / "z_ref_vs_zero_tokens_diff.png"
    )


def attach_noise_prediction_deltas(z_ref: dict[str, Any], zero: dict[str, Any]) -> None:
    z_results = z_ref.get("timestep_results") or {}
    zero_results = zero.get("timestep_results") or {}
    preview_key = str(z_ref["row"].get("preview_timestep_key", ""))
    for timestep_key in sorted(set(z_results) & set(zero_results)):
        left = z_results[timestep_key].get("noise_pred_flat")
        right = zero_results[timestep_key].get("noise_pred_flat")
        if left is None or right is None:
            continue
        stats = compare_flat_tensors(left, right)
        for name, value in stats.items():
            z_ref["row"][f"noise_pred_{name}_vs_zero_{timestep_key}"] = value
            zero["row"][f"noise_pred_{name}_vs_z_ref_{timestep_key}"] = value
        if timestep_key == preview_key:
            for name, value in stats.items():
                z_ref["row"][f"preview_noise_pred_{name}_vs_zero"] = value
                zero["row"][f"preview_noise_pred_{name}_vs_z_ref"] = value


def compare_flat_tensors(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left_f = left.detach().float().reshape(-1)
    right_f = right.detach().float().reshape(-1)
    if left_f.numel() != right_f.numel():
        raise ValueError(f"Tensor sizes differ: {left_f.numel()} vs {right_f.numel()}")
    diff = left_f - right_f
    left_norm = float(torch.linalg.vector_norm(left_f).item())
    right_norm = float(torch.linalg.vector_norm(right_f).item())
    diff_l2 = float(torch.linalg.vector_norm(diff).item())
    denom = max(left_norm * right_norm, 1e-12)
    cosine = float(torch.dot(left_f, right_f).item() / denom)
    return {
        "l1": float(diff.abs().mean().item()),
        "l2": diff_l2,
        "relative_l2": float(diff_l2 / max(right_norm, 1e-12)),
        "cosine": cosine,
    }


def make_reference_reconstruction_panel(
    *,
    reference: Image.Image,
    variant_results: list[dict[str, Any]],
    thumbnail_size: int,
    title: str,
) -> Image.Image:
    images: list[tuple[str, Image.Image]] = [("reference", reference.convert("RGB"))]
    for result in variant_results:
        label = f"{result['variant']} {result.get('preview_timestep_key', '')}".strip()
        images.append((label, result["prediction"].convert("RGB")))
    primary = next((result for result in variant_results if result["variant"] == Z_REF_ONLY_VARIANT), variant_results[0])
    images.append(
        (
            "abs_error",
            Image.fromarray(
                (np.clip(primary["abs_error"], 0.0, 1.0) * 255).astype(np.uint8),
                mode="L",
            ).convert("RGB"),
        )
    )
    thumbs = [(label, _thumbnail(image, thumbnail_size)) for label, image in images]
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


def _thumbnail(image: Image.Image, size: int) -> Image.Image:
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


def parse_fixed_t_eval_timesteps(value: str | None) -> list[float]:
    if value is None or not value.strip():
        return []
    timesteps: list[float] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        timestep = float(part)
        if not math.isfinite(timestep):
            raise ValueError(f"Fixed timestep must be finite, got {part!r}.")
        timesteps.append(timestep)
    if not timesteps:
        raise ValueError("--fixed-t-eval-timesteps must contain at least one value.")
    return timesteps


def tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    value = tensor.detach().float()
    return {
        "mean": float(value.mean().item()),
        "std": float(value.std().item()),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "l2_norm": float(torch.linalg.vector_norm(value).item()),
    }


def build_z_ref_reconstruction_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(str(row.get("variant", "")), []).append(row)

    summary: dict[str, Any] = {
        "variants": sorted(key for key in by_variant if key),
        "by_variant": {
            variant: aggregate_numeric_rows(variant_rows)
            for variant, variant_rows in sorted(by_variant.items())
        },
    }
    z_rows = by_variant.get(Z_REF_ONLY_VARIANT, [])
    summary["num_samples"] = float(len(z_rows))
    if z_rows:
        z_summary = summary["by_variant"].get(Z_REF_ONLY_VARIANT, {})
        full_l1 = float(z_summary.get("full_l1_mean", math.nan))
        full_psnr = float(z_summary.get("full_psnr_mean", math.nan))
        summary["reference_reconstruction_hint"] = interpret_reconstruction_quality(
            full_l1=full_l1,
            full_psnr=full_psnr,
        )
        zero_deltas = [
            float(row["prediction_l1_vs_zero_tokens"])
            for row in z_rows
            if "prediction_l1_vs_zero_tokens" in row
            and math.isfinite(float(row["prediction_l1_vs_zero_tokens"]))
        ]
        if zero_deltas:
            summary["prediction_l1_vs_zero_tokens_mean"] = float(np.mean(zero_deltas))
            summary["prediction_l1_vs_zero_tokens_std"] = float(np.std(zero_deltas))
            summary["z_ref_effect_hint"] = (
                "z_ref_tokens_have_little_visible_effect"
                if float(np.mean(zero_deltas)) < 0.01
                else "z_ref_tokens_change_generation"
            )
        loss_deltas = [
            float(row["preview_velocity_mse_delta_zero_minus_z_ref"])
            for row in z_rows
            if "preview_velocity_mse_delta_zero_minus_z_ref" in row
            and math.isfinite(float(row["preview_velocity_mse_delta_zero_minus_z_ref"]))
        ]
        if loss_deltas:
            mean_delta = float(np.mean(loss_deltas))
            summary["preview_velocity_mse_delta_zero_minus_z_ref_mean"] = mean_delta
            summary["preview_velocity_mse_delta_zero_minus_z_ref_std"] = float(np.std(loss_deltas))
            summary["z_ref_loss_hint"] = (
                "z_ref_reduces_reference_denoising_loss"
                if mean_delta > 1e-4
                else "no_clear_z_ref_loss_benefit"
            )
    return summary


def interpret_reconstruction_quality(*, full_l1: float, full_psnr: float) -> str:
    if math.isfinite(full_l1) and full_l1 <= 0.08:
        return "strong_reference_reconstruction"
    if math.isfinite(full_psnr) and full_psnr >= 22.0:
        return "strong_reference_reconstruction"
    if math.isfinite(full_l1) and full_l1 >= 0.16:
        return "poor_reference_reconstruction"
    if math.isfinite(full_psnr) and full_psnr <= 16.0:
        return "poor_reference_reconstruction"
    return "mixed_reference_reconstruction"


def aggregate_numeric_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (float, int)) and key not in {"index"}
        }
    )
    output: dict[str, float] = {"num_samples": float(len(rows))}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
        if values:
            output[f"{key}_mean"] = float(np.mean(values))
            output[f"{key}_std"] = float(np.std(values))
    return output


def aggregate_fixed_t_losses(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(str(row.get("variant", "")), []).append(row)
    return {
        variant: aggregate_numeric_rows(variant_rows)
        for variant, variant_rows in sorted(by_variant.items())
    }


def write_rows(output_dir: Path, rows: list[dict[str, Any]], *, stem: str = "metrics") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{stem}.jsonl").open("w", encoding="utf8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row if not isinstance(row.get(key), dict)})
    with (output_dir / f"{stem}.csv").open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key in fieldnames})


if __name__ == "__main__":
    raise SystemExit(main())

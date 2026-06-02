"""Diagnose whether Cross V3 reference tokens are used by the transformer.

This diagnostic runs the full Cross V3 one-step denoising path:

    target image latent + target ControlNet condition + reference tokens

It compares correct reference tokens against zero, mismatched, and token-mean
ablations. It also backpropagates the fixed-t denoising loss to the appended
text+reference context embeddings, which gives a functional saliency signal for
whether the frozen FLUX transformer is sensitive to reference tokens.

This intentionally does not depend on fragile attention-weight hooks. Modern
diffusers/torch attention often uses fused kernels that do not expose attention
probabilities. The forward ablations and context gradients answer the more
important question: do reference tokens affect the prediction/loss in the real
training path?
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

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


CORRECT_VARIANT = "correct"
ZERO_VARIANT = "zero"
MISMATCH_VARIANT = "mismatch"
MEAN_VARIANT = "token_mean"
VARIANTS = (CORRECT_VARIANT, ZERO_VARIANT, MISMATCH_VARIANT, MEAN_VARIANT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Full-path Cross V3 probe for whether reference tokens affect frozen "
            "FLUX denoising and receive loss gradient."
        )
    )
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--checkpoint", required=True, help="Cross V3 checkpoint dir.")
    parser.add_argument("--metadata", required=True, help="metadata_cross_{train,val}.json path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-sample-id", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--timesteps", default="500,700,900")
    parser.add_argument("--noise-seed", type=int, default=42)
    return parser


def parse_args(args=None) -> argparse.Namespace:
    return build_parser().parse_args(args)


def main(argv=None) -> int:
    args = parse_args(argv)

    from controlnet_train.cli.eval_controlnet_flux_cross import _safe_name, read_cross_metadata
    from controlnet_train.data.common import load_image_tensor, load_nuclei_mask, load_tissue_mask
    from controlnet_train.inference.pipeline_cross_v3 import load_cross_v3_bundle
    from scripts.diagnose_cross_v3_z_ref_reconstruction import parse_fixed_t_eval_timesteps

    dtype_by_name = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    records = unique_reference_records(read_cross_metadata(args.metadata))
    selected = select_records(
        records,
        reference_sample_ids=args.reference_sample_id,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    if len(records) < 2:
        raise ValueError("Need at least two unique references for mismatch-token probing.")
    timesteps = parse_fixed_t_eval_timesteps(args.timesteps)
    rng = random.Random(args.seed)

    bundle = load_cross_v3_bundle(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        checkpoint_path=args.checkpoint,
        device=args.device,
        torch_dtype=dtype_by_name[args.torch_dtype],
        num_inference_steps=1,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
    )
    freeze_bundle(bundle)

    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(selected):
        ref_id = reference_record_id(record)
        mismatch = select_mismatch_record(records, anchor=record, rng=rng)
        mismatch_id = reference_record_id(mismatch)
        sample_dir = samples_dir / f"{index:04d}_{_safe_name(record_id(record))}__ref_{_safe_name(ref_id)}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        tensors = load_usage_sample_tensors(
            record=record,
            mismatch=mismatch,
            load_image_tensor=load_image_tensor,
            load_tissue_mask=load_tissue_mask,
            load_nuclei_mask=load_nuclei_mask,
        )
        usage_results = run_full_path_usage_probe(
            bundle=bundle,
            tensors=tensors,
            timesteps=timesteps,
            seed=int(args.noise_seed) + index,
        )
        row = build_usage_row(
            index=index,
            record=record,
            mismatch=mismatch,
            usage_results=usage_results,
        )
        rows.append(row)
        (sample_dir / "metrics.json").write_text(
            json.dumps(row, indent=2, ensure_ascii=False, allow_nan=True),
            encoding="utf8",
        )
        preview_key = select_preview_timestep_key(usage_results)
        print(
            f"[{index + 1}/{len(selected)}] sample={record_id(record)} ref={ref_id} mismatch={mismatch_id} "
            f"{preview_key}_zero_minus_correct={row[f'zero_minus_correct_loss_{preview_key}']:.4f} "
            f"{preview_key}_mismatch_minus_correct={row[f'mismatch_minus_correct_loss_{preview_key}']:.4f} "
            f"{preview_key}_mean_minus_correct={row[f'token_mean_minus_correct_loss_{preview_key}']:.4f} "
            f"pred_rel_zero={row[f'noise_pred_correct_vs_zero_relative_l2_{preview_key}']:.4f} "
            f"ref_grad/text={row[f'grad_ref_vs_text_token_mean_ratio_{preview_key}']:.4f}"
        )

    write_rows(output_dir / "metrics.csv", rows)
    (output_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, allow_nan=True) for row in rows) + "\n",
        encoding="utf8",
    )
    summary = build_usage_summary(rows)
    summary["diagnostic"] = "cross_v3_ref_token_usage"
    summary["timesteps"] = timesteps
    summary["interpretation"] = interpret_usage_summary(summary)
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf8",
    )
    print(f"interpretation={summary['interpretation']}")
    print(
        "means: "
        f"zero_minus_correct={summary.get('zero_minus_correct_loss_all_mean', math.nan):.4f} "
        f"mismatch_minus_correct={summary.get('mismatch_minus_correct_loss_all_mean', math.nan):.4f} "
        f"mean_minus_correct={summary.get('token_mean_minus_correct_loss_all_mean', math.nan):.4f} "
        f"pred_rel_zero={summary.get('noise_pred_correct_vs_zero_relative_l2_all_mean', math.nan):.4f} "
        f"grad_ref/text={summary.get('grad_ref_vs_text_token_mean_ratio_all_mean', math.nan):.4f}"
    )
    print(f"wrote Cross V3 reference-token usage diagnostic outputs to {output_dir}")
    return 0


def freeze_bundle(bundle) -> None:
    modules = [bundle.flux_pipeline.vae, bundle.flux_pipeline.transformer, bundle.controlnet, *bundle.condition_modules.values()]
    for module in modules:
        module.eval()
        module.requires_grad_(False)


def load_usage_sample_tensors(
    *,
    record: dict[str, Any],
    mismatch: dict[str, Any],
    load_image_tensor,
    load_tissue_mask,
    load_nuclei_mask,
) -> dict[str, torch.Tensor]:
    return {
        "target_image": load_image_tensor(record["target_image"]),
        "target_tissue_mask": load_tissue_mask(record["target_tissue_mask"]),
        "target_nuclei_mask": load_nuclei_mask(record["target_nuclei_mask"]),
        "reference_image": load_image_tensor(record["reference_image"]),
        "reference_tissue_mask": load_tissue_mask(record["reference_tissue_mask"]),
        "reference_nuclei_mask": load_nuclei_mask(record["reference_nuclei_mask"]),
        "mismatch_reference_image": load_image_tensor(mismatch["reference_image"]),
        "mismatch_reference_tissue_mask": load_tissue_mask(mismatch["reference_tissue_mask"]),
        "mismatch_reference_nuclei_mask": load_nuclei_mask(mismatch["reference_nuclei_mask"]),
    }


def run_full_path_usage_probe(
    *,
    bundle,
    tensors: dict[str, torch.Tensor],
    timesteps: list[float],
    seed: int,
) -> dict[str, dict[str, Any]]:
    from controlnet_train.inference.pipeline_cross_v3 import (
        CROSS_V3_PROMPT,
        CROSS_V3_REFERENCE_WITH_REF,
        _build_cross_v3_inference_conditions,
        _encode_images_to_latents,
    )

    with torch.no_grad():
        target_latent = _encode_images_to_latents(
            bundle.flux_pipeline.vae,
            tensors["target_image"].unsqueeze(0),
            bundle.torch_dtype,
        )
        control_tensor, correct_tokens = _build_cross_v3_inference_conditions(
            bundle,
            reference_image=tensors["reference_image"],
            reference_tissue_mask=tensors["reference_tissue_mask"],
            reference_nuclei_mask=tensors["reference_nuclei_mask"],
            target_tissue_mask=tensors["target_tissue_mask"],
            target_nuclei_mask=tensors["target_nuclei_mask"],
            reference_condition_mode=CROSS_V3_REFERENCE_WITH_REF,
        )
        _, mismatch_tokens = _build_cross_v3_inference_conditions(
            bundle,
            reference_image=tensors["mismatch_reference_image"],
            reference_tissue_mask=tensors["mismatch_reference_tissue_mask"],
            reference_nuclei_mask=tensors["mismatch_reference_nuclei_mask"],
            target_tissue_mask=tensors["target_tissue_mask"],
            target_nuclei_mask=tensors["target_nuclei_mask"],
            reference_condition_mode=CROSS_V3_REFERENCE_WITH_REF,
        )
        variant_tokens = {
            CORRECT_VARIANT: correct_tokens,
            ZERO_VARIANT: torch.zeros_like(correct_tokens),
            MISMATCH_VARIANT: mismatch_tokens,
            MEAN_VARIANT: correct_tokens.mean(dim=1, keepdim=True).expand_as(correct_tokens).contiguous(),
        }

    return fixed_timestep_usage_with_flux_controlnet(
        pipe=bundle.flux_pipeline,
        controlnet=bundle.controlnet,
        prompt=CROSS_V3_PROMPT,
        pixel_latents=target_latent,
        control_tensor=control_tensor,
        variant_tokens=variant_tokens,
        timesteps=timesteps,
        device=bundle.device,
        torch_dtype=bundle.torch_dtype,
        guidance_scale=bundle.guidance_scale,
        controlnet_conditioning_scale=bundle.controlnet_conditioning_scale,
        seed=seed,
    )


def fixed_timestep_usage_with_flux_controlnet(
    *,
    pipe,
    controlnet,
    prompt: str,
    pixel_latents: torch.Tensor,
    control_tensor: torch.Tensor,
    variant_tokens: dict[str, torch.Tensor],
    timesteps: list[float],
    device: str,
    torch_dtype: torch.dtype,
    guidance_scale: float,
    controlnet_conditioning_scale: float,
    seed: int,
) -> dict[str, dict[str, Any]]:
    from diffusers import FluxControlNetPipeline

    from controlnet_train.inference.pipeline_cross_v3 import (
        _format_timestep_key,
        _per_sample_mse,
        _prepare_packed_latent_image_ids,
        _sigma_for_timestep,
    )
    from controlnet_train.modules.cross_v3_conditioning import append_cross_v3_reference_context

    torch_device = torch.device(device)
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=[prompt],
        prompt_2=[prompt],
        device=torch_device,
    )
    if text_ids.dim() == 3:
        text_ids = text_ids[0]

    bsz = int(pixel_latents.shape[0])
    packed_pixel_latents = FluxControlNetPipeline._pack_latents(
        pixel_latents,
        bsz,
        pixel_latents.shape[1],
        pixel_latents.shape[2],
        pixel_latents.shape[3],
    )
    control_image = FluxControlNetPipeline._pack_latents(
        control_tensor,
        bsz,
        control_tensor.shape[1],
        control_tensor.shape[2],
        control_tensor.shape[3],
    )
    latent_image_ids = _prepare_packed_latent_image_ids(
        packed_height=pixel_latents.shape[2] // 2,
        packed_width=pixel_latents.shape[3] // 2,
        device=torch_device,
        dtype=torch_dtype,
    )
    controlnet_blocks_repeat = False if getattr(controlnet, "input_hint_block", None) is None else True
    generator = torch.Generator(device=torch_device).manual_seed(seed)
    noise = torch.randn(
        packed_pixel_latents.shape,
        generator=generator,
        device=packed_pixel_latents.device,
        dtype=packed_pixel_latents.dtype,
    )

    results: dict[str, dict[str, Any]] = {}
    for timestep_value in timesteps:
        timestep_key = _format_timestep_key(timestep_value)
        timestep = torch.tensor([timestep_value], device=torch_device, dtype=torch.float32)
        sigma = _sigma_for_timestep(
            pipe.scheduler,
            timestep,
            n_dim=packed_pixel_latents.ndim,
            dtype=packed_pixel_latents.dtype,
        )
        noisy_model_input = (1.0 - sigma) * packed_pixel_latents + sigma * noise
        expanded_timestep = timestep.expand(bsz).to(dtype=packed_pixel_latents.dtype)
        target_velocity = noise - packed_pixel_latents

        with torch.no_grad():
            guidance = None
            if controlnet.config.guidance_embeds:
                guidance = torch.full((bsz,), guidance_scale, device=torch_device)
            controlnet_block_samples, controlnet_single_block_samples = controlnet(
                hidden_states=noisy_model_input,
                controlnet_cond=control_image,
                controlnet_mode=None,
                conditioning_scale=controlnet_conditioning_scale,
                timestep=expanded_timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )

        timestep_result: dict[str, Any] = {"timestep": float(timestep_value), "sigma": float(sigma.detach().float().flatten()[0].cpu().item())}
        predictions: dict[str, torch.Tensor] = {}
        for variant in VARIANTS:
            tokens = variant_tokens[variant].to(device=torch_device, dtype=prompt_embeds.dtype)
            context, context_ids = append_cross_v3_reference_context(
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                reference_tokens=tokens,
            )
            compute_grad = variant == CORRECT_VARIANT
            if compute_grad:
                context = context.detach().clone().requires_grad_(True)
            else:
                context = context.detach()

            transformer_guidance = None
            if pipe.transformer.config.guidance_embeds:
                transformer_guidance = torch.full((bsz,), guidance_scale, device=torch_device)
            with torch.enable_grad():
                noise_pred = pipe.transformer(
                    hidden_states=noisy_model_input.detach(),
                    timestep=expanded_timestep / 1000,
                    guidance=transformer_guidance,
                    pooled_projections=pooled_prompt_embeds.detach(),
                    encoder_hidden_states=context,
                    controlnet_block_samples=(
                        [sample.detach().to(dtype=torch_dtype) for sample in controlnet_block_samples]
                        if controlnet_block_samples is not None
                        else None
                    ),
                    controlnet_single_block_samples=(
                        [sample.detach().to(dtype=torch_dtype) for sample in controlnet_single_block_samples]
                        if controlnet_single_block_samples is not None
                        else None
                    ),
                    txt_ids=context_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                    controlnet_blocks_repeat=controlnet_blocks_repeat,
                )[0]
                loss = _per_sample_mse(noise_pred, target_velocity.detach()).mean()
            timestep_result[f"{variant}_loss"] = float(loss.detach().cpu().item())
            predictions[variant] = noise_pred.detach().float().reshape(-1).cpu()
            if compute_grad:
                pipe.transformer.zero_grad(set_to_none=True)
                loss.backward()
                if context.grad is None:
                    raise RuntimeError("No gradient was produced for appended context embeddings.")
                timestep_result.update(
                    context_gradient_stats(
                        context=context.detach(),
                        grad=context.grad.detach(),
                        text_token_count=int(prompt_embeds.shape[1]),
                    )
                )

        for right in (ZERO_VARIANT, MISMATCH_VARIANT, MEAN_VARIANT):
            timestep_result[f"{right}_minus_correct_loss"] = (
                timestep_result[f"{right}_loss"] - timestep_result[f"{CORRECT_VARIANT}_loss"]
            )
            stats = compare_flat_tensors(predictions[CORRECT_VARIANT], predictions[right])
            for name, value in stats.items():
                timestep_result[f"noise_pred_correct_vs_{right}_{name}"] = value

        results[timestep_key] = timestep_result
    return results


def context_gradient_stats(*, context: torch.Tensor, grad: torch.Tensor, text_token_count: int) -> dict[str, float]:
    if context.shape != grad.shape:
        raise ValueError(f"context and grad shapes differ: {tuple(context.shape)} vs {tuple(grad.shape)}")
    if context.ndim != 3:
        raise ValueError(f"context must have shape (B, N, C), got {tuple(context.shape)}")
    if text_token_count <= 0 or text_token_count >= context.shape[1]:
        raise ValueError(f"text_token_count must split context tokens, got {text_token_count}")

    grad_f = grad.detach().float()
    context_f = context.detach().float()
    text_grad = grad_f[:, :text_token_count, :]
    ref_grad = grad_f[:, text_token_count:, :]
    text_context = context_f[:, :text_token_count, :]
    ref_context = context_f[:, text_token_count:, :]

    text_token_norm = torch.linalg.vector_norm(text_grad, dim=-1)
    ref_token_norm = torch.linalg.vector_norm(ref_grad, dim=-1)
    text_sum = float(text_token_norm.sum().item())
    ref_sum = float(ref_token_norm.sum().item())
    saliency = torch.sum(torch.abs(ref_grad * ref_context), dim=-1)
    ref_norm_distribution = ref_token_norm.reshape(-1)
    return {
        "grad_text_token_mean": float(text_token_norm.mean().item()),
        "grad_ref_token_mean": float(ref_token_norm.mean().item()),
        "grad_ref_vs_text_token_mean_ratio": float(ref_token_norm.mean().item() / max(float(text_token_norm.mean().item()), 1e-12)),
        "grad_text_token_sum": text_sum,
        "grad_ref_token_sum": ref_sum,
        "grad_ref_token_sum_share": float(ref_sum / max(ref_sum + text_sum, 1e-12)),
        "grad_text_l2": float(torch.linalg.vector_norm(text_grad).item()),
        "grad_ref_l2": float(torch.linalg.vector_norm(ref_grad).item()),
        "grad_ref_vs_text_l2_ratio": float(torch.linalg.vector_norm(ref_grad).item() / max(float(torch.linalg.vector_norm(text_grad).item()), 1e-12)),
        "grad_ref_nonzero_frac": float((ref_token_norm > 0).float().mean().item()),
        "grad_ref_entropy_norm": normalized_entropy(ref_norm_distribution),
        "grad_x_ref_token_abs_mean": float(saliency.mean().item()),
        "grad_x_ref_token_abs_max": float(saliency.max().item()),
        "context_ref_token_abs_mean": float(ref_context.abs().mean().item()),
        "context_text_token_abs_mean": float(text_context.abs().mean().item()),
    }


def normalized_entropy(values: torch.Tensor) -> float:
    flat = values.detach().float().reshape(-1)
    if flat.numel() <= 1:
        return 0.0
    total = float(flat.sum().item())
    if total <= 1e-12:
        return 0.0
    probs = torch.clamp(flat / total, min=1e-12)
    entropy = -torch.sum(probs * torch.log(probs))
    return float((entropy / math.log(flat.numel())).item())


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


def build_usage_row(
    *,
    index: int,
    record: dict[str, Any],
    mismatch: dict[str, Any],
    usage_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "index": index,
        "sample_id": record_id(record),
        "reference_sample_id": reference_record_id(record),
        "mismatch_reference_sample_id": reference_record_id(mismatch),
        "dataset": record.get("dataset", ""),
    }
    for timestep_key, result in sorted(usage_results.items()):
        for key, value in result.items():
            if isinstance(value, (float, int)):
                row[f"{key}_{timestep_key}"] = float(value)
    return row


def select_preview_timestep_key(usage_results: dict[str, dict[str, Any]]) -> str:
    if not usage_results:
        raise ValueError("No timestep results available.")
    selected = min(usage_results.items(), key=lambda item: abs(float(item[1]["timestep"]) - 700.0))
    return str(selected[0])


def build_usage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"num_samples": 0.0}
    summary: dict[str, Any] = {"num_samples": float(len(rows))}
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (float, int)) and key != "index"
        }
    )
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))

    aggregate_prefixes = (
        "zero_minus_correct_loss",
        "mismatch_minus_correct_loss",
        "token_mean_minus_correct_loss",
        "noise_pred_correct_vs_zero_relative_l2",
        "noise_pred_correct_vs_mismatch_relative_l2",
        "noise_pred_correct_vs_token_mean_relative_l2",
        "grad_ref_vs_text_token_mean_ratio",
        "grad_ref_token_sum_share",
        "grad_ref_entropy_norm",
    )
    for prefix in aggregate_prefixes:
        values = [
            float(row[key])
            for row in rows
            for key in numeric_keys
            if key.startswith(f"{prefix}_") and key in row and math.isfinite(float(row[key]))
        ]
        if values:
            summary[f"{prefix}_all_mean"] = float(np.mean(values))
            summary[f"{prefix}_all_std"] = float(np.std(values))
    return summary


def interpret_usage_summary(summary: dict[str, Any]) -> str:
    zero_delta = float(summary.get("zero_minus_correct_loss_all_mean", math.nan))
    mismatch_delta = float(summary.get("mismatch_minus_correct_loss_all_mean", math.nan))
    mean_delta = float(summary.get("token_mean_minus_correct_loss_all_mean", math.nan))
    pred_zero = float(summary.get("noise_pred_correct_vs_zero_relative_l2_all_mean", math.nan))
    pred_mismatch = float(summary.get("noise_pred_correct_vs_mismatch_relative_l2_all_mean", math.nan))
    grad_ratio = float(summary.get("grad_ref_vs_text_token_mean_ratio_all_mean", math.nan))
    grad_share = float(summary.get("grad_ref_token_sum_share_all_mean", math.nan))

    affects_output = (
        (math.isfinite(pred_zero) and pred_zero >= 0.02)
        or (math.isfinite(pred_mismatch) and pred_mismatch >= 0.02)
        or (math.isfinite(grad_ratio) and grad_ratio >= 0.10)
        or (math.isfinite(grad_share) and grad_share >= 0.10)
    )
    productive = (
        math.isfinite(zero_delta)
        and math.isfinite(mismatch_delta)
        and zero_delta > 0.01
        and mismatch_delta > 0.01
    )
    if productive:
        return "reference_tokens_used_productively"
    if not affects_output:
        return "reference_tokens_barely_used_by_transformer"
    if math.isfinite(zero_delta) and zero_delta > 0.005 and math.isfinite(mean_delta) and abs(mean_delta) <= 0.003:
        return "transformer_uses_global_reference_signal_more_than_token_texture"
    if affects_output:
        return "reference_tokens_affect_transformer_but_not_productively"
    return "mixed_or_weak_reference_token_usage"


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


def select_records(
    records: list[dict[str, Any]],
    *,
    reference_sample_ids: list[str],
    num_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    if reference_sample_ids:
        by_id = {reference_record_id(record): record for record in records}
        missing = [sample_id for sample_id in reference_sample_ids if sample_id not in by_id]
        if missing:
            raise ValueError(f"reference sample_id(s) not found: {missing}")
        return [by_id[sample_id] for sample_id in reference_sample_ids]
    if num_samples <= 0 or num_samples >= len(records):
        return list(records)
    selected = list(records)
    random.Random(seed).shuffle(selected)
    return selected[:num_samples]


def select_mismatch_record(records: list[dict[str, Any]], *, anchor: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    anchor_id = reference_record_id(anchor)
    candidates = [record for record in records if reference_record_id(record) != anchor_id]
    if not candidates:
        raise ValueError("Need at least two unique references for mismatch-token probing.")
    return rng.choice(candidates)


def record_id(record: dict[str, Any]) -> str:
    return str(record.get("sample_id") or Path(record.get("target_image", record["reference_image"])).stem)


def reference_record_id(record: dict[str, Any]) -> str:
    return str(record.get("reference_sample_id") or Path(record["reference_image"]).stem)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())

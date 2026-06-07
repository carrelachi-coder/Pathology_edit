"""Trace Cross V2.1 z_ref contribution after ControlNet x_embedder.

This diagnostic compares two otherwise identical ControlNet passes:

* full: the packed condition is used as-is.
* no_z_ref: only the packed z_ref columns are removed at
  controlnet_x_embedder input.

Because controlnet_x_embedder is linear, the z_ref activation injected into
the image stream is measured exactly as ``full_cond_emb - no_z_ref_cond_emb``.
The script then runs both streams through the double blocks, single blocks, and
final ControlNet output projections, reporting how much the activations change.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


EPS = 1e-12


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trace Cross V2.1 z_ref activation contribution through Flux ControlNet."
    )
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--checkpoint", required=True, help="Cross V2.1 checkpoint dir.")
    parser.add_argument("--metadata", required=True, help="metadata_cross_{train,val}.json path.")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Inspect exactly one metadata index. Overrides --num-samples selection.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-seed", type=int, default=1234)
    parser.add_argument("--timesteps", default="100,500,900")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--prompt-source", choices=["metadata", "dataset"], default="dataset")
    parser.add_argument("--prompt", default=None, help="Override every sample with one prompt.")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Drop per-layer rows from the JSON and keep compact summaries.",
    )
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    from controlnet_train.data.common import (
        default_prompt_for_dataset,
        load_image_tensor,
        load_nuclei_mask,
        load_tissue_mask,
    )
    from controlnet_train.inference.pipeline_cross_v2_1 import (
        _encode_images_to_latents,
        load_cross_v2_1_bundle,
    )

    dtype_by_name = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_by_name[args.torch_dtype]
    timestep_values = parse_timestep_values(args.timesteps)
    records = read_cross_metadata(args.metadata)
    selected_records = select_records(
        records,
        sample_index=args.sample_index,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    bundle = load_cross_v2_1_bundle(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        checkpoint_path=args.checkpoint,
        device=args.device,
        torch_dtype=torch_dtype,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
    )
    bundle.controlnet.eval()
    bundle.flux_pipeline.vae.eval()
    for module in bundle.condition_modules.values():
        module.eval()

    sample_reports = []
    for selection_index, (record_index, record) in enumerate(selected_records):
        prompt = resolve_prompt(
            record=record,
            prompt_override=args.prompt,
            prompt_source=args.prompt_source,
            default_prompt_for_dataset=default_prompt_for_dataset,
        )
        report = diagnose_record(
            bundle=bundle,
            record=record,
            record_index=record_index,
            prompt=prompt,
            timesteps=timestep_values,
            noise_seed=int(args.noise_seed) + selection_index,
            torch_dtype=torch_dtype,
            load_image_tensor=load_image_tensor,
            load_tissue_mask=load_tissue_mask,
            load_nuclei_mask=load_nuclei_mask,
            encode_images_to_latents=_encode_images_to_latents,
            summary_only=bool(args.summary_only),
        )
        sample_reports.append(report)

    output = {
        "checkpoint": str(args.checkpoint),
        "pretrained_model_name_or_path": str(args.pretrained_model_name_or_path),
        "metadata": str(args.metadata),
        "device": str(bundle.device),
        "torch_dtype": str(torch_dtype).replace("torch.", ""),
        "timesteps": timestep_values,
        "control_spec": {
            "reference_latent_channels": int(bundle.control_spec.reference_latent_channels),
            "raw_channels": int(bundle.control_spec.raw_channels),
            "packed_channels": int(bundle.control_spec.packed_channels),
            "packed_reference_latent_channels": int(
                bundle.control_spec.packed_reference_latent_channels
            ),
            "packed_reference_mask_start": int(bundle.control_spec.packed_reference_mask_start),
            "packed_target_mask_start": int(bundle.control_spec.packed_target_mask_start),
        },
        "summary": aggregate_sample_reports(sample_reports),
        "samples": sample_reports,
    }

    text = json.dumps(output, indent=2, ensure_ascii=False, allow_nan=True)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf8")
    return 0


@torch.inference_mode()
def diagnose_record(
    *,
    bundle,
    record: dict[str, Any],
    record_index: int,
    prompt: str,
    timesteps: list[float],
    noise_seed: int,
    torch_dtype: torch.dtype,
    load_image_tensor,
    load_tissue_mask,
    load_nuclei_mask,
    encode_images_to_latents,
    summary_only: bool,
) -> dict[str, Any]:
    from controlnet_train.modules.cross_v2_1_conditioning import build_cross_v2_1_condition

    torch_device = torch.device(bundle.device)
    reference_image = load_image_tensor(record["reference_image"]).unsqueeze(0)
    target_image = load_image_tensor(record["target_image"]).unsqueeze(0)
    reference_tissue_mask = load_tissue_mask(record["reference_tissue_mask"]).unsqueeze(0)
    reference_nuclei_mask = load_nuclei_mask(record["reference_nuclei_mask"]).unsqueeze(0)
    target_tissue_mask = load_tissue_mask(record["target_tissue_mask"]).unsqueeze(0)
    target_nuclei_mask = load_nuclei_mask(record["target_nuclei_mask"]).unsqueeze(0)

    vae = bundle.flux_pipeline.vae
    target_latent = encode_images_to_latents(vae, target_image, torch_dtype)
    z_ref = encode_images_to_latents(vae, reference_image, torch_dtype)
    ref_tissue_feat = bundle.condition_modules["tissue_downsampler"](
        bundle.condition_modules["hte"](reference_tissue_mask.to(device=torch_device))
    ).to(dtype=torch_dtype)
    ref_nuclei_feat = bundle.condition_modules["nuclei_encoder"](
        reference_nuclei_mask.to(device=torch_device)
    ).to(dtype=torch_dtype)
    tar_tissue_feat = bundle.condition_modules["tissue_downsampler"](
        bundle.condition_modules["hte"](target_tissue_mask.to(device=torch_device))
    ).to(dtype=torch_dtype)
    tar_nuclei_feat = bundle.condition_modules["nuclei_encoder"](
        target_nuclei_mask.to(device=torch_device)
    ).to(dtype=torch_dtype)
    control_tensor = build_cross_v2_1_condition(
        z_ref=z_ref,
        ref_tissue_feat=ref_tissue_feat,
        ref_nuclei_feat=ref_nuclei_feat,
        tar_tissue_feat=tar_tissue_feat,
        tar_nuclei_feat=tar_nuclei_feat,
    )

    bsz = int(target_latent.shape[0])
    packed_target_latent = pack_latents(
        target_latent,
        bsz,
        int(target_latent.shape[1]),
        int(target_latent.shape[2]),
        int(target_latent.shape[3]),
    )
    packed_control = pack_latents(
        control_tensor,
        bsz,
        int(control_tensor.shape[1]),
        int(control_tensor.shape[2]),
        int(control_tensor.shape[3]),
    )
    latent_image_ids = prepare_packed_latent_image_ids(
        packed_height=int(target_latent.shape[2]) // 2,
        packed_width=int(target_latent.shape[3]) // 2,
        device=torch_device,
        dtype=torch_dtype,
    )
    prompt_embeds, pooled_prompt_embeds, text_ids = bundle.flux_pipeline.encode_prompt(
        prompt=[prompt],
        prompt_2=[prompt],
        device=torch_device,
    )
    if text_ids.dim() == 3:
        text_ids = text_ids[0]

    generator = torch.Generator(device=torch_device).manual_seed(noise_seed)
    noise = torch.randn(
        packed_target_latent.shape,
        generator=generator,
        device=packed_target_latent.device,
        dtype=packed_target_latent.dtype,
    )

    timestep_reports = []
    for timestep_value in timesteps:
        timestep = torch.tensor([timestep_value], device=torch_device, dtype=torch.float32)
        sigma = sigma_for_timestep(
            bundle.flux_pipeline.scheduler,
            timestep,
            n_dim=packed_target_latent.ndim,
            dtype=packed_target_latent.dtype,
        )
        noisy_model_input = (1.0 - sigma) * packed_target_latent + sigma * noise
        expanded_timestep = timestep.expand(bsz).to(dtype=packed_target_latent.dtype) / 1000
        guidance = None
        if bundle.controlnet.config.guidance_embeds:
            guidance = torch.full((bsz,), bundle.guidance_scale, device=torch_device)

        trace = trace_controlnet_z_ref_contribution(
            controlnet=bundle.controlnet,
            hidden_states=noisy_model_input,
            controlnet_cond=packed_control,
            z_width=int(bundle.control_spec.packed_reference_latent_channels),
            conditioning_scale=float(bundle.controlnet_conditioning_scale),
            timestep=expanded_timestep,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
        )
        timestep_reports.append(
            {
                "timestep": float(timestep_value),
                "sigma_mean": float(sigma.float().mean().item()),
                **trim_trace_for_output(trace, summary_only=summary_only),
            }
        )

    sample_report = {
        "record_index": int(record_index),
        "sample_id": str(record.get("sample_id") or Path(record["target_image"]).stem),
        "reference_sample_id": str(
            record.get("reference_sample_id") or Path(record["reference_image"]).stem
        ),
        "prompt": prompt,
        "control_tensor_shape": list(control_tensor.shape),
        "packed_control_shape": list(packed_control.shape),
        "control_input_stats": control_input_stats(
            control_tensor=control_tensor,
            packed_control=packed_control,
            raw_z_channels=int(bundle.control_spec.reference_latent_channels),
            packed_z_width=int(bundle.control_spec.packed_reference_latent_channels),
        ),
        "summary": aggregate_timestep_reports(timestep_reports),
        "timesteps": timestep_reports,
    }
    return sample_report


def trace_controlnet_z_ref_contribution(
    *,
    controlnet,
    hidden_states: torch.Tensor,
    controlnet_cond: torch.Tensor,
    z_width: int,
    conditioning_scale: float,
    timestep: torch.Tensor,
    guidance: torch.Tensor | None,
    pooled_projections: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    txt_ids: torch.Tensor,
    img_ids: torch.Tensor,
) -> dict[str, Any]:
    """Run full/no-z_ref streams through ControlNet and record differences."""

    if getattr(controlnet, "input_hint_block", None) is not None:
        raise NotImplementedError(
            "z_ref activation tracing currently supports direct controlnet_x_embedder "
            "conditions only; input_hint_block would mix channels before x_embedder."
        )

    image_base = controlnet.x_embedder(hidden_states)
    cond_full, cond_no_z, cond_z = split_controlnet_x_embedder_embedding(
        controlnet.controlnet_x_embedder,
        controlnet_cond,
        z_width=z_width,
    )
    image_full = image_base + cond_full
    image_no_z = image_base + cond_no_z

    timestep = timestep.to(image_full.dtype) * 1000
    guidance_for_temb = guidance.to(image_full.dtype) * 1000 if guidance is not None else None
    if guidance_for_temb is None:
        temb = controlnet.time_text_embed(timestep, pooled_projections)
    else:
        temb = controlnet.time_text_embed(timestep, guidance_for_temb, pooled_projections)

    text_base = controlnet.context_embedder(encoder_hidden_states)
    text_full = text_base
    text_no_z = text_base.clone()

    if getattr(controlnet, "union", False):
        raise NotImplementedError("ControlNet-Union mode is not supported by this z_ref diagnostic.")

    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        img_ids = img_ids[0]
    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = controlnet.pos_embed(ids)

    trace: dict[str, Any] = {
        "x_embedder": {
            "z_width": int(z_width),
            "base_image_hidden": tensor_stats(image_base),
            "full_control_embedding": tensor_stats(cond_full),
            "no_z_ref_control_embedding": tensor_stats(cond_no_z),
            "z_ref_control_embedding": tensor_stats(cond_z),
            "z_ref_vs_full_control_embedding": contribution_metrics(cond_full, cond_no_z),
            "z_ref_vs_injected_image_hidden": contribution_metrics(image_full, image_no_z),
        },
        "double_blocks": [],
        "single_blocks": [],
        "controlnet_blocks": [],
        "controlnet_single_blocks": [],
    }

    block_samples_full = []
    block_samples_no_z = []
    for index, block in enumerate(controlnet.transformer_blocks):
        text_full, image_full = block(
            hidden_states=image_full,
            encoder_hidden_states=text_full,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
        )
        text_no_z, image_no_z = block(
            hidden_states=image_no_z,
            encoder_hidden_states=text_no_z,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
        )
        block_samples_full.append(image_full)
        block_samples_no_z.append(image_no_z)
        trace["double_blocks"].append(
            {
                "index": int(index),
                "image_hidden": contribution_metrics(image_full, image_no_z),
                "text_hidden": contribution_metrics(text_full, text_no_z),
            }
        )

    text_seq_len = int(text_full.shape[1])
    combined_full = torch.cat([text_full, image_full], dim=1)
    combined_no_z = torch.cat([text_no_z, image_no_z], dim=1)

    single_samples_full = []
    single_samples_no_z = []
    for index, block in enumerate(controlnet.single_transformer_blocks):
        combined_full = block(
            hidden_states=combined_full,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
        )
        combined_no_z = block(
            hidden_states=combined_no_z,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
        )
        image_single_full = combined_full[:, text_seq_len:]
        image_single_no_z = combined_no_z[:, text_seq_len:]
        single_samples_full.append(image_single_full)
        single_samples_no_z.append(image_single_no_z)
        trace["single_blocks"].append(
            {
                "index": int(index),
                "combined_hidden": contribution_metrics(combined_full, combined_no_z),
                "image_hidden": contribution_metrics(image_single_full, image_single_no_z),
            }
        )

    control_outputs_full = []
    control_outputs_no_z = []
    for index, (full_sample, no_z_sample, controlnet_block) in enumerate(
        zip(block_samples_full, block_samples_no_z, controlnet.controlnet_blocks)
    ):
        out_full = controlnet_block(full_sample) * conditioning_scale
        out_no_z = controlnet_block(no_z_sample) * conditioning_scale
        control_outputs_full.append(out_full)
        control_outputs_no_z.append(out_no_z)
        trace["controlnet_blocks"].append(
            {
                "index": int(index),
                "output": contribution_metrics(out_full, out_no_z),
            }
        )

    single_outputs_full = []
    single_outputs_no_z = []
    for index, (full_sample, no_z_sample, controlnet_block) in enumerate(
        zip(single_samples_full, single_samples_no_z, controlnet.controlnet_single_blocks)
    ):
        out_full = controlnet_block(full_sample) * conditioning_scale
        out_no_z = controlnet_block(no_z_sample) * conditioning_scale
        single_outputs_full.append(out_full)
        single_outputs_no_z.append(out_no_z)
        trace["controlnet_single_blocks"].append(
            {
                "index": int(index),
                "output": contribution_metrics(out_full, out_no_z),
            }
        )

    trace["final_outputs"] = {
        "double_controlnet_outputs": tensor_list_contribution_metrics(
            control_outputs_full,
            control_outputs_no_z,
        ),
        "single_controlnet_outputs": tensor_list_contribution_metrics(
            single_outputs_full,
            single_outputs_no_z,
        ),
        "all_controlnet_outputs": tensor_list_contribution_metrics(
            [*control_outputs_full, *single_outputs_full],
            [*control_outputs_no_z, *single_outputs_no_z],
        ),
    }
    trace["summary"] = summarize_trace(trace)
    return trace


def split_controlnet_x_embedder_embedding(
    x_embedder: torch.nn.Linear,
    controlnet_cond: torch.Tensor,
    *,
    z_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return full, no-z_ref, and z_ref-only control embeddings.

    ``no_z_ref`` keeps the x_embedder bias and all non-z_ref input columns.
    ``z_ref`` is bias-free, so ``full == no_z_ref + z_ref`` up to floating
    point roundoff.
    """

    if z_width < 0 or z_width > x_embedder.in_features:
        raise ValueError(
            f"z_width must be in [0, {x_embedder.in_features}], got {z_width}."
        )
    z_input = controlnet_cond[..., :z_width]
    non_z_input = controlnet_cond[..., z_width:]
    z_weight = x_embedder.weight[:, :z_width]
    non_z_weight = x_embedder.weight[:, z_width:]

    if z_width == 0:
        z_embedding = torch.zeros(
            *controlnet_cond.shape[:-1],
            x_embedder.out_features,
            device=controlnet_cond.device,
            dtype=controlnet_cond.dtype,
        )
    else:
        z_embedding = F.linear(z_input, z_weight, None)
    no_z_embedding = F.linear(non_z_input, non_z_weight, x_embedder.bias)
    full_embedding = no_z_embedding + z_embedding
    return full_embedding, no_z_embedding, z_embedding


def contribution_metrics(full: torch.Tensor, baseline: torch.Tensor) -> dict[str, float]:
    """Measure how much full changes when baseline removes z_ref."""

    full_f = full.detach().float()
    baseline_f = baseline.detach().float()
    delta = full_f - baseline_f
    full_norm = float(torch.linalg.vector_norm(full_f).item())
    baseline_norm = float(torch.linalg.vector_norm(baseline_f).item())
    delta_norm = float(torch.linalg.vector_norm(delta).item())
    full_abs_mean = float(full_f.abs().mean().item())
    baseline_abs_mean = float(baseline_f.abs().mean().item())
    delta_abs_mean = float(delta.abs().mean().item())
    dot = float((full_f.flatten() * baseline_f.flatten()).sum().item())
    denom = full_norm * baseline_norm
    return {
        "full_norm": full_norm,
        "baseline_norm": baseline_norm,
        "delta_norm": delta_norm,
        "delta_over_full_norm": safe_div(delta_norm, full_norm),
        "delta_over_baseline_norm": safe_div(delta_norm, baseline_norm),
        "delta_energy_over_full_energy": safe_div(delta_norm * delta_norm, full_norm * full_norm),
        "full_abs_mean": full_abs_mean,
        "baseline_abs_mean": baseline_abs_mean,
        "delta_abs_mean": delta_abs_mean,
        "delta_over_full_abs_mean": safe_div(delta_abs_mean, full_abs_mean),
        "delta_over_baseline_abs_mean": safe_div(delta_abs_mean, baseline_abs_mean),
        "cosine_full_baseline": safe_div(dot, denom),
    }


def tensor_list_contribution_metrics(
    full_tensors: Iterable[torch.Tensor],
    baseline_tensors: Iterable[torch.Tensor],
) -> dict[str, float]:
    full_norm_sq = 0.0
    baseline_norm_sq = 0.0
    delta_norm_sq = 0.0
    dot = 0.0
    full_abs_sum = 0.0
    baseline_abs_sum = 0.0
    delta_abs_sum = 0.0
    numel = 0
    count = 0
    for full, baseline in zip(full_tensors, baseline_tensors):
        full_f = full.detach().float()
        baseline_f = baseline.detach().float()
        delta = full_f - baseline_f
        full_norm_sq += float(full_f.pow(2).sum().item())
        baseline_norm_sq += float(baseline_f.pow(2).sum().item())
        delta_norm_sq += float(delta.pow(2).sum().item())
        dot += float((full_f.flatten() * baseline_f.flatten()).sum().item())
        full_abs_sum += float(full_f.abs().sum().item())
        baseline_abs_sum += float(baseline_f.abs().sum().item())
        delta_abs_sum += float(delta.abs().sum().item())
        numel += int(full_f.numel())
        count += 1

    full_norm = math.sqrt(full_norm_sq)
    baseline_norm = math.sqrt(baseline_norm_sq)
    delta_norm = math.sqrt(delta_norm_sq)
    full_abs_mean = safe_div(full_abs_sum, numel)
    baseline_abs_mean = safe_div(baseline_abs_sum, numel)
    delta_abs_mean = safe_div(delta_abs_sum, numel)
    return {
        "tensor_count": int(count),
        "full_norm": full_norm,
        "baseline_norm": baseline_norm,
        "delta_norm": delta_norm,
        "delta_over_full_norm": safe_div(delta_norm, full_norm),
        "delta_over_baseline_norm": safe_div(delta_norm, baseline_norm),
        "delta_energy_over_full_energy": safe_div(delta_norm_sq, full_norm_sq),
        "full_abs_mean": full_abs_mean,
        "baseline_abs_mean": baseline_abs_mean,
        "delta_abs_mean": delta_abs_mean,
        "delta_over_full_abs_mean": safe_div(delta_abs_mean, full_abs_mean),
        "delta_over_baseline_abs_mean": safe_div(delta_abs_mean, baseline_abs_mean),
        "cosine_full_baseline": safe_div(dot, full_norm * baseline_norm),
    }


def tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    value = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "norm": float(torch.linalg.vector_norm(value).item()),
        "abs_mean": float(value.abs().mean().item()),
        "mean": float(value.mean().item()),
        "std": float(value.std(unbiased=False).item()),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
    }


def control_input_stats(
    *,
    control_tensor: torch.Tensor,
    packed_control: torch.Tensor,
    raw_z_channels: int,
    packed_z_width: int,
) -> dict[str, Any]:
    raw_z = control_tensor[:, :raw_z_channels]
    raw_mask = control_tensor[:, raw_z_channels:]
    packed_z = packed_control[..., :packed_z_width]
    packed_mask = packed_control[..., packed_z_width:]
    return {
        "raw_z_ref": tensor_stats(raw_z),
        "raw_masks": tensor_stats(raw_mask),
        "raw_z_over_mask_abs_mean": safe_div(
            float(raw_z.detach().float().abs().mean().item()),
            float(raw_mask.detach().float().abs().mean().item()),
        ),
        "packed_z_ref": tensor_stats(packed_z),
        "packed_masks": tensor_stats(packed_mask),
        "packed_z_over_mask_abs_mean": safe_div(
            float(packed_z.detach().float().abs().mean().item()),
            float(packed_mask.detach().float().abs().mean().item()),
        ),
    }


def summarize_trace(trace: dict[str, Any]) -> dict[str, float]:
    double_last = last_metric(trace.get("double_blocks", []), "image_hidden")
    single_last = last_metric(trace.get("single_blocks", []), "image_hidden")
    control_outputs = trace["final_outputs"]["double_controlnet_outputs"]
    single_outputs = trace["final_outputs"]["single_controlnet_outputs"]
    all_outputs = trace["final_outputs"]["all_controlnet_outputs"]
    x_metrics = trace["x_embedder"]["z_ref_vs_injected_image_hidden"]
    control_embed_metrics = trace["x_embedder"]["z_ref_vs_full_control_embedding"]
    return {
        "x_z_over_full_control_embedding_norm": control_embed_metrics["delta_over_full_norm"],
        "x_z_over_injected_image_hidden_norm": x_metrics["delta_over_full_norm"],
        "double_last_image_delta_over_full_norm": double_last.get("delta_over_full_norm", math.nan),
        "single_last_image_delta_over_full_norm": single_last.get("delta_over_full_norm", math.nan),
        "controlnet_double_outputs_delta_over_full_norm": control_outputs["delta_over_full_norm"],
        "controlnet_single_outputs_delta_over_full_norm": single_outputs["delta_over_full_norm"],
        "controlnet_all_outputs_delta_over_full_norm": all_outputs["delta_over_full_norm"],
        "controlnet_all_outputs_delta_over_full_abs_mean": all_outputs[
            "delta_over_full_abs_mean"
        ],
        "controlnet_all_outputs_cosine_full_no_z": all_outputs["cosine_full_baseline"],
    }


def trim_trace_for_output(trace: dict[str, Any], *, summary_only: bool) -> dict[str, Any]:
    if not summary_only:
        return trace
    return {
        "x_embedder": trace["x_embedder"],
        "final_outputs": trace["final_outputs"],
        "summary": trace["summary"],
    }


def aggregate_timestep_reports(timestep_reports: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "x_z_over_full_control_embedding_norm",
        "x_z_over_injected_image_hidden_norm",
        "double_last_image_delta_over_full_norm",
        "single_last_image_delta_over_full_norm",
        "controlnet_double_outputs_delta_over_full_norm",
        "controlnet_single_outputs_delta_over_full_norm",
        "controlnet_all_outputs_delta_over_full_norm",
        "controlnet_all_outputs_delta_over_full_abs_mean",
        "controlnet_all_outputs_cosine_full_no_z",
    ]
    rows = [row.get("summary", {}) for row in timestep_reports]
    return aggregate_numeric_rows(rows, keys)


def aggregate_sample_reports(sample_reports: list[dict[str, Any]]) -> dict[str, float]:
    if not sample_reports:
        return {}

    flattened_rows = []
    for report in sample_reports:
        for timestep_report in report.get("timesteps", []):
            flattened_rows.append(timestep_report.get("summary", {}))
    trace_keys = sorted({key for row in flattened_rows for key in row})
    return aggregate_numeric_rows(flattened_rows, trace_keys)


def aggregate_numeric_rows(rows: list[dict[str, Any]], keys: Iterable[str]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in keys:
        values = [
            float(row[key])
            for row in rows
            if key in row and isinstance(row[key], (int, float)) and math.isfinite(float(row[key]))
        ]
        if values:
            summary[f"{key}_mean"] = float(sum(values) / len(values))
            variance = sum((value - summary[f"{key}_mean"]) ** 2 for value in values) / len(values)
            summary[f"{key}_std"] = float(math.sqrt(variance))
            summary[f"{key}_min"] = float(min(values))
            summary[f"{key}_max"] = float(max(values))
    return summary


def last_metric(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    if not rows:
        return {}
    value = rows[-1].get(key, {})
    return value if isinstance(value, dict) else {}


def pack_latents(
    latents: torch.Tensor,
    batch_size: int,
    num_channels_latents: int,
    height: int,
    width: int,
) -> torch.Tensor:
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)


def prepare_packed_latent_image_ids(
    *,
    packed_height: int,
    packed_width: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    latent_image_ids = torch.zeros(packed_height, packed_width, 3, device=device, dtype=dtype)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(
        packed_height,
        device=device,
        dtype=dtype,
    )[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(
        packed_width,
        device=device,
        dtype=dtype,
    )[None, :]
    return latent_image_ids.reshape(packed_height * packed_width, 3)


def sigma_for_timestep(
    scheduler,
    timestep: torch.Tensor,
    *,
    n_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    sigmas = scheduler.sigmas.to(device=timestep.device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device=timestep.device, dtype=timestep.dtype)
    step_indices = []
    for value in timestep:
        matches = (schedule_timesteps == value).nonzero()
        if matches.numel() > 0:
            step_indices.append(int(matches[0].item()))
        else:
            step_indices.append(int(torch.argmin(torch.abs(schedule_timesteps - value)).item()))
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def parse_timestep_values(value: str) -> list[float]:
    timesteps = []
    for part in str(value).split(","):
        stripped = part.strip()
        if stripped:
            timesteps.append(float(stripped))
    if not timesteps:
        raise ValueError("--timesteps must contain at least one numeric value")
    return timesteps


def read_cross_metadata(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf8"))
    if isinstance(payload, dict):
        records = payload.get("pairs")
        if not isinstance(records, list):
            raise ValueError("cross metadata dict must contain a 'pairs' list")
        return records
    if isinstance(payload, list):
        return payload
    raise TypeError(f"unsupported cross metadata payload type: {type(payload)!r}")


def select_records(
    records: list[dict[str, Any]],
    *,
    sample_index: int | None,
    num_samples: int,
    seed: int,
) -> list[tuple[int, dict[str, Any]]]:
    if not records:
        raise ValueError("metadata contains no records")
    if sample_index is not None:
        index = int(sample_index) % len(records)
        return [(index, records[index])]
    indexed = list(enumerate(records))
    if num_samples is None or num_samples <= 0 or num_samples >= len(indexed):
        return indexed
    random.Random(seed).shuffle(indexed)
    return indexed[:num_samples]


def resolve_prompt(
    *,
    record: dict[str, Any],
    prompt_override: str | None,
    prompt_source: str,
    default_prompt_for_dataset,
) -> str:
    if prompt_override:
        return prompt_override
    if prompt_source == "metadata" and record.get("prompt"):
        return str(record["prompt"])
    return default_prompt_for_dataset(record.get("dataset"))


def safe_div(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) <= EPS:
        return math.nan
    return float(numerator) / float(denominator)


if __name__ == "__main__":
    raise SystemExit(main())

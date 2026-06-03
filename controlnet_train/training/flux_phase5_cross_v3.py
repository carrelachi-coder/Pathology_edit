"""Phase 5.3 Cross V3 training for Flux ControlNet.

Cross V3 separates structure and appearance:
- fixed text prompt: ``histopathology image``
- ControlNet: target tissue/nuclei masks only
- FLUX joint cross-attention: projected ``z_ref + ref masks`` tokens
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import Callable

import accelerate
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxControlNetModel,
    FluxControlNetPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, T5EncoderModel

from dataset_config import NUM_COARSE, NUM_FINE
from controlnet_train.data import CrossReconstructionDataset
from controlnet_train.modules import (
    FixedOneHotTissueEncoder,
    HierarchicalTissueEmbedding,
    NucleiConditionEncoder,
    TissueConditionDownsampler,
)
from controlnet_train.modules.cross_v3_conditioning import (
    CROSS_V3_PROMPT,
    CrossV3ControlSpec,
    CrossV3ReferenceContextEncoder,
    CrossV3ReferenceSpec,
    append_cross_v3_reference_context,
    build_cross_v3_control_condition,
    deterministic_latent_from_posterior,
)
from controlnet_train.modules.cross_v4_conditioning import (
    CrossV4ControlSpec,
    CrossV4CorrespondenceBiasConfig,
    CrossV4PriorTokenBank,
    CrossV4ReferenceContextEncoder,
    CrossV4ReferenceEncoding,
    CrossV4ReferenceSpec,
    append_cross_v4_context,
    apply_cross_v4_reference_encoding_mode,
    build_cross_v4_control_condition,
    build_cross_v4_correspondence_bias,
    build_cross_v4_token_metadata,
)
from controlnet_train.training.cross_v4_attention import (
    install_cross_v4_attention_processors,
    parse_cross_v4_block_indices,
)
from controlnet_train.training.conditioning import patch_controlnet_x_embedder
from controlnet_train.training.cross_v1_losses import (
    RegionalStainStyleLossConfig,
    ref_swap_sensitivity_loss,
    regional_stain_style_loss,
    unpack_flux_packed_latents,
)

if is_wandb_available():
    import wandb  # noqa: F401

logger = get_logger(__name__)
if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


def collate_cross_v3_batch(examples: list[dict]) -> dict:
    flattened: list[dict] = []
    for item in examples:
        if "paired_counterfactual" in item:
            flattened.extend(item["paired_counterfactual"])
        else:
            flattened.append(item)
    examples = flattened
    return {
        "sample_ids": [item["sample_id"] for item in examples],
        "reference_sample_ids": [item["reference_sample_id"] for item in examples],
        "target_image": torch.stack([item["target_image"] for item in examples]),
        "reference_image": torch.stack([item["reference_image"] for item in examples]),
        "target_tissue_mask": torch.stack([item["target_tissue_mask"] for item in examples]),
        "target_nuclei_mask": torch.stack([item["target_nuclei_mask"] for item in examples]),
        "reference_tissue_mask": torch.stack([item["reference_tissue_mask"] for item in examples]),
        "reference_nuclei_mask": torch.stack([item["reference_nuclei_mask"] for item in examples]),
        "prompts": [item["prompt"] for item in examples],
        "sample_modes": [item.get("sample_mode", "cross") for item in examples],
    }


def _configure_controlnet_trainable_params(
    controlnet: nn.Module,
    *,
    mode: str,
    train_x_embedder: bool = False,
    train_last_n_blocks: int = 0,
    train_last_n_single_blocks: int = 0,
) -> list[str]:
    mode = str(mode or "all").strip().lower()
    if mode not in {"all", "outputs"}:
        raise ValueError(f"Unsupported ControlNet train mode {mode!r}; choose 'all' or 'outputs'.")

    if mode == "all":
        controlnet.requires_grad_(True)
        return [name for name, param in controlnet.named_parameters() if param.requires_grad]

    controlnet.requires_grad_(False)
    _set_module_requires_grad(getattr(controlnet, "controlnet_blocks", None), True)
    _set_module_requires_grad(getattr(controlnet, "controlnet_single_blocks", None), True)
    if train_x_embedder:
        _set_module_requires_grad(getattr(controlnet, "controlnet_x_embedder", None), True)
    _set_last_n_modules_requires_grad(getattr(controlnet, "transformer_blocks", None), train_last_n_blocks)
    _set_last_n_modules_requires_grad(
        getattr(controlnet, "single_transformer_blocks", None),
        train_last_n_single_blocks,
    )
    return [name for name, param in controlnet.named_parameters() if param.requires_grad]


def _set_module_requires_grad(module: nn.Module | None, value: bool) -> None:
    if module is not None:
        module.requires_grad_(value)


def _set_last_n_modules_requires_grad(modules: nn.Module | None, count: int) -> None:
    if modules is None:
        return
    count = max(0, int(count or 0))
    if count == 0:
        return
    for module in list(modules)[-count:]:
        module.requires_grad_(True)


def _encode_images_to_latents(vae: AutoencoderKL, images: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    device = next(vae.parameters()).device
    images = images.to(device=device, dtype=dtype)
    images = images * 2.0 - 1.0
    latents = vae.encode(images).latent_dist.sample()
    return (latents - vae.config.shift_factor) * vae.config.scaling_factor


def _encode_images_to_deterministic_latents(
    vae: AutoencoderKL,
    images: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    device = next(vae.parameters()).device
    images = images.to(device=device, dtype=dtype)
    images = images * 2.0 - 1.0
    posterior = vae.encode(images).latent_dist
    latents = deterministic_latent_from_posterior(posterior)
    return (latents - vae.config.shift_factor) * vae.config.scaling_factor


def _build_cross_v3_control_batch(
    *,
    batch: dict,
    modules: dict[str, nn.Module],
    vae: AutoencoderKL,
    weight_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    device = next(vae.parameters()).device
    target_image_latent = _encode_images_to_latents(vae, batch["target_image"], weight_dtype)
    reference_image_latent = _encode_images_to_deterministic_latents(
        vae,
        batch["reference_image"],
        weight_dtype,
    )

    ref_tissue_feat = modules["tissue_downsampler"](
        modules["hte"](batch["reference_tissue_mask"].to(device=device))
    ).to(dtype=weight_dtype)
    ref_nuclei_feat = modules["nuclei_encoder"](
        batch["reference_nuclei_mask"].to(device=device)
    ).to(dtype=weight_dtype)
    tar_tissue_feat = _target_tissue_downsampler(modules)(
        _target_hte(modules)(batch["target_tissue_mask"].to(device=device))
    ).to(dtype=weight_dtype)
    tar_nuclei_feat = modules["nuclei_encoder"](
        batch["target_nuclei_mask"].to(device=device)
    ).to(dtype=weight_dtype)

    control_tensor = build_cross_v3_control_condition(
        tar_tissue_feat=tar_tissue_feat,
        tar_nuclei_feat=tar_nuclei_feat,
    )
    reference_tokens = modules["reference_context_encoder"](
        z_ref=reference_image_latent,
        ref_tissue_feat=ref_tissue_feat,
        ref_nuclei_feat=ref_nuclei_feat,
        ref_tissue_ids=batch["reference_tissue_mask"].to(device=device),
    )
    feature_stats = _cross_v3_feature_stats(
        tar_tissue_feat=tar_tissue_feat,
        tar_nuclei_feat=tar_nuclei_feat,
        ref_tissue_feat=ref_tissue_feat,
        ref_nuclei_feat=ref_nuclei_feat,
        reference_tokens=reference_tokens,
    )
    return target_image_latent, control_tensor, reference_tokens, feature_stats


def _build_cross_v4_control_batch(
    *,
    batch: dict,
    modules: dict[str, nn.Module],
    vae: AutoencoderKL,
    weight_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, CrossV4ReferenceEncoding, object, dict[str, float]]:
    device = next(vae.parameters()).device
    target_image_latent = _encode_images_to_latents(vae, batch["target_image"], weight_dtype)
    reference_image_latent = _encode_images_to_deterministic_latents(
        vae,
        batch["reference_image"],
        weight_dtype,
    )

    ref_tissue_feat = modules["tissue_downsampler"](
        modules["hte"](batch["reference_tissue_mask"].to(device=device))
    ).to(dtype=weight_dtype)
    ref_nuclei_feat = modules["nuclei_encoder"](
        batch["reference_nuclei_mask"].to(device=device)
    ).to(dtype=weight_dtype)
    tar_tissue_feat = _target_tissue_downsampler(modules)(
        _target_hte(modules)(batch["target_tissue_mask"].to(device=device))
    ).to(dtype=weight_dtype)
    tar_nuclei_feat = modules["nuclei_encoder"](
        batch["target_nuclei_mask"].to(device=device)
    ).to(dtype=weight_dtype)

    control_tensor = build_cross_v4_control_condition(
        tar_tissue_feat=tar_tissue_feat,
        tar_nuclei_feat=tar_nuclei_feat,
    )
    reference_encoding = modules["reference_context_encoder"](
        z_ref=reference_image_latent,
        ref_tissue_feat=ref_tissue_feat,
        ref_nuclei_feat=ref_nuclei_feat,
        ref_tissue_ids=batch["reference_tissue_mask"].to(device=device),
        ref_nuclei_ids=batch["reference_nuclei_mask"].to(device=device),
    )
    target_metadata = build_cross_v4_token_metadata(
        tissue_ids=batch["target_tissue_mask"].to(device=device),
        nuclei_ids=batch["target_nuclei_mask"].to(device=device),
        token_height=target_image_latent.shape[2] // 2,
        token_width=target_image_latent.shape[3] // 2,
    )
    feature_stats = _cross_v3_feature_stats(
        tar_tissue_feat=tar_tissue_feat,
        tar_nuclei_feat=tar_nuclei_feat,
        ref_tissue_feat=ref_tissue_feat,
        ref_nuclei_feat=ref_nuclei_feat,
        reference_tokens=reference_encoding.local_tokens,
    )
    feature_stats.update(
        {
            "reference_route_anchor_tokens": float(reference_encoding.route_anchor_tokens.shape[1]),
            "target_token_cell_density_mean": float(
                target_metadata.cell_density.detach().float().mean().cpu().item()
            ),
            "reference_token_cell_density_mean": float(
                reference_encoding.metadata.cell_density.detach().float().mean().cpu().item()
            ),
        }
    )
    return target_image_latent, control_tensor, reference_encoding, target_metadata, feature_stats


def _target_hte(modules: dict[str, nn.Module]) -> nn.Module:
    if "target_tissue_encoder" in modules:
        return modules["target_tissue_encoder"]
    return modules.get("target_hte") or modules["hte"]


def _target_tissue_downsampler(modules: dict[str, nn.Module]) -> nn.Module:
    if "target_tissue_encoder" in modules:
        return nn.Identity()
    return modules.get("target_tissue_downsampler") or modules["tissue_downsampler"]


def _cross_v3_feature_stats(
    *,
    tar_tissue_feat: torch.Tensor,
    tar_nuclei_feat: torch.Tensor,
    ref_tissue_feat: torch.Tensor,
    ref_nuclei_feat: torch.Tensor,
    reference_tokens: torch.Tensor,
) -> dict[str, float]:
    target_tissue_abs_mean = float(tar_tissue_feat.detach().float().abs().mean().cpu().item())
    reference_tissue_abs_mean = float(ref_tissue_feat.detach().float().abs().mean().cpu().item())
    return {
        "target_tissue_abs_mean": target_tissue_abs_mean,
        "target_tissue_abs_max": float(tar_tissue_feat.detach().float().abs().max().cpu().item()),
        "reference_tissue_abs_mean": reference_tissue_abs_mean,
        "reference_tissue_abs_max": float(ref_tissue_feat.detach().float().abs().max().cpu().item()),
        "target_to_reference_tissue_abs_mean_ratio": float(
            target_tissue_abs_mean / max(reference_tissue_abs_mean, 1e-12)
        ),
        "target_nuclei_abs_mean": float(tar_nuclei_feat.detach().float().abs().mean().cpu().item()),
        "target_nuclei_abs_max": float(tar_nuclei_feat.detach().float().abs().max().cpu().item()),
        "reference_nuclei_abs_mean": float(ref_nuclei_feat.detach().float().abs().mean().cpu().item()),
        "reference_nuclei_abs_max": float(ref_nuclei_feat.detach().float().abs().max().cpu().item()),
        "reference_token_abs_mean": float(reference_tokens.detach().float().abs().mean().cpu().item()),
        "reference_token_abs_max": float(reference_tokens.detach().float().abs().max().cpu().item()),
        "target_feature_height": float(tar_tissue_feat.shape[2]),
        "target_feature_width": float(tar_tissue_feat.shape[3]),
        "reference_feature_height": float(ref_tissue_feat.shape[2]),
        "reference_feature_width": float(ref_tissue_feat.shape[3]),
    }


def _use_self_reconstruction_reference(batch: dict) -> dict:
    warmup_batch = dict(batch)
    warmup_batch["reference_image"] = batch["target_image"]
    warmup_batch["reference_tissue_mask"] = batch["target_tissue_mask"]
    warmup_batch["reference_nuclei_mask"] = batch["target_nuclei_mask"]
    return warmup_batch


def _insert_self_reconstruction_samples(batch: dict, sample_mask: torch.Tensor) -> dict:
    mask = sample_mask.to(device=batch["target_image"].device, dtype=torch.bool)
    if mask.ndim != 1 or mask.shape[0] != batch["target_image"].shape[0]:
        raise ValueError(
            f"sample_mask must have shape ({batch['target_image'].shape[0]},), got {tuple(mask.shape)}"
        )
    if not bool(mask.any().item()):
        return batch

    mixed = dict(batch)
    for reference_key, target_key in (
        ("reference_image", "target_image"),
        ("reference_tissue_mask", "target_tissue_mask"),
        ("reference_nuclei_mask", "target_nuclei_mask"),
    ):
        reference = batch[reference_key].clone()
        reference[mask] = batch[target_key][mask].to(device=reference.device, dtype=reference.dtype)
        mixed[reference_key] = reference
    return mixed


def _batch_mode_mask(batch: dict, mode: str, *, device: torch.device | str) -> torch.Tensor:
    modes = batch.get("sample_modes")
    if modes is None:
        return torch.zeros(int(batch["target_image"].shape[0]), device=device, dtype=torch.bool)
    return torch.tensor([str(value) == mode for value in modes], device=device, dtype=torch.bool)


def _masked_mean_or_zero(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=torch.bool)
    if not bool(mask.any().item()):
        return values.new_zeros(())
    return values[mask].mean()


def _per_sample_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction.float() - target.float()).pow(2).flatten(1).mean(dim=1)


def _mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() - b.float()).abs().mean()


def _decode_packed_model_prediction(
    *,
    vae: AutoencoderKL,
    packed_noisy_latents: torch.Tensor,
    noise_pred: torch.Tensor,
    sigmas: torch.Tensor,
    latent_channels: int,
    latent_height: int,
    latent_width: int,
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    pred_original = packed_noisy_latents - sigmas * noise_pred
    pred_latents = unpack_flux_packed_latents(
        pred_original,
        channels=latent_channels,
        height=latent_height,
        width=latent_width,
    )
    pred_latents = (pred_latents / vae.config.scaling_factor) + vae.config.shift_factor
    decoded = vae.decode(
        pred_latents.to(device=next(vae.parameters()).device, dtype=weight_dtype),
        return_dict=False,
    )[0]
    return ((decoded.float() / 2.0) + 0.5).clamp(0.0, 1.0)


def _cast_control_samples(
    samples: list[torch.Tensor] | tuple[torch.Tensor, ...] | None,
    dtype: torch.dtype,
) -> list[torch.Tensor] | None:
    if samples is None:
        return None
    return [sample.to(dtype=dtype) for sample in samples]


def _zero_control_samples_like(samples: list[torch.Tensor] | None) -> list[torch.Tensor] | None:
    if samples is None:
        return None
    return [torch.zeros_like(sample) for sample in samples]


def _parse_ref_swap_variants(value: str | None) -> list[str]:
    variants: list[str] = []
    for raw_part in str(value or "").split(","):
        variant = raw_part.strip().lower().replace("-", "_")
        if not variant:
            continue
        if variant in {"shuffle", "batch_shuffle"}:
            variant = "random"
        if variant not in {"zero", "random"}:
            raise ValueError(f"Unsupported ref-swap variant {variant!r}; choose zero and/or random.")
        if variant not in variants:
            variants.append(variant)
    return variants


def _build_swapped_reference_tokens(
    reference_tokens: torch.Tensor,
    variant: str,
) -> torch.Tensor | None:
    variant = str(variant).lower()
    if variant == "zero":
        return torch.zeros_like(reference_tokens)
    if variant == "random":
        bsz = int(reference_tokens.shape[0])
        if bsz <= 1:
            return None
        order = torch.arange(bsz, device=reference_tokens.device).roll(1)
        return reference_tokens.index_select(0, order)
    raise ValueError(f"Unsupported ref-swap variant {variant!r}; choose zero and/or random.")


def _build_swapped_cross_v4_reference_encoding(
    reference_encoding: CrossV4ReferenceEncoding,
    variant: str,
) -> CrossV4ReferenceEncoding | None:
    variant = str(variant).lower()
    if variant == "zero":
        return apply_cross_v4_reference_encoding_mode(reference_encoding, "zero-ref")
    if variant != "random":
        raise ValueError(f"Unsupported ref-swap variant {variant!r}; choose zero and/or random.")
    bsz = int(reference_encoding.local_tokens.shape[0])
    if bsz <= 1:
        return None
    order = torch.arange(bsz, device=reference_encoding.local_tokens.device).roll(1)

    def select(value: torch.Tensor) -> torch.Tensor:
        return value.index_select(0, order) if value.shape[0] == bsz else value

    metadata = reference_encoding.metadata
    swapped_metadata = type(metadata)(
        tissue_fine_id=select(metadata.tissue_fine_id),
        tissue_coarse_id=select(metadata.tissue_coarse_id),
        tissue_confidence=select(metadata.tissue_confidence),
        cell_hist=select(metadata.cell_hist),
        cell_density=select(metadata.cell_density),
    )
    return CrossV4ReferenceEncoding(
        local_tokens=select(reference_encoding.local_tokens),
        route_anchor_tokens=select(reference_encoding.route_anchor_tokens),
        metadata=swapped_metadata,
    )


def _build_cross_v4_context_and_kwargs(
    *,
    prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    reference_encoding: CrossV4ReferenceEncoding,
    target_metadata,
    prior_token_bank: nn.Module,
    args: argparse.Namespace,
    global_step: int,
    dtype: torch.dtype,
    diagnose_attention: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict, object, dict | None]:
    prior_tokens = prior_token_bank(reference_encoding.local_tokens)
    context = append_cross_v4_context(
        prompt_embeds=prompt_embeds,
        text_ids=text_ids,
        reference_encoding=reference_encoding,
        prior_tokens=prior_tokens,
    )
    bias_config = CrossV4CorrespondenceBiasConfig(
        same_fine=float(getattr(args, "cross_v4_same_fine_bias", 3.0)),
        same_coarse=float(getattr(args, "cross_v4_same_coarse_bias", 2.0)),
        mismatch=float(getattr(args, "cross_v4_mismatch_bias", -2.0)),
        cell_similarity=float(getattr(args, "cross_v4_cell_similarity_bias", 1.0)),
        density_gap=float(getattr(args, "cross_v4_density_gap_bias", 0.5)),
        prior_when_ref_present=float(getattr(args, "cross_v4_prior_present_bias", 0.5)),
        prior_when_ref_missing=float(getattr(args, "cross_v4_prior_missing_bias", 3.0)),
        prior_wrong_class=float(getattr(args, "cross_v4_prior_wrong_class_bias", -2.0)),
        cell_prior=float(getattr(args, "cross_v4_cell_prior_bias", 1.0)),
        scale=1.0,
    )
    correspondence_bias = build_cross_v4_correspondence_bias(
        target_metadata=target_metadata,
        context=context,
        config=bias_config,
        dtype=dtype,
    )
    joint_attention_kwargs = {
        "cross_v4_bias": correspondence_bias,
        "cross_v4_bias_scale": _cross_v4_bias_scale(args, global_step),
    }
    diagnostics = None
    if diagnose_attention:
        diagnostics = _build_cross_v4_attention_diagnostics(
            context=context,
            target_metadata=target_metadata,
            correspondence_bias=correspondence_bias,
        )
        joint_attention_kwargs["cross_v4_diagnostics"] = diagnostics
    return context.encoder_hidden_states, context.txt_ids, joint_attention_kwargs, context, diagnostics


def _cross_v4_bias_scale(args: argparse.Namespace, global_step: int) -> float:
    base = max(0.0, float(getattr(args, "cross_v4_bias_scale", 1.0) or 0.0))
    warmup = max(0, int(getattr(args, "cross_v4_bias_warmup_steps", 1000) or 0))
    if base == 0.0 or warmup == 0:
        return base
    return base * min(1.0, float(max(0, global_step)) / float(warmup))


def _parse_step_set(value: str | None) -> set[int]:
    steps: set[int] = set()
    for raw_part in str(value or "").split(","):
        part = raw_part.strip()
        if not part:
            continue
        step = int(part)
        if step > 0:
            steps.add(step)
    return steps


def _should_run_cross_v4_diagnostics(args: argparse.Namespace, step: int) -> bool:
    if step <= 0:
        return False
    diagnose_steps = _parse_step_set(getattr(args, "cross_v4_diagnose_steps", "500,1000,1500,2000"))
    if step in diagnose_steps:
        return True
    interval = max(0, int(getattr(args, "cross_v4_diagnose_interval", 0) or 0))
    return interval > 0 and step % interval == 0


def _build_cross_v4_attention_diagnostics(
    *,
    context,
    target_metadata,
    correspondence_bias: torch.Tensor,
) -> dict:
    bucket_masks = _cross_v4_attention_bucket_masks(context=context, target_metadata=target_metadata)
    group_masks = _cross_v4_attention_group_masks(
        target_metadata=target_metadata,
        reference_metadata=context.reference_metadata,
    )
    return {
        "bucket_masks": bucket_masks,
        "group_masks": group_masks,
        "records": [],
        "static": _cross_v4_static_diagnostic_stats(
            context=context,
            target_metadata=target_metadata,
            group_masks=group_masks,
        ),
        "bias_stats": _summarize_cross_v4_bucket_tensor(
            correspondence_bias.detach().float(),
            bucket_masks=bucket_masks,
            group_masks=group_masks,
            prefix="cross_v4_bias",
        ),
    }


def _cross_v4_attention_bucket_masks(*, context, target_metadata) -> dict[str, torch.Tensor]:
    device = target_metadata.tissue_coarse_id.device
    batch_size, image_tokens = target_metadata.tissue_coarse_id.shape
    context_tokens = int(context.segments.total_tokens)

    def empty() -> torch.Tensor:
        return torch.zeros(batch_size, image_tokens, context_tokens, device=device, dtype=torch.bool)

    buckets: dict[str, torch.Tensor] = {
        "ref_same_fine": empty(),
        "ref_same_coarse": empty(),
        "ref_same_total": empty(),
        "ref_mismatch": empty(),
        "ref_all_local": empty(),
        "tissue_prior_target": empty(),
        "tissue_prior_other": empty(),
        "cell_prior_match": empty(),
        "cell_prior_other": empty(),
        "text_global": empty(),
        "route_anchor": empty(),
    }

    text_start, text_end = context.segments.text
    global_start, global_end = context.segments.global_style
    if text_end > text_start:
        buckets["text_global"][:, :, text_start:text_end] = True
    if global_end > global_start:
        buckets["text_global"][:, :, global_start:global_end] = True

    route_start, route_end = context.segments.route_anchor
    if route_end > route_start:
        buckets["route_anchor"][:, :, route_start:route_end] = True

    ref_start, ref_end = context.segments.reference_local
    if ref_end > ref_start:
        reference = context.reference_metadata
        same_fine = target_metadata.tissue_fine_id[:, :, None] == reference.tissue_fine_id[:, None, :]
        same_coarse = target_metadata.tissue_coarse_id[:, :, None] == reference.tissue_coarse_id[:, None, :]
        buckets["ref_same_fine"][:, :, ref_start:ref_end] = same_fine
        buckets["ref_same_coarse"][:, :, ref_start:ref_end] = same_coarse & ~same_fine
        buckets["ref_same_total"][:, :, ref_start:ref_end] = same_coarse
        buckets["ref_mismatch"][:, :, ref_start:ref_end] = ~same_coarse
        buckets["ref_all_local"][:, :, ref_start:ref_end] = True

    prior_start, prior_end = context.segments.tissue_prior
    if prior_end > prior_start:
        prior_ids = context.tissue_prior_class_ids.to(device=device)
        target_is_prior = target_metadata.tissue_coarse_id[:, :, None] == prior_ids.view(1, 1, -1)
        buckets["tissue_prior_target"][:, :, prior_start:prior_end] = target_is_prior
        buckets["tissue_prior_other"][:, :, prior_start:prior_end] = ~target_is_prior

    cell_start, cell_end = context.segments.cell_prior
    if cell_end > cell_start:
        cell_prior_ids = context.cell_prior_class_ids.to(device=device)
        dominant_cell = target_metadata.cell_hist.to(device=device).argmax(dim=-1)
        cell_match = dominant_cell[:, :, None] == cell_prior_ids.view(1, 1, -1)
        buckets["cell_prior_match"][:, :, cell_start:cell_end] = cell_match
        buckets["cell_prior_other"][:, :, cell_start:cell_end] = ~cell_match
    return buckets


def _cross_v4_attention_group_masks(*, target_metadata, reference_metadata) -> dict[str, torch.Tensor]:
    device = target_metadata.tissue_coarse_id.device
    ref_presence = _cross_v4_class_presence(reference_metadata.tissue_coarse_id.to(device=device), NUM_COARSE)
    target_class = target_metadata.tissue_coarse_id.to(device=device)
    covered = ref_presence.gather(1, target_class).bool()
    groups: dict[str, torch.Tensor] = {
        "all": torch.ones_like(covered, dtype=torch.bool),
        "covered": covered,
        "missing": ~covered,
    }
    for class_id in range(NUM_COARSE):
        class_mask = target_class == class_id
        groups[f"class_{class_id}"] = class_mask
        groups[f"covered_class_{class_id}"] = covered & class_mask
        groups[f"missing_class_{class_id}"] = (~covered) & class_mask
    return groups


def _cross_v4_class_presence(class_ids: torch.Tensor, class_count: int) -> torch.Tensor:
    return F.one_hot(class_ids.long(), num_classes=class_count).bool().any(dim=1)


def _cross_v4_static_diagnostic_stats(*, context, target_metadata, group_masks: dict[str, torch.Tensor]) -> dict[str, float]:
    target_class = target_metadata.tissue_coarse_id
    reference_class = context.reference_metadata.tissue_coarse_id.to(device=target_class.device)
    target_presence = _cross_v4_class_presence(target_class, NUM_COARSE)
    reference_presence = _cross_v4_class_presence(reference_class, NUM_COARSE)
    stats = {
        "cross_v4_context_tokens": float(context.segments.total_tokens),
        "cross_v4_text_tokens": float(context.segments.text[1] - context.segments.text[0]),
        "cross_v4_global_style_tokens": float(context.segments.global_style[1] - context.segments.global_style[0]),
        "cross_v4_tissue_prior_tokens": float(context.segments.tissue_prior[1] - context.segments.tissue_prior[0]),
        "cross_v4_cell_prior_tokens": float(context.segments.cell_prior[1] - context.segments.cell_prior[0]),
        "cross_v4_route_anchor_tokens": float(context.segments.route_anchor[1] - context.segments.route_anchor[0]),
        "cross_v4_reference_local_tokens": float(context.segments.reference_local[1] - context.segments.reference_local[0]),
        "cross_v4_target_tokens": float(target_class.numel()),
        "cross_v4_target_present_coarse_classes": float(target_presence.sum(dim=1).float().mean().item()),
        "cross_v4_reference_present_coarse_classes": float(reference_presence.sum(dim=1).float().mean().item()),
    }
    for group_name in ("covered", "missing"):
        group = group_masks[group_name]
        stats[f"cross_v4_{group_name}_target_token_fraction"] = float(group.float().mean().item())
        stats[f"cross_v4_{group_name}_target_tokens"] = float(group.sum().item())
    return stats


def _summarize_cross_v4_bucket_tensor(
    values: torch.Tensor,
    *,
    bucket_masks: dict[str, torch.Tensor],
    group_masks: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, float]:
    summary: dict[str, float] = {}
    values = values.detach().float()
    for bucket_name, raw_bucket_mask in bucket_masks.items():
        bucket_mask = raw_bucket_mask.to(device=values.device, dtype=torch.bool)
        if bucket_mask.shape != values.shape:
            continue
        for group_name, raw_group_mask in group_masks.items():
            if group_name.startswith("class_") or group_name.startswith("covered_class_") or group_name.startswith("missing_class_"):
                continue
            group_mask = raw_group_mask.to(device=values.device, dtype=torch.bool)
            expanded_group = group_mask[:, :, None].expand_as(bucket_mask)
            combined = bucket_mask & expanded_group
            if not bool(combined.any().item()):
                continue
            summary[f"{prefix}_{group_name}_{bucket_name}_mean"] = float(values[combined].mean().item())
    return summary


def _summarize_cross_v4_attention_records(diagnostics: dict | None) -> dict[str, float]:
    if not diagnostics:
        return {}
    records = list(diagnostics.get("records") or [])
    if not records:
        return {}
    keys = sorted({key for record in records for key in record})
    summary = {"cross_v4_attention_diagnostic_layers": float(len(records))}
    for key in keys:
        vals = [float(record[key]) for record in records if key in record]
        if vals:
            summary[key] = float(sum(vals) / len(vals))
    return summary


def _cross_v4_diagnostic_verdict(summary: dict[str, float]) -> tuple[str, list[str]]:
    issues: list[str] = []
    covered_fraction = float(summary.get("cross_v4_covered_target_token_fraction", 0.0))
    missing_fraction = float(summary.get("cross_v4_missing_target_token_fraction", 0.0))

    covered_same = float(summary.get("cross_v4_attention_covered_ref_same_total", 0.0))
    covered_all_ref = float(summary.get("cross_v4_attention_covered_ref_all_local", 0.0))
    covered_mismatch = float(summary.get("cross_v4_attention_covered_ref_mismatch", 0.0))
    covered_prior = float(summary.get("cross_v4_attention_covered_tissue_prior_target", 0.0))
    if covered_fraction > 0.0 and covered_all_ref > 0.0:
        same_ratio = covered_same / max(covered_all_ref, 1e-8)
        if same_ratio <= 0.6:
            issues.append(f"covered ref_same/ref_all={same_ratio:.3f} <= 0.6")
        if covered_same <= covered_prior:
            issues.append("covered same-class reference mass is not above matching tissue-prior mass")
        if covered_mismatch >= covered_same / 3.0:
            issues.append("covered mismatch reference mass is too high")

    missing_prior = float(summary.get("cross_v4_attention_missing_tissue_prior_target", 0.0))
    missing_mismatch = float(summary.get("cross_v4_attention_missing_ref_mismatch", 0.0))
    missing_prior_other = float(summary.get("cross_v4_attention_missing_tissue_prior_other", 0.0))
    if missing_fraction > 0.0:
        if missing_prior <= missing_mismatch:
            issues.append("missing-class matching tissue-prior mass is not above mismatch reference mass")
        if missing_prior_other > max(0.05, missing_prior * 0.5):
            issues.append("missing-class wrong-prior mass is high")

    if "cross_v4_attention_diagnostic_layers" not in summary:
        issues.append("no injected attention diagnostics were recorded")
    if issues:
        return "watch", issues
    return "pass", []


def _module_grad_norm(module: nn.Module | None) -> float:
    if module is None:
        return 0.0
    total = 0.0
    for param in module.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        total += float(grad.pow(2).sum().item())
    return math.sqrt(total)


def _cuda_memory_stats(device: torch.device | str) -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    torch_device = torch.device(device)
    return {
        "cuda_memory_allocated_gb": float(torch.cuda.memory_allocated(torch_device) / (1024**3)),
        "cuda_memory_reserved_gb": float(torch.cuda.memory_reserved(torch_device) / (1024**3)),
        "cuda_peak_memory_allocated_gb": float(torch.cuda.max_memory_allocated(torch_device) / (1024**3)),
        "cuda_peak_memory_reserved_gb": float(torch.cuda.max_memory_reserved(torch_device) / (1024**3)),
    }


def _enforce_cuda_memory_limit(
    *,
    device: torch.device | str,
    max_memory_gb: float,
    step: int,
) -> dict[str, float]:
    stats = _cuda_memory_stats(device)
    if max_memory_gb > 0.0 and stats:
        peak_reserved = float(stats["cuda_peak_memory_reserved_gb"])
        if peak_reserved > float(max_memory_gb):
            raise RuntimeError(
                f"CUDA peak reserved memory {peak_reserved:.2f} GiB exceeded "
                f"--max-cuda-memory-gb={float(max_memory_gb):.2f} at step {step}."
            )
    return stats


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _apply_training_prompt_policy(records: list[dict], args: argparse.Namespace) -> None:
    for record in records:
        record["prompt"] = CROSS_V3_PROMPT
    logger.info("Using fixed Cross V3 prompt %r for all %s records", CROSS_V3_PROMPT, len(records))


def _build_prompt_cache(
    *,
    pipeline: FluxControlNetPipeline,
    prompts: list[str],
    weight_dtype: torch.dtype,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    unique_prompts = sorted(set(prompts))
    logger.info("Encoding %s unique prompt(s) from %s training records", len(unique_prompts), len(prompts))
    prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    text_ids = None
    with torch.no_grad():
        for start in range(0, len(unique_prompts), batch_size):
            prompt_batch = unique_prompts[start : start + batch_size]
            prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                prompt_batch,
                prompt_2=prompt_batch,
                device=device,
            )
            for index, prompt in enumerate(prompt_batch):
                prompt_cache[prompt] = (
                    prompt_embeds[index].to(dtype=weight_dtype, device="cpu"),
                    pooled_prompt_embeds[index].to(dtype=weight_dtype, device="cpu"),
                )
        empty_prompt_embeds, empty_pooled, text_ids = pipeline.encode_prompt([""], prompt_2=[""], device=device)
    if text_ids.dim() == 3:
        text_ids = text_ids[0]
    empty_prompt = (
        empty_prompt_embeds[0].to(dtype=weight_dtype, device="cpu"),
        empty_pooled[0].to(dtype=weight_dtype, device="cpu"),
    )
    return prompt_cache, empty_prompt, text_ids.to(dtype=weight_dtype, device="cpu")


def _resolve_prompt_batch(
    *,
    prompts: list[str],
    prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
    empty_prompt_embeds: torch.Tensor,
    empty_pooled: torch.Tensor,
    proportion_empty_prompts: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_prompt = []
    batch_pooled = []
    for prompt in prompts:
        if random.random() < proportion_empty_prompts:
            batch_prompt.append(empty_prompt_embeds)
            batch_pooled.append(empty_pooled)
        else:
            prompt_embeds, pooled_prompt = prompt_cache[prompt]
            batch_prompt.append(prompt_embeds)
            batch_pooled.append(pooled_prompt)
    return torch.stack(batch_prompt), torch.stack(batch_pooled)


def _prepare_packed_latent_image_ids(
    *,
    packed_height: int,
    packed_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if packed_height <= 0 or packed_width <= 0:
        raise ValueError(f"packed latent grid must be positive, got {packed_height}x{packed_width}.")
    latent_image_ids = torch.zeros(packed_height, packed_width, 3)
    latent_image_ids[..., 1] = torch.arange(packed_height)[:, None]
    latent_image_ids[..., 2] = torch.arange(packed_width)[None, :]
    latent_image_ids = latent_image_ids.reshape(packed_height * packed_width, 3)
    return latent_image_ids.to(device=device, dtype=dtype)


def _latest_checkpoint(output_dir: str) -> str | None:
    dirs = [directory for directory in os.listdir(output_dir) if directory.startswith("checkpoint-")]
    if not dirs:
        return None
    latest = sorted(dirs, key=lambda item: int(item.split("-")[1]))[-1]
    return os.path.join(output_dir, latest)


def _save_checkpoint(accelerator: Accelerator, args: argparse.Namespace, global_step: int) -> str:
    if args.checkpoints_total_limit is not None:
        checkpoints = [
            directory for directory in os.listdir(args.output_dir) if directory.startswith("checkpoint-")
        ]
        checkpoints = sorted(checkpoints, key=lambda item: int(item.split("-")[1]))
        if len(checkpoints) >= args.checkpoints_total_limit:
            for stale_checkpoint in checkpoints[: len(checkpoints) - args.checkpoints_total_limit + 1]:
                shutil.rmtree(os.path.join(args.output_dir, stale_checkpoint))
    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    accelerator.save_state(save_path)
    logger.info("Saved state to %s", save_path)
    return save_path


def _save_condition_modules(
    output_dir: str,
    modules: dict[str, nn.Module],
    unwrap_model: Callable[[nn.Module], nn.Module],
    save_dtype: torch.dtype,
    *,
    control_spec: CrossV3ControlSpec,
    reference_spec: CrossV3ReferenceSpec,
    cross_version: str = "v3",
    args: argparse.Namespace | None = None,
) -> None:
    cross_version = str(cross_version or "v3").lower()
    state = {
        "cross_version": cross_version,
        "cross_v3_control_spec": {
            "tissue_channels": int(control_spec.tissue_channels),
            "nuclei_channels": int(control_spec.nuclei_channels),
            "raw_channels": int(control_spec.raw_channels),
            "packed_channels": int(control_spec.packed_channels),
            "target_tissue_path": (
                "fixed_one_hot"
                if "target_tissue_encoder" in modules
                else "target_hte,target_tissue_downsampler"
                if "target_hte" in modules or "target_tissue_downsampler" in modules
                else "shared_hte,tissue_downsampler"
            ),
            "target_one_hot_scale": (
                float(getattr(modules.get("target_tissue_encoder"), "scale"))
                if "target_tissue_encoder" in modules
                else None
            ),
            "condition_order": [
                "tar_tissue_feat",
                "tar_nuclei_feat",
            ],
        },
        "cross_v3_reference_spec": {
            "reference_latent_channels": int(reference_spec.reference_latent_channels),
            "tissue_channels": int(reference_spec.tissue_channels),
            "nuclei_channels": int(reference_spec.nuclei_channels),
            "raw_channels": int(reference_spec.raw_channels),
            "packed_channels": int(reference_spec.packed_channels),
            "token_dim": int(reference_spec.token_dim),
            "output_init_std": float(reference_spec.output_init_std),
            "route_anchor_mode": str(reference_spec.normalized_route_anchor_mode),
            "route_class_count": int(reference_spec.route_class_count),
            "route_embedding_init_std": float(reference_spec.route_embedding_init_std),
            "condition_order": [
                "z_ref",
                "ref_tissue_feat",
                "ref_nuclei_feat",
                "ref_tissue_ids_for_semantic_route",
            ],
            "injection_path": "flux_joint_cross_attention_context",
            "txt_ids": "zeros_no_reference_coordinates_route_is_learned_token_payload",
        },
        "prompt_policy": {
            "kind": "fixed",
            "prompt": CROSS_V3_PROMPT,
            "proportion_empty_prompts": 0.0,
        },
    }
    if cross_version == "v4":
        state["cross_v4_control_spec"] = dict(state["cross_v3_control_spec"])
        state["cross_v4_reference_spec"] = {
            "reference_latent_channels": int(reference_spec.reference_latent_channels),
            "tissue_channels": int(reference_spec.tissue_channels),
            "nuclei_channels": int(reference_spec.nuclei_channels),
            "raw_channels": int(reference_spec.raw_channels),
            "packed_channels": int(reference_spec.packed_channels),
            "token_dim": int(reference_spec.token_dim),
            "output_init_std": float(reference_spec.output_init_std),
            "route_anchor_mode": str(reference_spec.normalized_route_anchor_mode),
            "route_class_count": int(reference_spec.route_class_count),
            "route_embedding_init_std": float(reference_spec.route_embedding_init_std),
            "tissue_prior_tokens_per_class": int(
                getattr(reference_spec, "tissue_prior_tokens_per_class", 4)
            ),
            "cell_prior_tokens_per_class": int(getattr(reference_spec, "cell_prior_tokens_per_class", 0)),
            "global_style_tokens": int(getattr(reference_spec, "global_style_tokens", 0)),
            "prior_init_std": float(getattr(reference_spec, "prior_init_std", 0.02)),
            "condition_order": [
                "z_ref",
                "ref_tissue_feat",
                "ref_nuclei_feat",
                "ref_tissue_ids_for_token_metadata",
                "ref_nuclei_ids_for_token_metadata",
                "target_tissue_ids_for_token_metadata",
                "target_nuclei_ids_for_token_metadata",
            ],
            "injection_path": "flux_joint_cross_attention_context_with_cross_v4_bias",
            "context_order": [
                "text_tokens",
                "global_style_tokens",
                "tissue_prior_tokens",
                "cell_prior_tokens",
                "route_anchor_tokens",
                "reference_local_tokens",
            ],
        }
        if args is not None:
            state["cross_v4_attention_bias"] = {
                "biased_double_block_indices": str(
                    getattr(args, "cross_v4_biased_double_blocks", "last")
                ),
                "bias_scale": float(getattr(args, "cross_v4_bias_scale", 1.0)),
                "bias_warmup_steps": int(getattr(args, "cross_v4_bias_warmup_steps", 1000)),
                "same_fine": float(getattr(args, "cross_v4_same_fine_bias", 3.0)),
                "same_coarse": float(getattr(args, "cross_v4_same_coarse_bias", 2.0)),
                "mismatch": float(getattr(args, "cross_v4_mismatch_bias", -2.0)),
                "cell_similarity": float(getattr(args, "cross_v4_cell_similarity_bias", 1.0)),
                "density_gap": float(getattr(args, "cross_v4_density_gap_bias", 0.5)),
                "prior_when_ref_present": float(getattr(args, "cross_v4_prior_present_bias", 0.5)),
                "prior_when_ref_missing": float(getattr(args, "cross_v4_prior_missing_bias", 3.0)),
                "prior_wrong_class": float(getattr(args, "cross_v4_prior_wrong_class_bias", -2.0)),
                "cell_prior": float(getattr(args, "cross_v4_cell_prior_bias", 1.0)),
            }
    for name, module in modules.items():
        unwrapped = unwrap_model(module)
        state[name] = {key: value.detach().cpu().to(save_dtype) for key, value in unwrapped.state_dict().items()}
    torch.save(state, os.path.join(output_dir, "phase5_conditioning.pt"))


def _save_cross_v3_artifacts(
    output_dir: str,
    args: argparse.Namespace,
    *,
    flux_controlnet: nn.Module,
    modules: dict[str, nn.Module],
    unwrap_model: Callable[[nn.Module], nn.Module],
    control_spec: CrossV3ControlSpec,
    reference_spec: CrossV3ReferenceSpec,
    cross_version: str = "v3",
) -> None:
    save_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.save_weight_dtype, torch.float32)
    unwrapped_controlnet = unwrap_model(flux_controlnet)
    if args.save_weight_dtype != "fp32":
        unwrapped_controlnet.save_pretrained(output_dir, variant=args.save_weight_dtype)
    else:
        unwrapped_controlnet.save_pretrained(output_dir)
    _save_condition_modules(
        output_dir,
        modules,
        unwrap_model,
        save_dtype,
        control_spec=control_spec,
        reference_spec=reference_spec,
        cross_version=cross_version,
        args=args,
    )


def _load_cross_v3_controlnet_checkpoint(
    checkpoint_path: str | Path,
    control_spec: CrossV3ControlSpec,
) -> FluxControlNetModel:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        return FluxControlNetModel.from_pretrained(str(checkpoint_path))

    controlnet_config = FluxControlNetModel.load_config(checkpoint)
    controlnet = FluxControlNetModel.from_config(controlnet_config)
    patch_controlnet_x_embedder(controlnet, control_spec.packed_channels)
    state_dict = _load_diffusers_model_state_dict(checkpoint)
    source_layout = _load_source_layout(checkpoint)
    state_dict = _remap_x_embedder_state_dict(
        state_dict,
        source_layout=source_layout,
        target_spec=control_spec,
    )
    controlnet.load_state_dict(state_dict, strict=True)
    return controlnet


def _load_source_layout(checkpoint: Path) -> dict:
    state_path = checkpoint / "phase5_conditioning.pt"
    if not state_path.exists():
        return {}
    state = _torch_load_weights(state_path)
    if "cross_v3_control_spec" in state:
        return {"kind": "cross_v3", **dict(state["cross_v3_control_spec"])}
    if "cross_v2_1_control_spec" in state:
        return {"kind": "cross_v2_1", **dict(state["cross_v2_1_control_spec"])}
    if "cross_v1_control_spec" in state:
        return {"kind": "cross_v1", **dict(state["cross_v1_control_spec"])}
    if "cross_v1_spatial_mode" in state:
        return {"kind": "cross_v1", "spatial_mode": str(state["cross_v1_spatial_mode"])}
    return {}


def _remap_x_embedder_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    source_layout: dict,
    target_spec: CrossV3ControlSpec,
) -> dict[str, torch.Tensor]:
    weight_key = "controlnet_x_embedder.weight"
    if weight_key not in state_dict:
        return state_dict

    old_weight = state_dict[weight_key]
    new_in_features = target_spec.packed_channels
    if old_weight.shape[1] == new_in_features:
        return state_dict

    remapped = dict(state_dict)
    new_weight = old_weight.new_zeros((old_weight.shape[0], new_in_features))
    source_target_start = _source_packed_target_mask_start(source_layout, old_weight.shape[1])
    copy_width = min(new_in_features, max(0, old_weight.shape[1] - source_target_start))
    if copy_width > 0:
        new_weight[:, :copy_width] = old_weight[:, source_target_start : source_target_start + copy_width]

    if "controlnet_x_embedder.bias" in state_dict:
        remapped["controlnet_x_embedder.bias"] = state_dict["controlnet_x_embedder.bias"]
    remapped[weight_key] = new_weight
    return remapped


def _source_packed_target_mask_start(source_layout: dict, old_width: int) -> int:
    kind = str(source_layout.get("kind", "")).lower()
    spatial_mode = str(source_layout.get("spatial_mode", "reference_target")).lower()
    tissue_channels = int(source_layout.get("tissue_channels", 64))
    nuclei_channels = int(source_layout.get("nuclei_channels", 16))
    packed_mask_channels = (tissue_channels + nuclei_channels) * 4
    if kind == "cross_v3" or spatial_mode == "target_only":
        return 0
    if kind == "cross_v1":
        return packed_mask_channels
    if kind == "cross_v2_1":
        reference_latent_channels = int(source_layout.get("reference_latent_channels", 16))
        return reference_latent_channels * 4 + packed_mask_channels
    return max(0, old_width - packed_mask_channels)


def _load_diffusers_model_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    safetensors_indexes = sorted(checkpoint_path.glob("diffusion_pytorch_model*.safetensors.index.json"))
    bin_indexes = sorted(checkpoint_path.glob("diffusion_pytorch_model*.bin.index.json"))
    if safetensors_indexes:
        return _load_sharded_diffusers_state_dict(safetensors_indexes[0])
    if bin_indexes:
        return _load_sharded_diffusers_state_dict(bin_indexes[0])

    weight_candidates = [
        *sorted(checkpoint_path.glob("diffusion_pytorch_model*.safetensors")),
        *sorted(checkpoint_path.glob("diffusion_pytorch_model*.bin")),
        checkpoint_path / "pytorch_model.bin",
        checkpoint_path / "model.safetensors",
    ]
    for weight_path in weight_candidates:
        if weight_path.exists():
            return _load_single_diffusers_weight_file(weight_path)
    raise FileNotFoundError(f"No diffusers ControlNet weights found under: {checkpoint_path}")


def _load_sharded_diffusers_state_dict(index_path: Path) -> dict[str, torch.Tensor]:
    payload = json.loads(index_path.read_text(encoding="utf8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid diffusers weight index file: {index_path}")
    state_dict: dict[str, torch.Tensor] = {}
    for filename in sorted(set(weight_map.values())):
        state_dict.update(_load_single_diffusers_weight_file(index_path.parent / filename))
    return state_dict


def _load_single_diffusers_weight_file(weight_path: Path) -> dict[str, torch.Tensor]:
    if weight_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(weight_path)
    return _torch_load_weights(weight_path)


def _torch_load_weights(weight_path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(weight_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(weight_path, map_location="cpu")


def _resolve_conditioning_checkpoint_path(args: argparse.Namespace) -> Path | None:
    path = getattr(args, "conditioning_checkpoint", None)
    if path:
        return Path(path)
    path = getattr(args, "controlnet_model_name_or_path", None)
    return Path(path) if path else None


def _load_condition_modules_from_checkpoint(
    modules: dict[str, nn.Module],
    checkpoint_path: str | Path,
) -> None:
    checkpoint = Path(checkpoint_path)
    state_path = checkpoint / "phase5_conditioning.pt" if checkpoint.is_dir() else checkpoint
    if not state_path.exists():
        raise FileNotFoundError(f"Missing phase5_conditioning.pt: {state_path}")

    state = _torch_load_weights(state_path)
    for name in ("hte", "tissue_downsampler", "nuclei_encoder"):
        if name not in state:
            raise KeyError(f"Missing {name!r} in conditioning checkpoint: {state_path}")
        modules[name].load_state_dict(state[name])
    for name in ("target_hte", "target_tissue_downsampler"):
        if name in modules and name in state:
            modules[name].load_state_dict(state[name])
        elif name in modules:
            logger.info(
                "Conditioning checkpoint has no %s; keeping the newly initialized target-side low-capacity module.",
                name,
            )
    if "target_tissue_encoder" in modules:
        logger.info("Using fixed one-hot target tissue encoder; no target-side tissue weights are loaded.")
    if "reference_context_encoder" in state:
        _load_reference_context_encoder_state(
            modules["reference_context_encoder"],
            state["reference_context_encoder"],
            state_path=state_path,
        )
    else:
        logger.info("Conditioning checkpoint has no reference_context_encoder; initializing it from scratch.")
    if "prior_token_bank" in modules:
        if "prior_token_bank" in state:
            modules["prior_token_bank"].load_state_dict(state["prior_token_bank"], strict=True)
        else:
            logger.info("Conditioning checkpoint has no prior_token_bank; initializing Cross V4 priors from scratch.")


def _load_reference_context_encoder_state(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    state_path: Path,
) -> None:
    try:
        module.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError:
        if getattr(module, "route_class_count", 0) <= 0:
            raise
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    allowed_missing_prefixes = (
        "local_route_embedding.",
        "anchor_route_embedding.",
        "route_type_embedding.",
        "route_missing_anchor",
    )
    disallowed_missing = [
        key for key in missing if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "Could not load reference_context_encoder from "
            f"{state_path}: missing={list(missing)}, unexpected={list(unexpected)}"
        )
    logger.info(
        "Loaded reference_context_encoder from %s with newly initialized semantic route parameters: %s",
        state_path,
        ", ".join(missing),
    )


def run_cross_v3_training(args: argparse.Namespace) -> None:
    cross_version = str(getattr(args, "cross_version", "v3")).lower().replace("_", ".").replace("-", ".")
    if cross_version not in {"v3", "v4"}:
        raise NotImplementedError("This module implements cross V3 and Cross V4.")
    is_cross_v4 = cross_version == "v4"

    dataset = CrossReconstructionDataset(
        args.train_metadata,
        stain_augmentation=getattr(args, "stain_augmentation", "none"),
        stain_counterfactual_prob=float(getattr(args, "stain_counterfactual_prob", 0.0) or 0.0),
        hed_sigma=float(getattr(args, "hed_sigma", 0.2) or 0.0),
        hed_beta=float(getattr(args, "hed_beta", 0.02) or 0.0),
        hed_strong_alpha_sampling=bool(getattr(args, "hed_strong_alpha_sampling", False)),
        hed_alpha_min=float(getattr(args, "hed_alpha_min", 0.4)),
        hed_alpha_low=float(getattr(args, "hed_alpha_low", 0.75)),
        hed_alpha_high=float(getattr(args, "hed_alpha_high", 1.25)),
        hed_alpha_max=float(getattr(args, "hed_alpha_max", 1.8)),
    )
    if args.max_train_samples is not None:
        dataset.records = dataset.records[: args.max_train_samples]

    target_tissue_encoding = str(getattr(args, "target_tissue_encoding", "shared_hte") or "shared_hte").lower()
    if target_tissue_encoding not in {"shared_hte", "low_capacity_hte", "one_hot"}:
        raise ValueError(
            f"Unsupported target_tissue_encoding {target_tissue_encoding!r}; "
            "choose shared_hte, low_capacity_hte, or one_hot."
        )
    target_tissue_channels = NUM_FINE if target_tissue_encoding == "one_hot" else args.tissue_out_channels
    control_spec = (CrossV4ControlSpec if is_cross_v4 else CrossV3ControlSpec)(
        tissue_channels=target_tissue_channels,
        nuclei_channels=args.nuclei_out_channels,
    )
    reference_spec_kwargs = {
        "reference_latent_channels": int(getattr(args, "reference_latent_channels", 16)),
        "tissue_channels": args.tissue_out_channels,
        "nuclei_channels": args.nuclei_out_channels,
        "token_dim": int(getattr(args, "reference_token_dim", 4096)),
        "output_init_std": float(getattr(args, "reference_token_output_init_std", 0.02)),
        "route_anchor_mode": str(getattr(args, "reference_route_anchor_mode", "none")),
        "route_embedding_init_std": float(getattr(args, "reference_route_embedding_init_std", 0.02)),
    }
    if is_cross_v4:
        reference_spec = CrossV4ReferenceSpec(
            **reference_spec_kwargs,
            tissue_prior_tokens_per_class=int(getattr(args, "cross_v4_tissue_prior_tokens_per_class", 4)),
            cell_prior_tokens_per_class=int(getattr(args, "cross_v4_cell_prior_tokens_per_class", 0)),
            global_style_tokens=int(getattr(args, "cross_v4_global_style_tokens", 0)),
            prior_init_std=float(getattr(args, "cross_v4_prior_init_std", 0.02)),
        )
    else:
        reference_spec = CrossV3ReferenceSpec(**reference_spec_kwargs)
    target_tissue_embedding_dim = int(
        getattr(args, "target_tissue_embedding_dim", None) or args.tissue_embedding_dim
    )
    modules = {
        "hte": HierarchicalTissueEmbedding(embedding_dim=args.tissue_embedding_dim),
        "tissue_downsampler": TissueConditionDownsampler(
            in_channels=args.tissue_embedding_dim,
            hidden_channels=args.tissue_out_channels,
            num_blocks=args.condition_downsample_blocks,
        ),
        "nuclei_encoder": NucleiConditionEncoder(
            embedding_dim=args.nuclei_embedding_dim,
            out_channels=args.nuclei_out_channels,
            num_blocks=args.condition_downsample_blocks,
        ),
        "reference_context_encoder": (CrossV4ReferenceContextEncoder if is_cross_v4 else CrossV3ReferenceContextEncoder)(
            reference_latent_channels=reference_spec.reference_latent_channels,
            tissue_channels=reference_spec.tissue_channels,
            nuclei_channels=reference_spec.nuclei_channels,
            token_dim=reference_spec.token_dim,
            hidden_dim=int(getattr(args, "reference_token_hidden_dim", reference_spec.token_dim)),
            output_init_std=reference_spec.output_init_std,
            route_anchor_mode=reference_spec.route_anchor_mode,
            route_embedding_init_std=reference_spec.route_embedding_init_std,
        ),
    }
    if is_cross_v4:
        modules["prior_token_bank"] = CrossV4PriorTokenBank(
            token_dim=reference_spec.token_dim,
            tissue_prior_tokens_per_class=int(reference_spec.tissue_prior_tokens_per_class),
            cell_prior_tokens_per_class=int(reference_spec.cell_prior_tokens_per_class),
            global_style_tokens=int(reference_spec.global_style_tokens),
            init_std=float(reference_spec.prior_init_std),
        )
    if target_tissue_encoding == "one_hot":
        modules["target_tissue_encoder"] = FixedOneHotTissueEncoder(
            num_classes=NUM_FINE,
            downsample_factor=2 ** int(args.condition_downsample_blocks),
            scale=float(getattr(args, "target_one_hot_scale", 4.0)),
        )
    elif target_tissue_encoding == "low_capacity_hte":
        modules["target_hte"] = HierarchicalTissueEmbedding(embedding_dim=target_tissue_embedding_dim)
        modules["target_tissue_downsampler"] = TissueConditionDownsampler(
            in_channels=target_tissue_embedding_dim,
            hidden_channels=args.tissue_out_channels,
            num_blocks=args.condition_downsample_blocks,
        )
    if bool(getattr(args, "load_conditioning_from_checkpoint", False)):
        conditioning_checkpoint = _resolve_conditioning_checkpoint_path(args)
        if conditioning_checkpoint is None:
            raise ValueError("Loading conditioning modules requires a conditioning checkpoint.")
        _load_condition_modules_from_checkpoint(modules, conditioning_checkpoint)

    self_reconstruction_warmup_steps = max(0, int(getattr(args, "self_reconstruction_warmup_steps", 0) or 0))
    self_reconstruction_sample_prob = min(
        1.0,
        max(0.0, float(getattr(args, "self_reconstruction_sample_prob", 0.0) or 0.0)),
    )
    ref_check_step = int(getattr(args, "ref_check_step", 10) or 0)
    reference_style_loss_weight = max(
        0.0,
        float(getattr(args, "reference_style_loss_weight", 1.0) or 0.0),
    )
    reference_style_loss_interval = int(getattr(args, "reference_style_loss_interval", 1) or 0)
    reference_style_loss_config = RegionalStainStyleLossConfig(
        tissue_weight=float(getattr(args, "reference_style_tissue_weight", 1.0) or 0.0),
        nuclei_weight=float(getattr(args, "reference_style_nuclei_weight", 1.0) or 0.0),
        mean_weight=float(getattr(args, "reference_style_mean_weight", 1.0) or 0.0),
        std_weight=float(getattr(args, "reference_style_std_weight", 1.0) or 0.0),
        covariance_weight=float(getattr(args, "reference_style_cov_weight", 0.25) or 0.0),
        min_pixels=max(1, int(getattr(args, "reference_style_min_pixels", 32) or 1)),
        max_regions_per_sample=getattr(args, "reference_style_max_regions_per_sample", None),
    )
    ref_swap_loss_weight = max(0.0, float(getattr(args, "ref_swap_loss_weight", 0.1) or 0.0))
    ref_swap_loss_interval = int(getattr(args, "ref_swap_loss_interval", 1) or 0)
    ref_swap_margin = float(getattr(args, "ref_swap_margin", 0.08) or 0.0)
    ref_swap_variants = _parse_ref_swap_variants(getattr(args, "ref_swap_variants", "zero"))
    max_cuda_memory_gb = max(0.0, float(getattr(args, "max_cuda_memory_gb", 0.0) or 0.0))
    cuda_memory_check_interval = max(0, int(getattr(args, "cuda_memory_check_interval", 10) or 0))
    cross_v4_diagnose_jsonl = Path(
        getattr(args, "cross_v4_diagnose_jsonl", None)
        or os.path.join(args.output_dir, "cross_v4_diagnostics.jsonl")
    )

    logging_out_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=str(logging_out_dir),
    )
    from datetime import timedelta

    kwargs = accelerate.InitProcessGroupKwargs(timeout=timedelta(hours=5))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    logger.info(
        "Using Cross %s ControlNet target-only order [tar_tissue_feat, tar_nuclei_feat]: "
        "raw_channels=%s packed_channels=%s; reference [z_ref, ref_tissue_feat, ref_nuclei_feat] "
        "enters joint cross-attention with token_dim=%s route_anchor_mode=%s",
        cross_version.upper(),
        control_spec.raw_channels,
        control_spec.packed_channels,
        reference_spec.token_dim,
        reference_spec.normalized_route_anchor_mode,
    )
    if is_cross_v4:
        logger.info(
            "Cross V4 priors: tissue_per_class=%s cell_per_class=%s global_style=%s "
            "bias_scale=%s warmup_steps=%s biased_double_blocks=%s",
            reference_spec.tissue_prior_tokens_per_class,
            reference_spec.cell_prior_tokens_per_class,
            reference_spec.global_style_tokens,
            getattr(args, "cross_v4_bias_scale", 1.0),
            getattr(args, "cross_v4_bias_warmup_steps", 1000),
            getattr(args, "cross_v4_biased_double_blocks", "last"),
        )
        logger.info(
            "Cross V4 early diagnostics: steps=%s interval=%s jsonl=%s max_cuda_memory_gb=%s",
            getattr(args, "cross_v4_diagnose_steps", "500,1000,1500,2000"),
            getattr(args, "cross_v4_diagnose_interval", 0),
            cross_v4_diagnose_jsonl,
            max_cuda_memory_gb or "disabled",
        )
    if reference_style_loss_weight > 0.0:
        logger.info(
            "Using Cross V3 reference region stain/style loss: weight=%s interval=%s "
            "tissue=%s nuclei=%s mean/std/cov=%s/%s/%s",
            reference_style_loss_weight,
            reference_style_loss_interval,
            reference_style_loss_config.tissue_weight,
            reference_style_loss_config.nuclei_weight,
            reference_style_loss_config.mean_weight,
            reference_style_loss_config.std_weight,
            reference_style_loss_config.covariance_weight,
        )
    if ref_swap_loss_weight > 0.0:
        logger.info(
            "Using Cross V3 ref-swap sensitivity loss: weight=%s interval=%s margin=%s variants=%s",
            ref_swap_loss_weight,
            ref_swap_loss_interval,
            ref_swap_margin,
            ",".join(ref_swap_variants),
        )
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    _apply_training_prompt_policy(dataset.records, args)

    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    ).to(accelerator.device)
    text_encoder_two = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant,
    ).to(accelerator.device)

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    flux_transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.bfloat16,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )

    if args.controlnet_model_name_or_path:
        flux_controlnet = _load_cross_v3_controlnet_checkpoint(
            args.controlnet_model_name_or_path,
            control_spec=control_spec,
        )
    else:
        flux_controlnet = FluxControlNetModel.from_transformer(
            flux_transformer,
            attention_head_dim=flux_transformer.config["attention_head_dim"],
            num_attention_heads=flux_transformer.config["num_attention_heads"],
            num_layers=args.num_double_layers,
            num_single_layers=args.num_single_layers,
        )

    patch_controlnet_x_embedder(flux_controlnet, control_spec.packed_channels)
    logger.info("Patched controlnet_x_embedder to packed width %s for cross-v3", control_spec.packed_channels)

    tmp_pipeline = FluxControlNetPipeline(
        scheduler=noise_scheduler,
        vae=None,
        text_encoder=text_encoder_one,
        tokenizer=tokenizer_one,
        text_encoder_2=text_encoder_two,
        tokenizer_2=tokenizer_two,
        transformer=flux_transformer,
        controlnet=flux_controlnet,
    )
    tmp_pipeline.to(accelerator.device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    prompt_cache, empty_prompt, text_ids = _build_prompt_cache(
        pipeline=tmp_pipeline,
        prompts=[record["prompt"] for record in dataset.records],
        weight_dtype=weight_dtype,
        batch_size=args.prompt_batch_size,
        device=accelerator.device,
    )

    del tmp_pipeline, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    flux_transformer.to(accelerator.device, dtype=weight_dtype)
    flux_transformer.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)
    vae.eval()
    vae.requires_grad_(False)
    flux_controlnet.to(accelerator.device, dtype=weight_dtype)
    flux_controlnet.train()
    controlnet_trainable_names = _configure_controlnet_trainable_params(
        flux_controlnet,
        mode=getattr(args, "controlnet_train_mode", "all"),
        train_x_embedder=bool(getattr(args, "controlnet_train_x_embedder", False)),
        train_last_n_blocks=max(0, int(getattr(args, "controlnet_train_last_n_blocks", 0) or 0)),
        train_last_n_single_blocks=max(0, int(getattr(args, "controlnet_train_last_n_single_blocks", 0) or 0)),
    )
    logger.info(
        "ControlNet train mode=%s trainable_tensors=%s trainable_params=%s sample_names=%s",
        getattr(args, "controlnet_train_mode", "all"),
        len(controlnet_trainable_names),
        sum(param.numel() for param in flux_controlnet.parameters() if param.requires_grad),
        ", ".join(controlnet_trainable_names[:12]),
    )
    for module in modules.values():
        module.to(accelerator.device, dtype=weight_dtype)
        module.train()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if args.enable_xformers_memory_efficient_attention and is_xformers_available():
        flux_transformer.enable_xformers_memory_efficient_attention()
        flux_controlnet.enable_xformers_memory_efficient_attention()
    if is_cross_v4:
        biased_indices = parse_cross_v4_block_indices(
            getattr(args, "cross_v4_biased_double_blocks", "last"),
            total_blocks=len(getattr(flux_transformer, "transformer_blocks", []) or []),
        )
        install_summary = install_cross_v4_attention_processors(
            flux_transformer,
            biased_double_block_indices=biased_indices,
        )
        logger.info(
            "Installed Cross V4 attention processors: double_blocks=%s biased=%s single_blocks=%s",
            install_summary.double_blocks,
            list(install_summary.biased_double_blocks),
            install_summary.single_blocks,
        )
    if args.gradient_checkpointing:
        flux_controlnet.enable_gradient_checkpointing()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate *= (
            args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    conditioning_learning_rate = float(
        getattr(args, "conditioning_learning_rate", None)
        if getattr(args, "conditioning_learning_rate", None) is not None
        else args.learning_rate
    )

    if args.use_8bit_adam:
        import bitsandbytes as bnb

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    controlnet_lr_params = [param for param in flux_controlnet.parameters() if param.requires_grad]
    conditioning_lr_params = [
        param for module in modules.values() for param in module.parameters() if param.requires_grad
    ]
    optimizer_param_groups = []
    if controlnet_lr_params:
        optimizer_param_groups.append({"params": controlnet_lr_params, "lr": args.learning_rate})
    if conditioning_lr_params:
        optimizer_param_groups.append({"params": conditioning_lr_params, "lr": conditioning_learning_rate})
    if not optimizer_param_groups:
        raise ValueError("No trainable parameters were added to the optimizer.")
    logger.info(
        "Optimizer LR groups: controlnet_lr=%s params=%s, conditioning_lr=%s params=%s",
        args.learning_rate,
        sum(param.numel() for param in controlnet_lr_params),
        conditioning_learning_rate,
        sum(param.numel() for param in conditioning_lr_params),
    )
    optimizer = optimizer_class(
        optimizer_param_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    dataloader_kwargs = {
        "shuffle": True,
        "collate_fn": collate_cross_v3_batch,
        "batch_size": args.train_batch_size,
        "num_workers": args.dataloader_num_workers,
        "pin_memory": True,
    }
    if args.dataloader_num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = max(1, int(getattr(args, "dataloader_prefetch_factor", 2) or 2))
    train_dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=(
            (args.max_train_steps or args.num_train_epochs * num_update_steps_per_epoch)
            * accelerator.num_processes
        ),
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    trainable_modules = [flux_controlnet, *modules.values()]
    prepared = accelerator.prepare(*trainable_modules, optimizer, train_dataloader, lr_scheduler)
    prepared_models = prepared[: len(trainable_modules)]
    flux_controlnet = prepared_models[0]
    modules = dict(zip(modules.keys(), prepared_models[1:]))
    optimizer = prepared[len(trainable_modules)]
    train_dataloader = prepared[len(trainable_modules) + 1]
    lr_scheduler = prepared[len(trainable_modules) + 2]

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)))

    prompt_cache = {
        prompt: (embeds.to(device=accelerator.device), pooled.to(device=accelerator.device))
        for prompt, (embeds, pooled) in prompt_cache.items()
    }
    empty_prompt_embeds = empty_prompt[0].to(device=accelerator.device)
    empty_pooled = empty_prompt[1].to(device=accelerator.device)
    text_ids = text_ids.to(device=accelerator.device)

    logger.info("***** Running Phase 5.3 cross-v3 training *****")
    logger.info("  Num examples = %s", len(dataset))
    logger.info("  Num Epochs = %s", args.num_train_epochs)
    logger.info(
        "  Total batch size = %s",
        args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps,
    )
    logger.info("  Total optimization steps = %s", args.max_train_steps)

    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        checkpoint_path = (
            os.path.join(args.output_dir, args.resume_from_checkpoint)
            if args.resume_from_checkpoint != "latest"
            else _latest_checkpoint(args.output_dir)
        )
        if checkpoint_path is not None:
            accelerator.load_state(checkpoint_path)
            global_step = int(Path(checkpoint_path).name.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        total=args.max_train_steps,
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == timestep).nonzero().item() for timestep in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(flux_controlnet):
                bsz = int(batch["target_image"].shape[0])
                in_self_reconstruction_warmup = global_step < self_reconstruction_warmup_steps
                self_reconstruction_sample_mask = torch.zeros(bsz, device=accelerator.device, dtype=torch.bool)
                if in_self_reconstruction_warmup:
                    self_reconstruction_sample_mask.fill_(True)
                    training_batch = _use_self_reconstruction_reference(batch)
                elif self_reconstruction_sample_prob > 0.0:
                    self_reconstruction_sample_mask = (
                        torch.rand(bsz, device=accelerator.device) < self_reconstruction_sample_prob
                    )
                    training_batch = _insert_self_reconstruction_samples(batch, self_reconstruction_sample_mask)
                else:
                    training_batch = batch

                counterfactual_sample_mask = _batch_mode_mask(training_batch, "counterfactual", device=accelerator.device)
                cross_sample_mask = ~(counterfactual_sample_mask | self_reconstruction_sample_mask)

                reference_encoding = None
                target_metadata = None
                if is_cross_v4:
                    (
                        pixel_latents,
                        control_tensor,
                        reference_encoding,
                        target_metadata,
                        feature_stats,
                    ) = _build_cross_v4_control_batch(
                        batch=training_batch,
                        modules=modules,
                        vae=vae,
                        weight_dtype=weight_dtype,
                    )
                    reference_tokens = reference_encoding.local_tokens
                else:
                    pixel_latents, control_tensor, reference_tokens, feature_stats = _build_cross_v3_control_batch(
                        batch=training_batch,
                        modules=modules,
                        vae=vae,
                        weight_dtype=weight_dtype,
                    )
                if accelerator.is_main_process and global_step == 0:
                    logger.info(
                        "[SCALE-CHECK] step=%s target_tissue_abs_mean=%.8g "
                        "reference_tissue_abs_mean=%.8g target/ref_ratio=%.8g "
                        "target_tissue_abs_max=%.8g reference_tissue_abs_max=%.8g "
                        "target_nuclei_abs_mean=%.8g reference_nuclei_abs_mean=%.8g "
                        "reference_token_abs_mean=%.8g target_feature_hw=%sx%s "
                        "reference_feature_hw=%sx%s target_tissue_encoding=%s "
                        "target_one_hot_scale=%.8g",
                        global_step,
                        feature_stats["target_tissue_abs_mean"],
                        feature_stats["reference_tissue_abs_mean"],
                        feature_stats["target_to_reference_tissue_abs_mean_ratio"],
                        feature_stats["target_tissue_abs_max"],
                        feature_stats["reference_tissue_abs_max"],
                        feature_stats["target_nuclei_abs_mean"],
                        feature_stats["reference_nuclei_abs_mean"],
                        feature_stats["reference_token_abs_mean"],
                        int(feature_stats["target_feature_height"]),
                        int(feature_stats["target_feature_width"]),
                        int(feature_stats["reference_feature_height"]),
                        int(feature_stats["reference_feature_width"]),
                        target_tissue_encoding,
                        float(getattr(args, "target_one_hot_scale", 4.0)),
                    )
                bsz = pixel_latents.shape[0]

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
                batch_prompt, batch_pooled = _resolve_prompt_batch(
                    prompts=training_batch["prompts"],
                    prompt_cache=prompt_cache,
                    empty_prompt_embeds=empty_prompt_embeds,
                    empty_pooled=empty_pooled,
                    proportion_empty_prompts=0.0,
                )
                joint_attention_kwargs = None
                cross_v4_context = None
                cross_v4_diagnostics = None
                cross_v4_run_diagnostics = (
                    is_cross_v4
                    and accelerator.sync_gradients
                    and _should_run_cross_v4_diagnostics(args, global_step + 1)
                )
                if is_cross_v4:
                    if reference_encoding is None or target_metadata is None:
                        raise RuntimeError("Cross V4 requires reference encoding and target metadata.")
                    (
                        batch_context,
                        context_ids,
                        joint_attention_kwargs,
                        cross_v4_context,
                        cross_v4_diagnostics,
                    ) = _build_cross_v4_context_and_kwargs(
                        prompt_embeds=batch_prompt,
                        text_ids=text_ids,
                        reference_encoding=reference_encoding,
                        target_metadata=target_metadata,
                        prior_token_bank=modules["prior_token_bank"],
                        args=args,
                        global_step=global_step,
                        dtype=weight_dtype,
                        diagnose_attention=cross_v4_run_diagnostics,
                    )
                else:
                    batch_context, context_ids = append_cross_v3_reference_context(
                        prompt_embeds=batch_prompt,
                        text_ids=text_ids,
                        reference_tokens=reference_tokens,
                    )

                noise = torch.randn_like(packed_pixel_latents)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=packed_pixel_latents.device)
                sigmas = get_sigmas(timesteps, n_dim=packed_pixel_latents.ndim, dtype=packed_pixel_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * packed_pixel_latents + sigmas * noise

                guidance_vec = None
                if flux_transformer.config.guidance_embeds:
                    guidance_vec = torch.full((bsz,), args.guidance_scale, device=accelerator.device, dtype=weight_dtype)

                latent_image_ids = _prepare_packed_latent_image_ids(
                    packed_height=pixel_latents.shape[2] // 2,
                    packed_width=pixel_latents.shape[3] // 2,
                    device=accelerator.device,
                    dtype=weight_dtype,
                )
                if latent_image_ids.shape[0] != noisy_model_input.shape[1]:
                    raise ValueError(
                        "FLUX img_ids length must match packed latent sequence length: "
                        f"img_ids={tuple(latent_image_ids.shape)}, "
                        f"packed_latents={tuple(noisy_model_input.shape)}, "
                        f"unpacked_latents={tuple(pixel_latents.shape)}"
                    )

                controlnet_block_samples, controlnet_single_block_samples = flux_controlnet(
                    hidden_states=noisy_model_input,
                    controlnet_cond=control_image,
                    timestep=timesteps / 1000,
                    guidance=guidance_vec,
                    pooled_projections=batch_pooled,
                    encoder_hidden_states=batch_prompt,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )
                transformer_controlnet_block_samples = _cast_control_samples(
                    controlnet_block_samples,
                    weight_dtype,
                )
                transformer_controlnet_single_block_samples = _cast_control_samples(
                    controlnet_single_block_samples,
                    weight_dtype,
                )

                noise_pred = flux_transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance_vec,
                    pooled_projections=batch_pooled,
                    encoder_hidden_states=batch_context,
                    controlnet_block_samples=transformer_controlnet_block_samples,
                    controlnet_single_block_samples=transformer_controlnet_single_block_samples,
                    txt_ids=context_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )[0]

                ref_check_diff = None
                ref_check_logs: dict[str, float] = {}
                if ref_check_step > 0 and global_step == ref_check_step:
                    with torch.no_grad():
                        zero_joint_attention_kwargs = None
                        if is_cross_v4:
                            if reference_encoding is None or target_metadata is None:
                                raise RuntimeError("Cross V4 ref-check requires reference encoding and target metadata.")
                            zero_encoding = apply_cross_v4_reference_encoding_mode(reference_encoding, "zero-ref")
                            zero_context, zero_context_ids, zero_joint_attention_kwargs, _, _ = (
                                _build_cross_v4_context_and_kwargs(
                                    prompt_embeds=batch_prompt,
                                    text_ids=text_ids,
                                    reference_encoding=zero_encoding,
                                    target_metadata=target_metadata,
                                    prior_token_bank=modules["prior_token_bank"],
                                    args=args,
                                    global_step=global_step,
                                    dtype=weight_dtype,
                                )
                            )
                        else:
                            zero_context, zero_context_ids = append_cross_v3_reference_context(
                                prompt_embeds=batch_prompt,
                                text_ids=text_ids,
                                reference_tokens=torch.zeros_like(reference_tokens),
                            )
                        zero_ref_with_control = flux_transformer(
                            hidden_states=noisy_model_input,
                            timestep=timesteps / 1000,
                            guidance=guidance_vec,
                            pooled_projections=batch_pooled,
                            encoder_hidden_states=zero_context,
                            controlnet_block_samples=transformer_controlnet_block_samples,
                            controlnet_single_block_samples=transformer_controlnet_single_block_samples,
                            txt_ids=zero_context_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=zero_joint_attention_kwargs,
                            return_dict=False,
                        )[0]
                        zero_controlnet_block_samples = _zero_control_samples_like(
                            transformer_controlnet_block_samples
                        )
                        zero_controlnet_single_block_samples = _zero_control_samples_like(
                            transformer_controlnet_single_block_samples
                        )
                        with_ref_zero_control = flux_transformer(
                            hidden_states=noisy_model_input,
                            timestep=timesteps / 1000,
                            guidance=guidance_vec,
                            pooled_projections=batch_pooled,
                            encoder_hidden_states=batch_context,
                            controlnet_block_samples=zero_controlnet_block_samples,
                            controlnet_single_block_samples=zero_controlnet_single_block_samples,
                            txt_ids=context_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=joint_attention_kwargs,
                            return_dict=False,
                        )[0]
                        zero_ref_zero_control = flux_transformer(
                            hidden_states=noisy_model_input,
                            timestep=timesteps / 1000,
                            guidance=guidance_vec,
                            pooled_projections=batch_pooled,
                            encoder_hidden_states=zero_context,
                            controlnet_block_samples=zero_controlnet_block_samples,
                            controlnet_single_block_samples=zero_controlnet_single_block_samples,
                            txt_ids=zero_context_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=zero_joint_attention_kwargs,
                            return_dict=False,
                        )[0]
                        with_control_ref_effect = noise_pred.detach() - zero_ref_with_control.detach()
                        zero_control_ref_effect = (
                            with_ref_zero_control.detach() - zero_ref_zero_control.detach()
                        )
                        ref_check_diff = _mean_abs_diff(noise_pred.detach(), zero_ref_with_control.detach())
                        ref_check_logs = {
                            "ref_check_mean_abs_noise_pred_diff": ref_check_diff.detach().item(),
                            "control_check_mean_abs_noise_pred_diff": _mean_abs_diff(
                                noise_pred.detach(),
                                with_ref_zero_control.detach(),
                            ).detach().item(),
                            "ref_check_zero_control_mean_abs_noise_pred_diff": _mean_abs_diff(
                                with_ref_zero_control.detach(),
                                zero_ref_zero_control.detach(),
                            ).detach().item(),
                            "control_check_zero_ref_mean_abs_noise_pred_diff": _mean_abs_diff(
                                zero_ref_with_control.detach(),
                                zero_ref_zero_control.detach(),
                            ).detach().item(),
                            "ref_control_interaction_mean_abs_noise_pred_diff": _mean_abs_diff(
                                with_control_ref_effect,
                                zero_control_ref_effect,
                            ).detach().item(),
                        }
                    if accelerator.is_main_process:
                        logger.info(
                            "[REF-CHECK] step=%s ref_with_control_diff=%.8g "
                            "control_with_ref_diff=%.8g ref_zero_control_diff=%.8g "
                            "control_zero_ref_diff=%.8g ref_control_interaction_diff=%.8g "
                            "ref_token_abs_mean=%.8g ref_token_abs_max=%.8g output_init_std=%.8g",
                            global_step,
                            float(ref_check_logs["ref_check_mean_abs_noise_pred_diff"]),
                            float(ref_check_logs["control_check_mean_abs_noise_pred_diff"]),
                            float(ref_check_logs["ref_check_zero_control_mean_abs_noise_pred_diff"]),
                            float(ref_check_logs["control_check_zero_ref_mean_abs_noise_pred_diff"]),
                            float(ref_check_logs["ref_control_interaction_mean_abs_noise_pred_diff"]),
                            float(reference_tokens.detach().float().abs().mean().cpu().item()),
                            float(reference_tokens.detach().float().abs().max().cpu().item()),
                            float(getattr(args, "reference_token_output_init_std", 0.02)),
                        )

                target_velocity = noise - packed_pixel_latents
                per_sample_loss = _per_sample_mse(noise_pred, target_velocity)
                denoise_loss = per_sample_loss.mean()
                cross_denoise_loss = _masked_mean_or_zero(per_sample_loss, cross_sample_mask)
                counterfactual_denoise_loss = _masked_mean_or_zero(per_sample_loss, counterfactual_sample_mask)
                self_reconstruction_denoise_loss = _masked_mean_or_zero(
                    per_sample_loss,
                    self_reconstruction_sample_mask,
                )
                style_loss = noise_pred.new_zeros(())
                style_tissue_loss = noise_pred.new_zeros(())
                style_nuclei_loss = noise_pred.new_zeros(())
                style_tissue_regions = 0
                style_nuclei_regions = 0
                should_compute_style_loss = (
                    reference_style_loss_weight > 0.0
                    and reference_style_loss_interval > 0
                    and global_step % reference_style_loss_interval == 0
                )
                if should_compute_style_loss:
                    prediction_rgb = _decode_packed_model_prediction(
                        vae=vae,
                        packed_noisy_latents=noisy_model_input,
                        noise_pred=noise_pred,
                        sigmas=sigmas,
                        latent_channels=pixel_latents.shape[1],
                        latent_height=pixel_latents.shape[2],
                        latent_width=pixel_latents.shape[3],
                        weight_dtype=weight_dtype,
                    )
                    style_terms = regional_stain_style_loss(
                        prediction=prediction_rgb,
                        reference=training_batch["reference_image"].to(
                            device=accelerator.device,
                            dtype=prediction_rgb.dtype,
                        ),
                        target_tissue_mask=training_batch["target_tissue_mask"].to(accelerator.device),
                        reference_tissue_mask=training_batch["reference_tissue_mask"].to(accelerator.device),
                        target_nuclei_mask=training_batch["target_nuclei_mask"].to(accelerator.device),
                        reference_nuclei_mask=training_batch["reference_nuclei_mask"].to(accelerator.device),
                        config=reference_style_loss_config,
                    )
                    style_loss = style_terms["total"].to(dtype=denoise_loss.dtype)
                    style_tissue_loss = style_terms["tissue"].to(dtype=denoise_loss.dtype)
                    style_nuclei_loss = style_terms["nuclei"].to(dtype=denoise_loss.dtype)
                    style_tissue_regions = int(style_terms["tissue_regions"])
                    style_nuclei_regions = int(style_terms["nuclei_regions"])

                swap_loss = noise_pred.new_zeros(())
                ref_variant_loss_logs: dict[str, float] = {}
                should_compute_swap_loss = (
                    ref_swap_loss_weight > 0.0
                    and ref_swap_loss_interval > 0
                    and global_step % ref_swap_loss_interval == 0
                    and bool(ref_swap_variants)
                )
                if should_compute_swap_loss:
                    swapped_per_sample_losses = []
                    for variant in ref_swap_variants:
                        swapped_joint_attention_kwargs = None
                        if is_cross_v4:
                            if reference_encoding is None or target_metadata is None:
                                raise RuntimeError("Cross V4 ref-swap requires reference encoding and target metadata.")
                            swapped_encoding = _build_swapped_cross_v4_reference_encoding(reference_encoding, variant)
                            if swapped_encoding is None:
                                continue
                            swapped_context, swapped_context_ids, swapped_joint_attention_kwargs, _, _ = (
                                _build_cross_v4_context_and_kwargs(
                                    prompt_embeds=batch_prompt,
                                    text_ids=text_ids,
                                    reference_encoding=swapped_encoding,
                                    target_metadata=target_metadata,
                                    prior_token_bank=modules["prior_token_bank"],
                                    args=args,
                                    global_step=global_step,
                                    dtype=weight_dtype,
                                )
                            )
                        else:
                            swapped_reference_tokens = _build_swapped_reference_tokens(reference_tokens, variant)
                            if swapped_reference_tokens is None:
                                continue
                            swapped_context, swapped_context_ids = append_cross_v3_reference_context(
                                prompt_embeds=batch_prompt,
                                text_ids=text_ids,
                                reference_tokens=swapped_reference_tokens,
                            )
                        swapped_noise_pred = flux_transformer(
                            hidden_states=noisy_model_input,
                            timestep=timesteps / 1000,
                            guidance=guidance_vec,
                            pooled_projections=batch_pooled,
                            encoder_hidden_states=swapped_context,
                            controlnet_block_samples=transformer_controlnet_block_samples,
                            controlnet_single_block_samples=transformer_controlnet_single_block_samples,
                            txt_ids=swapped_context_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=swapped_joint_attention_kwargs,
                            return_dict=False,
                        )[0]
                        swapped_per_sample_losses.append(_per_sample_mse(swapped_noise_pred, target_velocity))
                        ref_variant_loss_logs[f"ref_{variant}_denoise_loss"] = (
                            swapped_per_sample_losses[-1].mean().detach().item()
                        )
                    swap_loss = ref_swap_sensitivity_loss(
                        per_sample_loss,
                        swapped_per_sample_losses,
                        margin=ref_swap_margin,
                    ).to(dtype=denoise_loss.dtype)
                loss = (
                    denoise_loss
                    + reference_style_loss_weight * style_loss
                    + ref_swap_loss_weight * swap_loss
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [param for model in [flux_controlnet, *modules.values()] for param in model.parameters() if param.requires_grad],
                        args.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    save_path = _save_checkpoint(accelerator, args, global_step)
                    _save_cross_v3_artifacts(
                        save_path,
                        args,
                        flux_controlnet=flux_controlnet,
                        modules=modules,
                        unwrap_model=unwrap_model,
                        control_spec=control_spec,
                        reference_spec=reference_spec,
                        cross_version=cross_version,
                    )
                    logger.info("Saved eval-ready Phase 5.3 cross-%s artifacts to %s", cross_version, save_path)

            logs = {
                "loss": loss.detach().item(),
                "denoise_loss": denoise_loss.detach().item(),
                "cross_denoise_loss": cross_denoise_loss.detach().item(),
                "counterfactual_denoise_loss": counterfactual_denoise_loss.detach().item(),
                "self_reconstruction_denoise_loss": self_reconstruction_denoise_loss.detach().item(),
                "style_loss": style_loss.detach().item(),
                "style_tissue_loss": style_tissue_loss.detach().item(),
                "style_nuclei_loss": style_nuclei_loss.detach().item(),
                "style_tissue_regions": style_tissue_regions,
                "style_nuclei_regions": style_nuclei_regions,
                "ref_swap_loss": swap_loss.detach().item(),
                "ref_normal_denoise_loss": denoise_loss.detach().item(),
                "self_reconstruction_samples": int(self_reconstruction_sample_mask.sum().detach().item()),
                "counterfactual_samples": int(counterfactual_sample_mask.sum().detach().item()),
                "cross_samples": int(cross_sample_mask.sum().detach().item()),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            logs.update(ref_variant_loss_logs)
            if ref_check_diff is not None:
                logs.update(ref_check_logs)
            if global_step <= 1:
                logs.update(feature_stats)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _save_cross_v3_artifacts(
            args.output_dir,
            args,
            flux_controlnet=flux_controlnet,
            modules=modules,
            unwrap_model=unwrap_model,
            control_spec=control_spec,
            reference_spec=reference_spec,
            cross_version=cross_version,
        )
        logger.info("Saved Phase 5.3 cross-%s artifacts to %s", cross_version, args.output_dir)

    accelerator.end_training()

"""Phase 5.3 Cross V1 training — IP-Adapter reference attention for Flux ControlNet.

This module is self-contained and does not modify any existing V0/inpaint code.
It duplicates the shared training loop from flux_phase5.py and adds:
- IP-Adapter attention installation on the frozen Flux transformer
- ReferenceImageEncoder (UNI2-h + Perceiver resampler) for reference appearance injection
- joint_attention_kwargs passing in the transformer forward call
- Separate save strategy for IP-Adapter and ref_encoder modules

Multi-GPU DDP fix: trainable sub-modules that can't be directly wrapped (because they
live inside frozen parents or contain huge frozen backbones) are extracted into small
wrapper nn.Modules and passed through accelerator.prepare() independently.
"""

from __future__ import annotations

import argparse
import contextlib
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
from packaging import version
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, T5EncoderModel

from controlnet_train.data import CrossReconstructionDataset
from controlnet_train.data.common import (
    default_prompt_for_dataset,
    load_image_tensor,
    load_nuclei_mask,
    load_tissue_mask,
)
from controlnet_train.modules import (
    HierarchicalTissueEmbedding,
    NucleiConditionEncoder,
    TissueConditionDownsampler,
)
from controlnet_train.modules.cross_v1_conditioning import (
    CROSS_V1_SPATIAL_REFERENCE_TARGET,
    CROSS_V1_SPATIAL_REFERENCE_TARGET_DELTA,
    CROSS_V1_SPATIAL_TARGET_ONLY,
    CrossV1ControlSpec,
    build_cross_v1_condition,
    normalize_cross_v1_spatial_mode,
)
from controlnet_train.modules.reference_image_encoder import ReferenceImageEncoder
from controlnet_train.training.conditioning import patch_controlnet_x_embedder
from controlnet_train.training.cross_v1_losses import (
    RegionalStainStyleLossConfig,
    per_sample_mse,
    ref_swap_sensitivity_loss,
    regional_stain_style_loss,
    self_reconstruction_l1_loss,
    uni_token_cosine_perceptual_loss,
    unpack_flux_packed_latents,
)

if is_wandb_available():
    import wandb  # noqa: F401

logger = get_logger(__name__)
if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


# ---------------------------------------------------------------------------
# IP-Adapter installation and helpers
# ---------------------------------------------------------------------------
class IPAdapterListProjection(nn.Module):
    """Wraps IPAdapterFullImageProjection to handle list input/output."""
    def __init__(self, proj: nn.Module):
        super().__init__()
        self.proj = proj

    def forward(self, image_embeds):
        target_dtype = next(self.proj.parameters()).dtype
        if isinstance(image_embeds, list):
            return [self.proj(embed).to(dtype=target_dtype) for embed in image_embeds]
        return self.proj(image_embeds).to(dtype=target_dtype)


class FluxSingleIPAdapterAttnProcessor2_0(nn.Module):
    """IP-Adapter processor for FLUX single-stream blocks.

    The single stream contains [text_tokens, image_tokens]. Reference attention is
    applied only to image-token queries; text-token outputs remain the native
    self-attention outputs from the frozen FLUX block.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        cross_attention_dim: int,
        num_tokens: int | tuple[int, ...] = (16,),
        scale: float | list[float] = 1.0,
    ) -> None:
        super().__init__()
        if not isinstance(num_tokens, (tuple, list)):
            num_tokens = (int(num_tokens),)
        self.num_tokens = tuple(int(value) for value in num_tokens)
        if not isinstance(scale, list):
            scale = [float(scale)] * len(self.num_tokens)
        if len(scale) != len(self.num_tokens):
            raise ValueError("scale must have the same length as num_tokens.")
        self.scale = [float(value) for value in scale]
        self.to_k_ip = nn.ModuleList(
            [nn.Linear(cross_attention_dim, hidden_size, bias=False) for _ in self.num_tokens]
        )
        self.to_v_ip = nn.ModuleList(
            [nn.Linear(cross_attention_dim, hidden_size, bias=False) for _ in self.num_tokens]
        )

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        ip_hidden_states: list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
        txt_seq_len: int | None = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None:
            raise ValueError("FluxSingleIPAdapterAttnProcessor2_0 expects pre-concatenated single-stream states.")
        batch_size, _, _ = hidden_states.shape
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        output = output.to(query.dtype)

        if ip_hidden_states:
            txt_seq_len = int(txt_seq_len or 0)
            if txt_seq_len < 0 or txt_seq_len > output.shape[1]:
                raise ValueError(
                    f"txt_seq_len must be within [0, {output.shape[1]}], got {txt_seq_len}."
                )
            image_query = query[:, :, txt_seq_len:, :]
            if image_query.shape[2] > 0:
                image_ip_output = output.new_zeros((batch_size, image_query.shape[2], output.shape[2]))
                for current_ip_hidden_states, scale, to_k_ip, to_v_ip in zip(
                    ip_hidden_states,
                    self.scale,
                    self.to_k_ip,
                    self.to_v_ip,
                ):
                    if scale == 0:
                        continue
                    ip_key = to_k_ip(current_ip_hidden_states)
                    ip_value = to_v_ip(current_ip_hidden_states)
                    ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                    ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                    ip_attn = torch.nn.functional.scaled_dot_product_attention(
                        image_query,
                        ip_key,
                        ip_value,
                        dropout_p=0.0,
                        is_causal=False,
                    )
                    ip_attn = ip_attn.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                    image_ip_output = image_ip_output + float(scale) * ip_attn.to(output.dtype)
                image_output = output[:, txt_seq_len:, :] + image_ip_output
                if txt_seq_len > 0:
                    output = torch.cat([output[:, :txt_seq_len, :], image_output], dim=1)
                else:
                    output = image_output

        return output.to(hidden_states.dtype)


def install_flux_ip_adapter_attention(
    transformer: FluxTransformer2DModel,
    hidden_dim: int = 3072,
    cross_attention_dim: int = 3072,
    num_tokens: int = 16,
    scale: float = 1.0,
    ip_init_gain: float = 0.1,
    num_single_layers: int = 0,
) -> None:
    """Install IP-Adapter attention processors on FLUX double and last-N single blocks."""
    from diffusers.models.attention_processor import FluxIPAdapterJointAttnProcessor2_0
    from diffusers.models.embeddings import IPAdapterFullImageProjection

    class FluxIPAdapterJointAttnProcessorWithTxtSeqLen(FluxIPAdapterJointAttnProcessor2_0):
        def __call__(
            self,
            attn,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            image_rotary_emb: torch.Tensor | None = None,
            ip_hidden_states: list[torch.Tensor] | None = None,
            ip_adapter_masks: torch.Tensor | None = None,
            txt_seq_len: int | None = None,
        ) -> torch.Tensor:
            return super().__call__(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
                ip_hidden_states=ip_hidden_states,
                ip_adapter_masks=ip_adapter_masks,
            )

    raw_proj = IPAdapterFullImageProjection(
        image_embed_dim=cross_attention_dim,
        cross_attention_dim=cross_attention_dim,
    )
    transformer.encoder_hid_proj = IPAdapterListProjection(raw_proj)

    for block in transformer.transformer_blocks:
        processor = FluxIPAdapterJointAttnProcessorWithTxtSeqLen(
            hidden_size=hidden_dim,
            cross_attention_dim=cross_attention_dim,
            num_tokens=(num_tokens,),
            scale=[scale],
        )
        for linear in processor.to_k_ip:
            _init_ip_adapter_linear(linear, gain=ip_init_gain)
        for linear in processor.to_v_ip:
            _init_ip_adapter_linear(linear, gain=ip_init_gain)
        block.attn.set_processor(processor)

    single_blocks = list(getattr(transformer, "single_transformer_blocks", []))
    if num_single_layers > 0:
        for block in single_blocks[-int(num_single_layers):]:
            processor = FluxSingleIPAdapterAttnProcessor2_0(
                hidden_size=hidden_dim,
                cross_attention_dim=cross_attention_dim,
                num_tokens=(num_tokens,),
                scale=[scale],
            )
            for linear in processor.to_k_ip:
                _init_ip_adapter_linear(linear, gain=ip_init_gain)
            for linear in processor.to_v_ip:
                _init_ip_adapter_linear(linear, gain=ip_init_gain)
            block.attn.set_processor(processor)


def _init_ip_adapter_linear(linear: nn.Linear, *, gain: float) -> None:
    """Initialize IP K/V projections so reference content can affect attention from step 0."""
    nn.init.xavier_uniform_(linear.weight, gain=gain)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _linear_init_stats(linear: nn.Linear) -> dict[str, float]:
    weight = linear.weight.detach().float()
    bias = linear.bias.detach().float() if linear.bias is not None else None
    return {
        "weight_l2": float(torch.linalg.vector_norm(weight).item()),
        "weight_max_abs": float(weight.abs().max().item()),
        "bias_max_abs": 0.0 if bias is None else float(bias.abs().max().item()),
    }


def _log_ip_adapter_initialization_stats(
    transformer: FluxTransformer2DModel,
    *,
    ip_init_gain: float,
) -> None:
    """Log initialization magnitudes for the IP projection and K/V branches."""
    encoder_hid_proj = getattr(transformer, "encoder_hid_proj", None)
    if encoder_hid_proj is None:
        logger.warning("IP init check: transformer has no encoder_hid_proj.")
    else:
        encoder_linears = [
            (name, module)
            for name, module in encoder_hid_proj.named_modules()
            if isinstance(module, nn.Linear)
        ]
        if not encoder_linears:
            logger.warning("IP init check: encoder_hid_proj has no Linear layers.")
        for name, linear in encoder_linears:
            stats = _linear_init_stats(linear)
            logger.info(
                "IP init check encoder_hid_proj.%s: weight_l2=%.6f "
                "weight_max_abs=%.6f bias_max_abs=%.6f",
                name or "<root>",
                stats["weight_l2"],
                stats["weight_max_abs"],
                stats["bias_max_abs"],
            )
            if stats["weight_max_abs"] == 0.0:
                logger.warning("IP init check encoder_hid_proj.%s appears zero-initialized.", name)

    kv_linears: list[tuple[str, nn.Linear]] = []
    for prefix, blocks in (
        ("block", getattr(transformer, "transformer_blocks", [])),
        ("single_block", getattr(transformer, "single_transformer_blocks", [])),
    ):
        for block_index, block in enumerate(blocks):
            processor = getattr(block.attn, "processor", None)
            for branch_name in ("to_k_ip", "to_v_ip"):
                branch = getattr(processor, branch_name, None)
                if branch is None:
                    continue
                for index, linear in enumerate(branch):
                    kv_linears.append((f"{prefix}_{block_index}.{branch_name}.{index}", linear))

    if not kv_linears:
        logger.warning("IP init check: no to_k_ip/to_v_ip Linear layers found.")
        return

    max_abs_values = []
    l2_values = []
    for _, linear in kv_linears:
        stats = _linear_init_stats(linear)
        max_abs_values.append(stats["weight_max_abs"])
        l2_values.append(stats["weight_l2"])
    logger.info(
        "IP init check to_k_ip/to_v_ip: gain=%.4f linears=%s "
        "weight_max_abs[min/mean/max]=%.6f/%.6f/%.6f "
        "weight_l2[min/mean/max]=%.6f/%.6f/%.6f",
        ip_init_gain,
        len(kv_linears),
        min(max_abs_values),
        sum(max_abs_values) / len(max_abs_values),
        max(max_abs_values),
        min(l2_values),
        sum(l2_values) / len(l2_values),
        max(l2_values),
    )


def _collect_ip_adapter_modules(transformer: FluxTransformer2DModel) -> dict[str, nn.Module]:
    """Collect all IP-Adapter trainable modules attached to the frozen transformer."""
    from diffusers.models.attention_processor import FluxIPAdapterJointAttnProcessor2_0

    modules: dict[str, nn.Module] = {}
    if hasattr(transformer, "encoder_hid_proj"):
        modules["encoder_hid_proj"] = transformer.encoder_hid_proj
    for i, block in enumerate(transformer.transformer_blocks):
        processor = block.attn.processor
        if isinstance(processor, FluxIPAdapterJointAttnProcessor2_0):
            modules[f"block_{i}_to_k_ip"] = processor.to_k_ip
            modules[f"block_{i}_to_v_ip"] = processor.to_v_ip
    for i, block in enumerate(getattr(transformer, "single_transformer_blocks", [])):
        processor = block.attn.processor
        if isinstance(processor, FluxSingleIPAdapterAttnProcessor2_0):
            modules[f"single_block_{i}_to_k_ip"] = processor.to_k_ip
            modules[f"single_block_{i}_to_v_ip"] = processor.to_v_ip
    return modules


def _split_ip_adapter_module_groups(
    ip_adapter_modules: dict[str, nn.Module],
) -> tuple[list[nn.Module], list[nn.Module]]:
    double_modules = []
    single_modules = []
    for name, module in ip_adapter_modules.items():
        if name.startswith("single_block_"):
            single_modules.append(module)
        else:
            double_modules.append(module)
    return double_modules, single_modules


def _configure_controlnet_trainable_params(
    controlnet: nn.Module,
    *,
    mode: str,
    train_x_embedder: bool = False,
    train_last_n_blocks: int = 0,
    train_last_n_single_blocks: int = 0,
) -> list[str]:
    """Apply a ControlNet unfreeze policy and return trainable parameter names."""
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

    _set_last_n_modules_requires_grad(
        getattr(controlnet, "transformer_blocks", None),
        train_last_n_blocks,
    )
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
    children = list(modules)
    for module in children[-count:]:
        module.requires_grad_(True)


def _sync_ip_adapter_to_transformer(
    ip_wrapper: "IPAdapterTrainableWrapper",
    transformer: FluxTransformer2DModel,
) -> None:
    """Sync trained IP-Adapter weights from wrapper back to transformer processors.

    After accelerator.prepare(), the wrapper holds the DDP-managed parameters.
    We need the transformer's forward pass to use these exact parameter objects
    so that gradients flow correctly.
    """
    transformer.encoder_hid_proj = ip_wrapper.encoder_hid_proj
    for i, block in enumerate(transformer.transformer_blocks):
        k_key = f"block_{i}_to_k_ip"
        v_key = f"block_{i}_to_v_ip"
        if hasattr(ip_wrapper, k_key):
            block.attn.processor.to_k_ip = getattr(ip_wrapper, k_key)
            block.attn.processor.to_v_ip = getattr(ip_wrapper, v_key)
    for i, block in enumerate(getattr(transformer, "single_transformer_blocks", [])):
        k_key = f"single_block_{i}_to_k_ip"
        v_key = f"single_block_{i}_to_v_ip"
        if hasattr(ip_wrapper, k_key):
            block.attn.processor.to_k_ip = getattr(ip_wrapper, k_key)
            block.attn.processor.to_v_ip = getattr(ip_wrapper, v_key)


def patch_flux_single_ip_forward(transformer: FluxTransformer2DModel) -> None:
    """Pass the text/image split index to single-stream IP processors."""
    if getattr(transformer, "_cross_v1_single_ip_forward_patched", False):
        return
    original_forward = transformer.forward

    def forward_with_single_ip_txt_seq_len(*args, **kwargs):
        joint_attention_kwargs = kwargs.get("joint_attention_kwargs")
        encoder_hidden_states = kwargs.get("encoder_hidden_states")
        if encoder_hidden_states is None and len(args) > 1:
            encoder_hidden_states = args[1]
        if joint_attention_kwargs is not None and encoder_hidden_states is not None:
            joint_attention_kwargs = dict(joint_attention_kwargs)
            joint_attention_kwargs.setdefault("txt_seq_len", int(encoder_hidden_states.shape[1]))
            kwargs["joint_attention_kwargs"] = joint_attention_kwargs
        return original_forward(*args, **kwargs)

    transformer.forward = forward_with_single_ip_txt_seq_len
    transformer._cross_v1_single_ip_forward_patched = True


# ---------------------------------------------------------------------------
# ★ FIX 1: Wrapper modules for DDP — 把散落的可训练参数包成 nn.Module
# ---------------------------------------------------------------------------

class IPAdapterTrainableWrapper(nn.Module):
    """Wraps all IP-Adapter trainable sub-modules into one nn.Module for DDP.

    This allows accelerator.prepare() to handle gradient synchronization across GPUs
    without needing to DDP-wrap the entire frozen transformer.
    """

    def __init__(self, ip_adapter_modules: dict[str, nn.Module]):
        super().__init__()
        # Register each module as a sub-module so DDP sees them
        for name, module in ip_adapter_modules.items():
            # nn.Module attribute names can't have dots, replace if needed
            safe_name = name.replace(".", "_")
            self.add_module(safe_name, module)


class RefEncoderTrainableWrapper(nn.Module):
    """Wraps the trainable parts of ReferenceImageEncoder for DDP.

    The frozen UNI2-h backbone (~1.9B params) stays outside DDP to avoid
    broadcast hangs. Only proj_mlp, perceiver_layers, latent_queries,
    and perceiver_norm go through accelerator.prepare().
    """

    def __init__(self, ref_encoder: ReferenceImageEncoder):
        super().__init__()
        self.proj_mlp = ref_encoder.proj_mlp
        self.skip_perceiver = bool(getattr(ref_encoder, "skip_perceiver", False))
        if not self.skip_perceiver:
            self.perceiver_layers = ref_encoder.perceiver_layers
            self.latent_queries = nn.Parameter(ref_encoder.latent_queries.data)
            self.perceiver_norm = ref_encoder.perceiver_norm

    def sync_back(self, ref_encoder: ReferenceImageEncoder) -> None:
        """After prepare(), point ref_encoder's trainable parts to our (DDP-managed) params."""
        ref_encoder.proj_mlp = self.proj_mlp
        if not self.skip_perceiver:
            ref_encoder.perceiver_layers = self.perceiver_layers
            ref_encoder.latent_queries = self.latent_queries
            ref_encoder.perceiver_norm = self.perceiver_norm


# ---------------------------------------------------------------------------
# Control batch builder
# ---------------------------------------------------------------------------

def _encode_images_to_latents(vae: AutoencoderKL, images: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    device = next(vae.parameters()).device
    images = images.to(device=device, dtype=dtype)
    images = images * 2.0 - 1.0
    latents = vae.encode(images).latent_dist.sample()
    return (latents - vae.config.shift_factor) * vae.config.scaling_factor


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
    """Decode one-step denoised model output to RGB in [0, 1] for style losses."""
    pred_original = packed_noisy_latents - sigmas * noise_pred
    pred_latents = _unpack_flux_packed_latents(
        pred_original,
        channels=latent_channels,
        height=latent_height,
        width=latent_width,
    )
    pred_latents = (pred_latents / vae.config.scaling_factor) + vae.config.shift_factor
    decoded = vae.decode(pred_latents.to(device=next(vae.parameters()).device, dtype=weight_dtype), return_dict=False)[0]
    return ((decoded.float() / 2.0) + 0.5).clamp(0.0, 1.0)


def _unpack_flux_packed_latents(
    packed_latents: torch.Tensor,
    *,
    channels: int,
    height: int,
    width: int,
) -> torch.Tensor:
    return unpack_flux_packed_latents(
        packed_latents,
        channels=channels,
        height=height,
        width=width,
    )


def _build_cross_v1_control_batch(
    *,
    batch: dict,
    modules: dict[str, torch.nn.Module],
    vae: AutoencoderKL,
    weight_dtype: torch.dtype,
    spatial_mode: str = CROSS_V1_SPATIAL_REFERENCE_TARGET,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(vae.parameters()).device
    target_image_latent = _encode_images_to_latents(vae, batch["target_image"], weight_dtype)

    target_tissue_feat = modules["tissue_downsampler"](
        modules["hte"](batch["target_tissue_mask"].to(device=device))
    ).to(dtype=weight_dtype)
    target_nuclei_feat = modules["nuclei_encoder"](
        batch["target_nuclei_mask"].to(device=device)
    ).to(dtype=weight_dtype)
    reference_tissue_feat = None
    reference_nuclei_feat = None
    if normalize_cross_v1_spatial_mode(spatial_mode) in {
        CROSS_V1_SPATIAL_REFERENCE_TARGET,
        CROSS_V1_SPATIAL_REFERENCE_TARGET_DELTA,
    }:
        reference_tissue_feat = modules["tissue_downsampler"](
            modules["hte"](batch["reference_tissue_mask"].to(device=device))
        ).to(dtype=weight_dtype)
        reference_nuclei_feat = modules["nuclei_encoder"](
            batch["reference_nuclei_mask"].to(device=device)
        ).to(dtype=weight_dtype)

    control_tensor = build_cross_v1_condition(
        reference_tissue_feat=reference_tissue_feat,
        reference_nuclei_feat=reference_nuclei_feat,
        target_tissue_feat=target_tissue_feat,
        target_nuclei_feat=target_nuclei_feat,
        spatial_mode=spatial_mode,
    )
    return target_image_latent, control_tensor


def _build_ip_adapter_kwargs(
    batch: dict,
    modules: dict[str, torch.nn.Module],
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    transformer: FluxTransformer2DModel,
) -> dict:
    """Build joint_attention_kwargs with pre-projected ip_hidden_states."""
    ref_encoder = modules["ref_encoder"]
    uni_dtype = next(ref_encoder.uni.parameters()).dtype
    ref_ip_features = ref_encoder(
        batch["reference_image"].to(device=accelerator.device, dtype=uni_dtype)
    ).to(dtype=weight_dtype)
    ip_hidden_states = transformer.encoder_hid_proj([ref_ip_features])
    ip_hidden_states = [hs.to(dtype=weight_dtype) for hs in ip_hidden_states]
    return {"ip_hidden_states": ip_hidden_states}


def _use_self_reconstruction_reference(batch: dict) -> dict:
    """Use the target patch as the reference patch for same-patch warmup."""
    warmup_batch = dict(batch)
    warmup_batch["reference_image"] = batch["target_image"]
    warmup_batch["reference_tissue_mask"] = batch["target_tissue_mask"]
    warmup_batch["reference_nuclei_mask"] = batch["target_nuclei_mask"]
    return warmup_batch


def _insert_self_reconstruction_samples(batch: dict, sample_mask: torch.Tensor) -> dict:
    """Use target patches as references for selected batch items."""
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
        return torch.zeros(
            int(batch["target_image"].shape[0]),
            device=device,
            dtype=torch.bool,
        )
    return torch.tensor([str(value) == mode for value in modes], device=device, dtype=torch.bool)


def _masked_mean_or_zero(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=torch.bool)
    if not bool(mask.any().item()):
        return values.new_zeros(())
    return values[mask].mean()


def _use_zero_reference(batch: dict) -> dict:
    swapped = dict(batch)
    swapped["reference_image"] = torch.zeros_like(batch["reference_image"])
    return swapped


def _use_random_reference(batch: dict, random_batch: dict | None = None) -> dict:
    swapped = dict(batch)
    bsz = int(batch["reference_image"].shape[0])
    if random_batch is not None:
        for key in ("reference_image", "reference_tissue_mask", "reference_nuclei_mask"):
            if key not in random_batch:
                raise KeyError(f"random reference batch is missing {key!r}")
            value = random_batch[key]
            if value.shape[0] != bsz:
                raise ValueError(
                    f"random reference batch size {value.shape[0]} does not match current batch size {bsz}"
                )
            swapped[key] = value.to(device=batch[key].device)
        return swapped
    if bsz <= 1:
        raise ValueError(
            "random ref-swap requires train-batch-size > 1 because it uses another "
            "sample in the same batch as the negative reference. Provide a dataset-level "
            "random_batch for batch size 1."
        )
    order = torch.arange(bsz, device=batch["reference_image"].device).roll(1)
    swapped["reference_image"] = batch["reference_image"].index_select(0, order)
    swapped["reference_tissue_mask"] = batch["reference_tissue_mask"].index_select(0, order)
    swapped["reference_nuclei_mask"] = batch["reference_nuclei_mask"].index_select(0, order)
    return swapped


def _parse_ref_swap_variants(value: str | None) -> list[str]:
    variants: list[str] = []
    for raw_part in str(value or "").split(","):
        variant = raw_part.strip().lower()
        if not variant:
            continue
        if variant not in {"zero", "random"}:
            raise ValueError(
                f"Unsupported ref-swap variant {variant!r}; choose zero and/or random."
            )
        if variant not in variants:
            variants.append(variant)
    return variants


def _validate_self_reconstruction_loss_config(
    *,
    sample_prob: float,
    l1_weight: float,
) -> None:
    if sample_prob > 0.0 and l1_weight <= 0.0:
        raise ValueError(
            "--self-reconstruction-sample-prob inserts reference=target samples, "
            "but --self-reconstruction-l1-weight is 0. Set "
            "--self-reconstruction-l1-weight > 0 to train with the reconstruction "
            "loss, or set --self-reconstruction-sample-prob 0 to disable these samples."
        )


class RandomReferenceSampler:
    """Sample real reference patches from metadata for ref-swap negatives."""

    def __init__(self, records: list[dict], *, seed: int | None = None) -> None:
        self.records = list(records)
        if not self.records:
            raise ValueError("RandomReferenceSampler requires at least one metadata record.")
        self.rng = random.Random(seed)

    def sample_for_batch(self, batch: dict, *, device: torch.device | str) -> dict:
        current_ids = set(str(sample_id) for sample_id in batch.get("reference_sample_ids", []))
        current_ids.update(str(sample_id) for sample_id in batch.get("sample_ids", []))
        references = []
        tissue_masks = []
        nuclei_masks = []
        for _ in range(int(batch["reference_image"].shape[0])):
            record = self._choose_record(exclude_sample_ids=current_ids)
            references.append(load_image_tensor(record["reference_image"]))
            tissue_masks.append(load_tissue_mask(record["reference_tissue_mask"]))
            nuclei_masks.append(load_nuclei_mask(record["reference_nuclei_mask"], remap=True))
            current_ids.add(str(record.get("reference_sample_id") or ""))
            current_ids.add(str(record.get("sample_id") or ""))
        return {
            "reference_image": torch.stack(references).to(device=device),
            "reference_tissue_mask": torch.stack(tissue_masks).to(device=device),
            "reference_nuclei_mask": torch.stack(nuclei_masks).to(device=device),
        }

    def _choose_record(self, *, exclude_sample_ids: set[str]) -> dict:
        for _ in range(16):
            record = self.rng.choice(self.records)
            if str(record.get("reference_sample_id") or "") not in exclude_sample_ids:
                return record
        return self.rng.choice(self.records)


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate_cross_batch(examples: list[dict]) -> dict:
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


# ---------------------------------------------------------------------------
# Prompt helpers (copied from flux_phase5.py for independence)
# ---------------------------------------------------------------------------

def _apply_training_prompt_policy(records: list[dict], args: argparse.Namespace) -> None:
    prompt_override = getattr(args, "prompt", None)
    prompt_source = getattr(args, "prompt_source", "dataset")
    if prompt_override:
        for record in records:
            record["prompt"] = prompt_override
        logger.info("Using one explicit training prompt for all %s records", len(records))
        return

    if prompt_source == "metadata":
        logger.info("Using prompts from training metadata")
        return

    if prompt_source != "dataset":
        raise ValueError(f"Unsupported prompt source: {prompt_source}")

    for record in records:
        record["prompt"] = default_prompt_for_dataset(record["dataset"])
    unique_prompts = sorted({record["prompt"] for record in records})
    logger.info(
        "Using dataset-level training prompts: %s unique prompt(s) for %s records",
        len(unique_prompts),
        len(records),
    )


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
                prompt_batch, prompt_2=prompt_batch, device=device,
            )
            for index, prompt in enumerate(prompt_batch):
                prompt_cache[prompt] = (
                    prompt_embeds[index].to(dtype=weight_dtype, device="cpu"),
                    pooled_prompt_embeds[index].to(dtype=weight_dtype, device="cpu"),
                )
        empty_prompt_embeds, empty_pooled, text_ids = pipeline.encode_prompt(
            [""], prompt_2=[""], device=device,
        )
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


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

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


# ★ FIX 3: 保存时不再原地改 dtype，而是拷贝 state_dict 后转换
def _save_condition_modules(
    output_dir: str,
    modules: dict[str, nn.Module],
    unwrap_model: Callable,
    save_dtype: torch.dtype,
    *,
    control_spec: CrossV1ControlSpec,
) -> None:
    state = {
        "cross_v1_spatial_mode": control_spec.spatial_mode,
        "cross_v1_control_spec": {
            "tissue_channels": int(control_spec.tissue_channels),
            "nuclei_channels": int(control_spec.nuclei_channels),
            "spatial_mode": control_spec.spatial_mode,
            "raw_channels": int(control_spec.raw_channels),
            "packed_channels": int(control_spec.packed_channels),
        },
    }
    for name, module in modules.items():
        unwrapped = unwrap_model(module)
        if name == "ref_encoder":
            # Only save trainable parts, skip frozen UNI2-h backbone (~4GB)
            state["ref_encoder_config"] = {
                "uni_embed_dim": int(getattr(unwrapped, "uni_embed_dim", 1536)),
                "hidden_dim": int(getattr(unwrapped, "hidden_dim", 3072)),
                "num_tokens": int(getattr(unwrapped, "num_tokens", 16)),
                "num_output_tokens": int(getattr(unwrapped, "num_output_tokens", getattr(unwrapped, "num_tokens", 16))),
                "num_perceiver_layers": int(
                    getattr(unwrapped, "num_perceiver_layers", len(unwrapped.perceiver_layers))
                ),
                "perceiver_heads": int(getattr(unwrapped, "perceiver_heads", 8)),
                "use_perceiver_self_attn": bool(
                    getattr(unwrapped, "use_perceiver_self_attn", True)
                ),
                "skip_perceiver": bool(getattr(unwrapped, "skip_perceiver", False)),
                "perceiver_cross_gate_init": getattr(
                    unwrapped, "perceiver_cross_gate_init", None
                ),
            }
            state["ref_encoder_proj_mlp"] = {
                k: v.to(save_dtype) for k, v in unwrapped.proj_mlp.state_dict().items()
            }
            if not bool(getattr(unwrapped, "skip_perceiver", False)):
                state["ref_encoder_perceiver_layers"] = {
                    k: v.to(save_dtype) for k, v in unwrapped.perceiver_layers.state_dict().items()
                }
                state["ref_encoder_latent_queries"] = unwrapped.latent_queries.data.cpu().to(save_dtype)
                state["ref_encoder_perceiver_norm"] = {
                    k: v.to(save_dtype) for k, v in unwrapped.perceiver_norm.state_dict().items()
                }
        else:
            state[name] = {
                k: v.to(save_dtype) for k, v in unwrapped.state_dict().items()
            }
    torch.save(state, os.path.join(output_dir, "phase5_conditioning.pt"))


def _save_ip_adapter_modules(
    output_dir: str,
    ip_wrapper: nn.Module,
    unwrap_model: Callable,
    save_dtype: torch.dtype,
    *,
    num_tokens: int,
    ip_init_gain: float,
) -> None:
    unwrapped = unwrap_model(ip_wrapper)
    state = {
        name: {k: v.to(save_dtype) for k, v in mod.state_dict().items()}
        for name, mod in unwrapped.named_modules()
        if name  # skip root module
    }
    state["scale"] = 1.0
    state["num_tokens"] = int(num_tokens)
    state["ip_init_gain"] = float(ip_init_gain)
    single_block_indices = sorted(
        {
            int(name.split("_")[2])
            for name in state
            if name.startswith("single_block_") and name.endswith(("_to_k_ip", "_to_v_ip"))
        }
    )
    state["single_block_indices"] = single_block_indices
    state["num_single_layers"] = len(single_block_indices)
    torch.save(state, os.path.join(output_dir, "phase5_ip_adapter.pt"))


def _save_cross_v1_artifacts(
    output_dir: str,
    args: argparse.Namespace,
    *,
    flux_controlnet: nn.Module,
    modules: dict[str, nn.Module],
    ip_trainable_wrapper: nn.Module,
    unwrap_model: Callable,
    control_spec: CrossV1ControlSpec,
) -> None:
    save_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
        args.save_weight_dtype, torch.float32,
    )
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
    )
    _save_ip_adapter_modules(
        output_dir,
        ip_trainable_wrapper,
        unwrap_model,
        save_dtype,
        num_tokens=args.reference_num_tokens,
        ip_init_gain=args.ip_init_gain,
    )


def _load_cross_v1_controlnet_checkpoint(
    checkpoint_path: str | Path,
    control_spec: CrossV1ControlSpec,
) -> FluxControlNetModel:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        return FluxControlNetModel.from_pretrained(str(checkpoint_path))

    controlnet_config = FluxControlNetModel.load_config(checkpoint)
    controlnet = FluxControlNetModel.from_config(controlnet_config)
    patch_controlnet_x_embedder(controlnet, control_spec.packed_channels)
    state_dict = _load_diffusers_model_state_dict(checkpoint)
    source_spec = _load_saved_cross_v1_control_spec(checkpoint)
    state_dict = _remap_cross_v1_x_embedder_state_dict(
        state_dict,
        source_spec=source_spec,
        target_spec=control_spec,
    )
    controlnet.load_state_dict(state_dict, strict=True)
    return controlnet


def _load_saved_cross_v1_control_spec(checkpoint: Path) -> CrossV1ControlSpec:
    state_path = checkpoint / "phase5_conditioning.pt"
    if not state_path.exists():
        return CrossV1ControlSpec()
    state = _torch_load_weights(state_path)
    saved_spec = state.get("cross_v1_control_spec") or {}
    return CrossV1ControlSpec(
        tissue_channels=int(saved_spec.get("tissue_channels", 64)),
        nuclei_channels=int(saved_spec.get("nuclei_channels", 16)),
        spatial_mode=str(
            saved_spec.get(
                "spatial_mode",
                state.get("cross_v1_spatial_mode", CROSS_V1_SPATIAL_REFERENCE_TARGET),
            )
        ),
    )


def _remap_cross_v1_x_embedder_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    source_spec: CrossV1ControlSpec,
    target_spec: CrossV1ControlSpec,
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
    if (
        source_spec.spatial_mode == CROSS_V1_SPATIAL_REFERENCE_TARGET
        and target_spec.spatial_mode == CROSS_V1_SPATIAL_TARGET_ONLY
    ):
        old_start = source_spec.packed_target_start
        copy_width = min(
            source_spec.packed_target_channels,
            target_spec.packed_target_channels,
            max(0, old_weight.shape[1] - old_start),
            new_in_features,
        )
        if copy_width > 0:
            new_weight[:, :copy_width] = old_weight[:, old_start : old_start + copy_width]
    elif (
        source_spec.spatial_mode == CROSS_V1_SPATIAL_TARGET_ONLY
        and target_spec.spatial_mode == CROSS_V1_SPATIAL_REFERENCE_TARGET
    ):
        new_start = target_spec.packed_target_start
        copy_width = min(
            source_spec.packed_target_channels,
            target_spec.packed_target_channels,
            old_weight.shape[1],
            max(0, new_in_features - new_start),
        )
        if copy_width > 0:
            new_weight[:, new_start : new_start + copy_width] = old_weight[:, :copy_width]
    else:
        copy_width = min(old_weight.shape[1], new_in_features)
        new_weight[:, :copy_width] = old_weight[:, :copy_width]
    remapped[weight_key] = new_weight
    return remapped


def _load_diffusers_model_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    safetensors_index = checkpoint_path / "diffusion_pytorch_model.safetensors.index.json"
    bin_index = checkpoint_path / "diffusion_pytorch_model.bin.index.json"

    if safetensors_index.exists():
        return _load_sharded_diffusers_state_dict(safetensors_index)
    if bin_index.exists():
        return _load_sharded_diffusers_state_dict(bin_index)

    for filename in (
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.bin",
        "pytorch_model.bin",
        "model.safetensors",
    ):
        weight_path = checkpoint_path / filename
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
    path = getattr(args, "a1_lite_conditioning_checkpoint", None)
    if path:
        return Path(path)
    controlnet_path = getattr(args, "controlnet_model_name_or_path", None)
    return Path(controlnet_path) if controlnet_path else None


def _load_condition_modules_from_checkpoint(
    modules: dict[str, nn.Module],
    checkpoint_path: str | Path,
    *,
    load_ref_encoder: bool = False,
) -> None:
    checkpoint = Path(checkpoint_path)
    state_path = checkpoint / "phase5_conditioning.pt" if checkpoint.is_dir() else checkpoint
    if not state_path.exists():
        raise FileNotFoundError(f"Missing phase5_conditioning.pt for A1-lite: {state_path}")

    state = _torch_load_weights(state_path)
    for name in ("hte", "tissue_downsampler", "nuclei_encoder"):
        if name not in state:
            raise KeyError(f"Missing {name!r} in conditioning checkpoint: {state_path}")
        modules[name].load_state_dict(state[name])

    if load_ref_encoder:
        ref_encoder = modules["ref_encoder"]
        ref_encoder.proj_mlp.load_state_dict(state["ref_encoder_proj_mlp"])
        if not bool(getattr(ref_encoder, "skip_perceiver", False)):
            ref_encoder.load_perceiver_layers_state_dict(state["ref_encoder_perceiver_layers"])
            ref_encoder.latent_queries.data.copy_(
                state["ref_encoder_latent_queries"].to(ref_encoder.latent_queries.device)
            )
            ref_encoder.perceiver_norm.load_state_dict(state["ref_encoder_perceiver_norm"])


def _resolve_ip_adapter_checkpoint_path(args: argparse.Namespace) -> Path | None:
    path = getattr(args, "ip_adapter_checkpoint", None)
    if path:
        return Path(path)
    if bool(getattr(args, "no_load_ip_adapter_from_controlnet", False)):
        return None
    controlnet_path = getattr(args, "controlnet_model_name_or_path", None)
    if not controlnet_path:
        return None
    checkpoint = Path(controlnet_path)
    state_path = checkpoint / "phase5_ip_adapter.pt" if checkpoint.is_dir() else checkpoint
    return checkpoint if state_path.exists() else None


def _load_ip_adapter_modules_from_checkpoint(
    transformer: FluxTransformer2DModel,
    checkpoint_path: str | Path,
    *,
    load_single_ip: bool = False,
) -> None:
    checkpoint = Path(checkpoint_path)
    state_path = checkpoint / "phase5_ip_adapter.pt" if checkpoint.is_dir() else checkpoint
    if not state_path.exists():
        raise FileNotFoundError(f"Missing phase5_ip_adapter.pt: {state_path}")

    state = _torch_load_weights(state_path)
    transformer.encoder_hid_proj.load_state_dict(state["encoder_hid_proj"])
    loaded_double = 0
    for i, block in enumerate(transformer.transformer_blocks):
        k_key = f"block_{i}_to_k_ip"
        v_key = f"block_{i}_to_v_ip"
        if k_key not in state or v_key not in state:
            continue
        block.attn.processor.to_k_ip.load_state_dict(state[k_key])
        block.attn.processor.to_v_ip.load_state_dict(state[v_key])
        loaded_double += 1

    loaded_single = 0
    if load_single_ip:
        for i, block in enumerate(getattr(transformer, "single_transformer_blocks", [])):
            k_key = f"single_block_{i}_to_k_ip"
            v_key = f"single_block_{i}_to_v_ip"
            if k_key not in state or v_key not in state:
                continue
            block.attn.processor.to_k_ip.load_state_dict(state[k_key])
            block.attn.processor.to_v_ip.load_state_dict(state[v_key])
            loaded_single += 1

    logger.info(
        "Loaded IP-Adapter checkpoint %s: double_blocks=%s single_blocks=%s",
        state_path,
        loaded_double,
        loaded_single,
    )


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def run_cross_v1_training(args: argparse.Namespace) -> None:
    if args.cross_version.lower() != "v1":
        raise NotImplementedError("This module implements only cross V1.")
    if args.uni_checkpoint_path is None:
        raise ValueError("--uni-checkpoint-path is required for cross V1")
    a1_lite = bool(getattr(args, "a1_lite", False))
    if a1_lite and not args.controlnet_model_name_or_path:
        raise ValueError("--a1-lite requires --controlnet_model_name_or_path with an existing Cross V1 checkpoint.")

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

    control_spec = CrossV1ControlSpec(
        tissue_channels=args.tissue_out_channels,
        nuclei_channels=args.nuclei_out_channels,
        spatial_mode=getattr(args, "cross_v1_spatial_mode", CROSS_V1_SPATIAL_REFERENCE_TARGET),
    )
    self_reconstruction_warmup_steps = max(
        0,
        int(getattr(args, "self_reconstruction_warmup_steps", 0) or 0),
    )
    self_reconstruction_sample_prob = min(
        1.0,
        max(0.0, float(getattr(args, "self_reconstruction_sample_prob", 0.0) or 0.0)),
    )
    self_reconstruction_l1_weight = max(
        0.0,
        float(getattr(args, "self_reconstruction_l1_weight", 0.0) or 0.0),
    )
    if self_reconstruction_sample_prob > 0.0 and self_reconstruction_l1_weight <= 0.0:
        raise ValueError(
            "--self-reconstruction-sample-prob inserts reference=target samples, "
            "but --self-reconstruction-l1-weight is 0. Set "
            "--self-reconstruction-l1-weight > 0 to train with the reconstruction "
            "loss, or set --self-reconstruction-sample-prob 0 to disable these samples."
        )
    reference_style_loss_weight = max(
        0.0,
        float(getattr(args, "reference_style_loss_weight", 0.0) or 0.0),
    )
    perceptual_loss_weight = max(
        0.0,
        float(getattr(args, "perceptual_loss_weight", 0.0) or 0.0),
    )
    perceptual_loss_interval = int(getattr(args, "perceptual_loss_interval", 1) or 0)
    reference_style_loss_interval = int(
        getattr(args, "reference_style_loss_interval", 1) or 0
    )
    reference_style_loss_config = RegionalStainStyleLossConfig(
        tissue_weight=float(getattr(args, "reference_style_tissue_weight", 1.0) or 0.0),
        nuclei_weight=float(getattr(args, "reference_style_nuclei_weight", 1.0) or 0.0),
        mean_weight=float(getattr(args, "reference_style_mean_weight", 1.0) or 0.0),
        std_weight=float(getattr(args, "reference_style_std_weight", 1.0) or 0.0),
        covariance_weight=float(getattr(args, "reference_style_cov_weight", 0.25) or 0.0),
        min_pixels=max(1, int(getattr(args, "reference_style_min_pixels", 32) or 1)),
        max_regions_per_sample=getattr(args, "reference_style_max_regions_per_sample", None),
    )
    ref_swap_loss_weight = max(0.0, float(getattr(args, "ref_swap_loss_weight", 0.0) or 0.0))
    ref_swap_loss_interval = int(getattr(args, "ref_swap_loss_interval", 1) or 0)
    ref_swap_margin = float(getattr(args, "ref_swap_margin", 0.02) or 0.0)
    ref_swap_variants = _parse_ref_swap_variants(getattr(args, "ref_swap_variants", "zero,random"))
    random_reference_sampler = (
        RandomReferenceSampler(dataset.records, seed=args.seed)
        if "random" in ref_swap_variants and args.train_batch_size <= 1
        else None
    )

    ref_encoder = ReferenceImageEncoder(
        uni_checkpoint_path=args.uni_checkpoint_path,
        num_tokens=args.reference_num_tokens,
        num_perceiver_layers=args.reference_num_perceiver_layers,
        perceiver_heads=args.reference_perceiver_heads,
        use_perceiver_self_attn=not bool(
            getattr(args, "disable_reference_perceiver_self_attn", False)
        ),
        perceiver_cross_gate_init=getattr(args, "reference_perceiver_cross_gate_init", None),
        skip_perceiver=bool(getattr(args, "skip_reference_perceiver", False)),
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
        "ref_encoder": ref_encoder,
    }
    should_load_conditioning = bool(
        a1_lite or getattr(args, "load_conditioning_from_checkpoint", False)
    )
    if should_load_conditioning:
        conditioning_checkpoint = _resolve_conditioning_checkpoint_path(args)
        if conditioning_checkpoint is None:
            raise ValueError("Loading conditioning modules requires a conditioning checkpoint.")
        _load_condition_modules_from_checkpoint(
            modules,
            conditioning_checkpoint,
            load_ref_encoder=bool(
                getattr(args, "a1_lite_load_ref_encoder", False)
                or getattr(args, "load_ref_encoder_from_checkpoint", False)
            ),
        )

    # ---- accelerator setup ----
    logging_out_dir = Path(args.output_dir, args.logging_dir)
    print(">>> BEFORE Accelerator init", flush=True)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=str(logging_out_dir),
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
    print(f">>> AFTER Accelerator init, rank={accelerator.process_index}", flush=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    logger.info(
        "Using Cross V1 spatial mode %s: raw_channels=%s packed_channels=%s",
        control_spec.spatial_mode,
        control_spec.raw_channels,
        control_spec.packed_channels,
    )
    if getattr(args, "stain_augmentation", "none") != "none":
        logger.info(
            "Using stain self-supervision: augmentation=%s hed_sigma=%s hed_beta=%s "
            "strong_alpha=%s alpha_ranges=[%s,%s] U [%s,%s]",
            getattr(args, "stain_augmentation", "none"),
            getattr(args, "hed_sigma", None),
            getattr(args, "hed_beta", None),
            bool(getattr(args, "hed_strong_alpha_sampling", False)),
            getattr(args, "hed_alpha_min", None),
            getattr(args, "hed_alpha_low", None),
            getattr(args, "hed_alpha_high", None),
            getattr(args, "hed_alpha_max", None),
        )
    if self_reconstruction_warmup_steps:
        logger.info(
            "Using same-patch self-reconstruction warmup for the first %s optimizer steps",
            self_reconstruction_warmup_steps,
        )
    if self_reconstruction_sample_prob > 0.0 and self_reconstruction_l1_weight > 0.0:
        logger.info(
            "Using persistent self-reconstruction samples: prob=%s l1_weight=%s",
            self_reconstruction_sample_prob,
            self_reconstruction_l1_weight,
        )
    if reference_style_loss_weight > 0.0:
        logger.info(
            "Using reference region stain/style loss: weight=%s interval=%s tissue=%s nuclei=%s mean/std/cov=%s/%s/%s",
            reference_style_loss_weight,
            reference_style_loss_interval,
            reference_style_loss_config.tissue_weight,
            reference_style_loss_config.nuclei_weight,
            reference_style_loss_config.mean_weight,
            reference_style_loss_config.std_weight,
            reference_style_loss_config.covariance_weight,
        )
    if perceptual_loss_weight > 0.0:
        logger.info(
            "Using frozen UNI perceptual loss against target image: weight=%s interval=%s",
            perceptual_loss_weight,
            perceptual_loss_interval,
        )
    if ref_swap_loss_weight > 0.0:
        logger.info(
            "Using ref-swap sensitivity loss: weight=%s interval=%s margin=%s variants=%s",
            ref_swap_loss_weight,
            ref_swap_loss_interval,
            ref_swap_margin,
            ",".join(ref_swap_variants),
        )
    if not ref_encoder.use_perceiver_self_attn:
        logger.info("Reference Perceiver self-attention is disabled.")
    if ref_encoder.skip_perceiver:
        logger.info("Reference Perceiver is skipped; projected UNI patch tokens feed IP-Adapter directly.")
    if ref_encoder.perceiver_cross_gate_init is not None:
        logger.info(
            "Reference Perceiver cross-attention gate enabled with init=%s "
            "(sigmoid=%s)",
            ref_encoder.perceiver_cross_gate_init,
            torch.sigmoid(torch.tensor(ref_encoder.perceiver_cross_gate_init)).item(),
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

    # ---- load models ----
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision,
    )
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
        revision=args.revision, variant=args.variant,
    ).to(accelerator.device)
    text_encoder_two = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2",
        revision=args.revision, variant=args.variant,
    ).to(accelerator.device)

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler",
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    flux_transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer",
        revision=args.revision, variant=args.variant, torch_dtype=torch.bfloat16,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
        revision=args.revision, variant=args.variant,
    )

    if args.controlnet_model_name_or_path:
        flux_controlnet = _load_cross_v1_controlnet_checkpoint(
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
    logger.info("Patched controlnet_x_embedder to packed width %s for cross-v1", control_spec.packed_channels)

    # V1: install IP-Adapter attention on transformer
    install_flux_ip_adapter_attention(
        flux_transformer,
        num_tokens=ref_encoder.num_output_tokens,
        ip_init_gain=args.ip_init_gain,
        num_single_layers=max(0, int(getattr(args, "ip_single_num_layers", 0) or 0)),
    )
    patch_flux_single_ip_forward(flux_transformer)
    ip_adapter_checkpoint = _resolve_ip_adapter_checkpoint_path(args)
    if ip_adapter_checkpoint is not None:
        _load_ip_adapter_modules_from_checkpoint(
            flux_transformer,
            ip_adapter_checkpoint,
            load_single_ip=bool(getattr(args, "load_single_ip_from_checkpoint", False)),
        )
    _log_ip_adapter_initialization_stats(
        flux_transformer,
        ip_init_gain=args.ip_init_gain,
    )
    ip_adapter_modules = _collect_ip_adapter_modules(flux_transformer)
    logger.info("Installed IP-Adapter attention (%s modules)", len(ip_adapter_modules))

    # ---- temporary pipeline for prompt encoding ----
    tmp_pipeline = FluxControlNetPipeline(
        scheduler=noise_scheduler, vae=None,
        text_encoder=text_encoder_one, tokenizer=tokenizer_one,
        text_encoder_2=text_encoder_two, tokenizer_2=tokenizer_two,
        transformer=flux_transformer, controlnet=flux_controlnet,
    )

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
    torch.cuda.empty_cache()

    # ---- freeze transformer, re-enable IP-Adapter modules ----
    flux_transformer.to(accelerator.device, dtype=weight_dtype)
    if hasattr(flux_transformer, 'encoder_hid_proj'):
        flux_transformer.encoder_hid_proj.to(dtype=weight_dtype)
    flux_transformer.requires_grad_(False)
    for module in ip_adapter_modules.values():
        module.requires_grad_(True)

    vae.to(accelerator.device, dtype=weight_dtype)
    vae.eval()
    vae.requires_grad_(False)
    flux_controlnet.to(accelerator.device, dtype=weight_dtype)
    if a1_lite:
        flux_controlnet.eval()
        flux_controlnet.requires_grad_(False)
        controlnet_trainable_names: list[str] = []
        for name, module in modules.items():
            module.to(accelerator.device, dtype=weight_dtype)
            if name == "ref_encoder":
                module.train()
            else:
                module.eval()
                module.requires_grad_(False)
    else:
        flux_controlnet.train()
        controlnet_trainable_names = _configure_controlnet_trainable_params(
            flux_controlnet,
            mode=getattr(args, "controlnet_train_mode", "all"),
            train_x_embedder=bool(getattr(args, "controlnet_train_x_embedder", False)),
            train_last_n_blocks=max(0, int(getattr(args, "controlnet_train_last_n_blocks", 0) or 0)),
            train_last_n_single_blocks=max(
                0,
                int(getattr(args, "controlnet_train_last_n_single_blocks", 0) or 0),
            ),
        )
        logger.info(
            "ControlNet train mode=%s trainable_tensors=%s trainable_params=%s sample_names=%s",
            getattr(args, "controlnet_train_mode", "all"),
            len(controlnet_trainable_names),
            sum(p.numel() for p in flux_controlnet.parameters() if p.requires_grad),
            ", ".join(controlnet_trainable_names[:12]),
        )
        for module in modules.values():
            module.train()
    # UNI2-h backbone inside ref_encoder stays frozen
    modules["ref_encoder"].uni.requires_grad_(False)
    modules["ref_encoder"].uni.eval()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if args.enable_xformers_memory_efficient_attention and is_xformers_available():
        flux_transformer.enable_xformers_memory_efficient_attention()
        flux_controlnet.enable_xformers_memory_efficient_attention()
    should_checkpoint_controlnet = (
        args.gradient_checkpointing
        and not a1_lite
        and (
            getattr(args, "controlnet_train_mode", "all") == "all"
            or int(getattr(args, "controlnet_train_last_n_blocks", 0) or 0) > 0
            or int(getattr(args, "controlnet_train_last_n_single_blocks", 0) or 0) > 0
        )
    )
    if should_checkpoint_controlnet:
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
    ip_ref_learning_rate = float(
        getattr(args, "ip_ref_learning_rate", None) or (args.learning_rate * 10.0)
    )
    raw_ip_single_learning_rate = getattr(args, "ip_single_learning_rate", None)
    ip_single_learning_rate = (
        ip_ref_learning_rate
        if raw_ip_single_learning_rate is None
        else float(raw_ip_single_learning_rate)
    )

    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # =========================================================================
    # ★ FIX 1 & 2: 用 wrapper 包住不能直接 prepare 的可训练参数
    # =========================================================================

    # --- FIX 2: ref_encoder 可训练部分包成 wrapper ---
    ref_encoder_raw = modules.pop("ref_encoder")
    ref_encoder_raw.to(accelerator.device)
    # 冻结的 UNI backbone 手动放到 device，不过 DDP
    ref_encoder_raw.uni.to(accelerator.device)

    ref_trainable_wrapper = RefEncoderTrainableWrapper(ref_encoder_raw)

    # --- FIX 1: IP-Adapter 可训练部分包成 wrapper ---
    ip_trainable_wrapper = IPAdapterTrainableWrapper(ip_adapter_modules)

    # ---- optimizer: 现在所有可训练参数都在可 prepare 的 module 里 ----
    trainable_modules_list = [
        flux_controlnet,
        *modules.values(),            # hte, tissue_downsampler, nuclei_encoder
        ref_trainable_wrapper,         # ref_encoder 可训练部分
        ip_trainable_wrapper,          # IP-Adapter 可训练部分
    ]
    if a1_lite:
        trainable_modules_list = [
            ref_trainable_wrapper,
            ip_trainable_wrapper,
        ]
    controlnet_lr_modules = [] if a1_lite else [flux_controlnet]
    conditioning_lr_modules = [] if a1_lite else list(modules.values())
    double_ip_modules, single_ip_modules = _split_ip_adapter_module_groups(ip_adapter_modules)
    if ip_single_learning_rate <= 0.0:
        for module in single_ip_modules:
            module.requires_grad_(False)
    controlnet_lr_params = [
        p for module in controlnet_lr_modules for p in module.parameters() if p.requires_grad
    ]
    conditioning_lr_params = [
        p for module in conditioning_lr_modules for p in module.parameters() if p.requires_grad
    ]
    ip_ref_lr_params = [
        p
        for module in [ref_trainable_wrapper, *double_ip_modules]
        for p in module.parameters()
        if p.requires_grad
    ]
    ip_single_lr_params = [
        p for module in single_ip_modules for p in module.parameters() if p.requires_grad
    ]
    optimizer_param_groups = []

    def add_optimizer_group(name: str, params: list[torch.nn.Parameter], lr: float) -> None:
        if not params:
            logger.info("Optimizer group %s: lr=%s trainable_tensors=0 trainable_params=0", name, lr)
            return
        optimizer_param_groups.append({"params": params, "lr": lr})
        logger.info(
            "Optimizer group %s: lr=%s trainable_tensors=%s trainable_params=%s",
            name,
            lr,
            len(params),
            sum(p.numel() for p in params),
        )

    add_optimizer_group("controlnet", controlnet_lr_params, args.learning_rate)
    add_optimizer_group("conditioning", conditioning_lr_params, conditioning_learning_rate)
    add_optimizer_group("ip_ref", ip_ref_lr_params, ip_ref_learning_rate)
    add_optimizer_group("ip_single", ip_single_lr_params, ip_single_learning_rate)
    if not optimizer_param_groups:
        raise ValueError("No trainable parameters were added to the optimizer.")
    logger.info(
        (
            "Optimizer LR groups: controlnet_lr=%s params=%s, "
            "conditioning_lr=%s params=%s, ip_ref_lr=%s params=%s, "
            "ip_single_lr=%s params=%s"
        ),
        args.learning_rate,
        sum(p.numel() for p in controlnet_lr_params),
        conditioning_learning_rate,
        sum(p.numel() for p in conditioning_lr_params),
        ip_ref_learning_rate,
        sum(p.numel() for p in ip_ref_lr_params),
        ip_single_learning_rate,
        sum(p.numel() for p in ip_single_lr_params),
    )
    optimizer = optimizer_class(
        optimizer_param_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    dataloader_kwargs = {
        "shuffle": True,
        "collate_fn": collate_cross_batch,
        "batch_size": args.train_batch_size,
        "num_workers": args.dataloader_num_workers,
        "pin_memory": True,
    }
    if args.dataloader_num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = max(
            1,
            int(getattr(args, "dataloader_prefetch_factor", 2) or 2),
        )
    train_dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=(
            (args.max_train_steps or args.num_train_epochs * num_update_steps_per_epoch)
            * accelerator.num_processes
        ),
        num_cycles=args.lr_num_cycles, power=args.lr_power,
    )

    # ---- accelerator.prepare ----
    # 所有可训练 module 都过 prepare，冻结的 UNI backbone 不在里面
    n_cond_modules = len(modules)  # hte, tissue_downsampler, nuclei_encoder
    all_to_prepare = [
        flux_controlnet,
        *modules.values(),
        ref_trainable_wrapper,
        ip_trainable_wrapper,
    ]
    prepared = accelerator.prepare(
        *all_to_prepare, optimizer, train_dataloader, lr_scheduler,
    )
    n_models = len(all_to_prepare)

    flux_controlnet = prepared[0]
    prepared_cond = prepared[1 : 1 + n_cond_modules]
    modules = dict(zip(modules.keys(), prepared_cond))
    ref_trainable_wrapper = prepared[1 + n_cond_modules]
    ip_trainable_wrapper = prepared[1 + n_cond_modules + 1]

    optimizer = prepared[n_models]
    train_dataloader = prepared[n_models + 1]
    lr_scheduler = prepared[n_models + 2]

    # ★ 关键：把 prepare 后的参数同步回原始对象
    # ref_encoder: wrapper 的参数指回 ref_encoder_raw
    unwrap_model(ref_trainable_wrapper).sync_back(ref_encoder_raw)
    modules["ref_encoder"] = ref_encoder_raw

    # ip_adapter: wrapper 的参数指回 transformer 的 processor
    _sync_ip_adapter_to_transformer(unwrap_model(ip_trainable_wrapper), flux_transformer)

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

    total_batch_size = (
        args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )
    logger.info("***** Running Phase 5.3 cross-v1 training *****")
    logger.info("  Num examples = %s", len(dataset))
    logger.info("  Num Epochs = %s", args.num_train_epochs)
    logger.info("  Total batch size = %s", total_batch_size)
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

    # ---- training loop ----
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            accumulate_model = ip_trainable_wrapper if a1_lite else flux_controlnet
            with accelerator.accumulate(accumulate_model):
                bsz = int(batch["target_image"].shape[0])
                in_self_reconstruction_warmup = global_step < self_reconstruction_warmup_steps
                self_reconstruction_sample_mask = torch.zeros(
                    bsz,
                    device=accelerator.device,
                    dtype=torch.bool,
                )
                if in_self_reconstruction_warmup:
                    self_reconstruction_sample_mask.fill_(True)
                    training_batch = _use_self_reconstruction_reference(batch)
                elif self_reconstruction_sample_prob > 0.0:
                    self_reconstruction_sample_mask = (
                        torch.rand(bsz, device=accelerator.device)
                        < self_reconstruction_sample_prob
                    )
                    training_batch = _insert_self_reconstruction_samples(
                        batch,
                        self_reconstruction_sample_mask,
                    )
                else:
                    training_batch = batch
                counterfactual_sample_mask = _batch_mode_mask(
                    training_batch,
                    "counterfactual",
                    device=accelerator.device,
                )
                cross_sample_mask = ~(counterfactual_sample_mask | self_reconstruction_sample_mask)

                with torch.no_grad() if a1_lite else contextlib.nullcontext():
                    pixel_latents, control_tensor = _build_cross_v1_control_batch(
                        batch=training_batch,
                        modules=modules,
                        vae=vae,
                        weight_dtype=weight_dtype,
                        spatial_mode=control_spec.spatial_mode,
                    )
                bsz = pixel_latents.shape[0]

                packed_pixel_latents = FluxControlNetPipeline._pack_latents(
                    pixel_latents, bsz, pixel_latents.shape[1],
                    pixel_latents.shape[2], pixel_latents.shape[3],
                )
                control_image = FluxControlNetPipeline._pack_latents(
                    control_tensor, bsz, control_tensor.shape[1],
                    control_tensor.shape[2], control_tensor.shape[3],
                )
                batch_prompt, batch_pooled = _resolve_prompt_batch(
                    prompts=batch["prompts"], prompt_cache=prompt_cache,
                    empty_prompt_embeds=empty_prompt_embeds, empty_pooled=empty_pooled,
                    proportion_empty_prompts=args.proportion_empty_prompts,
                )

                noise = torch.randn_like(packed_pixel_latents)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme, batch_size=bsz,
                    logit_mean=args.logit_mean, logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=packed_pixel_latents.device)
                sigmas = get_sigmas(timesteps, n_dim=packed_pixel_latents.ndim, dtype=packed_pixel_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * packed_pixel_latents + sigmas * noise

                guidance_vec = None
                if flux_transformer.config.guidance_embeds:
                    guidance_vec = torch.full(
                        (bsz,), args.guidance_scale,
                        device=accelerator.device, dtype=weight_dtype,
                    )

                latent_image_ids = _prepare_packed_latent_image_ids(
                    packed_height=pixel_latents.shape[2] // 2,
                    packed_width=pixel_latents.shape[3] // 2,
                    device=accelerator.device, dtype=weight_dtype,
                )
                if latent_image_ids.shape[0] != noisy_model_input.shape[1]:
                    raise ValueError(
                        "FLUX img_ids length must match packed latent sequence length: "
                        f"img_ids={tuple(latent_image_ids.shape)}, "
                        f"packed_latents={tuple(noisy_model_input.shape)}, "
                        f"unpacked_latents={tuple(pixel_latents.shape)}"
                    )

                with torch.no_grad() if a1_lite else contextlib.nullcontext():
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

                joint_attention_kwargs = _build_ip_adapter_kwargs(
                    training_batch, modules, accelerator, weight_dtype, flux_transformer,
                )
                transformer_controlnet_block_samples = (
                    [sample.to(dtype=weight_dtype) for sample in controlnet_block_samples]
                    if controlnet_block_samples is not None else None
                )
                transformer_controlnet_single_block_samples = (
                    [sample.to(dtype=weight_dtype) for sample in controlnet_single_block_samples]
                    if controlnet_single_block_samples is not None else None
                )

                noise_pred = flux_transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance_vec,
                    pooled_projections=batch_pooled,
                    encoder_hidden_states=batch_prompt,
                    controlnet_block_samples=transformer_controlnet_block_samples,
                    controlnet_single_block_samples=transformer_controlnet_single_block_samples,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=dict(joint_attention_kwargs),
                    return_dict=False,
                )[0]

                target_velocity = noise - packed_pixel_latents
                normal_per_sample_loss = per_sample_mse(noise_pred, target_velocity)
                denoising_loss = normal_per_sample_loss.mean()
                cross_denoising_loss = _masked_mean_or_zero(normal_per_sample_loss, cross_sample_mask)
                counterfactual_denoising_loss = _masked_mean_or_zero(
                    normal_per_sample_loss,
                    counterfactual_sample_mask,
                )
                self_reconstruction_denoising_loss = _masked_mean_or_zero(
                    normal_per_sample_loss,
                    self_reconstruction_sample_mask,
                )
                style_loss = noise_pred.new_zeros(())
                style_tissue_loss = noise_pred.new_zeros(())
                style_nuclei_loss = noise_pred.new_zeros(())
                style_tissue_regions = 0
                style_nuclei_regions = 0
                perceptual_loss = noise_pred.new_zeros(())
                self_reconstruction_l1 = noise_pred.new_zeros(())
                prediction_rgb = None
                should_compute_style_loss = (
                    reference_style_loss_weight > 0.0
                    and reference_style_loss_interval > 0
                    and global_step % reference_style_loss_interval == 0
                )
                should_compute_perceptual_loss = (
                    perceptual_loss_weight > 0.0
                    and perceptual_loss_interval > 0
                    and global_step % perceptual_loss_interval == 0
                )
                should_compute_self_reconstruction_l1 = bool(
                    self_reconstruction_sample_mask.any().item()
                )
                if (
                    should_compute_style_loss
                    or should_compute_perceptual_loss
                    or should_compute_self_reconstruction_l1
                ):
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
                if should_compute_perceptual_loss:
                    ref_encoder = modules["ref_encoder"]
                    uni_dtype = next(ref_encoder.uni.parameters()).dtype
                    prediction_features = ref_encoder.extract_uni_features(
                        prediction_rgb.to(device=accelerator.device, dtype=uni_dtype),
                        allow_input_grad=True,
                    )
                    target_features = ref_encoder.extract_uni_features(
                        training_batch["target_image"].to(device=accelerator.device, dtype=uni_dtype),
                    )
                    perceptual_loss = uni_token_cosine_perceptual_loss(
                        prediction_features=prediction_features,
                        target_features=target_features,
                    ).to(dtype=denoising_loss.dtype)
                if should_compute_style_loss:
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
                    style_loss = style_terms["total"].to(dtype=denoising_loss.dtype)
                    style_tissue_loss = style_terms["tissue"].to(dtype=denoising_loss.dtype)
                    style_nuclei_loss = style_terms["nuclei"].to(dtype=denoising_loss.dtype)
                    style_tissue_regions = int(style_terms["tissue_regions"])
                    style_nuclei_regions = int(style_terms["nuclei_regions"])
                if should_compute_self_reconstruction_l1:
                    self_reconstruction_l1 = self_reconstruction_l1_loss(
                        prediction=prediction_rgb,
                        reference=training_batch["reference_image"].to(
                            device=accelerator.device,
                            dtype=prediction_rgb.dtype,
                        ),
                        sample_mask=self_reconstruction_sample_mask,
                    ).to(dtype=denoising_loss.dtype)
                self_reconstruction_l1_weighted = (
                    self_reconstruction_l1_weight * self_reconstruction_l1
                )

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
                        if variant == "zero":
                            swapped_batch = _use_zero_reference(training_batch)
                        elif variant == "random":
                            random_batch = (
                                random_reference_sampler.sample_for_batch(
                                    training_batch,
                                    device=accelerator.device,
                                )
                                if random_reference_sampler is not None
                                and int(training_batch["reference_image"].shape[0]) <= 1
                                else None
                            )
                            swapped_batch = _use_random_reference(
                                training_batch,
                                random_batch=random_batch,
                            )
                        else:
                            continue
                        swapped_kwargs = _build_ip_adapter_kwargs(
                            swapped_batch, modules, accelerator, weight_dtype, flux_transformer,
                        )
                        swapped_noise_pred = flux_transformer(
                            hidden_states=noisy_model_input,
                            timestep=timesteps / 1000,
                            guidance=guidance_vec,
                            pooled_projections=batch_pooled,
                            encoder_hidden_states=batch_prompt,
                            controlnet_block_samples=transformer_controlnet_block_samples,
                            controlnet_single_block_samples=transformer_controlnet_single_block_samples,
                            txt_ids=text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=dict(swapped_kwargs),
                            return_dict=False,
                        )[0]
                        swapped_per_sample_losses.append(
                            per_sample_mse(swapped_noise_pred, target_velocity)
                        )
                        ref_variant_loss_logs[f"ref_{variant}_denoise_loss"] = (
                            swapped_per_sample_losses[-1].mean().detach().item()
                        )
                    swap_loss = ref_swap_sensitivity_loss(
                        normal_per_sample_loss,
                        swapped_per_sample_losses,
                        margin=ref_swap_margin,
                    ).to(dtype=denoising_loss.dtype)

                loss = (
                    denoising_loss
                    + perceptual_loss_weight * perceptual_loss
                    + reference_style_loss_weight * style_loss
                    + ref_swap_loss_weight * swap_loss
                    + self_reconstruction_l1_weighted
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # ★ 梯度裁剪也要包含 wrapper 里的参数
                    if a1_lite:
                        all_trainable = [ref_trainable_wrapper, ip_trainable_wrapper]
                    else:
                        all_trainable = [
                            flux_controlnet,
                            *modules.values(),
                            ref_trainable_wrapper,
                            ip_trainable_wrapper,
                        ]
                    accelerator.clip_grad_norm_(
                        [p for m in all_trainable for p in m.parameters() if p.requires_grad],
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
                    _save_cross_v1_artifacts(
                        save_path,
                        args,
                        flux_controlnet=flux_controlnet,
                        modules=modules,
                        ip_trainable_wrapper=ip_trainable_wrapper,
                        unwrap_model=unwrap_model,
                        control_spec=control_spec,
                    )
                    logger.info("Saved eval-ready Phase 5.3 cross-v1 artifacts to %s", save_path)

            logs = {
                "loss": loss.detach().item(),
                "denoise_loss": denoising_loss.detach().item(),
                "cross_denoise_loss": cross_denoising_loss.detach().item(),
                "counterfactual_denoise_loss": counterfactual_denoising_loss.detach().item(),
                "self_reconstruction_denoise_loss": self_reconstruction_denoising_loss.detach().item(),
                "perceptual_loss": perceptual_loss.detach().item(),
                "style_loss": style_loss.detach().item(),
                "style_tissue_loss": style_tissue_loss.detach().item(),
                "style_nuclei_loss": style_nuclei_loss.detach().item(),
                "self_reconstruction_l1": self_reconstruction_l1.detach().item(),
                "self_reconstruction_l1_weighted": self_reconstruction_l1_weighted.detach().item(),
                "self_reconstruction_samples": int(self_reconstruction_sample_mask.sum().detach().item()),
                "counterfactual_samples": int(counterfactual_sample_mask.sum().detach().item()),
                "cross_samples": int(cross_sample_mask.sum().detach().item()),
                "ref_swap_loss": swap_loss.detach().item(),
                "ref_normal_denoise_loss": denoising_loss.detach().item(),
                "style_tissue_regions": style_tissue_regions,
                "style_nuclei_regions": style_nuclei_regions,
                "lr": lr_scheduler.get_last_lr()[0],
            }
            logs.update(ref_variant_loss_logs)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    # ---- save final artifacts ----
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
            args.save_weight_dtype, torch.float32,
        )
        unwrapped_controlnet = unwrap_model(flux_controlnet)
        # ★ FIX 3: 不原地改 dtype
        if args.save_weight_dtype != "fp32":
            unwrapped_controlnet.save_pretrained(args.output_dir, variant=args.save_weight_dtype)
        else:
            unwrapped_controlnet.save_pretrained(args.output_dir)
        _save_condition_modules(
            args.output_dir,
            modules,
            unwrap_model,
            save_dtype,
            control_spec=control_spec,
        )
        _save_ip_adapter_modules(
            args.output_dir,
            ip_trainable_wrapper,
            unwrap_model,
            save_dtype,
            num_tokens=modules["ref_encoder"].num_output_tokens,
            ip_init_gain=args.ip_init_gain,
        )
        logger.info("Saved Phase 5.3 cross-v1 artifacts to %s", args.output_dir)

    accelerator.end_training()

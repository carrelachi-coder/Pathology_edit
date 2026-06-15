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
import hashlib
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
from packaging import version
from torch.optim.lr_scheduler import LambdaLR
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
from controlnet_train.modules.reference_image_encoder import (
    ReferenceImageEncoder,
    build_region_ip_token_labels,
    normalize_region_ip_label_mode,
    normalize_region_ip_token_mode,
    resize_mask_to_token_labels,
)
from controlnet_train.training.conditioning import patch_controlnet_x_embedder
from controlnet_train.training.cross_v1_losses import (
    RegionalFeatureLossConfig,
    RegionalRgbFftLossConfig,
    RegionalStainStyleLossConfig,
    per_sample_mse,
    ref_swap_sensitivity_loss,
    regional_feature_map_loss,
    regional_rgb_fft_loss,
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


CROSS_V1_IP_ARCH_GLOBAL = "global"
CROSS_V1_IP_ARCH_REGIONAL_HARD = "regional_hard"
CROSS_V1_IP_ARCH_GLOBAL_SOFT_BIAS = "global_soft_bias"
CROSS_V1_IP_ARCH_MODES = (
    CROSS_V1_IP_ARCH_GLOBAL,
    CROSS_V1_IP_ARCH_REGIONAL_HARD,
    CROSS_V1_IP_ARCH_GLOBAL_SOFT_BIAS,
)
REFERENCE_REGION_LOSS_BACKEND_RGB_FFT = "rgb_fft"
REFERENCE_REGION_LOSS_BACKEND_UNI = "uni"
REFERENCE_REGION_LOSS_BACKENDS = (
    REFERENCE_REGION_LOSS_BACKEND_RGB_FFT,
    REFERENCE_REGION_LOSS_BACKEND_UNI,
)


def normalize_cross_v1_ip_architecture(
    value: str | None,
    *,
    regional_ip_adapter: bool | None = None,
) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "": CROSS_V1_IP_ARCH_REGIONAL_HARD if regional_ip_adapter else CROSS_V1_IP_ARCH_GLOBAL,
        "global": CROSS_V1_IP_ARCH_GLOBAL,
        "dense": CROSS_V1_IP_ARCH_GLOBAL,
        "plain": CROSS_V1_IP_ARCH_GLOBAL,
        "regional": CROSS_V1_IP_ARCH_REGIONAL_HARD,
        "regional_hard": CROSS_V1_IP_ARCH_REGIONAL_HARD,
        "hard": CROSS_V1_IP_ARCH_REGIONAL_HARD,
        "hard_region": CROSS_V1_IP_ARCH_REGIONAL_HARD,
        "hard_regional": CROSS_V1_IP_ARCH_REGIONAL_HARD,
        "soft": CROSS_V1_IP_ARCH_GLOBAL_SOFT_BIAS,
        "soft_bias": CROSS_V1_IP_ARCH_GLOBAL_SOFT_BIAS,
        "global_soft": CROSS_V1_IP_ARCH_GLOBAL_SOFT_BIAS,
        "global_soft_bias": CROSS_V1_IP_ARCH_GLOBAL_SOFT_BIAS,
        "stats_soft_bias": CROSS_V1_IP_ARCH_GLOBAL_SOFT_BIAS,
        "v1_global_soft_bias": CROSS_V1_IP_ARCH_GLOBAL_SOFT_BIAS,
    }
    if raw not in aliases:
        raise ValueError(
            f"Unsupported Cross V1 IP architecture {value!r}; "
            f"choose from {', '.join(CROSS_V1_IP_ARCH_MODES)}."
        )
    return aliases[raw]


def _uses_soft_region_bias(architecture: str) -> bool:
    return normalize_cross_v1_ip_architecture(architecture) == CROSS_V1_IP_ARCH_GLOBAL_SOFT_BIAS


def normalize_reference_region_loss_backend(value: str | None) -> str:
    raw = str(value or REFERENCE_REGION_LOSS_BACKEND_UNI).strip().lower().replace("-", "_")
    aliases = {
        "rgb_fft": REFERENCE_REGION_LOSS_BACKEND_RGB_FFT,
        "rgbfft": REFERENCE_REGION_LOSS_BACKEND_RGB_FFT,
        "rgb": REFERENCE_REGION_LOSS_BACKEND_RGB_FFT,
        "fft": REFERENCE_REGION_LOSS_BACKEND_RGB_FFT,
        "independent": REFERENCE_REGION_LOSS_BACKEND_RGB_FFT,
        "uni": REFERENCE_REGION_LOSS_BACKEND_UNI,
        "feature": REFERENCE_REGION_LOSS_BACKEND_UNI,
        "features": REFERENCE_REGION_LOSS_BACKEND_UNI,
        "feature_map": REFERENCE_REGION_LOSS_BACKEND_UNI,
        "spatial_uni": REFERENCE_REGION_LOSS_BACKEND_UNI,
    }
    if raw not in aliases:
        raise ValueError(
            f"Unsupported reference region loss backend {value!r}; "
            f"choose from {', '.join(REFERENCE_REGION_LOSS_BACKENDS)}."
        )
    return aliases[raw]


def _reference_region_sigma_mask(
    sigmas: torch.Tensor,
    *,
    min_sigma: float,
    max_sigma: float,
) -> torch.Tensor:
    sigma_values = sigmas.detach().float().flatten()
    return (sigma_values >= float(min_sigma)) & (sigma_values <= float(max_sigma))


def _build_lr_scheduler(
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    *,
    num_training_steps: int,
    num_warmup_steps: int,
):
    scheduler_name = str(getattr(args, "lr_scheduler", "cosine") or "cosine")
    if scheduler_name != "cosine_with_min_lr":
        return get_scheduler(
            scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    min_factor = float(getattr(args, "lr_min_factor", 0.0) or 0.0)
    if not 0.0 <= min_factor <= 1.0:
        raise ValueError(f"--lr-min-factor must be in [0, 1], got {min_factor}.")
    cycles = float(getattr(args, "lr_num_cycles", 0.5) or 0.5)
    decay_start_step = max(0, int(getattr(args, "lr_decay_start_step", 0) or 0))
    warmup_steps = max(0, int(num_warmup_steps))
    total_steps = max(1, int(num_training_steps))

    def lr_lambda(current_step: int) -> float:
        if decay_start_step > 0:
            if current_step < decay_start_step:
                return 1.0
            progress = float(current_step - decay_start_step) / float(max(1, total_steps - decay_start_step))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * 2.0 * cycles * progress))
            return min_factor + (1.0 - min_factor) * cosine
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * 2.0 * cycles * progress))
        return min_factor + (1.0 - min_factor) * cosine

    return LambdaLR(optimizer, lr_lambda)


def _set_optimizer_group_lrs(optimizer: torch.optim.Optimizer, lrs: list[float]) -> None:
    if len(optimizer.param_groups) != len(lrs):
        raise ValueError(
            f"Optimizer group count changed: {len(optimizer.param_groups)} vs {len(lrs)}"
        )
    for group, lr in zip(optimizer.param_groups, lrs):
        group["lr"] = float(lr)
        group["initial_lr"] = float(lr)


def _advance_scheduler_to_step(scheduler, step: int) -> None:
    target_step = max(0, int(step))
    if hasattr(scheduler, "lr_lambdas") and hasattr(scheduler, "base_lrs"):
        scheduler.last_epoch = target_step
        for group, base_lr, lr_lambda in zip(
            scheduler.optimizer.param_groups,
            scheduler.base_lrs,
            scheduler.lr_lambdas,
        ):
            group["lr"] = float(base_lr) * float(lr_lambda(target_step))
        return
    for _ in range(target_step):
        scheduler.step()


class TrainableEMA:
    """EMA for the small trainable Cross V1 modules."""

    def __init__(
        self,
        modules: list[nn.Module],
        *,
        decay: float,
        device: str = "cpu",
    ) -> None:
        self.decay = float(decay)
        self.device = str(device)
        if not 0.0 < self.decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {self.decay}.")
        if self.device not in {"cpu", "model"}:
            raise ValueError(f"EMA device must be 'cpu' or 'model', got {self.device!r}.")
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        for name, param in self._named_params(modules):
            target_device = torch.device("cpu") if self.device == "cpu" else param.device
            self.shadow[name] = param.detach().float().to(device=target_device).clone()

    @staticmethod
    def _named_params(modules: list[nn.Module]):
        for module_index, module in enumerate(modules):
            for name, param in module.named_parameters():
                if param.requires_grad:
                    yield f"{module_index}.{name}", param

    def update(self, modules: list[nn.Module]) -> None:
        decay = self.decay
        with torch.no_grad():
            for name, param in self._named_params(modules):
                if name not in self.shadow:
                    target_device = torch.device("cpu") if self.device == "cpu" else param.device
                    self.shadow[name] = param.detach().float().to(device=target_device).clone()
                    continue
                value = param.detach().float()
                if self.device == "cpu":
                    value = value.cpu()
                else:
                    value = value.to(device=self.shadow[name].device)
                self.shadow[name].mul_(decay).add_(value, alpha=1.0 - decay)

    def copy_to(self, modules: list[nn.Module]) -> None:
        with torch.no_grad():
            for name, param in self._named_params(modules):
                if name not in self.shadow:
                    continue
                self.backup[name] = param.detach().clone()
                param.copy_(self.shadow[name].to(device=param.device, dtype=param.dtype))

    def restore(self, modules: list[nn.Module]) -> None:
        if not self.backup:
            return
        with torch.no_grad():
            for name, param in self._named_params(modules):
                backup = self.backup.get(name)
                if backup is not None:
                    param.copy_(backup.to(device=param.device, dtype=param.dtype))
        self.backup.clear()

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "device": self.device,
            "shadow": {name: value.cpu() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        shadow = state.get("shadow")
        if not isinstance(shadow, dict):
            raise ValueError("EMA state is missing shadow weights.")
        target_device = torch.device("cpu") if self.device == "cpu" else None
        loaded = {}
        for name, value in shadow.items():
            if not torch.is_tensor(value):
                continue
            loaded[str(name)] = (
                value.detach().float().to(device=target_device).clone()
                if target_device is not None
                else value.detach().float().clone()
            )
        if not loaded:
            raise ValueError("EMA state did not contain tensor shadow weights.")
        self.shadow = loaded


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
            return [
                self.proj(embed.to(dtype=target_dtype)).to(dtype=target_dtype)
                for embed in image_embeds
            ]
        return self.proj(image_embeds.to(dtype=target_dtype)).to(dtype=target_dtype)


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
        soft_bias_init: float = 4.0,
        use_soft_bias: bool = False,
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
        self.ip_null_tokens = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 1, cross_attention_dim)) for _ in self.num_tokens]
        )
        self.use_soft_bias = bool(use_soft_bias)
        self.ip_soft_bias = nn.ParameterList(
            [nn.Parameter(torch.tensor(float(soft_bias_init)))]
        )

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        ip_hidden_states: list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
        ip_adapter_masks: torch.Tensor | dict | None = None,
        ip_region_token_labels: torch.Tensor | None = None,
        ip_query_region_labels: torch.Tensor | None = None,
        ip_region_fallback_labels: torch.Tensor | None = None,
        ip_query_fallback_labels: torch.Tensor | None = None,
        ip_region_strict: bool = True,
        ip_region_soft_bias: torch.Tensor | float | None = None,
        ip_region_use_soft_bias: bool | None = None,
        txt_seq_len: int | None = None,
        ip_debug_collector: dict | None = None,
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
            (
                packed_key_labels,
                packed_query_labels,
                packed_strict,
                packed_key_fallback_labels,
                packed_query_fallback_labels,
            ) = _unpack_region_ip_adapter_masks(ip_adapter_masks)
            ip_region_token_labels = ip_region_token_labels if ip_region_token_labels is not None else packed_key_labels
            ip_query_region_labels = ip_query_region_labels if ip_query_region_labels is not None else packed_query_labels
            ip_region_fallback_labels = (
                ip_region_fallback_labels
                if ip_region_fallback_labels is not None
                else packed_key_fallback_labels
            )
            ip_query_fallback_labels = (
                ip_query_fallback_labels
                if ip_query_fallback_labels is not None
                else packed_query_fallback_labels
            )
            ip_region_strict = bool(ip_region_strict and packed_strict)
            packed_use_soft_bias = (
                ip_region_use_soft_bias
                if ip_region_use_soft_bias is not None
                else _unpack_region_ip_soft_bias_enabled(ip_adapter_masks)
            )
            soft_bias_value = (
                ip_region_soft_bias
                if ip_region_soft_bias is not None
                else _unpack_region_ip_soft_bias_value(ip_adapter_masks)
            )
            if soft_bias_value is None:
                soft_bias_value = self.ip_soft_bias[0]
            use_soft_bias = bool(self.use_soft_bias or packed_use_soft_bias)
            txt_seq_len = int(txt_seq_len or 0)
            if txt_seq_len < 0 or txt_seq_len > output.shape[1]:
                raise ValueError(
                    f"txt_seq_len must be within [0, {output.shape[1]}], got {txt_seq_len}."
                )
            image_query = query[:, :, txt_seq_len:, :]
            if image_query.shape[2] > 0:
                image_ip_output = output.new_zeros((batch_size, image_query.shape[2], output.shape[2]))
                for current_ip_hidden_states, scale, to_k_ip, to_v_ip, ip_null_token in zip(
                    ip_hidden_states,
                    self.scale,
                    self.to_k_ip,
                    self.to_v_ip,
                    self.ip_null_tokens,
                ):
                    if scale == 0:
                        continue
                    ip_input = current_ip_hidden_states.to(
                        device=image_query.device,
                        dtype=to_k_ip.weight.dtype,
                    )
                    ip_attn_mask, _, _ = _build_region_attention_mask_and_query_gate(
                        query_region_labels=ip_query_region_labels,
                        key_region_labels=ip_region_token_labels,
                        query_fallback_labels=ip_query_fallback_labels,
                        key_fallback_labels=ip_region_fallback_labels,
                        batch_size=batch_size,
                        query_len=image_query.shape[2],
                        key_len=ip_input.shape[1],
                        device=image_query.device,
                        dtype=image_query.dtype,
                        strict=bool(ip_region_strict),
                        soft_bias=soft_bias_value,
                        use_soft_bias=use_soft_bias,
                    )
                    if ip_attn_mask is not None and ip_attn_mask.shape[-1] == ip_input.shape[1] + 1:
                        null_input = ip_null_token.to(
                            device=ip_input.device,
                            dtype=ip_input.dtype,
                        ).expand(batch_size, -1, -1)
                        ip_input = torch.cat([ip_input, null_input], dim=1)
                    ip_key = to_k_ip(ip_input).to(dtype=image_query.dtype)
                    ip_value = to_v_ip(ip_input).to(dtype=image_query.dtype)
                    ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                    ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                    ip_attn = torch.nn.functional.scaled_dot_product_attention(
                        image_query,
                        ip_key,
                        ip_value,
                        attn_mask=ip_attn_mask,
                        dropout_p=0.0,
                        is_causal=False,
                    )
                    ip_attn = ip_attn.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                    image_ip_output = image_ip_output + float(scale) * ip_attn.to(output.dtype)
                _record_ip_attention_debug(
                    ip_debug_collector if ip_debug_collector is not None else getattr(self, "_ip_debug_collector", None),
                    getattr(self, "debug_name", "single_block"),
                    output[:, txt_seq_len:, :],
                    image_ip_output,
                )
                image_output = output[:, txt_seq_len:, :] + image_ip_output
                if txt_seq_len > 0:
                    output = torch.cat([output[:, :txt_seq_len, :], image_output], dim=1)
                else:
                    output = image_output

        return output.to(hidden_states.dtype)


class FluxRegionalIPAdapterJointAttnProcessor2_0(nn.Module):
    """FLUX double-stream IP processor with class-label-gated reference attention."""

    def __init__(
        self,
        *,
        hidden_size: int,
        cross_attention_dim: int,
        num_tokens: int | tuple[int, ...] = (16,),
        scale: float | list[float] = 1.0,
        soft_bias_init: float = 4.0,
        use_soft_bias: bool = False,
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
        self.ip_null_tokens = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 1, cross_attention_dim)) for _ in self.num_tokens]
        )
        self.use_soft_bias = bool(use_soft_bias)
        self.ip_soft_bias = nn.ParameterList(
            [nn.Parameter(torch.tensor(float(soft_bias_init)))]
        )

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        ip_hidden_states: list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
        ip_adapter_masks: torch.Tensor | dict | None = None,
        ip_region_token_labels: torch.Tensor | None = None,
        ip_query_region_labels: torch.Tensor | None = None,
        ip_region_fallback_labels: torch.Tensor | None = None,
        ip_query_fallback_labels: torch.Tensor | None = None,
        ip_region_strict: bool = True,
        ip_region_soft_bias: torch.Tensor | float | None = None,
        ip_region_use_soft_bias: bool | None = None,
        txt_seq_len: int | None = None,
        ip_debug_collector: dict | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if encoder_hidden_states is None:
            raise ValueError("FluxRegionalIPAdapterJointAttnProcessor2_0 expects double-stream states.")
        batch_size, _, _ = encoder_hidden_states.shape
        hidden_states_query_proj = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        hidden_states_query_proj = hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            hidden_states_query_proj = attn.norm_q(hidden_states_query_proj)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

        query = torch.cat([encoder_hidden_states_query_proj, hidden_states_query_proj], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        attn_output = attn_output.to(query.dtype)

        encoder_hidden_states, hidden_states = (
            attn_output[:, : encoder_hidden_states.shape[1]],
            attn_output[:, encoder_hidden_states.shape[1] :],
        )
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        ip_attn_output = hidden_states.new_zeros(hidden_states.shape)
        if ip_hidden_states:
            (
                packed_key_labels,
                packed_query_labels,
                packed_strict,
                packed_key_fallback_labels,
                packed_query_fallback_labels,
            ) = _unpack_region_ip_adapter_masks(ip_adapter_masks)
            ip_region_token_labels = ip_region_token_labels if ip_region_token_labels is not None else packed_key_labels
            ip_query_region_labels = ip_query_region_labels if ip_query_region_labels is not None else packed_query_labels
            ip_region_fallback_labels = (
                ip_region_fallback_labels
                if ip_region_fallback_labels is not None
                else packed_key_fallback_labels
            )
            ip_query_fallback_labels = (
                ip_query_fallback_labels
                if ip_query_fallback_labels is not None
                else packed_query_fallback_labels
            )
            ip_region_strict = bool(ip_region_strict and packed_strict)
            packed_use_soft_bias = (
                ip_region_use_soft_bias
                if ip_region_use_soft_bias is not None
                else _unpack_region_ip_soft_bias_enabled(ip_adapter_masks)
            )
            soft_bias_value = (
                ip_region_soft_bias
                if ip_region_soft_bias is not None
                else _unpack_region_ip_soft_bias_value(ip_adapter_masks)
            )
            if soft_bias_value is None:
                soft_bias_value = self.ip_soft_bias[0]
            use_soft_bias = bool(self.use_soft_bias or packed_use_soft_bias)
            for current_ip_hidden_states, scale, to_k_ip, to_v_ip, ip_null_token in zip(
                ip_hidden_states,
                self.scale,
                self.to_k_ip,
                self.to_v_ip,
                self.ip_null_tokens,
            ):
                if scale == 0:
                    continue
                ip_input = current_ip_hidden_states.to(
                    device=hidden_states_query_proj.device,
                    dtype=to_k_ip.weight.dtype,
                )
                ip_attn_mask, _, _ = _build_region_attention_mask_and_query_gate(
                    query_region_labels=ip_query_region_labels,
                    key_region_labels=ip_region_token_labels,
                    query_fallback_labels=ip_query_fallback_labels,
                    key_fallback_labels=ip_region_fallback_labels,
                    batch_size=batch_size,
                    query_len=hidden_states_query_proj.shape[2],
                    key_len=ip_input.shape[1],
                    device=hidden_states_query_proj.device,
                    dtype=hidden_states_query_proj.dtype,
                    strict=bool(ip_region_strict),
                    soft_bias=soft_bias_value,
                    use_soft_bias=use_soft_bias,
                )
                if ip_attn_mask is not None and ip_attn_mask.shape[-1] == ip_input.shape[1] + 1:
                    null_input = ip_null_token.to(
                        device=ip_input.device,
                        dtype=ip_input.dtype,
                    ).expand(batch_size, -1, -1)
                    ip_input = torch.cat([ip_input, null_input], dim=1)
                ip_key = to_k_ip(ip_input).to(dtype=hidden_states_query_proj.dtype)
                ip_value = to_v_ip(ip_input).to(dtype=hidden_states_query_proj.dtype)
                ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                current_output = F.scaled_dot_product_attention(
                    hidden_states_query_proj,
                    ip_key,
                    ip_value,
                    attn_mask=ip_attn_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )
                current_output = current_output.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                ip_attn_output = ip_attn_output + float(scale) * current_output.to(hidden_states.dtype)

        _record_ip_attention_debug(
            ip_debug_collector if ip_debug_collector is not None else getattr(self, "_ip_debug_collector", None),
            getattr(self, "debug_name", "block"),
            hidden_states,
            ip_attn_output,
        )
        return hidden_states, encoder_hidden_states, ip_attn_output


def _build_region_attention_mask(
    *,
    query_region_labels: torch.Tensor | None,
    key_region_labels: torch.Tensor | None,
    batch_size: int,
    query_len: int,
    key_len: int,
    device: torch.device,
    dtype: torch.dtype,
    strict: bool,
    query_fallback_labels: torch.Tensor | None = None,
    key_fallback_labels: torch.Tensor | None = None,
    soft_bias: torch.Tensor | float | None = None,
    use_soft_bias: bool = False,
) -> torch.Tensor | None:
    """Build an additive SDPA mask with an extra learned-null key column."""
    mask, _, _ = _build_region_attention_mask_and_query_gate(
        query_region_labels=query_region_labels,
        key_region_labels=key_region_labels,
        batch_size=batch_size,
        query_len=query_len,
        key_len=key_len,
        device=device,
        dtype=dtype,
        strict=strict,
        query_fallback_labels=query_fallback_labels,
        key_fallback_labels=key_fallback_labels,
        soft_bias=soft_bias,
        use_soft_bias=use_soft_bias,
    )
    return mask


def _build_region_attention_mask_and_query_gate(
    *,
    query_region_labels: torch.Tensor | None,
    key_region_labels: torch.Tensor | None,
    batch_size: int,
    query_len: int,
    key_len: int,
    device: torch.device,
    dtype: torch.dtype,
    strict: bool,
    query_fallback_labels: torch.Tensor | None = None,
    key_fallback_labels: torch.Tensor | None = None,
    soft_bias: torch.Tensor | float | None = None,
    use_soft_bias: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, float | int | bool]]:
    """Build an additive SDP mask for regional IP attention plus a learned null token.

    Internal ``allowed=True`` means "keep/attend". The returned additive mask uses
    SDPA semantics: 0 keeps a pair and a large negative value blocks it. Strict
    regional mode appends one learned null token column: queries with no exact
    matching key and unlabeled queries attend only to that null token.
    """
    empty_stats: dict[str, float | int | bool] = {
        "strict": bool(strict),
        "has_labels": bool(query_region_labels is not None and key_region_labels is not None),
        "query_tokens": int(query_len),
        "key_tokens": int(key_len),
        "valid_query_fraction": math.nan,
        "valid_key_fraction": math.nan,
        "allowed_pair_fraction": math.nan,
        "allowed_valid_pair_fraction": math.nan,
        "active_query_fraction": math.nan,
        "missing_query_fraction": math.nan,
        "fallback_query_fraction": math.nan,
        "null_query_fraction": math.nan,
        "allowed_tokens_per_query_mean": math.nan,
        "allowed_tokens_per_query_min": 0,
        "allowed_tokens_per_query_max": 0,
        "unique_query_labels": 0,
        "unique_key_labels": 0,
        "soft_bias_enabled": False,
        "soft_bias": math.nan,
        "same_label_pair_fraction": math.nan,
        "other_label_pair_fraction": math.nan,
    }
    if query_region_labels is None or key_region_labels is None:
        return None, None, empty_stats
    if not strict and not use_soft_bias:
        return None, None, empty_stats
    query_labels = query_region_labels.to(device=device, dtype=torch.long)
    key_labels = key_region_labels.to(device=device, dtype=torch.long)
    if query_labels.ndim != 2 or key_labels.ndim != 2:
        raise ValueError(
            "region labels must have shape (B,T), "
            f"got query={tuple(query_labels.shape)} key={tuple(key_labels.shape)}"
        )
    if query_labels.shape != (batch_size, query_len):
        raise ValueError(
            f"query region labels shape {tuple(query_labels.shape)} does not match "
            f"(B,Q)=({batch_size},{query_len})"
        )
    if key_labels.shape != (batch_size, key_len):
        raise ValueError(
            f"key region labels shape {tuple(key_labels.shape)} does not match "
            f"(B,K)=({batch_size},{key_len})"
        )
    valid_query = query_labels >= 0
    valid_key = key_labels >= 0
    allowed = (
        (query_labels[:, :, None] == key_labels[:, None, :])
        & valid_query[:, :, None]
        & valid_key[:, None, :]
    )
    if use_soft_bias:
        return _build_soft_region_attention_bias(
            query_labels=query_labels,
            key_labels=key_labels,
            allowed=allowed,
            valid_query=valid_query,
            valid_key=valid_key,
            batch_size=batch_size,
            query_len=query_len,
            key_len=key_len,
            device=device,
            dtype=dtype,
            soft_bias=soft_bias,
            strict=bool(strict),
        )
    exact_missing = valid_query & ~allowed.any(dim=-1)
    fallback_used = torch.zeros_like(valid_query)
    missing = exact_missing
    null_query = (~valid_query) | missing
    null_column = null_query[:, :, None]
    allowed_with_null = torch.cat([allowed, null_column], dim=-1)
    if not bool(allowed_with_null.any(dim=-1).all().item()):
        raise RuntimeError("regional IP attention mask construction produced an all-masked query row")
    stats = _region_attention_stats(
        allowed=allowed,
        valid_query=valid_query,
        valid_key=valid_key,
        missing=missing,
        fallback_used=fallback_used,
        null_query=null_query,
        allowed_with_null=allowed_with_null,
        query_labels=query_labels,
        key_labels=key_labels,
        strict=True,
    )
    mask = torch.zeros((batch_size, 1, query_len, key_len + 1), device=device, dtype=dtype)
    mask = mask.masked_fill(~allowed_with_null[:, None, :, :], -torch.finfo(mask.dtype).max)
    return mask, None, stats


def _build_soft_region_attention_bias(
    *,
    query_labels: torch.Tensor,
    key_labels: torch.Tensor,
    allowed: torch.Tensor,
    valid_query: torch.Tensor,
    valid_key: torch.Tensor,
    batch_size: int,
    query_len: int,
    key_len: int,
    device: torch.device,
    dtype: torch.dtype,
    soft_bias: torch.Tensor | float | None,
    strict: bool,
) -> tuple[torch.Tensor, None, dict[str, float | int | bool]]:
    if soft_bias is None:
        soft_bias_tensor = torch.tensor(1.0, device=device, dtype=dtype)
    elif torch.is_tensor(soft_bias):
        soft_bias_tensor = soft_bias.to(device=device, dtype=dtype)
    else:
        soft_bias_tensor = torch.tensor(float(soft_bias), device=device, dtype=dtype)
    soft_bias_tensor = soft_bias_tensor.reshape(())
    same = allowed
    valid_pairs = valid_query[:, :, None] & valid_key[:, None, :]
    other = valid_pairs & ~same
    bias = torch.zeros((batch_size, query_len, key_len), device=device, dtype=dtype)
    bias = bias + same.to(dtype=dtype) * soft_bias_tensor
    bias = bias - other.to(dtype=dtype) * soft_bias_tensor
    has_valid_key = valid_key.any(dim=-1)
    if bool(has_valid_key.any().item()):
        invalid_key = ~valid_key[:, None, :]
        key_padding = invalid_key & has_valid_key[:, None, None]
        bias = bias.masked_fill(key_padding, -torch.finfo(dtype).max)
    stats = _region_attention_stats(
        allowed=allowed,
        valid_query=valid_query,
        valid_key=valid_key,
        missing=valid_query & ~allowed.any(dim=-1),
        fallback_used=torch.zeros_like(valid_query),
        null_query=torch.zeros_like(valid_query),
        allowed_with_null=valid_pairs,
        query_labels=query_labels,
        key_labels=key_labels,
        strict=strict,
    )
    valid_pair_count = int(valid_pairs.sum().item())
    same_pair_count = int(same.sum().item())
    other_pair_count = int(other.sum().item())
    stats.update(
        {
            "soft_bias_enabled": True,
            "soft_bias": float(soft_bias_tensor.detach().float().item()),
            "same_label_pair_fraction": (
                float(same_pair_count / valid_pair_count) if valid_pair_count > 0 else 0.0
            ),
            "other_label_pair_fraction": (
                float(other_pair_count / valid_pair_count) if valid_pair_count > 0 else 0.0
            ),
            "allowed_tokens_per_query_mean": float(valid_key.detach().sum(dim=-1).float().mean().item()),
            "allowed_tokens_per_query_min": int(valid_key.detach().sum(dim=-1).min().item()),
            "allowed_tokens_per_query_max": int(valid_key.detach().sum(dim=-1).max().item()),
            "active_query_fraction": (
                float(valid_query.detach().float().mean().item()) if valid_query.numel() else math.nan
            ),
            "null_query_fraction": 0.0,
        }
    )
    return bias[:, None, :, :], None, stats


def _region_attention_stats(
    *,
    allowed: torch.Tensor,
    valid_query: torch.Tensor,
    valid_key: torch.Tensor,
    missing: torch.Tensor,
    fallback_used: torch.Tensor,
    null_query: torch.Tensor,
    allowed_with_null: torch.Tensor,
    query_labels: torch.Tensor,
    key_labels: torch.Tensor,
    strict: bool,
) -> dict[str, float | int | bool]:
    valid_pairs = valid_query[:, :, None] & valid_key[:, None, :]
    valid_query_count = int(valid_query.sum().item())
    valid_key_count = int(valid_key.sum().item())
    valid_pair_count = int(valid_pairs.sum().item())

    def fraction(mask: torch.Tensor, denom: int | None = None) -> float:
        if denom is None:
            return float(mask.detach().float().mean().item()) if mask.numel() else math.nan
        if denom <= 0:
            return 0.0
        return float(mask.detach().float().sum().item() / denom)

    query_valid_values = query_labels[valid_query]
    key_valid_values = key_labels[valid_key]
    allowed_counts = allowed_with_null.detach().sum(dim=-1)
    valid_allowed_counts = allowed_counts[valid_query] if bool(valid_query.any().item()) else allowed_counts.flatten()
    return {
        "strict": bool(strict),
        "has_labels": True,
        "query_tokens": int(query_labels.shape[1]),
        "key_tokens": int(key_labels.shape[1]),
        "valid_query_fraction": fraction(valid_query),
        "valid_key_fraction": fraction(valid_key),
        "allowed_pair_fraction": fraction(allowed),
        "allowed_valid_pair_fraction": fraction(allowed & valid_pairs, valid_pair_count),
        "active_query_fraction": fraction(valid_query & ~missing, valid_query_count),
        "missing_query_fraction": fraction(missing, valid_query_count),
        "fallback_query_fraction": fraction(fallback_used, valid_query_count),
        "null_query_fraction": fraction(null_query, int(null_query.numel())),
        "allowed_tokens_per_query_mean": float(valid_allowed_counts.float().mean().item()) if valid_allowed_counts.numel() else 0.0,
        "allowed_tokens_per_query_min": int(valid_allowed_counts.min().item()) if valid_allowed_counts.numel() else 0,
        "allowed_tokens_per_query_max": int(valid_allowed_counts.max().item()) if valid_allowed_counts.numel() else 0,
        "unique_query_labels": int(torch.unique(query_valid_values).numel()) if query_valid_values.numel() else 0,
        "unique_key_labels": int(torch.unique(key_valid_values).numel()) if key_valid_values.numel() else 0,
    }


def _format_region_attention_stats(stats: dict[str, float | int | bool]) -> str:
    base = (
        f"strict={bool(stats['strict'])} has_labels={bool(stats['has_labels'])} "
        f"query_tokens={int(stats['query_tokens'])} key_tokens={int(stats['key_tokens'])} "
        f"valid_q={float(stats['valid_query_fraction']):.3f} "
        f"valid_k={float(stats['valid_key_fraction']):.3f} "
        f"allowed_pairs={float(stats['allowed_valid_pair_fraction']):.5f} "
        f"active_q={float(stats['active_query_fraction']):.3f} "
        f"missing_q={float(stats['missing_query_fraction']):.3f} "
        f"fallback_q={float(stats['fallback_query_fraction']):.3f} "
        f"null_q={float(stats.get('null_query_fraction', math.nan)):.3f} "
        f"allowed_per_q={float(stats.get('allowed_tokens_per_query_mean', math.nan)):.2f}/"
        f"{int(stats.get('allowed_tokens_per_query_min', 0))}/"
        f"{int(stats.get('allowed_tokens_per_query_max', 0))} "
        f"unique_q={int(stats['unique_query_labels'])} "
        f"unique_k={int(stats['unique_key_labels'])}"
    )
    if bool(stats.get("soft_bias_enabled", False)):
        base += (
            f" soft_bias={float(stats.get('soft_bias', math.nan)):.6f} "
            f"same_pairs={float(stats.get('same_label_pair_fraction', math.nan)):.5f} "
            f"other_pairs={float(stats.get('other_label_pair_fraction', math.nan)):.5f}"
        )
    return base


def _format_region_token_label_summary(
    labels: torch.Tensor,
    *,
    max_samples: int = 4,
    max_labels: int = 12,
) -> str:
    label_values = labels.detach().to(device="cpu", dtype=torch.long)
    batch, token_slots = label_values.shape
    valid = label_values >= 0
    unique_valid = torch.unique(label_values[valid]).numel() if bool(valid.any().item()) else 0
    sample_parts = []
    for sample_index in range(min(batch, int(max_samples))):
        sample = label_values[sample_index]
        sample_valid = sample[sample >= 0]
        if sample_valid.numel():
            values, counts = torch.unique(sample_valid, sorted=True, return_counts=True)
            entries = [
                f"{int(value.item())}:{int(count.item())}"
                for value, count in zip(values[:max_labels], counts[:max_labels])
            ]
            if values.numel() > max_labels:
                entries.append("...")
            label_counts = ",".join(entries)
        else:
            label_counts = "none"
        sample_parts.append(
            f"s{sample_index}[valid={int(sample_valid.numel())}/{int(token_slots)} labels={label_counts}]"
        )
    if batch > max_samples:
        sample_parts.append("...")
    return (
        f"batch={int(batch)} token_slots={int(token_slots)} "
        f"valid={int(valid.sum().item())}/{int(label_values.numel())} "
        f"unique_labels={int(unique_valid)} samples=" + "; ".join(sample_parts)
    )


def _unpack_region_ip_adapter_masks(
    ip_adapter_masks: torch.Tensor | dict | None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    bool,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    if not isinstance(ip_adapter_masks, dict):
        return None, None, True, None, None
    key_labels = ip_adapter_masks.get("key_region_labels")
    query_labels = ip_adapter_masks.get("query_region_labels")
    strict = bool(ip_adapter_masks.get("strict", True))
    key_fallback_labels = ip_adapter_masks.get("key_fallback_region_labels")
    query_fallback_labels = ip_adapter_masks.get("query_fallback_region_labels")
    return key_labels, query_labels, strict, key_fallback_labels, query_fallback_labels


def _unpack_region_ip_soft_bias_enabled(ip_adapter_masks: torch.Tensor | dict | None) -> bool:
    if not isinstance(ip_adapter_masks, dict):
        return False
    return bool(ip_adapter_masks.get("use_soft_bias", False))


def _unpack_region_ip_soft_bias_value(
    ip_adapter_masks: torch.Tensor | dict | None,
) -> torch.Tensor | float | None:
    if not isinstance(ip_adapter_masks, dict):
        return None
    return ip_adapter_masks.get("soft_bias")


def _record_ip_attention_debug(
    collector: dict | None,
    block_name: str,
    hidden_states: torch.Tensor,
    scaled_ip_output: torch.Tensor,
) -> None:
    """Record detached IP residual magnitudes without keeping the graph alive."""
    if collector is None:
        return
    with torch.no_grad():
        hidden = hidden_states.detach().float()
        ip_output = scaled_ip_output.detach().float()
        hidden_norm = float(torch.linalg.vector_norm(hidden).item()) if hidden.numel() else 0.0
        ip_norm = float(torch.linalg.vector_norm(ip_output).item()) if ip_output.numel() else 0.0
        ratio = ip_norm / max(hidden_norm, 1e-12)
        records = collector.setdefault("records", [])
        records.append(
            {
                "block": str(block_name),
                "hidden_norm": hidden_norm,
                "ip_norm": ip_norm,
                "ratio": ratio,
            }
        )
        should_store = bool(collector.get("store_first_ip_output", False))
        is_double_block = str(block_name).startswith("block_") or str(block_name) == "block"
        if should_store and is_double_block and "first_ip_output" not in collector:
            collector["first_ip_output"] = ip_output.cpu()
            collector["first_ip_block"] = str(block_name)


def install_flux_ip_adapter_attention(
    transformer: FluxTransformer2DModel,
    hidden_dim: int = 3072,
    cross_attention_dim: int = 3072,
    num_tokens: int = 16,
    scale: float = 1.0,
    ip_init_gain: float = 0.1,
    num_single_layers: int = 0,
    regional: bool = False,
    use_soft_bias: bool = False,
    soft_bias_init: float = 4.0,
) -> None:
    """Install IP-Adapter attention processors on FLUX double and last-N single blocks."""
    from diffusers.models.embeddings import IPAdapterFullImageProjection

    raw_proj = IPAdapterFullImageProjection(
        image_embed_dim=cross_attention_dim,
        cross_attention_dim=cross_attention_dim,
    )
    transformer.encoder_hid_proj = IPAdapterListProjection(raw_proj)

    for block_index, block in enumerate(transformer.transformer_blocks):
        processor = FluxRegionalIPAdapterJointAttnProcessor2_0(
            hidden_size=hidden_dim,
            cross_attention_dim=cross_attention_dim,
            num_tokens=(num_tokens,),
            scale=[scale],
            use_soft_bias=use_soft_bias,
            soft_bias_init=soft_bias_init,
        )
        processor.debug_name = f"block_{block_index}"
        for linear in processor.to_k_ip:
            _init_ip_adapter_linear(linear, gain=ip_init_gain)
        for linear in processor.to_v_ip:
            _init_ip_adapter_linear(linear, gain=ip_init_gain)
        block.attn.set_processor(processor)

    single_blocks = list(getattr(transformer, "single_transformer_blocks", []))
    if num_single_layers > 0:
        first_single_index = max(0, len(single_blocks) - int(num_single_layers))
        for offset, block in enumerate(single_blocks[-int(num_single_layers):]):
            processor = FluxSingleIPAdapterAttnProcessor2_0(
                hidden_size=hidden_dim,
                cross_attention_dim=cross_attention_dim,
                num_tokens=(num_tokens,),
                scale=[scale],
                use_soft_bias=use_soft_bias,
                soft_bias_init=soft_bias_init,
            )
            processor.debug_name = f"single_block_{first_single_index + offset}"
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


def _named_parameter_norm_hash(
    named_parameters: list[tuple[str, nn.Parameter]],
) -> dict[str, float | int | str]:
    hasher = hashlib.sha256()
    param_count = 0
    tensor_count = 0
    square_sum = 0.0
    max_abs = 0.0
    dtypes: set[str] = set()
    for name, param in sorted(named_parameters, key=lambda item: item[0]):
        value = param.detach().cpu().contiguous()
        tensor_count += 1
        param_count += int(value.numel())
        dtypes.add(str(value.dtype).replace("torch.", ""))
        value_float = value.float()
        if value_float.numel():
            square_sum += float(torch.sum(value_float * value_float).item())
            max_abs = max(max_abs, float(value_float.abs().max().item()))
        hasher.update(str(name).encode("utf8"))
        hasher.update(str(tuple(value.shape)).encode("utf8"))
        hasher.update(str(value.dtype).encode("utf8"))
        hasher.update(value_float.contiguous().numpy().tobytes())
    return {
        "tensors": tensor_count,
        "params": param_count,
        "l2": math.sqrt(square_sum),
        "max_abs": max_abs,
        "dtypes": ",".join(sorted(dtypes)) if dtypes else "none",
        "sha256": hasher.hexdigest()[:16],
    }


def _collect_ip_soft_bias_values(transformer: FluxTransformer2DModel) -> list[float]:
    values: list[float] = []
    for blocks in (
        getattr(transformer, "transformer_blocks", []),
        getattr(transformer, "single_transformer_blocks", []),
    ):
        for block in blocks:
            processor = getattr(getattr(block, "attn", None), "processor", None)
            soft_bias = getattr(processor, "ip_soft_bias", None)
            if soft_bias is None:
                continue
            for param in soft_bias:
                if torch.is_tensor(param):
                    values.append(float(param.detach().float().mean().item()))
    return values


def _ip_soft_bias_summary(transformer: FluxTransformer2DModel) -> dict[str, float | int]:
    values = _collect_ip_soft_bias_values(transformer)
    if not values:
        return {
            "count": 0,
            "min": math.nan,
            "mean": math.nan,
            "max": math.nan,
        }
    return {
        "count": len(values),
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def _ip_soft_bias_log_values(transformer: FluxTransformer2DModel) -> dict[str, float]:
    summary = _ip_soft_bias_summary(transformer)
    if int(summary["count"]) <= 0:
        return {}
    return {
        "soft_bias_min": float(summary["min"]),
        "soft_bias_mean": float(summary["mean"]),
        "soft_bias_max": float(summary["max"]),
    }


def _ip_soft_bias_value_for_probe(transformer: FluxTransformer2DModel) -> torch.Tensor | float | None:
    values = _collect_ip_soft_bias_values(transformer)
    if not values:
        return None
    return torch.tensor(float(values[0]), device=next(transformer.parameters()).device)


def _log_cross_v1_step0_adapter_assert(
    *,
    accelerator: Accelerator,
    ref_trainable_wrapper: nn.Module,
    ip_trainable_wrapper: nn.Module,
    transformer: FluxTransformer2DModel,
    architecture: str,
    regional_ip_adapter: bool,
    regional_ip_strict: bool,
    regional_ip_token_mode: str,
    regional_ip_label_mode: str,
    use_soft_bias: bool,
    soft_bias_init: float,
    loaded_ip_adapter_checkpoint: str | Path | None,
    loaded_resume_checkpoint: str | Path | None,
) -> None:
    if not accelerator.is_local_main_process:
        return
    ref_stats = _named_parameter_norm_hash(
        [
            (_clean_wrapped_parameter_name(name), param)
            for name, param in ref_trainable_wrapper.named_parameters()
            if param.requires_grad
        ]
    )
    ip_stats = _named_parameter_norm_hash(
        [
            (_clean_wrapped_parameter_name(name), param)
            for name, param in ip_trainable_wrapper.named_parameters()
            if param.requires_grad
        ]
    )
    soft_summary = _ip_soft_bias_summary(transformer)
    logger.info(
        "STEP0_ADAPTER_ASSERT architecture=%s regional_ip_adapter=%s strict=%s "
        "token_mode=%s label_mode=%s use_soft_bias=%s soft_bias_init=%.6f "
        "soft_bias[count/min/mean/max]=%s/%.6f/%.6f/%.6f "
        "ip_loaded_from=%s resume_loaded_from=%s",
        normalize_cross_v1_ip_architecture(architecture),
        bool(regional_ip_adapter),
        bool(regional_ip_strict),
        normalize_region_ip_token_mode(regional_ip_token_mode),
        normalize_region_ip_label_mode(regional_ip_label_mode),
        bool(use_soft_bias),
        float(soft_bias_init),
        int(soft_summary["count"]),
        float(soft_summary["min"]),
        float(soft_summary["mean"]),
        float(soft_summary["max"]),
        str(loaded_ip_adapter_checkpoint) if loaded_ip_adapter_checkpoint else "fresh",
        str(loaded_resume_checkpoint) if loaded_resume_checkpoint else "none",
    )
    logger.info(
        "STEP0_ADAPTER_ASSERT ip_adapter tensors=%s params=%s dtypes=%s l2=%.6e max_abs=%.6e hash=%s",
        int(ip_stats["tensors"]),
        int(ip_stats["params"]),
        str(ip_stats["dtypes"]),
        float(ip_stats["l2"]),
        float(ip_stats["max_abs"]),
        str(ip_stats["sha256"]),
    )
    logger.info(
        "STEP0_ADAPTER_ASSERT ref_encoder_trainable tensors=%s params=%s dtypes=%s l2=%.6e max_abs=%.6e hash=%s",
        int(ref_stats["tensors"]),
        int(ref_stats["params"]),
        str(ref_stats["dtypes"]),
        float(ref_stats["l2"]),
        float(ref_stats["max_abs"]),
        str(ref_stats["sha256"]),
    )


def _grad_flow_stats(
    named_parameters: list[tuple[str, nn.Parameter]],
    *,
    optimizer_param_ids: set[int],
) -> dict[str, float | int]:
    trainable = [(name, param) for name, param in named_parameters if param.requires_grad]
    grad_tensors = 0
    none_grad_tensors = 0
    zero_grad_tensors = 0
    finite_values = 0
    total_values = 0
    grad_sq_sum = 0.0
    grad_max_abs = 0.0
    for _, param in trainable:
        grad = param.grad
        if grad is None:
            none_grad_tensors += 1
            continue
        grad_tensors += 1
        value = grad.detach().float()
        finite = torch.isfinite(value)
        finite_values += int(finite.sum().item())
        total_values += int(value.numel())
        if not bool(finite.all().item()):
            value = value[finite] if bool(finite.any().item()) else value.new_zeros((1,))
        max_abs = float(value.abs().max().item()) if value.numel() else 0.0
        grad_max_abs = max(grad_max_abs, max_abs)
        grad_sq_sum += float(torch.sum(value * value).item())
        if max_abs == 0.0:
            zero_grad_tensors += 1
    return {
        "trainable_tensors": len(trainable),
        "trainable_params": sum(param.numel() for _, param in trainable),
        "optimizer_tensors": sum(id(param) in optimizer_param_ids for _, param in trainable),
        "grad_tensors": grad_tensors,
        "none_grad_tensors": none_grad_tensors,
        "zero_grad_tensors": zero_grad_tensors,
        "grad_norm": math.sqrt(grad_sq_sum),
        "grad_max_abs": grad_max_abs,
        "grad_finite_fraction": (
            float(finite_values / total_values) if total_values > 0 else math.nan
        ),
    }


def _format_grad_flow_stats(name: str, stats: dict[str, float | int]) -> str:
    return (
        f"{name}: trainable_tensors={int(stats['trainable_tensors'])} "
        f"trainable_params={int(stats['trainable_params'])} "
        f"in_optimizer={int(stats['optimizer_tensors'])} "
        f"grad_tensors={int(stats['grad_tensors'])} "
        f"none_grad={int(stats['none_grad_tensors'])} "
        f"zero_grad={int(stats['zero_grad_tensors'])} "
        f"grad_norm={float(stats['grad_norm']):.6e} "
        f"grad_max_abs={float(stats['grad_max_abs']):.6e} "
        f"finite={float(stats['grad_finite_fraction']):.3f}"
    )


def _clean_wrapped_parameter_name(name: str) -> str:
    return name[7:] if name.startswith("module.") else name


def _select_named_parameters(
    params: list[tuple[str, nn.Parameter]],
    predicate: Callable[[str], bool],
) -> list[tuple[str, nn.Parameter]]:
    return [
        (name, param)
        for name, param in params
        if predicate(_clean_wrapped_parameter_name(name))
    ]


def _reference_ip_parameter_groups(
    *,
    ref_trainable_wrapper: nn.Module,
    ip_trainable_wrapper: nn.Module,
) -> dict[str, list[tuple[str, nn.Parameter]]]:
    ref_params = list(ref_trainable_wrapper.named_parameters())
    ip_params = list(ip_trainable_wrapper.named_parameters())
    return {
        "ref.proj_mlp": _select_named_parameters(
            ref_params,
            lambda name: name.startswith("proj_mlp."),
        ),
        "ref.latent_queries": _select_named_parameters(
            ref_params,
            lambda name: name == "latent_queries",
        ),
        "ref.perceiver": _select_named_parameters(
            ref_params,
            lambda name: name.startswith("perceiver_layers.") or name.startswith("perceiver_norm."),
        ),
        "ip.encoder_hid_proj": _select_named_parameters(
            ip_params,
            lambda name: name.startswith("encoder_hid_proj."),
        ),
        "ip.double_to_k_ip": _select_named_parameters(
            ip_params,
            lambda name: name.startswith("block_") and "_to_k_ip." in name,
        ),
        "ip.double_to_v_ip": _select_named_parameters(
            ip_params,
            lambda name: name.startswith("block_") and "_to_v_ip." in name,
        ),
        "ip.double_null": _select_named_parameters(
            ip_params,
            lambda name: name.startswith("block_") and "_ip_null_tokens." in name,
        ),
        "ip.double_soft_bias": _select_named_parameters(
            ip_params,
            lambda name: name.startswith("block_") and "_ip_soft_bias." in name,
        ),
        "ip.single_to_k_ip": _select_named_parameters(
            ip_params,
            lambda name: name.startswith("single_block_") and "_to_k_ip." in name,
        ),
        "ip.single_to_v_ip": _select_named_parameters(
            ip_params,
            lambda name: name.startswith("single_block_") and "_to_v_ip." in name,
        ),
        "ip.single_null": _select_named_parameters(
            ip_params,
            lambda name: name.startswith("single_block_") and "_ip_null_tokens." in name,
        ),
        "ip.single_soft_bias": _select_named_parameters(
            ip_params,
            lambda name: name.startswith("single_block_") and "_ip_soft_bias." in name,
        ),
    }


class IPTrainableHealthMonitor:
    """Track grad-ever-nonzero and fp32 parameter deltas for IP/ref modules."""

    def __init__(
        self,
        *,
        ref_trainable_wrapper: nn.Module,
        ip_trainable_wrapper: nn.Module,
        accelerator: Accelerator,
        warmup_steps: int,
    ) -> None:
        self.enabled = bool(accelerator.is_local_main_process)
        self.warmup_steps = max(1, int(warmup_steps))
        self.groups = (
            _reference_ip_parameter_groups(
                ref_trainable_wrapper=ref_trainable_wrapper,
                ip_trainable_wrapper=ip_trainable_wrapper,
            )
            if self.enabled
            else {}
        )
        self.initial: dict[str, list[tuple[str, torch.Tensor]]] = {}
        self.grad_ever_nonzero: dict[str, bool] = {}
        self._warned_zero_delta: set[str] = set()
        self._warned_zero_grad: set[str] = set()
        self._warned_dtype: set[str] = set()
        if not self.enabled:
            return
        for group_name, params in self.groups.items():
            self.initial[group_name] = [
                (
                    _clean_wrapped_parameter_name(name),
                    param.detach().float().cpu().clone(),
                )
                for name, param in params
                if param.requires_grad
            ]
            self.grad_ever_nonzero[group_name] = False

    def record_after_backward(self) -> None:
        if not self.enabled:
            return
        for group_name, params in self.groups.items():
            for _, param in params:
                grad = param.grad
                if grad is None:
                    continue
                grad_value = grad.detach()
                if grad_value.numel() and bool((grad_value != 0).any().item()):
                    self.grad_ever_nonzero[group_name] = True
                    break

    def log_param_delta(self, *, step: int) -> dict[str, float]:
        logs: dict[str, float] = {}
        if not self.enabled:
            return logs
        logger.info("IP/ref param delta health step=%s", step)
        for group_name, params in self.groups.items():
            trainable_params = [
                (_clean_wrapped_parameter_name(name), param)
                for name, param in params
                if param.requires_grad
            ]
            initial_by_name = dict(self.initial.get(group_name, []))
            delta_sq_sum = 0.0
            theta_sq_sum = 0.0
            max_abs = 0.0
            dtype_names = sorted({str(param.dtype).replace("torch.", "") for _, param in trainable_params})
            for name, param in trainable_params:
                initial = initial_by_name.get(name)
                if initial is None:
                    continue
                current = param.detach().float().cpu()
                delta = current - initial
                delta_sq_sum += float(torch.sum(delta * delta).item())
                theta_sq_sum += float(torch.sum(initial * initial).item())
                max_abs = max(max_abs, float(current.abs().max().item()) if current.numel() else 0.0)
            delta_norm = math.sqrt(delta_sq_sum)
            initial_norm = math.sqrt(theta_sq_sum)
            relative_delta = delta_norm / max(initial_norm, 1e-12)
            ever_nonzero = bool(self.grad_ever_nonzero.get(group_name, False))
            safe_key = group_name.replace(".", "_")
            logs[f"ip_health_{safe_key}_param_delta"] = delta_norm
            logs[f"ip_health_{safe_key}_relative_delta"] = relative_delta
            logs[f"ip_health_{safe_key}_grad_ever_nonzero"] = float(ever_nonzero)
            current_stats = _named_parameter_norm_hash(trainable_params)
            logger.info(
                "IP/ref param delta health %s: tensors=%s params=%s dtypes=%s "
                "delta_norm=%.6e relative_delta=%.6e max_abs=%.6e hash=%s grad_ever_nonzero=%s",
                group_name,
                len(trainable_params),
                sum(param.numel() for _, param in trainable_params),
                ",".join(dtype_names) if dtype_names else "none",
                delta_norm,
                relative_delta,
                max_abs,
                str(current_stats["sha256"]),
                ever_nonzero,
            )
            if trainable_params and dtype_names != ["float32"] and group_name not in self._warned_dtype:
                logger.warning(
                    "IP/ref param delta health: %s parameters are %s, expected float32. "
                    "bf16/fp16 trainable weights can fall into the AdamW rounding dead zone.",
                    group_name,
                    ",".join(dtype_names),
                )
                self._warned_dtype.add(group_name)
            if step >= self.warmup_steps and trainable_params and delta_norm == 0.0 and group_name not in self._warned_zero_delta:
                logger.warning(
                    "IP/ref param delta health: %s has zero ||theta_t - theta_0|| after %s steps. "
                    "This is a strong sign that optimizer updates are not changing the weights.",
                    group_name,
                    step,
                )
                self._warned_zero_delta.add(group_name)
            if step >= self.warmup_steps and trainable_params and not ever_nonzero and group_name not in self._warned_zero_grad:
                logger.warning(
                    "IP/ref param delta health: %s never received a nonzero gradient in the first %s steps.",
                    group_name,
                    step,
                )
                self._warned_zero_grad.add(group_name)
        return logs


def _log_gradient_flow_debug(
    *,
    ref_trainable_wrapper: nn.Module,
    ip_trainable_wrapper: nn.Module,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    step: int,
) -> None:
    if not accelerator.is_local_main_process:
        return
    optimizer_param_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group.get("params", [])
    }
    groups = _reference_ip_parameter_groups(
        ref_trainable_wrapper=ref_trainable_wrapper,
        ip_trainable_wrapper=ip_trainable_wrapper,
    )
    logger.info("Gradient flow debug step=%s", step)
    for name, params in groups.items():
        stats = _grad_flow_stats(params, optimizer_param_ids=optimizer_param_ids)
        logger.info("Gradient flow debug %s", _format_grad_flow_stats(name, stats))
        trainable_tensors = int(stats["trainable_tensors"])
        if trainable_tensors <= 0:
            continue
        if int(stats["optimizer_tensors"]) != trainable_tensors:
            logger.warning(
                "Gradient flow debug: %s has %s/%s trainable tensors in the optimizer.",
                name,
                int(stats["optimizer_tensors"]),
                trainable_tensors,
            )
        if int(stats["grad_tensors"]) == 0:
            logger.info(
                "Gradient flow debug: %s received no gradients on this backward; "
                "the 100-step param-delta monitor is the hard health check.",
                name,
            )
        elif float(stats["grad_norm"]) == 0.0:
            logger.info(
                "Gradient flow debug: %s gradients are exactly zero on this backward; "
                "this can be valid for zero-initialized downstream projections.",
                name,
            )


def _collect_ip_adapter_modules(transformer: FluxTransformer2DModel) -> dict[str, nn.Module]:
    """Collect all IP-Adapter trainable modules attached to the frozen transformer."""
    from diffusers.models.attention_processor import FluxIPAdapterJointAttnProcessor2_0

    modules: dict[str, nn.Module] = {}
    if hasattr(transformer, "encoder_hid_proj"):
        modules["encoder_hid_proj"] = transformer.encoder_hid_proj
    for i, block in enumerate(transformer.transformer_blocks):
        processor = block.attn.processor
        if isinstance(processor, (FluxIPAdapterJointAttnProcessor2_0, FluxRegionalIPAdapterJointAttnProcessor2_0)):
            modules[f"block_{i}_to_k_ip"] = processor.to_k_ip
            modules[f"block_{i}_to_v_ip"] = processor.to_v_ip
            if hasattr(processor, "ip_null_tokens"):
                modules[f"block_{i}_ip_null_tokens"] = processor.ip_null_tokens
            if bool(getattr(processor, "use_soft_bias", False)) and hasattr(processor, "ip_soft_bias"):
                modules[f"block_{i}_ip_soft_bias"] = processor.ip_soft_bias
    for i, block in enumerate(getattr(transformer, "single_transformer_blocks", [])):
        processor = block.attn.processor
        if isinstance(processor, FluxSingleIPAdapterAttnProcessor2_0):
            modules[f"single_block_{i}_to_k_ip"] = processor.to_k_ip
            modules[f"single_block_{i}_to_v_ip"] = processor.to_v_ip
            if hasattr(processor, "ip_null_tokens"):
                modules[f"single_block_{i}_ip_null_tokens"] = processor.ip_null_tokens
            if bool(getattr(processor, "use_soft_bias", False)) and hasattr(processor, "ip_soft_bias"):
                modules[f"single_block_{i}_ip_soft_bias"] = processor.ip_soft_bias
    return modules


@contextlib.contextmanager
def _temporary_ip_debug_collector(
    transformer: FluxTransformer2DModel,
    collector: dict | None,
):
    """Attach debug collection only to IP processors, avoiding FluxAttnProcessor noise."""
    processors: list[tuple[object, bool, dict | None]] = []
    if collector is not None:
        for block in transformer.transformer_blocks:
            processor = block.attn.processor
            if isinstance(processor, FluxRegionalIPAdapterJointAttnProcessor2_0):
                processors.append(
                    (
                        processor,
                        hasattr(processor, "_ip_debug_collector"),
                        getattr(processor, "_ip_debug_collector", None),
                    )
                )
                setattr(processor, "_ip_debug_collector", collector)
        for block in getattr(transformer, "single_transformer_blocks", []):
            processor = block.attn.processor
            if isinstance(processor, FluxSingleIPAdapterAttnProcessor2_0):
                processors.append(
                    (
                        processor,
                        hasattr(processor, "_ip_debug_collector"),
                        getattr(processor, "_ip_debug_collector", None),
                    )
                )
                setattr(processor, "_ip_debug_collector", collector)
    try:
        yield
    finally:
        for processor, had_value, old_value in processors:
            if had_value:
                setattr(processor, "_ip_debug_collector", old_value)
            elif hasattr(processor, "_ip_debug_collector"):
                delattr(processor, "_ip_debug_collector")


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
        null_key = f"block_{i}_ip_null_tokens"
        bias_key = f"block_{i}_ip_soft_bias"
        if hasattr(ip_wrapper, k_key):
            block.attn.processor.to_k_ip = getattr(ip_wrapper, k_key)
            block.attn.processor.to_v_ip = getattr(ip_wrapper, v_key)
        if hasattr(ip_wrapper, null_key):
            block.attn.processor.ip_null_tokens = getattr(ip_wrapper, null_key)
        if hasattr(ip_wrapper, bias_key):
            block.attn.processor.ip_soft_bias = getattr(ip_wrapper, bias_key)
    for i, block in enumerate(getattr(transformer, "single_transformer_blocks", [])):
        k_key = f"single_block_{i}_to_k_ip"
        v_key = f"single_block_{i}_to_v_ip"
        null_key = f"single_block_{i}_ip_null_tokens"
        bias_key = f"single_block_{i}_ip_soft_bias"
        if hasattr(ip_wrapper, k_key):
            block.attn.processor.to_k_ip = getattr(ip_wrapper, k_key)
            block.attn.processor.to_v_ip = getattr(ip_wrapper, v_key)
        if hasattr(ip_wrapper, null_key):
            block.attn.processor.ip_null_tokens = getattr(ip_wrapper, null_key)
        if hasattr(ip_wrapper, bias_key):
            block.attn.processor.ip_soft_bias = getattr(ip_wrapper, bias_key)


def patch_flux_single_ip_forward(transformer: FluxTransformer2DModel) -> None:
    """Pass the text/image split index to single-stream IP processors."""
    if getattr(transformer, "_cross_v1_single_ip_forward_patched", False):
        return
    has_single_ip_processor = any(
        isinstance(getattr(block.attn, "processor", None), FluxSingleIPAdapterAttnProcessor2_0)
        for block in getattr(transformer, "single_transformer_blocks", [])
    )
    if not has_single_ip_processor:
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


def _move_reference_encoder(
    ref_encoder: ReferenceImageEncoder,
    *,
    device: torch.device | str,
    train_dtype: torch.dtype = torch.float32,
) -> None:
    """Move ref encoder while keeping frozen UNI and trainable ref modules in fp32."""
    ref_encoder.to(device=device)
    ref_encoder.proj_mlp.to(device=device, dtype=train_dtype)
    ref_encoder.perceiver_layers.to(device=device, dtype=train_dtype)
    ref_encoder.perceiver_norm.to(device=device, dtype=train_dtype)
    ref_encoder.latent_queries.data = ref_encoder.latent_queries.data.to(
        device=device,
        dtype=train_dtype,
    )
    ref_encoder.uni.to(device=device, dtype=torch.float32)
    ref_encoder._lock_uni_backbone()


def _move_ip_adapter_modules(
    ip_adapter_modules: dict[str, nn.Module],
    *,
    device: torch.device | str,
    train_dtype: torch.dtype = torch.float32,
) -> None:
    """Keep trainable IP-Adapter parameters in fp32 under mixed-precision training."""
    for module in ip_adapter_modules.values():
        module.to(device=device, dtype=train_dtype)
        module.requires_grad_(True)


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
    """Decode one-step denoised model output to RGB in [0, 1] for RGB-space losses."""
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = next(vae.parameters()).device
    target_image_latent = _encode_images_to_latents(vae, batch["target_image"], weight_dtype)
    noising_image_latent = _encode_images_to_latents(
        vae,
        batch.get("clean_image_for_noising", batch["target_image"]),
        weight_dtype,
    )

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
    return target_image_latent, noising_image_latent, control_tensor


def _build_ip_adapter_kwargs(
    batch: dict,
    modules: dict[str, torch.nn.Module],
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    transformer: FluxTransformer2DModel,
    *,
    regional: bool = False,
    query_token_count: int | None = None,
    strict: bool = True,
    regional_token_mode: str = "spatial",
    regional_label_mode: str = "tissue",
    use_soft_bias: bool = False,
    soft_bias: torch.Tensor | float | None = None,
    ip_debug_collector: dict | None = None,
) -> dict:
    """Build joint_attention_kwargs with pre-projected ip_hidden_states."""
    ref_encoder = modules["ref_encoder"]
    uni_dtype = next(ref_encoder.uni.parameters()).dtype
    reference_images = batch["reference_image"].to(device=accelerator.device, dtype=uni_dtype)
    if regional:
        regional_token_mode = normalize_region_ip_token_mode(regional_token_mode)
        regional_label_mode = normalize_region_ip_label_mode(regional_label_mode)
        reference_tissue_mask = batch["reference_tissue_mask"].to(device=accelerator.device)
        reference_nuclei_mask = batch["reference_nuclei_mask"].to(device=accelerator.device)
        target_tissue_mask = batch["target_tissue_mask"].to(device=accelerator.device)
        target_nuclei_mask = batch["target_nuclei_mask"].to(device=accelerator.device)
        ref_ip_features, region_token_labels = ref_encoder.encode_region_ip_tokens(
            reference_images,
            reference_tissue_mask,
            nuclei_mask=reference_nuclei_mask,
            token_mode=regional_token_mode,
            label_mode=regional_label_mode,
        )
        if query_token_count is None:
            raise ValueError("query_token_count is required for regional IP-Adapter.")
        query_region_labels = build_region_ip_token_labels(
            tissue_mask=target_tissue_mask,
            num_tokens=int(query_token_count),
            nuclei_mask=target_nuclei_mask,
            label_mode=regional_label_mode,
        )
        key_fallback_region_labels = _tissue_fallback_region_labels(
            region_token_labels,
            label_mode=regional_label_mode,
        )
        query_fallback_region_labels = resize_mask_to_token_labels(
            target_tissue_mask,
            int(query_token_count),
        )
    else:
        ref_ip_features = ref_encoder(reference_images)
        region_token_labels = None
        query_region_labels = None
        key_fallback_region_labels = None
        query_fallback_region_labels = None
    ref_gate = ref_encoder.reference_presence_gate(
        reference_images,
        device=accelerator.device,
        dtype=next(transformer.encoder_hid_proj.parameters()).dtype,
    )
    ip_hidden_states = transformer.encoder_hid_proj([ref_ip_features])
    ip_hidden_states = [
        hs.to(device=accelerator.device) * ref_gate.to(device=accelerator.device, dtype=hs.dtype)
        for hs in ip_hidden_states
    ]
    kwargs = {"ip_hidden_states": ip_hidden_states}
    if regional:
        mask_payload = {
            "key_region_labels": region_token_labels.to(device=accelerator.device),
            "query_region_labels": query_region_labels.to(device=accelerator.device),
            "key_fallback_region_labels": key_fallback_region_labels.to(device=accelerator.device),
            "query_fallback_region_labels": query_fallback_region_labels.to(device=accelerator.device),
            "strict": bool(strict),
            "use_soft_bias": bool(use_soft_bias),
        }
        if soft_bias is not None:
            mask_payload["soft_bias"] = soft_bias
        kwargs.update(
            {
                "ip_adapter_masks": mask_payload,
            }
        )
    return kwargs


def _regional_ip_mask_stats_from_kwargs(
    kwargs: dict,
    *,
    batch_size: int,
    query_token_count: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, float | int | bool] | None:
    masks = kwargs.get("ip_adapter_masks")
    if not isinstance(masks, dict):
        return None
    key_region_labels = masks.get("key_region_labels")
    query_region_labels = masks.get("query_region_labels")
    if key_region_labels is None or query_region_labels is None:
        return None
    _, _, stats = _build_region_attention_mask_and_query_gate(
        query_region_labels=query_region_labels,
        key_region_labels=key_region_labels,
        query_fallback_labels=masks.get("query_fallback_region_labels"),
        key_fallback_labels=masks.get("key_fallback_region_labels"),
        batch_size=batch_size,
        query_len=int(query_token_count),
        key_len=int(key_region_labels.shape[1]),
        device=device,
        dtype=dtype,
        strict=bool(masks.get("strict", True)),
        soft_bias=masks.get("soft_bias"),
        use_soft_bias=bool(masks.get("use_soft_bias", False)),
    )
    return stats


def _tensor_signal_stats(tensor: torch.Tensor) -> dict[str, float | list[int]]:
    value = tensor.detach().float()
    if value.numel() == 0:
        return {
            "shape": list(tensor.shape),
            "mean": math.nan,
            "std": math.nan,
            "min": math.nan,
            "max": math.nan,
            "abs_mean": math.nan,
            "l2": math.nan,
            "finite_fraction": math.nan,
        }
    finite = torch.isfinite(value)
    finite_fraction = float(finite.float().mean().item())
    value = value[finite] if bool(finite.any().item()) else value.reshape(-1)
    return {
        "shape": list(tensor.shape),
        "mean": float(value.mean().item()),
        "std": float(value.std(unbiased=False).item()),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "abs_mean": float(value.abs().mean().item()),
        "l2": float(torch.linalg.vector_norm(value).item()),
        "finite_fraction": finite_fraction,
    }


def _tensor_cosine_against(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 or right.numel() == 0:
        return math.nan
    if left.shape == right.shape:
        left_flat = left.detach().float().reshape(1, -1)
        right_flat = right.detach().float().reshape(1, -1)
        return float(F.cosine_similarity(left_flat, right_flat, dim=1, eps=1e-8).item())
    if left.ndim == 3 and right.ndim == 3 and left.shape[0] == right.shape[0] and left.shape[-1] == right.shape[-1]:
        left_flat = left.detach().float().mean(dim=1)
        right_flat = right.detach().float().mean(dim=1)
        cosine = F.cosine_similarity(left_flat, right_flat, dim=1, eps=1e-8)
        return float(cosine.mean().item()) if cosine.numel() else math.nan
    return math.nan


def _format_signal_stats(name: str, stats: dict[str, float | list[int]], *, zero_cosine: float | None = None) -> str:
    text = (
        f"{name}: shape={stats['shape']} mean={float(stats['mean']):.5f} "
        f"std={float(stats['std']):.5f} min={float(stats['min']):.5f} "
        f"max={float(stats['max']):.5f} abs_mean={float(stats['abs_mean']):.5f} "
        f"l2={float(stats['l2']):.5f} finite={float(stats['finite_fraction']):.3f}"
    )
    if zero_cosine is not None:
        text += f" zero_cos={zero_cosine:.5f}"
    return text


def _log_sensitivity_warning(
    *,
    stage: str,
    zero_cosine: float,
    real_cosine: float,
    warn_threshold: float = 0.99,
) -> None:
    if math.isfinite(zero_cosine) and zero_cosine >= warn_threshold:
        logger.warning(
            "Reference signal debug: %s normal-vs-zero cosine %.5f >= %.2f; "
            "image information may already be invariant at this stage.",
            stage,
            zero_cosine,
            warn_threshold,
        )
    if math.isfinite(real_cosine) and real_cosine >= warn_threshold:
        logger.warning(
            "Reference signal debug: %s real-vs-real cosine %.5f >= %.2f; "
            "different real references look nearly identical at this stage.",
            stage,
            real_cosine,
            warn_threshold,
        )


def _token_variation_stats(tensor: torch.Tensor) -> dict[str, float]:
    if tensor.ndim < 3 or tensor.numel() == 0:
        return {
            "token_norm_mean": math.nan,
            "token_norm_std": math.nan,
            "token_norm_min": math.nan,
            "token_norm_max": math.nan,
            "within_sample_token_std": math.nan,
            "batch_centered_l2_mean": math.nan,
            "batch_centered_l2_std": math.nan,
        }
    value = tensor.detach().float()
    token_norm = torch.linalg.vector_norm(value, dim=-1)
    centered = value - value.mean(dim=1, keepdim=True)
    sample_pooled = value.mean(dim=1)
    batch_centered = sample_pooled - sample_pooled.mean(dim=0, keepdim=True)
    batch_centered_l2 = torch.linalg.vector_norm(batch_centered, dim=-1)
    return {
        "token_norm_mean": float(token_norm.mean().item()),
        "token_norm_std": float(token_norm.std(unbiased=False).item()),
        "token_norm_min": float(token_norm.min().item()),
        "token_norm_max": float(token_norm.max().item()),
        "within_sample_token_std": float(centered.std(unbiased=False).item()),
        "batch_centered_l2_mean": float(batch_centered_l2.mean().item()),
        "batch_centered_l2_std": float(batch_centered_l2.std(unbiased=False).item()),
    }


def _format_token_variation_stats(name: str, tensor: torch.Tensor) -> str:
    stats = _token_variation_stats(tensor)
    return (
        f"{name}: token_norm_mean={stats['token_norm_mean']:.5f} "
        f"token_norm_std={stats['token_norm_std']:.5f} "
        f"token_norm_min={stats['token_norm_min']:.5f} "
        f"token_norm_max={stats['token_norm_max']:.5f} "
        f"within_sample_token_std={stats['within_sample_token_std']:.5f} "
        f"batch_centered_l2_mean={stats['batch_centered_l2_mean']:.5f} "
        f"batch_centered_l2_std={stats['batch_centered_l2_std']:.5f}"
    )


def _format_sequence_values(values: list[str], *, limit: int = 4) -> str:
    if not values:
        return "[]"
    shown = [str(value) for value in values[:limit]]
    suffix = "" if len(values) <= limit else f", ... (+{len(values) - limit})"
    return "[" + ", ".join(shown) + suffix + "]"


def _log_reference_batch_data_debug(batch: dict, reference_images: torch.Tensor) -> None:
    target_images = batch["target_image"].to(device=reference_images.device, dtype=reference_images.dtype)
    sample_ids = [str(value) for value in batch.get("sample_ids", [])]
    reference_ids = [str(value) for value in batch.get("reference_sample_ids", [])]
    same_id_fraction = math.nan
    if sample_ids and reference_ids and len(sample_ids) == len(reference_ids):
        same_id_fraction = sum(a == b for a, b in zip(sample_ids, reference_ids)) / len(sample_ids)
    ref_flat = reference_images.detach().float().flatten(1)
    target_flat = target_images.detach().float().flatten(1)
    cosine = F.cosine_similarity(ref_flat, target_flat, dim=1, eps=1e-8)
    logger.info(
        "Reference data debug: batch=%s sample_ids=%s reference_ids=%s "
        "unique_samples=%s unique_references=%s same_id_fraction=%.3f "
        "ref_target_cos_mean=%.5f ref_target_cos_min=%.5f ref_target_cos_max=%.5f",
        int(reference_images.shape[0]),
        _format_sequence_values(sample_ids),
        _format_sequence_values(reference_ids),
        len(set(sample_ids)) if sample_ids else 0,
        len(set(reference_ids)) if reference_ids else 0,
        same_id_fraction,
        float(cosine.mean().item()) if cosine.numel() else math.nan,
        float(cosine.min().item()) if cosine.numel() else math.nan,
        float(cosine.max().item()) if cosine.numel() else math.nan,
    )


def _alternate_real_reference_batch(
    batch: dict,
    *,
    random_batch: dict | None = None,
) -> dict:
    """Build a real-reference contrast batch with the same target/noise tensors."""
    alternate = dict(batch)
    bsz = int(batch["reference_image"].shape[0])
    if random_batch is not None:
        for key in ("reference_image", "reference_tissue_mask", "reference_nuclei_mask"):
            if key not in random_batch:
                raise KeyError(f"random reference batch is missing {key!r}")
            alternate[key] = random_batch[key].to(device=batch[key].device)
        return alternate
    if bsz > 1:
        order = torch.arange(bsz, device=batch["reference_image"].device).roll(1)
        alternate["reference_image"] = batch["reference_image"].index_select(0, order)
        alternate["reference_tissue_mask"] = batch["reference_tissue_mask"].index_select(0, order)
        alternate["reference_nuclei_mask"] = batch["reference_nuclei_mask"].index_select(0, order)
        return alternate
    alternate["reference_image"] = batch["target_image"]
    alternate["reference_tissue_mask"] = batch["target_tissue_mask"]
    alternate["reference_nuclei_mask"] = batch["target_nuclei_mask"]
    return alternate


@torch.no_grad()
def _log_reference_signal_debug(
    *,
    batch: dict,
    modules: dict[str, torch.nn.Module],
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    transformer: FluxTransformer2DModel,
    regional: bool,
    regional_strict: bool,
    regional_token_mode: str,
    regional_label_mode: str,
    use_soft_bias: bool,
    soft_bias: torch.Tensor | float | None,
    query_token_count: int | None,
    step: int,
    real_contrast_batch: dict | None = None,
) -> None:
    if not accelerator.is_local_main_process:
        return

    ref_encoder = modules["ref_encoder"]
    uni_dtype = next(ref_encoder.uni.parameters()).dtype
    reference_images = batch["reference_image"].to(device=accelerator.device, dtype=uni_dtype)
    zero_images = torch.zeros_like(reference_images)
    reference_tissue_mask = batch["reference_tissue_mask"].to(device=accelerator.device)
    reference_nuclei_mask = batch["reference_nuclei_mask"].to(device=accelerator.device)
    if real_contrast_batch is None:
        real_contrast_batch = _alternate_real_reference_batch(batch)
    real_images = real_contrast_batch["reference_image"].to(device=accelerator.device, dtype=uni_dtype)
    real_tissue_mask = real_contrast_batch["reference_tissue_mask"].to(device=accelerator.device)
    real_nuclei_mask = real_contrast_batch["reference_nuclei_mask"].to(device=accelerator.device)
    _log_reference_batch_data_debug(batch, reference_images)

    def encode_variant(
        images: torch.Tensor,
        tissue_mask: torch.Tensor,
        nuclei_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        uni_features = ref_encoder.extract_uni_features(images)
        proj_dtype = next(ref_encoder.proj_mlp.parameters()).dtype
        projected = ref_encoder.proj_mlp(uni_features.to(dtype=proj_dtype))
        region_labels = None
        if regional:
            token_mode = normalize_region_ip_token_mode(regional_token_mode)
            label_mode = normalize_region_ip_label_mode(regional_label_mode)
            labels = build_region_ip_token_labels(
                tissue_mask=tissue_mask,
                num_tokens=int(projected.shape[1]),
                nuclei_mask=nuclei_mask,
                label_mode=label_mode,
            ).to(device=accelerator.device)
            if token_mode == "spatial":
                ref_tokens = projected
                region_labels = labels
            elif token_mode == "stats":
                ref_tokens, region_labels = ref_encoder._stats_by_region_labels(projected, labels)
            else:
                ref_tokens, region_labels = ref_encoder._resample_by_region_labels(projected, labels)
        else:
            ref_tokens = projected if ref_encoder.skip_perceiver else ref_encoder._resample(projected)
        ref_gate = ref_encoder.reference_presence_gate(
            images, device=ref_tokens.device, dtype=ref_tokens.dtype
        )
        ref_tokens = ref_tokens * ref_gate
        projected_ip = transformer.encoder_hid_proj([ref_tokens])[0]
        projected_ip = projected_ip * ref_gate.to(device=projected_ip.device, dtype=projected_ip.dtype)
        zero_token_ip = transformer.encoder_hid_proj([torch.zeros_like(ref_tokens)])[0]
        projected_ip_centered = projected_ip - zero_token_ip.to(
            device=projected_ip.device,
            dtype=projected_ip.dtype,
        )
        result = {
            "images": images,
            "uni": uni_features,
            "projected": projected,
            "ref_tokens": ref_tokens,
            "encoder_hid_proj": projected_ip,
            "encoder_hid_proj_zero_token": zero_token_ip,
            "encoder_hid_proj_centered": projected_ip_centered,
        }
        if region_labels is not None:
            result["region_token_labels"] = region_labels
        return result

    normal = encode_variant(reference_images, reference_tissue_mask, reference_nuclei_mask)
    zero = encode_variant(zero_images, reference_tissue_mask, reference_nuclei_mask)
    real = encode_variant(real_images, real_tissue_mask, real_nuclei_mask)
    logger.info(
        "Reference signal debug step=%s mode=%s regional=%s strict=%s token_mode=%s label_mode=%s "
        "use_soft_bias=%s soft_bias=%s skip_perceiver=%s uni_training=%s uni_dtype=%s "
        "proj_dtype=%s query_tokens=%s",
        step,
        "train" if ref_encoder.training else "eval",
        regional,
        regional_strict,
        normalize_region_ip_token_mode(regional_token_mode),
        normalize_region_ip_label_mode(regional_label_mode),
        bool(use_soft_bias),
        (
            f"{float(soft_bias.detach().float().mean().item()):.6f}"
            if torch.is_tensor(soft_bias)
            else ("none" if soft_bias is None else f"{float(soft_bias):.6f}")
        ),
        bool(ref_encoder.skip_perceiver),
        bool(ref_encoder.uni.training),
        str(uni_dtype).replace("torch.", ""),
        str(next(ref_encoder.proj_mlp.parameters()).dtype).replace("torch.", ""),
        query_token_count,
    )
    encoder_hid_proj_dtype = next(transformer.encoder_hid_proj.parameters()).dtype
    logger.info(
        "Reference signal debug dtype: train_weight_dtype=%s encoder_hid_proj_dtype=%s "
        "ref_tokens_dtype=%s ip_hidden_dtype=%s",
        str(weight_dtype).replace("torch.", ""),
        str(encoder_hid_proj_dtype).replace("torch.", ""),
        str(normal["ref_tokens"].dtype).replace("torch.", ""),
        str(normal["encoder_hid_proj"].dtype).replace("torch.", ""),
    )
    for name in ("images", "uni", "projected", "ref_tokens", "encoder_hid_proj"):
        stats = _tensor_signal_stats(normal[name])
        zero_cosine = _tensor_cosine_against(normal[name], zero[name])
        real_cosine = _tensor_cosine_against(normal[name], real[name])
        logger.info(
            "Reference signal debug %s real_cos=%.5f",
            _format_signal_stats(name, stats, zero_cosine=zero_cosine),
            real_cosine,
        )
        if name in {"projected", "ref_tokens", "encoder_hid_proj"}:
            logger.info("Reference token debug %s", _format_token_variation_stats(name, normal[name]))
            _log_sensitivity_warning(
                stage=name,
                zero_cosine=zero_cosine,
                real_cosine=real_cosine,
            )
    if regional and "region_token_labels" in normal:
        logger.info(
            "Reference region token debug: token_mode=%s label_mode=%s %s",
            normalize_region_ip_token_mode(regional_token_mode),
            normalize_region_ip_label_mode(regional_label_mode),
            _format_region_token_label_summary(normal["region_token_labels"]),
        )

    zero_token_basis = normal["encoder_hid_proj_zero_token"]
    logger.info(
        "Reference signal debug encoder_hid_proj_zero_token: %s normal_cos=%.5f real_cos=%.5f "
        "centered_real_cos=%.5f",
        _format_signal_stats(
            "encoder_hid_proj_zero_token",
            _tensor_signal_stats(zero_token_basis),
            zero_cosine=None,
        ),
        _tensor_cosine_against(normal["encoder_hid_proj"], zero_token_basis),
        _tensor_cosine_against(real["encoder_hid_proj"], real["encoder_hid_proj_zero_token"]),
        _tensor_cosine_against(
            normal["encoder_hid_proj_centered"],
            real["encoder_hid_proj_centered"],
        ),
    )

    if regional and "region_token_labels" in normal and query_token_count is not None:
        label_mode = normalize_region_ip_label_mode(regional_label_mode)
        probe_soft_bias = _ip_soft_bias_value_for_probe(transformer)
        query_region_labels = build_region_ip_token_labels(
            tissue_mask=batch["target_tissue_mask"].to(device=accelerator.device),
            num_tokens=int(query_token_count),
            nuclei_mask=batch["target_nuclei_mask"].to(device=accelerator.device),
            label_mode=label_mode,
        ).to(device=accelerator.device)
        key_fallback_region_labels = _tissue_fallback_region_labels(
            normal["region_token_labels"],
            label_mode=label_mode,
        )
        query_fallback_region_labels = resize_mask_to_token_labels(
            batch["target_tissue_mask"].to(device=accelerator.device),
            int(query_token_count),
        ).to(device=accelerator.device)
        _, _, region_stats = _build_region_attention_mask_and_query_gate(
            query_region_labels=query_region_labels,
            key_region_labels=normal["region_token_labels"],
            query_fallback_labels=query_fallback_region_labels,
            key_fallback_labels=key_fallback_region_labels,
            batch_size=int(reference_images.shape[0]),
            query_len=int(query_token_count),
            key_len=int(normal["region_token_labels"].shape[1]),
            device=accelerator.device,
            dtype=weight_dtype,
            strict=regional_strict,
            soft_bias=probe_soft_bias,
            use_soft_bias=use_soft_bias,
        )
        logger.info(
            "Reference region mask debug: %s",
            _format_region_attention_stats(region_stats),
        )
        missing_fraction = float(region_stats["missing_query_fraction"])
        active_fraction = float(region_stats["active_query_fraction"])
        fallback_fraction = float(region_stats["fallback_query_fraction"])
        allowed_fraction = float(region_stats["allowed_valid_pair_fraction"])
        if regional_strict and active_fraction <= 0.0:
            logger.warning(
                "Reference region mask debug: no query tokens have a matching reference region; "
                "all strict regional queries will attend the learned null IP token."
            )
        elif regional_strict and missing_fraction > 0.2:
            logger.warning(
                "Reference region mask debug: %.1f%% of valid query tokens have no matching "
                "reference region and will attend the learned null IP token.",
                100.0 * missing_fraction,
            )
        if regional_strict and fallback_fraction > 0.5:
            logger.warning(
                "Reference region mask debug: %.1f%% of valid query tokens used legacy fallback; "
                "strict null-token routing should normally keep this at 0.",
                100.0 * fallback_fraction,
            )
        if regional_strict and allowed_fraction > 0.5:
            logger.warning(
                "Reference region mask debug: allowed pair fraction %.3f is very high; "
                "region labels may be too coarse or mostly background.",
                allowed_fraction,
            )

    uni_std = float(_tensor_signal_stats(normal["uni"])["std"])
    projected_std = float(_tensor_signal_stats(normal["projected"])["std"])
    if uni_std < 0.01:
        logger.warning(
            "Reference signal debug: UNI std %.6f is very low; check image range, UNI dtype, and eval state.",
            uni_std,
        )
    if projected_std < 0.01:
        logger.warning(
            "Reference signal debug: projected std %.6f is very low; proj_mlp may have collapsed.",
            projected_std,
        )
    if regional and normalize_region_ip_token_mode(regional_token_mode) == "spatial":
        logger.warning(
            "Reference signal debug: regional spatial mode feeds %s IP tokens directly. "
            "If high IP scale produces outlines, retry with --regional-ip-token-mode perceiver.",
            int(normal["ref_tokens"].shape[1]),
        )


def _summarize_ip_attention_collector(
    collector: dict | None,
    *,
    step: int,
    variant: str,
    max_ratio_warn: float,
    min_ratio_warn: float,
    mask_stats: dict[str, float | int | bool] | None = None,
) -> dict[str, float]:
    logs: dict[str, float] = {}
    safe_variant = str(variant).replace(".", "_")
    if mask_stats is not None:
        logs[f"ip_health_{safe_variant}_region_active_q"] = float(mask_stats["active_query_fraction"])
        logs[f"ip_health_{safe_variant}_region_missing_q"] = float(mask_stats["missing_query_fraction"])
        logs[f"ip_health_{safe_variant}_region_fallback_q"] = float(mask_stats["fallback_query_fraction"])
        logs[f"ip_health_{safe_variant}_region_null_q"] = float(mask_stats["null_query_fraction"])
        logs[f"ip_health_{safe_variant}_region_allowed_pairs"] = float(mask_stats["allowed_valid_pair_fraction"])
        logs[f"ip_health_{safe_variant}_region_allowed_per_q"] = float(
            mask_stats["allowed_tokens_per_query_mean"]
        )
        logger.info(
            "IP attention health step=%s variant=%s region_mask: %s",
            step,
            variant,
            _format_region_attention_stats(mask_stats),
        )
    if collector is None:
        return logs
    records = list(collector.get("records", []))
    if not records:
        logger.warning(
            "IP attention health step=%s variant=%s: no IP attention records were collected.",
            step,
            variant,
        )
        return logs
    ratios = [float(record.get("ratio", math.nan)) for record in records]
    finite_ratios = [value for value in ratios if math.isfinite(value)]
    if not finite_ratios:
        logger.warning(
            "IP attention health step=%s variant=%s: all IP ratios are non-finite.",
            step,
            variant,
        )
        return logs
    ratio_min = min(finite_ratios)
    ratio_max = max(finite_ratios)
    ratio_mean = sum(finite_ratios) / len(finite_ratios)
    logs[f"ip_health_{safe_variant}_ip_ratio_min"] = ratio_min
    logs[f"ip_health_{safe_variant}_ip_ratio_mean"] = ratio_mean
    logs[f"ip_health_{safe_variant}_ip_ratio_max"] = ratio_max
    logger.info(
        "IP attention health step=%s variant=%s: blocks=%s "
        "ratio[min/mean/max]=%.6e/%.6e/%.6e",
        step,
        variant,
        len(records),
        ratio_min,
        ratio_mean,
        ratio_max,
    )
    for record in records:
        logger.info(
            "IP attention health step=%s variant=%s %s: "
            "scale_ip_out_norm=%.6e hidden_norm=%.6e ratio=%.6e",
            step,
            variant,
            record.get("block", "block"),
            float(record.get("ip_norm", math.nan)),
            float(record.get("hidden_norm", math.nan)),
            float(record.get("ratio", math.nan)),
        )
    if ratio_max > max_ratio_warn:
        logger.warning(
            "IP attention health step=%s variant=%s: max ||scale*ip_out||/||hidden|| "
            "%.6e exceeds %.6e; high IP scale may burn/outline the image.",
            step,
            variant,
            ratio_max,
            max_ratio_warn,
        )
    if ratio_max < min_ratio_warn:
        if safe_variant == "zero":
            logger.info(
                "IP attention health step=%s variant=%s: all IP residual ratios are below %.6e "
                "because zero-reference gating intentionally disables the IP branch.",
                step,
                variant,
                min_ratio_warn,
            )
        elif mask_stats is not None and float(mask_stats["active_query_fraction"]) <= 0.0:
            logger.warning(
                "IP attention health step=%s variant=%s: all IP residual ratios are below %.6e "
                "because strict regional masking routed every query to the learned null token. "
                "Check label overlap before treating this as an ignored reference branch.",
                step,
                variant,
                min_ratio_warn,
            )
        elif mask_stats is not None and float(mask_stats["missing_query_fraction"]) > 0.8:
            logger.warning(
                "IP attention health step=%s variant=%s: all IP residual ratios are below %.6e; "
                "%.1f%% of valid query tokens have no matching reference region and route to the learned null token.",
                step,
                variant,
                min_ratio_warn,
                100.0 * float(mask_stats["missing_query_fraction"]),
            )
        else:
            logger.warning(
                "IP attention health step=%s variant=%s: all IP residual ratios are below %.6e; "
                "the transformer may effectively ignore the reference branch.",
                step,
                variant,
                min_ratio_warn,
            )
    return logs


def _collector_first_ip_cosine(left: dict | None, right: dict | None) -> float:
    if left is None or right is None:
        return math.nan
    left_output = left.get("first_ip_output")
    right_output = right.get("first_ip_output")
    if not torch.is_tensor(left_output) or not torch.is_tensor(right_output):
        return math.nan
    return _tensor_cosine_against(left_output, right_output)


def _mix_reference_variant_batch(
    base_batch: dict,
    contrast_batch: dict,
    *,
    swap_image: bool,
    swap_labels: bool,
) -> dict:
    """Swap reference image and/or reference labels while keeping target/noise fixed."""
    variant = dict(base_batch)
    if swap_image:
        variant["reference_image"] = contrast_batch["reference_image"].to(
            device=base_batch["reference_image"].device,
            dtype=base_batch["reference_image"].dtype,
        )
    if swap_labels:
        for key in ("reference_tissue_mask", "reference_nuclei_mask"):
            variant[key] = contrast_batch[key].to(device=base_batch[key].device)
    return variant


def _loss_gap_stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().flatten()
    if values.numel() == 0:
        return {"mean": math.nan, "std": math.nan, "stderr": math.nan, "n": 0.0}
    std = values.std(unbiased=False)
    return {
        "mean": float(values.mean().item()),
        "std": float(std.item()),
        "stderr": float((std / math.sqrt(max(1, int(values.numel())))).item()),
        "n": float(values.numel()),
    }


def _record_loss_gap_stats(
    *,
    logs: dict[str, float],
    safe_variant: str,
    gap_values: torch.Tensor,
    timesteps: torch.Tensor,
) -> dict[str, float]:
    stats = _loss_gap_stats(gap_values)
    logs[f"ip_health_{safe_variant}_loss_gap"] = stats["mean"]
    logs[f"ip_health_{safe_variant}_loss_gap_std"] = stats["std"]
    logs[f"ip_health_{safe_variant}_loss_gap_stderr"] = stats["stderr"]
    logs[f"ip_health_{safe_variant}_loss_gap_n"] = stats["n"]

    t = timesteps.detach().float().flatten()
    flat_gap = gap_values.detach().float().flatten()
    if t.numel() == flat_gap.numel():
        if bool((t > 2.0).any().item()):
            t = t / 1000.0
        for bucket_name, mask in {
            "low": t < (1.0 / 3.0),
            "mid": (t >= (1.0 / 3.0)) & (t < (2.0 / 3.0)),
            "high": t >= (2.0 / 3.0),
        }.items():
            bucket_stats = _loss_gap_stats(flat_gap[mask])
            logs[f"ip_health_{safe_variant}_loss_gap_t_{bucket_name}"] = bucket_stats["mean"]
            logs[f"ip_health_{safe_variant}_loss_gap_t_{bucket_name}_stderr"] = bucket_stats["stderr"]
            logs[f"ip_health_{safe_variant}_loss_gap_t_{bucket_name}_n"] = bucket_stats["n"]
    return stats


@torch.no_grad()
def _run_ip_reference_health_diagnostics(
    *,
    step: int,
    training_batch: dict,
    real_contrast_batch: dict | None,
    modules: dict[str, torch.nn.Module],
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    transformer: FluxTransformer2DModel,
    noisy_model_input: torch.Tensor,
    timesteps: torch.Tensor,
    guidance_vec: torch.Tensor | None,
    batch_pooled: torch.Tensor,
    batch_prompt: torch.Tensor,
    text_ids: torch.Tensor,
    latent_image_ids: torch.Tensor,
    transformer_controlnet_block_samples: list[torch.Tensor] | None,
    transformer_controlnet_single_block_samples: list[torch.Tensor] | None,
    target_velocity: torch.Tensor,
    regional: bool,
    regional_strict: bool,
    regional_token_mode: str,
    regional_label_mode: str,
    use_soft_bias: bool,
    soft_bias: torch.Tensor | float | None,
    query_token_count: int,
    warmup_steps: int,
    min_ref_l2: float,
    min_swap_loss_gap: float,
    max_ip_ratio_warn: float,
    min_ip_ratio_warn: float,
) -> dict[str, float]:
    logs: dict[str, float] = {}
    if not accelerator.is_local_main_process:
        return logs

    normal_ip_collector = {"store_first_ip_output": True}
    probe_soft_bias = _ip_soft_bias_value_for_probe(transformer)
    normal_kwargs = _build_ip_adapter_kwargs(
        training_batch,
        modules,
        accelerator,
        weight_dtype,
        transformer,
        regional=regional,
        query_token_count=query_token_count,
        strict=regional_strict,
        regional_token_mode=regional_token_mode,
        regional_label_mode=regional_label_mode,
        use_soft_bias=use_soft_bias,
        soft_bias=probe_soft_bias,
        ip_debug_collector=normal_ip_collector,
    )
    normal_mask_stats = _regional_ip_mask_stats_from_kwargs(
        normal_kwargs,
        batch_size=int(noisy_model_input.shape[0]),
        query_token_count=query_token_count,
        dtype=weight_dtype,
        device=accelerator.device,
    )
    with _temporary_ip_debug_collector(transformer, normal_ip_collector):
        normal_noise_pred = transformer(
            hidden_states=noisy_model_input,
            timestep=timesteps / 1000,
            guidance=guidance_vec,
            pooled_projections=batch_pooled,
            encoder_hidden_states=batch_prompt,
            controlnet_block_samples=[
                sample.detach() for sample in transformer_controlnet_block_samples
            ] if transformer_controlnet_block_samples is not None else None,
            controlnet_single_block_samples=[
                sample.detach() for sample in transformer_controlnet_single_block_samples
            ] if transformer_controlnet_single_block_samples is not None else None,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=dict(normal_kwargs),
            return_dict=False,
        )[0]
    normal_per_sample_loss = per_sample_mse(normal_noise_pred, target_velocity)
    logs.update(
        _summarize_ip_attention_collector(
            normal_ip_collector,
            step=step,
            variant="normal",
            max_ratio_warn=max_ip_ratio_warn,
            min_ratio_warn=min_ip_ratio_warn,
            mask_stats=normal_mask_stats,
        )
    )
    real_contrast_batch = real_contrast_batch or _alternate_real_reference_batch(training_batch)
    variants = {
        "zero": _use_zero_reference(training_batch),
        "real_feature": _mix_reference_variant_batch(
            training_batch,
            real_contrast_batch,
            swap_image=True,
            swap_labels=False,
        ),
        "real_label": _mix_reference_variant_batch(
            training_batch,
            real_contrast_batch,
            swap_image=False,
            swap_labels=True,
        ),
        "real": real_contrast_batch,
    }
    for variant_name, variant_batch in variants.items():
        collector = {"store_first_ip_output": True}
        variant_kwargs = _build_ip_adapter_kwargs(
            variant_batch,
            modules,
            accelerator,
            weight_dtype,
            transformer,
            regional=regional,
            query_token_count=query_token_count,
            strict=regional_strict,
            regional_token_mode=regional_token_mode,
            regional_label_mode=regional_label_mode,
            use_soft_bias=use_soft_bias,
            soft_bias=probe_soft_bias,
            ip_debug_collector=collector,
        )
        variant_mask_stats = _regional_ip_mask_stats_from_kwargs(
            variant_kwargs,
            batch_size=int(noisy_model_input.shape[0]),
            query_token_count=query_token_count,
            dtype=weight_dtype,
            device=accelerator.device,
        )
        with _temporary_ip_debug_collector(transformer, collector):
            variant_pred = transformer(
                hidden_states=noisy_model_input,
                timestep=timesteps / 1000,
                guidance=guidance_vec,
                pooled_projections=batch_pooled,
                encoder_hidden_states=batch_prompt,
                controlnet_block_samples=[
                    sample.detach() for sample in transformer_controlnet_block_samples
                ] if transformer_controlnet_block_samples is not None else None,
                controlnet_single_block_samples=[
                    sample.detach() for sample in transformer_controlnet_single_block_samples
                ] if transformer_controlnet_single_block_samples is not None else None,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=dict(variant_kwargs),
                return_dict=False,
            )[0]
        variant_loss = per_sample_mse(variant_pred, target_velocity)
        pred_l2 = float(
            torch.sqrt(
                torch.mean((variant_pred.detach().float() - normal_noise_pred.detach().float()) ** 2)
            ).item()
        )
        loss_gap_values = variant_loss.detach() - normal_per_sample_loss.detach()
        ip_output_cosine = _collector_first_ip_cosine(normal_ip_collector, collector)
        safe_variant = variant_name.replace(".", "_")
        gap_stats = _record_loss_gap_stats(
            logs=logs,
            safe_variant=safe_variant,
            gap_values=loss_gap_values,
            timesteps=timesteps,
        )
        loss_gap = float(gap_stats["mean"])
        logs[f"ip_health_{safe_variant}_pred_l2"] = pred_l2
        logs[f"ip_health_{safe_variant}_first_ip_output_cos"] = ip_output_cosine
        logger.info(
            "Reference health step=%s variant=%s: pred_l2=%.6e "
            "loss_gap=%.6e±%.6e n=%d first_double_ip_output_cos=%.6f",
            step,
            variant_name,
            pred_l2,
            loss_gap,
            float(gap_stats["stderr"]),
            int(gap_stats["n"]),
            ip_output_cosine,
        )
        if step >= warmup_steps and pred_l2 <= min_ref_l2:
            logger.warning(
                "Reference health step=%s variant=%s: normal-vs-%s noise_pred L2 %.6e <= %.6e; "
                "reference swaps are still not changing the output.",
                step,
                variant_name,
                variant_name,
                pred_l2,
                min_ref_l2,
            )
        if (
            step >= warmup_steps
            and variant_name == "real"
            and loss_gap <= min_swap_loss_gap
        ):
            logger.warning(
                "Reference health step=%s: paired-ref vs shuffled/alternate-ref loss gap "
                "%.6e <= %.6e; the model is not yet penalizing wrong references.",
                step,
                loss_gap,
                min_swap_loss_gap,
            )
        if math.isfinite(ip_output_cosine) and ip_output_cosine >= 0.99:
            logger.warning(
                "Reference health step=%s variant=%s: first double-block IP output cosine "
                "%.6f >= 0.99; IP attention output is nearly invariant to this reference change.",
                step,
                variant_name,
                ip_output_cosine,
            )
        logs.update(
            _summarize_ip_attention_collector(
                collector,
                step=step,
                variant=variant_name,
                max_ratio_warn=max_ip_ratio_warn,
                min_ratio_warn=min_ip_ratio_warn,
                mask_stats=variant_mask_stats,
            )
        )
    return logs


def _tissue_fallback_region_labels(labels: torch.Tensor, *, label_mode: str) -> torch.Tensor:
    """Map exact IP labels back to tissue labels for strict fallback matching."""
    label_mode = normalize_region_ip_label_mode(label_mode)
    labels = labels.to(dtype=torch.long)
    if label_mode in {"tissue", "coarse_tissue"}:
        return labels
    fallback = torch.full_like(labels, -1)
    valid = labels >= 0
    fallback[valid] = labels[valid] // 256
    return fallback


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


def _apply_cross_v1_mask_augmentation(
    batch: dict,
    *,
    mode: str,
    prob: float,
    translate: float,
    scale: float,
    rotate_degrees: float,
    boundary_jitter: float,
    boundary_grid: int,
    coarse_prob: float,
    coarse_factor: int,
) -> dict:
    mode = str(mode or "none").strip().lower()
    prob = min(1.0, max(0.0, float(prob or 0.0)))
    if mode == "none" or prob <= 0.0:
        return batch
    if mode != "affine_coarse":
        raise ValueError(f"Unsupported mask augmentation mode {mode!r}.")
    if random.random() >= prob:
        return batch

    augmented = dict(batch)
    for tissue_key, nuclei_key in (
        ("target_tissue_mask", "target_nuclei_mask"),
        ("reference_tissue_mask", "reference_nuclei_mask"),
    ):
        if tissue_key not in batch or nuclei_key not in batch:
            continue
        tissue_aug, nuclei_aug = _augment_label_mask_group(
            [batch[tissue_key], batch[nuclei_key]],
            translate=float(translate or 0.0),
            scale=float(scale or 0.0),
            rotate_degrees=float(rotate_degrees or 0.0),
            boundary_jitter=float(boundary_jitter or 0.0),
            boundary_grid=int(boundary_grid or 1),
            coarse_prob=float(coarse_prob or 0.0),
            coarse_factor=int(coarse_factor or 1),
        )
        augmented[tissue_key] = tissue_aug
        augmented[nuclei_key] = nuclei_aug
    return augmented


def _augment_label_mask_batch(
    masks: torch.Tensor,
    *,
    translate: float,
    scale: float,
    rotate_degrees: float,
    boundary_jitter: float,
    boundary_grid: int,
    coarse_prob: float,
    coarse_factor: int,
) -> torch.Tensor:
    return _augment_label_mask_group(
        [masks],
        translate=translate,
        scale=scale,
        rotate_degrees=rotate_degrees,
        boundary_jitter=boundary_jitter,
        boundary_grid=boundary_grid,
        coarse_prob=coarse_prob,
        coarse_factor=coarse_factor,
    )[0]


def _augment_label_mask_group(
    masks: list[torch.Tensor],
    *,
    translate: float,
    scale: float,
    rotate_degrees: float,
    boundary_jitter: float,
    boundary_grid: int,
    coarse_prob: float,
    coarse_factor: int,
) -> list[torch.Tensor]:
    if not masks:
        return []
    first = masks[0]
    if first.ndim != 3:
        raise ValueError(f"Expected mask batch [B,H,W], got {tuple(first.shape)}")
    original_dtypes = [mask.dtype for mask in masks]
    device = first.device
    bsz, height, width = first.shape
    for mask in masks:
        if mask.ndim != 3 or mask.shape != first.shape:
            raise ValueError(
                "All grouped masks must have identical [B,H,W] shape, "
                f"got {tuple(mask.shape)} vs {tuple(first.shape)}"
            )
    masks_float = torch.stack([mask.float() for mask in masks], dim=1)
    theta = masks_float.new_zeros((bsz, 2, 3))
    max_translate = max(0.0, float(translate))
    max_scale = max(0.0, float(scale))
    max_rotate = max(0.0, float(rotate_degrees))
    for index in range(bsz):
        angle = math.radians(random.uniform(-max_rotate, max_rotate)) if max_rotate > 0.0 else 0.0
        zoom = 1.0 + (random.uniform(-max_scale, max_scale) if max_scale > 0.0 else 0.0)
        zoom = max(0.5, zoom)
        cos_a = math.cos(angle) / zoom
        sin_a = math.sin(angle) / zoom
        tx = random.uniform(-max_translate, max_translate) if max_translate > 0.0 else 0.0
        ty = random.uniform(-max_translate, max_translate) if max_translate > 0.0 else 0.0
        theta[index, 0, 0] = cos_a
        theta[index, 0, 1] = -sin_a
        theta[index, 1, 0] = sin_a
        theta[index, 1, 1] = cos_a
        theta[index, 0, 2] = tx
        theta[index, 1, 2] = ty
    grid = F.affine_grid(theta, size=masks_float.shape, align_corners=False)
    jitter_strength = max(0.0, float(boundary_jitter or 0.0))
    jitter_grid = max(2, int(boundary_grid or 2))
    if jitter_strength > 0.0:
        lowres = masks_float.new_empty((bsz, 2, jitter_grid, jitter_grid)).uniform_(
            -jitter_strength,
            jitter_strength,
        )
        displacement = F.interpolate(
            lowres,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        grid = (grid + displacement).clamp(-1.2, 1.2)
    warped = F.grid_sample(
        masks_float,
        grid,
        mode="nearest",
        padding_mode="border",
        align_corners=False,
    )
    coarse_prob = min(1.0, max(0.0, float(coarse_prob or 0.0)))
    coarse_factor = max(1, int(coarse_factor or 1))
    if coarse_factor > 1 and coarse_prob > 0.0 and random.random() < coarse_prob:
        coarse_h = max(1, height // coarse_factor)
        coarse_w = max(1, width // coarse_factor)
        warped = F.interpolate(warped, size=(coarse_h, coarse_w), mode="nearest")
        warped = F.interpolate(warped, size=(height, width), mode="nearest")
    return [
        warped[:, index].round().to(device=device, dtype=original_dtypes[index])
        for index in range(len(masks))
    ]


def _sample_timestep_indices_with_degraded_floor(
    *,
    initial_indices: torch.Tensor,
    degraded_sample_mask: torch.Tensor,
    sigmas_by_index: torch.Tensor,
    min_sigma: float,
    sample_indices: Callable[[int], torch.Tensor],
    max_resample_rounds: int = 8,
) -> torch.Tensor:
    if min_sigma <= 0.0 or not bool(degraded_sample_mask.any().item()):
        return initial_indices

    indices = initial_indices.clone()
    degraded_sample_mask = degraded_sample_mask.to(device=indices.device, dtype=torch.bool)
    sigmas_by_index = sigmas_by_index.to(device=indices.device)
    for _ in range(max(1, int(max_resample_rounds))):
        low_sigma_mask = degraded_sample_mask & (sigmas_by_index[indices].float() < float(min_sigma))
        if not bool(low_sigma_mask.any().item()):
            break
        indices[low_sigma_mask] = sample_indices(int(low_sigma_mask.sum().item())).to(
            device=indices.device,
            dtype=indices.dtype,
        )
    low_sigma_mask = degraded_sample_mask & (sigmas_by_index[indices].float() < float(min_sigma))
    if bool(low_sigma_mask.any().item()):
        valid_indices = torch.nonzero(sigmas_by_index.float() >= float(min_sigma), as_tuple=False).flatten()
        if valid_indices.numel() > 0:
            choice = torch.randint(
                0,
                int(valid_indices.numel()),
                (int(low_sigma_mask.sum().item()),),
                device=indices.device,
            )
            indices[low_sigma_mask] = valid_indices.to(device=indices.device, dtype=indices.dtype)[choice]
    return indices


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
        "clean_image_for_noising": torch.stack(
            [item.get("clean_image_for_noising", item["target_image"]) for item in examples]
        ),
        "reference_image": torch.stack([item["reference_image"] for item in examples]),
        "target_tissue_mask": torch.stack([item["target_tissue_mask"] for item in examples]),
        "target_nuclei_mask": torch.stack([item["target_nuclei_mask"] for item in examples]),
        "reference_tissue_mask": torch.stack([item["reference_tissue_mask"] for item in examples]),
        "reference_nuclei_mask": torch.stack([item["reference_nuclei_mask"] for item in examples]),
        "prompts": [item["prompt"] for item in examples],
        "sample_modes": [item.get("sample_mode", "cross") for item in examples],
        "uses_degraded_noising": torch.tensor(
            [bool(item.get("uses_degraded_noising", False)) for item in examples],
            dtype=torch.bool,
        ),
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
    cross_v1_ip_architecture: str = CROSS_V1_IP_ARCH_GLOBAL,
    regional_ip_token_mode: str = "spatial",
    regional_ip_label_mode: str = "tissue",
    regional_ip_soft_bias_init: float = 4.0,
) -> None:
    state = {
        "cross_v1_spatial_mode": control_spec.spatial_mode,
        "cross_v1_ip_architecture": normalize_cross_v1_ip_architecture(cross_v1_ip_architecture),
        "regional_ip_soft_bias_init": float(regional_ip_soft_bias_init),
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
                "regional_ip_token_mode": normalize_region_ip_token_mode(regional_ip_token_mode),
                "regional_ip_label_mode": normalize_region_ip_label_mode(regional_ip_label_mode),
                "cross_v1_ip_architecture": normalize_cross_v1_ip_architecture(cross_v1_ip_architecture),
                "regional_ip_soft_bias_init": float(regional_ip_soft_bias_init),
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
    cross_v1_ip_architecture: str = CROSS_V1_IP_ARCH_GLOBAL,
    regional_ip_adapter: bool = False,
    regional_ip_token_mode: str = "spatial",
    regional_ip_label_mode: str = "tissue",
    regional_ip_soft_bias_init: float = 4.0,
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
    state["cross_v1_ip_architecture"] = normalize_cross_v1_ip_architecture(cross_v1_ip_architecture)
    state["regional_ip_adapter"] = bool(regional_ip_adapter)
    state["regional_ip_token_mode"] = normalize_region_ip_token_mode(regional_ip_token_mode)
    state["regional_ip_label_mode"] = normalize_region_ip_label_mode(regional_ip_label_mode)
    state["regional_ip_soft_bias_init"] = float(regional_ip_soft_bias_init)
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
    ip_num_tokens: int | None = None,
    cross_v1_ip_architecture: str = CROSS_V1_IP_ARCH_GLOBAL,
    regional_ip_adapter: bool = False,
    regional_ip_token_mode: str = "spatial",
    regional_ip_label_mode: str = "tissue",
    regional_ip_soft_bias_init: float = 4.0,
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
        cross_v1_ip_architecture=cross_v1_ip_architecture,
        regional_ip_token_mode=regional_ip_token_mode,
        regional_ip_label_mode=regional_ip_label_mode,
        regional_ip_soft_bias_init=regional_ip_soft_bias_init,
    )
    _save_ip_adapter_modules(
        output_dir,
        ip_trainable_wrapper,
        unwrap_model,
        save_dtype,
        num_tokens=int(ip_num_tokens or args.reference_num_tokens),
        ip_init_gain=args.ip_init_gain,
        cross_v1_ip_architecture=cross_v1_ip_architecture,
        regional_ip_adapter=regional_ip_adapter,
        regional_ip_token_mode=regional_ip_token_mode,
        regional_ip_label_mode=regional_ip_label_mode,
        regional_ip_soft_bias_init=regional_ip_soft_bias_init,
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
    state_path = _resolve_phase5_conditioning_state_path(checkpoint_path)

    state = _torch_load_weights(state_path)
    for name in ("hte", "tissue_downsampler", "nuclei_encoder"):
        if name not in state:
            raise KeyError(f"Missing {name!r} in conditioning checkpoint: {state_path}")
        modules[name].load_state_dict(state[name])

    if load_ref_encoder:
        ref_encoder = modules["ref_encoder"]
        ref_encoder.proj_mlp.load_state_dict(state["ref_encoder_proj_mlp"])
        if not bool(getattr(ref_encoder, "skip_perceiver", False)):
            perceiver_keys = (
                "ref_encoder_perceiver_layers",
                "ref_encoder_latent_queries",
                "ref_encoder_perceiver_norm",
            )
            if all(key in state for key in perceiver_keys):
                ref_encoder.load_perceiver_layers_state_dict(state["ref_encoder_perceiver_layers"])
                ref_encoder.latent_queries.data.copy_(
                    state["ref_encoder_latent_queries"].to(ref_encoder.latent_queries.device)
                )
                ref_encoder.perceiver_norm.load_state_dict(state["ref_encoder_perceiver_norm"])
            else:
                logger.warning(
                    "Conditioning checkpoint %s does not contain reference Perceiver "
                    "weights; keeping the newly initialized Perceiver trainable.",
                    state_path,
                )


def _resolve_phase5_conditioning_state_path(checkpoint_path: str | Path) -> Path:
    checkpoint = Path(checkpoint_path)
    candidates = []
    if checkpoint.is_dir():
        candidates.append(checkpoint / "phase5_conditioning.pt")
        if checkpoint.name.startswith("checkpoint-"):
            candidates.append(checkpoint.parent / "phase5_conditioning.pt")
    else:
        candidates.append(checkpoint)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Missing phase5_conditioning.pt for A1-lite. Tried: {tried}")


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


def _saved_ip_architecture_from_state(state: dict[str, object]) -> str:
    if "cross_v1_ip_architecture" in state:
        return normalize_cross_v1_ip_architecture(str(state.get("cross_v1_ip_architecture")))
    return normalize_cross_v1_ip_architecture(
        None,
        regional_ip_adapter=bool(state.get("regional_ip_adapter", False)),
    )


def _load_ip_adapter_modules_from_checkpoint(
    transformer: FluxTransformer2DModel,
    checkpoint_path: str | Path,
    *,
    load_single_ip: bool = False,
    expected_regional_ip_adapter: bool | None = None,
    expected_cross_v1_ip_architecture: str | None = None,
) -> Path:
    checkpoint = Path(checkpoint_path)
    state_path = checkpoint / "phase5_ip_adapter.pt" if checkpoint.is_dir() else checkpoint
    if not state_path.exists():
        raise FileNotFoundError(f"Missing phase5_ip_adapter.pt: {state_path}")

    state = _torch_load_weights(state_path)
    if expected_regional_ip_adapter is not None and bool(state.get("regional_ip_adapter", False)) != bool(expected_regional_ip_adapter):
        raise RuntimeError(
            "IP-Adapter checkpoint architecture mismatch: "
            f"{state_path} has regional_ip_adapter={bool(state.get('regional_ip_adapter', False))}, "
            f"expected {bool(expected_regional_ip_adapter)}. Refusing to cold-start silently."
        )
    saved_architecture = _saved_ip_architecture_from_state(state)
    if expected_cross_v1_ip_architecture is not None:
        expected_architecture = normalize_cross_v1_ip_architecture(expected_cross_v1_ip_architecture)
        if saved_architecture != expected_architecture:
            raise RuntimeError(
                "IP-Adapter checkpoint architecture mismatch: "
                f"{state_path} has cross_v1_ip_architecture={saved_architecture!r}, "
                f"expected {expected_architecture!r}. Refusing to cold-start silently."
            )
    transformer.encoder_hid_proj.load_state_dict(state["encoder_hid_proj"])
    loaded_double = 0
    loaded_double_null = 0
    loaded_double_soft_bias = 0
    missing_soft_bias: list[str] = []
    for i, block in enumerate(transformer.transformer_blocks):
        k_key = f"block_{i}_to_k_ip"
        v_key = f"block_{i}_to_v_ip"
        null_key = f"block_{i}_ip_null_tokens"
        bias_key = f"block_{i}_ip_soft_bias"
        if k_key not in state or v_key not in state:
            continue
        block.attn.processor.to_k_ip.load_state_dict(state[k_key])
        block.attn.processor.to_v_ip.load_state_dict(state[v_key])
        if null_key in state and hasattr(block.attn.processor, "ip_null_tokens"):
            block.attn.processor.ip_null_tokens.load_state_dict(state[null_key])
            loaded_double_null += 1
        if hasattr(block.attn.processor, "ip_soft_bias"):
            if bias_key in state:
                block.attn.processor.ip_soft_bias.load_state_dict(state[bias_key])
                loaded_double_soft_bias += 1
            elif saved_architecture == CROSS_V1_IP_ARCH_GLOBAL_SOFT_BIAS:
                missing_soft_bias.append(bias_key)
        loaded_double += 1

    loaded_single = 0
    loaded_single_null = 0
    loaded_single_soft_bias = 0
    if load_single_ip:
        for i, block in enumerate(getattr(transformer, "single_transformer_blocks", [])):
            k_key = f"single_block_{i}_to_k_ip"
            v_key = f"single_block_{i}_to_v_ip"
            null_key = f"single_block_{i}_ip_null_tokens"
            bias_key = f"single_block_{i}_ip_soft_bias"
            if k_key not in state or v_key not in state:
                continue
            block.attn.processor.to_k_ip.load_state_dict(state[k_key])
            block.attn.processor.to_v_ip.load_state_dict(state[v_key])
            if null_key in state and hasattr(block.attn.processor, "ip_null_tokens"):
                block.attn.processor.ip_null_tokens.load_state_dict(state[null_key])
                loaded_single_null += 1
            if hasattr(block.attn.processor, "ip_soft_bias"):
                if bias_key in state:
                    block.attn.processor.ip_soft_bias.load_state_dict(state[bias_key])
                    loaded_single_soft_bias += 1
                elif saved_architecture == CROSS_V1_IP_ARCH_GLOBAL_SOFT_BIAS:
                    missing_soft_bias.append(bias_key)
            loaded_single += 1
    if missing_soft_bias:
        raise RuntimeError(
            "IP-Adapter checkpoint is marked global_soft_bias but is missing soft-bias weights: "
            f"{missing_soft_bias[:8]}{'...' if len(missing_soft_bias) > 8 else ''}"
        )

    logger.info(
        "Loaded IP-Adapter checkpoint %s: architecture=%s double_blocks=%s double_null=%s "
        "double_soft_bias=%s single_blocks=%s single_null=%s single_soft_bias=%s",
        state_path,
        saved_architecture,
        loaded_double,
        loaded_double_null,
        loaded_double_soft_bias,
        loaded_single,
        loaded_single_null,
        loaded_single_soft_bias,
    )
    return state_path


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
        noising_degradation=getattr(args, "noising_degradation", "none"),
        texture_blur_prob=float(getattr(args, "texture_blur_prob", 0.7)),
        texture_blur_sigma_min=float(getattr(args, "texture_blur_sigma_min", 0.4)),
        texture_blur_sigma_max=float(getattr(args, "texture_blur_sigma_max", 1.4)),
        texture_downsample_prob=float(getattr(args, "texture_downsample_prob", 0.7)),
        texture_downsample_scale_min=float(getattr(args, "texture_downsample_scale_min", 0.35)),
        texture_downsample_scale_max=float(getattr(args, "texture_downsample_scale_max", 0.75)),
        texture_noise_prob=float(getattr(args, "texture_noise_prob", 0.35)),
        texture_noise_std_min=float(getattr(args, "texture_noise_std_min", 0.005)),
        texture_noise_std_max=float(getattr(args, "texture_noise_std_max", 0.03)),
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
    regional_ip_adapter = bool(getattr(args, "regional_ip_adapter", False))
    regional_ip_strict = bool(getattr(args, "regional_ip_strict", True))
    cross_v1_ip_architecture = normalize_cross_v1_ip_architecture(
        getattr(args, "cross_v1_ip_architecture", None),
        regional_ip_adapter=regional_ip_adapter,
    )
    regional_ip_token_mode = normalize_region_ip_token_mode(
        getattr(args, "regional_ip_token_mode", "spatial")
    )
    regional_ip_label_mode = normalize_region_ip_label_mode(
        getattr(args, "regional_ip_label_mode", "tissue")
    )
    regional_ip_soft_bias_init = float(
        getattr(args, "regional_ip_soft_bias_init", 4.0) or 0.0
    )
    if cross_v1_ip_architecture == CROSS_V1_IP_ARCH_GLOBAL_SOFT_BIAS:
        regional_ip_adapter = True
        regional_ip_strict = False
        regional_ip_token_mode = "stats"
        regional_ip_label_mode = "coarse_tissue"
    elif cross_v1_ip_architecture == CROSS_V1_IP_ARCH_REGIONAL_HARD:
        regional_ip_adapter = True
    elif cross_v1_ip_architecture == CROSS_V1_IP_ARCH_GLOBAL:
        regional_ip_adapter = False
        regional_ip_strict = False
    regional_ip_use_soft_bias = _uses_soft_region_bias(cross_v1_ip_architecture)
    args.cross_v1_ip_architecture = cross_v1_ip_architecture
    args.regional_ip_adapter = regional_ip_adapter
    args.regional_ip_strict = regional_ip_strict
    args.regional_ip_token_mode = regional_ip_token_mode
    args.regional_ip_label_mode = regional_ip_label_mode
    args.regional_ip_soft_bias_init = regional_ip_soft_bias_init
    degraded_noising_min_sigma = max(
        0.0,
        float(getattr(args, "degraded_noising_min_sigma", 0.1) or 0.0),
    )
    reference_region_loss_weight = max(
        0.0,
        float(getattr(args, "reference_region_loss_weight", 0.0) or 0.0),
    )
    reference_region_loss_interval = int(
        getattr(args, "reference_region_loss_interval", 1) or 0
    )
    reference_region_loss_min_sigma = max(
        0.0,
        float(getattr(args, "reference_region_loss_min_sigma", 0.0) or 0.0),
    )
    reference_region_loss_max_sigma = float(
        getattr(args, "reference_region_loss_max_sigma", 0.6)
    )
    if reference_region_loss_max_sigma < reference_region_loss_min_sigma:
        raise ValueError(
            "--reference-region-loss-max-sigma must be >= "
            "--reference-region-loss-min-sigma."
        )
    args.reference_region_loss_min_sigma = reference_region_loss_min_sigma
    args.reference_region_loss_max_sigma = reference_region_loss_max_sigma
    reference_region_loss_backend = normalize_reference_region_loss_backend(
        getattr(args, "reference_region_loss_backend", REFERENCE_REGION_LOSS_BACKEND_UNI)
    )
    args.reference_region_loss_backend = reference_region_loss_backend
    if reference_region_loss_backend == REFERENCE_REGION_LOSS_BACKEND_UNI:
        reference_region_loss_config = RegionalFeatureLossConfig(
            tissue_weight=float(getattr(args, "reference_region_tissue_weight", 1.0) or 0.0),
            nuclei_weight=float(getattr(args, "reference_region_nuclei_weight", 0.0) or 0.0),
            composite_weight=float(getattr(args, "reference_region_composite_weight", 0.0) or 0.0),
            mean_weight=float(getattr(args, "reference_region_mean_weight", 1.0) or 0.0),
            std_weight=float(getattr(args, "reference_region_std_weight", 0.5) or 0.0),
            pooled_cosine_weight=float(getattr(args, "reference_region_cosine_weight", 0.25) or 0.0),
            min_tokens=max(1, int(getattr(args, "reference_region_min_tokens", 2) or 1)),
            max_regions_per_sample=getattr(args, "reference_region_max_regions_per_sample", None),
        )
    else:
        reference_region_loss_config = RegionalRgbFftLossConfig(
            tissue_weight=float(getattr(args, "reference_region_tissue_weight", 1.0) or 0.0),
            nuclei_weight=float(getattr(args, "reference_region_nuclei_weight", 0.0) or 0.0),
            composite_weight=float(getattr(args, "reference_region_composite_weight", 0.0) or 0.0),
            mean_weight=float(getattr(args, "reference_region_mean_weight", 1.0) or 0.0),
            std_weight=float(getattr(args, "reference_region_std_weight", 0.5) or 0.0),
            fft_weight=float(getattr(args, "reference_region_fft_weight", 0.25) or 0.0),
            fft_bins=max(1, int(getattr(args, "reference_region_fft_bins", 6) or 1)),
            fft_size=max(4, int(getattr(args, "reference_region_fft_size", 64) or 4)),
            min_pixels=max(1, int(getattr(args, "reference_region_min_pixels", 32) or 1)),
            max_regions_per_sample=getattr(args, "reference_region_max_regions_per_sample", None),
        )
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
    ip_health_debug_interval = max(0, int(getattr(args, "ip_health_debug_interval", 0) or 0))
    mask_augmentation_mode = str(getattr(args, "mask_augmentation", "none") or "none").strip().lower()
    mask_augment_prob = min(1.0, max(0.0, float(getattr(args, "mask_augment_prob", 0.0) or 0.0)))
    mask_augment_translate = max(0.0, float(getattr(args, "mask_augment_translate", 0.0) or 0.0))
    mask_augment_scale = max(0.0, float(getattr(args, "mask_augment_scale", 0.0) or 0.0))
    mask_augment_rotate_degrees = max(
        0.0,
        float(getattr(args, "mask_augment_rotate_degrees", 0.0) or 0.0),
    )
    mask_augment_boundary_jitter = max(
        0.0,
        float(getattr(args, "mask_augment_boundary_jitter", 0.0) or 0.0),
    )
    mask_augment_boundary_grid = max(2, int(getattr(args, "mask_augment_boundary_grid", 8) or 8))
    mask_augment_coarse_prob = min(
        1.0,
        max(0.0, float(getattr(args, "mask_augment_coarse_prob", 0.0) or 0.0)),
    )
    mask_augment_coarse_factor = max(1, int(getattr(args, "mask_augment_coarse_factor", 1) or 1))
    random_reference_sampler = (
        RandomReferenceSampler(dataset.records, seed=args.seed)
        if (("random" in ref_swap_variants) or ip_health_debug_interval > 0)
        and args.train_batch_size <= 1
        else None
    )

    skip_reference_perceiver = bool(getattr(args, "skip_reference_perceiver", False))
    if regional_ip_adapter and regional_ip_token_mode in {"spatial", "stats"}:
        skip_reference_perceiver = True
    elif regional_ip_adapter and regional_ip_token_mode == "perceiver":
        skip_reference_perceiver = False

    ref_encoder = ReferenceImageEncoder(
        uni_checkpoint_path=args.uni_checkpoint_path,
        num_tokens=args.reference_num_tokens,
        num_perceiver_layers=args.reference_num_perceiver_layers,
        perceiver_heads=args.reference_perceiver_heads,
        use_perceiver_self_attn=not bool(
            getattr(args, "disable_reference_perceiver_self_attn", False)
        ),
        perceiver_cross_gate_init=getattr(args, "reference_perceiver_cross_gate_init", None),
        skip_perceiver=skip_reference_perceiver,
    )
    ip_num_tokens = (
        ref_encoder.num_spatial_tokens
        if regional_ip_adapter and regional_ip_token_mode == "spatial"
        else ref_encoder.num_output_tokens
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
    if regional_ip_adapter:
        logger.info(
            "Using mask-guided regional IP-Adapter: tokens=%s strict=%s token_mode=%s label_mode=%s",
            ip_num_tokens,
            regional_ip_strict,
            regional_ip_token_mode,
            regional_ip_label_mode,
        )
    if mask_augmentation_mode != "none" and mask_augment_prob > 0.0:
        logger.info(
            "Using mask augmentation: mode=%s prob=%s translate=%s scale=%s rotate_deg=%s boundary_jitter=%s boundary_grid=%s coarse_prob=%s coarse_factor=%s",
            mask_augmentation_mode,
            mask_augment_prob,
            mask_augment_translate,
            mask_augment_scale,
            mask_augment_rotate_degrees,
            mask_augment_boundary_jitter,
            mask_augment_boundary_grid,
            mask_augment_coarse_prob,
            mask_augment_coarse_factor,
        )
    if getattr(args, "noising_degradation", "none") != "none":
        logger.info(
            "Using degraded target as noising source: mode=%s min_sigma=%s",
            getattr(args, "noising_degradation", "none"),
            degraded_noising_min_sigma,
        )
    if reference_region_loss_weight > 0.0:
        if reference_region_loss_backend == REFERENCE_REGION_LOSS_BACKEND_UNI:
            logger.info(
                "Using decode->frozen-UNI reference region loss: prediction RGB is VAE-decoded before UNI; "
                "reference RGB is encoded by frozen UNI; region stats weight=%s interval=%s sigma=[%s,%s] "
                "tissue=%s nuclei=%s composite=%s mean/std/cos=%s/%s/%s",
                reference_region_loss_weight,
                reference_region_loss_interval,
                reference_region_loss_min_sigma,
                reference_region_loss_max_sigma,
                reference_region_loss_config.tissue_weight,
                reference_region_loss_config.nuclei_weight,
                reference_region_loss_config.composite_weight,
                reference_region_loss_config.mean_weight,
                reference_region_loss_config.std_weight,
                reference_region_loss_config.pooled_cosine_weight,
            )
        else:
            logger.info(
                "Using independent RGB+FFT reference region loss: weight=%s interval=%s sigma=[%s,%s] tissue=%s nuclei=%s composite=%s mean/std/fft=%s/%s/%s fft_bins=%s fft_size=%s",
                reference_region_loss_weight,
                reference_region_loss_interval,
                reference_region_loss_min_sigma,
                reference_region_loss_max_sigma,
                reference_region_loss_config.tissue_weight,
                reference_region_loss_config.nuclei_weight,
                reference_region_loss_config.composite_weight,
                reference_region_loss_config.mean_weight,
                reference_region_loss_config.std_weight,
                reference_region_loss_config.fft_weight,
                reference_region_loss_config.fft_bins,
                reference_region_loss_config.fft_size,
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
    logger.info(
        "Using Cross V1 IP architecture %s: regional=%s strict=%s token_mode=%s "
        "label_mode=%s use_soft_bias=%s soft_bias_init=%s",
        cross_v1_ip_architecture,
        regional_ip_adapter,
        regional_ip_strict,
        regional_ip_token_mode,
        regional_ip_label_mode,
        regional_ip_use_soft_bias,
        regional_ip_soft_bias_init,
    )
    logger.info(
        "Loss configuration: denoise=1 perceptual=%s region=%s style=%s "
        "ref_swap=%s self_recon_l1=%s",
        perceptual_loss_weight,
        reference_region_loss_weight,
        reference_style_loss_weight,
        ref_swap_loss_weight,
        self_reconstruction_l1_weight,
    )
    if cross_v1_ip_architecture == CROSS_V1_IP_ARCH_GLOBAL_SOFT_BIAS and any(
        value > 0.0
        for value in (
            perceptual_loss_weight,
            reference_region_loss_weight,
            reference_style_loss_weight,
            ref_swap_loss_weight,
            self_reconstruction_l1_weight,
        )
    ):
        logger.info(
            "Cross V1 global_soft_bias has auxiliary losses enabled; this run is no longer "
            "a pure-denoise attribution probe."
        )
    if ref_encoder.perceiver_cross_gate_init is not None:
        logger.info(
            "Reference Perceiver uses cross-output-only mode; latent queries are not "
            "residual output content. cross_out_scale_init=%s (scale=%s)",
            ref_encoder.perceiver_cross_gate_init,
            1.0 - torch.sigmoid(torch.tensor(ref_encoder.perceiver_cross_gate_init)).item(),
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
        num_tokens=ip_num_tokens,
        ip_init_gain=args.ip_init_gain,
        num_single_layers=max(0, int(getattr(args, "ip_single_num_layers", 0) or 0)),
        regional=regional_ip_adapter,
        use_soft_bias=regional_ip_use_soft_bias,
        soft_bias_init=regional_ip_soft_bias_init,
    )
    patch_flux_single_ip_forward(flux_transformer)
    ip_adapter_checkpoint = _resolve_ip_adapter_checkpoint_path(args)
    loaded_ip_adapter_checkpoint: Path | None = None
    if ip_adapter_checkpoint is not None:
        loaded_ip_adapter_checkpoint = _load_ip_adapter_modules_from_checkpoint(
            flux_transformer,
            ip_adapter_checkpoint,
            load_single_ip=bool(getattr(args, "load_single_ip_from_checkpoint", False)),
            expected_regional_ip_adapter=regional_ip_adapter,
            expected_cross_v1_ip_architecture=cross_v1_ip_architecture,
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
    flux_transformer.requires_grad_(False)
    _move_ip_adapter_modules(
        ip_adapter_modules,
        device=accelerator.device,
        train_dtype=torch.float32,
    )

    vae.to(accelerator.device, dtype=weight_dtype)
    vae.eval()
    vae.requires_grad_(False)
    flux_controlnet.to(accelerator.device, dtype=weight_dtype)
    if a1_lite:
        flux_controlnet.eval()
        flux_controlnet.requires_grad_(False)
        controlnet_trainable_names: list[str] = []
        for name, module in modules.items():
            if name == "ref_encoder":
                _move_reference_encoder(
                    module,
                    device=accelerator.device,
                    train_dtype=torch.float32,
                )
                module.train()
            else:
                module.to(accelerator.device, dtype=weight_dtype)
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
        for name, module in modules.items():
            if name == "ref_encoder":
                _move_reference_encoder(
                    module,
                    device=accelerator.device,
                    train_dtype=torch.float32,
                )
            else:
                module.to(accelerator.device, dtype=weight_dtype)
            module.train()
    # UNI2-h backbone inside ref_encoder stays frozen and fp32.
    modules["ref_encoder"].uni.to(device=accelerator.device, dtype=torch.float32)
    modules["ref_encoder"]._lock_uni_backbone()

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
    _move_reference_encoder(
        ref_encoder_raw,
        device=accelerator.device,
        train_dtype=torch.float32,
    )

    ref_trainable_wrapper = RefEncoderTrainableWrapper(ref_encoder_raw)

    # --- FIX 1: IP-Adapter 可训练部分包成 wrapper ---
    ip_trainable_wrapper = IPAdapterTrainableWrapper(ip_adapter_modules)

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
    configured_group_lrs = [float(group["lr"]) for group in optimizer_param_groups]
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

    lr_training_steps = (
        (args.max_train_steps or args.num_train_epochs * num_update_steps_per_epoch)
        * (1 if str(getattr(args, "lr_scheduler", "")) == "cosine_with_min_lr" else accelerator.num_processes)
    )
    lr_warmup_steps = args.lr_warmup_steps * (
        1 if str(getattr(args, "lr_scheduler", "")) == "cosine_with_min_lr" else accelerator.num_processes
    )
    lr_scheduler = _build_lr_scheduler(
        args,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=lr_training_steps,
    )

    # ---- accelerator.prepare ----
    # Only trainable wrappers go through DDP in A1-lite; frozen ControlNet and
    # frozen spatial conditioning stay as ordinary modules on the accelerator.
    trainable_modules_to_prepare = (
        [ref_trainable_wrapper, ip_trainable_wrapper]
        if a1_lite
        else [
            flux_controlnet,
            *modules.values(),
            ref_trainable_wrapper,
            ip_trainable_wrapper,
        ]
    )
    n_cond_modules = len(modules) if not a1_lite else 0
    all_to_prepare = [*trainable_modules_to_prepare]
    prepared = accelerator.prepare(
        *all_to_prepare, optimizer, train_dataloader, lr_scheduler,
    )
    n_models = len(all_to_prepare)

    if a1_lite:
        ref_trainable_wrapper = prepared[0]
        ip_trainable_wrapper = prepared[1]
    else:
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

    ema_decay = float(getattr(args, "ema_decay", 0.0) or 0.0)
    trainable_ema = None

    def create_or_load_trainable_ema(checkpoint_path: str | None) -> TrainableEMA | None:
        if ema_decay <= 0.0:
            return None
        ema = TrainableEMA(
            [unwrap_model(ref_trainable_wrapper), unwrap_model(ip_trainable_wrapper)],
            decay=ema_decay,
            device=str(getattr(args, "ema_device", "cpu") or "cpu"),
        )
        ema_state_path = Path(args.output_dir) / "ema_state.pt"
        if checkpoint_path:
            candidate = Path(checkpoint_path) / "ema_state.pt"
            if candidate.exists():
                ema_state_path = candidate
        if ema_state_path.exists():
            try:
                ema.load_state_dict(torch.load(ema_state_path, map_location="cpu"))
                logger.info("Loaded EMA state from %s", ema_state_path)
            except Exception as exc:
                logger.warning("Could not load EMA state from %s: %s", ema_state_path, exc)
        logger.info(
            "Using trainable EMA: decay=%s device=%s shadow_tensors=%s",
            ema.decay,
            ema.device,
            len(ema.shadow),
        )
        return ema

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
    loaded_checkpoint_path = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            checkpoint_path = _latest_checkpoint(args.output_dir)
        else:
            resume_path = Path(str(args.resume_from_checkpoint))
            checkpoint_path = str(resume_path if resume_path.is_absolute() else Path(args.output_dir) / resume_path)
        if checkpoint_path is not None:
            accelerator.load_state(checkpoint_path)
            loaded_checkpoint_path = checkpoint_path
            global_step = int(Path(checkpoint_path).name.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            _set_optimizer_group_lrs(optimizer, configured_group_lrs)
            lr_scheduler = _build_lr_scheduler(
                args,
                optimizer=optimizer,
                num_warmup_steps=lr_warmup_steps,
                num_training_steps=lr_training_steps,
            )
            _advance_scheduler_to_step(
                lr_scheduler,
                global_step if str(getattr(args, "lr_scheduler", "")) == "cosine_with_min_lr"
                else global_step * accelerator.num_processes,
            )
            logger.info(
                "Resumed from %s at global_step=%s; reset optimizer LRs to %s and rebuilt %s scheduler.",
                checkpoint_path,
                global_step,
                configured_group_lrs,
                args.lr_scheduler,
            )
    unwrap_model(ref_trainable_wrapper).sync_back(ref_encoder_raw)
    modules["ref_encoder"] = ref_encoder_raw
    _sync_ip_adapter_to_transformer(unwrap_model(ip_trainable_wrapper), flux_transformer)
    _log_cross_v1_step0_adapter_assert(
        accelerator=accelerator,
        ref_trainable_wrapper=ref_trainable_wrapper,
        ip_trainable_wrapper=ip_trainable_wrapper,
        transformer=flux_transformer,
        architecture=cross_v1_ip_architecture,
        regional_ip_adapter=regional_ip_adapter,
        regional_ip_strict=regional_ip_strict,
        regional_ip_token_mode=regional_ip_token_mode,
        regional_ip_label_mode=regional_ip_label_mode,
        use_soft_bias=regional_ip_use_soft_bias,
        soft_bias_init=regional_ip_soft_bias_init,
        loaded_ip_adapter_checkpoint=loaded_ip_adapter_checkpoint,
        loaded_resume_checkpoint=loaded_checkpoint_path,
    )
    trainable_ema = create_or_load_trainable_ema(loaded_checkpoint_path)

    progress_bar = tqdm(
        total=args.max_train_steps,
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    ip_health_debug_warmup_steps = max(
        1,
        int(getattr(args, "ip_health_debug_warmup_steps", 100) or 100),
    )
    ip_health_min_ref_l2 = max(0.0, float(getattr(args, "ip_health_min_ref_l2", 1e-6) or 0.0))
    ip_health_min_swap_loss_gap = float(getattr(args, "ip_health_min_swap_loss_gap", 0.0) or 0.0)
    ip_health_max_ip_ratio = max(0.0, float(getattr(args, "ip_health_max_ip_ratio", 1.0) or 0.0))
    ip_health_min_ip_ratio = max(0.0, float(getattr(args, "ip_health_min_ip_ratio", 1e-8) or 0.0))
    ip_health_monitor = IPTrainableHealthMonitor(
        ref_trainable_wrapper=ref_trainable_wrapper,
        ip_trainable_wrapper=ip_trainable_wrapper,
        accelerator=accelerator,
        warmup_steps=ip_health_debug_warmup_steps,
    )
    reference_signal_debug_logged = False
    gradient_flow_debug_logged = False

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
                health_step = global_step + 1
                should_run_ip_health = (
                    ip_health_debug_interval > 0
                    and accelerator.sync_gradients
                    and (
                        global_step == 0
                        or health_step % ip_health_debug_interval == 0
                    )
                )
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
                training_batch = _apply_cross_v1_mask_augmentation(
                    training_batch,
                    mode=mask_augmentation_mode,
                    prob=mask_augment_prob,
                    translate=mask_augment_translate,
                    scale=mask_augment_scale,
                    rotate_degrees=mask_augment_rotate_degrees,
                    boundary_jitter=mask_augment_boundary_jitter,
                    boundary_grid=mask_augment_boundary_grid,
                    coarse_prob=mask_augment_coarse_prob,
                    coarse_factor=mask_augment_coarse_factor,
                )
                health_real_contrast_batch = None
                if should_run_ip_health:
                    random_batch = (
                        random_reference_sampler.sample_for_batch(
                            training_batch,
                            device=accelerator.device,
                        )
                        if random_reference_sampler is not None
                        and int(training_batch["reference_image"].shape[0]) <= 1
                        else None
                    )
                    health_real_contrast_batch = _alternate_real_reference_batch(
                        training_batch,
                        random_batch=random_batch,
                    )
                counterfactual_sample_mask = _batch_mode_mask(
                    training_batch,
                    "counterfactual",
                    device=accelerator.device,
                )
                appearance_degraded_sample_mask = _batch_mode_mask(
                    training_batch,
                    "appearance_degraded",
                    device=accelerator.device,
                )
                cross_sample_mask = ~(counterfactual_sample_mask | self_reconstruction_sample_mask)

                with torch.no_grad() if a1_lite else contextlib.nullcontext():
                    pixel_latents, noising_latents, control_tensor = _build_cross_v1_control_batch(
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
                packed_noising_latents = FluxControlNetPipeline._pack_latents(
                    noising_latents, bsz, noising_latents.shape[1],
                    noising_latents.shape[2], noising_latents.shape[3],
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
                degraded_noising_mask = training_batch.get("uses_degraded_noising")
                if degraded_noising_mask is None:
                    degraded_noising_mask = torch.zeros(bsz, device=accelerator.device, dtype=torch.bool)
                else:
                    degraded_noising_mask = degraded_noising_mask.to(device=accelerator.device, dtype=torch.bool)

                def _resample_indices(count: int) -> torch.Tensor:
                    new_u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme, batch_size=count,
                        logit_mean=args.logit_mean, logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    return (new_u * noise_scheduler_copy.config.num_train_timesteps).long()

                indices = _sample_timestep_indices_with_degraded_floor(
                    initial_indices=indices,
                    degraded_sample_mask=degraded_noising_mask,
                    sigmas_by_index=noise_scheduler_copy.sigmas,
                    min_sigma=degraded_noising_min_sigma,
                    sample_indices=_resample_indices,
                )
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=packed_pixel_latents.device)
                sigmas = get_sigmas(timesteps, n_dim=packed_pixel_latents.ndim, dtype=packed_pixel_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * packed_noising_latents + sigmas * noise

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
                    training_batch,
                    modules,
                    accelerator,
                    weight_dtype,
                    flux_transformer,
                    regional=regional_ip_adapter,
                    query_token_count=noisy_model_input.shape[1],
                    strict=regional_ip_strict,
                    regional_token_mode=regional_ip_token_mode,
                    regional_label_mode=regional_ip_label_mode,
                    use_soft_bias=regional_ip_use_soft_bias,
                )
                if not reference_signal_debug_logged and not should_run_ip_health:
                    _log_reference_signal_debug(
                        batch=training_batch,
                        modules=modules,
                        accelerator=accelerator,
                        weight_dtype=weight_dtype,
                        transformer=flux_transformer,
                        regional=regional_ip_adapter,
                        regional_strict=regional_ip_strict,
                        regional_token_mode=regional_ip_token_mode,
                        regional_label_mode=regional_ip_label_mode,
                        use_soft_bias=regional_ip_use_soft_bias,
                        soft_bias=None,
                        query_token_count=noisy_model_input.shape[1],
                        step=health_step,
                        real_contrast_batch=health_real_contrast_batch,
                    )
                    reference_signal_debug_logged = True
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

                target_velocity = (noisy_model_input - packed_pixel_latents) / sigmas.clamp_min(1e-6)
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
                reference_region_loss = noise_pred.new_zeros(())
                reference_region_tissue_loss = noise_pred.new_zeros(())
                reference_region_nuclei_loss = noise_pred.new_zeros(())
                reference_region_composite_loss = noise_pred.new_zeros(())
                reference_region_tissue_regions = 0
                reference_region_nuclei_regions = 0
                reference_region_composite_regions = 0
                reference_region_sigma_mask = _reference_region_sigma_mask(
                    sigmas,
                    min_sigma=reference_region_loss_min_sigma,
                    max_sigma=reference_region_loss_max_sigma,
                ).to(device=accelerator.device, dtype=torch.bool)
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
                should_compute_reference_region_loss = (
                    reference_region_loss_weight > 0.0
                    and reference_region_loss_interval > 0
                    and global_step % reference_region_loss_interval == 0
                    and bool(reference_region_sigma_mask.any().item())
                )
                should_compute_self_reconstruction_l1 = bool(
                    self_reconstruction_sample_mask.any().item()
                )
                if (
                    should_compute_style_loss
                    or should_compute_perceptual_loss
                    or should_compute_reference_region_loss
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
                if should_compute_reference_region_loss:
                    if reference_region_loss_backend == REFERENCE_REGION_LOSS_BACKEND_UNI:
                        ref_encoder = modules["ref_encoder"]
                        uni_dtype = next(ref_encoder.uni.parameters()).dtype
                        prediction_features = ref_encoder.extract_uni_features(
                            prediction_rgb.to(device=accelerator.device, dtype=uni_dtype),
                            allow_input_grad=True,
                        )
                        reference_features = ref_encoder.extract_uni_features(
                            training_batch["reference_image"].to(device=accelerator.device, dtype=uni_dtype),
                        )
                        reference_region_terms = regional_feature_map_loss(
                            prediction_features=prediction_features,
                            reference_features=reference_features,
                            target_tissue_mask=training_batch["target_tissue_mask"].to(accelerator.device),
                            reference_tissue_mask=training_batch["reference_tissue_mask"].to(accelerator.device),
                            target_nuclei_mask=training_batch["target_nuclei_mask"].to(accelerator.device),
                            reference_nuclei_mask=training_batch["reference_nuclei_mask"].to(accelerator.device),
                            sample_mask=reference_region_sigma_mask,
                            config=reference_region_loss_config,
                        )
                    else:
                        reference_region_terms = regional_rgb_fft_loss(
                            prediction=prediction_rgb,
                            reference=training_batch["reference_image"].to(
                                device=accelerator.device,
                                dtype=prediction_rgb.dtype,
                            ),
                            target_tissue_mask=training_batch["target_tissue_mask"].to(accelerator.device),
                            reference_tissue_mask=training_batch["reference_tissue_mask"].to(accelerator.device),
                            target_nuclei_mask=training_batch["target_nuclei_mask"].to(accelerator.device),
                            reference_nuclei_mask=training_batch["reference_nuclei_mask"].to(accelerator.device),
                            sample_mask=reference_region_sigma_mask,
                            config=reference_region_loss_config,
                        )
                    reference_region_loss = reference_region_terms["total"].to(dtype=denoising_loss.dtype)
                    reference_region_tissue_loss = reference_region_terms["tissue"].to(dtype=denoising_loss.dtype)
                    reference_region_nuclei_loss = reference_region_terms["nuclei"].to(dtype=denoising_loss.dtype)
                    reference_region_composite_loss = reference_region_terms["composite"].to(dtype=denoising_loss.dtype)
                    reference_region_tissue_regions = int(reference_region_terms["tissue_regions"])
                    reference_region_nuclei_regions = int(reference_region_terms["nuclei_regions"])
                    reference_region_composite_regions = int(reference_region_terms["composite_regions"])
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
                ip_health_logs: dict[str, float] = {}
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
                            swapped_batch,
                            modules,
                            accelerator,
                            weight_dtype,
                            flux_transformer,
                            regional=regional_ip_adapter,
                            query_token_count=noisy_model_input.shape[1],
                            strict=regional_ip_strict,
                            regional_token_mode=regional_ip_token_mode,
                            regional_label_mode=regional_ip_label_mode,
                            use_soft_bias=regional_ip_use_soft_bias,
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
                    + reference_region_loss_weight * reference_region_loss
                    + reference_style_loss_weight * style_loss
                    + ref_swap_loss_weight * swap_loss
                    + self_reconstruction_l1_weighted
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    ip_health_monitor.record_after_backward()
                if accelerator.sync_gradients and not gradient_flow_debug_logged:
                    _log_gradient_flow_debug(
                        ref_trainable_wrapper=ref_trainable_wrapper,
                        ip_trainable_wrapper=ip_trainable_wrapper,
                        optimizer=optimizer,
                        accelerator=accelerator,
                        step=global_step,
                    )
                    gradient_flow_debug_logged = True
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
                if accelerator.sync_gradients and trainable_ema is not None:
                    trainable_ema.update(
                        [unwrap_model(ref_trainable_wrapper), unwrap_model(ip_trainable_wrapper)]
                    )
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                if should_run_ip_health:
                    _log_reference_signal_debug(
                        batch=training_batch,
                        modules=modules,
                        accelerator=accelerator,
                        weight_dtype=weight_dtype,
                        transformer=flux_transformer,
                        regional=regional_ip_adapter,
                        regional_strict=regional_ip_strict,
                        regional_token_mode=regional_ip_token_mode,
                        regional_label_mode=regional_ip_label_mode,
                        use_soft_bias=regional_ip_use_soft_bias,
                        soft_bias=None,
                        query_token_count=noisy_model_input.shape[1],
                        step=health_step,
                        real_contrast_batch=health_real_contrast_batch,
                    )
                    reference_signal_debug_logged = True
                    ip_health_logs.update(
                        ip_health_monitor.log_param_delta(step=health_step)
                    )
                    ip_health_logs.update(
                        _run_ip_reference_health_diagnostics(
                            step=health_step,
                            training_batch=training_batch,
                            real_contrast_batch=health_real_contrast_batch,
                            modules=modules,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            transformer=flux_transformer,
                            noisy_model_input=noisy_model_input.detach(),
                            timesteps=timesteps.detach(),
                            guidance_vec=guidance_vec.detach() if guidance_vec is not None else None,
                            batch_pooled=batch_pooled.detach(),
                            batch_prompt=batch_prompt.detach(),
                            text_ids=text_ids.detach(),
                            latent_image_ids=latent_image_ids.detach(),
                            transformer_controlnet_block_samples=transformer_controlnet_block_samples,
                            transformer_controlnet_single_block_samples=transformer_controlnet_single_block_samples,
                            target_velocity=target_velocity.detach(),
                            regional=regional_ip_adapter,
                            regional_strict=regional_ip_strict,
                            regional_token_mode=regional_ip_token_mode,
                            regional_label_mode=regional_ip_label_mode,
                            use_soft_bias=regional_ip_use_soft_bias,
                            soft_bias=None,
                            query_token_count=int(noisy_model_input.shape[1]),
                            warmup_steps=ip_health_debug_warmup_steps,
                            min_ref_l2=ip_health_min_ref_l2,
                            min_swap_loss_gap=ip_health_min_swap_loss_gap,
                            max_ip_ratio_warn=ip_health_max_ip_ratio,
                            min_ip_ratio_warn=ip_health_min_ip_ratio,
                        )
                    )

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
                        ip_num_tokens=ip_num_tokens,
                        cross_v1_ip_architecture=cross_v1_ip_architecture,
                        regional_ip_adapter=regional_ip_adapter,
                        regional_ip_token_mode=regional_ip_token_mode,
                        regional_ip_label_mode=regional_ip_label_mode,
                        regional_ip_soft_bias_init=regional_ip_soft_bias_init,
                    )
                    if trainable_ema is not None:
                        torch.save(trainable_ema.state_dict(), os.path.join(save_path, "ema_state.pt"))
                        trainable_ema.copy_to(
                            [unwrap_model(ref_trainable_wrapper), unwrap_model(ip_trainable_wrapper)]
                        )
                        unwrap_model(ref_trainable_wrapper).sync_back(ref_encoder_raw)
                        _sync_ip_adapter_to_transformer(unwrap_model(ip_trainable_wrapper), flux_transformer)
                        ema_save_path = os.path.join(save_path, "ema")
                        _save_cross_v1_artifacts(
                            ema_save_path,
                            args,
                            flux_controlnet=flux_controlnet,
                            modules=modules,
                            ip_trainable_wrapper=ip_trainable_wrapper,
                            unwrap_model=unwrap_model,
                            control_spec=control_spec,
                            ip_num_tokens=ip_num_tokens,
                            cross_v1_ip_architecture=cross_v1_ip_architecture,
                            regional_ip_adapter=regional_ip_adapter,
                            regional_ip_token_mode=regional_ip_token_mode,
                            regional_ip_label_mode=regional_ip_label_mode,
                            regional_ip_soft_bias_init=regional_ip_soft_bias_init,
                        )
                        trainable_ema.restore(
                            [unwrap_model(ref_trainable_wrapper), unwrap_model(ip_trainable_wrapper)]
                        )
                        unwrap_model(ref_trainable_wrapper).sync_back(ref_encoder_raw)
                        _sync_ip_adapter_to_transformer(unwrap_model(ip_trainable_wrapper), flux_transformer)
                        logger.info("Saved EMA eval-ready Phase 5.3 cross-v1 artifacts to %s", ema_save_path)
                    logger.info("Saved raw eval-ready Phase 5.3 cross-v1 artifacts to %s", save_path)

            logs = {
                "loss": loss.detach().item(),
                "denoise_loss": denoising_loss.detach().item(),
                "cross_denoise_loss": cross_denoising_loss.detach().item(),
                "counterfactual_denoise_loss": counterfactual_denoising_loss.detach().item(),
                "self_reconstruction_denoise_loss": self_reconstruction_denoising_loss.detach().item(),
                "perceptual_loss": perceptual_loss.detach().item(),
                "reference_region_loss": reference_region_loss.detach().item(),
                "reference_region_loss_weighted": (reference_region_loss_weight * reference_region_loss).detach().item(),
                "reference_region_tissue_loss": reference_region_tissue_loss.detach().item(),
                "reference_region_nuclei_loss": reference_region_nuclei_loss.detach().item(),
                "reference_region_composite_loss": reference_region_composite_loss.detach().item(),
                "reference_region_tissue_regions": reference_region_tissue_regions,
                "reference_region_nuclei_regions": reference_region_nuclei_regions,
                "reference_region_composite_regions": reference_region_composite_regions,
                "reference_region_sigma_gated_samples": int(
                    reference_region_sigma_mask.sum().detach().item()
                ),
                "style_loss": style_loss.detach().item(),
                "style_tissue_loss": style_tissue_loss.detach().item(),
                "style_nuclei_loss": style_nuclei_loss.detach().item(),
                "self_reconstruction_l1": self_reconstruction_l1.detach().item(),
                "self_reconstruction_l1_weighted": self_reconstruction_l1_weighted.detach().item(),
                "self_reconstruction_samples": int(self_reconstruction_sample_mask.sum().detach().item()),
                "counterfactual_samples": int(counterfactual_sample_mask.sum().detach().item()),
                "appearance_degraded_samples": int(appearance_degraded_sample_mask.sum().detach().item()),
                "cross_samples": int(cross_sample_mask.sum().detach().item()),
                "ref_swap_loss": swap_loss.detach().item(),
                "ref_normal_denoise_loss": denoising_loss.detach().item(),
                "style_tissue_regions": style_tissue_regions,
                "style_nuclei_regions": style_nuclei_regions,
                "lr": lr_scheduler.get_last_lr()[0],
            }
            logs.update(ref_variant_loss_logs)
            logs.update(ip_health_logs)
            logs.update(_ip_soft_bias_log_values(flux_transformer))
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
            cross_v1_ip_architecture=cross_v1_ip_architecture,
            regional_ip_token_mode=regional_ip_token_mode,
            regional_ip_label_mode=regional_ip_label_mode,
            regional_ip_soft_bias_init=regional_ip_soft_bias_init,
        )
        _save_ip_adapter_modules(
            args.output_dir,
            ip_trainable_wrapper,
            unwrap_model,
            save_dtype,
            num_tokens=ip_num_tokens,
            ip_init_gain=args.ip_init_gain,
            cross_v1_ip_architecture=cross_v1_ip_architecture,
            regional_ip_adapter=regional_ip_adapter,
            regional_ip_token_mode=regional_ip_token_mode,
            regional_ip_label_mode=regional_ip_label_mode,
            regional_ip_soft_bias_init=regional_ip_soft_bias_init,
        )
        if trainable_ema is not None:
            torch.save(trainable_ema.state_dict(), os.path.join(args.output_dir, "ema_state.pt"))
            trainable_ema.copy_to(
                [unwrap_model(ref_trainable_wrapper), unwrap_model(ip_trainable_wrapper)]
            )
            unwrap_model(ref_trainable_wrapper).sync_back(ref_encoder_raw)
            _sync_ip_adapter_to_transformer(unwrap_model(ip_trainable_wrapper), flux_transformer)
            ema_output_dir = os.path.join(args.output_dir, "ema")
            _save_cross_v1_artifacts(
                ema_output_dir,
                args,
                flux_controlnet=flux_controlnet,
                modules=modules,
                ip_trainable_wrapper=ip_trainable_wrapper,
                unwrap_model=unwrap_model,
                control_spec=control_spec,
                ip_num_tokens=ip_num_tokens,
                cross_v1_ip_architecture=cross_v1_ip_architecture,
                regional_ip_adapter=regional_ip_adapter,
                regional_ip_token_mode=regional_ip_token_mode,
                regional_ip_label_mode=regional_ip_label_mode,
                regional_ip_soft_bias_init=regional_ip_soft_bias_init,
            )
            trainable_ema.restore(
                [unwrap_model(ref_trainable_wrapper), unwrap_model(ip_trainable_wrapper)]
            )
            unwrap_model(ref_trainable_wrapper).sync_back(ref_encoder_raw)
            _sync_ip_adapter_to_transformer(unwrap_model(ip_trainable_wrapper), flux_transformer)
            logger.info("Saved EMA Phase 5.3 cross-v1 artifacts to %s", ema_output_dir)
        logger.info("Saved raw Phase 5.3 cross-v1 artifacts to %s", args.output_dir)

    accelerator.end_training()

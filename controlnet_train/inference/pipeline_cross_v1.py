from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

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
    CrossV1ControlSpec,
    build_cross_v1_condition,
)
from controlnet_train.training.conditioning import patch_controlnet_x_embedder
from controlnet_train.inference.pipeline_cross_v2_1 import (
    _build_mask_change_map,
    _build_packed_change_gate,
    _pack_source_latents_for_sampling,
    _prepare_source_noised_latents,
    _sigma_for_timestep,
    _source_init_timesteps,
    _validate_nonnegative_float,
    _validate_source_latent_init_strength,
)
from controlnet_train.modules.cross_v2_1_conditioning import deterministic_latent_from_posterior
from controlnet_train.modules.reference_image_encoder import (
    build_region_ip_token_labels,
    normalize_region_ip_label_mode,
    normalize_region_ip_token_mode,
    resize_mask_to_token_labels,
)


@dataclass
class CrossV1InferenceBundle:
    pretrained_model_name_or_path: str | Path
    checkpoint_path: Path
    uni_checkpoint_path: str | Path
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.bfloat16
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    controlnet_conditioning_scale: float = 1.0
    ip_adapter_scale: float = 1.0
    flux_pipeline: object | None = None
    controlnet: object | None = None
    condition_modules: dict[str, nn.Module] = field(default_factory=dict)
    control_spec: CrossV1ControlSpec = field(default_factory=CrossV1ControlSpec)
    ip_adapter_modules: dict[str, nn.Module] = field(default_factory=dict)
    ref_encoder: ReferenceImageEncoder | None = None
    regional_ip_adapter: bool = False
    regional_ip_strict: bool = True
    regional_ip_token_mode: str = "spatial"
    regional_ip_label_mode: str = "tissue"
    cross_v1_ip_architecture: str = "global"
    regional_ip_use_soft_bias: bool = False
    regional_ip_soft_bias_init: float = 4.0


@dataclass
class FluxKVInjectionContext:
    reference_features: dict[str, Any]
    mode: str = "kv"
    strength: float = 0.2
    start_step: int = 18
    after_layer: int = 20
    inject_after_t: float | None = None
    image_token_count: int = 1024
    text_token_count: int | None = None
    step_index: int = -1
    timestep: float = 0.0
    second_order: bool = False
    enabled: bool = True
    events: list[dict[str, Any]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    regional_stats: list[dict[str, Any]] = field(default_factory=list)

    def should_inject(self, block_id: int) -> bool:
        return (
            bool(self.enabled)
            and int(self.step_index) >= int(self.start_step)
            and int(block_id) >= int(self.after_layer)
            and (
                self.inject_after_t is None
                or float(self.timestep) <= float(self.inject_after_t)
            )
            and float(self.strength) > 0.0
        )

    def get_reference(self, block_id: int) -> dict[str, torch.Tensor] | None:
        payload = _find_kv_reference_feature(
            self.reference_features,
            block_id=int(block_id),
            t=float(self.timestep),
            second_order=bool(self.second_order),
        )
        if payload is None:
            if len(self.missing) < 128:
                self.missing.append(
                    f"block={int(block_id)} t={float(self.timestep):.10f} "
                    f"second_order={bool(self.second_order)}"
                )
            return None
        return payload

    def record(self, event: dict[str, Any]) -> None:
        if len(self.events) < 2048:
            self.events.append(event)

    def regional_attention_mask(
        self,
        *,
        total_tokens: int,
        text_tokens: int,
        image_tokens: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
        payload = self.reference_features.get("regional_labels")
        if not isinstance(payload, dict):
            return None, None
        mask, stats = _build_kv_regional_attention_mask(
            total_tokens=total_tokens,
            text_tokens=text_tokens,
            image_tokens=image_tokens,
            labels=payload,
            device=device,
        )
        if stats is not None and len(self.regional_stats) < 256:
            self.regional_stats.append(stats)
        return mask, stats

    def query_injection_scale(
        self,
        *,
        total_tokens: int,
        text_tokens: int,
        image_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
        payload = self.reference_features.get("regional_labels")
        if not isinstance(payload, dict):
            return None, None
        scale, stats = _build_kv_query_injection_scale(
            total_tokens=total_tokens,
            text_tokens=text_tokens,
            image_tokens=image_tokens,
            labels=payload,
            device=device,
            dtype=dtype,
        )
        if stats is not None and len(self.regional_stats) < 256:
            self.regional_stats.append(stats)
        return scale, stats

    def summary(self) -> dict[str, Any]:
        by_block = self.reference_features.get("by_block", {})
        return {
            "enabled": bool(self.enabled),
            "mode": self.mode,
            "strength": float(self.strength),
            "start_step": int(self.start_step),
            "after_layer": int(self.after_layer),
            "inject_after_t": self.inject_after_t,
            "image_token_count": int(self.image_token_count),
            "text_token_count": self.text_token_count,
            "reference_feature_count": int(self.reference_features.get("feature_count", 0)),
            "reference_block_ids": sorted(int(key) for key in by_block.keys()),
            "event_count": len(self.events),
            "events_sample": self.events[:40],
            "missing_count": len(self.missing),
            "missing_sample": self.missing[:20],
            "regional_labels": _summarize_kv_regional_labels(
                self.reference_features.get("regional_labels")
            ),
            "regional_stats_sample": self.regional_stats[:20],
        }


def install_flux_single_kv_injection(
    transformer: nn.Module,
    context: FluxKVInjectionContext,
) -> dict[str, Any]:
    """Replace FLUX single-block processors with K/V-injecting processors."""
    installed: list[int] = []
    skipped: list[int] = []
    for block_id, block in enumerate(getattr(transformer, "single_transformer_blocks", [])):
        attn = getattr(block, "attn", None)
        processor = getattr(attn, "processor", None)
        if attn is None or processor is None:
            skipped.append(int(block_id))
            continue
        wrapped = FluxSingleKVCrossImageAttnProcessor2_0.from_existing(
            processor,
            context=context,
            block_id=block_id,
        )
        attn.set_processor(wrapped)
        installed.append(int(block_id))
    return {
        "installed_single_blocks": installed,
        "skipped_single_blocks": skipped,
        "after_layer": int(context.after_layer),
        "start_step": int(context.start_step),
        "mode": context.mode,
        "strength": float(context.strength),
    }


class FluxSingleKVCrossImageAttnProcessor2_0(nn.Module):
    """FLUX single-stream processor with late reference K/V image-token injection.

    The single-stream sequence is [text_tokens, image_tokens]. Text-token K/V
    stay native to the current target denoise. Only the image-token slice can be
    blended toward RF-Solver reference K/V.
    """

    def __init__(
        self,
        *,
        context: FluxKVInjectionContext,
        block_id: int,
        num_tokens: tuple[int, ...],
        scale: list[float],
        to_k_ip: nn.ModuleList | None = None,
        to_v_ip: nn.ModuleList | None = None,
        ip_null_tokens: nn.ParameterList | None = None,
        ip_soft_bias: nn.ParameterList | None = None,
        use_soft_bias: bool = False,
        debug_name: str | None = None,
    ) -> None:
        super().__init__()
        self.context = context
        self.block_id = int(block_id)
        self.num_tokens = tuple(int(value) for value in num_tokens)
        self.scale = [float(value) for value in scale]
        self.to_k_ip = to_k_ip if to_k_ip is not None else nn.ModuleList()
        self.to_v_ip = to_v_ip if to_v_ip is not None else nn.ModuleList()
        self.ip_null_tokens = ip_null_tokens if ip_null_tokens is not None else nn.ParameterList()
        self.ip_soft_bias = ip_soft_bias if ip_soft_bias is not None else nn.ParameterList(
            [nn.Parameter(torch.tensor(4.0))]
        )
        self.use_soft_bias = bool(use_soft_bias)
        self.debug_name = debug_name or f"single_block_{self.block_id}"

    @classmethod
    def from_existing(
        cls,
        processor: nn.Module,
        *,
        context: FluxKVInjectionContext,
        block_id: int,
    ) -> "FluxSingleKVCrossImageAttnProcessor2_0":
        num_tokens = getattr(processor, "num_tokens", (16,))
        if not isinstance(num_tokens, (tuple, list)):
            num_tokens = (int(num_tokens),)
        scale = getattr(processor, "scale", [0.0] * len(num_tokens))
        if not isinstance(scale, list):
            scale = [float(value) for value in scale] if isinstance(scale, tuple) else [float(scale)]
        if len(scale) != len(num_tokens):
            scale = [float(scale[0] if scale else 0.0)] * len(num_tokens)
        return cls(
            context=context,
            block_id=block_id,
            num_tokens=tuple(int(value) for value in num_tokens),
            scale=[float(value) for value in scale],
            to_k_ip=getattr(processor, "to_k_ip", None),
            to_v_ip=getattr(processor, "to_v_ip", None),
            ip_null_tokens=getattr(processor, "ip_null_tokens", None),
            ip_soft_bias=getattr(processor, "ip_soft_bias", None),
            use_soft_bias=bool(getattr(processor, "use_soft_bias", False)),
            debug_name=getattr(processor, "debug_name", None),
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
            raise ValueError(
                "FluxSingleKVCrossImageAttnProcessor2_0 expects pre-concatenated single-stream states."
            )
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

        base_key = key
        base_value = value
        base_attention_mask = attention_mask
        key, value, kv_attention_mask, kv_query_injection_scale = self._inject_reference_kv(
            key,
            value,
        )

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            if kv_query_injection_scale is not None:
                base_key = apply_rotary_emb(base_key, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        if kv_attention_mask is not None:
            if attention_mask is not None and len(self.context.missing) < 128:
                self.context.missing.append(
                    f"attention_mask_ignored_for_kv_regional:block={self.block_id}"
                )
            attention_mask = kv_attention_mask

        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        if kv_query_injection_scale is not None:
            base_output = F.scaled_dot_product_attention(
                query,
                base_key,
                base_value,
                attn_mask=base_attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            scale = kv_query_injection_scale.to(device=output.device, dtype=output.dtype)
            if scale.ndim != 2 or scale.shape[0] != output.shape[0] or scale.shape[1] != output.shape[2]:
                raise ValueError(
                    "K/V query injection scale must have shape "
                    f"({output.shape[0]},{output.shape[2]}), got {tuple(scale.shape)}"
                )
            output = torch.lerp(base_output, output, scale[:, None, :, None])
        output = output.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        output = output.to(query.dtype)

        output = self._apply_ip_residual(
            output=output,
            query=query,
            head_dim=head_dim,
            attn=attn,
            ip_hidden_states=ip_hidden_states,
            ip_adapter_masks=ip_adapter_masks,
            ip_region_token_labels=ip_region_token_labels,
            ip_query_region_labels=ip_query_region_labels,
            ip_region_fallback_labels=ip_region_fallback_labels,
            ip_query_fallback_labels=ip_query_fallback_labels,
            ip_region_strict=ip_region_strict,
            ip_region_soft_bias=ip_region_soft_bias,
            ip_region_use_soft_bias=ip_region_use_soft_bias,
            txt_seq_len=txt_seq_len,
            ip_debug_collector=ip_debug_collector,
        )
        return output.to(hidden_states.dtype)

    def _inject_reference_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        context = self.context
        if not context.should_inject(self.block_id):
            return key, value, None, None
        payload = context.get_reference(self.block_id)
        if payload is None:
            return key, value, None, None
        ref_k = payload["K"].to(device=key.device, dtype=key.dtype)
        ref_v = payload["V"].to(device=value.device, dtype=value.dtype)
        if ref_k.shape != key.shape or ref_v.shape != value.shape:
            if len(context.missing) < 128:
                context.missing.append(
                    f"shape_mismatch:block={self.block_id}:"
                    f"k={tuple(key.shape)} ref_k={tuple(ref_k.shape)} "
                    f"v={tuple(value.shape)} ref_v={tuple(ref_v.shape)}"
                )
            return key, value, None, None

        total_tokens = int(key.shape[2])
        image_tokens = int(context.image_token_count)
        if total_tokens <= image_tokens:
            if len(context.missing) < 128:
                context.missing.append(
                    f"bad_token_split:block={self.block_id}:total={total_tokens}:image={image_tokens}"
                )
            return key, value, None, None
        text_tokens = total_tokens - image_tokens
        context.text_token_count = int(text_tokens)
        target_slice = slice(text_tokens, total_tokens)
        strength = min(max(float(context.strength), 0.0), 1.0)
        mode = str(context.mode)
        next_key = key
        next_value = value
        if mode == "kv":
            next_key = key.clone()
            next_key[:, :, target_slice, :] = torch.lerp(
                next_key[:, :, target_slice, :],
                ref_k[:, :, target_slice, :],
                strength,
            )
        next_value = value.clone()
        next_value[:, :, target_slice, :] = torch.lerp(
            next_value[:, :, target_slice, :],
            ref_v[:, :, target_slice, :],
            strength,
        )
        regional_mask, regional_stats = context.regional_attention_mask(
            total_tokens=total_tokens,
            text_tokens=text_tokens,
            image_tokens=image_tokens,
            device=key.device,
        )
        query_injection_scale, query_injection_stats = context.query_injection_scale(
            total_tokens=total_tokens,
            text_tokens=text_tokens,
            image_tokens=image_tokens,
            device=key.device,
            dtype=key.dtype,
        )
        context.record(
            {
                "action": "inject_ref_kv" if mode == "kv" else "inject_ref_v",
                "block_id": int(self.block_id),
                "step_index": int(context.step_index),
                "t": float(context.timestep),
                "second_order": bool(context.second_order),
                "mode": mode,
                "strength": strength,
                "tokens_total": total_tokens,
                "text_tokens": text_tokens,
                "image_tokens": image_tokens,
                "regional_mode": (
                    regional_stats.get("mode", "none")
                    if isinstance(regional_stats, dict)
                    else "none"
                ),
                "nuclei_query_injection_scale": (
                    query_injection_stats.get("target_nuclei_inject_scale")
                    if isinstance(query_injection_stats, dict)
                    else None
                ),
                "nuclei_query_count": (
                    query_injection_stats.get("target_nuclei_query_count")
                    if isinstance(query_injection_stats, dict)
                    else None
                ),
                "reference_key": payload.get("key"),
            }
        )
        return next_key, next_value, regional_mask, query_injection_scale

    def _apply_ip_residual(
        self,
        *,
        output: torch.Tensor,
        query: torch.Tensor,
        head_dim: int,
        attn,
        ip_hidden_states: list[torch.Tensor] | tuple[torch.Tensor, ...] | None,
        ip_adapter_masks: torch.Tensor | dict | None,
        ip_region_token_labels: torch.Tensor | None,
        ip_query_region_labels: torch.Tensor | None,
        ip_region_fallback_labels: torch.Tensor | None,
        ip_query_fallback_labels: torch.Tensor | None,
        ip_region_strict: bool,
        ip_region_soft_bias: torch.Tensor | float | None,
        ip_region_use_soft_bias: bool | None,
        txt_seq_len: int | None,
        ip_debug_collector: dict | None,
    ) -> torch.Tensor:
        if not ip_hidden_states:
            return output
        if not self.to_k_ip or not self.to_v_ip:
            return output
        from controlnet_train.training.flux_phase5_cross_v1 import (
            _build_region_attention_mask_and_query_gate,
            _merge_ip_attention_pooling_debug,
            _record_ip_attention_debug,
            _summarize_ip_attention_pooling_debug,
            _unpack_region_ip_adapter_masks,
            _unpack_region_ip_soft_bias_enabled,
            _unpack_region_ip_soft_bias_value,
        )

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
        if soft_bias_value is None and len(self.ip_soft_bias) > 0:
            soft_bias_value = self.ip_soft_bias[0]
        use_soft_bias = bool(self.use_soft_bias or packed_use_soft_bias)
        txt_seq_len = int(txt_seq_len or 0)
        if txt_seq_len < 0 or txt_seq_len > output.shape[1]:
            raise ValueError(f"txt_seq_len must be within [0, {output.shape[1]}], got {txt_seq_len}.")
        image_query = query[:, :, txt_seq_len:, :]
        if image_query.shape[2] <= 0:
            return output

        debug_collector = (
            ip_debug_collector
            if ip_debug_collector is not None
            else getattr(self, "_ip_debug_collector", None)
        )
        collect_attention_pooling = bool(
            isinstance(debug_collector, dict)
            and debug_collector.get("store_attention_pooling", False)
        )
        batch_size = int(output.shape[0])
        image_ip_output = output.new_zeros((batch_size, image_query.shape[2], output.shape[2]))
        debug_uniform_ip_output = None
        debug_label_uniform_ip_output = None
        debug_attention_stats: list[dict[str, float | int]] = []
        debug_key_masses: list[torch.Tensor] = []
        for index, current_ip_hidden_states in enumerate(ip_hidden_states):
            if index >= len(self.scale) or index >= len(self.to_k_ip) or index >= len(self.to_v_ip):
                continue
            scale = float(self.scale[index])
            if scale == 0:
                continue
            to_k_ip = self.to_k_ip[index]
            to_v_ip = self.to_v_ip[index]
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
                if index >= len(self.ip_null_tokens):
                    raise RuntimeError("IP attention mask requires a null token, but none is installed.")
                null_input = self.ip_null_tokens[index].to(
                    device=ip_input.device,
                    dtype=ip_input.dtype,
                ).expand(batch_size, -1, -1)
                ip_input = torch.cat([ip_input, null_input], dim=1)
            ip_key = to_k_ip(ip_input).to(dtype=image_query.dtype)
            ip_value = to_v_ip(ip_input).to(dtype=image_query.dtype)
            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_attn = F.scaled_dot_product_attention(
                image_query,
                ip_key,
                ip_value,
                attn_mask=ip_attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            ip_attn = ip_attn.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            scaled_ip_attn = scale * ip_attn.to(output.dtype)
            image_ip_output = image_ip_output + scaled_ip_attn
            if collect_attention_pooling:
                pooling_debug = _summarize_ip_attention_pooling_debug(
                    query=image_query,
                    key=ip_key,
                    value=ip_value,
                    attn_mask=ip_attn_mask,
                    output=ip_attn,
                    query_region_labels=ip_query_region_labels,
                    key_region_labels=ip_region_token_labels,
                )
                debug_attention_stats.append(pooling_debug["stats"])
                debug_key_masses.append(pooling_debug["key_mass"])
                uniform_output = scale * pooling_debug["uniform_output"].to(output.dtype)
                debug_uniform_ip_output = (
                    uniform_output
                    if debug_uniform_ip_output is None
                    else debug_uniform_ip_output + uniform_output
                )
                label_uniform_output = pooling_debug.get("label_uniform_output")
                if torch.is_tensor(label_uniform_output):
                    label_uniform_output = scale * label_uniform_output.to(output.dtype)
                    debug_label_uniform_ip_output = (
                        label_uniform_output
                        if debug_label_uniform_ip_output is None
                        else debug_label_uniform_ip_output + label_uniform_output
                    )
        debug_payload = None
        if collect_attention_pooling:
            debug_payload = _merge_ip_attention_pooling_debug(
                debug_attention_stats,
                debug_key_masses,
                debug_uniform_ip_output,
                debug_label_uniform_ip_output,
            )
        _record_ip_attention_debug(
            debug_collector,
            getattr(self, "debug_name", "single_block"),
            output[:, txt_seq_len:, :],
            image_ip_output,
            debug_payload=debug_payload,
        )
        image_output = output[:, txt_seq_len:, :] + image_ip_output
        if txt_seq_len > 0:
            return torch.cat([output[:, :txt_seq_len, :], image_output], dim=1)
        return image_output


def _find_kv_reference_feature(
    features: dict[str, Any],
    *,
    block_id: int,
    t: float,
    second_order: bool,
) -> dict[str, torch.Tensor] | None:
    by_block = features.get("by_block")
    if not isinstance(by_block, dict):
        return None
    entries = by_block.get(int(block_id)) or by_block.get(str(int(block_id)))
    if not entries:
        return None

    def _candidate_score(entry: dict[str, Any], require_second_order: bool) -> tuple[float, dict[str, Any]] | None:
        if require_second_order and bool(entry.get("second_order")) != bool(second_order):
            return None
        return abs(float(entry.get("t", 0.0)) - float(t)), entry

    scored = [
        score
        for entry in entries
        if (score := _candidate_score(entry, require_second_order=True)) is not None
    ]
    if not scored:
        scored = [
            score
            for entry in entries
            if (score := _candidate_score(entry, require_second_order=False)) is not None
        ]
    if not scored:
        return None
    _, best = min(scored, key=lambda item: item[0])
    payload = best.get("payload")
    if not isinstance(payload, dict):
        return None
    result = dict(payload)
    result["key"] = best.get("key")
    return result


def _build_kv_regional_attention_mask(
    *,
    total_tokens: int,
    text_tokens: int,
    image_tokens: int,
    labels: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
    mode = str(labels.get("mode", "none"))
    suppress_ref_nuclei = bool(labels.get("block_ref_nuclei_to_target_non_nuclei", False))
    if mode == "none" and not suppress_ref_nuclei:
        return None, None
    if int(total_tokens) != int(text_tokens) + int(image_tokens):
        raise ValueError(
            "K/V regional token split mismatch: "
            f"total={total_tokens} text={text_tokens} image={image_tokens}"
        )

    if mode == "tissue":
        query_labels = _kv_label_tensor(labels, "target_tissue", image_tokens=image_tokens, device=device)
        key_labels = _kv_label_tensor(labels, "reference_tissue", image_tokens=image_tokens, device=device)
    elif mode == "nuclei":
        query_labels = _kv_label_tensor(labels, "target_nuclei", image_tokens=image_tokens, device=device)
        key_labels = _kv_label_tensor(labels, "reference_nuclei", image_tokens=image_tokens, device=device)
    elif mode == "tissue_nuclei":
        query_labels = _kv_label_tensor(labels, "target_composite", image_tokens=image_tokens, device=device)
        key_labels = _kv_label_tensor(labels, "reference_composite", image_tokens=image_tokens, device=device)
    elif mode == "none":
        query_labels = None
        key_labels = None
    else:
        raise ValueError(f"Unsupported K/V regional mode {mode!r}.")
    if mode != "none" and (query_labels is None or key_labels is None):
        raise ValueError(f"K/V regional mode {mode!r} is missing target/reference labels.")

    fallback_tissue = mode == "tissue_nuclei"
    target_tissue = (
        _kv_label_tensor(labels, "target_tissue", image_tokens=image_tokens, device=device)
        if fallback_tissue
        else None
    )
    reference_tissue = (
        _kv_label_tensor(labels, "reference_tissue", image_tokens=image_tokens, device=device)
        if fallback_tissue
        else None
    )
    target_nuclei_present = _kv_bool_token_tensor(
        labels,
        "target_nuclei_present",
        image_tokens=image_tokens,
        device=device,
    )
    reference_nuclei_present = _kv_bool_token_tensor(
        labels,
        "reference_nuclei_present",
        image_tokens=image_tokens,
        device=device,
    )
    if suppress_ref_nuclei and (target_nuclei_present is None or reference_nuclei_present is None):
        raise ValueError(
            "K/V ref-nuclei suppression requires target_nuclei_present and "
            "reference_nuclei_present occupancy labels."
        )

    mask = torch.zeros((1, 1, int(total_tokens), int(total_tokens)), dtype=torch.bool, device=device)
    target_slice = slice(int(text_tokens), int(total_tokens))
    mask[:, :, : int(text_tokens), : int(text_tokens)] = True
    if mode == "none":
        mask[:, :, : int(text_tokens), :] = True
    mask[:, :, target_slice, : int(text_tokens)] = True

    fallback_all_count = 0
    fallback_tissue_count = 0
    exact_count = 0
    ref_nuclei_blocked_pair_count = 0
    ref_nuclei_empty_fallback_count = 0
    for query_index in range(int(image_tokens)):
        if mode == "none":
            allowed = torch.ones((int(image_tokens),), dtype=torch.bool, device=device)
        else:
            label = int(query_labels[0, query_index].item())
            if label >= 0:
                allowed = key_labels[0] == label
            else:
                allowed = torch.zeros((int(image_tokens),), dtype=torch.bool, device=device)
            if bool(allowed.any().item()):
                exact_count += 1
            elif fallback_tissue and target_tissue is not None and reference_tissue is not None:
                tissue_label = int(target_tissue[0, query_index].item())
                if tissue_label >= 0:
                    allowed = reference_tissue[0] == tissue_label
                else:
                    allowed = torch.zeros((int(image_tokens),), dtype=torch.bool, device=device)
                if bool(allowed.any().item()):
                    fallback_tissue_count += 1
                else:
                    allowed = torch.ones((int(image_tokens),), dtype=torch.bool, device=device)
                    fallback_all_count += 1
            else:
                allowed = torch.ones((int(image_tokens),), dtype=torch.bool, device=device)
                fallback_all_count += 1
        if suppress_ref_nuclei and target_nuclei_present is not None and reference_nuclei_present is not None:
            query_has_nuclei = bool(target_nuclei_present[0, query_index].item())
            if not query_has_nuclei:
                suppress = reference_nuclei_present[0]
                suppressed_allowed = allowed & ~suppress
                ref_nuclei_blocked_pair_count += int((allowed & suppress).sum().item())
                if bool(suppressed_allowed.any().item()):
                    allowed = suppressed_allowed
                else:
                    ref_nuclei_empty_fallback_count += 1
        mask[:, :, int(text_tokens) + query_index, target_slice] = allowed

    allowed_per_image_query = mask[0, 0, target_slice].sum(dim=1)
    stats = {
        "mode": mode,
        "action": "regional_attention_mask",
        "text_tokens": int(text_tokens),
        "image_tokens": int(image_tokens),
        "exact_query_count": int(exact_count),
        "fallback_tissue_query_count": int(fallback_tissue_count),
        "fallback_all_query_count": int(fallback_all_count),
        "block_ref_nuclei_to_target_non_nuclei": bool(suppress_ref_nuclei),
        "ref_nuclei_blocked_pair_count": int(ref_nuclei_blocked_pair_count),
        "ref_nuclei_empty_fallback_count": int(ref_nuclei_empty_fallback_count),
        "allowed_tokens_per_image_query_min": int(allowed_per_image_query.min().item()),
        "allowed_tokens_per_image_query_max": int(allowed_per_image_query.max().item()),
        "allowed_tokens_per_image_query_mean": float(allowed_per_image_query.float().mean().item()),
    }
    return mask, stats


def _build_kv_query_injection_scale(
    *,
    total_tokens: int,
    text_tokens: int,
    image_tokens: int,
    labels: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
    if not bool(labels.get("protect_target_nuclei", False)):
        return None, None
    if int(total_tokens) != int(text_tokens) + int(image_tokens):
        raise ValueError(
            "K/V query injection scale token split mismatch: "
            f"total={total_tokens} text={text_tokens} image={image_tokens}"
        )
    target_nuclei_present = _kv_bool_token_tensor(
        labels,
        "target_nuclei_present",
        image_tokens=image_tokens,
        device=device,
    )
    if target_nuclei_present is None:
        raise ValueError("K/V target nuclei protection requires target_nuclei_present occupancy labels.")
    nuclei_scale = min(max(float(labels.get("target_nuclei_inject_scale", 0.0)), 0.0), 1.0)
    scale = torch.ones((1, int(total_tokens)), dtype=dtype, device=device)
    image_scale = scale[:, int(text_tokens): int(total_tokens)]
    image_scale[target_nuclei_present] = nuclei_scale
    nuclei_count = int(target_nuclei_present.sum().item())
    stats = {
        "mode": str(labels.get("mode", "none")),
        "action": "query_injection_scale",
        "protect_target_nuclei": True,
        "target_nuclei_inject_scale": float(nuclei_scale),
        "target_nuclei_query_count": nuclei_count,
        "target_non_nuclei_query_count": int(image_tokens) - nuclei_count,
    }
    if nuclei_count <= 0 or nuclei_scale >= 1.0:
        return None, stats
    return scale, stats


def _kv_label_tensor(
    labels: dict[str, Any],
    name: str,
    *,
    image_tokens: int,
    device: torch.device,
) -> torch.Tensor | None:
    value = labels.get(name)
    if value is None:
        return None
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    value = value.to(device=device, dtype=torch.long)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape[0] != 1 or value.shape[1] != int(image_tokens):
        raise ValueError(
            f"K/V regional labels {name!r} must have shape (1,{image_tokens}), "
            f"got {tuple(value.shape)}"
        )
    return value


def _kv_bool_token_tensor(
    labels: dict[str, Any],
    name: str,
    *,
    image_tokens: int,
    device: torch.device,
) -> torch.Tensor | None:
    value = labels.get(name)
    if value is None:
        return None
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    value = value.to(device=device)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape[0] != 1 or value.shape[1] != int(image_tokens):
        raise ValueError(
            f"K/V regional labels {name!r} must have shape (1,{image_tokens}), "
            f"got {tuple(value.shape)}"
        )
    return value.bool()


def _summarize_kv_regional_labels(labels: Any) -> dict[str, Any] | None:
    if not isinstance(labels, dict):
        return None

    def _counts(value: Any) -> dict[str, int] | None:
        if value is None:
            return None
        tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
        unique, counts = torch.unique(tensor.detach().cpu().to(dtype=torch.long), return_counts=True)
        return {str(int(label)): int(count) for label, count in zip(unique, counts)}

    return {
        "mode": str(labels.get("mode", "none")),
        "target_tissue_counts": _counts(labels.get("target_tissue")),
        "reference_tissue_counts": _counts(labels.get("reference_tissue")),
        "target_nuclei_counts": _counts(labels.get("target_nuclei")),
        "reference_nuclei_counts": _counts(labels.get("reference_nuclei")),
        "target_composite_counts": _counts(labels.get("target_composite")),
        "reference_composite_counts": _counts(labels.get("reference_composite")),
        "target_nuclei_present_counts": _counts(labels.get("target_nuclei_present")),
        "reference_nuclei_present_counts": _counts(labels.get("reference_nuclei_present")),
        "protect_target_nuclei": bool(labels.get("protect_target_nuclei", False)),
        "target_nuclei_inject_scale": labels.get("target_nuclei_inject_scale"),
        "block_ref_nuclei_to_target_non_nuclei": bool(
            labels.get("block_ref_nuclei_to_target_non_nuclei", False)
        ),
        "nuclei_occupancy": labels.get("nuclei_occupancy"),
    }


def load_cross_v1_bundle(
    *,
    pretrained_model_name_or_path: str | Path,
    checkpoint_path: str | Path,
    uni_checkpoint_path: str | Path,
    device: str = "cuda",
    torch_dtype: torch.dtype | None = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    controlnet_conditioning_scale: float = 1.0,
    ip_adapter_scale: float = 1.0,
) -> CrossV1InferenceBundle:
    device = _resolve_device(device)
    dtype = _resolve_torch_dtype(torch_dtype, device)
    checkpoint = _validate_checkpoint_dir(checkpoint_path)
    control_spec = _load_cross_v1_control_spec(checkpoint)
    ref_encoder_config = _load_ref_encoder_config(checkpoint)
    from controlnet_train.training.flux_phase5_cross_v1 import (
        CROSS_V1_IP_ARCH_GLOBAL_SOFT_BIAS,
        CROSS_V1_IP_ARCH_REGIONAL_HARD,
        _collect_ip_adapter_modules,
        install_flux_ip_adapter_attention,
        normalize_cross_v1_ip_architecture,
        patch_flux_single_ip_forward,
    )

    # Load Flux ControlNet pipeline with the checkpoint's V1 spatial layout.
    pipe, controlnet = _load_flux_controlnet_pipeline(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        checkpoint_path=checkpoint,
        packed_channels=control_spec.packed_channels,
        device=device,
        torch_dtype=dtype,
    )

    # Load IP-Adapter weights from checkpoint
    ip_state = _torch_load_weights(checkpoint / "phase5_ip_adapter.pt")
    if "cross_v1_ip_architecture" in ip_state:
        cross_v1_ip_architecture = normalize_cross_v1_ip_architecture(
            str(ip_state.get("cross_v1_ip_architecture"))
        )
    else:
        cross_v1_ip_architecture = normalize_cross_v1_ip_architecture(
            None,
            regional_ip_adapter=bool(ip_state.get("regional_ip_adapter", False)),
        )
    regional_ip_adapter = cross_v1_ip_architecture in {
        CROSS_V1_IP_ARCH_REGIONAL_HARD,
        CROSS_V1_IP_ARCH_GLOBAL_SOFT_BIAS,
    }
    regional_ip_use_soft_bias = cross_v1_ip_architecture == CROSS_V1_IP_ARCH_GLOBAL_SOFT_BIAS
    regional_ip_strict = cross_v1_ip_architecture == CROSS_V1_IP_ARCH_REGIONAL_HARD
    regional_ip_token_mode = normalize_region_ip_token_mode(
        ip_state.get(
            "regional_ip_token_mode",
            ref_encoder_config.get("regional_ip_token_mode", "spatial"),
        )
    )
    regional_ip_label_mode = normalize_region_ip_label_mode(
        ip_state.get(
            "regional_ip_label_mode",
            ref_encoder_config.get("regional_ip_label_mode", "tissue"),
        )
    )
    regional_ip_soft_bias_init = float(ip_state.get("regional_ip_soft_bias_init", 4.0) or 0.0)
    install_flux_ip_adapter_attention(
        pipe.transformer,
        num_tokens=int(ip_state.get("num_tokens", ref_encoder_config.get("num_output_tokens", ref_encoder_config["num_tokens"]))),
        num_single_layers=_resolve_saved_single_ip_layer_count(ip_state),
        regional=regional_ip_adapter,
        use_soft_bias=regional_ip_use_soft_bias,
        soft_bias_init=regional_ip_soft_bias_init,
    )
    patch_flux_single_ip_forward(pipe.transformer)
    pipe.transformer.encoder_hid_proj.load_state_dict(ip_state["encoder_hid_proj"])
    for i, block in enumerate(pipe.transformer.transformer_blocks):
        _load_state_dict_ignoring_legacy_bias(block.attn.processor.to_k_ip, ip_state[f"block_{i}_to_k_ip"])
        _load_state_dict_ignoring_legacy_bias(block.attn.processor.to_v_ip, ip_state[f"block_{i}_to_v_ip"])
        null_key = f"block_{i}_ip_null_tokens"
        if null_key in ip_state and hasattr(block.attn.processor, "ip_null_tokens"):
            block.attn.processor.ip_null_tokens.load_state_dict(ip_state[null_key])
        bias_key = f"block_{i}_ip_soft_bias"
        if bias_key in ip_state and hasattr(block.attn.processor, "ip_soft_bias"):
            block.attn.processor.ip_soft_bias.load_state_dict(ip_state[bias_key])
        elif regional_ip_use_soft_bias:
            raise RuntimeError(f"Missing soft-bias weights in Cross V1 IP checkpoint: {bias_key}")
    for i, block in enumerate(getattr(pipe.transformer, "single_transformer_blocks", [])):
        k_key = f"single_block_{i}_to_k_ip"
        v_key = f"single_block_{i}_to_v_ip"
        null_key = f"single_block_{i}_ip_null_tokens"
        bias_key = f"single_block_{i}_ip_soft_bias"
        if k_key in ip_state and v_key in ip_state:
            _load_state_dict_ignoring_legacy_bias(block.attn.processor.to_k_ip, ip_state[k_key])
            _load_state_dict_ignoring_legacy_bias(block.attn.processor.to_v_ip, ip_state[v_key])
            if null_key in ip_state and hasattr(block.attn.processor, "ip_null_tokens"):
                block.attn.processor.ip_null_tokens.load_state_dict(ip_state[null_key])
            if bias_key in ip_state and hasattr(block.attn.processor, "ip_soft_bias"):
                block.attn.processor.ip_soft_bias.load_state_dict(ip_state[bias_key])
            elif regional_ip_use_soft_bias:
                raise RuntimeError(f"Missing soft-bias weights in Cross V1 IP checkpoint: {bias_key}")
    _move_ip_adapter_modules(pipe.transformer, device=device, torch_dtype=dtype)
    set_ip_adapter_scale(pipe.transformer, ip_adapter_scale)

    ip_adapter_modules = _collect_ip_adapter_modules(pipe.transformer)

    # Load conditioning modules (hte, tissue_downsampler, nuclei_encoder, ref_encoder)
    modules = _load_condition_modules(
        checkpoint_path=checkpoint,
        uni_checkpoint_path=uni_checkpoint_path,
        device=device,
        torch_dtype=dtype,
        ref_encoder_config=ref_encoder_config,
    )

    return CrossV1InferenceBundle(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        checkpoint_path=checkpoint,
        uni_checkpoint_path=uni_checkpoint_path,
        device=device,
        torch_dtype=dtype,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        ip_adapter_scale=float(ip_adapter_scale),
        flux_pipeline=pipe,
        controlnet=controlnet,
        condition_modules=modules,
        control_spec=control_spec,
        ip_adapter_modules=ip_adapter_modules,
        ref_encoder=modules["ref_encoder"],
        regional_ip_adapter=regional_ip_adapter,
        regional_ip_strict=regional_ip_strict,
        regional_ip_token_mode=regional_ip_token_mode,
        regional_ip_label_mode=regional_ip_label_mode,
        cross_v1_ip_architecture=cross_v1_ip_architecture,
        regional_ip_use_soft_bias=regional_ip_use_soft_bias,
        regional_ip_soft_bias_init=regional_ip_soft_bias_init,
    )


def _resolve_saved_single_ip_layer_count(ip_state: dict[str, Any]) -> int:
    if "num_single_layers" in ip_state:
        return int(ip_state["num_single_layers"])
    indices = {
        int(key.split("_")[2])
        for key in ip_state
        if key.startswith("single_block_") and key.endswith(("_to_k_ip", "_to_v_ip"))
    }
    return len(indices)


def _load_state_dict_ignoring_legacy_bias(module: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    cleaned_state = {key: value for key, value in state_dict.items() if not str(key).endswith(".bias")}
    module.load_state_dict(cleaned_state)


def set_ip_adapter_scale(transformer: nn.Module, scale: float) -> None:
    """Set IP-Adapter scale on both double-stream and installed single-stream processors."""
    for blocks in (
        getattr(transformer, "transformer_blocks", []),
        getattr(transformer, "single_transformer_blocks", []),
    ):
        for block in blocks:
            processor = getattr(getattr(block, "attn", None), "processor", None)
            if processor is None or not hasattr(processor, "scale"):
                continue
            current = processor.scale
            if isinstance(current, list):
                processor.scale = [float(scale) for _ in current] or [float(scale)]
            elif isinstance(current, tuple):
                processor.scale = tuple(float(scale) for _ in current) or (float(scale),)
            elif torch.is_tensor(current):
                current.fill_(float(scale))
            else:
                processor.scale = float(scale)


def set_ip_soft_bias(transformer: nn.Module, value: float) -> dict[str, float | int | bool]:
    """Override learned global-soft-bias scalars on installed IP-Adapter processors.

    This only changes existing ``ip_soft_bias`` parameters. It does not enable
    soft-bias routing for checkpoints whose processors were installed without it.
    """
    bias_value = float(value)
    values: list[float] = []
    enabled_processors = 0
    processor_count = 0
    with torch.no_grad():
        for blocks in (
            getattr(transformer, "transformer_blocks", []),
            getattr(transformer, "single_transformer_blocks", []),
        ):
            for block in blocks:
                processor = getattr(getattr(block, "attn", None), "processor", None)
                soft_bias = getattr(processor, "ip_soft_bias", None)
                if soft_bias is None:
                    continue
                processor_count += 1
                if bool(getattr(processor, "use_soft_bias", False)):
                    enabled_processors += 1
                for parameter in soft_bias:
                    parameter.fill_(bias_value)
                    values.append(float(parameter.detach().float().mean().cpu().item()))
    if not values:
        return {
            "requested": bias_value,
            "applied": False,
            "parameter_count": 0,
            "processor_count": 0,
            "enabled_processor_count": 0,
        }
    return {
        "requested": bias_value,
        "applied": True,
        "parameter_count": len(values),
        "processor_count": processor_count,
        "enabled_processor_count": enabled_processors,
        "min": min(values),
        "mean": float(sum(values) / len(values)),
        "max": max(values),
    }


def _packed_flux_image_token_count(image: torch.Tensor, pipe) -> int:
    height, width = (int(v) for v in image.shape[-2:])
    vae_scale_factor = int(getattr(pipe, "vae_scale_factor", 8) or 8)
    latent_height = height // vae_scale_factor
    latent_width = width // vae_scale_factor
    return (latent_height // 2) * (latent_width // 2)


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


@torch.inference_mode()
def run_cross_v1_bundle(
    bundle: CrossV1InferenceBundle,
    reference_image: torch.Tensor,
    reference_tissue_mask: torch.Tensor,
    reference_nuclei_mask: torch.Tensor,
    target_tissue_mask: torch.Tensor,
    target_nuclei_mask: torch.Tensor,
    prompt: str,
    source_latent_init_strength: float = 0.0,
    mask_chord_scale: float = 0.0,
    mask_chord_use_gate: bool = False,
    mask_chord_gate_dilate_radius: int = 0,
    mask_chord_gate_feather_radius: int = 0,
    mask_chord_gate_outside_scale: float = 0.0,
    source_latent_init_image: torch.Tensor | None = None,
    seed: int = 42,
) -> Image.Image:
    source_latent_init_strength = _validate_source_latent_init_strength(source_latent_init_strength)
    mask_chord_scale = _validate_nonnegative_float(mask_chord_scale, "mask_chord_scale")
    # Encode reference image via UNI2-h + Perceiver resampler
    reference_batch = reference_image.unsqueeze(0).to(device=bundle.device, dtype=bundle.torch_dtype)
    if bundle.regional_ip_adapter:
        reference_tissue_batch = reference_tissue_mask.unsqueeze(0).to(device=bundle.device)
        reference_nuclei_batch = reference_nuclei_mask.unsqueeze(0).to(device=bundle.device)
        target_tissue_batch = target_tissue_mask.unsqueeze(0).to(device=bundle.device)
        target_nuclei_batch = target_nuclei_mask.unsqueeze(0).to(device=bundle.device)
        ref_features, region_token_labels = bundle.ref_encoder.encode_region_ip_tokens(
            reference_batch,
            reference_tissue_batch,
            nuclei_mask=reference_nuclei_batch,
            token_mode=bundle.regional_ip_token_mode,
            label_mode=bundle.regional_ip_label_mode,
        )
        query_token_count = _packed_flux_image_token_count(target_tissue_mask, bundle.flux_pipeline)
        query_region_labels = build_region_ip_token_labels(
            tissue_mask=target_tissue_batch,
            num_tokens=query_token_count,
            nuclei_mask=target_nuclei_batch,
            label_mode=bundle.regional_ip_label_mode,
        ).to(device=bundle.device)
        key_fallback_region_labels = _tissue_fallback_region_labels(
            region_token_labels,
            label_mode=bundle.regional_ip_label_mode,
        ).to(device=bundle.device)
        query_fallback_region_labels = resize_mask_to_token_labels(
            target_tissue_batch,
            query_token_count,
        ).to(device=bundle.device)
    else:
        ref_features = bundle.ref_encoder(reference_batch)
        region_token_labels = None
        query_region_labels = None
        key_fallback_region_labels = None
        query_fallback_region_labels = None
    ref_features = ref_features.to(device=bundle.device)
    ref_gate = bundle.ref_encoder.reference_presence_gate(
        reference_batch,
        device=bundle.device,
        dtype=next(bundle.flux_pipeline.transformer.encoder_hid_proj.parameters()).dtype,
    )
    ip_hidden_states = bundle.flux_pipeline.transformer.encoder_hid_proj([ref_features])
    ip_hidden_states = [
        hidden.to(device=bundle.device) * ref_gate.to(device=bundle.device, dtype=hidden.dtype)
        for hidden in ip_hidden_states
    ]

    # Build spatial control tensor (no reference_image_latent)
    target_tissue_feat = bundle.condition_modules["tissue_downsampler"](
        bundle.condition_modules["hte"](
            target_tissue_mask.unsqueeze(0).to(device=bundle.device)
        )
    ).to(dtype=bundle.torch_dtype)
    target_nuclei_feat = bundle.condition_modules["nuclei_encoder"](
        target_nuclei_mask.unsqueeze(0).to(device=bundle.device)
    ).to(dtype=bundle.torch_dtype)
    reference_tissue_feat = None
    reference_nuclei_feat = None
    needs_reference_mask_features = (
        mask_chord_scale > 0.0
        or bundle.control_spec.spatial_mode in {
            CROSS_V1_SPATIAL_REFERENCE_TARGET,
            CROSS_V1_SPATIAL_REFERENCE_TARGET_DELTA,
        }
    )
    if needs_reference_mask_features:
        reference_tissue_feat = bundle.condition_modules["tissue_downsampler"](
            bundle.condition_modules["hte"](
                reference_tissue_mask.unsqueeze(0).to(device=bundle.device)
            )
        ).to(dtype=bundle.torch_dtype)
        reference_nuclei_feat = bundle.condition_modules["nuclei_encoder"](
            reference_nuclei_mask.unsqueeze(0).to(device=bundle.device)
        ).to(dtype=bundle.torch_dtype)

    control_tensor = build_cross_v1_condition(
        reference_tissue_feat=reference_tissue_feat,
        reference_nuclei_feat=reference_nuclei_feat,
        target_tissue_feat=target_tissue_feat,
        target_nuclei_feat=target_nuclei_feat,
        spatial_mode=bundle.control_spec.spatial_mode,
    )
    source_control_tensor = None
    if mask_chord_scale > 0.0:
        if reference_tissue_feat is None or reference_nuclei_feat is None:
            raise ValueError("Cross V1 mask chord guidance requires reference mask features.")
        source_control_tensor = build_cross_v1_condition(
            reference_tissue_feat=reference_tissue_feat,
            reference_nuclei_feat=reference_nuclei_feat,
            target_tissue_feat=reference_tissue_feat,
            target_nuclei_feat=reference_nuclei_feat,
            spatial_mode=bundle.control_spec.spatial_mode,
        )

    change_mask = None
    if mask_chord_use_gate:
        change_mask = _build_mask_change_map(
            reference_tissue_mask=reference_tissue_mask,
            reference_nuclei_mask=reference_nuclei_mask,
            target_tissue_mask=target_tissue_mask,
            target_nuclei_mask=target_nuclei_mask,
        )

    source_latents = None
    if source_latent_init_strength > 0.0:
        source_image_for_latents = (
            reference_image
            if source_latent_init_image is None
            else source_latent_init_image
        )
        source_latents = _encode_images_to_latents(
            bundle.flux_pipeline.vae,
            source_image_for_latents.unsqueeze(0),
            bundle.torch_dtype,
        )

    output_size = tuple(int(v) for v in reference_image.shape[1:])
    joint_attention_kwargs = {"ip_hidden_states": ip_hidden_states}
    if bundle.regional_ip_adapter:
        joint_attention_kwargs.update(
            {
                "ip_adapter_masks": {
                    "key_region_labels": region_token_labels.to(device=bundle.device),
                    "query_region_labels": query_region_labels,
                    "key_fallback_region_labels": key_fallback_region_labels,
                    "query_fallback_region_labels": query_fallback_region_labels,
                    "strict": bool(bundle.regional_ip_strict),
                    "use_soft_bias": bool(bundle.regional_ip_use_soft_bias),
                },
            }
        )

    return _sample_with_flux_controlnet(
        pipe=bundle.flux_pipeline,
        controlnet=bundle.controlnet,
        prompt=prompt,
        control_tensor=control_tensor,
        source_control_tensor=source_control_tensor,
        source_latents=source_latents,
        source_latent_init_strength=source_latent_init_strength,
        mask_chord_scale=mask_chord_scale,
        mask_chord_change_mask=change_mask,
        mask_chord_gate_dilate_radius=mask_chord_gate_dilate_radius,
        mask_chord_gate_feather_radius=mask_chord_gate_feather_radius,
        mask_chord_gate_outside_scale=mask_chord_gate_outside_scale,
        output_size=output_size,
        device=bundle.device,
        torch_dtype=bundle.torch_dtype,
        num_inference_steps=bundle.num_inference_steps,
        guidance_scale=bundle.guidance_scale,
        controlnet_conditioning_scale=bundle.controlnet_conditioning_scale,
        joint_attention_kwargs=joint_attention_kwargs,
        seed=seed,
    )


@torch.inference_mode()
def run_cross_v1_controlnet_denoise_from_packed_latents(
    bundle: CrossV1InferenceBundle,
    *,
    initial_packed_latents: torch.Tensor,
    reference_image: torch.Tensor,
    reference_tissue_mask: torch.Tensor,
    reference_nuclei_mask: torch.Tensor,
    target_tissue_mask: torch.Tensor,
    target_nuclei_mask: torch.Tensor,
    prompt: str,
    output_size: tuple[int, int],
    timesteps: list[float],
    guidance_scale: float | None = None,
    controlnet_conditioning_scale: float | None = None,
    with_second_order: bool = True,
    controlnet_start_step: int = 0,
    kv_reference_features: dict[str, Any] | None = None,
    kv_inject_mode: str = "kv",
    kv_inject_strength: float = 0.2,
    kv_inject_start_step: int = 18,
    kv_inject_after_layer: int = 20,
    kv_inject_after_t: float | None = None,
    seed: int = 42,
    return_debug: bool = False,
) -> Image.Image | tuple[Image.Image, dict[str, Any]]:
    """Denoise RF-Solver inversion latents with Cross V1 ControlNet.

    By default this is the no-injection reconstruction path. When
    kv_reference_features is provided, late single-block image-token K/V are
    blended from the reference inversion during denoise.
    """
    if len(timesteps) < 2:
        raise ValueError("timesteps must contain at least two values")
    pipe = bundle.flux_pipeline
    controlnet = bundle.controlnet
    torch_device = torch.device(bundle.device)
    torch_dtype = bundle.torch_dtype
    guidance = bundle.guidance_scale if guidance_scale is None else float(guidance_scale)
    conditioning_scale = (
        bundle.controlnet_conditioning_scale
        if controlnet_conditioning_scale is None
        else float(controlnet_conditioning_scale)
    )

    reference_batch = reference_image.unsqueeze(0).to(device=bundle.device, dtype=bundle.torch_dtype)
    if bundle.regional_ip_adapter:
        reference_tissue_batch = reference_tissue_mask.unsqueeze(0).to(device=bundle.device)
        reference_nuclei_batch = reference_nuclei_mask.unsqueeze(0).to(device=bundle.device)
        target_tissue_batch = target_tissue_mask.unsqueeze(0).to(device=bundle.device)
        target_nuclei_batch = target_nuclei_mask.unsqueeze(0).to(device=bundle.device)
        ref_features, region_token_labels = bundle.ref_encoder.encode_region_ip_tokens(
            reference_batch,
            reference_tissue_batch,
            nuclei_mask=reference_nuclei_batch,
            token_mode=bundle.regional_ip_token_mode,
            label_mode=bundle.regional_ip_label_mode,
        )
        query_token_count = _packed_flux_image_token_count(target_tissue_mask, bundle.flux_pipeline)
        query_region_labels = build_region_ip_token_labels(
            tissue_mask=target_tissue_batch,
            num_tokens=query_token_count,
            nuclei_mask=target_nuclei_batch,
            label_mode=bundle.regional_ip_label_mode,
        ).to(device=bundle.device)
        key_fallback_region_labels = _tissue_fallback_region_labels(
            region_token_labels,
            label_mode=bundle.regional_ip_label_mode,
        ).to(device=bundle.device)
        query_fallback_region_labels = resize_mask_to_token_labels(
            target_tissue_batch,
            query_token_count,
        ).to(device=bundle.device)
    else:
        ref_features = bundle.ref_encoder(reference_batch)
        region_token_labels = None
        query_region_labels = None
        key_fallback_region_labels = None
        query_fallback_region_labels = None

    ref_features = ref_features.to(device=bundle.device)
    ref_gate = bundle.ref_encoder.reference_presence_gate(
        reference_batch,
        device=bundle.device,
        dtype=next(pipe.transformer.encoder_hid_proj.parameters()).dtype,
    )
    ip_hidden_states = pipe.transformer.encoder_hid_proj([ref_features])
    ip_hidden_states = [
        hidden.to(device=bundle.device) * ref_gate.to(device=bundle.device, dtype=hidden.dtype)
        for hidden in ip_hidden_states
    ]

    target_tissue_feat = bundle.condition_modules["tissue_downsampler"](
        bundle.condition_modules["hte"](
            target_tissue_mask.unsqueeze(0).to(device=bundle.device)
        )
    ).to(dtype=bundle.torch_dtype)
    target_nuclei_feat = bundle.condition_modules["nuclei_encoder"](
        target_nuclei_mask.unsqueeze(0).to(device=bundle.device)
    ).to(dtype=bundle.torch_dtype)
    reference_tissue_feat = None
    reference_nuclei_feat = None
    if bundle.control_spec.spatial_mode in {
        CROSS_V1_SPATIAL_REFERENCE_TARGET,
        CROSS_V1_SPATIAL_REFERENCE_TARGET_DELTA,
    }:
        reference_tissue_feat = bundle.condition_modules["tissue_downsampler"](
            bundle.condition_modules["hte"](
                reference_tissue_mask.unsqueeze(0).to(device=bundle.device)
            )
        ).to(dtype=bundle.torch_dtype)
        reference_nuclei_feat = bundle.condition_modules["nuclei_encoder"](
            reference_nuclei_mask.unsqueeze(0).to(device=bundle.device)
        ).to(dtype=bundle.torch_dtype)

    control_tensor = build_cross_v1_condition(
        reference_tissue_feat=reference_tissue_feat,
        reference_nuclei_feat=reference_nuclei_feat,
        target_tissue_feat=target_tissue_feat,
        target_nuclei_feat=target_nuclei_feat,
        spatial_mode=bundle.control_spec.spatial_mode,
    )

    joint_attention_kwargs = {"ip_hidden_states": ip_hidden_states}
    if bundle.regional_ip_adapter:
        joint_attention_kwargs.update(
            {
                "ip_adapter_masks": {
                    "key_region_labels": region_token_labels.to(device=bundle.device),
                    "query_region_labels": query_region_labels,
                    "key_fallback_region_labels": key_fallback_region_labels,
                    "query_fallback_region_labels": query_fallback_region_labels,
                    "strict": bool(bundle.regional_ip_strict),
                    "use_soft_bias": bool(bundle.regional_ip_use_soft_bias),
                },
            }
        )

    kv_injection_context = None
    kv_install_summary = None
    if kv_reference_features is not None:
        kv_injection_context = FluxKVInjectionContext(
            reference_features=kv_reference_features,
            mode=str(kv_inject_mode),
            strength=float(kv_inject_strength),
            start_step=int(kv_inject_start_step),
            after_layer=int(kv_inject_after_layer),
            inject_after_t=(
                None if kv_inject_after_t is None else float(kv_inject_after_t)
            ),
            image_token_count=int(kv_reference_features.get("image_token_count", 1024)),
        )
        kv_install_summary = install_flux_single_kv_injection(
            pipe.transformer,
            kv_injection_context,
        )

    return _denoise_packed_latents_with_flux_controlnet(
        pipe=pipe,
        controlnet=controlnet,
        initial_packed_latents=initial_packed_latents,
        prompt=prompt,
        control_tensor=control_tensor,
        output_size=output_size,
        timesteps=timesteps,
        device=bundle.device,
        torch_dtype=torch_dtype,
        guidance_scale=guidance,
        controlnet_conditioning_scale=conditioning_scale,
        joint_attention_kwargs=joint_attention_kwargs,
        with_second_order=with_second_order,
        controlnet_start_step=controlnet_start_step,
        kv_injection_context=kv_injection_context,
        kv_install_summary=kv_install_summary,
        seed=seed,
        return_debug=return_debug,
    )


# ---------------------------------------------------------------------------
# Internal helpers (adapted from pipeline.py for independence)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _denoise_packed_latents_with_flux_controlnet(
    *,
    pipe,
    controlnet,
    initial_packed_latents: torch.Tensor,
    prompt: str,
    control_tensor: torch.Tensor,
    output_size: tuple[int, int],
    timesteps: list[float],
    device: str,
    torch_dtype: torch.dtype,
    guidance_scale: float,
    controlnet_conditioning_scale: float,
    joint_attention_kwargs: dict | None,
    with_second_order: bool,
    controlnet_start_step: int = 0,
    kv_injection_context: FluxKVInjectionContext | None = None,
    kv_install_summary: dict[str, Any] | None = None,
    seed: int = 42,
    return_debug: bool = False,
) -> Image.Image | tuple[Image.Image, dict[str, Any]]:
    from diffusers import FluxControlNetPipeline

    torch_device = torch.device(device)
    height, width = output_size
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=[prompt], prompt_2=[prompt], device=torch_device,
    )
    if text_ids.dim() == 3:
        text_ids = text_ids[0]

    control_image = FluxControlNetPipeline._pack_latents(
        control_tensor,
        1,
        control_tensor.shape[1],
        control_tensor.shape[2],
        control_tensor.shape[3],
    )
    num_channels_latents = pipe.transformer.config.in_channels // 4
    latent_height = 2 * (int(height) // (pipe.vae_scale_factor * 2))
    latent_width = 2 * (int(width) // (pipe.vae_scale_factor * 2))
    expected_shape = (
        1,
        (latent_height // 2) * (latent_width // 2),
        num_channels_latents * 4,
    )
    latent_image_ids = FluxControlNetPipeline._prepare_latent_image_ids(
        1,
        latent_height // 2,
        latent_width // 2,
        torch_device,
        prompt_embeds.dtype,
    )
    latents = initial_packed_latents.to(device=torch_device, dtype=prompt_embeds.dtype)
    if tuple(latents.shape) != expected_shape:
        raise ValueError(
            "Initial RF packed latents do not match FluxControlNet packed latent shape: "
            f"initial={tuple(latents.shape)} expected={expected_shape}"
        )
    start_latent_summary = _tensor_debug_summary(latents)
    debug_summary: dict[str, Any] = {
        "latent_start_source": "initial_packed_latents",
        "uses_random_prepare_latents_start": False,
        "initial_packed_latents_shape": list(latents.shape),
        "expected_packed_latents_shape": list(expected_shape),
        "latent_image_ids_shape": list(latent_image_ids.shape),
        "latent_start": start_latent_summary,
        "timesteps_count": len(timesteps),
        "timesteps_head": [float(value) for value in timesteps[:5]],
        "timesteps_tail": [float(value) for value in timesteps[-5:]],
        "with_second_order": bool(with_second_order),
        "controlnet_start_step": int(controlnet_start_step),
        "controlnet_disabled_steps": min(int(controlnet_start_step), len(timesteps) - 1),
        "kv_injection": kv_install_summary,
        "seed_note": (
            "seed is kept for API compatibility; no random latent start is "
            "generated in this RF-inversion denoise path."
        ),
    }
    print(
        "ControlNet denoise latent start: source=initial_packed_latents "
        f"shape={list(latents.shape)} mean={start_latent_summary['mean']:.6f} "
        f"std={start_latent_summary['std']:.6f}"
    )

    controlnet_blocks_repeat = False if getattr(controlnet, "input_hint_block", None) is None else True
    guidance_vec = torch.full((latents.shape[0],), float(guidance_scale), device=torch_device, dtype=latents.dtype)
    controlnet_guidance = guidance_vec if controlnet.config.guidance_embeds else None
    transformer_guidance = guidance_vec if pipe.transformer.config.guidance_embeds else None

    def predict(
        hidden_states: torch.Tensor,
        t_value: float,
        *,
        use_controlnet: bool,
        step_index: int,
        second_order: bool,
    ) -> torch.Tensor:
        timestep = torch.full(
            (hidden_states.shape[0],),
            float(t_value),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        if kv_injection_context is not None:
            kv_injection_context.step_index = int(step_index)
            kv_injection_context.timestep = float(t_value)
            kv_injection_context.second_order = bool(second_order)
        if not use_controlnet:
            return pipe.transformer(
                hidden_states=hidden_states,
                timestep=timestep,
                guidance=transformer_guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                controlnet_block_samples=None,
                controlnet_single_block_samples=None,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=None,
                return_dict=False,
                controlnet_blocks_repeat=controlnet_blocks_repeat,
            )[0]
        return _predict_flux_controlnet_velocity(
            pipe=pipe,
            controlnet=controlnet,
            hidden_states=hidden_states,
            controlnet_cond=control_image,
            conditioning_scale=controlnet_conditioning_scale,
            timestep=timestep,
            controlnet_guidance=controlnet_guidance,
            transformer_guidance=transformer_guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            controlnet_blocks_repeat=controlnet_blocks_repeat,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    for step_index, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        use_controlnet = step_index >= int(controlnet_start_step)
        dt = float(t_prev) - float(t_curr)
        pred = predict(
            latents,
            float(t_curr),
            use_controlnet=use_controlnet,
            step_index=step_index,
            second_order=False,
        )
        if with_second_order:
            t_mid = float(t_curr) + dt / 2.0
            latents_mid = latents + (dt / 2.0) * pred
            pred_mid = predict(
                latents_mid,
                t_mid,
                use_controlnet=use_controlnet,
                step_index=step_index,
                second_order=True,
            )
            first_order = (pred_mid - pred) / (dt / 2.0)
            latents = latents + dt * pred + 0.5 * (dt ** 2) * first_order
        else:
            latents = latents + dt * pred

    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    debug_summary["final_unpacked_latents_before_vae_shift"] = _tensor_debug_summary(latents)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents.to(dtype=torch_dtype), return_dict=False)[0]
    output = pipe.image_processor.postprocess(image, output_type="pil")[0]
    if kv_injection_context is not None:
        debug_summary["kv_injection"] = kv_injection_context.summary()
    if return_debug:
        return output, debug_summary
    return output


def _tensor_debug_summary(tensor: torch.Tensor) -> dict[str, Any]:
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

def _resolve_torch_dtype(torch_dtype: torch.dtype | None, device: str) -> torch.dtype:
    if torch_dtype is not None:
        return torch_dtype
    return torch.bfloat16 if "cuda" in str(device).lower() else torch.float32


def _resolve_device(device: str | torch.device | None) -> str:
    value = str(device or "cuda").strip().lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cpu":
        return value
    if value == "cuda":
        _validate_cuda_device(value, index=0)
        return value
    if value.startswith("cuda:"):
        try:
            index = int(value.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device {device!r}; expected cuda or cuda:<index>.") from exc
        _validate_cuda_device(value, index=index)
        return value
    raise ValueError(f"Unsupported device {device!r}; choose auto, cpu, cuda, or cuda:<index>.")


def _validate_cuda_device(device: str, *, index: int) -> None:
    if index < 0:
        raise ValueError(f"Invalid CUDA device {device!r}; CUDA index must be non-negative.")
    if not torch.cuda.is_available():
        raise ValueError(f"CUDA device {device!r} was requested, but CUDA is not available.")
    visible_count = torch.cuda.device_count()
    if index >= visible_count:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        visible_msg = f" CUDA_VISIBLE_DEVICES={visible!r}." if visible is not None else ""
        raise ValueError(
            f"CUDA device {device!r} is not visible to this process; "
            f"torch sees {visible_count} CUDA device(s).{visible_msg} "
            "Use 'cuda'/'cuda:0' for the first visible GPU, or adjust CUDA_VISIBLE_DEVICES."
        )


def _validate_checkpoint_dir(checkpoint_path: str | Path) -> Path:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint}")
    for candidate in _cross_v1_checkpoint_candidates(checkpoint):
        if _is_eval_ready_cross_v1_checkpoint(candidate):
            if candidate != checkpoint:
                print(f"Resolved Cross V1 eval-ready checkpoint: {checkpoint} -> {candidate}")
            return candidate

    required = [
        "config.json",
        "phase5_conditioning.pt",
        "phase5_ip_adapter.pt",
    ]
    tried = []
    for candidate in _cross_v1_checkpoint_candidates(checkpoint):
        missing = [name for name in required if not (candidate / name).exists()]
        tried.append(f"{candidate} missing {missing}")
    raise FileNotFoundError(
        "Could not find an eval-ready Cross V1 checkpoint. The checkpoint "
        "directory must contain config.json, phase5_conditioning.pt, "
        "phase5_ip_adapter.pt, and ControlNet weights. This error often means "
        "you passed an accelerate resume-only directory, '.', '..', or an unset "
        "CONTROLNET_CHECKPOINT. Try the final training output dir, a raw "
        "checkpoint-N dir saved by the updated trainer, or checkpoint-N/ema. "
        "Checked: " + " | ".join(tried)
    )


def _cross_v1_checkpoint_candidates(checkpoint: Path) -> list[Path]:
    candidates = [
        checkpoint,
        checkpoint / "ema",
    ]
    if checkpoint.name.startswith("checkpoint-"):
        candidates.extend(
            [
                checkpoint.parent,
                checkpoint.parent / "ema",
            ]
        )
    if checkpoint.is_dir():
        checkpoint_children = sorted(
            [
                child
                for child in checkpoint.iterdir()
                if child.is_dir() and child.name.startswith("checkpoint-")
            ],
            key=lambda path: _checkpoint_step(path.name),
            reverse=True,
        )
        for child in checkpoint_children:
            candidates.extend([child, child / "ema"])

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def _checkpoint_step(name: str) -> int:
    try:
        return int(name.split("-", 1)[1])
    except Exception:
        return -1


def _is_eval_ready_cross_v1_checkpoint(path: Path) -> bool:
    if not path.is_dir():
        return False
    required = [
        path / "config.json",
        path / "phase5_conditioning.pt",
        path / "phase5_ip_adapter.pt",
    ]
    if not all(item.exists() for item in required):
        return False
    weight_patterns = (
        "diffusion_pytorch_model*.safetensors",
        "diffusion_pytorch_model*.bin",
        "pytorch_model.bin",
        "model.safetensors",
    )
    return any(path.glob(pattern) for pattern in weight_patterns)


def _move_ip_adapter_modules(transformer: nn.Module, *, device: str, torch_dtype: torch.dtype) -> None:
    train_dtype = torch.float32
    if hasattr(transformer, "encoder_hid_proj"):
        transformer.encoder_hid_proj.to(device=device, dtype=train_dtype)
    for blocks in (
        getattr(transformer, "transformer_blocks", []),
        getattr(transformer, "single_transformer_blocks", []),
    ):
        for block in blocks:
            processor = getattr(getattr(block, "attn", None), "processor", None)
            for name in ("to_k_ip", "to_v_ip", "ip_null_tokens", "ip_soft_bias"):
                module = getattr(processor, name, None)
                if module is not None:
                    module.to(device=device, dtype=train_dtype)


def _load_flux_controlnet_pipeline(
    *,
    pretrained_model_name_or_path: str | Path,
    checkpoint_path: Path,
    packed_channels: int,
    device: str,
    torch_dtype: torch.dtype,
) -> tuple:
    from diffusers import FluxControlNetModel, FluxControlNetPipeline

    controlnet_config = FluxControlNetModel.load_config(checkpoint_path)
    controlnet = FluxControlNetModel.from_config(controlnet_config)
    patch_controlnet_x_embedder(controlnet, packed_channels)
    controlnet.load_state_dict(_load_diffusers_model_state_dict(checkpoint_path), strict=True)
    controlnet.to(dtype=torch_dtype)

    pipe = FluxControlNetPipeline.from_pretrained(
        pretrained_model_name_or_path, controlnet=controlnet, torch_dtype=torch_dtype,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, controlnet


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


def _load_ref_encoder_config(checkpoint_path: Path) -> dict[str, Any]:
    state = _torch_load_weights(checkpoint_path / "phase5_conditioning.pt")
    config = dict(state.get("ref_encoder_config") or {})
    if "num_tokens" not in config:
        latent_queries = state.get("ref_encoder_latent_queries")
        config["num_tokens"] = int(latent_queries.shape[1]) if latent_queries is not None else 16
    if "num_perceiver_layers" not in config:
        config["num_perceiver_layers"] = _count_ref_perceiver_layers(
            state.get("ref_encoder_perceiver_layers", {})
        )
    config.setdefault("uni_embed_dim", 1536)
    config.setdefault("hidden_dim", 3072)
    config.setdefault("perceiver_heads", 8)
    config.setdefault("use_perceiver_self_attn", True)
    config.setdefault("skip_perceiver", False)
    config.setdefault("perceiver_cross_gate_init", None)
    config.setdefault("feature_layer", None)
    config.setdefault("regional_ip_token_mode", "spatial")
    config.setdefault("regional_ip_label_mode", "tissue")
    return {
        "uni_embed_dim": int(config["uni_embed_dim"]),
        "hidden_dim": int(config["hidden_dim"]),
        "num_tokens": int(config["num_tokens"]),
        "num_output_tokens": int(config.get("num_output_tokens", config["num_tokens"])),
        "num_perceiver_layers": int(config["num_perceiver_layers"]),
        "perceiver_heads": int(config["perceiver_heads"]),
        "use_perceiver_self_attn": bool(config["use_perceiver_self_attn"]),
        "skip_perceiver": bool(config["skip_perceiver"]),
        "perceiver_cross_gate_init": (
            None
            if config["perceiver_cross_gate_init"] is None
            else float(config["perceiver_cross_gate_init"])
        ),
        "feature_layer": None if config["feature_layer"] is None else int(config["feature_layer"]),
        "regional_ip_token_mode": normalize_region_ip_token_mode(config["regional_ip_token_mode"]),
        "regional_ip_label_mode": normalize_region_ip_label_mode(config["regional_ip_label_mode"]),
    }


def _load_cross_v1_control_spec(checkpoint_path: Path) -> CrossV1ControlSpec:
    state = _torch_load_weights(checkpoint_path / "phase5_conditioning.pt")
    saved_spec = state.get("cross_v1_control_spec") or {}
    return CrossV1ControlSpec(
        tissue_channels=int(saved_spec.get("tissue_channels", 64)),
        nuclei_channels=int(saved_spec.get("nuclei_channels", 16)),
        spatial_mode=str(
            saved_spec.get(
                "spatial_mode",
                state.get("cross_v1_spatial_mode", "reference_target"),
            )
        ),
    )


def _count_ref_perceiver_layers(state_dict: dict[str, torch.Tensor]) -> int:
    layer_indices = {
        int(key.split(".", 1)[0])
        for key in state_dict
        if key.split(".", 1)[0].isdigit()
    }
    if not layer_indices:
        return 2
    return max(layer_indices) + 1


def _load_condition_modules(
    *,
    checkpoint_path: Path,
    uni_checkpoint_path: str | Path,
    device: str,
    torch_dtype: torch.dtype,
    ref_encoder_config: dict[str, Any] | None = None,
) -> dict[str, nn.Module]:
    state = _torch_load_weights(checkpoint_path / "phase5_conditioning.pt")
    hte_state = state["hte"]
    tissue_state = state["tissue_downsampler"]
    nuclei_state = state["nuclei_encoder"]

    hte_dim = hte_state["parent_embeddings.weight"].shape[1]
    tissue_in = tissue_state["blocks.0.block.0.weight"].shape[1]
    tissue_hidden = tissue_state["blocks.0.block.0.weight"].shape[0]
    tissue_out = tissue_state[f"blocks.{_count_conv_blocks(tissue_state, 'blocks') - 1}.block.0.weight"].shape[0]
    nuclei_embed = nuclei_state["embedding.weight"].shape[1]
    nuclei_out = nuclei_state["downsampler.0.block.0.weight"].shape[0]
    nuclei_blocks = _count_conv_blocks(nuclei_state, "downsampler")

    modules: dict[str, nn.Module] = {
        "hte": HierarchicalTissueEmbedding(embedding_dim=hte_dim),
        "tissue_downsampler": TissueConditionDownsampler(
            in_channels=tissue_in,
            hidden_channels=tissue_hidden,
            out_channels=tissue_out,
            num_blocks=_count_conv_blocks(tissue_state, "blocks"),
        ),
        "nuclei_encoder": NucleiConditionEncoder(
            embedding_dim=nuclei_embed,
            out_channels=nuclei_out,
            num_blocks=nuclei_blocks,
        ),
    }

    for name, module in modules.items():
        module.load_state_dict(state[name])
        module.to(device=device, dtype=torch_dtype)
        module.eval()

    # Load ref_encoder
    from controlnet_train.modules.reference_image_encoder import ReferenceImageEncoder

    ref_config = ref_encoder_config or _load_ref_encoder_config(checkpoint_path)
    ref_encoder = ReferenceImageEncoder(
        uni_checkpoint_path=uni_checkpoint_path,
        uni_embed_dim=ref_config["uni_embed_dim"],
        hidden_dim=ref_config["hidden_dim"],
        num_tokens=ref_config["num_tokens"],
        num_perceiver_layers=ref_config["num_perceiver_layers"],
        perceiver_heads=ref_config["perceiver_heads"],
        use_perceiver_self_attn=ref_config.get("use_perceiver_self_attn", True),
        perceiver_cross_gate_init=ref_config.get("perceiver_cross_gate_init"),
        skip_perceiver=ref_config.get("skip_perceiver", False),
        feature_layer=ref_config.get("feature_layer"),
    )
    ref_encoder.proj_mlp.load_state_dict(state["ref_encoder_proj_mlp"])
    if not ref_encoder.skip_perceiver:
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
            warnings.warn(
                "phase5_conditioning.pt does not contain reference Perceiver weights; "
                "using the newly initialized Perceiver.",
                RuntimeWarning,
                stacklevel=2,
            )
    ref_encoder.to(device=device)
    ref_encoder.proj_mlp.to(device=device, dtype=torch.float32)
    ref_encoder.perceiver_layers.to(device=device, dtype=torch.float32)
    ref_encoder.perceiver_norm.to(device=device, dtype=torch.float32)
    ref_encoder.latent_queries.data = ref_encoder.latent_queries.data.to(
        device=device,
        dtype=torch.float32,
    )
    ref_encoder.uni.to(device=device, dtype=torch.float32)
    ref_encoder.eval()
    modules["ref_encoder"] = ref_encoder

    return modules


def _count_conv_blocks(state_dict: dict[str, torch.Tensor], prefix: str) -> int:
    return len(
        {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith(prefix) and key.endswith("block.0.weight")
        }
    )


@torch.inference_mode()
def _sample_with_flux_controlnet(
    *,
    pipe,
    controlnet,
    prompt: str,
    control_tensor: torch.Tensor,
    source_control_tensor: torch.Tensor | None = None,
    source_latents: torch.Tensor | None = None,
    source_latent_init_strength: float = 0.0,
    mask_chord_scale: float = 0.0,
    mask_chord_change_mask: torch.Tensor | None = None,
    mask_chord_gate_dilate_radius: int = 0,
    mask_chord_gate_feather_radius: int = 0,
    mask_chord_gate_outside_scale: float = 0.0,
    output_size: tuple[int, int],
    device: str,
    torch_dtype: torch.dtype,
    num_inference_steps: int,
    guidance_scale: float,
    controlnet_conditioning_scale: float,
    joint_attention_kwargs: dict | None = None,
    seed: int = 42,
) -> Image.Image:
    from diffusers import FluxControlNetPipeline
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

    torch_device = torch.device(device)
    height, width = output_size
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=[prompt], prompt_2=[prompt], device=torch_device,
    )
    if text_ids.dim() == 3:
        text_ids = text_ids[0]

    control_image = FluxControlNetPipeline._pack_latents(
        control_tensor, 1, control_tensor.shape[1],
        control_tensor.shape[2], control_tensor.shape[3],
    )
    source_control_image = None
    if source_control_tensor is not None:
        source_control_image = FluxControlNetPipeline._pack_latents(
            source_control_tensor,
            1,
            source_control_tensor.shape[1],
            source_control_tensor.shape[2],
            source_control_tensor.shape[3],
        )
    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents, latent_image_ids = pipe.prepare_latents(
        1, num_channels_latents, height, width,
        prompt_embeds.dtype, torch_device,
        generator=torch.Generator(device=torch_device).manual_seed(int(seed)),
        latents=None,
    )
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = _calculate_shift(
        image_seq_len=image_seq_len,
        base_seq_len=pipe.scheduler.config.get("base_image_seq_len", 256),
        max_seq_len=pipe.scheduler.config.get("max_image_seq_len", 4096),
        base_shift=pipe.scheduler.config.get("base_shift", 0.5),
        max_shift=pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, _ = retrieve_timesteps(
        pipe.scheduler, num_inference_steps, torch_device, sigmas=sigmas, mu=mu,
    )
    source_latent_init_strength = _validate_source_latent_init_strength(source_latent_init_strength)
    if source_latent_init_strength > 0.0:
        timesteps = _source_init_timesteps(timesteps, source_latent_init_strength)
        source_packed_latents = _pack_source_latents_for_sampling(
            source_latents=source_latents,
            expected_latents=latents,
            torch_dtype=prompt_embeds.dtype,
        )
        start_sigma = _sigma_for_timestep(
            pipe.scheduler,
            timesteps[:1].to(device=torch_device, dtype=torch.float32),
            n_dim=latents.ndim,
            dtype=latents.dtype,
        )
        latents = _prepare_source_noised_latents(
            source_latents=source_packed_latents,
            noise_latents=latents,
            sigma=start_sigma,
        )

    mask_chord_scale = _validate_nonnegative_float(mask_chord_scale, "mask_chord_scale")
    mask_chord_gate = None
    if mask_chord_change_mask is not None:
        mask_chord_gate = _build_packed_change_gate(
            change_mask=mask_chord_change_mask,
            latent_height=control_tensor.shape[2],
            latent_width=control_tensor.shape[3],
            packed_channels=latents.shape[-1],
            device=torch_device,
            dtype=latents.dtype,
            dilate_radius=mask_chord_gate_dilate_radius,
            feather_radius=mask_chord_gate_feather_radius,
            outside_scale=mask_chord_gate_outside_scale,
        )
    controlnet_blocks_repeat = False if getattr(controlnet, "input_hint_block", None) is None else True

    for timestep in timesteps:
        if mask_chord_scale > 0.0:
            if source_control_image is None:
                raise ValueError("mask_chord_scale > 0 requires source_control_tensor.")
            source_noise_pred, noise_pred = _predict_flux_controlnet_velocity_pair(
                pipe=pipe,
                controlnet=controlnet,
                hidden_states=latents,
                source_controlnet_cond=source_control_image,
                target_controlnet_cond=control_image,
                conditioning_scale=controlnet_conditioning_scale,
                timestep=timestep,
                guidance_scale=guidance_scale,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                controlnet_blocks_repeat=controlnet_blocks_repeat,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            delta = noise_pred - source_noise_pred
            if mask_chord_gate is not None:
                delta = delta * mask_chord_gate
            noise_pred = source_noise_pred + mask_chord_scale * delta
        else:
            expanded_timestep = timestep.expand(latents.shape[0]).to(latents.dtype)
            controlnet_guidance = None
            if controlnet.config.guidance_embeds:
                controlnet_guidance = torch.tensor([guidance_scale], device=torch_device).expand(latents.shape[0])
            transformer_guidance = None
            if pipe.transformer.config.guidance_embeds:
                transformer_guidance = torch.tensor([guidance_scale], device=torch_device).expand(latents.shape[0])
            noise_pred = _predict_flux_controlnet_velocity(
                pipe=pipe,
                controlnet=controlnet,
                hidden_states=latents,
                controlnet_cond=control_image,
                conditioning_scale=controlnet_conditioning_scale,
                timestep=expanded_timestep / 1000,
                controlnet_guidance=controlnet_guidance,
                transformer_guidance=transformer_guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                controlnet_blocks_repeat=controlnet_blocks_repeat,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        latents_dtype = latents.dtype
        latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
        if latents.dtype != latents_dtype:
            latents = latents.to(latents_dtype)

    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents.to(dtype=torch_dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def _encode_images_to_latents(vae, images: torch.Tensor, torch_dtype: torch.dtype) -> torch.Tensor:
    device = next(vae.parameters()).device
    images = images.to(device=device, dtype=torch_dtype)
    images = images * 2.0 - 1.0
    posterior = vae.encode(images).latent_dist
    latents = deterministic_latent_from_posterior(posterior)
    return (latents - vae.config.shift_factor) * vae.config.scaling_factor


def _predict_flux_controlnet_velocity_pair(
    *,
    pipe,
    controlnet,
    hidden_states: torch.Tensor,
    source_controlnet_cond: torch.Tensor,
    target_controlnet_cond: torch.Tensor,
    conditioning_scale: float,
    timestep: torch.Tensor,
    guidance_scale: float,
    pooled_projections: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    txt_ids: torch.Tensor,
    img_ids: torch.Tensor,
    controlnet_blocks_repeat: bool,
    joint_attention_kwargs: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(hidden_states.shape[0])
    batched_hidden_states = torch.cat([hidden_states, hidden_states], dim=0)
    batched_controlnet_cond = torch.cat([source_controlnet_cond, target_controlnet_cond], dim=0)
    batched_timestep = timestep.expand(batch_size * 2).to(dtype=hidden_states.dtype) / 1000
    batched_pooled = torch.cat([pooled_projections, pooled_projections], dim=0)
    batched_encoder_hidden = torch.cat([encoder_hidden_states, encoder_hidden_states], dim=0)
    device = hidden_states.device

    controlnet_guidance = None
    if controlnet.config.guidance_embeds:
        controlnet_guidance = torch.full(
            (batch_size * 2,),
            float(guidance_scale),
            device=device,
            dtype=hidden_states.dtype,
        )
    transformer_guidance = None
    if pipe.transformer.config.guidance_embeds:
        transformer_guidance = torch.full(
            (batch_size * 2,),
            float(guidance_scale),
            device=device,
            dtype=hidden_states.dtype,
        )

    batched_noise_pred = _predict_flux_controlnet_velocity(
        pipe=pipe,
        controlnet=controlnet,
        hidden_states=batched_hidden_states,
        controlnet_cond=batched_controlnet_cond,
        conditioning_scale=conditioning_scale,
        timestep=batched_timestep,
        controlnet_guidance=controlnet_guidance,
        transformer_guidance=transformer_guidance,
        pooled_projections=batched_pooled,
        encoder_hidden_states=batched_encoder_hidden,
        txt_ids=txt_ids,
        img_ids=img_ids,
        controlnet_blocks_repeat=controlnet_blocks_repeat,
        joint_attention_kwargs=_repeat_joint_attention_kwargs(joint_attention_kwargs, repeats=2),
    )
    source_noise_pred, target_noise_pred = batched_noise_pred.chunk(2, dim=0)
    return source_noise_pred, target_noise_pred


def _repeat_joint_attention_kwargs(kwargs: dict | None, *, repeats: int) -> dict | None:
    if kwargs is None:
        return None
    repeated: dict = {}
    for key, value in kwargs.items():
        if key == "ip_hidden_states" and isinstance(value, list):
            repeated[key] = [
                torch.cat([hidden_state] * repeats, dim=0)
                for hidden_state in value
            ]
        elif key == "ip_adapter_masks" and isinstance(value, dict):
            repeated[key] = {
                sub_key: (
                    torch.cat([sub_value] * repeats, dim=0)
                    if torch.is_tensor(sub_value) and sub_value.shape[:1] == (1,)
                    else sub_value
                )
                for sub_key, sub_value in value.items()
            }
        elif torch.is_tensor(value) and value.shape[:1] == (1,):
            repeated[key] = torch.cat([value] * repeats, dim=0)
        else:
            repeated[key] = value
    return repeated


def _predict_flux_controlnet_velocity(
    *,
    pipe,
    controlnet,
    hidden_states: torch.Tensor,
    controlnet_cond: torch.Tensor,
    conditioning_scale: float,
    timestep: torch.Tensor,
    controlnet_guidance: torch.Tensor | None,
    transformer_guidance: torch.Tensor | None,
    pooled_projections: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    txt_ids: torch.Tensor,
    img_ids: torch.Tensor,
    controlnet_blocks_repeat: bool,
    joint_attention_kwargs: dict | None = None,
) -> torch.Tensor:
    controlnet_block_samples, controlnet_single_block_samples = controlnet(
        hidden_states=hidden_states,
        controlnet_cond=controlnet_cond,
        controlnet_mode=None,
        conditioning_scale=conditioning_scale,
        timestep=timestep,
        guidance=controlnet_guidance,
        pooled_projections=pooled_projections,
        encoder_hidden_states=encoder_hidden_states,
        txt_ids=txt_ids,
        img_ids=img_ids,
        joint_attention_kwargs=None,
        return_dict=False,
    )
    return pipe.transformer(
        hidden_states=hidden_states,
        timestep=timestep,
        guidance=transformer_guidance,
        pooled_projections=pooled_projections,
        encoder_hidden_states=encoder_hidden_states,
        controlnet_block_samples=controlnet_block_samples,
        controlnet_single_block_samples=controlnet_single_block_samples,
        txt_ids=txt_ids,
        img_ids=img_ids,
        joint_attention_kwargs=dict(joint_attention_kwargs) if joint_attention_kwargs is not None else None,
        return_dict=False,
        controlnet_blocks_repeat=controlnet_blocks_repeat,
    )[0]


def _calculate_shift(
    *,
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
) -> float:
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    intercept = base_shift - slope * base_seq_len
    return image_seq_len * slope + intercept

'''
python -m controlnet_train.cli.eval_controlnet_flux_cross_v1   --pretrained-model-name-or-path /data/huggingface/FLUX.1-dev   --checkpoint /home/lyw/wqx-DL/flow-edit/FlowEdit-main/phase5_runs/controlnet_cross_v1/checkpoint-40000   --uni-checkpoint-path /home/lyw/wqx-DL/flow-edit/FlowEdit-main/UNI-2h/pytorch_model.bin   --metadata /home/lyw/wqx-DL/flow-edit/FlowEdit-main/phase5_runs/cross_meta/metadata_cross_val.json   --output-dir /home/lyw/wqx-DL/flow-edit/FlowEdit-main/phase5_runs/cross_v1_eval_10cases   --num-samples 10   --device cuda   --torch-dtype bf16   --prompt-source dataset
'''

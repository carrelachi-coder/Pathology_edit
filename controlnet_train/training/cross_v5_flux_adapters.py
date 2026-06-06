"""Cross V5 AdaLN adapters for real diffusers FLUX transformer blocks.

The generic V5 glue expects blocks to expose ``set_cross_v5_adaln_modulator``.
This module patches diffusers' current FLUX block forward paths in-place while
keeping the original module tree intact:

* double-stream ``FluxTransformerBlock``: image stream post ``norm2`` + FLUX
  MLP scale/shift, before ``ff``;
* single-stream ``FluxSingleTransformerBlock``: image-token slice immediately
  after ``norm`` on the MLP branch only; the attention branch keeps the
  unmodulated normalized states.

V5 conditioning is read from ``joint_attention_kwargs`` under the keys defined
below and stripped before the kwargs are forwarded to the attention processor.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from controlnet_train.modules.cross_v5_conditioning import CrossV5TissueBank
from controlnet_train.training.cross_v5_glue import (
    CrossV5AdaLNHookSpec,
    CrossV5AppearanceModulator,
    apply_cross_v5_adaln_to_hidden,
    install_cross_v5_adaln_hooks,
)


CROSS_V5_TARGET_CLASS_IDS_KEY = "cross_v5_target_class_ids"
CROSS_V5_TARGET_STRUCTURE_TOKENS_KEY = "cross_v5_target_structure_tokens"
CROSS_V5_BANK_KEY = "cross_v5_bank"
CROSS_V5_FALLBACK_PROTOTYPES_KEY = "cross_v5_fallback_prototypes"
CROSS_V5_IMAGE_TOKEN_START_KEY = "cross_v5_image_token_start"
CROSS_V5_ADALN_SCALE_KEY = "cross_v5_adaln_scale"

DOUBLE_IMAGE_POST_NORM_HOOK = "double_image_post_norm2_before_ff"
SINGLE_IMAGE_POST_NORM_HOOK = "single_image_post_norm_mlp_only"


@dataclass(frozen=True)
class CrossV5FluxAdapterInstallSummary:
    """Summary for FLUX-specific V5 block patching."""

    double_blocks: tuple[int, ...]
    double_strip_blocks: tuple[int, ...]
    single_blocks: tuple[int, ...]
    single_strip_blocks: tuple[int, ...]
    double_hook_point: str
    single_hook_point: str


def install_cross_v5_flux_adaln_adapters(
    *,
    transformer: nn.Module,
    modulator: CrossV5AppearanceModulator,
    double_block_indices: Sequence[int] = (-1,),
    single_block_indices: Sequence[int] = (),
    detach_bank: bool = False,
    require_nonzero_gamma: bool = True,
    require_conditioning: bool = True,
    strip_unselected_double_blocks: bool = True,
    strip_unselected_single_blocks: bool = True,
    double_hook_point: str = DOUBLE_IMAGE_POST_NORM_HOOK,
    single_hook_point: str = SINGLE_IMAGE_POST_NORM_HOOK,
) -> CrossV5FluxAdapterInstallSummary:
    """Patch real diffusers FLUX blocks and install V5 AdaLN modulation.

    The caller must pass V5 tensors through ``joint_attention_kwargs``:

    ``cross_v5_target_class_ids``: ``(B, N_img)`` target tissue IDs.
    ``cross_v5_target_structure_tokens``: optional ``(B, N_img, S)`` target
    structure tokens for SEAN-style spatial AdaLN.
    ``cross_v5_bank``: ``CrossV5TissueBank`` built from the reference image.
    ``cross_v5_fallback_prototypes``: optional prior prototypes.
    ``cross_v5_image_token_start``: optional text length for single-stream
    blocks that use the legacy pre-concatenated single-stream signature.
    ``cross_v5_adaln_scale``: optional temporary inference-time multiplier for
    the AdaLN delta, useful for diagnosing whether the reference path is active.
    """

    double_blocks = list(getattr(transformer, "transformer_blocks", []) or [])
    single_blocks = list(getattr(transformer, "single_transformer_blocks", []) or [])
    selected_double = _normalize_indices(double_block_indices, len(double_blocks), name="double")
    selected_single = _normalize_indices(single_block_indices, len(single_blocks), name="single")

    for index in selected_double:
        _patch_double_block(double_blocks[index], require_conditioning=require_conditioning)
    strip_double = ()
    if strip_unselected_double_blocks and (selected_double or selected_single):
        strip_double = tuple(index for index in range(len(double_blocks)) if index not in selected_double)
    for index in strip_double:
        _patch_double_block_strip_only(double_blocks[index])

    strip_single = ()
    if strip_unselected_single_blocks and (selected_double or selected_single):
        strip_single = tuple(index for index in range(len(single_blocks)) if index not in selected_single)

    for index in selected_single:
        _patch_single_block(single_blocks[index], require_conditioning=require_conditioning)
    for index in strip_single:
        _patch_single_block(single_blocks[index], require_conditioning=require_conditioning)

    if selected_double:
        install_cross_v5_adaln_hooks(
            transformer=transformer,
            modulator=modulator,
            spec=CrossV5AdaLNHookSpec(
                block_indices=selected_double,
                hook_point=double_hook_point,
                detach_bank=detach_bank,
                require_nonzero_gamma=require_nonzero_gamma,
            ),
            block_attr="transformer_blocks",
        )
    if selected_single:
        install_cross_v5_adaln_hooks(
            transformer=transformer,
            modulator=modulator,
            spec=CrossV5AdaLNHookSpec(
                block_indices=selected_single,
                hook_point=single_hook_point,
                detach_bank=detach_bank,
                require_nonzero_gamma=require_nonzero_gamma,
            ),
            block_attr="single_transformer_blocks",
        )

    return CrossV5FluxAdapterInstallSummary(
        double_blocks=tuple(selected_double),
        double_strip_blocks=tuple(strip_double),
        single_blocks=tuple(selected_single),
        single_strip_blocks=tuple(strip_single),
        double_hook_point=double_hook_point,
        single_hook_point=single_hook_point,
    )


def _patch_double_block(block: nn.Module, *, require_conditioning: bool) -> None:
    _ensure_flux_double_block_shape(block)
    _install_setter(block)
    block.cross_v5_require_conditioning = bool(require_conditioning)
    if getattr(block, "_cross_v5_forward_patched", False):
        return
    if getattr(block, "_cross_v5_strip_only_patched", False):
        block.forward = block._cross_v5_original_forward
        block._cross_v5_strip_only_patched = False
    block._cross_v5_original_forward = block.forward
    block.forward = MethodType(_cross_v5_double_block_forward, block)
    block._cross_v5_forward_patched = True


def _patch_double_block_strip_only(block: nn.Module) -> None:
    _ensure_flux_double_block_shape(block)
    if getattr(block, "_cross_v5_forward_patched", False) or getattr(block, "_cross_v5_strip_only_patched", False):
        return
    block._cross_v5_original_forward = block.forward
    block.forward = MethodType(_cross_v5_double_block_strip_only_forward, block)
    block._cross_v5_strip_only_patched = True


def _patch_single_block(block: nn.Module, *, require_conditioning: bool) -> None:
    _ensure_flux_single_block_shape(block)
    _install_setter(block)
    block.cross_v5_require_conditioning = bool(require_conditioning)
    if getattr(block, "_cross_v5_forward_patched", False):
        return
    block._cross_v5_original_forward = block.forward
    block.forward = MethodType(_cross_v5_single_block_forward, block)
    block._cross_v5_forward_patched = True


def _install_setter(block: nn.Module) -> None:
    if getattr(block, "set_cross_v5_adaln_modulator", None) is None:
        block.set_cross_v5_adaln_modulator = MethodType(_set_cross_v5_adaln_modulator, block)


def _set_cross_v5_adaln_modulator(
    self: nn.Module,
    modulator: CrossV5AppearanceModulator,
    *,
    hook_point: str,
    detach_bank: bool,
) -> None:
    self.cross_v5_adaln_modulator = modulator
    self.cross_v5_hook_point = str(hook_point)
    self.cross_v5_detach_bank = bool(detach_bank)


def _cross_v5_double_block_forward(
    self: nn.Module,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    image_rotary_emb=None,
    joint_attention_kwargs: Mapping[str, Any] | None = None,
):
    # Double-stream post-norm insertion point:
    # after self.norm2(hidden_states) and FLUX's own scale/shift, before self.ff(...).
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
        encoder_hidden_states, emb=temb
    )
    attn_kwargs, cross_v5_kwargs = _split_joint_kwargs(joint_attention_kwargs)
    attention_outputs = self.attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        image_rotary_emb=image_rotary_emb,
        **attn_kwargs,
    )

    if len(attention_outputs) == 2:
        attn_output, context_attn_output = attention_outputs
        ip_attn_output = None
    elif len(attention_outputs) == 3:
        attn_output, context_attn_output, ip_attn_output = attention_outputs
    else:
        raise ValueError(f"Unexpected FLUX attention output count: {len(attention_outputs)}.")

    hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    norm_hidden_states = _apply_cross_v5_from_kwargs(
        self,
        norm_hidden_states,
        cross_v5_kwargs,
        image_token_start=0,
    )
    ff_output = self.ff(norm_hidden_states)
    hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
    if ip_attn_output is not None:
        hidden_states = hidden_states + ip_attn_output

    encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
    norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
    context_ff_output = self.ff_context(norm_encoder_hidden_states)
    encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states


def _cross_v5_double_block_strip_only_forward(self: nn.Module, *args, **kwargs):
    if "joint_attention_kwargs" in kwargs:
        attention_kwargs, _ = _split_joint_kwargs(kwargs.get("joint_attention_kwargs"))
        kwargs = dict(kwargs)
        kwargs["joint_attention_kwargs"] = attention_kwargs or None
    return self._cross_v5_original_forward(*args, **kwargs)


def _cross_v5_single_block_forward(
    self: nn.Module,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor | None = None,
    temb: torch.FloatTensor | None = None,
    image_rotary_emb=None,
    joint_attention_kwargs: Mapping[str, Any] | None = None,
):
    # Single-stream post-norm insertion point:
    # after self.norm(...) on [text, image] tokens, on the MLP branch only.
    # New diffusers versions pass separated image/text streams to this block;
    # older versions pass the already concatenated stream. Support both.
    if encoder_hidden_states is not None and encoder_hidden_states.ndim != hidden_states.ndim:
        image_rotary_emb = temb
        temb = encoder_hidden_states
        encoder_hidden_states = None
    if temb is None:
        raise TypeError("FLUX single block V5 adapter requires temb.")

    attn_kwargs, cross_v5_kwargs = _split_joint_kwargs(joint_attention_kwargs)
    if encoder_hidden_states is not None:
        text_seq_len = int(encoder_hidden_states.shape[1])
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        image_token_start = text_seq_len
        return_split_streams = True
    else:
        image_token_start = _resolve_single_image_token_start(hidden_states, cross_v5_kwargs)
        return_split_streams = False

    residual = hidden_states
    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
    mlp_input = _apply_cross_v5_from_kwargs(
        self,
        norm_hidden_states,
        cross_v5_kwargs,
        image_token_start=image_token_start,
    )
    mlp_hidden_states = self.act_mlp(self.proj_mlp(mlp_input))
    attn_output = self.attn(
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
        **attn_kwargs,
    )

    hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
    hidden_states = gate.unsqueeze(1) * self.proj_out(hidden_states)
    hidden_states = residual + hidden_states
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    if return_split_streams:
        encoder_hidden_states = hidden_states[:, :text_seq_len, :]
        hidden_states = hidden_states[:, text_seq_len:, :]
        return encoder_hidden_states, hidden_states
    return hidden_states


def _apply_cross_v5_from_kwargs(
    block: nn.Module,
    hidden_states: torch.Tensor,
    cross_v5_kwargs: Mapping[str, Any],
    *,
    image_token_start: int,
) -> torch.Tensor:
    modulator = getattr(block, "cross_v5_adaln_modulator", None)
    bank = cross_v5_kwargs.get(CROSS_V5_BANK_KEY)
    target_class_ids = cross_v5_kwargs.get(CROSS_V5_TARGET_CLASS_IDS_KEY)
    if modulator is None:
        return hidden_states
    if bank is None or target_class_ids is None:
        if bool(getattr(block, "cross_v5_require_conditioning", False)):
            raise ValueError(
                "Cross V5 FLUX adapter is installed but joint_attention_kwargs is missing "
                f"{CROSS_V5_BANK_KEY!r} or {CROSS_V5_TARGET_CLASS_IDS_KEY!r}."
            )
        return hidden_states
    if not isinstance(bank, CrossV5TissueBank):
        raise TypeError(f"{CROSS_V5_BANK_KEY} must be a CrossV5TissueBank.")
    if not isinstance(target_class_ids, torch.Tensor):
        raise TypeError(f"{CROSS_V5_TARGET_CLASS_IDS_KEY} must be a tensor.")

    image_token_start = int(image_token_start)
    image_token_count = int(target_class_ids.shape[1])
    image_token_end = image_token_start + image_token_count
    if image_token_start < 0 or image_token_end > hidden_states.shape[1]:
        raise ValueError(
            f"V5 image token slice [{image_token_start}, {image_token_end}) is outside hidden token count "
            f"{hidden_states.shape[1]}."
        )
    image_hidden = hidden_states[:, image_token_start:image_token_end, :]
    target_structure_tokens = cross_v5_kwargs.get(CROSS_V5_TARGET_STRUCTURE_TOKENS_KEY)
    if target_structure_tokens is not None:
        if not isinstance(target_structure_tokens, torch.Tensor):
            raise TypeError(f"{CROSS_V5_TARGET_STRUCTURE_TOKENS_KEY} must be a tensor.")
        if target_structure_tokens.shape[:2] != target_class_ids.shape:
            raise ValueError(
                f"{CROSS_V5_TARGET_STRUCTURE_TOKENS_KEY} shape {tuple(target_structure_tokens.shape)} "
                f"must start with target_class_ids shape {tuple(target_class_ids.shape)}."
            )
    modulated = apply_cross_v5_adaln_to_hidden(
        hidden_states=image_hidden,
        target_class_ids=target_class_ids,
        bank=bank,
        modulator=modulator,
        fallback_prototypes=cross_v5_kwargs.get(CROSS_V5_FALLBACK_PROTOTYPES_KEY),
        target_structure_tokens=target_structure_tokens,
        detach_bank=bool(getattr(block, "cross_v5_detach_bank", False)),
        modulation_scale=cross_v5_kwargs.get(CROSS_V5_ADALN_SCALE_KEY, 1.0),
    )
    if image_token_start == 0 and image_token_end == hidden_states.shape[1]:
        return modulated
    return torch.cat(
        [
            hidden_states[:, :image_token_start, :],
            modulated,
            hidden_states[:, image_token_end:, :],
        ],
        dim=1,
    )


def _split_joint_kwargs(joint_attention_kwargs: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if joint_attention_kwargs is None:
        return {}, {}
    cross_keys = {
        CROSS_V5_TARGET_CLASS_IDS_KEY,
        CROSS_V5_TARGET_STRUCTURE_TOKENS_KEY,
        CROSS_V5_BANK_KEY,
        CROSS_V5_FALLBACK_PROTOTYPES_KEY,
        CROSS_V5_IMAGE_TOKEN_START_KEY,
        CROSS_V5_ADALN_SCALE_KEY,
    }
    attention_kwargs: dict[str, Any] = {}
    cross_v5_kwargs: dict[str, Any] = {}
    for key, value in dict(joint_attention_kwargs).items():
        if key in cross_keys:
            cross_v5_kwargs[key] = value
        else:
            attention_kwargs[key] = value
    return attention_kwargs, cross_v5_kwargs


def _resolve_single_image_token_start(
    hidden_states: torch.Tensor,
    cross_v5_kwargs: Mapping[str, Any],
) -> int:
    explicit = cross_v5_kwargs.get(CROSS_V5_IMAGE_TOKEN_START_KEY)
    if explicit is not None:
        return int(explicit)
    target_class_ids = cross_v5_kwargs.get(CROSS_V5_TARGET_CLASS_IDS_KEY)
    if isinstance(target_class_ids, torch.Tensor) and target_class_ids.ndim >= 2:
        return int(hidden_states.shape[1] - target_class_ids.shape[1])
    return 0


def _ensure_flux_double_block_shape(block: nn.Module) -> None:
    required = ("norm1", "norm1_context", "attn", "norm2", "ff", "norm2_context", "ff_context")
    missing = [name for name in required if not hasattr(block, name)]
    if missing:
        raise TypeError(f"Block does not look like a diffusers FluxTransformerBlock; missing {missing}.")


def _ensure_flux_single_block_shape(block: nn.Module) -> None:
    required = ("norm", "proj_mlp", "act_mlp", "attn", "proj_out")
    missing = [name for name in required if not hasattr(block, name)]
    if missing:
        raise TypeError(f"Block does not look like a diffusers FluxSingleTransformerBlock; missing {missing}.")


def _normalize_indices(indices: Sequence[int], total_blocks: int, *, name: str) -> tuple[int, ...]:
    if total_blocks <= 0:
        if indices:
            raise ValueError(f"No FLUX {name} blocks are available for V5 adapter installation.")
        return ()
    normalized: list[int] = []
    for raw in indices:
        index = int(raw)
        if index < 0:
            index = total_blocks + index
        if index < 0 or index >= total_blocks:
            raise ValueError(f"V5 {name} block index {raw} resolves to {index}, outside [0, {total_blocks}).")
        if index not in normalized:
            normalized.append(index)
    return tuple(normalized)


__all__ = [
    "CROSS_V5_ADALN_SCALE_KEY",
    "CROSS_V5_BANK_KEY",
    "CROSS_V5_FALLBACK_PROTOTYPES_KEY",
    "CROSS_V5_IMAGE_TOKEN_START_KEY",
    "CROSS_V5_TARGET_CLASS_IDS_KEY",
    "CROSS_V5_TARGET_STRUCTURE_TOKENS_KEY",
    "DOUBLE_IMAGE_POST_NORM_HOOK",
    "SINGLE_IMAGE_POST_NORM_HOOK",
    "CrossV5FluxAdapterInstallSummary",
    "install_cross_v5_flux_adaln_adapters",
]

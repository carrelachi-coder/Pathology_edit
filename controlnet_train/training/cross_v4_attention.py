"""Attention processor utilities for Cross V4 correspondence bias."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CrossV4AttentionInstallSummary:
    """Counts of FLUX attention processors replaced for V4."""

    double_blocks: int
    biased_double_blocks: tuple[int, ...]
    single_blocks: int


class FluxCrossV4AttnProcessor2_0:
    """FLUX attention processor with optional Cross V4 additive logits bias.

    The processor mirrors diffusers' ``FluxAttnProcessor2_0`` for the standard
    FLUX joint attention path. When ``apply_cross_v4_bias`` is true and
    ``cross_v4_bias`` is passed, it materializes attention logits and adds the
    bias only to the image-query/context-key slice:

    ``logits[:, :, context_len:, :context_len] += cross_v4_bias``.

    Processors installed on non-selected double blocks or single blocks accept
    the same kwargs but keep the default SDPA behavior.
    """

    def __init__(self, *, apply_cross_v4_bias: bool = False) -> None:
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxCrossV4AttnProcessor2_0 requires PyTorch 2.0 or newer.")
        self.apply_cross_v4_bias = bool(apply_cross_v4_bias)

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        cross_v4_bias: torch.Tensor | None = None,
        cross_v4_bias_scale: float | torch.Tensor = 1.0,
        cross_v4_diagnostics: dict | None = None,
        **kwargs,
    ) -> torch.FloatTensor:
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported FluxCrossV4AttnProcessor2_0 kwargs: {unsupported}")

        if not self.apply_cross_v4_bias or cross_v4_bias is None or encoder_hidden_states is None:
            return self._sdpa_forward(
                attn=attn,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
            )
        return self._biased_forward(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
            cross_v4_bias=cross_v4_bias,
            cross_v4_bias_scale=cross_v4_bias_scale,
            cross_v4_diagnostics=cross_v4_diagnostics,
        )

    def _project_joint_qkv(
        self,
        *,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
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

        context_len = 0
        image_len = int(hidden_states.shape[1])
        if encoder_hidden_states is not None:
            context_len = int(encoder_hidden_states.shape[1])
            context_query = attn.add_q_proj(encoder_hidden_states)
            context_key = attn.add_k_proj(encoder_hidden_states)
            context_value = attn.add_v_proj(encoder_hidden_states)

            context_query = context_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            context_key = context_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            context_value = context_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_added_q is not None:
                context_query = attn.norm_added_q(context_query)
            if attn.norm_added_k is not None:
                context_key = attn.norm_added_k(context_key)

            query = torch.cat([context_query, query], dim=2)
            key = torch.cat([context_key, key], dim=2)
            value = torch.cat([context_value, value], dim=2)
        return query, key, value, context_len, image_len

    def _apply_rotary(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        image_rotary_emb: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image_rotary_emb is None:
            return query, key
        from diffusers.models.embeddings import apply_rotary_emb

        return apply_rotary_emb(query, image_rotary_emb), apply_rotary_emb(key, image_rotary_emb)

    def _finish_output(
        self,
        *,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        context_len: int,
    ):
        if encoder_hidden_states is None:
            return hidden_states
        encoder_out, image_out = hidden_states[:, :context_len], hidden_states[:, context_len:]
        image_out = attn.to_out[0](image_out)
        image_out = attn.to_out[1](image_out)
        encoder_out = attn.to_add_out(encoder_out)
        return image_out, encoder_out

    def _sdpa_forward(
        self,
        *,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        image_rotary_emb: torch.Tensor | None,
    ):
        query, key, value, context_len, _image_len = self._project_joint_qkv(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )
        query, key = self._apply_rotary(query=query, key=key, image_rotary_emb=image_rotary_emb)
        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            query.shape[0],
            -1,
            query.shape[1] * query.shape[-1],
        )
        hidden_states = hidden_states.to(query.dtype)
        return self._finish_output(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            context_len=context_len,
        )

    def _biased_forward(
        self,
        *,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        image_rotary_emb: torch.Tensor | None,
        cross_v4_bias: torch.Tensor,
        cross_v4_bias_scale: float | torch.Tensor,
        cross_v4_diagnostics: dict | None,
    ):
        query, key, value, context_len, image_len = self._project_joint_qkv(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )
        query, key = self._apply_rotary(query=query, key=key, image_rotary_emb=image_rotary_emb)
        if cross_v4_bias.ndim == 3:
            cross_v4_bias = cross_v4_bias.unsqueeze(1)
        if cross_v4_bias.ndim != 4:
            raise ValueError(
                "cross_v4_bias must have shape (B,N_img,N_context) or (B,1,N_img,N_context), "
                f"got {tuple(cross_v4_bias.shape)}."
            )
        if cross_v4_bias.shape[0] != query.shape[0] or cross_v4_bias.shape[2] != image_len:
            raise ValueError(
                "cross_v4_bias batch/image dimensions must match attention, "
                f"got bias={tuple(cross_v4_bias.shape)} image_len={image_len} batch={query.shape[0]}."
            )
        if cross_v4_bias.shape[3] != context_len:
            raise ValueError(
                "cross_v4_bias context dimension must match encoder_hidden_states length, "
                f"got {cross_v4_bias.shape[3]} and {context_len}."
            )

        logits = torch.matmul(query, key.transpose(-2, -1)) * (1.0 / math.sqrt(query.shape[-1]))
        if attention_mask is not None:
            logits = logits + attention_mask
        scale = torch.as_tensor(cross_v4_bias_scale, device=logits.device, dtype=logits.dtype)
        logits[:, :, context_len : context_len + image_len, :context_len] += (
            cross_v4_bias.to(device=logits.device, dtype=logits.dtype) * scale
        )
        attention_probs = torch.softmax(logits.float(), dim=-1).to(dtype=value.dtype)
        _record_cross_v4_attention_diagnostics(
            attention_probs=attention_probs,
            context_len=context_len,
            image_len=image_len,
            diagnostics=cross_v4_diagnostics,
        )
        hidden_states = torch.matmul(attention_probs, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            query.shape[0],
            -1,
            query.shape[1] * query.shape[-1],
        )
        hidden_states = hidden_states.to(query.dtype)
        return self._finish_output(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            context_len=context_len,
        )


def install_cross_v4_attention_processors(
    transformer,
    *,
    biased_double_block_indices: Iterable[int] | None = None,
) -> CrossV4AttentionInstallSummary:
    """Install V4-compatible processors on all FLUX attention blocks.

    Only selected double transformer blocks materialize logits and apply
    ``cross_v4_bias``. All other double/single blocks accept the same kwargs and
    keep the default SDPA computation.
    """

    double_blocks = list(getattr(transformer, "transformer_blocks", []) or [])
    single_blocks = list(getattr(transformer, "single_transformer_blocks", []) or [])
    selected = _normalize_indices(biased_double_block_indices, len(double_blocks))
    for index, block in enumerate(double_blocks):
        block.attn.set_processor(FluxCrossV4AttnProcessor2_0(apply_cross_v4_bias=index in selected))
    for block in single_blocks:
        block.attn.set_processor(FluxCrossV4AttnProcessor2_0(apply_cross_v4_bias=False))
    return CrossV4AttentionInstallSummary(
        double_blocks=len(double_blocks),
        biased_double_blocks=tuple(sorted(selected)),
        single_blocks=len(single_blocks),
    )


def parse_cross_v4_block_indices(value: str | int | Iterable[int] | None, *, total_blocks: int | None = None) -> tuple[int, ...]:
    """Parse a block-index option such as ``"last"``, ``"-1"``, or ``"1,3"``."""

    if value is None:
        return (-1,)
    if isinstance(value, int):
        return (value,)
    if not isinstance(value, str):
        return tuple(int(item) for item in value)

    text = value.strip().lower().replace(" ", "")
    if not text or text in {"none", "off"}:
        return ()
    if text == "last":
        return (-1,)
    if text == "all":
        if total_blocks is None:
            raise ValueError("total_blocks is required to parse 'all'.")
        return tuple(range(total_blocks))
    return tuple(int(part) for part in text.split(",") if part)


def _normalize_indices(indices: Iterable[int] | None, total_blocks: int) -> set[int]:
    normalized: set[int] = set()
    for raw_index in parse_cross_v4_block_indices(indices):
        index = int(raw_index)
        if index < 0:
            index = total_blocks + index
        if index < 0 or index >= total_blocks:
            raise ValueError(
                f"Cross V4 biased double block index {raw_index} is out of range for {total_blocks} blocks."
            )
        normalized.add(index)
    return normalized


def _record_cross_v4_attention_diagnostics(
    *,
    attention_probs: torch.Tensor,
    context_len: int,
    image_len: int,
    diagnostics: dict | None,
) -> None:
    if diagnostics is None:
        return
    bucket_masks = diagnostics.get("bucket_masks")
    if not isinstance(bucket_masks, dict) or not bucket_masks:
        return
    attn_slice = attention_probs.detach()[:, :, context_len : context_len + image_len, :context_len].float()
    group_masks = dict(diagnostics.get("group_masks") or {})
    if "all" not in group_masks:
        group_masks["all"] = torch.ones(
            attn_slice.shape[0],
            attn_slice.shape[2],
            device=attn_slice.device,
            dtype=torch.bool,
        )
    record: dict[str, float] = {}
    for bucket_name, raw_bucket_mask in bucket_masks.items():
        bucket_mask = raw_bucket_mask.to(device=attn_slice.device, dtype=torch.bool)
        if bucket_mask.shape != (attn_slice.shape[0], attn_slice.shape[2], attn_slice.shape[3]):
            continue
        bucket_mass = (attn_slice * bucket_mask.unsqueeze(1)).sum(dim=-1)
        for group_name, raw_group_mask in group_masks.items():
            group_mask = raw_group_mask.to(device=attn_slice.device, dtype=torch.bool)
            if group_mask.shape != (attn_slice.shape[0], attn_slice.shape[2]):
                continue
            expanded_group = group_mask.unsqueeze(1).expand_as(bucket_mass)
            if not bool(expanded_group.any().item()):
                continue
            value = bucket_mass[expanded_group].mean().detach().cpu().item()
            record[f"cross_v4_attention_{group_name}_{bucket_name}"] = float(value)
    if record:
        diagnostics.setdefault("records", []).append(record)


__all__ = [
    "CrossV4AttentionInstallSummary",
    "FluxCrossV4AttnProcessor2_0",
    "install_cross_v4_attention_processors",
    "parse_cross_v4_block_indices",
]

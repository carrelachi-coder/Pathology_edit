"""Narrow runtime compatibility shims for the production inference stack."""

from __future__ import annotations

from functools import wraps

import torch
import torch.nn.functional as F


def install_sdpa_enable_gqa_compat() -> bool:
    """Let diffusers 0.35 call PyTorch 2.4 SDPA with ``enable_gqa``.

    PyTorch 2.5 added the keyword while diffusers 0.35 forwards it even when
    it is false. The production amax2 environment still uses PyTorch 2.4.1.
    Return ``True`` only when the compatibility wrapper is installed.
    """

    sdpa = F.scaled_dot_product_attention
    if getattr(sdpa, "_pathology_sdpa_enable_gqa_compat", False):
        return False
    if "enable_gqa" in (getattr(sdpa, "__doc__", "") or ""):
        return False

    @wraps(sdpa)
    def compatible_sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        if enable_gqa:
            query_heads = int(query.size(-3))
            key_heads = int(key.size(-3))
            value_heads = int(value.size(-3))
            if key_heads != value_heads or key_heads <= 0 or query_heads % key_heads:
                raise ValueError(
                    "Cannot emulate enable_gqa for incompatible query/key/value head counts: "
                    f"{query_heads}/{key_heads}/{value_heads}"
                )
            repeat = query_heads // key_heads
            key = key.repeat_interleave(repeat, dim=-3)
            value = value.repeat_interleave(repeat, dim=-3)
        return sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    compatible_sdpa._pathology_sdpa_enable_gqa_compat = True
    F.scaled_dot_product_attention = compatible_sdpa
    return True

#!/usr/bin/env python
"""Smoke test for Cross V5 adapters that patch FLUX double/single blocks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controlnet_train.modules.cross_v5_conditioning import CrossV5SpatialAdaLNModulator, CrossV5TissueBank  # noqa: E402
from controlnet_train.training.cross_v5_flux_adapters import (  # noqa: E402
    CROSS_V5_BANK_KEY,
    CROSS_V5_IMAGE_TOKEN_START_KEY,
    CROSS_V5_TARGET_CLASS_IDS_KEY,
    CROSS_V5_TARGET_STRUCTURE_TOKENS_KEY,
    install_cross_v5_flux_adaln_adapters,
)


class ToyAdaNorm(torch.nn.Module):
    def forward(self, hidden, emb):
        batch, _, dim = hidden.shape
        gate = torch.ones(batch, dim, device=hidden.device, dtype=hidden.dtype)
        shift = torch.zeros_like(gate)
        scale = torch.zeros_like(gate)
        return hidden, gate, shift, scale, gate


class ToySingleNorm(torch.nn.Module):
    def forward(self, hidden, emb):
        batch, _, dim = hidden.shape
        return hidden, torch.ones(batch, dim, device=hidden.device, dtype=hidden.dtype)


class ToyAttention(torch.nn.Module):
    def __init__(self, *, double: bool) -> None:
        super().__init__()
        self.double = double
        self.cross_keys_seen = False

    def forward(self, hidden_states, encoder_hidden_states=None, image_rotary_emb=None, **kwargs):
        self.cross_keys_seen = any(key.startswith("cross_v5_") for key in kwargs)
        if self.double:
            return hidden_states * 0.0, encoder_hidden_states * 0.0
        return hidden_states * 0.0


class ToyDoubleBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = ToyAdaNorm()
        self.norm1_context = ToyAdaNorm()
        self.attn = ToyAttention(double=True)
        self.norm2 = torch.nn.Identity()
        self.ff = torch.nn.Identity()
        self.norm2_context = torch.nn.Identity()
        self.ff_context = torch.nn.Identity()


class ToySingleBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = ToySingleNorm()
        self.proj_mlp = torch.nn.Linear(4, 4)
        self.act_mlp = torch.nn.Identity()
        self.attn = ToyAttention(double=False)
        self.proj_out = torch.nn.Linear(8, 4)


class ToyTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([ToyDoubleBlock()])
        self.single_transformer_blocks = torch.nn.ModuleList([ToySingleBlock()])


def _bank() -> CrossV5TissueBank:
    prototypes = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
    return CrossV5TissueBank(
        prototypes=prototypes,
        local_tokens=torch.zeros(1, 2, 1, 4),
        class_present=torch.ones(1, 2, dtype=torch.bool),
        class_mass=torch.ones(1, 2),
        token_class_ids=torch.zeros(1, 1, dtype=torch.long),
        token_class_confidence=torch.ones(1, 1),
    )


def main() -> None:
    torch.manual_seed(11)
    transformer = ToyTransformer()
    summary = install_cross_v5_flux_adaln_adapters(
        transformer=transformer,
        modulator=CrossV5SpatialAdaLNModulator(
            hidden_dim=4,
            prototype_dim=4,
            structure_dim=3,
            output_init_std=0.01,
        ),
        double_block_indices=(0,),
        single_block_indices=(0,),
    )
    kwargs = {
        CROSS_V5_TARGET_CLASS_IDS_KEY: torch.tensor([[0, 1, 0]]),
        CROSS_V5_TARGET_STRUCTURE_TOKENS_KEY: torch.tensor([[[1.0, 0.0, -1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]]]),
        CROSS_V5_BANK_KEY: _bank(),
    }

    hidden = torch.zeros(1, 3, 4)
    encoder = torch.zeros(1, 2, 4)
    _, double_out = transformer.transformer_blocks[0](
        hidden_states=hidden,
        encoder_hidden_states=encoder,
        temb=torch.zeros(1, 4),
        joint_attention_kwargs=kwargs,
    )
    if transformer.transformer_blocks[0].attn.cross_keys_seen:
        raise SystemExit("V5 kwargs leaked into double-block attention.")
    if torch.allclose(double_out, hidden):
        raise SystemExit("Double-block V5 modulation did not change image hidden states.")

    legacy_single_kwargs = dict(kwargs)
    legacy_single_kwargs[CROSS_V5_IMAGE_TOKEN_START_KEY] = 2
    legacy_single_out = transformer.single_transformer_blocks[0](
        hidden_states=torch.zeros(1, 5, 4),
        temb=torch.zeros(1, 4),
        joint_attention_kwargs=legacy_single_kwargs,
    )
    new_encoder_out, new_image_out = transformer.single_transformer_blocks[0](
        hidden_states=torch.zeros(1, 3, 4),
        encoder_hidden_states=torch.zeros(1, 2, 4),
        temb=torch.zeros(1, 4),
        joint_attention_kwargs=kwargs,
    )
    if transformer.single_transformer_blocks[0].attn.cross_keys_seen:
        raise SystemExit("V5 kwargs leaked into single-block attention.")
    missing_bank_raised = False
    try:
        transformer.transformer_blocks[0](
            hidden_states=hidden,
            encoder_hidden_states=encoder,
            temb=torch.zeros(1, 4),
            joint_attention_kwargs={},
        )
    except ValueError:
        missing_bank_raised = True
    if not missing_bank_raised:
        raise SystemExit("Missing V5 bank did not raise under strict adapter mode.")

    print(
        json.dumps(
            {
                "double_blocks": list(summary.double_blocks),
                "double_strip_blocks": list(summary.double_strip_blocks),
                "single_blocks": list(summary.single_blocks),
                "double_delta_abs_mean": float((double_out - hidden).detach().abs().mean().cpu().item()),
                "legacy_single_shape": list(legacy_single_out.shape),
                "new_single_encoder_shape": list(new_encoder_out.shape),
                "new_single_image_shape": list(new_image_out.shape),
                "missing_bank_raised": missing_bank_raised,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

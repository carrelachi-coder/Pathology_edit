#!/usr/bin/env python
"""Smoke test Cross V5 on a real diffusers FluxTransformer2DModel forward.

This does not load pretrained FLUX weights. It instantiates a tiny random
FluxTransformer2DModel so the smoke stays local and cheap while still checking
the real diffusers block signatures and full-model kwargs plumbing.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _prepare_local_diffusers_import() -> None:
    """Keep diffusers import local-cache-safe on this workstation."""

    cache_root = Path("/private/tmp/codex_hf_cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))

    # The installed diffusers version imports torchao quantizer symbols that
    # reference torch.float8 names. Older local torch builds may not expose
    # those attributes even though this smoke does not use quantization.
    for name in ("float8_e4m3fn", "float8_e5m2"):
        if not hasattr(torch, name):
            setattr(torch, name, torch.uint8)


_prepare_local_diffusers_import()

from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel  # noqa: E402

from controlnet_train.modules.cross_v5_conditioning import (  # noqa: E402
    CrossV5SpatialAdaLNModulator,
    CrossV5TissueBank,
    build_cross_v5_spatial_structure_tokens,
)
from controlnet_train.training.cross_v5_flux_adapters import (  # noqa: E402
    CROSS_V5_BANK_KEY,
    CROSS_V5_TARGET_CLASS_IDS_KEY,
    CROSS_V5_TARGET_STRUCTURE_TOKENS_KEY,
    install_cross_v5_flux_adaln_adapters,
)


def _bank(prototypes: torch.Tensor) -> CrossV5TissueBank:
    return CrossV5TissueBank(
        prototypes=prototypes,
        local_tokens=torch.zeros(prototypes.shape[0], prototypes.shape[1], 1, 2),
        class_present=torch.ones(prototypes.shape[:2], dtype=torch.bool),
        class_mass=torch.ones(prototypes.shape[:2]),
        token_class_ids=torch.zeros(prototypes.shape[0], 1, dtype=torch.long),
        token_class_confidence=torch.ones(prototypes.shape[0], 1),
    )


def main() -> None:
    torch.manual_seed(23)
    batch_size = 1
    token_height = 2
    token_width = 2
    image_tokens = token_height * token_width
    text_tokens = 2
    latent_channels = 4
    joint_attention_dim = 12
    pooled_projection_dim = 8
    hidden_dim = 16
    num_classes = 2

    transformer = FluxTransformer2DModel(
        patch_size=1,
        in_channels=latent_channels,
        out_channels=latent_channels,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=joint_attention_dim,
        pooled_projection_dim=pooled_projection_dim,
        guidance_embeds=False,
        axes_dims_rope=(2, 2, 4),
    )
    transformer.eval()
    noop_reference_transformer = copy.deepcopy(transformer).eval()
    noop_patched_transformer = copy.deepcopy(transformer).eval()

    target_class_ids_hw = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long)
    structure_tokens = build_cross_v5_spatial_structure_tokens(
        class_ids=target_class_ids_hw,
        num_classes=num_classes,
        token_height=token_height,
        token_width=token_width,
    )
    target_class_ids = target_class_ids_hw.reshape(batch_size, image_tokens)
    modulator = CrossV5SpatialAdaLNModulator(
        hidden_dim=hidden_dim,
        prototype_dim=4,
        structure_dim=structure_tokens.shape[-1],
        output_init_std=0.01,
        use_internal_norm=False,
    )
    summary = install_cross_v5_flux_adaln_adapters(
        transformer=transformer,
        modulator=modulator,
        double_block_indices=(0,),
        single_block_indices=(),
    )
    if summary.single_blocks:
        raise SystemExit("V5-min true-FLUX smoke should not install appearance modulation on single blocks.")
    if summary.single_strip_blocks != (0,):
        raise SystemExit("Single block was not patched to strip V5 kwargs in double-only mode.")

    hidden_states = torch.randn(batch_size, image_tokens, latent_channels) * 0.01
    encoder_hidden_states = torch.randn(batch_size, text_tokens, joint_attention_dim) * 0.01
    pooled_projections = torch.randn(batch_size, pooled_projection_dim) * 0.01
    timestep = torch.tensor([0.5])
    txt_ids = torch.zeros(text_tokens, 3)
    img_ids = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    bank_a = _bank(torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]))
    bank_b = _bank(torch.tensor([[[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]))

    noop_modulator = CrossV5SpatialAdaLNModulator(
        hidden_dim=hidden_dim,
        prototype_dim=4,
        structure_dim=structure_tokens.shape[-1],
        output_init_std=0.01,
        use_internal_norm=False,
    )
    with torch.no_grad():
        final = noop_modulator.mlp[-1]
        final.weight.zero_()
        final.bias.zero_()
    install_cross_v5_flux_adaln_adapters(
        transformer=noop_patched_transformer,
        modulator=noop_modulator,
        double_block_indices=(0,),
        single_block_indices=(),
        require_nonzero_gamma=False,
        require_conditioning=True,
    )

    def run(bank: CrossV5TissueBank) -> torch.Tensor:
        return transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            joint_attention_kwargs={
                CROSS_V5_TARGET_CLASS_IDS_KEY: target_class_ids,
                CROSS_V5_TARGET_STRUCTURE_TOKENS_KEY: structure_tokens,
                CROSS_V5_BANK_KEY: bank,
            },
            return_dict=False,
        )[0]

    with torch.no_grad():
        reference_noop = noop_reference_transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        patched_noop = noop_patched_transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            joint_attention_kwargs={
                CROSS_V5_TARGET_CLASS_IDS_KEY: target_class_ids,
                CROSS_V5_TARGET_STRUCTURE_TOKENS_KEY: structure_tokens,
                CROSS_V5_BANK_KEY: bank_a,
            },
            return_dict=False,
        )[0]
        output_a = run(bank_a)
        output_b = run(bank_b)

    noop_max_abs_diff = (reference_noop - patched_noop).detach().abs().max()
    if not torch.isfinite(noop_max_abs_diff) or noop_max_abs_diff.item() > 1e-6:
        raise SystemExit(
            "Zero-gamma V5 adapter changed true-FLUX output: "
            f"max_abs_diff={noop_max_abs_diff.item():.8g}"
        )

    if output_a.shape != (batch_size, image_tokens, latent_channels):
        raise SystemExit(f"Unexpected true-FLUX output shape: {tuple(output_a.shape)}.")
    bank_delta = (output_a - output_b).detach().abs().mean()
    if not torch.isfinite(bank_delta) or bank_delta.item() <= 1e-8:
        raise SystemExit("Changing the V5 bank did not affect the true-FLUX output.")

    missing_bank_raised = False
    try:
        with torch.no_grad():
            transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                joint_attention_kwargs={},
                return_dict=False,
            )
    except ValueError:
        missing_bank_raised = True
    if not missing_bank_raised:
        raise SystemExit("Missing V5 bank did not raise under strict true-FLUX adapter mode.")

    print(
        json.dumps(
            {
                "bank_swap_delta_abs_mean": float(bank_delta.cpu().item()),
                "double_blocks": list(summary.double_blocks),
                "noop_max_abs_diff": float(noop_max_abs_diff.cpu().item()),
                "output_shape": list(output_a.shape),
                "single_blocks": list(summary.single_blocks),
                "single_strip_blocks": list(summary.single_strip_blocks),
                "structure_shape": list(structure_tokens.shape),
                "missing_bank_raised": missing_bank_raised,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

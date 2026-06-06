#!/usr/bin/env python
"""Smoke the actual Cross V5 train-loop wiring without pretrained weights."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _prepare_local_diffusers_import() -> None:
    cache_root = Path("/private/tmp/codex_hf_cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
    for name in ("float8_e4m3fn", "float8_e5m2"):
        if not hasattr(torch, name):
            setattr(torch, name, torch.uint8)
    if not hasattr(torch, "compiler"):
        class _CompilerShim:
            @staticmethod
            def disable(fn=None, *args, **kwargs):
                del args, kwargs
                if fn is None:
                    return lambda inner: inner
                return fn

        torch.compiler = _CompilerShim()
    try:
        import transformers

        integrations = getattr(transformers, "integrations", None)
        if integrations is not None and not hasattr(integrations, "deepspeed"):
            class _DeepSpeedShim:
                @staticmethod
                def is_deepspeed_zero3_enabled() -> bool:
                    return False

            integrations.deepspeed = _DeepSpeedShim()
    except ModuleNotFoundError:
        pass


_prepare_local_diffusers_import()

from diffusers import FluxControlNetPipeline  # noqa: E402
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel  # noqa: E402

from controlnet_train.modules import (  # noqa: E402
    CrossV5PriorPrototypeBank,
    CrossV5RefBankBuilder,
    CrossV5SpatialAdaLNModulator,
    HierarchicalTissueEmbedding,
    NucleiConditionEncoder,
    TissueConditionDownsampler,
)
from controlnet_train.training.cross_v5_flux_adapters import (  # noqa: E402
    CROSS_V5_BANK_KEY,
    CROSS_V5_FALLBACK_PROTOTYPES_KEY,
    CROSS_V5_TARGET_CLASS_IDS_KEY,
    CROSS_V5_TARGET_STRUCTURE_TOKENS_KEY,
    install_cross_v5_flux_adaln_adapters,
)
from controlnet_train.training.cross_v5_glue import (  # noqa: E402
    CrossV5LatentDecodeConfig,
    CrossV5LossIntervals,
    CrossV5LossWeights,
    CrossV5StepContext,
    assemble_cross_v5_step_losses,
    decode_cross_v5_prediction_rgb,
)
from controlnet_train.training.cross_v5_losses import CrossV5AppearanceLossConfig  # noqa: E402
from controlnet_train.training.flux_phase5_cross_v3 import (  # noqa: E402
    _build_cross_v5_control_batch,
    _fine_tissue_to_coarse,
    _prepare_packed_latent_image_ids,
)


class _LatentPosterior:
    def __init__(self, latents: torch.Tensor) -> None:
        self._latents = latents

    def sample(self) -> torch.Tensor:
        return self._latents


class _EncodeOutput:
    def __init__(self, latents: torch.Tensor) -> None:
        self.latent_dist = _LatentPosterior(latents)


class ToyVAE(torch.nn.Module):
    def __init__(self, latent_channels: int = 4) -> None:
        super().__init__()
        self.proj = torch.nn.Conv2d(3, latent_channels, kernel_size=1)
        self.out = torch.nn.Conv2d(latent_channels, 3, kernel_size=1)
        self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)

    def encode(self, images: torch.Tensor) -> _EncodeOutput:
        latents = self.proj(F.adaptive_avg_pool2d(images, output_size=(4, 4)))
        return _EncodeOutput(latents)

    def decode(self, latents: torch.Tensor, return_dict: bool = False):
        rgb = self.out(F.interpolate(latents, size=(16, 16), mode="nearest")).tanh()
        return (rgb,) if not return_dict else SimpleNamespace(sample=rgb)


def main() -> None:
    torch.manual_seed(29)
    batch_size = 1
    latent_channels = 4
    packed_channels = latent_channels * 4
    hidden_dim = 16
    modules = {
        "hte": HierarchicalTissueEmbedding(embedding_dim=8),
        "tissue_downsampler": TissueConditionDownsampler(in_channels=8, hidden_channels=4, num_blocks=2),
        "nuclei_encoder": NucleiConditionEncoder(embedding_dim=4, out_channels=4, num_blocks=2),
        "cross_v5_ref_bank_builder": CrossV5RefBankBuilder(num_classes=8, local_tokens_per_class=2),
        "cross_v5_prior_bank": CrossV5PriorPrototypeBank(num_classes=8, prototype_dim=4, init_std=0.02),
        "cross_v5_adaln_modulator": CrossV5SpatialAdaLNModulator(
            hidden_dim=hidden_dim,
            prototype_dim=4,
            structure_dim=14,
            output_init_std=0.01,
            use_internal_norm=False,
        ),
    }
    vae = ToyVAE(latent_channels=latent_channels)
    transformer = FluxTransformer2DModel(
        patch_size=1,
        in_channels=packed_channels,
        out_channels=packed_channels,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=hidden_dim,
        pooled_projection_dim=8,
        guidance_embeds=False,
        axes_dims_rope=(2, 2, 4),
    )
    install_summary = install_cross_v5_flux_adaln_adapters(
        transformer=transformer,
        modulator=modules["cross_v5_adaln_modulator"],
        double_block_indices=(0,),
        single_block_indices=(),
    )

    image = torch.rand(batch_size, 3, 16, 16)
    reference = image.roll(shifts=2, dims=-1)
    tissue = torch.zeros(batch_size, 16, 16, dtype=torch.long)
    tissue[:, :, 8:] = 3
    nuclei = torch.zeros(batch_size, 16, 16, dtype=torch.long)
    nuclei[:, 4:10, 4:10] = 1
    drop_ids = [int(value) for value in _fine_tissue_to_coarse(tissue).unique().tolist()]
    batch = {
        "target_image": image,
        "reference_image": reference,
        "target_tissue_mask": tissue,
        "reference_tissue_mask": tissue.roll(shifts=2, dims=-1),
        "target_nuclei_mask": nuclei,
        "reference_nuclei_mask": nuclei.roll(shifts=2, dims=-1),
        "v5_reference_bank_drop_tissue_ids": [drop_ids],
    }
    (
        pixel_latents,
        _control_tensor,
        bank,
        target_class_ids,
        structure_tokens,
        fallback_prototypes,
        _feature_stats,
    ) = _build_cross_v5_control_batch(
        batch=batch,
        modules=modules,
        vae=vae,
        weight_dtype=torch.float32,
    )
    packed_latents = FluxControlNetPipeline._pack_latents(
        pixel_latents,
        batch_size,
        pixel_latents.shape[1],
        pixel_latents.shape[2],
        pixel_latents.shape[3],
    )
    noise = torch.randn_like(packed_latents)
    sigma = torch.full((batch_size, 1, 1), 0.3)
    noisy = (1.0 - sigma) * packed_latents + sigma * noise
    img_ids = _prepare_packed_latent_image_ids(
        packed_height=pixel_latents.shape[2] // 2,
        packed_width=pixel_latents.shape[3] // 2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    text = torch.zeros(batch_size, 2, hidden_dim)
    txt_ids = torch.zeros(2, 3)
    pooled = torch.zeros(batch_size, 8)
    pred = transformer(
        hidden_states=noisy,
        timestep=torch.full((batch_size,), 0.3),
        pooled_projections=pooled,
        encoder_hidden_states=text,
        txt_ids=txt_ids,
        img_ids=img_ids,
        joint_attention_kwargs={
            CROSS_V5_TARGET_CLASS_IDS_KEY: target_class_ids,
            CROSS_V5_TARGET_STRUCTURE_TOKENS_KEY: structure_tokens,
            CROSS_V5_BANK_KEY: bank,
            CROSS_V5_FALLBACK_PROTOTYPES_KEY: fallback_prototypes,
        },
        return_dict=False,
    )[0]
    denoise = (pred - (noise - packed_latents)).pow(2).mean()
    rgb = decode_cross_v5_prediction_rgb(
        vae=vae,
        noisy_latents=noisy,
        model_prediction=pred,
        sigma=sigma,
        config=CrossV5LatentDecodeConfig(
            prediction_type="velocity",
            packed_latents=True,
            latent_channels=latent_channels,
            latent_height=pixel_latents.shape[2],
            latent_width=pixel_latents.shape[3],
            clamp_rgb=True,
        ),
    )
    context = CrossV5StepContext(
        prediction_rgb=rgb,
        reference_rgb=reference,
        target_tissue_mask=_fine_tissue_to_coarse(tissue),
        reference_tissue_mask=_fine_tissue_to_coarse(tissue.roll(shifts=2, dims=-1)),
        target_nuclei_mask=nuclei,
        target_nuclei_binary=(nuclei > 0).float(),
    )
    bundle = assemble_cross_v5_step_losses(
        denoise_loss=denoise,
        context=context,
        weights=CrossV5LossWeights(denoise=1.0, appearance=0.75, geometry=0.0),
        global_step=0,
        timestep=torch.tensor([300.0]),
        intervals=CrossV5LossIntervals(appearance=1, geometry=0),
        appearance_config=CrossV5AppearanceLossConfig(min_pixels=4, texture_weight=0.0),
    )
    bundle.total.backward()
    adaln_grad = modules["cross_v5_adaln_modulator"].mlp[-1].weight.grad
    prior_grad = modules["cross_v5_prior_bank"].prototypes.grad
    if adaln_grad is None or float(adaln_grad.abs().mean().item()) <= 0.0:
        raise SystemExit("Cross V5 AdaLN modulator did not receive gradients.")
    if prior_grad is None:
        raise SystemExit("Cross V5 prior bank did not stay connected to the graph.")
    print(
        json.dumps(
            {
                "adaln_grad_abs_mean": float(adaln_grad.abs().mean().item()),
                "appearance_total": float(bundle.components["appearance_total"].detach().item()),
                "bank_present_classes": int(bank.class_present.sum().item()),
                "gate_appearance": int(bundle.components["gate_appearance"]),
                "output_shape": list(pred.shape),
                "single_strip_blocks": list(install_summary.single_strip_blocks),
                "total": float(bundle.total.detach().item()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

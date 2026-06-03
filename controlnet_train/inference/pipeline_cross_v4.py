"""Inference helpers for Cross V4 Flux ControlNet."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from controlnet_train.modules import (
    FixedOneHotTissueEncoder,
    HierarchicalTissueEmbedding,
    NucleiConditionEncoder,
    TissueConditionDownsampler,
)
from controlnet_train.modules.cross_v4_conditioning import (
    CROSS_V4_PROMPT,
    CROSS_V4_REFERENCE_WITH_REF,
    CROSS_V4_REFERENCE_ZERO_REF,
    CrossV4ControlSpec,
    CrossV4CorrespondenceBiasConfig,
    CrossV4PriorTokenBank,
    CrossV4ReferenceContextEncoder,
    CrossV4ReferenceSpec,
    append_cross_v4_context,
    apply_cross_v4_reference_encoding_mode,
    build_cross_v4_control_condition,
    build_cross_v4_correspondence_bias,
    build_cross_v4_token_metadata,
    deterministic_latent_from_posterior,
)
from controlnet_train.training.conditioning import patch_controlnet_x_embedder
from controlnet_train.training.cross_v4_attention import (
    install_cross_v4_attention_processors,
    parse_cross_v4_block_indices,
)
from controlnet_train.inference.pipeline_cross_v3 import (
    _calculate_shift,
    _count_conv_blocks,
    _load_diffusers_model_state_dict,
    _resolve_device,
    _resolve_torch_dtype,
    _target_hte,
    _target_tissue_downsampler,
    _torch_load_weights,
    _validate_checkpoint_dir,
)


@dataclass
class CrossV4InferenceBundle:
    pretrained_model_name_or_path: str | Path
    checkpoint_path: Path
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.bfloat16
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    controlnet_conditioning_scale: float = 1.0
    flux_pipeline: object | None = None
    controlnet: object | None = None
    condition_modules: dict[str, nn.Module] = field(default_factory=dict)
    control_spec: CrossV4ControlSpec = field(default_factory=CrossV4ControlSpec)
    reference_spec: CrossV4ReferenceSpec = field(default_factory=CrossV4ReferenceSpec)
    attention_bias_config: dict = field(default_factory=dict)


def load_cross_v4_bundle(
    *,
    pretrained_model_name_or_path: str | Path,
    checkpoint_path: str | Path,
    device: str = "cuda",
    torch_dtype: torch.dtype | None = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    controlnet_conditioning_scale: float = 1.0,
) -> CrossV4InferenceBundle:
    device = _resolve_device(device)
    dtype = _resolve_torch_dtype(torch_dtype, device)
    checkpoint = _validate_checkpoint_dir(checkpoint_path)
    state = _torch_load_weights(checkpoint / "phase5_conditioning.pt")
    control_spec = _load_cross_v4_control_spec(state)
    reference_spec = _load_cross_v4_reference_spec(state)
    pipe, controlnet = _load_flux_controlnet_pipeline(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        checkpoint_path=checkpoint,
        packed_channels=control_spec.packed_channels,
        device=device,
        torch_dtype=dtype,
    )
    attention_bias_config = dict(state.get("cross_v4_attention_bias") or {})
    biased_blocks = parse_cross_v4_block_indices(
        attention_bias_config.get("biased_double_block_indices", "last"),
        total_blocks=len(getattr(pipe.transformer, "transformer_blocks", []) or []),
    )
    install_cross_v4_attention_processors(pipe.transformer, biased_double_block_indices=biased_blocks)
    modules = _load_cross_v4_condition_modules(
        state=state,
        reference_spec=reference_spec,
        device=device,
        torch_dtype=dtype,
    )
    return CrossV4InferenceBundle(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        checkpoint_path=checkpoint,
        device=device,
        torch_dtype=dtype,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        flux_pipeline=pipe,
        controlnet=controlnet,
        condition_modules=modules,
        control_spec=control_spec,
        reference_spec=reference_spec,
        attention_bias_config=attention_bias_config,
    )


@torch.inference_mode()
def run_cross_v4_bundle(
    bundle: CrossV4InferenceBundle,
    reference_image: torch.Tensor,
    reference_tissue_mask: torch.Tensor,
    reference_nuclei_mask: torch.Tensor,
    target_tissue_mask: torch.Tensor,
    target_nuclei_mask: torch.Tensor,
    prompt: str = CROSS_V4_PROMPT,
    reference_condition_mode: str = CROSS_V4_REFERENCE_WITH_REF,
) -> Image.Image:
    control_tensor, reference_encoding, target_metadata = _build_cross_v4_inference_conditions(
        bundle,
        reference_image=reference_image,
        reference_tissue_mask=reference_tissue_mask,
        reference_nuclei_mask=reference_nuclei_mask,
        target_tissue_mask=target_tissue_mask,
        target_nuclei_mask=target_nuclei_mask,
        reference_condition_mode=reference_condition_mode,
    )
    return _sample_with_flux_controlnet_v4(
        bundle=bundle,
        prompt=CROSS_V4_PROMPT,
        control_tensor=control_tensor,
        reference_encoding=reference_encoding,
        target_metadata=target_metadata,
        output_size=tuple(int(v) for v in reference_image.shape[1:]),
    )


def _build_cross_v4_inference_conditions(
    bundle: CrossV4InferenceBundle,
    *,
    reference_image: torch.Tensor,
    reference_tissue_mask: torch.Tensor,
    reference_nuclei_mask: torch.Tensor,
    target_tissue_mask: torch.Tensor,
    target_nuclei_mask: torch.Tensor,
    reference_condition_mode: str,
) -> tuple[torch.Tensor, object, object]:
    reference_latent = _encode_images_to_latents(
        bundle.flux_pipeline.vae,
        reference_image.unsqueeze(0),
        bundle.torch_dtype,
    )
    ref_tissue_feat = bundle.condition_modules["tissue_downsampler"](
        bundle.condition_modules["hte"](
            reference_tissue_mask.unsqueeze(0).to(device=bundle.device)
        )
    ).to(dtype=bundle.torch_dtype)
    ref_nuclei_feat = bundle.condition_modules["nuclei_encoder"](
        reference_nuclei_mask.unsqueeze(0).to(device=bundle.device)
    ).to(dtype=bundle.torch_dtype)
    tar_tissue_feat = _target_tissue_downsampler(bundle.condition_modules)(
        _target_hte(bundle.condition_modules)(
            target_tissue_mask.unsqueeze(0).to(device=bundle.device)
        )
    ).to(dtype=bundle.torch_dtype)
    tar_nuclei_feat = bundle.condition_modules["nuclei_encoder"](
        target_nuclei_mask.unsqueeze(0).to(device=bundle.device)
    ).to(dtype=bundle.torch_dtype)
    control_tensor = build_cross_v4_control_condition(
        tar_tissue_feat=tar_tissue_feat,
        tar_nuclei_feat=tar_nuclei_feat,
    )
    reference_encoding = bundle.condition_modules["reference_context_encoder"](
        z_ref=reference_latent,
        ref_tissue_feat=ref_tissue_feat,
        ref_nuclei_feat=ref_nuclei_feat,
        ref_tissue_ids=reference_tissue_mask.unsqueeze(0).to(device=bundle.device),
        ref_nuclei_ids=reference_nuclei_mask.unsqueeze(0).to(device=bundle.device),
    )
    reference_encoding = apply_cross_v4_reference_encoding_mode(reference_encoding, reference_condition_mode)
    target_metadata = build_cross_v4_token_metadata(
        tissue_ids=target_tissue_mask.unsqueeze(0).to(device=bundle.device),
        nuclei_ids=target_nuclei_mask.unsqueeze(0).to(device=bundle.device),
        token_height=reference_latent.shape[2] // 2,
        token_width=reference_latent.shape[3] // 2,
    )
    return control_tensor, reference_encoding, target_metadata


@torch.inference_mode()
def _sample_with_flux_controlnet_v4(
    *,
    bundle: CrossV4InferenceBundle,
    prompt: str,
    control_tensor: torch.Tensor,
    reference_encoding,
    target_metadata,
    output_size: tuple[int, int],
) -> Image.Image:
    from diffusers import FluxControlNetPipeline
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
    import numpy as np

    pipe = bundle.flux_pipeline
    controlnet = bundle.controlnet
    torch_device = torch.device(bundle.device)
    height, width = output_size
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=[prompt],
        prompt_2=[prompt],
        device=torch_device,
    )
    if text_ids.dim() == 3:
        text_ids = text_ids[0]
    prior_tokens = bundle.condition_modules["prior_token_bank"](reference_encoding.local_tokens)
    context = append_cross_v4_context(
        prompt_embeds=prompt_embeds,
        text_ids=text_ids,
        reference_encoding=reference_encoding,
        prior_tokens=prior_tokens,
    )
    bias_cfg = _bias_config_from_bundle(bundle)
    correspondence_bias = build_cross_v4_correspondence_bias(
        target_metadata=target_metadata,
        context=context,
        config=bias_cfg,
        dtype=bundle.torch_dtype,
    )
    joint_attention_kwargs = {
        "cross_v4_bias": correspondence_bias,
        "cross_v4_bias_scale": float(bundle.attention_bias_config.get("bias_scale", 1.0)),
    }

    control_image = FluxControlNetPipeline._pack_latents(
        control_tensor,
        1,
        control_tensor.shape[1],
        control_tensor.shape[2],
        control_tensor.shape[3],
    )
    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents, latent_image_ids = pipe.prepare_latents(
        1,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        torch_device,
        generator=torch.Generator(device=torch_device).manual_seed(42),
        latents=None,
    )
    sigmas = np.linspace(1.0, 1 / bundle.num_inference_steps, bundle.num_inference_steps)
    mu = _calculate_shift(
        image_seq_len=latents.shape[1],
        base_seq_len=pipe.scheduler.config.get("base_image_seq_len", 256),
        max_seq_len=pipe.scheduler.config.get("max_image_seq_len", 4096),
        base_shift=pipe.scheduler.config.get("base_shift", 0.5),
        max_shift=pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, _ = retrieve_timesteps(
        pipe.scheduler,
        bundle.num_inference_steps,
        torch_device,
        sigmas=sigmas,
        mu=mu,
    )
    controlnet_blocks_repeat = False if getattr(controlnet, "input_hint_block", None) is None else True

    for timestep in timesteps:
        expanded_timestep = timestep.expand(latents.shape[0]).to(latents.dtype)
        guidance = None
        if controlnet.config.guidance_embeds:
            guidance = torch.tensor([bundle.guidance_scale], device=torch_device).expand(latents.shape[0])
        controlnet_block_samples, controlnet_single_block_samples = controlnet(
            hidden_states=latents,
            controlnet_cond=control_image,
            controlnet_mode=None,
            conditioning_scale=bundle.controlnet_conditioning_scale,
            timestep=expanded_timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        transformer_guidance = None
        if pipe.transformer.config.guidance_embeds:
            transformer_guidance = torch.tensor([bundle.guidance_scale], device=torch_device).expand(latents.shape[0])
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=expanded_timestep / 1000,
            guidance=transformer_guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=context.encoder_hidden_states,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
            txt_ids=context.txt_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
            controlnet_blocks_repeat=controlnet_blocks_repeat,
        )[0]
        latents_dtype = latents.dtype
        latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
        if latents.dtype != latents_dtype:
            latents = latents.to(latents_dtype)

    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents.to(dtype=bundle.torch_dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def _load_flux_controlnet_pipeline(
    *,
    pretrained_model_name_or_path: str | Path,
    checkpoint_path: Path,
    packed_channels: int,
    device: str,
    torch_dtype: torch.dtype,
) -> tuple[object, object]:
    from diffusers import FluxControlNetModel, FluxControlNetPipeline

    controlnet_config = FluxControlNetModel.load_config(checkpoint_path)
    controlnet = FluxControlNetModel.from_config(controlnet_config)
    patch_controlnet_x_embedder(controlnet, packed_channels)
    controlnet.load_state_dict(_load_diffusers_model_state_dict(checkpoint_path), strict=True)
    controlnet.to(dtype=torch_dtype)

    pipe = FluxControlNetPipeline.from_pretrained(
        pretrained_model_name_or_path,
        controlnet=controlnet,
        torch_dtype=torch_dtype,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, controlnet


def _load_cross_v4_control_spec(state: dict) -> CrossV4ControlSpec:
    saved_spec = state.get("cross_v4_control_spec") or state.get("cross_v3_control_spec") or {}
    return CrossV4ControlSpec(
        tissue_channels=int(saved_spec.get("tissue_channels", 64)),
        nuclei_channels=int(saved_spec.get("nuclei_channels", 16)),
    )


def _load_cross_v4_reference_spec(state: dict) -> CrossV4ReferenceSpec:
    saved_spec = state.get("cross_v4_reference_spec") or state.get("cross_v3_reference_spec") or {}
    return CrossV4ReferenceSpec(
        reference_latent_channels=int(saved_spec.get("reference_latent_channels", 16)),
        tissue_channels=int(saved_spec.get("tissue_channels", 64)),
        nuclei_channels=int(saved_spec.get("nuclei_channels", 16)),
        token_dim=int(saved_spec.get("token_dim", 4096)),
        output_init_std=float(saved_spec.get("output_init_std", 0.02)),
        route_anchor_mode=str(saved_spec.get("route_anchor_mode", "none")),
        route_embedding_init_std=float(saved_spec.get("route_embedding_init_std", 0.02)),
        tissue_prior_tokens_per_class=int(saved_spec.get("tissue_prior_tokens_per_class", 4)),
        cell_prior_tokens_per_class=int(saved_spec.get("cell_prior_tokens_per_class", 0)),
        global_style_tokens=int(saved_spec.get("global_style_tokens", 0)),
        prior_init_std=float(saved_spec.get("prior_init_std", 0.02)),
    )


def _load_cross_v4_condition_modules(
    *,
    state: dict,
    reference_spec: CrossV4ReferenceSpec,
    device: str,
    torch_dtype: torch.dtype,
) -> dict[str, nn.Module]:
    control_spec = state.get("cross_v4_control_spec") or state.get("cross_v3_control_spec") or {}
    hte_state = state["hte"]
    tissue_state = state["tissue_downsampler"]
    nuclei_state = state["nuclei_encoder"]
    reference_state = state["reference_context_encoder"]

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
        "reference_context_encoder": CrossV4ReferenceContextEncoder(
            reference_latent_channels=reference_spec.reference_latent_channels,
            tissue_channels=reference_spec.tissue_channels,
            nuclei_channels=reference_spec.nuclei_channels,
            token_dim=reference_spec.token_dim,
            hidden_dim=reference_state["proj_in.weight"].shape[0],
            output_init_std=reference_spec.output_init_std,
            route_anchor_mode=reference_spec.route_anchor_mode,
            route_embedding_init_std=reference_spec.route_embedding_init_std,
        ),
        "prior_token_bank": CrossV4PriorTokenBank(
            token_dim=reference_spec.token_dim,
            tissue_prior_tokens_per_class=reference_spec.tissue_prior_tokens_per_class,
            cell_prior_tokens_per_class=reference_spec.cell_prior_tokens_per_class,
            global_style_tokens=reference_spec.global_style_tokens,
            init_std=reference_spec.prior_init_std,
        ),
    }
    if str(control_spec.get("target_tissue_path", "")).lower() == "fixed_one_hot":
        modules["target_tissue_encoder"] = FixedOneHotTissueEncoder(
            num_classes=int(control_spec.get("tissue_channels", 16)),
            downsample_factor=2 ** _count_conv_blocks(tissue_state, "blocks"),
            scale=float(control_spec.get("target_one_hot_scale", 4.0) or 4.0),
        )
    elif "target_hte" in state or "target_tissue_downsampler" in state:
        target_hte_state = state["target_hte"]
        target_tissue_state = state["target_tissue_downsampler"]
        target_hte_dim = target_hte_state["parent_embeddings.weight"].shape[1]
        target_tissue_in = target_tissue_state["blocks.0.block.0.weight"].shape[1]
        target_tissue_hidden = target_tissue_state["blocks.0.block.0.weight"].shape[0]
        target_tissue_out = target_tissue_state[
            f"blocks.{_count_conv_blocks(target_tissue_state, 'blocks') - 1}.block.0.weight"
        ].shape[0]
        modules["target_hte"] = HierarchicalTissueEmbedding(embedding_dim=target_hte_dim)
        modules["target_tissue_downsampler"] = TissueConditionDownsampler(
            in_channels=target_tissue_in,
            hidden_channels=target_tissue_hidden,
            out_channels=target_tissue_out,
            num_blocks=_count_conv_blocks(target_tissue_state, "blocks"),
        )
    for name, module in modules.items():
        if name in state:
            module.load_state_dict(state[name])
        module.to(device=device, dtype=torch_dtype)
        module.eval()
    return modules


def _bias_config_from_bundle(bundle: CrossV4InferenceBundle) -> CrossV4CorrespondenceBiasConfig:
    saved = bundle.attention_bias_config
    return CrossV4CorrespondenceBiasConfig(
        same_fine=float(saved.get("same_fine", 3.0)),
        same_coarse=float(saved.get("same_coarse", 2.0)),
        mismatch=float(saved.get("mismatch", -2.0)),
        cell_similarity=float(saved.get("cell_similarity", 1.0)),
        density_gap=float(saved.get("density_gap", 0.5)),
        prior_when_ref_present=float(saved.get("prior_when_ref_present", 0.5)),
        prior_when_ref_missing=float(saved.get("prior_when_ref_missing", 3.0)),
        prior_wrong_class=float(saved.get("prior_wrong_class", -2.0)),
        cell_prior=float(saved.get("cell_prior", 1.0)),
        scale=1.0,
    )


def _encode_images_to_latents(vae, images: torch.Tensor, torch_dtype: torch.dtype) -> torch.Tensor:
    device = next(vae.parameters()).device
    images = images.to(device=device, dtype=torch_dtype)
    images = images * 2.0 - 1.0
    posterior = vae.encode(images).latent_dist
    latents = deterministic_latent_from_posterior(posterior)
    return (latents - vae.config.shift_factor) * vae.config.scaling_factor


__all__ = [
    "CROSS_V4_PROMPT",
    "CROSS_V4_REFERENCE_WITH_REF",
    "CROSS_V4_REFERENCE_ZERO_REF",
    "CrossV4InferenceBundle",
    "load_cross_v4_bundle",
    "run_cross_v4_bundle",
]

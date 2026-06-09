"""Inference helpers for Cross V2.2 Flux ControlNet."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from PIL import Image

from controlnet_train.modules import (
    HierarchicalTissueEmbedding,
    NucleiConditionEncoder,
    TissueConditionDownsampler,
)
from controlnet_train.modules.cross_v2_2_conditioning import (
    CROSS_V2_2_REFERENCE_WITH_REF,
    CROSS_V2_2_REFERENCE_ZERO_REF,
    CrossV22ControlSpec,
    apply_cross_v2_2_reference_mode,
    build_cross_v2_2_block_bank_reference_latent,
    build_cross_v2_2_condition,
    deterministic_latent_from_posterior,
)
from controlnet_train.training.conditioning import patch_controlnet_x_embedder


@dataclass
class CrossV22InferenceBundle:
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
    control_spec: CrossV22ControlSpec = field(default_factory=CrossV22ControlSpec)
    reference_bank_block_size: int = 4
    reference_bank_label_mode: str = "tissue_nuclei"
    zero_reference_mask_features: bool = True


def load_cross_v2_2_bundle(
    *,
    pretrained_model_name_or_path: str | Path,
    checkpoint_path: str | Path,
    device: str = "cuda",
    torch_dtype: torch.dtype | None = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    controlnet_conditioning_scale: float = 1.0,
    reference_bank_block_size: int | None = None,
    reference_bank_label_mode: str | None = None,
    keep_reference_mask_features: bool | None = None,
) -> CrossV22InferenceBundle:
    device = _resolve_device(device)
    dtype = _resolve_torch_dtype(torch_dtype, device)
    checkpoint = _validate_checkpoint_dir(checkpoint_path)
    control_spec = _load_cross_v2_2_control_spec(checkpoint)
    bank_config = _load_cross_v2_2_reference_bank_config(checkpoint)
    if reference_bank_block_size is not None:
        bank_config["reference_bank_block_size"] = max(1, int(reference_bank_block_size))
    if reference_bank_label_mode is not None:
        bank_config["reference_bank_label_mode"] = str(reference_bank_label_mode)
    if keep_reference_mask_features is not None:
        bank_config["zero_reference_mask_features"] = not bool(keep_reference_mask_features)
    pipe, controlnet = _load_flux_controlnet_pipeline(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        checkpoint_path=checkpoint,
        packed_channels=control_spec.packed_channels,
        device=device,
        torch_dtype=dtype,
    )
    modules = _load_condition_modules(
        checkpoint_path=checkpoint,
        device=device,
        torch_dtype=dtype,
    )
    return CrossV22InferenceBundle(
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
        reference_bank_block_size=int(bank_config["reference_bank_block_size"]),
        reference_bank_label_mode=str(bank_config["reference_bank_label_mode"]),
        zero_reference_mask_features=bool(bank_config["zero_reference_mask_features"]),
    )


@torch.inference_mode()
def run_cross_v2_2_bundle(
    bundle: CrossV22InferenceBundle,
    reference_image: torch.Tensor,
    reference_tissue_mask: torch.Tensor,
    reference_nuclei_mask: torch.Tensor,
    target_tissue_mask: torch.Tensor,
    target_nuclei_mask: torch.Tensor,
    prompt: str,
    reference_condition_mode: str = CROSS_V2_2_REFERENCE_WITH_REF,
    reference_condition_image: torch.Tensor | None = None,
    reference_latent_scale: float = 1.0,
) -> Image.Image:
    control_tensor = build_cross_v2_2_control_tensor(
        bundle,
        reference_image=reference_image,
        reference_tissue_mask=reference_tissue_mask,
        reference_nuclei_mask=reference_nuclei_mask,
        target_tissue_mask=target_tissue_mask,
        target_nuclei_mask=target_nuclei_mask,
        reference_condition_mode=reference_condition_mode,
        reference_condition_image=reference_condition_image,
        reference_latent_scale=reference_latent_scale,
    )

    return _sample_with_flux_controlnet(
        pipe=bundle.flux_pipeline,
        controlnet=bundle.controlnet,
        prompt=prompt,
        control_tensor=control_tensor,
        output_size=tuple(int(v) for v in reference_image.shape[1:]),
        device=bundle.device,
        torch_dtype=bundle.torch_dtype,
        num_inference_steps=bundle.num_inference_steps,
        guidance_scale=bundle.guidance_scale,
        controlnet_conditioning_scale=bundle.controlnet_conditioning_scale,
    )


@torch.inference_mode()
def compute_cross_v2_2_fixed_timestep_losses(
    bundle: CrossV22InferenceBundle,
    reference_image: torch.Tensor,
    reference_tissue_mask: torch.Tensor,
    reference_nuclei_mask: torch.Tensor,
    target_image: torch.Tensor,
    target_tissue_mask: torch.Tensor,
    target_nuclei_mask: torch.Tensor,
    prompt: str,
    timesteps: Iterable[int | float],
    reference_condition_mode: str = CROSS_V2_2_REFERENCE_WITH_REF,
    seed: int = 42,
    reference_condition_image: torch.Tensor | None = None,
    reference_latent_scale: float = 1.0,
) -> dict[str, float]:
    """Compute one-step denoising MSE at fixed training timesteps."""

    timestep_values = [float(timestep) for timestep in timesteps]
    if not timestep_values:
        return {}

    target_latent = _encode_images_to_latents(
        bundle.flux_pipeline.vae,
        target_image.unsqueeze(0),
        bundle.torch_dtype,
    )
    control_tensor = build_cross_v2_2_control_tensor(
        bundle,
        reference_image=reference_image,
        reference_tissue_mask=reference_tissue_mask,
        reference_nuclei_mask=reference_nuclei_mask,
        target_tissue_mask=target_tissue_mask,
        target_nuclei_mask=target_nuclei_mask,
        reference_condition_mode=reference_condition_mode,
        reference_condition_image=reference_condition_image,
        reference_latent_scale=reference_latent_scale,
    )
    return _fixed_timestep_losses_with_flux_controlnet(
        pipe=bundle.flux_pipeline,
        controlnet=bundle.controlnet,
        prompt=prompt,
        pixel_latents=target_latent,
        control_tensor=control_tensor,
        timesteps=timestep_values,
        device=bundle.device,
        torch_dtype=bundle.torch_dtype,
        guidance_scale=bundle.guidance_scale,
        controlnet_conditioning_scale=bundle.controlnet_conditioning_scale,
        seed=seed,
    )


@torch.inference_mode()
def build_cross_v2_2_control_tensor(
    bundle: CrossV22InferenceBundle,
    *,
    reference_image: torch.Tensor,
    reference_tissue_mask: torch.Tensor,
    reference_nuclei_mask: torch.Tensor,
    target_tissue_mask: torch.Tensor,
    target_nuclei_mask: torch.Tensor,
    reference_condition_mode: str = CROSS_V2_2_REFERENCE_WITH_REF,
    reference_condition_image: torch.Tensor | None = None,
    reference_latent_scale: float = 1.0,
) -> torch.Tensor:
    """Build the raw Cross V2.2 condition tensor for inference diagnostics."""

    reference_source = reference_image if reference_condition_image is None else reference_condition_image
    reference_latent = _encode_images_to_latents(
        bundle.flux_pipeline.vae,
        _ensure_batched_image(reference_source, name="reference_condition_image"),
        bundle.torch_dtype,
    )
    ref_tissue_mask_b = _ensure_batched_mask(reference_tissue_mask, name="reference_tissue_mask")
    ref_nuclei_mask_b = _ensure_batched_mask(reference_nuclei_mask, name="reference_nuclei_mask")
    target_tissue_mask_b = _ensure_batched_mask(target_tissue_mask, name="target_tissue_mask")
    target_nuclei_mask_b = _ensure_batched_mask(target_nuclei_mask, name="target_nuclei_mask")

    ref_tissue_feat = bundle.condition_modules["tissue_downsampler"](
        bundle.condition_modules["hte"](ref_tissue_mask_b.to(device=bundle.device))
    ).to(dtype=bundle.torch_dtype)
    ref_nuclei_feat = bundle.condition_modules["nuclei_encoder"](
        ref_nuclei_mask_b.to(device=bundle.device)
    ).to(dtype=bundle.torch_dtype)
    tar_tissue_feat = bundle.condition_modules["tissue_downsampler"](
        bundle.condition_modules["hte"](target_tissue_mask_b.to(device=bundle.device))
    ).to(dtype=bundle.torch_dtype)
    tar_nuclei_feat = bundle.condition_modules["nuclei_encoder"](
        target_nuclei_mask_b.to(device=bundle.device)
    ).to(dtype=bundle.torch_dtype)

    if reference_condition_image is None:
        reference_latent = build_cross_v2_2_block_bank_reference_latent(
            z_ref=reference_latent,
            reference_tissue_mask=ref_tissue_mask_b,
            reference_nuclei_mask=ref_nuclei_mask_b,
            target_tissue_mask=target_tissue_mask_b,
            target_nuclei_mask=target_nuclei_mask_b,
            block_size=bundle.reference_bank_block_size,
            label_mode=bundle.reference_bank_label_mode,
        ).to(dtype=bundle.torch_dtype)
    reference_latent = reference_latent * float(reference_latent_scale)

    if bundle.zero_reference_mask_features:
        ref_tissue_feat = torch.zeros_like(ref_tissue_feat)
        ref_nuclei_feat = torch.zeros_like(ref_nuclei_feat)
    reference_latent, ref_tissue_feat, ref_nuclei_feat = apply_cross_v2_2_reference_mode(
        z_ref=reference_latent,
        ref_tissue_feat=ref_tissue_feat,
        ref_nuclei_feat=ref_nuclei_feat,
        mode=reference_condition_mode,
    )

    return build_cross_v2_2_condition(
        z_ref=reference_latent,
        ref_tissue_feat=ref_tissue_feat,
        ref_nuclei_feat=ref_nuclei_feat,
        tar_tissue_feat=tar_tissue_feat,
        tar_nuclei_feat=tar_nuclei_feat,
    )


def _ensure_batched_image(image: torch.Tensor, *, name: str) -> torch.Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"{name} must have shape (C,H,W) or (B,C,H,W), got {tuple(image.shape)}.")
    if image.shape[1] != 3:
        raise ValueError(f"{name} must have 3 RGB channels, got {tuple(image.shape)}.")
    return image


def _ensure_batched_mask(mask: torch.Tensor, *, name: str) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"{name} must have shape (H,W) or (B,H,W), got {tuple(mask.shape)}.")
    return mask


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
    if not (checkpoint / "config.json").exists():
        raise FileNotFoundError(f"Missing ControlNet config.json under checkpoint path: {checkpoint}")
    if not (checkpoint / "phase5_conditioning.pt").exists():
        raise FileNotFoundError(f"Missing phase5_conditioning.pt under checkpoint path: {checkpoint}")
    return checkpoint


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


def _load_cross_v2_2_control_spec(checkpoint_path: Path) -> CrossV22ControlSpec:
    state = _torch_load_weights(checkpoint_path / "phase5_conditioning.pt")
    saved_spec = state.get("cross_v2_2_control_spec") or state.get("cross_v2_1_control_spec") or {}
    return CrossV22ControlSpec(
        reference_latent_channels=int(saved_spec.get("reference_latent_channels", 16)),
        tissue_channels=int(saved_spec.get("tissue_channels", 64)),
        nuclei_channels=int(saved_spec.get("nuclei_channels", 16)),
    )


def _load_cross_v2_2_reference_bank_config(checkpoint_path: Path) -> dict[str, object]:
    state = _torch_load_weights(checkpoint_path / "phase5_conditioning.pt")
    saved_spec = state.get("cross_v2_2_control_spec") or {}
    return {
        "reference_bank_block_size": max(1, int(saved_spec.get("reference_bank_block_size", 4) or 4)),
        "reference_bank_label_mode": str(saved_spec.get("reference_bank_label_mode", "tissue_nuclei")),
        "zero_reference_mask_features": bool(saved_spec.get("zero_reference_mask_features", True)),
    }


def _load_condition_modules(
    *,
    checkpoint_path: Path,
    device: str,
    torch_dtype: torch.dtype,
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
    return modules


def _count_conv_blocks(state_dict: dict[str, torch.Tensor], prefix: str) -> int:
    return len(
        {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith(prefix) and key.endswith("block.0.weight")
        }
    )


def _encode_images_to_latents(vae, images: torch.Tensor, torch_dtype: torch.dtype) -> torch.Tensor:
    device = next(vae.parameters()).device
    images = images.to(device=device, dtype=torch_dtype)
    images = images * 2.0 - 1.0
    posterior = vae.encode(images).latent_dist
    latents = deterministic_latent_from_posterior(posterior)
    return (latents - vae.config.shift_factor) * vae.config.scaling_factor


def _per_sample_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction.float() - target.float()).pow(2).flatten(1).mean(dim=1)


@torch.inference_mode()
def _sample_with_flux_controlnet(
    *,
    pipe,
    controlnet,
    prompt: str,
    control_tensor: torch.Tensor,
    output_size: tuple[int, int],
    device: str,
    torch_dtype: torch.dtype,
    num_inference_steps: int,
    guidance_scale: float,
    controlnet_conditioning_scale: float,
) -> Image.Image:
    from diffusers import FluxControlNetPipeline
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
    import numpy as np

    torch_device = torch.device(device)
    height, width = output_size
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=[prompt],
        prompt_2=[prompt],
        device=torch_device,
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
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = _calculate_shift(
        image_seq_len=image_seq_len,
        base_seq_len=pipe.scheduler.config.get("base_image_seq_len", 256),
        max_seq_len=pipe.scheduler.config.get("max_image_seq_len", 4096),
        base_shift=pipe.scheduler.config.get("base_shift", 0.5),
        max_shift=pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, _ = retrieve_timesteps(pipe.scheduler, num_inference_steps, torch_device, sigmas=sigmas, mu=mu)
    controlnet_blocks_repeat = False if getattr(controlnet, "input_hint_block", None) is None else True

    for timestep in timesteps:
        expanded_timestep = timestep.expand(latents.shape[0]).to(latents.dtype)
        guidance = None
        if controlnet.config.guidance_embeds:
            guidance = torch.tensor([guidance_scale], device=torch_device).expand(latents.shape[0])
        controlnet_block_samples, controlnet_single_block_samples = controlnet(
            hidden_states=latents,
            controlnet_cond=control_image,
            controlnet_mode=None,
            conditioning_scale=controlnet_conditioning_scale,
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
            transformer_guidance = torch.tensor([guidance_scale], device=torch_device).expand(latents.shape[0])
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=expanded_timestep / 1000,
            guidance=transformer_guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
            controlnet_blocks_repeat=controlnet_blocks_repeat,
        )[0]
        latents_dtype = latents.dtype
        latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
        if latents.dtype != latents_dtype:
            latents = latents.to(latents_dtype)

    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents.to(dtype=torch_dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


@torch.inference_mode()
def _fixed_timestep_losses_with_flux_controlnet(
    *,
    pipe,
    controlnet,
    prompt: str,
    pixel_latents: torch.Tensor,
    control_tensor: torch.Tensor,
    timesteps: list[float],
    device: str,
    torch_dtype: torch.dtype,
    guidance_scale: float,
    controlnet_conditioning_scale: float,
    seed: int,
) -> dict[str, float]:
    from diffusers import FluxControlNetPipeline

    torch_device = torch.device(device)
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=[prompt],
        prompt_2=[prompt],
        device=torch_device,
    )
    if text_ids.dim() == 3:
        text_ids = text_ids[0]

    bsz = int(pixel_latents.shape[0])
    packed_pixel_latents = FluxControlNetPipeline._pack_latents(
        pixel_latents,
        bsz,
        pixel_latents.shape[1],
        pixel_latents.shape[2],
        pixel_latents.shape[3],
    )
    control_image = FluxControlNetPipeline._pack_latents(
        control_tensor,
        bsz,
        control_tensor.shape[1],
        control_tensor.shape[2],
        control_tensor.shape[3],
    )
    latent_image_ids = _prepare_packed_latent_image_ids(
        packed_height=pixel_latents.shape[2] // 2,
        packed_width=pixel_latents.shape[3] // 2,
        device=torch_device,
        dtype=torch_dtype,
    )
    controlnet_blocks_repeat = False if getattr(controlnet, "input_hint_block", None) is None else True
    generator = torch.Generator(device=torch_device).manual_seed(seed)
    noise = torch.randn(
        packed_pixel_latents.shape,
        generator=generator,
        device=packed_pixel_latents.device,
        dtype=packed_pixel_latents.dtype,
    )

    results: dict[str, float] = {}
    for timestep_value in timesteps:
        timestep = torch.tensor([timestep_value], device=torch_device, dtype=torch.float32)
        sigma = _sigma_for_timestep(pipe.scheduler, timestep, n_dim=packed_pixel_latents.ndim, dtype=packed_pixel_latents.dtype)
        noisy_model_input = (1.0 - sigma) * packed_pixel_latents + sigma * noise
        expanded_timestep = timestep.expand(bsz).to(dtype=packed_pixel_latents.dtype)

        guidance = None
        if controlnet.config.guidance_embeds:
            guidance = torch.full((bsz,), guidance_scale, device=torch_device)
        controlnet_block_samples, controlnet_single_block_samples = controlnet(
            hidden_states=noisy_model_input,
            controlnet_cond=control_image,
            controlnet_mode=None,
            conditioning_scale=controlnet_conditioning_scale,
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
            transformer_guidance = torch.full((bsz,), guidance_scale, device=torch_device)
        noise_pred = pipe.transformer(
            hidden_states=noisy_model_input,
            timestep=expanded_timestep / 1000,
            guidance=transformer_guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
            controlnet_blocks_repeat=controlnet_blocks_repeat,
        )[0]
        target_velocity = noise - packed_pixel_latents
        loss = _per_sample_mse(noise_pred, target_velocity).mean()
        results[_format_timestep_key(timestep_value)] = float(loss.detach().cpu().item())
    return results


def _sigma_for_timestep(scheduler, timestep: torch.Tensor, *, n_dim: int, dtype: torch.dtype) -> torch.Tensor:
    sigmas = scheduler.sigmas.to(device=timestep.device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device=timestep.device, dtype=timestep.dtype)
    step_indices = []
    for value in timestep:
        matches = (schedule_timesteps == value).nonzero()
        if matches.numel() > 0:
            step_indices.append(int(matches[0].item()))
        else:
            step_indices.append(int(torch.argmin(torch.abs(schedule_timesteps - value)).item()))
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def _prepare_packed_latent_image_ids(
    *,
    packed_height: int,
    packed_width: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    latent_image_ids = torch.zeros(packed_height, packed_width, 3, device=device, dtype=dtype)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(
        packed_height,
        device=device,
        dtype=dtype,
    )[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(
        packed_width,
        device=device,
        dtype=dtype,
    )[None, :]
    return latent_image_ids.reshape(packed_height * packed_width, 3)


def _format_timestep_key(timestep: float) -> str:
    if float(timestep).is_integer():
        return f"t{int(timestep)}"
    return f"t{str(timestep).replace('.', 'p')}"


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

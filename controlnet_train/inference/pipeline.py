"""Unified Phase 5.4 edit pipeline orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from controlnet_train.data.common import (
    default_prompt_for_dataset,
    load_image_tensor,
    load_nuclei_mask,
    load_tissue_mask,
)
from controlnet_train.modules import (
    ChangeMaskEncoder,
    HierarchicalTissueEmbedding,
    NucleiConditionEncoder,
    TissueConditionDownsampler,
    build_cross_v0_condition,
    build_inpaint_condition,
)
from controlnet_train.training.conditioning import (
    CrossV0ControlSpec,
    InpaintControlSpec,
    patch_controlnet_x_embedder,
)

from .router import EditRoutingConfig, EditRoutingDecision, route_edit_request

GenericRunner = Callable[..., Image.Image]


@dataclass(frozen=True)
class EditPipelineInputs:
    reference_image: str | Path
    reference_tissue_mask: str | Path
    reference_nuclei_mask: str | Path
    target_tissue_mask: str | Path
    target_nuclei_mask: str | Path
    output_dir: str | Path
    prompt: str | None = None
    dataset: str | None = None
    force_mode: str | None = None
    save_debug_artifacts: bool = False


@dataclass(frozen=True)
class LoadedEditInputs:
    reference_image_path: Path
    reference_tissue_mask_path: Path
    reference_nuclei_mask_path: Path
    target_tissue_mask_path: Path
    target_nuclei_mask_path: Path
    output_dir: Path
    reference_image: torch.Tensor
    reference_tissue_mask: torch.Tensor
    reference_nuclei_mask: torch.Tensor
    target_tissue_mask: torch.Tensor
    target_nuclei_mask: torch.Tensor
    prompt: str | None
    dataset: str | None
    force_mode: str | None
    save_debug_artifacts: bool


@dataclass(frozen=True)
class EditPipelineResult:
    image: Image.Image
    selected_mode: str
    change_region_mask: torch.Tensor
    change_ratio: float
    prompt: str
    output_dir: Path


@dataclass
class InpaintInferenceBundle:
    pretrained_model_name_or_path: str | Path
    checkpoint_path: Path
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.bfloat16
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    controlnet_conditioning_scale: float = 1.0
    flux_pipeline: object | None = None
    controlnet: object | None = None
    condition_modules: dict[str, torch.nn.Module] = field(default_factory=dict)
    control_spec: InpaintControlSpec = field(default_factory=InpaintControlSpec)


@dataclass
class CrossV0InferenceBundle:
    pretrained_model_name_or_path: str | Path
    checkpoint_path: Path
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.bfloat16
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    controlnet_conditioning_scale: float = 1.0
    flux_pipeline: object | None = None
    controlnet: object | None = None
    condition_modules: dict[str, torch.nn.Module] = field(default_factory=dict)
    control_spec: CrossV0ControlSpec = field(default_factory=CrossV0ControlSpec)


def resolve_prompt(prompt: str | None, dataset: str | None) -> str:
    if prompt:
        return prompt
    if dataset:
        return default_prompt_for_dataset(dataset)
    return "H&E stained cancer histopathology at 40x magnification"


def run_edit_pipeline(
    *,
    inputs: EditPipelineInputs,
    inpaint_bundle: InpaintInferenceBundle | object,
    cross_bundle: CrossV0InferenceBundle | object,
    inpaint_runner: GenericRunner,
    cross_runner: GenericRunner,
    routing_config: EditRoutingConfig | None = None,
) -> EditPipelineResult:
    loaded_inputs = _load_inputs(inputs)
    decision = route_edit_request(
        loaded_inputs.reference_tissue_mask,
        loaded_inputs.target_tissue_mask,
        config=routing_config,
    )
    selected_mode = loaded_inputs.force_mode or decision.selected_mode
    if selected_mode not in {"inpaint", "cross"}:
        raise ValueError(f"Unsupported force_mode: {selected_mode}")

    prompt = resolve_prompt(loaded_inputs.prompt, loaded_inputs.dataset)

    if selected_mode == "inpaint":
        image = inpaint_runner(inpaint_bundle, loaded_inputs, prompt, decision.change_region_mask)
    else:
        image = cross_runner(cross_bundle, loaded_inputs, prompt)

    _save_outputs(
        loaded_inputs=loaded_inputs,
        decision=decision,
        selected_mode=selected_mode,
        prompt=prompt,
        image=image,
    )

    return EditPipelineResult(
        image=image,
        selected_mode=selected_mode,
        change_region_mask=decision.change_region_mask,
        change_ratio=decision.change_ratio,
        prompt=prompt,
        output_dir=loaded_inputs.output_dir,
    )


def load_inpaint_bundle(
    *,
    pretrained_model_name_or_path: str | Path,
    checkpoint_path: str | Path,
    device: str = "cuda",
    torch_dtype: torch.dtype | None = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    controlnet_conditioning_scale: float = 1.0,
) -> InpaintInferenceBundle:
    dtype = _resolve_torch_dtype(torch_dtype, device)
    checkpoint = _validate_checkpoint_dir(checkpoint_path)
    pipe, controlnet = _load_flux_controlnet_pipeline(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        checkpoint_path=checkpoint,
        packed_channels=InpaintControlSpec().packed_channels,
        device=device,
        torch_dtype=dtype,
    )
    modules = _load_condition_modules(
        checkpoint_path=checkpoint,
        device=device,
        torch_dtype=dtype,
        include_change_encoder=True,
    )
    return InpaintInferenceBundle(
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
    )


def load_cross_bundle(
    *,
    pretrained_model_name_or_path: str | Path,
    checkpoint_path: str | Path,
    device: str = "cuda",
    torch_dtype: torch.dtype | None = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    controlnet_conditioning_scale: float = 1.0,
) -> CrossV0InferenceBundle:
    dtype = _resolve_torch_dtype(torch_dtype, device)
    checkpoint = _validate_checkpoint_dir(checkpoint_path)
    pipe, controlnet = _load_flux_controlnet_pipeline(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        checkpoint_path=checkpoint,
        packed_channels=CrossV0ControlSpec().packed_channels,
        device=device,
        torch_dtype=dtype,
    )
    modules = _load_condition_modules(
        checkpoint_path=checkpoint,
        device=device,
        torch_dtype=dtype,
        include_change_encoder=False,
    )
    return CrossV0InferenceBundle(
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
    )


def run_inpaint_bundle(
    bundle: InpaintInferenceBundle,
    inputs: LoadedEditInputs,
    prompt: str,
    change_region_mask: torch.Tensor,
) -> Image.Image:
    reference_image = inputs.reference_image.unsqueeze(0)
    source_image = _erase_reference_image(reference_image, change_region_mask)
    source_latent = _encode_images_to_latents(
        bundle.flux_pipeline.vae,
        source_image,
        bundle.torch_dtype,
    )
    target_tissue_feat = bundle.condition_modules["tissue_downsampler"](
        bundle.condition_modules["hte"](
            inputs.target_tissue_mask.unsqueeze(0).to(device=bundle.device)
        )
    ).to(dtype=bundle.torch_dtype)
    target_nuclei_feat = bundle.condition_modules["nuclei_encoder"](
        inputs.target_nuclei_mask.unsqueeze(0).to(device=bundle.device)
    ).to(dtype=bundle.torch_dtype)
    resized_change_mask = F.interpolate(
        change_region_mask.unsqueeze(0).unsqueeze(0).to(
            device=bundle.device,
            dtype=bundle.torch_dtype,
        ),
        size=source_latent.shape[2:],
        mode="nearest",
    )
    change_mask_feat = bundle.condition_modules["change_encoder"](resized_change_mask).to(
        dtype=bundle.torch_dtype
    )
    control_tensor = build_inpaint_condition(
        source_image_latent=source_latent,
        target_tissue_feat=target_tissue_feat,
        target_nuclei_feat=target_nuclei_feat,
        change_mask_feat=change_mask_feat,
    )
    return _sample_with_flux_controlnet(
        pipe=bundle.flux_pipeline,
        controlnet=bundle.controlnet,
        prompt=prompt,
        control_tensor=control_tensor,
        output_size=tuple(int(v) for v in inputs.reference_image.shape[1:]),
        device=bundle.device,
        torch_dtype=bundle.torch_dtype,
        num_inference_steps=bundle.num_inference_steps,
        guidance_scale=bundle.guidance_scale,
        controlnet_conditioning_scale=bundle.controlnet_conditioning_scale,
    )


def run_cross_v0_bundle(
    bundle: CrossV0InferenceBundle,
    inputs: LoadedEditInputs,
    prompt: str,
) -> Image.Image:
    reference_image = inputs.reference_image.unsqueeze(0)
    reference_latent = _encode_images_to_latents(
        bundle.flux_pipeline.vae,
        reference_image,
        bundle.torch_dtype,
    )
    reference_tissue_feat = bundle.condition_modules["tissue_downsampler"](
        bundle.condition_modules["hte"](
            inputs.reference_tissue_mask.unsqueeze(0).to(device=bundle.device)
        )
    ).to(dtype=bundle.torch_dtype)
    reference_nuclei_feat = bundle.condition_modules["nuclei_encoder"](
        inputs.reference_nuclei_mask.unsqueeze(0).to(device=bundle.device)
    ).to(dtype=bundle.torch_dtype)
    target_tissue_feat = bundle.condition_modules["tissue_downsampler"](
        bundle.condition_modules["hte"](
            inputs.target_tissue_mask.unsqueeze(0).to(device=bundle.device)
        )
    ).to(dtype=bundle.torch_dtype)
    target_nuclei_feat = bundle.condition_modules["nuclei_encoder"](
        inputs.target_nuclei_mask.unsqueeze(0).to(device=bundle.device)
    ).to(dtype=bundle.torch_dtype)
    control_tensor = build_cross_v0_condition(
        reference_image_latent=reference_latent,
        reference_tissue_feat=reference_tissue_feat,
        reference_nuclei_feat=reference_nuclei_feat,
        target_tissue_feat=target_tissue_feat,
        target_nuclei_feat=target_nuclei_feat,
    )
    return _sample_with_flux_controlnet(
        pipe=bundle.flux_pipeline,
        controlnet=bundle.controlnet,
        prompt=prompt,
        control_tensor=control_tensor,
        output_size=tuple(int(v) for v in inputs.reference_image.shape[1:]),
        device=bundle.device,
        torch_dtype=bundle.torch_dtype,
        num_inference_steps=bundle.num_inference_steps,
        guidance_scale=bundle.guidance_scale,
        controlnet_conditioning_scale=bundle.controlnet_conditioning_scale,
    )


def _load_inputs(inputs: EditPipelineInputs) -> LoadedEditInputs:
    reference_image_path = _ensure_existing_file(inputs.reference_image, "reference_image")
    reference_tissue_mask_path = _ensure_existing_file(
        inputs.reference_tissue_mask, "reference_tissue_mask"
    )
    reference_nuclei_mask_path = _ensure_existing_file(
        inputs.reference_nuclei_mask, "reference_nuclei_mask"
    )
    target_tissue_mask_path = _ensure_existing_file(inputs.target_tissue_mask, "target_tissue_mask")
    target_nuclei_mask_path = _ensure_existing_file(inputs.target_nuclei_mask, "target_nuclei_mask")

    reference_image = load_image_tensor(reference_image_path)
    reference_tissue_mask = load_tissue_mask(reference_tissue_mask_path)
    reference_nuclei_mask = load_nuclei_mask(reference_nuclei_mask_path)
    target_tissue_mask = load_tissue_mask(target_tissue_mask_path)
    target_nuclei_mask = load_nuclei_mask(target_nuclei_mask_path)

    image_size = tuple(int(v) for v in reference_image.shape[1:])
    _validate_mask_size(reference_tissue_mask, image_size, "reference_tissue_mask")
    _validate_mask_size(reference_nuclei_mask, image_size, "reference_nuclei_mask")
    _validate_mask_size(target_tissue_mask, image_size, "target_tissue_mask")
    _validate_mask_size(target_nuclei_mask, image_size, "target_nuclei_mask")

    output_dir = Path(inputs.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return LoadedEditInputs(
        reference_image_path=reference_image_path,
        reference_tissue_mask_path=reference_tissue_mask_path,
        reference_nuclei_mask_path=reference_nuclei_mask_path,
        target_tissue_mask_path=target_tissue_mask_path,
        target_nuclei_mask_path=target_nuclei_mask_path,
        output_dir=output_dir,
        reference_image=reference_image,
        reference_tissue_mask=reference_tissue_mask,
        reference_nuclei_mask=reference_nuclei_mask,
        target_tissue_mask=target_tissue_mask,
        target_nuclei_mask=target_nuclei_mask,
        prompt=inputs.prompt,
        dataset=inputs.dataset,
        force_mode=inputs.force_mode,
        save_debug_artifacts=inputs.save_debug_artifacts,
    )


def _save_outputs(
    *,
    loaded_inputs: LoadedEditInputs,
    decision: EditRoutingDecision,
    selected_mode: str,
    prompt: str,
    image: Image.Image,
) -> None:
    output_dir = loaded_inputs.output_dir
    image.save(output_dir / "final.png")
    _save_mask_image(output_dir / "change_region_mask.png", decision.change_region_mask)
    if loaded_inputs.save_debug_artifacts:
        _save_mask_image(output_dir / "reference_tissue_mask.png", loaded_inputs.reference_tissue_mask)
        _save_mask_image(output_dir / "target_tissue_mask.png", loaded_inputs.target_tissue_mask)
        _save_mask_image(output_dir / "reference_nuclei_mask.png", loaded_inputs.reference_nuclei_mask)
        _save_mask_image(output_dir / "target_nuclei_mask.png", loaded_inputs.target_nuclei_mask)

    summary = {
        "selected_mode": selected_mode,
        "change_ratio": decision.change_ratio,
        "prompt": prompt,
        "reference_image": str(loaded_inputs.reference_image_path),
        "reference_tissue_mask": str(loaded_inputs.reference_tissue_mask_path),
        "reference_nuclei_mask": str(loaded_inputs.reference_nuclei_mask_path),
        "target_tissue_mask": str(loaded_inputs.target_tissue_mask_path),
        "target_nuclei_mask": str(loaded_inputs.target_nuclei_mask_path),
        "changed_tissue_ids_from": decision.changed_tissue_ids_from,
        "changed_tissue_ids_to": decision.changed_tissue_ids_to,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf8",
    )


def _save_mask_image(path: Path, mask: torch.Tensor) -> None:
    array = mask.detach().cpu().numpy()
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.dtype.kind in {"f", "b"}:
        array = (array > 0).astype(np.uint8) * 255
    else:
        array = array.astype(np.uint8)
    Image.fromarray(array).save(path)


def _ensure_existing_file(path_value: str | Path, label: str) -> Path:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _validate_mask_size(mask: torch.Tensor, image_size: tuple[int, int], label: str) -> None:
    if tuple(int(v) for v in mask.shape) != image_size:
        raise ValueError(
            f"{label} must match reference_image spatial size {image_size}, "
            f"got {tuple(int(v) for v in mask.shape)}."
        )


def _resolve_torch_dtype(torch_dtype: torch.dtype | None, device: str) -> torch.dtype:
    if torch_dtype is not None:
        return torch_dtype
    return torch.bfloat16 if "cuda" in str(device).lower() else torch.float32


def _validate_checkpoint_dir(checkpoint_path: str | Path) -> Path:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint}")
    if not (checkpoint / "config.json").exists():
        raise FileNotFoundError(f"Missing ControlNet config.json under checkpoint path: {checkpoint}")
    if not (checkpoint / "phase5_conditioning.pt").exists():
        raise FileNotFoundError(
            f"Missing phase5_conditioning.pt under checkpoint path: {checkpoint}"
        )
    return checkpoint


def _load_flux_controlnet_pipeline(
    *,
    pretrained_model_name_or_path: str | Path,
    checkpoint_path: Path,
    packed_channels: int,
    device: str,
    torch_dtype: torch.dtype,
) -> tuple[object, object]:
    from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
    from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline

    controlnet = FluxControlNetModel.from_pretrained(checkpoint_path, torch_dtype=torch_dtype)
    patch_controlnet_x_embedder(controlnet, packed_channels)
    pipe = FluxControlNetPipeline.from_pretrained(
        pretrained_model_name_or_path,
        controlnet=controlnet,
        torch_dtype=torch_dtype,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, controlnet


def _load_condition_modules(
    *,
    checkpoint_path: Path,
    device: str,
    torch_dtype: torch.dtype,
    include_change_encoder: bool,
) -> dict[str, torch.nn.Module]:
    state = torch.load(checkpoint_path / "phase5_conditioning.pt", map_location="cpu")
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

    modules: dict[str, torch.nn.Module] = {
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
    if include_change_encoder:
        change_state = state["change_encoder"]
        change_out = change_state["encoder.0.weight"].shape[0]
        modules["change_encoder"] = ChangeMaskEncoder(out_channels=change_out)

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


def _erase_reference_image(reference_image: torch.Tensor, change_region_mask: torch.Tensor) -> torch.Tensor:
    mask = change_region_mask.unsqueeze(0).unsqueeze(0).to(dtype=reference_image.dtype)
    return reference_image * (1.0 - mask) + 0.5 * mask


def _encode_images_to_latents(vae, images: torch.Tensor, torch_dtype: torch.dtype) -> torch.Tensor:
    device = next(vae.parameters()).device
    images = images.to(device=device, dtype=torch_dtype)
    images = images * 2.0 - 1.0
    latents = vae.encode(images).latent_dist.sample()
    return (latents - vae.config.shift_factor) * vae.config.scaling_factor


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
    from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

    height, width = output_size
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=[prompt],
        prompt_2=[prompt],
        device=device,
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
        device,
        generator=torch.Generator(device=device).manual_seed(42),
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
        pipe.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    controlnet_blocks_repeat = False if getattr(controlnet, "input_hint_block", None) is None else True

    for timestep in timesteps:
        expanded_timestep = timestep.expand(latents.shape[0]).to(latents.dtype)
        guidance = None
        if controlnet.config.guidance_embeds:
            guidance = torch.tensor([guidance_scale], device=device).expand(latents.shape[0])
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
            transformer_guidance = torch.tensor([guidance_scale], device=device).expand(
                latents.shape[0]
            )
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

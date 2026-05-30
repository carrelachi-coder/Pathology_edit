from __future__ import annotations

import json
import os
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
    CrossV1ControlSpec,
    build_cross_v1_condition,
)
from controlnet_train.modules.reference_image_encoder import ReferenceImageEncoder
from controlnet_train.training.conditioning import patch_controlnet_x_embedder
from controlnet_train.training.flux_phase5_cross_v1 import (
    install_flux_ip_adapter_attention,
    patch_flux_single_ip_forward,
    _collect_ip_adapter_modules,
    _sync_ip_adapter_to_transformer,
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
    flux_pipeline: object | None = None
    controlnet: object | None = None
    condition_modules: dict[str, nn.Module] = field(default_factory=dict)
    control_spec: CrossV1ControlSpec = field(default_factory=CrossV1ControlSpec)
    ip_adapter_modules: dict[str, nn.Module] = field(default_factory=dict)
    ref_encoder: ReferenceImageEncoder | None = None


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
) -> CrossV1InferenceBundle:
    device = _resolve_device(device)
    dtype = _resolve_torch_dtype(torch_dtype, device)
    checkpoint = _validate_checkpoint_dir(checkpoint_path)
    control_spec = _load_cross_v1_control_spec(checkpoint)
    ref_encoder_config = _load_ref_encoder_config(checkpoint)

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
    install_flux_ip_adapter_attention(
        pipe.transformer,
        num_tokens=ref_encoder_config.get("num_output_tokens", ref_encoder_config["num_tokens"]),
        num_single_layers=_resolve_saved_single_ip_layer_count(ip_state),
    )
    patch_flux_single_ip_forward(pipe.transformer)
    pipe.transformer.encoder_hid_proj.load_state_dict(ip_state["encoder_hid_proj"])
    for i, block in enumerate(pipe.transformer.transformer_blocks):
        block.attn.processor.to_k_ip.load_state_dict(ip_state[f"block_{i}_to_k_ip"])
        block.attn.processor.to_v_ip.load_state_dict(ip_state[f"block_{i}_to_v_ip"])
    for i, block in enumerate(getattr(pipe.transformer, "single_transformer_blocks", [])):
        k_key = f"single_block_{i}_to_k_ip"
        v_key = f"single_block_{i}_to_v_ip"
        if k_key in ip_state and v_key in ip_state:
            block.attn.processor.to_k_ip.load_state_dict(ip_state[k_key])
            block.attn.processor.to_v_ip.load_state_dict(ip_state[v_key])
    _move_ip_adapter_modules(pipe.transformer, device=device, torch_dtype=dtype)

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
        flux_pipeline=pipe,
        controlnet=controlnet,
        condition_modules=modules,
        control_spec=control_spec,
        ip_adapter_modules=ip_adapter_modules,
        ref_encoder=modules["ref_encoder"],
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


def run_cross_v1_bundle(
    bundle: CrossV1InferenceBundle,
    reference_image: torch.Tensor,
    reference_tissue_mask: torch.Tensor,
    reference_nuclei_mask: torch.Tensor,
    target_tissue_mask: torch.Tensor,
    target_nuclei_mask: torch.Tensor,
    prompt: str,
) -> Image.Image:
    # Encode reference image via UNI2-h + Perceiver resampler
    ref_features = bundle.ref_encoder(
        reference_image.unsqueeze(0).to(device=bundle.device, dtype=bundle.torch_dtype)
    )
    ref_features = ref_features.to(device=bundle.device, dtype=bundle.torch_dtype)
    ip_hidden_states = bundle.flux_pipeline.transformer.encoder_hid_proj([ref_features])
    ip_hidden_states = [
        hidden.to(device=bundle.device, dtype=bundle.torch_dtype)
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
    if bundle.control_spec.spatial_mode == CROSS_V1_SPATIAL_REFERENCE_TARGET:
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

    output_size = tuple(int(v) for v in reference_image.shape[1:])
    return _sample_with_flux_controlnet(
        pipe=bundle.flux_pipeline,
        controlnet=bundle.controlnet,
        prompt=prompt,
        control_tensor=control_tensor,
        output_size=output_size,
        device=bundle.device,
        torch_dtype=bundle.torch_dtype,
        num_inference_steps=bundle.num_inference_steps,
        guidance_scale=bundle.guidance_scale,
        controlnet_conditioning_scale=bundle.controlnet_conditioning_scale,
        joint_attention_kwargs={"ip_hidden_states": ip_hidden_states},
    )


# ---------------------------------------------------------------------------
# Internal helpers (adapted from pipeline.py for independence)
# ---------------------------------------------------------------------------

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
        raise FileNotFoundError(
            "Missing ControlNet config.json under checkpoint path: "
            f"{checkpoint}. This usually means the directory is an older "
            "accelerate resume-only checkpoint. Run diagnose/eval against the "
            "final output directory, or a newly saved checkpoint that includes "
            "eval-ready Cross V1 artifacts."
        )
    if not (checkpoint / "phase5_conditioning.pt").exists():
        raise FileNotFoundError(f"Missing phase5_conditioning.pt under checkpoint path: {checkpoint}")
    return checkpoint


def _move_ip_adapter_modules(transformer: nn.Module, *, device: str, torch_dtype: torch.dtype) -> None:
    if hasattr(transformer, "encoder_hid_proj"):
        transformer.encoder_hid_proj.to(device=device, dtype=torch_dtype)
    for blocks in (
        getattr(transformer, "transformer_blocks", []),
        getattr(transformer, "single_transformer_blocks", []),
    ):
        for block in blocks:
            processor = getattr(getattr(block, "attn", None), "processor", None)
            for name in ("to_k_ip", "to_v_ip"):
                module = getattr(processor, name, None)
                if module is not None:
                    module.to(device=device, dtype=torch_dtype)


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
        config["num_tokens"] = int(state["ref_encoder_latent_queries"].shape[1])
    if "num_perceiver_layers" not in config:
        config["num_perceiver_layers"] = _count_ref_perceiver_layers(
            state["ref_encoder_perceiver_layers"]
        )
    config.setdefault("uni_embed_dim", 1536)
    config.setdefault("hidden_dim", 3072)
    config.setdefault("perceiver_heads", 8)
    config.setdefault("use_perceiver_self_attn", True)
    config.setdefault("skip_perceiver", False)
    config.setdefault("perceiver_cross_gate_init", None)
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
    )
    ref_encoder.proj_mlp.load_state_dict(state["ref_encoder_proj_mlp"])
    if not ref_encoder.skip_perceiver:
        ref_encoder.load_perceiver_layers_state_dict(state["ref_encoder_perceiver_layers"])
        ref_encoder.latent_queries.data.copy_(
            state["ref_encoder_latent_queries"].to(ref_encoder.latent_queries.device)
        )
        ref_encoder.perceiver_norm.load_state_dict(state["ref_encoder_perceiver_norm"])
    ref_encoder.to(device=device, dtype=torch_dtype)
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
    joint_attention_kwargs: dict | None = None,
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
    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents, latent_image_ids = pipe.prepare_latents(
        1, num_channels_latents, height, width,
        prompt_embeds.dtype, torch_device,
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
    timesteps, _ = retrieve_timesteps(
        pipe.scheduler, num_inference_steps, torch_device, sigmas=sigmas, mu=mu,
    )
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
            joint_attention_kwargs=dict(joint_attention_kwargs) if joint_attention_kwargs is not None else None,
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

'''
python -m controlnet_train.cli.eval_controlnet_flux_cross_v1   --pretrained-model-name-or-path /data/huggingface/FLUX.1-dev   --checkpoint /home/lyw/wqx-DL/flow-edit/FlowEdit-main/phase5_runs/controlnet_cross_v1/checkpoint-40000   --uni-checkpoint-path /home/lyw/wqx-DL/flow-edit/FlowEdit-main/UNI-2h/pytorch_model.bin   --metadata /home/lyw/wqx-DL/flow-edit/FlowEdit-main/phase5_runs/cross_meta/metadata_cross_val.json   --output-dir /home/lyw/wqx-DL/flow-edit/FlowEdit-main/phase5_runs/cross_v1_eval_10cases   --num-samples 10   --device cuda   --torch-dtype bf16   --prompt-source dataset
'''

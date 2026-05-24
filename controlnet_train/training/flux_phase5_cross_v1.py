"""Phase 5.3 Cross V1 training — IP-Adapter reference attention for Flux ControlNet.

This module is self-contained and does not modify any existing V0/inpaint code.
It duplicates the shared training loop from flux_phase5.py and adds:
- IP-Adapter attention installation on the frozen Flux transformer
- ReferenceImageEncoder (UNI2-h + Perceiver resampler) for reference appearance injection
- joint_attention_kwargs passing in the transformer forward call
- Separate save strategy for IP-Adapter and ref_encoder modules

Multi-GPU DDP fix: trainable sub-modules that can't be directly wrapped (because they
live inside frozen parents or contain huge frozen backbones) are extracted into small
wrapper nn.Modules and passed through accelerator.prepare() independently.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import Callable

import accelerate
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxControlNetModel,
    FluxControlNetPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from packaging import version
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, T5EncoderModel

from controlnet_train.data import CrossReconstructionDataset
from controlnet_train.data.common import default_prompt_for_dataset
from controlnet_train.modules import (
    HierarchicalTissueEmbedding,
    NucleiConditionEncoder,
    TissueConditionDownsampler,
)
from controlnet_train.modules.cross_v1_conditioning import (
    CROSS_V1_SPATIAL_REFERENCE_TARGET,
    CROSS_V1_SPATIAL_TARGET_ONLY,
    CrossV1ControlSpec,
    build_cross_v1_condition,
    normalize_cross_v1_spatial_mode,
)
from controlnet_train.modules.reference_image_encoder import ReferenceImageEncoder
from controlnet_train.training.conditioning import patch_controlnet_x_embedder

if is_wandb_available():
    import wandb  # noqa: F401

logger = get_logger(__name__)
if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


# ---------------------------------------------------------------------------
# IP-Adapter installation and helpers
# ---------------------------------------------------------------------------
class IPAdapterListProjection(nn.Module):
    """Wraps IPAdapterFullImageProjection to handle list input/output."""
    def __init__(self, proj: nn.Module):
        super().__init__()
        self.proj = proj

    def forward(self, image_embeds):
        target_dtype = next(self.proj.parameters()).dtype
        if isinstance(image_embeds, list):
            return [self.proj(embed).to(dtype=target_dtype) for embed in image_embeds]
        return self.proj(image_embeds).to(dtype=target_dtype)


def install_flux_ip_adapter_attention(
    transformer: FluxTransformer2DModel,
    hidden_dim: int = 3072,
    cross_attention_dim: int = 3072,
    num_tokens: int = 16,
    scale: float = 1.0,
    ip_init_gain: float = 0.02,
) -> None:
    """Install IP-Adapter attention processors on all double-stream blocks."""
    from diffusers.models.attention_processor import FluxIPAdapterJointAttnProcessor2_0
    from diffusers.models.embeddings import IPAdapterFullImageProjection

    raw_proj = IPAdapterFullImageProjection(
        image_embed_dim=cross_attention_dim,
        cross_attention_dim=cross_attention_dim,
    )
    transformer.encoder_hid_proj = IPAdapterListProjection(raw_proj)

    for block in transformer.transformer_blocks:
        processor = FluxIPAdapterJointAttnProcessor2_0(
            hidden_size=hidden_dim,
            cross_attention_dim=cross_attention_dim,
            num_tokens=(num_tokens,),
            scale=[scale],
        )
        for linear in processor.to_k_ip:
            _init_ip_adapter_linear(linear, gain=ip_init_gain)
        for linear in processor.to_v_ip:
            _init_ip_adapter_linear(linear, gain=ip_init_gain)
        block.attn.set_processor(processor)


def _init_ip_adapter_linear(linear: nn.Linear, *, gain: float) -> None:
    """Initialize IP K/V projections so reference content can affect attention from step 0."""
    nn.init.xavier_uniform_(linear.weight, gain=gain)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _collect_ip_adapter_modules(transformer: FluxTransformer2DModel) -> dict[str, nn.Module]:
    """Collect all IP-Adapter trainable modules attached to the frozen transformer."""
    from diffusers.models.attention_processor import FluxIPAdapterJointAttnProcessor2_0

    modules: dict[str, nn.Module] = {}
    if hasattr(transformer, "encoder_hid_proj"):
        modules["encoder_hid_proj"] = transformer.encoder_hid_proj
    for i, block in enumerate(transformer.transformer_blocks):
        processor = block.attn.processor
        if isinstance(processor, FluxIPAdapterJointAttnProcessor2_0):
            modules[f"block_{i}_to_k_ip"] = processor.to_k_ip
            modules[f"block_{i}_to_v_ip"] = processor.to_v_ip
    return modules


def _sync_ip_adapter_to_transformer(
    ip_wrapper: "IPAdapterTrainableWrapper",
    transformer: FluxTransformer2DModel,
) -> None:
    """Sync trained IP-Adapter weights from wrapper back to transformer processors.

    After accelerator.prepare(), the wrapper holds the DDP-managed parameters.
    We need the transformer's forward pass to use these exact parameter objects
    so that gradients flow correctly.
    """
    transformer.encoder_hid_proj = ip_wrapper.encoder_hid_proj
    for i, block in enumerate(transformer.transformer_blocks):
        k_key = f"block_{i}_to_k_ip"
        v_key = f"block_{i}_to_v_ip"
        if hasattr(ip_wrapper, k_key):
            block.attn.processor.to_k_ip = getattr(ip_wrapper, k_key)
            block.attn.processor.to_v_ip = getattr(ip_wrapper, v_key)


# ---------------------------------------------------------------------------
# ★ FIX 1: Wrapper modules for DDP — 把散落的可训练参数包成 nn.Module
# ---------------------------------------------------------------------------

class IPAdapterTrainableWrapper(nn.Module):
    """Wraps all IP-Adapter trainable sub-modules into one nn.Module for DDP.

    This allows accelerator.prepare() to handle gradient synchronization across GPUs
    without needing to DDP-wrap the entire frozen transformer.
    """

    def __init__(self, ip_adapter_modules: dict[str, nn.Module]):
        super().__init__()
        # Register each module as a sub-module so DDP sees them
        for name, module in ip_adapter_modules.items():
            # nn.Module attribute names can't have dots, replace if needed
            safe_name = name.replace(".", "_")
            self.add_module(safe_name, module)


class RefEncoderTrainableWrapper(nn.Module):
    """Wraps the trainable parts of ReferenceImageEncoder for DDP.

    The frozen UNI2-h backbone (~1.9B params) stays outside DDP to avoid
    broadcast hangs. Only proj_mlp, perceiver_layers, latent_queries,
    and perceiver_norm go through accelerator.prepare().
    """

    def __init__(self, ref_encoder: ReferenceImageEncoder):
        super().__init__()
        self.proj_mlp = ref_encoder.proj_mlp
        self.perceiver_layers = ref_encoder.perceiver_layers
        self.latent_queries = nn.Parameter(ref_encoder.latent_queries.data)
        self.perceiver_norm = ref_encoder.perceiver_norm

    def sync_back(self, ref_encoder: ReferenceImageEncoder) -> None:
        """After prepare(), point ref_encoder's trainable parts to our (DDP-managed) params."""
        ref_encoder.proj_mlp = self.proj_mlp
        ref_encoder.perceiver_layers = self.perceiver_layers
        ref_encoder.latent_queries = self.latent_queries
        ref_encoder.perceiver_norm = self.perceiver_norm


# ---------------------------------------------------------------------------
# Control batch builder
# ---------------------------------------------------------------------------

def _encode_images_to_latents(vae: AutoencoderKL, images: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    device = next(vae.parameters()).device
    images = images.to(device=device, dtype=dtype)
    images = images * 2.0 - 1.0
    latents = vae.encode(images).latent_dist.sample()
    return (latents - vae.config.shift_factor) * vae.config.scaling_factor


def _build_cross_v1_control_batch(
    *,
    batch: dict,
    modules: dict[str, torch.nn.Module],
    vae: AutoencoderKL,
    weight_dtype: torch.dtype,
    spatial_mode: str = CROSS_V1_SPATIAL_REFERENCE_TARGET,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(vae.parameters()).device
    target_image_latent = _encode_images_to_latents(vae, batch["target_image"], weight_dtype)

    target_tissue_feat = modules["tissue_downsampler"](
        modules["hte"](batch["target_tissue_mask"].to(device=device))
    ).to(dtype=weight_dtype)
    target_nuclei_feat = modules["nuclei_encoder"](
        batch["target_nuclei_mask"].to(device=device)
    ).to(dtype=weight_dtype)
    reference_tissue_feat = None
    reference_nuclei_feat = None
    if normalize_cross_v1_spatial_mode(spatial_mode) == CROSS_V1_SPATIAL_REFERENCE_TARGET:
        reference_tissue_feat = modules["tissue_downsampler"](
            modules["hte"](batch["reference_tissue_mask"].to(device=device))
        ).to(dtype=weight_dtype)
        reference_nuclei_feat = modules["nuclei_encoder"](
            batch["reference_nuclei_mask"].to(device=device)
        ).to(dtype=weight_dtype)

    control_tensor = build_cross_v1_condition(
        reference_tissue_feat=reference_tissue_feat,
        reference_nuclei_feat=reference_nuclei_feat,
        target_tissue_feat=target_tissue_feat,
        target_nuclei_feat=target_nuclei_feat,
        spatial_mode=spatial_mode,
    )
    return target_image_latent, control_tensor


def _build_ip_adapter_kwargs(
    batch: dict,
    modules: dict[str, torch.nn.Module],
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    transformer: FluxTransformer2DModel,
) -> dict:
    """Build joint_attention_kwargs with pre-projected ip_hidden_states."""
    ref_encoder = modules["ref_encoder"]
    uni_dtype = next(ref_encoder.uni.parameters()).dtype
    ref_ip_features = ref_encoder(
        batch["reference_image"].to(device=accelerator.device, dtype=uni_dtype)
    ).to(dtype=weight_dtype)
    ip_hidden_states = transformer.encoder_hid_proj([ref_ip_features])
    ip_hidden_states = [hs.to(dtype=weight_dtype) for hs in ip_hidden_states]
    return {"ip_hidden_states": ip_hidden_states}


def _use_self_reconstruction_reference(batch: dict) -> dict:
    """Use the target patch as the reference patch for same-patch warmup."""
    warmup_batch = dict(batch)
    warmup_batch["reference_image"] = batch["target_image"]
    warmup_batch["reference_tissue_mask"] = batch["target_tissue_mask"]
    warmup_batch["reference_nuclei_mask"] = batch["target_nuclei_mask"]
    return warmup_batch


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate_cross_batch(examples: list[dict]) -> dict:
    return {
        "target_image": torch.stack([item["target_image"] for item in examples]),
        "reference_image": torch.stack([item["reference_image"] for item in examples]),
        "target_tissue_mask": torch.stack([item["target_tissue_mask"] for item in examples]),
        "target_nuclei_mask": torch.stack([item["target_nuclei_mask"] for item in examples]),
        "reference_tissue_mask": torch.stack([item["reference_tissue_mask"] for item in examples]),
        "reference_nuclei_mask": torch.stack([item["reference_nuclei_mask"] for item in examples]),
        "prompts": [item["prompt"] for item in examples],
    }


# ---------------------------------------------------------------------------
# Prompt helpers (copied from flux_phase5.py for independence)
# ---------------------------------------------------------------------------

def _apply_training_prompt_policy(records: list[dict], args: argparse.Namespace) -> None:
    prompt_override = getattr(args, "prompt", None)
    prompt_source = getattr(args, "prompt_source", "dataset")
    if prompt_override:
        for record in records:
            record["prompt"] = prompt_override
        logger.info("Using one explicit training prompt for all %s records", len(records))
        return

    if prompt_source == "metadata":
        logger.info("Using prompts from training metadata")
        return

    if prompt_source != "dataset":
        raise ValueError(f"Unsupported prompt source: {prompt_source}")

    for record in records:
        record["prompt"] = default_prompt_for_dataset(record["dataset"])
    unique_prompts = sorted({record["prompt"] for record in records})
    logger.info(
        "Using dataset-level training prompts: %s unique prompt(s) for %s records",
        len(unique_prompts),
        len(records),
    )


def _build_prompt_cache(
    *,
    pipeline: FluxControlNetPipeline,
    prompts: list[str],
    weight_dtype: torch.dtype,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    unique_prompts = sorted(set(prompts))
    logger.info("Encoding %s unique prompt(s) from %s training records", len(unique_prompts), len(prompts))
    prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    text_ids = None
    with torch.no_grad():
        for start in range(0, len(unique_prompts), batch_size):
            prompt_batch = unique_prompts[start : start + batch_size]
            prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                prompt_batch, prompt_2=prompt_batch, device=device,
            )
            for index, prompt in enumerate(prompt_batch):
                prompt_cache[prompt] = (
                    prompt_embeds[index].to(dtype=weight_dtype, device="cpu"),
                    pooled_prompt_embeds[index].to(dtype=weight_dtype, device="cpu"),
                )
        empty_prompt_embeds, empty_pooled, text_ids = pipeline.encode_prompt(
            [""], prompt_2=[""], device=device,
        )
    if text_ids.dim() == 3:
        text_ids = text_ids[0]
    empty_prompt = (
        empty_prompt_embeds[0].to(dtype=weight_dtype, device="cpu"),
        empty_pooled[0].to(dtype=weight_dtype, device="cpu"),
    )
    return prompt_cache, empty_prompt, text_ids.to(dtype=weight_dtype, device="cpu")


def _resolve_prompt_batch(
    *,
    prompts: list[str],
    prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
    empty_prompt_embeds: torch.Tensor,
    empty_pooled: torch.Tensor,
    proportion_empty_prompts: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_prompt = []
    batch_pooled = []
    for prompt in prompts:
        if random.random() < proportion_empty_prompts:
            batch_prompt.append(empty_prompt_embeds)
            batch_pooled.append(empty_pooled)
        else:
            prompt_embeds, pooled_prompt = prompt_cache[prompt]
            batch_prompt.append(prompt_embeds)
            batch_pooled.append(pooled_prompt)
    return torch.stack(batch_prompt), torch.stack(batch_pooled)


def _prepare_packed_latent_image_ids(
    *,
    packed_height: int,
    packed_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if packed_height <= 0 or packed_width <= 0:
        raise ValueError(f"packed latent grid must be positive, got {packed_height}x{packed_width}.")
    latent_image_ids = torch.zeros(packed_height, packed_width, 3)
    latent_image_ids[..., 1] = torch.arange(packed_height)[:, None]
    latent_image_ids[..., 2] = torch.arange(packed_width)[None, :]
    latent_image_ids = latent_image_ids.reshape(packed_height * packed_width, 3)
    return latent_image_ids.to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _latest_checkpoint(output_dir: str) -> str | None:
    dirs = [directory for directory in os.listdir(output_dir) if directory.startswith("checkpoint-")]
    if not dirs:
        return None
    latest = sorted(dirs, key=lambda item: int(item.split("-")[1]))[-1]
    return os.path.join(output_dir, latest)


def _save_checkpoint(accelerator: Accelerator, args: argparse.Namespace, global_step: int) -> None:
    if args.checkpoints_total_limit is not None:
        checkpoints = [
            directory for directory in os.listdir(args.output_dir) if directory.startswith("checkpoint-")
        ]
        checkpoints = sorted(checkpoints, key=lambda item: int(item.split("-")[1]))
        if len(checkpoints) >= args.checkpoints_total_limit:
            for stale_checkpoint in checkpoints[: len(checkpoints) - args.checkpoints_total_limit + 1]:
                shutil.rmtree(os.path.join(args.output_dir, stale_checkpoint))
    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    accelerator.save_state(save_path)
    logger.info("Saved state to %s", save_path)


# ★ FIX 3: 保存时不再原地改 dtype，而是拷贝 state_dict 后转换
def _save_condition_modules(
    output_dir: str,
    modules: dict[str, nn.Module],
    unwrap_model: Callable,
    save_dtype: torch.dtype,
    *,
    control_spec: CrossV1ControlSpec,
) -> None:
    state = {
        "cross_v1_spatial_mode": control_spec.spatial_mode,
        "cross_v1_control_spec": {
            "tissue_channels": int(control_spec.tissue_channels),
            "nuclei_channels": int(control_spec.nuclei_channels),
            "spatial_mode": control_spec.spatial_mode,
            "raw_channels": int(control_spec.raw_channels),
            "packed_channels": int(control_spec.packed_channels),
        },
    }
    for name, module in modules.items():
        unwrapped = unwrap_model(module)
        if name == "ref_encoder":
            # Only save trainable parts, skip frozen UNI2-h backbone (~4GB)
            state["ref_encoder_config"] = {
                "uni_embed_dim": int(getattr(unwrapped, "uni_embed_dim", 1536)),
                "hidden_dim": int(getattr(unwrapped, "hidden_dim", 3072)),
                "num_tokens": int(getattr(unwrapped, "num_tokens", unwrapped.latent_queries.shape[1])),
                "num_perceiver_layers": int(
                    getattr(unwrapped, "num_perceiver_layers", len(unwrapped.perceiver_layers))
                ),
                "perceiver_heads": int(getattr(unwrapped, "perceiver_heads", 8)),
            }
            state["ref_encoder_proj_mlp"] = {
                k: v.to(save_dtype) for k, v in unwrapped.proj_mlp.state_dict().items()
            }
            state["ref_encoder_perceiver_layers"] = {
                k: v.to(save_dtype) for k, v in unwrapped.perceiver_layers.state_dict().items()
            }
            state["ref_encoder_latent_queries"] = unwrapped.latent_queries.data.cpu().to(save_dtype)
            state["ref_encoder_perceiver_norm"] = {
                k: v.to(save_dtype) for k, v in unwrapped.perceiver_norm.state_dict().items()
            }
        else:
            state[name] = {
                k: v.to(save_dtype) for k, v in unwrapped.state_dict().items()
            }
    torch.save(state, os.path.join(output_dir, "phase5_conditioning.pt"))


def _save_ip_adapter_modules(
    output_dir: str,
    ip_wrapper: nn.Module,
    unwrap_model: Callable,
    save_dtype: torch.dtype,
    *,
    num_tokens: int,
    ip_init_gain: float,
) -> None:
    unwrapped = unwrap_model(ip_wrapper)
    state = {
        name: {k: v.to(save_dtype) for k, v in mod.state_dict().items()}
        for name, mod in unwrapped.named_modules()
        if name  # skip root module
    }
    state["scale"] = 1.0
    state["num_tokens"] = int(num_tokens)
    state["ip_init_gain"] = float(ip_init_gain)
    torch.save(state, os.path.join(output_dir, "phase5_ip_adapter.pt"))


def _load_cross_v1_controlnet_checkpoint(
    checkpoint_path: str | Path,
    control_spec: CrossV1ControlSpec,
) -> FluxControlNetModel:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        return FluxControlNetModel.from_pretrained(str(checkpoint_path))

    controlnet_config = FluxControlNetModel.load_config(checkpoint)
    controlnet = FluxControlNetModel.from_config(controlnet_config)
    patch_controlnet_x_embedder(controlnet, control_spec.packed_channels)
    state_dict = _load_diffusers_model_state_dict(checkpoint)
    source_spec = _load_saved_cross_v1_control_spec(checkpoint)
    state_dict = _remap_cross_v1_x_embedder_state_dict(
        state_dict,
        source_spec=source_spec,
        target_spec=control_spec,
    )
    controlnet.load_state_dict(state_dict, strict=True)
    return controlnet


def _load_saved_cross_v1_control_spec(checkpoint: Path) -> CrossV1ControlSpec:
    state_path = checkpoint / "phase5_conditioning.pt"
    if not state_path.exists():
        return CrossV1ControlSpec()
    state = _torch_load_weights(state_path)
    saved_spec = state.get("cross_v1_control_spec") or {}
    return CrossV1ControlSpec(
        tissue_channels=int(saved_spec.get("tissue_channels", 64)),
        nuclei_channels=int(saved_spec.get("nuclei_channels", 16)),
        spatial_mode=str(
            saved_spec.get(
                "spatial_mode",
                state.get("cross_v1_spatial_mode", CROSS_V1_SPATIAL_REFERENCE_TARGET),
            )
        ),
    )


def _remap_cross_v1_x_embedder_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    source_spec: CrossV1ControlSpec,
    target_spec: CrossV1ControlSpec,
) -> dict[str, torch.Tensor]:
    weight_key = "controlnet_x_embedder.weight"
    if weight_key not in state_dict:
        return state_dict

    old_weight = state_dict[weight_key]
    new_in_features = target_spec.packed_channels
    if old_weight.shape[1] == new_in_features:
        return state_dict

    remapped = dict(state_dict)
    new_weight = old_weight.new_zeros((old_weight.shape[0], new_in_features))
    if (
        source_spec.spatial_mode == CROSS_V1_SPATIAL_REFERENCE_TARGET
        and target_spec.spatial_mode == CROSS_V1_SPATIAL_TARGET_ONLY
    ):
        old_start = source_spec.packed_target_start
        copy_width = min(
            source_spec.packed_target_channels,
            target_spec.packed_target_channels,
            max(0, old_weight.shape[1] - old_start),
            new_in_features,
        )
        if copy_width > 0:
            new_weight[:, :copy_width] = old_weight[:, old_start : old_start + copy_width]
    elif (
        source_spec.spatial_mode == CROSS_V1_SPATIAL_TARGET_ONLY
        and target_spec.spatial_mode == CROSS_V1_SPATIAL_REFERENCE_TARGET
    ):
        new_start = target_spec.packed_target_start
        copy_width = min(
            source_spec.packed_target_channels,
            target_spec.packed_target_channels,
            old_weight.shape[1],
            max(0, new_in_features - new_start),
        )
        if copy_width > 0:
            new_weight[:, new_start : new_start + copy_width] = old_weight[:, :copy_width]
    else:
        copy_width = min(old_weight.shape[1], new_in_features)
        new_weight[:, :copy_width] = old_weight[:, :copy_width]
    remapped[weight_key] = new_weight
    return remapped


def _load_diffusers_model_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    safetensors_index = checkpoint_path / "diffusion_pytorch_model.safetensors.index.json"
    bin_index = checkpoint_path / "diffusion_pytorch_model.bin.index.json"

    if safetensors_index.exists():
        return _load_sharded_diffusers_state_dict(safetensors_index)
    if bin_index.exists():
        return _load_sharded_diffusers_state_dict(bin_index)

    for filename in (
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.bin",
        "pytorch_model.bin",
        "model.safetensors",
    ):
        weight_path = checkpoint_path / filename
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


def _resolve_conditioning_checkpoint_path(args: argparse.Namespace) -> Path | None:
    path = getattr(args, "conditioning_checkpoint", None)
    if path:
        return Path(path)
    path = getattr(args, "a1_lite_conditioning_checkpoint", None)
    if path:
        return Path(path)
    controlnet_path = getattr(args, "controlnet_model_name_or_path", None)
    return Path(controlnet_path) if controlnet_path else None


def _load_condition_modules_from_checkpoint(
    modules: dict[str, nn.Module],
    checkpoint_path: str | Path,
    *,
    load_ref_encoder: bool = False,
) -> None:
    checkpoint = Path(checkpoint_path)
    state_path = checkpoint / "phase5_conditioning.pt" if checkpoint.is_dir() else checkpoint
    if not state_path.exists():
        raise FileNotFoundError(f"Missing phase5_conditioning.pt for A1-lite: {state_path}")

    state = _torch_load_weights(state_path)
    for name in ("hte", "tissue_downsampler", "nuclei_encoder"):
        if name not in state:
            raise KeyError(f"Missing {name!r} in conditioning checkpoint: {state_path}")
        modules[name].load_state_dict(state[name])

    if load_ref_encoder:
        ref_encoder = modules["ref_encoder"]
        ref_encoder.proj_mlp.load_state_dict(state["ref_encoder_proj_mlp"])
        ref_encoder.perceiver_layers.load_state_dict(state["ref_encoder_perceiver_layers"])
        ref_encoder.latent_queries.data.copy_(
            state["ref_encoder_latent_queries"].to(ref_encoder.latent_queries.device)
        )
        ref_encoder.perceiver_norm.load_state_dict(state["ref_encoder_perceiver_norm"])


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def run_cross_v1_training(args: argparse.Namespace) -> None:
    if args.cross_version.lower() != "v1":
        raise NotImplementedError("This module implements only cross V1.")
    if args.uni_checkpoint_path is None:
        raise ValueError("--uni-checkpoint-path is required for cross V1")
    a1_lite = bool(getattr(args, "a1_lite", False))
    if a1_lite and not args.controlnet_model_name_or_path:
        raise ValueError("--a1-lite requires --controlnet_model_name_or_path with an existing Cross V1 checkpoint.")

    dataset = CrossReconstructionDataset(args.train_metadata)
    if args.max_train_samples is not None:
        dataset.records = dataset.records[: args.max_train_samples]

    control_spec = CrossV1ControlSpec(
        tissue_channels=args.tissue_out_channels,
        nuclei_channels=args.nuclei_out_channels,
        spatial_mode=getattr(args, "cross_v1_spatial_mode", CROSS_V1_SPATIAL_REFERENCE_TARGET),
    )
    self_reconstruction_warmup_steps = max(
        0,
        int(getattr(args, "self_reconstruction_warmup_steps", 0) or 0),
    )

    ref_encoder = ReferenceImageEncoder(
        uni_checkpoint_path=args.uni_checkpoint_path,
        num_tokens=args.reference_num_tokens,
        num_perceiver_layers=args.reference_num_perceiver_layers,
        perceiver_heads=args.reference_perceiver_heads,
    )

    modules = {
        "hte": HierarchicalTissueEmbedding(embedding_dim=args.tissue_embedding_dim),
        "tissue_downsampler": TissueConditionDownsampler(
            in_channels=args.tissue_embedding_dim,
            hidden_channels=args.tissue_out_channels,
            num_blocks=args.condition_downsample_blocks,
        ),
        "nuclei_encoder": NucleiConditionEncoder(
            embedding_dim=args.nuclei_embedding_dim,
            out_channels=args.nuclei_out_channels,
            num_blocks=args.condition_downsample_blocks,
        ),
        "ref_encoder": ref_encoder,
    }
    should_load_conditioning = bool(
        a1_lite or getattr(args, "load_conditioning_from_checkpoint", False)
    )
    if should_load_conditioning:
        conditioning_checkpoint = _resolve_conditioning_checkpoint_path(args)
        if conditioning_checkpoint is None:
            raise ValueError("Loading conditioning modules requires a conditioning checkpoint.")
        _load_condition_modules_from_checkpoint(
            modules,
            conditioning_checkpoint,
            load_ref_encoder=bool(getattr(args, "a1_lite_load_ref_encoder", False)),
        )

    # ---- accelerator setup ----
    logging_out_dir = Path(args.output_dir, args.logging_dir)
    print(">>> BEFORE Accelerator init", flush=True)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=str(logging_out_dir),
    )
    from datetime import timedelta
    kwargs = accelerate.InitProcessGroupKwargs(timeout=timedelta(hours=5))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
    print(f">>> AFTER Accelerator init, rank={accelerator.process_index}", flush=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    logger.info(
        "Using Cross V1 spatial mode %s: raw_channels=%s packed_channels=%s",
        control_spec.spatial_mode,
        control_spec.raw_channels,
        control_spec.packed_channels,
    )
    if self_reconstruction_warmup_steps:
        logger.info(
            "Using same-patch self-reconstruction warmup for the first %s optimizer steps",
            self_reconstruction_warmup_steps,
        )
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    _apply_training_prompt_policy(dataset.records, args)

    # ---- load models ----
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision,
    )
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
        revision=args.revision, variant=args.variant,
    ).to(accelerator.device)
    text_encoder_two = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2",
        revision=args.revision, variant=args.variant,
    ).to(accelerator.device)

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler",
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    flux_transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer",
        revision=args.revision, variant=args.variant, torch_dtype=torch.bfloat16,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
        revision=args.revision, variant=args.variant,
    )

    if args.controlnet_model_name_or_path:
        flux_controlnet = _load_cross_v1_controlnet_checkpoint(
            args.controlnet_model_name_or_path,
            control_spec=control_spec,
        )
    else:
        flux_controlnet = FluxControlNetModel.from_transformer(
            flux_transformer,
            attention_head_dim=flux_transformer.config["attention_head_dim"],
            num_attention_heads=flux_transformer.config["num_attention_heads"],
            num_layers=args.num_double_layers,
            num_single_layers=args.num_single_layers,
        )

    patch_controlnet_x_embedder(flux_controlnet, control_spec.packed_channels)
    logger.info("Patched controlnet_x_embedder to packed width %s for cross-v1", control_spec.packed_channels)

    # V1: install IP-Adapter attention on transformer
    install_flux_ip_adapter_attention(
        flux_transformer,
        num_tokens=args.reference_num_tokens,
        ip_init_gain=args.ip_init_gain,
    )
    ip_adapter_modules = _collect_ip_adapter_modules(flux_transformer)
    logger.info("Installed IP-Adapter attention (%s modules)", len(ip_adapter_modules))

    # ---- temporary pipeline for prompt encoding ----
    tmp_pipeline = FluxControlNetPipeline(
        scheduler=noise_scheduler, vae=None,
        text_encoder=text_encoder_one, tokenizer=tokenizer_one,
        text_encoder_2=text_encoder_two, tokenizer_2=tokenizer_two,
        transformer=flux_transformer, controlnet=flux_controlnet,
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    prompt_cache, empty_prompt, text_ids = _build_prompt_cache(
        pipeline=tmp_pipeline,
        prompts=[record["prompt"] for record in dataset.records],
        weight_dtype=weight_dtype,
        batch_size=args.prompt_batch_size,
        device=accelerator.device,
    )

    del tmp_pipeline, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two
    torch.cuda.empty_cache()

    # ---- freeze transformer, re-enable IP-Adapter modules ----
    flux_transformer.to(accelerator.device, dtype=weight_dtype)
    if hasattr(flux_transformer, 'encoder_hid_proj'):
        flux_transformer.encoder_hid_proj.to(dtype=weight_dtype)
    flux_transformer.requires_grad_(False)
    for module in ip_adapter_modules.values():
        module.requires_grad_(True)

    vae.to(accelerator.device, dtype=weight_dtype)
    vae.eval()
    vae.requires_grad_(False)
    flux_controlnet.to(accelerator.device, dtype=weight_dtype)
    if a1_lite:
        flux_controlnet.eval()
        flux_controlnet.requires_grad_(False)
        for name, module in modules.items():
            module.to(accelerator.device, dtype=weight_dtype)
            if name == "ref_encoder":
                module.train()
            else:
                module.eval()
                module.requires_grad_(False)
    else:
        flux_controlnet.train()
        for module in modules.values():
            module.train()
    # UNI2-h backbone inside ref_encoder stays frozen
    modules["ref_encoder"].uni.requires_grad_(False)
    modules["ref_encoder"].uni.eval()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if args.enable_xformers_memory_efficient_attention and is_xformers_available():
        flux_transformer.enable_xformers_memory_efficient_attention()
        flux_controlnet.enable_xformers_memory_efficient_attention()
    if args.gradient_checkpointing and not a1_lite:
        flux_controlnet.enable_gradient_checkpointing()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate *= (
            args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # =========================================================================
    # ★ FIX 1 & 2: 用 wrapper 包住不能直接 prepare 的可训练参数
    # =========================================================================

    # --- FIX 2: ref_encoder 可训练部分包成 wrapper ---
    ref_encoder_raw = modules.pop("ref_encoder")
    ref_encoder_raw.to(accelerator.device)
    # 冻结的 UNI backbone 手动放到 device，不过 DDP
    ref_encoder_raw.uni.to(accelerator.device)

    ref_trainable_wrapper = RefEncoderTrainableWrapper(ref_encoder_raw)

    # --- FIX 1: IP-Adapter 可训练部分包成 wrapper ---
    ip_trainable_wrapper = IPAdapterTrainableWrapper(ip_adapter_modules)

    # ---- optimizer: 现在所有可训练参数都在可 prepare 的 module 里 ----
    trainable_modules_list = [
        flux_controlnet,
        *modules.values(),            # hte, tissue_downsampler, nuclei_encoder
        ref_trainable_wrapper,         # ref_encoder 可训练部分
        ip_trainable_wrapper,          # IP-Adapter 可训练部分
    ]
    if a1_lite:
        trainable_modules_list = [
            ref_trainable_wrapper,
            ip_trainable_wrapper,
        ]
    optimizer = optimizer_class(
        [p for m in trainable_modules_list for p in m.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, collate_fn=collate_cross_batch,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers, pin_memory=True,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=(
            (args.max_train_steps or args.num_train_epochs * num_update_steps_per_epoch)
            * accelerator.num_processes
        ),
        num_cycles=args.lr_num_cycles, power=args.lr_power,
    )

    # ---- accelerator.prepare ----
    # 所有可训练 module 都过 prepare，冻结的 UNI backbone 不在里面
    n_cond_modules = len(modules)  # hte, tissue_downsampler, nuclei_encoder
    all_to_prepare = [
        flux_controlnet,
        *modules.values(),
        ref_trainable_wrapper,
        ip_trainable_wrapper,
    ]
    prepared = accelerator.prepare(
        *all_to_prepare, optimizer, train_dataloader, lr_scheduler,
    )
    n_models = len(all_to_prepare)

    flux_controlnet = prepared[0]
    prepared_cond = prepared[1 : 1 + n_cond_modules]
    modules = dict(zip(modules.keys(), prepared_cond))
    ref_trainable_wrapper = prepared[1 + n_cond_modules]
    ip_trainable_wrapper = prepared[1 + n_cond_modules + 1]

    optimizer = prepared[n_models]
    train_dataloader = prepared[n_models + 1]
    lr_scheduler = prepared[n_models + 2]

    # ★ 关键：把 prepare 后的参数同步回原始对象
    # ref_encoder: wrapper 的参数指回 ref_encoder_raw
    unwrap_model(ref_trainable_wrapper).sync_back(ref_encoder_raw)
    modules["ref_encoder"] = ref_encoder_raw

    # ip_adapter: wrapper 的参数指回 transformer 的 processor
    _sync_ip_adapter_to_transformer(unwrap_model(ip_trainable_wrapper), flux_transformer)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)))

    prompt_cache = {
        prompt: (embeds.to(device=accelerator.device), pooled.to(device=accelerator.device))
        for prompt, (embeds, pooled) in prompt_cache.items()
    }
    empty_prompt_embeds = empty_prompt[0].to(device=accelerator.device)
    empty_pooled = empty_prompt[1].to(device=accelerator.device)
    text_ids = text_ids.to(device=accelerator.device)

    total_batch_size = (
        args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )
    logger.info("***** Running Phase 5.3 cross-v1 training *****")
    logger.info("  Num examples = %s", len(dataset))
    logger.info("  Num Epochs = %s", args.num_train_epochs)
    logger.info("  Total batch size = %s", total_batch_size)
    logger.info("  Total optimization steps = %s", args.max_train_steps)

    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        checkpoint_path = (
            os.path.join(args.output_dir, args.resume_from_checkpoint)
            if args.resume_from_checkpoint != "latest"
            else _latest_checkpoint(args.output_dir)
        )
        if checkpoint_path is not None:
            accelerator.load_state(checkpoint_path)
            global_step = int(Path(checkpoint_path).name.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        initial=global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == timestep).nonzero().item() for timestep in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # ---- training loop ----
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            accumulate_model = ip_trainable_wrapper if a1_lite else flux_controlnet
            with accelerator.accumulate(accumulate_model):
                training_batch = (
                    _use_self_reconstruction_reference(batch)
                    if global_step < self_reconstruction_warmup_steps
                    else batch
                )
                with torch.no_grad() if a1_lite else contextlib.nullcontext():
                    pixel_latents, control_tensor = _build_cross_v1_control_batch(
                        batch=training_batch,
                        modules=modules,
                        vae=vae,
                        weight_dtype=weight_dtype,
                        spatial_mode=control_spec.spatial_mode,
                    )
                bsz = pixel_latents.shape[0]

                packed_pixel_latents = FluxControlNetPipeline._pack_latents(
                    pixel_latents, bsz, pixel_latents.shape[1],
                    pixel_latents.shape[2], pixel_latents.shape[3],
                )
                control_image = FluxControlNetPipeline._pack_latents(
                    control_tensor, bsz, control_tensor.shape[1],
                    control_tensor.shape[2], control_tensor.shape[3],
                )
                batch_prompt, batch_pooled = _resolve_prompt_batch(
                    prompts=batch["prompts"], prompt_cache=prompt_cache,
                    empty_prompt_embeds=empty_prompt_embeds, empty_pooled=empty_pooled,
                    proportion_empty_prompts=args.proportion_empty_prompts,
                )

                noise = torch.randn_like(packed_pixel_latents)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme, batch_size=bsz,
                    logit_mean=args.logit_mean, logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=packed_pixel_latents.device)
                sigmas = get_sigmas(timesteps, n_dim=packed_pixel_latents.ndim, dtype=packed_pixel_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * packed_pixel_latents + sigmas * noise

                guidance_vec = None
                if flux_transformer.config.guidance_embeds:
                    guidance_vec = torch.full(
                        (bsz,), args.guidance_scale,
                        device=accelerator.device, dtype=weight_dtype,
                    )

                latent_image_ids = _prepare_packed_latent_image_ids(
                    packed_height=pixel_latents.shape[2] // 2,
                    packed_width=pixel_latents.shape[3] // 2,
                    device=accelerator.device, dtype=weight_dtype,
                )
                if latent_image_ids.shape[0] != noisy_model_input.shape[1]:
                    raise ValueError(
                        "FLUX img_ids length must match packed latent sequence length: "
                        f"img_ids={tuple(latent_image_ids.shape)}, "
                        f"packed_latents={tuple(noisy_model_input.shape)}, "
                        f"unpacked_latents={tuple(pixel_latents.shape)}"
                    )

                with torch.no_grad() if a1_lite else contextlib.nullcontext():
                    controlnet_block_samples, controlnet_single_block_samples = flux_controlnet(
                        hidden_states=noisy_model_input,
                        controlnet_cond=control_image,
                        timestep=timesteps / 1000,
                        guidance=guidance_vec,
                        pooled_projections=batch_pooled,
                        encoder_hidden_states=batch_prompt,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        return_dict=False,
                )

                joint_attention_kwargs = _build_ip_adapter_kwargs(
                    training_batch, modules, accelerator, weight_dtype, flux_transformer,
                )

                noise_pred = flux_transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance_vec,
                    pooled_projections=batch_pooled,
                    encoder_hidden_states=batch_prompt,
                    controlnet_block_samples=(
                        [sample.to(dtype=weight_dtype) for sample in controlnet_block_samples]
                        if controlnet_block_samples is not None else None
                    ),
                    controlnet_single_block_samples=(
                        [sample.to(dtype=weight_dtype) for sample in controlnet_single_block_samples]
                        if controlnet_single_block_samples is not None else None
                    ),
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=dict(joint_attention_kwargs),
                    return_dict=False,
                )[0]

                loss = F.mse_loss(
                    noise_pred.float(), (noise - packed_pixel_latents).float(), reduction="mean",
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # ★ 梯度裁剪也要包含 wrapper 里的参数
                    if a1_lite:
                        all_trainable = [ref_trainable_wrapper, ip_trainable_wrapper]
                    else:
                        all_trainable = [
                            flux_controlnet,
                            *modules.values(),
                            ref_trainable_wrapper,
                            ip_trainable_wrapper,
                        ]
                    accelerator.clip_grad_norm_(
                        [p for m in all_trainable for p in m.parameters() if p.requires_grad],
                        args.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    _save_checkpoint(accelerator, args, global_step)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    # ---- save final artifacts ----
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
            args.save_weight_dtype, torch.float32,
        )
        unwrapped_controlnet = unwrap_model(flux_controlnet)
        # ★ FIX 3: 不原地改 dtype
        if args.save_weight_dtype != "fp32":
            unwrapped_controlnet.save_pretrained(args.output_dir, variant=args.save_weight_dtype)
        else:
            unwrapped_controlnet.save_pretrained(args.output_dir)
        _save_condition_modules(
            args.output_dir,
            modules,
            unwrap_model,
            save_dtype,
            control_spec=control_spec,
        )
        _save_ip_adapter_modules(
            args.output_dir,
            ip_trainable_wrapper,
            unwrap_model,
            save_dtype,
            num_tokens=args.reference_num_tokens,
            ip_init_gain=args.ip_init_gain,
        )
        logger.info("Saved Phase 5.3 cross-v1 artifacts to %s", args.output_dir)

    accelerator.end_training()

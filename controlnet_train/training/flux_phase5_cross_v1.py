"""Phase 5.3 Cross V1 training — IP-Adapter reference attention for Flux ControlNet.

This module is self-contained and does not modify any existing V0/inpaint code.
It duplicates the shared training loop from flux_phase5.py and adds:
- IP-Adapter attention installation on the frozen Flux transformer
- ReferenceImageEncoder (UNI2-h + Perceiver resampler) for reference appearance injection
- joint_attention_kwargs passing in the transformer forward call
- Separate save strategy for IP-Adapter and ref_encoder modules
"""

from __future__ import annotations

import argparse
import copy
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
from controlnet_train.modules.cross_v1_conditioning import CrossV1ControlSpec, build_cross_v1_condition
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
        # Force output dtype to match proj weights — mixed precision autocast
        # can produce float32 output even when weights are bfloat16.
        target_dtype = next(self.proj.parameters()).dtype
        if isinstance(image_embeds, list):
            return [self.proj(embed).to(dtype=target_dtype) for embed in image_embeds]
        return self.proj(image_embeds).to(dtype=target_dtype)
    
def install_flux_ip_adapter_attention(
    transformer: FluxTransformer2DModel,
    hidden_dim: int = 3072,
    cross_attention_dim: int = 3072,
    num_tokens: int = 16,
    scale: float = 0.1,
) -> None:
    """Install IP-Adapter attention processors on all double-stream blocks."""
    from diffusers.models.attention_processor import FluxIPAdapterJointAttnProcessor2_0
    from diffusers.models.embeddings import IPAdapterFullImageProjection

    # 1. Add encoder_hid_proj
    raw_proj = IPAdapterFullImageProjection(
        image_embed_dim=cross_attention_dim,
        cross_attention_dim=cross_attention_dim,
    )
    with torch.no_grad():
        ff_net = raw_proj.ff.net
        linear2 = ff_net[-1]
        linear2.weight.zero_()
        if linear2.bias is not None:
            linear2.bias.zero_()
    transformer.encoder_hid_proj = IPAdapterListProjection(raw_proj)

    # 2. Replace attention processors on double-stream blocks
    for block in transformer.transformer_blocks:
        processor = FluxIPAdapterJointAttnProcessor2_0(
            hidden_size=hidden_dim,
            cross_attention_dim=cross_attention_dim,
            num_tokens=(num_tokens,),
            scale=[scale],
        )
        with torch.no_grad():
            for linear in processor.to_k_ip:
                linear.weight.zero_()
                if linear.bias is not None:
                    linear.bias.zero_()
            for linear in processor.to_v_ip:
                linear.weight.zero_()
                if linear.bias is not None:
                    linear.bias.zero_()


        block.attn.set_processor(processor)


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
    ip_adapter_modules: dict[str, nn.Module],
    transformer: FluxTransformer2DModel,
) -> None:
    """Sync trained IP-Adapter weights from detached modules back to transformer."""
    transformer.encoder_hid_proj = ip_adapter_modules["encoder_hid_proj"]
    for i, block in enumerate(transformer.transformer_blocks):
        block.attn.processor.to_k_ip = ip_adapter_modules[f"block_{i}_to_k_ip"]
        block.attn.processor.to_v_ip = ip_adapter_modules[f"block_{i}_to_v_ip"]


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
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(vae.parameters()).device
    target_image_latent = _encode_images_to_latents(vae, batch["target_image"], weight_dtype)

    reference_tissue_feat = modules["tissue_downsampler"](
        modules["hte"](batch["reference_tissue_mask"].to(device=device))
    ).to(dtype=weight_dtype)
    reference_nuclei_feat = modules["nuclei_encoder"](
        batch["reference_nuclei_mask"].to(device=device)
    ).to(dtype=weight_dtype)
    target_tissue_feat = modules["tissue_downsampler"](
        modules["hte"](batch["target_tissue_mask"].to(device=device))
    ).to(dtype=weight_dtype)
    target_nuclei_feat = modules["nuclei_encoder"](
        batch["target_nuclei_mask"].to(device=device)
    ).to(dtype=weight_dtype)

    control_tensor = build_cross_v1_condition(
        reference_tissue_feat=reference_tissue_feat,
        reference_nuclei_feat=reference_nuclei_feat,
        target_tissue_feat=target_tissue_feat,
        target_nuclei_feat=target_nuclei_feat,
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
    # Project through encoder_hid_proj and cast to weight_dtype
    ip_hidden_states = transformer.encoder_hid_proj([ref_ip_features])
    ip_hidden_states = [hs.to(dtype=weight_dtype) for hs in ip_hidden_states]
    return {"ip_hidden_states": ip_hidden_states}


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
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    unique_prompts = sorted(set(prompts))
    logger.info("Encoding %s unique prompt(s) from %s training records", len(unique_prompts), len(prompts))
    prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    text_ids = None
    with torch.no_grad():
        for start in range(0, len(unique_prompts), batch_size):
            prompt_batch = unique_prompts[start : start + batch_size]
            prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                prompt_batch, prompt_2=prompt_batch,
            )
            for index, prompt in enumerate(prompt_batch):
                prompt_cache[prompt] = (
                    prompt_embeds[index].to(dtype=weight_dtype, device="cpu"),
                    pooled_prompt_embeds[index].to(dtype=weight_dtype, device="cpu"),
                )
        empty_prompt_embeds, empty_pooled, text_ids = pipeline.encode_prompt([""], prompt_2=[""])
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


def _save_condition_modules(
    output_dir: str,
    modules: dict[str, nn.Module],
    unwrap_model: Callable,
    save_dtype: torch.dtype,
) -> None:
    state = {}
    for name, module in modules.items():
        unwrapped = unwrap_model(module)
        if name == "ref_encoder":
            # Only save trainable parts, skip frozen UNI2-h backbone (~4GB)
            unwrapped.to(save_dtype)
            state["ref_encoder_proj_mlp"] = unwrapped.proj_mlp.state_dict()
            state["ref_encoder_perceiver_layers"] = unwrapped.perceiver_layers.state_dict()
            state["ref_encoder_latent_queries"] = unwrapped.latent_queries.data.cpu()
            state["ref_encoder_perceiver_norm"] = unwrapped.perceiver_norm.state_dict()
        else:
            unwrapped.to(save_dtype)
            state[name] = unwrapped.state_dict()
    torch.save(state, os.path.join(output_dir, "phase5_conditioning.pt"))


def _save_ip_adapter_modules(
    output_dir: str,
    ip_adapter_modules: dict[str, nn.Module],
    unwrap_model: Callable,
    save_dtype: torch.dtype,
) -> None:
    state = {}
    for name, module in ip_adapter_modules.items():
        unwrapped = unwrap_model(module)
        unwrapped.to(save_dtype)
        state[name] = unwrapped.state_dict()
    state["scale"] = 0.1  # fixed value, saved for reference
    torch.save(state, os.path.join(output_dir, "phase5_ip_adapter.pt"))


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def run_cross_v1_training(args: argparse.Namespace) -> None:
    if args.cross_version.lower() != "v1":
        raise NotImplementedError("This module implements only cross V1.")
    if args.uni_checkpoint_path is None:
        raise ValueError("--uni-checkpoint-path is required for cross V1")

    dataset = CrossReconstructionDataset(args.train_metadata)
    if args.max_train_samples is not None:
        dataset.records = dataset.records[: args.max_train_samples]

    control_spec = CrossV1ControlSpec(
        tissue_channels=args.tissue_out_channels,
        nuclei_channels=args.nuclei_out_channels,
    )

    ref_encoder = ReferenceImageEncoder(uni_checkpoint_path=args.uni_checkpoint_path)

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

    # ---- accelerator setup ----
    logging_out_dir = Path(args.output_dir, args.logging_dir)
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

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
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
        flux_controlnet = FluxControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
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
    install_flux_ip_adapter_attention(flux_transformer)
    ip_adapter_modules = _collect_ip_adapter_modules(flux_transformer)
    logger.info("Installed IP-Adapter attention (%s modules)", len(ip_adapter_modules))

    # ---- temporary pipeline for prompt encoding ----
    tmp_pipeline = FluxControlNetPipeline(
        scheduler=noise_scheduler, vae=None,
        text_encoder=text_encoder_one, tokenizer=tokenizer_one,
        text_encoder_2=text_encoder_two, tokenizer_2=tokenizer_two,
        transformer=flux_transformer, controlnet=flux_controlnet,
    )
    tmp_pipeline.to(accelerator.device)

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
    )

    del tmp_pipeline, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two
    torch.cuda.empty_cache()

    # ---- freeze transformer, re-enable IP-Adapter modules ----
    flux_transformer.to(accelerator.device, dtype=weight_dtype)
    # Ensure IP-Adapter modules (added after transformer creation) are also cast
    if hasattr(flux_transformer, 'encoder_hid_proj'):
        flux_transformer.encoder_hid_proj.to(dtype=weight_dtype)
    logger.info("=== DEBUG dtype check ===")
    for n, p in flux_transformer.encoder_hid_proj.named_parameters():
        logger.info(f"encoder_hid_proj {n}: {p.dtype}")
    for block_key in list(ip_adapter_modules.keys())[:3]:
        mod = ip_adapter_modules[block_key]
        for n, p in mod.named_parameters():
            logger.info(f"{block_key} {n}: {p.dtype}")
            break
    flux_transformer.requires_grad_(False)
    for module in ip_adapter_modules.values():
        module.requires_grad_(True)

    vae.to(accelerator.device, dtype=weight_dtype)
    vae.eval()
    vae.requires_grad_(False)
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
    if args.gradient_checkpointing:
        # Do NOT enable gradient checkpointing on transformer — diffusers 0.32.2's
        # checkpointing wrapper doesn't pass joint_attention_kwargs to blocks,
        # which breaks IP-Adapter. ControlNet checkpointing is fine.
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

    # ---- optimizer: include IP-Adapter modules, filter frozen params ----
    trainable_modules_list = [flux_controlnet, *modules.values(), *ip_adapter_modules.values()]
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

    if args.max_train_steps is None:
        num_update_steps_per_epoch = math.ceil(
            math.ceil(len(train_dataloader) / accelerator.num_processes)
            / args.gradient_accumulation_steps
        )
    else:
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
    n_modules = len(modules)
    n_ip_adapter = len(ip_adapter_modules)
    all_modules_to_prepare = [
        flux_controlnet,
        *modules.values(),
        *ip_adapter_modules.values(),
    ]
    prepared = accelerator.prepare(*all_modules_to_prepare, optimizer, train_dataloader, lr_scheduler)
    prepared_models = prepared[: len(all_modules_to_prepare)]
    flux_controlnet = prepared_models[0]

    prepared_module_values = prepared_models[1 : 1 + n_modules]
    modules = dict(zip(modules.keys(), prepared_module_values))

    ip_adapter_prepared = prepared_models[1 + n_modules : 1 + n_modules + n_ip_adapter]
    ip_adapter_modules = dict(zip(ip_adapter_modules.keys(), ip_adapter_prepared))

    # Sync IP-Adapter modules back to the frozen transformer after accelerator wrapping
    _sync_ip_adapter_to_transformer(ip_adapter_modules, flux_transformer)

    optimizer = prepared[len(all_modules_to_prepare)]
    train_dataloader = prepared[len(all_modules_to_prepare) + 1]
    lr_scheduler = prepared[len(all_modules_to_prepare) + 2]

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
            with accelerator.accumulate(flux_controlnet):
                pixel_latents, control_tensor = _build_cross_v1_control_batch(
                    batch=batch, modules=modules, vae=vae, weight_dtype=weight_dtype,
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

                # V1: build joint_attention_kwargs for IP-Adapter reference injection
                joint_attention_kwargs = _build_ip_adapter_kwargs(
                    batch, modules, accelerator, weight_dtype, flux_transformer,
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
                    all_trainable = [flux_controlnet, *modules.values(), *ip_adapter_modules.values()]
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
        unwrapped_controlnet.to(save_dtype)
        if args.save_weight_dtype != "fp32":
            unwrapped_controlnet.save_pretrained(args.output_dir, variant=args.save_weight_dtype)
        else:
            unwrapped_controlnet.save_pretrained(args.output_dir)
        _save_condition_modules(args.output_dir, modules, unwrap_model, save_dtype)
        _save_ip_adapter_modules(args.output_dir, ip_adapter_modules, unwrap_model, save_dtype)
        logger.info("Saved Phase 5.3 cross-v1 artifacts to %s", args.output_dir)

    accelerator.end_training()
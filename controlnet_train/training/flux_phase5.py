"""Phase 5 training flows built on top of the official Flux ControlNet implementation."""

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

from controlnet_train.data import InpaintDataset
from controlnet_train.data.common import default_prompt_for_dataset
from controlnet_train.modules import (
    ChangeMaskEncoder,
    HierarchicalTissueEmbedding,
    NucleiConditionEncoder,
    TissueConditionDownsampler,
)
from controlnet_train.modules.conditioning import build_inpaint_condition

from .conditioning import InpaintControlSpec, patch_controlnet_x_embedder

if is_wandb_available():
    import wandb  # noqa: F401

logger = get_logger(__name__)
if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


def collate_inpaint_batch(examples: list[dict]) -> dict:
    return {
        "target_image": torch.stack([item["target_image"] for item in examples]),
        "erased_source_image": torch.stack([item["erased_source_image"] for item in examples]),
        "target_tissue_mask": torch.stack([item["target_tissue_mask"] for item in examples]),
        "target_nuclei_mask": torch.stack([item["target_nuclei_mask"] for item in examples]),
        "change_region_mask": torch.stack([item["change_region_mask"] for item in examples]),
        "prompts": [item["prompt"] for item in examples],
    }


def run_inpaint_training(args: argparse.Namespace) -> None:
    dataset = InpaintDataset(args.train_metadata)
    if args.max_train_samples is not None:
        dataset.records = dataset.records[: args.max_train_samples]
    control_spec = InpaintControlSpec(
        tissue_channels=args.tissue_out_channels,
        nuclei_channels=args.nuclei_out_channels,
        change_channels=args.change_out_channels,
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
        "change_encoder": ChangeMaskEncoder(out_channels=args.change_out_channels),
    }
    _run_training(
        args=args,
        task_name="inpaint",
        dataset=dataset,
        collate_fn=collate_inpaint_batch,
        control_spec=control_spec,
        modules=modules,
        control_builder=lambda batch, modules, vae, weight_dtype: _build_inpaint_control_batch(
            batch=batch,
            modules=modules,
            vae=vae,
            weight_dtype=weight_dtype,
        ),
    )


def _run_training(
    *,
    args: argparse.Namespace,
    task_name: str,
    dataset,
    collate_fn: Callable[[list[dict]], dict],
    control_spec,
    modules: dict[str, torch.nn.Module],
    control_builder: Callable[[dict, dict[str, torch.nn.Module], AutoencoderKL, torch.dtype], tuple[torch.Tensor, torch.Tensor]],
) -> None:
    logging_out_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=str(logging_out_dir),
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
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
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

    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    ).to(accelerator.device)
    text_encoder_two = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant,
    ).to(accelerator.device)

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    flux_transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.bfloat16,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
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
    logger.info(
        "Patched controlnet_x_embedder to packed width %s for %s",
        control_spec.packed_channels,
        task_name,
    )

    tmp_pipeline = FluxControlNetPipeline(
        scheduler=noise_scheduler,
        vae=None,
        text_encoder=text_encoder_one,
        tokenizer=tokenizer_one,
        text_encoder_2=text_encoder_two,
        tokenizer_2=tokenizer_two,
        transformer=flux_transformer,
        controlnet=flux_controlnet,
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

    flux_transformer.to(accelerator.device, dtype=weight_dtype)
    flux_transformer.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)
    vae.eval()
    vae.requires_grad_(False)
    flux_controlnet.train()
    for module in modules.values():
        module.train()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if args.enable_xformers_memory_efficient_attention and is_xformers_available():
        flux_transformer.enable_xformers_memory_efficient_attention()
        flux_controlnet.enable_xformers_memory_efficient_attention()
    if args.gradient_checkpointing:
        flux_transformer.enable_gradient_checkpointing()
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

    trainable_modules = [flux_controlnet, *modules.values()]
    optimizer = optimizer_class(
        [parameter for model in trainable_modules for parameter in model.parameters()],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
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
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=(
            (args.max_train_steps or args.num_train_epochs * num_update_steps_per_epoch)
            * accelerator.num_processes
        ),
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    prepared = accelerator.prepare(*trainable_modules, optimizer, train_dataloader, lr_scheduler)
    prepared_models = prepared[: len(trainable_modules)]
    flux_controlnet = prepared_models[0]
    prepared_module_values = prepared_models[1:]
    modules = dict(zip(modules.keys(), prepared_module_values))
    optimizer = prepared[len(trainable_modules)]
    train_dataloader = prepared[len(trainable_modules) + 1]
    lr_scheduler = prepared[len(trainable_modules) + 2]

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)))

    prompt_cache = {
        prompt: (
            embeds.to(device=accelerator.device),
            pooled.to(device=accelerator.device),
        )
        for prompt, (embeds, pooled) in prompt_cache.items()
    }
    empty_prompt_embeds = empty_prompt[0].to(device=accelerator.device)
    empty_pooled = empty_prompt[1].to(device=accelerator.device)
    text_ids = text_ids.to(device=accelerator.device)

    total_batch_size = (
        args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )
    logger.info("***** Running Phase 5.3 %s training *****", task_name)
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
        initial=global_step,
        desc="Steps",
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

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(flux_controlnet):
                pixel_latents, control_tensor = control_builder(
                    batch,
                    modules,
                    vae,
                    weight_dtype,
                )
                bsz = pixel_latents.shape[0]

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
                batch_prompt, batch_pooled = _resolve_prompt_batch(
                    prompts=batch["prompts"],
                    prompt_cache=prompt_cache,
                    empty_prompt_embeds=empty_prompt_embeds,
                    empty_pooled=empty_pooled,
                    proportion_empty_prompts=args.proportion_empty_prompts,
                )

                noise = torch.randn_like(packed_pixel_latents)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=packed_pixel_latents.device)
                sigmas = get_sigmas(
                    timesteps,
                    n_dim=packed_pixel_latents.ndim,
                    dtype=packed_pixel_latents.dtype,
                )
                noisy_model_input = (1.0 - sigmas) * packed_pixel_latents + sigmas * noise

                guidance_vec = None
                if flux_transformer.config.guidance_embeds:
                    guidance_vec = torch.full(
                        (bsz,),
                        args.guidance_scale,
                        device=accelerator.device,
                        dtype=weight_dtype,
                    )

                latent_image_ids = _prepare_packed_latent_image_ids(
                    packed_height=pixel_latents.shape[2] // 2,
                    packed_width=pixel_latents.shape[3] // 2,
                    device=accelerator.device,
                    dtype=weight_dtype,
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

                noise_pred = flux_transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance_vec,
                    pooled_projections=batch_pooled,
                    encoder_hidden_states=batch_prompt,
                    controlnet_block_samples=(
                        [sample.to(dtype=weight_dtype) for sample in controlnet_block_samples]
                        if controlnet_block_samples is not None
                        else None
                    ),
                    controlnet_single_block_samples=(
                        [sample.to(dtype=weight_dtype) for sample in controlnet_single_block_samples]
                        if controlnet_single_block_samples is not None
                        else None
                    ),
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]

                loss = F.mse_loss(
                    noise_pred.float(),
                    (noise - packed_pixel_latents).float(),
                    reduction="mean",
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [parameter for model in [flux_controlnet, *modules.values()] for parameter in model.parameters()],
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

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
            args.save_weight_dtype,
            torch.float32,
        )
        unwrapped_controlnet = unwrap_model(flux_controlnet)
        unwrapped_controlnet.to(save_dtype)
        if args.save_weight_dtype != "fp32":
            unwrapped_controlnet.save_pretrained(args.output_dir, variant=args.save_weight_dtype)
        else:
            unwrapped_controlnet.save_pretrained(args.output_dir)
        _save_condition_modules(args.output_dir, modules, unwrap_model, save_dtype)
        logger.info("Saved Phase 5.3 %s artifacts to %s", task_name, args.output_dir)

    accelerator.end_training()


def _build_prompt_cache(
    *,
    pipeline: FluxControlNetPipeline,
    prompts: list[str],
    weight_dtype: torch.dtype,
    batch_size: int,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    unique_prompts = sorted(set(prompts))
    logger.info(
        "Encoding %s unique prompt(s) from %s training records",
        len(unique_prompts),
        len(prompts),
    )
    prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    text_ids = None
    with torch.no_grad():
        for start in range(0, len(unique_prompts), batch_size):
            prompt_batch = unique_prompts[start : start + batch_size]
            logger.info(
                "Encoding prompt cache batch %s-%s/%s",
                start + 1,
                start + len(prompt_batch),
                len(unique_prompts),
            )
            prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                prompt_batch,
                prompt_2=prompt_batch,
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


def _apply_training_prompt_policy(records: list[dict], args: argparse.Namespace) -> None:
    prompt_override = getattr(args, "prompt", None)
    prompt_source = getattr(args, "prompt_source", "metadata")
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
    """Build FLUX image ids for already-packed latent tokens.

    Diffusers helper semantics have changed across releases, so Phase 5 keeps
    this tied directly to the packed latent grid used by `_pack_latents`.
    """
    if packed_height <= 0 or packed_width <= 0:
        raise ValueError(
            f"packed latent grid must be positive, got {packed_height}x{packed_width}."
        )
    latent_image_ids = torch.zeros(packed_height, packed_width, 3)
    latent_image_ids[..., 1] = torch.arange(packed_height)[:, None]
    latent_image_ids[..., 2] = torch.arange(packed_width)[None, :]
    latent_image_ids = latent_image_ids.reshape(packed_height * packed_width, 3)
    return latent_image_ids.to(device=device, dtype=dtype)


def _encode_images_to_latents(vae: AutoencoderKL, images: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    device = next(vae.parameters()).device
    images = images.to(device=device, dtype=dtype)
    images = images * 2.0 - 1.0
    latents = vae.encode(images).latent_dist.sample()
    return (latents - vae.config.shift_factor) * vae.config.scaling_factor


def _build_inpaint_control_batch(
    *,
    batch: dict,
    modules: dict[str, torch.nn.Module],
    vae: AutoencoderKL,
    weight_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(vae.parameters()).device
    target_image_latent = _encode_images_to_latents(vae, batch["target_image"], weight_dtype)
    source_image_latent = _encode_images_to_latents(vae, batch["erased_source_image"], weight_dtype)

    target_tissue_feat = modules["tissue_downsampler"](
        modules["hte"](batch["target_tissue_mask"].to(device=device))
    ).to(dtype=weight_dtype)
    target_nuclei_feat = modules["nuclei_encoder"](
        batch["target_nuclei_mask"].to(device=device)
    ).to(dtype=weight_dtype)
    resized_change_mask = F.interpolate(
        batch["change_region_mask"].to(device=device, dtype=weight_dtype),
        size=target_image_latent.shape[2:],
        mode="nearest",
    )
    change_mask_feat = modules["change_encoder"](resized_change_mask).to(dtype=weight_dtype)

    control_tensor = build_inpaint_condition(
        source_image_latent=source_image_latent,
        target_tissue_feat=target_tissue_feat,
        target_nuclei_feat=target_nuclei_feat,
        change_mask_feat=change_mask_feat,
    )
    return target_image_latent, control_tensor


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
    modules: dict[str, torch.nn.Module],
    unwrap_model: Callable[[torch.nn.Module], torch.nn.Module],
    save_dtype: torch.dtype,
) -> None:
    state = {}
    for name, module in modules.items():
        unwrapped = unwrap_model(module)
        unwrapped.to(save_dtype)
        state[name] = unwrapped.state_dict()
    torch.save(state, os.path.join(output_dir, "phase5_conditioning.pt"))

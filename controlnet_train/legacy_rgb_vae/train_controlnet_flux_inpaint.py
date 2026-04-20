#!/usr/bin/env python
# coding=utf-8
"""
train_flux_controlnet_6ch_v2.py
================================
6通道 ControlNet: → 生成 target_image

和 inpaint 版的区别:
  - erased_image → reference_image (同WSI的另一个patch)
  - prompt 只算一次然后释放 text encoder

数据: training_pairs.json + 预计算 latent 目录
"""

import argparse
import copy
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, T5EncoderModel

import diffusers
from diffusers import FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.training_utils import compute_density_for_timestep_sampling
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

if is_wandb_available():
    import wandb

logger = get_logger(__name__)
if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False

VAE_LATENT_CH = 16


# =============================================================================
# Args
# =============================================================================
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="6ch ControlNet (ref_image + target_mask)")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--controlnet_model_name_or_path", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="controlnet-6ch-output")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--latent_dir", type=str, required=True)
    parser.add_argument("--num_double_layers", type=int, default=4)
    parser.add_argument("--num_single_layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true")
    parser.add_argument("--save_weight_dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--weighting_scheme", type=str, default="logit_normal",
                        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--default_caption", type=str, default="a H&E stained breast cancer pathology image")
    parser.add_argument("--proportion_empty_prompts", type=float, default=0)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--tracker_project_name", type=str, default="flux_controlnet_6ch_ref")
    args = parser.parse_args(input_args)
    return args


# =============================================================================
# Dataset
# =============================================================================
class PrecomputedLatentDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, latent_dir, max_samples=None):
        with open(json_path, "r") as f:
            data = json.load(f)
        self.pairs = data["pairs"]
        if max_samples is not None:
            self.pairs = self.pairs[:max_samples]
        self.image_latent_dir = os.path.join(latent_dir, "images")
        self.mask_latent_dir = os.path.join(latent_dir, "masks")

    def _load_latent(self, filepath, is_mask=False):
        pt_name = os.path.basename(filepath).rsplit(".", 1)[0] + ".pt"
        pt_path = os.path.join(
            self.mask_latent_dir if is_mask else self.image_latent_dir, pt_name
        )
        return torch.load(pt_path, map_location="cpu", weights_only=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        return {
            "target_img_latent": self._load_latent(pair["target_image"], is_mask=False),
            "target_mask_latent": self._load_latent(pair["target_mask"], is_mask=True),
            "ref_img_latent": self._load_latent(pair["reference_image"], is_mask=False),
            "ref_mask_latent": self._load_latent(pair["reference_mask"], is_mask=True),
        }


def collate_fn(examples):
    return {
        "target_img_latent": torch.stack([e["target_img_latent"] for e in examples]),
        "target_mask_latent": torch.stack([e["target_mask_latent"] for e in examples]),
        "ref_img_latent": torch.stack([e["ref_img_latent"] for e in examples]),
        "ref_mask_latent": torch.stack([e["ref_mask_latent"] for e in examples]),
    }


# =============================================================================
# Main
# =============================================================================
def main(args):
    logging_out_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=str(logging_out_dir))
    from datetime import timedelta
    kwargs = accelerate.InitProcessGroupKwargs(timeout=timedelta(hours=5))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # =========================================================================
    # 1) Text encoder → precompute prompt → release
    # =========================================================================
    logger.info("Loading text encoders...")
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision)
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
        revision=args.revision, variant=args.variant).to(accelerator.device)
    text_encoder_two = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2",
        revision=args.revision, variant=args.variant).to(accelerator.device)

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    flux_transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer",
        revision=args.revision, variant=args.variant, torch_dtype=torch.bfloat16)

    # ControlNet (6ch: ref_image + target_mask)
    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        flux_controlnet = FluxControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet from transformer")
        flux_controlnet = FluxControlNetModel.from_transformer(
            flux_transformer,
            attention_head_dim=flux_transformer.config["attention_head_dim"],
            num_attention_heads=flux_transformer.config["num_attention_heads"],
            num_layers=args.num_double_layers,
            num_single_layers=args.num_single_layers,
        )

    # === 6ch PATCH: controlnet_x_embedder 64 → 128 ===
    old_x_embedder = flux_controlnet.controlnet_x_embedder
    old_in = old_x_embedder.in_features
    new_in = old_in * 3
    out_features = old_x_embedder.out_features
    new_x_embedder = nn.Linear(new_in, out_features)
    with torch.no_grad():
        new_x_embedder.weight.zero_()
        new_x_embedder.weight[:, :old_in] = old_x_embedder.weight
        if old_x_embedder.bias is not None:
            new_x_embedder.bias.copy_(old_x_embedder.bias)
    flux_controlnet.controlnet_x_embedder = new_x_embedder
    logger.info(f"=== 6ch PATCH: controlnet_x_embedder {old_in} → {new_in} ===")

    # Tmp pipeline for encode_prompt
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

    logger.info("Precomputing prompt embeddings...")
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = tmp_pipeline.encode_prompt(
            [args.default_caption], prompt_2=[args.default_caption])
        prompt_embeds = prompt_embeds.to(dtype=weight_dtype, device="cpu")
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype, device="cpu")
        text_ids = text_ids.to(dtype=weight_dtype, device="cpu")
        empty_prompt_embeds, empty_pooled, _ = tmp_pipeline.encode_prompt([""], prompt_2=[""])
        empty_prompt_embeds = empty_prompt_embeds.to(dtype=weight_dtype, device="cpu")
        empty_pooled = empty_pooled.to(dtype=weight_dtype, device="cpu")

    if text_ids.dim() == 3:
        text_ids = text_ids[0]

    del tmp_pipeline, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two
    torch.cuda.empty_cache()
    import gc; gc.collect()
    logger.info("Text encoders released!")

    # =========================================================================
    # 2) Setup training
    # =========================================================================
    flux_transformer.to(accelerator.device, dtype=weight_dtype)
    flux_transformer.requires_grad_(False)
    flux_controlnet.train()

    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Save/load hooks
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for model in models:
                    unwrapped = unwrap_model(model)
                    if isinstance(unwrapped, FluxControlNetModel):
                        unwrapped.save_pretrained(os.path.join(output_dir, "flux_controlnet"))
                while len(weights) > 0:
                    weights.pop()

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                model = models.pop()
                load_model = FluxControlNetModel.from_pretrained(input_dir, subfolder="flux_controlnet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict(), strict=False)
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            flux_transformer.enable_xformers_memory_efficient_attention()
            flux_controlnet.enable_xformers_memory_efficient_attention()
    if args.gradient_checkpointing:
        flux_transformer.enable_gradient_checkpointing()
        flux_controlnet.enable_gradient_checkpointing()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate *= args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        flux_controlnet.parameters(), lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)

    # =========================================================================
    # Dataset & DataLoader
    # =========================================================================
    train_dataset = PrecomputedLatentDataset(
        json_path=args.train_json, latent_dir=args.latent_dir,
        max_samples=args.max_train_samples)
    logger.info(f"Training samples: {len(train_dataset)}")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn,
        batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers,
        pin_memory=True)

    # LR scheduler
    if args.max_train_steps is None:
        num_update_steps_per_epoch = math.ceil(
            math.ceil(len(train_dataloader) / accelerator.num_processes)
            / args.gradient_accumulation_steps)
    else:
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps)

    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=(
            (args.max_train_steps or args.num_train_epochs * num_update_steps_per_epoch)
            * accelerator.num_processes),
        num_cycles=args.lr_num_cycles, power=args.lr_power)

    flux_controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        flux_controlnet, optimizer, train_dataloader, lr_scheduler)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)))

    # Move prompt tensors to GPU
    prompt_embeds = prompt_embeds.to(device=accelerator.device)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=accelerator.device)
    text_ids = text_ids.to(device=accelerator.device)
    empty_prompt_embeds = empty_prompt_embeds.to(device=accelerator.device)
    empty_pooled = empty_pooled.to(device=accelerator.device)

    # =========================================================================
    # Training loop
    # =========================================================================
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running 6ch ControlNet training (ref_image + target_mask) *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total batch size = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if dirs else None
        if path is None:
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(range(0, args.max_train_steps), initial=initial_global_step,
                        desc="Steps", disable=not accelerator.is_local_main_process)

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(flux_controlnet):
                bsz = batch["target_img_latent"].shape[0]

                # 1. Load precomputed latents
                target_img_latent = batch["target_img_latent"].to(
                    device=accelerator.device, dtype=weight_dtype)
                target_mask_latent = batch["target_mask_latent"].to(
                    device=accelerator.device, dtype=weight_dtype)
                ref_img_latent = batch["ref_img_latent"].to(
                    device=accelerator.device, dtype=weight_dtype)
                ref_mask_latent = batch["ref_mask_latent"].to(
                    device=accelerator.device, dtype=weight_dtype)


                # 2. Pack target image (GT)
                pixel_latents = FluxControlNetPipeline._pack_latents(
                    target_img_latent, bsz,
                    target_img_latent.shape[1],
                    target_img_latent.shape[2],
                    target_img_latent.shape[3])

                # 3. 6ch control = [ref_image_latent, target_mask_latent] concat
                control_9ch = torch.cat([ref_img_latent, ref_mask_latent, target_mask_latent], dim=1)
# (B, 32, H, W)

                control_image = FluxControlNetPipeline._pack_latents(
                    control_9ch, bsz,
                    VAE_LATENT_CH * 3,
                    ref_img_latent.shape[2],
                    ref_img_latent.shape[3])

                # 4. Prompt
                batch_prompt = []
                batch_pooled_list = []
                for _ in range(bsz):
                    if random.random() < args.proportion_empty_prompts:
                        batch_prompt.append(empty_prompt_embeds[0])
                        batch_pooled_list.append(empty_pooled[0])
                    else:
                        batch_prompt.append(prompt_embeds[0])
                        batch_pooled_list.append(pooled_prompt_embeds[0])
                batch_prompt = torch.stack(batch_prompt)
                batch_pooled_t = torch.stack(batch_pooled_list)

                # 5. Noise + timesteps
                noise = torch.randn_like(pixel_latents)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme, batch_size=bsz,
                    logit_mean=args.logit_mean, logit_std=args.logit_std,
                    mode_scale=args.mode_scale)
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)
                sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise

                if flux_transformer.config.guidance_embeds:
                    guidance_vec = torch.full(
                        (bsz,), args.guidance_scale,
                        device=accelerator.device, dtype=weight_dtype)
                else:
                    guidance_vec = None

                latent_image_ids = FluxControlNetPipeline._prepare_latent_image_ids(
                    batch_size=bsz,
                    height=target_img_latent.shape[2] // 2,
                    width=target_img_latent.shape[3] // 2,
                    device=accelerator.device, dtype=weight_dtype)

                # 6. ControlNet forward
                controlnet_block_samples, controlnet_single_block_samples = flux_controlnet(
                    hidden_states=noisy_model_input,
                    controlnet_cond=control_image,
                    timestep=timesteps / 1000,
                    guidance=guidance_vec,
                    pooled_projections=batch_pooled_t,
                    encoder_hidden_states=batch_prompt,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False)

                # 7. Transformer forward
                noise_pred = flux_transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance_vec,
                    pooled_projections=batch_pooled_t,
                    encoder_hidden_states=batch_prompt,
                    controlnet_block_samples=(
                        [s.to(dtype=weight_dtype) for s in controlnet_block_samples]
                        if controlnet_block_samples is not None else None),
                    controlnet_single_block_samples=(
                        [s.to(dtype=weight_dtype) for s in controlnet_single_block_samples]
                        if controlnet_single_block_samples is not None else None),
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False)[0]

                # 8. Loss
                loss = F.mse_loss(noise_pred.float(), (noise - pixel_latents).float(), reduction="mean")
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(flux_controlnet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if (accelerator.distributed_type == DistributedType.DEEPSPEED
                        or accelerator.is_main_process):
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = [d for d in os.listdir(args.output_dir)
                                           if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                for rc in checkpoints[:len(checkpoints) - args.checkpoints_total_limit + 1]:
                                    shutil.rmtree(os.path.join(args.output_dir, rc))
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # =========================================================================
    # Save final
    # =========================================================================
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        cn = unwrap_model(flux_controlnet)
        save_wt = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.save_weight_dtype, torch.float32)
        cn.to(save_wt)
        if args.save_weight_dtype != "fp32":
            cn.save_pretrained(args.output_dir, variant=args.save_weight_dtype)
        else:
            cn.save_pretrained(args.output_dir)
        logger.info(f"Saved ControlNet to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
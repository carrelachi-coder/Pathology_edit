#!/usr/bin/env python
# coding=utf-8
"""
flowedit_reference_controlnet.py
=================================
FlowEdit + Reference-based ControlNet 病理图像编辑

核心思路:
  原图 + 原mask → Reference Encoder → reference tokens (两边共享)
  原mask → ControlNet condition → V_src (源速度)
  新mask → ControlNet condition → V_tar (目标速度)
  V_delta = V_tar - V_src → 驱动编辑

  未变化区域: V_delta ≈ 0, 图像不变
  变化区域: V_delta ≠ 0, 按新mask生成

用法:
  CUDA_VISIBLE_DEVICES=6 python /home/lyw/wqx-DL/flow-edit/FlowEdit-main/edit_plan/flowedit_controlnet.py \
    --pretrained_model_name_or_path /data/huggingface/FLUX.1-dev \
    --controlnet_path /data/huggingface/controlnet_reference_output/checkpoint-11000/flux_controlnet \
    --ref_encoder_path /data/huggingface/controlnet_reference_output/checkpoint-11000/reference_encoder.pt \
    --ref_image /home/lyw/wqx-DL/flow-edit/FlowEdit-main/BCSS_dataset/images/TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500_y0_x0.png \
    --ref_mask /home/lyw/wqx-DL/flow-edit/FlowEdit-main/BCSS_dataset/conditioning/TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500_y0_x0.png \
    --target_mask /home/lyw/wqx-DL/flow-edit/FlowEdit-main/mask_data_generate/mask_edit_output/necrosis_expansion/final_with_generated_cells.png \
    --output_dir /home/lyw/wqx-DL/flow-edit/FlowEdit-main/edit_result/flowedit_ref_results \
    --tar_guidance_scale 10 \
    --n_avg 3 --n_min 4 --n_max 28
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
)
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from transformers import AutoTokenizer, CLIPTextModel, T5EncoderModel


# =============================================================================
# Reference Encoder (和训练代码一致)
# =============================================================================
class ReferenceEncoder(nn.Module):
    def __init__(self, in_features, hidden_size=3072, num_ref_tokens=64,
                 num_layers=4, num_heads=8, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_ref_tokens = num_ref_tokens
        self.input_proj = nn.Linear(in_features, hidden_size)
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=num_heads,
                dim_feedforward=hidden_size * 4, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(num_layers)
        ])
        self.query_tokens = nn.Parameter(torch.randn(1, num_ref_tokens, hidden_size) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.cross_attn_norm_q = nn.LayerNorm(hidden_size)
        self.cross_attn_norm_kv = nn.LayerNorm(hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, ref_packed_latents):
        B = ref_packed_latents.shape[0]
        x = self.input_proj(ref_packed_latents)
        for layer in self.self_attn_layers:
            x = layer(x)
        queries = self.query_tokens.expand(B, -1, -1)
        ref_tokens, _ = self.cross_attn(
            query=self.cross_attn_norm_q(queries),
            key=self.cross_attn_norm_kv(x),
            value=self.cross_attn_norm_kv(x),
        )
        return self.output_norm(ref_tokens + queries)


VAE_LATENT_CH = 16
PACK_FACTOR = 4


# =============================================================================
# FlowEdit 辅助
# =============================================================================
def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096,
                    base_shift=0.5, max_shift=1.16):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def calc_velocity(transformer, controlnet, latents, control_image,
                  prompt_embeds, pooled_prompt_embeds, guidance_vec,
                  text_ids, img_ids, t, dtype):
    """计算单次 velocity: ControlNet + Transformer"""
    timestep = t.expand(latents.shape[0]).to(dtype) / 1000

    controlnet_block_samples, controlnet_single_block_samples = controlnet(
        hidden_states=latents,
        controlnet_cond=control_image,
        timestep=timestep,
        guidance=guidance_vec,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=img_ids,
        return_dict=False,
    )

    noise_pred = transformer(
        hidden_states=latents,
        timestep=timestep,
        guidance=guidance_vec,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        controlnet_block_samples=(
            [s.to(dtype=dtype) for s in controlnet_block_samples]
            if controlnet_block_samples is not None else None
        ),
        controlnet_single_block_samples=(
            [s.to(dtype=dtype) for s in controlnet_single_block_samples]
            if controlnet_single_block_samples is not None else None
        ),
        txt_ids=text_ids,
        img_ids=img_ids,
        return_dict=False,
    )[0]

    return noise_pred


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--controlnet_path", type=str, required=True)
    parser.add_argument("--ref_encoder_path", type=str, required=True)
    parser.add_argument("--ref_encoder_config", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)

    # Input
    parser.add_argument("--ref_image", type=str, required=True)
    parser.add_argument("--ref_mask", type=str, required=True)
    parser.add_argument("--target_mask", type=str, required=True)

    # FlowEdit params
    parser.add_argument("--caption", type=str, default="a H&E stained breast cancer pathology image")
    parser.add_argument("--T_steps", type=int, default=28)
    parser.add_argument("--n_avg", type=int, default=3, help="Number of noise averages per step")
    parser.add_argument("--n_min", type=int, default=4, help="Last n_min steps: pure denoise with target")
    parser.add_argument("--n_max", type=int, default=28, help="First T-n_max steps: skip")
    parser.add_argument("--src_guidance_scale", type=float, default=1.5)
    parser.add_argument("--tar_guidance_scale", type=float, default=5.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=2)

    parser.add_argument("--output_dir", type=str, default="./flowedit_ref_results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    os.makedirs(args.output_dir, exist_ok=True)

    # =========================================================================
    # Load models
    # =========================================================================
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
        revision=args.revision, variant=args.variant,
    ).to(device, dtype=dtype)
    vae.eval()

    print("Loading Transformer...")
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer",
        revision=args.revision, variant=args.variant, torch_dtype=dtype,
    ).to(device)
    transformer.eval()

    print("Loading ControlNet...")
    controlnet = FluxControlNetModel.from_pretrained(
        args.controlnet_path
    ).to(device, dtype=dtype)
    controlnet.eval()

    print("Loading Reference Encoder...")
    if args.ref_encoder_config and os.path.exists(args.ref_encoder_config):
        with open(args.ref_encoder_config) as f:
            ref_config = json.load(f)
    else:
        ref_config = {"in_features": 128, "hidden_size": 4096,
                      "num_ref_tokens": 64, "num_layers": 4, "num_heads": 8}
    ref_encoder = ReferenceEncoder(**ref_config).to(device, dtype=dtype)
    ref_encoder.load_state_dict(
        torch.load(args.ref_encoder_path, map_location=device, weights_only=True)
    )
    ref_encoder.eval()

    print("Loading Text Encoders...")
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
        variant=args.variant).to(device)
    text_encoder_two = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2",
        variant=args.variant).to(device)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")

    # Pipeline for encode_prompt only
    pipeline = FluxControlNetPipeline(
        scheduler=scheduler, vae=vae,
        text_encoder=text_encoder_one, tokenizer=tokenizer_one,
        text_encoder_2=text_encoder_two, tokenizer_2=tokenizer_two,
        transformer=transformer, controlnet=controlnet,
    )
    pipeline.to(device)

    # =========================================================================
    # Encode prompt
    # =========================================================================
    print("Encoding prompt...")
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            [args.caption], prompt_2=[args.caption])
        prompt_embeds = prompt_embeds.to(dtype=dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=dtype)
        text_ids = text_ids.to(dtype=dtype)

    if text_ids.dim() == 3:
        text_ids = text_ids[0]

    # Release text encoders
    del text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two, pipeline
    torch.cuda.empty_cache()

    # =========================================================================
    # Load and encode images
    # =========================================================================
    print("Encoding images...")
    image_transforms = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    mask_transforms = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    ref_img_tensor = image_transforms(
        Image.open(args.ref_image).convert("RGB")
    ).unsqueeze(0).to(device, dtype=dtype)

    ref_mask_tensor = mask_transforms(
        Image.open(args.ref_mask).convert("RGB")
    ).unsqueeze(0).to(device, dtype=dtype)

    target_mask_tensor = mask_transforms(
        Image.open(args.target_mask).convert("RGB")
    ).unsqueeze(0).to(device, dtype=dtype)

    with torch.no_grad():
        # Source image latent
        x0_src_latent = vae.encode(ref_img_tensor).latent_dist.mode()
        x0_src_latent = (x0_src_latent - vae.config.shift_factor) * vae.config.scaling_factor

        # Source mask (original) latent
        src_mask_latent = vae.encode(ref_mask_tensor).latent_dist.sample()
        src_mask_latent = (src_mask_latent - vae.config.shift_factor) * vae.config.scaling_factor

        # Target mask (edited) latent
        tar_mask_latent = vae.encode(target_mask_tensor).latent_dist.sample()
        tar_mask_latent = (tar_mask_latent - vae.config.shift_factor) * vae.config.scaling_factor

        # Reference image latent (same as source)
        ref_img_latent = x0_src_latent.clone()
        ref_mask_latent = src_mask_latent.clone()

    # =========================================================================
    # Prepare packed conditions
    # =========================================================================
    print("Preparing conditions...")
    H_lat = x0_src_latent.shape[2]
    W_lat = x0_src_latent.shape[3]

    # Pack source image
    x0_src_packed = FluxControlNetPipeline._pack_latents(
        x0_src_latent, 1, VAE_LATENT_CH, H_lat, W_lat)

    # Pack source mask condition (original mask)
    src_control = FluxControlNetPipeline._pack_latents(
        src_mask_latent, 1, VAE_LATENT_CH, H_lat, W_lat)

    # Pack target mask condition (edited mask)
    tar_control = FluxControlNetPipeline._pack_latents(
        tar_mask_latent, 1, VAE_LATENT_CH, H_lat, W_lat)

    # Reference tokens (shared between src and tar)
    ref_concat = torch.cat([ref_img_latent, ref_mask_latent], dim=1)
    ref_packed = FluxControlNetPipeline._pack_latents(
        ref_concat, 1, VAE_LATENT_CH * 2, H_lat, W_lat)
    with torch.no_grad():
        ref_tokens = ref_encoder(ref_packed).to(dtype=prompt_embeds.dtype)

    # Augmented prompt = text + reference tokens
    augmented_prompt = torch.cat([prompt_embeds, ref_tokens], dim=1)

    # Text ids
    ref_text_ids = torch.zeros(
        ref_config["num_ref_tokens"], text_ids.shape[-1],
        device=device, dtype=text_ids.dtype)
    augmented_text_ids = torch.cat([text_ids, ref_text_ids], dim=0)

    # Latent image ids
    latent_image_ids = FluxControlNetPipeline._prepare_latent_image_ids(
        batch_size=1, height=H_lat // 2, width=W_lat // 2,
        device=device, dtype=dtype)

    # Guidance
    if transformer.config.guidance_embeds:
        src_guidance = torch.full((1,), args.src_guidance_scale, device=device, dtype=dtype)
        tar_guidance = torch.full((1,), args.tar_guidance_scale, device=device, dtype=dtype)
    else:
        src_guidance = tar_guidance = None

    # =========================================================================
    # Setup scheduler
    # =========================================================================
    image_seq_len = x0_src_packed.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.16),
    )

    sigmas = np.linspace(1.0, 1.0 / args.T_steps, args.T_steps)
    scheduler.set_timesteps(args.T_steps, device=device, mu=mu)
    timesteps = scheduler.timesteps

    # Release VAE temporarily
    del vae
    torch.cuda.empty_cache()

    # =========================================================================
    # FlowEdit loop
    # =========================================================================
    print(f"\nRunning FlowEdit (T={args.T_steps}, n_avg={args.n_avg}, "
          f"n_min={args.n_min}, n_max={args.n_max})")

    for sample_idx in range(args.num_samples):
        seed = args.seed + sample_idx
        torch.manual_seed(seed)

        zt_edit = x0_src_packed.clone()

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps),
                         desc=f"Sample {sample_idx+1}"):
            # Skip early steps
            if args.T_steps - i > args.n_max:
                continue

            scheduler._init_step_index(t)
            t_i = scheduler.sigmas[scheduler.step_index]
            if scheduler.step_index + 1 < len(scheduler.sigmas):
                t_im1 = scheduler.sigmas[scheduler.step_index + 1]
            else:
                t_im1 = t_i

            if args.T_steps - i > args.n_min:
                # --- FlowEdit velocity difference phase ---
                V_delta_avg = torch.zeros_like(x0_src_packed)

                for k in range(args.n_avg):
                    fwd_noise = torch.randn_like(x0_src_packed)
                    zt_src = (1 - t_i) * x0_src_packed + t_i * fwd_noise
                    zt_tar = zt_edit + zt_src - x0_src_packed

                    with torch.no_grad():
                        V_src = calc_velocity(
                            transformer, controlnet, zt_src, src_control,
                            augmented_prompt, pooled_prompt_embeds, src_guidance,
                            augmented_text_ids, latent_image_ids, t, dtype)

                        V_tar = calc_velocity(
                            transformer, controlnet, zt_tar, tar_control,
                            augmented_prompt, pooled_prompt_embeds, tar_guidance,
                            augmented_text_ids, latent_image_ids, t, dtype)

                    V_delta_avg += (1.0 / args.n_avg) * (V_tar - V_src)

                if i % 5 == 0:
                    print(f"  step {i}: ||V_delta|| = {V_delta_avg.norm().item():.4f}")

                zt_edit = zt_edit.float()
                zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg.float()
                zt_edit = zt_edit.to(dtype)

            else:
                # --- Pure denoise phase with target condition ---
                if i == args.T_steps - args.n_min:
                    fwd_noise = torch.randn_like(x0_src_packed)
                    xt_src = (1 - t_i) * x0_src_packed + t_i * fwd_noise
                    xt_tar = zt_edit + xt_src - x0_src_packed

                with torch.no_grad():
                    V_tar = calc_velocity(
                        transformer, controlnet, xt_tar, tar_control,
                        augmented_prompt, pooled_prompt_embeds, tar_guidance,
                        augmented_text_ids, latent_image_ids, t, dtype)

                xt_tar = xt_tar.float()
                xt_tar = xt_tar + (t_im1 - t_i) * V_tar.float()
                xt_tar = xt_tar.to(dtype)

        # Final output
        out_packed = zt_edit if args.n_min == 0 else xt_tar

        # Unpack
        out_latent = out_packed.view(
            1, H_lat // 2, W_lat // 2, VAE_LATENT_CH, 2, 2
        ).permute(0, 1, 4, 2, 5, 3).contiguous().view(
            1, H_lat, W_lat, VAE_LATENT_CH
        ).permute(0, 3, 1, 2).contiguous()

        # VAE decode
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae",
            revision=args.revision, variant=args.variant,
        ).to(device, dtype=dtype)
        vae.eval()

        with torch.no_grad():
            out_denorm = out_latent / vae.config.scaling_factor + vae.config.shift_factor
            image = vae.decode(out_denorm, return_dict=False)[0]

        image = (image / 2 + 0.5).clamp(0, 1)
        image_np = image[0].permute(1, 2, 0).cpu().float().numpy()
        image_np = (image_np * 255).round().astype(np.uint8)

        out_path = os.path.join(args.output_dir, f"edited_seed{seed}.png")
        Image.fromarray(image_np).save(out_path)
        print(f"  Saved: {out_path}")

        del vae
        torch.cuda.empty_cache()

    # Save references for comparison
    ref_img = Image.open(args.ref_image).convert("RGB").resize(
        (args.resolution, args.resolution), Image.LANCZOS)
    ref_img.save(os.path.join(args.output_dir, "reference_image.png"))

    ref_msk = Image.open(args.ref_mask).convert("RGB").resize(
        (args.resolution, args.resolution), Image.NEAREST)
    ref_msk.save(os.path.join(args.output_dir, "reference_mask.png"))

    tar_msk = Image.open(args.target_mask).convert("RGB").resize(
        (args.resolution, args.resolution), Image.NEAREST)
    tar_msk.save(os.path.join(args.output_dir, "target_mask.png"))

    print(f"\nDone! Results in {args.output_dir}")


if __name__ == "__main__":
    main()
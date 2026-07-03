#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed

from controlnet_train.models.flux_vae import get_flux_vae
from controlnet_train.models.uni_encoder import get_uni_encoder
from controlnet_train.pix2pix_transfer_v2.dataset import Pix2PixV2Dataset, collate_fn
from controlnet_train.pix2pix_transfer_v2.dit_backbone import Pix2PixV2DiT
from controlnet_train.pix2pix_transfer_v2.flow_matching import FlowMatching
from controlnet_train.pix2pix_transfer_v2.loss import Pix2PixV2Loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pix2Pix V2 Training: Flow-Matching DiT Refinement")
    # Data
    parser.add_argument("--train_metadata", type=str, required=True, help="Path to training metadata JSON")
    parser.add_argument("--val_metadata", type=str, required=True, help="Path to validation metadata JSON")
    parser.add_argument("--image_size", type=int, default=512, help="Input image size")
    parser.add_argument("--latent_size", type=int, default=64, help="FLUX VAE latent size (image_size / 8)")
    parser.add_argument("--stain_augment_prob", type=float, default=0.7, help="Probability of applying stain augmentation")
    parser.add_argument("--use_cache", action="store_true", default=True, help="Use cached I0 latents and ref tokens")
    parser.add_argument("--i0_cache_root", type=str, default="/data/wqx/flowedit/pix2pix_i0_lazy_cache", help="I0 latent cache root")
    parser.add_argument("--ref_cache_root", type=str, default="/data/wqx/flowedit/pix2pix_ref_token_cache", help="Ref token cache root")
    # Model
    parser.add_argument("--dit_hidden_size", type=int, default=384, help="DiT hidden dimension")
    parser.add_argument("--dit_depth", type=int, default=12, help="Number of DiT layers")
    parser.add_argument("--dit_num_heads", type=int, default=6, help="Number of DiT attention heads")
    parser.add_argument("--ref_cross_attn_start_layer", type=int, default=6, help="Start adding ref cross-attn from this layer")
    parser.add_argument("--ref_token_dim", type=int, default=1024, help="UNI/Virchow2 token dimension")
    # Training
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    # Loss weights
    parser.add_argument("--repa_weight", type=float, default=0.3, help="Weight for REPA alignment loss")
    parser.add_argument("--gram_weight", type=float, default=0.1, help="Weight for regional Gram loss")
    parser.add_argument("--latent_l1_weight", type=float, default=0.05, help="Weight for latent L1 loss")
    # Logging & Saving
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--log_every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--eval_every", type=int, default=1000, help="Evaluate every N steps")
    parser.add_argument("--save_every", type=int, default=5000, help="Save checkpoint every N steps")
    parser.add_argument("--num_eval_samples", type=int, default=8, help="Number of samples to generate during eval")
    # Resume
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint path")
    return parser.parse_args()


def encode_i0_latent(vae: nn.Module, i0_tensor: torch.Tensor) -> torch.Tensor:
    """Encode I0 RGB image to FLUX VAE latent, scaled correctly."""
    with torch.no_grad():
        latent = vae.encode(i0_tensor).latent_dist.sample()
        latent = latent * vae.scaling_factor
    return latent


def encode_ref_tokens(uni_encoder: nn.Module, ref_tensor: torch.Tensor) -> torch.Tensor:
    """Encode reference RGB image to UNI patch tokens, spatial grid flattened."""
    with torch.no_grad():
        tokens = uni_encoder.encode_patch_tokens(ref_tensor)  # (B, H, W, D)
        tokens = tokens.flatten(1, 2)  # (B, H*W, D)
    return tokens


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    accelerator = Accelerator(mixed_precision=args.mixed_precision, project_dir=args.output_dir)
    device = accelerator.device
    is_main = accelerator.is_main_process

    # Create output directories
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
        # Save config
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    # Load frozen pre-trained models
    vae = get_flux_vae(pretrained=True, freeze=True).to(device, dtype=torch.float16)
    uni_encoder = get_uni_encoder(pretrained=True, freeze=True).to(device, dtype=torch.float16)

    # Initialize model, flow matching, loss
    model = Pix2PixV2DiT(
        latent_size=args.latent_size,
        patch_size=2,
        in_channels=32,
        out_channels=16,
        hidden_size=args.dit_hidden_size,
        depth=args.dit_depth,
        num_heads=args.dit_num_heads,
        ref_cross_attn_start_layer=args.ref_cross_attn_start_layer,
        ref_token_dim=args.ref_token_dim,
    ).to(device)
    flow_matching = FlowMatching()
    loss_fn = Pix2PixV2Loss(
        dit_hidden_dim=args.dit_hidden_size,
        conch_feature_dim=512,
        repa_weight=args.repa_weight,
        gram_weight=args.gram_weight,
        latent_l1_weight=args.latent_l1_weight,
    ).to(device)

    # Load datasets
    train_dataset = Pix2PixV2Dataset(
        metadata_path=args.train_metadata,
        image_size=args.image_size,
        latent_size=args.latent_size,
        stain_augment_prob=args.stain_augment_prob,
        use_cache=args.use_cache,
        i0_latent_cache_root=args.i0_cache_root,
        ref_token_cache_root=args.ref_cache_root,
        split="train",
    )
    val_dataset = Pix2PixV2Dataset(
        metadata_path=args.val_metadata,
        image_size=args.image_size,
        latent_size=args.latent_size,
        stain_augment_prob=0.0,
        use_cache=args.use_cache,
        i0_latent_cache_root=args.i0_cache_root,
        ref_token_cache_root=args.ref_cache_root,
        split="val",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.num_eval_samples,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    # Optimizer
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] + [p for p in loss_fn.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Prepare for distributed training
    model, loss_fn, optimizer, train_loader, val_loader = accelerator.prepare(
        model, loss_fn, optimizer, train_loader, val_loader
    )
    # Sync batch norm
    if accelerator.num_processes > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Resume from checkpoint
    global_step = 0
    if args.resume:
        if is_main:
            print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = checkpoint["global_step"]

    # Get target latent for loss
    @torch.no_grad()
    def get_target_latent(target_tensor: torch.Tensor) -> torch.Tensor:
        target_tensor = target_tensor.to(dtype=torch.float16)
        return vae.encode(target_tensor).latent_dist.sample() * vae.scaling_factor

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        loss_fn.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", disable=not is_main)

        for batch in pbar:
            optimizer.zero_grad()
            batch_size = batch["i0"].shape[0]

            # Encode I0 latent and ref tokens if not cached
            i0_latent = batch.get("i0_latent_cached")
            if i0_latent is None:
                i0_latent = encode_i0_latent(vae, batch["i0"].to(device, dtype=torch.float16))
            else:
                i0_latent = i0_latent.to(device, dtype=torch.float16)

            ref_tokens = batch.get("ref_tokens_cached")
            if ref_tokens is None:
                ref_tokens = encode_ref_tokens(uni_encoder, batch["reference"].to(device, dtype=torch.float16))
            else:
                ref_tokens = [t.to(device, dtype=torch.float16) for t in ref_tokens]
                ref_tokens = torch.stack(ref_tokens)

            # Get target latent
            target_latent = get_target_latent(batch["target"].to(device))

            # Sample timesteps and noisy latent
            t = flow_matching.sample_timesteps(batch_size, device=device)
            z_t, target_v = flow_matching.get_noisy_latent_and_velocity(target_latent, t)

            # Forward pass
            pred_v = model(z_t, i0_latent, t, ref_tokens)

            # Compute loss
            loss_dict = loss_fn(
                pred_v=pred_v,
                target_v=target_v,
                pred_latent=z_t + pred_v * (1 - t.view(batch_size, 1, 1, 1)),  # predict z1 from z_t + v * (1 - t)
                target_latent=target_latent,
                target_image=batch["target"].to(device),
                tissue_mask=batch["tissue_mask"].to(device),
            )
            total_loss = loss_dict["total_loss"]

            # Backward pass
            accelerator.backward(total_loss)
            optimizer.step()

            # Log
            if is_main and global_step % args.log_every == 0:
                for k, v in loss_dict.items():
                    writer.add_scalar(f"train/{k}", v.item(), global_step)
                pbar.set_postfix({k: f"{v.item():.4f}" for k, v in loss_dict.items()})

            # Evaluate
            if is_main and global_step % args.eval_every == 0:
                model.eval()
                loss_fn.eval()
                with torch.no_grad():
                    val_batch = next(iter(val_loader))
                    # Encode
                    val_i0_latent = encode_i0_latent(vae, val_batch["i0"].to(device, dtype=torch.float16))
                    val_ref_tokens = encode_ref_tokens(uni_encoder, val_batch["reference"].to(device, dtype=torch.float16))
                    # Sample
                    val_gen_latent = flow_matching.sample(
                        model=accelerator.unwrap_model(model),
                        i0_latent=val_i0_latent,
                        ref_tokens=val_ref_tokens,
                        num_steps=16,
                        device=device,
                        dtype=torch.float16,
                    )
                    # Decode to RGB
                    val_gen_image = vae.decode(val_gen_latent / vae.scaling_factor).sample
                    val_gen_image = torch.clamp(val_gen_image, -1, 1)
                    # Save samples
                    for i in range(min(args.num_eval_samples, val_gen_image.shape[0])):
                        img = (val_gen_image[i].permute(1, 2, 0).cpu().numpy() + 1) / 2
                        img = (img * 255).astype(np.uint8)
                        Image.fromarray(img).save(os.path.join(args.output_dir, "samples", f"step_{global_step}_sample_{i}.png"))
                model.train()
                loss_fn.train()

            # Save checkpoint
            if is_main and global_step % args.save_every == 0:
                checkpoint = {
                    "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                    "config": vars(args),
                }
                torch.save(checkpoint, os.path.join(args.output_dir, "checkpoints", f"checkpoint_step_{global_step}.pt"))

            global_step += 1
            pbar.update(1)

        pbar.close()

    # Save final checkpoint
    if is_main:
        checkpoint = {
            "model_state_dict": accelerator.unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            "config": vars(args),
        }
        torch.save(checkpoint, os.path.join(args.output_dir, "checkpoints", "final_checkpoint.pt"))
        writer.close()
    accelerator.end_training()


if __name__ == "__main__":
    main()

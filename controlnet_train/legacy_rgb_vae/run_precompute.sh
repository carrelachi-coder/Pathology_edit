#!/bin/bash
# ============================================================
# Step 1: 预计算 VAE latent (只需跑一次, 约10-15分钟)
# ============================================================
# 用一张空闲卡就行

export MODEL_DIR="/data/huggingface/FLUX.1-dev"  # 改成你的本地路径
export IMAGE_DIR="/home/lyw/wqx-DL/flow-edit/FlowEdit-main/BCSS_dataset/images"
export MASK_DIR="/home/lyw/wqx-DL/flow-edit/FlowEdit-main/BCSS_dataset/conditioning"
export LATENT_DIR="/data/huggingface/controlnet_reference_output/latents"

CUDA_VISIBLE_DEVICES=1,2,4,7 python /home/lyw/wqx-DL/flow-edit/FlowEdit-main/controlnet_reference_train/precompute_vae_latents.py \
  --pretrained_model_name_or_path=$MODEL_DIR \
  --image_dir=$IMAGE_DIR \
  --mask_dir=$MASK_DIR \
  --latent_dir=$LATENT_DIR \
  --resolution=512 \
  --batch_size=32





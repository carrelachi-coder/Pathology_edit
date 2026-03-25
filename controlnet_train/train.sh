#!/bin/bash
# =============================================================
# train_inpaint_controlnet.sh
# 多卡训练 Inpainting ControlNet
# GPU 4, 6, 7 空闲，共 3 卡
# =============================================================
export CUDA_VISIBLE_DEVICES=2,3,4,7
export MODEL_DIR="/data/huggingface/FLUX.1-dev"  # 改成你的路径
export OUTPUT_DIR="/data/huggingface/controlnet_6ch_v2_output"
export TRAIN_JSON="/home/lyw/wqx-DL/flow-edit/FlowEdit-main/controlnet_train/training_pairs.json"
export LATENT_DIR="/data/huggingface/controlnet_output/latents"
echo "=============================="
echo "训练 Inpainting ControlNet (4x A800)"
echo "GPUs: 2,3,4,7"
echo "=============================="

NCCL_TIMEOUT=3600 \
NCCL_P2P_DISABLE=0 \
TORCH_NCCL_BLOCKING_WAIT=0 \
NCCL_ASYNC_ERROR_HANDLING=1 \
accelerate launch --multi_gpu --num_processes=4 /home/lyw/wqx-DL/flow-edit/FlowEdit-main/controlnet_train/train_controlnet_flux.py \
  --pretrained_model_name_or_path=$MODEL_DIR \
  --train_json=$TRAIN_JSON \
  --latent_dir=$LATENT_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --num_train_epochs=15 \
  --learning_rate=1e-5 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=200 \
  --checkpointing_steps=2000 \
  --checkpoints_total_limit=3 \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --use_8bit_adam \
  --allow_tf32 \
  --seed=42 \
  --dataloader_num_workers=8 \
  --num_double_layers=4 \
  --num_single_layers=4 \
  --guidance_scale=3.5 \
  --default_caption="a H&E stained breast cancer pathology image" \
  --report_to="tensorboard" \
  --resume_from_checkpoint=latest \
  --tracker_project_name="flux_controlnet_6ch_ref"
 
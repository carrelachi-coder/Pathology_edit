#!/usr/bin/env python3
"""
ProbNet 训练入口 (Phase 4.1 — Embedding-based)

任务：给定组织层 + 已有细胞核（编辑区域内清零）+ 编辑mask + 癌种ID
     预测编辑区域内每个像素的核类型概率

输入: tissue_map(int 0-15) + cell_map(int 0-5) + mask(float) + cancer_id(int 0-5)
     → ProbNetInputEncoder (Embedding lookup) → (B, 17, H, W) → UNet
输出: 核类型概率 (6ch): [背景, neoplastic, inflammatory, connective, dead, epithelial]

用法:
    # 训练 (单数据集, 旧格式兼容)
    CUDA_VISIBLE_DEVICES=5 python inpaint_cells/train.py \
        --data-dir /data/huggingface/dataset_for_mask_edit \
        --output-dir /data/huggingface/pathology_edit/prob_net \
        --batch-size 16 --num-epochs 100

    # 训练 (多数据集)
    python inpaint_cells/train.py \
        --datasets BCSS:/data/bcss_probnet PANDA:/data/panda_probnet \
        --output-dir /data/huggingface/pathology_edit/prob_net \
        --batch-size 16 --num-epochs 100

    # 从 checkpoint 恢复训练
    python inpaint_cells/train.py \
        --data-dir ... --output-dir ... \
        --resume-from-checkpoint latest

    # 从旧 flat-embedding checkpoint 启动层级化短程微调
    python inpaint_cells/train.py \
        --datasets ... --output-dir /path/to/new_run \
        --init-from-checkpoint /path/to/old/best.pt \
        --fine-to-parent-dropout 0.25 --validate-coarse-fallback

    # 推理（结合实例库）
    python inpaint_cells/train.py --mode inference \
        --ckpt /path/to/best.pt \
        --library /path/to/nuclei_library \
        --data-dir /path/to/data
"""

import os
import sys
import glob
import random
import logging
import argparse
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# 允许从项目根目录直接 python inpaint_cells/train.py 运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inpaint_cells.models.prob_unet import (
    ProbUNet,
    apply_fine_to_parent_dropout,
    collapse_fine_to_parent,
    freeze_non_density_parameters,
)
from inpaint_cells.data.prob_dataset import (
    NucleiProbDatasetLayered, NucleiProbDatasetLegacy,
    build_multi_dataset,
)
from inpaint_cells.losses.focal_dice import CenterDensityLoss, FocalDiceLoss
from inpaint_cells.utils.mask_utils import (
    NUM_TISSUE, NUM_NUCLEI, NUCLEI_CLASSES,
    NUCLEI_RAW_TO_INDEX, NUCLEI_INDEX_TO_RAW,
    overlay, index_to_rgb, NUCLEI_RGB, TISSUE_RGB_MAP,
    load_tissue_mask, load_nuclei_mask, save_nuclei_mask,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
#  Checkpoint 解析
# ============================================================

def _resolve_resume_checkpoint(args):
    """
    解析 --resume-from-checkpoint 参数。
    支持: "latest" → 最新 epoch_*.pt, 具体路径, 或 None
    """
    resume = args.resume_from_checkpoint
    if resume is None:
        return None

    if resume == "latest":
        ckpt_dir = os.path.join(args.output_dir, "checkpoints")
        if not os.path.isdir(ckpt_dir):
            logger.warning(f"No checkpoints dir found at {ckpt_dir}, training from scratch")
            return None

        epoch_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")))
        best_pt = os.path.join(ckpt_dir, "best.pt")

        candidates = []
        for p in epoch_ckpts:
            try:
                ep = int(os.path.basename(p).replace("epoch_", "").replace(".pt", ""))
                candidates.append((ep, p))
            except ValueError:
                pass
        if os.path.exists(best_pt):
            try:
                ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
                candidates.append((ckpt.get("epoch", -1), best_pt))
            except Exception:
                pass

        if not candidates:
            logger.warning(f"No checkpoint files found in {ckpt_dir}, training from scratch")
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        chosen = candidates[0][1]
        logger.info(f"Resolved 'latest' → {chosen}")
        return chosen

    if os.path.exists(resume):
        return resume

    logger.warning(f"Checkpoint not found: {resume}, training from scratch")
    return None


def _checkpoint_uses_flat_tissue_embedding(state_dict):
    return (
        'input_encoder.tissue_emb.weight' in state_dict
        and 'input_encoder.tissue_emb.parent_embeddings.weight' not in state_dict
    )


def _load_model_weights(model, checkpoint_path, device):
    """Load model weights and report whether the legacy tissue table was migrated."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model', checkpoint)
    migrated = _checkpoint_uses_flat_tissue_embedding(state_dict)
    checkpoint_has_density = any(key.startswith('density_head.') for key in state_dict)
    if model.with_density_head and not checkpoint_has_density:
        result = model.load_state_dict(state_dict, strict=False)
        invalid_missing = [
            key for key in result.missing_keys
            if not key.startswith('density_head.')
        ]
        if invalid_missing or result.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint mismatch: missing={invalid_missing}, "
                f"unexpected={result.unexpected_keys}."
            )
    else:
        model.load_state_dict(state_dict)
    return checkpoint, migrated


def _checkpoint_metadata(args):
    return {
        'model_format_version': 5 if args.center_density_head else 2,
        'tissue_embedding': 'hierarchical_parent_delta',
        'center_density_head': args.center_density_head,
        'density_channels': 5 if args.center_density_head else 0,
        'density_sigma': args.density_sigma,
        'density_loss_weight': args.density_loss_weight,
        'count_loss_weight': args.count_loss_weight,
        'total_count_loss_weight': args.total_count_loss_weight,
        'density_init_bias': list(args.density_init_bias),
        'density_empty_group_weight': args.density_empty_group_weight,
        'density_high_count_threshold': args.density_high_count_threshold,
        'density_high_count_weight': args.density_high_count_weight,
        'empty_sample_fp_loss_weight': args.empty_sample_fp_loss_weight,
        'density_head_only': args.density_head_only,
        'complete_instance_erasure': args.complete_instance_erasure,
        'instance_definition': 'per_class_8_connected_components',
        'checkpoint_metric': args.checkpoint_metric,
        'base_ch': args.base_ch,
        'fine_to_parent_dropout': args.fine_to_parent_dropout,
        'tissue_delta_l2_weight': args.tissue_delta_l2_weight,
        'early_stopping_patience': args.early_stopping_patience,
        'seed': args.seed,
        'initialization_checkpoint': args.init_from_checkpoint,
        'training_from_scratch': args.init_from_checkpoint is None,
    }


# ============================================================
#  训练
# ============================================================

def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.checkpoint_metric == 'auto':
        args.checkpoint_metric = (
            'count_rmae_macro' if args.center_density_head else 'native_loss'
        )
    val_num_workers = 0 if args.num_workers == 0 else max(1, args.num_workers // 2)

    # ---- 数据 ----
    if args.datasets:
        # 多数据集模式: --datasets BCSS:/data/bcss PANDA:/data/panda
        dataset_configs = []
        for spec in args.datasets:
            name, path = spec.split(':', 1)
            dataset_configs.append({'dataset_name': name, 'data_dir': path})
        train_dataset, train_sampler = build_multi_dataset(
            dataset_configs, split='train', out_size=args.img_size, augment=True,
            crop_mode=args.crop_mode,
            center_density_targets=args.center_density_head,
            density_sigma=args.density_sigma,
            complete_instance_erasure=args.complete_instance_erasure)
        val_dataset, _ = build_multi_dataset(
            dataset_configs, split='val', out_size=args.img_size, augment=False,
            crop_mode=args.crop_mode,
            center_density_targets=args.center_density_head,
            density_sigma=args.density_sigma,
            complete_instance_erasure=args.complete_instance_erasure)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  sampler=train_sampler, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=val_num_workers,
                                pin_memory=True)
    else:
        # 单数据集 (兼容旧目录结构)
        cancer_idx = args.cancer_type_index
        # Auto-detect format
        has_layered = (
            os.path.isdir(os.path.join(args.data_dir, 'gt_tissue'))
            or os.path.isdir(os.path.join(args.data_dir, 'train', 'gt_tissue'))
            or os.path.isdir(os.path.join(args.data_dir, 'val', 'gt_tissue'))
            or len(glob.glob(os.path.join(args.data_dir, '*', 'tissue_mask.png'))) > 0
        )
        if has_layered:
            has_train_val = (
                os.path.isdir(os.path.join(args.data_dir, 'train'))
                and os.path.isdir(os.path.join(args.data_dir, 'val'))
            )
            if args.mode == 'train' and not has_train_val and not args.allow_flat_single_dataset:
                raise ValueError(
                    "Single-dataset layered training requires train/ and val/ subdirectories. "
                    f"Got flat data_dir={args.data_dir}. Re-run prepare_dataset.py, or pass "
                    "--allow-flat-single-dataset only for intentional debugging."
                )
            train_dataset = NucleiProbDatasetLayered(
                data_dir=os.path.join(args.data_dir, 'train') if os.path.isdir(os.path.join(args.data_dir, 'train')) else args.data_dir,
                cancer_type_index=cancer_idx, out_size=args.img_size, augment=True,
                crop_mode=args.crop_mode,
                dataset_name='single',
                center_density_targets=args.center_density_head,
                density_sigma=args.density_sigma,
                complete_instance_erasure=args.complete_instance_erasure)
            val_dataset = NucleiProbDatasetLayered(
                data_dir=os.path.join(args.data_dir, 'val') if os.path.isdir(os.path.join(args.data_dir, 'val')) else args.data_dir,
                cancer_type_index=cancer_idx, out_size=args.img_size, augment=False,
                crop_mode=args.crop_mode,
                dataset_name='single',
                center_density_targets=args.center_density_head,
                density_sigma=args.density_sigma,
                complete_instance_erasure=args.complete_instance_erasure)
        else:
            train_dataset = NucleiProbDatasetLegacy(
                gt_dir=os.path.join(args.data_dir, 'ground_truth'),
                train_dir=os.path.join(args.data_dir, 'train'),
                cancer_type_index=cancer_idx, out_size=args.img_size, augment=True,
                crop_mode=args.crop_mode,
                dataset_name='single',
                center_density_targets=args.center_density_head,
                density_sigma=args.density_sigma,
                complete_instance_erasure=args.complete_instance_erasure)
            val_dataset = NucleiProbDatasetLegacy(
                gt_dir=os.path.join(args.data_dir, 'ground_truth'),
                train_dir=os.path.join(args.data_dir, 'val'),
                cancer_type_index=cancer_idx, out_size=args.img_size, augment=False,
                crop_mode=args.crop_mode,
                dataset_name='single',
                center_density_targets=args.center_density_head,
                density_sigma=args.density_sigma,
                complete_instance_erasure=args.complete_instance_erasure)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=val_num_workers, pin_memory=True)

    # ---- 模型 (Phase 4.1: Embedding-based, 无需 in_ch 参数) ----
    model = ProbUNet(
        out_ch=NUM_NUCLEI,
        base_ch=args.base_ch,
        with_density_head=args.center_density_head,
        density_init_bias=args.density_init_bias,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"ProbUNet parameters: {num_params:.1f}M (Embedding input, 17ch)")

    # Loss + 优化器
    criterion = FocalDiceLoss(num_classes=NUM_NUCLEI, mask_weight=args.mask_weight).to(device)
    density_criterion = CenterDensityLoss(
        num_tissues=NUM_TISSUE,
        empty_group_weight=args.density_empty_group_weight,
        high_count_threshold=args.density_high_count_threshold,
        high_count_weight=args.density_high_count_weight,
    ).to(device)
    if args.density_head_only:
        freeze_non_density_parameters(model)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError('No trainable ProbNet parameters remain.')
    logger.info(
        'Trainable parameters: %d/%d (%s)',
        sum(parameter.numel() for parameter in trainable_parameters),
        sum(parameter.numel() for parameter in model.parameters()),
        ', '.join(trainable_names),
    )
    optimizer = AdamW(trainable_parameters, lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    # 输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
    vis_dir = os.path.join(args.output_dir, 'vis')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(args.output_dir, 'tb_logs'))
    except ImportError:
        logger.warning('tensorboard is unavailable; continuing without event logs.')

        class _NoOpSummaryWriter:
            def add_scalar(self, *args, **kwargs):
                return None

            def close(self):
                return None

        writer = _NoOpSummaryWriter()

    global_step = 0
    best_val_loss = float('inf')
    best_selection_score = float('inf')
    start_epoch = 0
    epochs_without_improvement = 0

    if args.init_from_checkpoint and args.resume_from_checkpoint:
        raise ValueError(
            "Use either --init-from-checkpoint for a new fine-tuning run or "
            "--resume-from-checkpoint to continue the same run, not both."
        )

    # Weights-only initialization deliberately starts a fresh optimizer and LR schedule.
    if args.init_from_checkpoint:
        if not os.path.isfile(args.init_from_checkpoint):
            raise FileNotFoundError(f"Initialization checkpoint not found: {args.init_from_checkpoint}")
        logger.info(f"Initializing model weights from: {args.init_from_checkpoint}")
        init_ckpt, migrated = _load_model_weights(model, args.init_from_checkpoint, device)
        if migrated:
            logger.info("  Losslessly migrated legacy flat 16x8 tissue embedding to parent + delta.")
        logger.info(
            f"  Loaded source epoch={init_ckpt.get('epoch', 'unknown')}; "
            "optimizer and scheduler start fresh."
        )

    # Resume
    resume_path = _resolve_resume_checkpoint(args)
    if resume_path is not None:
        logger.info(f"Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        state_dict = ckpt['model']
        if _checkpoint_uses_flat_tissue_embedding(state_dict):
            raise ValueError(
                "A legacy flat-embedding checkpoint cannot safely restore optimizer state. "
                "Start a new output directory with --init-from-checkpoint instead."
            )
        model.load_state_dict(state_dict)
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        else:
            for _ in range(ckpt.get('epoch', 0) + 1):
                scheduler.step()
        start_epoch = ckpt.get('epoch', 0) + 1
        global_step = ckpt.get('global_step', start_epoch * len(train_loader))
        best_val_loss = ckpt.get('best_val_loss', ckpt.get('val_loss', best_val_loss))
        best_selection_score = ckpt.get(
            'best_selection_score',
            ckpt.get('selection_score', best_val_loss),
        )
        logger.info(f"  Resuming from epoch {start_epoch}, global_step={global_step}")

    # 训练循环
    for epoch in range(start_epoch, args.num_epochs):
        stop_training = False
        model.train()
        epoch_loss = 0
        epoch_focal = 0
        epoch_dice = 0
        epoch_delta_l2 = 0
        epoch_density = 0
        epoch_count = 0
        epoch_total_count = 0
        epoch_empty_sample = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}')
        for batch in pbar:
            tissue_map = batch['tissue_map'].to(device)    # (B, H, W) int64
            cell_map = batch['cell_map'].to(device)        # (B, H, W) int64
            mask = batch['mask'].to(device)                # (B, 1, H, W) float
            cancer_id = batch['cancer_id'].to(device)      # (B,) int64
            target = batch['target'].to(device)            # (B, H, W) int64

            model_tissue_map = apply_fine_to_parent_dropout(
                tissue_map,
                probability=args.fine_to_parent_dropout,
            )
            logits, density_prediction = model(
                model_tissue_map,
                cell_map,
                mask,
                cancer_id,
                return_density=True,
            )
            task_loss, loss_dict = criterion(logits, target, mask)
            density_loss = logits.sum() * 0.0
            count_loss = logits.sum() * 0.0
            total_count_loss = logits.sum() * 0.0
            empty_sample_loss = logits.sum() * 0.0
            if args.center_density_head:
                density_target = batch['density_target'].to(device)
                _, density_loss_dict = density_criterion(
                    density_prediction,
                    density_target,
                    model_tissue_map,
                    mask,
                )
                density_loss = density_loss_dict['density']
                count_loss = density_loss_dict['count']
                total_count_loss = density_loss_dict['total_count']
                empty_sample_loss = density_loss_dict['empty_sample']
            delta_l2 = model.input_encoder.tissue_emb.fine_delta_l2()
            loss = (
                task_loss
                + args.density_loss_weight * density_loss
                + args.count_loss_weight * count_loss
                + args.total_count_loss_weight * total_count_loss
                + args.empty_sample_fp_loss_weight * empty_sample_loss
                + args.tissue_delta_l2_weight * delta_l2
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_focal += loss_dict['focal'].item()
            epoch_dice += loss_dict['dice'].item()
            epoch_delta_l2 += delta_l2.item()
            epoch_density += density_loss.item()
            epoch_count += count_loss.item()
            epoch_total_count += total_count_loss.item()
            epoch_empty_sample += empty_sample_loss.item()
            global_step += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                density=f"{density_loss.item():.4f}",
                count=f"{count_loss.item():.4f}",
                total_count=f"{total_count_loss.item():.4f}",
                empty_fp=f"{empty_sample_loss.item():.4f}",
            )

            if global_step % 50 == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/focal', loss_dict['focal'].item(), global_step)
                writer.add_scalar('train/dice', loss_dict['dice'].item(), global_step)
                writer.add_scalar('train/tissue_delta_l2', delta_l2.item(), global_step)
                if args.center_density_head:
                    writer.add_scalar('train/density', density_loss.item(), global_step)
                    writer.add_scalar('train/count', count_loss.item(), global_step)
                    writer.add_scalar(
                        'train/total_count', total_count_loss.item(), global_step
                    )
                    writer.add_scalar(
                        'train/empty_sample_fp', empty_sample_loss.item(), global_step
                    )

        n_batches = len(train_loader)
        avg_loss = epoch_loss / n_batches
        scheduler.step()

        logger.info(f'Epoch {epoch+1}: loss={avg_loss:.4f}, '
                    f'focal={epoch_focal/n_batches:.4f}, dice={epoch_dice/n_batches:.4f}, '
                    f'density={epoch_density/n_batches:.4f}, count={epoch_count/n_batches:.4f}, '
                    f'total_count={epoch_total_count/n_batches:.4f}, '
                    f'empty_sample_fp={epoch_empty_sample/n_batches:.4f}, '
                    f'tissue_delta_l2={epoch_delta_l2/n_batches:.6f}, '
                    f'lr={scheduler.get_last_lr()[0]:.6f}')
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)

        # 验证
        if (epoch + 1) % args.val_every == 0:
            val_loss, val_metrics = validate(
                model,
                criterion,
                val_loader,
                device,
                density_criterion=density_criterion,
                density_loss_weight=args.density_loss_weight,
                count_loss_weight=args.count_loss_weight,
                total_count_loss_weight=args.total_count_loss_weight,
                empty_sample_fp_loss_weight=args.empty_sample_fp_loss_weight,
            )
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/mask_acc', val_metrics['mask_acc'], epoch)
            writer.add_scalar('val/mask_nuclei_recall', val_metrics['nuclei_recall'], epoch)
            writer.add_scalar('val/mask_nuclei_precision', val_metrics['nuclei_precision'], epoch)
            writer.add_scalar('val/mask_nuclei_type_acc', val_metrics['nuclei_type_acc'], epoch)

            logger.info(f'  val: loss={val_loss:.4f}, mask_acc={val_metrics["mask_acc"]:.4f}, '
                        f'nuclei_recall={val_metrics["nuclei_recall"]:.4f}, '
                        f'nuclei_precision={val_metrics["nuclei_precision"]:.4f}, '
                        f'nuclei_type_acc={val_metrics["nuclei_type_acc"]:.4f}')
            if args.center_density_head:
                writer.add_scalar('val/count_rmae_macro', val_metrics['count_rmae_macro'], epoch)
                writer.add_scalar('val/count_signed_relative_macro',
                                  val_metrics['count_signed_relative_macro'], epoch)
                logger.info(
                    f'  val density: count_rMAE_macro={val_metrics["count_rmae_macro"]:.4f}, '
                    f'signed_relative_macro={val_metrics["count_signed_relative_macro"]:.4f}, '
                    f'generated/target={val_metrics["count_ratio"]:.4f}'
                )

            coarse_val_loss = None
            coarse_val_metrics = None
            if args.validate_coarse_fallback:
                coarse_val_loss, coarse_val_metrics = validate(
                    model,
                    criterion,
                    val_loader,
                    device,
                    force_parent_tissue=True,
                    density_criterion=density_criterion,
                    density_loss_weight=args.density_loss_weight,
                    count_loss_weight=args.count_loss_weight,
                    total_count_loss_weight=args.total_count_loss_weight,
                    empty_sample_fp_loss_weight=args.empty_sample_fp_loss_weight,
                )
                writer.add_scalar('val_coarse/loss', coarse_val_loss, epoch)
                writer.add_scalar('val_coarse/mask_acc', coarse_val_metrics['mask_acc'], epoch)
                writer.add_scalar(
                    'val_coarse/mask_nuclei_recall',
                    coarse_val_metrics['nuclei_recall'],
                    epoch,
                )
                writer.add_scalar(
                    'val_coarse/mask_nuclei_precision',
                    coarse_val_metrics['nuclei_precision'],
                    epoch,
                )
                writer.add_scalar(
                    'val_coarse/mask_nuclei_type_acc',
                    coarse_val_metrics['nuclei_type_acc'],
                    epoch,
                )
                logger.info(
                    f'  val coarse fallback: loss={coarse_val_loss:.4f}, '
                    f'mask_acc={coarse_val_metrics["mask_acc"]:.4f}, '
                    f'nuclei_recall={coarse_val_metrics["nuclei_recall"]:.4f}, '
                    f'nuclei_precision={coarse_val_metrics["nuclei_precision"]:.4f}, '
                    f'nuclei_type_acc={coarse_val_metrics["nuclei_type_acc"]:.4f}'
                )

            best_val_loss = min(best_val_loss, val_loss)
            selection_score = (
                val_loss
                if args.checkpoint_metric == 'native_loss'
                else val_metrics[args.checkpoint_metric]
            )
            if selection_score < best_selection_score:
                best_selection_score = selection_score
                epochs_without_improvement = 0
                payload = {
                    'epoch': epoch, 'global_step': global_step,
                    'model': model.state_dict(),
                    'val_loss': val_loss, 'val_metrics': val_metrics,
                    'selection_score': selection_score,
                    'best_selection_score': best_selection_score,
                    'best_val_loss': best_val_loss,
                    **_checkpoint_metadata(args),
                }
                if coarse_val_metrics is not None:
                    payload['coarse_fallback_val_loss'] = coarse_val_loss
                    payload['coarse_fallback_val_metrics'] = coarse_val_metrics
                torch.save(payload, os.path.join(ckpt_dir, 'best.pt'))
                logger.info(
                    f'  Saved best model ({args.checkpoint_metric}={selection_score:.4f}, '
                    f'val_loss={val_loss:.4f})'
                )
            elif args.early_stopping_patience > 0:
                epochs_without_improvement += 1
                logger.info(
                    f'  No {args.checkpoint_metric} improvement for '
                    f'{epochs_without_improvement}/{args.early_stopping_patience} checks.'
                )
                stop_training = epochs_without_improvement >= args.early_stopping_patience

        # 可视化
        if (epoch + 1) % args.vis_every == 0:
            visualize(model, val_loader, device, vis_dir, epoch)

        # 定期保存
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch, 'global_step': global_step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_selection_score': best_selection_score,
                **_checkpoint_metadata(args),
            }, os.path.join(ckpt_dir, f'epoch_{epoch+1}.pt'))
            logger.info(f'  Saved checkpoint epoch_{epoch+1}.pt')

        if stop_training:
            logger.info(f'Early stopping after epoch {epoch+1}.')
            break

    writer.close()
    logger.info('Training done!')


# ============================================================
#  验证
# ============================================================

@torch.no_grad()
def validate(
    model,
    criterion,
    val_loader,
    device,
    force_parent_tissue=False,
    density_criterion=None,
    density_loss_weight=0.0,
    count_loss_weight=0.0,
    total_count_loss_weight=0.0,
    empty_sample_fp_loss_weight=0.0,
):
    was_training = model.training
    model.eval()
    if density_criterion is None and getattr(model, 'with_density_head', False):
        density_criterion = CenterDensityLoss().to(device)
    total_loss = 0
    total_mask_correct = 0
    total_mask_pixels = 0
    total_nuclei_tp = 0
    total_nuclei_gt = 0
    total_nuclei_pred = 0
    total_nuclei_type_correct = 0
    total_semantic_loss = 0.0
    total_density_loss = 0.0
    total_count_loss = 0.0
    total_patch_count_loss = 0.0
    total_empty_sample_loss = 0.0
    density_stats = defaultdict(
        lambda: {
            'samples': 0,
            'relative_abs_sum': 0.0,
            'relative_signed_sum': 0.0,
            'predicted_count': 0.0,
            'target_count': 0.0,
        }
    )
    density_group_stats = defaultdict(
        lambda: {
            'samples': 0,
            'absolute_error_sum': 0.0,
            'signed_error_sum': 0.0,
            'predicted_count': 0.0,
            'target_count': 0.0,
        }
    )
    count_bin_stats = defaultdict(
        lambda: {
            'samples': 0,
            'absolute_error_sum': 0.0,
            'relative_absolute_error_sum': 0.0,
            'signed_error_sum': 0.0,
            'predicted_count': 0.0,
            'target_count': 0.0,
            'rounded_false_positive_samples': 0,
        }
    )
    n = 0

    for batch in val_loader:
        tissue_map = batch['tissue_map'].to(device)
        cell_map = batch['cell_map'].to(device)
        mask = batch['mask'].to(device)
        cancer_id = batch['cancer_id'].to(device)
        target = batch['target'].to(device)

        if force_parent_tissue:
            tissue_map = collapse_fine_to_parent(tissue_map)

        logits, density_prediction = model(
            tissue_map,
            cell_map,
            mask,
            cancer_id,
            return_density=True,
        )
        semantic_loss, _ = criterion(logits, target, mask)
        density_loss = semantic_loss * 0.0
        count_loss = semantic_loss * 0.0
        patch_count_loss = semantic_loss * 0.0
        empty_sample_loss = semantic_loss * 0.0
        if density_prediction is not None and 'density_target' in batch:
            density_target = batch['density_target'].to(device)
            _, density_loss_dict = density_criterion(
                density_prediction,
                density_target,
                tissue_map,
                mask,
            )
            density_loss = density_loss_dict['density']
            count_loss = density_loss_dict['count']
            patch_count_loss = density_loss_dict['total_count']
            empty_sample_loss = density_loss_dict['empty_sample']

            mask_float = mask
            predicted_counts = (density_prediction * mask_float).sum(dim=(1, 2, 3))
            target_counts = (density_target * mask_float).sum(dim=(1, 2, 3))
            dataset_names = batch.get('dataset_name', ['unknown'] * tissue_map.shape[0])
            for sample_index, dataset_name in enumerate(dataset_names):
                predicted_count = float(predicted_counts[sample_index].item())
                target_count = float(target_counts[sample_index].item())
                denominator = max(target_count, 1.0)
                stats = density_stats[str(dataset_name)]
                stats['samples'] += 1
                stats['relative_abs_sum'] += abs(predicted_count - target_count) / denominator
                stats['relative_signed_sum'] += (predicted_count - target_count) / denominator
                stats['predicted_count'] += predicted_count
                stats['target_count'] += target_count

                if target_count < 0.5:
                    count_bin = '0'
                elif target_count <= 5.5:
                    count_bin = '1-5'
                elif target_count <= 20.5:
                    count_bin = '6-20'
                else:
                    count_bin = '>20'
                bin_stats = count_bin_stats[(str(dataset_name), count_bin)]
                absolute_error = abs(predicted_count - target_count)
                bin_stats['samples'] += 1
                bin_stats['absolute_error_sum'] += absolute_error
                bin_stats['relative_absolute_error_sum'] += (
                    absolute_error / max(target_count, 1.0)
                )
                bin_stats['signed_error_sum'] += predicted_count - target_count
                bin_stats['predicted_count'] += predicted_count
                bin_stats['target_count'] += target_count
                bin_stats['rounded_false_positive_samples'] += int(
                    count_bin == '0' and round(predicted_count) > 0
                )

                changed_sample = mask[sample_index, 0] > 0.5
                for tissue_id in torch.unique(tissue_map[sample_index][changed_sample]):
                    tissue_region = changed_sample & (
                        tissue_map[sample_index] == tissue_id
                    )
                    predicted_by_class = density_prediction[
                        sample_index, :, tissue_region
                    ].sum(dim=1)
                    target_by_class = density_target[
                        sample_index, :, tissue_region
                    ].sum(dim=1)
                    for class_index in range(density_prediction.shape[1]):
                        predicted_value = float(predicted_by_class[class_index].item())
                        target_value = float(target_by_class[class_index].item())
                        group = density_group_stats[
                            (str(dataset_name), int(tissue_id.item()), class_index + 1)
                        ]
                        group['samples'] += 1
                        group['absolute_error_sum'] += abs(predicted_value - target_value)
                        group['signed_error_sum'] += predicted_value - target_value
                        group['predicted_count'] += predicted_value
                        group['target_count'] += target_value

        loss = (
            semantic_loss
            + density_loss_weight * density_loss
            + count_loss_weight * count_loss
            + total_count_loss_weight * patch_count_loss
            + empty_sample_fp_loss_weight * empty_sample_loss
        )
        total_loss += loss.item() * tissue_map.shape[0]
        total_semantic_loss += semantic_loss.item() * tissue_map.shape[0]
        total_density_loss += density_loss.item() * tissue_map.shape[0]
        total_count_loss += count_loss.item() * tissue_map.shape[0]
        total_patch_count_loss += patch_count_loss.item() * tissue_map.shape[0]
        total_empty_sample_loss += empty_sample_loss.item() * tissue_map.shape[0]
        n += tissue_map.shape[0]

        pred = logits.argmax(dim=1)
        mask_bool = mask[:, 0] > 0.5

        total_mask_correct += (pred[mask_bool] == target[mask_bool]).sum().item()
        total_mask_pixels += mask_bool.sum().item()

        gt_has_nuc = (target > 0) & mask_bool
        pred_has_nuc = (pred > 0) & mask_bool
        occupancy_tp = pred_has_nuc & gt_has_nuc
        total_nuclei_tp += occupancy_tp.sum().item()
        total_nuclei_gt += gt_has_nuc.sum().item()
        total_nuclei_pred += pred_has_nuc.sum().item()
        total_nuclei_type_correct += ((pred == target) & gt_has_nuc).sum().item()

    model.train(was_training)
    mask_acc = total_mask_correct / max(total_mask_pixels, 1)
    nuclei_recall = total_nuclei_tp / max(total_nuclei_gt, 1)
    nuclei_precision = total_nuclei_tp / max(total_nuclei_pred, 1)
    nuclei_type_acc = total_nuclei_type_correct / max(total_nuclei_gt, 1)
    metrics = {
        'mask_acc': mask_acc,
        'nuclei_recall': nuclei_recall,
        'nuclei_precision': nuclei_precision,
        'nuclei_type_acc': nuclei_type_acc,
        'samples': n,
        'mask_correct': total_mask_correct,
        'mask_pixels': total_mask_pixels,
        'nuclei_tp': total_nuclei_tp,
        'nuclei_gt': total_nuclei_gt,
        'nuclei_pred': total_nuclei_pred,
        'nuclei_type_correct': total_nuclei_type_correct,
        'semantic_loss': total_semantic_loss / max(n, 1),
        'density_loss': total_density_loss / max(n, 1),
        'count_loss': total_count_loss / max(n, 1),
        'total_count_loss': total_patch_count_loss / max(n, 1),
        'empty_sample_fp_loss': total_empty_sample_loss / max(n, 1),
    }
    if density_stats:
        per_dataset = {}
        for dataset_name, stats in sorted(density_stats.items()):
            sample_count = max(stats['samples'], 1)
            per_dataset[dataset_name] = {
                'samples': stats['samples'],
                'count_rmae': stats['relative_abs_sum'] / sample_count,
                'count_signed_relative': stats['relative_signed_sum'] / sample_count,
                'count_ratio': (
                    stats['predicted_count'] / max(stats['target_count'], 1e-8)
                ),
                'predicted_count': stats['predicted_count'],
                'target_count': stats['target_count'],
            }
        metrics['count_rmae_macro'] = float(np.mean([
            row['count_rmae'] for row in per_dataset.values()
        ]))
        metrics['count_signed_relative_macro'] = float(np.mean([
            row['count_signed_relative'] for row in per_dataset.values()
        ]))
        predicted_total = sum(row['predicted_count'] for row in per_dataset.values())
        target_total = sum(row['target_count'] for row in per_dataset.values())
        metrics['count_ratio'] = predicted_total / max(target_total, 1e-8)
        metrics['count_by_dataset'] = per_dataset
        group_rows = []
        for (dataset_name, tissue_id, class_id), stats in sorted(
            density_group_stats.items()
        ):
            sample_count = max(stats['samples'], 1)
            group_rows.append({
                'dataset': dataset_name,
                'tissue_id': tissue_id,
                'cell_class_id': class_id,
                'samples': stats['samples'],
                'count_mae': stats['absolute_error_sum'] / sample_count,
                'count_signed_error': stats['signed_error_sum'] / sample_count,
                'predicted_count': stats['predicted_count'],
                'target_count': stats['target_count'],
            })
        metrics['count_by_tissue_class'] = group_rows
        bin_rows = []
        for dataset_name in sorted({key[0] for key in count_bin_stats}):
            for count_bin in ('0', '1-5', '6-20', '>20'):
                stats = count_bin_stats.get((dataset_name, count_bin))
                if not stats or stats['samples'] == 0:
                    continue
                sample_count = stats['samples']
                bin_rows.append({
                    'dataset': dataset_name,
                    'count_bin': count_bin,
                    'samples': sample_count,
                    'count_mae': stats['absolute_error_sum'] / sample_count,
                    'count_rmae': (
                        stats['relative_absolute_error_sum'] / sample_count
                    ),
                    'count_signed_error': stats['signed_error_sum'] / sample_count,
                    'predicted_count': stats['predicted_count'],
                    'target_count': stats['target_count'],
                    'rounded_false_positive_rate': (
                        stats['rounded_false_positive_samples'] / sample_count
                        if count_bin == '0'
                        else None
                    ),
                })
        metrics['count_by_bin'] = bin_rows
        metrics['count_bin_rmae_macro'] = float(np.mean([
            row['count_rmae'] for row in bin_rows
        ]))

    return total_loss / max(n, 1), metrics


# ============================================================
#  可视化
# ============================================================

@torch.no_grad()
def visualize(model, val_loader, device, vis_dir, epoch):
    model.eval()
    batch = next(iter(val_loader))

    tissue_map = batch['tissue_map'][:4].to(device)   # (B, H, W) int64
    cell_map = batch['cell_map'][:4].to(device)       # (B, H, W) int64
    mask = batch['mask'][:4].to(device)               # (B, 1, H, W)
    cancer_id = batch['cancer_id'][:4].to(device)     # (B,)
    target = batch['target'][:4].to(device)           # (B, H, W)

    logits = model(tissue_map, cell_map, mask, cancer_id)
    pred = logits.argmax(dim=1).cpu().numpy()

    gt_np = target.cpu().numpy()
    tissue_np = tissue_map.cpu().numpy()     # already int IDs, no argmax needed
    input_nuc_np = cell_map.cpu().numpy()    # already int indices, no argmax needed
    mask_np = mask[:, 0].cpu().numpy()

    rows = []
    for i in range(min(4, pred.shape[0])):
        vis_input = overlay(tissue_np[i], input_nuc_np[i])
        vis_gt = overlay(tissue_np[i], gt_np[i])
        vis_pred = overlay(tissue_np[i], pred[i])

        m = (mask_np[i] * 255).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for img in [vis_input, vis_gt, vis_pred]:
            cv2.drawContours(img, contours, -1, (255, 255, 255), 2)

        row = np.concatenate([vis_input, vis_gt, vis_pred], axis=1)
        rows.append(row)

    vis = np.concatenate(rows, axis=0)

    h_title = 25
    w = vis.shape[1]
    titled = np.zeros((h_title + vis.shape[0], w, 3), dtype=np.uint8)
    titled[:h_title] = 40
    titled[h_title:] = vis

    col_w = vis.shape[1] // 3
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(titled, 'Input (erased)', (5, 18), font, 0.5, (255,255,255), 1)
    cv2.putText(titled, 'GT', (col_w+5, 18), font, 0.5, (255,255,255), 1)
    cv2.putText(titled, 'Predicted', (col_w*2+5, 18), font, 0.5, (255,255,255), 1)

    cv2.imwrite(os.path.join(vis_dir, f'epoch_{epoch+1:03d}.png'),
                cv2.cvtColor(titled, cv2.COLOR_RGB2BGR))
    model.train()


# ============================================================
#  推理 (ProbNet + 实例库)
# ============================================================

@torch.no_grad()
def inference_with_library(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- 加载模型 (Phase 4.1: 无 in_ch 参数) ----
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state_dict = ckpt.get('model', ckpt)
    with_density_head = bool(
        ckpt.get('center_density_head')
        or any(key.startswith('density_head.') for key in state_dict)
    )
    model = ProbUNet(
        out_ch=NUM_NUCLEI,
        base_ch=int(ckpt.get('base_ch', args.base_ch)),
        with_density_head=with_density_head,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Loaded model from {args.ckpt}")

    from inpaint_cells.nuclei_library.library import NucleiLibrary, poisson_disk_sampling
    library = NucleiLibrary(args.library)

    # 推理数据集 — 兼容新旧格式
    cancer_idx = args.cancer_type_index
    has_layered = (
        os.path.isdir(os.path.join(args.data_dir, 'gt_tissue'))
        or len(glob.glob(os.path.join(args.data_dir, '*', 'tissue_mask.png'))) > 0
    )
    if has_layered:
        val_dir = os.path.join(args.data_dir, 'val') if os.path.isdir(os.path.join(args.data_dir, 'val')) else args.data_dir
        val_dataset = NucleiProbDatasetLayered(
            data_dir=val_dir, cancer_type_index=cancer_idx,
            out_size=args.img_size, augment=False)
    else:
        val_dataset = NucleiProbDatasetLegacy(
            gt_dir=os.path.join(args.data_dir, 'ground_truth'),
            train_dir=os.path.join(args.data_dir, 'val'),
            cancer_type_index=cancer_idx, out_size=args.img_size, augment=False)

    output_dir = os.path.join(args.output_dir, 'inference_results')
    os.makedirs(output_dir, exist_ok=True)

    for idx in range(min(args.n_samples, len(val_dataset))):
        sample = val_dataset[idx]
        tissue_map = sample['tissue_map'].unsqueeze(0).to(device)    # (1, H, W) int64
        cell_map = sample['cell_map'].unsqueeze(0).to(device)        # (1, H, W) int64
        mask = sample['mask'].unsqueeze(0).to(device)                # (1, 1, H, W)
        cancer_id = sample['cancer_id'].unsqueeze(0).to(device)      # (1,)
        target = sample['target'].numpy()                            # (H, W)

        logits = model(tissue_map, cell_map, mask, cancer_id)
        prob = F.softmax(logits, dim=1)[0].cpu().numpy()  # (6, H, W)

        tissue_np = tissue_map[0].cpu().numpy()           # (H, W) int, 0-15
        mask_np = mask[0, 0].cpu().numpy() > 0.5          # (H, W) bool
        input_nuc_np = cell_map[0].cpu().numpy()          # (H, W) int, 0-5

        # 输出: 在 edit 区域外保留原有核, 区域内由 ProbNet + Library 填充
        output_nuclei = input_nuc_np.copy()

        for tissue_id in np.unique(tissue_np[mask_np]):
            tissue_id = int(tissue_id)
            tissue_region = mask_np & (tissue_np == tissue_id)
            if tissue_region.sum() < 50:
                continue

            nuc_prob = 1.0 - prob[0]  # P(any nucleus)
            avg_nuc_prob = nuc_prob[tissue_region].mean()
            region_area = tissue_region.sum()
            num_nuclei = int(avg_nuc_prob * region_area / 80)
            num_nuclei = max(0, int(num_nuclei * random.uniform(0.8, 1.2)))
            if num_nuclei == 0:
                continue

            stats = library.stats.get(str(tissue_id), {})
            mean_areas = [info['mean_area'] for info in stats.get('nuclei_types', {}).values()
                          if info.get('mean_area', 0) > 0]
            avg_area = np.mean(mean_areas) if mean_areas else 100
            min_distance = max(np.sqrt(avg_area / np.pi) * 3, 10)

            centers = poisson_disk_sampling(tissue_region, min_distance)
            if len(centers) > num_nuclei:
                random.shuffle(centers)
                centers = centers[:num_nuclei]

            for cy, cx in centers:
                type_probs = prob[1:, cy, cx]
                if type_probs.sum() < 0.05:
                    continue
                type_probs = type_probs / type_probs.sum()
                nuc_type_idx = np.random.choice(5, p=type_probs)
                nuc_type_raw = NUCLEI_CLASSES[nuc_type_idx]

                instance = library.sample_instance(tissue_id, nuc_type_raw)
                if instance is None:
                    continue

                # 放置 (output_nuclei 使用 index 0-5)
                _place_nucleus_simple(output_nuclei, cy, cx, instance)

        # 可视化: tissue + nuclei overlay
        vis_input = overlay(tissue_np, input_nuc_np)
        vis_gt = overlay(tissue_np, target)
        vis_pred = overlay(tissue_np, output_nuclei)

        m = (mask_np.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for img in [vis_input, vis_gt, vis_pred]:
            cv2.drawContours(img, contours, -1, (255, 255, 255), 2)

        row = np.concatenate([vis_input, vis_gt, vis_pred], axis=1)
        cv2.imwrite(os.path.join(output_dir, f'result_{idx:03d}.png'),
                    cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
        logger.info(f'[{idx+1}] saved')

    logger.info(f'Results saved to {output_dir}')


def _place_nucleus_simple(nuclei_map, cy, cx, instance, augment=True):
    """推理时使用的简化放置函数 (nuclei_map 值域 0-5, internal index)"""
    nuc_mask = instance['mask'].copy()
    nuc_type_raw = instance['type']
    nuc_type_idx = NUCLEI_RAW_TO_INDEX.get(nuc_type_raw, 0)  # 101→1, 102→2, ..., 105→5

    if augment:
        k = random.randint(0, 3)
        nuc_mask = np.rot90(nuc_mask, k)
        if random.random() > 0.5:
            nuc_mask = np.fliplr(nuc_mask)
        if random.random() > 0.5:
            nuc_mask = np.flipud(nuc_mask)
        scale = random.uniform(0.8, 1.2)
        if abs(scale - 1.0) > 0.05:
            new_h = max(1, int(nuc_mask.shape[0] * scale))
            new_w = max(1, int(nuc_mask.shape[1] * scale))
            nuc_mask = cv2.resize(nuc_mask.astype(np.uint8), (new_w, new_h),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)

    h, w = nuc_mask.shape
    H, W = nuclei_map.shape

    y1, x1 = cy - h//2, cx - w//2
    y2, x2 = y1 + h, x1 + w

    sy1, sx1 = max(0, -y1), max(0, -x1)
    sy2, sx2 = h - max(0, y2-H), w - max(0, x2-W)
    dy1, dx1 = max(0, y1), max(0, x1)
    dy2, dx2 = min(H, y2), min(W, x2)

    if dy2 <= dy1 or dx2 <= dx1:
        return False

    local = nuc_mask[sy1:sy2, sx1:sx2]
    target = nuclei_map[dy1:dy2, dx1:dx2]

    overlap = (target > 0) & local
    if overlap.sum() > local.sum() * 0.2:
        return False

    target[local] = nuc_type_idx
    return True


# ============================================================
#  主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='ProbNet 训练/推理 (Phase 4.1)')
    parser.add_argument('--mode', choices=['train', 'inference'], default='train')
    parser.add_argument('--data-dir', type=str,
                        default='/home/lyw/wqx-DL/flow-edit/FlowEdit-main/inpaint_cells/lama_dataset',
                        help='Single dataset dir (legacy compat)')
    parser.add_argument('--datasets', type=str, nargs='*', default=None,
                        help='Multi-dataset specs: NAME:PATH [NAME:PATH ...]')
    parser.add_argument('--cancer-type-index', type=int, default=0,
                        help='Cancer type index (0-5) for single dataset mode')
    parser.add_argument('--output-dir', type=str,
                        default='/data/huggingface/pathology_edit/prob_net')
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--base-ch', type=int, default=64)
    parser.add_argument('--crop-mode', choices=['mask', 'random'], default='mask',
                        help='Crop strategy for images larger than --img-size')
    parser.add_argument('--allow-flat-single-dataset', action='store_true',
                        help='Allow single-dataset layered train/val to reuse a flat data_dir')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--mask-weight', type=float, default=5.0)
    parser.add_argument('--center-density-head', action='store_true',
                        help='Add five non-negative class-wise nucleus-center density channels')
    parser.add_argument('--density-sigma', type=float, default=2.0,
                        help='Gaussian sigma in pixels for unit-mass center-density targets')
    parser.add_argument('--density-loss-weight', type=float, default=0.0)
    parser.add_argument('--count-loss-weight', type=float, default=0.0)
    parser.add_argument('--total-count-loss-weight', type=float, default=0.0)
    parser.add_argument(
        '--density-init-bias',
        type=float,
        nargs='+',
        default=[-9.0],
        help='One scalar or five per-class pre-softplus density-head biases',
    )
    parser.add_argument(
        '--density-empty-group-weight',
        type=float,
        default=1.0,
        help='Relative weight of empty tissue/class groups after separate averaging',
    )
    parser.add_argument(
        '--density-high-count-threshold', type=float, default=20.0,
        help='True patch count above which the total-count loss is upweighted',
    )
    parser.add_argument(
        '--density-high-count-weight', type=float, default=1.0,
        help='Multiplier for patch-count loss when true count exceeds the threshold',
    )
    parser.add_argument(
        '--empty-sample-fp-loss-weight', type=float, default=0.0,
        help='Independent weight for predicted nuclei on true-zero edit regions',
    )
    parser.add_argument(
        '--density-head-only', action='store_true',
        help='Freeze all ProbNet parameters except density_head.*',
    )
    parser.add_argument('--complete-instance-erasure', action='store_true', default=True,
                        help='Expand edit masks to every intersecting same-class component')
    parser.add_argument('--no-complete-instance-erasure',
                        dest='complete_instance_erasure', action='store_false')
    parser.add_argument(
        '--checkpoint-metric',
        choices=['auto', 'native_loss', 'count_rmae_macro', 'count_bin_rmae_macro'],
        default='auto',
        help='Metric minimized for best checkpoint and early stopping',
    )
    parser.add_argument('--val-every', type=int, default=2)
    parser.add_argument('--vis-every', type=int, default=5)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--resume-from-checkpoint', type=str, default=None,
                        help="Resume: 'latest' or path to .pt")
    parser.add_argument('--init-from-checkpoint', type=str, default=None,
                        help='Load model weights only and start a fresh optimizer/schedule')
    parser.add_argument('--fine-to-parent-dropout', type=float, default=0.0,
                        help='Per-sample probability of replacing fine tissue IDs with coarse parents')
    parser.add_argument('--tissue-delta-l2-weight', type=float, default=0.0,
                        help='L2 regularization weight for dataset-specific tissue residuals')
    parser.add_argument('--validate-coarse-fallback', action='store_true',
                        help='Also validate with every fine tissue label collapsed to its parent')
    parser.add_argument('--early-stopping-patience', type=int, default=0,
                        help='Stop after this many validation checks without improvement; 0 disables')
    # 推理
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--library', type=str, default=None)
    parser.add_argument('--n-samples', type=int, default=10)
    args = parser.parse_args()

    if not 0.0 <= args.fine_to_parent_dropout <= 1.0:
        parser.error('--fine-to-parent-dropout must be in [0, 1]')
    if args.tissue_delta_l2_weight < 0:
        parser.error('--tissue-delta-l2-weight must be non-negative')
    if args.early_stopping_patience < 0:
        parser.error('--early-stopping-patience must be non-negative')
    if args.num_workers < 0:
        parser.error('--num-workers must be non-negative')
    if args.base_ch <= 0:
        parser.error('--base-ch must be positive')
    if args.density_sigma <= 0:
        parser.error('--density-sigma must be positive')
    if (
        args.density_loss_weight < 0
        or args.count_loss_weight < 0
        or args.total_count_loss_weight < 0
    ):
        parser.error('density/count loss weights must be non-negative')
    if len(args.density_init_bias) not in (1, 5):
        parser.error('--density-init-bias requires one scalar or five class values')
    if not all(np.isfinite(value) for value in args.density_init_bias):
        parser.error('--density-init-bias values must be finite')
    if args.density_empty_group_weight < 0:
        parser.error('--density-empty-group-weight must be non-negative')
    if args.density_high_count_threshold < 0:
        parser.error('--density-high-count-threshold must be non-negative')
    if args.density_high_count_weight <= 0:
        parser.error('--density-high-count-weight must be positive')
    if args.empty_sample_fp_loss_weight < 0:
        parser.error('--empty-sample-fp-loss-weight must be non-negative')
    if not args.center_density_head and (
        args.density_loss_weight > 0
        or args.count_loss_weight > 0
        or args.total_count_loss_weight > 0
    ):
        parser.error('density/count losses require --center-density-head')
    if not args.center_density_head and args.checkpoint_metric in {
        'count_rmae_macro', 'count_bin_rmae_macro'
    }:
        parser.error('density count checkpoint metrics require --center-density-head')
    if args.density_head_only and not args.center_density_head:
        parser.error('--density-head-only requires --center-density-head')
    if args.init_from_checkpoint and args.resume_from_checkpoint:
        parser.error('--init-from-checkpoint and --resume-from-checkpoint are mutually exclusive')

    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        if not args.ckpt or not args.library:
            print("Inference requires --ckpt and --library")
            return
        inference_with_library(args)


if __name__ == '__main__':
    main()

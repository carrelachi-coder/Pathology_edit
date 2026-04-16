#!/usr/bin/env python3
"""
ProbNet 训练入口

任务：给定组织层 + 已有细胞核（编辑区域内清零）+ 编辑mask
     预测编辑区域内每个像素的核类型概率

输入: tissue one-hot (22ch) + nuclei one-hot (6ch) + mask (1ch) = 29ch
输出: 核类型概率 (6ch): [背景, neoplastic, inflammatory, connective, dead, epithelial]

用法:
    # 训练
    CUDA_VISIBLE_DEVICES=5 python inpaint_cells/train.py \
        --data-dir /data/huggingface/dataset_for_mask_edit \
        --output-dir /data/huggingface/pathology_edit/prob_net \
        --batch-size 16 --num-epochs 100

    # 从 checkpoint 恢复训练
    python inpaint_cells/train.py \
        --data-dir ... --output-dir ... \
        --resume-from-checkpoint latest

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

from inpaint_cells.models.prob_unet import ProbUNet
from inpaint_cells.data.prob_dataset import NucleiProbDataset
from inpaint_cells.losses.focal_dice import FocalDiceLoss
from inpaint_cells.utils.mask_utils import (
    NUM_TISSUE, NUM_NUCLEI, NUCLEI_CLASSES,
    overlay, index_to_rgb, NUCLEI_RGB, TISSUE_RGB_MAP,
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


# ============================================================
#  训练
# ============================================================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据
    train_dataset = NucleiProbDataset(
        gt_dir=os.path.join(args.data_dir, 'ground_truth'),
        train_dir=os.path.join(args.data_dir, 'train'),
        out_size=args.img_size, augment=True,
    )
    val_dataset = NucleiProbDataset(
        gt_dir=os.path.join(args.data_dir, 'ground_truth'),
        train_dir=os.path.join(args.data_dir, 'val'),
        out_size=args.img_size, augment=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # 模型
    model = ProbUNet(in_ch=NUM_TISSUE + NUM_NUCLEI + 1, out_ch=NUM_NUCLEI, base_ch=64).to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"ProbUNet parameters: {num_params:.1f}M")

    # Loss + 优化器
    criterion = FocalDiceLoss(num_classes=NUM_NUCLEI, mask_weight=args.mask_weight).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    # 输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
    vis_dir = os.path.join(args.output_dir, 'vis')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(os.path.join(args.output_dir, 'tb_logs'))

    global_step = 0
    best_val_loss = float('inf')
    start_epoch = 0

    # Resume
    resume_path = _resolve_resume_checkpoint(args)
    if resume_path is not None:
        logger.info(f"Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        else:
            for _ in range(ckpt.get('epoch', 0) + 1):
                scheduler.step()
        start_epoch = ckpt.get('epoch', 0) + 1
        global_step = ckpt.get('global_step', start_epoch * len(train_loader))
        if 'val_loss' in ckpt:
            best_val_loss = ckpt['val_loss']
        logger.info(f"  Resuming from epoch {start_epoch}, global_step={global_step}")

    # 训练循环
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0
        epoch_focal = 0
        epoch_dice = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}')
        for batch in pbar:
            tissue = batch['tissue'].to(device)
            nuclei_input = batch['nuclei_input'].to(device)
            mask = batch['mask'].to(device)
            target = batch['target'].to(device)

            logits = model(tissue, nuclei_input, mask)
            loss, loss_dict = criterion(logits, target, mask)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_focal += loss_dict['focal'].item()
            epoch_dice += loss_dict['dice'].item()
            global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             focal=f"{loss_dict['focal'].item():.4f}",
                             dice=f"{loss_dict['dice'].item():.4f}")

            if global_step % 50 == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/focal', loss_dict['focal'].item(), global_step)
                writer.add_scalar('train/dice', loss_dict['dice'].item(), global_step)

        n_batches = len(train_loader)
        avg_loss = epoch_loss / n_batches
        scheduler.step()

        logger.info(f'Epoch {epoch+1}: loss={avg_loss:.4f}, '
                    f'focal={epoch_focal/n_batches:.4f}, dice={epoch_dice/n_batches:.4f}, '
                    f'lr={scheduler.get_last_lr()[0]:.6f}')
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)

        # 验证
        if (epoch + 1) % args.val_every == 0:
            val_loss, val_metrics = validate(model, criterion, val_loader, device)
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/mask_acc', val_metrics['mask_acc'], epoch)
            writer.add_scalar('val/mask_nuclei_recall', val_metrics['nuclei_recall'], epoch)

            logger.info(f'  val: loss={val_loss:.4f}, mask_acc={val_metrics["mask_acc"]:.4f}, '
                        f'nuclei_recall={val_metrics["nuclei_recall"]:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch, 'global_step': global_step,
                    'model': model.state_dict(),
                    'val_loss': val_loss, 'val_metrics': val_metrics,
                }, os.path.join(ckpt_dir, 'best.pt'))
                logger.info(f'  Saved best model (val_loss={val_loss:.4f})')

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
            }, os.path.join(ckpt_dir, f'epoch_{epoch+1}.pt'))
            logger.info(f'  Saved checkpoint epoch_{epoch+1}.pt')

    writer.close()
    logger.info('Training done!')


# ============================================================
#  验证
# ============================================================

@torch.no_grad()
def validate(model, criterion, val_loader, device):
    model.eval()
    total_loss = 0
    total_mask_correct = 0
    total_mask_pixels = 0
    total_nuclei_tp = 0
    total_nuclei_gt = 0
    n = 0

    for batch in val_loader:
        tissue = batch['tissue'].to(device)
        nuclei_input = batch['nuclei_input'].to(device)
        mask = batch['mask'].to(device)
        target = batch['target'].to(device)

        logits = model(tissue, nuclei_input, mask)
        loss, _ = criterion(logits, target, mask)
        total_loss += loss.item() * tissue.shape[0]
        n += tissue.shape[0]

        pred = logits.argmax(dim=1)
        mask_bool = mask[:, 0] > 0.5

        total_mask_correct += (pred[mask_bool] == target[mask_bool]).sum().item()
        total_mask_pixels += mask_bool.sum().item()

        gt_has_nuc = (target > 0) & mask_bool
        pred_has_nuc = (pred > 0) & gt_has_nuc
        total_nuclei_tp += pred_has_nuc.sum().item()
        total_nuclei_gt += gt_has_nuc.sum().item()

    model.train()
    mask_acc = total_mask_correct / max(total_mask_pixels, 1)
    nuclei_recall = total_nuclei_tp / max(total_nuclei_gt, 1)
    return total_loss / n, {'mask_acc': mask_acc, 'nuclei_recall': nuclei_recall}


# ============================================================
#  可视化
# ============================================================

@torch.no_grad()
def visualize(model, val_loader, device, vis_dir, epoch):
    model.eval()
    batch = next(iter(val_loader))

    tissue = batch['tissue'][:4].to(device)
    nuclei_input = batch['nuclei_input'][:4].to(device)
    mask = batch['mask'][:4].to(device)
    target = batch['target'][:4].to(device)

    logits = model(tissue, nuclei_input, mask)
    pred = logits.argmax(dim=1).cpu().numpy()

    gt_np = target.cpu().numpy()
    tissue_np = tissue.argmax(dim=1).cpu().numpy()
    input_nuc_np = nuclei_input.argmax(dim=1).cpu().numpy()
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

    model = ProbUNet(in_ch=NUM_TISSUE + NUM_NUCLEI + 1, out_ch=NUM_NUCLEI, base_ch=64).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    logger.info(f"Loaded model from {args.ckpt}")

    from inpaint_cells.nuclei_library.library import NucleiLibrary, poisson_disk_sampling
    library = NucleiLibrary(args.library)

    val_dataset = NucleiProbDataset(
        gt_dir=os.path.join(args.data_dir, 'ground_truth'),
        train_dir=os.path.join(args.data_dir, 'val'),
        out_size=args.img_size, augment=False,
    )

    output_dir = os.path.join(args.output_dir, 'inference_results')
    os.makedirs(output_dir, exist_ok=True)

    for idx in range(min(args.n_samples, len(val_dataset))):
        sample = val_dataset[idx]
        tissue = sample['tissue'].unsqueeze(0).to(device)
        nuclei_input = sample['nuclei_input'].unsqueeze(0).to(device)
        mask = sample['mask'].unsqueeze(0).to(device)
        target = sample['target'].numpy()

        logits = model(tissue, nuclei_input, mask)
        prob = F.softmax(logits, dim=1)[0].cpu().numpy()

        tissue_map = tissue[0].argmax(dim=0).cpu().numpy()
        mask_np = mask[0, 0].cpu().numpy() > 0.5

        output_nuclei = nuclei_input[0].argmax(dim=0).cpu().numpy()

        for tissue_id in np.unique(tissue_map[mask_np]):
            tissue_id = int(tissue_id)
            tissue_region = mask_np & (tissue_map == tissue_id)
            if tissue_region.sum() < 50:
                continue

            nuc_prob = 1.0 - prob[0]
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
                nuc_type = NUCLEI_CLASSES[nuc_type_idx]

                instance = library.sample_instance(tissue_id, nuc_type)
                if instance is None:
                    continue

                # 简易放置
                _place_nucleus_simple(output_nuclei, cy, cx, instance)

        # 可视化
        vis_input = overlay(tissue_map, nuclei_input[0].argmax(dim=0).cpu().numpy())
        vis_gt = overlay(tissue_map, target)
        vis_pred = overlay(tissue_map, output_nuclei)

        m = (mask_np * 255).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for img in [vis_input, vis_gt, vis_pred]:
            cv2.drawContours(img, contours, -1, (255, 255, 255), 2)

        row = np.concatenate([vis_input, vis_gt, vis_pred], axis=1)
        cv2.imwrite(os.path.join(output_dir, f'result_{idx:03d}.png'),
                    cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
        logger.info(f'[{idx+1}] saved')

    logger.info(f'Results saved to {output_dir}')


def _place_nucleus_simple(nuclei_map, cy, cx, instance, augment=True):
    """推理时使用的简化放置函数 (nuclei_map 值域 0-5)"""
    nuc_mask = instance['mask'].copy()
    nuc_type_raw = instance['type']
    nuc_type_idx = NUCLEI_CLASSES.index(nuc_type_raw) + 1

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
    parser = argparse.ArgumentParser(description='ProbNet 训练/推理')
    parser.add_argument('--mode', choices=['train', 'inference'], default='train')
    parser.add_argument('--data-dir', type=str,
                        default='/home/lyw/wqx-DL/flow-edit/FlowEdit-main/inpaint_cells/lama_dataset')
    parser.add_argument('--output-dir', type=str,
                        default='/data/huggingface/pathology_edit/prob_net')
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--mask-weight', type=float, default=5.0)
    parser.add_argument('--val-every', type=int, default=2)
    parser.add_argument('--vis-every', type=int, default=5)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--resume-from-checkpoint', type=str, default=None,
                        help="Resume: 'latest' or path to .pt")
    # 推理
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--library', type=str, default=None)
    parser.add_argument('--n-samples', type=int, default=10)
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        if not args.ckpt or not args.library:
            print("Inference requires --ckpt and --library")
            return
        inference_with_library(args)


if __name__ == '__main__':
    main()

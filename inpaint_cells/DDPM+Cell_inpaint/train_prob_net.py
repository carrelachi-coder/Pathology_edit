#!/usr/bin/env python3
"""
细胞核概率图预测网络

任务：给定组织层 + 已有细胞核（编辑区域内置零）+ 编辑mask
     预测编辑区域内每个像素的核类型概率

输入: 组织层 one-hot (22ch) + 细胞核 one-hot (6ch) + mask (1ch) = 29ch
输出: 核类型概率 (6ch): [背景, neoplastic, inflammatory, connective, dead, epithelial]

训练数据: lama_dataset 中的 ground_truth + train(带mask的擦除数据)

用法:
    # 建库（先跑这个）
    python build_nuclei_library.py --gt-dir ... --output-dir ...

    # 训练概率图网络
    CUDA_VISIBLE_DEVICES=5 python /home/lyw/wqx-DL/flow-edit/FlowEdit-main/inpaint_cells/DDPM+Cell_inpaint/train_prob_net.py \
        --data-dir /data/huggingface/dataset_for_mask_edit \
        --output-dir /data/huggingface/pathology_edit/prob_net \
        --batch-size 16 \
        --num-epochs 100 \
        --resume-from-checkpoint /data/huggingface/pathology_edit/prob_net/checkpoints/epoch_50.pt


    # 推理（结合实例库）
    x
"""
#!/usr/bin/env python3
"""
细胞核概率图预测网络

任务：给定组织层 + 已有细胞核（编辑区域内置零）+ 编辑mask
     预测编辑区域内每个像素的核类型概率

输入: 组织层 one-hot (22ch) + 细胞核 one-hot (6ch) + mask (1ch) = 29ch
输出: 核类型概率 (6ch): [背景, neoplastic, inflammatory, connective, dead, epithelial]

训练数据: lama_dataset 中的 ground_truth + train(带mask的擦除数据)

用法:
    # 训练概率图网络
    CUDA_VISIBLE_DEVICES=5 python train_prob_net.py \
        --data-dir /data/huggingface/dataset_for_mask_edit \
        --output-dir /data/huggingface/pathology_edit/prob_net \
        --batch-size 16 --num-epochs 100

    # 从 checkpoint 恢复训练
    python train_prob_net.py \
        --data-dir ... --output-dir ... \
        --resume-from-checkpoint latest
    # 或指定具体 checkpoint:
        --resume-from-checkpoint /path/to/checkpoints/epoch_50.pt

    # 推理（结合实例库）
    python train_prob_net.py --mode inference --ckpt best.pt --library ...
"""

import os
import sys
import json
import argparse
import glob
import random
import logging

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
#  颜色/类别定义
# ============================================================

COLOR_MAP = {
    0: [30,30,30], 1: [180,60,60], 2: [60,150,60], 3: [140,60,180],
    4: [60,60,180], 5: [180,180,80], 6: [160,40,40], 7: [40,40,40],
    8: [80,150,150], 9: [200,170,100], 10: [180,120,150], 11: [120,120,190],
    12: [100,190,190], 13: [200,140,60], 14: [140,200,100], 15: [140,140,140],
    16: [200,200,130], 17: [150,80,60], 18: [60,140,100], 19: [190,40,40],
    20: [80,60,150], 21: [170,170,170],
    101: [255,0,0], 102: [0,255,0], 103: [0,80,255], 104: [255,255,0], 105: [255,0,255],
}

NUCLEI_CLASSES = [101, 102, 103, 104, 105]
NUM_TISSUE = 22
NUM_NUCLEI = 6  # 背景(0) + 5类核

_rgb_to_val = {}
for val, rgb in COLOR_MAP.items():
    key = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
    _rgb_to_val[key] = val


def rgb_to_class_map(rgb_img):
    encoded = rgb_img[:,:,0].astype(np.int64)*65536 + rgb_img[:,:,1].astype(np.int64)*256 + rgb_img[:,:,2].astype(np.int64)
    result = np.zeros(rgb_img.shape[:2], dtype=np.int64)
    for key, val in _rgb_to_val.items():
        result[encoded == key] = val
    return result


def split_tissue_nuclei(class_map):
    tissue = class_map.copy()
    nuclei = np.zeros_like(class_map)
    
    for i, nuc_val in enumerate(NUCLEI_CLASSES):
        mask = class_map == nuc_val
        nuclei[mask] = i + 1
    
    nuc_mask = class_map >= 100
    if nuc_mask.any():
        from scipy.ndimage import distance_transform_edt
        _, nearest_idx = distance_transform_edt(nuc_mask, return_distances=True, return_indices=True)
        tissue[nuc_mask] = class_map[nearest_idx[0][nuc_mask], nearest_idx[1][nuc_mask]]
        tissue = np.clip(tissue, 0, 21)
    
    return tissue, nuclei


def to_onehot(index_map, num_classes):
    oh = np.zeros((num_classes, index_map.shape[0], index_map.shape[1]), dtype=np.float32)
    for c in range(num_classes):
        oh[c] = (index_map == c).astype(np.float32)
    return oh


# ============================================================
#  数据集
# ============================================================

class NucleiProbDataset(Dataset):
    def __init__(self, gt_dir, train_dir, out_size=256, augment=True):
        self.gt_dir = gt_dir
        self.train_dir = train_dir
        self.out_size = out_size
        self.augment = augment
        
        all_gt = sorted(glob.glob(os.path.join(gt_dir, '*.png')))
        self.samples = []
        for gt_path in all_gt:
            fname = os.path.basename(gt_path)
            train_path = os.path.join(train_dir, fname)
            mask_path = os.path.join(train_dir, fname.replace('.png', '_mask001.png'))
            if os.path.exists(train_path) and os.path.exists(mask_path):
                self.samples.append({'gt': gt_path, 'input': train_path, 'mask': mask_path})
        
        logger.info(f"NucleiProbDataset: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        
        gt_rgb = cv2.cvtColor(cv2.imread(s['gt']), cv2.COLOR_BGR2RGB)
        input_rgb = cv2.cvtColor(cv2.imread(s['input']), cv2.COLOR_BGR2RGB)
        mask_bin = cv2.imread(s['mask'], cv2.IMREAD_GRAYSCALE)
        
        h, w = gt_rgb.shape[:2]
        
        if h > self.out_size or w > self.out_size:
            y = random.randint(0, max(0, h - self.out_size))
            x = random.randint(0, max(0, w - self.out_size))
            gt_rgb = gt_rgb[y:y+self.out_size, x:x+self.out_size]
            input_rgb = input_rgb[y:y+self.out_size, x:x+self.out_size]
            mask_bin = mask_bin[y:y+self.out_size, x:x+self.out_size]
        
        gt_map = rgb_to_class_map(gt_rgb)
        gt_tissue, gt_nuclei = split_tissue_nuclei(gt_map)
        
        input_map = rgb_to_class_map(input_rgb)
        input_tissue, input_nuclei = split_tissue_nuclei(input_map)
        
        edit_mask = (mask_bin > 128).astype(np.float32)
        
        input_nuclei_masked = input_nuclei.copy()
        input_nuclei_masked[edit_mask > 0.5] = 0
        
        tissue_oh = to_onehot(gt_tissue, NUM_TISSUE)
        nuclei_input_oh = to_onehot(input_nuclei_masked, NUM_NUCLEI)
        
        target = gt_nuclei.astype(np.int64)
        
        if self.augment:
            if random.random() > 0.5:
                tissue_oh = tissue_oh[:, :, ::-1].copy()
                nuclei_input_oh = nuclei_input_oh[:, :, ::-1].copy()
                target = target[:, ::-1].copy()
                edit_mask = edit_mask[:, ::-1].copy()
            if random.random() > 0.5:
                tissue_oh = tissue_oh[:, ::-1, :].copy()
                nuclei_input_oh = nuclei_input_oh[:, ::-1, :].copy()
                target = target[::-1, :].copy()
                edit_mask = edit_mask[::-1, :].copy()
            if random.random() > 0.5:
                tissue_oh = np.rot90(tissue_oh, 1, axes=(1,2)).copy()
                nuclei_input_oh = np.rot90(nuclei_input_oh, 1, axes=(1,2)).copy()
                target = np.rot90(target, 1).copy()
                edit_mask = np.rot90(edit_mask, 1).copy()
        
        edit_mask = edit_mask[np.newaxis, :, :]
        
        return {
            'tissue': torch.from_numpy(tissue_oh),
            'nuclei_input': torch.from_numpy(nuclei_input_oh),
            'mask': torch.from_numpy(edit_mask),
            'target': torch.from_numpy(target),
        }


# ============================================================
#  轻量 UNet
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        return self.conv(x) + self.skip(x)


class ProbUNet(nn.Module):
    def __init__(self, in_ch=29, out_ch=6, base_ch=64):
        super().__init__()
        
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch*2)
        self.enc3 = ConvBlock(base_ch*2, base_ch*4)
        self.enc4 = ConvBlock(base_ch*4, base_ch*8)
        self.enc5 = ConvBlock(base_ch*8, base_ch*8)
        
        self.pool = nn.MaxPool2d(2)
        
        self.bottleneck = ConvBlock(base_ch*8, base_ch*8)
        
        self.up5 = nn.ConvTranspose2d(base_ch*8, base_ch*8, 2, stride=2)
        self.dec5 = ConvBlock(base_ch*16, base_ch*8)
        
        self.up4 = nn.ConvTranspose2d(base_ch*8, base_ch*8, 2, stride=2)
        self.dec4 = ConvBlock(base_ch*16, base_ch*4)
        
        self.up3 = nn.ConvTranspose2d(base_ch*4, base_ch*4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch*8, base_ch*2)
        
        self.up2 = nn.ConvTranspose2d(base_ch*2, base_ch*2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch*4, base_ch)
        
        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch*2, base_ch)
        
        self.out_conv = nn.Conv2d(base_ch, out_ch, 1)
    
    def forward(self, tissue, nuclei_input, mask):
        x = torch.cat([tissue, nuclei_input, mask], dim=1)
        
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        
        b = self.bottleneck(self.pool(e5))
        
        d5 = self.dec5(torch.cat([self.up5(b), e5], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.out_conv(d1)


# ============================================================
#  Loss: Focal + Dice
# ============================================================

class FocalDiceLoss(nn.Module):
    def __init__(self, num_classes=6, focal_gamma=2.0, weight_focal=1.0, weight_dice=1.0,
                 mask_weight=5.0):
        super().__init__()
        self.num_classes = num_classes
        self.focal_gamma = focal_gamma
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice
        self.mask_weight = mask_weight
        
        self.register_buffer('class_weights', torch.tensor([0.2, 2.0, 2.0, 2.0, 3.0, 2.0]))
    
    def focal_loss(self, logits, target, mask):
        ce = F.cross_entropy(logits, target, weight=self.class_weights, reduction='none')
        pt = torch.exp(-F.cross_entropy(logits, target, reduction='none'))
        focal = ((1 - pt) ** self.focal_gamma) * ce
        weight_map = 1.0 + mask[:, 0] * self.mask_weight
        focal = focal * weight_map
        return focal.mean()
    
    def dice_loss(self, logits, target, mask):
        pred = F.softmax(logits, dim=1)
        target_oh = F.one_hot(target, self.num_classes).permute(0,3,1,2).float()
        mask_expanded = mask.expand_as(pred)
        pred_masked = pred * mask_expanded
        target_masked = target_oh * mask_expanded
        dims = (0, 2, 3)
        intersection = (pred_masked * target_masked).sum(dim=dims)
        cardinality = (pred_masked + target_masked).sum(dim=dims)
        dice = (2 * intersection + 1e-6) / (cardinality + 1e-6)
        return (1 - dice).mean()
    
    def forward(self, logits, target, mask):
        focal = self.focal_loss(logits, target, mask) * self.weight_focal
        dice = self.dice_loss(logits, target, mask) * self.weight_dice
        return focal + dice, {'focal': focal, 'dice': dice}


# ============================================================
#  训练
# ============================================================

def _resolve_resume_checkpoint(args):
    """
    解析 --resume-from-checkpoint 参数。

    支持:
      "latest"          → 找 checkpoints/ 下最新的 epoch_*.pt
      "/path/to/xxx.pt" → 直接用
      None              → 不恢复

    返回: checkpoint 文件路径 (str) 或 None
    """
    resume = args.resume_from_checkpoint
    if resume is None:
        return None

    if resume == "latest":
        ckpt_dir = os.path.join(args.output_dir, "checkpoints")
        if not os.path.isdir(ckpt_dir):
            logger.warning(f"No checkpoints dir found at {ckpt_dir}, training from scratch")
            return None

        # 找 epoch_*.pt，取编号最大的
        epoch_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")))
        # 也考虑 best.pt
        best_pt = os.path.join(ckpt_dir, "best.pt")

        candidates = []
        for p in epoch_ckpts:
            try:
                ep = int(os.path.basename(p).replace("epoch_", "").replace(".pt", ""))
                candidates.append((ep, p))
            except ValueError:
                pass
        if os.path.exists(best_pt):
            # 读 best.pt 的 epoch
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

    # 直接路径
    if os.path.exists(resume):
        return resume

    logger.warning(f"Checkpoint not found: {resume}, training from scratch")
    return None


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据
    train_dataset = NucleiProbDataset(
        gt_dir=os.path.join(args.data_dir, 'ground_truth'),
        train_dir=os.path.join(args.data_dir, 'train'),
        out_size=args.img_size,
        augment=True,
    )
    val_dataset = NucleiProbDataset(
        gt_dir=os.path.join(args.data_dir, 'ground_truth'),
        train_dir=os.path.join(args.data_dir, 'val'),
        out_size=args.img_size,
        augment=False,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=2, pin_memory=True)
    
    # 模型
    model = ProbUNet(in_ch=NUM_TISSUE + NUM_NUCLEI + 1, out_ch=NUM_NUCLEI, base_ch=64).to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"ProbUNet parameters: {num_params:.1f}M")
    
    # Loss
    criterion = FocalDiceLoss(num_classes=NUM_NUCLEI, mask_weight=args.mask_weight).to(device)
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    
    # 输出
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

    # ================================================================
    # Resume from checkpoint
    # ================================================================
    resume_path = _resolve_resume_checkpoint(args)
    if resume_path is not None:
        logger.info(f"Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)

        # 模型权重 (必须有)
        model.load_state_dict(ckpt['model'])
        logger.info(f"  Loaded model weights")

        # optimizer 状态 (epoch_*.pt 有, best.pt 没有)
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
            logger.info(f"  Loaded optimizer state")
        else:
            logger.info(f"  No optimizer state in checkpoint, using fresh optimizer")

        # scheduler 状态
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
            logger.info(f"  Loaded scheduler state")
        else:
            # 手动 step scheduler 到正确位置
            resumed_epoch = ckpt.get('epoch', 0)
            for _ in range(resumed_epoch + 1):
                scheduler.step()
            logger.info(f"  No scheduler state, stepped to epoch {resumed_epoch + 1}")

        # epoch / global_step
        start_epoch = ckpt.get('epoch', 0) + 1
        global_step = ckpt.get('global_step', start_epoch * len(train_loader))

        # best_val_loss
        if 'val_loss' in ckpt:
            best_val_loss = ckpt['val_loss']
            logger.info(f"  Restored best_val_loss={best_val_loss:.4f}")

        logger.info(f"  Resuming from epoch {start_epoch}, global_step={global_step}")
    
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
                    'epoch': epoch,
                    'global_step': global_step,
                    'model': model.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                }, os.path.join(ckpt_dir, 'best.pt'))
                logger.info(f'  Saved best model (val_loss={val_loss:.4f})')
        
        # 可视化
        if (epoch + 1) % args.vis_every == 0:
            visualize(model, val_loader, device, vis_dir, epoch)
        
        # 保存 checkpoint (包含 optimizer + scheduler 用于恢复训练)
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, os.path.join(ckpt_dir, f'epoch_{epoch+1}.pt'))
            logger.info(f'  Saved checkpoint epoch_{epoch+1}.pt')
    
    writer.close()
    logger.info('Training done!')


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
    
    return total_loss / n, {
        'mask_acc': mask_acc,
        'nuclei_recall': nuclei_recall,
    }


NUCLEI_RGB = {
    0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 255, 0],
    3: [0, 80, 255], 4: [255, 255, 0], 5: [255, 0, 255],
}

TISSUE_RGB_MAP = {i: COLOR_MAP[i] for i in range(22)}


def index_to_rgb(index_map, color_map):
    h, w = index_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in color_map.items():
        rgb[index_map == idx] = color
    return rgb


def overlay(tissue_map, nuclei_map):
    tissue_rgb = index_to_rgb(tissue_map, TISSUE_RGB_MAP)
    nuc_rgb = index_to_rgb(nuclei_map, NUCLEI_RGB)
    result = tissue_rgb.copy()
    result[nuclei_map > 0] = nuc_rgb[nuclei_map > 0]
    return result


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
#  推理
# ============================================================

@torch.no_grad()
def inference_with_library(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ProbUNet(in_ch=NUM_TISSUE + NUM_NUCLEI + 1, out_ch=NUM_NUCLEI, base_ch=64).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    logger.info(f"Loaded model from {args.ckpt}")
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from generate_nuclei import NucleiLibrary, poisson_disk_sampling, place_nucleus, class_map_to_rgb
    library = NucleiLibrary(args.library)
    
    val_dataset = NucleiProbDataset(
        gt_dir=os.path.join(args.data_dir, 'ground_truth'),
        train_dir=os.path.join(args.data_dir, 'val'),
        out_size=args.img_size,
        augment=False,
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
        
        H, W = tissue_map.shape
        
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
                
                place_nucleus_simple(output_nuclei, cy, cx, instance)
        
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


def place_nucleus_simple(nuclei_map, cy, cx, instance, augment=True):
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
    parser = argparse.ArgumentParser()
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
    # Resume
    parser.add_argument('--resume-from-checkpoint', type=str, default=None,
                        help="Resume training: 'latest' or path to .pt file")
    # 推理参数
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
"""
FocalDiceLoss — ProbNet 训练 loss

组合 Focal loss + Dice loss，支持编辑区域加权。

- Focal: 解决类别不平衡 (背景像素远多于核像素)
- Dice: 提升小目标 (核) 的分割质量
- mask_weight: 编辑区域内的 loss 权重加大，聚焦于需要预测的区域
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalDiceLoss(nn.Module):
    """
    Focal + Dice 组合 loss。

    Args:
        num_classes: 核类型数 (默认 6 = bg + 5)
        focal_gamma: Focal loss 的 gamma 参数
        weight_focal: Focal loss 权重
        weight_dice: Dice loss 权重
        mask_weight: 编辑区域内像素的 loss 加权系数
    """
    def __init__(self, num_classes=6, focal_gamma=2.0, weight_focal=1.0, weight_dice=1.0,
                 mask_weight=5.0):
        super().__init__()
        self.num_classes = num_classes
        self.focal_gamma = focal_gamma
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice
        self.mask_weight = mask_weight

        # 类别权重：背景低，核类高，dead 核最高 (最稀有)
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
        """
        Args:
            logits: (B, num_classes, H, W)
            target: (B, H, W) int64
            mask: (B, 1, H, W) 编辑区域 mask

        Returns:
            (total_loss, {'focal': focal_loss, 'dice': dice_loss})
        """
        focal = self.focal_loss(logits, target, mask) * self.weight_focal
        dice = self.dice_loss(logits, target, mask) * self.weight_dice
        return focal + dice, {'focal': focal, 'dice': dice}

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


class CenterDensityLoss(nn.Module):
    """Decouple center placement from group and patch count calibration.

    ``high_count_weight`` only changes the patch-level count term for samples
    above ``high_count_threshold``.  ``empty_sample`` is returned separately so
    callers can penalize patch-level false positives without conflating them
    with empty tissue/class groups.
    """

    def __init__(
        self,
        num_tissues=16,
        empty_group_weight=1.0,
        high_count_threshold=20.0,
        high_count_weight=1.0,
    ):
        super().__init__()
        self.num_tissues = num_tissues
        if empty_group_weight < 0:
            raise ValueError("empty_group_weight must be non-negative.")
        if high_count_threshold < 0:
            raise ValueError("high_count_threshold must be non-negative.")
        if high_count_weight <= 0:
            raise ValueError("high_count_weight must be positive.")
        self.empty_group_weight = float(empty_group_weight)
        self.high_count_threshold = float(high_count_threshold)
        self.high_count_weight = float(high_count_weight)

    def _balanced_mean(self, positive_terms, empty_terms, zero):
        positive = torch.cat(positive_terms).mean() if positive_terms else None
        empty = torch.cat(empty_terms).mean() if empty_terms else None
        if positive is None and empty is None:
            return zero, zero, zero
        if positive is None:
            return empty, zero, empty
        if empty is None:
            return positive, positive, zero
        weight = self.empty_group_weight
        combined = (positive + weight * empty) / (1.0 + weight)
        return combined, positive, empty

    @staticmethod
    def _count_error(predicted_count, target_count):
        log_error = F.smooth_l1_loss(
            torch.log1p(predicted_count),
            torch.log1p(target_count),
            reduction='none',
        )
        relative_error = (predicted_count - target_count) / (target_count + 1.0)
        relative_error = F.smooth_l1_loss(
            relative_error,
            torch.zeros_like(relative_error),
            reduction='none',
        )
        return 0.5 * (log_error + relative_error)

    def forward(self, prediction, target, tissue_map, mask):
        if prediction is None:
            raise ValueError("CenterDensityLoss requires a density prediction.")
        if prediction.shape != target.shape:
            raise ValueError(
                f"Density prediction and target shapes differ: "
                f"{tuple(prediction.shape)} vs {tuple(target.shape)}."
            )

        changed = mask[:, 0] > 0.5
        positive_density_terms = []
        positive_count_terms = []
        empty_count_terms = []
        for batch_index in range(prediction.shape[0]):
            tissue_ids = torch.unique(tissue_map[batch_index][changed[batch_index]])
            for tissue_id in tissue_ids:
                region = changed[batch_index] & (tissue_map[batch_index] == tissue_id)
                if not torch.any(region):
                    continue
                pred_counts = prediction[batch_index, :, region].sum(dim=1)
                target_counts = target[batch_index, :, region].sum(dim=1)

                positive = target_counts > 0.5
                if torch.any(positive):
                    predicted_distribution = prediction[
                        batch_index, positive, :
                    ][:, region]
                    target_distribution = target[
                        batch_index, positive, :
                    ][:, region]
                    predicted_distribution = predicted_distribution / pred_counts[
                        positive, None
                    ].clamp_min(1e-8)
                    target_distribution = target_distribution / target_counts[
                        positive, None
                    ].clamp_min(1e-8)
                    positive_density_terms.append(
                        torch.abs(
                            predicted_distribution - target_distribution
                        ).sum(dim=1)
                    )
                count_values = self._count_error(pred_counts, target_counts)
                if torch.any(positive):
                    positive_count_terms.append(count_values[positive])
                if torch.any(~positive):
                    empty_count_terms.append(count_values[~positive])

        zero = prediction.sum() * 0.0
        density_loss = (
            torch.cat(positive_density_terms).mean()
            if positive_density_terms
            else zero
        )
        count_loss, count_positive, count_empty = self._balanced_mean(
            positive_count_terms,
            empty_count_terms,
            zero,
        )
        predicted_total = (prediction * mask).sum(dim=(1, 2, 3))
        target_total = (target * mask).sum(dim=(1, 2, 3))
        total_count_errors = self._count_error(
            predicted_total,
            target_total,
        )
        total_count_weights = torch.ones_like(total_count_errors)
        total_count_weights = torch.where(
            target_total > self.high_count_threshold,
            total_count_weights * self.high_count_weight,
            total_count_weights,
        )
        total_count_loss = (
            total_count_errors * total_count_weights
        ).sum() / total_count_weights.sum().clamp_min(1e-8)

        empty_samples = target_total <= 0.5
        if torch.any(empty_samples):
            # Direct count-space regression is intentionally stronger than the
            # log/relative calibration term for false nuclei on truly empty
            # edit regions.
            empty_sample_loss = F.smooth_l1_loss(
                predicted_total[empty_samples],
                torch.zeros_like(predicted_total[empty_samples]),
            )
        else:
            empty_sample_loss = zero

        high_count_samples = target_total > self.high_count_threshold
        if torch.any(high_count_samples):
            high_count_loss = total_count_errors[high_count_samples].mean()
        else:
            high_count_loss = zero
        return density_loss + count_loss + total_count_loss, {
            'density': density_loss,
            'count': count_loss,
            'total_count': total_count_loss,
            'empty_sample': empty_sample_loss,
            'high_count': high_count_loss,
            'density_positive': density_loss,
            'density_empty': zero,
            'count_positive': count_positive,
            'count_empty': count_empty,
        }

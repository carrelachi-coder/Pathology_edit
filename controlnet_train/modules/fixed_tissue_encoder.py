"""Fixed target tissue encoders without learnable label memory."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_config import NUM_FINE


class FixedOneHotTissueEncoder(nn.Module):
    """Encode tissue IDs as fixed one-hot layout features.

    This is intended for target-side ControlNet structure conditioning when the
    target mask should provide only class routing and geometry, not a learned
    per-class texture prior.
    """

    def __init__(
        self,
        num_classes: int = NUM_FINE,
        downsample_factor: int = 8,
        scale: float = 4.0,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}.")
        if downsample_factor <= 0:
            raise ValueError(f"downsample_factor must be positive, got {downsample_factor}.")
        if scale <= 0.0:
            raise ValueError(f"scale must be positive, got {scale}.")
        self.num_classes = int(num_classes)
        self.downsample_factor = int(downsample_factor)
        self.scale = float(scale)

    @property
    def out_channels(self) -> int:
        return self.num_classes

    def forward(self, tissue_ids: torch.Tensor) -> torch.Tensor:
        squeeze_batch = tissue_ids.ndim == 2
        if squeeze_batch:
            tissue_ids = tissue_ids.unsqueeze(0)
        if tissue_ids.ndim != 3:
            raise ValueError(
                f"Expected tissue_ids with shape (B, H, W) or (H, W), got {tuple(tissue_ids.shape)}."
            )

        tissue_ids = tissue_ids.long()
        if tissue_ids.numel() > 0:
            min_id = int(tissue_ids.min().item())
            max_id = int(tissue_ids.max().item())
            if min_id < 0 or max_id >= self.num_classes:
                raise ValueError(
                    f"tissue_ids out of range: got [{min_id}, {max_id}], "
                    f"expected [0, {self.num_classes - 1}]."
                )

        one_hot = F.one_hot(tissue_ids, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        if self.downsample_factor > 1:
            one_hot = F.avg_pool2d(
                one_hot,
                kernel_size=self.downsample_factor,
                stride=self.downsample_factor,
            )
        one_hot = one_hot * self.scale
        if squeeze_batch:
            return one_hot.squeeze(0)
        return one_hot

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, "
            f"downsample_factor={self.downsample_factor}, scale={self.scale}"
        )

"""Downsample HTE tissue features to the FLUX latent resolution."""

from __future__ import annotations

import torch
import torch.nn as nn


def _resolve_group_count(num_channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


class _DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_resolve_group_count(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TissueConditionDownsampler(nn.Module):
    """Learned 8x spatial downsampling for HTE feature maps."""

    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 64,
        out_channels: int | None = None,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}.")

        out_channels = hidden_channels if out_channels is None else out_channels

        blocks = []
        current_in = in_channels
        for block_idx in range(num_blocks):
            current_out = out_channels if block_idx == num_blocks - 1 else hidden_channels
            blocks.append(_DownsampleBlock(current_in, current_out))
            current_in = current_out
        self.blocks = nn.Sequential(*blocks)

    def forward(self, tissue_feat: torch.Tensor) -> torch.Tensor:
        if tissue_feat.ndim != 4:
            raise ValueError(
                f"Expected tissue_feat with shape (B, C, H, W), got {tuple(tissue_feat.shape)}."
            )
        return self.blocks(tissue_feat)

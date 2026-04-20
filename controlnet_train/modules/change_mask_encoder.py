"""Encode a binary change mask into a light learned feature map."""

from __future__ import annotations

import torch
import torch.nn as nn

from .tissue_condition_downsampler import _resolve_group_count


class ChangeMaskEncoder(nn.Module):
    """Project a 1-channel binary edit mask to a small learned feature space."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}.")

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_resolve_group_count(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, change_mask: torch.Tensor) -> torch.Tensor:
        if change_mask.ndim != 4:
            raise ValueError(
                f"Expected change_mask with shape (B, C, H, W), got {tuple(change_mask.shape)}."
            )
        return self.encoder(change_mask.float())

"""Encode nuclei ID maps into learned condition features."""

from __future__ import annotations

import torch
import torch.nn as nn

from .tissue_condition_downsampler import _resolve_group_count


class _NucleiDownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_resolve_group_count(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class NucleiConditionEncoder(nn.Module):
    """Encode raw or remapped nuclei IDs into a latent-resolution feature map."""

    RAW_TO_INTERNAL = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        101: 1,
        102: 2,
        103: 3,
        104: 4,
        105: 5,
    }

    def __init__(
        self,
        num_embeddings: int = 6,
        embedding_dim: int = 16,
        out_channels: int = 16,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}.")

        mapping = torch.full((106,), -1, dtype=torch.long)
        for raw_id, internal_id in self.RAW_TO_INTERNAL.items():
            mapping[raw_id] = internal_id

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.register_buffer("id_lookup", mapping, persistent=False)

        blocks = []
        current_in = embedding_dim
        for _ in range(num_blocks):
            blocks.append(_NucleiDownsampleBlock(current_in, out_channels))
            current_in = out_channels
        self.downsampler = nn.Sequential(*blocks)

    def remap_ids(self, nuclei_ids: torch.Tensor) -> torch.Tensor:
        nuclei_ids = nuclei_ids.long()
        if nuclei_ids.numel() == 0:
            return nuclei_ids

        min_id = int(nuclei_ids.min().item())
        max_id = int(nuclei_ids.max().item())
        if min_id < 0 or max_id >= self.id_lookup.numel():
            raise ValueError(
                f"nuclei_ids out of supported range: got [{min_id}, {max_id}], expected IDs within [0, 105]."
            )

        remapped = self.id_lookup[nuclei_ids]
        if (remapped < 0).any():
            invalid_ids = torch.unique(nuclei_ids[remapped < 0]).tolist()
            raise ValueError(f"Unsupported nuclei IDs encountered: {invalid_ids}")
        return remapped

    def forward(self, nuclei_ids: torch.Tensor) -> torch.Tensor:
        squeeze_batch = nuclei_ids.ndim == 2
        if squeeze_batch:
            nuclei_ids = nuclei_ids.unsqueeze(0)

        if nuclei_ids.ndim != 3:
            raise ValueError(
                f"Expected nuclei_ids with shape (B, H, W) or (H, W), got {tuple(nuclei_ids.shape)}."
            )

        remapped = self.remap_ids(nuclei_ids)
        embedded = self.embedding(remapped).permute(0, 3, 1, 2).contiguous()
        features = self.downsampler(embedded)

        if squeeze_batch:
            return features.squeeze(0)
        return features

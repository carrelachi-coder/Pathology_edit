"""Hierarchical Tissue Embedding used by Phase 5 ControlNet training."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from dataset_config import FINE_TO_PARENT, NUM_COARSE, NUM_FINE


class HierarchicalTissueEmbedding(nn.Module):
    """Encode unified fine tissue IDs into hierarchical embeddings.

    Each fine embedding is defined as:

        fine_emb[fine_id] = parent_emb[parent_id] + delta_emb[fine_id]

    For coarse-only datasets, the fine IDs are usually in ``0..7`` and the
    zero-initialized delta table lets training update parent embeddings
    directly. Fine-grained datasets can then learn residual offsets on top of
    the shared parent representation.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        num_coarse: int = NUM_COARSE,
        num_fine: int = NUM_FINE,
    ) -> None:
        super().__init__()
        if num_coarse != NUM_COARSE:
            raise ValueError(f"Expected {NUM_COARSE} coarse labels, got {num_coarse}.")
        if num_fine != NUM_FINE:
            raise ValueError(f"Expected {NUM_FINE} fine labels, got {num_fine}.")

        parent_lookup = [FINE_TO_PARENT[fine_id] for fine_id in range(num_fine)]

        self.embedding_dim = embedding_dim
        self.num_coarse = num_coarse
        self.num_fine = num_fine

        self.parent_embeddings = nn.Embedding(num_coarse, embedding_dim)
        self.delta_embeddings = nn.Embedding(num_fine, embedding_dim)
        self.register_buffer(
            "fine_to_parent",
            torch.tensor(parent_lookup, dtype=torch.long),
            persistent=False,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.parent_embeddings.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.delta_embeddings.weight)

    def embedding_table(self) -> torch.Tensor:
        """Return the current fine embedding table with shape (num_fine, dim)."""
        parent_ids = self.fine_to_parent
        return self.parent_embeddings(parent_ids) + self.delta_embeddings.weight

    def forward(self, tissue_ids: torch.Tensor) -> torch.Tensor:
        """Encode a tissue ID map.

        Args:
            tissue_ids: ``(B, H, W)`` or ``(H, W)`` tensor of unified fine IDs.

        Returns:
            Tensor of shape ``(B, D, H, W)`` or ``(D, H, W)``.
        """
        squeeze_batch = tissue_ids.ndim == 2
        if squeeze_batch:
            tissue_ids = tissue_ids.unsqueeze(0)

        if tissue_ids.ndim != 3:
            raise ValueError(
                f"Expected tissue_ids to have shape (B, H, W) or (H, W), got {tuple(tissue_ids.shape)}."
            )

        tissue_ids = tissue_ids.long()
        if tissue_ids.numel() > 0:
            min_id = int(tissue_ids.min().item())
            max_id = int(tissue_ids.max().item())
            if min_id < 0 or max_id >= self.num_fine:
                raise ValueError(
                    f"tissue_ids out of range: got [{min_id}, {max_id}], expected [0, {self.num_fine - 1}]."
                )

        parent_ids = self.fine_to_parent[tissue_ids]
        parent_feat = self.parent_embeddings(parent_ids)
        delta_feat = self.delta_embeddings(tissue_ids)
        features = parent_feat + delta_feat
        features = features.permute(0, 3, 1, 2).contiguous()

        if squeeze_batch:
            return features.squeeze(0)
        return features

    def extra_repr(self) -> str:
        return (
            f"embedding_dim={self.embedding_dim}, "
            f"num_coarse={self.num_coarse}, num_fine={self.num_fine}"
        )

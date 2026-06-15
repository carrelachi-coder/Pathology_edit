"""Frozen CONCH image encoder for region-level scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConchFeatureEncoder(nn.Module):
    """Extract frozen CONCH ViT spatial patch tokens from RGB images.

    Inputs are expected as ``(B, 3, H, W)`` tensors in ``[0, 1]``. The encoder
    returns unprojected spatial patch tokens shaped ``(B, T, C)`` so the existing
    region-stat loss can compare target/reference regions by label.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        conch_root: str | Path | None = None,
        model_cfg: str = "conch_ViT-B-16",
    ) -> None:
        super().__init__()
        checkpoint = Path(checkpoint_path)
        root = Path(conch_root) if conch_root is not None else _infer_conch_root(checkpoint)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from conch.open_clip_custom import create_model_from_pretrained

        self.model = create_model_from_pretrained(
            model_cfg,
            str(checkpoint),
            device="cpu",
            return_transform=False,
        )
        self.model.requires_grad_(False)
        self.model.eval()

        image_size = getattr(self.model.visual, "image_size", (448, 448))
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        image_mean = getattr(self.model.visual, "image_mean", (0.48145466, 0.4578275, 0.40821073))
        image_std = getattr(self.model.visual, "image_std", (0.26862954, 0.26130258, 0.27577711))
        self.register_buffer("mean", torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def extract_features(self, images: torch.Tensor, *, allow_input_grad: bool = False) -> torch.Tensor:
        def _extract() -> torch.Tensor:
            self.model.eval()
            x = self._prepare_input(images)
            tokens = self.model.visual.trunk(x)
            if tokens.ndim != 3:
                raise ValueError(f"CONCH visual trunk must return (B,T,C), got {tuple(tokens.shape)}")
            return tokens[:, 1:, :] if tokens.shape[1] > self.num_spatial_tokens else tokens

        if allow_input_grad:
            return _extract()
        with torch.no_grad():
            return _extract()

    @property
    def num_spatial_tokens(self) -> int:
        raw_patch_size = getattr(
            getattr(self.model.visual.trunk, "patch_embed", None),
            "patch_size",
            (16, 16),
        )
        patch_size = int(raw_patch_size[0] if isinstance(raw_patch_size, (tuple, list)) else raw_patch_size)
        return (self.image_size[0] // patch_size) * (self.image_size[1] // patch_size)

    def _prepare_input(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"CONCH input must have shape (B,3,H,W), got {tuple(images.shape)}")
        param = next(self.model.parameters())
        x = images.to(device=param.device, dtype=param.dtype)
        x = F.interpolate(x, size=self.image_size, mode="bicubic", align_corners=False)
        x = x.clamp(0.0, 1.0)
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std


def _infer_conch_root(checkpoint_path: Path) -> Path:
    for parent in [checkpoint_path.parent, *checkpoint_path.parents]:
        if (parent / "conch" / "open_clip_custom").is_dir():
            return parent
    return checkpoint_path.parent

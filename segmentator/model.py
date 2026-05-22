from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _load_uni2h_model(local_repo: str | Path, checkpoint_name: str = "pytorch_model.bin", device: torch.device | None = None) -> nn.Module:
    local_repo = Path(local_repo)
    checkpoint_path = local_repo / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    import timm

    model = timm.create_model(
        model_name="vit_giant_patch14_224",
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1e-5,
        embed_dim=1536,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    if device is not None:
        model.to(device)
    return model


class Uni2hFeatureEncoder(nn.Module):
    def __init__(self, local_repo: str | Path, checkpoint_name: str = "pytorch_model.bin", freeze: bool = True) -> None:
        super().__init__()
        self.backbone = _load_uni2h_model(local_repo, checkpoint_name=checkpoint_name)
        self.freeze = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.feature_channels = (1536, 1536, 1536, 1536)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if hasattr(self.backbone, "get_intermediate_layers"):
            kwargs: dict[str, Any] = {"n": [5, 11, 17, 23], "reshape": True}
            signature = inspect.signature(self.backbone.get_intermediate_layers)
            if "return_class_token" in signature.parameters:
                kwargs["return_class_token"] = False
            elif "return_prefix_tokens" in signature.parameters:
                kwargs["return_prefix_tokens"] = False
            features = self.backbone.get_intermediate_layers(x, **kwargs)
            if isinstance(features, tuple):
                features = list(features)
            if len(features) == 4:
                return list(features)

        if hasattr(self.backbone, "forward_features"):
            features = self.backbone.forward_features(x)
            if isinstance(features, (list, tuple)) and len(features) == 4:
                maps: list[torch.Tensor] = []
                for feat in features:
                    if feat.ndim == 3:
                        feat = feat[:, 1:, :]
                        b, n, c = feat.shape
                        side = int(n**0.5)
                        feat = feat[:, : side * side, :].transpose(1, 2).reshape(b, c, side, side)
                    elif feat.ndim == 2:
                        feat = feat[:, :, None, None]
                    maps.append(feat)
                return maps

        feat = self.backbone(x)
        if feat.ndim == 2:
            feat = feat[:, :, None, None]
        return [feat, feat, feat, feat]


class UPerLikeDecoder(nn.Module):
    def __init__(self, feature_channels: tuple[int, int, int, int], num_classes: int) -> None:
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(c) for c in feature_channels)
        self.lateral = nn.ModuleList(nn.Conv2d(c, 256, 1) for c in feature_channels)
        self.fuse = nn.Sequential(
            ConvBlock(256 * 4, 512),
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        target_size = feats[0].shape[-2:]
        fused = []
        for feat, norm, proj in zip(feats, self.norms, self.lateral):
            feat = norm(feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
            x = proj(feat)
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            fused.append(x)
        return self.fuse(torch.cat(fused, dim=1))


@dataclass(frozen=True)
class Uni2hConfig:
    local_repo: str | Path = "UNI-2h"
    checkpoint_name: str = "pytorch_model.bin"
    num_classes: int = 8
    freeze_encoder: bool = True


class BaselineSegmenter(nn.Module):
    def __init__(self, num_classes: int = 8, freeze_encoder: bool = True, local_repo: str | Path = "UNI-2h") -> None:
        super().__init__()
        self.encoder = Uni2hFeatureEncoder(local_repo=local_repo, freeze=freeze_encoder)
        self.decoder = UPerLikeDecoder((1536, 1536, 1536, 1536), num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        pad_h = (14 - x.shape[-2] % 14) % 14
        pad_w = (14 - x.shape[-1] % 14) % 14
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        feats = self.encoder(x)
        if len(feats) != 4:
            raise RuntimeError(f"expected 4 feature maps from UNI2-h, got {len(feats)}")
        for feat in feats:
            if feat.shape[1] != 1536:
                raise RuntimeError(f"expected UNI2-h channel dimension 1536, got {feat.shape[1]}")
        logits = self.decoder(feats)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        logits = logits[..., : input_size[0], : input_size[1]]
        probs = logits.softmax(dim=1)
        entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=1)
        pred = probs.argmax(dim=1)
        return {"logits": logits, "probs": probs, "entropy": entropy, "pred": pred}

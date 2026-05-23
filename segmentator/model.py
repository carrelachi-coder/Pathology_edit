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
    def __init__(
        self,
        local_repo: str | Path,
        checkpoint_name: str = "pytorch_model.bin",
        freeze: bool = True,
        intermediate_layers: tuple[int, ...] = (5, 11, 17, 23),
    ) -> None:
        super().__init__()
        self.backbone = _load_uni2h_model(local_repo, checkpoint_name=checkpoint_name)
        self.freeze = freeze
        self.intermediate_layers = intermediate_layers
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.feature_channels = tuple(1536 for _ in intermediate_layers)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if hasattr(self.backbone, "get_intermediate_layers"):
            kwargs: dict[str, Any] = {"n": list(self.intermediate_layers), "reshape": True}
            signature = inspect.signature(self.backbone.get_intermediate_layers)
            if "return_class_token" in signature.parameters:
                kwargs["return_class_token"] = False
            elif "return_prefix_tokens" in signature.parameters:
                kwargs["return_prefix_tokens"] = False
            features = self.backbone.get_intermediate_layers(x, **kwargs)
            if isinstance(features, tuple):
                features = list(features)
            if len(features) == len(self.intermediate_layers):
                return list(features)

        if hasattr(self.backbone, "forward_features"):
            features = self.backbone.forward_features(x)
            if isinstance(features, (list, tuple)) and len(features) >= len(self.intermediate_layers):
                maps: list[torch.Tensor] = []
                for feat in features[-len(self.intermediate_layers) :]:
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
        return [feat for _ in self.intermediate_layers]


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


class SimpleFeaturePyramid(nn.Module):
    """Build patch14-compatible multi-scale features from a single ViT map."""

    def __init__(self, in_channels: int = 1536, out_channels: int = 256) -> None:
        super().__init__()
        self.out_channels = (out_channels, out_channels, out_channels, out_channels)
        self.strides = (7, 14, 28, 56)
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                    nn.GroupNorm(32, out_channels),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.GroupNorm(32, out_channels),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.GroupNorm(32, out_channels),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=4, stride=4),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.GroupNorm(32, out_channels),
                    nn.GELU(),
                ),
            ]
        )

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        if not feats:
            raise RuntimeError("SimpleFeaturePyramid requires at least one feature map")
        base = feats[-1]
        return [stage(base) for stage in self.stages]


class OfficialMask2FormerDecoder(nn.Module):
    """Thin wrapper around MMSegmentation's official Mask2FormerHead."""

    def __init__(
        self,
        feature_channels: tuple[int, int, int, int],
        num_classes: int,
        num_queries: int = 100,
        feature_strides: tuple[int, int, int, int] = (7, 14, 28, 56),
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        try:
            from mmseg.registry import MODELS
        except ImportError as exc:
            raise ImportError(
                "decoder='mask2former' requires MMSegmentation. Install mmsegmentation "
                "and compatible mmcv/mmengine/mmdet packages in the training environment."
            ) from exc

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.ignore_index = ignore_index
        self.head = MODELS.build(
            dict(
                type="Mask2FormerHead",
                in_channels=list(feature_channels),
                strides=list(feature_strides),
                feat_channels=256,
                out_channels=256,
                num_classes=num_classes,
                num_queries=num_queries,
                ignore_index=ignore_index,
                num_transformer_feat_level=3,
                align_corners=False,
                pixel_decoder=dict(
                    type="mmdet.MSDeformAttnPixelDecoder",
                    num_outs=3,
                    norm_cfg=dict(type="GN", num_groups=32),
                    act_cfg=dict(type="ReLU"),
                    encoder=dict(
                        num_layers=6,
                        layer_cfg=dict(
                            self_attn_cfg=dict(
                                embed_dims=256,
                                num_heads=8,
                                num_levels=3,
                                num_points=4,
                                im2col_step=64,
                                dropout=0.0,
                                batch_first=True,
                                norm_cfg=None,
                                init_cfg=None,
                            ),
                            ffn_cfg=dict(
                                embed_dims=256,
                                feedforward_channels=1024,
                                num_fcs=2,
                                ffn_drop=0.0,
                                act_cfg=dict(type="ReLU", inplace=True),
                            ),
                        ),
                        init_cfg=None,
                    ),
                    positional_encoding=dict(num_feats=128, normalize=True),
                    init_cfg=None,
                ),
                enforce_decoder_input_project=False,
                positional_encoding=dict(num_feats=128, normalize=True),
                transformer_decoder=dict(
                    return_intermediate=True,
                    num_layers=9,
                    layer_cfg=dict(
                        self_attn_cfg=dict(
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.0,
                            proj_drop=0.0,
                            dropout_layer=None,
                            batch_first=True,
                        ),
                        cross_attn_cfg=dict(
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.0,
                            proj_drop=0.0,
                            dropout_layer=None,
                            batch_first=True,
                        ),
                        ffn_cfg=dict(
                            embed_dims=256,
                            feedforward_channels=2048,
                            num_fcs=2,
                            act_cfg=dict(type="ReLU", inplace=True),
                            ffn_drop=0.0,
                            dropout_layer=None,
                            add_identity=True,
                        ),
                    ),
                    init_cfg=None,
                ),
                loss_cls=dict(
                    type="mmdet.CrossEntropyLoss",
                    use_sigmoid=False,
                    loss_weight=2.0,
                    reduction="mean",
                    class_weight=[1.0] * num_classes + [0.1],
                ),
                loss_mask=dict(
                    type="mmdet.CrossEntropyLoss",
                    use_sigmoid=True,
                    reduction="mean",
                    loss_weight=5.0,
                ),
                loss_dice=dict(
                    type="mmdet.DiceLoss",
                    use_sigmoid=True,
                    activate=True,
                    reduction="mean",
                    naive_dice=True,
                    eps=1.0,
                    loss_weight=5.0,
                ),
                train_cfg=dict(
                    num_points=12544,
                    oversample_ratio=3.0,
                    importance_sample_ratio=0.75,
                    assigner=dict(
                        type="mmdet.HungarianAssigner",
                        match_costs=[
                            dict(type="mmdet.ClassificationCost", weight=2.0),
                            dict(type="mmdet.CrossEntropyLossCost", weight=5.0, use_sigmoid=True),
                            dict(type="mmdet.DiceCost", weight=5.0, pred_act=True, eps=1.0),
                        ],
                    ),
                    sampler=dict(type="mmdet.MaskPseudoSampler"),
                ),
            )
        )

    def forward(self, feats: list[torch.Tensor], image_shape: tuple[int, int]) -> torch.Tensor:
        batch_size = feats[0].shape[0]
        batch_img_metas = [
            {
                "batch_input_shape": image_shape,
                "img_shape": image_shape,
                "ori_shape": image_shape,
                "pad_shape": image_shape,
            }
            for _ in range(batch_size)
        ]
        results = self.head.predict(tuple(feats), batch_img_metas, test_cfg=dict(mode="whole"))
        if torch.is_tensor(results):
            return results
        logits = []
        for result in results:
            if hasattr(result, "seg_logits"):
                logits.append(result.seg_logits.data)
            elif isinstance(result, dict) and "seg_logits" in result:
                seg_logits = result["seg_logits"]
                logits.append(seg_logits.data if hasattr(seg_logits, "data") else seg_logits)
            else:
                raise TypeError(
                    "Mask2FormerHead.predict returned an unsupported result type. "
                    f"Expected SegDataSample with seg_logits, got {type(result)!r}."
                )
        return torch.stack(logits, dim=0)

    def loss(self, feats: list[torch.Tensor], target: torch.Tensor, image_shape: tuple[int, int]) -> dict[str, torch.Tensor]:
        try:
            from mmengine.structures import PixelData
            from mmseg.structures import SegDataSample
        except ImportError as exc:
            raise ImportError("Mask2Former loss requires mmengine and mmsegmentation.") from exc

        batch_samples = []
        for idx in range(target.shape[0]):
            sample = SegDataSample()
            sample.gt_sem_seg = PixelData(data=target[idx : idx + 1].long())
            sample.set_metainfo(
                {
                    "batch_input_shape": image_shape,
                    "img_shape": image_shape,
                    "ori_shape": image_shape,
                    "pad_shape": image_shape,
                }
            )
            batch_samples.append(sample)
        return self.head.loss(tuple(feats), batch_samples, train_cfg=self.head.train_cfg)


@dataclass(frozen=True)
class Uni2hConfig:
    local_repo: str | Path = "UNI-2h"
    checkpoint_name: str = "pytorch_model.bin"
    num_classes: int = 8
    freeze_encoder: bool = True


class BaselineSegmenter(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        freeze_encoder: bool = True,
        local_repo: str | Path = "UNI-2h",
        decoder: str = "upernet",
        mask2former_queries: int = 100,
        mask2former_ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.decoder_name = decoder
        encoder_layers = (23,) if decoder == "mask2former" else (5, 11, 17, 23)
        self.encoder = Uni2hFeatureEncoder(local_repo=local_repo, freeze=freeze_encoder, intermediate_layers=encoder_layers)
        if decoder == "upernet":
            self.feature_pyramid = None
            self.decoder = UPerLikeDecoder((1536, 1536, 1536, 1536), num_classes)
        elif decoder == "mask2former":
            self.feature_pyramid = SimpleFeaturePyramid(1536, 256)
            self.decoder = OfficialMask2FormerDecoder(
                self.feature_pyramid.out_channels,
                num_classes,
                num_queries=mask2former_queries,
                feature_strides=self.feature_pyramid.strides,
                ignore_index=mask2former_ignore_index,
            )
        else:
            raise ValueError(f"unsupported decoder: {decoder}")

    def _input_alignment(self) -> int:
        return 56 if isinstance(self.decoder, OfficialMask2FormerDecoder) else 14

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        align = self._input_alignment()
        pad_h = (align - x.shape[-2] % align) % align
        pad_w = (align - x.shape[-1] % align) % align
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        feats = self.encoder(x)
        if self.feature_pyramid is not None:
            feats = self.feature_pyramid(feats)
        if len(feats) != 4:
            raise RuntimeError(f"expected 4 feature maps from UNI2-h, got {len(feats)}")
        for feat in feats:
            expected_channels = 256 if isinstance(self.decoder, OfficialMask2FormerDecoder) else 1536
            if feat.shape[1] != expected_channels:
                raise RuntimeError(f"expected decoder feature channel dimension {expected_channels}, got {feat.shape[1]}")
        if isinstance(self.decoder, OfficialMask2FormerDecoder):
            logits = self.decoder(feats, x.shape[-2:])
        else:
            logits = self.decoder(feats)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        logits = logits[..., : input_size[0], : input_size[1]]
        probs = logits.softmax(dim=1)
        entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=1)
        pred = probs.argmax(dim=1)
        return {"logits": logits, "probs": probs, "entropy": entropy, "pred": pred}

    def loss(self, x: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        align = self._input_alignment()
        pad_h = (align - x.shape[-2] % align) % align
        pad_w = (align - x.shape[-1] % align) % align
        ignore_index = self.decoder.ignore_index if isinstance(self.decoder, OfficialMask2FormerDecoder) else 255
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            target = F.pad(target[:, None].long(), (0, pad_w, 0, pad_h), mode="constant", value=ignore_index).squeeze(1)
        feats = self.encoder(x)
        if self.feature_pyramid is not None:
            feats = self.feature_pyramid(feats)
        if hasattr(self.decoder, "loss"):
            target = target.clone()
            invalid = ((target < 0) | (target >= self.num_classes)) & (target != ignore_index)
            target[invalid] = self.decoder.ignore_index
            losses = self.decoder.loss(feats, target, x.shape[-2:])
            total = sum(value for value in losses.values() if torch.is_tensor(value))
            losses = dict(losses)
            losses["total"] = total
            return losses
        logits = self.decoder(feats)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        logits = logits[..., : input_size[0], : input_size[1]]
        from .losses import segmentation_loss

        return segmentation_loss(logits, target[..., : input_size[0], : input_size[1]], self.num_classes, invalid_to=ignore_index)

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from dataset_config.unified_labels import FINE_TO_PARENT, NUM_COARSE, NUM_FINE


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
        self.trainable_block_count = 0
        self.intermediate_layers = intermediate_layers
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.feature_channels = tuple(1536 for _ in intermediate_layers)

    def select_trainable_blocks(self, count: int) -> list[nn.Parameter]:
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None or count <= 0:
            return []
        selected_modules = list(blocks)[-min(count, len(blocks)) :]
        norm = getattr(self.backbone, "norm", None)
        if norm is not None:
            selected_modules.append(norm)
        parameters: list[nn.Parameter] = []
        for module in selected_modules:
            parameters.extend(module.parameters())
        return list(dict.fromkeys(parameters))

    def set_selected_blocks_trainable(self, count: int, trainable: bool) -> list[nn.Parameter]:
        selected = self.select_trainable_blocks(count)
        for parameter in selected:
            parameter.requires_grad = trainable
        self.freeze = not trainable
        self.trainable_block_count = count if trainable else 0
        return selected

    def train(self, mode: bool = True) -> Uni2hFeatureEncoder:
        super().train(mode)
        self.backbone.eval()
        if mode and self.trainable_block_count > 0:
            blocks = getattr(self.backbone, "blocks", ())
            for block in list(blocks)[-self.trainable_block_count :]:
                block.train(True)
            norm = getattr(self.backbone, "norm", None)
            if norm is not None:
                norm.train(True)
        return self

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
    """Build patch14-compatible feature levels from distinct ViT depths."""

    def __init__(self, in_channels: int | tuple[int, int, int, int] = 1536, out_channels: int = 256) -> None:
        super().__init__()
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels, in_channels, in_channels)
        if len(in_channels) != 4:
            raise ValueError(f"SimpleFeaturePyramid expects 4 input channel counts, got {len(in_channels)}")
        self.out_channels = (out_channels, out_channels, out_channels, out_channels)
        self.strides = (7, 14, 28, 56)
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels[0], out_channels, kernel_size=2, stride=2),
                    nn.GroupNorm(32, out_channels),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels[1], out_channels, kernel_size=1),
                    nn.GroupNorm(32, out_channels),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(in_channels[2], out_channels, kernel_size=1),
                    nn.GroupNorm(32, out_channels),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=4, stride=4),
                    nn.Conv2d(in_channels[3], out_channels, kernel_size=1),
                    nn.GroupNorm(32, out_channels),
                    nn.GELU(),
                ),
            ]
        )

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(feats) != 4:
            raise RuntimeError(f"SimpleFeaturePyramid requires 4 feature maps from distinct depths, got {len(feats)}")
        return [stage(feat) for stage, feat in zip(self.stages, feats)]


class BoundaryRefinementHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_channels: int = 64,
        gate_width: int = 4,
        gate_threshold: float = 0.15,
    ) -> None:
        super().__init__()
        if gate_width < 1:
            raise ValueError("gate_width must be positive")
        if not 0.0 <= gate_threshold < 1.0:
            raise ValueError("gate_threshold must be in [0, 1)")
        self.gate_width = gate_width
        self.gate_threshold = gate_threshold
        self.image_stem = nn.Sequential(
            nn.Conv2d(3, hidden_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
        )
        self.logit_projection = nn.Conv2d(num_classes, hidden_channels, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, num_classes, 1),
        )
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def boundary_gate(self, logits: torch.Tensor) -> torch.Tensor:
        probabilities = logits.detach().softmax(dim=1)
        kernel = 2 * self.gate_width + 1
        prediction = probabilities.argmax(dim=1)
        hard_classes = F.one_hot(prediction, num_classes=probabilities.shape[1]).permute(0, 3, 1, 2).float()
        dilation = F.max_pool2d(hard_classes, kernel_size=kernel, stride=1, padding=self.gate_width)
        erosion = -F.max_pool2d(-hard_classes, kernel_size=kernel, stride=1, padding=self.gate_width)
        boundary_band = ((dilation - erosion).amax(dim=1, keepdim=True) > 0).to(probabilities.dtype)
        if probabilities.shape[1] > 1:
            top_two = probabilities.topk(2, dim=1).values
            uncertainty = 1.0 - (top_two[:, :1] - top_two[:, 1:2])
        else:
            uncertainty = 1.0 - probabilities
        uncertainty = ((uncertainty - self.gate_threshold) / (1.0 - self.gate_threshold)).clamp(0.0, 1.0)
        return boundary_band * (0.25 + 0.75 * uncertainty)

    def forward(
        self,
        image: torch.Tensor,
        logits: torch.Tensor,
        *,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        image_features = self.image_stem(image)
        logit_features = self.logit_projection(F.interpolate(logits, size=image_features.shape[-2:], mode="bilinear", align_corners=False))
        residual = self.fuse(torch.cat([image_features, logit_features], dim=1))
        residual = F.interpolate(residual, size=logits.shape[-2:], mode="bilinear", align_corners=False)
        gate = self.boundary_gate(logits)
        refined = logits + self.residual_scale * gate * residual
        return (refined, gate) if return_gate else refined


class HierarchicalFineHead(nn.Module):
    """Predict unified fine labels from the highest-resolution shared feature map."""

    def __init__(self, in_channels: int, num_subtypes: int, hidden_channels: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(16, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(16, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, num_subtypes, 1),
        )

    def forward(self, feature: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        logits = self.net(feature)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


def compose_hierarchical_prediction(coarse_pred: torch.Tensor, fine_pred: torch.Tensor) -> torch.Tensor:
    """Return fine IDs while falling back to coarse labels on any parent mismatch."""
    if coarse_pred.shape != fine_pred.shape:
        raise ValueError(f"coarse and fine predictions must have the same shape, got {coarse_pred.shape} and {fine_pred.shape}")
    parent_lookup = torch.tensor([FINE_TO_PARENT[idx] for idx in range(NUM_FINE)], device=fine_pred.device)
    parent_consistent = parent_lookup[fine_pred.long()] == coarse_pred
    return torch.where(parent_consistent, fine_pred, coarse_pred)


class CellPriorEncoder(nn.Module):
    def __init__(self, out_channels: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, out_channels, 3, padding=1),
        )

    def forward(self, density: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        density = F.interpolate(density, size=size, mode="bilinear", align_corners=False)
        return self.net(density)


class CellDensityHead(nn.Module):
    def __init__(self, in_channels: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.Conv2d(128, 6, 1),
        )

    def forward(self, features: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return torch.sigmoid(F.interpolate(self.net(features), size=size, mode="bilinear", align_corners=False))


class FineCellTeacherAdapter(nn.Module):
    """A zero-initialized residual adapter shared by the Fine and density heads."""

    def __init__(self, channels: int = 256, hidden_channels: int = 64) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.residual(features)


class OfficialMask2FormerDecoder(nn.Module):
    """Thin wrapper around MMSegmentation's official Mask2FormerHead."""

    def __init__(
        self,
        feature_channels: tuple[int, int, int, int],
        num_classes: int,
        num_queries: int = 100,
        feature_strides: tuple[int, int, int, int] = (7, 14, 28, 56),
        ignore_index: int = 255,
        class_weights: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        try:
            from mmengine.config import ConfigDict
            from mmseg.registry import MODELS
        except ImportError as exc:
            raise ImportError(
                "decoder='mask2former' requires MMSegmentation. Install mmsegmentation "
                "and compatible mmcv/mmengine/mmdet packages in the training environment."
            ) from exc

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.ignore_index = ignore_index
        if class_weights is not None and len(class_weights) != num_classes:
            raise ValueError(f"expected {num_classes} class weights, got {len(class_weights)}")
        query_class_weights = list(class_weights or (1.0,) * num_classes) + [0.1]
        self.head = MODELS.build(
            ConfigDict(
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
                    class_weight=query_class_weights,
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
        mask2former_class_weights: tuple[float, ...] | None = None,
        symmetric_padding: bool = False,
        boundary_refinement: bool = False,
        refinement_loss_weight: float = 1.0,
        refinement_boundary_weight: float = 0.5,
        refinement_boundary_widths: tuple[int, ...] = (2, 4, 8),
        refinement_boundary_ce_weight: float = 0.0,
        refinement_consistency_weight: float = 0.0,
        refinement_gate_width: int = 4,
        refinement_gate_threshold: float = 0.15,
        cellvit_mode: str = "none",
        cell_prior_dropout: float = 0.2,
        cell_aux_loss_weight: float = 0.2,
        hierarchical_fine: bool = False,
        fine_loss_weight: float = 1.0,
        fine_only_loss: bool = False,
        refinement_only_loss: bool = False,
        fine_class_weights: tuple[float, ...] | None = None,
        fine_supported_ids: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.decoder_name = decoder
        self.symmetric_padding = symmetric_padding
        self.refinement_loss_weight = refinement_loss_weight
        self.refinement_boundary_weight = refinement_boundary_weight
        self.refinement_boundary_widths = refinement_boundary_widths
        self.refinement_boundary_ce_weight = refinement_boundary_ce_weight
        self.refinement_consistency_weight = refinement_consistency_weight
        self.refinement_gate_width = refinement_gate_width
        self.cellvit_mode = cellvit_mode
        self.cell_prior_dropout = cell_prior_dropout
        self.cell_aux_loss_weight = cell_aux_loss_weight
        self.hierarchical_fine = hierarchical_fine
        self.fine_loss_weight = fine_loss_weight
        self.fine_only_loss = fine_only_loss
        self.refinement_only_loss = refinement_only_loss
        coarse_weights = (
            torch.tensor(mask2former_class_weights, dtype=torch.float32)
            if mask2former_class_weights is not None
            else torch.ones(num_classes, dtype=torch.float32)
        )
        self.register_buffer("coarse_class_weights", coarse_weights, persistent=False)
        if fine_only_loss and not hierarchical_fine:
            raise ValueError("fine_only_loss requires hierarchical_fine")
        if refinement_only_loss and not boundary_refinement:
            raise ValueError("refinement_only_loss requires boundary_refinement")
        if fine_only_loss and refinement_only_loss:
            raise ValueError("fine_only_loss and refinement_only_loss are mutually exclusive")
        if cellvit_mode not in {"none", "teacher", "input"}:
            raise ValueError(f"unsupported CellViT mode: {cellvit_mode}")
        encoder_layers = (5, 11, 17, 23)
        self.encoder = Uni2hFeatureEncoder(local_repo=local_repo, freeze=freeze_encoder, intermediate_layers=encoder_layers)
        if decoder == "upernet":
            self.feature_pyramid = None
            self.decoder = UPerLikeDecoder((1536, 1536, 1536, 1536), num_classes)
        elif decoder == "mask2former":
            self.feature_pyramid = SimpleFeaturePyramid(self.encoder.feature_channels, 256)
            self.decoder = OfficialMask2FormerDecoder(
                self.feature_pyramid.out_channels,
                num_classes,
                num_queries=mask2former_queries,
                feature_strides=self.feature_pyramid.strides,
                ignore_index=mask2former_ignore_index,
                class_weights=mask2former_class_weights,
            )
        else:
            raise ValueError(f"unsupported decoder: {decoder}")
        self.refinement_head = (
            BoundaryRefinementHead(
                num_classes,
                gate_width=refinement_gate_width,
                gate_threshold=refinement_gate_threshold,
            )
            if boundary_refinement
            else None
        )
        self.cell_prior_encoder = CellPriorEncoder(256) if cellvit_mode == "input" else None
        self.cell_density_head = CellDensityHead(256) if cellvit_mode in {"teacher", "input"} else None
        fine_channels = 256 if self.feature_pyramid is not None else self.encoder.feature_channels[0]
        self.fine_head = HierarchicalFineHead(fine_channels, NUM_FINE) if hierarchical_fine else None
        self.cell_teacher_adapter = (
            FineCellTeacherAdapter(fine_channels)
            if cellvit_mode == "teacher" and hierarchical_fine
            else None
        )
        supported_ids = set(fine_supported_ids or range(NUM_FINE))
        supported_ids.update(range(NUM_COARSE))
        supported_mask = torch.tensor([fine_id in supported_ids for fine_id in range(NUM_FINE)], dtype=torch.bool)
        if hierarchical_fine and not supported_mask.any():
            raise ValueError("hierarchical fine training requires at least one supported fine label")
        self.register_buffer("fine_supported_mask", supported_mask, persistent=hierarchical_fine)
        if fine_class_weights is not None and len(fine_class_weights) != NUM_FINE:
            raise ValueError(f"expected {NUM_FINE} fine class weights, got {len(fine_class_weights)}")
        fine_weights = (
            torch.tensor(fine_class_weights, dtype=torch.float32)
            if fine_class_weights is not None
            else torch.ones(NUM_FINE, dtype=torch.float32)
        )
        self.register_buffer("fine_class_weights", fine_weights, persistent=hierarchical_fine)

    def _input_alignment(self) -> int:
        return 56 if isinstance(self.decoder, OfficialMask2FormerDecoder) else 14

    def _padding(self, height: int, width: int) -> tuple[int, int, int, int]:
        align = self._input_alignment()
        pad_h = (align - height % align) % align
        pad_w = (align - width % align) % align
        if self.symmetric_padding:
            left = pad_w // 2
            top = pad_h // 2
            return left, pad_w - left, top, pad_h - top
        return 0, pad_w, 0, pad_h

    @staticmethod
    def _crop_padding(values: torch.Tensor, padding: tuple[int, int, int, int], size: tuple[int, int]) -> torch.Tensor:
        left, _, top, _ = padding
        return values[..., top : top + size[0], left : left + size[1]]

    def _encode(self, x: torch.Tensor, nuclei_density: torch.Tensor | None) -> list[torch.Tensor]:
        feats = self.encoder(x)
        if self.feature_pyramid is not None:
            feats = self.feature_pyramid(feats)
        if len(feats) != 4:
            raise RuntimeError(f"expected 4 feature maps from UNI2-h, got {len(feats)}")
        if self.cell_prior_encoder is not None:
            if nuclei_density is None:
                nuclei_density = x.new_zeros((x.shape[0], 6, *x.shape[-2:]))
            if self.training and self.cell_prior_dropout > 0:
                keep = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) >= self.cell_prior_dropout).to(nuclei_density.dtype)
                nuclei_density = nuclei_density * keep
            feats[0] = feats[0] + self.cell_prior_encoder(nuclei_density, feats[0].shape[-2:])
        return feats

    def _fine_branch_features(self, feats: list[torch.Tensor]) -> torch.Tensor:
        features = feats[0]
        if self.cell_teacher_adapter is not None:
            features = self.cell_teacher_adapter(features)
        return features

    def _effective_fine_allowed(
        self,
        fine_allowed: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        supported = self.fine_supported_mask.to(device=device)
        if fine_allowed is None:
            allowed = torch.zeros((batch_size, NUM_COARSE, NUM_FINE), dtype=torch.bool, device=device)
            allowed[:, torch.arange(NUM_COARSE, device=device), torch.arange(NUM_COARSE, device=device)] = True
        else:
            if fine_allowed.ndim == 2:
                fine_allowed = fine_allowed.unsqueeze(0)
            expected_shape = (batch_size, NUM_COARSE, NUM_FINE)
            if fine_allowed.shape != expected_shape:
                raise ValueError(f"expected fine_allowed shape {expected_shape}, got {tuple(fine_allowed.shape)}")
            allowed = fine_allowed.to(device=device, dtype=torch.bool)
        allowed = allowed & supported[None, None, :]
        empty = ~allowed.any(dim=2)
        if empty.any():
            allowed = allowed.clone()
            batch_index, parent_index = torch.where(empty)
            allowed[batch_index, parent_index, parent_index] = True
        return allowed

    def _masked_fine_logits(
        self,
        logits: torch.Tensor,
        parent_map: torch.Tensor,
        fine_allowed: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        allowed = self._effective_fine_allowed(fine_allowed, logits.shape[0], logits.device)
        safe_parent = parent_map.clamp(0, NUM_COARSE - 1).long()
        batch_index = torch.arange(logits.shape[0], device=logits.device)[:, None, None]
        pixel_allowed = allowed[batch_index, safe_parent].permute(0, 3, 1, 2)
        logits = logits.masked_fill(~pixel_allowed, torch.finfo(logits.dtype).min)
        return logits, allowed

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None = None,
        nuclei_density: torch.Tensor | None = None,
        nuclei_available: torch.Tensor | None = None,
        fine_target: torch.Tensor | None = None,
        fine_allowed: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if nuclei_density is not None and nuclei_density.ndim == 3:
            nuclei_density = nuclei_density.unsqueeze(0)
        if target is not None:
            return self._loss_impl(
                x,
                target,
                nuclei_density=nuclei_density,
                nuclei_available=nuclei_available,
                fine_target=fine_target,
                fine_allowed=fine_allowed,
            )

        input_size = x.shape[-2:]
        padding = self._padding(*input_size)
        if any(padding):
            x = F.pad(x, padding, mode="reflect")
            if nuclei_density is not None:
                nuclei_density = F.pad(nuclei_density, padding)
        feats = self._encode(x, nuclei_density)
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
        if self.refinement_head is not None:
            logits = self.refinement_head(x, logits)
        logits = self._crop_padding(logits, padding, input_size)
        probs = logits.softmax(dim=1)
        entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=1)
        pred = probs.argmax(dim=1)
        output = {"logits": logits, "probs": probs, "entropy": entropy, "pred": pred}
        fine_features = self._fine_branch_features(feats)
        if self.fine_head is not None:
            fine_logits = self.fine_head(fine_features, x.shape[-2:])
            fine_logits = self._crop_padding(fine_logits, padding, input_size)
            fine_logits, _ = self._masked_fine_logits(fine_logits, pred, fine_allowed)
            fine_probs = fine_logits.softmax(dim=1)
            fine_pred = fine_probs.argmax(dim=1)
            output.update(
                {
                    "fine_logits": fine_logits,
                    "fine_probs": fine_probs,
                    "fine_pred": fine_pred,
                    "hierarchical_pred": compose_hierarchical_prediction(pred, fine_pred),
                }
            )
        if self.cell_density_head is not None:
            cell_density = self.cell_density_head(fine_features, x.shape[-2:])
            output["cell_density"] = self._crop_padding(cell_density, padding, input_size)
        return output

    def loss(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        nuclei_density: torch.Tensor | None = None,
        nuclei_available: torch.Tensor | None = None,
        fine_target: torch.Tensor | None = None,
        fine_allowed: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self._loss_impl(
            x,
            target,
            nuclei_density=nuclei_density,
            nuclei_available=nuclei_available,
            fine_target=fine_target,
            fine_allowed=fine_allowed,
        )

    def _loss_impl(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        nuclei_density: torch.Tensor | None = None,
        nuclei_available: torch.Tensor | None = None,
        fine_target: torch.Tensor | None = None,
        fine_allowed: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        padding = self._padding(*input_size)
        ignore_index = self.decoder.ignore_index if isinstance(self.decoder, OfficialMask2FormerDecoder) else 255
        if any(padding):
            x = F.pad(x, padding, mode="reflect")
            target = F.pad(target[:, None].long(), padding, mode="constant", value=ignore_index).squeeze(1)
            if nuclei_density is not None:
                nuclei_density = F.pad(nuclei_density, padding)
            if fine_target is not None:
                fine_target = F.pad(fine_target[:, None].long(), padding, mode="constant", value=ignore_index).squeeze(1)
        feats = self._encode(x, nuclei_density)
        if self.refinement_only_loss:
            losses = {}
            if isinstance(self.decoder, OfficialMask2FormerDecoder):
                logits = self.decoder(feats, x.shape[-2:])
            else:
                logits = self.decoder(feats)
            total = sum(
                (parameter.sum() * 0.0 for parameter in self.refinement_head.parameters()),
                x.new_zeros(()),
            )
        elif self.fine_only_loss:
            losses = {}
            total = sum((parameter.sum() * 0.0 for parameter in self.fine_head.parameters()), x.new_zeros(()))
        elif hasattr(self.decoder, "loss"):
            target = target.clone()
            invalid = ((target < 0) | (target >= self.num_classes)) & (target != ignore_index)
            target[invalid] = self.decoder.ignore_index
            losses = self.decoder.loss(feats, target, x.shape[-2:])
            losses = dict(losses)
            total = sum(value for name, value in losses.items() if torch.is_tensor(value) and "loss" in name)
        else:
            logits = self.decoder(feats)
            if logits.shape[-2:] != x.shape[-2:]:
                logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
            from .losses import segmentation_loss

            losses = segmentation_loss(logits, target, self.num_classes, invalid_to=ignore_index)
            total = losses["total"]

        if self.refinement_head is not None and not self.fine_only_loss:
            if isinstance(self.decoder, OfficialMask2FormerDecoder):
                logits = self.decoder(feats, x.shape[-2:])
            if logits.shape[-2:] != x.shape[-2:]:
                logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
            refined_logits, refinement_gate = self.refinement_head(x, logits, return_gate=True)
            from .losses import (
                boundary_band_cross_entropy,
                multi_scale_soft_boundary_loss,
                outside_boundary_consistency_loss,
                segmentation_loss,
            )

            refine_losses = segmentation_loss(
                refined_logits,
                target,
                self.num_classes,
                class_weights=self.coarse_class_weights,
                invalid_to=ignore_index,
            )
            refine_boundary = multi_scale_soft_boundary_loss(
                refined_logits,
                target,
                self.num_classes,
                widths=self.refinement_boundary_widths,
                ignore_index=ignore_index,
            )
            refine_boundary_ce = boundary_band_cross_entropy(
                refined_logits,
                target,
                self.num_classes,
                width=self.refinement_gate_width,
                class_weights=self.coarse_class_weights,
                ignore_index=ignore_index,
            )
            refine_consistency = outside_boundary_consistency_loss(
                refined_logits,
                logits,
                refinement_gate,
                target,
                ignore_index=ignore_index,
            )
            losses["loss_refine_semantic"] = refine_losses["total"] * self.refinement_loss_weight
            losses["loss_refine_boundary"] = refine_boundary * self.refinement_boundary_weight
            losses["loss_refine_boundary_ce"] = refine_boundary_ce * self.refinement_boundary_ce_weight
            losses["loss_refine_consistency"] = refine_consistency * self.refinement_consistency_weight
            losses["refinement_gate_mean"] = refinement_gate.detach().mean()
            total = (
                total
                + losses["loss_refine_semantic"]
                + losses["loss_refine_boundary"]
                + losses["loss_refine_boundary_ce"]
                + losses["loss_refine_consistency"]
            )

        fine_features = self._fine_branch_features(feats)
        if self.fine_head is not None and fine_target is not None and not self.refinement_only_loss:
            from .losses import masked_segmentation_loss

            fine_logits = self.fine_head(fine_features, x.shape[-2:])
            fine_logits, fine_allowed_effective = self._masked_fine_logits(fine_logits, target, fine_allowed)
            fine_target = fine_target.clone()
            fine_valid = (fine_target >= 0) & (fine_target < NUM_FINE)
            if fine_valid.any():
                safe_parent = target.clamp(0, NUM_COARSE - 1).long()
                batch_index = torch.arange(target.shape[0], device=target.device)[:, None, None]
                pixel_allowed = fine_allowed_effective[batch_index, safe_parent]
                target_allowed = pixel_allowed.gather(3, fine_target.clamp(0, NUM_FINE - 1).unsqueeze(-1)).squeeze(-1)
                supported = self.fine_supported_mask[fine_target.clamp(0, NUM_FINE - 1)]
                fine_target[fine_valid & (~target_allowed | ~supported)] = ignore_index
            fine_losses = masked_segmentation_loss(
                fine_logits,
                fine_target,
                NUM_FINE,
                class_weights=self.fine_class_weights,
                ignore_index=ignore_index,
            )
            losses["loss_fine_ce"] = fine_losses["ce"] * self.fine_loss_weight
            losses["loss_fine_dice"] = fine_losses["dice"] * self.fine_loss_weight
            losses["fine_valid_pixels"] = fine_losses["valid_pixels"]
            losses["loss_fine"] = fine_losses["total"] * self.fine_loss_weight
            total = total + losses["loss_fine"]

        if self.cell_density_head is not None and nuclei_density is not None:
            predicted_density = self.cell_density_head(fine_features, x.shape[-2:])
            elementwise = F.smooth_l1_loss(predicted_density, nuclei_density, reduction="none")
            density_weight = 1.0 + 4.0 * nuclei_density
            per_sample = (elementwise * density_weight).sum(dim=(1, 2, 3)) / density_weight.sum(
                dim=(1, 2, 3)
            ).clamp_min(1.0)
            availability = nuclei_available.to(per_sample.device, per_sample.dtype) if nuclei_available is not None else torch.ones_like(per_sample)
            cell_loss = (per_sample * availability).sum() / availability.sum().clamp_min(1.0)
            losses["loss_cell_density"] = cell_loss * self.cell_aux_loss_weight
            total = total + losses["loss_cell_density"]
        losses["total"] = total
        return losses

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
        feature_layer: int | None = None,
    ) -> None:
        super().__init__()
        self.feature_layer = None if feature_layer is None else int(feature_layer)
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
        self._validate_feature_layer()

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def extract_features(self, images: torch.Tensor, *, allow_input_grad: bool = False) -> torch.Tensor:
        def _extract() -> torch.Tensor:
            self.model.eval()
            x = self._prepare_input(images)
            if self.feature_layer is None:
                tokens = self.model.visual.trunk(x)
            else:
                tokens = self._extract_intermediate_features(x, self.feature_layer)
            return self._normalize_token_output(tokens)

        if allow_input_grad:
            return _extract()
        with torch.no_grad():
            return _extract()

    def _validate_feature_layer(self) -> None:
        if self.feature_layer is None:
            return
        if self.feature_layer <= 0:
            raise ValueError(f"CONCH feature_layer must be 1-based and positive, got {self.feature_layer}.")
        blocks = self._find_blocks(required=False)
        if blocks is not None and self.feature_layer > len(blocks):
            raise ValueError(f"CONCH feature_layer exceeds backbone depth {len(blocks)}: {self.feature_layer}.")

    def _extract_intermediate_features(self, x: torch.Tensor, layer_number: int) -> torch.Tensor:
        block_index = int(layer_number) - 1
        trunk = self.model.visual.trunk
        if hasattr(trunk, "get_intermediate_layers"):
            return self._call_get_intermediate_layers(x, block_index=block_index, layer_number=layer_number)
        return self._extract_layer_with_hook(x, block_index=block_index, layer_number=layer_number)

    def _call_get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        block_index: int,
        layer_number: int,
    ) -> torch.Tensor:
        trunk = self.model.visual.trunk
        attempts = (
            {"n": [block_index], "reshape": False, "return_prefix_tokens": False},
            {"n": [block_index], "reshape": False},
            {"n": [block_index]},
        )
        last_error: Exception | None = None
        for kwargs in attempts:
            try:
                outputs = trunk.get_intermediate_layers(x, **kwargs)
                break
            except TypeError as exc:
                last_error = exc
        else:
            raise RuntimeError("CONCH trunk.get_intermediate_layers call failed") from last_error
        if torch.is_tensor(outputs):
            return outputs
        if isinstance(outputs, (tuple, list)) and outputs:
            features = outputs[0]
            if isinstance(features, (tuple, list)) and features:
                features = features[0]
            if torch.is_tensor(features):
                return features
        raise RuntimeError(f"CONCH get_intermediate_layers returned no tensor for layer {layer_number}.")

    def _extract_layer_with_hook(
        self,
        x: torch.Tensor,
        *,
        block_index: int,
        layer_number: int,
    ) -> torch.Tensor:
        blocks = self._find_blocks(required=True)
        if block_index >= len(blocks):
            raise ValueError(f"CONCH feature_layer exceeds backbone depth {len(blocks)}: {layer_number}.")
        captured: dict[str, torch.Tensor] = {}

        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(tensor):
                raise TypeError(f"CONCH block {layer_number} hook output is not a tensor: {type(output)!r}")
            captured["features"] = tensor

        handle = blocks[block_index].register_forward_hook(hook)
        try:
            _ = self.model.visual.trunk(x)
        finally:
            handle.remove()
        if "features" not in captured:
            raise RuntimeError(f"CONCH block hook did not capture layer {layer_number}.")
        return captured["features"]

    def _find_blocks(self, *, required: bool) -> object | None:
        trunk = self.model.visual.trunk
        candidates = (
            getattr(trunk, "blocks", None),
            getattr(getattr(trunk, "transformer", None), "resblocks", None),
            getattr(trunk, "resblocks", None),
            getattr(getattr(trunk, "model", None), "blocks", None),
        )
        for blocks in candidates:
            if blocks is not None and hasattr(blocks, "__len__") and len(blocks) > 0:
                return blocks
        if required:
            raise RuntimeError(
                "Could not find CONCH visual transformer blocks. Expected trunk.blocks, "
                "trunk.transformer.resblocks, trunk.resblocks, or trunk.model.blocks."
            )
        return None

    def _normalize_token_output(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim == 4:
            tokens = tokens.flatten(2).transpose(1, 2)
        if tokens.ndim != 3:
            raise ValueError(f"CONCH visual trunk must return (B,T,C) or (B,C,H,W), got {tuple(tokens.shape)}")
        return tokens[:, -self.num_spatial_tokens :, :] if tokens.shape[1] > self.num_spatial_tokens else tokens

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

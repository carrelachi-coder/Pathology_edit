"""Frozen external modules for Cross V5 decoded-image losses."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


_VGG16_LAYER_ALIASES = {
    "relu1_1": 1,
    "relu1_2": 3,
    "relu2_1": 6,
    "relu2_2": 8,
    "relu3_1": 11,
    "relu3_2": 13,
    "relu3_3": 15,
    "relu4_1": 18,
    "relu4_2": 20,
    "relu4_3": 22,
}


class CrossV5VGGTextureExtractor(nn.Module):
    """Frozen shallow VGG16 features for masked Gram appearance loss.

    The forward pass is intentionally differentiable with respect to input
    images. Parameters are frozen, but calls must not run under ``no_grad`` for
    generated RGB.
    """

    def __init__(
        self,
        features: nn.Sequential,
        *,
        layer_indices: Mapping[str, int],
    ) -> None:
        super().__init__()
        if not layer_indices:
            raise ValueError("At least one VGG layer is required.")
        self.features = features
        self.layer_indices = dict(sorted(layer_indices.items(), key=lambda item: item[1]))
        self.max_layer = max(self.layer_indices.values())
        self.register_buffer(
            "mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.eval()
        self.requires_grad_(False)

    @classmethod
    def from_torchvision(
        cls,
        *,
        layers: str = "relu1_2,relu2_2,relu3_3",
        weights_path: str | Path | None = None,
        allow_download: bool = True,
    ) -> "CrossV5VGGTextureExtractor":
        try:
            from torchvision.models import VGG16_Weights, vgg16
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ImportError("Cross V5 VGG texture loss requires torchvision.") from exc

        if weights_path:
            model = vgg16(weights=None)
            state = torch.load(Path(weights_path).expanduser(), map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            if not isinstance(state, dict):
                raise ValueError(f"Unsupported VGG16 checkpoint format: {weights_path}")
            cleaned = {_strip_known_prefixes(str(key), ("module.", "_orig_mod.", "vgg.")): value for key, value in state.items()}
            model.load_state_dict(cleaned, strict=False)
        else:
            if not allow_download:
                raise ValueError(
                    "VGG texture loss requested but no --cross-v5-vgg-weights-path was provided "
                    "and VGG download is disabled."
                )
            model = vgg16(weights=VGG16_Weights.DEFAULT)

        layer_indices = _parse_vgg16_layers(layers)
        features = nn.Sequential(*list(model.features.children())[: max(layer_indices.values()) + 1])
        return cls(features, layer_indices=layer_indices)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"VGG texture extractor expects RGB BCHW, got {tuple(images.shape)}.")
        x = images.float().clamp(0.0, 1.0)
        x = (x - self.mean.to(device=x.device)) / self.std.to(device=x.device)
        outputs: dict[str, torch.Tensor] = {}
        capture_by_index = {index: name for name, index in self.layer_indices.items()}
        for index, layer in enumerate(self.features):
            x = layer(x)
            name = capture_by_index.get(index)
            if name is not None:
                outputs[name] = x
            if index >= self.max_layer:
                break
        return outputs


class CrossV5SegmentatorGeometryPredictor(nn.Module):
    """Frozen tissue geometry predictor backed by the local segmentator model."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.register_buffer(
            "mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.eval()
        self.requires_grad_(False)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        num_classes: int = 8,
        decoder: str = "mask2former",
        local_repo: str | Path = "UNI-2h",
        mask2former_queries: int = 100,
        mask2former_ignore_index: int = 255,
    ) -> "CrossV5SegmentatorGeometryPredictor":
        from segmentator.model import BaselineSegmenter

        resolved = _resolve_segmentator_checkpoint(checkpoint_path)
        model = BaselineSegmenter(
            num_classes=int(num_classes),
            freeze_encoder=True,
            local_repo=local_repo,
            decoder=str(decoder),
            mask2former_queries=int(mask2former_queries),
            mask2former_ignore_index=int(mask2former_ignore_index),
        )
        state = torch.load(resolved, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported segmentator checkpoint format: {resolved}")
        cleaned = {_strip_known_prefixes(str(key), ("module.", "_orig_mod.")): value for key, value in state.items()}
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Segmentator geometry checkpoint mismatch: "
                f"missing={list(missing)[:10]} unexpected={list(unexpected)[:10]}"
            )
        model.eval()
        model.requires_grad_(False)
        return cls(model)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Geometry predictor expects RGB BCHW, got {tuple(images.shape)}.")
        x = images.float().clamp(0.0, 1.0)
        x = (x - self.mean.to(device=x.device)) / self.std.to(device=x.device)
        outputs = self.model(x)
        if not isinstance(outputs, Mapping) or "logits" not in outputs:
            raise TypeError("Segmentator geometry predictor must return a mapping with 'logits'.")
        logits = outputs["logits"]
        return {
            "tissue_logits": logits,
        }


def _parse_vgg16_layers(layers: str) -> dict[str, int]:
    names = [part.strip() for part in str(layers or "").split(",") if part.strip()]
    if not names:
        raise ValueError("VGG layer list cannot be empty.")
    parsed: dict[str, int] = {}
    for name in names:
        if name.isdigit():
            parsed[f"layer_{name}"] = int(name)
            continue
        if name not in _VGG16_LAYER_ALIASES:
            raise ValueError(
                f"Unsupported VGG16 layer {name!r}; expected one of {sorted(_VGG16_LAYER_ALIASES)}."
            )
        parsed[name] = _VGG16_LAYER_ALIASES[name]
    return parsed


def _resolve_segmentator_checkpoint(path: str | Path) -> Path:
    checkpoint = Path(path).expanduser()
    if checkpoint.is_file():
        return checkpoint
    if checkpoint.is_dir():
        candidates = (
            checkpoint / "checkpoint_best.pt",
            checkpoint / "checkpoint_last.pt",
            checkpoint / "best.pt",
            checkpoint / "model.pt",
            checkpoint / "pytorch_model.bin",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Segmentator checkpoint not found: {checkpoint}")


def _strip_known_prefixes(key: str, prefixes: tuple[str, ...]) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


__all__ = [
    "CrossV5SegmentatorGeometryPredictor",
    "CrossV5VGGTextureExtractor",
]

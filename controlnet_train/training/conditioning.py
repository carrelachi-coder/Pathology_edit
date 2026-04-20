"""Channel specs and ControlNet projection helpers for Phase 5 training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def packed_control_channels(raw_control_channels: int) -> int:
    if raw_control_channels <= 0:
        raise ValueError(f"raw_control_channels must be positive, got {raw_control_channels}.")
    return raw_control_channels * 4


@dataclass(frozen=True)
class InpaintControlSpec:
    image_latent_channels: int = 16
    tissue_channels: int = 64
    nuclei_channels: int = 16
    change_channels: int = 4

    @property
    def raw_channels(self) -> int:
        return (
            self.image_latent_channels
            + self.tissue_channels
            + self.nuclei_channels
            + self.change_channels
        )

    @property
    def packed_channels(self) -> int:
        return packed_control_channels(self.raw_channels)


@dataclass(frozen=True)
class CrossV0ControlSpec:
    image_latent_channels: int = 16
    tissue_channels: int = 64
    nuclei_channels: int = 16

    @property
    def raw_channels(self) -> int:
        return self.image_latent_channels + self.tissue_channels + self.nuclei_channels + self.tissue_channels + self.nuclei_channels

    @property
    def packed_channels(self) -> int:
        return packed_control_channels(self.raw_channels)


def patch_controlnet_x_embedder(controlnet: nn.Module, packed_control_channels: int) -> nn.Module:
    if not hasattr(controlnet, "controlnet_x_embedder"):
        raise AttributeError("controlnet must expose a controlnet_x_embedder module.")

    old_x_embedder = controlnet.controlnet_x_embedder
    if not isinstance(old_x_embedder, nn.Linear):
        raise TypeError(
            "controlnet_x_embedder must be an nn.Linear for Phase 5 width patching, "
            f"got {type(old_x_embedder)!r}."
        )

    if packed_control_channels <= 0:
        raise ValueError(
            f"packed_control_channels must be positive, got {packed_control_channels}."
        )

    if old_x_embedder.in_features == packed_control_channels:
        return controlnet

    new_x_embedder = nn.Linear(packed_control_channels, old_x_embedder.out_features)
    with torch.no_grad():
        new_x_embedder.weight.zero_()
        copy_width = min(old_x_embedder.in_features, packed_control_channels)
        new_x_embedder.weight[:, :copy_width] = old_x_embedder.weight[:, :copy_width]
        if old_x_embedder.bias is not None:
            new_x_embedder.bias.copy_(old_x_embedder.bias)

    controlnet.controlnet_x_embedder = new_x_embedder
    return controlnet

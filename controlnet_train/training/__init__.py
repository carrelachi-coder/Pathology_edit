"""Phase 5 training helpers built on top of the official Flux ControlNet flow."""

from .conditioning import InpaintControlSpec, patch_controlnet_x_embedder

__all__ = [
    "InpaintControlSpec",
    "patch_controlnet_x_embedder",
]

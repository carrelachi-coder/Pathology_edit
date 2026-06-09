"""UNI2-h reference image encoder with Perceiver resampler for Cross V1 IP-Adapter."""

from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


UNI2H_KWARGS = {
    "model_name": "vit_giant_patch14_224",
    "img_size": 224,
    "patch_size": 14,
    "depth": 24,
    "num_heads": 24,
    "init_values": 1e-5,
    "embed_dim": 1536,
    "mlp_ratio": 2.66667 * 2,
    "num_classes": 0,
    "no_embed_class": True,
    "mlp_layer": timm.layers.SwiGLUPacked,
    "act_layer": torch.nn.SiLU,
    "reg_tokens": 8,
    "dynamic_img_size": True,
}


class ReferenceImageEncoder(nn.Module):
    """Encodes reference image via frozen UNI2-h ViT -> projection MLP -> Perceiver resampler.

    The Perceiver resampler compresses UNI2-h's spatial tokens into a smaller set of
    appearance tokens. This filters out spatial layout and retains appearance semantics
    (staining/texture), which is what we want for cross-reference injection.
    """

    def __init__(
        self,
        uni_checkpoint_path: str | Path,
        uni_embed_dim: int = 1536,
        hidden_dim: int = 3072,
        num_tokens: int = 16,
        num_perceiver_layers: int = 2,
        perceiver_heads: int = 8,
        use_perceiver_self_attn: bool = True,
        perceiver_cross_gate_init: float | None = None,
        skip_perceiver: bool = False,
    ):
        super().__init__()
        self.uni_embed_dim = int(uni_embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_tokens = int(num_tokens)
        self.num_perceiver_layers = int(num_perceiver_layers)
        self.perceiver_heads = int(perceiver_heads)
        self.use_perceiver_self_attn = bool(use_perceiver_self_attn)
        self.skip_perceiver = bool(skip_perceiver)
        self.perceiver_cross_gate_init = (
            None if perceiver_cross_gate_init is None else float(perceiver_cross_gate_init)
        )
        self.uni = self._load_uni(uni_checkpoint_path)
        self.uni.requires_grad_(False)
        self.uni.eval()

        mean_vals = (0.485, 0.456, 0.406)
        std_vals = (0.229, 0.224, 0.225)
        self.register_buffer("mean", torch.tensor(mean_vals).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std_vals).view(1, 3, 1, 1))

        self.proj_mlp = nn.Sequential(
            nn.Linear(uni_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.latent_queries = nn.Parameter(
            torch.randn(1, num_tokens, hidden_dim) * 0.02
        )
        self.perceiver_layers = nn.ModuleList([
            PerceiverCrossAttentionLayer(
                latent_dim=hidden_dim,
                input_dim=hidden_dim,
                num_heads=perceiver_heads,
                use_self_attn=self.use_perceiver_self_attn,
                cross_gate_init=self.perceiver_cross_gate_init,
            )
            for _ in range(num_perceiver_layers)
        ])
        self.perceiver_norm = nn.LayerNorm(hidden_dim)

    @property
    def num_output_tokens(self) -> int:
        if self.skip_perceiver:
            return self.num_spatial_tokens
        return self.num_tokens

    @property
    def num_spatial_tokens(self) -> int:
        patch_size = int(UNI2H_KWARGS["patch_size"])
        image_size = int(UNI2H_KWARGS["img_size"])
        return (image_size // patch_size) * (image_size // patch_size)

    def _load_uni(self, checkpoint_path: str | Path):
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else -1
        print(f"[rank {rank}] >>> BEFORE torch.load UNI: {checkpoint_path}", flush=True)

        model = timm.create_model(**UNI2H_KWARGS, pretrained=False)
        state_dict = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

        print(f"[rank {rank}] >>> AFTER torch.load, num_keys={len(state_dict)}", flush=True)

        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"UNI2-h checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )

        print(f"[rank {rank}] >>> UNI model loaded successfully", flush=True)
        return model

    def extract_uni_features(
        self,
        images: torch.Tensor,
        *,
        allow_input_grad: bool = False,
    ) -> torch.Tensor:
        def _extract() -> torch.Tensor:
            x = F.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False)
            x = (x - self.mean) / self.std
            features = self.uni.forward_features(x)
            if features.ndim == 3:
                patch_size = int(UNI2H_KWARGS["patch_size"])
                num_patch_tokens = (x.shape[-2] // patch_size) * (x.shape[-1] // patch_size)
                if features.shape[1] > num_patch_tokens:
                    features = features[:, -num_patch_tokens:, :]
            return features

        if allow_input_grad:
            return _extract()
        with torch.no_grad():
            return _extract()

    def _resample(self, projected: torch.Tensor) -> torch.Tensor:
        B = projected.shape[0]
        latents = self.latent_queries.expand(B, -1, -1)
        for layer in self.perceiver_layers:
            latents = layer(latents, projected)
        return self.perceiver_norm(latents)

    def load_perceiver_layers_state_dict(
        self, state_dict: dict[str, torch.Tensor]
    ) -> None:
        incompatible = self.perceiver_layers.load_state_dict(state_dict, strict=False)
        allowed_missing = set()
        if self.perceiver_cross_gate_init is not None:
            allowed_missing = {
                f"{index}.cross_gate" for index in range(len(self.perceiver_layers))
            }
        missing = [
            key for key in incompatible.missing_keys if key not in allowed_missing
        ]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Reference Perceiver checkpoint mismatch: "
                f"missing={missing}, unexpected={incompatible.unexpected_keys}"
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        projected = self.encode_projected_patch_tokens(images)
        if self.skip_perceiver:
            return projected
        resampled = self._resample(projected)
        return resampled

    def encode_projected_patch_tokens(
        self,
        images: torch.Tensor,
        *,
        allow_input_grad: bool = False,
    ) -> torch.Tensor:
        """Return projected UNI spatial patch tokens before global Perceiver pooling."""
        uni_features = self.extract_uni_features(
            images,
            allow_input_grad=allow_input_grad,
        )
        return self.proj_mlp(uni_features)

    def encode_region_ip_tokens(
        self,
        images: torch.Tensor,
        region_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return spatial IP tokens plus one tissue-region label per token."""
        projected = self.encode_projected_patch_tokens(images)
        labels = resize_mask_to_token_labels(region_mask, projected.shape[1])
        return projected, labels.to(device=projected.device)


class PerceiverCrossAttentionLayer(nn.Module):
    """Single Perceiver cross-attention layer: latent queries attend to input tokens."""

    def __init__(
        self,
        latent_dim: int,
        input_dim: int,
        num_heads: int,
        use_self_attn: bool = True,
        cross_gate_init: float | None = None,
    ):
        super().__init__()
        self.use_self_attn = bool(use_self_attn)
        if cross_gate_init is None:
            self.register_parameter("cross_gate", None)
        else:
            self.cross_gate = nn.Parameter(torch.tensor(float(cross_gate_init)))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=num_heads, batch_first=True,
            kdim=input_dim, vdim=input_dim,
        )
        # Keep these modules even when disabled so old/new checkpoints share keys.
        self.self_attn = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=num_heads, batch_first=True,
        )
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.self_attn_norm = nn.LayerNorm(latent_dim)
        self.input_norm = nn.LayerNorm(input_dim)
        self.ff_norm = nn.LayerNorm(latent_dim)
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Linear(latent_dim * 4, latent_dim),
        )

    def forward(self, latents: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        normed_latents = self.latent_norm(latents)
        normed_inputs = self.input_norm(inputs)
        cross_out, _ = self.cross_attn(
            query=normed_latents, key=normed_inputs, value=normed_inputs,
        )
        if self.cross_gate is None:
            latents = latents + cross_out
        else:
            gate = torch.sigmoid(self.cross_gate).to(dtype=latents.dtype)
            latents = gate * latents + (1.0 - gate) * cross_out
        if self.use_self_attn:
            normed_latents2 = self.self_attn_norm(latents)
            self_out, _ = self.self_attn(
                query=normed_latents2, key=normed_latents2, value=normed_latents2,
            )
            latents = latents + self_out
        latents = latents + self.ff(self.ff_norm(latents))
        return latents


def resize_mask_to_token_labels(mask: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Nearest-resize a BHW/B1HW mask to the square token grid and flatten to BTN labels."""
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 3:
        raise ValueError(f"mask must have shape (B,H,W) or (B,1,H,W), got {tuple(mask.shape)}")
    grid_h, grid_w = _infer_square_token_grid(int(num_tokens))
    labels = F.interpolate(
        mask.unsqueeze(1).float(),
        size=(grid_h, grid_w),
        mode="nearest",
    )[:, 0]
    return labels.to(dtype=torch.long).flatten(1)


def _infer_square_token_grid(num_tokens: int) -> tuple[int, int]:
    side = int(round(float(num_tokens) ** 0.5))
    if side * side != int(num_tokens):
        raise ValueError(f"expected a square spatial token grid, got num_tokens={num_tokens}")
    return side, side

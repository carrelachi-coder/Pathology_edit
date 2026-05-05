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
    ):
        super().__init__()
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
            )
            for _ in range(num_perceiver_layers)
        ])
        self.perceiver_norm = nn.LayerNorm(hidden_dim)

    def _load_uni(self, checkpoint_path: str | Path):
        model = timm.create_model(**UNI2H_KWARGS, pretrained=False)
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"UNI2-h checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
        return model

    @torch.no_grad()
    def extract_uni_features(self, images: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        features = self.uni.forward_features(x)
        return features

    def _resample(self, projected: torch.Tensor) -> torch.Tensor:
        B = projected.shape[0]
        latents = self.latent_queries.expand(B, -1, -1)
        for layer in self.perceiver_layers:
            latents = layer(latents, projected)
        return self.perceiver_norm(latents)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        uni_features = self.extract_uni_features(images)
        projected = self.proj_mlp(uni_features)
        resampled = self._resample(projected)
        return resampled


class PerceiverCrossAttentionLayer(nn.Module):
    """Single Perceiver cross-attention layer: latent queries attend to input tokens."""

    def __init__(self, latent_dim: int, input_dim: int, num_heads: int):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=num_heads, batch_first=True,
            kdim=input_dim, vdim=input_dim,
        )
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
        latents = latents + cross_out
        normed_latents2 = self.self_attn_norm(latents)
        self_out, _ = self.self_attn(
            query=normed_latents2, key=normed_latents2, value=normed_latents2,
        )
        latents = latents + self_out
        latents = latents + self.ff(self.ff_norm(latents))
        return latents
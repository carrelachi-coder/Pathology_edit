from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from controlnet_train.modules.pos_embedding import RoPE, get_1d_sincos_pos_embed, get_2d_sincos_pos_embed


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding, supports arbitrary input channels."""
    def __init__(self, img_size: int = 64, patch_size: int = 2, in_chans: int = 32, embed_dim: int = 384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, E, H/P, W/P)
        x = rearrange(x, "b e h w -> b (h w) e")
        return x


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None, act_layer: type[nn.Module] = nn.GELU, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 6, qkv_bias: bool = False, qk_norm: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = RoPE(dim=self.head_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, H, N, D)

        # Apply RoPE
        q = self.rope(q)
        k = self.rope(k)

        # QK norm (MuPaD default)
        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -torch.inf)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RefCrossAttention(nn.Module):
    """
    Independent Decoupled Cross-Attention (DCA, MuPaD style) for reference image tokens.
    - 100% independent K/V projection for ref tokens, no weight sharing with self-attention
    - QK norm for training stability
    - Learnable modality scale to adjust reference influence strength
    - No mask, fully free attention between latent tokens and ref tokens
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ref_dim: int = 1024,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Fully independent projections: no sharing with self-attention weights
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(ref_dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(ref_dim, dim, bias=qkv_bias)

        # QK norm (MuPaD default)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = RoPE(dim=self.head_dim)

        # Learnable modality scale (MuPaD DCA feature)
        self.modality_scale = nn.Parameter(torch.tensor(init_scale))

    def forward(self, x: torch.Tensor, ref_tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, C = x.shape
        _, M, _ = ref_tokens.shape

        # Independent Q from latent, K/V from ref
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, N, D)
        k = self.k_proj(ref_tokens).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, M, D)
        v = self.v_proj(ref_tokens).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, M, D)

        # RoPE on Q only (ref tokens have their own pos embed)
        q = self.rope(q)

        # QK norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -torch.inf)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Apply modality scale
        x = x * self.modality_scale

        return x


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        has_ref_cross_attn: bool = False,
        ref_dim: int = 1024,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )

        # Optional independent reference cross-attention (DCA, MuPaD style)
        self.has_ref_cross_attn = has_ref_cross_attn
        self.ref_cross_attn: RefCrossAttention | None = None
        self.norm_ref: nn.LayerNorm | None = None
        self.adaLN_modulation_ref: nn.Sequential | None = None
        if self.has_ref_cross_attn:
            self.norm_ref = norm_layer(dim, elementwise_affine=False, eps=1e-6)
            self.adaLN_modulation_ref = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, 3 * dim, bias=True)
            )
            self.ref_cross_attn = RefCrossAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                ref_dim=ref_dim,
            )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm3 = norm_layer(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, ref_tokens: torch.Tensor | None = None) -> torch.Tensor:
        # AdaLN modulation for self-attention
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(t).chunk(6, dim=-1)
        )
        # Self-Attention
        x = x + gate_msa.unsqueeze(1) * self.attn(
            self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        )

        # Reference Decoupled Cross-Attention (if enabled)
        if self.has_ref_cross_attn and ref_tokens is not None and self.ref_cross_attn is not None:
            shift_ref, scale_ref, gate_ref = (self.adaLN_modulation_ref(t).chunk(3, dim=-1))
            x = x + gate_ref.unsqueeze(1) * self.ref_cross_attn(
                self.norm_ref(x) * (1 + scale_ref.unsqueeze(1)) + shift_ref.unsqueeze(1),
                ref_tokens
            )

        # Feed-Forward Network
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            self.norm3(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        )

        return x


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def timestep_embedding(self, t: torch.Tensor, max_period: float = 10000) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t)
        t_emb = self.mlp(t_freq)
        return t_emb


class Pix2PixV2DiT(nn.Module):
    """
    Small DiT backbone for pix2pix V2 refinement, fully aligned with MuPaD design.
    Core features:
    1. Input: z_t (16ch) + I0 latent (16ch) concat = 32ch total, no I0 erasing
    2. Independent Decoupled Cross-Attention (DCA) for reference, added to last 6 layers
    3. Output: velocity prediction (16ch), flow-matching objective, aligned with FLUX
    4. AdaLN-zero modulation, consistent with modern DiT designs
    """
    def __init__(
        self,
        latent_size: int = 64,  # FLUX VAE 8x downsample
        patch_size: int = 2,
        in_channels: int = 32,  # 16 z_t + 16 I0 latent
        out_channels: int = 16,  # velocity prediction
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        ref_cross_attn_start_layer: int = 6,  # start adding ref cross-attn from layer 6 (last 6 layers, MuPaD default)
        ref_token_dim: int = 1024,  # UNI/Virchow2 patch token dim
        num_classes: int | None = None,  # optional class condition
        qkv_bias: bool = True,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.ref_cross_attn_start_layer = ref_cross_attn_start_layer

        # Input patch embedding
        self.patch_embed = PatchEmbed(img_size=latent_size, patch_size=patch_size, in_chans=in_channels, embed_dim=hidden_size)
        self.num_patches = self.patch_embed.num_patches

        # Positional embedding (fixed sincos, MuPaD default)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        # Timestep embedding
        self.time_embed = TimestepEmbedder(hidden_size)

        # Optional class embedding
        self.class_embedding: nn.Embedding | None = None
        if num_classes is not None:
            self.class_embedding = nn.Embedding(num_classes, hidden_size)

        # DiT blocks: last `depth - ref_cross_attn_start_layer` blocks have ref DCA
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                has_ref_cross_attn=(i >= ref_cross_attn_start_layer),
                ref_dim=ref_token_dim,
            ) for i in range(depth)
        ])

        # Output head (zero initialized, MuPaD default)
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.head = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self) -> None:
        # Initialize pos embed as fixed 2D sincos
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch embedding
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.zeros_(self.patch_embed.proj.bias)

        # Initialize timestep embedding
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.time_embed.mlp[0].bias)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.time_embed.mlp[2].bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, P*P*C) -> (B, C, H, W)"""
        p = self.patch_size
        h = w = int(self.num_patches ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.out_channels))
        x = torch.einsum("bhwpqc->bchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], self.out_channels, h * p, w * p))
        return imgs

    def forward(self, z_t: torch.Tensor, i0_latent: torch.Tensor, t: torch.Tensor, ref_tokens: torch.Tensor, class_label: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            z_t: (B, 16, L, L) noised latent at timestep t
            i0_latent: (B, 16, L, L) I0 latent (fixed, concat to z_t, full structure preserved)
            t: (B,) timestep in [0, 1]
            ref_tokens: (B, M, D) reference image patch tokens (from UNI/Virchow2, stain-agnostic)
            class_label: (B,) optional class label
        Returns:
            v: (B, 16, L, L) predicted velocity for flow-matching
        """
        # Concat z_t and I0 latent: 16 + 16 = 32 channels, no modification to I0
        x = torch.cat([z_t, i0_latent], dim=1)  # (B, 32, L, L)

        # Patch embedding + fixed pos embed
        x = self.patch_embed(x)  # (B, N, C)
        x = x + self.pos_embed

        # Timestep embedding, broadcast to sequence length
        t_emb = self.time_embed(t)
        t_emb = repeat(t_emb, "b c -> b n c", n=x.shape[1])
        x = x + t_emb

        # Optional class embedding
        if self.class_embedding is not None and class_label is not None:
            class_emb = self.class_embedding(class_label)
            class_emb = repeat(class_emb, "b c -> b n c", n=x.shape[1])
            x = x + class_emb

        # DiT blocks
        for block in self.blocks:
            x = block(x, t_emb[:, 0, :], ref_tokens=ref_tokens)

        # Output head
        x = self.norm_final(x)
        x = self.head(x)
        x = self.unpatchify(x)

        return x

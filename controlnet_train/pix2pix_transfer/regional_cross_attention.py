"""UNet generator with mask-guided regional cross attention."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .identity_adapter import FamilyWSIIdentityAdapter
from .reference_augmentation import rotate_reference_bundle


NUCLEI_LABEL_OFFSET = 256


def downsample_label_grid(label_map: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    labels = F.interpolate(label_map.float(), size=size, mode="nearest")
    return labels[:, 0].long()


def downsample_label_map(label_map: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    labels = downsample_label_grid(label_map, size)
    return labels.flatten(1)


def scale_pixel_radius(
    radius: int,
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> int:
    radius = int(radius)
    if radius <= 0:
        return 0
    scale_y = float(target_size[0]) / max(1.0, float(source_size[0]))
    scale_x = float(target_size[1]) / max(1.0, float(source_size[1]))
    return max(1, int(math.ceil(float(radius) * max(scale_y, scale_x))))


def build_local_label_context_mask(
    query_label_grid: torch.Tensor,
    key_labels: torch.Tensor,
    *,
    radius: int,
) -> torch.Tensor:
    """Allow reference labels that occur near each query token in target layout."""

    if query_label_grid.ndim != 3:
        raise ValueError(f"query_label_grid must be BxHxW, got {tuple(query_label_grid.shape)}")
    if key_labels.ndim != 2:
        raise ValueError(f"key_labels must be BxN, got {tuple(key_labels.shape)}")
    b, h, w = query_label_grid.shape
    if key_labels.shape[0] != b:
        raise ValueError("query_label_grid and key_labels batch size must match")
    radius = int(radius)
    n_q = h * w
    n_k = key_labels.shape[1]
    context = torch.zeros((b, n_q, n_k), dtype=torch.bool, device=query_label_grid.device)
    if radius <= 0:
        return context

    kernel = radius * 2 + 1
    for batch_index in range(b):
        for label in torch.unique(key_labels[batch_index]).tolist():
            key_mask = key_labels[batch_index].eq(int(label))
            support = query_label_grid[batch_index].eq(int(label)).float().view(1, 1, h, w)
            if not bool(support.any().item()):
                continue
            nearby = F.max_pool2d(support, kernel_size=kernel, stride=1, padding=radius)
            nearby = nearby.view(n_q).bool()
            if bool(nearby.any().item()):
                context[batch_index] |= nearby[:, None] & key_mask[None, :]
    return context


def build_region_attention_mask(
    query_labels: torch.Tensor,
    key_labels: torch.Tensor,
) -> torch.Tensor:
    allow, _ = build_region_attention_mask_and_strength(query_labels, key_labels)
    return allow


def _clamped_scale(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def build_label_family_compatibility(
    query_labels: torch.Tensor,
    key_labels: torch.Tensor,
) -> torch.Tensor:
    query_is_nuclei = query_labels.ge(NUCLEI_LABEL_OFFSET)
    key_is_nuclei = key_labels.ge(NUCLEI_LABEL_OFFSET)
    return query_is_nuclei.unsqueeze(2) == key_is_nuclei.unsqueeze(1)


def build_region_attention_mask_and_strength(
    query_labels: torch.Tensor,
    key_labels: torch.Tensor,
    *,
    fallback_scale: float = 1.0,
    query_trust: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build exact region attention and per-query reference strength.

    Matched query tokens attend only same-label reference tokens.  Unmatched
    query tokens may attend all reference tokens, but their update is scaled by
    ``fallback_scale`` so I0 remains dominant.
    """

    allow_exact = query_labels.unsqueeze(2) == key_labels.unsqueeze(1)
    has_match = allow_exact.any(dim=2)
    family_compatible = build_label_family_compatibility(query_labels, key_labels)
    has_family_fallback = family_compatible.any(dim=2)
    fallback_allow = torch.where(
        has_family_fallback.unsqueeze(2),
        family_compatible,
        torch.ones_like(family_compatible),
    )
    allow = torch.where(has_match.unsqueeze(2), allow_exact, fallback_allow)
    fallback = float(max(0.0, min(1.0, fallback_scale)))
    strength = torch.where(
        has_match,
        torch.ones_like(query_labels, dtype=torch.float32),
        torch.full_like(query_labels, fallback, dtype=torch.float32),
    )
    strength = torch.where(
        (~has_match) & (~has_family_fallback),
        torch.zeros_like(strength),
        strength,
    )
    if query_trust is not None:
        trust = query_trust.to(device=strength.device, dtype=strength.dtype)
        if trust.shape != strength.shape:
            raise ValueError(f"query_trust must match query labels, got {tuple(trust.shape)} vs {tuple(strength.shape)}")
        strength = torch.minimum(strength, trust.clamp(0.0, 1.0))
    return allow, strength


def build_region_attention_bias_and_strength(
    query_labels: torch.Tensor,
    key_labels: torch.Tensor,
    *,
    fallback_scale: float = 1.0,
    query_trust: torch.Tensor | None = None,
    query_label_grid: torch.Tensor | None = None,
    soft_context_scale: float = 0.0,
    nuclei_context_scale: float = 0.0,
    soft_context_radius: int = 0,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build additive attention bias plus per-query reference update strength.

    Same-label reference tokens remain the high-confidence path.  Query tokens
    near a target-layout boundary can also see nearby semantic labels through a
    negative logit bias, so microenvironment context is visible without making
    it as strong as exact region matching.
    """

    allow_exact, strength = build_region_attention_mask_and_strength(
        query_labels,
        key_labels,
        fallback_scale=fallback_scale,
        query_trust=query_trust,
    )
    has_match = allow_exact.any(dim=2)
    very_negative = -torch.finfo(dtype).max
    bias = torch.full(allow_exact.shape, very_negative, dtype=dtype, device=query_labels.device)
    bias = bias.masked_fill(allow_exact | ~has_match.unsqueeze(2), 0.0)

    tissue_context_scale = _clamped_scale(soft_context_scale)
    nuclei_context_scale = _clamped_scale(nuclei_context_scale)
    if (
        query_label_grid is not None
        and int(soft_context_radius) > 0
        and (tissue_context_scale > 0.0 or nuclei_context_scale > 0.0)
    ):
        if query_label_grid.shape[0] != query_labels.shape[0]:
            raise ValueError("query_label_grid and query_labels batch size must match")
        if query_label_grid.numel() // query_label_grid.shape[0] != query_labels.shape[1]:
            raise ValueError("query_label_grid spatial size must match query_labels length")
        local_context = build_local_label_context_mask(
            query_label_grid,
            key_labels,
            radius=int(soft_context_radius),
        )
        family_compatible = build_label_family_compatibility(query_labels, key_labels)
        context = local_context & ~allow_exact & has_match.unsqueeze(2) & family_compatible
        query_is_nuclei = query_labels.ge(NUCLEI_LABEL_OFFSET)
        key_is_nuclei = key_labels.ge(NUCLEI_LABEL_OFFSET)
        nuclei_context = context & query_is_nuclei.unsqueeze(2) & key_is_nuclei.unsqueeze(1)
        tissue_context = context & ~query_is_nuclei.unsqueeze(2) & ~key_is_nuclei.unsqueeze(1)
        if tissue_context_scale > 0.0:
            bias = bias.masked_fill(tissue_context, math.log(tissue_context_scale))
        if nuclei_context_scale > 0.0:
            bias = bias.masked_fill(nuclei_context, math.log(nuclei_context_scale))
    return bias, strength


def downsample_strength_map(strength_map: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if strength_map.ndim == 3:
        strength_map = strength_map.unsqueeze(1)
    if strength_map.ndim != 4 or strength_map.shape[1] != 1:
        raise ValueError(f"strength_map must be Bx1xHxW or BxHxW, got {tuple(strength_map.shape)}")
    values = F.interpolate(strength_map.float(), size=size, mode="area")
    return values[:, 0].flatten(1).clamp(0.0, 1.0)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, groups: int = 8) -> None:
        super().__init__()
        group_count = min(groups, out_ch)
        while out_ch % group_count != 0:
            group_count -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(group_count, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(group_count, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.block = ConvBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x))


def normalize_upsample_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    aliases = {
        "bilinear": "bilinear",
        "linear": "bilinear",
        "nearest": "nearest",
        "nearest-exact": "nearest",
    }
    if value not in aliases:
        raise ValueError(f"upsample mode must be bilinear or nearest (got {mode!r})")
    return aliases[value]


def resize_feature(x: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
    if mode == "nearest":
        return F.interpolate(x, size=size, mode="nearest")
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        *,
        upsample_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        self.upsample_mode = normalize_upsample_mode(upsample_mode)
        self.up_proj = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.block = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = resize_feature(x, skip.shape[-2:], self.upsample_mode)
        x = self.up_proj(x)
        return self.block(torch.cat([x, skip], dim=1))


class RegionalCrossAttention(nn.Module):
    """Multi-head cross attention with optional region mask."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        use_region_mask: bool = True,
        reference_pool_size: int | None = None,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.use_region_mask = bool(use_region_mask)
        self.reference_pool_size = (
            None if reference_pool_size is None else max(1, int(reference_pool_size))
        )

        self.norm_q = nn.GroupNorm(1, dim)
        self.norm_kv = nn.GroupNorm(1, dim)
        self.to_q = nn.Conv2d(dim, dim, 1)
        self.to_k = nn.Conv2d(dim, dim, 1)
        self.to_v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(()))

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        return (
            x.view(b, self.num_heads, self.head_dim, h * w)
            .permute(0, 1, 3, 2)
            .contiguous()
        )

    def _attention_update(
        self,
        query_feat: torch.Tensor,
        reference_feat: torch.Tensor,
        *,
        query_region_map: torch.Tensor | None = None,
        reference_region_map: torch.Tensor | None = None,
        query_trust_map: torch.Tensor | None = None,
        ref_fallback_scale: float = 1.0,
        ref_soft_context_scale: float = 0.0,
        ref_nuclei_context_scale: float = 0.0,
        ref_soft_context_radius: int = 0,
    ) -> torch.Tensor:
        b, c, hq, wq = query_feat.shape
        if self.reference_pool_size is not None and (
            reference_feat.shape[-2] > self.reference_pool_size
            or reference_feat.shape[-1] > self.reference_pool_size
        ):
            reference_feat = F.adaptive_avg_pool2d(
                reference_feat,
                output_size=(self.reference_pool_size, self.reference_pool_size),
            )
        _, _, hk, wk = reference_feat.shape
        q = self._heads(self.to_q(self.norm_q(query_feat)))
        k = self._heads(self.to_k(self.norm_kv(reference_feat)))
        v = self._heads(self.to_v(self.norm_kv(reference_feat)))

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        ref_strength = None
        if (
            self.use_region_mask
            and query_region_map is not None
            and reference_region_map is not None
        ):
            query_label_grid = downsample_label_grid(query_region_map, (hq, wq))
            query_labels = query_label_grid.flatten(1)
            key_labels = downsample_label_map(reference_region_map, (hk, wk))
            query_trust = None
            if query_trust_map is not None:
                query_trust = downsample_strength_map(query_trust_map, (hq, wq))
            scaled_radius = scale_pixel_radius(
                int(ref_soft_context_radius),
                source_size=tuple(query_region_map.shape[-2:]),
                target_size=(hq, wq),
            )
            attention_bias, ref_strength = build_region_attention_bias_and_strength(
                query_labels,
                key_labels,
                fallback_scale=ref_fallback_scale,
                query_trust=query_trust,
                query_label_grid=query_label_grid,
                soft_context_scale=ref_soft_context_scale,
                nuclei_context_scale=ref_nuclei_context_scale,
                soft_context_radius=scaled_radius,
                dtype=logits.dtype,
            )
            logits = logits + attention_bias.unsqueeze(1)

        attn = logits.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(b, c, hq, wq)
        out = self.proj(out)
        if ref_strength is not None:
            out = out * ref_strength.to(device=out.device, dtype=out.dtype).view(b, 1, hq, wq)
        return out

    def forward(
        self,
        query_feat: torch.Tensor,
        reference_feat: torch.Tensor,
        *,
        query_region_map: torch.Tensor | None = None,
        reference_region_map: torch.Tensor | None = None,
        query_trust_map: torch.Tensor | None = None,
        ref_fallback_scale: float = 1.0,
        ref_soft_context_scale: float = 0.0,
        ref_nuclei_context_scale: float = 0.0,
        ref_soft_context_radius: int = 0,
    ) -> torch.Tensor:
        update = self._attention_update(
            query_feat,
            reference_feat,
            query_region_map=query_region_map,
            reference_region_map=reference_region_map,
            query_trust_map=query_trust_map,
            ref_fallback_scale=ref_fallback_scale,
            ref_soft_context_scale=ref_soft_context_scale,
            ref_nuclei_context_scale=ref_nuclei_context_scale,
            ref_soft_context_radius=ref_soft_context_radius,
        )
        return query_feat + self.gamma * update

    def forward_dual_reference(
        self,
        query_feat: torch.Tensor,
        reference_feat: torch.Tensor,
        reference_feat_rot90: torch.Tensor,
        *,
        rot90_gate: torch.Tensor,
        query_region_map: torch.Tensor | None = None,
        reference_region_map: torch.Tensor | None = None,
        reference_region_map_rot90: torch.Tensor | None = None,
        query_trust_map: torch.Tensor | None = None,
        ref_fallback_scale: float = 1.0,
        ref_soft_context_scale: float = 0.0,
        ref_nuclei_context_scale: float = 0.0,
        ref_soft_context_radius: int = 0,
    ) -> torch.Tensor:
        update = self._attention_update(
            query_feat,
            reference_feat,
            query_region_map=query_region_map,
            reference_region_map=reference_region_map,
            query_trust_map=query_trust_map,
            ref_fallback_scale=ref_fallback_scale,
            ref_soft_context_scale=ref_soft_context_scale,
            ref_nuclei_context_scale=ref_nuclei_context_scale,
            ref_soft_context_radius=ref_soft_context_radius,
        )
        gate = rot90_gate
        if gate.ndim == 3:
            gate = gate.unsqueeze(1)
        if gate.ndim != 4 or gate.shape[1] != 1 or gate.shape[0] != query_feat.shape[0]:
            raise ValueError("rot90_gate must have shape [B,1,H,W] or [B,H,W]")
        if tuple(gate.shape[-2:]) != tuple(query_feat.shape[-2:]):
            gate = F.interpolate(gate.float(), size=query_feat.shape[-2:], mode="bilinear", align_corners=False)
        gate = gate.to(device=query_feat.device, dtype=query_feat.dtype).clamp(0.0, 1.0)
        if not bool(gate.gt(0.0).any().item()):
            return query_feat + self.gamma * update
        update_rot90 = self._attention_update(
            query_feat,
            reference_feat_rot90,
            query_region_map=query_region_map,
            reference_region_map=reference_region_map_rot90,
            query_trust_map=query_trust_map,
            ref_fallback_scale=ref_fallback_scale,
            ref_soft_context_scale=ref_soft_context_scale,
            ref_nuclei_context_scale=ref_nuclei_context_scale,
            ref_soft_context_radius=ref_soft_context_radius,
        )
        blended_update = update * (1.0 - gate) + update_rot90 * gate
        return query_feat + self.gamma * blended_update

    def forward_multi_reference(
        self,
        query_feat: torch.Tensor,
        reference_features: Sequence[torch.Tensor],
        *,
        reference_weights: torch.Tensor,
        query_region_map: torch.Tensor | None = None,
        reference_region_maps: Sequence[torch.Tensor | None] | None = None,
        query_trust_map: torch.Tensor | None = None,
        ref_fallback_scale: float = 1.0,
        ref_soft_context_scale: float = 0.0,
        ref_nuclei_context_scale: float = 0.0,
        ref_soft_context_radius: int = 0,
        update_gain: float = 1.0,
    ) -> torch.Tensor:
        """Blend attention updates from a bank of rotated references per query location."""

        features = tuple(reference_features)
        if not features:
            raise ValueError("reference_features cannot be empty")
        if reference_region_maps is None:
            region_maps: tuple[torch.Tensor | None, ...] = (None,) * len(features)
        else:
            region_maps = tuple(reference_region_maps)
        if len(region_maps) != len(features):
            raise ValueError("reference_region_maps must match reference_features")
        weights = reference_weights
        if weights.ndim != 4 or weights.shape[:2] != (query_feat.shape[0], len(features)):
            raise ValueError(
                "reference_weights must have shape [B,K,H,W] matching the reference bank"
            )
        if tuple(weights.shape[-2:]) != tuple(query_feat.shape[-2:]):
            weights = F.interpolate(
                weights.float(),
                size=query_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        weights = weights.to(device=query_feat.device, dtype=query_feat.dtype).clamp_min(0.0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        blended_update = torch.zeros_like(query_feat)
        for index, (reference_feat, reference_region_map) in enumerate(
            zip(features, region_maps, strict=True)
        ):
            update = self._attention_update(
                query_feat,
                reference_feat,
                query_region_map=query_region_map,
                reference_region_map=reference_region_map,
                query_trust_map=query_trust_map,
                ref_fallback_scale=ref_fallback_scale,
                ref_soft_context_scale=ref_soft_context_scale,
                ref_nuclei_context_scale=ref_nuclei_context_scale,
                ref_soft_context_radius=ref_soft_context_radius,
            )
            blended_update = blended_update + update * weights[:, index : index + 1]
        return query_feat + (self.gamma * float(update_gain)) * blended_update


def normalize_cross_attn_scales(scales: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(scales, str):
        raw_values = [value.strip() for value in scales.split(",")]
    else:
        raw_values = [str(value).strip() for value in scales]
    aliases = {
        "4": "1/4",
        "1/4": "1/4",
        "quarter": "1/4",
        "8": "1/8",
        "1/8": "1/8",
        "eighth": "1/8",
        "mid": "1/8",
        "16": "1/16",
        "1/16": "1/16",
        "sixteenth": "1/16",
        "bottleneck": "1/16",
    }
    normalized: list[str] = []
    for value in raw_values:
        if not value:
            continue
        key = value.lower()
        if key not in aliases:
            raise ValueError(
                "cross-attn scale must be one of 1/4, 1/8, 1/16 "
                f"(got {value!r})"
            )
        scale = aliases[key]
        if scale not in normalized:
            normalized.append(scale)
    if not normalized:
        raise ValueError("At least one cross-attn scale is required.")
    return tuple(normalized)


def normalize_texture_steering_scales(
    scales: str | tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    if isinstance(scales, str):
        raw_values = [value.strip() for value in scales.split(",")]
    else:
        raw_values = [str(value).strip() for value in scales]
    aliases = {
        "1": "1/1",
        "1/1": "1/1",
        "full": "1/1",
        "2": "1/2",
        "1/2": "1/2",
        "half": "1/2",
        "4": "1/4",
        "1/4": "1/4",
        "quarter": "1/4",
        "8": "1/8",
        "1/8": "1/8",
        "eighth": "1/8",
        "16": "1/16",
        "1/16": "1/16",
        "bottleneck": "1/16",
    }
    normalized: list[str] = []
    for value in raw_values:
        if not value:
            continue
        key = value.lower()
        if key not in aliases:
            raise ValueError(
                "texture steering scale must be one of 1/1, 1/2, 1/4, 1/8, 1/16 "
                f"(got {value!r})"
            )
        scale = aliases[key]
        if scale not in normalized:
            normalized.append(scale)
    if not normalized:
        raise ValueError("At least one texture steering scale is required.")
    return tuple(normalized)


class Pix2PixCrossAttnUNet(nn.Module):
    """A compact supervised I0/reference -> target generator.

    target_cond contains I0 plus target masks. reference_cond contains reference
    image plus reference masks. Multi-scale cross-attention injects reference
    appearance into target-structure features, constrained by region labels.
    """

    def __init__(
        self,
        *,
        in_ch: int,
        out_ch: int = 3,
        base: int = 64,
        num_heads: int = 4,
        use_region_mask: bool = True,
        residual_output: bool = True,
        cross_attn_scales: str | tuple[str, ...] | list[str] = ("1/4", "1/8", "1/16"),
        upsample_mode: str = "bilinear",
        use_wsi_identity: bool = False,
        identity_gamma_max: float = 0.30,
        identity_gamma_init: float = 0.10,
        identity_min_tissue_pixels: int = 256,
        identity_min_nuclei_pixels: int = 64,
        full_pyramid_texture_steering: bool = False,
        steering_highres_reference_size: int = 8,
    ) -> None:
        super().__init__()
        self.residual_output = bool(residual_output)
        self.cross_attn_scales = normalize_cross_attn_scales(cross_attn_scales)
        self.upsample_mode = normalize_upsample_mode(upsample_mode)
        self.full_pyramid_texture_steering = bool(full_pyramid_texture_steering)

        self.target_in = ConvBlock(in_ch, base)
        self.target_down1 = DownBlock(base, base * 2)
        self.target_down2 = DownBlock(base * 2, base * 4)
        self.target_down3 = DownBlock(base * 4, base * 8)
        self.target_down4 = DownBlock(base * 8, base * 8)

        self.ref_in = ConvBlock(in_ch, base)
        self.ref_down1 = DownBlock(base, base * 2)
        self.ref_down2 = DownBlock(base * 2, base * 4)
        self.ref_down3 = DownBlock(base * 4, base * 8)
        self.ref_down4 = DownBlock(base * 8, base * 8)

        self.bottleneck = ConvBlock(base * 8, base * 8)
        self.cross_4 = (
            RegionalCrossAttention(base * 4, num_heads=num_heads, use_region_mask=use_region_mask)
            if "1/4" in self.cross_attn_scales
            else None
        )
        self.cross_8 = (
            RegionalCrossAttention(base * 8, num_heads=num_heads, use_region_mask=use_region_mask)
            if "1/8" in self.cross_attn_scales
            else None
        )
        self.cross_16 = (
            RegionalCrossAttention(base * 8, num_heads=num_heads, use_region_mask=use_region_mask)
            if "1/16" in self.cross_attn_scales
            else None
        )
        self.steering_cross_2 = None
        self.steering_cross_1 = None
        if self.full_pyramid_texture_steering:
            self.steering_cross_2 = RegionalCrossAttention(
                base * 2,
                num_heads=num_heads,
                use_region_mask=use_region_mask,
                reference_pool_size=steering_highres_reference_size,
            )
            self.steering_cross_1 = RegionalCrossAttention(
                base,
                num_heads=num_heads,
                use_region_mask=use_region_mask,
                reference_pool_size=steering_highres_reference_size,
            )
            # Exact legacy output at initialization, but non-zero gamma lets the
            # zero projection receive gradients immediately after old weights load.
            for module in (self.steering_cross_2, self.steering_cross_1):
                nn.init.zeros_(module.proj.weight)
                nn.init.zeros_(module.proj.bias)
                module.gamma.data.fill_(0.10)
        self.identity_adapter = (
            FamilyWSIIdentityAdapter(
                channels_by_scale={
                    "1/4": base * 4,
                    "1/8": base * 8,
                    "1/16": base * 8,
                },
                tissue_scales=("1/4", "1/8", "1/16"),
                nuclei_scales=("1/4",),
                gamma_max=identity_gamma_max,
                gamma_init=identity_gamma_init,
                min_tissue_pixels=identity_min_tissue_pixels,
                min_nuclei_pixels=identity_min_nuclei_pixels,
            )
            if use_wsi_identity
            else None
        )

        self.up3 = UpBlock(base * 8, base * 8, base * 8, upsample_mode=self.upsample_mode)
        self.up2 = UpBlock(base * 8, base * 4, base * 4, upsample_mode=self.upsample_mode)
        self.up1 = UpBlock(base * 4, base * 2, base * 2, upsample_mode=self.upsample_mode)
        self.up0 = UpBlock(base * 2, base, base, upsample_mode=self.upsample_mode)
        self.out = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base, out_ch, 1),
        )

    def _encode_target(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        e0 = self.target_in(x)
        e1 = self.target_down1(e0)
        e2 = self.target_down2(e1)
        e3 = self.target_down3(e2)
        e4 = self.target_down4(e3)
        return e0, e1, e2, e3, e4

    def _encode_reference(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        r0 = self.ref_in(x)
        r1 = self.ref_down1(r0)
        r2 = self.ref_down2(r1)
        r3 = self.ref_down3(r2)
        r4 = self.ref_down4(r3)
        return r0, r1, r2, r3, r4

    def forward(
        self,
        target_cond: torch.Tensor,
        reference_cond: torch.Tensor,
        *,
        target_region: torch.Tensor | None = None,
        reference_region: torch.Tensor | None = None,
        target_trust_map: torch.Tensor | None = None,
        highres_nuclei_trust_map: torch.Tensor | None = None,
        ref_fallback_scale: float = 1.0,
        ref_soft_context_scale: float = 0.0,
        ref_nuclei_context_scale: float = 0.0,
        ref_soft_context_radius: int = 0,
        target_tissue_mask: torch.Tensor | None = None,
        target_nuclei_mask: torch.Tensor | None = None,
        reference_tissue_mask: torch.Tensor | None = None,
        reference_nuclei_mask: torch.Tensor | None = None,
        cross4_rot90_gate: torch.Tensor | None = None,
        cross4_rotation_weights: torch.Tensor | None = None,
        cross4_rotation_angles: Sequence[float] = (0.0, 45.0, 90.0, 135.0),
        texture_steering_scales: Sequence[str] = ("1/4",),
        cross4_steering_gain: float = 1.0,
        cross8_steering_gain: float = 1.0,
        cross16_steering_gain: float = 1.0,
        cross2_steering_gain: float = 1.0,
        cross1_steering_gain: float = 1.0,
    ) -> torch.Tensor:
        e0, e1, e2, e3, e4 = self._encode_target(target_cond)
        r0, r1, r2, r3, r4 = self._encode_reference(reference_cond)
        highres_trust_map = target_trust_map
        if highres_nuclei_trust_map is not None:
            if highres_trust_map is None:
                highres_trust_map = highres_nuclei_trust_map
            else:
                if tuple(highres_trust_map.shape) != tuple(highres_nuclei_trust_map.shape):
                    raise ValueError(
                        "target_trust_map and highres_nuclei_trust_map must have identical shapes"
                    )
                highres_trust_map = torch.minimum(
                    highres_trust_map.float(), highres_nuclei_trust_map.float()
                )
        steered_features: dict[str, list[torch.Tensor]] = {}
        steered_region_maps: list[torch.Tensor | None] | None = None
        steering_scales: set[str] = set()
        if cross4_rot90_gate is not None and cross4_rotation_weights is not None:
            raise ValueError("cross4_rot90_gate and cross4_rotation_weights are mutually exclusive")
        if cross4_rotation_weights is not None:
            angles = tuple(float(value) for value in cross4_rotation_angles)
            if not angles or abs(angles[0]) > 1.0e-6:
                raise ValueError("cross4_rotation_angles must begin with 0 degrees")
            if cross4_rotation_weights.shape[1] != len(angles):
                raise ValueError(
                    "cross4_rotation_weights channel count must match cross4_rotation_angles"
                )
            steering_scales = set(normalize_texture_steering_scales(texture_steering_scales))
            if not self.full_pyramid_texture_steering and steering_scales.difference(
                {"1/4", "1/8"}
            ):
                raise ValueError(
                    "1/1, 1/2, and 1/16 steering require full_pyramid_texture_steering"
                )
            unavailable = steering_scales.intersection({"1/4", "1/8", "1/16"}).difference(
                self.cross_attn_scales
            )
            if unavailable:
                raise ValueError(
                    f"texture steering scales are disabled in cross_attn_scales: {sorted(unavailable)}"
                )
            if any(
                value is None
                for value in (
                    reference_region,
                    reference_tissue_mask,
                    reference_nuclei_mask,
                )
            ):
                raise ValueError(
                    "steered texture attention requires reference region, tissue, and nuclei masks"
                )
            all_features = {
                "1/1": r0,
                "1/2": r1,
                "1/4": r2,
                "1/8": r3,
                "1/16": r4,
            }
            steered_features = {
                scale: [all_features[scale]] for scale in steering_scales
            }
            steered_region_maps = [reference_region]
            for angle in angles[1:]:
                rotated = rotate_reference_bundle(
                    reference_cond,
                    reference_region,
                    reference_tissue_mask,
                    reference_nuclei_mask,
                    angles_degrees=angle,
                )
                rotated_r0 = self.ref_in(rotated.reference_cond)
                rotated_r1 = self.ref_down1(rotated_r0)
                rotated_r2 = self.ref_down2(rotated_r1)
                rotated_r3 = (
                    self.ref_down3(rotated_r2)
                    if steering_scales.intersection({"1/8", "1/16"})
                    else None
                )
                rotated_r4 = (
                    self.ref_down4(rotated_r3)
                    if "1/16" in steering_scales and rotated_r3 is not None
                    else None
                )
                rotated_features = {
                    "1/1": rotated_r0,
                    "1/2": rotated_r1,
                    "1/4": rotated_r2,
                    "1/8": rotated_r3,
                    "1/16": rotated_r4,
                }
                for scale in steering_scales:
                    feature = rotated_features[scale]
                    if feature is None:
                        raise RuntimeError(f"failed to build rotated reference feature at {scale}")
                    steered_features[scale].append(feature)
                steered_region_maps.append(rotated.reference_region)
        z = self.bottleneck(e4)
        if self.cross_16 is not None:
            if cross4_rotation_weights is not None and "1/16" in steering_scales:
                assert steered_region_maps is not None
                z = self.cross_16.forward_multi_reference(
                    z,
                    steered_features["1/16"],
                    reference_weights=cross4_rotation_weights,
                    query_region_map=target_region,
                    reference_region_maps=steered_region_maps,
                    query_trust_map=target_trust_map,
                    ref_fallback_scale=ref_fallback_scale,
                    ref_soft_context_scale=ref_soft_context_scale,
                    ref_nuclei_context_scale=ref_nuclei_context_scale,
                    ref_soft_context_radius=ref_soft_context_radius,
                    update_gain=cross16_steering_gain,
                )
            else:
                z = self.cross_16(
                    z,
                    r4,
                    query_region_map=target_region,
                    reference_region_map=reference_region,
                    query_trust_map=target_trust_map,
                    ref_fallback_scale=ref_fallback_scale,
                    ref_soft_context_scale=ref_soft_context_scale,
                    ref_nuclei_context_scale=ref_nuclei_context_scale,
                    ref_soft_context_radius=ref_soft_context_radius,
                )
        identity_masks = (
            target_tissue_mask,
            target_nuclei_mask,
            reference_tissue_mask,
            reference_nuclei_mask,
        )
        use_identity = self.identity_adapter is not None and all(mask is not None for mask in identity_masks)
        if use_identity:
            z, _ = self.identity_adapter.forward_scale(
                "1/16",
                z,
                r4,
                target_tissue_mask=target_tissue_mask,
                target_nuclei_mask=target_nuclei_mask,
                reference_tissue_mask=reference_tissue_mask,
                reference_nuclei_mask=reference_nuclei_mask,
            )
        x = self.up3(z, e3)
        if self.cross_8 is not None:
            if cross4_rotation_weights is not None and "1/8" in steering_scales:
                assert steered_region_maps is not None
                x = self.cross_8.forward_multi_reference(
                    x,
                    steered_features["1/8"],
                    reference_weights=cross4_rotation_weights,
                    query_region_map=target_region,
                    reference_region_maps=steered_region_maps,
                    query_trust_map=target_trust_map,
                    ref_fallback_scale=ref_fallback_scale,
                    ref_soft_context_scale=ref_soft_context_scale,
                    ref_nuclei_context_scale=ref_nuclei_context_scale,
                    ref_soft_context_radius=ref_soft_context_radius,
                    update_gain=cross8_steering_gain,
                )
            else:
                x = self.cross_8(
                    x,
                    r3,
                    query_region_map=target_region,
                    reference_region_map=reference_region,
                    query_trust_map=target_trust_map,
                    ref_fallback_scale=ref_fallback_scale,
                    ref_soft_context_scale=ref_soft_context_scale,
                    ref_nuclei_context_scale=ref_nuclei_context_scale,
                    ref_soft_context_radius=ref_soft_context_radius,
                )
        if use_identity:
            x, _ = self.identity_adapter.forward_scale(
                "1/8",
                x,
                r3,
                target_tissue_mask=target_tissue_mask,
                target_nuclei_mask=target_nuclei_mask,
                reference_tissue_mask=reference_tissue_mask,
                reference_nuclei_mask=reference_nuclei_mask,
            )
        x = self.up2(x, e2)
        if self.cross_4 is not None:
            if cross4_rotation_weights is not None and "1/4" in steering_scales:
                assert steered_region_maps is not None
                x = self.cross_4.forward_multi_reference(
                    x,
                    steered_features["1/4"],
                    reference_weights=cross4_rotation_weights,
                    query_region_map=target_region,
                    reference_region_maps=steered_region_maps,
                    query_trust_map=target_trust_map,
                    ref_fallback_scale=ref_fallback_scale,
                    ref_soft_context_scale=ref_soft_context_scale,
                    ref_nuclei_context_scale=ref_nuclei_context_scale,
                    ref_soft_context_radius=ref_soft_context_radius,
                    update_gain=cross4_steering_gain,
                )
            elif cross4_rot90_gate is None:
                x = self.cross_4(
                    x,
                    r2,
                    query_region_map=target_region,
                    reference_region_map=reference_region,
                    query_trust_map=target_trust_map,
                    ref_fallback_scale=ref_fallback_scale,
                    ref_soft_context_scale=ref_soft_context_scale,
                    ref_nuclei_context_scale=ref_nuclei_context_scale,
                    ref_soft_context_radius=ref_soft_context_radius,
                )
            else:
                reference_rot90 = torch.rot90(reference_cond, 1, dims=(-2, -1))
                r0_rot90 = self.ref_in(reference_rot90)
                r1_rot90 = self.ref_down1(r0_rot90)
                r2_rot90 = self.ref_down2(r1_rot90)
                reference_region_rot90 = (
                    torch.rot90(reference_region, 1, dims=(-2, -1))
                    if reference_region is not None
                    else None
                )
                x = self.cross_4.forward_dual_reference(
                    x,
                    r2,
                    r2_rot90,
                    rot90_gate=cross4_rot90_gate,
                    query_region_map=target_region,
                    reference_region_map=reference_region,
                    reference_region_map_rot90=reference_region_rot90,
                    query_trust_map=target_trust_map,
                    ref_fallback_scale=ref_fallback_scale,
                    ref_soft_context_scale=ref_soft_context_scale,
                    ref_nuclei_context_scale=ref_nuclei_context_scale,
                    ref_soft_context_radius=ref_soft_context_radius,
                )
        if use_identity:
            x, _ = self.identity_adapter.forward_scale(
                "1/4",
                x,
                r2,
                target_tissue_mask=target_tissue_mask,
                target_nuclei_mask=target_nuclei_mask,
                reference_tissue_mask=reference_tissue_mask,
                reference_nuclei_mask=reference_nuclei_mask,
            )
        x = self.up1(x, e1)
        if cross4_rotation_weights is not None and "1/2" in steering_scales:
            if self.steering_cross_2 is None or steered_region_maps is None:
                raise RuntimeError("1/2 steering requested without full-pyramid module")
            x = self.steering_cross_2.forward_multi_reference(
                x,
                steered_features["1/2"],
                reference_weights=cross4_rotation_weights,
                query_region_map=target_region,
                reference_region_maps=steered_region_maps,
                query_trust_map=highres_trust_map,
                ref_fallback_scale=ref_fallback_scale,
                ref_soft_context_scale=ref_soft_context_scale,
                ref_nuclei_context_scale=ref_nuclei_context_scale,
                ref_soft_context_radius=ref_soft_context_radius,
                update_gain=cross2_steering_gain,
            )
        x = self.up0(x, e0)
        if cross4_rotation_weights is not None and "1/1" in steering_scales:
            if self.steering_cross_1 is None or steered_region_maps is None:
                raise RuntimeError("1/1 steering requested without full-pyramid module")
            x = self.steering_cross_1.forward_multi_reference(
                x,
                steered_features["1/1"],
                reference_weights=cross4_rotation_weights,
                query_region_map=target_region,
                reference_region_maps=steered_region_maps,
                query_trust_map=highres_trust_map,
                ref_fallback_scale=ref_fallback_scale,
                ref_soft_context_scale=ref_soft_context_scale,
                ref_nuclei_context_scale=ref_nuclei_context_scale,
                ref_soft_context_radius=ref_soft_context_radius,
                update_gain=cross1_steering_gain,
            )
        residual_or_image = self.out(x)
        if self.residual_output:
            i0 = target_cond[:, :3]
            return torch.tanh(i0 + residual_or_image)
        return torch.tanh(residual_or_image)


def model_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

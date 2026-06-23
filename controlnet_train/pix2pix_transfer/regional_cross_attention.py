"""UNet generator with mask-guided regional cross attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def downsample_label_map(label_map: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    labels = F.interpolate(label_map.float(), size=size, mode="nearest")
    b, _, h, w = labels.shape
    return labels.long().view(b, h * w)


def build_region_attention_mask(
    query_labels: torch.Tensor,
    key_labels: torch.Tensor,
) -> torch.Tensor:
    allow = query_labels.unsqueeze(2) == key_labels.unsqueeze(1)
    has_match = allow.any(dim=2, keepdim=True)
    return torch.where(has_match, allow, torch.ones_like(allow))


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

    def __init__(self, dim: int, num_heads: int = 4, use_region_mask: bool = True) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.use_region_mask = bool(use_region_mask)

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

    def forward(
        self,
        query_feat: torch.Tensor,
        reference_feat: torch.Tensor,
        *,
        query_region_map: torch.Tensor | None = None,
        reference_region_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, c, hq, wq = query_feat.shape
        _, _, hk, wk = reference_feat.shape
        q = self._heads(self.to_q(self.norm_q(query_feat)))
        k = self._heads(self.to_k(self.norm_kv(reference_feat)))
        v = self._heads(self.to_v(self.norm_kv(reference_feat)))

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if (
            self.use_region_mask
            and query_region_map is not None
            and reference_region_map is not None
        ):
            query_labels = downsample_label_map(query_region_map, (hq, wq))
            key_labels = downsample_label_map(reference_region_map, (hk, wk))
            region_mask = build_region_attention_mask(query_labels, key_labels)
            logits = logits.masked_fill(~region_mask.unsqueeze(1), -torch.finfo(logits.dtype).max)

        attn = logits.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(b, c, hq, wq)
        out = self.proj(out)
        return query_feat + self.gamma * out


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
    ) -> None:
        super().__init__()
        self.residual_output = bool(residual_output)
        self.cross_attn_scales = normalize_cross_attn_scales(cross_attn_scales)
        self.upsample_mode = normalize_upsample_mode(upsample_mode)

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
    ) -> torch.Tensor:
        e0, e1, e2, e3, e4 = self._encode_target(target_cond)
        _, _, r2, r3, r4 = self._encode_reference(reference_cond)
        z = self.bottleneck(e4)
        if self.cross_16 is not None:
            z = self.cross_16(
                z,
                r4,
                query_region_map=target_region,
                reference_region_map=reference_region,
            )
        x = self.up3(z, e3)
        if self.cross_8 is not None:
            x = self.cross_8(
                x,
                r3,
                query_region_map=target_region,
                reference_region_map=reference_region,
            )
        x = self.up2(x, e2)
        if self.cross_4 is not None:
            x = self.cross_4(
                x,
                r2,
                query_region_map=target_region,
                reference_region_map=reference_region,
            )
        x = self.up1(x, e1)
        x = self.up0(x, e0)
        residual_or_image = self.out(x)
        if self.residual_output:
            i0 = target_cond[:, :3]
            return torch.tanh(i0 + residual_or_image)
        return torch.tanh(residual_or_image)


def model_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

"""
ProbUNet — 细胞核概率图预测网络

轻量 UNet 架构，5层编码器 + 5层解码器 + skip connections。

输入: tissue_onehot + nuclei_onehot + mask + (可选)cancer_type
输出: 核类型概率 (6ch): [背景, neoplastic, inflammatory, connective, dead, epithelial]

NOTE: 当前 in_ch=29 (22 tissue + 6 nuclei + 1 mask)，
Phase 4 适配后将改为 29 (16 fine + 6 nuclei + 1 mask + 6 cancer_type)。
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """双层卷积 + GroupNorm + GELU + 残差跳连"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.skip(x)


class ProbUNet(nn.Module):
    """
    轻量 UNet，用于预测编辑区域内每个像素的细胞核类型概率。

    Args:
        in_ch: 输入通道数 (默认 29 = 22 tissue + 6 nuclei + 1 mask)
        out_ch: 输出通道数 (默认 6 = bg + 5 类核)
        base_ch: 基础通道数 (默认 64)
    """
    def __init__(self, in_ch=29, out_ch=6, base_ch=64):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch*2)
        self.enc3 = ConvBlock(base_ch*2, base_ch*4)
        self.enc4 = ConvBlock(base_ch*4, base_ch*8)
        self.enc5 = ConvBlock(base_ch*8, base_ch*8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch*8, base_ch*8)

        # Decoder
        self.up5 = nn.ConvTranspose2d(base_ch*8, base_ch*8, 2, stride=2)
        self.dec5 = ConvBlock(base_ch*16, base_ch*8)

        self.up4 = nn.ConvTranspose2d(base_ch*8, base_ch*8, 2, stride=2)
        self.dec4 = ConvBlock(base_ch*16, base_ch*4)

        self.up3 = nn.ConvTranspose2d(base_ch*4, base_ch*4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch*8, base_ch*2)

        self.up2 = nn.ConvTranspose2d(base_ch*2, base_ch*2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch*4, base_ch)

        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch*2, base_ch)

        # Output
        self.out_conv = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, tissue, nuclei_input, mask):
        """
        Args:
            tissue: (B, num_tissue, H, W) — tissue one-hot
            nuclei_input: (B, num_nuclei, H, W) — nuclei one-hot (编辑区域内清零)
            mask: (B, 1, H, W) — 编辑区域二值 mask

        Returns:
            (B, out_ch, H, W) — 核类型 logits
        """
        x = torch.cat([tissue, nuclei_input, mask], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        b = self.bottleneck(self.pool(e5))

        d5 = self.dec5(torch.cat([self.up5(b), e5], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)

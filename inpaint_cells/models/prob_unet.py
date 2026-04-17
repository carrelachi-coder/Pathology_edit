"""
ProbUNet — Cell nuclei probability prediction network.

Lightweight UNet (5-layer encoder + 5-layer decoder + skip connections)
with an Embedding-based input encoder (AD-4).

Input:  tissue_map (int, 0-15) + cell_map (int, 0-5) + mask (float) + cancer_id (int, 0-5)
        → ProbNetInputEncoder → (B, 17, H, W) dense features → UNet
Output: nuclei type probabilities (6ch): [bg, neoplastic, inflammatory, connective, dead, epithelial]

Phase 4.1 changes (AD-4: Embedding lookup replaces one-hot):
  - Added ProbNetInputEncoder: learnable Embedding tables for tissue/cell/cancer
  - ProbUNet now takes 4 separate integer inputs instead of pre-concatenated one-hot tensors
  - Input channels: 17 = tissue_emb(8) + cell_emb(4) + mask(1) + cancer_emb(4)
  - Embeddings are trained end-to-end with the UNet
"""
"""
new version
"""
import torch
import torch.nn as nn

from ..utils.mask_utils import (
    NUM_TISSUE, NUM_NUCLEI, NUM_CANCER_TYPES,
    TISSUE_EMB_DIM, CELL_EMB_DIM, CANCER_EMB_DIM,
    PROBNET_IN_CH,
)


class ProbNetInputEncoder(nn.Module):
    """
    Encode discrete ID maps into dense feature maps via learnable Embedding tables.

    Replaces one-hot encoding (AD-4):
      - Storage: 4 integer maps (tissue, cell, mask, cancer_id) — compact
      - Encoding: Embedding lookup → ~17ch dense features (vs one-hot 29ch)
      - No spurious ordinal relationships between IDs
      - Consistent with HTE paradigm in Phase 5 ControlNet

    Args:
        tissue_classes: number of tissue fine classes (default 16)
        cell_classes: number of cell classes including bg=0 (default 6, rows=7 with padding)
        cancer_types: number of cancer types (default 6)
        tissue_dim: embedding dimension for tissue (default 8)
        cell_dim: embedding dimension for cell (default 4)
        cancer_dim: embedding dimension for cancer type (default 4)
    """

    def __init__(
        self,
        tissue_classes: int = NUM_TISSUE,
        cell_classes: int = NUM_NUCLEI,   # 6 (bg=0 + 5 types)
        cancer_types: int = NUM_CANCER_TYPES,
        tissue_dim: int = TISSUE_EMB_DIM,
        cell_dim: int = CELL_EMB_DIM,
        cancer_dim: int = CANCER_EMB_DIM,
    ):
        super().__init__()
        self.tissue_emb = nn.Embedding(tissue_classes, tissue_dim)   # 16 -> 8
        self.cell_emb = nn.Embedding(cell_classes + 1, cell_dim)     # 7 (0=no cell, 1-5=types, +1 padding) -> 4
        self.cancer_emb = nn.Embedding(cancer_types, cancer_dim)     # 6 -> 4

        self.out_channels = tissue_dim + cell_dim + 1 + cancer_dim   # 8+4+1+4 = 17

        # Initialize embeddings
        nn.init.xavier_uniform_(self.tissue_emb.weight)
        nn.init.xavier_uniform_(self.cell_emb.weight)
        nn.init.xavier_uniform_(self.cancer_emb.weight)

    def forward(self, tissue_map, cell_map, mask, cancer_id):
        """
        Args:
            tissue_map: (B, H, W) int64, values [0, 15]
            cell_map:   (B, H, W) int64, values [0, 5] (0=no cell)
            mask:       (B, 1, H, W) float32, edit region binary mask
            cancer_id:  (B,) int64, values [0, 5]

        Returns:
            (B, 17, H, W) float32 — dense feature map for UNet input
        """
        # Embedding lookup: (B, H, W) -> (B, H, W, D) -> (B, D, H, W)
        t = self.tissue_emb(tissue_map).permute(0, 3, 1, 2)    # (B, 8, H, W)
        c = self.cell_emb(cell_map).permute(0, 3, 1, 2)        # (B, 4, H, W)

        # Cancer type: scalar -> embedding -> spatial broadcast
        cv = self.cancer_emb(cancer_id)                          # (B, 4)
        cm = cv[:, :, None, None].expand(
            -1, -1, tissue_map.shape[1], tissue_map.shape[2]
        )                                                        # (B, 4, H, W)

        return torch.cat([t, c, mask, cm], dim=1)                # (B, 17, H, W)


class ConvBlock(nn.Module):
    """Double convolution + GroupNorm + GELU + residual skip."""

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
    Lightweight UNet for predicting per-pixel nuclei type probabilities
    in edited regions.

    Architecture changes from Phase 4.1:
      - Integrates ProbNetInputEncoder (Embedding-based)
      - forward() now accepts 4 separate inputs: tissue_map, cell_map, mask, cancer_id
      - Input channels: PROBNET_IN_CH = 17 (vs old 29 one-hot)

    Args:
        out_ch: output channels (default 6 = bg + 5 nuclei types)
        base_ch: base channel width (default 64)
    """

    def __init__(self, out_ch=NUM_NUCLEI, base_ch=64):
        super().__init__()

        in_ch = PROBNET_IN_CH  # 17

        # Input encoder: discrete IDs -> dense features
        self.input_encoder = ProbNetInputEncoder()

        # Encoder
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)
        self.enc5 = ConvBlock(base_ch * 8, base_ch * 8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 8)

        # Decoder
        self.up5 = nn.ConvTranspose2d(base_ch * 8, base_ch * 8, 2, stride=2)
        self.dec5 = ConvBlock(base_ch * 16, base_ch * 8)

        self.up4 = nn.ConvTranspose2d(base_ch * 8, base_ch * 8, 2, stride=2)
        self.dec4 = ConvBlock(base_ch * 16, base_ch * 4)

        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 2)

        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch)

        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        # Output
        self.out_conv = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, tissue_map, cell_map, mask, cancer_id):
        """
        Args:
            tissue_map: (B, H, W) int64, unified fine tissue IDs [0, 15]
            cell_map:   (B, H, W) int64, nuclei indices [0, 5] (0=no cell, edit region zeroed)
            mask:       (B, 1, H, W) float32, edit region binary mask
            cancer_id:  (B,) int64, cancer type index [0, 5]

        Returns:
            (B, out_ch, H, W) — nuclei type logits
        """
        # Encode discrete inputs to dense features
        x = self.input_encoder(tissue_map, cell_map, mask, cancer_id)  # (B, 17, H, W)

        # UNet forward
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

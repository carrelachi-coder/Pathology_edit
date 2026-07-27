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

Hierarchical tissue update:
  - Fine tissue IDs use parent + residual embeddings
  - Legacy flat 16 x 8 tissue embeddings migrate losslessly on load
  - The UNet input width and inference interface remain unchanged
"""
import torch
import torch.nn as nn

from dataset_config import FINE_TO_PARENT, NUM_CELL_CLASSES, NUM_COARSE, NUM_FINE


# Keep the model definition independent of mask image I/O dependencies such as OpenCV.
NUM_TISSUE = NUM_FINE
NUM_NUCLEI = NUM_CELL_CLASSES + 1
NUM_CANCER_TYPES = 6
TISSUE_EMB_DIM = 8
CELL_EMB_DIM = 4
CANCER_EMB_DIM = 4
PROBNET_IN_CH = TISSUE_EMB_DIM + CELL_EMB_DIM + 1 + CANCER_EMB_DIM


_FINE_TO_PARENT = tuple(FINE_TO_PARENT[index] for index in range(NUM_FINE))


def freeze_non_density_parameters(model):
    """Freeze a ProbNet except for parameters under ``density_head.*``."""
    if not getattr(model, 'with_density_head', False) or model.density_head is None:
        raise ValueError('Density-head-only training requires a density head.')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('density_head.'))
    return [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]


def collapse_fine_to_parent(tissue_map: torch.Tensor) -> torch.Tensor:
    """Map unified fine tissue IDs to their coarse parent IDs."""
    if tissue_map.numel() == 0:
        return tissue_map
    if tissue_map.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError(f"tissue_map must contain integer IDs, got {tissue_map.dtype}.")

    min_id = int(tissue_map.min().item())
    max_id = int(tissue_map.max().item())
    if min_id < 0 or max_id >= NUM_FINE:
        raise ValueError(
            f"tissue_map IDs must be in [0, {NUM_FINE - 1}], got [{min_id}, {max_id}]."
        )

    lookup = torch.as_tensor(_FINE_TO_PARENT, dtype=torch.long, device=tissue_map.device)
    return lookup[tissue_map.long()]


def apply_fine_to_parent_dropout(
    tissue_map: torch.Tensor,
    probability: float,
) -> torch.Tensor:
    """Collapse all fine labels in selected samples to simulate coarse Segmentator input.

    Dropout is sampled once per batch item instead of once per pixel so it does not
    create artificial salt-and-pepper tissue boundaries.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}.")
    if tissue_map.ndim != 3:
        raise ValueError(f"Expected tissue_map with shape (B, H, W), got {tuple(tissue_map.shape)}.")
    if probability == 0.0 or tissue_map.shape[0] == 0:
        return tissue_map

    collapsed = collapse_fine_to_parent(tissue_map)
    if probability == 1.0:
        return collapsed

    collapse_sample = torch.rand(tissue_map.shape[0], device=tissue_map.device) < probability
    return torch.where(collapse_sample[:, None, None], collapsed, tissue_map)


class HierarchicalTissueEmbedding(nn.Module):
    """Parent + residual tissue embedding with legacy flat-table compatibility."""

    def __init__(
        self,
        embedding_dim: int = TISSUE_EMB_DIM,
        num_coarse: int = NUM_COARSE,
        num_fine: int = NUM_FINE,
    ) -> None:
        super().__init__()
        if num_coarse != NUM_COARSE or num_fine != NUM_FINE:
            raise ValueError(
                f"ProbNet expects {NUM_COARSE} coarse and {NUM_FINE} fine tissue labels, "
                f"got {num_coarse} and {num_fine}."
            )

        self.embedding_dim = embedding_dim
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.parent_embeddings = nn.Embedding(num_coarse, embedding_dim)
        self.delta_embeddings = nn.Embedding(num_fine, embedding_dim)
        self.register_buffer(
            "fine_to_parent",
            torch.tensor(_FINE_TO_PARENT, dtype=torch.long),
            persistent=False,
        )
        self.last_load_migrated_flat_embedding = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.parent_embeddings.weight)
        nn.init.zeros_(self.delta_embeddings.weight)

    def embedding_table(self) -> torch.Tensor:
        """Return the effective 16 x D fine-label embedding table."""
        residual_mask = (
            torch.arange(self.num_fine, device=self.delta_embeddings.weight.device)
            >= self.num_coarse
        )[:, None]
        residuals = self.delta_embeddings.weight * residual_mask
        return self.parent_embeddings(self.fine_to_parent) + residuals

    def fine_delta_l2(self) -> torch.Tensor:
        """L2 penalty over dataset-specific fine residuals only."""
        return self.delta_embeddings.weight[NUM_COARSE:].square().mean()

    def forward(self, tissue_map: torch.Tensor) -> torch.Tensor:
        if tissue_map.ndim != 3:
            raise ValueError(
                f"Expected tissue_map with shape (B, H, W), got {tuple(tissue_map.shape)}."
            )
        if tissue_map.numel() > 0:
            min_id = int(tissue_map.min().item())
            max_id = int(tissue_map.max().item())
            if min_id < 0 or max_id >= self.num_fine:
                raise ValueError(
                    f"tissue_map IDs must be in [0, {self.num_fine - 1}], "
                    f"got [{min_id}, {max_id}]."
                )

        tissue_ids = tissue_map.long()
        parent_ids = self.fine_to_parent[tissue_ids]
        residual_mask = (tissue_ids >= self.num_coarse).unsqueeze(-1)
        residuals = self.delta_embeddings(tissue_ids) * residual_mask
        features = self.parent_embeddings(parent_ids) + residuals
        return features.permute(0, 3, 1, 2).contiguous()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        flat_key = prefix + "weight"
        parent_key = prefix + "parent_embeddings.weight"
        delta_key = prefix + "delta_embeddings.weight"
        self.last_load_migrated_flat_embedding = False

        if flat_key in state_dict and parent_key not in state_dict and delta_key not in state_dict:
            flat_weight = state_dict.pop(flat_key)
            expected_shape = (self.num_fine, self.embedding_dim)
            if tuple(flat_weight.shape) != expected_shape:
                error_msgs.append(
                    f"Cannot migrate {flat_key}: expected shape {expected_shape}, "
                    f"got {tuple(flat_weight.shape)}."
                )
            else:
                parent_weight = flat_weight[:self.num_coarse].clone()
                parent_ids = self.fine_to_parent.to(flat_weight.device)
                delta_weight = flat_weight - parent_weight.index_select(0, parent_ids)
                state_dict[parent_key] = parent_weight
                state_dict[delta_key] = delta_weight
                self.last_load_migrated_flat_embedding = True

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
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
        if tissue_classes != NUM_FINE:
            raise ValueError(f"Expected {NUM_FINE} tissue classes, got {tissue_classes}.")
        self.tissue_emb = HierarchicalTissueEmbedding(tissue_dim)    # (8 parents + 16 deltas) -> 8
        self.cell_emb = nn.Embedding(cell_classes + 1, cell_dim)     # 7 (0=no cell, 1-5=types, +1 padding) -> 4
        self.cancer_emb = nn.Embedding(cancer_types, cancer_dim)     # 6 -> 4

        self.out_channels = tissue_dim + cell_dim + 1 + cancer_dim   # 8+4+1+4 = 17

        # Initialize embeddings
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
        t = self.tissue_emb(tissue_map)                         # (B, 8, H, W)
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
      - Optional five-channel non-negative nucleus-center density head

    Args:
        out_ch: output channels (default 6 = bg + 5 nuclei types)
        base_ch: base channel width (default 64)
        with_density_head: enable class-wise center-density prediction
        density_channels: number of center-density channels (default 5)
        density_init_bias: scalar or per-class pre-softplus output bias
    """

    def __init__(
        self,
        out_ch=NUM_NUCLEI,
        base_ch=64,
        with_density_head: bool = False,
        density_channels: int = NUM_NUCLEI - 1,
        density_init_bias=-9.0,
    ):
        super().__init__()

        in_ch = PROBNET_IN_CH  # 17

        # Input encoder: discrete IDs -> dense features
        self.input_encoder = ProbNetInputEncoder()
        self.with_density_head = with_density_head
        self.density_channels = density_channels

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
        if with_density_head:
            self.density_head = nn.Sequential(
                nn.Conv2d(base_ch, base_ch, 3, padding=1),
                nn.GroupNorm(min(8, base_ch), base_ch),
                nn.GELU(),
                nn.Conv2d(base_ch, density_channels, 1),
            )
            nn.init.zeros_(self.density_head[-1].weight)
            initial_bias = torch.as_tensor(density_init_bias, dtype=torch.float32).flatten()
            if initial_bias.numel() == 1:
                initial_bias = initial_bias.expand(density_channels)
            if initial_bias.numel() != density_channels:
                raise ValueError(
                    "density_init_bias must be a scalar or contain one value "
                    f"per density channel ({density_channels}), got {initial_bias.numel()}."
                )
            with torch.no_grad():
                self.density_head[-1].bias.copy_(initial_bias)
        else:
            self.density_head = None

    def forward(self, tissue_map, cell_map, mask, cancer_id, return_density=False):
        """
        Args:
            tissue_map: (B, H, W) int64, unified fine tissue IDs [0, 15]
            cell_map:   (B, H, W) int64, nuclei indices [0, 5] (0=no cell, edit region zeroed)
            mask:       (B, 1, H, W) float32, edit region binary mask
            cancer_id:  (B,) int64, cancer type index [0, 5]

        Returns:
            Semantic logits by default. With ``return_density=True``, returns
            ``(semantic_logits, density)``; density is ``None`` when the
            optional head is disabled.
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

        semantic_logits = self.out_conv(d1)
        if not return_density:
            return semantic_logits

        density = None
        if self.density_head is not None:
            density = torch.nn.functional.softplus(self.density_head(d1))
        return semantic_logits, density

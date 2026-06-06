"""Cross V4 mask-guided correspondence conditioning.

Cross V4 keeps the Cross V3 separation of target structure and reference
appearance, then adds explicit token metadata, per-class prior tokens, and
mask-guided attention bias for semantic correspondence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_config import FINE_TO_PARENT, NUM_COARSE, NUM_FINE
from controlnet_train.modules.cross_v3_conditioning import (
    CROSS_V3_PROMPT,
    CROSS_V3_REFERENCE_WITH_REF,
    CROSS_V3_REFERENCE_ZERO_REF,
    CROSS_V3_ROUTE_COARSE,
    CROSS_V3_ROUTE_FINE,
    CROSS_V3_ROUTE_NONE,
    CrossV3ControlSpec,
    CrossV3ReferenceSpec,
    apply_cross_v3_reference_mode,
    apply_cross_v3_reference_token_mode,
    build_cross_v3_control_condition,
    deterministic_latent_from_posterior,
    normalize_cross_v3_reference_mode,
    normalize_cross_v3_reference_route_mode,
    pack_cross_v3_reference_grid,
)
from controlnet_train.training.conditioning import packed_control_channels


CROSS_V4_PROMPT = CROSS_V3_PROMPT
CROSS_V4_REFERENCE_WITH_REF = CROSS_V3_REFERENCE_WITH_REF
CROSS_V4_REFERENCE_ZERO_REF = CROSS_V3_REFERENCE_ZERO_REF
CROSS_V4_ROUTE_NONE = CROSS_V3_ROUTE_NONE
CROSS_V4_ROUTE_COARSE = CROSS_V3_ROUTE_COARSE
CROSS_V4_ROUTE_FINE = CROSS_V3_ROUTE_FINE
NUM_CELL_WITH_BG = 6

_FINE_TO_PARENT = tuple(int(FINE_TO_PARENT[fine_id]) for fine_id in range(NUM_FINE))
_RAW_CELL_TO_INTERNAL = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    101: 1,
    102: 2,
    103: 3,
    104: 4,
    105: 5,
}


@dataclass(frozen=True)
class CrossV4ControlSpec(CrossV3ControlSpec):
    """Target-only ControlNet structure layout shared with Cross V3."""


@dataclass(frozen=True)
class CrossV4ReferenceSpec(CrossV3ReferenceSpec):
    """Reference context and prior-token layout for Cross V4."""

    tissue_prior_tokens_per_class: int = 4
    cell_prior_tokens_per_class: int = 0
    global_style_tokens: int = 0
    prior_init_std: float = 0.02

    @property
    def tissue_prior_token_count(self) -> int:
        return NUM_COARSE * max(0, int(self.tissue_prior_tokens_per_class))

    @property
    def cell_prior_token_count(self) -> int:
        return NUM_CELL_WITH_BG * max(0, int(self.cell_prior_tokens_per_class))


@dataclass(frozen=True)
class CrossV4ContextSegments:
    """Token offsets inside the final FLUX context sequence."""

    text: tuple[int, int]
    global_style: tuple[int, int]
    tissue_prior: tuple[int, int]
    cell_prior: tuple[int, int]
    route_anchor: tuple[int, int]
    reference_local: tuple[int, int]

    @property
    def total_tokens(self) -> int:
        return self.reference_local[1]


@dataclass
class CrossV4TokenMetadata:
    """Per-token semantic metadata downsampled from masks."""

    tissue_fine_id: torch.Tensor
    tissue_coarse_id: torch.Tensor
    tissue_confidence: torch.Tensor
    cell_hist: torch.Tensor
    cell_density: torch.Tensor


@dataclass
class CrossV4ReferenceEncoding:
    """Projected reference tokens plus metadata for local tokens."""

    local_tokens: torch.Tensor
    route_anchor_tokens: torch.Tensor
    metadata: CrossV4TokenMetadata

    @property
    def tokens(self) -> torch.Tensor:
        if self.route_anchor_tokens.shape[1] == 0:
            return self.local_tokens
        return torch.cat([self.route_anchor_tokens, self.local_tokens], dim=1)


@dataclass
class CrossV4PriorTokens:
    """Prior/style token tensors and per-token class IDs."""

    global_style_tokens: torch.Tensor
    tissue_prior_tokens: torch.Tensor
    tissue_prior_class_ids: torch.Tensor
    cell_prior_tokens: torch.Tensor
    cell_prior_class_ids: torch.Tensor


@dataclass
class CrossV4Context:
    """Final context sequence consumed by FLUX."""

    encoder_hidden_states: torch.Tensor
    txt_ids: torch.Tensor
    segments: CrossV4ContextSegments
    reference_metadata: CrossV4TokenMetadata
    tissue_prior_class_ids: torch.Tensor
    cell_prior_class_ids: torch.Tensor


@dataclass(frozen=True)
class CrossV4CorrespondenceBiasConfig:
    """Scalar weights for mask-guided image-to-context attention bias."""

    same_fine: float = 3.0
    same_coarse: float = 2.0
    mismatch: float = -2.0
    cell_similarity: float = 1.0
    density_gap: float = 0.5
    prior_when_ref_present: float = 0.5
    prior_when_ref_missing: float = 3.0
    prior_wrong_class: float = -2.0
    cell_prior: float = 1.0
    scale: float = 1.0


class CrossV4ReferenceContextEncoder(nn.Module):
    """Project local reference patches and optionally build route anchors."""

    def __init__(
        self,
        *,
        reference_latent_channels: int = 16,
        tissue_channels: int = 64,
        nuclei_channels: int = 16,
        token_dim: int = 4096,
        hidden_dim: int | None = None,
        output_init_std: float = 0.02,
        route_anchor_mode: str = CROSS_V4_ROUTE_NONE,
        route_embedding_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        route_anchor_mode = normalize_cross_v3_reference_route_mode(route_anchor_mode)
        self.spec = CrossV4ReferenceSpec(
            reference_latent_channels=reference_latent_channels,
            tissue_channels=tissue_channels,
            nuclei_channels=nuclei_channels,
            token_dim=token_dim,
            output_init_std=output_init_std,
            route_anchor_mode=route_anchor_mode,
            route_embedding_init_std=route_embedding_init_std,
        )
        hidden_dim = int(hidden_dim or token_dim)
        self.norm = nn.LayerNorm(self.spec.packed_channels)
        self.proj_in = nn.Linear(self.spec.packed_channels, hidden_dim)
        self.act = nn.GELU()
        self.proj_out = nn.Linear(hidden_dim, token_dim)
        self.output_init_std = float(output_init_std)
        self.route_embedding_init_std = float(route_embedding_init_std)
        self.route_class_count = self.spec.route_class_count
        if self.route_class_count > 0:
            self.local_route_embedding = nn.Embedding(self.route_class_count, token_dim)
            self.anchor_route_embedding = nn.Embedding(self.route_class_count, token_dim)
            self.route_type_embedding = nn.Embedding(2, token_dim)
            self.route_missing_anchor = nn.Parameter(torch.zeros(1, 1, token_dim))
        else:
            self.local_route_embedding = None
            self.anchor_route_embedding = None
            self.route_type_embedding = None
            self.register_parameter("route_missing_anchor", None)
        self._init_output_projection()
        self._init_route_embeddings()

    def _init_output_projection(self) -> None:
        if self.output_init_std < 0.0:
            raise ValueError(f"output_init_std must be non-negative, got {self.output_init_std}.")
        with torch.no_grad():
            if self.output_init_std == 0.0:
                self.proj_out.weight.zero_()
            else:
                nn.init.normal_(self.proj_out.weight, mean=0.0, std=self.output_init_std)
            self.proj_out.bias.zero_()

    def _init_route_embeddings(self) -> None:
        if self.route_class_count <= 0:
            return
        if self.route_embedding_init_std < 0.0:
            raise ValueError(
                f"route_embedding_init_std must be non-negative, got {self.route_embedding_init_std}."
            )
        with torch.no_grad():
            for embedding in (
                self.local_route_embedding,
                self.anchor_route_embedding,
                self.route_type_embedding,
            ):
                if embedding is None:
                    continue
                if self.route_embedding_init_std == 0.0:
                    embedding.weight.zero_()
                else:
                    nn.init.normal_(embedding.weight, mean=0.0, std=self.route_embedding_init_std)
            if self.route_missing_anchor is not None:
                self.route_missing_anchor.zero_()

    def forward(
        self,
        *,
        z_ref: torch.Tensor,
        ref_tissue_feat: torch.Tensor,
        ref_nuclei_feat: torch.Tensor,
        ref_tissue_ids: torch.Tensor,
        ref_nuclei_ids: torch.Tensor,
    ) -> CrossV4ReferenceEncoding:
        packed = pack_cross_v3_reference_grid(
            z_ref=z_ref,
            ref_tissue_feat=ref_tissue_feat,
            ref_nuclei_feat=ref_nuclei_feat,
        )
        local_tokens = self.proj_out(self.act(self.proj_in(self.norm(packed))))
        token_height = z_ref.shape[2] // 2
        token_width = z_ref.shape[3] // 2
        metadata = build_cross_v4_token_metadata(
            tissue_ids=ref_tissue_ids,
            nuclei_ids=ref_nuclei_ids,
            token_height=token_height,
            token_width=token_width,
        )
        route_anchor_tokens = local_tokens.new_zeros((local_tokens.shape[0], 0, local_tokens.shape[2]))
        if self.route_class_count > 0:
            local_tokens, route_anchor_tokens = self._build_route_anchor_tokens(
                local_tokens=local_tokens,
                route_ids=(
                    metadata.tissue_coarse_id
                    if self.spec.normalized_route_anchor_mode == CROSS_V4_ROUTE_COARSE
                    else metadata.tissue_fine_id
                ),
                route_confidence=metadata.tissue_confidence,
            )
        return CrossV4ReferenceEncoding(
            local_tokens=local_tokens,
            route_anchor_tokens=route_anchor_tokens,
            metadata=metadata,
        )

    def _build_route_anchor_tokens(
        self,
        *,
        local_tokens: torch.Tensor,
        route_ids: torch.Tensor,
        route_confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.local_route_embedding is None or self.anchor_route_embedding is None or self.route_type_embedding is None:
            return local_tokens, local_tokens.new_zeros((local_tokens.shape[0], 0, local_tokens.shape[2]))
        batch_size, token_count, token_dim = local_tokens.shape
        route_ids = route_ids.reshape(batch_size, token_count).to(device=local_tokens.device)
        route_confidence = route_confidence.reshape(batch_size, token_count).to(
            device=local_tokens.device,
            dtype=local_tokens.dtype,
        )
        local_type_id = torch.zeros((), device=local_tokens.device, dtype=torch.long)
        anchor_type_id = torch.ones((), device=local_tokens.device, dtype=torch.long)
        local_tokens = (
            local_tokens
            + self.local_route_embedding(route_ids)
            + self.route_type_embedding(local_type_id).view(1, 1, token_dim)
        )

        class_weights = F.one_hot(route_ids, num_classes=self.route_class_count).to(dtype=local_tokens.dtype)
        class_weights = class_weights * route_confidence.unsqueeze(-1)
        class_mass = class_weights.sum(dim=1)
        anchors = torch.einsum("bnc,bnd->bcd", class_weights, local_tokens)
        present = class_mass > 1e-6
        anchors = anchors / class_mass.clamp_min(1e-6).unsqueeze(-1)

        class_ids = torch.arange(self.route_class_count, device=local_tokens.device, dtype=torch.long)
        anchors = (
            anchors
            + self.anchor_route_embedding(class_ids).unsqueeze(0)
            + self.route_type_embedding(anchor_type_id).view(1, 1, token_dim)
        )
        if self.route_missing_anchor is not None:
            anchors = torch.where(
                present.unsqueeze(-1),
                anchors,
                self.route_missing_anchor.to(device=anchors.device, dtype=anchors.dtype),
            )
        return local_tokens, anchors


class CrossV4PriorTokenBank(nn.Module):
    """Learned tissue/cell prior tokens plus optional weak global style tokens."""

    def __init__(
        self,
        *,
        token_dim: int = 4096,
        tissue_prior_tokens_per_class: int = 4,
        cell_prior_tokens_per_class: int = 0,
        global_style_tokens: int = 0,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.tissue_prior_tokens_per_class = max(0, int(tissue_prior_tokens_per_class))
        self.cell_prior_tokens_per_class = max(0, int(cell_prior_tokens_per_class))
        self.global_style_token_count = max(0, int(global_style_tokens))
        self.init_std = float(init_std)
        if self.tissue_prior_tokens_per_class > 0:
            self.tissue_prior_tokens = nn.Parameter(
                torch.empty(NUM_COARSE, self.tissue_prior_tokens_per_class, self.token_dim)
            )
        else:
            self.register_parameter("tissue_prior_tokens", None)
        if self.cell_prior_tokens_per_class > 0:
            self.cell_prior_tokens = nn.Parameter(
                torch.empty(NUM_CELL_WITH_BG, self.cell_prior_tokens_per_class, self.token_dim)
            )
        else:
            self.register_parameter("cell_prior_tokens", None)
        if self.global_style_token_count > 0:
            self.global_style_offsets = nn.Parameter(torch.empty(self.global_style_token_count, self.token_dim))
        else:
            self.register_parameter("global_style_offsets", None)
        self.global_style_proj = (
            nn.Linear(self.token_dim, self.token_dim) if self.global_style_token_count > 0 else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.init_std < 0.0:
            raise ValueError(f"init_std must be non-negative, got {self.init_std}.")
        with torch.no_grad():
            for value in (self.tissue_prior_tokens, self.cell_prior_tokens, self.global_style_offsets):
                if value is None or value.numel() == 0:
                    continue
                if self.init_std == 0.0:
                    value.zero_()
                else:
                    nn.init.normal_(value, mean=0.0, std=self.init_std)
            if self.global_style_proj is not None:
                if self.init_std == 0.0:
                    self.global_style_proj.weight.zero_()
                else:
                    nn.init.normal_(self.global_style_proj.weight, mean=0.0, std=self.init_std)
                self.global_style_proj.bias.zero_()

    def forward(self, reference_local_tokens: torch.Tensor) -> CrossV4PriorTokens:
        if reference_local_tokens.ndim != 3:
            raise ValueError(
                "reference_local_tokens must have shape (B, N, C), "
                f"got {tuple(reference_local_tokens.shape)}."
            )
        batch_size = int(reference_local_tokens.shape[0])
        dtype = reference_local_tokens.dtype
        device = reference_local_tokens.device

        if self.tissue_prior_tokens is not None:
            tissue_tokens = self.tissue_prior_tokens.to(device=device, dtype=dtype)
            tissue_tokens = tissue_tokens.reshape(1, -1, self.token_dim).expand(batch_size, -1, -1)
            tissue_ids = torch.arange(NUM_COARSE, device=device, dtype=torch.long).repeat_interleave(
                self.tissue_prior_tokens_per_class
            )
        else:
            tissue_tokens = reference_local_tokens.new_zeros((batch_size, 0, self.token_dim))
            tissue_ids = torch.empty(0, device=device, dtype=torch.long)

        if self.cell_prior_tokens is not None:
            cell_tokens = self.cell_prior_tokens.to(device=device, dtype=dtype)
            cell_tokens = cell_tokens.reshape(1, -1, self.token_dim).expand(batch_size, -1, -1)
            cell_ids = torch.arange(NUM_CELL_WITH_BG, device=device, dtype=torch.long).repeat_interleave(
                self.cell_prior_tokens_per_class
            )
        else:
            cell_tokens = reference_local_tokens.new_zeros((batch_size, 0, self.token_dim))
            cell_ids = torch.empty(0, device=device, dtype=torch.long)

        if self.global_style_token_count > 0:
            pooled = reference_local_tokens.mean(dim=1)
            assert self.global_style_proj is not None
            assert self.global_style_offsets is not None
            projected = self.global_style_proj(pooled.to(dtype=self.global_style_proj.weight.dtype)).to(dtype=dtype)
            offsets = self.global_style_offsets.to(device=device, dtype=dtype).unsqueeze(0)
            global_tokens = projected.unsqueeze(1) + offsets
        else:
            global_tokens = reference_local_tokens.new_zeros((batch_size, 0, self.token_dim))

        return CrossV4PriorTokens(
            global_style_tokens=global_tokens,
            tissue_prior_tokens=tissue_tokens,
            tissue_prior_class_ids=tissue_ids,
            cell_prior_tokens=cell_tokens,
            cell_prior_class_ids=cell_ids,
        )


def build_cross_v4_control_condition(
    *,
    tar_tissue_feat: torch.Tensor,
    tar_nuclei_feat: torch.Tensor,
) -> torch.Tensor:
    """Concatenate target-only spatial skeleton consumed by ControlNet."""

    return build_cross_v3_control_condition(
        tar_tissue_feat=tar_tissue_feat,
        tar_nuclei_feat=tar_nuclei_feat,
    )


def build_cross_v4_token_metadata(
    *,
    tissue_ids: torch.Tensor,
    nuclei_ids: torch.Tensor,
    token_height: int,
    token_width: int,
) -> CrossV4TokenMetadata:
    """Downsample tissue/cell masks to one metadata record per packed FLUX token."""

    if token_height <= 0 or token_width <= 0:
        raise ValueError(f"token grid must be positive, got {token_height}x{token_width}.")
    tissue_ids = _ensure_batched_hw(tissue_ids, name="tissue_ids").long()
    nuclei_ids = _ensure_batched_hw(nuclei_ids, name="nuclei_ids").long()
    if tissue_ids.shape[0] != nuclei_ids.shape[0] or tissue_ids.shape[1:] != nuclei_ids.shape[1:]:
        raise ValueError(
            "tissue_ids and nuclei_ids must match on batch/spatial dims, "
            f"got {tuple(tissue_ids.shape)} vs {tuple(nuclei_ids.shape)}."
        )
    _validate_id_range(tissue_ids, low=0, high=NUM_FINE - 1, name="tissue_ids")

    fine_one_hot = F.one_hot(tissue_ids, num_classes=NUM_FINE).permute(0, 3, 1, 2).float()
    fine_pooled = F.adaptive_avg_pool2d(fine_one_hot, output_size=(int(token_height), int(token_width)))
    tissue_confidence, fine_ids = fine_pooled.max(dim=1)

    lookup = torch.tensor(_FINE_TO_PARENT, device=tissue_ids.device, dtype=torch.long)
    coarse_ids_per_pixel = lookup[tissue_ids]
    coarse_one_hot = F.one_hot(coarse_ids_per_pixel, num_classes=NUM_COARSE).permute(0, 3, 1, 2).float()
    coarse_pooled = F.adaptive_avg_pool2d(coarse_one_hot, output_size=(int(token_height), int(token_width)))
    _, coarse_ids = coarse_pooled.max(dim=1)

    cell_ids = remap_cross_v4_cell_ids(nuclei_ids)
    cell_one_hot = F.one_hot(cell_ids, num_classes=NUM_CELL_WITH_BG).permute(0, 3, 1, 2).float()
    cell_pooled = F.adaptive_avg_pool2d(cell_one_hot, output_size=(int(token_height), int(token_width)))
    cell_hist = cell_pooled.permute(0, 2, 3, 1).reshape(tissue_ids.shape[0], token_height * token_width, -1)
    cell_hist = cell_hist / cell_hist.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    cell_density = (1.0 - cell_hist[..., 0]).clamp(0.0, 1.0)

    return CrossV4TokenMetadata(
        tissue_fine_id=fine_ids.reshape(tissue_ids.shape[0], token_height * token_width).long(),
        tissue_coarse_id=coarse_ids.reshape(tissue_ids.shape[0], token_height * token_width).long(),
        tissue_confidence=tissue_confidence.reshape(tissue_ids.shape[0], token_height * token_width),
        cell_hist=cell_hist,
        cell_density=cell_density,
    )


def remap_cross_v4_cell_ids(nuclei_ids: torch.Tensor) -> torch.Tensor:
    """Map raw CellViT-style IDs to ``0..5`` internal cell classes."""

    nuclei_ids = nuclei_ids.long()
    if nuclei_ids.numel() == 0:
        return nuclei_ids
    max_raw = max(_RAW_CELL_TO_INTERNAL)
    mapping = torch.full((max_raw + 1,), -1, device=nuclei_ids.device, dtype=torch.long)
    for raw_id, internal_id in _RAW_CELL_TO_INTERNAL.items():
        mapping[raw_id] = internal_id
    min_id = int(nuclei_ids.min().item())
    max_id = int(nuclei_ids.max().item())
    if min_id < 0 or max_id >= mapping.numel():
        raise ValueError(
            f"nuclei_ids out of supported range: got [{min_id}, {max_id}], expected IDs within [0, {max_raw}]."
        )
    remapped = mapping[nuclei_ids]
    if (remapped < 0).any():
        invalid_ids = torch.unique(nuclei_ids[remapped < 0]).tolist()
        raise ValueError(f"Unsupported nuclei IDs encountered: {invalid_ids}")
    return remapped


def append_cross_v4_context(
    *,
    prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    reference_encoding: CrossV4ReferenceEncoding,
    prior_tokens: CrossV4PriorTokens,
) -> CrossV4Context:
    """Append style/prior/route/local-reference tokens and record segment offsets."""

    if prompt_embeds.ndim != 3:
        raise ValueError(f"prompt_embeds must have shape (B, N, C), got {tuple(prompt_embeds.shape)}.")
    tensors = [
        prior_tokens.global_style_tokens,
        prior_tokens.tissue_prior_tokens,
        prior_tokens.cell_prior_tokens,
        reference_encoding.route_anchor_tokens,
        reference_encoding.local_tokens,
    ]
    for tensor in tensors:
        if tensor.ndim != 3:
            raise ValueError(f"all context tensors must have shape (B, N, C), got {tuple(tensor.shape)}.")
        if tensor.shape[0] != prompt_embeds.shape[0] or tensor.shape[2] != prompt_embeds.shape[2]:
            raise ValueError(
                "all context tensors must match prompt_embeds on batch/embedding dims, "
                f"got {tuple(tensor.shape)} vs {tuple(prompt_embeds.shape)}."
            )
    if text_ids.ndim == 3:
        text_ids = text_ids[0]
    if text_ids.ndim != 2 or text_ids.shape[1] != 3:
        raise ValueError(f"text_ids must have shape (N, 3), got {tuple(text_ids.shape)}.")

    offset = int(prompt_embeds.shape[1])
    text_segment = (0, offset)

    def take_segment(count: int) -> tuple[int, int]:
        nonlocal offset
        start = offset
        offset += int(count)
        return start, offset

    segments = CrossV4ContextSegments(
        text=text_segment,
        global_style=take_segment(prior_tokens.global_style_tokens.shape[1]),
        tissue_prior=take_segment(prior_tokens.tissue_prior_tokens.shape[1]),
        cell_prior=take_segment(prior_tokens.cell_prior_tokens.shape[1]),
        route_anchor=take_segment(reference_encoding.route_anchor_tokens.shape[1]),
        reference_local=take_segment(reference_encoding.local_tokens.shape[1]),
    )
    encoder_hidden_states = torch.cat([prompt_embeds, *tensors], dim=1)
    extra_ids = torch.zeros(
        segments.total_tokens - int(text_ids.shape[0]),
        3,
        device=text_ids.device,
        dtype=text_ids.dtype,
    )
    return CrossV4Context(
        encoder_hidden_states=encoder_hidden_states,
        txt_ids=torch.cat([text_ids, extra_ids], dim=0),
        segments=segments,
        reference_metadata=reference_encoding.metadata,
        tissue_prior_class_ids=prior_tokens.tissue_prior_class_ids,
        cell_prior_class_ids=prior_tokens.cell_prior_class_ids,
    )


def build_cross_v4_correspondence_bias(
    *,
    target_metadata: CrossV4TokenMetadata,
    context: CrossV4Context,
    config: CrossV4CorrespondenceBiasConfig | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build additive image-query to context-key attention bias.

    Returned shape is ``(B, N_img, N_context)``. The custom FLUX V4 attention
    processor places it into the image-query/context-key slice of joint
    attention logits.
    """

    config = config or CrossV4CorrespondenceBiasConfig()
    target = target_metadata
    reference = context.reference_metadata
    device = target.tissue_coarse_id.device
    batch_size, image_tokens = target.tissue_coarse_id.shape
    context_tokens = context.segments.total_tokens
    out_dtype = dtype or target.cell_hist.dtype
    bias = torch.zeros(batch_size, image_tokens, context_tokens, device=device, dtype=out_dtype)
    if config.scale == 0.0:
        return bias

    if reference.tissue_coarse_id.shape[0] != batch_size:
        raise ValueError(
            "target/reference metadata batch sizes must match, "
            f"got {batch_size} and {reference.tissue_coarse_id.shape[0]}."
        )
    _add_reference_local_bias(bias, target=target, reference=reference, context=context, config=config)
    _add_tissue_prior_bias(bias, target=target, reference=reference, context=context, config=config)
    _add_cell_prior_bias(bias, target=target, context=context, config=config)
    if config.scale != 1.0:
        bias.mul_(float(config.scale))
    return bias


def normalize_cross_v4_reference_mode(mode: str) -> str:
    return normalize_cross_v3_reference_mode(mode)


def apply_cross_v4_reference_mode(
    *,
    z_ref: torch.Tensor,
    ref_tissue_feat: torch.Tensor,
    ref_nuclei_feat: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return apply_cross_v3_reference_mode(
        z_ref=z_ref,
        ref_tissue_feat=ref_tissue_feat,
        ref_nuclei_feat=ref_nuclei_feat,
        mode=mode,
    )


def apply_cross_v4_reference_encoding_mode(
    reference_encoding: CrossV4ReferenceEncoding,
    mode: str,
) -> CrossV4ReferenceEncoding:
    """Zero reference appearance tokens for ablations while preserving metadata."""

    normalized = normalize_cross_v4_reference_mode(mode)
    if normalized == CROSS_V4_REFERENCE_WITH_REF:
        return reference_encoding
    return CrossV4ReferenceEncoding(
        local_tokens=torch.zeros_like(reference_encoding.local_tokens),
        route_anchor_tokens=torch.zeros_like(reference_encoding.route_anchor_tokens),
        metadata=reference_encoding.metadata,
    )


def apply_cross_v4_reference_token_mode(reference_tokens: torch.Tensor, mode: str) -> torch.Tensor:
    return apply_cross_v3_reference_token_mode(reference_tokens, mode)


def _add_reference_local_bias(
    bias: torch.Tensor,
    *,
    target: CrossV4TokenMetadata,
    reference: CrossV4TokenMetadata,
    context: CrossV4Context,
    config: CrossV4CorrespondenceBiasConfig,
) -> None:
    start, end = context.segments.reference_local
    if end <= start:
        return
    same_fine = target.tissue_fine_id[:, :, None] == reference.tissue_fine_id[:, None, :]
    same_coarse = target.tissue_coarse_id[:, :, None] == reference.tissue_coarse_id[:, None, :]
    ref_bias = torch.where(
        same_fine,
        torch.as_tensor(config.same_fine, device=bias.device, dtype=bias.dtype),
        torch.where(
            same_coarse,
            torch.as_tensor(config.same_coarse, device=bias.device, dtype=bias.dtype),
            torch.as_tensor(config.mismatch, device=bias.device, dtype=bias.dtype),
        ),
    )
    target_hist = target.cell_hist.to(device=bias.device, dtype=bias.dtype)
    ref_hist = reference.cell_hist.to(device=bias.device, dtype=bias.dtype)
    cell_sim = (target_hist[:, :, None, :] * ref_hist[:, None, :, :]).sum(dim=-1)
    density_gap = (
        target.cell_density.to(device=bias.device, dtype=bias.dtype)[:, :, None]
        - reference.cell_density.to(device=bias.device, dtype=bias.dtype)[:, None, :]
    ).abs()
    ref_bias = (
        ref_bias
        + float(config.cell_similarity) * cell_sim
        - float(config.density_gap) * density_gap
    )
    bias[:, :, start:end] += ref_bias


def _add_tissue_prior_bias(
    bias: torch.Tensor,
    *,
    target: CrossV4TokenMetadata,
    reference: CrossV4TokenMetadata,
    context: CrossV4Context,
    config: CrossV4CorrespondenceBiasConfig,
) -> None:
    start, end = context.segments.tissue_prior
    if end <= start:
        return
    prior_ids = context.tissue_prior_class_ids.to(device=bias.device)
    if prior_ids.numel() != end - start:
        raise ValueError(
            "tissue prior class IDs must match tissue prior segment length, "
            f"got {prior_ids.numel()} IDs for segment {start}:{end}."
        )
    ref_presence = _class_presence(reference.tissue_coarse_id.to(device=bias.device), NUM_COARSE)
    target_is_class = target.tissue_coarse_id.to(device=bias.device)[:, :, None] == prior_ids.view(1, 1, -1)
    present_for_prior = ref_presence.gather(1, prior_ids.view(1, -1).expand(ref_presence.shape[0], -1))
    class_bonus = torch.where(
        present_for_prior,
        torch.as_tensor(config.prior_when_ref_present, device=bias.device, dtype=bias.dtype),
        torch.as_tensor(config.prior_when_ref_missing, device=bias.device, dtype=bias.dtype),
    )
    prior_bias = torch.where(
        target_is_class,
        class_bonus[:, None, :],
        torch.as_tensor(config.prior_wrong_class, device=bias.device, dtype=bias.dtype),
    )
    bias[:, :, start:end] += prior_bias


def _add_cell_prior_bias(
    bias: torch.Tensor,
    *,
    target: CrossV4TokenMetadata,
    context: CrossV4Context,
    config: CrossV4CorrespondenceBiasConfig,
) -> None:
    start, end = context.segments.cell_prior
    if end <= start:
        return
    prior_ids = context.cell_prior_class_ids.to(device=bias.device)
    if prior_ids.numel() != end - start:
        raise ValueError(
            "cell prior class IDs must match cell prior segment length, "
            f"got {prior_ids.numel()} IDs for segment {start}:{end}."
        )
    cell_hist = target.cell_hist.to(device=bias.device, dtype=bias.dtype)
    bias[:, :, start:end] += float(config.cell_prior) * cell_hist.index_select(dim=-1, index=prior_ids)


def _class_presence(class_ids: torch.Tensor, class_count: int) -> torch.Tensor:
    one_hot = F.one_hot(class_ids.long(), num_classes=class_count).bool()
    return one_hot.any(dim=1)


def _ensure_batched_hw(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.ndim == 2:
        return value.unsqueeze(0)
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape (B,H,W) or (H,W), got {tuple(value.shape)}.")
    return value


def _validate_id_range(value: torch.Tensor, *, low: int, high: int, name: str) -> None:
    if value.numel() == 0:
        return
    min_id = int(value.min().item())
    max_id = int(value.max().item())
    if min_id < low or max_id > high:
        raise ValueError(f"{name} out of range: got [{min_id}, {max_id}], expected [{low}, {high}].")


__all__ = [
    "CROSS_V4_PROMPT",
    "CROSS_V4_REFERENCE_WITH_REF",
    "CROSS_V4_REFERENCE_ZERO_REF",
    "CROSS_V4_ROUTE_COARSE",
    "CROSS_V4_ROUTE_FINE",
    "CROSS_V4_ROUTE_NONE",
    "NUM_CELL_WITH_BG",
    "CrossV4Context",
    "CrossV4ContextSegments",
    "CrossV4ControlSpec",
    "CrossV4CorrespondenceBiasConfig",
    "CrossV4PriorTokenBank",
    "CrossV4PriorTokens",
    "CrossV4ReferenceContextEncoder",
    "CrossV4ReferenceEncoding",
    "CrossV4ReferenceSpec",
    "CrossV4TokenMetadata",
    "append_cross_v4_context",
    "apply_cross_v4_reference_encoding_mode",
    "apply_cross_v4_reference_mode",
    "apply_cross_v4_reference_token_mode",
    "build_cross_v4_control_condition",
    "build_cross_v4_correspondence_bias",
    "build_cross_v4_token_metadata",
    "deterministic_latent_from_posterior",
    "normalize_cross_v4_reference_mode",
    "pack_cross_v3_reference_grid",
    "packed_control_channels",
    "remap_cross_v4_cell_ids",
]

"""Cross V3 conditioning with separated structure and appearance pathways.

ControlNet receives only target tissue/nuclei structure. Reference appearance is
encoded as content-addressable context tokens from ``z_ref`` plus reference
masks and enters FLUX through joint cross-attention.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_config import FINE_TO_PARENT, NUM_COARSE, NUM_FINE
from controlnet_train.training.conditioning import packed_control_channels


CROSS_V3_PROMPT = "histopathology image"
CROSS_V3_REFERENCE_WITH_REF = "with_ref"
CROSS_V3_REFERENCE_ZERO_REF = "zero_ref"
CROSS_V3_ROUTE_NONE = "none"
CROSS_V3_ROUTE_COARSE = "coarse"
CROSS_V3_ROUTE_FINE = "fine"

_FINE_TO_PARENT = tuple(int(FINE_TO_PARENT[fine_id]) for fine_id in range(NUM_FINE))


@dataclass(frozen=True)
class CrossV3ControlSpec:
    """Target-only ControlNet structure layout: [tar_tissue_feat, tar_nuclei_feat]."""

    tissue_channels: int = 64
    nuclei_channels: int = 16

    @property
    def raw_channels(self) -> int:
        return self.tissue_channels + self.nuclei_channels

    @property
    def packed_channels(self) -> int:
        return packed_control_channels(self.raw_channels)


@dataclass(frozen=True)
class CrossV3ReferenceSpec:
    """Reference cross-attention token layout before projection."""

    reference_latent_channels: int = 16
    tissue_channels: int = 64
    nuclei_channels: int = 16
    token_dim: int = 4096
    output_init_std: float = 0.02
    route_anchor_mode: str = CROSS_V3_ROUTE_NONE
    route_embedding_init_std: float = 0.02

    @property
    def raw_channels(self) -> int:
        return self.reference_latent_channels + self.tissue_channels + self.nuclei_channels

    @property
    def packed_channels(self) -> int:
        return packed_control_channels(self.raw_channels)

    @property
    def normalized_route_anchor_mode(self) -> str:
        return normalize_cross_v3_reference_route_mode(self.route_anchor_mode)

    @property
    def route_class_count(self) -> int:
        return cross_v3_route_class_count(self.route_anchor_mode)


class CrossV3ReferenceContextEncoder(nn.Module):
    """Project packed local reference patches into FLUX joint-attention tokens."""

    def __init__(
        self,
        *,
        reference_latent_channels: int = 16,
        tissue_channels: int = 64,
        nuclei_channels: int = 16,
        token_dim: int = 4096,
        hidden_dim: int | None = None,
        output_init_std: float = 0.02,
        route_anchor_mode: str = CROSS_V3_ROUTE_NONE,
        route_embedding_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        route_anchor_mode = normalize_cross_v3_reference_route_mode(route_anchor_mode)
        self.spec = CrossV3ReferenceSpec(
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
        ref_tissue_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        packed = pack_cross_v3_reference_grid(
            z_ref=z_ref,
            ref_tissue_feat=ref_tissue_feat,
            ref_nuclei_feat=ref_nuclei_feat,
        )
        local_tokens = self.proj_out(self.act(self.proj_in(self.norm(packed))))
        if self.route_class_count <= 0:
            return local_tokens
        if ref_tissue_ids is None:
            raise ValueError(
                "ref_tissue_ids are required when CrossV3ReferenceContextEncoder uses semantic route anchors."
            )
        return self._append_route_anchor_tokens(
            local_tokens=local_tokens,
            ref_tissue_ids=ref_tissue_ids,
            token_height=z_ref.shape[2] // 2,
            token_width=z_ref.shape[3] // 2,
        )

    def _append_route_anchor_tokens(
        self,
        *,
        local_tokens: torch.Tensor,
        ref_tissue_ids: torch.Tensor,
        token_height: int,
        token_width: int,
    ) -> torch.Tensor:
        if self.local_route_embedding is None or self.anchor_route_embedding is None or self.route_type_embedding is None:
            return local_tokens
        route_ids, route_confidence = build_cross_v3_reference_route_ids(
            ref_tissue_ids=ref_tissue_ids,
            token_height=token_height,
            token_width=token_width,
            route_anchor_mode=self.spec.route_anchor_mode,
        )
        batch_size, token_count, token_dim = local_tokens.shape
        route_ids_flat = route_ids.reshape(batch_size, token_count).to(device=local_tokens.device)
        route_conf_flat = route_confidence.reshape(batch_size, token_count).to(
            device=local_tokens.device,
            dtype=local_tokens.dtype,
        )

        local_type_id = torch.zeros((), device=local_tokens.device, dtype=torch.long)
        anchor_type_id = torch.ones((), device=local_tokens.device, dtype=torch.long)
        local_tokens = (
            local_tokens
            + self.local_route_embedding(route_ids_flat)
            + self.route_type_embedding(local_type_id).view(1, 1, token_dim)
        )

        class_weights = F.one_hot(route_ids_flat, num_classes=self.route_class_count).to(dtype=local_tokens.dtype)
        class_weights = class_weights * route_conf_flat.unsqueeze(-1)
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
        return torch.cat([anchors, local_tokens], dim=1)


def normalize_cross_v3_reference_route_mode(mode: str | None) -> str:
    value = str(mode or CROSS_V3_ROUTE_NONE).strip().lower().replace("-", "_")
    aliases = {
        "": CROSS_V3_ROUTE_NONE,
        "off": CROSS_V3_ROUTE_NONE,
        "no": CROSS_V3_ROUTE_NONE,
        "none": CROSS_V3_ROUTE_NONE,
        "coarse": CROSS_V3_ROUTE_COARSE,
        "coarse_anchor": CROSS_V3_ROUTE_COARSE,
        "coarse_anchors": CROSS_V3_ROUTE_COARSE,
        "fine": CROSS_V3_ROUTE_FINE,
        "fine_anchor": CROSS_V3_ROUTE_FINE,
        "fine_anchors": CROSS_V3_ROUTE_FINE,
    }
    if value not in aliases:
        raise ValueError(
            f"Unsupported Cross V3 reference route mode {mode!r}; "
            f"choose {CROSS_V3_ROUTE_NONE!r}, {CROSS_V3_ROUTE_COARSE!r}, or {CROSS_V3_ROUTE_FINE!r}."
        )
    return aliases[value]


def cross_v3_route_class_count(mode: str | None) -> int:
    normalized = normalize_cross_v3_reference_route_mode(mode)
    if normalized == CROSS_V3_ROUTE_NONE:
        return 0
    if normalized == CROSS_V3_ROUTE_COARSE:
        return NUM_COARSE
    if normalized == CROSS_V3_ROUTE_FINE:
        return NUM_FINE
    raise AssertionError(f"unhandled route mode: {normalized}")


def build_cross_v3_reference_route_ids(
    *,
    ref_tissue_ids: torch.Tensor,
    token_height: int,
    token_width: int,
    route_anchor_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a reference tissue mask into per-reference-token route IDs."""

    route_anchor_mode = normalize_cross_v3_reference_route_mode(route_anchor_mode)
    if route_anchor_mode == CROSS_V3_ROUTE_NONE:
        raise ValueError("route_anchor_mode='none' does not produce route IDs.")
    if token_height <= 0 or token_width <= 0:
        raise ValueError(f"token grid must be positive, got {token_height}x{token_width}.")

    if ref_tissue_ids.ndim == 2:
        ref_tissue_ids = ref_tissue_ids.unsqueeze(0)
    if ref_tissue_ids.ndim != 3:
        raise ValueError(
            f"ref_tissue_ids must have shape (B,H,W) or (H,W), got {tuple(ref_tissue_ids.shape)}."
        )
    tissue_ids = ref_tissue_ids.long()
    if tissue_ids.numel() > 0:
        min_id = int(tissue_ids.min().item())
        max_id = int(tissue_ids.max().item())
        if min_id < 0 or max_id >= NUM_FINE:
            raise ValueError(
                f"ref_tissue_ids out of range: got [{min_id}, {max_id}], expected [0, {NUM_FINE - 1}]."
            )

    if route_anchor_mode == CROSS_V3_ROUTE_COARSE:
        lookup = torch.tensor(_FINE_TO_PARENT, device=tissue_ids.device, dtype=torch.long)
        class_ids = lookup[tissue_ids]
        class_count = NUM_COARSE
    else:
        class_ids = tissue_ids
        class_count = NUM_FINE
    one_hot = F.one_hot(class_ids, num_classes=class_count).permute(0, 3, 1, 2).float()
    pooled = F.adaptive_avg_pool2d(one_hot, output_size=(int(token_height), int(token_width)))
    confidence, route_ids = pooled.max(dim=1)
    return route_ids.long(), confidence


def build_cross_v3_control_condition(
    *,
    tar_tissue_feat: torch.Tensor,
    tar_nuclei_feat: torch.Tensor,
) -> torch.Tensor:
    """Concatenate the target-only spatial skeleton consumed by ControlNet."""

    _validate_feature_grid(
        {
            "tar_tissue_feat": tar_tissue_feat,
            "tar_nuclei_feat": tar_nuclei_feat,
        },
        reference_name="tar_tissue_feat",
    )
    return torch.cat([tar_tissue_feat, tar_nuclei_feat], dim=1)


def pack_cross_v3_reference_grid(
    *,
    z_ref: torch.Tensor,
    ref_tissue_feat: torch.Tensor,
    ref_nuclei_feat: torch.Tensor,
) -> torch.Tensor:
    """Pack local reference latent/mask features into a cross-attention sequence."""

    _validate_feature_grid(
        {
            "z_ref": z_ref,
            "ref_tissue_feat": ref_tissue_feat,
            "ref_nuclei_feat": ref_nuclei_feat,
        },
        reference_name="z_ref",
    )
    reference_grid = torch.cat([z_ref, ref_tissue_feat, ref_nuclei_feat], dim=1)
    batch_size, channels, height, width = reference_grid.shape
    if height % 2 or width % 2:
        raise ValueError(f"Reference feature grid must have even H/W for 2x2 packing, got {height}x{width}.")
    reference_grid = reference_grid.view(batch_size, channels, height // 2, 2, width // 2, 2)
    reference_grid = reference_grid.permute(0, 2, 4, 1, 3, 5)
    return reference_grid.reshape(batch_size, (height // 2) * (width // 2), channels * 4)


def normalize_cross_v3_reference_mode(mode: str) -> str:
    value = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "normal": CROSS_V3_REFERENCE_WITH_REF,
        "ref": CROSS_V3_REFERENCE_WITH_REF,
        "reference": CROSS_V3_REFERENCE_WITH_REF,
        "with_ref": CROSS_V3_REFERENCE_WITH_REF,
        "zero": CROSS_V3_REFERENCE_ZERO_REF,
        "zero_ref": CROSS_V3_REFERENCE_ZERO_REF,
    }
    if value not in aliases:
        raise ValueError(
            f"Unsupported Cross V3 reference mode {mode!r}; "
            f"choose {CROSS_V3_REFERENCE_WITH_REF!r} or {CROSS_V3_REFERENCE_ZERO_REF!r}."
        )
    return aliases[value]


def apply_cross_v3_reference_mode(
    *,
    z_ref: torch.Tensor,
    ref_tissue_feat: torch.Tensor,
    ref_nuclei_feat: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Optionally ablate the full reference cross-attention input."""

    normalized = normalize_cross_v3_reference_mode(mode)
    if normalized == CROSS_V3_REFERENCE_WITH_REF:
        return z_ref, ref_tissue_feat, ref_nuclei_feat
    return (
        torch.zeros_like(z_ref),
        torch.zeros_like(ref_tissue_feat),
        torch.zeros_like(ref_nuclei_feat),
    )


def apply_cross_v3_reference_token_mode(reference_tokens: torch.Tensor, mode: str) -> torch.Tensor:
    """Zero all reference context tokens for an explicit inference ablation."""

    normalized = normalize_cross_v3_reference_mode(mode)
    if normalized == CROSS_V3_REFERENCE_WITH_REF:
        return reference_tokens
    return torch.zeros_like(reference_tokens)


def deterministic_latent_from_posterior(posterior) -> torch.Tensor:
    """Return a stable latent from a VAE posterior for reference conditioning."""

    return posterior.mode() if hasattr(posterior, "mode") else posterior.mean


def append_cross_v3_reference_context(
    *,
    prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    reference_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append content-addressable reference tokens without reference coordinates."""

    if prompt_embeds.ndim != 3 or reference_tokens.ndim != 3:
        raise ValueError("prompt_embeds and reference_tokens must both have shape (B, N, C).")
    if prompt_embeds.shape[0] != reference_tokens.shape[0] or prompt_embeds.shape[2] != reference_tokens.shape[2]:
        raise ValueError(
            "reference_tokens must match prompt_embeds on batch/embedding dims, "
            f"got {tuple(reference_tokens.shape)} vs {tuple(prompt_embeds.shape)}."
        )
    if text_ids.ndim == 3:
        text_ids = text_ids[0]
    if text_ids.ndim != 2 or text_ids.shape[1] != 3:
        raise ValueError(f"text_ids must have shape (N, 3), got {tuple(text_ids.shape)}.")
    reference_ids = torch.zeros(
        reference_tokens.shape[1],
        3,
        device=text_ids.device,
        dtype=text_ids.dtype,
    )
    return torch.cat([prompt_embeds, reference_tokens], dim=1), torch.cat([text_ids, reference_ids], dim=0)


def _validate_feature_grid(features: dict[str, torch.Tensor], *, reference_name: str) -> None:
    reference = features[reference_name]
    for name, value in features.items():
        if value.ndim != 4:
            raise ValueError(f"{name} must have shape (B, C, H, W), got {tuple(value.shape)}.")
        if value.shape[0] != reference.shape[0] or value.shape[2:] != reference.shape[2:]:
            raise ValueError(
                f"{name} must match {reference_name} on batch/spatial dims, "
                f"got {tuple(value.shape)} vs {tuple(reference.shape)}."
            )

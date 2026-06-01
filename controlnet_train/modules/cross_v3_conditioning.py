"""Cross V3 conditioning with separated structure and appearance pathways.

ControlNet receives only target tissue/nuclei structure. Reference appearance is
encoded as content-addressable context tokens from ``z_ref`` plus reference
masks and enters FLUX through joint cross-attention.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from controlnet_train.training.conditioning import packed_control_channels


CROSS_V3_PROMPT = "histopathology image"
CROSS_V3_REFERENCE_WITH_REF = "with_ref"
CROSS_V3_REFERENCE_ZERO_REF = "zero_ref"


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

    @property
    def raw_channels(self) -> int:
        return self.reference_latent_channels + self.tissue_channels + self.nuclei_channels

    @property
    def packed_channels(self) -> int:
        return packed_control_channels(self.raw_channels)


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
    ) -> None:
        super().__init__()
        self.spec = CrossV3ReferenceSpec(
            reference_latent_channels=reference_latent_channels,
            tissue_channels=tissue_channels,
            nuclei_channels=nuclei_channels,
            token_dim=token_dim,
        )
        hidden_dim = int(hidden_dim or token_dim)
        self.norm = nn.LayerNorm(self.spec.packed_channels)
        self.proj_in = nn.Linear(self.spec.packed_channels, hidden_dim)
        self.act = nn.GELU()
        self.proj_out = nn.Linear(hidden_dim, token_dim)
        self.output_init_std = float(output_init_std)
        self._init_output_projection()

    def _init_output_projection(self) -> None:
        if self.output_init_std < 0.0:
            raise ValueError(f"output_init_std must be non-negative, got {self.output_init_std}.")
        with torch.no_grad():
            if self.output_init_std == 0.0:
                self.proj_out.weight.zero_()
            else:
                nn.init.normal_(self.proj_out.weight, mean=0.0, std=self.output_init_std)
            self.proj_out.bias.zero_()

    def forward(
        self,
        *,
        z_ref: torch.Tensor,
        ref_tissue_feat: torch.Tensor,
        ref_nuclei_feat: torch.Tensor,
    ) -> torch.Tensor:
        packed = pack_cross_v3_reference_grid(
            z_ref=z_ref,
            ref_tissue_feat=ref_tissue_feat,
            ref_nuclei_feat=ref_nuclei_feat,
        )
        return self.proj_out(self.act(self.proj_in(self.norm(packed))))


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

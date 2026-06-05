"""Cross V5-min bank-conditioned appearance interfaces.

V5-min keeps nuclei/cell masks in the geometry path, but keeps cell-type
appearance routing out of the first implementation. The modules here are the
small reusable pieces needed for tissue-level bank rendering:

* stain-stat prototypes from reference RGB and masks;
* masked pooling from low-level texture tokens into per-class local banks;
* hard class gathering for target tokens;
* AdaLN-style appearance modulation that does not compete with ControlNet
  residuals on the same additive path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_config import NUM_COARSE
from controlnet_train.training.conditioning import packed_control_channels


@dataclass(frozen=True)
class CrossV5GeometryControlSpec:
    """Class-agnostic V5 ControlNet geometry condition layout."""

    geometry_channels: int = 4

    @property
    def raw_channels(self) -> int:
        return int(self.geometry_channels)

    @property
    def packed_channels(self) -> int:
        return packed_control_channels(self.raw_channels)

    @property
    def condition_order(self) -> tuple[str, ...]:
        return (
            "target_tissue_boundary",
            "target_nuclei_binary",
            "target_nuclei_density",
            "target_nuclei_distance",
        )


@dataclass
class CrossV5TissueBank:
    """Reference-derived tissue appearance bank.

    Shapes:
    - ``prototypes``: ``(B, C, D)``
    - ``local_tokens``: ``(B, C, K, D)``
    - ``class_present``: ``(B, C)``
    - ``class_mass``: ``(B, C)`` in effective full-token units after pooling thresholding
    - ``token_class_ids``: ``(B, N)``
    - ``token_class_confidence``: ``(B, N)``
    """

    prototypes: torch.Tensor
    local_tokens: torch.Tensor
    class_present: torch.Tensor
    class_mass: torch.Tensor
    token_class_ids: torch.Tensor
    token_class_confidence: torch.Tensor


@dataclass
class CrossV5AdaLNOutput:
    """Output of class-wise appearance modulation."""

    hidden_states: torch.Tensor
    gamma: torch.Tensor
    beta: torch.Tensor
    source_prototypes: torch.Tensor


class CrossV5PriorPrototypeBank(nn.Module):
    """Learned fallback prototypes for tissue classes missing from reference."""

    def __init__(
        self,
        *,
        num_classes: int = NUM_COARSE,
        prototype_dim: int = 4,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}.")
        if prototype_dim <= 0:
            raise ValueError(f"prototype_dim must be positive, got {prototype_dim}.")
        self.num_classes = int(num_classes)
        self.prototype_dim = int(prototype_dim)
        self.init_std = float(init_std)
        self.prototypes = nn.Parameter(torch.empty(self.num_classes, self.prototype_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.normal_(self.prototypes, mean=0.0, std=self.init_std)

    def forward(self, batch_size: int | None = None) -> torch.Tensor:
        if batch_size is None:
            return self.prototypes
        return self.prototypes.unsqueeze(0).expand(int(batch_size), -1, -1)


class CrossV5RefBankBuilder(nn.Module):
    """Build tissue appearance banks from low-level reference appearance cues.

    V5-min defaults to HED stain-stat prototypes from reference RGB masks, not
    semantic UNI-style tokens. ``reference_tokens`` are expected to be shallow
    VGG/texture tokens and are used for local high-frequency tokens, with an
    optional class-pooled texture prototype concatenated to the HED stats.
    """

    def __init__(
        self,
        *,
        num_classes: int = NUM_COARSE,
        local_tokens_per_class: int = 4,
        min_presence_mass: float = 1.0,
        prototype_confidence_threshold: float = 0.5,
        prototype_pooling: str = "hard",
        prototype_source: str = "hed_stats",
        hed_channels: int = 2,
        include_hed_covariance: bool = False,
        texture_stat_kind: str = "var",
        eps: float = 1e-6,
        stain_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}.")
        if local_tokens_per_class <= 0:
            raise ValueError(
                f"local_tokens_per_class must be positive, got {local_tokens_per_class}."
            )
        self.num_classes = int(num_classes)
        self.local_tokens_per_class = int(local_tokens_per_class)
        self.min_presence_mass = float(min_presence_mass)
        self.prototype_confidence_threshold = float(prototype_confidence_threshold)
        self.prototype_pooling = _normalize_prototype_pooling(prototype_pooling)
        self.prototype_source = _normalize_prototype_source(prototype_source)
        self.hed_channels = _validate_hed_channels(hed_channels)
        self.include_hed_covariance = bool(include_hed_covariance)
        self.texture_stat_kind = _normalize_texture_stat_kind(texture_stat_kind)
        self.eps = float(eps)
        self.stain_eps = float(stain_eps)

    def forward(
        self,
        *,
        reference_tokens: torch.Tensor,
        reference_image: torch.Tensor | None = None,
        reference_class_ids: torch.Tensor | None = None,
        token_class_probs: torch.Tensor | None = None,
        token_height: int | None = None,
        token_width: int | None = None,
    ) -> CrossV5TissueBank:
        """Return a tissue bank for ``reference_tokens``.

        Args:
            reference_tokens: shallow texture tokens with shape ``(B, N, D)``.
            reference_class_ids: pixel/grid class IDs with shape ``(B,H,W)`` or
                ``(B,1,H,W)``. Required unless ``token_class_probs`` is provided.
            reference_image: reference RGB image with shape ``(B,3,H,W)``.
                Required when ``prototype_source`` includes ``hed_stats``.
            token_class_probs: optional precomputed class probabilities with
                shape ``(B,N,C)``.
            token_height/token_width: token grid size used to downsample
                ``reference_class_ids``. If omitted and possible, it is inferred.
        """
        if reference_tokens.ndim != 3:
            raise ValueError(
                "reference_tokens must have shape (B,N,D), "
                f"got {tuple(reference_tokens.shape)}."
            )
        batch_size, token_count, token_dim = reference_tokens.shape
        if token_class_probs is None:
            if reference_class_ids is None:
                raise ValueError("reference_class_ids are required when token_class_probs is not provided.")
            token_height, token_width = _resolve_token_grid(
                token_count=token_count,
                class_ids=reference_class_ids,
                token_height=token_height,
                token_width=token_width,
            )
            token_class_ids, token_confidence, token_class_probs = build_cross_v5_token_class_probs(
                class_ids=reference_class_ids,
                num_classes=self.num_classes,
                token_height=token_height,
                token_width=token_width,
            )
        else:
            if token_class_probs.ndim != 3:
                raise ValueError(
                    "token_class_probs must have shape (B,N,C), "
                    f"got {tuple(token_class_probs.shape)}."
                )
            if token_class_probs.shape[:2] != (batch_size, token_count):
                raise ValueError(
                    "token_class_probs must match reference_tokens on batch/token dims, "
                    f"got {tuple(token_class_probs.shape)} vs {tuple(reference_tokens.shape)}."
                )
            if token_class_probs.shape[2] != self.num_classes:
                raise ValueError(
                    f"token_class_probs has {token_class_probs.shape[2]} classes; "
                    f"expected {self.num_classes}."
                )
            token_confidence, token_class_ids = token_class_probs.max(dim=-1)

        token_class_probs = token_class_probs.to(device=reference_tokens.device, dtype=reference_tokens.dtype)
        token_class_ids = token_class_ids.to(device=reference_tokens.device, dtype=torch.long)
        token_confidence = token_confidence.to(device=reference_tokens.device, dtype=reference_tokens.dtype)

        prototype_weights = _build_prototype_pooling_weights(
            token_class_probs=token_class_probs,
            token_class_ids=token_class_ids,
            token_confidence=token_confidence,
            mode=self.prototype_pooling,
            confidence_threshold=self.prototype_confidence_threshold,
        )
        class_mass = prototype_weights.sum(dim=1)
        token_prototypes = torch.einsum("bnc,bnd->bcd", prototype_weights, reference_tokens)
        token_prototypes = token_prototypes / class_mass.clamp_min(1e-6).unsqueeze(-1)
        class_present = class_mass >= self.min_presence_mass
        prototypes = self._build_prototypes(
            reference_tokens=reference_tokens,
            token_prototypes=token_prototypes,
            prototype_weights=prototype_weights,
            class_mass=class_mass,
            reference_image=reference_image,
            reference_class_ids=reference_class_ids,
        )

        local_tokens = _select_topk_class_tokens(
            reference_tokens=reference_tokens,
            token_class_probs=token_class_probs,
            fallback_tokens=token_prototypes,
            class_present=class_present,
            k=self.local_tokens_per_class,
        )
        return CrossV5TissueBank(
            prototypes=prototypes,
            local_tokens=local_tokens,
            class_present=class_present,
            class_mass=class_mass,
            token_class_ids=token_class_ids,
            token_class_confidence=token_confidence,
        )

    def _build_prototypes(
        self,
        *,
        reference_tokens: torch.Tensor,
        token_prototypes: torch.Tensor,
        prototype_weights: torch.Tensor,
        class_mass: torch.Tensor,
        reference_image: torch.Tensor | None,
        reference_class_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if "hed_stats" in self.prototype_source:
            if reference_image is None:
                raise ValueError("reference_image is required when prototype_source includes 'hed_stats'.")
            if reference_class_ids is None:
                raise ValueError("reference_class_ids are required when building HED stain-stat prototypes.")
            hed = build_cross_v5_hed_stat_prototypes(
                reference_image=reference_image,
                reference_class_ids=reference_class_ids,
                num_classes=self.num_classes,
                hed_channels=self.hed_channels,
                include_covariance=self.include_hed_covariance,
                eps=self.eps,
                stain_eps=self.stain_eps,
            ).to(device=reference_tokens.device, dtype=reference_tokens.dtype)
            parts.append(hed)
        if "texture_stats" in self.prototype_source:
            # Low-dimensional texture STRENGTH statistics from the same shallow
            # texture features the Gram texture loss uses. Per-class per-channel
            # variance (and optionally mean) is the diagonal-energy proxy: it
            # captures "how strong/coarse" the texture is without the spatial
            # layout, and it stays the same dim as the feature channels (not the
            # high-dim flattened token), so it does not swamp the HED color dims.
            tex = build_cross_v5_texture_stat_prototypes(
                reference_tokens=reference_tokens,
                prototype_weights=prototype_weights,
                token_prototypes=token_prototypes,
                kind=self.texture_stat_kind,
                eps=self.eps,
            )
            parts.append(tex)
        if "token_pool" in self.prototype_source:
            parts.append(token_prototypes)
        if not parts:
            raise ValueError(f"prototype_source {self.prototype_source!r} produced no prototype parts.")
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]


class CrossV5AdaLNModulator(nn.Module):
    """Apply class-wise bank prototypes through normalization scale/shift.

    This module is deliberately not a residual cross-attention block. It is the
    V5-min interface for wiring per-class appearance into AdaLN-style locations
    so ControlNet remains the geometry/content residual path.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        prototype_dim: int | None = None,
        mlp_hidden_dim: int | None = None,
        initial_gamma: float = 0.05,
        output_init_std: float = 0.02,
        use_internal_norm: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        self.hidden_dim = int(hidden_dim)
        self.prototype_dim = int(prototype_dim or hidden_dim)
        self.mlp_hidden_dim = int(mlp_hidden_dim or max(self.hidden_dim, self.prototype_dim))
        self.initial_gamma = float(initial_gamma)
        self.output_init_std = float(output_init_std)
        self.prototype_norm = nn.LayerNorm(self.prototype_dim)
        self.hidden_norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False) if use_internal_norm else None
        self.mlp = nn.Sequential(
            nn.Linear(self.prototype_dim, self.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden_dim, 2 * self.hidden_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.output_init_std <= 0.0:
            raise ValueError(f"output_init_std must be > 0 for V5 appearance modulation, got {self.output_init_std}.")
        final = self.mlp[-1]
        assert isinstance(final, nn.Linear)
        with torch.no_grad():
            nn.init.normal_(final.weight, mean=0.0, std=self.output_init_std)
            final.bias.zero_()
            final.bias[: self.hidden_dim].fill_(self.initial_gamma)

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        target_class_ids: torch.Tensor,
        bank: CrossV5TissueBank,
        fallback_prototypes: torch.Tensor | None = None,
        target_structure_tokens: torch.Tensor | None = None,
    ) -> CrossV5AdaLNOutput:
        del target_structure_tokens
        if hidden_states.ndim != 3:
            raise ValueError(f"hidden_states must have shape (B,N,D), got {tuple(hidden_states.shape)}.")
        if hidden_states.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"hidden_states last dim {hidden_states.shape[-1]} does not match hidden_dim {self.hidden_dim}."
            )
        class_ids = _ensure_batched_tokens(target_class_ids, name="target_class_ids").to(
            device=hidden_states.device,
            dtype=torch.long,
        )
        if class_ids.shape != hidden_states.shape[:2]:
            raise ValueError(
                f"target_class_ids shape {tuple(class_ids.shape)} must match hidden token shape {tuple(hidden_states.shape[:2])}."
            )

        prototypes = bank.prototypes.to(device=hidden_states.device, dtype=hidden_states.dtype)
        if prototypes.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                f"bank batch size {prototypes.shape[0]} does not match hidden batch size {hidden_states.shape[0]}."
            )
        if prototypes.shape[-1] != self.prototype_dim:
            raise ValueError(
                f"bank prototype dim {prototypes.shape[-1]} does not match prototype_dim {self.prototype_dim}."
            )
        if fallback_prototypes is not None:
            prototypes = _merge_with_fallback_prototypes(
                prototypes=prototypes,
                class_present=bank.class_present.to(device=hidden_states.device),
                fallback_prototypes=fallback_prototypes.to(device=hidden_states.device, dtype=hidden_states.dtype),
            )

        source = gather_cross_v5_class_values(prototypes, class_ids)
        gamma_beta = self.mlp(self.prototype_norm(source.to(dtype=self.prototype_norm.weight.dtype))).to(
            dtype=hidden_states.dtype
        )
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        base = self.hidden_norm(hidden_states) if self.hidden_norm is not None else hidden_states
        modulated = base * (1.0 + gamma) + beta
        return CrossV5AdaLNOutput(
            hidden_states=modulated,
            gamma=gamma,
            beta=beta,
            source_prototypes=source,
        )


class CrossV5SpatialAdaLNModulator(nn.Module):
    """SEAN-style spatial AdaLN from style code plus target structure tokens.

    This is the V5 texture path that avoids residual cross-attention. The
    modulator generates per-token ``gamma/beta`` from reference style
    prototypes and target geometry/structure tokens, so texture information
    enters through normalization scale/shift instead of an additive residual.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        prototype_dim: int,
        structure_dim: int,
        mlp_hidden_dim: int | None = None,
        initial_gamma: float = 0.05,
        output_init_std: float = 0.02,
        use_internal_norm: bool = True,
        require_structure_tokens: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if prototype_dim <= 0:
            raise ValueError(f"prototype_dim must be positive, got {prototype_dim}.")
        if structure_dim <= 0:
            raise ValueError(f"structure_dim must be positive, got {structure_dim}.")
        self.hidden_dim = int(hidden_dim)
        self.prototype_dim = int(prototype_dim)
        self.structure_dim = int(structure_dim)
        self.mlp_hidden_dim = int(mlp_hidden_dim or max(self.hidden_dim, self.prototype_dim + self.structure_dim))
        self.initial_gamma = float(initial_gamma)
        self.output_init_std = float(output_init_std)
        self.require_structure_tokens = bool(require_structure_tokens)
        self.prototype_norm = nn.LayerNorm(self.prototype_dim)
        self.structure_norm = nn.LayerNorm(self.structure_dim)
        self.hidden_norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False) if use_internal_norm else None
        self.mlp = nn.Sequential(
            nn.Linear(self.prototype_dim + self.structure_dim, self.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden_dim, 2 * self.hidden_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.output_init_std <= 0.0:
            raise ValueError(f"output_init_std must be > 0 for V5 spatial appearance modulation, got {self.output_init_std}.")
        final = self.mlp[-1]
        assert isinstance(final, nn.Linear)
        with torch.no_grad():
            nn.init.normal_(final.weight, mean=0.0, std=self.output_init_std)
            final.bias.zero_()
            final.bias[: self.hidden_dim].fill_(self.initial_gamma)

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        target_class_ids: torch.Tensor,
        bank: CrossV5TissueBank,
        fallback_prototypes: torch.Tensor | None = None,
        target_structure_tokens: torch.Tensor | None = None,
    ) -> CrossV5AdaLNOutput:
        if hidden_states.ndim != 3:
            raise ValueError(f"hidden_states must have shape (B,N,D), got {tuple(hidden_states.shape)}.")
        if hidden_states.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"hidden_states last dim {hidden_states.shape[-1]} does not match hidden_dim {self.hidden_dim}."
            )
        class_ids = _ensure_batched_tokens(target_class_ids, name="target_class_ids").to(
            device=hidden_states.device,
            dtype=torch.long,
        )
        if class_ids.shape != hidden_states.shape[:2]:
            raise ValueError(
                f"target_class_ids shape {tuple(class_ids.shape)} must match hidden token shape {tuple(hidden_states.shape[:2])}."
            )
        if target_structure_tokens is None:
            if self.require_structure_tokens:
                raise ValueError("target_structure_tokens are required for CrossV5SpatialAdaLNModulator.")
            target_structure_tokens = hidden_states.new_zeros(hidden_states.shape[:2] + (self.structure_dim,))
        target_structure_tokens = target_structure_tokens.to(device=hidden_states.device, dtype=hidden_states.dtype)
        if target_structure_tokens.shape != hidden_states.shape[:2] + (self.structure_dim,):
            raise ValueError(
                "target_structure_tokens must have shape "
                f"{tuple(hidden_states.shape[:2] + (self.structure_dim,))}, got {tuple(target_structure_tokens.shape)}."
            )

        prototypes = bank.prototypes.to(device=hidden_states.device, dtype=hidden_states.dtype)
        if prototypes.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                f"bank batch size {prototypes.shape[0]} does not match hidden batch size {hidden_states.shape[0]}."
            )
        if prototypes.shape[-1] != self.prototype_dim:
            raise ValueError(
                f"bank prototype dim {prototypes.shape[-1]} does not match prototype_dim {self.prototype_dim}."
            )
        if fallback_prototypes is not None:
            prototypes = _merge_with_fallback_prototypes(
                prototypes=prototypes,
                class_present=bank.class_present.to(device=hidden_states.device),
                fallback_prototypes=fallback_prototypes.to(device=hidden_states.device, dtype=hidden_states.dtype),
            )

        source = gather_cross_v5_class_values(prototypes, class_ids)
        source_norm = self.prototype_norm(source.to(dtype=self.prototype_norm.weight.dtype))
        structure_norm = self.structure_norm(target_structure_tokens.to(dtype=self.structure_norm.weight.dtype))
        gamma_beta = self.mlp(torch.cat([source_norm, structure_norm], dim=-1)).to(dtype=hidden_states.dtype)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        base = self.hidden_norm(hidden_states) if self.hidden_norm is not None else hidden_states
        modulated = base * (1.0 + gamma) + beta
        return CrossV5AdaLNOutput(
            hidden_states=modulated,
            gamma=gamma,
            beta=beta,
            source_prototypes=source,
        )


def build_cross_v5_token_class_probs(
    *,
    class_ids: torch.Tensor,
    num_classes: int,
    token_height: int,
    token_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Downsample class IDs into token-level class probabilities."""

    if token_height <= 0 or token_width <= 0:
        raise ValueError(f"token grid must be positive, got {token_height}x{token_width}.")
    class_ids = _ensure_batched_hw(class_ids, name="class_ids").long()
    _validate_id_range(class_ids, low=0, high=int(num_classes) - 1, name="class_ids")
    one_hot = F.one_hot(class_ids, num_classes=int(num_classes)).permute(0, 3, 1, 2).float()
    pooled = F.adaptive_avg_pool2d(one_hot, output_size=(int(token_height), int(token_width)))
    probs = pooled.permute(0, 2, 3, 1).reshape(class_ids.shape[0], int(token_height) * int(token_width), int(num_classes))
    confidence, token_ids = probs.max(dim=-1)
    return token_ids.long(), confidence, probs


def build_cross_v5_hed_stat_prototypes(
    *,
    reference_image: torch.Tensor,
    reference_class_ids: torch.Tensor,
    num_classes: int,
    hed_channels: int = 2,
    include_covariance: bool = False,
    eps: float = 1e-6,
    stain_eps: float = 1e-3,
) -> torch.Tensor:
    """Build per-class HED stain statistics for V5 appearance prototypes.

    The default prototype is ``[mean_H, mean_E, std_H, std_E]``. Enabling
    covariance appends the H/E covariance, still staying in low-level stain
    space aligned with the V5 color fidelity loss.
    """

    channels = _validate_hed_channels(hed_channels)
    if reference_image.ndim != 4 or reference_image.shape[1] != 3:
        raise ValueError(f"reference_image must have shape (B,3,H,W), got {tuple(reference_image.shape)}.")
    class_ids = _ensure_batched_hw(reference_class_ids, name="reference_class_ids").to(
        device=reference_image.device,
        dtype=torch.long,
    )
    if class_ids.shape[0] != reference_image.shape[0]:
        raise ValueError(
            f"reference_class_ids batch size {class_ids.shape[0]} does not match image batch size {reference_image.shape[0]}."
        )
    _validate_id_range(class_ids, low=0, high=int(num_classes) - 1, name="reference_class_ids")
    class_ids = F.interpolate(
        class_ids.unsqueeze(1).float(),
        size=tuple(int(v) for v in reference_image.shape[-2:]),
        mode="nearest",
    ).squeeze(1).long()

    hed = cross_v5_rgb_to_hed_concentrations(
        reference_image,
        eps=float(eps),
        stain_eps=float(stain_eps),
    )[:, :channels]
    one_hot = F.one_hot(class_ids, num_classes=int(num_classes)).permute(0, 3, 1, 2).to(dtype=hed.dtype)
    mass = one_hot.flatten(2).sum(dim=-1).clamp_min(float(eps))
    means = torch.einsum("bkhw,bchw->bkc", one_hot, hed) / mass.unsqueeze(-1)
    second = torch.einsum("bkhw,bchw->bkc", one_hot, hed * hed) / mass.unsqueeze(-1)
    stds = (second - means * means).clamp_min(0.0).sqrt()
    parts = [means, stds]
    if include_covariance:
        if channels < 2:
            covariance = means.new_zeros(means.shape[:2] + (1,))
        else:
            he = hed[:, 0:1] * hed[:, 1:2]
            mean_he = torch.einsum("bkhw,bchw->bkc", one_hot, he) / mass.unsqueeze(-1)
            covariance = mean_he - means[:, :, 0:1] * means[:, :, 1:2]
        parts.append(covariance)
    return torch.cat(parts, dim=-1)


def build_cross_v5_texture_stat_prototypes(
    *,
    reference_tokens: torch.Tensor,
    prototype_weights: torch.Tensor,
    token_prototypes: torch.Tensor,
    kind: str = "var",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-class low-dim texture STRENGTH statistics from shallow texture tokens.

    Uses the SAME class-assignment weights as the prototype pooling, so the
    statistics describe exactly the tokens that define each class bank. The
    per-channel variance is a lightweight diagonal-energy proxy aligned with the
    Gram texture loss (Gram diagonal == per-channel energy), capturing texture
    coarseness/strength without spatial layout. Output dim is the feature-channel
    count (``var``) or twice it (``mean_var``) -- both stay low relative to the
    flattened token, so they do not swamp the few HED color dims.

    Args:
        reference_tokens: shallow texture tokens ``(B, N, D)``.
        prototype_weights: per-token per-class weights ``(B, N, C)`` (hard or
            soft), the same weights used to build ``token_prototypes``.
        token_prototypes: per-class weighted token mean ``(B, C, D)``.
        kind: ``"var"`` -> per-channel variance only; ``"mean_var"`` -> mean
            concatenated with variance.
    """

    if reference_tokens.ndim != 3:
        raise ValueError(f"reference_tokens must have shape (B,N,D), got {tuple(reference_tokens.shape)}.")
    if prototype_weights.ndim != 3:
        raise ValueError(f"prototype_weights must have shape (B,N,C), got {tuple(prototype_weights.shape)}.")
    if token_prototypes.ndim != 3:
        raise ValueError(f"token_prototypes must have shape (B,C,D), got {tuple(token_prototypes.shape)}.")
    weights = prototype_weights.to(device=reference_tokens.device, dtype=reference_tokens.dtype)
    mean = token_prototypes.to(device=reference_tokens.device, dtype=reference_tokens.dtype)
    mass = weights.sum(dim=1).clamp_min(float(eps))                      # (B, C)
    # E[x^2] per class per channel
    second = torch.einsum("bnc,bnd->bcd", weights, reference_tokens * reference_tokens)
    second = second / mass.unsqueeze(-1)                                 # (B, C, D)
    variance = (second - mean * mean).clamp_min(0.0)                     # (B, C, D)
    if kind == "var":
        return variance
    if kind == "mean_var":
        return torch.cat([mean, variance], dim=-1)
    raise ValueError(f"Unsupported texture_stat_kind {kind!r}; choose 'var' or 'mean_var'.")


def _normalize_texture_stat_kind(value: str) -> str:
    normalized = str(value or "var").strip().lower().replace("-", "_")
    if normalized in {"var", "variance"}:
        return "var"
    if normalized in {"mean_var", "meanvar", "mean_variance"}:
        return "mean_var"
    raise ValueError(f"Unsupported texture_stat_kind {value!r}; choose 'var' or 'mean_var'.")


def cross_v5_rgb_to_hed_concentrations(
    rgb: torch.Tensor,
    *,
    eps: float = 1e-6,
    stain_eps: float = 1e-3,
) -> torch.Tensor:
    """Approximate RGB to HED concentrations, shared by V5 bank construction."""

    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must have shape (B,3,H,W), got {tuple(rgb.shape)}.")
    x = rgb.float()
    if float(x.detach().min().item()) < -0.01:
        x = (x + 1.0) * 0.5
    x = x.clamp(0.0, 1.0)
    od = -torch.log(x + max(float(stain_eps), float(eps)))
    stain_matrix = x.new_tensor(
        [
            [0.650, 0.704, 0.286],
            [0.072, 0.990, 0.105],
            [0.268, 0.570, 0.776],
        ]
    )
    inv = torch.linalg.pinv(stain_matrix)
    return torch.einsum("cd,bdhw->bchw", inv, od)


def build_cross_v5_geometry_control_condition(
    *,
    target_tissue_mask: torch.Tensor,
    target_nuclei_mask: torch.Tensor,
    output_height: int,
    output_width: int,
    distance_iterations: int = 16,
) -> torch.Tensor:
    """Build class-agnostic geometry maps for V5 ControlNet.

    The returned channels are ``[tissue_boundary, nuclei_binary,
    nuclei_density, nuclei_distance]``. They deliberately discard tissue and
    cell type IDs so ControlNet can only carry target structure, while class IDs
    remain available to the V5 bank/AdaLN path.
    """

    output_height = int(output_height)
    output_width = int(output_width)
    if output_height <= 0 or output_width <= 0:
        raise ValueError(f"output size must be positive, got {output_height}x{output_width}.")

    tissue = _ensure_mask_bhw(target_tissue_mask, name="target_tissue_mask")
    nuclei = _ensure_mask_bhw(target_nuclei_mask, name="target_nuclei_mask")
    if tissue.shape[0] != nuclei.shape[0]:
        raise ValueError(
            f"target_tissue_mask batch size {tissue.shape[0]} does not match target_nuclei_mask {nuclei.shape[0]}."
        )

    tissue_low = _resize_label_mask(tissue, output_height, output_width)
    tissue_boundary = _label_boundary_map(tissue_low)

    nuclei_binary_full = (nuclei > 0).float().unsqueeze(1)
    nuclei_binary = F.interpolate(
        nuclei_binary_full,
        size=(output_height, output_width),
        mode="nearest",
    ).float()
    nuclei_density = F.interpolate(
        nuclei_binary_full,
        size=(output_height, output_width),
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    nuclei_density = _blur_unit_map(nuclei_density)
    nuclei_distance = _approx_normalized_distance_to_foreground(
        nuclei_binary,
        iterations=max(1, int(distance_iterations)),
    )

    return torch.cat(
        [
            tissue_boundary,
            nuclei_binary.clamp(0.0, 1.0),
            nuclei_density.clamp(0.0, 1.0),
            nuclei_distance.clamp(0.0, 1.0),
        ],
        dim=1,
    )


def build_cross_v5_spatial_structure_tokens(
    *,
    class_ids: torch.Tensor,
    num_classes: int,
    token_height: int,
    token_width: int,
    geometry_maps: torch.Tensor | None = None,
    include_xy: bool = True,
) -> torch.Tensor:
    """Build token-level target structure features for SEAN-style AdaLN.

    The output contains class probabilities on the token grid, optional pooled
    geometry maps such as boundary/distance/centroid heatmaps, and normalized
    XY coordinates. It is a structure/layout signal, not an appearance residual.
    """

    _, _, class_probs = build_cross_v5_token_class_probs(
        class_ids=class_ids,
        num_classes=num_classes,
        token_height=token_height,
        token_width=token_width,
    )
    parts = [class_probs]
    batch_size = class_probs.shape[0]
    device = class_probs.device
    dtype = class_probs.dtype
    if geometry_maps is not None:
        if geometry_maps.ndim == 3:
            geometry_maps = geometry_maps.unsqueeze(1)
        if geometry_maps.ndim != 4:
            raise ValueError(f"geometry_maps must have shape (B,G,H,W) or (B,H,W), got {tuple(geometry_maps.shape)}.")
        if geometry_maps.shape[0] != batch_size:
            raise ValueError(
                f"geometry_maps batch size {geometry_maps.shape[0]} does not match class_ids batch size {batch_size}."
            )
        pooled = F.adaptive_avg_pool2d(
            geometry_maps.to(device=device, dtype=dtype),
            output_size=(int(token_height), int(token_width)),
        )
        parts.append(pooled.permute(0, 2, 3, 1).reshape(batch_size, int(token_height) * int(token_width), -1))
    if include_xy:
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, int(token_height), device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, int(token_width), device=device, dtype=dtype),
            indexing="ij",
        )
        xy = torch.stack([xx, yy], dim=-1).reshape(1, int(token_height) * int(token_width), 2)
        parts.append(xy.expand(batch_size, -1, -1))
    return torch.cat(parts, dim=-1)


def _ensure_mask_bhw(mask: torch.Tensor, *, name: str) -> torch.Tensor:
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 3:
        raise ValueError(f"{name} must have shape (B,H,W) or (B,1,H,W), got {tuple(mask.shape)}.")
    return mask


def _resize_label_mask(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return F.interpolate(
        mask.float().unsqueeze(1),
        size=(int(height), int(width)),
        mode="nearest",
    )[:, 0].long()


def _label_boundary_map(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 3:
        raise ValueError(f"mask must have shape (B,H,W), got {tuple(mask.shape)}.")
    boundary = torch.zeros_like(mask, dtype=torch.bool)
    boundary[:, :, 1:] |= mask[:, :, 1:] != mask[:, :, :-1]
    boundary[:, :, :-1] |= mask[:, :, 1:] != mask[:, :, :-1]
    boundary[:, 1:, :] |= mask[:, 1:, :] != mask[:, :-1, :]
    boundary[:, :-1, :] |= mask[:, 1:, :] != mask[:, :-1, :]
    return F.max_pool2d(boundary.float().unsqueeze(1), kernel_size=3, stride=1, padding=1).clamp(0.0, 1.0)


def _blur_unit_map(values: torch.Tensor) -> torch.Tensor:
    height, width = values.shape[-2:]
    if min(height, width) >= 5:
        kernel = 5
    elif min(height, width) >= 3:
        kernel = 3
    else:
        return values
    return F.avg_pool2d(values, kernel_size=kernel, stride=1, padding=kernel // 2)


def _approx_normalized_distance_to_foreground(binary: torch.Tensor, *, iterations: int) -> torch.Tensor:
    if binary.ndim != 4 or binary.shape[1] != 1:
        raise ValueError(f"binary must have shape (B,1,H,W), got {tuple(binary.shape)}.")
    visited = binary > 0.5
    distance = binary.new_zeros(binary.shape)
    remaining = ~visited
    if not bool(visited.any().item()):
        return binary.new_ones(binary.shape)
    if not bool(remaining.any().item()):
        return distance
    steps = max(1, min(int(iterations), max(int(binary.shape[-2]), int(binary.shape[-1]))))
    for step in range(1, steps + 1):
        expanded = F.max_pool2d(visited.float(), kernel_size=3, stride=1, padding=1) > 0.0
        newly_reached = expanded & remaining
        if bool(newly_reached.any().item()):
            distance = torch.where(
                newly_reached,
                distance.new_full(distance.shape, float(step) / float(steps)),
                distance,
            )
        visited = expanded
        remaining = remaining & ~newly_reached
        if not bool(remaining.any().item()):
            break
    return torch.where(remaining, distance.new_ones(distance.shape), distance)


def gather_cross_v5_class_values(values: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
    """Gather ``values[:, class_ids]`` for batched target token class IDs."""

    if values.ndim != 3:
        raise ValueError(f"values must have shape (B,C,D), got {tuple(values.shape)}.")
    class_ids = _ensure_batched_tokens(class_ids, name="class_ids").to(device=values.device, dtype=torch.long)
    if class_ids.shape[0] != values.shape[0]:
        raise ValueError(
            f"class_ids batch size {class_ids.shape[0]} does not match values batch size {values.shape[0]}."
        )
    _validate_id_range(class_ids, low=0, high=values.shape[1] - 1, name="class_ids")
    gather_index = class_ids.unsqueeze(-1).expand(-1, -1, values.shape[-1])
    return torch.gather(values, dim=1, index=gather_index)


def _select_topk_class_tokens(
    *,
    reference_tokens: torch.Tensor,
    token_class_probs: torch.Tensor,
    fallback_tokens: torch.Tensor,
    class_present: torch.Tensor,
    k: int,
) -> torch.Tensor:
    batch_size, token_count, token_dim = reference_tokens.shape
    class_count = token_class_probs.shape[-1]
    effective_k = min(int(k), int(token_count))
    scores = token_class_probs.transpose(1, 2)
    top_scores, top_indices = torch.topk(scores, k=effective_k, dim=-1)
    expanded_tokens = reference_tokens.unsqueeze(1).expand(batch_size, class_count, token_count, token_dim)
    gather_index = top_indices.unsqueeze(-1).expand(batch_size, class_count, effective_k, token_dim)
    selected = torch.gather(expanded_tokens, dim=2, index=gather_index)
    selected = torch.where(
        ((top_scores > 0.0) & class_present.unsqueeze(-1)).unsqueeze(-1),
        selected,
        fallback_tokens.unsqueeze(2).expand_as(selected),
    )
    if effective_k == k:
        return selected
    padding = fallback_tokens.unsqueeze(2).expand(batch_size, class_count, int(k) - effective_k, token_dim)
    return torch.cat([selected, padding], dim=2)


def _build_prototype_pooling_weights(
    *,
    token_class_probs: torch.Tensor,
    token_class_ids: torch.Tensor,
    token_confidence: torch.Tensor,
    mode: str,
    confidence_threshold: float,
) -> torch.Tensor:
    class_count = token_class_probs.shape[-1]
    if mode == "soft":
        weights = token_class_probs
        if confidence_threshold > 0.0:
            weights = torch.where(
                weights >= float(confidence_threshold),
                weights,
                torch.zeros_like(weights),
            )
        return weights
    if mode == "hard":
        hard = F.one_hot(token_class_ids, num_classes=class_count).to(dtype=token_class_probs.dtype)
        keep = token_confidence >= float(confidence_threshold)
        return hard * keep.unsqueeze(-1).to(dtype=token_class_probs.dtype)
    raise ValueError(f"Unsupported prototype pooling mode {mode!r}.")


def _normalize_prototype_pooling(value: str) -> str:
    normalized = str(value or "hard").strip().lower().replace("-", "_")
    aliases = {
        "hard": "hard",
        "argmax": "hard",
        "dominant": "hard",
        "soft": "soft",
        "weighted": "soft",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported prototype_pooling {value!r}; choose 'hard' or 'soft'.")
    return aliases[normalized]


def _normalize_prototype_source(value: str) -> str:
    normalized = str(value or "hed_stats").strip().lower().replace("-", "_").replace("+", "_")
    aliases = {
        "token": "token_pool",
        "tokens": "token_pool",
        "token_pool": "token_pool",
        "texture": "token_pool",
        "texture_pool": "token_pool",
        "hed": "hed_stats",
        "he": "hed_stats",
        "hed_stats": "hed_stats",
        "stain": "hed_stats",
        "stain_stats": "hed_stats",
        "hed_token": "hed_stats_token_pool",
        "hed_tokens": "hed_stats_token_pool",
        "hed_token_pool": "hed_stats_token_pool",
        "hed_stats_token": "hed_stats_token_pool",
        "hed_stats_tokens": "hed_stats_token_pool",
        "hed_stats_token_pool": "hed_stats_token_pool",
        # Low-dimensional texture STATISTICS (not the high-dim token pool).
        # These stay in the same low-level space as the Gram texture loss and
        # do not numerically swamp the few HED color dims.
        "texstat": "texture_stats",
        "texture_stat": "texture_stats",
        "texture_stats": "texture_stats",
        "hed_texture": "hed_stats_texture_stats",
        "hed_texstat": "hed_stats_texture_stats",
        "hed_stats_texture": "hed_stats_texture_stats",
        "hed_stats_texstat": "hed_stats_texture_stats",
        "hed_stats_texture_stats": "hed_stats_texture_stats",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported prototype_source {value!r}; choose 'hed_stats', 'token_pool', "
            "'hed_stats+token_pool', 'texture_stats', or 'hed_stats+texture_stats'."
        )
    return aliases[normalized]


def _validate_hed_channels(value: int) -> int:
    channels = int(value)
    if channels < 1 or channels > 3:
        raise ValueError(f"hed_channels must be in [1, 3], got {value}.")
    return channels


def _merge_with_fallback_prototypes(
    *,
    prototypes: torch.Tensor,
    class_present: torch.Tensor,
    fallback_prototypes: torch.Tensor,
) -> torch.Tensor:
    if fallback_prototypes.ndim == 2:
        fallback_prototypes = fallback_prototypes.unsqueeze(0).expand(prototypes.shape[0], -1, -1)
    if fallback_prototypes.shape != prototypes.shape:
        raise ValueError(
            "fallback_prototypes must have shape (C,D) or (B,C,D), "
            f"got {tuple(fallback_prototypes.shape)} for prototypes {tuple(prototypes.shape)}."
        )
    return torch.where(class_present.unsqueeze(-1), prototypes, fallback_prototypes)


def _resolve_token_grid(
    *,
    token_count: int,
    class_ids: torch.Tensor,
    token_height: int | None,
    token_width: int | None,
) -> tuple[int, int]:
    if token_height is not None and token_width is not None:
        if int(token_height) * int(token_width) != int(token_count):
            raise ValueError(
                f"token_height*token_width must equal token count {token_count}, got {token_height}x{token_width}."
            )
        return int(token_height), int(token_width)
    spatial = class_ids.shape[-2:]
    if int(spatial[0]) * int(spatial[1]) == int(token_count):
        return int(spatial[0]), int(spatial[1])
    side = int(round(float(token_count) ** 0.5))
    if side * side == int(token_count):
        return side, side
    raise ValueError(
        "token_height/token_width are required when token count is not square and class_ids are not already token-grid sized."
    )


def _ensure_batched_hw(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape (B,H,W), (B,1,H,W), or (H,W), got {tuple(value.shape)}.")
    return value


def _ensure_batched_tokens(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape (B,N) or (N,), got {tuple(value.shape)}.")
    return value


def _validate_id_range(value: torch.Tensor, *, low: int, high: int, name: str) -> None:
    if value.numel() == 0:
        return
    min_id = int(value.min().item())
    max_id = int(value.max().item())
    if min_id < int(low) or max_id > int(high):
        raise ValueError(f"{name} IDs out of range: got [{min_id}, {max_id}], expected [{low}, {high}].")


__all__ = [
    "CrossV5AdaLNModulator",
    "CrossV5AdaLNOutput",
    "CrossV5GeometryControlSpec",
    "CrossV5PriorPrototypeBank",
    "CrossV5RefBankBuilder",
    "CrossV5SpatialAdaLNModulator",
    "CrossV5TissueBank",
    "build_cross_v5_geometry_control_condition",
    "build_cross_v5_hed_stat_prototypes",
    "build_cross_v5_spatial_structure_tokens",
    "build_cross_v5_texture_stat_prototypes",
    "build_cross_v5_token_class_probs",
    "cross_v5_rgb_to_hed_concentrations",
    "gather_cross_v5_class_values",
]

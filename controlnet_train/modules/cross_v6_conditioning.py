"""Cross V6-min-0b gamma-attention latent composer.

V6 keeps target class IDs out of the final ControlNet condition. Tissue and
nuclei labels are used only inside this composer to route reference-derived
VAE latent values into the target layout. The exposed condition is:

``z_ref_to_target`` plus class-agnostic geometry and retrieval diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_config import NUM_COARSE
from controlnet_train.training.conditioning import packed_control_channels


@dataclass(frozen=True)
class CrossV6Min0bControlSpec:
    """Final V6 ControlNet condition layout.

    The raw condition deliberately contains no target tissue/cell one-hot maps:
    ``latent_channels`` channels of reference-derived latent memory plus six
    scalar maps.
    """

    latent_channels: int = 16
    scalar_geometry_channels: int = 6

    @property
    def raw_channels(self) -> int:
        return int(self.latent_channels) + int(self.scalar_geometry_channels)

    @property
    def packed_channels(self) -> int:
        return packed_control_channels(self.raw_channels)

    @property
    def condition_order(self) -> tuple[str, ...]:
        return (
            "z_ref_to_target",
            "nuclei_binary",
            "nuclei_boundary",
            "nuclei_distance_map",
            "tissue_boundary",
            "retrieval_confidence_map",
            "missing_class_map",
        )


@dataclass
class CrossV6Min0bComposerCondition:
    ref_latent: torch.Tensor
    ref_tissue_mask: torch.Tensor
    ref_nuclei_mask: torch.Tensor
    target_tissue_mask: torch.Tensor
    target_nuclei_mask: torch.Tensor
    nuclei_binary: torch.Tensor
    nuclei_boundary: torch.Tensor
    nuclei_distance_map: torch.Tensor
    tissue_boundary: torch.Tensor


@dataclass
class CrossV6Min0bComposerOutput:
    z_tissue_pool: torch.Tensor
    z_tissue_attn: torch.Tensor
    gamma: torch.Tensor
    z_tissue_target: torch.Tensor
    z_nuclei_target: torch.Tensor
    z_ref_to_target: torch.Tensor
    retrieval_confidence_map: torch.Tensor
    missing_class_map: torch.Tensor


@dataclass
class CrossV6Min0bControlCondition:
    z_ref_to_target: torch.Tensor
    nuclei_binary: torch.Tensor
    nuclei_boundary: torch.Tensor
    nuclei_distance_map: torch.Tensor
    tissue_boundary: torch.Tensor
    retrieval_confidence_map: torch.Tensor
    missing_class_map: torch.Tensor


@dataclass
class CrossV6ClassInternalVarianceDiagnostics:
    class_mass: torch.Tensor
    token_variance: torch.Tensor
    pca_top1_energy: torch.Tensor
    mean_pairwise_distance: torch.Tensor


class CrossV6TargetLayoutEncoder(nn.Module):
    """Per-position query encoder from class-agnostic target geometry."""

    def __init__(
        self,
        *,
        geometry_channels: int = 4,
        query_dim: int = 64,
        include_xy: bool = True,
        output_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if geometry_channels <= 0:
            raise ValueError(f"geometry_channels must be positive, got {geometry_channels}.")
        if query_dim <= 0:
            raise ValueError(f"query_dim must be positive, got {query_dim}.")
        self.geometry_channels = int(geometry_channels)
        self.query_dim = int(query_dim)
        self.include_xy = bool(include_xy)
        self.input_channels = self.geometry_channels + (2 if self.include_xy else 0)
        self.proj = nn.Conv2d(self.input_channels, self.query_dim, kernel_size=1)
        self.output_init_std = float(output_init_std)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.normal_(self.proj.weight, mean=0.0, std=self.output_init_std)
            self.proj.bias.zero_()

    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        if geometry.ndim != 4:
            raise ValueError(f"geometry must have shape (B,G,H,W), got {tuple(geometry.shape)}.")
        if geometry.shape[1] != self.geometry_channels:
            raise ValueError(
                f"geometry has {geometry.shape[1]} channels; expected {self.geometry_channels}."
            )
        x = geometry
        if self.include_xy:
            x = torch.cat([x, _normalized_xy_grid_like(geometry)], dim=1)
        return self.proj(x)


class CrossV6Min0bLatentComposer(nn.Module):
    """Compose reference VAE latents onto the target mask layout.

    Tissue appearance uses a stable masked-pooling base plus a small non-zero
    gamma same-class attention residual. Nuclei appearance is a low-frequency
    mask-pooled residual, not an attention branch.
    """

    def __init__(
        self,
        *,
        latent_channels: int = 16,
        num_tissue_classes: int = NUM_COARSE,
        attention_dim: int | None = None,
        max_ref_tokens_per_class: int = 64,
        gamma_init: float = 5e-3,
        alpha: float = 0.3,
        min_tissue_pixels: float = 1.0,
        min_nuclei_pixels: float = 1.0,
        include_xy: bool = True,
        same_class_attention: bool = True,
        mask_probability_threshold: float = 1e-6,
        query_output_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if latent_channels <= 0:
            raise ValueError(f"latent_channels must be positive, got {latent_channels}.")
        if num_tissue_classes <= 0:
            raise ValueError(f"num_tissue_classes must be positive, got {num_tissue_classes}.")
        if max_ref_tokens_per_class <= 0:
            raise ValueError(
                f"max_ref_tokens_per_class must be positive, got {max_ref_tokens_per_class}."
            )
        attention_dim = int(attention_dim or latent_channels)
        if attention_dim <= 0:
            raise ValueError(f"attention_dim must be positive, got {attention_dim}.")
        if gamma_init == 0.0:
            raise ValueError("gamma_init must be non-zero so attention parameters receive first-step gradients.")

        self.latent_channels = int(latent_channels)
        self.num_tissue_classes = int(num_tissue_classes)
        self.attention_dim = int(attention_dim)
        self.max_ref_tokens_per_class = int(max_ref_tokens_per_class)
        self.gamma_init = float(gamma_init)
        self.alpha = float(alpha)
        self.min_tissue_pixels = float(min_tissue_pixels)
        self.min_nuclei_pixels = float(min_nuclei_pixels)
        self.same_class_attention = bool(same_class_attention)
        self.mask_probability_threshold = float(mask_probability_threshold)

        self.query_encoder = CrossV6TargetLayoutEncoder(
            geometry_channels=4,
            query_dim=self.attention_dim,
            include_xy=include_xy,
            output_init_std=query_output_init_std,
        )
        self.q_proj = nn.Linear(self.attention_dim, self.attention_dim)
        self.k_proj = nn.Linear(self.latent_channels, self.attention_dim)
        self.v_proj = nn.Linear(self.latent_channels, self.attention_dim)
        self.out_proj = nn.Linear(self.attention_dim, self.latent_channels)
        self.gamma = nn.Parameter(torch.full((1, self.latent_channels, 1, 1), self.gamma_init))
        self._reset_attention_parameters()

    def _reset_attention_parameters(self) -> None:
        with torch.no_grad():
            for module in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
                nn.init.xavier_uniform_(module.weight)
                module.bias.zero_()
            self.gamma.fill_(self.gamma_init)

    def forward(self, condition: CrossV6Min0bComposerCondition) -> CrossV6Min0bComposerOutput:
        ref_latent = condition.ref_latent
        if ref_latent.ndim != 4:
            raise ValueError(f"ref_latent must have shape (B,C,H,W), got {tuple(ref_latent.shape)}.")
        batch_size, latent_channels, height, width = ref_latent.shape
        if latent_channels != self.latent_channels:
            raise ValueError(
                f"ref_latent has {latent_channels} channels; composer expects {self.latent_channels}."
            )

        ref_tissue_probs = resize_cross_v6_class_mask_to_probs(
            condition.ref_tissue_mask,
            num_classes=self.num_tissue_classes,
            output_height=height,
            output_width=width,
            device=ref_latent.device,
            dtype=ref_latent.dtype,
            name="ref_tissue_mask",
        )
        target_tissue_probs = resize_cross_v6_class_mask_to_probs(
            condition.target_tissue_mask,
            num_classes=self.num_tissue_classes,
            output_height=height,
            output_width=width,
            device=ref_latent.device,
            dtype=ref_latent.dtype,
            name="target_tissue_mask",
        )
        _validate_batch_size(ref_tissue_probs, batch_size, "ref_tissue_mask")
        _validate_batch_size(target_tissue_probs, batch_size, "target_tissue_mask")

        tissue_prototypes, tissue_mass, tissue_present = masked_pool_cross_v6_latent_by_probs(
            ref_latent,
            ref_tissue_probs,
            min_pixels=self.min_tissue_pixels,
        )
        z_tissue_pool = torch.einsum("bkc,bkhw->bchw", tissue_prototypes, target_tissue_probs)
        retrieval_confidence_map = torch.einsum(
            "bk,bkhw->bhw",
            tissue_present.to(dtype=ref_latent.dtype),
            target_tissue_probs,
        ).unsqueeze(1)
        missing_class_map = torch.einsum(
            "bk,bkhw->bhw",
            (~tissue_present).to(dtype=ref_latent.dtype),
            target_tissue_probs,
        ).unsqueeze(1)

        if self.same_class_attention:
            geometry = _build_query_geometry_from_condition(
                condition,
                output_height=height,
                output_width=width,
                dtype=ref_latent.dtype,
                device=ref_latent.device,
            )
            query_map = self.query_encoder(geometry)
            z_tissue_attn = self._same_class_attention(
                ref_latent=ref_latent,
                ref_tissue_probs=ref_tissue_probs,
                target_tissue_probs=target_tissue_probs,
                tissue_present=tissue_present,
                query_map=query_map,
                fallback=z_tissue_pool,
            )
        else:
            z_tissue_attn = z_tissue_pool
        z_tissue_target = z_tissue_pool + self.gamma * (z_tissue_attn - z_tissue_pool.detach())

        target_nuclei_lat = resize_cross_v6_binary_mask(
            condition.target_nuclei_mask,
            output_height=height,
            output_width=width,
            device=ref_latent.device,
            dtype=ref_latent.dtype,
            name="target_nuclei_mask",
        )
        ref_nuclei_lat = resize_cross_v6_binary_mask(
            condition.ref_nuclei_mask,
            output_height=height,
            output_width=width,
            device=ref_latent.device,
            dtype=ref_latent.dtype,
            name="ref_nuclei_mask",
        )
        _validate_batch_size(target_nuclei_lat, batch_size, "target_nuclei_mask")
        _validate_batch_size(ref_nuclei_lat, batch_size, "ref_nuclei_mask")
        z_nuclei_target, nuclei_present = self._compose_nuclei_pooling(
            ref_latent=ref_latent,
            ref_nuclei_mask=ref_nuclei_lat,
            target_nuclei_mask=target_nuclei_lat,
        )
        nuclei_missing = ((~nuclei_present).to(dtype=ref_latent.dtype).view(batch_size, 1, 1, 1)) * target_nuclei_lat
        missing_class_map = torch.maximum(missing_class_map, nuclei_missing.clamp(0.0, 1.0))

        z_ref_to_target = z_tissue_target + self.alpha * target_nuclei_lat * (
            z_nuclei_target - z_tissue_target.detach()
        )
        return CrossV6Min0bComposerOutput(
            z_tissue_pool=z_tissue_pool,
            z_tissue_attn=z_tissue_attn,
            gamma=self.gamma,
            z_tissue_target=z_tissue_target,
            z_nuclei_target=z_nuclei_target,
            z_ref_to_target=z_ref_to_target,
            retrieval_confidence_map=retrieval_confidence_map.clamp(0.0, 1.0),
            missing_class_map=missing_class_map.clamp(0.0, 1.0),
        )

    def _same_class_attention(
        self,
        *,
        ref_latent: torch.Tensor,
        ref_tissue_probs: torch.Tensor,
        target_tissue_probs: torch.Tensor,
        tissue_present: torch.Tensor,
        query_map: torch.Tensor,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, channels, height, width = ref_latent.shape
        target_flat = target_tissue_probs.flatten(2)
        ref_flat = ref_tissue_probs.flatten(2)
        ref_tokens = ref_latent.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        query_tokens = query_map.permute(0, 2, 3, 1).reshape(batch_size, height * width, self.attention_dim)
        output = fallback.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels).clone()
        scale = 1.0 / math.sqrt(float(self.attention_dim))

        for batch_index in range(batch_size):
            for class_index in range(self.num_tissue_classes):
                target_indices = torch.nonzero(
                    target_flat[batch_index, class_index] > self.mask_probability_threshold,
                    as_tuple=False,
                ).flatten()
                if target_indices.numel() == 0:
                    continue
                if not bool(tissue_present[batch_index, class_index].item()):
                    continue
                ref_indices = torch.nonzero(
                    ref_flat[batch_index, class_index] > self.mask_probability_threshold,
                    as_tuple=False,
                ).flatten()
                if ref_indices.numel() == 0:
                    continue
                ref_indices = _uniform_subsample_indices(ref_indices, self.max_ref_tokens_per_class)
                queries = self.q_proj(query_tokens[batch_index, target_indices])
                keys = self.k_proj(ref_tokens[batch_index, ref_indices])
                values = self.v_proj(ref_tokens[batch_index, ref_indices])
                logits = torch.matmul(queries, keys.transpose(0, 1)) * scale
                attn = torch.softmax(logits, dim=-1)
                attended = torch.matmul(attn, values)
                output[batch_index, target_indices] = self.out_proj(attended)
        return output.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()

    def _compose_nuclei_pooling(
        self,
        *,
        ref_latent: torch.Tensor,
        ref_nuclei_mask: torch.Tensor,
        target_nuclei_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, _, _ = ref_latent.shape
        ref_weight = ref_nuclei_mask.clamp(0.0, 1.0)
        mass = ref_weight.flatten(2).sum(dim=-1).squeeze(1)
        present = mass >= self.min_nuclei_pixels
        proto = (ref_latent * ref_weight).flatten(2).sum(dim=-1) / mass.clamp_min(1e-6).unsqueeze(-1)
        proto = torch.where(present.unsqueeze(-1), proto, torch.zeros_like(proto))
        z_nuclei_target = proto.view(batch_size, channels, 1, 1) * target_nuclei_mask.clamp(0.0, 1.0)
        return z_nuclei_target, present


def build_cross_v6_geometry_maps(
    *,
    target_tissue_mask: torch.Tensor,
    target_nuclei_mask: torch.Tensor,
    output_height: int,
    output_width: int,
    distance_iterations: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return class-agnostic geometry maps in V6 order.

    Order: ``nuclei_binary, nuclei_boundary, nuclei_distance_map,
    tissue_boundary``.
    """

    output_height = int(output_height)
    output_width = int(output_width)
    if output_height <= 0 or output_width <= 0:
        raise ValueError(f"output size must be positive, got {output_height}x{output_width}.")
    tissue = _ensure_batched_hw(target_tissue_mask, name="target_tissue_mask")
    nuclei = _ensure_batched_hw(target_nuclei_mask, name="target_nuclei_mask")
    if tissue.shape[0] != nuclei.shape[0]:
        raise ValueError(
            f"target_tissue_mask batch size {tissue.shape[0]} does not match target_nuclei_mask {nuclei.shape[0]}."
        )

    tissue_low = _resize_label_mask(tissue, output_height, output_width)
    tissue_boundary = _label_boundary_map(tissue_low)
    nuclei_binary = resize_cross_v6_binary_mask(
        nuclei,
        output_height=output_height,
        output_width=output_width,
        device=nuclei.device,
        dtype=torch.float32,
        name="target_nuclei_mask",
    )
    nuclei_label_low = _resize_label_mask((nuclei > 0).long(), output_height, output_width)
    nuclei_boundary = _label_boundary_map(nuclei_label_low)
    nuclei_distance = _approx_normalized_distance_to_foreground(
        (nuclei_binary > 0.5).float(),
        iterations=max(1, int(distance_iterations)),
    )
    return (
        nuclei_binary.clamp(0.0, 1.0),
        nuclei_boundary.clamp(0.0, 1.0),
        nuclei_distance.clamp(0.0, 1.0),
        tissue_boundary.clamp(0.0, 1.0),
    )


def build_cross_v6_control_condition_tensor(
    condition: CrossV6Min0bControlCondition,
    *,
    reference_latent: torch.Tensor | None = None,
    target_latent: torch.Tensor | None = None,
    normalize_z: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Concatenate the final V6 condition without target class one-hot maps."""

    z = condition.z_ref_to_target
    if z.ndim != 4:
        raise ValueError(f"z_ref_to_target must have shape (B,C,H,W), got {tuple(z.shape)}.")
    if normalize_z:
        z = normalize_cross_v6_latent_condition(
            z,
            reference_latent=reference_latent,
            target_latent=target_latent,
            eps=eps,
        )
    maps = [
        _resize_scalar_map(condition.nuclei_binary, z, name="nuclei_binary"),
        _resize_scalar_map(condition.nuclei_boundary, z, name="nuclei_boundary"),
        _resize_scalar_map(condition.nuclei_distance_map, z, name="nuclei_distance_map"),
        _resize_scalar_map(condition.tissue_boundary, z, name="tissue_boundary"),
        _resize_scalar_map(condition.retrieval_confidence_map, z, name="retrieval_confidence_map"),
        _resize_scalar_map(condition.missing_class_map, z, name="missing_class_map"),
    ]
    return torch.cat([z, *maps], dim=1)


def build_cross_v6_control_condition_from_output(
    *,
    composer_output: CrossV6Min0bComposerOutput,
    nuclei_binary: torch.Tensor,
    nuclei_boundary: torch.Tensor,
    nuclei_distance_map: torch.Tensor,
    tissue_boundary: torch.Tensor,
    reference_latent: torch.Tensor | None = None,
    target_latent: torch.Tensor | None = None,
    normalize_z: bool = True,
) -> torch.Tensor:
    return build_cross_v6_control_condition_tensor(
        CrossV6Min0bControlCondition(
            z_ref_to_target=composer_output.z_ref_to_target,
            nuclei_binary=nuclei_binary,
            nuclei_boundary=nuclei_boundary,
            nuclei_distance_map=nuclei_distance_map,
            tissue_boundary=tissue_boundary,
            retrieval_confidence_map=composer_output.retrieval_confidence_map,
            missing_class_map=composer_output.missing_class_map,
        ),
        reference_latent=reference_latent,
        target_latent=target_latent,
        normalize_z=normalize_z,
    )


def normalize_cross_v6_latent_condition(
    z_ref_to_target: torch.Tensor,
    *,
    reference_latent: torch.Tensor | None = None,
    target_latent: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Match composed latent stats to reference or target latent scale."""

    if z_ref_to_target.ndim != 4:
        raise ValueError(f"z_ref_to_target must have shape (B,C,H,W), got {tuple(z_ref_to_target.shape)}.")
    source = reference_latent if reference_latent is not None else target_latent
    z_mean = z_ref_to_target.mean(dim=(1, 2, 3), keepdim=True)
    z_std = z_ref_to_target.std(dim=(1, 2, 3), keepdim=True, unbiased=False).clamp_min(float(eps))
    normalized = (z_ref_to_target - z_mean) / z_std
    if source is None:
        return normalized
    if source.shape != z_ref_to_target.shape:
        raise ValueError(
            f"normalization source shape {tuple(source.shape)} must match z_ref_to_target {tuple(z_ref_to_target.shape)}."
        )
    source = source.to(device=z_ref_to_target.device, dtype=z_ref_to_target.dtype)
    source_mean = source.mean(dim=(1, 2, 3), keepdim=True)
    source_std = source.std(dim=(1, 2, 3), keepdim=True, unbiased=False).clamp_min(float(eps))
    return normalized * source_std + source_mean


def masked_pool_cross_v6_latent_by_probs(
    latent: torch.Tensor,
    class_probs: torch.Tensor,
    *,
    min_pixels: float = 1.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool latent features into per-class prototypes with soft masks."""

    if latent.ndim != 4:
        raise ValueError(f"latent must have shape (B,C,H,W), got {tuple(latent.shape)}.")
    if class_probs.ndim != 4:
        raise ValueError(f"class_probs must have shape (B,K,H,W), got {tuple(class_probs.shape)}.")
    if class_probs.shape[0] != latent.shape[0] or class_probs.shape[-2:] != latent.shape[-2:]:
        raise ValueError(
            f"class_probs shape {tuple(class_probs.shape)} is incompatible with latent {tuple(latent.shape)}."
        )
    weights = class_probs.to(device=latent.device, dtype=latent.dtype).clamp_min(0.0)
    mass = weights.flatten(2).sum(dim=-1)
    proto = torch.einsum("bkhw,bchw->bkc", weights, latent) / mass.clamp_min(float(eps)).unsqueeze(-1)
    present = mass >= float(min_pixels)
    proto = torch.where(present.unsqueeze(-1), proto, torch.zeros_like(proto))
    return proto, mass, present


def resize_cross_v6_class_mask_to_probs(
    mask: torch.Tensor,
    *,
    num_classes: int,
    output_height: int,
    output_width: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    name: str = "class_mask",
) -> torch.Tensor:
    mask = _ensure_batched_hw(mask, name=name).to(device=device, dtype=torch.long)
    _validate_id_range(mask, low=0, high=int(num_classes) - 1, name=name)
    one_hot = F.one_hot(mask, num_classes=int(num_classes)).permute(0, 3, 1, 2).to(dtype=torch.float32)
    resized = F.interpolate(one_hot, size=(int(output_height), int(output_width)), mode="area")
    denom = resized.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return (resized / denom).to(device=device, dtype=dtype)


def resize_cross_v6_binary_mask(
    mask: torch.Tensor,
    *,
    output_height: int,
    output_width: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    name: str = "binary_mask",
) -> torch.Tensor:
    mask = _ensure_batched_hw(mask, name=name)
    binary = (mask > 0).to(device=device, dtype=torch.float32).unsqueeze(1)
    resized = F.interpolate(binary, size=(int(output_height), int(output_width)), mode="area")
    return resized.clamp(0.0, 1.0).to(device=device, dtype=dtype)


def diagnose_cross_v6_vae_class_internal_variance(
    *,
    ref_latent: torch.Tensor,
    ref_tissue_mask: torch.Tensor,
    num_classes: int = NUM_COARSE,
    max_pairwise_tokens: int = 256,
    eps: float = 1e-6,
) -> CrossV6ClassInternalVarianceDiagnostics:
    """Measure whether same-class reference latent tokens contain retrievable variation."""

    if ref_latent.ndim != 4:
        raise ValueError(f"ref_latent must have shape (B,C,H,W), got {tuple(ref_latent.shape)}.")
    batch_size, channels, height, width = ref_latent.shape
    probs = resize_cross_v6_class_mask_to_probs(
        ref_tissue_mask,
        num_classes=int(num_classes),
        output_height=height,
        output_width=width,
        device=ref_latent.device,
        dtype=ref_latent.dtype,
        name="ref_tissue_mask",
    )
    tokens = ref_latent.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
    probs_flat = probs.flatten(2)
    class_mass = probs_flat.sum(dim=-1)
    token_variance = ref_latent.new_zeros(batch_size, int(num_classes))
    pca_top1_energy = ref_latent.new_zeros(batch_size, int(num_classes))
    pairwise = ref_latent.new_zeros(batch_size, int(num_classes))

    for batch_index in range(batch_size):
        for class_index in range(int(num_classes)):
            indices = torch.nonzero(probs_flat[batch_index, class_index] > 0.5, as_tuple=False).flatten()
            if indices.numel() < 2:
                continue
            class_tokens = tokens[batch_index, indices].float()
            centered = class_tokens - class_tokens.mean(dim=0, keepdim=True)
            variance_per_channel = centered.var(dim=0, unbiased=False)
            token_variance[batch_index, class_index] = variance_per_channel.mean().to(dtype=ref_latent.dtype)
            total_energy = (centered * centered).sum().clamp_min(float(eps))
            try:
                singular_values = torch.linalg.svdvals(centered)
                top_energy = singular_values[0] * singular_values[0]
                pca_top1_energy[batch_index, class_index] = (top_energy / total_energy).to(dtype=ref_latent.dtype)
            except RuntimeError:
                pca_top1_energy[batch_index, class_index] = ref_latent.new_tensor(0.0)
            sampled = class_tokens[_uniform_subsample_indices(torch.arange(indices.numel(), device=indices.device), max_pairwise_tokens)]
            if sampled.shape[0] >= 2:
                pairwise[batch_index, class_index] = torch.pdist(sampled, p=2).mean().to(dtype=ref_latent.dtype)

    return CrossV6ClassInternalVarianceDiagnostics(
        class_mass=class_mass,
        token_variance=token_variance,
        pca_top1_energy=pca_top1_energy,
        mean_pairwise_distance=pairwise,
    )


def cross_v6_gamma_diagnostics(composer: CrossV6Min0bLatentComposer) -> dict[str, float]:
    delta = composer.gamma.detach().float() - float(composer.gamma_init)
    return {
        "cross_v6_gamma_init": float(composer.gamma_init),
        "cross_v6_gamma_mean": float(composer.gamma.detach().float().mean().cpu().item()),
        "cross_v6_gamma_abs_delta_mean": float(delta.abs().mean().cpu().item()),
        "cross_v6_gamma_delta_l2": float(torch.linalg.vector_norm(delta).cpu().item()),
    }


def _build_query_geometry_from_condition(
    condition: CrossV6Min0bComposerCondition,
    *,
    output_height: int,
    output_width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    maps = [
        _resize_map_to_latent(condition.nuclei_binary, output_height, output_width, device, dtype, "nuclei_binary"),
        _resize_map_to_latent(condition.nuclei_boundary, output_height, output_width, device, dtype, "nuclei_boundary"),
        _resize_map_to_latent(
            condition.nuclei_distance_map,
            output_height,
            output_width,
            device,
            dtype,
            "nuclei_distance_map",
        ),
        _resize_map_to_latent(condition.tissue_boundary, output_height, output_width, device, dtype, "tissue_boundary"),
    ]
    return torch.cat(maps, dim=1)


def _resize_scalar_map(value: torch.Tensor, target: torch.Tensor, *, name: str) -> torch.Tensor:
    return _resize_map_to_latent(
        value,
        int(target.shape[-2]),
        int(target.shape[-1]),
        target.device,
        target.dtype,
        name,
    )


def _resize_map_to_latent(
    value: torch.Tensor,
    output_height: int,
    output_width: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError(f"{name} must have shape (B,1,H,W) or (B,H,W), got {tuple(value.shape)}.")
    value = value.to(device=device, dtype=dtype)
    if value.shape[-2:] != (int(output_height), int(output_width)):
        value = F.interpolate(value, size=(int(output_height), int(output_width)), mode="bilinear", align_corners=False)
    return value


def _normalized_xy_grid_like(value: torch.Tensor) -> torch.Tensor:
    batch_size, _, height, width = value.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, int(height), device=value.device, dtype=value.dtype),
        torch.linspace(-1.0, 1.0, int(width), device=value.device, dtype=value.dtype),
        indexing="ij",
    )
    return torch.stack([xx, yy], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)


def _uniform_subsample_indices(indices: torch.Tensor, max_count: int) -> torch.Tensor:
    if indices.numel() <= int(max_count):
        return indices
    positions = torch.linspace(
        0,
        indices.numel() - 1,
        steps=int(max_count),
        device=indices.device,
    ).round().long()
    return indices.index_select(0, positions)


def _validate_batch_size(value: torch.Tensor, batch_size: int, name: str) -> None:
    if value.shape[0] != int(batch_size):
        raise ValueError(f"{name} batch size {value.shape[0]} does not match ref_latent batch size {batch_size}.")


def _ensure_batched_hw(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape (B,H,W), (B,1,H,W), or (H,W), got {tuple(value.shape)}.")
    return value


def _resize_label_mask(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return F.interpolate(mask.float().unsqueeze(1), size=(int(height), int(width)), mode="nearest")[:, 0].long()


def _label_boundary_map(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 3:
        raise ValueError(f"mask must have shape (B,H,W), got {tuple(mask.shape)}.")
    boundary = torch.zeros_like(mask, dtype=torch.bool)
    boundary[:, :, 1:] |= mask[:, :, 1:] != mask[:, :, :-1]
    boundary[:, :, :-1] |= mask[:, :, 1:] != mask[:, :, :-1]
    boundary[:, 1:, :] |= mask[:, 1:, :] != mask[:, :-1, :]
    boundary[:, :-1, :] |= mask[:, 1:, :] != mask[:, :-1, :]
    return F.max_pool2d(boundary.float().unsqueeze(1), kernel_size=3, stride=1, padding=1).clamp(0.0, 1.0)


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


def _validate_id_range(value: torch.Tensor, *, low: int, high: int, name: str) -> None:
    if value.numel() == 0:
        return
    min_id = int(value.min().item())
    max_id = int(value.max().item())
    if min_id < int(low) or max_id > int(high):
        raise ValueError(f"{name} IDs out of range: got [{min_id}, {max_id}], expected [{low}, {high}].")


__all__ = [
    "CrossV6ClassInternalVarianceDiagnostics",
    "CrossV6Min0bComposerCondition",
    "CrossV6Min0bComposerOutput",
    "CrossV6Min0bControlCondition",
    "CrossV6Min0bControlSpec",
    "CrossV6Min0bLatentComposer",
    "CrossV6TargetLayoutEncoder",
    "build_cross_v6_control_condition_from_output",
    "build_cross_v6_control_condition_tensor",
    "build_cross_v6_geometry_maps",
    "cross_v6_gamma_diagnostics",
    "diagnose_cross_v6_vae_class_internal_variance",
    "masked_pool_cross_v6_latent_by_probs",
    "normalize_cross_v6_latent_condition",
    "resize_cross_v6_binary_mask",
    "resize_cross_v6_class_mask_to_probs",
]

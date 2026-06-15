"""UNI2-h reference image encoder with Perceiver resampler for Cross V1 IP-Adapter."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_config import FINE_TO_PARENT


UNI2H_KWARGS = {
    "model_name": "vit_giant_patch14_224",
    "img_size": 224,
    "patch_size": 14,
    "depth": 24,
    "num_heads": 24,
    "init_values": 1e-5,
    "embed_dim": 1536,
    "mlp_ratio": 2.66667 * 2,
    "num_classes": 0,
    "no_embed_class": True,
    "act_layer": torch.nn.SiLU,
    "reg_tokens": 8,
    "dynamic_img_size": True,
}


logger = logging.getLogger(__name__)


class ReferenceImageEncoder(nn.Module):
    """Encodes reference image via frozen UNI2-h ViT -> projection MLP -> Perceiver resampler.

    The Perceiver resampler compresses UNI2-h's spatial tokens into a smaller set of
    appearance tokens. This filters out spatial layout and retains appearance semantics
    (staining/texture), which is what we want for cross-reference injection.
    """

    def __init__(
        self,
        uni_checkpoint_path: str | Path,
        uni_embed_dim: int = 1536,
        hidden_dim: int = 3072,
        num_tokens: int = 16,
        num_perceiver_layers: int = 2,
        perceiver_heads: int = 8,
        use_perceiver_self_attn: bool = True,
        perceiver_cross_gate_init: float | None = None,
        skip_perceiver: bool = False,
    ):
        super().__init__()
        self.uni_embed_dim = int(uni_embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_tokens = int(num_tokens)
        self.num_perceiver_layers = int(num_perceiver_layers)
        self.perceiver_heads = int(perceiver_heads)
        self.use_perceiver_self_attn = bool(use_perceiver_self_attn)
        self.skip_perceiver = bool(skip_perceiver)
        self._input_range_kind: str | None = None
        self.perceiver_cross_gate_init = (
            None if perceiver_cross_gate_init is None else float(perceiver_cross_gate_init)
        )
        self.uni = self._load_uni(uni_checkpoint_path)
        self._lock_uni_backbone()

        mean_vals = (0.485, 0.456, 0.406)
        std_vals = (0.229, 0.224, 0.225)
        self.register_buffer("mean", torch.tensor(mean_vals).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std_vals).view(1, 3, 1, 1))

        self.proj_mlp = nn.Sequential(
            nn.Linear(uni_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.latent_queries = nn.Parameter(
            torch.randn(1, num_tokens, hidden_dim) * 0.02
        )
        self.perceiver_layers = nn.ModuleList([
            PerceiverCrossAttentionLayer(
                latent_dim=hidden_dim,
                input_dim=hidden_dim,
                num_heads=perceiver_heads,
                use_self_attn=self.use_perceiver_self_attn,
                cross_gate_init=self.perceiver_cross_gate_init,
            )
            for _ in range(num_perceiver_layers)
        ])
        self.perceiver_norm = nn.LayerNorm(hidden_dim)

    def train(self, mode: bool = True):
        """Keep the frozen UNI backbone in eval mode even when the wrapper trains."""
        super().train(mode)
        self._lock_uni_backbone()
        return self

    def _lock_uni_backbone(self) -> None:
        self.uni.requires_grad_(False)
        self.uni.eval()

    @property
    def num_output_tokens(self) -> int:
        if self.skip_perceiver:
            return self.num_spatial_tokens
        return self.num_tokens

    @property
    def num_spatial_tokens(self) -> int:
        patch_size = int(UNI2H_KWARGS["patch_size"])
        image_size = int(UNI2H_KWARGS["img_size"])
        return (image_size // patch_size) * (image_size // patch_size)

    def _load_uni(self, checkpoint_path: str | Path):
        import torch.distributed as dist
        import timm

        rank = dist.get_rank() if dist.is_initialized() else -1
        print(f"[rank {rank}] >>> BEFORE torch.load UNI: {checkpoint_path}", flush=True)

        model = timm.create_model(
            **UNI2H_KWARGS,
            mlp_layer=timm.layers.SwiGLUPacked,
            pretrained=False,
        )
        state_dict = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

        print(f"[rank {rank}] >>> AFTER torch.load, num_keys={len(state_dict)}", flush=True)

        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"UNI2-h checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )

        print(f"[rank {rank}] >>> UNI model loaded successfully", flush=True)
        return model

    def extract_uni_features(
        self,
        images: torch.Tensor,
        *,
        allow_input_grad: bool = False,
    ) -> torch.Tensor:
        def _extract() -> torch.Tensor:
            self._lock_uni_backbone()
            x = self._prepare_uni_input(images)
            features = self.uni.forward_features(x)
            if features.ndim == 3:
                patch_size = int(UNI2H_KWARGS["patch_size"])
                num_patch_tokens = (x.shape[-2] // patch_size) * (x.shape[-1] // patch_size)
                if features.shape[1] > num_patch_tokens:
                    features = features[:, -num_patch_tokens:, :]
            return features

        if allow_input_grad:
            return _extract()
        with torch.no_grad():
            return _extract()

    def _prepare_uni_input(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"UNI input must have shape (B,3,H,W), got {tuple(images.shape)}")
        uni_param = next(self.uni.parameters())
        x = images.to(device=uni_param.device, dtype=uni_param.dtype)
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        x = self._coerce_image_range(x)
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def _coerce_image_range(self, images: torch.Tensor) -> torch.Tensor:
        """Accept common RGB ranges while keeping the training [0, 1] path unchanged."""
        x = images
        if self._input_range_kind is not None:
            if self._input_range_kind == "minus_one_one":
                return (x + 1.0) * 0.5
            if self._input_range_kind == "zero_255":
                return x / 255.0
            if self._input_range_kind == "clamp":
                return x.clamp(0.0, 1.0)
            return x

        detached = x.detach()
        if detached.numel() == 0:
            return x
        min_value = float(detached.amin().item())
        max_value = float(detached.amax().item())
        if min_value >= -1.05 and max_value <= 1.05 and min_value < -0.05:
            self._input_range_kind = "minus_one_one"
            return (x + 1.0) * 0.5
        if max_value > 2.0 and min_value >= -1e-3:
            self._input_range_kind = "zero_255"
            return x / 255.0
        if min_value < -1e-3 or max_value > 1.0 + 1e-3:
            self._input_range_kind = "clamp"
            if min_value < -0.05 or max_value > 1.05:
                logger.warning(
                    "ReferenceImageEncoder UNI input outside expected [0, 1] range: "
                    "min=%.4f max=%.4f; clamping before ImageNet normalization.",
                    min_value,
                    max_value,
                )
            else:
                logger.info(
                    "ReferenceImageEncoder UNI input had minor interpolation overshoot: "
                    "min=%.4f max=%.4f; clamping before ImageNet normalization.",
                    min_value,
                    max_value,
                )
            return x.clamp(0.0, 1.0)
        if max_value - min_value > 1e-6:
            self._input_range_kind = "zero_one"
        return x

    def _resample(self, projected: torch.Tensor) -> torch.Tensor:
        B = projected.shape[0]
        latents = self.latent_queries.expand(B, -1, -1)
        for layer in self.perceiver_layers:
            latents = layer(latents, projected)
        return self.perceiver_norm(latents)

    def reference_presence_gate(
        self,
        images: torch.Tensor,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Return a Bx1x1 gate that is 0 only for deliberately zeroed references."""
        if images.ndim != 4:
            raise ValueError(f"reference images must have shape (B,C,H,W), got {tuple(images.shape)}")
        detached = images.detach()
        gate = (detached.abs().amax(dim=(1, 2, 3)) > float(eps)).to(
            device=detached.device if device is None else device,
            dtype=detached.dtype if dtype is None else dtype,
        )
        return gate.view(-1, 1, 1)

    def load_perceiver_layers_state_dict(
        self, state_dict: dict[str, torch.Tensor]
    ) -> None:
        incompatible = self.perceiver_layers.load_state_dict(state_dict, strict=False)
        allowed_missing = set()
        if self.perceiver_cross_gate_init is not None:
            allowed_missing = {
                f"{index}.cross_gate" for index in range(len(self.perceiver_layers))
            }
        missing = [
            key for key in incompatible.missing_keys if key not in allowed_missing
        ]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Reference Perceiver checkpoint mismatch: "
                f"missing={missing}, unexpected={incompatible.unexpected_keys}"
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        projected = self.encode_projected_patch_tokens(images)
        ref_gate = self.reference_presence_gate(
            images, device=projected.device, dtype=projected.dtype
        )
        if self.skip_perceiver:
            return projected * ref_gate
        resampled = self._resample(projected)
        return resampled * ref_gate

    def encode_projected_patch_tokens(
        self,
        images: torch.Tensor,
        *,
        allow_input_grad: bool = False,
    ) -> torch.Tensor:
        """Return projected UNI spatial patch tokens before global Perceiver pooling."""
        uni_features = self.extract_uni_features(
            images,
            allow_input_grad=allow_input_grad,
        )
        proj_dtype = next(self.proj_mlp.parameters()).dtype
        uni_features = uni_features.to(dtype=proj_dtype)
        return self.proj_mlp(uni_features)

    def encode_region_ip_tokens(
        self,
        images: torch.Tensor,
        region_mask: torch.Tensor,
        *,
        nuclei_mask: torch.Tensor | None = None,
        token_mode: str = "spatial",
        label_mode: str = "tissue",
        background_label: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return regional IP tokens plus one class label per token.

        ``token_mode='spatial'`` preserves the old behavior: every UNI patch token
        is passed directly to the IP-Adapter. ``token_mode='perceiver'`` runs the
        existing Perceiver independently inside each region label, which gives the
        IP-Adapter a region-level bottleneck instead of raw high-frequency patch
        tokens. ``token_mode='stats'`` emits two tokens per label: the mean and
        standard deviation of projected patch tokens inside that label.
        """
        projected = self.encode_projected_patch_tokens(images)
        ref_gate = self.reference_presence_gate(
            images, device=projected.device, dtype=projected.dtype
        )
        labels = build_region_ip_token_labels(
            tissue_mask=region_mask,
            num_tokens=projected.shape[1],
            nuclei_mask=nuclei_mask,
            label_mode=label_mode,
        )
        token_mode = normalize_region_ip_token_mode(token_mode)
        if token_mode == "spatial":
            return projected * ref_gate, labels.to(device=projected.device)
        if token_mode == "stats":
            tokens, pooled_labels = self._stats_by_region_labels(
                projected,
                labels.to(device=projected.device),
                background_label=int(background_label),
            )
            return tokens * ref_gate, pooled_labels.to(device=projected.device)
        tokens, pooled_labels = self._resample_by_region_labels(
            projected,
            labels.to(device=projected.device),
            background_label=int(background_label),
        )
        return tokens * ref_gate, pooled_labels.to(device=projected.device)

    def _resample_by_region_labels(
        self,
        projected: torch.Tensor,
        labels: torch.Tensor,
        *,
        background_label: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress each mask label into ``num_tokens`` Perceiver tokens."""
        if projected.ndim != 3:
            raise ValueError(f"projected tokens must have shape (B,T,C), got {tuple(projected.shape)}")
        if labels.ndim != 2 or labels.shape[:2] != projected.shape[:2]:
            raise ValueError(
                "labels must have shape (B,T) matching projected tokens, "
                f"got labels={tuple(labels.shape)} projected={tuple(projected.shape)}"
            )
        batch_tokens: list[torch.Tensor] = []
        batch_labels: list[torch.Tensor] = []
        for sample_index in range(projected.shape[0]):
            sample_labels = labels[sample_index].to(dtype=torch.long)
            unique_labels = torch.unique(sample_labels).sort().values

            region_tokens = []
            region_labels = []
            for label in unique_labels:
                if int(label.item()) < 0:
                    continue
                region_mask = sample_labels == label
                if not bool(region_mask.any().item()):
                    continue
                pooled = self._resample(projected[sample_index : sample_index + 1, region_mask, :])
                region_tokens.append(pooled[0])
                region_labels.append(
                    torch.full(
                        (pooled.shape[1],),
                        int(label.item()),
                        dtype=torch.long,
                        device=projected.device,
                    )
                )
            if not region_tokens:
                pooled = projected.new_zeros((1, 1, projected.shape[-1]))
                region_tokens.append(pooled[0])
                region_labels.append(
                    torch.full(
                        (pooled.shape[1],),
                        -1,
                        dtype=torch.long,
                        device=projected.device,
                    )
                )
            batch_tokens.append(torch.cat(region_tokens, dim=0))
            batch_labels.append(torch.cat(region_labels, dim=0))

        max_tokens = max(tokens.shape[0] for tokens in batch_tokens)
        padded_tokens = projected.new_zeros((projected.shape[0], max_tokens, projected.shape[-1]))
        padded_labels = torch.full(
            (projected.shape[0], max_tokens),
            -1,
            dtype=torch.long,
            device=projected.device,
        )
        for sample_index, (tokens, token_labels) in enumerate(zip(batch_tokens, batch_labels)):
            padded_tokens[sample_index, : tokens.shape[0]] = tokens
            padded_labels[sample_index, : token_labels.shape[0]] = token_labels
        return padded_tokens, padded_labels

    def _stats_by_region_labels(
        self,
        projected: torch.Tensor,
        labels: torch.Tensor,
        *,
        background_label: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress each mask label into mean and std statistics tokens."""
        if projected.ndim != 3:
            raise ValueError(f"projected tokens must have shape (B,T,C), got {tuple(projected.shape)}")
        if labels.ndim != 2 or labels.shape[:2] != projected.shape[:2]:
            raise ValueError(
                "labels must have shape (B,T) matching projected tokens, "
                f"got labels={tuple(labels.shape)} projected={tuple(projected.shape)}"
            )
        batch_tokens: list[torch.Tensor] = []
        batch_labels: list[torch.Tensor] = []
        for sample_index in range(projected.shape[0]):
            sample_labels = labels[sample_index].to(dtype=torch.long)
            unique_labels = torch.unique(sample_labels).sort().values

            region_tokens = []
            region_labels = []
            for label in unique_labels:
                label_value = int(label.item())
                if label_value < 0 or label_value == int(background_label):
                    continue
                region_mask = sample_labels == label
                if not bool(region_mask.any().item()):
                    continue
                region_values = projected[sample_index, region_mask, :]
                mean_token = region_values.mean(dim=0)
                std_token = region_values.float().std(dim=0, unbiased=False).to(dtype=projected.dtype)
                region_tokens.append(torch.stack([mean_token, std_token], dim=0))
                region_labels.append(
                    torch.full(
                        (2,),
                        label_value,
                        dtype=torch.long,
                        device=projected.device,
                    )
                )
            if not region_tokens:
                region_tokens.append(projected.new_zeros((1, projected.shape[-1])))
                region_labels.append(
                    torch.full(
                        (1,),
                        -1,
                        dtype=torch.long,
                        device=projected.device,
                    )
                )
            batch_tokens.append(torch.cat(region_tokens, dim=0))
            batch_labels.append(torch.cat(region_labels, dim=0))

        max_tokens = max(tokens.shape[0] for tokens in batch_tokens)
        padded_tokens = projected.new_zeros((projected.shape[0], max_tokens, projected.shape[-1]))
        padded_labels = torch.full(
            (projected.shape[0], max_tokens),
            -1,
            dtype=torch.long,
            device=projected.device,
        )
        for sample_index, (tokens, token_labels) in enumerate(zip(batch_tokens, batch_labels)):
            padded_tokens[sample_index, : tokens.shape[0]] = tokens
            padded_labels[sample_index, : token_labels.shape[0]] = token_labels
        return padded_tokens, padded_labels


def normalize_region_ip_token_mode(mode: str) -> str:
    mode = str(mode or "spatial").strip().lower().replace("-", "_")
    aliases = {
        "spatial": "spatial",
        "patch": "spatial",
        "patches": "spatial",
        "direct": "spatial",
        "perceiver": "perceiver",
        "masked_perceiver": "perceiver",
        "region_perceiver": "perceiver",
        "region_wise_perceiver": "perceiver",
        "regionwise_perceiver": "perceiver",
        "stats": "stats",
        "stat": "stats",
        "statistics": "stats",
        "mean_std": "stats",
        "mean+std": "stats",
        "region_stats": "stats",
        "label_stats": "stats",
    }
    if mode not in aliases:
        raise ValueError("regional IP token mode must be 'spatial', 'perceiver', or 'stats'.")
    return aliases[mode]


def normalize_region_ip_label_mode(mode: str) -> str:
    mode = str(mode or "tissue").strip().lower().replace("-", "_")
    aliases = {
        "tissue": "tissue",
        "tissue_only": "tissue",
        "coarse": "coarse_tissue",
        "coarse_tissue": "coarse_tissue",
        "parent": "coarse_tissue",
        "parent_tissue": "coarse_tissue",
        "tissue_nuclei": "tissue_nuclei",
        "tissue+nuclei": "tissue_nuclei",
        "composite": "tissue_nuclei",
        "nuclei": "tissue_nuclei",
        "nuclei_aware": "tissue_nuclei",
    }
    if mode not in aliases:
        raise ValueError(
            "regional IP label mode must be 'tissue', 'coarse_tissue', or 'tissue_nuclei'."
        )
    return aliases[mode]


def build_region_ip_token_labels(
    *,
    tissue_mask: torch.Tensor,
    num_tokens: int,
    nuclei_mask: torch.Tensor | None = None,
    label_mode: str = "tissue",
) -> torch.Tensor:
    """Build tissue or tissue+nuclei labels on the UNI token grid.

    Tissue label 0 is background/unlabeled in the training masks, so regional IP
    attention represents it as -1. The attention mask builder treats only -1 as
    the pad/null-route id.
    """
    label_mode = normalize_region_ip_label_mode(label_mode)
    tissue_labels = resize_mask_to_token_labels(tissue_mask, num_tokens)
    valid_tissue = tissue_labels > 0
    if label_mode == "tissue":
        return torch.where(valid_tissue, tissue_labels, torch.full_like(tissue_labels, -1))
    if label_mode == "coarse_tissue":
        parent_lookup = torch.full(
            (max(FINE_TO_PARENT) + 1,),
            -1,
            dtype=torch.long,
            device=tissue_labels.device,
        )
        for fine_id, parent_id in FINE_TO_PARENT.items():
            parent_lookup[int(fine_id)] = int(parent_id)
        clamped = tissue_labels.clamp(min=0, max=parent_lookup.shape[0] - 1)
        coarse_labels = parent_lookup[clamped]
        return torch.where(valid_tissue, coarse_labels, torch.full_like(coarse_labels, -1))
    if nuclei_mask is None:
        raise ValueError("nuclei_mask is required when regional IP label mode is tissue_nuclei.")
    nuclei_labels = resize_mask_to_token_labels(nuclei_mask, num_tokens)
    combined = combine_tissue_nuclei_labels(tissue_labels, nuclei_labels)
    return torch.where(valid_tissue, combined, torch.full_like(combined, -1))


def combine_tissue_nuclei_labels(
    tissue_labels: torch.Tensor,
    nuclei_labels: torch.Tensor,
    *,
    nuclei_stride: int = 256,
) -> torch.Tensor:
    """Return a stable composite label that separates nuclei classes inside tissue."""
    if tissue_labels.shape != nuclei_labels.shape:
        raise ValueError(
            "tissue and nuclei labels must have the same shape, "
            f"got tissue={tuple(tissue_labels.shape)} nuclei={tuple(nuclei_labels.shape)}"
        )
    tissue_labels = tissue_labels.to(dtype=torch.long)
    nuclei_labels = nuclei_labels.to(dtype=torch.long)
    return tissue_labels * int(nuclei_stride) + nuclei_labels


class PerceiverCrossAttentionLayer(nn.Module):
    """Single Perceiver cross-attention layer: latent queries attend to input tokens."""

    def __init__(
        self,
        latent_dim: int,
        input_dim: int,
        num_heads: int,
        use_self_attn: bool = True,
        cross_gate_init: float | None = None,
    ):
        super().__init__()
        self.use_self_attn = bool(use_self_attn)
        if cross_gate_init is None:
            self.register_parameter("cross_gate", None)
        else:
            self.cross_gate = nn.Parameter(torch.tensor(float(cross_gate_init)))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=num_heads, batch_first=True,
            kdim=input_dim, vdim=input_dim,
        )
        # Keep these modules even when disabled so old/new checkpoints share keys.
        self.self_attn = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=num_heads, batch_first=True,
        )
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.self_attn_norm = nn.LayerNorm(latent_dim)
        self.input_norm = nn.LayerNorm(input_dim)
        self.ff_norm = nn.LayerNorm(latent_dim)
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Linear(latent_dim * 4, latent_dim),
        )

    def forward(self, latents: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        normed_latents = self.latent_norm(latents)
        normed_inputs = self.input_norm(inputs)
        cross_out, _ = self.cross_attn(
            query=normed_latents, key=normed_inputs, value=normed_inputs,
        )
        latents = cross_out
        if self.cross_gate is not None:
            gate = torch.sigmoid(self.cross_gate).to(dtype=latents.dtype)
            latents = (1.0 - gate) * latents
        if self.use_self_attn:
            normed_latents2 = self.self_attn_norm(latents)
            self_out, _ = self.self_attn(
                query=normed_latents2, key=normed_latents2, value=normed_latents2,
            )
            latents = latents + self_out
        latents = latents + self.ff(self.ff_norm(latents))
        return latents


def resize_mask_to_token_labels(mask: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Nearest-resize a BHW/B1HW mask to the square token grid and flatten to BTN labels."""
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 3:
        raise ValueError(f"mask must have shape (B,H,W) or (B,1,H,W), got {tuple(mask.shape)}")
    grid_h, grid_w = _infer_square_token_grid(int(num_tokens))
    labels = F.interpolate(
        mask.unsqueeze(1).float(),
        size=(grid_h, grid_w),
        mode="nearest",
    )[:, 0]
    return labels.to(dtype=torch.long).flatten(1)


def _infer_square_token_grid(num_tokens: int) -> tuple[int, int]:
    side = int(round(float(num_tokens) ** 0.5))
    if side * side != int(num_tokens):
        raise ValueError(f"expected a square spatial token grid, got num_tokens={num_tokens}")
    return side, side

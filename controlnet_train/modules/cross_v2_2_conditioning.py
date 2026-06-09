"""Cross V2.2 conditioning: block-bank reference latent plus target masks.

V2.2 deliberately removes IP-Adapter reference attention. Reference appearance
is carried by the first ControlNet condition block, ``z_ref``. Unlike V2.1,
``z_ref`` is rebuilt from same-label reference latent blocks sampled onto the
target mask layout, which preserves local texture while breaking reference
spatial layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from controlnet_train.training.conditioning import packed_control_channels


CROSS_V2_2_REFERENCE_WITH_REF = "with_ref"
CROSS_V2_2_REFERENCE_ZERO_REF = "zero_ref"
CROSS_V2_2_REFERENCE_BANK_LABEL_MODES = ("tissue", "nuclei", "tissue_nuclei")


@dataclass(frozen=True)
class CrossV22ControlSpec:
    """Fixed Cross V2.2 ControlNet condition layout.

    Raw condition order:
    [z_ref_bank, ref_tissue_feat, ref_nuclei_feat, tar_tissue_feat, tar_nuclei_feat]

    The reference mask feature slots are retained for V2.1 checkpoint width
    compatibility. V2.2 training can zero them while still using raw reference
    masks to build ``z_ref_bank``.
    """

    reference_latent_channels: int = 16
    tissue_channels: int = 64
    nuclei_channels: int = 16

    @property
    def raw_channels(self) -> int:
        mask_channels = self.tissue_channels + self.nuclei_channels
        return self.reference_latent_channels + mask_channels * 2

    @property
    def packed_channels(self) -> int:
        return packed_control_channels(self.raw_channels)

    @property
    def packed_reference_latent_channels(self) -> int:
        return packed_control_channels(self.reference_latent_channels)

    @property
    def packed_mask_channels(self) -> int:
        return packed_control_channels(self.tissue_channels + self.nuclei_channels)

    @property
    def packed_reference_mask_start(self) -> int:
        return self.packed_reference_latent_channels

    @property
    def packed_target_mask_start(self) -> int:
        return self.packed_reference_latent_channels + self.packed_mask_channels


def build_cross_v2_2_condition(
    *,
    z_ref: torch.Tensor,
    ref_tissue_feat: torch.Tensor,
    ref_nuclei_feat: torch.Tensor,
    tar_tissue_feat: torch.Tensor,
    tar_nuclei_feat: torch.Tensor,
) -> torch.Tensor:
    """Concatenate Cross V2.2 ControlNet conditions in the planned order."""

    features = {
        "z_ref": z_ref,
        "ref_tissue_feat": ref_tissue_feat,
        "ref_nuclei_feat": ref_nuclei_feat,
        "tar_tissue_feat": tar_tissue_feat,
        "tar_nuclei_feat": tar_nuclei_feat,
    }
    for name, value in features.items():
        if value.ndim != 4:
            raise ValueError(f"{name} must have shape (B, C, H, W), got {tuple(value.shape)}.")

    for name, value in features.items():
        if z_ref.shape[0] != value.shape[0] or z_ref.shape[2:] != value.shape[2:]:
            raise ValueError(
                f"{name} must match z_ref on batch/spatial dims, "
                f"got {tuple(value.shape)} vs {tuple(z_ref.shape)}."
            )

    return torch.cat(
        [
            z_ref,
            ref_tissue_feat,
            ref_nuclei_feat,
            tar_tissue_feat,
            tar_nuclei_feat,
        ],
        dim=1,
    )


def build_cross_v2_2_block_bank_reference_latent(
    *,
    z_ref: torch.Tensor,
    reference_tissue_mask: torch.Tensor,
    reference_nuclei_mask: torch.Tensor,
    target_tissue_mask: torch.Tensor,
    target_nuclei_mask: torch.Tensor,
    block_size: int = 4,
    label_mode: str = "tissue_nuclei",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample reference latent blocks by label and broadcast them to target layout.

    For each sample, reference latent tiles are grouped into label-specific
    pools using the downsampled reference masks. Each target tile samples one
    block, with replacement, from the corresponding reference pool. Missing
    exact labels fall back to tissue-only, nuclei-only, then the full reference
    block pool.
    """

    if z_ref.ndim != 4:
        raise ValueError(f"z_ref must have shape (B, C, H, W), got {tuple(z_ref.shape)}.")
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    label_mode = _normalize_reference_bank_label_mode(label_mode)

    height, width = z_ref.shape[-2:]
    ref_tissue = _downsample_label_mask(reference_tissue_mask, size=(height, width), device=z_ref.device)
    ref_nuclei = _downsample_label_mask(reference_nuclei_mask, size=(height, width), device=z_ref.device)
    tar_tissue = _downsample_label_mask(target_tissue_mask, size=(height, width), device=z_ref.device)
    tar_nuclei = _downsample_label_mask(target_nuclei_mask, size=(height, width), device=z_ref.device)
    _validate_label_grid_batch(
        z_ref=z_ref,
        reference_tissue_mask=ref_tissue,
        reference_nuclei_mask=ref_nuclei,
        target_tissue_mask=tar_tissue,
        target_nuclei_mask=tar_nuclei,
    )

    rebuilt = torch.empty_like(z_ref)
    for batch_index in range(z_ref.shape[0]):
        pools = _build_reference_block_pools(
            z_ref=z_ref[batch_index],
            tissue_labels=ref_tissue[batch_index],
            nuclei_labels=ref_nuclei[batch_index],
            block_size=block_size,
            label_mode=label_mode,
        )
        for y0, y1, x0, x1 in _iter_block_slices(height, width, block_size):
            tissue_label = _majority_label(tar_tissue[batch_index, y0:y1, x0:x1])
            nuclei_label = _majority_label(tar_nuclei[batch_index, y0:y1, x0:x1])
            source_block = _sample_reference_block(
                pools=pools,
                tissue_label=tissue_label,
                nuclei_label=nuclei_label,
                label_mode=label_mode,
                generator=generator,
                device=z_ref.device,
            )
            rebuilt[batch_index, :, y0:y1, x0:x1] = _fit_block_to_slice(
                source_block,
                height=y1 - y0,
                width=x1 - x0,
            )
    return rebuilt


def normalize_cross_v2_2_reference_mode(mode: str) -> str:
    value = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "normal": CROSS_V2_2_REFERENCE_WITH_REF,
        "ref": CROSS_V2_2_REFERENCE_WITH_REF,
        "reference": CROSS_V2_2_REFERENCE_WITH_REF,
        "with_ref": CROSS_V2_2_REFERENCE_WITH_REF,
        "zero": CROSS_V2_2_REFERENCE_ZERO_REF,
        "zero_ref": CROSS_V2_2_REFERENCE_ZERO_REF,
    }
    if value not in aliases:
        raise ValueError(
            f"Unsupported Cross V2.2 reference mode {mode!r}; "
            f"choose {CROSS_V2_2_REFERENCE_WITH_REF!r} or {CROSS_V2_2_REFERENCE_ZERO_REF!r}."
        )
    return aliases[value]


def apply_cross_v2_2_reference_mode(
    *,
    z_ref: torch.Tensor,
    ref_tissue_feat: torch.Tensor,
    ref_nuclei_feat: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Optionally ablate the complete reference-side ControlNet condition."""

    normalized = normalize_cross_v2_2_reference_mode(mode)
    if normalized == CROSS_V2_2_REFERENCE_WITH_REF:
        return z_ref, ref_tissue_feat, ref_nuclei_feat
    return (
        torch.zeros_like(z_ref),
        torch.zeros_like(ref_tissue_feat),
        torch.zeros_like(ref_nuclei_feat),
    )


def deterministic_latent_from_posterior(posterior) -> torch.Tensor:
    """Return a stable latent from a VAE posterior for ControlNet conditioning."""

    return posterior.mode() if hasattr(posterior, "mode") else posterior.mean


def _normalize_reference_bank_label_mode(label_mode: str) -> str:
    value = str(label_mode or "tissue_nuclei").strip().lower().replace("-", "_")
    if value not in CROSS_V2_2_REFERENCE_BANK_LABEL_MODES:
        raise ValueError(
            f"Unsupported Cross V2.2 reference bank label mode {label_mode!r}; "
            f"choose one of {CROSS_V2_2_REFERENCE_BANK_LABEL_MODES}."
        )
    return value


def _downsample_label_mask(
    mask: torch.Tensor,
    *,
    size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"label mask must have shape (B, H, W) or (H, W), got {tuple(mask.shape)}.")
    mask = mask.to(device=device)
    if tuple(mask.shape[-2:]) == tuple(size):
        return mask.long()
    downsampled = F.interpolate(
        mask.unsqueeze(1).float(),
        size=size,
        mode="nearest",
    ).squeeze(1)
    return downsampled.long()


def _validate_label_grid_batch(*, z_ref: torch.Tensor, **label_grids: torch.Tensor) -> None:
    for name, value in label_grids.items():
        if value.shape[0] != z_ref.shape[0] or value.shape[-2:] != z_ref.shape[-2:]:
            raise ValueError(
                f"{name} must match z_ref on batch/spatial dims after downsampling, "
                f"got {tuple(value.shape)} vs {tuple(z_ref.shape)}."
            )


def _iter_block_slices(height: int, width: int, block_size: int):
    for y0 in range(0, height, block_size):
        y1 = min(y0 + block_size, height)
        for x0 in range(0, width, block_size):
            x1 = min(x0 + block_size, width)
            yield y0, y1, x0, x1


def _build_reference_block_pools(
    *,
    z_ref: torch.Tensor,
    tissue_labels: torch.Tensor,
    nuclei_labels: torch.Tensor,
    block_size: int,
    label_mode: str,
) -> dict[str, object]:
    height, width = z_ref.shape[-2:]
    exact: dict[object, list[torch.Tensor]] = {}
    tissue: dict[int, list[torch.Tensor]] = {}
    nuclei: dict[int, list[torch.Tensor]] = {}
    all_blocks: list[torch.Tensor] = []
    for y0, y1, x0, x1 in _iter_block_slices(height, width, block_size):
        block = z_ref[:, y0:y1, x0:x1]
        tissue_label = _majority_label(tissue_labels[y0:y1, x0:x1])
        nuclei_label = _majority_label(nuclei_labels[y0:y1, x0:x1])
        all_blocks.append(block)
        tissue.setdefault(tissue_label, []).append(block)
        nuclei.setdefault(nuclei_label, []).append(block)
        exact.setdefault(_label_key(tissue_label, nuclei_label, label_mode), []).append(block)
    return {
        "exact": exact,
        "tissue": tissue,
        "nuclei": nuclei,
        "all": all_blocks,
    }


def _majority_label(labels: torch.Tensor) -> int:
    flat = labels.reshape(-1).long()
    if flat.numel() == 0:
        return 0
    values, counts = torch.unique(flat, sorted=True, return_counts=True)
    return int(values[counts.argmax()].item())


def _label_key(tissue_label: int, nuclei_label: int, label_mode: str) -> object:
    if label_mode == "tissue":
        return int(tissue_label)
    if label_mode == "nuclei":
        return int(nuclei_label)
    return int(tissue_label), int(nuclei_label)


def _sample_reference_block(
    *,
    pools: dict[str, object],
    tissue_label: int,
    nuclei_label: int,
    label_mode: str,
    generator: torch.Generator | None,
    device: torch.device,
) -> torch.Tensor:
    exact_pools = pools["exact"]
    tissue_pools = pools["tissue"]
    nuclei_pools = pools["nuclei"]
    all_blocks = pools["all"]
    candidates = exact_pools.get(_label_key(tissue_label, nuclei_label, label_mode), [])
    if not candidates and label_mode == "tissue_nuclei":
        candidates = tissue_pools.get(int(tissue_label), [])
    if not candidates and label_mode == "tissue_nuclei":
        candidates = nuclei_pools.get(int(nuclei_label), [])
    if not candidates:
        candidates = all_blocks
    if not candidates:
        raise ValueError("reference block pool is empty; z_ref must have positive spatial dimensions.")
    index = int(
        torch.randint(
            0,
            len(candidates),
            (1,),
            device=device,
            generator=generator,
        ).item()
    )
    return candidates[index]


def _fit_block_to_slice(block: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    if block.shape[-2:] == (height, width):
        return block
    if block.shape[-2] >= height and block.shape[-1] >= width:
        return block[:, :height, :width]
    return F.interpolate(
        block.unsqueeze(0).float(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).to(dtype=block.dtype)

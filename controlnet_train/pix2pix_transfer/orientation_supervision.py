"""Target-context supervision for locally directional tissue texture."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .losses import boundary_band_mask, gaussian_blur_image


DEFAULT_ORIENTATION_SCALES = (
    (0.75, 2.0, 3.0),
    (1.5, 4.0, 5.0),
)


@dataclass(frozen=True)
class TargetOrientationLossResult:
    orientation: torch.Tensor
    anisotropy: torch.Tensor
    valid_fraction: torch.Tensor
    mean_coherence: torch.Tensor


@dataclass(frozen=True)
class WindowedI0OrientationLossResult:
    direction: torch.Tensor
    directionality: torch.Tensor
    valid_window_fraction: torch.Tensor
    mean_angle_degrees: torch.Tensor
    mean_i0_resultant: torch.Tensor
    mean_prediction_resultant: torch.Tensor


@dataclass(frozen=True)
class FineTextureEnergyFloorLossResult:
    loss: torch.Tensor
    mean_energy_ratio: torch.Tensor
    under_floor_fraction: torch.Tensor
    over_ceiling_fraction: torch.Tensor
    mean_prediction_energy: torch.Tensor
    mean_baseline_energy: torch.Tensor


def _single_channel_mask(
    value: torch.Tensor,
    *,
    name: str,
    spatial_size: tuple[int, int],
) -> torch.Tensor:
    if value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B,1,H,W] or [B,H,W]")
    if tuple(value.shape[-2:]) != spatial_size:
        value = F.interpolate(value.float(), size=spatial_size, mode="nearest")
    return value


def _grayscale(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"image must have shape [B,3,H,W], got {tuple(image.shape)}")
    weights = image.new_tensor((0.2989, 0.5870, 0.1140), dtype=torch.float32).view(1, 3, 1, 1)
    return (image.float() * weights).sum(dim=1, keepdim=True)


def _structure_tensor(
    gray: torch.Tensor,
    *,
    inner_sigma: float,
    outer_sigma: float,
    integration_sigma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    band = gaussian_blur_image(gray, inner_sigma) - gaussian_blur_image(gray, outer_sigma)
    sobel_x = gray.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=torch.float32,
    ).view(1, 1, 3, 3) / 8.0
    padded = F.pad(band, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(padded, sobel_x)
    gy = F.conv2d(padded, sobel_x.transpose(-1, -2))
    jxx = gaussian_blur_image(gx.square(), integration_sigma)
    jyy = gaussian_blur_image(gy.square(), integration_sigma)
    jxy = gaussian_blur_image(gx * gy, integration_sigma)
    vector = torch.cat((jxx - jyy, 2.0 * jxy), dim=1)
    anisotropy = torch.linalg.vector_norm(vector, dim=1, keepdim=True)
    trace = jxx + jyy
    coherence = anisotropy / trace.clamp_min(1.0e-8)
    unit_vector = vector / anisotropy.clamp_min(1.0e-8)
    return unit_vector, anisotropy, coherence


def _tissue_weight(
    target_tissue_mask: torch.Tensor,
    target_nuclei_mask: torch.Tensor | None,
    trust_map: torch.Tensor | None,
    *,
    spatial_size: tuple[int, int],
    boundary_exclusion_radius: int,
    nuclei_exclusion_radius: int,
    min_trust: float,
) -> torch.Tensor:
    tissue_labels = _single_channel_mask(
        target_tissue_mask,
        name="target_tissue_mask",
        spatial_size=spatial_size,
    )
    weight = tissue_labels.ne(0).float()
    if boundary_exclusion_radius > 0:
        boundary = boundary_band_mask(
            tissue_labels,
            radius=int(boundary_exclusion_radius),
        )
        weight = weight * boundary.eq(0).to(weight.dtype)
    if target_nuclei_mask is not None:
        nuclei = _single_channel_mask(
            target_nuclei_mask,
            name="target_nuclei_mask",
            spatial_size=spatial_size,
        ).ne(0).float()
        radius = max(0, int(nuclei_exclusion_radius))
        if radius > 0:
            nuclei = F.max_pool2d(
                nuclei,
                kernel_size=2 * radius + 1,
                stride=1,
                padding=radius,
            )
        weight = weight * nuclei.eq(0).to(weight.dtype)
    if trust_map is not None:
        trust = _single_channel_mask(
            trust_map,
            name="trust_map",
            spatial_size=spatial_size,
        ).float().clamp(0.0, 1.0)
        trust = torch.where(trust.ge(float(min_trust)), trust, torch.zeros_like(trust))
        weight = weight * trust
    return weight


def multiscale_target_orientation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    target_tissue_mask: torch.Tensor,
    target_nuclei_mask: torch.Tensor | None = None,
    trust_map: torch.Tensor | None = None,
    scales: Sequence[tuple[float, float, float]] = DEFAULT_ORIENTATION_SCALES,
    min_coherence: float = 0.20,
    min_trust: float = 0.50,
    boundary_exclusion_radius: int = 1,
    nuclei_exclusion_radius: int = 2,
) -> TargetOrientationLossResult:
    """Align directional tissue bands to target GT while ignoring isotropic texture."""

    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have the same shape, got {prediction.shape} and {target.shape}"
        )
    if not scales:
        raise ValueError("at least one orientation scale is required")
    coherence_floor = min(max(float(min_coherence), 0.0), 0.999)
    spatial_size = tuple(prediction.shape[-2:])
    base_weight = _tissue_weight(
        target_tissue_mask,
        target_nuclei_mask,
        trust_map,
        spatial_size=spatial_size,
        boundary_exclusion_radius=boundary_exclusion_radius,
        nuclei_exclusion_radius=nuclei_exclusion_radius,
        min_trust=min_trust,
    ).to(device=prediction.device, dtype=torch.float32)
    pred_gray = _grayscale(prediction)
    target_gray = _grayscale(target.detach())
    orientation_values = []
    anisotropy_values = []
    valid_fractions = []
    coherence_values = []
    eligible_count = base_weight.gt(0).float().sum()

    for inner_sigma, outer_sigma, integration_sigma in scales:
        if not 0.0 <= float(inner_sigma) < float(outer_sigma):
            raise ValueError("orientation scales require 0 <= inner_sigma < outer_sigma")
        pred_vector, pred_anisotropy, _ = _structure_tensor(
            pred_gray,
            inner_sigma=float(inner_sigma),
            outer_sigma=float(outer_sigma),
            integration_sigma=float(integration_sigma),
        )
        with torch.no_grad():
            target_vector, target_anisotropy, target_coherence = _structure_tensor(
                target_gray,
                inner_sigma=float(inner_sigma),
                outer_sigma=float(outer_sigma),
                integration_sigma=float(integration_sigma),
            )
            target_energy_scale = (
                (target_anisotropy * base_weight).sum()
                / base_weight.sum().clamp_min(1.0)
            ).clamp_min(1.0e-8)
            relative_energy = (target_anisotropy / target_energy_scale).clamp(0.0, 3.0)
            coherence_weight = (
                (target_coherence - coherence_floor) / (1.0 - coherence_floor)
            ).clamp(0.0, 1.0)
            directional = (
                base_weight.gt(0)
                & target_coherence.ge(coherence_floor)
                & relative_energy.ge(0.50)
            )
            weight = base_weight * coherence_weight * relative_energy

        denominator = weight.sum().clamp_min(1.0e-8)
        cosine = (pred_vector * target_vector).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        orientation_map = 0.5 * (1.0 - cosine)
        orientation_values.append((orientation_map * weight).sum() / denominator)
        pred_relative = pred_anisotropy / target_energy_scale
        target_relative = target_anisotropy / target_energy_scale
        anisotropy_map = F.smooth_l1_loss(
            pred_relative,
            target_relative,
            reduction="none",
        )
        anisotropy_values.append((anisotropy_map * weight).sum() / denominator)
        valid_fractions.append(
            directional.float().sum() / eligible_count.clamp_min(1.0)
        )
        coherence_values.append(
            (target_coherence * directional.float()).sum()
            / directional.float().sum().clamp_min(1.0)
        )

    return TargetOrientationLossResult(
        orientation=torch.stack(orientation_values).mean(),
        anisotropy=torch.stack(anisotropy_values).mean(),
        valid_fraction=torch.stack(valid_fractions).mean().detach(),
        mean_coherence=torch.stack(coherence_values).mean().detach(),
    )


def windowed_i0_mean_orientation_loss(
    prediction: torch.Tensor,
    i0: torch.Tensor,
    *,
    target_tissue_mask: torch.Tensor,
    target_nuclei_mask: torch.Tensor | None = None,
    trust_map: torch.Tensor | None = None,
    fine_scale: tuple[float, float, float] = DEFAULT_ORIENTATION_SCALES[0],
    window_sizes: Sequence[int] = (32, 64),
    window_strides: Sequence[int] = (16, 32),
    min_coherence: float = 0.20,
    min_relative_energy: float = 0.50,
    min_valid_fraction: float = 0.25,
    min_resultant: float = 0.15,
    directionality_floor_ratio: float = 0.50,
    min_trust: float = 0.50,
    boundary_exclusion_radius: int = 3,
    nuclei_exclusion_radius: int = 5,
) -> WindowedI0OrientationLossResult:
    """Match fine-texture mean direction to I0 without copying individual lines.

    The fine-scale structure tensor is pooled as a doubled-angle vector inside
    local windows. I0 determines which windows are directional; prediction
    gradients may occur at different pixels as long as their window-level mean
    orientation follows I0. Nuclei neighborhoods and tissue-label boundaries
    are excluded before pooling.
    """

    if prediction.shape != i0.shape:
        raise ValueError(
            f"prediction and i0 must have the same shape, got {prediction.shape} and {i0.shape}"
        )
    if len(window_sizes) != len(window_strides) or not window_sizes:
        raise ValueError("window_sizes and window_strides must have the same non-zero length")
    normalized_windows = tuple(int(value) for value in window_sizes)
    normalized_strides = tuple(int(value) for value in window_strides)
    if any(value <= 0 for value in normalized_windows + normalized_strides):
        raise ValueError("window sizes and strides must be positive")
    if any(value > min(prediction.shape[-2:]) for value in normalized_windows):
        raise ValueError("orientation window size cannot exceed the prediction spatial size")
    inner_sigma, outer_sigma, integration_sigma = (float(value) for value in fine_scale)
    if not 0.0 <= inner_sigma < outer_sigma:
        raise ValueError("fine_scale requires 0 <= inner_sigma < outer_sigma")

    coherence_floor = min(max(float(min_coherence), 0.0), 0.999)
    relative_energy_floor = max(0.0, float(min_relative_energy))
    valid_fraction_floor = min(max(float(min_valid_fraction), 0.0), 1.0)
    resultant_floor = min(max(float(min_resultant), 0.0), 0.999)
    directionality_floor_ratio = max(0.0, float(directionality_floor_ratio))
    spatial_size = tuple(prediction.shape[-2:])
    base_weight = _tissue_weight(
        target_tissue_mask,
        target_nuclei_mask,
        trust_map,
        spatial_size=spatial_size,
        boundary_exclusion_radius=boundary_exclusion_radius,
        nuclei_exclusion_radius=nuclei_exclusion_radius,
        min_trust=min_trust,
    ).to(device=prediction.device, dtype=torch.float32)

    pred_vector, pred_anisotropy, _ = _structure_tensor(
        _grayscale(prediction),
        inner_sigma=inner_sigma,
        outer_sigma=outer_sigma,
        integration_sigma=integration_sigma,
    )
    with torch.no_grad():
        i0_vector, i0_anisotropy, i0_coherence = _structure_tensor(
            _grayscale(i0.detach()),
            inner_sigma=inner_sigma,
            outer_sigma=outer_sigma,
            integration_sigma=integration_sigma,
        )
        base_denominator = base_weight.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        i0_energy_scale = (
            (i0_anisotropy * base_weight).sum(dim=(2, 3), keepdim=True)
            / base_denominator
        ).clamp_min(1.0e-8)
        i0_relative_energy = (i0_anisotropy / i0_energy_scale).clamp(0.0, 3.0)
        i0_directional = (
            base_weight.gt(0)
            & i0_coherence.ge(coherence_floor)
            & i0_relative_energy.ge(relative_energy_floor)
        )
        i0_coherence_weight = (
            (i0_coherence - coherence_floor) / (1.0 - coherence_floor)
        ).clamp(0.0, 1.0)
        i0_pixel_weight = (
            base_weight
            * i0_directional.to(base_weight.dtype)
            * i0_coherence_weight
            * i0_relative_energy
        )

    pred_energy_scale = (
        (pred_anisotropy * base_weight).sum(dim=(2, 3), keepdim=True)
        / base_weight.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    ).clamp_min(1.0e-8)
    pred_relative_energy = (pred_anisotropy / pred_energy_scale).clamp(0.0, 3.0)
    pred_pixel_weight = base_weight * pred_relative_energy

    direction_values = []
    directionality_values = []
    valid_fractions = []
    angle_values = []
    i0_resultant_values = []
    pred_resultant_values = []
    base_pixels = base_weight.gt(0).float()
    directional_pixels = i0_directional.float()

    for window_size, stride in zip(normalized_windows, normalized_strides, strict=True):
        pool = lambda value: F.avg_pool2d(
            value,
            kernel_size=window_size,
            stride=stride,
        )
        base_fraction = pool(base_pixels)
        directional_fraction = pool(directional_pixels) / base_fraction.clamp_min(1.0e-8)

        i0_weight_sum = pool(i0_pixel_weight)
        i0_mean_vector = pool(i0_vector * i0_pixel_weight) / i0_weight_sum.clamp_min(1.0e-8)
        i0_resultant = torch.linalg.vector_norm(i0_mean_vector, dim=1, keepdim=True)

        pred_weight_sum = pool(pred_pixel_weight)
        pred_mean_vector = (
            pool(pred_vector * pred_pixel_weight) / pred_weight_sum.clamp_min(1.0e-8)
        )
        pred_resultant = torch.linalg.vector_norm(pred_mean_vector, dim=1, keepdim=True)

        valid = (
            base_fraction.ge(valid_fraction_floor)
            & directional_fraction.ge(valid_fraction_floor)
            & i0_weight_sum.gt(0)
            & i0_resultant.ge(resultant_floor)
        )
        valid_weight = (
            valid.to(i0_resultant.dtype)
            * i0_resultant.detach()
            * i0_weight_sum.detach()
        )
        denominator = valid_weight.sum().clamp_min(1.0e-8)
        cosine = (
            (pred_mean_vector * i0_mean_vector.detach()).sum(dim=1, keepdim=True)
            / (
                pred_resultant
                * i0_resultant.detach()
            ).clamp_min(1.0e-8)
        ).clamp(-1.0, 1.0)
        direction_map = 0.5 * (1.0 - cosine)
        directionality_map = F.relu(
            directionality_floor_ratio * i0_resultant.detach() - pred_resultant
        )
        direction_values.append((direction_map * valid_weight).sum() / denominator)
        directionality_values.append(
            (directionality_map * valid_weight).sum() / denominator
        )
        valid_fractions.append(valid.float().mean())
        angle_map = 0.5 * torch.rad2deg(torch.acos(cosine))
        angle_values.append((angle_map * valid_weight).sum() / denominator)
        i0_resultant_values.append((i0_resultant * valid_weight).sum() / denominator)
        pred_resultant_values.append((pred_resultant * valid_weight).sum() / denominator)

    return WindowedI0OrientationLossResult(
        direction=torch.stack(direction_values).mean(),
        directionality=torch.stack(directionality_values).mean(),
        valid_window_fraction=torch.stack(valid_fractions).mean().detach(),
        mean_angle_degrees=torch.stack(angle_values).mean().detach(),
        mean_i0_resultant=torch.stack(i0_resultant_values).mean().detach(),
        mean_prediction_resultant=torch.stack(pred_resultant_values).mean().detach(),
    )


def windowed_fine_texture_energy_floor_loss(
    prediction: torch.Tensor,
    baseline_prediction: torch.Tensor,
    *,
    target_tissue_mask: torch.Tensor,
    target_nuclei_mask: torch.Tensor | None = None,
    trust_map: torch.Tensor | None = None,
    fine_scale: tuple[float, float, float] = DEFAULT_ORIENTATION_SCALES[0],
    window_sizes: Sequence[int] = (32, 64),
    window_strides: Sequence[int] = (16, 32),
    energy_floor_ratio: float = 0.95,
    energy_ceiling_ratio: float | None = None,
    min_baseline_relative_energy: float = 0.50,
    min_valid_fraction: float = 0.25,
    min_trust: float = 0.50,
    boundary_exclusion_radius: int = 3,
    nuclei_exclusion_radius: int = 5,
) -> FineTextureEnergyFloorLossResult:
    """Prevent orientation supervision from winning by erasing fine texture.

    The frozen epoch25 output supplies only an absolute fine-band energy floor;
    it does not supply the desired direction.  Direction can therefore move
    toward I0 while the amount of transferred texture remains comparable to
    the sharp epoch25 baseline.
    """

    if prediction.shape != baseline_prediction.shape:
        raise ValueError(
            "prediction and baseline_prediction must have the same shape, got "
            f"{prediction.shape} and {baseline_prediction.shape}"
        )
    if len(window_sizes) != len(window_strides) or not window_sizes:
        raise ValueError("window_sizes and window_strides must have the same non-zero length")
    normalized_windows = tuple(int(value) for value in window_sizes)
    normalized_strides = tuple(int(value) for value in window_strides)
    if any(value <= 0 for value in normalized_windows + normalized_strides):
        raise ValueError("window sizes and strides must be positive")
    if any(value > min(prediction.shape[-2:]) for value in normalized_windows):
        raise ValueError("energy window size cannot exceed the prediction spatial size")
    inner_sigma, outer_sigma, integration_sigma = (float(value) for value in fine_scale)
    if not 0.0 <= inner_sigma < outer_sigma:
        raise ValueError("fine_scale requires 0 <= inner_sigma < outer_sigma")

    floor_ratio = max(0.0, float(energy_floor_ratio))
    ceiling_ratio = (
        None
        if energy_ceiling_ratio is None or float(energy_ceiling_ratio) <= 0.0
        else max(floor_ratio, float(energy_ceiling_ratio))
    )
    relative_energy_floor = max(0.0, float(min_baseline_relative_energy))
    valid_fraction_floor = min(max(float(min_valid_fraction), 0.0), 1.0)
    spatial_size = tuple(prediction.shape[-2:])
    base_weight = _tissue_weight(
        target_tissue_mask,
        target_nuclei_mask,
        trust_map,
        spatial_size=spatial_size,
        boundary_exclusion_radius=boundary_exclusion_radius,
        nuclei_exclusion_radius=nuclei_exclusion_radius,
        min_trust=min_trust,
    ).to(device=prediction.device, dtype=torch.float32)
    _, prediction_energy, _ = _structure_tensor(
        _grayscale(prediction),
        inner_sigma=inner_sigma,
        outer_sigma=outer_sigma,
        integration_sigma=integration_sigma,
    )
    with torch.no_grad():
        _, baseline_energy, _ = _structure_tensor(
            _grayscale(baseline_prediction.detach()),
            inner_sigma=inner_sigma,
            outer_sigma=outer_sigma,
            integration_sigma=integration_sigma,
        )
        baseline_image_energy = (
            (baseline_energy * base_weight).sum(dim=(2, 3), keepdim=True)
            / base_weight.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        ).clamp_min(1.0e-8)

    loss_values = []
    ratio_values = []
    under_values = []
    over_values = []
    prediction_values = []
    baseline_values = []
    base_pixels = base_weight.gt(0).float()
    for window_size, stride in zip(normalized_windows, normalized_strides, strict=True):
        def pool(value: torch.Tensor) -> torch.Tensor:
            return F.avg_pool2d(value, kernel_size=window_size, stride=stride)

        base_fraction = pool(base_pixels)
        pooled_weight = pool(base_weight).clamp_min(1.0e-8)
        prediction_window_energy = pool(prediction_energy * base_weight) / pooled_weight
        baseline_window_energy = pool(baseline_energy * base_weight) / pooled_weight
        baseline_relative = baseline_window_energy / baseline_image_energy
        valid = (
            base_fraction.ge(valid_fraction_floor)
            & baseline_relative.ge(relative_energy_floor)
            & baseline_window_energy.gt(1.0e-8)
        )
        valid_weight = valid.to(torch.float32) * baseline_relative.detach().clamp(0.0, 3.0)
        denominator = valid_weight.sum().clamp_min(1.0e-8)
        energy_ratio = prediction_window_energy / baseline_window_energy.detach().clamp_min(1.0e-8)
        energy_deficit = F.relu(floor_ratio - energy_ratio)
        energy_excess = (
            torch.zeros_like(energy_ratio)
            if ceiling_ratio is None
            else F.relu(energy_ratio - ceiling_ratio)
        )
        loss_values.append(
            ((energy_deficit + energy_excess) * valid_weight).sum() / denominator
        )
        ratio_values.append((energy_ratio * valid_weight).sum() / denominator)
        under_values.append(
            (energy_ratio.lt(floor_ratio).to(valid_weight.dtype) * valid_weight).sum()
            / denominator
        )
        over_values.append(
            (
                torch.zeros_like(valid_weight)
                if ceiling_ratio is None
                else energy_ratio.gt(ceiling_ratio).to(valid_weight.dtype)
            ).mul(valid_weight).sum()
            / denominator
        )
        prediction_values.append(
            (prediction_window_energy * valid_weight).sum() / denominator
        )
        baseline_values.append(
            (baseline_window_energy * valid_weight).sum() / denominator
        )

    return FineTextureEnergyFloorLossResult(
        loss=torch.stack(loss_values).mean(),
        mean_energy_ratio=torch.stack(ratio_values).mean().detach(),
        under_floor_fraction=torch.stack(under_values).mean().detach(),
        over_ceiling_fraction=torch.stack(over_values).mean().detach(),
        mean_prediction_energy=torch.stack(prediction_values).mean().detach(),
        mean_baseline_energy=torch.stack(baseline_values).mean().detach(),
    )

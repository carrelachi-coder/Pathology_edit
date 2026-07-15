"""Differentiable WSI appearance descriptors and structure-only losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_weight(mask: torch.Tensor, size: tuple[int, int], device: torch.device) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask must have shape [B,1,H,W] or [B,H,W], got {tuple(mask.shape)}")
    weight = mask.to(device=device).ne(0).float()
    if tuple(weight.shape[-2:]) != tuple(size):
        weight = F.interpolate(weight, size=size, mode="area")
    return weight.clamp(0.0, 1.0)


def _masked_moments(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight = _as_weight(mask, tuple(values.shape[-2:]), values.device)
    denominator = weight.sum(dim=(2, 3)).clamp_min(1.0)
    mean = (values.float() * weight).sum(dim=(2, 3)) / denominator
    centered = values.float() - mean[:, :, None, None]
    variance = (centered.square() * weight).sum(dim=(2, 3)) / denominator
    return mean, variance.clamp_min(1.0e-6).sqrt()


def optical_density(image: torch.Tensor) -> torch.Tensor:
    rgb = ((image.float() + 1.0) * 0.5).clamp(1.0 / 255.0, 1.0)
    return -torch.log(rgb)


def _masked_covariance(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = _as_weight(mask, tuple(values.shape[-2:]), values.device)
    denominator = weight.sum(dim=(2, 3)).clamp_min(1.0)
    mean = (values.float() * weight).sum(dim=(2, 3)) / denominator
    centered = (values.float() - mean[:, :, None, None]) * weight.sqrt()
    flattened = centered.flatten(2)
    return torch.matmul(flattened, flattened.transpose(1, 2)) / denominator[:, :, None]


def optical_density_moment_loss(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    prediction_mask: torch.Tensor,
    reference_mask: torch.Tensor,
) -> torch.Tensor:
    pred_od = optical_density(prediction)
    ref_od = optical_density(reference)
    pred_mean, _ = _masked_moments(pred_od, prediction_mask)
    ref_mean, _ = _masked_moments(ref_od, reference_mask)
    pred_cov = _masked_covariance(pred_od, prediction_mask)
    ref_cov = _masked_covariance(ref_od, reference_mask)
    return F.l1_loss(pred_mean, ref_mean.detach()) + F.l1_loss(pred_cov, ref_cov.detach())


def feature_moment_loss(
    prediction_feature: torch.Tensor,
    reference_feature: torch.Tensor,
    prediction_mask: torch.Tensor,
    reference_mask: torch.Tensor,
) -> torch.Tensor:
    pred_mean, pred_std = _masked_moments(prediction_feature, prediction_mask)
    ref_mean, ref_std = _masked_moments(reference_feature, reference_mask)
    return F.l1_loss(pred_mean, ref_mean.detach()) + F.l1_loss(pred_std, ref_std.detach())


def laplacian_band_descriptor(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    descriptors = []
    for kernel in (3, 5, 9):
        low = F.avg_pool2d(image.float(), kernel_size=kernel, stride=1, padding=kernel // 2)
        high = image.float() - low
        weight = _as_weight(mask, tuple(image.shape[-2:]), image.device)
        denominator = weight.sum(dim=(2, 3)).clamp_min(1.0)
        energy = (high.square() * weight).sum(dim=(2, 3)) / denominator
        descriptors.append(torch.log1p(energy))
    return torch.cat(descriptors, dim=1)


def family_image_descriptor(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Describe color, feature moments and multi-scale local frequency energy."""

    od = optical_density(image)
    mean, std = _masked_moments(od, mask)
    covariance = _masked_covariance(od, mask).flatten(1)
    band = laplacian_band_descriptor(image, mask)
    return torch.cat([mean, std, covariance, band], dim=1)


def selected_reference_ranking_loss(
    output_descriptor: torch.Tensor,
    selected_descriptor: torch.Tensor,
    negative_descriptor: torch.Tensor,
    *,
    margin: float = 0.10,
) -> torch.Tensor:
    selected_distance = (output_descriptor.float() - selected_descriptor.detach().float()).abs().mean(dim=1)
    negative_distance = (output_descriptor.float() - negative_descriptor.detach().float()).abs().mean(dim=1)
    return F.relu(selected_distance - negative_distance + float(margin)).mean()


def contrast_normalized_grayscale(image: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"image must have shape [B,3,H,W], got {tuple(image.shape)}")
    weights = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    gray = (image.float() * weights.float()).sum(dim=1, keepdim=True)
    mean = gray.mean(dim=(2, 3), keepdim=True)
    std = gray.var(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(eps).sqrt()
    return (gray - mean) / std


def _sobel(gray: torch.Tensor) -> torch.Tensor:
    kernel_x = gray.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    kernel_y = kernel_x.transpose(0, 1)
    kernels = torch.stack([kernel_x, kernel_y], dim=0).unsqueeze(1)
    return F.conv2d(gray, kernels, padding=1)


def grayscale_structure_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_gray = contrast_normalized_grayscale(prediction)
    target_gray = contrast_normalized_grayscale(target).detach()
    gray_loss = F.l1_loss(pred_gray, target_gray)
    edge_loss = F.l1_loss(_sobel(pred_gray), _sobel(target_gray))
    return gray_loss, edge_loss

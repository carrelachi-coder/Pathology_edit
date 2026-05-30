"""HED stain perturbation for H&E self-supervised stain transfer."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HEDPerturbation:
    """One sampled HED perturbation shared by reference and target images."""

    alpha: torch.Tensor
    beta: torch.Tensor


class HEDStainAugment:
    """Apply H&E stain concentration perturbations to RGB tensors in [0, 1].

    The transform uses a fixed Ruifrok-Johnston-style HED stain matrix. For the
    stain-transfer setup, sample params once and apply the same params to the
    reference and target patch so both receive the same random stain style.
    """

    def __init__(
        self,
        *,
        sigma: float = 0.2,
        beta: float = 0.02,
        strong_alpha_sampling: bool = False,
        alpha_min: float = 0.4,
        alpha_low: float = 0.75,
        alpha_high: float = 1.25,
        alpha_max: float = 1.8,
        eps: float = 1.0 / 255.0,
        max_condition_number: float = 100.0,
    ) -> None:
        if sigma < 0:
            raise ValueError("sigma must be non-negative.")
        if beta < 0:
            raise ValueError("beta must be non-negative.")
        if not (0 < alpha_min <= alpha_low < 1.0 < alpha_high <= alpha_max):
            raise ValueError(
                "alpha bounds must satisfy 0 < alpha_min <= alpha_low < 1 < alpha_high <= alpha_max."
            )
        if eps <= 0 or eps >= 1:
            raise ValueError("eps must be within (0, 1).")
        self.sigma = float(sigma)
        self.beta = float(beta)
        self.strong_alpha_sampling = bool(strong_alpha_sampling)
        self.alpha_min = float(alpha_min)
        self.alpha_low = float(alpha_low)
        self.alpha_high = float(alpha_high)
        self.alpha_max = float(alpha_max)
        self.eps = float(eps)

        stain_rows = torch.tensor(
            [
                [0.650, 0.704, 0.286],  # Hematoxylin
                [0.072, 0.990, 0.105],  # Eosin
                [0.268, 0.570, 0.776],  # DAB / residual channel
            ],
            dtype=torch.float32,
        )
        stain_rows = stain_rows / stain_rows.norm(dim=1, keepdim=True).clamp_min(eps)
        condition_number = float(torch.linalg.cond(stain_rows).item())
        if condition_number > float(max_condition_number):
            raise ValueError(
                "HED stain matrix is poorly conditioned: "
                f"condition_number={condition_number:.3f}, "
                f"max_condition_number={float(max_condition_number):.3f}"
            )
        self.condition_number = condition_number
        self._stain_rows = stain_rows
        self._inv_stain_rows = torch.linalg.inv(stain_rows)

    def sample(self, *, device: torch.device | str | None = None) -> HEDPerturbation:
        device = torch.device("cpu") if device is None else torch.device(device)
        alpha = torch.ones(3, device=device, dtype=torch.float32)
        beta = torch.zeros(3, device=device, dtype=torch.float32)
        if self.strong_alpha_sampling:
            alpha[:2] = self._sample_strong_alpha(device=device)
        elif self.sigma > 0:
            alpha[:2] = 1.0 + (torch.rand(2, device=device) * 2.0 - 1.0) * self.sigma
            alpha[:2] = alpha[:2].clamp_min(0.05)
        if self.beta > 0:
            beta[:2] = (torch.rand(2, device=device) * 2.0 - 1.0) * self.beta
        return HEDPerturbation(alpha=alpha, beta=beta)

    def _sample_strong_alpha(self, *, device: torch.device) -> torch.Tensor:
        side = torch.randint(0, 2, (2,), device=device, dtype=torch.int64)
        low = self.alpha_min + torch.rand(2, device=device) * (self.alpha_low - self.alpha_min)
        high = self.alpha_high + torch.rand(2, device=device) * (self.alpha_max - self.alpha_high)
        return torch.where(side.bool(), high, low)

    def __call__(self, image: torch.Tensor, params: HEDPerturbation) -> torch.Tensor:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"expected CHW RGB tensor with 3 channels, got {tuple(image.shape)}")

        dtype = image.dtype
        device = image.device
        stain_rows = self._stain_rows.to(device=device, dtype=torch.float32)
        inv_stain_rows = self._inv_stain_rows.to(device=device, dtype=torch.float32)
        rgb = image.to(dtype=torch.float32).clamp(self.eps, 1.0)

        chw = rgb.shape
        rgb_flat = rgb.permute(1, 2, 0).reshape(-1, 3)
        od = -torch.log(rgb_flat)
        concentrations = od @ inv_stain_rows
        alpha = params.alpha.to(device=device, dtype=torch.float32)
        beta = params.beta.to(device=device, dtype=torch.float32)
        concentrations = concentrations * alpha + beta
        concentrations = concentrations.clamp_min(0.0)
        perturbed_od = concentrations @ stain_rows
        perturbed_rgb = torch.exp(-perturbed_od).reshape(chw[1], chw[2], 3).permute(2, 0, 1)
        return perturbed_rgb.clamp(0.0, 1.0).to(dtype=dtype)

    def apply_pair(self, reference: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.sample(device=reference.device)
        return self(reference, params), self(target, params)

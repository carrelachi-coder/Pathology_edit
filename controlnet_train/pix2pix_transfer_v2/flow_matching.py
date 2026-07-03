from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from controlnet_train.pix2pix_transfer_v2.dit_backbone import Pix2PixV2DiT


class FlowMatching:
    """
    Flow-matching implementation for pix2pix V2, velocity prediction objective.
    Aligns with FLUX paradigm for future tool reuse (inversion, injection, etc.)
    """
    def __init__(self, sigma_min: float = 1e-4):
        self.sigma_min = sigma_min

    @torch.no_grad()
    def sample_timesteps(self, batch_size: int, device: torch.device | str = "cuda") -> torch.Tensor:
        """Sample t ~ Uniform(0, 1) for training."""
        return torch.rand(batch_size, device=device)

    def get_noisy_latent_and_velocity(self, z1: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z1: (B, C, H, W) ground truth clean latent (target)
            t: (B,) timesteps in [0, 1]
        Returns:
            z_t: (B, C, H, W) noised latent at timestep t
            v: (B, C, H, W) target velocity = z1 - ε
        """
        b, c, h, w = z1.shape
        eps = torch.randn_like(z1)
        t = t.view(b, 1, 1, 1)
        # Linear interpolation: z_t = t * z1 + (1 - t) * eps
        z_t = t * z1 + (1 - t) * eps
        # Velocity target: v = d z_t / dt = z1 - eps
        v = z1 - eps
        return z_t, v

    def velocity_loss(self, pred_v: torch.Tensor, target_v: torch.Tensor) -> torch.Tensor:
        """Main flow-matching loss: MSE between predicted and target velocity."""
        return F.mse_loss(pred_v, target_v)

    @torch.no_grad()
    def sample(
        self,
        model: Pix2PixV2DiT,
        i0_latent: torch.Tensor,
        ref_tokens: torch.Tensor,
        num_steps: int = 16,
        class_label: torch.Tensor | None = None,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """
        Sampling: start from pure noise, integrate velocity for num_steps.
        No SDEdit, structure fully guided by concat I0 latent.

        Args:
            model: trained Pix2PixV2DiT
            i0_latent: (B, 16, L, L) I0 latent (fixed)
            ref_tokens: (B, M, D) reference image patch tokens
            num_steps: number of sampling steps, 12~16 recommended
            class_label: optional class label
        Returns:
            z1: (B, 16, L, L) generated clean latent
        """
        b, _, h, w = i0_latent.shape
        # Initialize from pure noise
        z_t = torch.randn(b, 16, h, w, device=device, dtype=dtype)
        # Uniform time steps from 0 to 1
        steps = torch.linspace(0, 1, num_steps + 1, device=device, dtype=dtype)

        for i in range(num_steps):
            t = steps[i].expand(b)
            dt = steps[i+1] - steps[i]
            # Predict velocity
            v = model(z_t, i0_latent, t, ref_tokens, class_label=class_label)
            # Euler update: z_{t+dt} = z_t + v * dt
            z_t = z_t + v * dt

        return z_t

    @torch.no_grad()
    def sample_sdedit(
        self,
        model: Pix2PixV2DiT,
        i0_latent: torch.Tensor,
        ref_tokens: torch.Tensor,
        noise_level: float = 0.6,
        num_steps: int = 16,
        class_label: torch.Tensor | None = None,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """
        Optional SDEdit sampling: start from noised I0 latent, for tighter structure control.
        Not recommended for V0, provided as fallback knob.
        """
        b, _, h, w = i0_latent.shape
        # Initialize from noised I0 latent
        eps = torch.randn_like(i0_latent)
        z_t = noise_level * eps + (1 - noise_level) * i0_latent
        # Time steps from noise_level to 1
        steps = torch.linspace(noise_level, 1, num_steps + 1, device=device, dtype=dtype)

        for i in range(num_steps):
            t = steps[i].expand(b)
            dt = steps[i+1] - steps[i]
            v = model(z_t, i0_latent, t, ref_tokens, class_label=class_label)
            z_t = z_t + v * dt

        return z_t

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from controlnet_train.models.conch import get_conch_model
from controlnet_train.pix2pix_transfer_v2.dit_backbone import Pix2PixV2DiT


class CNNREPAAligner(nn.Module):
    """
    CNN-REPA spatial alignment to CONCH teacher features.
    Core feature: uses light CNN projector (not MLP) to preserve spatial structure,
    as proven in MuPaD to boost OOD image similarity from 0.24 to 0.42.
    """
    def __init__(self, dit_hidden_dim: int = 384, conch_feature_dim: int = 512, projector_depth: int = 3):
        super().__init__()
        # Freeze CONCH teacher
        self.conch = get_conch_model(pretrained=True, freeze=True)
        self.conch_feature_dim = conch_feature_dim

        # Light CNN projector: maps DiT intermediate features to CONCH feature space
        layers = []
        in_dim = dit_hidden_dim
        for i in range(projector_depth):
            out_dim = conch_feature_dim if i == projector_depth - 1 else dit_hidden_dim
            layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1))
            layers.append(nn.GELU())
            if i < projector_depth - 1:
                layers.append(nn.InstanceNorm2d(out_dim))
        self.projector = nn.Sequential(*layers)

    def get_conch_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Get CONCH spatial feature map from RGB image.
        Args:
            image: (B, 3, H, W) [-1, 1]
        Returns:
            feat: (B, D, L, L) CONCH spatial features, latent resolution
        """
        # Normalize to CONCH input range [0, 1]
        image = (image + 1) / 2
        # Get CONCH patch features (spatial grid)
        with torch.no_grad():
            feat = self.conch.encode_image_patch_features(image, return_spatial=True)
        return feat

    def forward(self, dit_intermediate_features: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:
        """
        Compute REPA alignment loss: MSE between projected DiT features and CONCH target features.
        Args:
            dit_intermediate_features: (B, C, L, L) DiT mid-layer features
            target_image: (B, 3, H, W) target RGB image
        Returns:
            loss: scalar REPA alignment loss
        """
        target_feat = self.get_conch_features(target_image)
        projected_feat = self.projector(dit_intermediate_features)
        # Align spatial size if needed
        if projected_feat.shape[-2:] != target_feat.shape[-2:]:
            projected_feat = F.interpolate(projected_feat, size=target_feat.shape[-2:], mode="bilinear", align_corners=False)
        return F.mse_loss(projected_feat, target_feat)


def gram_matrix(feat: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Compute Gram matrix for style loss, optionally masked to a region.
    Args:
        feat: (C, H, W) feature map
        mask: (H, W) binary mask, 1 for region to compute, 0 otherwise
    Returns:
        gram: (C, C) Gram matrix
    """
    c, h, w = feat.shape
    feat = rearrange(feat, "c h w -> c (h w)")
    if mask is not None:
        mask = rearrange(mask, "h w -> 1 (h w)")
        feat = feat * mask
    gram = torch.mm(feat, feat.t())
    # Normalize by number of elements
    gram = gram / (c * h * w)
    return gram


def regional_gram_loss(gen_feat: torch.Tensor, target_feat: torch.Tensor, mask: torch.Tensor, num_classes: int = 17) -> torch.Tensor:
    """
    Regional Gram/style loss, computed per tissue class mask.
    Enforces that texture statistics match within each tissue region,
    prevents cross-region texture leakage, and reduces edge artifacts.
    Args:
        gen_feat: (B, C, H, W) generated image features
        target_feat: (B, C, H, W) target image features
        mask: (B, H, W) tissue mask, int class labels
        num_classes: number of tissue classes
    Returns:
        loss: scalar regional Gram loss
    """
    b, c, h, w = gen_feat.shape
    loss = 0.0
    count = 0

    for batch_idx in range(b):
        for class_idx in range(num_classes):
            # Get binary mask for current class
            class_mask = (mask[batch_idx] == class_idx).float()
            if class_mask.sum() < 10:  # skip small regions
                continue
            # Compute Gram matrices for generated and target
            gen_gram = gram_matrix(gen_feat[batch_idx], class_mask)
            target_gram = gram_matrix(target_feat[batch_idx], class_mask)
            loss += F.mse_loss(gen_gram, target_gram)
            count += 1

    return loss / count if count > 0 else 0.0


class Pix2PixV2Loss(nn.Module):
    """
    Combined loss for pix2pix V2:
    1. Flow-matching velocity loss (main)
    2. CNN-REPA alignment loss (core OOD robustness)
    3. Regional Gram loss (texture consistency + region control)
    4. Optional small latent L1 (early training stability)
    """
    def __init__(
        self,
        dit_hidden_dim: int = 384,
        conch_feature_dim: int = 512,
        repa_weight: float = 0.3,
        gram_weight: float = 0.1,
        latent_l1_weight: float = 0.05,
        num_tissue_classes: int = 17,
    ):
        super().__init__()
        self.repa_aligner = CNNREPAAligner(dit_hidden_dim=dit_hidden_dim, conch_feature_dim=conch_feature_dim)
        self.repa_weight = repa_weight
        self.gram_weight = gram_weight
        self.latent_l1_weight = latent_l1_weight
        self.num_tissue_classes = num_tissue_classes

    def forward(
        self,
        pred_v: torch.Tensor,
        target_v: torch.Tensor,
        pred_latent: torch.Tensor,
        target_latent: torch.Tensor,
        target_image: torch.Tensor,
        tissue_mask: torch.Tensor,
        dit_intermediate_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pred_v: (B, 16, L, L) predicted velocity
            target_v: (B, 16, L, L) target velocity
            pred_latent: (B, 16, L, L) predicted latent (z1)
            target_latent: (B, 16, L, L) ground truth latent (z1)
            target_image: (B, 3, H, W) ground truth RGB image
            tissue_mask: (B, L, L) tissue mask for regional loss
            dit_intermediate_features: (B, C, L, L) DiT mid-layer features for REPA
        Returns:
            loss_dict: dict of individual losses + total loss
        """
        loss_dict = {}

        # 1. Main flow-matching velocity loss
        loss_dict["flow_loss"] = F.mse_loss(pred_v, target_v)

        # 2. CNN-REPA alignment loss (if intermediate features available)
        if dit_intermediate_features is not None and self.repa_weight > 0:
            loss_dict["repa_loss"] = self.repa_aligner(dit_intermediate_features, target_image) * self.repa_weight
        else:
            loss_dict["repa_loss"] = 0.0

        # 3. Regional Gram loss (on CONCH features)
        if self.gram_weight > 0:
            gen_conch_feat = self.repa_aligner.get_conch_features(
                self.repa_aligner.conch.vae.decode(pred_latent / self.repa_aligner.conch.vae.scaling_factor).sample
            )
            target_conch_feat = self.repa_aligner.get_conch_features(target_image)
            loss_dict["gram_loss"] = regional_gram_loss(gen_conch_feat, target_conch_feat, tissue_mask, self.num_tissue_classes) * self.gram_weight
        else:
            loss_dict["gram_loss"] = 0.0

        # 4. Optional latent L1 loss (small weight for early stability)
        if self.latent_l1_weight > 0:
            loss_dict["latent_l1_loss"] = F.l1_loss(pred_latent, target_latent) * self.latent_l1_weight
        else:
            loss_dict["latent_l1_loss"] = 0.0

        # Total loss
        loss_dict["total_loss"] = loss_dict["flow_loss"] + loss_dict["repa_loss"] + loss_dict["gram_loss"] + loss_dict["latent_l1_loss"]

        return loss_dict

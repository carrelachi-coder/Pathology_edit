from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from controlnet_train.data.common import load_nuclei_mask, load_tissue_mask
from controlnet_train.data.augmentations import randstainna, hed_jitter

# 固定类别数，和v1兼容
NUM_TISSUE_CLASSES: int = 17
NUM_NUCLEI_CLASSES: int = 8


class Pix2PixV2Dataset(Dataset):
    """
    Pix2Pix V2 Dataset for flow-matching DiT refinement.
    Core features:
      1. On-the-fly stain augmentation (RandStainNA + HED jitter) for ref images, no caching
      2. Supports pre-cached I0 latent and ref patch tokens to avoid repeated encoding
      3. Loads I0 (ControlNet generation), reference image, tissue/nuclei masks
    """
    def __init__(
        self,
        metadata_path: str | Path,
        image_size: int = 512,
        latent_size: int = 64,  # FLUX VAE 8x downsample: 512 / 8 = 64
        stain_augment_prob: float = 0.7,
        use_cache: bool = True,
        i0_latent_cache_root: str | Path | None = "/data/wqx/flowedit/pix2pix_i0_lazy_cache",
        ref_token_cache_root: str | Path | None = "/data/wqx/flowedit/pix2pix_ref_token_cache",
        split: str = "train",
    ):
        super().__init__()
        self.metadata_path = Path(metadata_path)
        self.image_size = image_size
        self.latent_size = latent_size
        self.stain_augment_prob = stain_augment_prob
        self.use_cache = use_cache
        self.i0_cache_root = Path(i0_latent_cache_root) if i0_latent_cache_root else None
        self.ref_cache_root = Path(ref_token_cache_root) if ref_token_cache_root else None
        self.split = split

        # Load metadata
        payload = json.loads(self.metadata_path.read_text())
        self.records = payload.get("pairs", payload) if isinstance(payload, dict) else payload
        if not isinstance(self.records, list):
            self.records = [self.records]
        print(f"Loaded {len(self.records)} records from {self.metadata_path}")

        # Base transforms
        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1, 1] for VAE
        ])
        self.mask_resize = T.Resize((latent_size, latent_size), interpolation=T.InterpolationMode.NEAREST)

    def __len__(self) -> int:
        return len(self.records)

    def _load_cache(self, cache_root: Path, sample_id: str, key: str) -> torch.Tensor | None:
        if not self.use_cache or not cache_root:
            return None
        cache_path = cache_root / f"{sample_id}_{key}.pt"
        if cache_path.exists():
            try:
                return torch.load(cache_path, map_location="cpu", weights_only=True)
            except Exception:
                return None
        return None

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        sample_id = record.get("sample_id", f"sample_{idx:06d}")
        ref_sample_id = record.get("reference_sample_id", sample_id)

        # Load I0 (ControlNet generation, fixed)
        i0_path = Path(record["i0_image"] if "i0_image" in record else record["target_image"])
        i0 = Image.open(i0_path).convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        i0_tensor = self.to_tensor(i0)

        # Load reference image + apply ON-THE-FLY stain augmentation (never cached)
        ref_path = Path(record["reference_image"])
        ref = Image.open(ref_path).convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        if self.split == "train" and random.random() < self.stain_augment_prob:
            # Apply stain augment: RandStainNA (H&E optimized) + HED jitter
            ref_np = np.asarray(ref)
            if random.random() < 0.5:
                ref_np = randstainna(ref_np, stain_matrix="H&E")
            if random.random() < 0.5:
                ref_np = hed_jitter(ref_np, hue_shift=0.02, sat_scale=0.1, val_scale=0.1)
            ref = Image.fromarray(ref_np)
        ref_tensor = self.to_tensor(ref)

        # Load ground truth target (for training supervision)
        target_path = Path(record["target_image"])
        target = Image.open(target_path).convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        target_tensor = self.to_tensor(target)

        # Load masks (tissue + nuclei, downsampled to latent resolution for loss)
        tissue_mask = load_tissue_mask(record["target_tissue_mask"])
        nuclei_mask = load_nuclei_mask(record["target_nuclei_mask"])
        tissue_mask_tensor = self.mask_resize(torch.from_numpy(tissue_mask).unsqueeze(0)).long().squeeze(0)
        nuclei_mask_tensor = self.mask_resize(torch.from_numpy(nuclei_mask).unsqueeze(0)).long().squeeze(0)

        # Load cached latents/tokens if available
        i0_latent = self._load_cache(self.i0_cache_root, sample_id, "i0_latent")
        ref_tokens = self._load_cache(self.ref_cache_root, ref_sample_id, "ref_patch_tokens")

        return {
            "sample_id": sample_id,
            "reference_sample_id": ref_sample_id,
            "i0": i0_tensor,  # (3, H, W) [-1, 1]
            "reference": ref_tensor,  # (3, H, W) [-1, 1], stain augmented on-the-fly
            "target": target_tensor,  # (3, H, W) [-1, 1], ground truth
            "tissue_mask": tissue_mask_tensor,  # (L, L) int, latent resolution
            "nuclei_mask": nuclei_mask_tensor,  # (L, L) int, latent resolution
            "i0_latent_cached": i0_latent,  # (16, L, L) float, optional cached
            "ref_tokens_cached": ref_tokens,  # (N, D) float, optional cached
            "prompt": record.get("prompt", ""),
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate to handle optional cached fields."""
    out = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if k in ("i0", "reference", "target", "tissue_mask", "nuclei_mask"):
            out[k] = torch.stack(vals)
        elif k in ("i0_latent_cached", "ref_tokens_cached"):
            if all(v is not None for v in vals):
                out[k] = torch.stack(vals) if k == "i0_latent_cached" else vals
            else:
                out[k] = None
        else:
            out[k] = vals
    return out

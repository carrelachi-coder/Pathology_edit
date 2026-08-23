"""Source-preserving compositing for local pathology generation."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

GENERATION_SUPPORT_FEATHER_PX = 12


def source_exact_generation_composite(
    reference_image: np.ndarray,
    generated_image: Image.Image,
    generation_support: np.ndarray,
    *,
    feather_px: int = GENERATION_SUPPORT_FEATHER_PX,
) -> tuple[Image.Image, dict[str, Any]]:
    """Blend only inside support and keep every exterior source pixel exact."""

    source = np.asarray(reference_image, dtype=np.uint8)
    generated = np.asarray(generated_image.convert("RGB"), dtype=np.uint8)
    support = np.asarray(generation_support, dtype=bool)
    if source.shape != generated.shape or source.shape[:2] != support.shape:
        raise ValueError(
            "reference, generated image and generation support must align"
        )
    if feather_px < 0:
        raise ValueError("feather_px must be non-negative")
    support_pixels = int(np.count_nonzero(support))
    if not support_pixels:
        composited = source.copy()
        alpha = np.zeros(support.shape, dtype=np.float32)
    elif feather_px == 0:
        composited = source.copy()
        composited[support] = generated[support]
        alpha = support.astype(np.float32)
    else:
        distance_inside = ndimage.distance_transform_edt(support)
        alpha = np.clip(
            distance_inside / float(feather_px),
            0.0,
            1.0,
        ).astype(np.float32)
        alpha[~support] = 0.0
        blended = (
            generated.astype(np.float32) * alpha[..., None]
            + source.astype(np.float32) * (1.0 - alpha[..., None])
        )
        composited = np.rint(blended).clip(0, 255).astype(np.uint8)
        composited[~support] = source[~support]
    outside_changed_pixels = int(
        np.count_nonzero(np.any(composited != source, axis=2) & ~support)
    )
    return Image.fromarray(composited, mode="RGB"), {
        "policy": "source_exact_outside_feathered_inside_support_v1",
        "support_pixels": support_pixels,
        "feather_px": int(feather_px),
        "fully_generated_pixels": int(np.count_nonzero(alpha >= 1.0)),
        "feathered_pixels": int(np.count_nonzero((alpha > 0.0) & (alpha < 1.0))),
        "outside_support_changed_pixels": outside_changed_pixels,
        "outside_support_source_exact": outside_changed_pixels == 0,
    }

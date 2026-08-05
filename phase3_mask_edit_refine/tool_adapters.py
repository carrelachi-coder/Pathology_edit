"""Explicit read-only adapters to deterministic legacy drawing tools."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from phase3_mask_edit.backends.organic_projection import (
    ORGANIC_PROJECTION_BACKEND,
    apply_organic_projected_label_write,
)
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.core.morphology import multi_scale_smooth_noise

TOOL_ADAPTER_VERSION = "mask-edit-refine-adapter-v2"


def smooth_noise(
    shape: tuple[int, int],
    *,
    seed: int,
    amplitude: float = 1.0,
    correlation_px: float | None = None,
) -> np.ndarray:
    """Call the legacy multi-scale noise helper without changing its implementation."""

    if correlation_px is None:
        scales = (2.0, 8.0, 24.0)
    else:
        correlation = max(3.0, float(correlation_px))
        scales = (max(1.0, correlation / 12.0), correlation / 3.0, correlation)
    return amplitude * multi_scale_smooth_noise(
        shape,
        scales=scales,
        amplitudes=(0.20, 0.35, 0.45),
        seed=seed,
    )


def organic_v2_projection(
    source_mask: np.ndarray,
    raw_template: np.ndarray,
    *,
    schema: MaskProfileSchema,
    source_labels: Sequence[str],
    target_label: str,
    primitive_name: str,
    target_pixels: int,
    seed: int,
    primitive_config: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run organic_v2 through an adapter and return only deterministic artifacts."""

    config = {
        "name": primitive_name,
        "organic_projection": {
            "allow_fallback_when_empty": False,
            "template_spillover_fraction": 0.10,
        },
        **dict(primitive_config or {}),
    }
    result = apply_organic_projected_label_write(
        source_mask,
        raw_template,
        schema=schema,
        source_labels=source_labels,
        target_label=target_label,
        primitive_config=config,
        target_pixels=target_pixels,
        seed=seed,
        allow_fallback_when_empty=False,
    )
    return (
        np.asarray(result.target_mask),
        np.asarray(result.change_region, dtype=bool),
        {
            "legacy_backend": ORGANIC_PROJECTION_BACKEND,
            "legacy_ops_log": dict(result.ops_log),
            "legacy_warnings": list(result.warnings),
        },
    )

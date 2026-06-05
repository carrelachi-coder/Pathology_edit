#!/usr/bin/env python
"""Toy smoke test for Cross V5 four-family loss assembly."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controlnet_train.training.cross_v5_glue import (  # noqa: E402
    CrossV5LossWeights,
    CrossV5StepContext,
    assemble_cross_v5_losses,
)
from controlnet_train.training.cross_v5_losses import CrossV5AppearanceLossConfig  # noqa: E402


def main() -> None:
    prediction = torch.zeros(1, 3, 8, 8, requires_grad=True)
    reference = torch.ones(1, 3, 8, 8) * 0.75
    target_tissue = torch.ones(1, 8, 8, dtype=torch.long)
    reference_tissue = torch.ones(1, 8, 8, dtype=torch.long)
    binary_target = torch.zeros(1, 8, 8)
    binary_target[:, 2:6, 2:6] = 1.0
    geometry_predictions = {
        "tissue_logits": torch.randn(1, 3, 8, 8, requires_grad=True),
        "nuclei_binary_logits": torch.randn(1, 1, 8, 8, requires_grad=True),
        "distance": torch.zeros(1, 1, 8, 8, requires_grad=True),
    }
    context = CrossV5StepContext(
        prediction_rgb=prediction,
        reference_rgb=reference,
        target_tissue_mask=target_tissue,
        reference_tissue_mask=reference_tissue,
        target_nuclei_binary=binary_target,
        target_dense_geometry={"distance": torch.ones(1, 1, 8, 8)},
    )
    bundle = assemble_cross_v5_losses(
        denoise_loss=torch.tensor(0.5, requires_grad=True),
        context=context,
        weights=CrossV5LossWeights(),
        appearance_config=CrossV5AppearanceLossConfig(min_pixels=4, color_space="rgb"),
        geometry_predictions=geometry_predictions,
    )
    bundle.total.backward()
    metrics = {
        key: float(value.detach().cpu().item())
        for key, value in bundle.components.items()
        if isinstance(value, torch.Tensor) and value.ndim == 0
    }
    metrics["total"] = float(bundle.total.detach().cpu().item())
    metrics["prediction_grad_abs_mean"] = float(prediction.grad.detach().abs().mean().cpu().item())
    if metrics["prediction_grad_abs_mean"] <= 0.0:
        raise SystemExit(f"Prediction gradient is zero: {metrics}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

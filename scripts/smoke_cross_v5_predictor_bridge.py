#!/usr/bin/env python
"""Smoke test for Cross V5 generated RGB -> frozen predictor gradient bridge."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controlnet_train.training.cross_v5_glue import (  # noqa: E402
    freeze_predictor_for_v5_loss,
    validate_cross_v5_predictor_grad_bridge,
)


class ToyDensePredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=1, bias=False)
        nn.init.constant_(self.conv.weight, 0.25)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.conv(images)
        return {
            "tissue_logits": logits,
            "nuclei_binary_logits": logits[:, :1],
        }


def main() -> None:
    predictor = freeze_predictor_for_v5_loss(ToyDensePredictor())
    prediction_rgb = torch.rand(2, 3, 16, 16, requires_grad=True)
    metrics = validate_cross_v5_predictor_grad_bridge(
        predictor=predictor,
        prediction_rgb=prediction_rgb,
    )
    if metrics["rgb_grad_abs_mean"] <= 0.0:
        raise SystemExit(f"Bridge gradient is zero: {metrics}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

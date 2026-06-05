#!/usr/bin/env python
"""Smoke test for Cross V5 SEAN-style spatial AdaLN modulation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controlnet_train.modules.cross_v5_conditioning import (  # noqa: E402
    CrossV5SpatialAdaLNModulator,
    CrossV5TissueBank,
    build_cross_v5_spatial_structure_tokens,
)


def main() -> None:
    torch.manual_seed(13)
    hidden = torch.zeros(1, 4, 4)
    class_ids = torch.zeros(1, 4, dtype=torch.long)
    structure = build_cross_v5_spatial_structure_tokens(
        class_ids=torch.zeros(1, 2, 2, dtype=torch.long),
        num_classes=1,
        token_height=2,
        token_width=2,
    )
    bank = CrossV5TissueBank(
        prototypes=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        local_tokens=torch.zeros(1, 1, 1, 2),
        class_present=torch.ones(1, 1, dtype=torch.bool),
        class_mass=torch.ones(1, 1),
        token_class_ids=torch.zeros(1, 1, dtype=torch.long),
        token_class_confidence=torch.ones(1, 1),
    )
    modulator = CrossV5SpatialAdaLNModulator(
        hidden_dim=4,
        prototype_dim=4,
        structure_dim=structure.shape[-1],
        output_init_std=0.01,
    )
    output = modulator(
        hidden_states=hidden,
        target_class_ids=class_ids,
        target_structure_tokens=structure,
        bank=bank,
    )
    gamma_delta = (output.gamma[:, 0] - output.gamma[:, -1]).detach().abs().mean()
    if gamma_delta.item() <= 1e-6:
        raise SystemExit("Spatial AdaLN gamma did not vary with target structure.")

    print(
        json.dumps(
            {
                "structure_shape": list(structure.shape),
                "gamma_delta_abs_mean": float(gamma_delta.cpu().item()),
                "hidden_shape": list(output.hidden_states.shape),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

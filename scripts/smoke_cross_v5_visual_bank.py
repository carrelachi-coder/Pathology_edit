#!/usr/bin/env python
"""Smoke test for Cross V5 low-level visual reference bank construction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controlnet_train.modules.cross_v5_conditioning import (  # noqa: E402
    CrossV5RefBankBuilder,
    build_cross_v5_hed_stat_prototypes,
)


def main() -> None:
    texture_tokens = torch.tensor(
        [
            [
                [1.0, 0.0],
                [3.0, 0.0],
                [0.0, 2.0],
                [0.0, 4.0],
            ]
        ]
    )
    reference_image = torch.tensor(
        [
            [
                [[0.8, 0.7], [0.2, 0.3]],
                [[0.6, 0.5], [0.4, 0.4]],
                [[0.7, 0.6], [0.5, 0.5]],
            ]
        ]
    )
    class_ids = torch.tensor([[[1, 1], [2, 2]]])
    builder = CrossV5RefBankBuilder(num_classes=3, local_tokens_per_class=2)
    bank = builder(
        reference_tokens=texture_tokens,
        reference_image=reference_image,
        reference_class_ids=class_ids,
    )
    expected = build_cross_v5_hed_stat_prototypes(
        reference_image=reference_image,
        reference_class_ids=class_ids,
        num_classes=3,
    )
    if not torch.allclose(bank.prototypes, expected.to(dtype=bank.prototypes.dtype)):
        raise SystemExit("V5 bank prototypes are not HED stain statistics.")
    if bank.local_tokens.shape[-1] != texture_tokens.shape[-1]:
        raise SystemExit("V5 local tokens did not preserve texture-token dimensionality.")

    print(
        json.dumps(
            {
                "prototype_shape": list(bank.prototypes.shape),
                "local_tokens_shape": list(bank.local_tokens.shape),
                "class_present": bank.class_present.int().tolist(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

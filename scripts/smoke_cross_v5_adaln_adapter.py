#!/usr/bin/env python
"""Toy smoke test for Cross V5 AdaLN hook installation and bank causality."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controlnet_train.modules.cross_v5_conditioning import (  # noqa: E402
    CrossV5AdaLNModulator,
    CrossV5TissueBank,
)
from controlnet_train.training.cross_v5_glue import (  # noqa: E402
    CrossV5AdaLNAdapterMixin,
    CrossV5AdaLNHookSpec,
    install_cross_v5_adaln_hooks,
)


class V5ReadyBlock(CrossV5AdaLNAdapterMixin, torch.nn.Module):
    def forward(self, hidden: torch.Tensor, *, target_class_ids: torch.Tensor, bank: CrossV5TissueBank) -> torch.Tensor:
        return self._apply_cross_v5_adaln(hidden, target_class_ids=target_class_ids, bank=bank)


class ToyTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([V5ReadyBlock(), V5ReadyBlock()])


def _bank(prototypes: torch.Tensor) -> CrossV5TissueBank:
    return CrossV5TissueBank(
        prototypes=prototypes,
        local_tokens=torch.zeros(prototypes.shape[0], prototypes.shape[1], 1, prototypes.shape[2]),
        class_present=torch.ones(prototypes.shape[0], prototypes.shape[1], dtype=torch.bool),
        class_mass=torch.ones(prototypes.shape[0], prototypes.shape[1]),
        token_class_ids=torch.zeros(prototypes.shape[0], 1, dtype=torch.long),
        token_class_confidence=torch.ones(prototypes.shape[0], 1),
    )


def main() -> None:
    torch.manual_seed(7)
    transformer = ToyTransformer()
    modulator = CrossV5AdaLNModulator(hidden_dim=4, output_init_std=0.01)
    summary = install_cross_v5_adaln_hooks(
        transformer=transformer,
        modulator=modulator,
        spec=CrossV5AdaLNHookSpec(block_indices=(-1,), hook_point="post_norm_hidden"),
    )

    hidden = torch.randn(1, 3, 4)
    target_class_ids = torch.tensor([[0, 1, 0]])
    bank_a = _bank(torch.zeros(1, 2, 4))
    bank_b = _bank(torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]))

    block = transformer.transformer_blocks[1]
    out_a = block(hidden, target_class_ids=target_class_ids, bank=bank_a)
    out_b = block(hidden, target_class_ids=target_class_ids, bank=bank_b)
    hidden_delta = (out_a - hidden).detach().abs().mean()
    bank_swap_delta = (out_b - out_a).detach().abs().mean()
    if hidden_delta.item() <= 0.0:
        raise SystemExit("AdaLN hook did not change hidden states.")
    if bank_swap_delta.item() <= 1e-6:
        raise SystemExit("Changing the reference bank did not change AdaLN output.")

    zero_gamma_rejected = False
    bad_modulator = CrossV5AdaLNModulator(hidden_dim=4, output_init_std=0.01)
    with torch.no_grad():
        final = bad_modulator.mlp[-1]
        final.weight[:4].zero_()
        final.bias[:4].zero_()
    try:
        install_cross_v5_adaln_hooks(
            transformer=ToyTransformer(),
            modulator=bad_modulator,
            spec=CrossV5AdaLNHookSpec(block_indices=(0,), require_nonzero_gamma=True),
        )
    except ValueError:
        zero_gamma_rejected = True
    if not zero_gamma_rejected:
        raise SystemExit("Zero-initialized AdaLN gamma path was not rejected.")

    print(
        json.dumps(
            {
                "installed_block_indices": list(summary.installed_block_indices),
                "hidden_delta_abs_mean": float(hidden_delta.cpu().item()),
                "bank_swap_delta_abs_mean": float(bank_swap_delta.cpu().item()),
                "zero_gamma_rejected": zero_gamma_rejected,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

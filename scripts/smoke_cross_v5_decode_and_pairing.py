#!/usr/bin/env python
"""Smoke test for Cross V5 differentiable decode bridge and pairing sampler."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controlnet_train.data.cross_v5_pairing import CrossV5PairingSampler  # noqa: E402
from controlnet_train.training.cross_v5_glue import (  # noqa: E402
    CrossV5LatentDecodeConfig,
    CrossV5PairingPolicy,
    decode_cross_v5_prediction_rgb,
)


class ToyVAE(torch.nn.Module):
    class Config:
        scaling_factor = 2.0
        shift_factor = 0.25

    config = Config()

    def decode(self, latents, return_dict=False):
        return (latents * 2.0,)


def main() -> None:
    noisy = torch.ones(1, 1, 2, 2, requires_grad=True)
    prediction = torch.full((1, 1, 2, 2), 0.5, requires_grad=True)
    rgb = decode_cross_v5_prediction_rgb(
        vae=ToyVAE(),
        noisy_latents=noisy,
        model_prediction=prediction,
        sigma=torch.tensor([0.2]),
        config=CrossV5LatentDecodeConfig(prediction_type="velocity"),
    )
    rgb.mean().backward()
    noisy_grad = float(noisy.grad.detach().abs().mean().cpu().item())
    prediction_grad = float(prediction.grad.detach().abs().mean().cpu().item())
    if noisy_grad <= 0.0 or prediction_grad <= 0.0:
        raise SystemExit("Decode bridge did not propagate gradients to latent inputs.")

    sampler = CrossV5PairingSampler(
        [
            {
                "reference_sample_id": "same_low_gap",
                "case_id": "wsi_a",
                "reference_case_id": "wsi_a",
                "pair_difficulty": "full",
                "appearance_gap": 0.1,
                "covered_target_tissue_ids": [1, 2],
            },
            {
                "reference_sample_id": "cross_partial",
                "case_id": "wsi_a",
                "reference_case_id": "wsi_b",
                "pair_difficulty": "partial",
                "appearance_gap": 0.4,
                "covered_target_tissue_ids": [1],
            },
            {
                "reference_sample_id": "cross_high_full",
                "case_id": "wsi_a",
                "reference_case_id": "wsi_c",
                "pair_difficulty": "full",
                "appearance_gap": 0.9,
                "covered_target_tissue_ids": [1, 2],
            },
        ],
        policy=CrossV5PairingPolicy(
            same_wsi_fraction=0.0,
            cross_wsi_fraction=0.0,
            high_appearance_gap_fraction=1.0,
            full_coverage_fraction=1.0,
            partial_coverage_fraction=0.0,
            low_coverage_fraction=0.0,
            class_bank_dropout_prob=1.0,
        ),
        seed=3,
    )
    sampled = sampler.sample()
    if sampled["reference_sample_id"] != "cross_high_full":
        raise SystemExit(f"Unexpected sampled pair: {sampled}")
    if not sampled["v5_reference_bank_keep_tissue_ids"]:
        raise SystemExit("Bank dropout removed every available class.")

    print(
        json.dumps(
            {
                "rgb_mean": float(rgb.detach().mean().cpu().item()),
                "noisy_grad_abs_mean": noisy_grad,
                "prediction_grad_abs_mean": prediction_grad,
                "sampled_reference": sampled["reference_sample_id"],
                "v5_pair_mode": sampled["v5_pair_mode"],
                "v5_coverage_mode": sampled["v5_coverage_mode"],
                "bank_keep_ids": sampled["v5_reference_bank_keep_tissue_ids"],
                "bank_drop_ids": sampled["v5_reference_bank_drop_tissue_ids"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

"""Train Phase 5.4 Cross V4 Flux ControlNet."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controlnet_train.cli.train_controlnet_flux_cross_v3 import parse_args as parse_cross_v3_args


def _has_cli_arg(input_args, name: str) -> bool:
    args = list(sys.argv[1:] if input_args is None else input_args)
    return any(arg == name or arg.startswith(f"{name}=") for arg in args)


def main(input_args=None) -> None:
    args = parse_args(input_args)
    from controlnet_train.training.flux_phase5_cross_v3 import run_cross_v3_training

    run_cross_v3_training(args)


def parse_args(input_args=None, description: str | None = None):
    args = parse_cross_v3_args(
        input_args,
        description=description
        or (
            "Train Phase 5.4 Cross V4 Flux ControlNet: target-only ControlNet, "
            "reference local tokens, per-class priors, and mask-guided correspondence bias."
        ),
    )
    args.cross_version = "v4"
    if args.output_dir == "phase5-controlnet-cross-v3":
        args.output_dir = "phase5-controlnet-cross-v4"
    if args.tracker_project_name == "flux_controlnet_phase5_cross_v3":
        args.tracker_project_name = "flux_controlnet_phase5_cross_v4"
    v4_defaults = {
        "--ref-swap-loss-weight": ("ref_swap_loss_weight", 0.0),
        "--ref-swap-loss-interval": ("ref_swap_loss_interval", 0),
        "--ref-swap-variants": ("ref_swap_variants", ""),
        "--cross-v4-cell-similarity-bias": ("cross_v4_cell_similarity_bias", 0.0),
        "--cross-v4-density-gap-bias": ("cross_v4_density_gap_bias", 0.0),
        "--cross-v4-cell-prior-bias": ("cross_v4_cell_prior_bias", 0.0),
    }
    for option_name, (attr_name, default_value) in v4_defaults.items():
        if not _has_cli_arg(input_args, option_name):
            setattr(args, attr_name, default_value)
    return args


if __name__ == "__main__":
    main()

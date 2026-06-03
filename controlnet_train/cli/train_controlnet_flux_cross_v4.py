"""Train Phase 5.4 Cross V4 Flux ControlNet."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controlnet_train.cli.train_controlnet_flux_cross_v3 import parse_args


def main(input_args=None) -> None:
    args = parse_args(
        input_args,
        description=(
            "Train Phase 5.4 Cross V4 Flux ControlNet: target-only ControlNet, "
            "reference local tokens, per-class priors, and mask-guided correspondence bias."
        ),
    )
    args.cross_version = "v4"
    if args.output_dir == "phase5-controlnet-cross-v3":
        args.output_dir = "phase5-controlnet-cross-v4"
    if args.tracker_project_name == "flux_controlnet_phase5_cross_v3":
        args.tracker_project_name = "flux_controlnet_phase5_cross_v4"
    from controlnet_train.training.flux_phase5_cross_v3 import run_cross_v3_training

    run_cross_v3_training(args)


if __name__ == "__main__":
    main()

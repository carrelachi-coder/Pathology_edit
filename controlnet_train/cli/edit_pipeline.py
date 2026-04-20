"""Run the unified Phase 5.4 edit pipeline from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controlnet_train.inference import (
    EditPipelineInputs,
    load_cross_bundle,
    load_inpaint_bundle,
    run_cross_v0_bundle,
    run_edit_pipeline,
    run_inpaint_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the unified Phase 5.4 edit pipeline.")
    parser.add_argument("--reference-image", required=True)
    parser.add_argument("--reference-tissue-mask", required=True)
    parser.add_argument("--reference-nuclei-mask", required=True)
    parser.add_argument("--target-tissue-mask", required=True)
    parser.add_argument("--target-nuclei-mask", required=True)
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--inpaint-checkpoint", required=True)
    parser.add_argument("--cross-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--force-mode", choices=["inpaint", "cross"], default=None)
    parser.add_argument("--save-debug-artifacts", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def parse_args(args=None) -> argparse.Namespace:
    return build_parser().parse_args(args)


def main(argv=None) -> int:
    args = parse_args(argv)
    inputs = EditPipelineInputs(
        reference_image=args.reference_image,
        reference_tissue_mask=args.reference_tissue_mask,
        reference_nuclei_mask=args.reference_nuclei_mask,
        target_tissue_mask=args.target_tissue_mask,
        target_nuclei_mask=args.target_nuclei_mask,
        output_dir=args.output_dir,
        prompt=args.prompt,
        dataset=args.dataset,
        force_mode=args.force_mode,
        save_debug_artifacts=args.save_debug_artifacts,
    )
    run_edit_pipeline(
        inputs=inputs,
        inpaint_bundle=load_inpaint_bundle(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            checkpoint_path=args.inpaint_checkpoint,
            device=args.device,
        ),
        cross_bundle=load_cross_bundle(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            checkpoint_path=args.cross_checkpoint,
            device=args.device,
        ),
        inpaint_runner=run_inpaint_bundle,
        cross_runner=run_cross_v0_bundle,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

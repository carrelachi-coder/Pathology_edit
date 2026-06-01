"""Train Phase 5.3 Cross V3 Flux ControlNet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse_args(input_args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Phase 5.3 Cross V3 Flux ControlNet: fixed prompt, target-only "
            "ControlNet mask, and reference latent+mask cross-attention tokens."
        )
    )
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--controlnet_model_name_or_path", type=str, default=None)
    parser.add_argument(
        "--conditioning-checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint directory containing phase5_conditioning.pt for initializing "
            "HTE/tissue/nuclei modules. Defaults to --controlnet_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--load-conditioning-from-checkpoint",
        action="store_true",
        help="Initialize HTE/tissue/nuclei conditioning modules from a checkpoint.",
    )
    parser.add_argument("--train-metadata", type=str, required=True)
    parser.add_argument(
        "--stain-augmentation",
        type=str,
        default="none",
        choices=["none", "hed_aggressive"],
        help="Self-supervised stain perturbation applied in the dataloader.",
    )
    parser.add_argument(
        "--stain-counterfactual-prob",
        type=float,
        default=0.0,
        help="Probability of expanding a pair into two independently stained HED variants.",
    )
    parser.add_argument("--hed-sigma", type=float, default=0.2)
    parser.add_argument("--hed-beta", type=float, default=0.02)
    parser.add_argument("--hed-strong-alpha-sampling", action="store_true")
    parser.add_argument("--hed-alpha-min", type=float, default=0.4)
    parser.add_argument("--hed-alpha-low", type=float, default=0.75)
    parser.add_argument("--hed-alpha-high", type=float, default=1.25)
    parser.add_argument("--hed-alpha-max", type=float, default=1.8)
    parser.add_argument(
        "--prompt-source",
        type=str,
        default="fixed",
        choices=["fixed", "dataset", "metadata"],
        help="Deprecated for Cross V3; prompts are always fixed to 'histopathology image'.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Deprecated for Cross V3; ignored because the prompt is fixed to 'histopathology image'.",
    )
    parser.add_argument("--output-dir", type=str, default="phase5-controlnet-cross-v3")
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-double-layers", type=int, default=4)
    parser.add_argument("--num-single-layers", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lr-scheduler", type=str, default="cosine")
    parser.add_argument("--lr-warmup-steps", type=int, default=500)
    parser.add_argument("--lr-num-cycles", type=int, default=1)
    parser.add_argument("--lr-power", type=float, default=1.0)
    parser.add_argument("--scale-lr", action="store_true", default=False)
    parser.add_argument(
        "--conditioning-learning-rate",
        type=float,
        default=None,
        help="Learning rate for HTE, tissue downsampler, and nuclei encoder.",
    )
    parser.add_argument(
        "--controlnet-train-mode",
        type=str,
        default="all",
        choices=["all", "outputs"],
        help="'all' trains the whole ControlNet; 'outputs' trains residual outputs by default.",
    )
    parser.add_argument("--controlnet-train-x-embedder", action="store_true")
    parser.add_argument("--controlnet-train-last-n-blocks", type=int, default=0)
    parser.add_argument("--controlnet-train-last-n-single-blocks", type=int, default=0)
    parser.add_argument("--use-8bit-adam", action="store_true")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-weight-decay", type=float, default=1e-2)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=2)
    parser.add_argument("--checkpointing-steps", type=int, default=500)
    parser.add_argument("--checkpoints-total-limit", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--mixed-precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--enable-xformers-memory-efficient-attention", action="store_true")
    parser.add_argument("--set-grads-to-none", action="store_true")
    parser.add_argument("--save-weight-dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument(
        "--weighting-scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
    )
    parser.add_argument("--logit-mean", type=float, default=0.0)
    parser.add_argument("--logit-std", type=float, default=1.0)
    parser.add_argument("--mode-scale", type=float, default=1.29)
    parser.add_argument(
        "--proportion-empty-prompts",
        type=float,
        default=0.0,
        help="Deprecated for Cross V3; ignored so every sample uses the fixed prompt.",
    )
    parser.add_argument("--report-to", type=str, default="tensorboard")
    parser.add_argument("--tracker-project-name", type=str, default="flux_controlnet_phase5_cross_v3")
    parser.add_argument("--prompt-batch-size", type=int, default=8)
    parser.add_argument(
        "--self-reconstruction-warmup-steps",
        type=int,
        default=0,
        help="For the first N optimizer steps, set reference=image/masks to target=image/masks.",
    )
    parser.add_argument(
        "--self-reconstruction-sample-prob",
        type=float,
        default=0.0,
        help="After warmup, randomly convert this fraction of the batch to reference=target samples.",
    )
    parser.add_argument("--reference-latent-channels", type=int, default=16)
    parser.add_argument(
        "--reference-token-dim",
        type=int,
        default=4096,
        help="Joint-attention embedding width for projected reference latent+mask tokens.",
    )
    parser.add_argument(
        "--reference-token-hidden-dim",
        type=int,
        default=4096,
        help="Hidden width inside the reference context projection MLP.",
    )
    parser.add_argument(
        "--reference-token-output-init-std",
        type=float,
        default=0.02,
        help="Normal init std for the reference token output projection; bias is zeroed.",
    )
    parser.add_argument(
        "--ref-check-step",
        type=int,
        default=10,
        help="Optimizer step for [REF-CHECK] with-ref vs zero-ref noise-pred diff; set 0 to disable.",
    )
    parser.add_argument("--tissue-embedding-dim", type=int, default=64)
    parser.add_argument("--tissue-out-channels", type=int, default=64)
    parser.add_argument("--nuclei-embedding-dim", type=int, default=16)
    parser.add_argument("--nuclei-out-channels", type=int, default=16)
    parser.add_argument("--condition-downsample-blocks", type=int, default=3)
    parser.add_argument("--cross-version", type=str, default="v3")
    return parser.parse_args(input_args)


def main(input_args=None) -> None:
    args = parse_args(input_args)
    args.cross_version = "v3"
    from controlnet_train.training.flux_phase5_cross_v3 import run_cross_v3_training

    run_cross_v3_training(args)


if __name__ == "__main__":
    main()

"""Train Phase 5.3 Cross V3 Flux ControlNet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse_args(input_args=None, description: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=description
        or (
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
    parser.add_argument(
        "--pair-difficulty-sampler",
        action="store_true",
        help="Sample cross pairs by pair_difficulty target ratios instead of natural metadata frequency.",
    )
    parser.add_argument(
        "--no-pair-difficulty-sampler",
        dest="pair_difficulty_sampler",
        action="store_false",
        help="Disable pair_difficulty weighted sampling.",
    )
    parser.add_argument(
        "--pair-difficulty-target-full",
        type=float,
        default=0.70,
        help="Target sampling fraction for pair_difficulty=full when --pair-difficulty-sampler is enabled.",
    )
    parser.add_argument(
        "--pair-difficulty-target-partial",
        type=float,
        default=0.25,
        help="Target sampling fraction for pair_difficulty=partial when --pair-difficulty-sampler is enabled.",
    )
    parser.add_argument(
        "--pair-difficulty-target-low",
        type=float,
        default=0.05,
        help="Target sampling fraction for pair_difficulty=low when --pair-difficulty-sampler is enabled.",
    )
    parser.add_argument("--checkpointing-steps", type=int, default=500)
    parser.add_argument("--checkpoints-total-limit", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--mixed-precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument(
        "--max-cuda-memory-gb",
        type=float,
        default=0.0,
        help="Abort training if per-process peak CUDA reserved memory exceeds this many GiB. 0 disables.",
    )
    parser.add_argument(
        "--cuda-memory-check-interval",
        type=int,
        default=10,
        help="Check/log peak CUDA memory every N optimizer steps. Use 0 to check only on diagnose steps.",
    )
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
        "--reference-route-anchor-mode",
        type=str,
        default="none",
        choices=["none", "coarse", "fine"],
        help=(
            "Optional semantic route anchors for reference tokens. Keep 'none' for pure appearance "
            "transfer; 'coarse' or 'fine' are explicit semantic-routing experiments."
        ),
    )
    parser.add_argument(
        "--reference-route-embedding-init-std",
        type=float,
        default=0.02,
        help="Normal init std for learned reference route/type embeddings.",
    )
    parser.add_argument(
        "--reference-style-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Weight for region-level reference stain/style loss. The loss matches RGB "
            "mean/std/covariance on target/reference regions that share tissue or nuclei labels."
        ),
    )
    parser.add_argument(
        "--reference-style-loss-interval",
        type=int,
        default=1,
        help="Compute reference style loss every N optimizer steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--reference-style-tissue-weight",
        type=float,
        default=1.0,
        help="Relative weight for shared tissue-mask region style terms.",
    )
    parser.add_argument(
        "--reference-style-nuclei-weight",
        type=float,
        default=1.0,
        help="Relative weight for shared nuclei-mask region style terms.",
    )
    parser.add_argument(
        "--reference-style-mean-weight",
        type=float,
        default=1.0,
        help="Relative weight for per-region RGB mean matching.",
    )
    parser.add_argument(
        "--reference-style-std-weight",
        type=float,
        default=1.0,
        help="Relative weight for per-region RGB standard-deviation matching.",
    )
    parser.add_argument(
        "--reference-style-cov-weight",
        type=float,
        default=0.25,
        help="Relative weight for per-region RGB covariance matching.",
    )
    parser.add_argument(
        "--reference-style-min-pixels",
        type=int,
        default=32,
        help="Minimum target/reference pixels required for a tissue or nuclei class to enter style loss.",
    )
    parser.add_argument(
        "--reference-style-max-regions-per-sample",
        type=int,
        default=None,
        help="Optional cap on style-loss regions per sample, keeping largest labels first.",
    )
    parser.add_argument(
        "--ref-swap-loss-weight",
        type=float,
        default=0.1,
        help=(
            "Weight for reference-swap sensitivity loss. It compares normal-reference denoising "
            "loss against zero/shuffled-reference denoising losses with a margin."
        ),
    )
    parser.add_argument(
        "--ref-swap-loss-interval",
        type=int,
        default=1,
        help="Compute reference-swap sensitivity loss every N optimizer steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--ref-swap-margin",
        type=float,
        default=0.08,
        help="Required per-sample denoising-loss margin between normal and swapped references.",
    )
    parser.add_argument(
        "--ref-swap-variants",
        type=str,
        default="zero",
        help="Comma-separated swapped-reference variants for ref-swap loss: zero, random.",
    )
    parser.add_argument(
        "--ref-check-step",
        type=int,
        default=10,
        help=(
            "Optimizer step for [REF-CHECK] reference/control 2x2 noise-pred diffs; "
            "set 0 to disable."
        ),
    )
    parser.add_argument("--tissue-embedding-dim", type=int, default=64)
    parser.add_argument(
        "--target-tissue-encoding",
        type=str,
        default="shared_hte",
        choices=["shared_hte", "low_capacity_hte", "one_hot"],
        help=(
            "Target-side tissue condition for ControlNet. 'one_hot' is fixed "
            "non-learnable class layout; reference-side HTE remains learnable."
        ),
    )
    parser.add_argument(
        "--target-tissue-embedding-dim",
        type=int,
        default=None,
        help=(
            "Embedding dim for --target-tissue-encoding=low_capacity_hte. "
            "Defaults to --tissue-embedding-dim."
        ),
    )
    parser.add_argument(
        "--target-one-hot-scale",
        type=float,
        default=4.0,
        help=(
            "Fixed multiplier for --target-tissue-encoding=one_hot. This keeps "
            "target layout feature magnitude comparable to learned tissue features."
        ),
    )
    parser.add_argument("--tissue-out-channels", type=int, default=64)
    parser.add_argument("--nuclei-embedding-dim", type=int, default=16)
    parser.add_argument("--nuclei-out-channels", type=int, default=16)
    parser.add_argument("--condition-downsample-blocks", type=int, default=3)
    parser.add_argument(
        "--cross-v4-tissue-prior-tokens-per-class",
        type=int,
        default=4,
        help="Cross V4 only: number of learned coarse tissue prior tokens per class.",
    )
    parser.add_argument(
        "--cross-v4-cell-prior-tokens-per-class",
        type=int,
        default=0,
        help="Cross V4 only: number of learned cell prior tokens per class, including background.",
    )
    parser.add_argument(
        "--cross-v4-global-style-tokens",
        type=int,
        default=0,
        help="Cross V4 only: weak global style tokens derived from pooled reference local tokens.",
    )
    parser.add_argument("--cross-v4-prior-init-std", type=float, default=0.02)
    parser.add_argument(
        "--cross-v4-biased-double-blocks",
        type=str,
        default="last",
        help="Cross V4 only: double transformer blocks receiving correspondence bias, e.g. last, all, 1,3, or off.",
    )
    parser.add_argument("--cross-v4-bias-scale", type=float, default=1.0)
    parser.add_argument("--cross-v4-bias-warmup-steps", type=int, default=1000)
    parser.add_argument("--cross-v4-same-fine-bias", type=float, default=3.0)
    parser.add_argument("--cross-v4-same-coarse-bias", type=float, default=2.0)
    parser.add_argument("--cross-v4-mismatch-bias", type=float, default=-2.0)
    parser.add_argument("--cross-v4-cell-similarity-bias", type=float, default=1.0)
    parser.add_argument("--cross-v4-density-gap-bias", type=float, default=0.5)
    parser.add_argument("--cross-v4-prior-present-bias", type=float, default=0.5)
    parser.add_argument("--cross-v4-prior-missing-bias", type=float, default=3.0)
    parser.add_argument("--cross-v4-prior-wrong-class-bias", type=float, default=-2.0)
    parser.add_argument("--cross-v4-cell-prior-bias", type=float, default=1.0)
    parser.add_argument(
        "--cross-v4-diagnose-steps",
        type=str,
        default="1,10,100,500,1000,1500,2000",
        help="Cross V4 only: comma-separated optimizer steps for strong early diagnostics.",
    )
    parser.add_argument(
        "--cross-v4-diagnose-interval",
        type=int,
        default=0,
        help="Cross V4 only: additionally emit diagnostics every N optimizer steps. 0 disables.",
    )
    parser.add_argument(
        "--cross-v4-diagnose-jsonl",
        type=str,
        default=None,
        help="Cross V4 only: path for JSONL diagnostic snapshots. Defaults to output_dir/cross_v4_diagnostics.jsonl.",
    )
    parser.add_argument(
        "--cross-v4-extreme-bias-smoke",
        action="store_true",
        help="Cross V4 only: apply strict pass/fail thresholds for the +50/-50 attention-bias smoke test.",
    )
    parser.add_argument("--cross-version", type=str, default="v3")
    return parser.parse_args(input_args)


def main(input_args=None) -> None:
    args = parse_args(input_args)
    args.cross_version = "v3"
    from controlnet_train.training.flux_phase5_cross_v3 import run_cross_v3_training

    run_cross_v3_training(args)


if __name__ == "__main__":
    main()

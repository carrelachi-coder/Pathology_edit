"""Train Phase 5.3 Cross V1 Flux ControlNet with IP-Adapter reference attention."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse_args(input_args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Phase 5.3 Cross V1 Flux ControlNet with IP-Adapter reference attention."
    )
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--controlnet_model_name_or_path", type=str, default=None)
    parser.add_argument(
        "--a1-lite",
        action="store_true",
        help=(
            "Freeze an existing Cross V1 ControlNet and spatial conditioning modules; "
            "train only freshly initialized A1 IP-Adapter modules plus ref_encoder."
        ),
    )
    parser.add_argument(
        "--a1-lite-conditioning-checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint directory containing phase5_conditioning.pt for A1-lite. "
            "Defaults to --controlnet_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--conditioning-checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint directory containing phase5_conditioning.pt for initializing "
            "spatial conditioning modules. Defaults to --controlnet_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--load-conditioning-from-checkpoint",
        action="store_true",
        help=(
            "Initialize HTE/tissue/nuclei conditioning modules from --conditioning-checkpoint "
            "or --controlnet_model_name_or_path. Useful for A3 target-only probes from an "
            "existing Cross V1 checkpoint."
        ),
    )
    parser.add_argument(
        "--a1-lite-load-ref-encoder",
        action="store_true",
        help="Also initialize ref_encoder trainable parts from the A1-lite conditioning checkpoint.",
    )
    parser.add_argument("--train-metadata", type=str, required=True)
    parser.add_argument(
        "--stain-augmentation",
        type=str,
        default="none",
        choices=["none", "hed_aggressive"],
        help=(
            "Self-supervised stain perturbation applied in the dataloader. "
            "'hed_aggressive' applies the same sampled HED perturbation to "
            "reference and target images."
        ),
    )
    parser.add_argument(
        "--hed-sigma",
        type=float,
        default=0.2,
        help="H/E concentration scale jitter for --stain-augmentation hed_aggressive.",
    )
    parser.add_argument(
        "--hed-beta",
        type=float,
        default=0.02,
        help="H/E concentration additive jitter for --stain-augmentation hed_aggressive.",
    )
    parser.add_argument(
        "--hed-strong-alpha-sampling",
        action="store_true",
        help=(
            "Sample H/E alpha from two ranges away from 1 instead of uniform 1±sigma. "
            "This prevents near-identity stain perturbations."
        ),
    )
    parser.add_argument("--hed-alpha-min", type=float, default=0.4)
    parser.add_argument("--hed-alpha-low", type=float, default=0.75)
    parser.add_argument("--hed-alpha-high", type=float, default=1.25)
    parser.add_argument("--hed-alpha-max", type=float, default=1.8)
    parser.add_argument(
        "--uni-checkpoint-path", type=str, required=True,
        help="Path to UNI2-h ViT-Giant/14 checkpoint (pytorch_model.bin).",
    )
    parser.add_argument(
        "--prompt-source", type=str, default="dataset",
        choices=["dataset", "metadata"],
        help="Use dataset-level default prompts or the prompt stored in metadata.",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Override every training sample with one prompt.")
    parser.add_argument("--output-dir", type=str, default="phase5-controlnet-cross-v1")
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
    parser.add_argument("--use-8bit-adam", action="store_true")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-weight-decay", type=float, default=1e-2)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument(
        "--dataloader-prefetch-factor",
        type=int,
        default=2,
        help="Number of batches each DataLoader worker prefetches when workers are enabled.",
    )
    parser.add_argument("--checkpointing-steps", type=int, default=500)
    parser.add_argument("--checkpoints-total-limit", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--mixed-precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--enable-xformers-memory-efficient-attention", action="store_true")
    parser.add_argument("--set-grads-to-none", action="store_true")
    parser.add_argument("--save-weight-dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--weighting-scheme", type=str, default="logit_normal",
                        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    parser.add_argument("--logit-mean", type=float, default=0.0)
    parser.add_argument("--logit-std", type=float, default=1.0)
    parser.add_argument("--mode-scale", type=float, default=1.29)
    parser.add_argument("--proportion-empty-prompts", type=float, default=0.0)
    parser.add_argument("--report-to", type=str, default="tensorboard")
    parser.add_argument("--tracker-project-name", type=str, default="flux_controlnet_phase5_cross_v1")
    parser.add_argument("--prompt-batch-size", type=int, default=8)
    parser.add_argument("--reference-num-tokens", type=int, default=16)
    parser.add_argument("--reference-num-perceiver-layers", type=int, default=2)
    parser.add_argument("--reference-perceiver-heads", type=int, default=8)
    parser.add_argument(
        "--reference-perceiver-cross-gate-init",
        type=float,
        default=None,
        help=(
            "Enable gated Cross-Attn mixing in the reference Perceiver. The gate is sigmoid(init): "
            "latents = gate * latents + (1 - gate) * cross_out. Try -2.0 to make inputs dominate."
        ),
    )
    parser.add_argument(
        "--disable-reference-perceiver-self-attn",
        action="store_true",
        help=(
            "Skip the self-attention sub-block inside each reference Perceiver layer. "
            "This keeps reference-specific UNI/proj_mlp signal from being dominated by latent queries."
        ),
    )
    parser.add_argument(
        "--skip-reference-perceiver",
        action="store_true",
        help=(
            "Bypass the reference Perceiver entirely and feed projected UNI patch tokens "
            "directly to the IP-Adapter."
        ),
    )
    parser.add_argument(
        "--self-reconstruction-warmup-steps",
        type=int,
        default=0,
        help=(
            "For the first N optimizer steps, replace reference image/masks with the target "
            "image/masks so the IP-Adapter gets a direct same-patch reconstruction signal."
        ),
    )
    parser.add_argument(
        "--self-reconstruction-sample-prob",
        type=float,
        default=0.0,
        help=(
            "After warmup, randomly convert this fraction of each batch to same-patch "
            "self-reconstruction samples by setting reference=target."
        ),
    )
    parser.add_argument(
        "--self-reconstruction-l1-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for pixel-level L1 loss between decoded prediction and reference image "
            "on self-reconstruction samples."
        ),
    )
    parser.add_argument(
        "--cross-v1-spatial-mode",
        type=str,
        default="reference_target",
        choices=["reference_target", "target_only"],
        help=(
            "Spatial ControlNet conditioning mode. 'reference_target' is the original Cross V1 "
            "layout; 'target_only' is the A3 probe that removes reference masks from ControlNet."
        ),
    )
    parser.add_argument(
        "--ip-init-gain",
        type=float,
        default=0.1,
        help="Xavier gain for freshly initialized IP-Adapter K/V projections.",
    )
    parser.add_argument(
        "--ip-ref-learning-rate",
        type=float,
        default=None,
        help=(
            "Learning rate for ref_encoder and double-stream IP-Adapter modules. "
            "Defaults to 10x --learning-rate after optional LR scaling."
        ),
    )
    parser.add_argument(
        "--ip-adapter-checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint directory or phase5_ip_adapter.pt used to initialize shared projection "
            "and double-stream IP modules. Defaults to --controlnet_model_name_or_path when "
            "that directory contains phase5_ip_adapter.pt."
        ),
    )
    parser.add_argument(
        "--no-load-ip-adapter-from-controlnet",
        action="store_true",
        help=(
            "Do not auto-initialize IP-Adapter modules from --controlnet_model_name_or_path. "
            "Use this when the ControlNet should be warm-started but IP/ref modules should "
            "train from scratch."
        ),
    )
    parser.add_argument(
        "--load-single-ip-from-checkpoint",
        action="store_true",
        help=(
            "Also load saved single-stream IP modules from --ip-adapter-checkpoint. By default "
            "single-stream IP modules are freshly initialized."
        ),
    )
    parser.add_argument(
        "--ip-single-learning-rate",
        type=float,
        default=None,
        help=(
            "Learning rate for single-stream IP-Adapter modules. Defaults to "
            "--ip-ref-learning-rate."
        ),
    )
    parser.add_argument(
        "--ip-single-num-layers",
        type=int,
        default=10,
        help="Install single-stream IP-Adapter processors on the last N FLUX single blocks.",
    )
    parser.add_argument(
        "--perceptual-loss-weight",
        type=float,
        default=0.5,
        help="Weight for frozen UNI token cosine perceptual loss against the target image.",
    )
    parser.add_argument(
        "--perceptual-loss-interval",
        type=int,
        default=1,
        help="Compute perceptual loss every N optimizer steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--reference-style-loss-weight",
        type=float,
        default=5.0,
        help=(
            "Weight for region-level reference stain/style loss. The loss matches RGB "
            "mean/std/covariance on target/reference regions that share tissue or nuclei labels."
        ),
    )
    parser.add_argument(
        "--reference-style-loss-interval",
        type=int,
        default=1,
        help=(
            "Compute reference style loss every N optimizer steps. Use 1 for every step; "
            "use 0 to disable the style loss without changing its configured weight."
        ),
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
            "loss against zero/random-reference denoising losses with a margin."
        ),
    )
    parser.add_argument(
        "--ref-swap-loss-interval",
        type=int,
        default=1,
        help=(
            "Compute reference-swap sensitivity loss every N optimizer steps. Use 1 for every step; "
            "use 0 to disable the swap loss without changing its configured weight."
        ),
    )
    parser.add_argument(
        "--ref-swap-margin",
        type=float,
        default=0.02,
        help="Required per-sample denoising-loss margin between normal and swapped references.",
    )
    parser.add_argument(
        "--ref-swap-variants",
        type=str,
        default="zero,random",
        help="Comma-separated swapped-reference variants for ref-swap loss: zero, random.",
    )
    parser.add_argument("--tissue-embedding-dim", type=int, default=64)
    parser.add_argument("--tissue-out-channels", type=int, default=64)
    parser.add_argument("--nuclei-embedding-dim", type=int, default=16)
    parser.add_argument("--nuclei-out-channels", type=int, default=16)
    parser.add_argument("--condition-downsample-blocks", type=int, default=3)
    # Internal: cross-version is always v1 for this CLI
    parser.add_argument("--cross-version", type=str, default="v1")
    return parser.parse_args(input_args)


def main(input_args=None) -> None:
    args = parse_args(input_args)
    args.cross_version = "v1"
    from controlnet_train.training.flux_phase5_cross_v1 import run_cross_v1_training
    run_cross_v1_training(args)


if __name__ == "__main__":
    main()

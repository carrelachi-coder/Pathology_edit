"""Train Phase 5.3 Cross V2.2 Flux ControlNet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse_args(input_args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Phase 5.3 Cross V2.2 Flux ControlNet with condition order "
            "[z_ref, ref_tissue_feat, ref_nuclei_feat, tar_tissue_feat, tar_nuclei_feat]."
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
        "--noising-degradation",
        type=str,
        default="none",
        choices=["none", "hed", "stain", "texture", "hed_texture", "stain_texture"],
        help=(
            "Clean image used to build noisy_input before denoising. "
            "Use hed_texture to train from wrong stain + degraded texture while "
            "keeping reference/ground truth as the original patch."
        ),
    )
    parser.add_argument("--texture-blur-prob", type=float, default=0.7)
    parser.add_argument("--texture-blur-sigma-min", type=float, default=0.4)
    parser.add_argument("--texture-blur-sigma-max", type=float, default=1.4)
    parser.add_argument("--texture-downsample-prob", type=float, default=0.7)
    parser.add_argument("--texture-downsample-scale-min", type=float, default=0.35)
    parser.add_argument("--texture-downsample-scale-max", type=float, default=0.75)
    parser.add_argument("--texture-noise-prob", type=float, default=0.35)
    parser.add_argument("--texture-noise-std-min", type=float, default=0.005)
    parser.add_argument("--texture-noise-std-max", type=float, default=0.03)
    parser.add_argument(
        "--degraded-noising-min-sigma",
        type=float,
        default=0.1,
        help=(
            "Minimum training sigma for samples whose noisy_input starts from a degraded image. "
            "Very low sigma makes degraded->ground-truth one-step restoration numerically sharp."
        ),
    )
    parser.add_argument(
        "--prompt-source",
        type=str,
        default="dataset",
        choices=["dataset", "metadata"],
        help="Use dataset-level default prompts or the prompt stored in metadata.",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Override every training sample with one prompt.")
    parser.add_argument("--output-dir", type=str, default="phase5-controlnet-cross-v2-2")
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
    parser.add_argument("--proportion-empty-prompts", type=float, default=0.0)
    parser.add_argument("--report-to", type=str, default="tensorboard")
    parser.add_argument("--tracker-project-name", type=str, default="flux_controlnet_phase5_cross_v2_2")
    parser.add_argument("--prompt-batch-size", type=int, default=8)
    parser.add_argument(
        "--reference-bank-block-size",
        type=int,
        default=4,
        help=(
            "Latent-grid block size for Cross V2.2 reference block-bank sampling. "
            "Larger blocks preserve more local texture; smaller blocks break reference layout more aggressively."
        ),
    )
    parser.add_argument(
        "--reference-bank-label-mode",
        type=str,
        default="tissue_nuclei",
        choices=["tissue", "nuclei", "tissue_nuclei"],
        help="Labels used to build reference latent block pools before target-mask broadcast.",
    )
    parser.add_argument(
        "--keep-reference-mask-features",
        action="store_true",
        help=(
            "Keep V2.1-style reference mask feature channels. By default V2.2 zeros these "
            "channels and uses reference masks only to construct the latent block bank."
        ),
    )
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
    parser.add_argument(
        "--reference-region-loss-weight",
        type=float,
        default=0.05,
        help=(
            "Small weight for region-level reference stain/style loss. The loss matches "
            "prediction target regions to reference regions with the same tissue/nuclei labels."
        ),
    )
    parser.add_argument(
        "--reference-region-loss-warmup-steps",
        type=int,
        default=500,
        help="Linearly warm up reference-region loss weight over this many optimizer steps.",
    )
    parser.add_argument(
        "--reference-region-loss-interval",
        type=int,
        default=1,
        help="Compute reference-region loss every N optimizer steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--reference-region-loss-min-sigma",
        type=float,
        default=0.0,
        help="Minimum sigma for reference-region loss timestep gating.",
    )
    parser.add_argument(
        "--reference-region-loss-max-sigma",
        type=float,
        default=0.6,
        help="Maximum sigma for reference-region loss timestep gating; keeps the loss in low/mid noise.",
    )
    parser.add_argument("--reference-region-tissue-weight", type=float, default=1.0)
    parser.add_argument("--reference-region-nuclei-weight", type=float, default=1.0)
    parser.add_argument("--reference-region-mean-weight", type=float, default=1.0)
    parser.add_argument("--reference-region-std-weight", type=float, default=1.0)
    parser.add_argument("--reference-region-cov-weight", type=float, default=0.25)
    parser.add_argument("--reference-region-min-pixels", type=int, default=32)
    parser.add_argument("--reference-region-max-regions-per-sample", type=int, default=None)
    parser.add_argument(
        "--uni-checkpoint-path",
        type=str,
        default=None,
        help="Optional UNI2-h checkpoint when --reference-perceptual-backend=uni.",
    )
    parser.add_argument(
        "--same-wsi-appearance-checkpoint",
        type=str,
        default=None,
        help="Frozen same-WSI appearance encoder checkpoint for reference perceptual texture loss.",
    )
    parser.add_argument(
        "--reference-perceptual-backend",
        type=str,
        default="vgg",
        choices=["vgg", "same_wsi", "uni"],
        help="Feature backend for reference perceptual texture loss.",
    )
    parser.add_argument(
        "--reference-vgg-weights",
        type=str,
        default="imagenet",
        choices=["imagenet", "none"],
        help="Weights for --reference-perceptual-backend=vgg.",
    )
    parser.add_argument(
        "--reference-vgg-weights-path",
        type=str,
        default=None,
        help="Optional local VGG16 state dict path; avoids downloading torchvision weights.",
    )
    parser.add_argument(
        "--reference-vgg-layers",
        type=str,
        default="relu1_1,relu1_2,relu2_1,relu2_2",
        help="Comma-separated VGG16 feature layers for --reference-perceptual-backend=vgg.",
    )
    parser.add_argument(
        "--reference-vgg-loss-type",
        type=str,
        default="gram",
        choices=["gram", "feature_l1"],
        help="VGG loss for --reference-perceptual-backend=vgg.",
    )
    parser.add_argument(
        "--reference-vgg-rgb",
        action="store_true",
        help="Disable default grayscale conversion before VGG; intended only for ablations.",
    )
    parser.add_argument(
        "--reference-vgg-input-size",
        type=int,
        default=256,
        help="Resize generated/reference images to this square size before VGG features. Use 0 to disable.",
    )
    parser.add_argument(
        "--reference-perceptual-loss-weight",
        type=float,
        default=0.0,
        help="Small weight for frozen reference perceptual texture loss against the reference image.",
    )
    parser.add_argument("--reference-perceptual-loss-warmup-steps", type=int, default=500)
    parser.add_argument("--reference-perceptual-loss-interval", type=int, default=1)
    parser.add_argument("--reference-perceptual-loss-min-sigma", type=float, default=0.0)
    parser.add_argument("--reference-perceptual-loss-max-sigma", type=float, default=0.4)
    parser.add_argument("--reference-perceptual-min-pixels", type=int, default=8)
    parser.add_argument("--reference-perceptual-mean-weight", type=float, default=1.0)
    parser.add_argument("--reference-perceptual-std-weight", type=float, default=1.0)
    parser.add_argument("--reference-perceptual-pooled-cosine-weight", type=float, default=0.25)
    parser.add_argument(
        "--reference-grad-ratio-interval",
        type=int,
        default=0,
        help=(
            "When > 0, log ControlNet grad-norm ratios every N batches where "
            "reference perceptual loss is active. Uses weighted "
            "region+perceptual appearance loss versus denoise loss."
        ),
    )
    parser.add_argument("--reference-latent-channels", type=int, default=16)
    parser.add_argument("--tissue-embedding-dim", type=int, default=64)
    parser.add_argument("--tissue-out-channels", type=int, default=64)
    parser.add_argument("--nuclei-embedding-dim", type=int, default=16)
    parser.add_argument("--nuclei-out-channels", type=int, default=16)
    parser.add_argument("--condition-downsample-blocks", type=int, default=3)
    parser.add_argument("--cross-version", type=str, default="v2.2")
    return parser.parse_args(input_args)


def main(input_args=None) -> None:
    args = parse_args(input_args)
    args.cross_version = "v2.2"
    from controlnet_train.training.flux_phase5_cross_v2_2 import run_cross_v2_2_training

    run_cross_v2_2_training(args)


if __name__ == "__main__":
    main()

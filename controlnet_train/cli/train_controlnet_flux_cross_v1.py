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
    parser.add_argument(
        "--load-ref-encoder-from-checkpoint",
        action="store_true",
        help=(
            "Initialize ref_encoder trainable parts from the conditioning checkpoint "
            "without enabling A1-lite freezing."
        ),
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
        "--stain-counterfactual-prob",
        type=float,
        default=0.0,
        help=(
            "Probability that a metadata pair is expanded into two HED variants with "
            "identical masks but independently sampled reference/target stain. This "
            "forces stain differences to be explained by the reference image."
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
        "--noising-degradation",
        type=str,
        default="none",
        choices=["none", "hed", "stain", "texture", "hed_texture", "stain_texture"],
        help=(
            "Clean image used to build noisy_model_input. Use hed_texture to start "
            "training from a degraded target while the supervision target remains clean."
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
        help="Minimum sigma for samples whose noisy_model_input starts from a degraded target.",
    )
    parser.add_argument(
        "--uni-checkpoint-path", type=str, default=None,
        help=(
            "Path to UNI2-h ViT-Giant/14 checkpoint (pytorch_model.bin). Required only "
            "when using UNI reference IP, UNI perceptual loss, or UNI region loss."
        ),
    )
    parser.add_argument(
        "--conch-checkpoint-path",
        type=str,
        default=None,
        help=(
            "Path to CONCH checkpoint (pytorch_model.bin). Required when "
            "--reference-region-loss-backend=conch and region loss weight > 0."
        ),
    )
    parser.add_argument(
        "--conch-root",
        type=str,
        default=None,
        help=(
            "Path to local CONCH repository root. If omitted, it is inferred from "
            "--conch-checkpoint-path."
        ),
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
    parser.add_argument("--lr-num-cycles", type=float, default=1.0)
    parser.add_argument("--lr-power", type=float, default=1.0)
    parser.add_argument(
        "--lr-min-factor",
        type=float,
        default=0.0,
        help=(
            "Minimum LR as a fraction of each optimizer group's initial LR. "
            "Only used by --lr-scheduler cosine_with_min_lr."
        ),
    )
    parser.add_argument(
        "--lr-decay-start-step",
        type=int,
        default=0,
        help=(
            "Global optimizer step where cosine_with_min_lr starts decaying. "
            "Use the resume checkpoint step for a phase-local cosine schedule."
        ),
    )
    parser.add_argument("--scale-lr", action="store_true", default=False)
    parser.add_argument(
        "--conditioning-learning-rate",
        type=float,
        default=None,
        help=(
            "Learning rate for HTE, tissue downsampler, and nuclei conditioning modules. "
            "Defaults to --learning-rate."
        ),
    )
    parser.add_argument(
        "--controlnet-train-mode",
        type=str,
        default="all",
        choices=["all", "outputs"],
        help=(
            "ControlNet unfreeze policy for non-A1-lite training. 'all' trains the whole "
            "ControlNet; 'outputs' trains only residual output projections by default."
        ),
    )
    parser.add_argument(
        "--controlnet-train-x-embedder",
        action="store_true",
        help="With --controlnet-train-mode outputs, also train controlnet_x_embedder.",
    )
    parser.add_argument(
        "--controlnet-train-last-n-blocks",
        type=int,
        default=0,
        help=(
            "With --controlnet-train-mode outputs, also train the last N double-stream "
            "ControlNet transformer blocks."
        ),
    )
    parser.add_argument(
        "--controlnet-train-last-n-single-blocks",
        type=int,
        default=0,
        help=(
            "With --controlnet-train-mode outputs, also train the last N single-stream "
            "ControlNet transformer blocks."
        ),
    )
    parser.add_argument("--use-8bit-adam", action="store_true")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-weight-decay", type=float, default=1e-2)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="Enable EMA for trainable Cross V1 modules when > 0, e.g. 0.999.",
    )
    parser.add_argument(
        "--ema-device",
        type=str,
        default="cpu",
        choices=["cpu", "model"],
        help="Keep EMA shadow weights on CPU for memory safety or beside model params for speed.",
    )
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
    parser.add_argument(
        "--mask-augmentation",
        type=str,
        default="none",
        choices=["none", "affine_coarse"],
        help="Train-time label-mask augmentation for target/reference tissue and nuclei masks.",
    )
    parser.add_argument("--mask-augment-prob", type=float, default=0.0)
    parser.add_argument("--mask-augment-translate", type=float, default=0.03)
    parser.add_argument("--mask-augment-scale", type=float, default=0.04)
    parser.add_argument("--mask-augment-rotate-degrees", type=float, default=3.0)
    parser.add_argument("--mask-augment-boundary-jitter", type=float, default=0.0)
    parser.add_argument("--mask-augment-boundary-grid", type=int, default=8)
    parser.add_argument("--mask-augment-coarse-prob", type=float, default=0.0)
    parser.add_argument("--mask-augment-coarse-factor", type=int, default=4)
    parser.add_argument("--report-to", type=str, default="tensorboard")
    parser.add_argument("--tracker-project-name", type=str, default="flux_controlnet_phase5_cross_v1")
    parser.add_argument("--prompt-batch-size", type=int, default=8)
    parser.add_argument(
        "--reference-ip-embedding-backend",
        type=str,
        default="uni",
        choices=["uni", "vae", "vae_latent", "latent"],
        help=(
            "Reference embedding source for IP-Adapter injection. 'uni' uses frozen UNI "
            "spatial tokens; 'vae_latent' encodes the reference image through the frozen "
            "FLUX VAE and projects packed latent cells into IP tokens."
        ),
    )
    parser.add_argument(
        "--reference-vae-latent-channels",
        type=int,
        default=16,
        help="Channel count expected from the frozen FLUX VAE latent grid for ref VAE tokens.",
    )
    parser.add_argument(
        "--reference-vae-token-grid-size",
        type=int,
        default=32,
        help=(
            "Square token grid for ref VAE latent tokens. For 512px images and 64x64 VAE "
            "latents, 32 with pack factor 2 gives 1024 reference tokens."
        ),
    )
    parser.add_argument(
        "--reference-vae-pack-factor",
        type=int,
        default=2,
        help="Local latent grid pack factor before projecting ref VAE latent tokens.",
    )
    parser.add_argument(
        "--reference-uni-feature-layer",
        type=int,
        default=None,
        help=(
            "Use a 1-based UNI transformer block output for reference IP tokens instead "
            "of final UNI tokens. For the layer-06 texture run, set this to 6."
        ),
    )
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
        "--regional-ip-adapter",
        action="store_true",
        help=(
            "Use mask-guided regional IP-Adapter attention. Reference UNI spatial tokens "
            "are labeled by reference tissue mask, and target image tokens can only attend "
            "same-label reference tokens."
        ),
    )
    parser.add_argument(
        "--cross-v1-ip-architecture",
        type=str,
        default=None,
        choices=[
            "global",
            "regional_hard",
            "regional-hard",
            "global_soft_bias",
            "global-soft-bias",
            "soft_bias",
            "soft-bias",
        ],
        help=(
            "Reference-attention architecture. 'global_soft_bias' uses global "
            "dense IP attention with finite label-logit bias instead of hard "
            "regional masking."
        ),
    )
    parser.add_argument(
        "--regional-ip-token-mode",
        type=str,
        default="spatial",
        choices=[
            "spatial",
            "perceiver",
            "masked_perceiver",
            "region_perceiver",
            "stats",
            "mean_std",
            "region_stats",
        ],
        help=(
            "Reference token bank for regional IP. 'spatial' feeds raw projected UNI patch "
            "tokens; 'perceiver' compresses each mask label separately through the "
            "reference Perceiver before IP attention; 'stats' emits one mean+std "
            "statistics token per tissue label."
        ),
    )
    parser.add_argument(
        "--regional-ip-label-mode",
        type=str,
        default="tissue",
        choices=[
            "tissue",
            "tissue_only",
            "coarse",
            "coarse_tissue",
            "coarse-tissue",
            "parent_tissue",
            "parent-tissue",
            "tissue_nuclei",
            "tissue-nuclei",
            "tissue+nuclei",
            "composite",
            "nuclei",
            "nuclei_aware",
        ],
        help=(
            "Region labels used for regional IP attention. 'tissue' gates tumor/stroma "
            "labels only; 'tissue-nuclei' builds composite tissue+nuclei labels so nuclei "
            "classes are decoupled inside each tissue region."
        ),
    )
    parser.add_argument(
        "--regional-ip-soft-bias-init",
        type=float,
        default=4.0,
        help=(
            "Initial finite same-label attention logit bias b for "
            "--cross-v1-ip-architecture global_soft_bias. Same-label pairs get +b; "
            "different-label pairs get -b."
        ),
    )
    parser.add_argument(
        "--no-regional-ip-strict",
        dest="regional_ip_strict",
        action="store_false",
        help=(
            "Disable same-label region gating and let IP attention attend globally. "
            "By default strict mode routes unmatched or unlabeled query regions to "
            "a learned null IP token instead of silently falling back to global attention."
        ),
    )
    parser.set_defaults(regional_ip_strict=True)
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
        choices=["reference_target", "reference_target_delta", "target_only"],
        help=(
            "Spatial ControlNet conditioning mode. 'reference_target' is the original Cross V1 "
            "layout; 'reference_target_delta' appends target-reference feature deltas; "
            "'target_only' is the A3 probe that removes reference masks from ControlNet."
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
        "--ip-health-debug-interval",
        type=int,
        default=100,
        help=(
            "Every N optimizer steps, log IP/ref parameter deltas, normal-vs-zero and "
            "real-vs-real sensitivity, per-block IP residual ratios, and swap-reference "
            "health metrics. Use 0 to disable periodic health diagnostics."
        ),
    )
    parser.add_argument(
        "--ip-health-debug-warmup-steps",
        type=int,
        default=100,
        help="Delay hard IP/ref health warnings until this many optimizer steps have completed.",
    )
    parser.add_argument(
        "--ip-health-min-ref-l2",
        type=float,
        default=1e-6,
        help="Warning threshold for normal-vs-swapped noise_pred RMS distance at health checks.",
    )
    parser.add_argument(
        "--ip-health-min-swap-loss-gap",
        type=float,
        default=0.0,
        help="Warning threshold for paired-ref minus shuffled-ref denoising-loss gap.",
    )
    parser.add_argument(
        "--ip-health-max-ip-ratio",
        type=float,
        default=1.0,
        help="Warn when any block has ||scale*ip_out|| / ||hidden|| above this value.",
    )
    parser.add_argument(
        "--ip-health-min-ip-ratio",
        type=float,
        default=1e-8,
        help="Warn when every block has ||scale*ip_out|| / ||hidden|| below this value.",
    )
    parser.add_argument(
        "--perceptual-loss-weight",
        type=float,
        default=0.0,
        help="Weight for frozen UNI token cosine perceptual loss against the target image.",
    )
    parser.add_argument(
        "--perceptual-loss-interval",
        type=int,
        default=1,
        help="Compute perceptual loss every N optimizer steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--reference-region-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for class-matched reference region loss. With feature-map backends "
            "('uni' or 'conch'), the model prediction is VAE-decoded to RGB and compared "
            "to reference RGB through frozen encoder region statistics."
        ),
    )
    parser.add_argument(
        "--reference-region-loss-backend",
        type=str,
        default="uni",
        help=(
            "Reference region loss descriptor backend. Use 'rgb_fft' for an independent "
            "RGB/statistical + FFT descriptor, or 'uni' for decoded-RGB -> frozen-UNI "
            "region mean/std/cosine statistics, or 'conch' for decoded-RGB -> frozen-CONCH "
            "region mean/std/cosine statistics."
        ),
    )
    parser.add_argument("--reference-region-loss-interval", type=int, default=1)
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
        help="Maximum sigma for reference-region loss timestep gating; default targets low/mid noise.",
    )
    parser.add_argument("--reference-region-tissue-weight", type=float, default=1.0)
    parser.add_argument("--reference-region-nuclei-weight", type=float, default=0.0)
    parser.add_argument(
        "--reference-region-composite-weight",
        type=float,
        default=0.0,
        help=(
            "Extra reference region loss on composite tissue+nuclei labels. "
            "This keeps nuclei texture matching inside the correct tissue class."
        ),
    )
    parser.add_argument("--reference-region-mean-weight", type=float, default=1.0)
    parser.add_argument("--reference-region-std-weight", type=float, default=0.5)
    parser.add_argument(
        "--reference-region-gram-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for regional feature Gram-matrix matching in UNI/CONCH region loss. "
            "Use a small value because Gram is a channel-correlation statistic."
        ),
    )
    parser.add_argument(
        "--reference-region-conch-layer",
        type=int,
        default=None,
        help=(
            "Use a 1-based CONCH visual transformer block output for CONCH region loss "
            "instead of final CONCH tokens. For CONCH layer-06 mean/std+Gram, set this to 6."
        ),
    )
    parser.add_argument("--reference-region-fft-weight", type=float, default=0.25)
    parser.add_argument("--reference-region-fft-bins", type=int, default=6)
    parser.add_argument("--reference-region-fft-size", type=int, default=64)
    parser.add_argument("--reference-region-cosine-weight", type=float, default=0.25)
    parser.add_argument("--reference-region-min-pixels", type=int, default=32)
    parser.add_argument("--reference-region-min-tokens", type=int, default=2)
    parser.add_argument("--reference-region-max-regions-per-sample", type=int, default=None)
    parser.add_argument(
        "--reference-texture-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for regional VGG low-level texture loss. This compares masked VGG "
            "Gram matrices between decoded prediction regions and reference regions."
        ),
    )
    parser.add_argument("--reference-texture-loss-interval", type=int, default=1)
    parser.add_argument("--reference-texture-loss-min-sigma", type=float, default=0.0)
    parser.add_argument("--reference-texture-loss-max-sigma", type=float, default=0.6)
    parser.add_argument("--reference-texture-tissue-weight", type=float, default=1.0)
    parser.add_argument("--reference-texture-nuclei-weight", type=float, default=0.25)
    parser.add_argument("--reference-texture-composite-weight", type=float, default=0.0)
    parser.add_argument("--reference-texture-min-pixels", type=int, default=8)
    parser.add_argument(
        "--reference-vgg-weights",
        type=str,
        default="imagenet",
        choices=["imagenet", "default", "pretrained", "none", "random", "untrained"],
        help="Weights for regional VGG texture loss.",
    )
    parser.add_argument(
        "--reference-vgg-weights-path",
        type=str,
        default=None,
        help="Optional local VGG16 weights file for regional VGG texture loss.",
    )
    parser.add_argument(
        "--reference-vgg-layers",
        type=str,
        default="relu1_1,relu1_2,relu2_1,relu2_2",
        help="Comma-separated VGG16 feature layers for regional VGG texture loss.",
    )
    parser.add_argument(
        "--reference-vgg-loss-type",
        type=str,
        default="gram",
        choices=["gram", "style", "feature_l1", "l1", "feature"],
        help="VGG regional texture comparison type. 'gram' is position-independent.",
    )
    parser.add_argument(
        "--reference-vgg-rgb",
        action="store_true",
        help="Use RGB VGG inputs instead of grayscale replication for texture loss.",
    )
    parser.add_argument("--reference-vgg-input-size", type=int, default=256)
    parser.add_argument(
        "--reference-style-loss-weight",
        type=float,
        default=0.0,
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
        default=0.0,
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

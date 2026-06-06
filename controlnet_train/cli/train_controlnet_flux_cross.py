"""Train Phase 5.3 Flux ControlNet for same-WSI cross reconstruction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse_args(input_args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 5.3 Flux ControlNet for cross reconstruction.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--controlnet_model_name_or_path", type=str, default=None)
    parser.add_argument("--train-metadata", type=str, required=True)
    parser.add_argument(
        "--cross-version",
        type=str,
        default="v0",
        choices=["v0", "v1", "v2.1", "v2_1", "v21", "v3"],
    )
    parser.add_argument(
        "--prompt-source",
        type=str,
        default="dataset",
        choices=["dataset", "metadata"],
        help=(
            "Use dataset-level default prompts or the prompt stored in metadata. "
            "Cross defaults to dataset prompts to keep prompt caching small."
        ),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Override every training sample with one prompt.",
    )
    parser.add_argument("--output-dir", type=str, default="phase5-controlnet-cross")
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
    parser.add_argument("--checkpointing-steps", type=int, default=500)
    parser.add_argument("--checkpoints-total-limit", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--mixed-precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--enable-xformers-memory-efficient-attention", action="store_true")
    parser.add_argument("--set-grads-to-none", action="store_true")
    parser.add_argument("--save-weight-dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--weighting-scheme", type=str, default="logit_normal", choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    parser.add_argument("--logit-mean", type=float, default=0.0)
    parser.add_argument("--logit-std", type=float, default=1.0)
    parser.add_argument("--mode-scale", type=float, default=1.29)
    parser.add_argument("--proportion-empty-prompts", type=float, default=0.0)
    parser.add_argument("--report-to", type=str, default="tensorboard")
    parser.add_argument("--tracker-project-name", type=str, default="flux_controlnet_phase5_cross")
    parser.add_argument("--prompt-batch-size", type=int, default=8)
    parser.add_argument("--tissue-embedding-dim", type=int, default=64)
    parser.add_argument("--tissue-out-channels", type=int, default=64)
    parser.add_argument("--nuclei-embedding-dim", type=int, default=16)
    parser.add_argument("--nuclei-out-channels", type=int, default=16)
    parser.add_argument("--condition-downsample-blocks", type=int, default=3)
    return parser.parse_args(input_args)


def main(input_args=None) -> None:
    args = parse_args(input_args)
    normalized = args.cross_version.lower().replace("-", ".").replace("_", ".")
    if normalized in {"v2.1", "v21"}:
        from controlnet_train.training.flux_phase5_cross_v2_1 import run_cross_v2_1_training

        args.cross_version = "v2.1"
        run_cross_v2_1_training(args)
        return
    if normalized == "v3":
        from controlnet_train.training.flux_phase5_cross_v3 import run_cross_v3_training

        args.cross_version = "v3"
        if not hasattr(args, "reference_latent_channels"):
            args.reference_latent_channels = 16
        if not hasattr(args, "reference_token_dim"):
            args.reference_token_dim = 4096
        if not hasattr(args, "reference_token_hidden_dim"):
            args.reference_token_hidden_dim = 4096
        if not hasattr(args, "reference_token_output_init_std"):
            args.reference_token_output_init_std = 0.02
        if not hasattr(args, "reference_route_anchor_mode"):
            args.reference_route_anchor_mode = "none"
        if not hasattr(args, "reference_route_embedding_init_std"):
            args.reference_route_embedding_init_std = 0.02
        for name, value in (
            ("reference_style_loss_weight", 1.0),
            ("reference_style_loss_interval", 1),
            ("reference_style_tissue_weight", 1.0),
            ("reference_style_nuclei_weight", 1.0),
            ("reference_style_mean_weight", 1.0),
            ("reference_style_std_weight", 1.0),
            ("reference_style_cov_weight", 0.25),
            ("reference_style_min_pixels", 32),
            ("reference_style_max_regions_per_sample", None),
            ("ref_swap_loss_weight", 0.1),
            ("ref_swap_loss_interval", 1),
            ("ref_swap_margin", 0.08),
            ("ref_swap_variants", "zero"),
        ):
            if not hasattr(args, name):
                setattr(args, name, value)
        if not hasattr(args, "ref_check_step"):
            args.ref_check_step = 10
        if not hasattr(args, "conditioning_checkpoint"):
            args.conditioning_checkpoint = None
        if not hasattr(args, "load_conditioning_from_checkpoint"):
            args.load_conditioning_from_checkpoint = False
        if not hasattr(args, "controlnet_train_mode"):
            args.controlnet_train_mode = "all"
        if not hasattr(args, "controlnet_train_x_embedder"):
            args.controlnet_train_x_embedder = False
        if not hasattr(args, "controlnet_train_last_n_blocks"):
            args.controlnet_train_last_n_blocks = 0
        if not hasattr(args, "controlnet_train_last_n_single_blocks"):
            args.controlnet_train_last_n_single_blocks = 0
        if not hasattr(args, "conditioning_learning_rate"):
            args.conditioning_learning_rate = None
        if not hasattr(args, "stain_augmentation"):
            args.stain_augmentation = "none"
        if not hasattr(args, "stain_counterfactual_prob"):
            args.stain_counterfactual_prob = 0.0
        for name, value in (
            ("hed_sigma", 0.2),
            ("hed_beta", 0.02),
            ("hed_strong_alpha_sampling", False),
            ("hed_alpha_min", 0.4),
            ("hed_alpha_low", 0.75),
            ("hed_alpha_high", 1.25),
            ("hed_alpha_max", 1.8),
        ):
            if not hasattr(args, name):
                setattr(args, name, value)
        if not hasattr(args, "dataloader_prefetch_factor"):
            args.dataloader_prefetch_factor = 2
        if not hasattr(args, "self_reconstruction_warmup_steps"):
            args.self_reconstruction_warmup_steps = 0
        if not hasattr(args, "self_reconstruction_sample_prob"):
            args.self_reconstruction_sample_prob = 0.0
        run_cross_v3_training(args)
        return
    if normalized == "v1":
        raise NotImplementedError(
            "Use controlnet_train/cli/train_controlnet_flux_cross_v1.py for Cross V1."
        )

    from controlnet_train.training.flux_phase5 import run_cross_v0_training

    args.cross_version = "v0"
    run_cross_v0_training(args)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms.functional as TF

from .config import BaselineConfig
from .data import load_mask, normalize_image_tensor, nuclei_mask_to_density
from .inference import load_checkpoint, save_prediction
from .training import run_stage4_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 4 tissue segmentation baseline.")
    parser.add_argument("--dataset-root", required=True, help="Root containing patches/images and patches/tissue_masks.")
    parser.add_argument("--uni2h-repo", default="UNI-2h", help="Local UNI2-h repository path.")
    parser.add_argument("--output-dir", default="segmentator_runs/stage4_baseline")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--remap-invalid-to", type=int, default=7)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--mask-remap", choices=["auto", "fine_to_coarse", "coarse", "ignore_invalid"], default="auto")
    parser.add_argument("--balanced-datasets", action="store_true")
    parser.add_argument("--dataset-sampling-temperature", type=float, default=0.5)
    parser.add_argument("--rare-class-sampling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rare-class-ids", type=int, nargs="+", default=[3, 4, 6])
    parser.add_argument("--rare-class-sample-boost", type=float, default=2.0)
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--train-split", type=int, default=1000)
    parser.add_argument("--val-split", type=int, default=200)
    parser.add_argument("--manifest", type=Path, default=None, help="Optional fixed split manifest JSON.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--lr-scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument(
        "--backbone-unfreeze-epoch",
        type=int,
        default=-1,
        help="One-based epoch that starts partial backbone training; -1 disables unfreezing.",
    )
    parser.add_argument("--backbone-unfreeze-blocks", type=int, default=0)
    parser.add_argument("--min-free-gpu-memory-gb-before-unfreeze", type=float, default=0.0)
    parser.add_argument("--gpu-memory-poll-seconds", type=float, default=60.0)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--checkpoint-boundary-weight", type=float, default=0.25)
    parser.add_argument("--checkpoint-fine-weight", type=float, default=0.25)
    parser.add_argument("--decoder", choices=["upernet", "mask2former"], default="upernet")
    parser.add_argument("--mask2former-queries", type=int, default=100)
    parser.add_argument("--mask2former-ignore-index", type=int, default=255)
    parser.add_argument("--symmetric-padding", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--boundary-refinement", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--refinement-loss-weight", type=float, default=1.0)
    parser.add_argument("--refinement-boundary-weight", type=float, default=0.5)
    parser.add_argument("--refinement-boundary-widths", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--refinement-boundary-ce-weight", type=float, default=0.0)
    parser.add_argument("--refinement-consistency-weight", type=float, default=0.0)
    parser.add_argument("--refinement-gate-width", type=int, default=4)
    parser.add_argument("--refinement-gate-threshold", type=float, default=0.15)
    parser.add_argument("--refinement-gate-mode", choices=["hard", "learned_soft"], default="hard")
    parser.add_argument("--refinement-gate-loss-weight", type=float, default=0.0)
    parser.add_argument("--refinement-gate-target-width", type=int, default=8)
    parser.add_argument("--boundary-aware-sampling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--boundary-sampling-boost", type=float, default=3.0)
    parser.add_argument("--boundary-sampling-min-pixels", type=int, default=512)
    parser.add_argument("--boundary-sampling-width", type=int, default=4)
    parser.add_argument("--boundary-sampling-mode", choices=["threshold", "dataset_quantile"], default="threshold")
    parser.add_argument("--joint-sampling-fine-fraction", type=float, default=0.6)
    parser.add_argument("--cell-input-fine-fraction", type=float, default=0.6)
    parser.add_argument("--cellvit-mode", choices=["none", "teacher", "input"], default="none")
    parser.add_argument("--cell-density-sigma", type=float, default=8.0)
    parser.add_argument("--cell-prior-dropout", type=float, default=0.2)
    parser.add_argument("--cell-aux-loss-weight", type=float, default=0.2)
    parser.add_argument("--hierarchical-fine", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fine-loss-weight", type=float, default=1.0)
    parser.add_argument("--class-weighting", choices=["none", "inverse_sqrt"], default="none")
    parser.add_argument("--fine-class-weighting", choices=["none", "inverse_sqrt"], default=None)
    parser.add_argument("--fine-class-weight-min", type=float, default=0.5)
    parser.add_argument("--fine-class-weight-max", type=float, default=4.0)
    parser.add_argument(
        "--fine-supervision-sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sample only patches containing valid dataset-specific fine supervision.",
    )
    parser.add_argument("--fine-sampling-rare-class-boost", type=float, default=4.0)
    parser.add_argument("--fine-sampling-min-valid-pixels", type=int, default=1)
    parser.add_argument("--fine-sampling-require-nuclei", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--freeze-shared-for-fine", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fine-only-loss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--trainable-scope",
        choices=["all", "fine", "boundary", "teacher", "input", "boundary_teacher"],
        default="all",
    )
    parser.add_argument("--refinement-only-loss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--label-space-summary", type=Path, default=None, help="Reuse a prior config.json or label-space summary to avoid rescanning all masks.")
    parser.add_argument("--stain-augmentation", choices=["none", "randstainna"], default="none")
    parser.add_argument("--stain-augmentation-prob", type=float, default=0.0)
    parser.add_argument("--augment-vflip", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--augment-rot90", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--augment-scale-crop", type=float, default=0.0)
    parser.add_argument("--randstainna-root", type=Path, default=Path("third_party/RandStainNA"))
    parser.add_argument("--randstainna-yaml", type=Path, default=None)
    parser.add_argument("--randstainna-std-hyper", type=float, default=-0.3)
    parser.add_argument("--randstainna-distribution", choices=["normal", "laplace", "uniform"], default="normal")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--export-val-predictions", action="store_true")
    parser.add_argument("--export-val-tensors", action="store_true")
    parser.add_argument("--boundary-width", type=int, default=2)
    parser.add_argument("--metric-sample-limit", type=int, default=2048)
    parser.add_argument(
        "--ddp-timeout-seconds",
        type=float,
        default=600.0,
        help="Timeout for distributed collectives, including ranks waiting during rank-zero validation.",
    )
    parser.add_argument(
        "--rank-zero-validation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the full validation set on rank 0 without the DDP wrapper to avoid validation collectives.",
    )
    parser.add_argument(
        "--checkpoint-mode",
        choices=["composite", "fine_dataset_macro", "boundary_f1_4", "joint"],
        default="composite",
    )
    parser.add_argument("--checkpoint-coarse-miou-floor", type=float, default=None)
    parser.add_argument("--checkpoint-coarse-boundary-f1-4-floor", type=float, default=None)
    parser.add_argument("--checkpoint-fine-dataset-macro-floor", type=float, default=None)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Resume from 'latest' or a segmentator training checkpoint path.")
    parser.add_argument("--init-from-checkpoint", type=str, default=None, help="Initialize model weights only, for example from a coarse Segmentator checkpoint.")
    parser.add_argument(
        "--init-refinement-from-checkpoint",
        type=str,
        default=None,
        help="Overlay refinement_head weights from a Boundary checkpoint after the primary initialization.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fine_loss_weight < 0 or args.checkpoint_fine_weight < 0:
        raise SystemExit("fine loss and checkpoint weights must be non-negative")
    if args.fine_class_weight_min <= 0 or args.fine_class_weight_max < args.fine_class_weight_min:
        raise SystemExit("fine class weight bounds must satisfy 0 < min <= max")
    if args.fine_sampling_rare_class_boost < 1:
        raise SystemExit("--fine-sampling-rare-class-boost must be at least 1")
    if args.fine_sampling_min_valid_pixels < 1:
        raise SystemExit("--fine-sampling-min-valid-pixels must be at least 1")
    if args.freeze_shared_for_fine and args.trainable_scope not in {"all", "fine"}:
        raise SystemExit("--freeze-shared-for-fine conflicts with --trainable-scope")
    if args.trainable_scope == "boundary" and not args.boundary_refinement:
        raise SystemExit("--trainable-scope boundary requires --boundary-refinement")
    if args.trainable_scope == "teacher" and args.cellvit_mode != "teacher":
        raise SystemExit("--trainable-scope teacher requires --cellvit-mode teacher")
    if args.trainable_scope == "input" and args.cellvit_mode != "input":
        raise SystemExit("--trainable-scope input requires --cellvit-mode input")
    if args.trainable_scope == "boundary_teacher" and (
        not args.boundary_refinement or args.cellvit_mode != "teacher"
    ):
        raise SystemExit("--trainable-scope boundary_teacher requires --boundary-refinement and --cellvit-mode teacher")
    if args.refinement_only_loss and not args.boundary_refinement:
        raise SystemExit("--refinement-only-loss requires --boundary-refinement")
    if args.boundary_aware_sampling and not args.boundary_refinement:
        raise SystemExit("--boundary-aware-sampling requires --boundary-refinement")
    if (
        args.boundary_aware_sampling
        and args.fine_supervision_sampling
        and args.trainable_scope != "boundary_teacher"
    ):
        raise SystemExit("--boundary-aware-sampling and --fine-supervision-sampling are mutually exclusive")
    if any(width < 1 for width in args.refinement_boundary_widths):
        raise SystemExit("--refinement-boundary-widths must contain positive integers")
    if (
        args.refinement_gate_width < 1
        or args.refinement_gate_target_width < 1
        or args.boundary_sampling_width < 1
    ):
        raise SystemExit("boundary gate and sampling widths must be positive")
    if not 0.0 <= args.refinement_gate_threshold < 1.0:
        raise SystemExit("--refinement-gate-threshold must be in [0, 1)")
    if args.boundary_sampling_boost < 1.0 or args.boundary_sampling_min_pixels < 1:
        raise SystemExit("boundary sampling requires boost >= 1 and min pixels >= 1")
    if min(
        args.refinement_loss_weight,
        args.refinement_boundary_weight,
        args.refinement_boundary_ce_weight,
        args.refinement_consistency_weight,
        args.refinement_gate_loss_weight,
    ) < 0:
        raise SystemExit("refinement loss weights must be non-negative")
    if not 0.0 <= args.joint_sampling_fine_fraction <= 1.0:
        raise SystemExit("--joint-sampling-fine-fraction must be in [0, 1]")
    if not 0.0 <= args.cell_input_fine_fraction <= 1.0:
        raise SystemExit("--cell-input-fine-fraction must be in [0, 1]")
    if args.fine_only_loss and args.refinement_only_loss:
        raise SystemExit("--fine-only-loss and --refinement-only-loss are mutually exclusive")
    if args.fine_sampling_require_nuclei and args.cellvit_mode == "none":
        raise SystemExit("--fine-sampling-require-nuclei requires a CellViT mode")
    if (
        args.freeze_shared_for_fine
        or args.fine_only_loss
        or args.fine_supervision_sampling
        or args.trainable_scope in {"fine", "teacher", "input", "boundary_teacher"}
        or args.checkpoint_mode == "fine_dataset_macro"
        or args.checkpoint_fine_dataset_macro_floor is not None
    ) and not args.hierarchical_fine:
        raise SystemExit("fine-only training and checkpoint options require --hierarchical-fine")
    if args.ddp_timeout_seconds <= 0:
        raise SystemExit("--ddp-timeout-seconds must be positive")
    if args.resume_from_checkpoint and args.init_from_checkpoint:
        raise SystemExit("use only one of --resume-from-checkpoint and --init-from-checkpoint")
    if args.resume_from_checkpoint and args.init_refinement_from_checkpoint:
        raise SystemExit("--init-refinement-from-checkpoint cannot be combined with --resume-from-checkpoint")
    config = BaselineConfig(
        image_size=args.image_size,
        remap_invalid_to=args.remap_invalid_to,
        ignore_index=args.ignore_index,
        mask_remap=args.mask_remap,
        balanced_datasets=args.balanced_datasets,
        dataset_sampling_temperature=args.dataset_sampling_temperature,
        rare_class_sampling=args.rare_class_sampling,
        rare_class_ids=tuple(args.rare_class_ids),
        rare_class_sample_boost=args.rare_class_sample_boost,
        samples_per_epoch=args.samples_per_epoch,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_workers=args.num_workers,
        epochs=args.epochs,
        seed=args.seed,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        lr_scheduler=args.lr_scheduler,
        backbone_unfreeze_epoch=args.backbone_unfreeze_epoch,
        backbone_unfreeze_blocks=args.backbone_unfreeze_blocks,
        min_free_gpu_memory_gb_before_unfreeze=args.min_free_gpu_memory_gb_before_unfreeze,
        gpu_memory_poll_seconds=args.gpu_memory_poll_seconds,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        checkpoint_boundary_weight=args.checkpoint_boundary_weight,
        checkpoint_fine_weight=args.checkpoint_fine_weight,
        train_split=args.train_split,
        val_split=args.val_split,
        manifest_path=args.manifest,
        decoder=args.decoder,
        mask2former_queries=args.mask2former_queries,
        mask2former_ignore_index=args.mask2former_ignore_index,
        symmetric_padding=args.symmetric_padding,
        boundary_refinement=args.boundary_refinement,
        refinement_loss_weight=args.refinement_loss_weight,
        refinement_boundary_weight=args.refinement_boundary_weight,
        refinement_boundary_widths=tuple(args.refinement_boundary_widths),
        refinement_boundary_ce_weight=args.refinement_boundary_ce_weight,
        refinement_consistency_weight=args.refinement_consistency_weight,
        refinement_gate_width=args.refinement_gate_width,
        refinement_gate_threshold=args.refinement_gate_threshold,
        refinement_gate_mode=args.refinement_gate_mode,
        refinement_gate_loss_weight=args.refinement_gate_loss_weight,
        refinement_gate_target_width=args.refinement_gate_target_width,
        boundary_aware_sampling=args.boundary_aware_sampling,
        boundary_sampling_boost=args.boundary_sampling_boost,
        boundary_sampling_min_pixels=args.boundary_sampling_min_pixels,
        boundary_sampling_width=args.boundary_sampling_width,
        boundary_sampling_mode=args.boundary_sampling_mode,
        joint_sampling_fine_fraction=args.joint_sampling_fine_fraction,
        cell_input_fine_fraction=args.cell_input_fine_fraction,
        cellvit_mode=args.cellvit_mode,
        cell_density_sigma=args.cell_density_sigma,
        cell_prior_dropout=args.cell_prior_dropout,
        cell_aux_loss_weight=args.cell_aux_loss_weight,
        hierarchical_fine=args.hierarchical_fine,
        fine_loss_weight=args.fine_loss_weight,
        fine_class_weighting=args.fine_class_weighting,
        fine_class_weight_min=args.fine_class_weight_min,
        fine_class_weight_max=args.fine_class_weight_max,
        fine_supervision_sampling=args.fine_supervision_sampling,
        fine_sampling_rare_class_boost=args.fine_sampling_rare_class_boost,
        fine_sampling_min_valid_pixels=args.fine_sampling_min_valid_pixels,
        fine_sampling_require_nuclei=args.fine_sampling_require_nuclei,
        freeze_shared_for_fine=args.freeze_shared_for_fine,
        fine_only_loss=args.fine_only_loss,
        trainable_scope=args.trainable_scope,
        refinement_only_loss=args.refinement_only_loss,
        amp=args.amp,
        disable_cudnn=args.disable_cudnn,
        class_weighting=args.class_weighting,
        label_space_summary_path=args.label_space_summary,
        stain_augmentation=args.stain_augmentation,
        stain_augmentation_prob=args.stain_augmentation_prob,
        augment_vflip=args.augment_vflip,
        augment_rot90=args.augment_rot90,
        augment_scale_crop=args.augment_scale_crop,
        randstainna_root=args.randstainna_root,
        randstainna_yaml=args.randstainna_yaml,
        randstainna_std_hyper=args.randstainna_std_hyper,
        randstainna_distribution=args.randstainna_distribution,
        export_val_predictions=args.export_val_predictions or args.export_val_tensors,
        export_val_tensors=args.export_val_tensors,
        boundary_width=args.boundary_width,
        metric_sample_limit=args.metric_sample_limit,
        ddp_timeout_seconds=args.ddp_timeout_seconds,
        rank_zero_validation=args.rank_zero_validation,
        checkpoint_mode=args.checkpoint_mode,
        checkpoint_coarse_miou_floor=args.checkpoint_coarse_miou_floor,
        checkpoint_coarse_boundary_f1_4_floor=args.checkpoint_coarse_boundary_f1_4_floor,
        checkpoint_fine_dataset_macro_floor=args.checkpoint_fine_dataset_macro_floor,
        resume_from_checkpoint=args.resume_from_checkpoint,
        init_from_checkpoint=args.init_from_checkpoint,
        init_refinement_from_checkpoint=args.init_refinement_from_checkpoint,
        output_dir=Path(args.output_dir),
    )
    metrics = run_stage4_baseline(args.dataset_root, config, uni2h_repo=args.uni2h_repo)
    print(metrics)
    return 0


def main_predict(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 4 tissue segmentation inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--decoder", choices=["upernet", "mask2former"], default="upernet")
    parser.add_argument("--mask2former-queries", type=int, default=100)
    parser.add_argument("--mask2former-ignore-index", type=int, default=255)
    parser.add_argument("--symmetric-padding", action="store_true")
    parser.add_argument("--boundary-refinement", action="store_true")
    parser.add_argument("--refinement-gate-mode", choices=["hard", "learned_soft"], default="hard")
    parser.add_argument("--cellvit-mode", choices=["none", "teacher", "input"], default="none")
    parser.add_argument("--hierarchical-fine", action="store_true")
    parser.add_argument("--dataset-id", default=None, help="Dataset/organ profile used to constrain hierarchical fine outputs.")
    parser.add_argument("--nuclei-mask", default=None)
    parser.add_argument("--cell-density-sigma", type=float, default=8.0)
    args = parser.parse_args(argv)

    model = load_checkpoint(
        args.checkpoint,
        num_classes=args.num_classes,
        decoder=args.decoder,
        mask2former_queries=args.mask2former_queries,
        mask2former_ignore_index=args.mask2former_ignore_index,
        symmetric_padding=args.symmetric_padding,
        boundary_refinement=args.boundary_refinement,
        refinement_gate_mode=args.refinement_gate_mode,
        cellvit_mode=args.cellvit_mode,
        hierarchical_fine=args.hierarchical_fine,
    )
    image = normalize_image_tensor(TF.to_tensor(Image.open(args.input).convert("RGB")))
    nuclei_density = None
    if args.nuclei_mask:
        nuclei_density = nuclei_mask_to_density(load_mask(Path(args.nuclei_mask)), sigma=args.cell_density_sigma)
    fine_allowed = None
    if args.hierarchical_fine and args.dataset_id:
        from .data import fine_supervision_for_dataset

        fine_allowed = fine_supervision_for_dataset(args.dataset_id)
    outputs = model(
        image.unsqueeze(0) if image.ndim == 3 else image,
        nuclei_density=nuclei_density,
        fine_allowed=fine_allowed,
    )
    prediction = outputs["hierarchical_pred"] if args.hierarchical_fine else outputs["pred"]
    save_prediction(prediction[0], args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

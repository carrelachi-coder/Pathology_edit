from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms.functional as TF

from .config import BaselineConfig
from .data import normalize_image_tensor
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
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--remap-invalid-to", type=int, default=7)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--mask-remap", choices=["auto", "fine_to_coarse", "coarse", "ignore_invalid"], default="auto")
    parser.add_argument("--balanced-datasets", action="store_true")
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--train-split", type=int, default=1000)
    parser.add_argument("--val-split", type=int, default=200)
    parser.add_argument("--manifest", type=Path, default=None, help="Optional fixed split manifest JSON.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decoder", choices=["upernet", "mask2former"], default="upernet")
    parser.add_argument("--mask2former-queries", type=int, default=100)
    parser.add_argument("--mask2former-ignore-index", type=int, default=255)
    parser.add_argument("--class-weighting", choices=["none", "inverse_sqrt"], default="none")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--export-val-predictions", action="store_true")
    parser.add_argument("--export-val-tensors", action="store_true")
    parser.add_argument("--boundary-width", type=int, default=2)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Resume from 'latest' or a segmentator training checkpoint path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BaselineConfig(
        image_size=args.image_size,
        remap_invalid_to=args.remap_invalid_to,
        ignore_index=args.ignore_index,
        mask_remap=args.mask_remap,
        balanced_datasets=args.balanced_datasets,
        samples_per_epoch=args.samples_per_epoch,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        epochs=args.epochs,
        seed=args.seed,
        train_split=args.train_split,
        val_split=args.val_split,
        manifest_path=args.manifest,
        decoder=args.decoder,
        mask2former_queries=args.mask2former_queries,
        mask2former_ignore_index=args.mask2former_ignore_index,
        amp=args.amp,
        disable_cudnn=args.disable_cudnn,
        class_weighting=args.class_weighting,
        export_val_predictions=args.export_val_predictions or args.export_val_tensors,
        export_val_tensors=args.export_val_tensors,
        boundary_width=args.boundary_width,
        resume_from_checkpoint=args.resume_from_checkpoint,
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
    args = parser.parse_args(argv)

    model = load_checkpoint(
        args.checkpoint,
        num_classes=args.num_classes,
        decoder=args.decoder,
        mask2former_queries=args.mask2former_queries,
        mask2former_ignore_index=args.mask2former_ignore_index,
    )
    image = normalize_image_tensor(TF.to_tensor(Image.open(args.input).convert("RGB")))
    outputs = model(image.unsqueeze(0) if image.ndim == 3 else image)
    save_prediction(outputs["pred"][0], args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

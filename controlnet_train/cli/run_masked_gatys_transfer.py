"""Run masked Gatys texture/stain transfer on an existing generated image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controlnet_train.postprocess import GatysTransferConfig, parse_region_labels, run_masked_gatys_transfer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Masked Gatys texture/stain transfer postprocess.")
    parser.add_argument("--initial-image", required=True, help="ControlNet generated structure image I0.")
    parser.add_argument("--target-mask", required=True, help="Target tissue label mask aligned to I0.")
    parser.add_argument("--target-nuclei-mask", default=None, help="Optional target nuclei mask; nuclei labels overwrite tissue labels.")
    parser.add_argument("--reference-image", required=True, help="Reference image providing texture/stain.")
    parser.add_argument("--reference-mask", required=True, help="Reference tissue label mask.")
    parser.add_argument("--reference-nuclei-mask", default=None, help="Optional reference nuclei mask; nuclei labels overwrite tissue labels.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--regions", default=None, help="Comma-separated tissue labels to transfer. Default: shared nonzero labels.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--optimizer", choices=["lbfgs", "adam"], default="lbfgs")
    parser.add_argument("--lr", type=float, default=1.0, help="LBFGS learning rate.")
    parser.add_argument("--adam-lr", type=float, default=0.02)
    parser.add_argument("--content-weight", type=float, default=1.0)
    parser.add_argument("--no-content-loss", action="store_true", help="Disable the VGG content anchor and run style/TV only.")
    parser.add_argument("--style-weight", type=float, default=1e4)
    parser.add_argument("--tv-weight", type=float, default=0.0)
    parser.add_argument("--style-layers", default="conv1_1,conv2_1,conv3_1,conv4_1,conv5_1")
    parser.add_argument("--layer-weights", default="conv1_1=1.0,conv2_1=1.0,conv3_1=0.5,conv4_1=0.25,conv5_1=0.0")
    parser.add_argument("--content-layer", default="conv4_2")
    parser.add_argument("--min-region-pixels", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--no-preserve-background", action="store_true")
    parser.add_argument("--background-label", type=int, default=0)
    parser.add_argument("--optimize-background", action="store_true", help="Allow label 0/background pixels to be optimized too.")
    parser.add_argument(
        "--missing-region-fallback",
        choices=["pooled", "skip"],
        default="pooled",
        help="How to style target labels missing from the reference mask.",
    )
    parser.add_argument("--no-save-mask-debug", action="store_true")
    parser.add_argument(
        "--pre-gatys-color-match",
        choices=["none", "lab", "macenko"],
        default="none",
        help="Apply stain/color match to I0 before masked Gatys.",
    )
    parser.add_argument(
        "--color-match-scope",
        choices=["region", "global"],
        default="region",
        help="For Macenko, match each composite mask label separately or match globally.",
    )
    parser.add_argument("--color-match-strength", type=float, default=1.0)
    parser.add_argument("--macenko-io", type=float, default=240.0)
    parser.add_argument("--macenko-beta", type=float, default=0.15)
    parser.add_argument("--macenko-alpha", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--vgg-weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--vgg-weights-path", default=None)
    return parser


def parse_args(args=None) -> argparse.Namespace:
    return build_parser().parse_args(args)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = GatysTransferConfig(
        steps=args.steps,
        optimizer=args.optimizer,
        lr=args.lr,
        adam_lr=args.adam_lr,
        content_weight=args.content_weight,
        use_content_loss=not args.no_content_loss,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight,
        style_layers=_parse_csv(args.style_layers),
        content_layer=args.content_layer,
        layer_weights=_parse_layer_weights(args.layer_weights),
        min_region_pixels=args.min_region_pixels,
        log_every=args.log_every,
        save_every=args.save_every,
        preserve_background=not args.no_preserve_background,
        background_label=args.background_label,
        optimize_background=args.optimize_background,
        missing_region_fallback=args.missing_region_fallback,
        save_mask_debug=not args.no_save_mask_debug,
        pre_color_match=args.pre_gatys_color_match,
        color_match_scope=args.color_match_scope,
        color_match_strength=args.color_match_strength,
        macenko_io=args.macenko_io,
        macenko_beta=args.macenko_beta,
        macenko_alpha=args.macenko_alpha,
        device=args.device,
        torch_dtype=args.torch_dtype,
        vgg_weights=args.vgg_weights,
        vgg_weights_path=args.vgg_weights_path,
    )
    result = run_masked_gatys_transfer(
        initial_image_path=args.initial_image,
        target_mask_path=args.target_mask,
        target_nuclei_mask_path=args.target_nuclei_mask,
        reference_image_path=args.reference_image,
        reference_mask_path=args.reference_mask,
        reference_nuclei_mask_path=args.reference_nuclei_mask,
        output_dir=args.output_dir,
        regions=parse_region_labels(args.regions),
        config=config,
    )
    print(f"wrote {result.output_path}")
    if result.metrics_path is not None:
        print(f"wrote {result.metrics_path}")
    return 0


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _parse_layer_weights(value: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        name, raw_weight = item.split("=", 1)
        weights[name.strip()] = float(raw_weight)
    return weights


if __name__ == "__main__":
    raise SystemExit(main())

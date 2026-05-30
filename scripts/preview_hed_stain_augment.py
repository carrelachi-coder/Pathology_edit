"""Preview HED stain augmentation on one real H&E image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controlnet_train.data.hed_stain_augment import HEDStainAugment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render HED stain augmentation previews.")
    parser.add_argument("--image", required=True, help="Path to one RGB H&E patch image.")
    parser.add_argument("--output-dir", default="hed_aug_preview", help="Directory for preview PNGs.")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument(
        "--strong-alpha-sampling",
        action="store_true",
        help="Sample alpha away from 1 using two ranges instead of uniform 1±sigma.",
    )
    parser.add_argument("--alpha-min", type=float, default=0.4)
    parser.add_argument("--alpha-low", type=float, default=0.75)
    parser.add_argument("--alpha-high", type=float, default=1.25)
    parser.add_argument("--alpha-max", type=float, default=1.8)
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help=(
            "Optional comma-separated sigma values. When set, renders previews "
            "for every sigma, for example: --sweep 0.2,0.3,0.4,0.5."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255.0

    print(f"input={image_path} shape={tuple(image_tensor.shape)}")

    original_out = output_dir / "original.png"
    image.save(original_out)
    print(f"saved {original_out}")

    sigmas = [args.sigma]
    if args.sweep:
        sigmas = [float(part.strip()) for part in args.sweep.split(",") if part.strip()]

    for sigma in sigmas:
        augment = HEDStainAugment(
            sigma=sigma,
            beta=args.beta,
            strong_alpha_sampling=args.strong_alpha_sampling,
            alpha_min=args.alpha_min,
            alpha_low=args.alpha_low,
            alpha_high=args.alpha_high,
            alpha_max=args.alpha_max,
        )
        print(
            f"sigma={sigma:.3f} beta={args.beta:.3f} "
            f"strong_alpha={args.strong_alpha_sampling} "
            f"alpha_ranges=[{args.alpha_min:.3f},{args.alpha_low:.3f}] "
            f"U [{args.alpha_high:.3f},{args.alpha_max:.3f}] "
            f"condition_number={augment.condition_number:.3f}"
        )
        prefix_base = f"sigma_{sigma:.2f}".replace(".", "p") if len(sigmas) > 1 else "aug"
        prefix = f"strong_{prefix_base}" if args.strong_alpha_sampling else prefix_base
        for index in range(args.num_samples):
            params = augment.sample()
            output = augment(image_tensor, params)
            array = (output.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
            output_path = output_dir / f"{prefix}_{index:02d}.png"
            Image.fromarray(array).save(output_path)
            print(
                f"saved {output_path} "
                f"alpha_h={float(params.alpha[0]):.4f} alpha_e={float(params.alpha[1]):.4f} "
                f"beta_h={float(params.beta[0]):.4f} beta_e={float(params.beta[1]):.4f} "
                f"range=[{float(output.min()):.3f},{float(output.max()):.3f}]"
            )


if __name__ == "__main__":
    main()

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

    augment = HEDStainAugment(sigma=args.sigma, beta=args.beta)
    print(f"condition_number={augment.condition_number:.3f}")
    print(f"input={image_path} shape={tuple(image_tensor.shape)}")

    original_out = output_dir / "original.png"
    image.save(original_out)
    print(f"saved {original_out}")

    for index in range(args.num_samples):
        params = augment.sample()
        output = augment(image_tensor, params)
        array = (output.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        output_path = output_dir / f"aug_{index:02d}.png"
        Image.fromarray(array).save(output_path)
        print(
            f"saved {output_path} "
            f"alpha_h={float(params.alpha[0]):.4f} alpha_e={float(params.alpha[1]):.4f} "
            f"beta_h={float(params.beta[0]):.4f} beta_e={float(params.beta[1]):.4f} "
            f"range=[{float(output.min()):.3f},{float(output.max()):.3f}]"
        )


if __name__ == "__main__":
    main()

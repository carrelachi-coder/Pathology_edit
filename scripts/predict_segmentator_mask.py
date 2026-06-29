#!/usr/bin/env python3
"""Run Stage 4 segmentator inference for one image."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--decoder", choices=("upernet", "mask2former"), default="mask2former")
    parser.add_argument("--mask2former-queries", type=int, default=100)
    parser.add_argument("--mask2former-ignore-index", type=int, default=255)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import torch
    from PIL import Image
    import torchvision.transforms.functional as TF

    from segmentator.data import normalize_image_tensor
    from segmentator.inference import load_checkpoint, save_prediction

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_checkpoint(
        args.checkpoint,
        num_classes=args.num_classes,
        decoder=args.decoder,
        mask2former_queries=args.mask2former_queries,
        mask2former_ignore_index=args.mask2former_ignore_index,
    ).to(device)
    image = normalize_image_tensor(TF.to_tensor(Image.open(args.input).convert("RGB"))).to(device)
    with torch.inference_mode():
        outputs = model(image.unsqueeze(0))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_prediction(outputs["pred"][0], output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

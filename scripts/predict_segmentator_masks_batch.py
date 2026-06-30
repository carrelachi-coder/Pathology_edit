#!/usr/bin/env python3
"""Run Stage 4 segmentator inference for a directory of images."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--decoder", choices=("upernet", "mask2former"), default="mask2former")
    parser.add_argument("--mask2former-queries", type=int, default=100)
    parser.add_argument("--mask2former-ignore-index", type=int, default=255)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def iter_images(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def predict_batch(model, image_paths: list[Path], output_dir: Path, device) -> None:
    import torch
    from PIL import Image, PngImagePlugin
    import torchvision.transforms.functional as TF

    from segmentator.data import normalize_image_tensor
    from segmentator.inference import save_prediction

    PngImagePlugin.MAX_TEXT_CHUNK = max(PngImagePlugin.MAX_TEXT_CHUNK, 256 * 1024 * 1024)
    PngImagePlugin.MAX_TEXT_MEMORY = max(PngImagePlugin.MAX_TEXT_MEMORY, 1024 * 1024 * 1024)
    tensors = [normalize_image_tensor(TF.to_tensor(Image.open(path).convert("RGB"))) for path in image_paths]
    shapes = {tuple(tensor.shape) for tensor in tensors}
    if len(shapes) != 1:
        for image_path, tensor in zip(image_paths, tensors):
            with torch.inference_mode():
                outputs = model(tensor.to(device).unsqueeze(0))
            save_prediction(outputs["pred"][0], output_dir / f"{image_path.stem}.png")
        return

    images = torch.stack(tensors, dim=0).to(device)
    with torch.inference_mode():
        outputs = model(images)
    for image_path, prediction in zip(image_paths, outputs["pred"]):
        save_prediction(prediction, output_dir / f"{image_path.stem}.png")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import torch
    from segmentator.inference import load_checkpoint

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)

    image_paths = iter_images(input_dir)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise RuntimeError(f"no images found in {input_dir}")

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_checkpoint(
        args.checkpoint,
        num_classes=args.num_classes,
        decoder=args.decoder,
        mask2former_queries=args.mask2former_queries,
        mask2former_ignore_index=args.mask2former_ignore_index,
    ).to(device)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    failed: list[dict[str, str]] = []
    batch: list[Path] = []
    for index, image_path in enumerate(image_paths, start=1):
        if args.skip_existing and (output_dir / f"{image_path.stem}.png").exists():
            skipped += 1
            continue
        batch.append(image_path)
        if len(batch) < args.batch_size and index != len(image_paths):
            continue
        try:
            predict_batch(model, batch, output_dir, device)
            written += len(batch)
            if index == 1 or index % 25 == 0 or index == len(image_paths):
                print(f"[{index}/{len(image_paths)}] wrote {written} masks", flush=True)
        except Exception as exc:  # noqa: BLE001 - keep batch jobs moving and report failures.
            print(f"[{index}/{len(image_paths)}] failed batch ending at {image_path}: {exc!r}; retrying singly", file=sys.stderr, flush=True)
            for single_path in batch:
                try:
                    predict_batch(model, [single_path], output_dir, device)
                    written += 1
                except Exception as single_exc:  # noqa: BLE001 - keep batch jobs moving and report failures.
                    failed.append({"images": [str(single_path)], "error": repr(single_exc)})
                    print(f"[{index}/{len(image_paths)}] failed {single_path}: {single_exc!r}", file=sys.stderr, flush=True)
        finally:
            batch = []

    summary = {
        "checkpoint": str(Path(args.checkpoint)),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "requested": len(image_paths),
        "written": written,
        "skipped": skipped,
        "failed": failed,
    }
    (output_dir / "prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

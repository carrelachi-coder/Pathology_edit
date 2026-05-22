from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms.functional as TF

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segmentator.data import load_manifest, normalize_image_tensor
from segmentator.metrics import _boundary
from segmentator.model import BaselineSegmenter


PALETTE = np.array(
    [
        [30, 30, 30],
        [180, 60, 60],
        [60, 150, 60],
        [60, 60, 180],
        [140, 60, 180],
        [180, 180, 80],
        [60, 140, 100],
        [170, 170, 170],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize GT/pred segmentator boundaries on val samples.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--uni2h-repo", default="UNI-2h")
    parser.add_argument("--output-dir", default="segmentator_runs/boundary_viz")
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--decoder", choices=["upernet", "mask2former"], default="upernet")
    parser.add_argument("--mask2former-queries", type=int, default=100)
    parser.add_argument("--boundary-width", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-cudnn", action="store_true")
    return parser.parse_args()


def colorize(mask: np.ndarray) -> Image.Image:
    mask = np.clip(mask, 0, len(PALETTE) - 1)
    return Image.fromarray(PALETTE[mask], mode="RGB")


def overlay_boundaries(image: Image.Image, gt: np.ndarray, pred: np.ndarray, width: int) -> Image.Image:
    canvas = image.convert("RGB").copy()
    arr = np.array(canvas).astype(np.float32)
    gt_b = _boundary(torch.from_numpy(gt[None, ...]), width=width)[0].numpy()
    pred_b = _boundary(torch.from_numpy(pred[None, ...]), width=width)[0].numpy()
    hit = gt_b & pred_b
    gt_only = gt_b & ~pred_b
    pred_only = pred_b & ~gt_b
    arr[gt_only] = arr[gt_only] * 0.25 + np.array([0, 255, 255]) * 0.75
    arr[pred_only] = arr[pred_only] * 0.25 + np.array([255, 0, 0]) * 0.75
    arr[hit] = arr[hit] * 0.25 + np.array([255, 255, 0]) * 0.75
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def panel(sample_id: str, image: Image.Image, gt: np.ndarray, pred: np.ndarray, width: int) -> Image.Image:
    gt_img = colorize(gt)
    pred_img = colorize(pred)
    boundary_img = overlay_boundaries(image, gt, pred, width)
    w, h = image.size
    label_h = 24
    out = Image.new("RGB", (w * 4, h + label_h), "white")
    labels = ["image", "gt", "pred", "boundary cyan=GT red=Pred yellow=Hit"]
    for idx, tile in enumerate([image.convert("RGB"), gt_img, pred_img, boundary_img]):
        out.paste(tile, (idx * w, label_h))
    draw = ImageDraw.Draw(out)
    draw.text((8, 4), sample_id, fill=(0, 0, 0))
    for idx, label in enumerate(labels):
        draw.text((idx * w + 8, 4), label, fill=(0, 0, 0))
    return out


def main() -> int:
    args = parse_args()
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False

    manifest = load_manifest(Path(args.manifest), root=Path(args.dataset_root))
    records = list(manifest.val)[: args.num_samples]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineSegmenter(
        num_classes=args.num_classes,
        freeze_encoder=True,
        local_repo=args.uni2h_repo,
        decoder=args.decoder,
        mask2former_queries=args.mask2former_queries,
    ).to(device)
    state = torch.load(Path(args.checkpoint), map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for record in records:
            image = Image.open(record.image_path).convert("RGB")
            mask = Image.open(record.mask_path).convert("L")
            image_r = TF.resize(image, [args.image_size, args.image_size])
            mask_r = TF.resize(mask, [args.image_size, args.image_size], interpolation=TF.InterpolationMode.NEAREST)
            image_t = normalize_image_tensor(TF.to_tensor(image_r)).unsqueeze(0).to(device)
            pred = model(image_t)["pred"][0].cpu().numpy().astype(np.uint8)
            gt = np.array(mask_r, dtype=np.uint8)
            panel(record.sample_id, image_r, gt, pred, args.boundary_width).save(out_dir / f"{record.sample_id}_boundary.png")
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

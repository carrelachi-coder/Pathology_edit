from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segmentator.config import DatasetManifest
from segmentator.data import TissueSegmentationDataset, build_manifest, load_manifest
from segmentator.metrics import confusion_matrix, segmentation_metrics
from segmentator.model import BaselineSegmenter


DEFAULT_LABELS = (
    "background",
    "tumor",
    "stroma",
    "necrosis",
    "immune_infiltrate",
    "normal_epithelium",
    "blood_vessel",
    "other_tissue",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Stage 4 segmentator checkpoint per class.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", type=Path, default=None, help="Optional fixed split manifest JSON.")
    parser.add_argument("--uni2h-repo", default="UNI-2h")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--decoder", choices=["upernet", "mask2former"], default="upernet")
    parser.add_argument("--mask2former-queries", type=int, default=100)
    parser.add_argument("--remap-invalid-to", type=int, default=7)
    parser.add_argument("--boundary-width", type=int, default=2)
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--output-json")
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = (
        load_manifest(args.manifest, root=Path(args.dataset_root))
        if args.manifest is not None
        else build_manifest(Path(args.dataset_root), args.train_count, args.val_count, seed=args.seed)
    )
    val_ds = TissueSegmentationDataset(
        list(manifest.val),
        image_size=args.image_size,
        augment=False,
        num_classes=args.num_classes,
        remap_invalid_to=args.remap_invalid_to,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

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

    mat = torch.zeros(args.num_classes, args.num_classes, dtype=torch.long)
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="eval", dynamic_ncols=True):
            image = batch["image"].to(device)
            target = batch["mask"]
            pred = model(image)["pred"].cpu()
            mat += confusion_matrix(pred, target, args.num_classes)
            preds.append(pred)
            targets.append(target)

    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)
    class_names = _class_names(manifest, args.num_classes)
    metrics = segmentation_metrics(
        pred_all,
        target_all,
        args.num_classes,
        class_names=class_names,
        boundary_width=args.boundary_width,
    )
    return {
        "device": str(device),
        "checkpoint": str(Path(args.checkpoint)),
        "dataset_root": str(Path(args.dataset_root)),
        "image_size": args.image_size,
        "train_count": args.train_count,
        "val_count": args.val_count,
        "manifest": str(args.manifest) if args.manifest is not None else None,
        "decoder": args.decoder,
        "mask2former_queries": args.mask2former_queries,
        "seed": args.seed,
        "metrics": metrics,
        "confusion_matrix": mat.tolist(),
    }


def _class_names(manifest: DatasetManifest, num_classes: int) -> tuple[str, ...]:
    names = manifest.classes or DEFAULT_LABELS
    if len(names) >= num_classes:
        return tuple(names[:num_classes])
    return tuple(names) + tuple(f"class_{idx}" for idx in range(len(names), num_classes))


def print_report(result: dict[str, object]) -> None:
    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    confusion = torch.tensor(result["confusion_matrix"])
    total_pixels = int(confusion.sum().item())
    print(
        json.dumps(
            {
                "mIoU": metrics["mIoU"],
                "mDice": metrics["mDice"],
                "foreground_recall": metrics["foreground_recall"],
                "boundary_f1": metrics["boundary_f1"],
            },
            ensure_ascii=False,
        )
    )
    print("\nPer-class:")
    per_class = metrics["per_class"]
    assert isinstance(per_class, dict)
    for name, values in per_class.items():
        support = int(values["support_pixels"])
        ratio = support / max(total_pixels, 1)
        print(
            f"{name:18s} "
            f"IoU={values['iou']:.4f} "
            f"Dice={values['dice']:.4f} "
            f"Recall={values['recall']:.4f} "
            f"pixels={support:>10d} "
            f"ratio={ratio:.6f}"
        )
    print("\nConfusion matrix rows=target cols=pred:")
    print(confusion.tolist())


def main() -> int:
    args = parse_args()
    result = evaluate(args)
    print_report(result)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

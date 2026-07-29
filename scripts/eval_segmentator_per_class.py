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

from dataset_config.unified_labels import NUM_FINE
from segmentator.config import (
    DatasetManifest,
    SEGMENTATOR_FINE_CLASSES,
    dataset_annotated_coarse_class_ids,
    dataset_supported_fine_class_ids,
)
from segmentator.data import TissueSegmentationDataset, build_manifest, load_manifest
from segmentator.metrics import (
    dataset_group_macro_iou,
    fine_segmentation_metrics,
    group_macro_iou,
    segmentation_metrics,
)
from segmentator.inference import load_checkpoint


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
    parser.add_argument("--split", choices=["val", "test"], default="val")
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
    parser.add_argument("--mask2former-ignore-index", type=int, default=255)
    parser.add_argument("--symmetric-padding", action="store_true")
    parser.add_argument("--boundary-refinement", action="store_true")
    parser.add_argument("--refinement-gate-mode", choices=["hard", "learned_soft"], default="hard")
    parser.add_argument("--cellvit-mode", choices=["none", "teacher", "input"], default="none")
    parser.add_argument("--hierarchical-fine", action="store_true")
    parser.add_argument("--cell-density-sigma", type=float, default=8.0)
    parser.add_argument("--metric-sample-limit", type=int, default=0)
    parser.add_argument("--skip-spatial-metrics", action="store_true")
    parser.add_argument("--skip-group-macro", action="store_true")
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
    records = list(manifest.test if args.split == "test" else manifest.val)
    if not records:
        raise ValueError(f"manifest has no records for split={args.split}")
    val_ds = TissueSegmentationDataset(
        records,
        image_size=args.image_size,
        augment=False,
        num_classes=args.num_classes,
        remap_invalid_to=args.remap_invalid_to,
        cellvit_mode=args.cellvit_mode,
        cell_density_sigma=args.cell_density_sigma,
        hierarchical_fine=args.hierarchical_fine,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = load_checkpoint(
        args.checkpoint,
        num_classes=args.num_classes,
        local_repo=args.uni2h_repo,
        decoder=args.decoder,
        mask2former_queries=args.mask2former_queries,
        mask2former_ignore_index=args.mask2former_ignore_index,
        symmetric_padding=args.symmetric_padding,
        boundary_refinement=args.boundary_refinement,
        refinement_gate_mode=args.refinement_gate_mode,
        cellvit_mode=args.cellvit_mode,
        hierarchical_fine=args.hierarchical_fine,
    ).to(device)

    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    dataset_ids: list[str] = []
    group_ids: list[str] = []
    fine_preds: list[torch.Tensor] = []
    fine_targets: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="eval", dynamic_ncols=True):
            image = batch["image"].to(device)
            target = batch["mask"].clone()
            nuclei_density = batch.get("nuclei_density")
            if torch.is_tensor(nuclei_density):
                nuclei_density = nuclei_density.to(device)
            fine_allowed = batch.get("fine_allowed")
            if torch.is_tensor(fine_allowed):
                fine_allowed = fine_allowed.to(device)
            outputs = model(image, nuclei_density=nuclei_density, fine_allowed=fine_allowed)
            pred = outputs["pred"].cpu()
            preds.append(pred)
            targets.append(target)
            if args.hierarchical_fine:
                fine_preds.append(outputs["fine_pred"].cpu())
                fine_targets.append(batch["fine_mask"].clone())
            dataset_ids.extend(str(value) for value in batch["dataset_id"])
            group_ids.extend(str(value) for value in batch["group_id"])

    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)
    class_names = _class_names(manifest, args.num_classes)
    metrics = segmentation_metrics(
        pred_all,
        target_all,
        args.num_classes,
        class_names=class_names,
        boundary_width=args.boundary_width,
        metric_sample_limit=args.metric_sample_limit,
        include_spatial_metrics=not args.skip_spatial_metrics,
    )
    metrics["pooled_unified_mIoU"] = float(metrics["mIoU"])
    metrics["pooled_unified_mDice"] = float(metrics["mDice"])
    metrics["pooled_unified_case_macro"] = (
        {"skipped": True}
        if args.skip_group_macro
        else group_macro_iou(
            pred_all,
            target_all,
            group_ids,
            args.num_classes,
        )
    )
    fine_pred_all = torch.cat(fine_preds, dim=0) if args.hierarchical_fine else None
    fine_target_all = torch.cat(fine_targets, dim=0) if args.hierarchical_fine else None
    if args.hierarchical_fine:
        supported_fine_class_ids = tuple(
            sorted(
                {
                    class_id
                    for dataset_id in set(dataset_ids)
                    for class_id in dataset_supported_fine_class_ids(dataset_id)
                }
            )
        )
        metrics["fine"] = fine_segmentation_metrics(
            fine_pred_all,
            fine_target_all,
            NUM_FINE,
            class_names=SEGMENTATOR_FINE_CLASSES,
            evaluated_class_ids=supported_fine_class_ids,
        )
    per_dataset = {}
    evaluated_class_ids_by_dataset: dict[str, tuple[int, ...]] = {}
    for dataset_id in sorted(set(dataset_ids)):
        indices = [index for index, value in enumerate(dataset_ids) if value == dataset_id]
        index_tensor = torch.tensor(indices, dtype=torch.long)
        evaluated_class_ids = dataset_annotated_coarse_class_ids(
            dataset_id,
            num_classes=args.num_classes,
        )
        evaluated_class_ids_by_dataset[dataset_id] = evaluated_class_ids
        dataset_metrics = segmentation_metrics(
            pred_all.index_select(0, index_tensor),
            target_all.index_select(0, index_tensor),
            args.num_classes,
            class_names=class_names,
            boundary_width=args.boundary_width,
            metric_sample_limit=args.metric_sample_limit,
            evaluated_class_ids=evaluated_class_ids,
            include_spatial_metrics=not args.skip_spatial_metrics,
        )
        dataset_metrics["case_macro"] = (
            {"skipped": True}
            if args.skip_group_macro
            else group_macro_iou(
                pred_all.index_select(0, index_tensor),
                target_all.index_select(0, index_tensor),
                [group_ids[index] for index in indices],
                args.num_classes,
                evaluated_class_ids=evaluated_class_ids,
            )
        )
        if args.hierarchical_fine:
            supported_fine_class_ids = dataset_supported_fine_class_ids(dataset_id)
            dataset_metrics["fine"] = fine_segmentation_metrics(
                fine_pred_all.index_select(0, index_tensor),
                fine_target_all.index_select(0, index_tensor),
                NUM_FINE,
                class_names=SEGMENTATOR_FINE_CLASSES,
                evaluated_class_ids=supported_fine_class_ids or None,
            )
        per_dataset[dataset_id] = dataset_metrics
    metrics["per_dataset"] = per_dataset
    coarse_miou_values = [float(value["mIoU"]) for value in per_dataset.values()]
    coarse_mdice_values = [float(value["mDice"]) for value in per_dataset.values()]
    metrics["mIoU"] = sum(coarse_miou_values) / len(coarse_miou_values)
    metrics["mDice"] = sum(coarse_mdice_values) / len(coarse_mdice_values)
    metrics["coarse_dataset_macro_mIoU"] = float(metrics["mIoU"])
    metrics["case_macro"] = (
        {"skipped": True}
        if args.skip_group_macro
        else dataset_group_macro_iou(
            pred_all,
            target_all,
            group_ids,
            dataset_ids,
            args.num_classes,
            evaluated_class_ids_by_dataset,
        )
    )
    if args.hierarchical_fine:
        fine_dataset_values = [
            float(value["fine"]["mIoU"])
            for value in per_dataset.values()
            if value["fine"]["available"]
        ]
        metrics["fine_dataset_macro_mIoU"] = (
            sum(fine_dataset_values) / len(fine_dataset_values)
            if fine_dataset_values
            else float("nan")
        )
    return {
        "device": str(device),
        "checkpoint": str(Path(args.checkpoint)),
        "dataset_root": str(Path(args.dataset_root)),
        "image_size": args.image_size,
        "train_count": args.train_count,
        "val_count": args.val_count,
        "manifest": str(args.manifest) if args.manifest is not None else None,
        "split": args.split,
        "decoder": args.decoder,
        "mask2former_queries": args.mask2former_queries,
        "mask2former_ignore_index": args.mask2former_ignore_index,
        "hierarchical_fine": args.hierarchical_fine,
        "cellvit_mode": args.cellvit_mode,
        "boundary_refinement": args.boundary_refinement,
        "refinement_gate_mode": args.refinement_gate_mode,
        "spatial_metrics_included": not args.skip_spatial_metrics,
        "group_macro_included": not args.skip_group_macro,
        "seed": args.seed,
        "metrics": metrics,
    }


def _class_names(manifest: DatasetManifest, num_classes: int) -> tuple[str, ...]:
    names = manifest.classes or DEFAULT_LABELS
    if len(names) >= num_classes:
        return tuple(names[:num_classes])
    return tuple(names) + tuple(f"class_{idx}" for idx in range(len(names), num_classes))


def print_report(result: dict[str, object]) -> None:
    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    total_pixels = sum(int(values["support_pixels"]) for values in metrics["per_class"].values())
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


def main() -> int:
    args = parse_args()
    result = evaluate(args)
    print_report(result)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

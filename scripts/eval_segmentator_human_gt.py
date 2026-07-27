#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
from scipy import ndimage
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from segmentator.patch_selection import organ_valid_confusion
from segmentator.metrics import fragmentation_metrics


CLASS_NAMES = (
    "background",
    "tumor",
    "stroma",
    "necrosis",
    "immune_infiltrate",
    "normal_epithelium",
    "blood_vessel",
    "other_tissue",
)


def _raw_confusion(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    valid = (gt >= 0) & (gt < 8) & (pred >= 0) & (pred < 8)
    return np.bincount((gt[valid] * 8 + pred[valid]).reshape(-1), minlength=64).reshape(8, 8)


def _boundary(mask: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    result = np.zeros(mask.shape, dtype=bool)
    horizontal = mask[:, 1:] != mask[:, :-1]
    vertical = mask[1:, :] != mask[:-1, :]
    if valid is not None:
        horizontal &= valid[:, 1:] & valid[:, :-1]
        vertical &= valid[1:, :] & valid[:-1, :]
    result[:, 1:] |= horizontal
    result[:, :-1] |= horizontal
    result[1:, :] |= vertical
    result[:-1, :] |= vertical
    return result


def _boundary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    valid = gt != 255
    pred_boundary = _boundary(pred, valid)
    gt_boundary = _boundary(gt, valid)
    if not np.any(pred_boundary) and not np.any(gt_boundary):
        return {"boundary_f1_2": 1.0, "boundary_f1_4": 1.0, "boundary_f1_8": 1.0, "hd95": 0.0}
    pred_distance = ndimage.distance_transform_edt(~pred_boundary)
    gt_distance = ndimage.distance_transform_edt(~gt_boundary)
    result: dict[str, float] = {}
    for tolerance in (2, 4, 8):
        precision = float(np.mean(gt_distance[pred_boundary] <= tolerance)) if np.any(pred_boundary) else 0.0
        recall = float(np.mean(pred_distance[gt_boundary] <= tolerance)) if np.any(gt_boundary) else 0.0
        result[f"boundary_f1_{tolerance}"] = 2 * precision * recall / max(precision + recall, 1e-8)
    distances = []
    if np.any(pred_boundary):
        distances.extend(gt_distance[pred_boundary].tolist())
    if np.any(gt_boundary):
        distances.extend(pred_distance[gt_boundary].tolist())
    result["hd95"] = float(np.percentile(distances, 95)) if distances else 0.0
    return result


def _report(matrix: np.ndarray) -> dict[str, object]:
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    per_class = {}
    ious = []
    dices = []
    for class_id, name in enumerate(CLASS_NAMES):
        union = support[class_id] + predicted[class_id] - true_positive[class_id]
        denom_dice = support[class_id] + predicted[class_id]
        iou = float(true_positive[class_id] / union) if union else float("nan")
        dice = float(2 * true_positive[class_id] / denom_dice) if denom_dice else float("nan")
        precision = float(true_positive[class_id] / predicted[class_id]) if predicted[class_id] else float("nan")
        recall = float(true_positive[class_id] / support[class_id]) if support[class_id] else float("nan")
        per_class[name] = {
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "support_pixels": int(support[class_id]),
        }
        if support[class_id] > 0:
            ious.append(iou)
            dices.append(dice)
    return {
        "mIoU": float(np.nanmean(ious)) if ious else float("nan"),
        "mDice": float(np.nanmean(dices)) if dices else float("nan"),
        "per_class": per_class,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def _bootstrap(records: list[dict[str, object]], key: str, iterations: int, seed: int) -> dict[str, float]:
    by_case: dict[str, list[np.ndarray]] = defaultdict(list)
    for record in records:
        by_case[str(record["case_id"])].append(np.asarray(record[key], dtype=np.int64))
    case_ids = sorted(by_case)
    if not case_ids:
        return {}
    rng = random.Random(seed)
    miou_values = []
    mdice_values = []
    for _ in range(iterations):
        sampled = [rng.choice(case_ids) for _ in case_ids]
        matrix = sum((sum(by_case[case_id], np.zeros((8, 8), dtype=np.int64)) for case_id in sampled), np.zeros((8, 8), dtype=np.int64))
        report = _report(matrix)
        miou_values.append(float(report["mIoU"]))
        mdice_values.append(float(report["mDice"]))
    return {
        "mIoU_mean": float(np.nanmean(miou_values)),
        "mIoU_ci_low": float(np.nanpercentile(miou_values, 2.5)),
        "mIoU_ci_high": float(np.nanpercentile(miou_values, 97.5)),
        "mDice_mean": float(np.nanmean(mdice_values)),
        "mDice_ci_low": float(np.nanpercentile(mdice_values, 2.5)),
        "mDice_ci_high": float(np.nanpercentile(mdice_values, 97.5)),
        "iterations": iterations,
        "case_count": len(case_ids),
    }


def _evaluation_groups(records: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = {"overall": records}
    strata = sorted({str(record.get("stratum", "unknown")) for record in records})
    organs = sorted({str(record["organ"]) for record in records})
    for stratum in strata:
        groups[f"stratum:{stratum}"] = [
            record for record in records if str(record.get("stratum", "unknown")) == stratum
        ]
    for organ in organs:
        groups[f"organ:{organ}"] = [record for record in records if record["organ"] == organ]
    for stratum in strata:
        for organ in organs:
            group = [
                record
                for record in records
                if str(record.get("stratum", "unknown")) == stratum and record["organ"] == organ
            ]
            if group:
                groups[f"stratum:{stratum}/organ:{organ}"] = group
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate segmentator predictions against human PNG masks.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--secondary-ground-truth-dir", type=Path, default=None)
    parser.add_argument("--double-annotation-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = list(csv.DictReader(args.manifest.open()))
    records: list[dict[str, object]] = []
    missing = []
    invalid = []
    for row in manifest:
        filename = row["filename"]
        pred_path = args.prediction_dir / filename
        gt_path = args.ground_truth_dir / filename
        if not pred_path.exists() or not gt_path.exists():
            missing.append(filename)
            continue
        pred = np.asarray(Image.open(pred_path), dtype=np.uint8)
        gt = np.asarray(Image.open(gt_path), dtype=np.uint8)
        if pred.shape != gt.shape or not set(np.unique(gt)).issubset(set(range(8)) | {255}):
            invalid.append(filename)
            continue
        record = dict(row)
        record["raw_confusion"] = _raw_confusion(pred, gt)
        record["organ_confusion"] = organ_valid_confusion(pred, gt, row["organ"])
        record["boundary"] = _boundary_metrics(pred, gt)
        record["prediction"] = pred
        records.append(record)

    groups = _evaluation_groups(records)

    reports = {}
    for name, group in groups.items():
        raw = sum((record["raw_confusion"] for record in group), np.zeros((8, 8), dtype=np.int64))
        organ_valid = sum((record["organ_confusion"] for record in group), np.zeros((8, 8), dtype=np.int64))
        boundary = {
            key: float(np.mean([record["boundary"][key] for record in group])) if group else float("nan")
            for key in ("boundary_f1_2", "boundary_f1_4", "boundary_f1_8", "hd95")
        }
        prediction_tensor = torch.from_numpy(np.stack([record["prediction"] for record in group])) if group else torch.empty(0, 1, 1, dtype=torch.long)
        reports[name] = {
            "samples": len(group),
            "raw_8class": _report(raw),
            "organ_valid": _report(organ_valid),
            "boundary": boundary,
            "fragmentation": fragmentation_metrics(prediction_tensor, num_classes=8),
            "raw_bootstrap": _bootstrap(group, "raw_confusion", args.bootstrap_iterations, args.seed),
            "organ_valid_bootstrap": _bootstrap(group, "organ_confusion", args.bootstrap_iterations, args.seed),
        }

    interannotator = None
    if args.secondary_ground_truth_dir and args.double_annotation_manifest:
        double_rows = list(csv.DictReader(args.double_annotation_manifest.open()))
        agreement_matrix = np.zeros((8, 8), dtype=np.int64)
        agreement_boundary = []
        agreement_missing = []
        for row in double_rows:
            filename = row["filename"]
            primary_path = args.ground_truth_dir / filename
            secondary_path = args.secondary_ground_truth_dir / filename
            if not primary_path.exists() or not secondary_path.exists():
                agreement_missing.append(filename)
                continue
            primary = np.asarray(Image.open(primary_path), dtype=np.uint8)
            secondary = np.asarray(Image.open(secondary_path), dtype=np.uint8)
            if primary.shape != secondary.shape:
                agreement_missing.append(filename)
                continue
            agreement_matrix += _raw_confusion(secondary, primary)
            agreement_boundary.append(_boundary_metrics(secondary, primary))
        interannotator = {
            "evaluated": len(double_rows) - len(agreement_missing),
            "missing": agreement_missing,
            "raw_8class": _report(agreement_matrix),
            "boundary": {
                key: float(np.mean([metrics[key] for metrics in agreement_boundary]))
                if agreement_boundary
                else float("nan")
                for key in ("boundary_f1_2", "boundary_f1_4", "boundary_f1_8", "hd95")
            },
        }

    result = {
        "evaluated": len(records),
        "missing": missing,
        "invalid": invalid,
        "reports": reports,
        "interannotator": interannotator,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n")
    print(json.dumps({"evaluated": len(records), "missing": len(missing), "invalid": len(invalid)}, indent=2))
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())

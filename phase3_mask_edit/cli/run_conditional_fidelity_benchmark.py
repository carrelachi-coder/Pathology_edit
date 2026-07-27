#!/usr/bin/env python3
"""Evaluate generated images against frozen tissue and nuclei conditions."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shlex
import sys

import numpy as np
import yaml

from phase3_mask_edit.benchmark.conditional_fidelity import (
    cell_distribution_metrics,
    detections_from_cellvit_json,
    detections_from_conic,
    load_label_mask,
    rescale_detections,
    spatial_matching_metrics,
    tissue_fidelity_metrics,
)
from phase3_mask_edit.benchmark.pathokid import sha256_file, stable_digest


DEFAULT_CONFIG = Path("benchmark_configs/conditional_fidelity.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--tissue-pred-root", type=Path, required=True)
    parser.add_argument("--cellvit-target-root", type=Path, required=True)
    parser.add_argument("--cellvit-pred-root", type=Path, required=True)
    parser.add_argument("--conic-target-root", type=Path, required=True)
    parser.add_argument("--conic-pred-root", type=Path, required=True)
    parser.add_argument("--sample-id-field", default="sample_id")
    parser.add_argument("--organ-field", default="organ")
    parser.add_argument("--group-field", default="wsi_id")
    parser.add_argument("--target-tissue-field", default="target_tissue_mask")
    parser.add_argument("--target-annotation-field", default="target_annotation_id")
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--max-items", type=int)
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    if suffix in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise TypeError(f"unsupported manifest structure: {path}")
    return records


def require(record: dict, field: str, sample_id: str) -> str:
    value = record.get(field)
    if value in (None, ""):
        raise ValueError(f"{sample_id}: missing field {field!r}")
    return str(value)


def prediction_path(
    root: Path, model_id: str, organ: str, sample_id: str, *suffix: str
) -> Path:
    """Accept both legacy organ-nested outputs and one-load-per-model flat outputs."""
    nested = root / model_id / organ / sample_id
    flat = root / model_id / sample_id
    if suffix:
        nested = nested.joinpath(*suffix)
        flat = flat.joinpath(*suffix)
    return nested if nested.exists() else flat


def scalar_metrics(prefix: str, payload: dict) -> dict[str, float]:
    values = {}
    for key, value in payload.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values[f"{prefix}.{key}"] = float(value)
        elif isinstance(value, dict):
            values.update(scalar_metrics(f"{prefix}.{key}", value))
    return values


def cluster_bootstrap_mean(
    values: np.ndarray, groups: list[str], *, repeats: int, seed: int
) -> dict:
    values = np.asarray(values, dtype=np.float64)
    names = sorted(set(groups))
    if len(values) != len(groups) or not names:
        raise ValueError("bootstrap values/groups mismatch")
    by_group = {name: np.flatnonzero(np.asarray(groups) == name) for name in names}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        selected = rng.choice(names, size=len(names), replace=True)
        indices = np.concatenate([by_group[name] for name in selected])
        draws.append(float(values[indices].mean()))
    draws = np.asarray(draws)
    return {
        "mean": float(values.mean()),
        "bootstrap_mean": float(draws.mean()),
        "bootstrap_std": float(draws.std(ddof=1)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    records = load_records(args.manifest)
    if args.max_items is not None:
        records = records[: args.max_items]
    model_configs = config["models"]
    eligible_models = [
        model_id
        for model_id, model_config in model_configs.items()
        if model_config.get("condition_structure_metrics")
    ]
    models = args.models or eligible_models
    unknown = sorted(set(models) - set(model_configs))
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    ineligible = sorted(set(models) - set(eligible_models))
    if ineligible:
        raise ValueError(
            "target-structure metrics require target geometry; ineligible models: "
            f"{ineligible}"
        )
    if not records:
        raise ValueError("manifest is empty")

    args.output_root.mkdir(parents=True, exist_ok=True)
    result_rows = []
    failures = []
    model_record_counts = {}
    tissue_config = config["tissue"]
    cellvit_config = config["cellvit"]
    conic_config = config["conic"]
    evaluation_config = config["evaluation_frame"]
    evaluation_size = tuple(int(value) for value in evaluation_config["image_size"])
    evaluation_mpp = float(evaluation_config["mpp"])
    if tuple(int(value) for value in cellvit_config["image_size"]) != evaluation_size:
        raise ValueError("CellViT image_size must match the 512 evaluation frame")
    if not np.isclose(float(cellvit_config["evaluation_mpp"]), evaluation_mpp):
        raise ValueError("CellViT MPP must match the evaluation frame")

    for model_id in models:
        model_config = model_configs[model_id]
        allowed_organs = set(model_config.get("organs", []))
        model_records = [
            record
            for record in records
            if not allowed_organs or record[args.organ_field] in allowed_organs
        ]
        model_record_counts[model_id] = len(model_records)
        for record in model_records:
            sample_id = require(record, args.sample_id_field, "unknown")
            organ = require(record, args.organ_field, sample_id)
            target_annotation_id = require(record, args.target_annotation_field, sample_id)
            group = str(record.get(args.group_field) or record.get("pair_id") or sample_id)
            row = {
                "model_id": model_id,
                "sample_id": sample_id,
                "organ": organ,
                "bootstrap_group": group,
                "tissue_fairness": model_config["tissue_fairness"],
                "strict_spatial_evaluator": model_config.get("strict_spatial_evaluator"),
            }
            try:
                target_tissue_path = Path(require(record, args.target_tissue_field, sample_id))
                predicted_tissue_path = prediction_path(
                    args.tissue_pred_root, model_id, organ, f"{sample_id}.png"
                )
                tissue = tissue_fidelity_metrics(
                    load_label_mask(target_tissue_path, expected_size=evaluation_size),
                    load_label_mask(predicted_tissue_path, expected_size=evaluation_size),
                    class_ids=tissue_config["class_ids"],
                    ignore_index=int(tissue_config["ignore_index"]),
                    background_id=int(tissue_config["background_id"]),
                    presence_min_fraction=float(tissue_config["presence_min_fraction"]),
                )
                row["tissue"] = tissue

                if model_config.get("cellvit_global"):
                    image_size = evaluation_size
                    target_cellvit = detections_from_cellvit_json(
                        args.cellvit_target_root / f"{target_annotation_id}.json",
                        mpp=float(cellvit_config["evaluation_mpp"]),
                        image_size=image_size,
                    )
                    predicted_cellvit = detections_from_cellvit_json(
                        prediction_path(
                            args.cellvit_pred_root, model_id, organ, f"{sample_id}.json"
                        ),
                        mpp=float(cellvit_config["evaluation_mpp"]),
                        image_size=image_size,
                    )
                    row["cellvit_global"] = cell_distribution_metrics(
                        target_cellvit,
                        predicted_cellvit,
                        class_ids=cellvit_config["class_ids"],
                    )
                    if model_config.get("strict_spatial_evaluator") == "cellvit":
                        row["strict_spatial"] = {
                            "evaluator": "cellvit",
                            "fairness": "fair_main",
                            "class_agnostic": spatial_matching_metrics(
                                target_cellvit,
                                predicted_cellvit,
                                max_distance_um=float(cellvit_config["matching_distance_um"]),
                                class_aware=False,
                            ),
                            "class_aware": spatial_matching_metrics(
                                target_cellvit,
                                predicted_cellvit,
                                max_distance_um=float(cellvit_config["matching_distance_um"]),
                                class_aware=True,
                            ),
                        }

                if model_config.get("strict_spatial_evaluator") == "conic":
                    detector_size = tuple(
                        int(value) for value in conic_config["detector_image_size"]
                    )
                    target_conic = detections_from_conic(
                        args.conic_target_root / target_annotation_id / "conic.npy",
                        mpp=float(conic_config["detector_mpp"]),
                    )
                    predicted_conic = detections_from_conic(
                        prediction_path(
                            args.conic_pred_root, model_id, organ, sample_id, "conic.npy"
                        ),
                        mpp=float(conic_config["detector_mpp"]),
                    )
                    if target_conic.image_size != detector_size:
                        raise ValueError(
                            f"unexpected target CoNIC detector frame {target_conic.image_size}"
                        )
                    if predicted_conic.image_size != detector_size:
                        raise ValueError(
                            f"unexpected predicted CoNIC detector frame {predicted_conic.image_size}"
                        )
                    target_conic = rescale_detections(
                        target_conic, image_size=evaluation_size, mpp=evaluation_mpp
                    )
                    predicted_conic = rescale_detections(
                        predicted_conic, image_size=evaluation_size, mpp=evaluation_mpp
                    )
                    row["conic_distribution"] = cell_distribution_metrics(
                        target_conic,
                        predicted_conic,
                        class_ids=conic_config["class_ids"],
                    )
                    row["strict_spatial"] = {
                        "evaluator": "conic_hovernet",
                        "fairness": "fair_main",
                        "class_agnostic": spatial_matching_metrics(
                            target_conic,
                            predicted_conic,
                            max_distance_um=float(conic_config["matching_distance_um"]),
                            class_aware=False,
                        ),
                        "class_aware": spatial_matching_metrics(
                            target_conic,
                            predicted_conic,
                            max_distance_um=float(conic_config["matching_distance_um"]),
                            class_aware=True,
                        ),
                    }
                result_rows.append(row)
            except Exception as exc:
                failures.append(
                    {
                        "model_id": model_id,
                        "sample_id": sample_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    rows_path = args.output_root / "conditional_fidelity_rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result_rows),
        encoding="utf-8",
    )

    summary_rows = []
    for model_index, model_id in enumerate(models):
        selected = [row for row in result_rows if row["model_id"] == model_id]
        if not selected:
            continue
        metric_values: dict[str, list[float]] = {}
        metric_groups: dict[str, list[str]] = {}
        for row in selected:
            flattened = {}
            flattened.update(scalar_metrics("tissue", row["tissue"]))
            if "cellvit_global" in row:
                flattened.update(scalar_metrics("cellvit_global", row["cellvit_global"]))
            if "conic_distribution" in row:
                flattened.update(scalar_metrics("conic_distribution", row["conic_distribution"]))
            if "strict_spatial" in row:
                flattened.update(
                    scalar_metrics(
                        "strict_spatial.class_agnostic", row["strict_spatial"]["class_agnostic"]
                    )
                )
                flattened.update(
                    scalar_metrics(
                        "strict_spatial.class_aware", row["strict_spatial"]["class_aware"]
                    )
                )
            for metric, value in flattened.items():
                if np.isfinite(value):
                    metric_values.setdefault(metric, []).append(value)
                    metric_groups.setdefault(metric, []).append(row["bootstrap_group"])
        for metric_index, (metric, values) in enumerate(sorted(metric_values.items())):
            summary_rows.append(
                {
                    "model_id": model_id,
                    "metric": metric,
                    "sample_count": len(values),
                    **cluster_bootstrap_mean(
                        np.asarray(values),
                        metric_groups[metric],
                        repeats=args.bootstrap_repeats,
                        seed=args.seed + 1009 * model_index + metric_index,
                    ),
                }
            )

    summary_csv = args.output_root / "conditional_fidelity_summary.csv"
    if summary_rows:
        with summary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
    report = {
        "schema_version": 1,
        "status": "failed" if failures else "completed",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "models": models,
        "requested_rows": sum(model_record_counts.values()),
        "model_record_counts": model_record_counts,
        "completed_rows": len(result_rows),
        "failures": failures,
        "rows": str(rows_path),
        "summary_csv": str(summary_csv),
        "bootstrap": {
            "group_field": args.group_field,
            "repeats": args.bootstrap_repeats,
            "seed": args.seed,
        },
        "metric_policy": {
            "evaluation_frame": {
                "image_size": list(evaluation_size),
                "mpp": evaluation_mpp,
                "fov_um": evaluation_size[0] * evaluation_mpp,
                "source": "MPP-normalized generated patches only",
            },
            "eligible_models": eligible_models,
            "ineligible_models": sorted(set(model_configs) - set(eligible_models)),
            "common_cell_distribution": "CellViT for eligible models only",
            "strict_spatial": {
                "cross_v1_project": "CellViT",
                "pixcell_controlnet": "CellViT",
                "pathdiff_conic": "HoVer-Net CoNIC",
                "models_without_target_geometry": "not_applicable",
            },
            "claim_name": "segmentator/CellViT/HoVer-Net-derived condition consistency",
        },
        "runtime": {
            "command": shlex.join(sys.argv),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_ids_sha256": stable_digest(
            [row[args.sample_id_field] for row in records]
        ),
    }
    report_path = args.output_root / "conditional_fidelity_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(report_path)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Analyze paired moderate-to-significant embedding editing trajectories."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from phase3_mask_edit.benchmark.embedding_utility import (
    compute_embedding_dose_response_scores,
    summarize_scores,
)
from phase3_mask_edit.benchmark.pathokid import sha256_file


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moderate-manifest", type=Path, required=True)
    parser.add_argument("--moderate-cache-root", type=Path, required=True)
    parser.add_argument("--significant-manifest", type=Path, required=True)
    parser.add_argument("--significant-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--feature-extractor-name", default="uni2h")
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--dose-field",
        default="changed_area_fraction",
        help="Significant-row field containing the realized significant dose.",
    )
    parser.add_argument(
        "--moderate-dose-field",
        default="moderate_changed_area_fraction",
        help="Significant-row field containing the paired realized moderate dose.",
    )
    return parser.parse_args(argv)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_features(cache_root: Path, set_name: str) -> tuple[np.ndarray, np.ndarray]:
    path = cache_root / f"{set_name}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = np.load(path)
    return payload["sample_ids"].astype(str), payload["features"].astype(np.float64)


def _index_by_id(ids: np.ndarray, *, label: str) -> dict[str, int]:
    mapping = {str(sample_id): index for index, sample_id in enumerate(ids)}
    if len(mapping) != len(ids):
        raise ValueError(f"{label} feature cache sample IDs must be unique")
    return mapping


def _unit(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("cannot normalize zero-norm feature row")
    return values / norms


def _summary(
    values: np.ndarray,
    groups: list[str],
    *,
    repeats: int,
    seed: int,
) -> dict:
    return summarize_scores(
        np.asarray(values, dtype=np.float64),
        groups,
        bootstrap_repeats=repeats,
        seed=seed,
    )


def _spearman(x: np.ndarray, y: np.ndarray) -> dict[str, float | str | None]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        return {"rho": None, "pvalue": None, "status": "undefined_constant_input"}
    result = spearmanr(x, y)
    return {
        "rho": float(result.statistic),
        "pvalue": float(result.pvalue),
        "status": "complete",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    moderate_rows = _read_jsonl(args.moderate_manifest)
    significant_rows = _read_jsonl(args.significant_manifest)
    if args.expected_count is not None and len(significant_rows) != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} significant rows, found {len(significant_rows)}"
        )
    if len(significant_rows) < 3:
        raise ValueError("dose-response analysis requires at least three paired rows")
    moderate_manifest_ids = {str(row["sample_id"]) for row in moderate_rows}
    significant_ids = [str(row["sample_id"]) for row in significant_rows]
    pair_ids = [
        str(row.get("moderate_sample_id") or row.get("pair_id") or "")
        for row in significant_rows
    ]
    if any(not pair_id for pair_id in pair_ids):
        raise ValueError("every significant row requires pair_id or moderate_sample_id")
    if len(set(significant_ids)) != len(significant_ids) or len(set(pair_ids)) != len(pair_ids):
        raise ValueError("significant sample IDs and moderate pair IDs must be unique")
    missing_pairs = sorted(set(pair_ids) - moderate_manifest_ids)
    if missing_pairs:
        raise ValueError(f"moderate manifest is missing paired IDs: {missing_pairs[:10]}")
    groups = [str(row["wsi_id"]) for row in significant_rows]
    moderate_fraction = np.asarray(
        [float(row[args.moderate_dose_field]) for row in significant_rows]
    )
    significant_fraction = np.asarray(
        [float(row[args.dose_field]) for row in significant_rows]
    )
    dose_increase = significant_fraction - moderate_fraction
    if np.any(dose_increase <= 0):
        raise ValueError("every significant row must have positive realized dose increase")

    moderate_cache: dict[str, np.ndarray] = {}
    significant_cache: dict[str, np.ndarray] = {}
    moderate_ids, moderate_reference_all = _load_features(
        args.moderate_cache_root, "reference"
    )
    significant_cache_ids, significant_reference_all = _load_features(
        args.significant_cache_root, "reference"
    )
    moderate_index = _index_by_id(moderate_ids, label="moderate")
    significant_index = _index_by_id(significant_cache_ids, label="significant")
    moderate_order = np.asarray([moderate_index[pair_id] for pair_id in pair_ids])
    significant_order = np.asarray(
        [significant_index[sample_id] for sample_id in significant_ids]
    )
    shared_reference = moderate_reference_all[moderate_order]
    significant_reference = significant_reference_all[significant_order]
    reference_cosine = np.sum(
        _unit(shared_reference) * _unit(significant_reference), axis=1
    )
    for backend in ("inpaint", "cross"):
        ids, values = _load_features(args.moderate_cache_root, backend)
        if not np.array_equal(ids, moderate_ids):
            raise ValueError(f"moderate {backend} cache order differs from reference")
        moderate_cache[backend] = values[moderate_order]
        ids, values = _load_features(args.significant_cache_root, backend)
        if not np.array_equal(ids, significant_cache_ids):
            raise ValueError(f"significant {backend} cache order differs from reference")
        significant_cache[backend] = values[significant_order]

    metrics_by_backend: dict[str, dict[str, np.ndarray]] = {}
    report_backends: dict[str, dict] = {}
    for backend_index, backend in enumerate(("inpaint", "cross")):
        scores = compute_embedding_dose_response_scores(
            shared_reference,
            moderate_cache[backend],
            significant_cache[backend],
            groups,
        )
        arrays = {
            field: np.asarray(getattr(scores, field), dtype=np.float64)
            for field in scores.__dataclass_fields__
        }
        metrics_by_backend[backend] = arrays
        summary_fields = (
            "moderate_directional_consistency",
            "significant_directional_consistency",
            "directional_consistency_change",
            "matched_cross_strength_agreement",
            "significant_to_moderate_centroid_alignment",
            "incremental_to_moderate_centroid_alignment",
            "moderate_centroid_projection",
            "significant_centroid_projection",
            "incremental_centroid_projection",
            "moderate_displacement_norm",
            "significant_displacement_norm",
            "displacement_norm_change",
            "displacement_norm_ratio",
        )
        summaries = {
            field: _summary(
                arrays[field],
                groups,
                repeats=args.bootstrap_repeats,
                seed=args.seed + backend_index * 10000 + field_index * 101,
            )
            for field_index, field in enumerate(summary_fields)
        }
        summaries["positive_incremental_projection_fraction"] = _summary(
            (arrays["incremental_centroid_projection"] > 0).astype(np.float64),
            groups,
            repeats=args.bootstrap_repeats,
            seed=args.seed + backend_index * 10000 + 7001,
        )
        summaries["increased_displacement_norm_fraction"] = _summary(
            (arrays["displacement_norm_change"] > 0).astype(np.float64),
            groups,
            repeats=args.bootstrap_repeats,
            seed=args.seed + backend_index * 10000 + 8009,
        )
        report_backends[backend] = {
            "metrics": summaries,
            "dose_correlations": {
                "dose_increase_vs_incremental_projection": _spearman(
                    dose_increase, arrays["incremental_centroid_projection"]
                ),
                "dose_increase_vs_displacement_norm_change": _spearman(
                    dose_increase, arrays["displacement_norm_change"]
                ),
                "significant_fraction_vs_significant_displacement_norm": _spearman(
                    significant_fraction, arrays["significant_displacement_norm"]
                ),
            },
        }

    args.output_root.mkdir(parents=True, exist_ok=True)
    row_path = args.output_root / "embedding_utility_dose_response_rows.csv"
    metric_names = list(metrics_by_backend["inpaint"])
    with row_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "sample_id",
            "pair_id",
            "wsi_id",
            "moderate_changed_area_fraction",
            "significant_changed_area_fraction",
            "dose_increase_fraction",
        ] + [
            f"{backend}_{metric}"
            for backend in ("inpaint", "cross")
            for metric in metric_names
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, sample_id in enumerate(significant_ids):
            output = {
                "sample_id": sample_id,
                "pair_id": pair_ids[index],
                "wsi_id": groups[index],
                "moderate_changed_area_fraction": float(moderate_fraction[index]),
                "significant_changed_area_fraction": float(significant_fraction[index]),
                "dose_increase_fraction": float(dose_increase[index]),
            }
            for backend in ("inpaint", "cross"):
                for metric in metric_names:
                    output[f"{backend}_{metric}"] = float(
                        metrics_by_backend[backend][metric][index]
                    )
            writer.writerow(output)
    report = {
        "schema_version": 1,
        "status": "complete",
        "analysis": f"paired_moderate_to_significant_tumor_increase_{args.feature_extractor_name}",
        "feature_extractor": args.feature_extractor_name,
        "sample_count": len(significant_rows),
        "wsi_count": len(set(groups)),
        "moderate_manifest": str(args.moderate_manifest),
        "moderate_manifest_sha256": sha256_file(args.moderate_manifest),
        "significant_manifest": str(args.significant_manifest),
        "significant_manifest_sha256": sha256_file(args.significant_manifest),
        "reference_cache_alignment": {
            "mean_cosine": float(reference_cosine.mean()),
            "minimum_cosine": float(reference_cosine.min()),
            "shared_reference_policy": "moderate cache used for both strength displacements",
        },
        "mask_dose": {
            "dose_field": args.dose_field,
            "moderate_dose_field": args.moderate_dose_field,
            "moderate_changed_area_fraction": _summary(
                moderate_fraction,
                groups,
                repeats=args.bootstrap_repeats,
                seed=args.seed + 9001,
            ),
            "significant_changed_area_fraction": _summary(
                significant_fraction,
                groups,
                repeats=args.bootstrap_repeats,
                seed=args.seed + 9011,
            ),
            "dose_increase_fraction": _summary(
                dose_increase,
                groups,
                repeats=args.bootstrap_repeats,
                seed=args.seed + 9029,
            ),
        },
        "backends": report_backends,
        "bootstrap": {
            "unit": "wsi_id",
            "repeats": args.bootstrap_repeats,
            "seed": args.seed,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = args.output_root / "embedding_utility_dose_response_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

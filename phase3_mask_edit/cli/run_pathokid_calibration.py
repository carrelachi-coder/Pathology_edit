#!/usr/bin/env python3
"""Calibrate Patho-KID with organ-preserving real-vs-real draws."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time

import numpy as np
import yaml

from phase3_mask_edit.benchmark.pathokid import (
    build_conch_extractor,
    build_uni2h_extractor,
    real_vs_real_kid_curve,
    sha256_file,
    stable_digest,
    summarize_values,
)
from phase3_mask_edit.cli.run_pathokid_benchmark import (
    get_or_extract_features,
    load_records,
    runtime_provenance,
    validate_image_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("benchmark_configs/pathokid.yaml"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--real-path-field", default="target_image")
    parser.add_argument("--sample-id-field", default="target_annotation_id")
    parser.add_argument("--organ-field", default="organ")
    parser.add_argument("--feature-extractors", nargs="+", choices=("conch", "uni2h"))
    parser.add_argument("--sample-sizes", nargs="+", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--overwrite-cache", action="store_true")
    return parser.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    evaluation = config["evaluation"]
    evaluation_frame = config["evaluation_frame"]
    calibration = config["calibration"]
    extractor_configs = config["feature_extractors"]
    extractor_names = args.feature_extractors or list(extractor_configs)
    sample_sizes = args.sample_sizes or [int(value) for value in calibration["sample_sizes"]]
    repeats = args.repeats or int(calibration["repeats"])
    seed = args.seed if args.seed is not None else int(calibration["seed"])
    dtype_name = args.dtype or evaluation["dtype"]
    batch_size = args.batch_size or int(evaluation["batch_size"])

    records = load_records(args.manifest)
    sample_ids = [str(record[args.sample_id_field]) for record in records]
    organs = [str(record[args.organ_field]) for record in records]
    image_paths = [Path(record[args.real_path_field]) for record in records]
    if len(records) < max(sample_sizes):
        raise ValueError(
            f"largest calibration size {max(sample_sizes)} exceeds real pool {len(records)}"
        )
    if len(sample_ids) != len(set(sample_ids)) or len(image_paths) != len(set(map(str, image_paths))):
        raise ValueError("calibration requires unique sample IDs and real image paths")
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} real images: {missing[:10]}")
    validate_image_frame(
        image_paths,
        expected_size=tuple(int(value) for value in evaluation_frame["image_size"]),
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "status": "running",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "sample_count": len(records),
        "sample_ids_sha256": stable_digest(sample_ids),
        "organ_counts": {name: organs.count(name) for name in sorted(set(organs))},
        "sample_sizes": sample_sizes,
        "repeats": repeats,
        "seed": seed,
        "feature_normalization": "none_raw_activations",
        "kernel": "(x^T y / d + 1)^3",
        "evaluation_frame": evaluation_frame,
        "interpretation": {
            "disjoint": "two organ-preserving real subsets with no shared source image",
            "bootstrap": (
                "two independent organ-preserving empirical bootstrap samples; source overlap "
                "is reported and this row is not a disjoint real-set floor"
            ),
        },
        "feature_extractors": {},
        "results": {},
        "runtime": runtime_provenance(args.device),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = args.output_root / "pathokid_calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_rows = []
    draw_payload = {}

    for extractor_index, name in enumerate(extractor_names):
        cfg = extractor_configs[name]
        checkpoint = Path(cfg["checkpoint"])
        checkpoint_hash = sha256_file(checkpoint)
        if name == "uni2h":
            extractor = build_uni2h_extractor(
                Path(cfg["root"]),
                device=args.device,
                dtype_name=dtype_name,
                checkpoint_sha256=checkpoint_hash,
            )
        else:
            extractor = build_conch_extractor(
                Path(cfg["root"]),
                checkpoint,
                device=args.device,
                dtype_name=dtype_name,
                checkpoint_sha256=checkpoint_hash,
            )
        extractor_digest = stable_digest(extractor.metadata)
        report["feature_extractors"][name] = {
            **extractor.metadata,
            "extractor_digest": extractor_digest,
        }
        raw, cache_hit = get_or_extract_features(
            extractor=extractor,
            extractor_digest=extractor_digest,
            set_name="real_calibration_pool",
            sample_ids=sample_ids,
            image_paths=image_paths,
            cache_root=args.output_root / "cache",
            batch_size=batch_size,
            overwrite=args.overwrite_cache,
        )
        features = np.asarray(raw, dtype=np.float64)
        curve = real_vs_real_kid_curve(
            features,
            organs,
            sample_sizes=sample_sizes,
            repeats=repeats,
            seed=seed + 1009 * extractor_index,
        )
        extractor_results = {}
        draw_payload[name] = {}
        for sample_size in sample_sizes:
            item = curve[sample_size]
            summary = summarize_values(item["values"])
            overlap = item["source_overlap_count"]
            result = {
                "sample_size_per_side": sample_size,
                "sampling_mode": item["sampling_mode"],
                "stratum_counts_per_side": item["stratum_counts_per_side"],
                "kid": {
                    **summary,
                    "mean_x1000": summary["mean"] * 1000.0,
                    "std_x1000": summary["std"] * 1000.0,
                    "ci95_low_x1000": summary["ci95_low"] * 1000.0,
                    "ci95_high_x1000": summary["ci95_high"] * 1000.0,
                },
                "source_overlap": {
                    "mean": float(overlap.mean()),
                    "min": int(overlap.min()),
                    "max": int(overlap.max()),
                },
            }
            extractor_results[str(sample_size)] = result
            draw_payload[name][str(sample_size)] = {
                "kid": item["values"].tolist(),
                "source_overlap_count": overlap.tolist(),
            }
            summary_rows.append(
                {
                    "extractor": name,
                    "sample_size_per_side": sample_size,
                    "sampling_mode": item["sampling_mode"],
                    "kid_mean": summary["mean"],
                    "kid_std": summary["std"],
                    "kid_ci95_low": summary["ci95_low"],
                    "kid_ci95_high": summary["ci95_high"],
                    "kid_mean_x1000": summary["mean"] * 1000.0,
                    "source_overlap_mean": float(overlap.mean()),
                }
            )
        report["results"][name] = {
            "feature_dimension": int(features.shape[1]),
            "cache_hit": cache_hit,
            "curve": extractor_results,
        }
        del extractor
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_path = args.output_root / "pathokid_calibration_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    draws_path = args.output_root / "pathokid_calibration_draws.json"
    draws_path.write_text(json.dumps(draw_payload), encoding="utf-8")
    report.update(
        {
            "status": "completed",
            "summary_csv": str(summary_path),
            "draws": str(draws_path),
            "runtime_seconds": round(time.time() - started, 3),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

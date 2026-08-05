#!/usr/bin/env python3
"""Stratify paired Patho-KID comparisons and render guarded-Cross QA sheets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase3_mask_edit.benchmark.pathokid import (
    cluster_bootstrap_kid,
    make_cluster_bootstrap_draws,
    paired_bootstrap_delta,
    summarize_values,
    unbiased_kid,
)


def parse_mapping(values: list[str], option: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} requires NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        if not name or name in result:
            raise ValueError(f"invalid or duplicate {option} name {name!r}")
        result[name] = Path(raw_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--annotation-manifest", type=Path, required=True)
    parser.add_argument(
        "--feature-cache-root",
        action="append",
        required=True,
        help="EXTRACTOR=directory containing real.npz and model caches",
    )
    parser.add_argument(
        "--model-cache",
        action="append",
        required=True,
        help="DISPLAY_NAME=cache basename without .npz",
    )
    parser.add_argument("--guarded-model", required=True)
    parser.add_argument("--old-model", required=True)
    parser.add_argument("--autocond-model", required=True)
    parser.add_argument(
        "--generation-root",
        action="append",
        default=[],
        help="DISPLAY_NAME=Cross output root used for worst-case sheets",
    )
    parser.add_argument("--guarded-mask-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--worst-count", type=int, default=12)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_features(path: Path, expected_ids: list[str]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        sample_ids = payload["sample_ids"].astype(str).tolist()
        features = payload["features"].astype(np.float64)
    if sample_ids != expected_ids:
        raise ValueError(f"sample order mismatch in {path}")
    if not np.isfinite(features).all():
        raise ValueError(f"non-finite features in {path}")
    return features


def load_project_ids(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {
        str(row["annotation_id"]): str(row.get("project_id") or "unknown")
        for row in rows
    }
    if len(result) != len(rows):
        raise ValueError("annotation manifest has duplicate annotation_id values")
    return result


def bootstrap_summary(values: np.ndarray) -> dict[str, float | int]:
    summary = summarize_values(values)
    return {
        **summary,
        "mean_x1000": summary["mean"] * 1000.0,
        "ci95_low_x1000": summary["ci95_low"] * 1000.0,
        "ci95_high_x1000": summary["ci95_high"] * 1000.0,
    }


def analyze_stratum(
    indices: np.ndarray,
    real: np.ndarray,
    models: dict[str, np.ndarray],
    groups: np.ndarray,
    *,
    guarded_model: str,
    old_model: str,
    autocond_model: str,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    selected_groups = groups[indices]
    draws = make_cluster_bootstrap_draws(
        selected_groups, repeats=repeats, seed=seed
    )
    model_bootstrap: dict[str, np.ndarray] = {}
    model_results = {}
    for name, features in models.items():
        generated = features[indices]
        bootstrap = cluster_bootstrap_kid(
            real[indices], generated, selected_groups, draws
        )
        model_bootstrap[name] = bootstrap
        full = unbiased_kid(real[indices], generated)
        model_results[name] = {
            "kid_full": full,
            "kid_full_x1000": full * 1000.0,
            "cluster_bootstrap": bootstrap_summary(bootstrap),
        }

    deltas = {}
    for baseline in (old_model, autocond_model):
        paired = paired_bootstrap_delta(
            model_bootstrap[guarded_model], model_bootstrap[baseline]
        )
        deltas[f"{guarded_model}__minus__{baseline}"] = {
            "delta_definition": "KID(guarded) - KID(baseline); negative favors guarded",
            "kid_full_delta": (
                model_results[guarded_model]["kid_full"]
                - model_results[baseline]["kid_full"]
            ),
            **paired,
        }
    return {
        "sample_count": int(len(indices)),
        "pair_cluster_count": int(len(set(selected_groups.tolist()))),
        "models": model_results,
        "paired_deltas": deltas,
    }


def rank_percentile(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(len(values) - 1, 1)


def fit_label(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    return text[: max(width - 3, 1)] + "..."


def render_generation_sheet(
    destination: Path,
    records: dict[str, dict[str, Any]],
    selected: list[dict[str, Any]],
    generation_roots: dict[str, Path],
) -> None:
    names = ["reference", "target", *generation_roots]
    tile = 256
    label_height = 28
    canvas = Image.new(
        "RGB", (tile * len(names), (tile + label_height) * len(selected)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, item in enumerate(selected):
        record = records[item["sample_id"]]
        paths = [Path(record["reference_image"]), Path(record["target_image"])]
        paths.extend(
            root / record["organ"] / record["sample_id"] / "generated.png"
            for root in generation_roots.values()
        )
        y = row_index * (tile + label_height)
        for column, (name, path) in enumerate(zip(names, paths)):
            with Image.open(path) as image:
                panel = image.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
            x = column * tile
            canvas.paste(panel, (x, y + label_height))
            label = fit_label(
                f"{item['sample_id']} | {name}" if column == 0 else name, 38
            )
            draw.text((x + 4, y + 7), label, fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=94)


def render_mask_sheet(
    destination: Path,
    selected: list[dict[str, Any]],
    guarded_mask_root: Path,
) -> None:
    if not selected:
        return
    tile = 256
    label_height = 28
    columns = ("RGB", "V2 profile", "Guarded")
    canvas = Image.new(
        "RGB", (tile * len(columns), (tile + label_height) * len(selected)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    palette = np.asarray(
        [
            [255, 255, 255], [215, 25, 28], [253, 174, 97], [75, 0, 130],
            [69, 117, 180], [145, 207, 96], [0, 150, 136], [128, 128, 128],
        ],
        dtype=np.uint8,
    )
    for row_index, row in enumerate(selected):
        prediction = guarded_mask_root / "predictions" / row["organ"] / row["annotation_id"]
        with Image.open(row["image"]) as image:
            rgb = image.convert("RGB")
        panels = [rgb]
        for name in ("v2_mask.png", "coarse_mask.png"):
            with Image.open(prediction / name) as image:
                mask = np.asarray(image, dtype=np.uint8)
            panels.append(Image.fromarray(palette[mask], mode="RGB"))
        y = row_index * (tile + label_height)
        for column, (name, panel) in enumerate(zip(columns, panels)):
            x = column * tile
            canvas.paste(
                panel.resize((tile, tile), Image.Resampling.NEAREST),
                (x, y + label_height),
            )
            label = (
                f"{row['annotation_id']} scale={row['guard_scale']:.3f}"
                if column == 0
                else name
            )
            draw.text((x + 4, y + 7), fit_label(label, 38), fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=94)


def render_mask_audit_sheets(
    output_root: Path,
    rows: list[dict[str, Any]],
    guarded_mask_root: Path,
) -> None:
    guard_rows = sorted(
        (row for row in rows if row["guard_triggered"]),
        key=lambda row: (row["guard_scale"], -row["changed_pixel_fraction_vs_v2"]),
    )[:12]
    render_mask_sheet(
        output_root / "visualizations" / "guard_trigger_cases.jpg",
        guard_rows,
        guarded_mask_root,
    )
    lung_vessel_rows = sorted(
        (row for row in rows if row["organ"] == "lung"),
        key=lambda row: (
            row["final_class_fractions"]["6"]
            - row["v2_class_fractions"]["6"]
        ),
        reverse=True,
    )[:12]
    render_mask_sheet(
        output_root / "visualizations" / "lung_vessel_risk_cases.jpg",
        lung_vessel_rows,
        guarded_mask_root,
    )
    changed_rows = sorted(
        rows,
        key=lambda row: row["changed_pixel_fraction_vs_v2"],
        reverse=True,
    )[:12]
    render_mask_sheet(
        output_root / "visualizations" / "largest_mask_changes.jpg",
        changed_rows,
        guarded_mask_root,
    )


def main() -> int:
    args = parse_args()
    feature_roots = parse_mapping(args.feature_cache_root, "--feature-cache-root")
    model_caches = parse_mapping(args.model_cache, "--model-cache")
    generation_roots = parse_mapping(args.generation_root, "--generation-root")
    for required in (args.guarded_model, args.old_model, args.autocond_model):
        if required not in model_caches:
            raise ValueError(f"comparison model {required!r} is not in --model-cache")

    manifest_payload = json.loads(args.evaluation_manifest.read_text(encoding="utf-8"))
    records = manifest_payload["records"]
    sample_ids = [str(record["sample_id"]) for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("evaluation manifest has duplicate sample IDs")
    record_by_id = {str(record["sample_id"]): record for record in records}
    project_by_annotation = load_project_ids(args.annotation_manifest)
    organs = np.asarray([str(record["organ"]) for record in records])
    groups = np.asarray([str(record["pair_id"]) for record in records])
    profiles = np.asarray(
        [project_by_annotation[str(record["target_annotation_id"])] for record in records]
    )

    result: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_manifest": str(args.evaluation_manifest.resolve()),
        "sample_count": len(records),
        "bootstrap": {"group": "pair_id", "repeats": args.bootstrap_repeats, "seed": args.seed},
        "feature_normalization": "none_raw_activations",
        "kernel": "(x^T y / d + 1)^3",
        "extractors": {},
    }
    sample_metrics = {
        sample_id: {
            "sample_id": sample_id,
            "organ": str(organs[index]),
            "profile": str(profiles[index]),
            "pair_id": str(groups[index]),
        }
        for index, sample_id in enumerate(sample_ids)
    }

    for extractor, cache_root in feature_roots.items():
        real = load_features(cache_root / "real.npz", sample_ids)
        models = {
            name: load_features(cache_root / f"{cache_name}.npz", sample_ids)
            for name, cache_name in model_caches.items()
        }
        strata = {
            "overall": {"all": np.arange(len(sample_ids), dtype=np.int64)},
            "organ": {
                value: np.flatnonzero(organs == value)
                for value in sorted(set(organs.tolist()))
            },
            "profile": {
                value: np.flatnonzero(profiles == value)
                for value in sorted(set(profiles.tolist()))
            },
        }
        extractor_result = {"stratifications": {}}
        for stratification, entries in strata.items():
            extractor_result["stratifications"][stratification] = {
                name: analyze_stratum(
                    indices,
                    real,
                    models,
                    groups,
                    guarded_model=args.guarded_model,
                    old_model=args.old_model,
                    autocond_model=args.autocond_model,
                    repeats=args.bootstrap_repeats,
                    seed=args.seed,
                )
                for name, indices in entries.items()
            }
        result["extractors"][extractor] = extractor_result

        mse = {
            name: np.mean(np.square(features - real), axis=1)
            for name, features in models.items()
        }
        delta_old = mse[args.guarded_model] - mse[args.old_model]
        delta_auto = mse[args.guarded_model] - mse[args.autocond_model]
        percentile_old = rank_percentile(delta_old)
        percentile_auto = rank_percentile(delta_auto)
        for index, sample_id in enumerate(sample_ids):
            row = sample_metrics[sample_id]
            row[f"{extractor}_guarded_mse"] = float(mse[args.guarded_model][index])
            row[f"{extractor}_delta_mse_vs_old"] = float(delta_old[index])
            row[f"{extractor}_delta_mse_vs_autocond"] = float(delta_auto[index])
            row[f"{extractor}_worse_rank_vs_old"] = float(percentile_old[index])
            row[f"{extractor}_worse_rank_vs_autocond"] = float(percentile_auto[index])

    rank_keys = [
        f"{extractor}_worse_rank_vs_old" for extractor in feature_roots
    ]
    for row in sample_metrics.values():
        row["mean_worse_rank_vs_old"] = float(np.mean([row[key] for key in rank_keys]))
    ordered_metrics = sorted(
        sample_metrics.values(),
        key=lambda row: row["mean_worse_rank_vs_old"],
        reverse=True,
    )
    worst = ordered_metrics[: args.worst_count]
    result["worst_case_definition"] = (
        "Auxiliary per-sample raw-feature MSE degradation, guarded minus old Cross, "
        "rank-averaged across extractors; this is not per-image KID."
    )
    result["worst_cases"] = worst

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "stratified_pathokid.json", result)
    fieldnames = sorted({key for row in ordered_metrics for key in row})
    with (args.output_root / "sample_feature_distance_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered_metrics)

    if generation_roots:
        render_generation_sheet(
            args.output_root / "visualizations" / "worst_generation_cases.jpg",
            record_by_id,
            worst,
            generation_roots,
        )
    if args.guarded_mask_root:
        rows_path = args.guarded_mask_root / "segmentation_rows.json"
        render_mask_audit_sheets(
            args.output_root,
            json.loads(rows_path.read_text(encoding="utf-8")),
            args.guarded_mask_root,
        )
    print(json.dumps({"status": "complete", "output_root": str(args.output_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Spot-check descriptor margins on visually structured probe subsets.

The script joins true-target calibration rows with the probe manifest/metadata,
computes simple image-structure heuristics on target crops, and reports
mean/std descriptor margin for high-anisotropy and high-periodicity subsets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze descriptor margin by structured tissue subsets.")
    parser.add_argument("--metadata", required=True, help="metadata_cross_{train,val}.json path.")
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--calibration-csv", required=True, help="descriptor_true_target_calibration.csv")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--tumor-label", type=int, default=1)
    parser.add_argument("--mask-source", choices=("target_tissue", "all"), default="target_tissue")
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.25,
        help="Top fraction used for high-anisotropy/high-periodicity buckets.",
    )
    parser.add_argument("--anisotropy-threshold", type=float, default=None)
    parser.add_argument("--periodicity-threshold", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = read_metadata(Path(args.metadata))
    manifest = json.loads(Path(args.selection_manifest).read_text(encoding="utf8"))
    calibration_rows = read_csv(Path(args.calibration_csv))
    probes = build_probe_lookup(manifest, records)
    feature_by_probe = {
        probe_index: compute_probe_features(
            probe,
            tumor_label=int(args.tumor_label),
            mask_source=args.mask_source,
        )
        for probe_index, probe in probes.items()
    }
    joined = join_calibration_rows(calibration_rows, feature_by_probe)
    if not joined:
        raise ValueError("No calibration rows joined to probe features.")

    top_fraction = min(1.0, max(0.0, float(args.top_fraction)))
    anisotropy_threshold = (
        float(args.anisotropy_threshold)
        if args.anisotropy_threshold is not None
        else quantile([row["anisotropy"] for row in joined], 1.0 - top_fraction)
    )
    periodicity_threshold = (
        float(args.periodicity_threshold)
        if args.periodicity_threshold is not None
        else quantile([row["periodicity_peak_ratio"] for row in joined], 1.0 - top_fraction)
    )
    for row in joined:
        row["high_anisotropy"] = row["anisotropy"] >= anisotropy_threshold
        row["high_periodicity"] = row["periodicity_peak_ratio"] >= periodicity_threshold
        row["structured_union"] = bool(row["high_anisotropy"] or row["high_periodicity"])

    summary = {
        "metadata": str(args.metadata),
        "selection_manifest": str(args.selection_manifest),
        "calibration_csv": str(args.calibration_csv),
        "mask_source": args.mask_source,
        "tumor_label": int(args.tumor_label),
        "top_fraction": top_fraction,
        "thresholds": {
            "anisotropy": anisotropy_threshold,
            "periodicity_peak_ratio": periodicity_threshold,
        },
        "overall": summarize_rows(joined),
        "subsets": {
            "high_anisotropy": summarize_rows([row for row in joined if row["high_anisotropy"]]),
            "not_high_anisotropy": summarize_rows([row for row in joined if not row["high_anisotropy"]]),
            "high_periodicity": summarize_rows([row for row in joined if row["high_periodicity"]]),
            "not_high_periodicity": summarize_rows([row for row in joined if not row["high_periodicity"]]),
            "structured_union": summarize_rows([row for row in joined if row["structured_union"]]),
            "structured_complement": summarize_rows([row for row in joined if not row["structured_union"]]),
        },
        "by_alternate_mode": summarize_by_key(joined, "alternate_mode"),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf8")
    if args.output_csv:
        write_csv(Path(args.output_csv), joined)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))
    return 0


def read_metadata(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf8"))
    if isinstance(payload, dict):
        pairs = payload.get("pairs")
        if isinstance(pairs, list):
            return pairs
        raise ValueError("metadata dict must contain a 'pairs' list")
    if isinstance(payload, list):
        return payload
    raise TypeError(f"unsupported metadata payload type: {type(payload)!r}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_probe_lookup(manifest: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_target_ref = {
        (path_key(row.get("target_image")), path_key(row.get("reference_image"))): row
        for row in records
        if row.get("target_image") and row.get("reference_image")
    }
    by_target = defaultdict(list)
    for row in records:
        if row.get("target_image"):
            by_target[path_key(row.get("target_image"))].append(row)

    probes = {}
    for probe_index, item in enumerate(manifest):
        target_key = path_key(item.get("target_image"))
        paired_ref_key = path_key(item.get("paired_reference_image"))
        record = by_target_ref.get((target_key, paired_ref_key))
        if record is None:
            candidates = by_target.get(target_key) or []
            if candidates:
                record = candidates[0]
        if record is None:
            raise ValueError(f"manifest target not found in metadata: {item.get('target_image')}")
        merged = dict(record)
        merged["probe_index"] = probe_index
        merged["manifest_sample_id"] = str(item.get("sample_id") or "")
        merged["target_image"] = str(item.get("target_image") or merged.get("target_image"))
        merged["reference_image"] = str(item.get("paired_reference_image") or merged.get("reference_image"))
        probes[probe_index] = merged
    return probes


def compute_probe_features(probe: dict[str, Any], *, tumor_label: int, mask_source: str) -> dict[str, Any]:
    image = np.asarray(Image.open(probe["target_image"]).convert("RGB"), dtype=np.float32) / 255.0
    gray = image.mean(axis=-1)
    if mask_source == "target_tissue":
        tissue = np.asarray(Image.open(probe["target_tissue_mask"]))
        mask = tissue == int(tumor_label)
    else:
        mask = np.ones(gray.shape, dtype=bool)
    if mask.shape != gray.shape:
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(gray.shape[::-1], Image.Resampling.NEAREST)) > 0
    if int(mask.sum()) < 8:
        mask = np.ones(gray.shape, dtype=bool)
    anisotropy, gradient_energy = structure_tensor_anisotropy(gray, mask)
    periodicity = spectral_peak_ratio(gray, mask)
    return {
        "probe_index": int(probe["probe_index"]),
        "sample_id": str(probe.get("sample_id") or probe.get("manifest_sample_id") or Path(probe["target_image"]).stem),
        "target_image": str(probe["target_image"]),
        "target_tissue_mask": str(probe["target_tissue_mask"]),
        "mask_fraction": float(mask.mean()),
        "anisotropy": anisotropy,
        "gradient_energy": gradient_energy,
        "periodicity_peak_ratio": periodicity,
    }


def structure_tensor_anisotropy(gray: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    gy, gx = np.gradient(gray.astype(np.float32))
    valid = mask.astype(bool)
    if not np.any(valid):
        return math.nan, math.nan
    gxv = gx[valid]
    gyv = gy[valid]
    jxx = float(np.mean(gxv * gxv))
    jyy = float(np.mean(gyv * gyv))
    jxy = float(np.mean(gxv * gyv))
    denom = jxx + jyy
    if denom <= 1e-12:
        return 0.0, 0.0
    anisotropy = math.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / denom
    return float(anisotropy), float(denom)


def spectral_peak_ratio(gray: np.ndarray, mask: np.ndarray) -> float:
    values = gray.astype(np.float32).copy()
    if np.any(mask):
        mean = float(values[mask].mean())
    else:
        mean = float(values.mean())
    values = values - mean
    values = values * mask.astype(np.float32)
    height, width = values.shape
    window = np.outer(np.hanning(height), np.hanning(width)).astype(np.float32)
    spectrum = np.fft.rfft2(values * window)
    power = np.abs(spectrum) ** 2
    if power.size <= 1:
        return math.nan
    yy = np.fft.fftfreq(height)[:, None]
    xx = np.fft.rfftfreq(width)[None, :]
    radius = np.sqrt(xx * xx + yy * yy)
    band = radius > 0.04
    band[0, 0] = False
    band_power = power[band]
    if band_power.size == 0:
        return math.nan
    total = float(band_power.sum())
    if total <= 1e-12:
        return 0.0
    return float(band_power.max() / (total + 1e-12))


def join_calibration_rows(
    calibration_rows: list[dict[str, str]],
    feature_by_probe: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    joined = []
    for row in calibration_rows:
        try:
            probe_index = int(row.get("probe_index", ""))
        except ValueError:
            continue
        features = feature_by_probe.get(probe_index)
        if features is None:
            continue
        margin = parse_float(row.get("cosine_margin"))
        paired_win = str(row.get("paired_win", "")).lower() in {"1", "true", "yes"}
        merged = {
            **features,
            "alternate_mode": str(row.get("alternate_mode") or ""),
            "paired_reference_sample_id": str(row.get("paired_reference_sample_id") or ""),
            "alternate_reference_sample_id": str(row.get("alternate_reference_sample_id") or ""),
            "cosine_margin": margin,
            "paired_win": paired_win,
            "target_paired_cosine": parse_float(row.get("target_paired_cosine")),
            "target_alternate_cosine": parse_float(row.get("target_alternate_cosine")),
        }
        joined.append(merged)
    return joined


def summarize_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    out = {}
    for value in sorted({str(row.get(key) or "") for row in rows}):
        out[value] = summarize_rows([row for row in rows if str(row.get(key) or "") == value])
    return out


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    margins = [float(row["cosine_margin"]) for row in rows if math.isfinite(float(row["cosine_margin"]))]
    wins = [1.0 if bool(row["paired_win"]) else 0.0 for row in rows]
    return {
        "n": len(rows),
        "win_rate": finite_mean(wins),
        "mean_margin": finite_mean(margins),
        "std_margin": sample_std(margins),
        "mean_anisotropy": finite_mean([row["anisotropy"] for row in rows]),
        "mean_periodicity_peak_ratio": finite_mean([row["periodicity_peak_ratio"] for row in rows]),
        "mean_mask_fraction": finite_mean([row["mask_fraction"] for row in rows]),
    }


def parse_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def finite_mean(values: list[float] | Any) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else math.nan


def sample_std(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) <= 1:
        return math.nan
    mean = sum(finite) / len(finite)
    return math.sqrt(sum((value - mean) ** 2 for value in finite) / (len(finite) - 1))


def quantile(values: list[float], q: float) -> float:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return math.nan
    q = min(1.0, max(0.0, q))
    index = int(round(q * (len(finite) - 1)))
    return finite[index]


def path_key(value: Any) -> str:
    if value is None:
        return ""
    return str(Path(str(value).replace("\\", "/")).expanduser())


if __name__ == "__main__":
    raise SystemExit(main())

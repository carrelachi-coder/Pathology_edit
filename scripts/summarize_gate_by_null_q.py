"""Summarize Cross V1 generation-gate wins by regional null-query buckets.

For non-regional control runs, pass --bucket-source-metrics from a regional
run to reuse the same probe-level null_q buckets.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bucket generation-gate win rate by null_q.")
    parser.add_argument("--metrics", required=True, help="generation_gate_metrics.csv to score.")
    parser.add_argument(
        "--bucket-source-metrics",
        default=None,
        help=(
            "Optional generation_gate_metrics.csv used only for region_null_q/active_q/missing_q. "
            "Use this when scoring a non-regional control whose own region stats are NaN."
        ),
    )
    parser.add_argument(
        "--win-metric",
        default="target_tumor_l1",
        choices=("target_tumor_l1", "target_full_l1"),
        help="Lower is better; paired wins when paired < alternate_feature.",
    )
    parser.add_argument("--low-threshold", type=float, default=0.1)
    parser.add_argument("--high-threshold", type=float, default=0.3)
    parser.add_argument("--output-csv", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metric_rows = read_csv(Path(args.metrics))
    bucket_rows = (
        read_csv(Path(args.bucket_source_metrics))
        if args.bucket_source_metrics
        else metric_rows
    )
    bucket_stats = extract_bucket_stats(bucket_rows)
    probe_rows = build_probe_rows(metric_rows, bucket_stats, win_metric=args.win_metric)
    print_summary(
        probe_rows,
        low_threshold=float(args.low_threshold),
        high_threshold=float(args.high_threshold),
    )
    if args.output_csv:
        write_probe_rows(
            Path(args.output_csv),
            probe_rows,
            low_threshold=float(args.low_threshold),
            high_threshold=float(args.high_threshold),
        )
        print(f"wrote {args.output_csv}")
    return 0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf8", newline="") as handle:
        return list(csv.DictReader(handle))


def probe_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("sample_id") or ""),
        str(row.get("paired_reference_sample_id") or ""),
        str(row.get("alternate_reference_sample_id") or ""),
        str(row.get("alternate_mode") or ""),
        str(row.get("prompt_mode") or ""),
        scale_key(row.get("scale")),
    )


def scale_key(value: Any) -> str:
    parsed = parse_float(value)
    if math.isfinite(parsed):
        return f"{parsed:.12g}"
    return str(value or "")


def parse_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def extract_bucket_stats(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str, str, str], dict[str, float]]:
    stats = {}
    for row in rows:
        key = probe_key(row)
        if key in stats:
            continue
        stats[key] = {
            "region_null_q": parse_float(row.get("region_null_q")),
            "region_active_q": parse_float(row.get("region_active_q")),
            "region_missing_q": parse_float(row.get("region_missing_q")),
            "region_fallback_q": parse_float(row.get("region_fallback_q")),
            "region_allowed_pairs": parse_float(row.get("region_allowed_pairs")),
        }
    return stats


def build_probe_rows(
    rows: list[dict[str, str]],
    bucket_stats: dict[tuple[str, str, str, str, str, str], dict[str, float]],
    *,
    win_metric: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[probe_key(row)][str(row.get("variant") or "")] = row

    probe_rows = []
    for key, variants in sorted(grouped.items()):
        paired = variants.get("paired")
        alternate = variants.get("alternate_feature")
        if paired is None or alternate is None:
            continue
        paired_value = parse_float(paired.get(win_metric))
        alternate_value = parse_float(alternate.get(win_metric))
        if not math.isfinite(paired_value) or not math.isfinite(alternate_value):
            continue
        stats = bucket_stats.get(key, {})
        advantage = alternate_value - paired_value
        probe_rows.append(
            {
                "sample_id": key[0],
                "paired_reference_sample_id": key[1],
                "alternate_reference_sample_id": key[2],
                "alternate_mode": key[3],
                "prompt_mode": key[4],
                "scale": key[5],
                "region_null_q": stats.get("region_null_q", math.nan),
                "region_active_q": stats.get("region_active_q", math.nan),
                "region_missing_q": stats.get("region_missing_q", math.nan),
                "region_fallback_q": stats.get("region_fallback_q", math.nan),
                "region_allowed_pairs": stats.get("region_allowed_pairs", math.nan),
                "paired_metric": paired_value,
                "alternate_metric": alternate_value,
                "paired_advantage": advantage,
                "paired_win": advantage > 0.0,
            }
        )
    return probe_rows


def print_summary(
    rows: list[dict[str, Any]],
    *,
    low_threshold: float,
    high_threshold: float,
) -> None:
    print(f"effective probe rows = {len(rows)}")
    print()
    for name, items in bucket_items(rows, low_threshold=low_threshold, high_threshold=high_threshold):
        print(format_summary(name, items))


def bucket_items(
    rows: list[dict[str, Any]],
    *,
    low_threshold: float,
    high_threshold: float,
) -> list[tuple[str, list[dict[str, Any]]]]:
    buckets = [
        (f"null_q < {low_threshold:g}", []),
        (f"{low_threshold:g} <= null_q <= {high_threshold:g}", []),
        (f"null_q > {high_threshold:g}", []),
        ("null_q is NaN", []),
    ]
    for row in rows:
        null_q = float(row.get("region_null_q", math.nan))
        if not math.isfinite(null_q):
            buckets[3][1].append(row)
        elif null_q < low_threshold:
            buckets[0][1].append(row)
        elif null_q <= high_threshold:
            buckets[1][1].append(row)
        else:
            buckets[2][1].append(row)
    return buckets


def format_summary(name: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"{name}: n=0"
    wins = sum(1 for row in rows if bool(row["paired_win"]))
    return (
        f"{name}: n={len(rows)} wins={wins} win_rate={wins / len(rows):.3f} "
        f"mean_null_q={finite_mean(row['region_null_q'] for row in rows):.3f} "
        f"mean_active_q={finite_mean(row['region_active_q'] for row in rows):.3f} "
        f"mean_advantage={finite_mean(row['paired_advantage'] for row in rows):.6e}"
    )


def finite_mean(values: Any) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return math.nan
    return sum(finite) / len(finite)


def write_probe_rows(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    low_threshold: float,
    high_threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "paired_reference_sample_id",
        "alternate_reference_sample_id",
        "alternate_mode",
        "prompt_mode",
        "scale",
        "null_q_bucket",
        "region_null_q",
        "region_active_q",
        "region_missing_q",
        "region_fallback_q",
        "region_allowed_pairs",
        "paired_metric",
        "alternate_metric",
        "paired_advantage",
        "paired_win",
    ]
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["null_q_bucket"] = bucket_name(
                float(row.get("region_null_q", math.nan)),
                low_threshold=low_threshold,
                high_threshold=high_threshold,
            )
            writer.writerow(output)


def bucket_name(value: float, *, low_threshold: float, high_threshold: float) -> str:
    if not math.isfinite(value):
        return "null_q is NaN"
    if value < low_threshold:
        return f"null_q < {low_threshold:g}"
    if value <= high_threshold:
        return f"{low_threshold:g} <= null_q <= {high_threshold:g}"
    return f"null_q > {high_threshold:g}"


if __name__ == "__main__":
    raise SystemExit(main())

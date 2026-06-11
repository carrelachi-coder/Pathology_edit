"""Analyze Cross V1 early-learning directionality.

This script measures two separate signals:

1. Loss-side gap trend from training logs:
   Reference health real_feature loss_gap over a step interval.
2. Generation-side directionality from generation gate metrics:
   paired wins when paired target error is lower than alternate_feature target
   error for the same sample, scale, alternate mode, and prompt mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


REFERENCE_HEALTH_RE = re.compile(
    r"Reference health step=(?P<step>\d+) variant=(?P<variant>[\w.]+): "
    r"pred_l2=(?P<pred_l2>[-+0-9.eE]+) "
    r"loss_gap=(?P<loss_gap>[-+0-9.eE]+)"
    r"(?:[±+/-]+(?P<stderr>[-+0-9.eE]+) n=(?P<n>\d+))? "
    r"first_double_ip_output_cos=(?P<cos>[-+0-9.eE]+)"
)
CHECKPOINT_STEP_RE = re.compile(r"checkpoint-(\d+)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Cross V1 learning trend signals.")
    parser.add_argument("--run-dir", default=None, help="CROSS_V1_OUTPUT_DIR. Defaults log to <run-dir>/logs/latest.log.")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--gap-variant", default="real_feature")
    parser.add_argument("--gap-start", type=int, default=2000)
    parser.add_argument("--gap-end", type=int, default=5000)
    parser.add_argument(
        "--generation-dir",
        action="append",
        default=[],
        help=(
            "Generation gate output dir. May be repeated. Use STEP:PATH to force "
            "the checkpoint step; otherwise parsed from generation_gate_summary.json."
        ),
    )
    parser.add_argument(
        "--win-metric",
        default="target_tumor_l1",
        choices=("target_tumor_l1", "target_full_l1"),
        help="Lower is better; paired wins when paired < alternate_feature.",
    )
    parser.add_argument(
        "--group-by",
        default="combo",
        choices=("combo", "mode", "prompt", "none"),
        help="How to group generation metrics before summarizing directionality.",
    )
    parser.add_argument(
        "--paired-win-threshold",
        type=float,
        default=0.5,
        help="Null win rate used for z-score sanity checks.",
    )
    parser.add_argument(
        "--same-class-threshold",
        type=float,
        default=0.60,
        help="Pre-registered same-dataset win-rate gate.",
    )
    parser.add_argument(
        "--cross-class-threshold",
        type=float,
        default=0.85,
        help="Pre-registered different-dataset win-rate gate.",
    )
    parser.add_argument(
        "--write-comparisons-csv",
        default=None,
        help="Optional path for per-sample paired-vs-alternate comparison rows.",
    )
    parser.add_argument(
        "--write-probes-csv",
        default=None,
        help="Optional path for independent probe-pair rows after scale aggregation.",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=2000,
        help="Bootstrap iterations over independent probe pairs for generation SE.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260611,
        help="Seed for generation probe-pair bootstrap.",
    )
    parser.add_argument(
        "--permutation-iters",
        type=int,
        default=2000,
        help=(
            "Permutation/sign-flip iterations over independent probe pairs. "
            "A healthy null sanity check should return about 50%% wins and zero advantage."
        ),
    )
    parser.add_argument(
        "--permutation-seed",
        type=int,
        default=20260612,
        help="Seed for the paired-vs-alternate permutation sanity check.",
    )
    parser.add_argument("--output-json", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report: dict[str, Any] = {}

    log_file = resolve_log_file(args)
    if log_file is not None and log_file.exists():
        gap_points = parse_loss_gap_points(log_file, variant=args.gap_variant)
        gap_points = [
            point for point in gap_points
            if args.gap_start <= int(point["step"]) <= args.gap_end
        ]
        report["loss_gap_trend"] = summarize_gap_points(gap_points)
    else:
        report["loss_gap_trend"] = {
            "status": "missing",
            "note": f"log file not found: {log_file}",
        }

    generation_dirs = [parse_generation_dir(value) for value in args.generation_dir]
    if generation_dirs:
        report["generation_directionality"] = summarize_generation_dirs(
            generation_dirs,
            win_metric=args.win_metric,
            group_by=args.group_by,
            paired_win_threshold=float(args.paired_win_threshold),
            same_class_threshold=float(args.same_class_threshold),
            cross_class_threshold=float(args.cross_class_threshold),
            bootstrap_iters=int(args.bootstrap_iters),
            bootstrap_seed=int(args.bootstrap_seed),
            permutation_iters=int(args.permutation_iters),
            permutation_seed=int(args.permutation_seed),
        )
    else:
        report["generation_directionality"] = {
            "status": "missing",
            "note": "no --generation-dir provided",
        }

    text = format_report(report)
    print(text)
    if args.write_comparisons_csv:
        write_generation_comparisons(
            Path(args.write_comparisons_csv),
            report.get("generation_directionality", {}),
        )
    if args.write_probes_csv:
        write_generation_probes(
            Path(args.write_probes_csv),
            report.get("generation_directionality", {}),
        )
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    return 0


def resolve_log_file(args: argparse.Namespace) -> Path | None:
    if args.log_file:
        return Path(args.log_file)
    if args.run_dir:
        return Path(args.run_dir) / "logs" / "latest.log"
    return None


def parse_loss_gap_points(path: Path, *, variant: str) -> list[dict[str, float]]:
    points = []
    for line in path.read_text(encoding="utf8", errors="replace").splitlines():
        match = REFERENCE_HEALTH_RE.search(line)
        if not match or match.group("variant") != variant:
            continue
        points.append(
            {
                "step": int(match.group("step")),
                "pred_l2": parse_float(match.group("pred_l2")),
                "loss_gap": parse_float(match.group("loss_gap")),
                "stderr": parse_float(match.group("stderr")),
                "n": parse_float(match.group("n")),
                "first_ip_output_cos": parse_float(match.group("cos")),
            }
        )
    return points


def summarize_gap_points(points: list[dict[str, float]]) -> dict[str, Any]:
    gaps = [float(point["loss_gap"]) for point in points if math.isfinite(float(point["loss_gap"]))]
    steps = [float(point["step"]) for point in points if math.isfinite(float(point["loss_gap"]))]
    if not gaps:
        return {"status": "missing", "n": 0, "note": "no finite gap points"}
    positive = sum(1 for gap in gaps if gap > 0)
    mean = finite_mean(gaps)
    stderr = sample_stderr(gaps)
    slope_per_1k = linear_slope(steps, gaps) * 1000.0 if len(gaps) >= 2 else math.nan
    split = max(1, len(gaps) // 2)
    first_mean = finite_mean(gaps[:split])
    last_mean = finite_mean(gaps[split:])
    status = "PASS" if positive / len(gaps) > 0.5 and slope_per_1k > 0 else "WARN"
    strong = mean - 2.0 * stderr > 0 if math.isfinite(stderr) else False
    return {
        "status": "PASS_STRONG" if status == "PASS" and strong else status,
        "n": len(gaps),
        "step_start": int(min(steps)),
        "step_end": int(max(steps)),
        "mean": mean,
        "stderr": stderr,
        "mean_minus_2stderr": mean - 2.0 * stderr if math.isfinite(stderr) else math.nan,
        "positive": positive,
        "negative_or_zero": len(gaps) - positive,
        "win_rate_positive_gap": positive / len(gaps),
        "slope_per_1k_steps": slope_per_1k,
        "first_half_mean": first_mean,
        "second_half_mean": last_mean,
        "second_minus_first": last_mean - first_mean,
        "points": points,
    }


def parse_generation_dir(value: str) -> tuple[int | None, Path]:
    if ":" in value:
        maybe_step, path = value.split(":", 1)
        if maybe_step.strip().isdigit():
            return int(maybe_step.strip()), Path(path)
    return None, Path(value)


def summarize_generation_dirs(
    generation_dirs: list[tuple[int | None, Path]],
    *,
    win_metric: str,
    group_by: str,
    paired_win_threshold: float,
    same_class_threshold: float,
    cross_class_threshold: float,
    bootstrap_iters: int,
    bootstrap_seed: int,
    permutation_iters: int,
    permutation_seed: int,
) -> dict[str, Any]:
    per_step = []
    group_series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for forced_step, path in generation_dirs:
        step = forced_step if forced_step is not None else infer_generation_step(path)
        metrics_path = path / "generation_gate_metrics.csv"
        if not metrics_path.exists():
            per_step.append(
                {
                    "step": step,
                    "path": str(path),
                    "status": "missing",
                    "note": f"missing {metrics_path.name}",
                }
            )
            continue
        rows = read_csv_rows(metrics_path)
        per_step.append(
            summarize_generation_step(
                step,
                path,
                rows,
                win_metric=win_metric,
                group_by=group_by,
                paired_win_threshold=paired_win_threshold,
                same_class_threshold=same_class_threshold,
                cross_class_threshold=cross_class_threshold,
                bootstrap_iters=bootstrap_iters,
                bootstrap_seed=bootstrap_seed + int(step or 0),
                permutation_iters=permutation_iters,
                permutation_seed=permutation_seed + int(step or 0),
            )
        )
        step_report = per_step[-1]
        for group, stats in (step_report.get("by_group") or {}).items():
            if step_report.get("step") is None:
                continue
            group_series[group].append(
                {
                    "step": float(step_report["step"]),
                    "win_rate": float(stats.get("win_rate", math.nan)),
                    "mean_paired_advantage": float(stats.get("mean_paired_advantage", math.nan)),
                }
            )
    valid_steps = [item for item in per_step if item.get("status") != "missing"]
    trend = summarize_generation_trend(valid_steps)
    trend_by_group = {
        group: summarize_generation_trend(items)
        for group, items in sorted(group_series.items())
    }
    return {
        "status": trend["status"],
        "win_metric": win_metric,
        "group_by": group_by,
        "paired_win_threshold": paired_win_threshold,
        "same_class_threshold": same_class_threshold,
        "cross_class_threshold": cross_class_threshold,
        "steps": per_step,
        "trend": trend,
        "trend_by_group": trend_by_group,
    }


def summarize_generation_step(
    step: int | None,
    path: Path,
    rows: list[dict[str, str]],
    *,
    win_metric: str,
    group_by: str,
    paired_win_threshold: float,
    same_class_threshold: float,
    cross_class_threshold: float,
    bootstrap_iters: int,
    bootstrap_seed: int,
    permutation_iters: int,
    permutation_seed: int,
) -> dict[str, Any]:
    grouped_rows = group_generation_rows(rows, group_by=group_by)
    comparisons = []
    for group_key, group_rows in sorted(grouped_rows.items()):
        by_pair: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
        for row in group_rows:
            key = (str(row.get("sample_id", "")), str(row.get("scale", "")))
            by_pair[key][str(row.get("variant", ""))] = row

        group_comparisons = []
        for (sample_id, scale), variants in sorted(by_pair.items()):
            paired = variants.get("paired")
            alternate = variants.get("alternate_feature")
            if not paired or not alternate:
                continue
            paired_value = parse_float(paired.get(win_metric))
            alternate_value = parse_float(alternate.get(win_metric))
            if not math.isfinite(paired_value) or not math.isfinite(alternate_value):
                continue
            advantage = alternate_value - paired_value
            group_comparisons.append(
                {
                    "step": step,
                    "path": str(path),
                    "group": group_key,
                    "sample_id": sample_id,
                    "scale": parse_float(scale),
                    "alternate_mode": str(paired.get("alternate_mode") or "same_dataset"),
                    "prompt_mode": str(paired.get("prompt_mode") or "dataset"),
                    "paired_reference_sample_id": str(paired.get("paired_reference_sample_id") or ""),
                    "alternate_reference_sample_id": str(paired.get("alternate_reference_sample_id") or ""),
                    "paired": paired_value,
                    "alternate_feature": alternate_value,
                    "paired_advantage": advantage,
                    "paired_win": advantage > 0.0,
                }
            )
        if group_comparisons:
            comparisons.extend(
                group_comparisons
            )

    if not comparisons:
        return {
            "step": step,
            "path": str(path),
            "status": "missing",
            "note": "no paired-vs-alternate comparisons found",
        }
    cluster_summary = summarize_clustered_comparisons(
        comparisons,
        paired_win_threshold=paired_win_threshold,
        threshold=0.5,
        bootstrap_iters=bootstrap_iters,
        bootstrap_seed=bootstrap_seed,
        permutation_iters=permutation_iters,
        permutation_seed=permutation_seed,
    )
    by_scale = {}
    for scale in sorted({float(item["scale"]) for item in comparisons if math.isfinite(float(item["scale"]))}):
        scale_items = [item for item in comparisons if float(item["scale"]) == scale]
        scale_summary = summarize_clustered_comparisons(
            scale_items,
            paired_win_threshold=paired_win_threshold,
            threshold=0.5,
            bootstrap_iters=bootstrap_iters,
            bootstrap_seed=bootstrap_seed + int(round(scale * 1000.0)),
            permutation_iters=permutation_iters,
            permutation_seed=permutation_seed + int(round(scale * 1000.0)),
        )
        by_scale[str(scale)] = {
            "n": scale_summary["n_observations"],
            "effective_probe_pairs": scale_summary["n_probe_pairs"],
            "win_rate": scale_summary["win_rate"],
            "mean_paired_advantage": scale_summary["mean_paired_advantage"],
            "stderr_paired_advantage": scale_summary["stderr_paired_advantage"],
            "win_rate_se": scale_summary["win_rate_se"],
            "win_rate_null_se": scale_summary["win_rate_null_se"],
            "win_rate_se_ratio_to_null": scale_summary["win_rate_se_ratio_to_null"],
            "win_rate_se_sanity_ok": scale_summary["win_rate_se_sanity_ok"],
            "win_rate_z_vs_threshold": scale_summary["win_rate_z_vs_0.5"],
        }
    by_group = {}
    for group in sorted({str(item.get("group", "")) for item in comparisons}):
        group_items = [item for item in comparisons if str(item.get("group", "")) == group]
        threshold = threshold_for_group(
            group,
            same_class_threshold=same_class_threshold,
            cross_class_threshold=cross_class_threshold,
        )
        group_summary = summarize_clustered_comparisons(
            group_items,
            paired_win_threshold=paired_win_threshold,
            threshold=threshold,
            bootstrap_iters=bootstrap_iters,
            bootstrap_seed=bootstrap_seed + stable_int_hash(group),
            permutation_iters=permutation_iters,
            permutation_seed=permutation_seed + stable_int_hash(group),
        )
        by_group[group] = {
            "n": group_summary["n_observations"],
            "effective_sample_ids": len({str(item.get("sample_id", "")) for item in group_items}),
            "effective_probe_pairs": group_summary["n_probe_pairs"],
            "paired_reference_ids": len({str(item.get("paired_reference_sample_id", "")) for item in group_items}),
            "alternate_reference_ids": len({str(item.get("alternate_reference_sample_id", "")) for item in group_items}),
            "win_rate": group_summary["win_rate"],
            "win_rate_se": group_summary["win_rate_se"],
            "win_rate_null_se": group_summary["win_rate_null_se"],
            "win_rate_se_ratio_to_null": group_summary["win_rate_se_ratio_to_null"],
            "win_rate_se_sanity_ok": group_summary["win_rate_se_sanity_ok"],
            "win_rate_z_vs_0.5": group_summary["win_rate_z_vs_0.5"],
            "pre_registered_threshold": threshold,
            "passes_pre_registered_threshold": group_summary["passes_pre_registered_threshold"],
            "mean_paired_advantage": group_summary["mean_paired_advantage"],
            "stderr_paired_advantage": group_summary["stderr_paired_advantage"],
            "permutation": group_summary["permutation"],
            "clustered_probe_rows": group_summary["probe_rows"],
        }
    status = (
        "PASS"
        if by_group
        and all(bool(stats.get("passes_pre_registered_threshold", False)) for stats in by_group.values())
        and float(cluster_summary["mean_paired_advantage"]) > 0
        else "WARN"
    )
    return {
        "step": step,
        "path": str(path),
        "status": status,
        "group_by": group_by,
        "n": cluster_summary["n_observations"],
        "effective_sample_ids": len({str(item.get("sample_id", "")) for item in comparisons}),
        "effective_probe_pairs": cluster_summary["n_probe_pairs"],
        "paired_reference_ids": len({str(item.get("paired_reference_sample_id", "")) for item in comparisons}),
        "alternate_reference_ids": len({str(item.get("alternate_reference_sample_id", "")) for item in comparisons}),
        "wins": cluster_summary["wins"],
        "losses_or_ties": cluster_summary["losses_or_ties"],
        "win_rate": cluster_summary["win_rate"],
        "win_rate_se": cluster_summary["win_rate_se"],
        "win_rate_null_se": cluster_summary["win_rate_null_se"],
        "win_rate_se_ratio_to_null": cluster_summary["win_rate_se_ratio_to_null"],
        "win_rate_se_sanity_ok": cluster_summary["win_rate_se_sanity_ok"],
        "win_rate_z_vs_0.5": cluster_summary["win_rate_z_vs_0.5"],
        "mean_paired_advantage": cluster_summary["mean_paired_advantage"],
        "stderr_paired_advantage": cluster_summary["stderr_paired_advantage"],
        "permutation": cluster_summary["permutation"],
        "by_scale": by_scale,
        "by_group": by_group,
        "comparisons": comparisons,
        "clustered_probe_rows": cluster_summary["probe_rows"],
    }


def summarize_clustered_comparisons(
    items: list[dict[str, Any]],
    *,
    paired_win_threshold: float,
    threshold: float,
    bootstrap_iters: int,
    bootstrap_seed: int,
    permutation_iters: int,
    permutation_seed: int,
) -> dict[str, Any]:
    probe_rows = build_probe_rows(items)
    win_rates = [float(row["probe_win_rate"]) for row in probe_rows]
    advantages = [float(row["probe_mean_advantage"]) for row in probe_rows]
    win_rate = finite_mean(win_rates)
    mean_advantage = finite_mean(advantages)
    n_probe_pairs = len(probe_rows)
    win_rate_se = bootstrap_stderr(
        win_rates,
        iters=bootstrap_iters,
        seed=bootstrap_seed,
    )
    advantage_se = bootstrap_stderr(
        advantages,
        iters=bootstrap_iters,
        seed=bootstrap_seed + 1009,
    )
    # z uses the honest independent probe-pair count, not raw scale rows.
    win_rate_null_se = binomial_null_se(
        paired_win_threshold,
        n_probe_pairs,
    )
    win_rate_se_ratio = (
        win_rate_se / win_rate_null_se
        if math.isfinite(win_rate_se)
        and math.isfinite(win_rate_null_se)
        and win_rate_null_se > 0
        else math.nan
    )
    se_sanity_checked = bool(
        math.isfinite(win_rate)
        and math.isfinite(win_rate_se_ratio)
        and abs(win_rate - paired_win_threshold) <= 0.15
    )
    se_sanity_ok = bool(
        (not se_sanity_checked)
        or (0.5 <= win_rate_se_ratio <= 2.0)
    )
    wins = sum(1 for row in probe_rows if float(row["probe_win_rate"]) > 0.5)
    losses_or_ties = n_probe_pairs - wins
    permutation = permutation_probe_test(
        probe_rows,
        iters=permutation_iters,
        seed=permutation_seed,
        paired_win_threshold=paired_win_threshold,
        observed_win_rate=win_rate,
        observed_mean_advantage=mean_advantage,
    )
    return {
        "n_observations": len(items),
        "n_probe_pairs": n_probe_pairs,
        "wins": wins,
        "losses_or_ties": losses_or_ties,
        "win_rate": win_rate,
        "win_rate_se": win_rate_se,
        "win_rate_null_se": win_rate_null_se,
        "win_rate_se_ratio_to_null": win_rate_se_ratio,
        "win_rate_se_sanity_checked": se_sanity_checked,
        "win_rate_se_sanity_ok": se_sanity_ok,
        "win_rate_z_vs_0.5": (
            (win_rate - paired_win_threshold) / win_rate_null_se
            if math.isfinite(win_rate) and math.isfinite(win_rate_null_se) and win_rate_null_se > 0
            else math.nan
        ),
        "mean_paired_advantage": mean_advantage,
        "stderr_paired_advantage": advantage_se,
        "passes_pre_registered_threshold": bool(math.isfinite(win_rate) and win_rate >= threshold),
        "permutation": permutation,
        "probe_rows": probe_rows,
    }


def build_probe_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        key = (
            str(item.get("group", "")),
            str(item.get("sample_id", "")),
            str(item.get("paired_reference_sample_id", "")),
            str(item.get("alternate_reference_sample_id", "")),
        )
        grouped[key].append(item)
    rows = []
    for (group, sample_id, paired_ref, alternate_ref), group_items in sorted(grouped.items()):
        wins = [1.0 if bool(item["paired_win"]) else 0.0 for item in group_items]
        advantages = [float(item["paired_advantage"]) for item in group_items]
        rows.append(
            {
                "group": group,
                "sample_id": sample_id,
                "paired_reference_sample_id": paired_ref,
                "alternate_reference_sample_id": alternate_ref,
                "n_observations": len(group_items),
                "probe_win_rate": finite_mean(wins),
                "probe_mean_advantage": finite_mean(advantages),
                "scales": ",".join(
                    format_float_for_csv(float(item["scale"]))
                    for item in sorted(group_items, key=lambda row: float(row["scale"]))
                ),
            }
        )
    return rows


def permutation_probe_test(
    probe_rows: list[dict[str, Any]],
    *,
    iters: int,
    seed: int,
    paired_win_threshold: float,
    observed_win_rate: float,
    observed_mean_advantage: float,
) -> dict[str, Any]:
    pairs = [
        (float(row["probe_win_rate"]), float(row["probe_mean_advantage"]))
        for row in probe_rows
        if math.isfinite(float(row["probe_win_rate"]))
        and math.isfinite(float(row["probe_mean_advantage"]))
    ]
    win_rates = [pair[0] for pair in pairs]
    advantages = [pair[1] for pair in pairs]
    if len(win_rates) <= 1:
        return {
            "status": "missing",
            "note": "need at least two finite independent probe rows",
        }
    iters = max(0, int(iters))
    if iters <= 0:
        return {
            "status": "disabled",
            "iters": 0,
        }
    rng = random.Random(int(seed))
    permuted_win_rates = []
    permuted_advantages = []
    n = len(win_rates)
    for _ in range(iters):
        win_total = 0.0
        advantage_total = 0.0
        for win_rate, advantage in zip(win_rates, advantages):
            if rng.randrange(2):
                win_total += 1.0 - win_rate
                advantage_total -= advantage
            else:
                win_total += win_rate
                advantage_total += advantage
        permuted_win_rates.append(win_total / n)
        permuted_advantages.append(advantage_total / n)

    observed_win_delta = (
        observed_win_rate - paired_win_threshold
        if math.isfinite(observed_win_rate)
        else math.nan
    )
    win_p = permutation_two_sided_pvalue(
        [value - paired_win_threshold for value in permuted_win_rates],
        observed_win_delta,
    )
    advantage_p = permutation_two_sided_pvalue(
        permuted_advantages,
        observed_mean_advantage,
    )
    null_win_mean = finite_mean(permuted_win_rates)
    null_advantage_mean = finite_mean(permuted_advantages)
    null_win_se = sample_std(permuted_win_rates)
    null_advantage_se = sample_std(permuted_advantages)
    sanity_ok = bool(
        math.isfinite(null_win_mean)
        and math.isfinite(null_win_se)
        and abs(null_win_mean - paired_win_threshold) <= max(0.02, 2.0 * null_win_se)
        and math.isfinite(null_advantage_mean)
        and math.isfinite(null_advantage_se)
        and abs(null_advantage_mean) <= max(1e-8, 2.0 * null_advantage_se)
    )
    return {
        "status": "ok" if sanity_ok else "WARN",
        "iters": iters,
        "null_win_rate_mean": null_win_mean,
        "null_win_rate_se": null_win_se,
        "null_win_rate_delta": null_win_mean - paired_win_threshold,
        "win_rate_two_sided_p": win_p,
        "null_advantage_mean": null_advantage_mean,
        "null_advantage_se": null_advantage_se,
        "advantage_two_sided_p": advantage_p,
        "sanity_ok": sanity_ok,
    }


def permutation_two_sided_pvalue(null_values: list[float], observed_value: float) -> float:
    finite = [abs(float(value)) for value in null_values if math.isfinite(float(value))]
    if not finite or not math.isfinite(observed_value):
        return math.nan
    observed_abs = abs(float(observed_value))
    exceed = sum(1 for value in finite if value >= observed_abs)
    return float((exceed + 1) / (len(finite) + 1))


def group_generation_rows(rows: list[dict[str, str]], *, group_by: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if group_by == "none":
            key = "all"
        elif group_by == "mode":
            key = str(row.get("alternate_mode") or "same_dataset")
        elif group_by == "prompt":
            key = str(row.get("prompt_mode") or "dataset")
        else:
            key = f"{row.get('alternate_mode') or 'same_dataset'}/{row.get('prompt_mode') or 'dataset'}"
        grouped[key].append(row)
    return grouped


def threshold_for_group(
    group: str,
    *,
    same_class_threshold: float,
    cross_class_threshold: float,
) -> float:
    if "different_dataset" in group:
        return cross_class_threshold
    if "same_dataset" in group:
        return same_class_threshold
    return 0.5


def summarize_generation_trend(items: list[dict[str, Any]]) -> dict[str, Any]:
    stepped = [
        item for item in items
        if item.get("step") is not None
        and math.isfinite(float(item.get("win_rate", math.nan)))
    ]
    if not stepped:
        return {"status": "missing", "note": "no generation steps with inferred checkpoint step"}
    stepped = sorted(stepped, key=lambda item: int(item["step"]))
    steps = [float(item["step"]) for item in stepped]
    win_rates = [float(item["win_rate"]) for item in stepped]
    advantages = [float(item["mean_paired_advantage"]) for item in stepped]
    win_rate_slope = linear_slope(steps, win_rates) * 1000.0 if len(stepped) >= 2 else math.nan
    advantage_slope = linear_slope(steps, advantages) * 1000.0 if len(stepped) >= 2 else math.nan
    latest = stepped[-1]
    if len(stepped) == 1:
        status = "INFO"
    elif latest["win_rate"] > 0.5 and (win_rate_slope > 0 or advantage_slope > 0):
        status = "TREND_UP"
    else:
        status = "WARN"
    return {
        "status": status,
        "latest_step": latest["step"],
        "latest_win_rate": latest["win_rate"],
        "latest_mean_paired_advantage": latest["mean_paired_advantage"],
        "win_rate_slope_per_1k_steps": win_rate_slope,
        "advantage_slope_per_1k_steps": advantage_slope,
    }


def infer_generation_step(path: Path) -> int | None:
    summary_path = path / "generation_gate_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf8"))
            checkpoint = str(summary.get("checkpoint", ""))
            match = CHECKPOINT_STEP_RE.search(checkpoint)
            if match:
                return int(match.group(1))
        except (json.JSONDecodeError, OSError):
            pass
    match = CHECKPOINT_STEP_RE.search(str(path))
    if match:
        return int(match.group(1))
    return None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_generation_comparisons(path: Path, generation_report: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for item in generation_report.get("steps", []):
        rows.extend(item.get("comparisons", []))
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_generation_probes(path: Path, generation_report: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for item in generation_report.get("steps", []):
        for row in item.get("clustered_probe_rows", []):
            rows.append(dict(row, step=item.get("step"), path=item.get("path")))
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: str | float | int | None) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(sum(finite) / len(finite)) if finite else math.nan


def sample_stderr(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) <= 1:
        return math.nan
    mean = finite_mean(finite)
    variance = sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
    return math.sqrt(variance) / math.sqrt(len(finite))


def sample_std(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) <= 1:
        return math.nan
    mean = finite_mean(finite)
    variance = sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
    return math.sqrt(variance)


def binomial_se(rate: float, n: int) -> float:
    if n <= 0 or not math.isfinite(rate):
        return math.nan
    return math.sqrt(max(rate * (1.0 - rate), 0.0) / n)


def binomial_null_se(null_rate: float, n: int) -> float:
    if n <= 0 or not math.isfinite(null_rate):
        return math.nan
    return math.sqrt(max(null_rate * (1.0 - null_rate), 0.0) / n)


def binomial_z(rate: float, n: int, null_rate: float) -> float:
    if n <= 0 or not math.isfinite(rate):
        return math.nan
    denom = math.sqrt(max(null_rate * (1.0 - null_rate), 1e-12) / n)
    return (rate - null_rate) / denom


def bootstrap_stderr(values: list[float], *, iters: int, seed: int) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) <= 1:
        return math.nan
    iters = max(0, int(iters))
    if iters <= 0:
        return sample_stderr(finite)
    rng = random.Random(int(seed))
    n = len(finite)
    means = []
    for _ in range(iters):
        total = 0.0
        for _ in range(n):
            total += finite[rng.randrange(n)]
        means.append(total / n)
    # The bootstrap standard error is the standard deviation of the
    # resampled statistic distribution. Do not divide by sqrt(B) again.
    return sample_std(means)


def stable_int_hash(value: str) -> int:
    result = 0
    for character in value:
        result = (result * 131 + ord(character)) % 1_000_000_007
    return result


def format_float_for_csv(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:g}"


def linear_slope(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) <= 1:
        return math.nan
    x_values = [pair[0] for pair in pairs]
    y_values = [pair[1] for pair in pairs]
    x_mean = finite_mean(x_values)
    y_mean = finite_mean(y_values)
    denom = sum((x - x_mean) ** 2 for x in x_values)
    if denom <= 0:
        return math.nan
    return sum((x - x_mean) * (y - y_mean) for x, y in pairs) / denom


def format_report(report: dict[str, Any]) -> str:
    lines = ["Cross V1 learning trend report"]
    gap = report.get("loss_gap_trend", {})
    lines.append("Loss gap trend:")
    if gap.get("status") == "missing":
        lines.append(f"  missing - {gap.get('note', '')}")
    else:
        lines.append(
            "  "
            f"status={gap['status']} n={gap['n']} "
            f"mean={gap['mean']:.6e} se={gap['stderr']:.6e} "
            f"mean-2se={gap['mean_minus_2stderr']:.6e} "
            f"positive={gap['positive']}/{gap['n']} "
            f"win_rate={gap['win_rate_positive_gap']:.3f} "
            f"slope_per_1k={gap['slope_per_1k_steps']:.6e} "
            f"second-first={gap['second_minus_first']:.6e}"
        )

    generation = report.get("generation_directionality", {})
    lines.append("Generation directionality:")
    if generation.get("status") == "missing":
        lines.append(f"  missing - {generation.get('note', '')}")
    else:
        trend = generation.get("trend", {})
        lines.append(
            "  "
            f"status={trend.get('status')} latest_step={trend.get('latest_step')} "
            f"latest_win_rate={float(trend.get('latest_win_rate', math.nan)):.3f} "
            f"latest_advantage={float(trend.get('latest_mean_paired_advantage', math.nan)):.6e} "
            f"win_rate_slope_per_1k={float(trend.get('win_rate_slope_per_1k_steps', math.nan)):.6e} "
            f"advantage_slope_per_1k={float(trend.get('advantage_slope_per_1k_steps', math.nan)):.6e}"
        )
        for group, group_trend in sorted((generation.get("trend_by_group") or {}).items()):
            lines.append(
                "  "
                f"group={group} status={group_trend.get('status')} "
                f"latest_step={group_trend.get('latest_step')} "
                f"latest_win_rate={float(group_trend.get('latest_win_rate', math.nan)):.3f} "
                f"latest_advantage={float(group_trend.get('latest_mean_paired_advantage', math.nan)):.6e} "
                f"win_rate_slope_per_1k={float(group_trend.get('win_rate_slope_per_1k_steps', math.nan)):.6e} "
                f"advantage_slope_per_1k={float(group_trend.get('advantage_slope_per_1k_steps', math.nan)):.6e}"
            )
        if generation.get("steps"):
            for item in generation.get("steps", []):
                if item.get("status") == "missing":
                    lines.append(f"  step={item.get('step')} missing - {item.get('note')}")
                    continue
                by_group = item.get("by_group") or {}
                if by_group:
                    group_text = ", ".join(
                        (
                        f"{group}:obs={stats['n']} eff_probe={stats.get('effective_probe_pairs')} "
                        f"samples={stats.get('effective_sample_ids')} "
                        f"win={float(stats['win_rate']):.3f}±{float(stats.get('win_rate_se', math.nan)):.3f}boot "
                        f"null_se={float(stats.get('win_rate_null_se', math.nan)):.3f} "
                        f"se_ratio={float(stats.get('win_rate_se_ratio_to_null', math.nan)):.2f} "
                        f"z={float(stats.get('win_rate_z_vs_0.5', math.nan)):.2f} "
                        f"gate={float(stats.get('pre_registered_threshold', math.nan)):.2f} "
                        f"adv={float(stats['mean_paired_advantage']):.6e}±{float(stats.get('stderr_paired_advantage', math.nan)):.6e}boot"
                        f" perm={format_permutation_inline(stats.get('permutation', {}))}"
                        + ("" if stats.get("win_rate_se_sanity_ok", True) else " SE_BAD")
                        )
                        for group, stats in sorted(by_group.items())
                    )
                    lines.append(f"  step={item.get('step')} groups={group_text}")
                else:
                    lines.append(
                        "  "
                        f"step={item.get('step')} status={item.get('status')} "
                        f"wins={item.get('wins')}/{item.get('effective_probe_pairs')} "
                        f"obs={item.get('n')} samples={item.get('effective_sample_ids')} "
                        f"win_rate={float(item.get('win_rate', math.nan)):.3f}±{float(item.get('win_rate_se', math.nan)):.3f}boot "
                        f"null_se={float(item.get('win_rate_null_se', math.nan)):.3f} "
                        f"se_ratio={float(item.get('win_rate_se_ratio_to_null', math.nan)):.2f} "
                        f"z={float(item.get('win_rate_z_vs_0.5', math.nan)):.2f} "
                        f"mean_advantage={float(item.get('mean_paired_advantage', math.nan)):.6e}±{float(item.get('stderr_paired_advantage', math.nan)):.6e}boot"
                        f" perm={format_permutation_inline(item.get('permutation', {}))}"
                        + ("" if item.get("win_rate_se_sanity_ok", True) else " SE_BAD")
                    )
    return "\n".join(lines)


def format_permutation_inline(permutation: dict[str, Any]) -> str:
    if not permutation:
        return "missing"
    status = permutation.get("status")
    if status != "ok":
        note = permutation.get("note")
        return f"{status}:{note}" if note else str(status)
    return (
        f"ok win0={float(permutation.get('null_win_rate_mean', math.nan)):.3f}"
        f"±{float(permutation.get('null_win_rate_se', math.nan)):.3f}"
        f" adv0={float(permutation.get('null_advantage_mean', math.nan)):.2e}"
        f"±{float(permutation.get('null_advantage_se', math.nan)):.2e}"
    )


if __name__ == "__main__":
    raise SystemExit(main())

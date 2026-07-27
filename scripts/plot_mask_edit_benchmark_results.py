#!/usr/bin/env python3
"""Create publication-style mask-edit benchmark and agentic-loop figures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


MODE_COLORS = {
    "GT": "#F79256",
    "Instruction": "#7DCFB6",
}
MODE_EDGES = {"GT": "#B95E32", "Instruction": "#3D8E78"}
MODE_MARKERS = {"GT": "o", "Instruction": "D"}
STRENGTH_ORDER = ["mild", "moderate", "significant", "xlarge_deid"]
STRENGTH_LABELS = {
    "mild": "Mild",
    "moderate": "Moderate",
    "significant": "Significant",
    "xlarge_deid": "Extra-large",
}
PRIMITIVE_LABELS = {
    "adenoma_to_carcinoma": "Adenoma to Carcinoma",
    "benign_atrophy": "Benign Atrophy",
    "benign_to_gleason3": "Benign to Gleason 3",
    "gleason_downgrade_4to3": "Gleason Downgrade, 4 to 3",
    "gleason_upgrade_3to4": "Gleason Upgrade, 3 to 4",
    "gleason_upgrade_4to5": "Gleason Upgrade, 4 to 5",
    "grade_upgrade": "Grade Upgrade",
    "immune_infiltration_decrease": "Immune Infiltration Decrease",
    "intratumoral_immune_infiltration": "Intratumoral Immune Infiltration",
    "necrosis_appearance": "Necrosis Appearance",
    "necrosis_resolution": "Necrosis Resolution",
    "normal_to_adenomatous": "Normal to Adenomatous",
    "stroma_decrease": "Stroma Decrease",
    "stromal_desmoplasia": "Stromal Desmoplasia",
    "stromal_immune_infiltration": "Stromal Immune Infiltration",
    "treatment_dedifferentiation": "Treatment Dedifferentiation",
    "tumor_burden_decrease": "Tumor Burden Decrease",
    "tumor_burden_increase": "Tumor Burden Increase",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--instruction", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def _as_bool(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower().isin({"true", "1", "yes"})


def load_results(path: Path, display_mode: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame["display_mode"] = display_mode
    frame["completed"] = frame["status"].eq("completed")
    for column in ("class_ok", "direction_ok", "location_ok"):
        frame[column] = _as_bool(frame[column])
    magnitude_values = frame["magnitude_bucket_pass"]
    if "intended_magnitude_bucket_agreement" in frame:
        modern = frame["intended_magnitude_bucket_agreement"]
        normalized = modern.astype(str).str.strip().str.lower()
        has_modern_value = ~normalized.isin({"", "null", "none", "nan"})
        magnitude_values = modern.where(has_modern_value, magnitude_values)
    frame["magnitude_agreement"] = _as_bool(magnitude_values)
    frame["semantic_core"] = (
        frame["class_ok"] & frame["direction_ok"] & frame["location_ok"]
    )
    for column in (
        "on_target_transition_ratio",
        "off_target_change_ratio",
        "spatial_containment_ratio",
        "measured_area_fraction",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["off_target_preservation"] = 1.0 - frame["off_target_change_ratio"]
    return frame


def clustered_ci(
    frame: pd.DataFrame,
    values: pd.Series,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    valid = values.notna()
    data = pd.DataFrame(
        {
            "cluster": frame.loc[valid, "wsi_id"].replace("", np.nan),
            "value": values.loc[valid].astype(float),
            "fallback": frame.loc[valid, "sample_id"],
        }
    )
    data["cluster"] = data["cluster"].fillna(data["fallback"])
    grouped = data.groupby("cluster", sort=True)["value"].agg(["sum", "count"])
    estimate = float(data["value"].mean())
    if grouped.empty:
        return estimate, math.nan, math.nan
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    remaining = max(1, iterations)
    while remaining:
        batch = min(256, remaining)
        sampled = rng.integers(0, len(grouped), size=(batch, len(grouped)))
        draws.append(sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1))
        remaining -= batch
    boot = np.concatenate(draws)
    low, high = np.quantile(boot, [0.025, 0.975])
    return estimate, float(low), float(high)


def ordinal_summary(frame: pd.DataFrame) -> dict[str, float]:
    details: list[tuple[bool, int, int, int]] = []
    completed = frame.loc[frame["completed"]].copy()
    completed = completed.loc[completed["strength"].isin(STRENGTH_ORDER)]
    for _, group in completed.groupby("ordinal_group_id"):
        means = group.groupby("strength")["measured_area_fraction"].mean()
        values = [means[item] for item in STRENGTH_ORDER if item in means]
        if len(values) < 2:
            continue
        concordant = tied = reversed_pairs = 0
        for left_index, left in enumerate(values):
            for right in values[left_index + 1 :]:
                if right > left:
                    concordant += 1
                elif right == left:
                    tied += 1
                else:
                    reversed_pairs += 1
        nondecreasing = all(right >= left for left, right in zip(values, values[1:]))
        details.append((nondecreasing, concordant, tied, reversed_pairs))
    total_pairs = sum(item[1] + item[2] + item[3] for item in details)
    return {
        "Nondecreasing groups": np.mean([item[0] for item in details]),
        "Concordant pairs": sum(item[1] for item in details) / total_pairs,
        "Tied pairs": sum(item[2] for item in details) / total_pairs,
        "Reversed pairs": sum(item[3] for item in details) / total_pairs,
        "n_groups": float(len(details)),
        "n_pairs": float(total_pairs),
    }


def strength_response(frame: pd.DataFrame) -> dict[str, Any]:
    completed = frame.loc[
        frame["completed"] & frame["strength"].isin(STRENGTH_ORDER)
    ]
    table = completed.groupby(
        ["ordinal_group_id", "strength"]
    )["measured_area_fraction"].mean().unstack()
    table = table.dropna(subset=STRENGTH_ORDER)
    maxima = table[STRENGTH_ORDER].max(axis=1).replace(0, np.nan)
    normalized = table[STRENGTH_ORDER].div(maxima, axis=0)
    return {
        "n_groups": len(normalized),
        "median": normalized.median(axis=0).to_numpy(dtype=float),
        "q1": normalized.quantile(0.25, axis=0).to_numpy(dtype=float),
        "q3": normalized.quantile(0.75, axis=0).to_numpy(dtype=float),
    }


def create_overview(
    frames: dict[str, pd.DataFrame],
    output_dir: Path,
    *,
    iterations: int,
    seed: int,
) -> None:
    fig = plt.figure(figsize=(17.2, 6.8))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.24, 1.06, 0.98),
        left=0.09,
        right=0.985,
        top=0.94,
        bottom=0.15,
        wspace=0.34,
    )
    ax_performance = fig.add_subplot(grid[0, 0])
    ax_strength = fig.add_subplot(grid[0, 1])
    agentic_grid = grid[0, 2].subgridspec(
        2,
        1,
        height_ratios=(1.30, 0.78),
        hspace=0.62,
    )
    ax_curve = fig.add_subplot(agentic_grid[0, 0])
    ax_outcomes = fig.add_subplot(agentic_grid[1, 0])

    ax = ax_performance
    metrics = [
        ("Completion", "completed", False),
        ("Semantic core", "semantic_core", False),
        ("Class transition", "class_ok", False),
        ("Direction", "direction_ok", False),
        ("Location", "location_ok", False),
        ("On-target transition", "on_target_transition_ratio", True),
        ("Off-target preservation", "off_target_preservation", True),
        ("Spatial containment", "spatial_containment_ratio", True),
    ]
    y = np.arange(len(metrics))[::-1]
    for row_index, y_position in enumerate(y):
        if row_index % 2:
            ax.axhspan(
                y_position - 0.5,
                y_position + 0.5,
                color="#F6F7F6",
                zorder=0,
            )
    for boundary in np.arange(0.5, len(metrics) - 0.5, 1.0):
        ax.axhline(
            boundary,
            color="#D4D7D8",
            linestyle=(0, (2.2, 3.2)),
            linewidth=0.9,
            zorder=0,
        )
    offset_values = np.linspace(0.14, -0.14, len(frames))
    offsets = dict(zip(frames, offset_values))
    mode_statistics: dict[str, dict[str, list[float]]] = {}
    for mode_index, (mode, frame) in enumerate(frames.items()):
        estimates = []
        lows = []
        highs = []
        for metric_index, (_, field, completed_only) in enumerate(metrics):
            subset = frame.loc[frame["completed"]] if completed_only else frame
            estimate, low, high = clustered_ci(
                subset,
                subset[field].astype(float),
                iterations=iterations,
                seed=seed + 101 * mode_index + metric_index,
            )
            estimates.append(100 * estimate)
            lows.append(100 * low)
            highs.append(100 * high)
        mode_statistics[mode] = {
            "estimates": estimates,
            "lows": lows,
            "highs": highs,
        }

    for metric_index, y_position in enumerate(y):
        row_estimates = [
            mode_statistics[mode]["estimates"][metric_index] for mode in frames
        ]
        ax.plot(
            [min(row_estimates), max(row_estimates)],
            [y_position, y_position],
            color="#B7C0C1",
            linewidth=3.0,
            solid_capstyle="round",
            zorder=1,
        )

    for mode, frame in frames.items():
        estimates = mode_statistics[mode]["estimates"]
        lows = mode_statistics[mode]["lows"]
        highs = mode_statistics[mode]["highs"]
        positions = y + offsets[mode]
        ax.hlines(
            positions,
            lows,
            highs,
            color=MODE_COLORS[mode],
            linewidth=2.2,
            zorder=2,
        )
        ax.scatter(
            estimates,
            positions,
            s=52,
            marker=MODE_MARKERS[mode],
            color=MODE_COLORS[mode],
            edgecolor=MODE_EDGES[mode],
            linewidth=1.0,
            label=mode,
            zorder=3,
        )
    ax.set_yticks(y, [item[0] for item in metrics])
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_xlim(96, 100.5)
    ax.set_xticks([96, 97, 98, 99, 100])
    ax.set_xlabel("Rate or ratio (%)", fontweight="bold")
    ax.grid(axis="x", color="#E0E2E3", linewidth=0.75)
    ax.grid(axis="y", visible=False)
    ax.legend(
        frameon=False,
        loc="lower right",
        ncol=len(frames),
        bbox_to_anchor=(1.0, 1.015),
        handletextpad=0.4,
        columnspacing=1.0,
    )
    ax.set_title("A  Overall metrics", loc="left", fontweight="bold", pad=15)

    ax = ax_strength
    x = np.arange(len(STRENGTH_ORDER))
    response = {mode: strength_response(frame) for mode, frame in frames.items()}
    for mode, values in response.items():
        median = 100 * values["median"]
        q1 = 100 * values["q1"]
        q3 = 100 * values["q3"]
        ax.fill_between(x, q1, q3, color=MODE_COLORS[mode], alpha=0.10, linewidth=0)
        ax.plot(
            x,
            median,
            color=MODE_COLORS[mode],
            marker=MODE_MARKERS[mode],
            markersize=7.0,
            markeredgecolor=MODE_EDGES[mode],
            markeredgewidth=0.9,
            linewidth=2.5,
            label=mode,
            zorder=3,
        )
    ax.set_xticks(x, ["Mild", "Moderate", "Significant", "Extra-large"])
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel(
        "Within-reference response (% of group maximum)",
        fontweight="bold",
    )
    ax.set_xlabel("Instruction strength", fontweight="bold")
    ax.grid(color="#E0E2E3", linewidth=0.75)
    ax.set_title("B  Ordinal strength trajectory", loc="left", fontweight="bold")
    ax.legend(
        frameon=False,
        loc="lower right",
        ncol=1,
        handletextpad=0.5,
        borderaxespad=0.8,
    )

    _plot_agentic_panels(ax_curve, ax_outcomes, frames["Instruction"])
    save_figure(fig, output_dir / "mask_edit_benchmark_overview")


def create_primitive_comparison(
    frames: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    primitives = sorted(set().union(*(set(frame["primitive"]) for frame in frames.values())))
    labels = [PRIMITIVE_LABELS.get(item, item.replace("_", " ").title()) for item in primitives]
    metric_specs = [
        ("Completion", lambda frame: frame["completed"].mean(), (90, 101)),
        ("Semantic core", lambda frame: frame["semantic_core"].mean(), (88, 101)),
        (
            "Nondecreasing strength response",
            lambda frame: ordinal_summary(frame)["Nondecreasing groups"],
            (15, 101),
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 9.4), sharey=True)
    fig.subplots_adjust(left=0.23, right=0.98, top=0.86, bottom=0.11, wspace=0.18)
    y = np.arange(len(primitives))[::-1]
    for row_index in range(len(primitives)):
        if row_index % 2 == 0:
            for ax in axes:
                ax.axhspan(y[row_index] - 0.5, y[row_index] + 0.5, color="#F5F5F3", zorder=0)
    for ax, (title, metric, limits) in zip(axes, metric_specs):
        values: dict[str, list[float]] = {}
        for mode, frame in frames.items():
            values[mode] = [
                100 * metric(frame.loc[frame["primitive"].eq(primitive)])
                for primitive in primitives
            ]
        for row_index in range(len(primitives)):
            row_values = [values[mode][row_index] for mode in frames]
            ax.plot(
                [min(row_values), max(row_values)],
                [y[row_index], y[row_index]],
                color="#BFC3C5",
                linewidth=1.2,
                zorder=1,
            )
        for mode in frames:
            ax.scatter(
                values[mode],
                y,
                s=35,
                marker=MODE_MARKERS[mode],
                color=MODE_COLORS[mode],
                edgecolor="white",
                linewidth=0.7,
                label=mode,
                zorder=2,
            )
        ax.set_xlim(*limits)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
        ax.set_xlabel("Rate (%)")
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.65)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, labels)
    axes[0].tick_params(axis="y", labelsize=8.5)
    axes[0].legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=len(frames),
        handletextpad=0.4,
        columnspacing=1.0,
    )
    fig.suptitle(
        "Mask-Edit Performance by Semantic Primitive",
        x=0.20,
        ha="left",
        fontsize=15.5,
        fontweight="bold",
    )
    fig.text(
        0.20,
        0.925,
        "Horizontal ranges compare GT and Instruction. Strength uses monotonic ordering within the same reference.",
        ha="left",
        fontsize=9.5,
        color="#444444",
    )
    fig.text(
        0.20,
        0.035,
        "GT stromal-desmoplasia strength calibration remains under review.",
        ha="left",
        fontsize=8.8,
        color="#444444",
    )
    save_figure(fig, output_dir / "mask_edit_benchmark_by_primitive")


def _cumulative_success_rates(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    attempted = frame.loc[pd.to_numeric(frame["attempt_count"], errors="coerce").fillna(0) > 0]
    successes = np.zeros(10, dtype=float)
    for raw in attempted["cumulative_success_at_k"]:
        try:
            values = json.loads(raw)
        except (TypeError, ValueError):
            values = {}
        successes += np.array(
            [float(bool(values.get(str(k), values.get(k, False)))) for k in range(1, 11)]
        )
    return np.arange(1, 11), successes / len(attempted)


def _plot_agentic_panels(
    ax_curve: plt.Axes,
    ax_outcomes: plt.Axes,
    frame: pd.DataFrame,
) -> None:
    ks, rates = _cumulative_success_rates(frame)
    percentages = 100 * rates
    ax_curve.plot(
        ks,
        percentages,
        color=MODE_COLORS["Instruction"],
        linewidth=2.5,
        marker="o",
        markersize=7.0,
        markeredgecolor=MODE_EDGES["Instruction"],
        markeredgewidth=0.9,
        zorder=3,
    )
    ax_curve.fill_between(
        ks,
        percentages,
        100,
        color=MODE_COLORS["Instruction"],
        alpha=0.08,
        linewidth=0,
    )
    label_layout = {
        1: ((5, -15), "left"),
        2: ((12, -22), "left"),
        5: ((0, -21), "center"),
        10: ((-3, -21), "right"),
    }
    for k, value in zip(ks, percentages):
        if int(k) not in label_layout:
            continue
        offset, alignment = label_layout[int(k)]
        ax_curve.annotate(
            f"{value:.3f}%" if k > 1 else f"{value:.2f}%",
            (k, value),
            xytext=offset,
            textcoords="offset points",
            ha=alignment,
            fontsize=9.5,
            color="#303030",
        )
    gain = percentages[-1] - percentages[0]
    ax_curve.text(
        0.98,
        0.11,
        f"+{gain:.2f} percentage points\nfrom replanning",
        transform=ax_curve.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=MODE_EDGES["Instruction"],
    )
    ax_curve.set_xlim(0.7, 10.3)
    ax_curve.set_xticks([1, 2, 3, 5, 7, 10])
    ax_curve.set_xlabel("Maximum attempts (k)", fontweight="bold")
    ax_curve.set_ylim(97.0, 100.12)
    ax_curve.set_yticks([97.0, 98.0, 99.0, 100.0])
    ax_curve.set_ylabel("Cumulative success (%)", fontweight="bold")
    ax_curve.grid(color="#E0E2E3", linewidth=0.75)
    ax_curve.set_title("C  Cumulative success at k", loc="left", fontweight="bold")

    replanned = _as_bool(frame["replanned"])
    repaired = _as_bool(frame["repair_success"])
    n_replanned = int(replanned.sum())
    n_recovered = int((replanned & repaired).sum())
    n_terminal = n_replanned - n_recovered
    recovered_pct = 100 * n_recovered / n_replanned
    terminal_pct = 100 - recovered_pct

    ax_outcomes.barh(
        0,
        recovered_pct,
        color=MODE_COLORS["Instruction"],
        height=0.30,
        label="Recovered",
    )
    ax_outcomes.barh(
        0,
        terminal_pct,
        left=recovered_pct,
        color=MODE_COLORS["GT"],
        height=0.30,
        label="Terminal",
    )
    ax_outcomes.text(
        recovered_pct / 2,
        0,
        f"Recovered  {n_recovered}  ({recovered_pct:.1f}%)",
        ha="center",
        va="center",
        color="#23483F",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_outcomes.annotate(
        f"Terminal  {n_terminal} ({terminal_pct:.1f}%)",
        xy=(recovered_pct + terminal_pct / 2, 0.16),
        xytext=(95, 0.43),
        textcoords="data",
        arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": 0.8},
        ha="right",
        va="center",
        fontsize=9.5,
        color="#303030",
    )
    ax_outcomes.set_xlim(0, 100)
    ax_outcomes.set_xticks([0, 25, 50, 75, 100])
    ax_outcomes.set_xlabel("Replanned cohort (%)", fontweight="bold")
    ax_outcomes.set_yticks([0], [f"Replanned cohort\n(n={n_replanned})"])
    ax_outcomes.set_ylim(-0.58, 0.58)
    ax_outcomes.grid(axis="x", color="#E0E2E3", linewidth=0.75)
    ax_outcomes.grid(axis="y", visible=False)
    ax_outcomes.set_title("D  Repair outcomes", loc="left", fontweight="bold")


def save_figure(fig: plt.Figure, stem: Path) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 12,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "axes.labelsize": 13,
            "axes.labelweight": "bold",
            "axes.labelcolor": "#202020",
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "xtick.color": "#303030",
            "ytick.color": "#303030",
            "legend.fontsize": 11.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    frames = {
        "GT": load_results(args.gt, "GT"),
        "Instruction": load_results(args.instruction, "Instruction"),
    }
    create_overview(
        frames,
        args.output_dir,
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    create_primitive_comparison(frames, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

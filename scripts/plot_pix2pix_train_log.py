#!/usr/bin/env python3
"""Parse pix2pix torchrun text logs and plot loss/gamma training curves."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRIC_RE = re.compile(
    r"^epoch=(?P<epoch>\d+)\s+step=(?P<step>\d+)\s+"
    r"loss=(?P<loss>[-+\d.eE]+)\s+l1=(?P<l1>[-+\d.eE]+)\s+"
    r"perc=(?P<perc>[-+\d.eE]+)\s+elapsed=(?P<elapsed>[-+\d.eE]+)m"
)
GAMMA_RE = re.compile(
    r"\[gamma\]\s+cross_8\.gamma=(?P<gamma8>[-+\d.eE]+)\s+\|\s+"
    r"cross_16\.gamma=(?P<gamma16>[-+\d.eE]+)"
)
TEXTURE_RE = re.compile(
    r"\[texture-loss\]\s+gram=(?P<gram>[-+\d.eE]+)\s+"
    r"contextual=(?P<contextual>[-+\d.eE]+)\s+"
    r"norm_l1=(?P<norm_l1>[-+\d.eE]+)\s+"
    r"norm_content=(?P<norm_content>[-+\d.eE]+)\s+"
    r"norm_gram=(?P<norm_gram>[-+\d.eE]+)\s+"
    r"norm_contextual=(?P<norm_contextual>[-+\d.eE]+)"
)


def parse_log(path: Path) -> list[dict[str, float | int]]:
    records: list[dict[str, float | int]] = []
    pending: dict[str, float | int] | None = None

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        metric_match = METRIC_RE.search(line)
        if metric_match:
            if pending is not None:
                records.append(pending)
            pending = {
                "epoch": int(metric_match["epoch"]),
                "step": int(metric_match["step"]),
                "loss": float(metric_match["loss"]),
                "l1": float(metric_match["l1"]),
                "perc": float(metric_match["perc"]),
                "elapsed_min": float(metric_match["elapsed"]),
                "gamma8": np.nan,
                "gamma16": np.nan,
                "gram": np.nan,
                "contextual": np.nan,
                "norm_l1": np.nan,
                "norm_content": np.nan,
                "norm_gram": np.nan,
                "norm_contextual": np.nan,
            }
            continue

        gamma_match = GAMMA_RE.search(line)
        if gamma_match and pending is not None:
            pending["gamma8"] = float(gamma_match["gamma8"])
            pending["gamma16"] = float(gamma_match["gamma16"])
            continue

        texture_match = TEXTURE_RE.search(line)
        if texture_match and pending is not None:
            for key in (
                "gram",
                "contextual",
                "norm_l1",
                "norm_content",
                "norm_gram",
                "norm_contextual",
            ):
                pending[key] = float(texture_match[key])

    if pending is not None:
        records.append(pending)

    if not records:
        raise ValueError(f"No training metrics found in {path}")

    # If a log contains duplicate steps after a restart, keep the latest entry.
    deduplicated = {int(record["step"]): record for record in records}
    return [deduplicated[step] for step in sorted(deduplicated)]


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    result = np.full(values.shape, np.nan, dtype=float)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result[index] = np.nanmean(values[start : index + 1])
    return result


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
        }
    )


def plot_summary(records: list[dict[str, float | int]], output: Path, window: int) -> None:
    steps = np.asarray([record["step"] for record in records], dtype=float)
    gamma8 = np.asarray([record["gamma8"] for record in records], dtype=float)
    gamma16 = np.asarray([record["gamma16"] for record in records], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(10.5, 10.5), sharex=True)
    fig.suptitle("Pix2Pix Texture Transfer Training", fontsize=15, fontweight="bold", y=0.98)

    loss_panels = (
        ("loss", "Total loss", "#4C78A8"),
        ("l1", "L1 loss", "#E45756"),
        ("perc", "Perceptual loss", "#72B7B2"),
    )
    for index, (key, label, color) in enumerate(loss_panels):
        values = np.asarray([record[key] for record in records], dtype=float)
        axes[index].plot(
            steps,
            values,
            color=color,
            alpha=0.28,
            linewidth=1.2,
            marker="o",
            markersize=2.5,
            label="raw",
        )
        axes[index].plot(
            steps,
            moving_average(values, window),
            color=color,
            linewidth=2.3,
            label=f"moving avg ({window} points)",
        )
        axes[index].set_ylabel(label)
        axes[index].set_title(label, loc="left")

    axes[0].legend(frameon=False, loc="upper right", ncol=2)

    gamma_axis = axes[3]
    gamma_axis.axhline(0.0, color="#777777", linewidth=0.9, alpha=0.65)
    gamma_axis.plot(steps, gamma8, color="#F58518", linewidth=2.1, marker="o", markersize=3, label="cross_8.gamma")
    gamma_axis.plot(steps, gamma16, color="#54A24B", linewidth=2.1, marker="o", markersize=3, label="cross_16.gamma")
    gamma_axis.set_xlabel("Training step")
    gamma_axis.set_ylabel("Gamma")
    gamma_axis.set_title("Learned gamma", loc="left")
    gamma_axis.legend(frameon=False, loc="best")

    fig.text(
        0.99,
        0.012,
        f"{len(records)} logged points · latest step {int(steps[-1]):,}",
        ha="right",
        va="bottom",
        color="#666666",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_components(records: list[dict[str, float | int]], output: Path, window: int) -> None:
    steps = np.asarray([record["step"] for record in records], dtype=float)
    candidate_metrics = (
        ("loss", "Total loss", "#4C78A8"),
        ("l1", "L1 loss", "#E45756"),
        ("perc", "Perceptual loss", "#72B7B2"),
        ("gram", "Regional Gram loss", "#F58518"),
        ("contextual", "Regional contextual loss", "#54A24B"),
    )
    metrics = tuple(
        metric
        for metric in candidate_metrics
        if not np.isnan(np.asarray([record.get(metric[0], np.nan) for record in records], dtype=float)).all()
    )

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10.5, 2.65 * len(metrics) + 0.5), sharex=True)
    axes = np.atleast_1d(axes)
    fig.suptitle("Pix2Pix Loss Components", fontsize=15, fontweight="bold", y=0.985)

    for axis, (key, title, color) in zip(axes, metrics):
        values = np.asarray([record[key] for record in records], dtype=float)
        axis.plot(steps, values, color=color, alpha=0.28, linewidth=1.1, marker="o", markersize=2.3)
        axis.plot(steps, moving_average(values, window), color=color, linewidth=2.2)
        axis.set_ylabel(title)
        axis.set_title(title, loc="left")

    axes[-1].set_xlabel("Training step")
    fig.tight_layout(rect=(0, 0.02, 1, 0.965))
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_csv(records: list[dict[str, float | int]], path: Path) -> None:
    fieldnames = [
        "epoch",
        "step",
        "loss",
        "l1",
        "perc",
        "gram",
        "contextual",
        "norm_l1",
        "norm_content",
        "norm_gram",
        "norm_contextual",
        "gamma8",
        "gamma16",
        "elapsed_min",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Path to train.log")
    parser.add_argument("--output-dir", type=Path, default=Path("training_curves"))
    parser.add_argument("--smooth-window", type=int, default=5, help="Moving-average window in logged points")
    args = parser.parse_args()

    if args.smooth_window < 1:
        parser.error("--smooth-window must be >= 1")

    configure_style()
    records = parse_log(args.log)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(records, args.output_dir / "metrics.csv")
    plot_summary(records, args.output_dir / "loss_gamma", args.smooth_window)
    plot_components(records, args.output_dir / "loss_components", args.smooth_window)

    latest = records[-1]
    print(
        f"Parsed {len(records)} records through step {latest['step']}; "
        f"latest loss={latest['loss']:.5f}, gamma8={latest['gamma8']:.5f}, "
        f"gamma16={latest['gamma16']:.5f}"
    )


if __name__ == "__main__":
    main()

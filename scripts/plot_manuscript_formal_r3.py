#!/usr/bin/env python3
"""Render the frozen formal R3 paired effects, ordering, and sham drift."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(analysis: dict, output: Path) -> dict:
    if analysis.get("schema_version") != "formal-r3-paired-analysis-v1":
        raise ValueError("unsupported formal R3 analysis schema")
    primary = analysis.get("primary_holm_family", [])
    order = analysis.get("order_outcomes", [])
    drift = analysis.get("sham_drift", [])
    if len(primary) != 4 or len(order) != 2 or len(drift) != 2:
        raise ValueError("formal R3 analysis is incomplete")
    output.mkdir(parents=True, exist_ok=False)

    primary_rows = []
    for row in primary:
        lo, hi = row["patient_bootstrap_95_interval"]
        primary_rows.append(
            {
                "endpoint": row["endpoint"],
                "edit_state": row["edit_state"],
                "requested_patients": row["requested_patients"],
                "valid_pairs": row["valid_pairs"],
                "mean_edit_minus_sham": row["mean_edit_minus_sham"],
                "ci95_low": lo,
                "ci95_high": hi,
                "holm_adjusted_p_value": row["holm_adjusted_p_value"],
                "positive_delta": row["positive_delta"],
                "zero_delta": row["zero_delta"],
                "negative_delta": row["negative_delta"],
            }
        )
    primary_csv = output / "formal_r3_primary_effects.csv"
    write_csv(primary_csv, primary_rows, list(primary_rows[0]))

    order_csv = output / "formal_r3_order_outcomes.csv"
    write_csv(order_csv, order, list(order[0]))
    drift_csv = output / "formal_r3_sham_drift.csv"
    write_csv(drift_csv, drift, list(drift[0]))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    labels = [row["edit_state"].replace("_", "\n") for row in primary]
    means = np.asarray([row["mean_edit_minus_sham"] for row in primary], float)
    cis = np.asarray([row["patient_bootstrap_95_interval"] for row in primary], float)
    x = np.arange(4)
    axes[0].errorbar(
        x,
        means,
        yerr=np.vstack([means - cis[:, 0], cis[:, 1] - means]),
        fmt="o",
        color="#274690",
        capsize=4,
    )
    axes[0].axhline(0, color="0.5", linestyle="--", linewidth=1)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Edit − sham tissue proportion")
    axes[0].set_title("A  Prespecified paired effects")

    x2 = np.arange(2)
    valid_rates = np.asarray([row["strict_order_rate_among_valid"] for row in order], float)
    axes[1].bar(x2, valid_rates, color=["#4C956C", "#F4A261"], width=0.55)
    for index, row in enumerate(order):
        axes[1].vlines(
            index,
            row["strict_order_all_requested_lower_bound"],
            row["strict_order_all_requested_upper_bound"],
            color="black",
            linewidth=2,
        )
        axes[1].text(
            index,
            min(1.04, valid_rates[index] + 0.04),
            f"{row['valid_triplets']}/{row['requested_patients']}",
            ha="center",
            fontsize=8,
        )
    axes[1].set_xticks(x2, [row["endpoint"].upper() for row in order])
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("Strict sham < moderate < significant")
    axes[1].set_title("B  Ordered response")

    drift_means = np.asarray([row["mean_sham_minus_original"] for row in drift], float)
    drift_ci = np.asarray([row["patient_bootstrap_95_interval"] for row in drift], float)
    axes[2].errorbar(
        x2,
        drift_means,
        yerr=np.vstack([drift_means - drift_ci[:, 0], drift_ci[:, 1] - drift_means]),
        fmt="o",
        color="#8E5572",
        capsize=4,
    )
    axes[2].axhline(0, color="0.5", linestyle="--", linewidth=1)
    axes[2].set_xticks(x2, [row["endpoint"].upper() for row in drift])
    axes[2].set_ylabel("Sham − original tissue proportion")
    axes[2].set_title("C  Reconstruction drift")
    fig.suptitle(
        f"Formal R3 · {analysis['requested_patients']} requested patients · patient-level intervals",
        fontsize=11,
    )
    fig.tight_layout()
    png = output / "formal_r3_results.png"
    pdf = output / "formal_r3_results.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {
        "primary_csv": identity(primary_csv),
        "order_csv": identity(order_csv),
        "sham_drift_csv": identity(drift_csv),
        "png": identity(png),
        "pdf": identity(pdf),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text())
    outputs = plot(analysis, args.output)
    manifest = {
        "schema_version": "formal-r3-figure-v1",
        "analysis": identity(args.analysis),
        "outputs": outputs,
        "values_recomputed_or_selected": False,
    }
    (args.output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

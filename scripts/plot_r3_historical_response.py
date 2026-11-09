"""Plot frozen patient-cluster historical R3 summaries from a complete report."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

METRICS = (
    ("strict_original_moderate_significant_order", "Strict ordering (%)", 100, None),
    ("mean_incremental_projection", "Moderate-to-Significant projection", 1, 0),
    ("displacement_norm_ratio", "Significant / Moderate displacement", 1, 1),
)


def extract(report):
    inference = report.get("inference", {})
    if report.get("status") != "complete" or report.get("schema_version") != 2:
        raise ValueError("complete corrected analysis required")
    if any(
        inference.get(k) != "patient_id"
        for k in ("bootstrap_unit", "permutation_unit", "correlation_unit")
    ):
        raise ValueError("patient-level inference required")
    rows = []
    for enc, label in [("uni2h", "UNI-2h"), ("conch", "CONCH")]:
        analysis = report["encoders"][enc]
        if (
            analysis["sample_count"] != 300
            or analysis["cluster_count"] != 90
            or analysis["cluster_unit"] != "patient_id"
        ):
            raise ValueError("frozen historical cohort mismatch")
        for primitive, operation in [("u1", "Tumor"), ("u2", "Immune")]:
            for backend, route in [("inpaint", "Local"), ("cross", "Global")]:
                values = analysis["primitives"][primitive]["backends"][backend]
                for metric, title, scale, null in METRICS:
                    s = values[metric]
                    if (
                        s["count"] != 90
                        or not all(
                            math.isfinite(s[k])
                            for k in ("mean", "ci95_low", "ci95_high")
                        )
                        or s["ci95_low"] > s["ci95_high"]
                    ):
                        raise ValueError("invalid or missing patient summary")
                    rows.append(
                        {
                            "encoder": enc,
                            "operation": primitive,
                            "backend": backend,
                            "label": f"{label} / {operation} / {route}",
                            "metric": metric,
                            "mean": s["mean"],
                            "ci95_low": s["ci95_low"],
                            "ci95_high": s["ci95_high"],
                            "patients": 90,
                            "patches": 300,
                        }
                    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    report = json.loads(raw)
    rows = extract(report)
    args.output.mkdir(parents=True, exist_ok=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12, 5.6), sharey=True)
    for axis, (metric, title, scale, null) in zip(axes, METRICS):
        selected = [r for r in rows if r["metric"] == metric]
        for i, row in enumerate(selected):
            color = "#b54735" if row["operation"] == "u1" else "#176a9b"
            lo, hi = row["ci95_low"] * scale, row["ci95_high"] * scale
            axis.plot([lo, hi], [i, i], color=color, lw=1.8)
            axis.scatter(row["mean"] * scale, i, color=color, s=30, zorder=3)
        axis.set_xlabel(title, fontsize=9)
        axis.set_yticks(range(8))
        axis.set_yticklabels([r["label"] for r in selected], fontsize=9)
        if null is not None:
            axis.axvline(null, color="#888888", ls="--", lw=0.8)
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    figure.suptitle("Historical representation response", fontsize=14)
    figure.text(
        0.5,
        0.02,
        "300 patches from 90 patients · Patient means and 95% patient-bootstrap intervals",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    figure.savefig(args.output / "r3_historical_response.pdf")
    figure.savefig(args.output / "r3_historical_response.png", dpi=250)
    plt.close(figure)
    with (args.output / "r3_historical_response.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "source_report_sha256": hashlib.sha256(raw).hexdigest(),
        "inference": report["inference"],
        "figure_scope": "historical computed representation response",
        "rows": len(rows),
    }
    (args.output / "figure_identity.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

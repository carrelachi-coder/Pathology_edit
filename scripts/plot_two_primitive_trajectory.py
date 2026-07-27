#!/usr/bin/env python3
"""Plot two controlled pathology-edit trajectories in a shared feature space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import proj3d
from scipy.stats import gaussian_kde

ENCODERS = (("uni2h", "UNI-2h"), ("conch", "CONCH"))
BACKENDS = (
    ("inpaint", "Local synthesis"),
    ("cross", "Reference-guided global synthesis"),
)
PRIMITIVES = ("u1", "u2")
STRENGTHS = ("moderate", "significant")
REFERENCE_COLOR = "#98A2B3"
BACKGROUND_COLOR = "#FBFCFE"
GRID_COLOR = "#D8DEE8"
COLORS = {
    ("u1", "moderate"): "#F59E0B",
    ("u1", "significant"): "#D94841",
    ("u2", "moderate"): "#36BFFA",
    ("u2", "significant"): "#1570EF",
}
MEAN_COLORS = {"u1": "#B42318", "u2": "#175CD3"}


def reference_neighborhoods(
    reference_points: np.ndarray,
    clusters: int,
    seed: int,
) -> np.ndarray:
    """Small deterministic k-means implementation for the 3D display only."""
    values = reference_points / np.maximum(reference_points.std(axis=0), 1e-8)
    best_labels: np.ndarray | None = None
    best_inertia = np.inf
    for restart in range(20):
        rng = np.random.default_rng(seed + restart * 1009)
        centers = [values[int(rng.integers(len(values)))]]
        while len(centers) < clusters:
            distance = np.min(
                np.stack(
                    [
                        np.sum((values - center) ** 2, axis=1)
                        for center in centers
                    ],
                    axis=1,
                ),
                axis=1,
            )
            total = float(distance.sum())
            if total <= 1e-12:
                candidates = [
                    index
                    for index in range(len(values))
                    if not any(np.array_equal(values[index], item) for item in centers)
                ]
                centers.append(values[candidates[0]])
            else:
                centers.append(values[int(rng.choice(len(values), p=distance / total))])
        centers_array = np.asarray(centers)
        for _ in range(100):
            squared = np.stack(
                [
                    np.sum((values - center) ** 2, axis=1)
                    for center in centers_array
                ],
                axis=1,
            )
            labels = np.argmin(squared, axis=1)
            updated = np.stack(
                [
                    values[labels == cluster].mean(axis=0)
                    if np.any(labels == cluster)
                    else centers_array[cluster]
                    for cluster in range(clusters)
                ]
            )
            if np.allclose(updated, centers_array, rtol=0.0, atol=1e-9):
                centers_array = updated
                break
            centers_array = updated
        squared = np.stack(
            [
                np.sum((values - center) ** 2, axis=1)
                for center in centers_array
            ],
            axis=1,
        )
        labels = np.argmin(squared, axis=1)
        inertia = float(squared[np.arange(len(values)), labels].sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
    if best_labels is None:
        raise RuntimeError("reference clustering failed")
    return best_labels


def bundled_paths(
    reference_points: np.ndarray,
    moderate_points: np.ndarray,
    significant_points: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "count": int(np.count_nonzero(labels == label)),
            "reference": reference_points[labels == label].mean(axis=0),
            "moderate": moderate_points[labels == label].mean(axis=0),
            "significant": significant_points[labels == label].mean(axis=0),
        }
        for label in sorted(np.unique(labels))
    ]


def representative_indices(
    labels: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    count = min(max(int(count), 1), len(labels))
    groups = np.unique(labels)
    exact = np.asarray(
        [(labels == group).sum() * count / len(labels) for group in groups]
    )
    allocation = np.maximum(np.floor(exact).astype(int), 1)
    while allocation.sum() > count:
        candidates = np.flatnonzero(allocation > 1)
        selected = candidates[
            np.argmin(exact[candidates] - allocation[candidates])
        ]
        allocation[selected] -= 1
    while allocation.sum() < count:
        capacity = (
            np.asarray([(labels == group).sum() for group in groups])
            - allocation
        )
        candidates = np.flatnonzero(capacity > 0)
        selected = candidates[
            np.argmax(exact[candidates] - allocation[candidates])
        ]
        allocation[selected] += 1
    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []
    for group, group_count in zip(groups, allocation):
        candidates = np.flatnonzero(labels == group)
        selected_indices.extend(
            rng.choice(
                candidates, size=int(group_count), replace=False
            ).tolist()
        )
    return np.asarray(sorted(selected_indices), dtype=int)


def draw_segment(
    axis: Any,
    start: np.ndarray,
    end: np.ndarray,
    color: str,
    linewidth: float,
    alpha: float,
    zorder: int,
) -> None:
    axis.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        [start[2], end[2]],
        color="white",
        linewidth=linewidth + 1.5,
        alpha=min(alpha + 0.08, 0.9),
        solid_capstyle="round",
        zorder=zorder - 1,
    )
    axis.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        [start[2], end[2]],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        solid_capstyle="round",
        zorder=zorder,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--static-points", type=int, default=36)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--dpi", type=int, default=320)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return vector / norm


def load_cache(root: Path, set_name: str) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(root / f"{set_name}.npz")
    return payload["sample_ids"].astype(str), payload["features"].astype(np.float64)


def shared_two_primitive_basis(
    reference: np.ndarray,
    values: dict[str, dict[str, dict[str, np.ndarray]]],
) -> np.ndarray:
    axes_by_primitive: dict[str, np.ndarray] = {}
    for primitive in PRIMITIVES:
        local_axis = unit(
            (values[primitive]["moderate"]["inpaint"] - reference).mean(axis=0)
        )
        cross_axis = unit(
            (values[primitive]["moderate"]["cross"] - reference).mean(axis=0)
        )
        axes_by_primitive[primitive] = unit(local_axis + cross_axis)
    u1_axis = axes_by_primitive["u1"]
    u2_axis = axes_by_primitive["u2"]
    common = unit(u1_axis + u2_axis)
    contrast = unit(u1_axis - u2_axis)
    if np.dot(u1_axis, contrast) < 0:
        contrast *= -1
    if abs(float(np.dot(common, contrast))) > 1e-8:
        raise RuntimeError("common and contrast axes should be orthogonal")
    centered_reference = reference - reference.mean(axis=0)
    residual_reference = centered_reference - (
        centered_reference @ np.stack((common, contrast), axis=1)
    ) @ np.stack((common, contrast), axis=0)
    _, _, residual_vh = np.linalg.svd(residual_reference, full_matrices=False)
    reference_variation = unit(residual_vh[0])
    pivot = int(np.argmax(np.abs(reference_variation)))
    if reference_variation[pivot] < 0:
        reference_variation *= -1
    basis = np.stack((common, contrast, reference_variation), axis=0)
    if not np.allclose(basis @ basis.T, np.eye(3), atol=1e-7, rtol=0.0):
        raise RuntimeError("display basis should be orthonormal")
    return basis


def limits_for(*arrays: np.ndarray) -> tuple[tuple[float, float], ...]:
    stacked = np.concatenate(arrays, axis=0)
    limits = []
    for axis in range(3):
        low, high = np.quantile(stacked[:, axis], [0.01, 0.99])
        span = max(float(high - low), 1e-6)
        limits.append((float(low - 0.035 * span), float(high + 0.035 * span)))
    return tuple(limits)


def load_encoder(
    run_root: Path,
    encoder: str,
    *,
    clusters: int,
    static_points: int,
    seed: int,
) -> dict[str, Any]:
    values: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    reference_ids: list[str] | None = None
    reference: np.ndarray | None = None
    for primitive in PRIMITIVES:
        values[primitive] = {}
        for strength in STRENGTHS:
            cell = f"{primitive}_{strength}"
            manifest_path = (
                run_root / "manifests" / f"{cell}_evaluation_manifest.jsonl"
            )
            rows = read_jsonl(manifest_path)
            expected_ids = [str(row["sample_id"]) for row in rows]
            current_reference_ids = [str(row["reference_id"]) for row in rows]
            cache_root = (
                run_root
                / "embeddings"
                / encoder
                / cell
                / "cache"
                / encoder
            )
            cell_values: dict[str, np.ndarray] = {}
            for set_name in ("reference", "inpaint", "cross"):
                ids, features = load_cache(cache_root, set_name)
                if list(ids) != expected_ids:
                    raise ValueError(f"{encoder}/{cell}/{set_name}: order mismatch")
                cell_values[set_name] = features
            values[primitive][strength] = cell_values
            if reference is None:
                reference = cell_values["reference"]
                reference_ids = current_reference_ids
            else:
                if current_reference_ids != reference_ids:
                    raise ValueError(f"{encoder}/{cell}: reference pairing mismatch")
                cosine = np.sum(
                    reference
                    / np.linalg.norm(reference, axis=1, keepdims=True)
                    * cell_values["reference"]
                    / np.linalg.norm(
                        cell_values["reference"], axis=1, keepdims=True
                    ),
                    axis=1,
                )
                if float(cosine.min()) < 0.999:
                    raise ValueError(f"{encoder}/{cell}: reference feature mismatch")
    assert reference is not None
    basis = shared_two_primitive_basis(reference, values)
    center = reference.mean(axis=0)
    projected_reference = (reference - center) @ basis.T
    raw_projected: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for primitive in PRIMITIVES:
        raw_projected[primitive] = {}
        for strength in STRENGTHS:
            raw_projected[primitive][strength] = {
                backend: (values[primitive][strength][backend] - center)
                @ basis.T
                for backend, _ in BACKENDS
            }
    pooled_significant_delta = np.concatenate(
        [
            raw_projected[primitive]["significant"][backend]
            - projected_reference
            for primitive in PRIMITIVES
            for backend, _ in BACKENDS
        ],
        axis=0,
    )
    median_significant_length = float(
        np.median(np.linalg.norm(pooled_significant_delta, axis=1))
    )
    if median_significant_length <= 1e-10:
        raise RuntimeError("zero edit displacements")
    displacement_magnification = 1.0
    projected: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for primitive in PRIMITIVES:
        projected[primitive] = {}
        for strength in STRENGTHS:
            projected[primitive][strength] = {}
            for backend, _ in BACKENDS:
                delta = (
                    raw_projected[primitive][strength][backend]
                    - projected_reference
                )
                projected[primitive][strength][backend] = (
                    projected_reference
                    + displacement_magnification * delta
                )

    centered_reference = reference - reference.mean(axis=0)
    u, singular_values, _ = np.linalg.svd(
        centered_reference, full_matrices=False
    )
    reference_scores = u[:, :3] * singular_values[:3]
    labels = reference_neighborhoods(reference_scores, clusters, seed)
    indices = representative_indices(labels, static_points, seed)
    paths = {
        backend: {
            primitive: bundled_paths(
                projected_reference,
                projected[primitive]["moderate"][backend],
                projected[primitive]["significant"][backend],
                labels,
            )
            for primitive in PRIMITIVES
        }
        for backend, _ in BACKENDS
    }
    limits = {
        backend: limits_for(
            projected_reference,
            *[
                projected[primitive][strength][backend]
                for primitive in PRIMITIVES
                for strength in STRENGTHS
            ],
        )
        for backend, _ in BACKENDS
    }
    return {
        "reference": projected_reference,
        "values": projected,
        "paths": paths,
        "indices": indices,
        "limits": limits,
        "basis": basis,
        "displacement_magnification": displacement_magnification,
    }


def style_axis(
    axis: Any,
    limits: tuple[tuple[float, float], ...],
) -> None:
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_zlim(*limits[2])
    axis.set_xlabel(
        "Shared edit direction",
        labelpad=8,
        fontsize=8.2,
        fontweight="bold",
    )
    axis.set_ylabel(
        "Tumor–immune contrast",
        labelpad=8,
        fontsize=8.2,
        fontweight="bold",
    )
    axis.set_zlabel(
        "Reference variation",
        labelpad=8,
        fontsize=8.2,
        fontweight="bold",
    )
    axis.set_box_aspect((1.0, 1.0, 1.0), zoom=1.05)
    axis.view_init(elev=23.0, azim=-61.0)
    axis.tick_params(axis="both", which="major", labelsize=6.8, colors="#475467")
    axis.grid(True)
    for coordinate in ("xaxis", "yaxis", "zaxis"):
        component = getattr(axis, coordinate)
        component.pane.set_facecolor(mpl.colors.to_rgba(BACKGROUND_COLOR, 0.96))
        component.pane.set_edgecolor(mpl.colors.to_rgba("#C8D0DC", 0.68))
        component._axinfo["grid"]["color"] = mpl.colors.to_rgba(
            GRID_COLOR, 0.56
        )
        component._axinfo["grid"]["linewidth"] = 0.55
    for spine in axis.spines.values():
        spine.set_color("#C8D0DC")
        spine.set_linewidth(0.7)


def density_rendering(
    points: np.ndarray,
    color: str,
    *,
    alpha_low: float,
    alpha_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    density = gaussian_kde(points.T)(points.T)
    order = np.argsort(density)
    ranks = np.empty(len(points), dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(points))
    # Compensate for overplotting: isolated points remain visible while dense
    # cores receive less per-point opacity and therefore do not bury paths.
    alpha = alpha_high - (alpha_high - alpha_low) * ranks**0.82
    rgba = np.tile(mpl.colors.to_rgba(color), (len(points), 1))
    rgba[:, 3] = alpha
    return order, rgba


def annotate_projected_point(
    axis: Any,
    point: np.ndarray,
    label: str,
    *,
    offset: tuple[float, float],
    color: str,
    fontsize: float,
) -> None:
    x_2d, y_2d, _ = proj3d.proj_transform(
        float(point[0]),
        float(point[1]),
        float(point[2]),
        axis.get_proj(),
    )
    annotation = axis.annotate(
        label,
        xy=(x_2d, y_2d),
        xycoords="data",
        xytext=offset,
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        zorder=30,
        annotation_clip=False,
    )
    annotation.set_path_effects(
        [path_effects.withStroke(linewidth=2.4, foreground="white")]
    )


def draw_panel(
    axis: Any,
    encoder: dict[str, Any],
    backend: str,
    metrics: dict[str, Any],
) -> None:
    reference = encoder["reference"]
    axis.set_facecolor(BACKGROUND_COLOR)
    style_axis(axis, encoder["limits"][backend])
    axis.scatter(
        reference[:, 0],
        reference[:, 1],
        reference[:, 2],
        s=4,
        c=REFERENCE_COLOR,
        alpha=0.075,
        edgecolors="none",
        depthshade=False,
        rasterized=True,
        zorder=1,
    )
    for primitive in PRIMITIVES:
        for strength, size, alpha_high in (
            ("moderate", 4.5, 0.14),
            ("significant", 5.5, 0.18),
        ):
            points = encoder["values"][primitive][strength][backend]
            order, rgba = density_rendering(
                points,
                COLORS[(primitive, strength)],
                alpha_low=0.018,
                alpha_high=alpha_high,
            )
            axis.scatter(
                points[order, 0],
                points[order, 1],
                points[order, 2],
                s=size,
                c=rgba[order],
                edgecolors="none",
                depthshade=False,
                rasterized=True,
                zorder=2 if strength == "moderate" else 3,
            )
    original_center = reference.mean(axis=0)
    axis.scatter(
        original_center[0],
        original_center[1],
        original_center[2],
        s=38,
        marker="o",
        c=REFERENCE_COLOR,
        edgecolors="white",
        linewidths=1.35,
        depthshade=False,
        zorder=12,
    )
    for primitive in PRIMITIVES:
        moderate_center = encoder["values"][primitive]["moderate"][backend].mean(
            axis=0
        )
        significant_center = encoder["values"][primitive][
            "significant"
        ][backend].mean(axis=0)
        draw_segment(
            axis,
            original_center,
            moderate_center,
            MEAN_COLORS[primitive],
            2.20,
            0.96,
            20,
        )
        draw_segment(
            axis,
            moderate_center,
            significant_center,
            MEAN_COLORS[primitive],
            2.40,
            0.98,
            21,
        )
        for strength, center in (
            ("moderate", moderate_center),
            ("significant", significant_center),
        ):
            axis.scatter(
                center[0],
                center[1],
                center[2],
                s=38 if strength == "moderate" else 46,
                marker="o",
                c=COLORS[(primitive, strength)],
                edgecolors="white",
                linewidths=1.45,
                depthshade=False,
                zorder=24,
            )
            label = "M" if strength == "moderate" else "S"
            offsets = {
                ("u1", "moderate"): (8, -10),
                ("u1", "significant"): (10, 7),
                ("u2", "moderate"): (-11, 8),
                ("u2", "significant"): (-12, -10),
            }
            annotate_projected_point(
                axis,
                center,
                label,
                offset=offsets[(primitive, strength)],
                color=MEAN_COLORS[primitive],
                fontsize=8.8,
            )
    annotate_projected_point(
        axis,
        original_center,
        "O",
        offset=(-12, -10),
        color="#667085",
        fontsize=8.6,
    )

    u1_order = (
        metrics["primitives"]["u1"]["backends"][backend][
            "strict_original_moderate_significant_order"
        ]["mean"]
        * 100
    )
    u2_order = (
        metrics["primitives"]["u2"]["backends"][backend][
            "strict_original_moderate_significant_order"
        ]["mean"]
        * 100
    )
    own_significant = (
        metrics["between_primitive"]["significant"]["backends"][backend][
            "both_primitives_prefer_own_axis_fraction"
        ]["mean"]
        * 100
    )
    axis.text2D(
        0.97,
        0.965,
        "WSI-held-out O < M < S\n"
        f"Tumor {u1_order:.1f}% · Immune {u2_order:.1f}%\n"
        f"Own-axis at Significant {own_significant:.1f}% · P < 5×10⁻⁵",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.45,
        fontweight="bold",
        color="#344054",
        linespacing=1.28,
        zorder=20,
    )


def main() -> int:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    encoder_data = {
        encoder: load_encoder(
            args.run_root,
            encoder,
            clusters=args.clusters,
            static_points=args.static_points,
            seed=args.seed + index * 10000,
        )
        for index, (encoder, _) in enumerate(ENCODERS)
    }
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelcolor": "#344054",
            "figure.facecolor": BACKGROUND_COLOR,
            "savefig.facecolor": BACKGROUND_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(12.8, 8.7), constrained_layout=False)
    positions = {
        (0, 0): [0.055, 0.545, 0.435, 0.335],
        (0, 1): [0.515, 0.545, 0.435, 0.335],
        (1, 0): [0.055, 0.125, 0.435, 0.335],
        (1, 1): [0.515, 0.125, 0.435, 0.335],
    }
    for row_index, (encoder, encoder_label) in enumerate(ENCODERS):
        for column_index, (backend, _) in enumerate(BACKENDS):
            axis = figure.add_axes(
                positions[(row_index, column_index)],
                projection="3d",
                computed_zorder=False,
            )
            draw_panel(
                axis,
                encoder_data[encoder],
                backend,
                report["encoders"][encoder],
            )
        figure.text(
            0.507,
            0.885 if row_index == 0 else 0.475,
            f"{encoder_label} paired-response space",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="#475467",
        )

    figure.text(
        0.507,
        0.988,
        "Representation-space response to controlled pathology edits",
        ha="center",
        va="top",
        fontsize=16.2,
        fontweight="bold",
        color="#101828",
    )
    for x, (_, title) in zip((0.275, 0.740), BACKENDS):
        figure.text(
            x,
            0.955,
            title,
            ha="center",
            va="top",
            fontsize=13.2,
            fontweight="bold",
            color="#101828",
        )
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=REFERENCE_COLOR,
            markersize=5.5,
            label="Original",
        ),
        *[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=COLORS[(primitive, strength)],
                markersize=5.5,
                label=(
                    "Tumor-burden increase"
                    if primitive == "u1"
                    else "Stromal immune infiltration"
                )
                + f" · {strength.title()}",
            )
            for primitive in PRIMITIVES
            for strength in STRENGTHS
        ],
    ]
    figure.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.507, 0.932),
        ncol=5,
        frameon=False,
        fontsize=8.2,
        handletextpad=0.45,
        columnspacing=1.35,
    )
    figure.text(
        0.507,
        0.006,
        "Absolute 3D coordinates; no edit displacement magnification. Cubic "
        "boxes change display aspect only.\nEach panel shows its central "
        "1st-99th percentile window; compare numeric ticks, not screen length.\n"
        "Density-compensated opacity reduces overplotting.",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="#667085",
        linespacing=1.25,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "uni2h_conch_two_primitive_trajectory"
    figure.savefig(
        stem.with_suffix(".png"),
        dpi=args.dpi,
        bbox_inches="tight",
        pad_inches=0.20,
    )
    figure.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.20,
    )
    figure.savefig(
        stem.with_suffix(".svg"),
        bbox_inches="tight",
        pad_inches=0.20,
    )
    plt.close(figure)
    metadata = {
        "status": "complete",
        "layout": "rows_are_encoders_columns_are_synthesis_methods",
        "basis": (
            "absolute three-dimensional feature projection; common axis is "
            "the unit sum of the two shared Moderate directions, contrast "
            "axis is their unit difference, and the third axis is the leading "
            "reference variation orthogonal to both; all coordinates and edit "
            "displacements are shown at true scale"
        ),
        "rendering": (
            "points are plotted at true coordinate scale in a cubic axis box; "
            "each synthesis panel uses separate 1st-to-99th percentile limits "
            "per display axis, so extreme points can fall outside the frame; "
            "within-group density-compensated opacity suppresses overplotting; "
            "only mean trajectories and centroid labels are foregrounded"
        ),
        "displacement_magnification": {
            encoder_label: encoder_data[encoder]["displacement_magnification"]
            for encoder, encoder_label in ENCODERS
        },
        "input_references_per_panel": 300,
        "input_generated_points_per_panel": 1200,
        "displayed_reference_bundles_per_primitive": 0,
        "report": str(args.report),
        "files": {
            suffix: str(stem.with_suffix(f".{suffix}"))
            for suffix in ("pdf", "png", "svg")
        },
    }
    (args.output_dir / "two_primitive_trajectory_figure_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

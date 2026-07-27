#!/usr/bin/env python3
"""Cross-fitted trajectory analysis for paired tumor and immune edits."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from phase3_mask_edit.benchmark.embedding_utility import (
    compute_embedding_dose_response_scores,
    summarize_scores,
)
from phase3_mask_edit.benchmark.pathokid import sha256_file


ENCODERS = ("uni2h", "conch")
PRIMITIVES = ("u1", "u2")
STRENGTHS = ("moderate", "significant")
BACKENDS = ("inpaint", "cross")
PRIMITIVE_NAMES = {
    "u1": "tumor_burden_increase",
    "u2": "stromal_immune_infiltration",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--permutation-repeats", type=int, default=20000)
    parser.add_argument("--high-overlap-quantile", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260724)
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
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("cannot normalize a zero or non-finite vector")
    return vector / norm


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
        raise ValueError("cannot normalize zero or non-finite rows")
    return values / norms[:, None]


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(unit(left), unit(right)))


def load_cache(root: Path, set_name: str) -> tuple[np.ndarray, np.ndarray]:
    path = root / f"{set_name}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = np.load(path)
    return payload["sample_ids"].astype(str), payload["features"].astype(np.float64)


def cluster_summary(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    return summarize_scores(
        values,
        groups,
        bootstrap_repeats=repeats,
        seed=seed,
    )


def wsi_sign_flip_pvalue(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    wsi_means = np.asarray(
        [values[groups == group].mean() for group in np.unique(groups)]
    )
    observed = float(wsi_means.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(
        (-1.0, 1.0), size=(repeats, len(wsi_means)), replace=True
    )
    null = np.mean(signs * wsi_means[None, :], axis=1)
    return float((np.count_nonzero(null >= observed) + 1) / (repeats + 1))


def spearman_summary(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if np.ptp(left) <= 1e-12 or np.ptp(right) <= 1e-12:
        return {
            "rho": None,
            "pvalue": None,
            "status": "undefined_constant_input",
        }
    result = spearmanr(left, right)
    return {
        "rho": float(result.statistic),
        "pvalue": float(result.pvalue),
        "status": "complete",
    }


def load_encoder(
    run_root: Path,
    encoder: str,
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], dict[str, list[dict]]]:
    features: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    manifests: dict[str, list[dict]] = {}
    canonical_reference_ids: list[str] | None = None
    canonical_groups: list[str] | None = None
    canonical_reference: np.ndarray | None = None
    for primitive in PRIMITIVES:
        features[primitive] = {}
        for strength in STRENGTHS:
            cell = f"{primitive}_{strength}"
            manifest_path = (
                run_root / "manifests" / f"{cell}_evaluation_manifest.jsonl"
            )
            rows = read_jsonl(manifest_path)
            manifests[cell] = rows
            expected_ids = [str(row["sample_id"]) for row in rows]
            reference_ids = [str(row["reference_id"]) for row in rows]
            groups = [str(row["wsi_id"]) for row in rows]
            cache_root = (
                run_root
                / "embeddings"
                / encoder
                / cell
                / "cache"
                / encoder
            )
            cell_features: dict[str, np.ndarray] = {}
            for set_name in ("reference", *BACKENDS):
                ids, values = load_cache(cache_root, set_name)
                if list(ids) != expected_ids:
                    raise ValueError(
                        f"{encoder}/{cell}/{set_name}: cache order differs "
                        "from manifest"
                    )
                cell_features[set_name] = values
            features[primitive][strength] = cell_features
            if canonical_reference_ids is None:
                canonical_reference_ids = reference_ids
                canonical_groups = groups
                canonical_reference = cell_features["reference"]
            else:
                if reference_ids != canonical_reference_ids:
                    raise ValueError(f"{cell}: reference order is not paired")
                if groups != canonical_groups:
                    raise ValueError(f"{cell}: WSI order is not paired")
                alignment = np.sum(
                    unit_rows(canonical_reference)
                    * unit_rows(cell_features["reference"]),
                    axis=1,
                )
                if float(alignment.min()) < 0.999:
                    raise ValueError(
                        f"{encoder}/{cell}: reference cache mismatch, "
                        f"minimum cosine={alignment.min()}"
                    )
    return features, manifests


def cross_fitted_axes(
    features: dict[str, dict[str, dict[str, np.ndarray]]],
    groups: np.ndarray,
) -> dict[str, np.ndarray]:
    count = len(groups)
    axes = {
        primitive: np.zeros(
            (count, features[primitive]["moderate"]["reference"].shape[1]),
            dtype=np.float64,
        )
        for primitive in PRIMITIVES
    }
    for held_out_group in np.unique(groups):
        train = groups != held_out_group
        test = ~train
        for primitive in PRIMITIVES:
            reference = features[primitive]["moderate"]["reference"]
            local_delta = features[primitive]["moderate"]["inpaint"] - reference
            cross_delta = features[primitive]["moderate"]["cross"] - reference
            local_axis = unit(local_delta[train].mean(axis=0))
            cross_axis = unit(cross_delta[train].mean(axis=0))
            axes[primitive][test] = unit(local_axis + cross_axis)
    return axes


def analyze_encoder(
    features: dict[str, dict[str, dict[str, np.ndarray]]],
    manifests: dict[str, list[dict]],
    *,
    bootstrap_repeats: int,
    permutation_repeats: int,
    high_overlap_quantile: float,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    canonical_rows = manifests["u1_moderate"]
    groups = np.asarray([str(row["wsi_id"]) for row in canonical_rows])
    axes = cross_fitted_axes(features, groups)
    axis_cosine = np.sum(axes["u1"] * axes["u2"], axis=1)
    output: dict[str, Any] = {
        "sample_count": int(len(groups)),
        "wsi_count": int(len(np.unique(groups))),
        "cross_fitted_mean_direction_cosine_u1_vs_u2": cluster_summary(
            axis_cosine,
            groups,
            repeats=bootstrap_repeats,
            seed=seed + 11,
        ),
        "primitives": {},
        "between_primitive": {},
    }
    rows_out: list[dict[str, Any]] = [
        {
            "reference_id": str(row["reference_id"]),
            "wsi_id": str(row["wsi_id"]),
            "cross_fitted_axis_cosine_u1_vs_u2": float(axis_cosine[index]),
        }
        for index, row in enumerate(canonical_rows)
    ]

    deltas: dict[str, dict[str, dict[str, np.ndarray]]] = {
        primitive: {strength: {} for strength in STRENGTHS}
        for primitive in PRIMITIVES
    }
    unit_deltas: dict[str, dict[str, dict[str, np.ndarray]]] = {
        primitive: {strength: {} for strength in STRENGTHS}
        for primitive in PRIMITIVES
    }
    for primitive_index, primitive in enumerate(PRIMITIVES):
        primitive_report: dict[str, Any] = {
            "primitive": PRIMITIVE_NAMES[primitive],
            "backends": {},
            "backend_mean_direction_agreement": {},
        }
        reference = features[primitive]["moderate"]["reference"]
        significant_reference = features[primitive]["significant"]["reference"]
        reference_alignment = np.sum(
            unit_rows(reference) * unit_rows(significant_reference), axis=1
        )
        primitive_report["reference_alignment"] = {
            "mean_cosine": float(reference_alignment.mean()),
            "minimum_cosine": float(reference_alignment.min()),
        }
        moderate_rows = manifests[f"{primitive}_moderate"]
        significant_rows = manifests[f"{primitive}_significant"]
        moderate_dose = np.asarray(
            [float(row["realized_dose_fraction"]) for row in moderate_rows]
        )
        significant_dose = np.asarray(
            [float(row["realized_dose_fraction"]) for row in significant_rows]
        )
        for strength in STRENGTHS:
            for backend in BACKENDS:
                deltas[primitive][strength][backend] = (
                    features[primitive][strength][backend] - reference
                )
                unit_deltas[primitive][strength][backend] = unit_rows(
                    deltas[primitive][strength][backend]
                )
        for strength in STRENGTHS:
            primitive_report["backend_mean_direction_agreement"][strength] = cosine(
                deltas[primitive][strength]["inpaint"].mean(axis=0),
                deltas[primitive][strength]["cross"].mean(axis=0),
            )
        for backend_index, backend in enumerate(BACKENDS):
            moderate = features[primitive]["moderate"][backend]
            significant = features[primitive]["significant"][backend]
            scores = compute_embedding_dose_response_scores(
                reference, moderate, significant, groups
            )
            moderate_projection = np.sum(
                deltas[primitive]["moderate"][backend] * axes[primitive],
                axis=1,
            )
            significant_projection = np.sum(
                deltas[primitive]["significant"][backend] * axes[primitive],
                axis=1,
            )
            incremental_projection = significant_projection - moderate_projection
            strict_order = (moderate_projection > 0) & (
                incremental_projection > 0
            )
            norm_change = (
                scores.significant_displacement_norm
                - scores.moderate_displacement_norm
            )
            prefix_seed = seed + primitive_index * 10000 + backend_index * 1000
            backend_report = {
                "strict_original_moderate_significant_order": cluster_summary(
                    strict_order.astype(float),
                    groups,
                    repeats=bootstrap_repeats,
                    seed=prefix_seed + 101,
                ),
                "positive_moderate_projection": cluster_summary(
                    (moderate_projection > 0).astype(float),
                    groups,
                    repeats=bootstrap_repeats,
                    seed=prefix_seed + 211,
                ),
                "positive_incremental_projection": cluster_summary(
                    (incremental_projection > 0).astype(float),
                    groups,
                    repeats=bootstrap_repeats,
                    seed=prefix_seed + 307,
                ),
                "mean_moderate_projection": cluster_summary(
                    moderate_projection,
                    groups,
                    repeats=bootstrap_repeats,
                    seed=prefix_seed + 401,
                ),
                "mean_incremental_projection": cluster_summary(
                    incremental_projection,
                    groups,
                    repeats=bootstrap_repeats,
                    seed=prefix_seed + 503,
                ),
                "incremental_projection_wsi_sign_flip_pvalue": (
                    wsi_sign_flip_pvalue(
                        incremental_projection,
                        groups,
                        repeats=permutation_repeats,
                        seed=prefix_seed + 601,
                    )
                ),
                "moderate_directional_consistency": cluster_summary(
                    scores.moderate_directional_consistency,
                    groups,
                    repeats=bootstrap_repeats,
                    seed=prefix_seed + 701,
                ),
                "significant_directional_consistency": cluster_summary(
                    scores.significant_directional_consistency,
                    groups,
                    repeats=bootstrap_repeats,
                    seed=prefix_seed + 809,
                ),
                "matched_cross_strength_agreement": cluster_summary(
                    scores.matched_cross_strength_agreement,
                    groups,
                    repeats=bootstrap_repeats,
                    seed=prefix_seed + 907,
                ),
                "displacement_norm_ratio": cluster_summary(
                    scores.displacement_norm_ratio,
                    groups,
                    repeats=bootstrap_repeats,
                    seed=prefix_seed + 1009,
                ),
                "increased_displacement_norm_fraction": cluster_summary(
                    (norm_change > 0).astype(float),
                    groups,
                    repeats=bootstrap_repeats,
                    seed=prefix_seed + 1103,
                ),
                "dose_correlations": {
                    "dose_increase_vs_incremental_projection": spearman_summary(
                        significant_dose - moderate_dose,
                        incremental_projection,
                    ),
                    "significant_dose_vs_displacement_norm": spearman_summary(
                        significant_dose,
                        scores.significant_displacement_norm,
                    ),
                },
            }
            primitive_report["backends"][backend] = backend_report
            for index, row in enumerate(rows_out):
                row.update(
                    {
                        f"{primitive}_{backend}_moderate_projection": float(
                            moderate_projection[index]
                        ),
                        f"{primitive}_{backend}_incremental_projection": float(
                            incremental_projection[index]
                        ),
                        f"{primitive}_{backend}_matched_strength_cosine": float(
                            scores.matched_cross_strength_agreement[index]
                        ),
                        f"{primitive}_{backend}_norm_ratio": float(
                            scores.displacement_norm_ratio[index]
                        ),
                    }
                )
        for strength_index, strength in enumerate(STRENGTHS):
            paired_backend_cosine = np.sum(
                unit_deltas[primitive][strength]["inpaint"]
                * unit_deltas[primitive][strength]["cross"],
                axis=1,
            )
            primitive_report.setdefault("paired_backend_agreement", {})[
                strength
            ] = cluster_summary(
                paired_backend_cosine,
                groups,
                repeats=bootstrap_repeats,
                seed=seed
                + primitive_index * 10000
                + strength_index * 1000
                + 1201,
            )
        output["primitives"][primitive] = primitive_report

    for strength_index, strength in enumerate(STRENGTHS):
        overlap = np.asarray(
            [
                float(row[f"cross_primitive_mask_iou_{strength}"])
                for row in canonical_rows
            ]
        )
        threshold = float(np.quantile(overlap, high_overlap_quantile))
        high_overlap = overlap >= threshold
        strength_report: dict[str, Any] = {
            "mask_iou": cluster_summary(
                overlap,
                groups,
                repeats=bootstrap_repeats,
                seed=seed + strength_index * 10000 + 1301,
            ),
            "high_overlap_sensitivity": {
                "quantile": high_overlap_quantile,
                "threshold": threshold,
                "sample_count": int(np.count_nonzero(high_overlap)),
                "wsi_count": int(len(np.unique(groups[high_overlap]))),
            },
            "backends": {},
        }
        for backend_index, backend in enumerate(BACKENDS):
            u1 = unit_deltas["u1"][strength][backend]
            u2 = unit_deltas["u2"][strength][backend]
            u1_on_u1 = np.sum(u1 * axes["u1"], axis=1)
            u1_on_u2 = np.sum(u1 * axes["u2"], axis=1)
            u2_on_u1 = np.sum(u2 * axes["u1"], axis=1)
            u2_on_u2 = np.sum(u2 * axes["u2"], axis=1)
            u1_margin = u1_on_u1 - u1_on_u2
            u2_margin = u2_on_u2 - u2_on_u1
            paired_discrimination = 0.5 * (u1_margin + u2_margin)
            paired_cosine = np.sum(u1 * u2, axis=1)
            own_preference = (u1_margin > 0) & (u2_margin > 0)
            prefix_seed = (
                seed + strength_index * 10000 + backend_index * 1000 + 1401
            )

            def selected_summary(
                values: np.ndarray, offset: int
            ) -> dict[str, float | int]:
                return cluster_summary(
                    values[high_overlap],
                    groups[high_overlap],
                    repeats=bootstrap_repeats,
                    seed=prefix_seed + offset,
                )

            backend_report = {
                "matched_reference_u1_u2_displacement_cosine": cluster_summary(
                    paired_cosine,
                    groups,
                    repeats=bootstrap_repeats,
                    seed=prefix_seed + 1,
                ),
                "cross_fitted_direction_matrix": {
                    "u1_displacement_on_u1_axis": cluster_summary(
                        u1_on_u1,
                        groups,
                        repeats=bootstrap_repeats,
                        seed=prefix_seed + 101,
                    ),
                    "u1_displacement_on_u2_axis": cluster_summary(
                        u1_on_u2,
                        groups,
                        repeats=bootstrap_repeats,
                        seed=prefix_seed + 211,
                    ),
                    "u2_displacement_on_u1_axis": cluster_summary(
                        u2_on_u1,
                        groups,
                        repeats=bootstrap_repeats,
                        seed=prefix_seed + 307,
                    ),
                    "u2_displacement_on_u2_axis": cluster_summary(
                        u2_on_u2,
                        groups,
                        repeats=bootstrap_repeats,
                        seed=prefix_seed + 401,
                    ),
                },
                "paired_own_axis_margin": cluster_summary(
                    paired_discrimination,
                    groups,
                    repeats=bootstrap_repeats,
                    seed=prefix_seed + 503,
                ),
                "both_primitives_prefer_own_axis_fraction": cluster_summary(
                    own_preference.astype(float),
                    groups,
                    repeats=bootstrap_repeats,
                    seed=prefix_seed + 601,
                ),
                "wsi_level_direction_label_permutation_pvalue": (
                    wsi_sign_flip_pvalue(
                        paired_discrimination,
                        groups,
                        repeats=permutation_repeats,
                        seed=prefix_seed + 701,
                    )
                ),
                "high_overlap_sensitivity": {
                    "matched_reference_u1_u2_displacement_cosine": (
                        selected_summary(paired_cosine, 809)
                    ),
                    "paired_own_axis_margin": selected_summary(
                        paired_discrimination, 907
                    ),
                    "both_primitives_prefer_own_axis_fraction": (
                        selected_summary(own_preference.astype(float), 1009)
                    ),
                    "wsi_level_direction_label_permutation_pvalue": (
                        wsi_sign_flip_pvalue(
                            paired_discrimination[high_overlap],
                            groups[high_overlap],
                            repeats=permutation_repeats,
                            seed=prefix_seed + 1103,
                        )
                    ),
                },
            }
            strength_report["backends"][backend] = backend_report
            for index, row in enumerate(rows_out):
                row.update(
                    {
                        f"{strength}_{backend}_mask_iou": float(overlap[index]),
                        f"{strength}_{backend}_u1_u2_cosine": float(
                            paired_cosine[index]
                        ),
                        f"{strength}_{backend}_u1_own_margin": float(
                            u1_margin[index]
                        ),
                        f"{strength}_{backend}_u2_own_margin": float(
                            u2_margin[index]
                        ),
                        f"{strength}_{backend}_paired_discrimination": float(
                            paired_discrimination[index]
                        ),
                    }
                )
        output["between_primitive"][strength] = strength_report
    return output, rows_out


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "analysis": "paired_two_primitive_wsi_cross_fitted_trajectory",
        "run_root": str(args.run_root),
        "inference": {
            "direction_fit": (
                "leave_one_wsi_out; within primitive, equal-weight unit mean "
                "of local and global Moderate mean directions"
            ),
            "bootstrap_unit": "wsi_id",
            "bootstrap_repeats": args.bootstrap_repeats,
            "direction_label_null": (
                "paired primitive-label swap implemented as WSI-level sign flip"
            ),
            "permutation_repeats": args.permutation_repeats,
            "high_overlap_quantile": args.high_overlap_quantile,
        },
        "encoders": {},
        "input_manifests": {},
    }
    for primitive in PRIMITIVES:
        for strength in STRENGTHS:
            cell = f"{primitive}_{strength}"
            path = args.run_root / "manifests" / f"{cell}_evaluation_manifest.jsonl"
            report["input_manifests"][cell] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }

    all_rows: dict[str, list[dict[str, Any]]] = {}
    for encoder_index, encoder in enumerate(ENCODERS):
        features, manifests = load_encoder(args.run_root, encoder)
        encoder_report, encoder_rows = analyze_encoder(
            features,
            manifests,
            bootstrap_repeats=args.bootstrap_repeats,
            permutation_repeats=args.permutation_repeats,
            high_overlap_quantile=args.high_overlap_quantile,
            seed=args.seed + encoder_index * 100000,
        )
        report["encoders"][encoder] = encoder_report
        all_rows[encoder] = encoder_rows

    for encoder, rows in all_rows.items():
        row_path = args.output_dir / f"{encoder}_two_primitive_trajectory_rows.csv"
        with row_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    report["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    report_path = args.output_dir / "two_primitive_trajectory_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

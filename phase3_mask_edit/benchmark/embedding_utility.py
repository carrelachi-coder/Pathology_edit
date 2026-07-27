"""Directional embedding-utility metrics for paired pathology edits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class EmbeddingUtilityScores:
    """Per-sample scores for two generators applied to the same target edit."""

    inpaint_directional_consistency: np.ndarray
    cross_directional_consistency: np.ndarray
    paired_backend_agreement: np.ndarray
    inpaint_displacement_norm: np.ndarray
    cross_displacement_norm: np.ndarray


@dataclass(frozen=True)
class EmbeddingDoseResponseScores:
    """Per-sample paired moderate-to-significant UNI displacement endpoints."""

    moderate_directional_consistency: np.ndarray
    significant_directional_consistency: np.ndarray
    directional_consistency_change: np.ndarray
    matched_cross_strength_agreement: np.ndarray
    significant_to_moderate_centroid_alignment: np.ndarray
    incremental_to_moderate_centroid_alignment: np.ndarray
    moderate_centroid_projection: np.ndarray
    significant_centroid_projection: np.ndarray
    incremental_centroid_projection: np.ndarray
    moderate_displacement_norm: np.ndarray
    significant_displacement_norm: np.ndarray
    displacement_norm_change: np.ndarray
    displacement_norm_ratio: np.ndarray


def normalize_displacements(
    generated: np.ndarray,
    reference: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract raw embeddings and return unit directions plus raw norms."""

    generated = np.asarray(generated, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if generated.shape != reference.shape or generated.ndim != 2:
        raise ValueError(
            "generated and reference embeddings must be same-shape rank-2 arrays"
        )
    displacement = generated - reference
    norms = np.linalg.norm(displacement, axis=1)
    if np.any(~np.isfinite(displacement)) or np.any(norms <= epsilon):
        raise ValueError("embedding displacement contains a non-finite or zero-norm row")
    return displacement / norms[:, None], norms


def leave_one_out_directional_consistency(
    directions: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Cosine of each unit direction with the other samples' mean direction."""

    directions = np.asarray(directions, dtype=np.float64)
    if directions.ndim != 2 or len(directions) < 3:
        raise ValueError("leave-one-out consistency requires at least three rows")
    row_norms = np.linalg.norm(directions, axis=1)
    if np.any(~np.isfinite(directions)) or np.any(row_norms <= epsilon):
        raise ValueError("directions contain a non-finite or zero-norm row")
    unit = directions / row_norms[:, None]
    leave_one_out_sum = unit.sum(axis=0, keepdims=True) - unit
    centroid_norms = np.linalg.norm(leave_one_out_sum, axis=1)
    if np.any(centroid_norms <= epsilon):
        raise ValueError("a leave-one-out mean direction has zero norm")
    centroids = leave_one_out_sum / centroid_norms[:, None]
    return np.sum(unit * centroids, axis=1)


def leave_group_out_centroids(
    directions: np.ndarray,
    groups: Sequence[str],
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Return a unit population direction excluding each row's whole group."""

    directions = np.asarray(directions, dtype=np.float64)
    groups = np.asarray([str(group) for group in groups], dtype=str)
    if directions.ndim != 2 or len(directions) != len(groups):
        raise ValueError("directions and groups must be aligned rank-2/row inputs")
    if len(np.unique(groups)) < 2:
        raise ValueError("leave-group-out centroids require at least two groups")
    row_norms = np.linalg.norm(directions, axis=1)
    if np.any(~np.isfinite(directions)) or np.any(row_norms <= epsilon):
        raise ValueError("directions contain a non-finite or zero-norm row")
    unit = directions / row_norms[:, None]
    total = unit.sum(axis=0)
    centroids = np.empty_like(unit)
    for group in np.unique(groups):
        selected = groups == group
        excluded_sum = total - unit[selected].sum(axis=0)
        norm = float(np.linalg.norm(excluded_sum))
        if norm <= epsilon:
            raise ValueError(f"leave-group-out centroid is zero for group {group!r}")
        centroids[selected] = excluded_sum / norm
    return centroids


def compute_embedding_dose_response_scores(
    reference: np.ndarray,
    moderate: np.ndarray,
    significant: np.ndarray,
    groups: Sequence[str],
    *,
    epsilon: float = 1e-12,
) -> EmbeddingDoseResponseScores:
    """Compute paired strength-response endpoints with group-held-out centroids."""

    reference = np.asarray(reference, dtype=np.float64)
    moderate = np.asarray(moderate, dtype=np.float64)
    significant = np.asarray(significant, dtype=np.float64)
    if not (
        reference.shape == moderate.shape == significant.shape
        and reference.ndim == 2
    ):
        raise ValueError(
            "reference, moderate, and significant embeddings must be aligned rank-2 arrays"
        )
    moderate_directions, moderate_norms = normalize_displacements(
        moderate, reference, epsilon=epsilon
    )
    significant_directions, significant_norms = normalize_displacements(
        significant, reference, epsilon=epsilon
    )
    incremental = significant - moderate
    incremental_norms = np.linalg.norm(incremental, axis=1)
    if np.any(~np.isfinite(incremental)) or np.any(incremental_norms <= epsilon):
        raise ValueError("moderate-to-significant displacement has a non-finite or zero row")
    incremental_directions = incremental / incremental_norms[:, None]
    centroids = leave_group_out_centroids(moderate_directions, groups, epsilon=epsilon)
    moderate_consistency = leave_one_out_directional_consistency(
        moderate_directions, epsilon=epsilon
    )
    significant_consistency = leave_one_out_directional_consistency(
        significant_directions, epsilon=epsilon
    )
    moderate_displacement = moderate - reference
    significant_displacement = significant - reference
    moderate_projection = np.sum(moderate_displacement * centroids, axis=1)
    significant_projection = np.sum(significant_displacement * centroids, axis=1)
    incremental_projection = np.sum(incremental * centroids, axis=1)
    return EmbeddingDoseResponseScores(
        moderate_directional_consistency=moderate_consistency,
        significant_directional_consistency=significant_consistency,
        directional_consistency_change=(
            significant_consistency - moderate_consistency
        ),
        matched_cross_strength_agreement=np.sum(
            moderate_directions * significant_directions, axis=1
        ),
        significant_to_moderate_centroid_alignment=np.sum(
            significant_directions * centroids, axis=1
        ),
        incremental_to_moderate_centroid_alignment=np.sum(
            incremental_directions * centroids, axis=1
        ),
        moderate_centroid_projection=moderate_projection,
        significant_centroid_projection=significant_projection,
        incremental_centroid_projection=incremental_projection,
        moderate_displacement_norm=moderate_norms,
        significant_displacement_norm=significant_norms,
        displacement_norm_change=significant_norms - moderate_norms,
        displacement_norm_ratio=significant_norms / moderate_norms,
    )


def compute_embedding_utility_scores(
    reference: np.ndarray,
    inpaint: np.ndarray,
    cross: np.ndarray,
) -> EmbeddingUtilityScores:
    """Compute the frozen exploratory UNI-2h utility endpoints."""

    inpaint_directions, inpaint_norms = normalize_displacements(inpaint, reference)
    cross_directions, cross_norms = normalize_displacements(cross, reference)
    if inpaint_directions.shape != cross_directions.shape:
        raise ValueError("inpaint and Cross feature matrices must align")
    return EmbeddingUtilityScores(
        inpaint_directional_consistency=leave_one_out_directional_consistency(
            inpaint_directions
        ),
        cross_directional_consistency=leave_one_out_directional_consistency(
            cross_directions
        ),
        paired_backend_agreement=np.sum(
            inpaint_directions * cross_directions, axis=1
        ),
        inpaint_displacement_norm=inpaint_norms,
        cross_displacement_norm=cross_norms,
    )


def cluster_bootstrap_mean(
    values: np.ndarray,
    groups: Sequence[str],
    *,
    repeats: int,
    seed: int,
) -> np.ndarray:
    """Bootstrap the sample mean by resampling whole WSI clusters."""

    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray([str(group) for group in groups], dtype=str)
    if values.ndim != 1 or len(values) != len(groups) or not len(values):
        raise ValueError("values and groups must be aligned non-empty vectors")
    if not np.isfinite(values).all():
        raise ValueError("values contain non-finite entries")
    if repeats <= 0:
        raise ValueError("bootstrap repeats must be positive")
    group_names = np.unique(groups)
    if len(group_names) < 2:
        raise ValueError("cluster bootstrap requires at least two WSI groups")
    indices = {name: np.flatnonzero(groups == name) for name in group_names}
    rng = np.random.default_rng(seed)
    bootstrapped = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sampled_groups = rng.choice(group_names, size=len(group_names), replace=True)
        sampled_rows = np.concatenate([indices[name] for name in sampled_groups])
        bootstrapped[repeat] = float(values[sampled_rows].mean())
    return bootstrapped


def summarize_scores(
    values: np.ndarray,
    groups: Sequence[str],
    *,
    bootstrap_repeats: int,
    seed: int,
) -> dict[str, float | int]:
    """Return descriptive statistics and a WSI-cluster bootstrap CI for the mean."""

    values = np.asarray(values, dtype=np.float64)
    bootstrap = cluster_bootstrap_mean(
        values,
        groups,
        repeats=bootstrap_repeats,
        seed=seed,
    )
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "bootstrap_repeats": int(bootstrap_repeats),
        "bootstrap_mean_std": float(bootstrap.std(ddof=1)),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
    }

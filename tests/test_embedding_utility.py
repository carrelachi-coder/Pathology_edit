import json

import numpy as np
import pytest

from phase3_mask_edit.benchmark.embedding_utility import (
    cluster_bootstrap_mean,
    compute_embedding_dose_response_scores,
    compute_embedding_utility_scores,
    leave_group_out_centroids,
    leave_one_out_directional_consistency,
    normalize_displacements,
    summarize_scores,
)
from phase3_mask_edit.cli.run_embedding_utility_dose_response import (
    main as run_dose_response,
)


def test_normalize_displacements_uses_raw_feature_differences():
    reference = np.asarray([[1.0, 2.0], [2.0, 4.0]])
    generated = np.asarray([[4.0, 6.0], [2.0, 9.0]])

    directions, norms = normalize_displacements(generated, reference)

    np.testing.assert_allclose(norms, [5.0, 5.0])
    np.testing.assert_allclose(directions, [[0.6, 0.8], [0.0, 1.0]])


def test_leave_one_out_consistency_excludes_the_scored_row():
    directions = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    scores = leave_one_out_directional_consistency(directions)

    np.testing.assert_allclose(scores[:2], np.sqrt(0.5))
    assert np.isclose(scores[2], 0.0)


def test_compute_scores_recovers_shared_and_opposing_backend_directions():
    reference = np.zeros((4, 3), dtype=np.float64)
    inpaint = np.asarray([[1.0, 0.0, 0.0]] * 4)
    cross = np.asarray([[2.0, 0.0, 0.0]] * 3 + [[-2.0, 0.0, 0.0]])

    scores = compute_embedding_utility_scores(reference, inpaint, cross)

    np.testing.assert_allclose(scores.inpaint_directional_consistency, 1.0)
    np.testing.assert_allclose(scores.paired_backend_agreement, [1.0, 1.0, 1.0, -1.0])
    np.testing.assert_allclose(scores.inpaint_displacement_norm, 1.0)
    np.testing.assert_allclose(scores.cross_displacement_norm, 2.0)


def test_zero_displacement_is_rejected():
    features = np.ones((3, 4), dtype=np.float64)

    with pytest.raises(ValueError, match="zero-norm"):
        normalize_displacements(features, features)


def test_cluster_bootstrap_is_deterministic_and_resamples_whole_groups():
    values = np.asarray([0.0, 2.0, 10.0, 12.0])
    groups = ["a", "a", "b", "b"]

    first = cluster_bootstrap_mean(values, groups, repeats=20, seed=7)
    second = cluster_bootstrap_mean(values, groups, repeats=20, seed=7)

    np.testing.assert_array_equal(first, second)
    assert set(first).issubset({1.0, 6.0, 11.0})


def test_score_summary_reports_cluster_bootstrap_interval():
    result = summarize_scores(
        np.asarray([0.1, 0.2, 0.8, 0.9]),
        ["a", "a", "b", "b"],
        bootstrap_repeats=100,
        seed=11,
    )

    assert result["count"] == 4
    assert np.isclose(result["mean"], 0.5)
    assert result["ci95_low"] <= result["mean"] <= result["ci95_high"]


def test_leave_group_out_centroids_excludes_the_whole_wsi():
    directions = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )

    centroids = leave_group_out_centroids(directions, ["a", "a", "b", "b"])

    np.testing.assert_allclose(centroids[:2], [[0.0, 1.0], [0.0, 1.0]])
    np.testing.assert_allclose(centroids[2:], [[1.0, 0.0], [1.0, 0.0]])


def test_dose_response_recovers_monotonic_shared_trajectory():
    reference = np.zeros((4, 3), dtype=np.float64)
    moderate = np.asarray([[1.0, 0.0, 0.0]] * 4)
    significant = np.asarray([[3.0, 0.0, 0.0]] * 4)

    scores = compute_embedding_dose_response_scores(
        reference,
        moderate,
        significant,
        ["a", "a", "b", "b"],
    )

    np.testing.assert_allclose(scores.matched_cross_strength_agreement, 1.0)
    np.testing.assert_allclose(
        scores.incremental_to_moderate_centroid_alignment, 1.0
    )
    np.testing.assert_allclose(scores.incremental_centroid_projection, 2.0)
    np.testing.assert_allclose(scores.displacement_norm_change, 2.0)
    np.testing.assert_allclose(scores.displacement_norm_ratio, 3.0)


def test_dose_response_cli_aligns_significant_rows_to_moderate_pairs(tmp_path):
    moderate_ids = np.asarray(["m0", "m1", "m2", "m3"])
    significant_ids = np.asarray(["s2", "s0", "s3", "s1"])
    pair_ids = ["m2", "m0", "m3", "m1"]
    groups = ["b", "a", "b", "a"]
    reference = np.asarray(
        [
            [10.0, 1.0, 1.0],
            [9.0, 2.0, 1.0],
            [8.0, 1.0, 2.0],
            [7.0, 2.0, 2.0],
        ]
    )
    moderate = reference + np.asarray([1.0, 0.0, 0.0])
    significant_reference = reference[[2, 0, 3, 1]]
    significant = significant_reference + np.asarray([3.0, 0.0, 0.0])
    moderate_cache = tmp_path / "moderate_cache"
    significant_cache = tmp_path / "significant_cache"
    moderate_cache.mkdir()
    significant_cache.mkdir()
    for name, values in (
        ("reference", reference),
        ("inpaint", moderate),
        ("cross", moderate),
    ):
        np.savez(
            moderate_cache / f"{name}.npz",
            sample_ids=moderate_ids,
            features=values,
        )
    for name, values in (
        ("reference", significant_reference),
        ("inpaint", significant),
        ("cross", significant),
    ):
        np.savez(
            significant_cache / f"{name}.npz",
            sample_ids=significant_ids,
            features=values,
        )
    moderate_manifest = tmp_path / "moderate.jsonl"
    moderate_manifest.write_text(
        "".join(
            json.dumps({"sample_id": sample_id, "wsi_id": "a"}) + "\n"
            for sample_id in moderate_ids
        ),
        encoding="utf-8",
    )
    significant_manifest = tmp_path / "significant.jsonl"
    significant_manifest.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "pair_id": pair_id,
                    "wsi_id": group,
                    "moderate_changed_area_fraction": 0.19,
                    "changed_area_fraction": 0.32,
                }
            )
            + "\n"
            for sample_id, pair_id, group in zip(
                significant_ids, pair_ids, groups, strict=True
            )
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "output"

    assert (
        run_dose_response(
            [
                "--moderate-manifest",
                str(moderate_manifest),
                "--moderate-cache-root",
                str(moderate_cache),
                "--significant-manifest",
                str(significant_manifest),
                "--significant-cache-root",
                str(significant_cache),
                "--output-root",
                str(output_root),
                "--expected-count",
                "4",
                "--bootstrap-repeats",
                "20",
            ]
        )
        == 0
    )
    report = json.loads(
        (output_root / "embedding_utility_dose_response_report.json").read_text()
    )
    assert np.isclose(
        report["backends"]["cross"]["metrics"][
            "incremental_centroid_projection"
        ]["mean"],
        2.0,
    )

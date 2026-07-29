from types import SimpleNamespace

import numpy as np

from inpaint_cells.generate import (
    choose_weighted_centers,
    quota_coverage_radius,
)
from inpaint_cells.sampling_policy import (
    retry_pool_target,
    valid_biological_tissue_mask,
)


def _retry_args():
    return SimpleNamespace(
        retry_candidate_multiplier=12.0,
        retry_candidate_floor=64,
        dense_retry_quota_threshold=20,
        dense_retry_occupancy_threshold=0.12,
        dense_retry_candidate_multiplier=24.0,
        dense_retry_candidate_floor=128,
    )


def test_retry_pool_expands_for_dense_components():
    ordinary = retry_pool_target(
        quota=5,
        component_area=10000,
        expected_nucleus_area=80,
        args=_retry_args(),
    )
    dense = retry_pool_target(
        quota=25,
        component_area=10000,
        expected_nucleus_area=80,
        args=_retry_args(),
    )

    assert ordinary == (64, False, 0.04)
    assert dense == (600, True, 0.2)


def test_valid_biological_tissue_excludes_background_and_skipped_labels():
    tissue = np.array([[0, 1, 2], [3, 4, 0]], dtype=np.uint8)

    allowed = valid_biological_tissue_mask(tissue, {2, 4})

    np.testing.assert_array_equal(
        allowed,
        np.array([[False, True, False], [True, False, False]]),
    )


def test_probnet_score_orders_the_full_retry_queue():
    candidates = [(0, 0), (0, 1), (0, 2), (0, 3)]
    probability = np.array([[0.2, 0.9, 0.4, 0.7]], dtype=np.float32)

    ranked = choose_weighted_centers(
        candidates,
        probability,
        target_count=len(candidates),
        gamma=1.5,
    )

    assert ranked == [(0, 1), (0, 3), (0, 2), (0, 0)]


def test_probnet_quota_prefix_defers_crowded_high_score_candidates():
    candidates = [(0, x) for x in range(10)]
    probability = np.array(
        [[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]],
        dtype=np.float32,
    )

    ranked = choose_weighted_centers(
        candidates,
        probability,
        target_count=len(candidates),
        gamma=1.5,
        coverage_count=3,
        coverage_radius=3.0,
    )

    assert ranked[:3] == [(0, 0), (0, 3), (0, 6)]
    assert ranked[3:] == [
        (0, 1),
        (0, 2),
        (0, 4),
        (0, 5),
        (0, 7),
        (0, 8),
        (0, 9),
    ]


def test_quota_coverage_radius_is_generic_area_per_count_spacing():
    radius = quota_coverage_radius(
        region_area=10000,
        quota=4,
        candidate_min_distance=8.0,
    )

    assert radius == 37.5

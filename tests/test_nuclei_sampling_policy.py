from types import SimpleNamespace

import numpy as np

from inpaint_cells.generate import (
    _require_complete_target_count,
    allocate_type_counts,
    choose_weighted_centers,
    choose_type_with_remaining_quota_at_center,
    compute_patch_adaptive_priors,
    count_retained_centers_by_type,
    fuse_density_head_with_tissue_prior,
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


def test_exact_target_count_rejects_silent_generation_shortfall():
    try:
        _require_complete_target_count(
            tissue_id=2,
            target_count=14,
            placed=13,
        )
    except RuntimeError as exc:
        assert "target=14" in str(exc)
        assert "placed=13" in str(exc)
        assert "shortfall=1" in str(exc)
    else:
        raise AssertionError("Expected a target-count shortfall to fail")


def test_exact_target_count_accepts_complete_quota():
    assert (
        _require_complete_target_count(
            tissue_id=2,
            target_count=14,
            placed=14,
        )
        == 0
    )


def test_density_head_and_tissue_prior_are_normalized_then_equally_weighted():
    fused, audit = fuse_density_head_with_tissue_prior(
        [6.0222, 1.5911, 3.9192, 0.0153, 0.0463],
        {
            101: 0.2019,
            102: 0.2485,
            103: 0.5289,
            104: 0.0035,
            105: 0.0171,
        },
        density_weight=0.5,
    )

    quota = allocate_type_counts(fused, 14)

    assert quota == {101: 5, 102: 3, 103: 6}
    assert audit["density_head_weight"] == 0.5
    assert abs(sum(fused.values()) - 1.0) < 1e-9


def test_density_head_can_fully_determine_type_quota():
    fused, audit = fuse_density_head_with_tissue_prior(
        [6.0222, 1.5911, 3.9192, 0.0153, 0.0463],
        {
            101: 0.2019,
            102: 0.2485,
            103: 0.5289,
            104: 0.0035,
            105: 0.0171,
        },
        density_weight=1.0,
    )

    assert allocate_type_counts(fused, 14) == {101: 7, 102: 2, 103: 5}
    assert audit["density_head_weight"] == 1.0


def test_generation_count_subtracts_retained_buffer_nuclei_by_centroid():
    nuclei = np.zeros((20, 20), dtype=np.uint8)
    nuclei[3:6, 3:6] = 1
    nuclei[10:13, 10:13] = 3
    region = np.zeros((20, 20), dtype=bool)
    region[1:9, 1:9] = True

    counts = count_retained_centers_by_type(nuclei, region)

    assert counts[101] == 1
    assert counts[103] == 0


def test_count_density_uses_full_pre_edit_source_tissue_not_deletion_mask():
    class FakeLibrary:
        def get_density(self, tissue_id):
            return 2.0

        def get_type_distribution(self, tissue_id):
            return {101: 1.0}

    source_tissue = np.full((100, 200), 2, dtype=np.uint8)
    target_tissue = np.full((100, 200), 2, dtype=np.uint8)
    source_nuclei = np.zeros_like(source_tissue)
    for index in range(20):
        row = 2 + (index // 10) * 40
        col = 2 + (index % 10) * 18
        source_nuclei[row : row + 3, col : col + 3] = 101
    deletion = np.ones_like(source_tissue, dtype=bool)
    generation = np.ones_like(source_tissue, dtype=bool)

    scales, _, audit = compute_patch_adaptive_priors(
        reference_nuclei_raw=source_nuclei,
        reference_tissue=source_tissue,
        density_exclusion_region=deletion,
        target_tissue=target_tissue,
        generation_region=generation,
        library=FakeLibrary(),
        dataset_name="BCSS",
    )

    tissue_audit = audit["tissues"]["2"]
    assert tissue_audit["density_reference_image"] == "pre_edit_source_patch"
    assert tissue_audit["density_reference_tissue_ids"] == [2]
    assert tissue_audit["density_reference_deletion_exclusion_applied"] is False
    assert tissue_audit["local_centroid_count"] == 20
    assert tissue_audit["reference_area_px"] == 20000
    assert tissue_audit["target_density_per_10k_px"] == 10.0
    assert scales[2] == 5.0


def test_absent_glas_grade_uses_target_prior_with_pre_edit_gland_calibration():
    class FakeLibrary:
        def get_density(self, tissue_id):
            return {11: 2.0, 12: 4.0, 13: 5.0}.get(int(tissue_id), 2.0)

        def get_type_distribution(self, tissue_id):
            return {101: 1.0}

    source_tissue = np.full((100, 200), 12, dtype=np.uint8)
    target_tissue = np.full((100, 200), 13, dtype=np.uint8)
    source_nuclei = np.zeros_like(source_tissue)
    for index in range(20):
        row = 2 + (index // 10) * 40
        col = 2 + (index % 10) * 18
        source_nuclei[row : row + 3, col : col + 3] = 101
    deletion = np.ones_like(source_tissue, dtype=bool)
    generation = np.ones_like(source_tissue, dtype=bool)

    scales, _, audit = compute_patch_adaptive_priors(
        reference_nuclei_raw=source_nuclei,
        reference_tissue=source_tissue,
        density_exclusion_region=deletion,
        target_tissue=target_tissue,
        generation_region=generation,
        library=FakeLibrary(),
        dataset_name="GlaS",
    )

    tissue_audit = audit["tissues"]["13"]
    calibration = tissue_audit[
        "dataset_prior_calibration_from_pre_edit_source"
    ]
    assert tissue_audit["reference_area_px"] == 0
    assert tissue_audit["density_mode"] == (
        "target_dataset_prior_times_pre_edit_family_calibration"
    )
    assert calibration["source_tissue_ids"] == [5, 11, 12, 13]
    assert calibration["scale"] == 2.5
    assert tissue_audit["target_density_per_10k_px"] == 12.5
    assert scales[13] == 2.5


def test_center_type_assignment_uses_probnet_score_subject_to_exact_quota():
    probability = np.zeros((6, 1, 2), dtype=np.float64)
    probability[1, 0, :] = [0.9, 0.1]
    probability[3, 0, :] = [0.2, 0.8]
    limits = {101: 1, 103: 1}
    placed = {101: 0, 103: 0}

    first = choose_type_with_remaining_quota_at_center(
        limits,
        placed,
        probability,
        0,
        0,
    )
    placed[first] += 1
    second = choose_type_with_remaining_quota_at_center(
        limits,
        placed,
        probability,
        0,
        1,
    )

    assert first == 101
    assert second == 103


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


def test_flat_probnet_quota_prefix_uses_quality_diversity_spacing():
    candidates = [(0, x) for x in range(10)]
    probability = np.full((1, 10), 0.6, dtype=np.float32)

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


def test_strong_probnet_quality_can_outweigh_diversity_bonus():
    candidates = [(0, 0), (0, 1), (0, 9)]
    probability = np.array(
        [[0.95, 0.90, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55]],
        dtype=np.float32,
    )

    ranked = choose_weighted_centers(
        candidates,
        probability,
        target_count=len(candidates),
        gamma=1.5,
        coverage_count=2,
        coverage_radius=3.0,
    )

    assert ranked[:2] == [(0, 0), (0, 1)]
    assert ranked[2:] == [(0, 9)]

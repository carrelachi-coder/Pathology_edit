import random

import numpy as np

try:
    import cv2
except ImportError:
    import pytest

    pytest.skip("OpenCV is required for nuclei instance extraction", allow_module_level=True)

from inpaint_cells.generate import sample_instance_for_center
from inpaint_cells.nuclei_library.library import (
    ReferenceFirstNucleiSampler,
    ReferenceNucleiInstancePool,
    place_nucleus_layered,
)
from phase3_joint_edit_refine.nuclei import iter_instances


class FakeLibrary:
    def __init__(self):
        self.calls = []

    def sample_instance(self, tissue_id, nuc_type=None, allow_cross_tissue=True):
        self.calls.append((tissue_id, nuc_type, allow_cross_tissue))
        resolved_type = 105 if nuc_type is None else nuc_type
        return {
            "mask": np.ones((3, 3), dtype=bool),
            "type": resolved_type,
            "area": 9,
            "source": "library",
        }


def test_reference_pool_extracts_tight_same_class_components_and_filters_border():
    nuclei = np.zeros((32, 32), dtype=np.uint8)
    nuclei[4:7, 5:9] = 101
    nuclei[12:16, 14:16] = 102
    nuclei[0:3, 20:23] = 101  # Clipped by the top patch boundary.

    pool = ReferenceNucleiInstancePool.from_mask(
        nuclei,
        min_area=2,
        max_area_ratio_to_median=0,
    )

    assert pool.counts()[101] == 1
    assert pool.counts()[102] == 1
    assert pool.rejected["border"] == 1
    assert pool.instances[101][0]["mask"].shape == (3, 4)
    assert pool.instances[101][0]["type"] == 101
    assert pool.instances[101][0]["area"] == 12


def test_reference_pool_accepts_internal_class_indices():
    nuclei = np.zeros((20, 20), dtype=np.uint8)
    nuclei[3:6, 3:7] = 1
    nuclei[10:14, 11:15] = 5

    pool = ReferenceNucleiInstancePool.from_mask(
        nuclei,
        min_area=2,
        max_area_ratio_to_median=0,
    )

    assert pool.counts()[101] == 1
    assert pool.counts()[105] == 1


def test_reference_pool_uses_same_touching_instance_recovery_as_joint_scene():
    nuclei = np.zeros((48, 48), dtype=np.uint8)
    cv2.circle(nuclei, (19, 24), 8, 103, thickness=-1)
    cv2.circle(nuclei, (29, 24), 8, 103, thickness=-1)

    pool = ReferenceNucleiInstancePool.from_mask(
        nuclei,
        min_area=2,
        max_area_ratio_to_median=0,
    )
    joint_areas = sorted(
        int(np.count_nonzero(component))
        for _, class_id, component in iter_instances(nuclei)
        if class_id == 3
    )

    assert len(joint_areas) == 2
    assert sorted(pool.area_samples(103)) == joint_areas


def test_reference_pool_can_be_partitioned_by_original_component_region():
    nuclei = np.zeros((24, 24), dtype=np.uint8)
    nuclei[3:6, 3:7] = 101
    nuclei[15:19, 16:20] = 101
    pool = ReferenceNucleiInstancePool.from_mask(nuclei, min_area=2)
    left = np.zeros_like(nuclei, dtype=bool)
    left[:, :12] = True

    left_pool = pool.subset_by_center_region(left)

    assert pool.counts()[101] == 2
    assert left_pool.counts()[101] == 1
    assert left_pool.instances[101][0]["area"] == 12


def test_reference_shapes_are_used_without_replacement_before_library_fallback():
    nuclei = np.zeros((32, 32), dtype=np.uint8)
    nuclei[3:6, 3:6] = 101
    nuclei[12:16, 12:15] = 101
    pool = ReferenceNucleiInstancePool.from_mask(
        nuclei,
        min_area=2,
        max_area_ratio_to_median=0,
    )
    library = FakeLibrary()
    random.seed(7)
    sampler = ReferenceFirstNucleiSampler(library, pool)

    first, first_source = sampler.sample_instance(1, 101)
    second, second_source = sampler.sample_instance(1, 101)
    third, third_source = sampler.sample_instance(1, 101)
    other_class, other_source = sampler.sample_instance(1, 102)

    assert first["type"] == second["type"] == third["type"] == 101
    assert first_source == second_source == "reference"
    assert third_source == "library"
    assert other_class["type"] == 102
    assert other_source == "library"
    assert library.calls == [(1, 101, True), (1, 102, True)]

    diagnostics = sampler.diagnostics()
    assert diagnostics["initial_reference_counts_by_type"]["101"] == 2
    assert diagnostics["remaining_reference_counts_by_type"]["101"] == 0
    assert diagnostics["selected_by_source"] == {"reference": 2, "library": 2}
    assert diagnostics["library_fallback_by_type"] == {"101": 1, "102": 1}


def test_failed_reference_shape_is_returned_to_pool():
    nuclei = np.zeros((24, 24), dtype=np.uint8)
    nuclei[5:8, 5:8] = 101
    pool = ReferenceNucleiInstancePool.from_mask(nuclei, min_area=2)
    sampler = ReferenceFirstNucleiSampler(FakeLibrary(), pool)

    instance, source = sampler.sample_instance(13, 101)
    assert source == "reference"
    assert sampler.diagnostics()["remaining_reference_counts_by_type"]["101"] == 0

    sampler.release_failed_instance(instance, source)

    assert sampler.diagnostics()["remaining_reference_counts_by_type"]["101"] == 1
    retry, retry_source = sampler.sample_instance(13, 101)
    assert retry_source == "reference"
    assert retry["area"] == instance["area"]


def test_obvious_same_class_area_outlier_is_not_used_as_one_nucleus():
    nuclei = np.zeros((48, 48), dtype=np.uint8)
    for y, x in [(3, 3), (3, 12), (12, 3), (12, 12)]:
        nuclei[y:y + 2, x:x + 2] = 103
    nuclei[25:30, 25:30] = 103

    pool = ReferenceNucleiInstancePool.from_mask(
        nuclei,
        min_area=2,
        max_area_ratio_to_median=4.0,
    )

    assert pool.counts()[103] == 4
    assert pool.rejected["area_outlier"] == 1


def test_size_calibration_filter_excludes_merge_without_changing_shape_pool():
    instances = {
        103: [
            {
                "mask": np.ones((1, area), dtype=bool),
                "type": 103,
                "area": area,
                "source": "reference",
            }
            for area in (100, 110, 120, 130, 500)
        ]
    }
    shape_pool = ReferenceNucleiInstancePool(instances=instances)

    size_pool = shape_pool.filtered_for_size_calibration()

    assert shape_pool.area_samples(103) == [100, 110, 120, 130, 500]
    assert size_pool.area_samples(103) == [100, 110, 120, 130]
    assert size_pool.rejected["area_outlier"] == 1


def test_library_fallback_is_resized_to_same_class_reference_area():
    nuclei = np.zeros((40, 40), dtype=np.uint8)
    nuclei[4:8, 4:8] = 101
    nuclei[20:24, 20:24] = 101
    pool = ReferenceNucleiInstancePool.from_mask(
        nuclei,
        min_area=2,
        max_area_ratio_to_median=0,
    )
    library = FakeLibrary()
    random.seed(13)
    sampler = ReferenceFirstNucleiSampler(
        library,
        pool,
        library_size_min_scale=0.25,
        library_size_max_scale=4.0,
        library_size_log_area_jitter=0.0,
    )

    sampler.sample_instance(1, 101)
    sampler.sample_instance(1, 101)
    fallback, source = sampler.sample_instance(1, 101)

    assert source == "library"
    assert fallback["size_calibrated"] is True
    assert fallback["size_calibration"]["basis"] == "same_class_reference_empirical_area"
    assert fallback["size_calibration"]["source_area"] == 9
    assert fallback["size_calibration"]["target_area"] == 16
    assert fallback["mask"].sum() == fallback["size_calibration"]["actual_area"]
    assert abs(int(fallback["mask"].sum()) - 16) <= 4

    diagnostics = sampler.diagnostics()["library_size_calibration"]
    assert diagnostics["calibrated_by_type"] == {"101": 1}
    assert diagnostics["uncalibrated_no_reference_by_type"] == {}


def test_library_fallback_without_same_class_reference_is_recorded_unscaled():
    nuclei = np.zeros((24, 24), dtype=np.uint8)
    nuclei[5:9, 5:9] = 101
    pool = ReferenceNucleiInstancePool.from_mask(nuclei, min_area=2)
    library = FakeLibrary()
    sampler = ReferenceFirstNucleiSampler(
        library,
        pool,
        library_size_log_area_jitter=0.0,
    )

    fallback, source = sampler.sample_instance(1, 102)

    assert source == "library"
    assert "size_calibrated" not in fallback
    diagnostics = sampler.diagnostics()["library_size_calibration"]
    assert diagnostics["uncalibrated_no_reference_by_type"] == {"102": 1}


def test_tissue_exact_library_shape_can_skip_patch_size_calibration():
    nuclei = np.zeros((24, 24), dtype=np.uint8)
    nuclei[5:9, 5:9] = 101
    pool = ReferenceNucleiInstancePool.from_mask(nuclei, min_area=2)
    library = FakeLibrary()
    sampler = ReferenceFirstNucleiSampler(
        library,
        pool,
        library_size_log_area_jitter=0.0,
    )

    instance, source = sampler.sample_library_instance(
        2,
        101,
        allow_cross_tissue=False,
        calibrate_size=False,
    )

    assert source == "library"
    assert instance["area"] == 9
    assert "size_calibrated" not in instance
    assert library.calls == [(2, 101, False)]


def test_new_tissue_library_shape_uses_target_tissue_size_reference():
    local = np.zeros((40, 40), dtype=np.uint8)
    local[4:7, 4:7] = 102
    target_tissue = np.zeros_like(local)
    target_tissue[10:16, 10:16] = 101
    patch = target_tissue.copy()
    patch[25:29, 25:29] = 102
    local_pool = ReferenceNucleiInstancePool.from_mask(local, min_area=2)
    target_pool = ReferenceNucleiInstancePool.from_mask(
        target_tissue, min_area=2
    )
    patch_pool = ReferenceNucleiInstancePool.from_mask(patch, min_area=2)
    sampler = ReferenceFirstNucleiSampler(
        FakeLibrary(),
        local_pool,
        size_reference_pool=target_pool,
        fallback_size_reference_pool=patch_pool,
        library_size_min_scale=0.25,
        library_size_max_scale=4.0,
        library_size_log_area_jitter=0.0,
    )

    instance, source = sample_instance_for_center(
        sampler,
        1,
        101,
        force_tissue_library=True,
    )

    assert source == "library"
    assert instance["size_calibration"]["target_area"] == 36
    calibration = sampler.diagnostics()["library_size_calibration"]
    assert calibration["calibrated_by_type"] == {"101": 1}
    assert calibration["reference_basis_by_type"]["101"] == (
        "target_tissue_same_class_complete_instance_reference"
    )
    assert calibration["reference_basis_by_type"]["102"] == (
        "patch_same_class_complete_instance_reference_fallback"
    )


def test_strict_layered_placement_never_overwrites_retained_nucleus():
    nuclei = np.zeros((12, 12), dtype=np.int64)
    nuclei[5, 5] = 2
    original = nuclei.copy()
    instance = {
        "mask": np.ones((3, 3), dtype=bool),
        "type": 101,
        "source": "reference",
    }

    placed = place_nucleus_layered(
        nuclei,
        6,
        6,
        instance,
        augment=False,
    )

    assert placed is False
    np.testing.assert_array_equal(nuclei, original)


def test_strict_layered_placement_requires_complete_tissue_containment():
    nuclei = np.zeros((12, 12), dtype=np.int64)
    original = nuclei.copy()
    tissue = np.zeros_like(nuclei, dtype=bool)
    tissue[5:8, 5:8] = True
    instance = {
        "mask": np.ones((5, 5), dtype=bool),
        "type": 101,
        "source": "reference",
    }

    placed = place_nucleus_layered(
        nuclei,
        6,
        6,
        instance,
        augment=False,
        valid_tissue_mask=tissue,
        require_full_tissue_containment=True,
    )

    assert placed is False
    np.testing.assert_array_equal(nuclei, original)


def test_layered_placement_spacing_margin_prevents_component_merging():
    nuclei = np.zeros((12, 12), dtype=np.int64)
    nuclei[5, 5] = 1
    original = nuclei.copy()
    instance = {
        "mask": np.ones((1, 1), dtype=bool),
        "type": 101,
        "source": "reference",
    }

    placed = place_nucleus_layered(
        nuclei,
        5,
        6,
        instance,
        augment=False,
        minimum_separation_px=1,
    )

    assert placed is False
    np.testing.assert_array_equal(nuclei, original)


def test_layered_placement_removes_disconnected_shape_satellites():
    nuclei = np.zeros((20, 20), dtype=np.int64)
    mask = np.zeros((7, 7), dtype=bool)
    mask[2:5, 2:5] = True
    mask[0, 0] = True
    instance = {
        "mask": mask,
        "type": 101,
        "source": "reference",
    }

    placed = place_nucleus_layered(
        nuclei,
        10,
        10,
        instance,
        augment=False,
    )

    assert placed is True
    assert np.count_nonzero(nuclei) == 9


def test_layered_placement_records_realized_transformed_footprint():
    nuclei = np.zeros((20, 20), dtype=np.int64)
    instance = {
        "mask": np.ones((3, 5), dtype=bool),
        "type": 101,
        "source": "reference",
    }
    placement = {}

    placed = place_nucleus_layered(
        nuclei,
        10,
        10,
        instance,
        augment=False,
        placement_metadata=placement,
    )

    assert placed is True
    assert placement["area_px"] == 15
    assert placement["class_id"] == 1
    assert placement["nucleus_type"] == 101
    assert placement["boundary_truncated"] is False


def test_reference_shape_ignores_retry_scale_and_preserves_patch_area():
    nuclei = np.zeros((20, 20), dtype=np.int64)
    instance = {
        "mask": np.ones((4, 4), dtype=bool),
        "type": 101,
        "source": "reference",
    }

    placed = place_nucleus_layered(
        nuclei,
        10,
        10,
        instance,
        augment=True,
        rotation_quarters=0,
        flip_horizontal=False,
        flip_vertical=False,
        scale=0.5,
    )

    assert placed is True
    assert np.count_nonzero(nuclei) == 16

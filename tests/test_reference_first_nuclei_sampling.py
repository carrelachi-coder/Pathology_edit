import random

import numpy as np

try:
    import cv2  # noqa: F401
except ImportError:
    import pytest

    pytest.skip("OpenCV is required for nuclei instance extraction", allow_module_level=True)

from inpaint_cells.nuclei_library.library import (
    ReferenceFirstNucleiSampler,
    ReferenceNucleiInstancePool,
    place_nucleus_layered,
)


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

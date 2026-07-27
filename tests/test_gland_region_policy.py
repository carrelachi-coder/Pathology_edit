import numpy as np

from phase3_mask_edit.core.gland_region import (
    expand_region_to_intersecting_components,
    glas_gland_mask,
    glas_whole_gland_generation_region,
)


def test_glas_boundary_change_expands_only_the_affected_gland_instance():
    source = np.full((12, 12), 2, dtype=np.uint8)
    source[2:5, 2:5] = 11
    source[8:10, 8:10] = 5
    target = source.copy()
    target[1:6, 1:6] = 11
    semantic = source != target

    generation, metadata = glas_whole_gland_generation_region(
        source,
        target,
        semantic,
        profile="GlaS",
    )

    assert metadata["applied"] is True
    assert metadata["reason"] == "gland_boundary_changed"
    assert np.all(generation[1:6, 1:6])
    assert not np.any(generation[8:10, 8:10])
    assert np.all(generation[semantic])


def test_glas_fine_relabel_without_boundary_change_is_not_overexpanded():
    source = np.full((8, 8), 2, dtype=np.uint8)
    source[2:6, 2:6] = 11
    target = source.copy()
    target[2:6, 2:6] = 12
    semantic = source != target

    generation, metadata = glas_whole_gland_generation_region(
        source,
        target,
        semantic,
        profile="GLAS",
    )

    assert metadata["applied"] is False
    assert metadata["reason"] == "gland_footprint_unchanged"
    assert np.array_equal(generation, semantic)


def test_candidate_region_expands_to_complete_intersecting_glas_components():
    tissue = np.full((10, 10), 2, dtype=np.uint8)
    tissue[1:4, 1:4] = 5
    tissue[6:9, 6:9] = 13
    candidate = np.zeros_like(tissue, dtype=bool)
    candidate[2, 2] = True
    candidate[5, 5] = True

    expanded, metadata = expand_region_to_intersecting_components(
        candidate,
        glas_gland_mask(tissue),
    )

    assert metadata["touched_component_count"] == 1
    assert np.all(expanded[1:4, 1:4])
    assert expanded[5, 5]
    assert not np.any(expanded[6:9, 6:9])

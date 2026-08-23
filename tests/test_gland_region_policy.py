import numpy as np

from phase3_mask_edit.core.gland_region import (
    bound_generation_context_region,
    expand_region_to_intersecting_components,
    generation_context_max_extra_fraction,
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
    assert metadata["reason"] == "gland_change_whole_connected_component"
    assert np.all(generation[1:6, 1:6])
    assert not np.any(generation[8:10, 8:10])
    assert np.all(generation[semantic])


def test_glas_fine_relabel_expands_to_whole_connected_gland():
    source = np.full((8, 8), 2, dtype=np.uint8)
    source[2:6, 2:6] = 11
    target = source.copy()
    target[3:5, 3:5] = 12
    semantic = source != target

    generation, metadata = glas_whole_gland_generation_region(
        source,
        target,
        semantic,
        profile="GLAS",
    )

    assert metadata["applied"] is True
    assert metadata["reason"] == "gland_change_whole_connected_component"
    assert np.all(generation[2:6, 2:6])
    assert int(np.count_nonzero(generation)) == 16


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


def test_generation_context_caps_large_connected_component_expansion():
    semantic = np.zeros((128, 128), dtype=bool)
    semantic[48:80, 48:80] = True
    candidate = np.ones_like(semantic)

    bounded, metadata = bound_generation_context_region(semantic, candidate)

    semantic_pixels = int(np.count_nonzero(semantic))
    assert metadata["capped"] is True
    assert metadata["policy"] == "bounded_generation_context_v2"
    assert metadata["extra_budget_pixels"] == semantic_pixels
    assert int(np.count_nonzero(bounded)) == semantic_pixels * 2
    assert np.all(bounded[semantic])
    assert not bounded[0, 0]


def test_generation_context_keeps_a_stromal_collar_around_a_thin_edit():
    semantic = np.zeros((128, 128), dtype=bool)
    semantic[60:68, 28:100] = True
    candidate = np.zeros_like(semantic)
    candidate[44:84, 12:116] = True

    bounded, metadata = bound_generation_context_region(semantic, candidate)

    semantic_pixels = int(np.count_nonzero(semantic))
    assert metadata["extra_budget_pixels"] == semantic_pixels
    assert int(np.count_nonzero(bounded)) == semantic_pixels * 2
    assert np.all(bounded[semantic])
    # The retained nearest context must exist on both normal sides of the
    # horizontal edit instead of spending the entire budget along its tips.
    assert np.any(bounded[56:60, 40:88])
    assert np.any(bounded[68:72, 40:88])


def test_cord_primitive_retains_a_larger_stromal_context_budget():
    semantic = np.zeros((128, 128), dtype=bool)
    semantic[60:68, 28:100] = True
    candidate = np.ones_like(semantic)

    bounded, metadata = bound_generation_context_region(
        semantic,
        candidate,
        primitive_id="invasive-cord-formation-v1",
    )

    semantic_pixels = int(np.count_nonzero(semantic))
    assert generation_context_max_extra_fraction(
        "invasive-cord-formation-v1"
    ) == 1.5
    assert metadata["primitive_id"] == "invasive-cord-formation-v1"
    assert metadata["max_extra_fraction"] == 1.5
    assert metadata["extra_budget_pixels"] == semantic_pixels * 3 // 2
    assert int(np.count_nonzero(bounded)) == semantic_pixels * 5 // 2


def test_current_infiltrative_cord_retains_larger_context_budget():
    semantic = np.zeros((128, 128), dtype=bool)
    semantic[60:68, 28:100] = True
    candidate = np.ones_like(semantic)

    bounded, metadata = bound_generation_context_region(
        semantic,
        candidate,
        primitive_id="infiltrative-nest-cord-extension-v1",
    )

    semantic_pixels = int(np.count_nonzero(semantic))
    assert metadata["max_extra_fraction"] == 1.5
    assert int(np.count_nonzero(bounded)) == semantic_pixels * 5 // 2


def test_cell_decrease_freezes_a_four_pixel_cleanup_collar():
    semantic = np.zeros((64, 64), dtype=bool)
    semantic[28:36, 28:36] = True
    candidate = np.ones_like(semantic)

    bounded, metadata = bound_generation_context_region(
        semantic,
        candidate,
        primitive_id="cell-type-abundance-decrease-v1",
    )

    required = np.zeros_like(semantic)
    required[24:40, 24:40] = True
    assert generation_context_max_extra_fraction(
        "cell-type-abundance-decrease-v1"
    ) == 2.0
    assert metadata["minimum_dilation_pixels"] == 4
    assert np.all(bounded[required])
    assert np.all(bounded[semantic])
    assert metadata["required_dilation_extra_pixels"] == 192
    assert int(np.count_nonzero(bounded)) == int(np.count_nonzero(required))


def test_glas_large_union_preserves_the_complete_connected_component():
    source = np.full((128, 128), 2, dtype=np.uint8)
    source[8:120, 8:120] = 12
    target = source.copy()
    target[40:88, 40:88] = 2
    semantic = source != target

    generation, metadata = glas_whole_gland_generation_region(
        source,
        target,
        semantic,
        profile="GlaS",
    )

    assert metadata["reason"] == "gland_change_whole_connected_component"
    assert metadata["context_bound"]["capped"] is False
    assert np.all(generation[8:120, 8:120])
    assert np.all(generation[semantic])

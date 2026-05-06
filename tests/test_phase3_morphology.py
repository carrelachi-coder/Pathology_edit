"""Tests for phase3_mask_edit/core/morphology.py — base + extensions."""

import unittest

import numpy as np

from phase3_mask_edit.core.morphology import (
    binary_dilate,
    binary_erode,
    boundary_ring,
    distance_to_boundary,
    distance_to_label,
    fill_small_holes,
    generate_islands,
    keep_only_touching,
    multi_scale_smooth_noise,
    nearest_label_backfill,
    remove_small_components,
    select_boundary_band_by_fraction,
    select_connected_region_by_fraction,
    select_region_by_fraction,
    signed_distance_field,
)


class Phase3MorphologyTests(unittest.TestCase):
    def test_binary_dilate_uses_square_radius(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True

        dilated = binary_dilate(mask, radius=1)

        expected = np.zeros((5, 5), dtype=bool)
        expected[1:4, 1:4] = True
        np.testing.assert_array_equal(dilated, expected)

    def test_binary_erode_shrinks_with_false_border(self):
        mask = np.ones((5, 5), dtype=bool)

        eroded = binary_erode(mask, radius=1)

        expected = np.zeros((5, 5), dtype=bool)
        expected[1:4, 1:4] = True
        np.testing.assert_array_equal(eroded, expected)

    def test_boundary_ring_limits_dilation_to_candidate_mask(self):
        source = np.zeros((5, 5), dtype=bool)
        source[2, 2] = True
        candidate = np.zeros((5, 5), dtype=bool)
        candidate[:, 2:] = True

        ring = boundary_ring(source, candidate, radius=1)

        expected = np.zeros((5, 5), dtype=bool)
        expected[1:4, 2:4] = True
        expected[2, 2] = False
        np.testing.assert_array_equal(ring, expected)

    def test_select_region_by_fraction_is_seeded_and_patch_fraction_based(self):
        candidate = np.zeros((4, 5), dtype=bool)
        candidate[1:3, :] = True

        selected_a = select_region_by_fraction(candidate, target_fraction=0.25, seed=7)
        selected_b = select_region_by_fraction(candidate, target_fraction=0.25, seed=7)

        self.assertEqual(int(selected_a.sum()), 5)
        np.testing.assert_array_equal(selected_a, selected_b)
        self.assertTrue(np.all(selected_a <= candidate))

    def test_select_region_by_fraction_caps_at_available_candidates(self):
        candidate = np.zeros((4, 5), dtype=bool)
        candidate[0, 0] = True
        candidate[1, 1] = True

        selected = select_region_by_fraction(candidate, target_fraction=0.90, seed=1)

        self.assertEqual(int(selected.sum()), 2)
        np.testing.assert_array_equal(selected, candidate)

    def test_select_connected_region_by_fraction_grows_single_connected_region_from_seed(self):
        candidate = np.zeros((6, 6), dtype=bool)
        candidate[1:5, 1:5] = True
        seed = np.zeros((6, 6), dtype=bool)
        seed[1, 1:5] = True

        selected = select_connected_region_by_fraction(
            candidate, seed, target_fraction=8 / 36, seed_value=3
        )

        self.assertEqual(int(selected.sum()), 8)
        self.assertTrue(np.all(selected <= candidate))
        self.assertTrue(np.any(selected & seed))
        self.assertEqual(_count_components_for_test(selected), 1)

    def test_select_connected_region_by_fraction_caps_at_seed_reachable_candidates(self):
        candidate = np.zeros((5, 7), dtype=bool)
        candidate[1:4, 1:3] = True
        candidate[1:4, 5:7] = True
        seed = np.zeros((5, 7), dtype=bool)
        seed[1, 1] = True

        selected = select_connected_region_by_fraction(
            candidate, seed, target_fraction=0.90, seed_value=11
        )

        self.assertEqual(int(selected.sum()), 6)
        self.assertTrue(np.all(selected[:, 5:7] == 0))
        self.assertEqual(_count_components_for_test(selected), 1)

    def test_select_connected_region_by_fraction_requires_seed_inside_candidate(self):
        candidate = np.zeros((4, 4), dtype=bool)
        candidate[1:3, 1:3] = True
        seed = np.zeros((4, 4), dtype=bool)

        with self.assertRaisesRegex(ValueError, "seed_mask"):
            select_connected_region_by_fraction(candidate, seed, 0.25, seed_value=1)

    def test_select_boundary_band_by_fraction_searches_radius_for_target_area(self):
        source = np.zeros((9, 9), dtype=bool)
        source[4, 4] = True
        candidate = ~source

        selected, info = select_boundary_band_by_fraction(
            source,
            candidate,
            target_fraction=8 / 81,
            min_radius=1,
            max_radius=4,
        )

        self.assertEqual(info["radius"], 1)
        self.assertEqual(info["selected_pixels"], 8)
        self.assertAlmostEqual(info["actual_fraction"], 8 / 81)
        np.testing.assert_array_equal(selected, boundary_ring(source, candidate, radius=1))

    def test_select_boundary_band_by_fraction_chooses_closest_available_radius(self):
        source = np.zeros((9, 9), dtype=bool)
        source[4, 4] = True
        candidate = ~source

        selected, info = select_boundary_band_by_fraction(
            source,
            candidate,
            target_fraction=20 / 81,
            min_radius=1,
            max_radius=4,
        )

        self.assertEqual(info["radius"], 2)
        self.assertEqual(int(selected.sum()), 24)
        self.assertTrue(np.all(selected <= candidate))

    def test_select_boundary_band_by_fraction_reports_candidate_shortfall(self):
        source = np.zeros((5, 5), dtype=bool)
        source[2, 2] = True
        candidate = np.zeros((5, 5), dtype=bool)
        candidate[1, 2] = True
        candidate[2, 1] = True

        selected, info = select_boundary_band_by_fraction(
            source,
            candidate,
            target_fraction=0.50,
            min_radius=1,
            max_radius=3,
        )

        self.assertEqual(int(selected.sum()), 2)
        self.assertTrue(info["candidate_shortfall"])

    def test_invalid_radius_and_fraction_raise_clear_errors(self):
        mask = np.zeros((3, 3), dtype=bool)

        with self.assertRaisesRegex(ValueError, "radius"):
            binary_dilate(mask, radius=-1)

        with self.assertRaisesRegex(ValueError, "target_fraction"):
            select_region_by_fraction(mask, target_fraction=1.5, seed=1)

        with self.assertRaisesRegex(ValueError, "max_radius"):
            select_boundary_band_by_fraction(mask, ~mask, 0.2, min_radius=3, max_radius=1)


# ── SDF / distance map tests ──────────────────────────────────────

class SDFAndDistanceTests(unittest.TestCase):
    def test_sdf_positive_outside_negative_inside(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True

        sdf = signed_distance_field(mask, metric="euclidean")
        self.assertTrue(sdf[2, 2] < 0)
        self.assertTrue(sdf[0, 0] > 0)
        # A single-pixel mask has interior distance = 1.0 from boundary
        self.assertAlmostEqual(sdf[2, 2], -1.0)

    def test_sdf_chessboard_metric(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[1:4, 1:4] = True

        sdf = signed_distance_field(mask, metric="chessboard")
        self.assertTrue(sdf[2, 2] < 0)
        self.assertTrue(sdf[0, 0] > 0)
        self.assertEqual(int(sdf[0, 0]), 1)

    def test_distance_to_boundary_inside(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True

        dist = distance_to_boundary(mask)
        self.assertAlmostEqual(dist[5, 5], 2.0, places=1)
        # Corner pixels of the mask are on the boundary, distance = 1.0
        self.assertAlmostEqual(dist[3, 3], 1.0, places=1)

    def test_distance_to_label_from_tumor(self):
        id_mask = np.array([[1, 2, 2], [2, 2, 2], [0, 0, 0]], dtype=np.int64)

        dist = distance_to_label(id_mask, target_ids=[1])
        self.assertAlmostEqual(dist[0, 0], 0.0)
        self.assertAlmostEqual(dist[0, 1], 1.0)
        self.assertAlmostEqual(dist[2, 0], 2.0, places=1)

    def test_distance_to_label_invalid_metric_raises(self):
        mask = np.ones((3, 3), dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "unknown metric"):
            distance_to_label(mask, target_ids=[1], metric="invalid")

    def test_distance_to_label_3d_raises(self):
        mask = np.ones((3, 3, 3), dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "2D"):
            distance_to_label(mask, target_ids=[1])


# ── multi-scale smooth noise tests ────────────────────────────────

class MultiScaleNoiseTests(unittest.TestCase):
    def test_output_shape_matches_input(self):
        noise = multi_scale_smooth_noise((10, 10), scales=(2.0, 8.0), seed=42)
        self.assertEqual(noise.shape, (10, 10))

    def test_deterministic_with_same_seed(self):
        a = multi_scale_smooth_noise((8, 8), scales=(2.0, 8.0), seed=7)
        b = multi_scale_smooth_noise((8, 8), scales=(2.0, 8.0), seed=7)
        np.testing.assert_array_almost_equal(a, b)

    def test_different_seed_gives_different_result(self):
        a = multi_scale_smooth_noise((8, 8), scales=(2.0, 8.0), seed=1)
        b = multi_scale_smooth_noise((8, 8), scales=(2.0, 8.0), seed=2)
        self.assertFalse(np.allclose(a, b))

    def test_custom_amplitudes(self):
        noise = multi_scale_smooth_noise(
            (8, 8), scales=(2.0, 8.0), amplitudes=(0.7, 0.3), seed=5
        )
        self.assertEqual(noise.shape, (8, 8))

    def test_mismatched_amplitudes_raises(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            multi_scale_smooth_noise((8, 8), scales=(2.0, 8.0), amplitudes=(1.0,))

    def test_empty_scales_raises(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            multi_scale_smooth_noise((8, 8), scales=())

    def test_bad_shape_raises(self):
        with self.assertRaisesRegex(ValueError, "2-element"):
            multi_scale_smooth_noise((8,))


# ── island generation tests ────────────────────────────────────────

class IslandGenerationTests(unittest.TestCase):
    def test_generate_islands_near_source(self):
        source = np.zeros((20, 20), dtype=bool)
        source[10, 10] = True
        candidate = ~source

        islands, info = generate_islands(
            candidate, source,
            max_distance_px=5,
            max_island_area_px=10,
            max_islands=3,
            target_fraction=3 / 400,
            seed=42,
        )

        self.assertTrue(np.all(islands <= candidate))
        self.assertTrue(np.all(islands <= ~source))
        self.assertTrue(info["islands_generated"] > 0)
        self.assertTrue(info["total_island_pixels"] > 0)

    def test_no_candidates_returns_empty(self):
        source = np.zeros((10, 10), dtype=bool)
        source[5, 5] = True
        candidate = np.zeros((10, 10), dtype=bool)

        islands, info = generate_islands(
            candidate, source,
            max_distance_px=3,
            max_island_area_px=5,
            max_islands=3,
            target_fraction=0.1,
            seed=1,
        )

        self.assertEqual(int(np.count_nonzero(islands)), 0)
        self.assertEqual(info["islands_generated"], 0)

    def test_islands_respect_max_distance(self):
        source = np.zeros((30, 30), dtype=bool)
        source[15, 15] = True
        candidate = ~source

        islands, info = generate_islands(
            candidate, source,
            max_distance_px=3,
            max_island_area_px=8,
            max_islands=5,
            target_fraction=0.05,
            seed=7,
        )

        if np.any(islands):
            dist_map = np.zeros((30, 30), dtype=float)
            from scipy import ndimage
            dist_map = ndimage.distance_transform_edt(~source)
            self.assertTrue(np.all(dist_map[islands] <= 3.0))

    def test_islands_respect_max_count(self):
        source = np.zeros((20, 20), dtype=bool)
        source[10, 10] = True
        candidate = ~source

        islands, info = generate_islands(
            candidate, source,
            max_distance_px=6,
            max_island_area_px=5,
            max_islands=2,
            target_fraction=0.10,
            seed=3,
        )

        self.assertLessEqual(info["islands_generated"], 2)

    def test_shape_mismatch_raises(self):
        source = np.zeros((5, 5), dtype=bool)
        candidate = np.zeros((6, 6), dtype=bool)
        with self.assertRaisesRegex(ValueError, "same shape"):
            generate_islands(candidate, source, max_distance_px=3, max_island_area_px=5, max_islands=1, target_fraction=0.01)


# ── topology cleanup tests ─────────────────────────────────────────

class TopologyCleanupTests(unittest.TestCase):
    def test_remove_small_components_keeps_large(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[1, 1] = True
        mask[3:6, 3:6] = True

        cleaned = remove_small_components(mask, min_area_px=5)
        self.assertFalse(cleaned[1, 1])
        self.assertTrue(np.all(cleaned[3:6, 3:6]))

    def test_remove_small_components_zero_threshold_returns_copy(self):
        mask = np.array([[True, False], [False, True]], dtype=bool)
        cleaned = remove_small_components(mask, min_area_px=0)
        np.testing.assert_array_equal(cleaned, mask)

    def test_fill_small_holes(self):
        mask = np.ones((10, 10), dtype=bool)
        mask[3, 3] = False
        mask[5:7, 5:7] = False

        filled = fill_small_holes(mask, max_hole_area_px=2)
        self.assertTrue(filled[3, 3])
        self.assertFalse(np.all(filled[5:7, 5:7]))

    def test_fill_small_holes_zero_threshold_returns_copy(self):
        mask = np.ones((5, 5), dtype=bool)
        filled = fill_small_holes(mask, max_hole_area_px=0)
        np.testing.assert_array_equal(filled, mask)

    def test_keep_only_touching(self):
        components = np.zeros((10, 10), dtype=bool)
        components[1, 1] = True
        components[5:7, 3:5] = True
        context = np.zeros((10, 10), dtype=bool)
        context[4, 3] = True

        kept = keep_only_touching(components, context)
        self.assertFalse(kept[1, 1])
        self.assertTrue(np.all(kept[5:7, 3:5]))


# ── nearest label backfill tests ───────────────────────────────────

class NearestBackfillTests(unittest.TestCase):
    def test_nearest_backfill_assigns_closest_source_id(self):
        id_mask = np.array([[1, 2, 2], [2, 2, 2], [0, 0, 0]], dtype=np.int64)
        change = np.array([[False, True, False], [False, True, False], [True, True, True]], dtype=bool)

        result = nearest_label_backfill(id_mask, source_labels=[2], change_region=change)

        self.assertEqual(result.shape, (int(np.count_nonzero(change)),))
        self.assertTrue(all(r == 2 for r in result))

    def test_nearest_backfill_chooses_closest_for_multiple_labels(self):
        id_mask = np.array([[1, 2, 7], [1, 2, 7], [0, 0, 0]], dtype=np.int64)
        change = np.zeros((3, 3), dtype=bool)
        change[2, 1] = True

        result = nearest_label_backfill(id_mask, source_labels=[1, 2], change_region=change)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], 2)

    def test_nearest_backfill_raises_on_missing_source(self):
        id_mask = np.ones((3, 3), dtype=np.int64)
        change = np.zeros((3, 3), dtype=bool)
        change[1, 1] = True

        with self.assertRaisesRegex(ValueError, "source_labels"):
            nearest_label_backfill(id_mask, source_labels=[99], change_region=change)

    def test_nearest_backfill_shape_mismatch_raises(self):
        id_mask = np.ones((3, 3), dtype=np.int64)
        change = np.ones((4, 4), dtype=bool)

        with self.assertRaisesRegex(ValueError, "same shape"):
            nearest_label_backfill(id_mask, source_labels=[1], change_region=change)


def _count_components_for_test(mask):
    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros(mask.shape, dtype=bool)
    components = 0
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if visited[row, col] or not mask[row, col]:
                continue
            components += 1
            stack = [(row, col)]
            visited[row, col] = True
            while stack:
                r, c = stack.pop()
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]:
                        if mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
    return components


if __name__ == "__main__":
    unittest.main()
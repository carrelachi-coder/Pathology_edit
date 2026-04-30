import unittest

import numpy as np

from phase3_mask_edit.core.morphology import (
    binary_dilate,
    binary_erode,
    boundary_ring,
    select_boundary_band_by_fraction,
    select_connected_region_by_fraction,
    select_region_by_fraction,
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

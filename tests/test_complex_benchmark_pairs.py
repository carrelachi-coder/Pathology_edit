from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image, PngImagePlugin

from scripts.build_complex_benchmark_pairs import (
    _cell_features,
    _copy_png_without_ancillary_chunks,
    _select_double_annotation_rows,
    _select_non_conflicting_wsi_pairs,
    _validate_selected_pair_set,
    coordinate_boxes_conflict,
    jensen_shannon_distance,
    symmetric_relative_difference,
)


class CoordinateConflictTests(unittest.TestCase):
    def test_identical_and_partially_overlapping_boxes_conflict(self):
        self.assertTrue(coordinate_boxes_conflict(100, 200, 100, 200, span=512))
        self.assertTrue(coordinate_boxes_conflict(100, 200, 611, 200, span=512))

    def test_edge_touching_boxes_do_not_overlap(self):
        self.assertFalse(coordinate_boxes_conflict(100, 200, 612, 200, span=512))
        self.assertFalse(coordinate_boxes_conflict(100, 200, 100, 712, span=512))

    def test_requested_gap_is_enforced(self):
        self.assertTrue(coordinate_boxes_conflict(100, 200, 612, 200, span=512, minimum_gap=1))
        self.assertFalse(coordinate_boxes_conflict(100, 200, 613, 200, span=512, minimum_gap=1))


class PairMetricTests(unittest.TestCase):
    def test_jsd_is_symmetric_and_bounded(self):
        left = [0.75, 0.25, 0.0]
        right = [0.25, 0.75, 0.0]
        forward = jensen_shannon_distance(left, right)
        reverse = jensen_shannon_distance(right, left)
        self.assertAlmostEqual(forward, reverse)
        self.assertGreater(forward, 0.0)
        self.assertLessEqual(forward, 1.0)
        self.assertEqual(jensen_shannon_distance(left, left), 0.0)

    def test_symmetric_density_difference(self):
        self.assertAlmostEqual(symmetric_relative_difference(0.8, 1.2), 0.4)
        self.assertEqual(symmetric_relative_difference(0.0, 0.0), 0.0)

    def test_cell_features_count_components_and_capture_spatial_profile(self):
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[1:4, 1:4] = 101
        mask[10:13, 10:13] = 101
        mask[5:8, 12:15] = 102
        counts, spatial = _cell_features(mask, min_component_area=5, grid_size=4)

        np.testing.assert_array_equal(counts, np.array([2, 1, 0, 0, 0], dtype=np.float64))
        self.assertAlmostEqual(float(spatial.sum()), 1.0)
        self.assertEqual(len(spatial), 16)
        self.assertTrue(np.all(spatial[:-1] >= spatial[1:]))

    def test_jsd_handles_near_ties_without_a_dominant_class_discontinuity(self):
        self.assertLess(jensen_shannon_distance([0.49, 0.51], [0.51, 0.49]), 0.01)


class WsiMatchingTests(unittest.TestCase):
    @staticmethod
    def edge(name: str, a: tuple[str, int, int], b: tuple[str, int, int], score: float):
        return {
            "pair_name": name,
            "pair_score": score,
            "a_stem": a[0],
            "a_x": a[1],
            "a_y": a[2],
            "b_stem": b[0],
            "b_x": b[1],
            "b_y": b[2],
        }

    def test_matching_uses_multiple_pairs_without_patch_or_spatial_reuse(self):
        edges = [
            self.edge("one", ("a", 0, 0), ("b", 512, 0), 0.10),
            self.edge("two", ("c", 1024, 0), ("d", 1536, 0), 0.20),
            # This cheap edge would make patch e overlap patch c from pair two.
            self.edge("spatial-conflict", ("a", 0, 0), ("e", 1100, 0), 0.01),
        ]
        selected = _select_non_conflicting_wsi_pairs(
            edges,
            coordinate_span=512,
            coordinate_gap=0,
            max_pairs=0,
        )
        self.assertEqual({row["pair_name"] for row in selected}, {"one", "two"})

    def test_pair_cap_is_respected(self):
        edges = [
            self.edge("one", ("a", 0, 0), ("b", 512, 0), 0.10),
            self.edge("two", ("c", 1024, 0), ("d", 1536, 0), 0.20),
        ]
        selected = _select_non_conflicting_wsi_pairs(
            edges,
            coordinate_span=512,
            coordinate_gap=0,
            max_pairs=1,
        )
        self.assertEqual([row["pair_name"] for row in selected], ["one"])

    def test_final_validator_rejects_cross_pair_coordinate_overlap(self):
        pairs = [
            {
                "wsi": "slide",
                "a_stem": "a",
                "a_x": 0,
                "a_y": 0,
                "b_stem": "b",
                "b_x": 512,
                "b_y": 0,
            },
            {
                "wsi": "slide",
                "a_stem": "c",
                "a_x": 1000,
                "a_y": 0,
                "b_stem": "d",
                "b_x": 1536,
                "b_y": 0,
            },
        ]
        with self.assertRaisesRegex(RuntimeError, "coordinates conflict"):
            _validate_selected_pair_set(pairs, coordinate_span=512, coordinate_gap=0)


class AnnotationPackageTests(unittest.TestCase):
    def test_annotation_png_copy_strips_text_metadata_without_changing_pixels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.png"
            destination = root / "destination.png"
            pixels = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("large_note", "x" * 1000)
            Image.fromarray(pixels).save(source, pnginfo=metadata)

            _copy_png_without_ancillary_chunks(source, destination)

            with Image.open(destination) as copied:
                np.testing.assert_array_equal(np.asarray(copied), pixels)
                self.assertNotIn("large_note", copied.info)

    def test_double_annotation_selection_balances_organs_and_uses_unique_pairs(self):
        rows = []
        for organ in ("breast", "lung"):
            for pair_index in range(10):
                for side in ("a", "b"):
                    rows.append(
                        {
                            "annotation_id": f"{organ}-{pair_index}-{side}",
                            "pair_id": f"{organ}-{pair_index}",
                            "side": side,
                            "organ": organ,
                            "pair_score": pair_index / 10,
                        }
                    )

        selected = _select_double_annotation_rows(rows, target=8)

        self.assertEqual(len(selected), 8)
        self.assertEqual({row["organ"] for row in selected}, {"breast", "lung"})
        self.assertEqual(len({row["pair_id"] for row in selected}), 8)
        self.assertEqual(
            {organ: sum(row["organ"] == organ for row in selected) for organ in ("breast", "lung")},
            {"breast": 4, "lung": 4},
        )


if __name__ == "__main__":
    unittest.main()

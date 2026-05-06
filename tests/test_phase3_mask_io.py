"""Tests for phase3_mask_edit/core/mask_io.py."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from phase3_mask_edit.core.mask_io import (
    MaskIOError,
    id_to_rgb,
    load_change_region,
    load_id_mask,
    load_metadata,
    load_rgb_mask,
    rgb_to_id,
    save_change_region,
    save_edit_output,
    save_id_mask,
    save_metadata,
    save_rgb_mask,
)


class IdMaskLoadSaveTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def test_save_and_load_roundtrip(self):
        mask = np.array([[0, 1, 2], [3, 7, 15]], dtype=np.int64)
        path = Path(self.tmpdir.name) / "mask.png"
        save_id_mask(mask, path)

        loaded = load_id_mask(path)
        np.testing.assert_array_equal(loaded, mask)

    def test_load_missing_file_raises(self):
        with self.assertRaises(MaskIOError):
            load_id_mask("/nonexistent/mask.png")

    def test_save_3d_mask_raises(self):
        mask = np.zeros((4, 4, 3), dtype=np.int64)
        with self.assertRaises(MaskIOError):
            save_id_mask(mask, Path(self.tmpdir.name) / "bad.png")

    def test_save_creates_parent_dirs(self):
        path = Path(self.tmpdir.name) / "deep" / "nested" / "mask.png"
        mask = np.zeros((3, 3), dtype=np.int64)
        save_id_mask(mask, path)
        self.assertTrue(path.exists())


class RGBMaskConversionTests(unittest.TestCase):
    def test_id_to_rgb_known_ids(self):
        mask = np.array([[0, 1, 2], [3, 4, 7]], dtype=np.int64)
        rgb = id_to_rgb(mask)

        from dataset_config.unified_labels import UNIFIED_COLOR_MAP
        for id_val, expected_color in UNIFIED_COLOR_MAP.items():
            if id_val in {0, 1, 2, 3, 4, 7}:
                pixels = rgb[mask == id_val]
                if pixels.size > 0:
                    np.testing.assert_array_equal(pixels[0], expected_color)

    def test_id_to_rgb_unknown_id_renders_white(self):
        mask = np.array([[99]], dtype=np.int64)
        rgb = id_to_rgb(mask)
        np.testing.assert_array_equal(rgb[0, 0], [255, 255, 255])

    def test_rgb_to_id_roundtrip(self):
        mask = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)
        rgb = id_to_rgb(mask)
        recovered = rgb_to_id(rgb)
        np.testing.assert_array_equal(recovered, mask)

    def test_rgb_to_id_preserves_fine_ids(self):
        mask = np.array([[8, 9, 10], [14, 15, 0]], dtype=np.int64)
        rgb = id_to_rgb(mask)
        recovered = rgb_to_id(rgb)
        np.testing.assert_array_equal(recovered, mask)

    def test_rgb_to_id_3d_required(self):
        rgb_2d = np.zeros((4, 4), dtype=np.uint8)
        with self.assertRaises(MaskIOError):
            rgb_to_id(rgb_2d)


class RGBMaskLoadSaveTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def test_save_and_load_rgb_roundtrip(self):
        mask = np.array([[0, 1, 2], [3, 7, 15]], dtype=np.int64)
        path = Path(self.tmpdir.name) / "rgb.png"
        save_rgb_mask(mask, path)

        loaded = load_rgb_mask(path)
        np.testing.assert_array_equal(loaded, mask)

    def test_load_missing_rgb_raises(self):
        with self.assertRaises(MaskIOError):
            load_rgb_mask("/nonexistent/rgb.png")


class ChangeRegionTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def test_save_bool_load_bool_roundtrip(self):
        region = np.array([[True, False], [False, True]], dtype=bool)
        path = Path(self.tmpdir.name) / "change.png"
        save_change_region(region, path)

        loaded = load_change_region(path)
        np.testing.assert_array_equal(loaded, region)

    def test_save_numeric_any_positive_is_changed(self):
        region = np.array([[0, 1], [128, 0]], dtype=np.uint8)
        path = Path(self.tmpdir.name) / "change.png"
        save_change_region(region, path)

        loaded = load_change_region(path)
        expected = np.array([[False, True], [True, False]], dtype=bool)
        np.testing.assert_array_equal(loaded, expected)

    def test_load_missing_raises(self):
        with self.assertRaises(MaskIOError):
            load_change_region("/nonexistent/change.png")

    def test_save_3d_raises(self):
        region = np.ones((4, 4, 2), dtype=bool)
        with self.assertRaises(MaskIOError):
            save_change_region(region, Path(self.tmpdir.name) / "bad.png")


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def test_save_and_load_roundtrip(self):
        meta = {"primitive": "tumor_burden_increase", "pixels": 42, "fraction": 0.15}
        path = Path(self.tmpdir.name) / "metadata.json"
        save_metadata(meta, path)

        loaded = load_metadata(path)
        self.assertEqual(loaded["primitive"], "tumor_burden_increase")
        self.assertEqual(loaded["pixels"], 42)
        self.assertAlmostEqual(loaded["fraction"], 0.15)

    def test_numpy_types_serialized(self):
        meta = {
            "count": np.int64(10),
            "fraction": np.float64(0.25),
            "array": np.array([1, 2, 3]),
        }
        path = Path(self.tmpdir.name) / "meta.json"
        save_metadata(meta, path)

        loaded = load_metadata(path)
        self.assertEqual(loaded["count"], 10)
        self.assertAlmostEqual(loaded["fraction"], 0.25)
        self.assertEqual(loaded["array"], [1, 2, 3])

    def test_load_missing_raises(self):
        with self.assertRaises(MaskIOError):
            load_metadata("/nonexistent/meta.json")

    def test_save_creates_parent_dirs(self):
        path = Path(self.tmpdir.name) / "deep" / "meta.json"
        save_metadata({"key": "val"}, path)
        self.assertTrue(path.exists())


class SaveEditOutputTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def test_full_bundle_roundtrip(self):
        src_mask = np.array([[0, 1, 2], [3, 7, 15]], dtype=np.int64)
        tar_mask = np.array([[1, 1, 2], [3, 7, 0]], dtype=np.int64)
        change = np.array([[True, True, False], [False, False, True]], dtype=bool)
        ops_log = {"primitive": "test", "selected_pixels": 3}
        warnings = ("warning_a",)

        paths = save_edit_output(
            src_mask=src_mask,
            target_mask=tar_mask,
            change_region=change,
            ops_log=ops_log,
            warnings=warnings,
            output_dir=self.tmpdir.name,
        )

        self.assertTrue(paths["src_mask"].exists())
        self.assertTrue(paths["tar_mask"].exists())
        self.assertTrue(paths["change_region"].exists())
        self.assertTrue(paths["src_mask_rgb"].exists())
        self.assertTrue(paths["tar_mask_rgb"].exists())
        self.assertTrue(paths["metadata"].exists())

        loaded_src = load_id_mask(paths["src_mask"])
        loaded_tar = load_id_mask(paths["tar_mask"])
        loaded_change = load_change_region(paths["change_region"])

        np.testing.assert_array_equal(loaded_src, src_mask)
        np.testing.assert_array_equal(loaded_tar, tar_mask)
        np.testing.assert_array_equal(loaded_change, change)

        meta = load_metadata(paths["metadata"])
        self.assertEqual(meta["ops_log"]["primitive"], "test")
        self.assertEqual(meta["warnings"], ["warning_a"])
        self.assertEqual(meta["change_region_pixels"], 3)
        self.assertAlmostEqual(meta["changed_area_fraction"], 3 / 6)

    def test_rgb_files_are_valid_images(self):
        src_mask = np.zeros((4, 4), dtype=np.int64)
        tar_mask = np.ones((4, 4), dtype=np.int64)
        change = np.zeros((4, 4), dtype=bool)
        ops_log = {"primitive": "empty"}

        paths = save_edit_output(
            src_mask=src_mask,
            target_mask=tar_mask,
            change_region=change,
            ops_log=ops_log,
            output_dir=self.tmpdir.name,
        )

        src_rgb = Image.open(paths["src_mask_rgb"])
        tar_rgb = Image.open(paths["tar_mask_rgb"])
        self.assertEqual(src_rgb.mode, "RGB")
        self.assertEqual(tar_rgb.mode, "RGB")
        self.assertEqual(src_rgb.size, (4, 4))
        self.assertEqual(tar_rgb.size, (4, 4))
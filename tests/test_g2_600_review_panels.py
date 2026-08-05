from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.build_g2_600_review_panels import (
    draw_changed_boundary,
    preflight_records,
    tissue_nuclei_overlay,
)


class G2600ReviewPanelTests(unittest.TestCase):
    def test_preflight_rejects_change_below_full_image_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = np.full((100, 100), 2, dtype=np.uint8)
            target = source.copy()
            target[:4] = 1
            nuclei = np.zeros_like(source)
            image = np.full((100, 100, 3), 180, dtype=np.uint8)
            paths = {}
            for name, array in (
                ("source", source),
                ("target", target),
                ("nuclei", nuclei),
                ("image", image),
            ):
                path = root / f"{name}.png"
                Image.fromarray(array).save(path)
                paths[name] = path

            def digest(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()

            record = {
                "case_id": "case-a",
                "source_tissue_mask": str(paths["source"]),
                "source_tissue_sha256": digest(paths["source"]),
                "target_tissue_mask": str(paths["target"]),
                "target_tissue_sha256": digest(paths["target"]),
                "target_nuclei_mask": str(paths["nuclei"]),
                "target_nuclei_sha256": digest(paths["nuclei"]),
                "selected_image": str(paths["image"]),
                "selected_image_sha256": digest(paths["image"]),
            }
            preflight = preflight_records(
                [record], minimum_change_fraction=0.05
            )
            self.assertFalse(preflight["passed"])
            self.assertIn(
                "changed fraction", preflight["failure_reasons"][0]
            )

    def test_nuclei_are_opaque_over_tissue_overlay(self):
        image = Image.fromarray(np.full((8, 8, 3), 200, dtype=np.uint8))
        tissue = np.full((8, 8), 2, dtype=np.uint8)
        nuclei = np.zeros((8, 8), dtype=np.uint8)
        nuclei[3, 4] = 102
        result = np.asarray(tissue_nuclei_overlay(image, tissue, nuclei))
        self.assertEqual(result[3, 4].tolist(), [0, 255, 0])

    def test_boundary_visualization_does_not_fill_changed_region(self):
        image = Image.fromarray(np.full((32, 32, 3), 200, dtype=np.uint8))
        changed = np.zeros((32, 32), dtype=bool)
        changed[8:24, 8:24] = True
        result = np.asarray(draw_changed_boundary(image, changed))
        self.assertEqual(result[16, 16].tolist(), [200, 200, 200])
        self.assertEqual(result[8, 16].tolist(), [255, 220, 0])


if __name__ == "__main__":
    unittest.main()

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "prototype_cross_v2_2_layered_condition.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "prototype_cross_v2_2_layered_condition",
    _MODULE_PATH,
)
prototype = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = prototype
_SPEC.loader.exec_module(prototype)


class CrossV22LayeredConditionPrototypeTests(unittest.TestCase):
    def test_same_tissue_fill_does_not_cross_label_when_sources_exist(self):
        rgb = np.zeros((3, 4, 3), dtype=np.uint8)
        rgb[:, :2] = [10, 20, 30]
        rgb[:, 2:] = [200, 210, 220]
        tissue = np.array(
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [1, 1, 2, 2],
            ],
            dtype=np.int64,
        )
        hole = np.zeros((3, 4), dtype=bool)
        hole[1, 1] = True
        hole[1, 2] = True

        filled, report = prototype.remove_nuclei_with_same_tissue_fill(
            rgb,
            hole,
            tissue,
            feather_radius=0.0,
        )

        self.assertTrue(np.array_equal(filled[1, 1], [10, 20, 30]))
        self.assertTrue(np.array_equal(filled[1, 2], [200, 210, 220]))
        self.assertEqual(report["same_tissue_filled_pixels"], 2)

    def test_fill_patch_outside_mask_uses_nearest_nucleus_pixels(self):
        rgb = np.zeros((3, 3, 3), dtype=np.uint8)
        rgb[1, 1] = [100, 50, 25]
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True

        filled = prototype.fill_patch_outside_mask_with_nearest(rgb, mask)

        self.assertTrue(np.all(filled == np.array([100, 50, 25], dtype=np.uint8)))

    def test_synthesize_nuclei_layer_pastes_matching_label(self):
        proto_rgb = np.zeros((3, 3, 3), dtype=np.uint8)
        proto_rgb[:, :] = [80, 20, 120]
        alpha = np.zeros((3, 3), dtype=np.uint8)
        alpha[1, 1] = 255
        prototypes = [
            prototype.NucleusPrototype(
                label=2,
                area=1,
                bbox=(0, 3, 0, 3),
                rgb=proto_rgb,
                alpha=alpha,
            )
        ]
        target = np.zeros((5, 5), dtype=np.int64)
        target[2, 2] = 2

        rgba, report = prototype.synthesize_nuclei_layer(
            target_nuclei=target,
            prototypes=prototypes,
            seed=1,
            alpha_feather=0.0,
        )

        self.assertEqual(report["pasted_component_count"], 1)
        self.assertGreater(int(rgba[..., 3].max()), 0)
        self.assertTrue(np.array_equal(rgba[2, 2, :3], [80, 20, 120]))


if __name__ == "__main__":
    unittest.main()

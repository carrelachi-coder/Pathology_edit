from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from segmentator.release import (
    load_segmentator_release,
    release_model_kwargs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_PATH = (
    REPO_ROOT
    / "benchmark_configs"
    / "releases"
    / "segmentator_fine_c_epoch2.json"
)


class SegmentatorReleaseTests(unittest.TestCase):
    def test_c_line_release_reconstructs_the_strict_runtime_architecture(self):
        release = load_segmentator_release(
            RELEASE_PATH,
            verify_checkpoint=False,
        )
        kwargs = release_model_kwargs(release)

        self.assertEqual(
            release["release_id"],
            "segmentator-fine-c-joint-epoch2-v1",
        )
        self.assertEqual(release["runtime"]["runtime_inputs"], "image_only")
        self.assertFalse(release["runtime"]["cellvit_required_at_inference"])
        self.assertTrue(release["runtime"]["strict_checkpoint_load"])
        self.assertEqual(kwargs["decoder"], "mask2former")
        self.assertTrue(kwargs["hierarchical_fine"])
        self.assertTrue(kwargs["boundary_refinement"])
        self.assertEqual(kwargs["refinement_gate_mode"], "learned_soft")
        self.assertEqual(kwargs["cellvit_mode"], "teacher")

    def test_checkpoint_path_can_be_overridden_for_hub_deployment(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "segmentator.pt"
            checkpoint.write_bytes(b"hub-checkpoint")
            with patch.dict(
                os.environ,
                {"PATHOLOGY_SEGMENTATOR_CHECKPOINT": str(checkpoint)},
            ):
                release = load_segmentator_release(
                    RELEASE_PATH,
                    verify_checkpoint=False,
                )

        self.assertEqual(release["checkpoint"], str(checkpoint))
        self.assertEqual(
            release["_checkpoint_environment_selector"],
            "PATHOLOGY_SEGMENTATOR_CHECKPOINT",
        )


if __name__ == "__main__":
    unittest.main()

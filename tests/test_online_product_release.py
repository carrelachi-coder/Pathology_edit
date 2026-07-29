import json
import unittest
from pathlib import Path

from controlnet_train.inference.model_paths import (
    DEFAULT_CROSS_V1_CHECKPOINT,
    DEFAULT_INPAINT_CHECKPOINT,
    FROZEN_CELLVIT_SHA256,
    FROZEN_PROBNET_SHA256,
    PRODUCTION_CONTROLNET_RELEASES,
    PRODUCTION_PIX2PIX_EPOCH,
    PRODUCTION_PIX2PIX_GLOBAL_STEP,
    PRODUCTION_PIX2PIX_SHA256,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCT_RELEASE = (
    REPO_ROOT
    / "benchmark_configs"
    / "releases"
    / "online_agent_product_v1.json"
)
SEGMENTATOR_RELEASE = (
    REPO_ROOT
    / "benchmark_configs"
    / "releases"
    / "segmentator_fine_c_epoch2.json"
)


class OnlineProductReleaseTests(unittest.TestCase):
    def test_manifest_matches_runtime_model_pins(self):
        release = json.loads(PRODUCT_RELEASE.read_text(encoding="utf-8"))
        segmentator = json.loads(SEGMENTATOR_RELEASE.read_text(encoding="utf-8"))

        nuclei = release["nuclei_generation"]
        generation = release["image_generation"]
        verification = release["verification"]
        self.assertRegex(release["code_commit"], r"^[0-9a-f]{40}$")
        self.assertEqual(release["code_commit"], segmentator["code_commit"])
        self.assertEqual(nuclei["checkpoint_sha256"], FROZEN_PROBNET_SHA256)
        self.assertEqual(
            nuclei["candidate_queue_policy"],
            "stable_descending_probnet_score",
        )
        self.assertEqual(
            generation["pix2pix"]["checkpoint_sha256"],
            PRODUCTION_PIX2PIX_SHA256,
        )
        self.assertEqual(
            generation["pix2pix"]["epoch"],
            PRODUCTION_PIX2PIX_EPOCH,
        )
        self.assertEqual(
            generation["pix2pix"]["global_step"],
            PRODUCTION_PIX2PIX_GLOBAL_STEP,
        )
        self.assertEqual(
            verification["cellvit_checkpoint_sha256"],
            FROZEN_CELLVIT_SHA256,
        )
        self.assertEqual(
            verification["segmentator_release_id"],
            segmentator["release_id"],
        )
        self.assertEqual(
            verification["segmentator_checkpoint_sha256"],
            segmentator["checkpoint_sha256"],
        )

    def test_manifest_matches_packaged_controlnet_defaults(self):
        release = json.loads(PRODUCT_RELEASE.read_text(encoding="utf-8"))
        generation = release["image_generation"]

        self.assertEqual(
            generation["inpaint"]["checkpoint"],
            DEFAULT_INPAINT_CHECKPOINT,
        )
        self.assertEqual(
            generation["cross_v1"]["checkpoint"],
            DEFAULT_CROSS_V1_CHECKPOINT,
        )
        for mode, key in (("inpaint", "inpaint"), ("cross-v1", "cross_v1")):
            expected = PRODUCTION_CONTROLNET_RELEASES[mode]
            actual = generation[key]
            self.assertEqual(
                actual["weight_size_bytes"],
                expected["weight_size_bytes"],
            )
            self.assertEqual(
                actual["weight_sha256"],
                expected["weight_sha256"],
            )

    def test_release_entrypoints_exist(self):
        release = json.loads(PRODUCT_RELEASE.read_text(encoding="utf-8"))

        for relative_path in release["entrypoints"].values():
            self.assertTrue((REPO_ROOT / relative_path).is_file(), relative_path)


if __name__ == "__main__":
    unittest.main()

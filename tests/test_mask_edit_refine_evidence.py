"""Evidence manifest and annotation-profile statistics tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from phase3_mask_edit_refine.evidence import (
    EvidenceManifest,
    build_annotation_profile_statistics,
    sha256_file,
    verify_case_run_bundle,
)
from phase3_mask_edit_refine.models import RefineContractError
from phase3_mask_edit_refine.skills import SkillRepository


class EvidenceStatisticsTests(unittest.TestCase):
    def test_run_bundle_requires_all_original_artifacts_and_digests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = {}
            for name in (
                "source_image",
                "source_mask",
                "target_mask",
                "instruction",
                "planner_response",
                "run_manifest",
                "code_snapshot",
            ):
                path = root / f"{name}.json"
                path.write_text("{}", encoding="utf-8")
                artifacts[name] = {"path": str(path), "sha256": sha256_file(path)}
            manifest_path = root / "bundle.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "mask-edit-refine-run-bundle-v1",
                        "case_id": "152",
                        "artifacts": artifacts,
                    }
                ),
                encoding="utf-8",
            )
            report = verify_case_run_bundle(manifest_path)
            self.assertTrue(report["passed"])
            self.assertEqual(len(report["verified_artifacts"]), 7)

    def test_statistics_capture_fragmented_background_and_patient_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks = []
            for index in range(2):
                mask = np.full((32, 32), 2, dtype=np.int64)
                mask[8:24, 8:24] = 12
                mask[::4, ::4] = 0
                path = root / f"mask_{index}.npy"
                np.save(path, mask, allow_pickle=False)
                masks.append(path)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "annotation_profile_id": "glas-gland-v1",
                        "dataset_revision": "synthetic-v1",
                        "protocol_sources": ["fixture"],
                        "records": [
                            {
                                "record_id": f"r{index}",
                                "mask_uri": str(path),
                                "patient_id": f"p{index}",
                                "wsi_id": f"w{index}",
                                "split": "train",
                            }
                            for index, path in enumerate(masks)
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest = EvidenceManifest.load(manifest_path)
            schema = SkillRepository().annotation_schema("glas-gland-v1")
            stats = build_annotation_profile_statistics(manifest, schema=schema)
            self.assertEqual(stats["record_count"], 2)
            self.assertEqual(stats["patient_count"], 2)
            self.assertGreater(stats["background_components_per_mpx"]["p50"], 0)
            self.assertIn("Stroma|Tumor", stats["adjacency_patch_counts"])

    def test_manifest_rejects_patient_split_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "annotation_profile_id": "glas-gland-v1",
                        "dataset_revision": "x",
                        "protocol_sources": [],
                        "records": [
                            {"record_id": "a", "mask_uri": "a.npy", "patient_id": "p", "wsi_id": "w1", "split": "train"},
                            {"record_id": "b", "mask_uri": "b.npy", "patient_id": "p", "wsi_id": "w2", "split": "test"}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RefineContractError, "split leakage"):
                EvidenceManifest.load(path)


if __name__ == "__main__":
    unittest.main()

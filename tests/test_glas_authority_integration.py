"""Integration regressions for the GLaS two-level authority contract.

These tests are intended to run inside the complete Pathology_edit checkout.
They deliberately exercise the existing joint scene/auxiliary pipeline rather
than reimplementing it in test fixtures.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from phase3_joint_edit_refine.authority import (
    GLAND_AUTHORITY_SEMANTIC_PROXY,
    NUCLEUS_AUTHORITY_HYBRID,
    gland_instance_authority_status,
)
from phase3_joint_edit_refine.auxiliary import materialize_profile_auxiliaries
from phase3_joint_edit_refine.models import CellCountExtentBudget, JointCaseContext
from phase3_joint_edit_refine.scene import build_joint_scene_analysis
from phase3_mask_edit.core.labels import MaskProfileSchema


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(root: Path, *, tissue_path: Path, nuclei_path: Path) -> JointCaseContext:
    image_path = root / "image.png"
    Image.new("RGB", (32, 32), "white").save(image_path)
    return JointCaseContext(
        case_id="glas-authority-regression",
        instruction="Add scattered tumor cells near the tumor boundary.",
        source_image_uri=str(image_path),
        source_tissue_mask_uri=str(tissue_path),
        source_nuclei_mask_uri=str(nuclei_path),
        pathology_domain_id="colorectal-adenocarcinoma-v1",
        annotation_profile_id="glas-gland-v1",
        cell_observation_profile_id="cellvit-five-class-v1",
        cell_population_profile_id="colorectal-cellvit-source-first-v1",
        primitive_id="peritumoral-neoplastic-scatter-increase-v1",
        joint_area_budget=None,
        cell_count_extent_budget=CellCountExtentBudget(
            6, 4, 8, 96, 4, 48, 20, 3
        ),
        seed=17,
        provenance={
            "source_image_sha256": _sha(image_path),
            "source_tissue_mask_sha256": _sha(tissue_path),
            "source_nuclei_mask_sha256": _sha(nuclei_path),
            "preprocessing_revision": "test-v1",
            "original_label_map_digest": _sha(tissue_path),
            "patch_grade": "moderately_differentiated",
        },
    )


class GlaSAuthorityIntegrationTests(unittest.TestCase):
    def test_generated_gland_component_map_is_proxy_not_original_instance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tissue = np.full((32, 32), 2, dtype=np.uint8)
            tissue[8:24, 8:24] = 12
            nuclei = np.zeros_like(tissue)
            tissue_path = root / "tissue.png"
            nuclei_path = root / "nuclei.png"
            Image.fromarray(tissue).save(tissue_path)
            Image.fromarray(nuclei).save(nuclei_path)
            case = _case(root, tissue_path=tissue_path, nuclei_path=nuclei_path)

            effective, produced = materialize_profile_auxiliaries(
                case,
                source_tissue=tissue,
                output_dir=root / "auxiliary",
            )

            gland = next(
                item
                for item in produced
                if item.structure_id == "native_gland_instance_map"
            )
            self.assertEqual(
                effective.provenance["gland_instance_authority_kind"],
                GLAND_AUTHORITY_SEMANTIC_PROXY,
            )
            self.assertEqual(
                effective.provenance["derived_gland_component_map_sha256"],
                gland.sha256,
            )
            self.assertNotIn(
                "original_instance_mask_digest", effective.provenance
            )
            self.assertTrue(
                gland_instance_authority_status(effective.provenance)["valid"]
            )

    def test_hybrid_raster_scene_reports_hybrid_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tissue = np.full((32, 32), 2, dtype=np.uint8)
            semantic = np.zeros_like(tissue)
            semantic[5:10, 5:10] = 1
            semantic[20:24, 20:24] = 1
            labels = np.zeros_like(tissue, dtype=np.int32)
            labels[5:10, 5:10] = 1
            labels[20:24, 20:24] = 2
            label_map = root / "instances.npy"
            np.save(label_map, labels, allow_pickle=False)
            manifest = root / "instances.json"
            manifest.write_text(
                json.dumps(
                    {
                        "raster_instance_authority": {
                            "label_map_uri": label_map.name,
                            "label_map_sha256": _sha(label_map),
                            "instances": [
                                {
                                    "label_id": 1,
                                    "type": 1,
                                    "seed_source": "cellvit",
                                },
                                {
                                    "label_id": 2,
                                    "type": 1,
                                    "seed_source": "semantic_unseeded",
                                },
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )

            scene = build_joint_scene_analysis(
                tissue,
                semantic,
                schema=MaskProfileSchema.from_reference_profile("GLaS"),
                pixel_size_um=0.25,
                nuclei_instances_path=str(manifest),
            )

            self.assertEqual(
                scene.cells.observation_quality,
                NUCLEUS_AUTHORITY_HYBRID,
            )
            authority = scene.to_metadata()["nucleus_instance_authority"]
            self.assertEqual(authority["native_seed_instance_count"], 1)
            self.assertEqual(authority["semantic_residual_instance_count"], 1)

    def test_review_runner_exposes_probnet_device(self):
        from scripts.run_glas_primitive_mask_review import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "--cross-meta-eval",
                "cross-meta.json",
                "--output-dir",
                "output",
                "--probnet-checkpoint",
                "probnet.pt",
                "--nuclei-instance-library",
                "library",
                "--cellvit-model",
                "cellvit.pt",
                "--cellvit-root",
                "cellvit",
                "--cellvit-python",
                "python",
                "--probnet-device",
                "cuda:0",
            ]
        )

        self.assertEqual(args.probnet_device, "cuda:0")


if __name__ == "__main__":
    unittest.main()

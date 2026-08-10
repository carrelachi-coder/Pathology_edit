"""Read-only contracts for G2 pair qualification."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from phase3_joint_edit_refine.g2_qualification import qualify_g2_manifest
from phase3_mask_edit.core.labels import MaskProfileSchema


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class G2PairQualificationTests(unittest.TestCase):
    def test_qualification_writes_only_shadow_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            image_path = source / "image.png"
            tissue_path = source / "tissue.png"
            nuclei_path = source / "nuclei.png"
            Image.fromarray(
                np.full((32, 32, 3), [170, 95, 130], dtype=np.uint8)
            ).save(image_path)
            schema = MaskProfileSchema.from_reference_profile("BCSS")
            tumor_id = schema.resolve_fine_ids("Tumor")[0]
            stroma_id = schema.resolve_fine_ids("Stroma")[0]
            tissue = np.full((32, 32), stroma_id, dtype=np.uint8)
            tissue[:, :16] = tumor_id
            Image.fromarray(tissue).save(tissue_path)
            nuclei = np.zeros((32, 32), dtype=np.uint8)
            nuclei[5:8, 5:8] = 101
            nuclei[18:21, 21:24] = 103
            Image.fromarray(nuclei).save(nuclei_path)
            source_digests = {
                path: _sha(path)
                for path in (image_path, tissue_path, nuclei_path)
            }
            base = {
                "sample_id": "fixture",
                "organ": "breast",
                "dataset": "BCSS",
                "profile": "BCSS",
                "source_image": str(image_path),
                "source_tissue_mask": str(tissue_path),
                "source_nuclei_mask": str(nuclei_path),
                "source_mask_sha256": source_digests[tissue_path],
            }
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                **base,
                                "case_id": "g2-fixture-tumor",
                                "instruction": "Increase tumor burden",
                                "primitive": "tumor_burden_increase",
                                "g2_primitive": "tumor_increase",
                            },
                            {
                                **base,
                                "case_id": "g2-fixture-immune",
                                "instruction": "Increase immune infiltrate",
                                "primitive": "stromal_immune_infiltration",
                                "g2_primitive": "immune_increase",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = root / "shadow"
            result = qualify_g2_manifest(
                manifest,
                output_dir=output,
                board_page_size=2,
                tile_size=96,
            )

            self.assertEqual(result["records"], 2)
            rows = [
                json.loads(line)
                for line in Path(result["jsonl"]).read_text().splitlines()
            ]
            by_id = {row["case_id"]: row for row in rows}
            self.assertEqual(
                by_id["g2-fixture-tumor"]["qualification_status"],
                "ready_for_h&e_review",
            )
            self.assertEqual(
                by_id["g2-fixture-immune"]["qualification_status"],
                "requires_cell_only_review",
            )
            self.assertEqual(
                by_id["g2-fixture-immune"]["requested_semantics"][
                    "explicit_cell_class"
                ],
                "immune",
            )
            for path, digest in source_digests.items():
                self.assertEqual(_sha(path), digest)
            summary = json.loads(Path(result["summary"]).read_text())
            self.assertFalse(summary["h&e_decision_complete"])
            self.assertFalse(summary["read_only_contract"]["target_masks_generated"])


if __name__ == "__main__":
    unittest.main()

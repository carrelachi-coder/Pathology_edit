import json
import math
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np


_TMP_ROOT = Path("tmp_test_textbook_nuclei_summary")


def _write_npz(path: Path, area: int, nuc_type: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    side = max(1, int(round(math.sqrt(area))))
    mask = np.ones((side, side), dtype=bool)
    np.savez_compressed(
        path,
        mask=mask,
        type=np.array(nuc_type, dtype=np.int32),
        area=np.array(area, dtype=np.int32),
    )


class TextbookNucleiSummaryTests(unittest.TestCase):
    def test_summarize_nuclei_library_uses_instance_percentiles_and_density(self):
        tmpdir = _TMP_ROOT / f"case_{uuid.uuid4().hex}"
        try:
            library_dir = tmpdir / "BCSS"
            _write_npz(
                library_dir / "nuclei_instances" / "tissue_01_Tumor" / "000000.npz",
                area=100,
                nuc_type=101,
            )
            _write_npz(
                library_dir / "nuclei_instances" / "tissue_01_Tumor" / "000001.npz",
                area=400,
                nuc_type=101,
            )
            _write_npz(
                library_dir / "nuclei_instances" / "tissue_02_Stroma" / "000000.npz",
                area=225,
                nuc_type=103,
            )
            (library_dir / "statistics.json").write_text(
                json.dumps(
                    {
                        "dataset": "BCSS",
                        "cancer_type": "breast",
                        "cancer_type_index": 0,
                        "statistics": {
                            "1": {
                                "name": "Tumor",
                                "total_area_pixels": 10000,
                                "total_nuclei": 2,
                                "density_per_10k_px": 2.0,
                                "nuclei_types": {"101": {"count": 2, "fraction": 1.0}},
                            },
                            "2": {
                                "name": "Stroma",
                                "total_area_pixels": 20000,
                                "total_nuclei": 1,
                                "density_per_10k_px": 0.5,
                                "nuclei_types": {"103": {"count": 1, "fraction": 1.0}},
                            },
                        },
                    }
                ),
                encoding="utf8",
            )

            from scripts.textbook_nuclei_summary import summarize_nuclei_library

            summary = summarize_nuclei_library(library_dir, target_mpp=0.25)

            self.assertEqual(summary["dataset"], "BCSS")
            self.assertEqual(summary["cancer_type"], "breast")
            self.assertEqual(summary["library_key"], "breast")
            self.assertEqual(summary["stored_instance_count"], 3)
            self.assertEqual(summary["stored_instance_count_by_type"]["101"], 2)
            self.assertEqual(summary["stored_instance_count_by_type"]["103"], 1)
            self.assertEqual(summary["nucleus_area_px_median"], 225.0)
            self.assertEqual(summary["neoplastic_area_px_median"], 250.0)
            self.assertAlmostEqual(summary["density_per_10k_px_weighted"], 1.0)
            self.assertAlmostEqual(
                summary["expected_nuclei_per_512_patch_weighted"],
                512 * 512 / 10000,
            )
            self.assertEqual(summary["tumor_density_per_10k_px"], 2.0)
            self.assertEqual(summary["tumor_neoplastic_fraction"], 1.0)
            self.assertGreater(summary["nucleus_diameter_px_median"], 0)
            self.assertAlmostEqual(
                summary["nucleus_diameter_um_median"],
                summary["nucleus_diameter_px_median"] * 0.25,
                places=4,
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_write_nuclei_summary_files_creates_cancer_key_json(self):
        tmpdir = _TMP_ROOT / f"case_{uuid.uuid4().hex}"
        try:
            library_root = tmpdir / "libraries"
            library_dir = library_root / "ORCA"
            _write_npz(
                library_dir / "nuclei_instances" / "tissue_01_Tumor" / "000000.npz",
                area=144,
                nuc_type=101,
            )
            (library_dir / "statistics.json").write_text(
                json.dumps(
                    {
                        "dataset": "ORCA",
                        "cancer_type": "oral_scc",
                        "cancer_type_index": 5,
                        "statistics": {
                            "1": {
                                "name": "Tumor",
                                "total_area_pixels": 10000,
                                "total_nuclei": 1,
                                "density_per_10k_px": 1.0,
                                "nuclei_types": {"101": {"count": 1, "fraction": 1.0}},
                            }
                        },
                    }
                ),
                encoding="utf8",
            )

            from scripts.textbook_nuclei_summary import write_nuclei_summary_files

            output_dir = tmpdir / "out"
            written = write_nuclei_summary_files(library_root, output_dir, target_mpp=0.25)

            self.assertEqual([path.name for path in written], ["oral.json"])
            payload = json.loads((output_dir / "oral.json").read_text(encoding="utf8"))
            self.assertEqual(payload["library_key"], "oral")
            self.assertEqual(payload["dataset"], "ORCA")
            self.assertEqual(payload["target_mpp"], 0.25)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

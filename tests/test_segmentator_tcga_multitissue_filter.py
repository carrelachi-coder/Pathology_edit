import unittest
from pathlib import Path

import numpy as np

from scripts.run_segmentator_tcga_multitissue_filter import (
    DEFAULT_LABELS,
    build_segmentator_prompt,
    parse_tcga_barcode,
    project_from_path,
    resolve_organ_info,
    summarize_prediction_counts,
)


class SegmentatorTcgaMultiTissueFilterTests(unittest.TestCase):
    def test_parse_tcga_slide_and_case_barcode(self):
        parsed = parse_tcga_barcode(
            "TCGA-2A-A8VO-01Z-00-DX1.3A69CC37-B066-4529-B1BE-AD155983AAA7_7488_42336.png"
        )

        self.assertEqual(parsed.slide_barcode, "TCGA-2A-A8VO-01Z-00-DX1")
        self.assertEqual(parsed.case_barcode, "TCGA-2A-A8VO")

    def test_project_from_path_resolves_tcga_project_code(self):
        project = project_from_path(Path("/data/TCGA-PRAD/images/TCGA-2A-A8VO-01Z-00-DX1.png"))

        self.assertEqual(project, "PRAD")

    def test_resolve_organ_from_path_project(self):
        info = resolve_organ_info(
            Path("/data/TCGA-PRAD/images/TCGA-2A-A8VO-01Z-00-DX1.png"),
            metadata={},
            gdc_cases={},
        )

        self.assertEqual(info.organ, "prostate")
        self.assertEqual(info.project_id, "PRAD")
        self.assertEqual(info.source, "path_project")

    def test_prompt_contains_organ_specific_disease(self):
        prompt = build_segmentator_prompt("prostate", "PRAD")

        self.assertIn("prostate adenocarcinoma", prompt)
        self.assertIn("tumor, stroma, necrosis", prompt)

    def test_rejects_tiny_second_tissue_fraction(self):
        counts = np.zeros(len(DEFAULT_LABELS), dtype=np.int64)
        counts[1] = 9990
        counts[2] = 10

        summary = summarize_prediction_counts(
            counts,
            min_class_fraction=0.01,
            min_class_pixels=1,
            min_foreground_fraction=0.0,
        )

        self.assertFalse(summary["selected"])
        self.assertEqual(summary["qualifying_tissue_count"], 1)

    def test_accepts_two_nontrivial_tissue_classes(self):
        counts = np.zeros(len(DEFAULT_LABELS), dtype=np.int64)
        counts[1] = 8500
        counts[2] = 1500

        summary = summarize_prediction_counts(
            counts,
            min_class_fraction=0.01,
            min_class_pixels=256,
            min_foreground_fraction=0.0,
        )

        self.assertTrue(summary["selected"])
        self.assertEqual(summary["qualifying_tissue_count"], 2)


if __name__ == "__main__":
    unittest.main()

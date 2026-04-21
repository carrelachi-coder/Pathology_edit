import json
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
from PIL import Image


_TMP_ROOT = Path.cwd() / ".tmp_testdata"
_TMP_ROOT.mkdir(exist_ok=True)


def _write_mask(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8)).save(path)


def _write_rgb(path: Path, value: int) -> None:
    arr = np.full((8, 8, 3), value, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


class Phase5PatchAnalysisTests(unittest.TestCase):
    def test_summarize_patch_categories_detects_single_stroma_mixed_and_tumor_rich(self):
        from controlnet_train.cli.analyze_phase5_patch_distribution import summarize_patch_categories

        pure_stroma = np.full((8, 8), 2, dtype=np.uint8)
        mixed = np.full((8, 8), 2, dtype=np.uint8)
        mixed[:, :4] = 1
        tumor_rich = np.full((8, 8), 2, dtype=np.uint8)
        tumor_rich[:6, :] = 1

        pure_stroma_summary = summarize_patch_categories("BCSS", pure_stroma)
        mixed_summary = summarize_patch_categories("BCSS", mixed)
        tumor_rich_summary = summarize_patch_categories("BCSS", tumor_rich)

        self.assertTrue(pure_stroma_summary["single_tissue_patch"])
        self.assertTrue(pure_stroma_summary["pure_stroma_patch"])
        self.assertFalse(pure_stroma_summary["mixed_patch"])

        self.assertFalse(mixed_summary["single_tissue_patch"])
        self.assertFalse(mixed_summary["pure_stroma_patch"])
        self.assertTrue(mixed_summary["mixed_patch"])

        self.assertTrue(tumor_rich_summary["tumor_rich_patch"])
        self.assertGreaterEqual(tumor_rich_summary["tumor_ratio"], 0.5)

    def test_analyze_dataset_root_counts_patch_types_and_missing_pairs(self):
        from controlnet_train.cli.analyze_phase5_patch_distribution import analyze_dataset_root

        tmpdir = _TMP_ROOT / f"case_{uuid.uuid4().hex}"
        try:
            root = tmpdir / "BCSS"
            (root / "images").mkdir(parents=True)
            (root / "tissue_masks").mkdir()
            (root / "nuclei_masks").mkdir()

            _write_mask(root / "tissue_masks" / "pure_stroma_py0_px0.png", np.full((8, 8), 2, dtype=np.uint8))
            _write_mask(root / "tissue_masks" / "mixed_py0_px0.png", np.block([
                [np.full((4, 8), 1, dtype=np.uint8)],
                [np.full((4, 8), 2, dtype=np.uint8)],
            ]))
            _write_mask(root / "tissue_masks" / "tumor_rich_py0_px0.png", np.full((8, 8), 1, dtype=np.uint8))

            for name in ("pure_stroma_py0_px0", "mixed_py0_px0"):
                _write_rgb(root / "images" / f"{name}.png", 80)
                _write_mask(root / "nuclei_masks" / f"{name}.png", np.full((8, 8), 101, dtype=np.uint8))

            summary = analyze_dataset_root("BCSS", root)

            self.assertEqual(summary["patch_count"], 3)
            self.assertEqual(summary["paired_patch_count"], 2)
            self.assertEqual(summary["missing_image_count"], 1)
            self.assertEqual(summary["missing_nuclei_count"], 1)
            self.assertEqual(summary["single_tissue_patch_count"], 2)
            self.assertEqual(summary["pure_stroma_patch_count"], 1)
            self.assertEqual(summary["mixed_patch_count"], 1)
            self.assertEqual(summary["tumor_rich_patch_count"], 1)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_write_summary_writes_json_payload(self):
        from controlnet_train.cli.analyze_phase5_patch_distribution import write_summary

        tmpdir = _TMP_ROOT / f"case_{uuid.uuid4().hex}"
        try:
            output_path = tmpdir / "summary.json"
            payload = {"datasets": [{"dataset": "BCSS", "patch_count": 3}]}

            write_summary(output_path, payload)

            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf8")),
                payload,
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

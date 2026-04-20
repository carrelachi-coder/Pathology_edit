import json
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from controlnet_train.data.common import load_layered_dataset_samples
from controlnet_train.data.cross import CrossReconstructionDataset, build_cross_metadata
from controlnet_train.data.inpaint import InpaintDataset, build_inpaint_metadata

_TMP_ROOT = Path(__file__).resolve().parents[3] / ".tmp_testdata"
_TMP_ROOT.mkdir(exist_ok=True)


def _write_rgb(path: Path, value: int) -> None:
    arr = np.full((8, 8, 3), value, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _write_mask(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8)).save(path)


class CommonLayeredDataTests(unittest.TestCase):
    def test_load_layered_dataset_samples_reads_metadata_and_parses_case_id(self):
        tmpdir = _TMP_ROOT / f"case_{uuid.uuid4().hex}"
        try:
            root = tmpdir / "BCSS"
            (root / "images").mkdir(parents=True)
            (root / "tissue_masks").mkdir()
            (root / "nuclei_masks").mkdir()

            image_name = "caseA_region1_py0_px256.png"
            _write_rgb(root / "images" / image_name, 32)
            _write_mask(root / "tissue_masks" / image_name, np.full((8, 8), 1, dtype=np.uint8))
            _write_mask(root / "nuclei_masks" / image_name, np.full((8, 8), 101, dtype=np.uint8))

            metadata_path = root / "metadata.jsonl"
            metadata_path.write_text(
                json.dumps(
                    {
                        "image": f"images\\{image_name}",
                        "conditioning_image": f"conditioning\\{image_name}",
                        "text": "custom prompt",
                    }
                )
                + "\n",
                encoding="utf8",
            )

            samples = load_layered_dataset_samples("BCSS", root)

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].prompt, "custom prompt")
            self.assertEqual(samples[0].case_id, "caseA_region1")
            self.assertEqual(samples[0].patch_y, 0)
            self.assertEqual(samples[0].patch_x, 256)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class InpaintDatasetTests(unittest.TestCase):
    def test_build_metadata_and_dataset_load_remapped_nuclei(self):
        tmpdir = _TMP_ROOT / f"case_{uuid.uuid4().hex}"
        try:
            root = Path(tmpdir)
            source_image = root / "source.png"
            target_image = root / "target.png"
            target_tissue = root / "target_tissue_mask.png"
            target_nuclei = root / "target_nuclei_mask.png"
            change_mask = root / "change_region_mask.png"
            input_jsonl = root / "input.jsonl"
            output_dir = root / "normalized"

            _write_rgb(source_image, 64)
            _write_rgb(target_image, 96)
            _write_mask(target_tissue, np.full((8, 8), 8, dtype=np.uint8))
            _write_mask(target_nuclei, np.full((8, 8), 105, dtype=np.uint8))

            binary = np.zeros((8, 8), dtype=np.uint8)
            binary[:, :4] = 255
            _write_mask(change_mask, binary)

            input_jsonl.write_text(
                json.dumps(
                    {
                        "dataset": "PANDA",
                        "source_image": str(source_image),
                        "target_image": str(target_image),
                        "target_tissue_mask": str(target_tissue),
                        "target_nuclei_mask": str(target_nuclei),
                        "change_region_mask": str(change_mask),
                        "edit_type": "gleason_upgrade_3to4",
                        "change_ratio": 0.5,
                    }
                )
                + "\n",
                encoding="utf8",
            )

            train_path, _ = build_inpaint_metadata(
                input_jsonl_paths=[input_jsonl],
                output_dir=output_dir,
                val_ratio=0.0,
                seed=7,
            )

            dataset = InpaintDataset(train_path)
            sample = dataset[0]

            self.assertEqual(sample["target_nuclei_mask"].dtype, torch.int64)
            self.assertEqual(int(sample["target_nuclei_mask"].max().item()), 5)
            self.assertEqual(tuple(sample["change_region_mask"].shape), (1, 8, 8))
            self.assertTrue(Path(sample["erased_source_image_path"]).exists())
            self.assertIn("prostate", sample["prompt"].lower())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class CrossDatasetTests(unittest.TestCase):
    def test_build_cross_metadata_pairs_same_case_with_required_fields(self):
        tmpdir = _TMP_ROOT / f"case_{uuid.uuid4().hex}"
        try:
            root = Path(tmpdir) / "BCSS"
            (root / "images").mkdir(parents=True)
            (root / "tissue_masks").mkdir()
            (root / "nuclei_masks").mkdir()

            sample_defs = [
                ("case1_py0_px0.png", 1, 101, 32),
                ("case1_py0_px256.png", 1, 101, 48),
                ("case1_py256_px0.png", 2, 103, 64),
                ("case2_py0_px0.png", 1, 101, 80),
            ]
            for name, tissue_id, nuclei_id, image_value in sample_defs:
                _write_rgb(root / "images" / name, image_value)
                _write_mask(root / "tissue_masks" / name, np.full((8, 8), tissue_id, dtype=np.uint8))
                _write_mask(root / "nuclei_masks" / name, np.full((8, 8), nuclei_id, dtype=np.uint8))

            metadata_path = root / "metadata.jsonl"
            with metadata_path.open("w", encoding="utf8") as f:
                for name, _, _, _ in sample_defs:
                    f.write(
                        json.dumps(
                            {
                                "image": f"images\\{name}",
                                "conditioning_image": f"conditioning\\{name}",
                                "text": "breast prompt",
                            }
                        )
                        + "\n"
                    )

            train_path, _ = build_cross_metadata(
                dataset_roots={"BCSS": root},
                output_dir=root / "cross_output",
                num_ref_per_target=1,
                val_ratio=0.0,
                seed=11,
                top_k=2,
            )

            dataset = CrossReconstructionDataset(train_path)
            sample = dataset[0]

            self.assertIn("reference_tissue_mask", sample)
            self.assertIn("reference_nuclei_mask", sample)
            self.assertEqual(sample["case_id"], "case1")
            self.assertNotEqual(sample["sample_id"], sample["reference_sample_id"])
            self.assertEqual(tuple(sample["reference_tissue_mask"].shape), (8, 8))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

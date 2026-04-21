import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from controlnet_train.data.common import load_layered_dataset_samples
from controlnet_train.data.cross import CrossReconstructionDataset, build_cross_metadata
from controlnet_train.data.inpaint import InpaintDataset, build_inpaint_metadata
from controlnet_train.data.inpaint_synthesis import (
    expand_band,
    _build_near_identity_mask,
    replace_like_blob,
    shrink_band,
    synthesize_change_region,
    build_synthetic_inpaint_metadata,
)

_TMP_ROOT = Path.cwd() / ".tmp_testdata"
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


class InpaintSynthesisTests(unittest.TestCase):
    def _make_component_mask(self) -> np.ndarray:
        tissue_mask = np.zeros((8, 8), dtype=np.uint8)
        tissue_mask[2:6, 2:6] = 1
        return tissue_mask

    def test_build_synthetic_metadata_identity_writes_trace_fields(self):
        tmpdir = _TMP_ROOT / f"case_{uuid.uuid4().hex}"
        try:
            root = Path(tmpdir) / "PANDA"
            (root / "images").mkdir(parents=True)
            (root / "tissue_masks").mkdir()
            (root / "nuclei_masks").mkdir()

            sample_name = "case1_py0_px0.png"
            _write_rgb(root / "images" / sample_name, 72)
            _write_mask(root / "tissue_masks" / sample_name, np.full((8, 8), 1, dtype=np.uint8))
            _write_mask(root / "nuclei_masks" / sample_name, np.full((8, 8), 101, dtype=np.uint8))

            metadata_path = root / "metadata.jsonl"
            metadata_path.write_text(
                json.dumps({"image": f"images\\{sample_name}", "text": "prostate prompt"}) + "\n",
                encoding="utf8",
            )

            train_path, _ = build_synthetic_inpaint_metadata(
                dataset_roots={"PANDA": root},
                output_dir=root / "synthetic_output",
                forced_mode="identity",
                val_ratio=0.0,
                seed=13,
            )

            rows = json.loads(train_path.read_text(encoding="utf8").splitlines()[0])

            self.assertEqual(rows["mask_mode"], "identity")
            self.assertEqual(rows["size_bucket"], "identity")
            self.assertEqual(float(rows["change_ratio"]), 0.0)
            self.assertEqual(rows["target_image"], rows["source_image"])
            self.assertEqual(rows["erased_source_image"], rows["source_image"])
            self.assertEqual(int(np.asarray(Image.open(rows["change_region_mask"])).sum()), 0)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_build_synthetic_metadata_rejects_invalid_forced_mode(self):
        tmpdir = _TMP_ROOT / f"case_{uuid.uuid4().hex}"
        try:
            root = Path(tmpdir) / "PANDA"
            (root / "images").mkdir(parents=True)
            (root / "tissue_masks").mkdir()
            (root / "nuclei_masks").mkdir()

            sample_name = "case_invalid_py0_px0.png"
            _write_rgb(root / "images" / sample_name, 72)
            _write_mask(root / "tissue_masks" / sample_name, np.full((8, 8), 1, dtype=np.uint8))
            _write_mask(root / "nuclei_masks" / sample_name, np.full((8, 8), 101, dtype=np.uint8))

            metadata_path = root / "metadata.jsonl"
            metadata_path.write_text(
                json.dumps({"image": f"images\\{sample_name}", "text": "prostate prompt"}) + "\n",
                encoding="utf8",
            )

            with patch(
                "controlnet_train.data.inpaint_synthesis.load_layered_dataset_samples",
                side_effect=AssertionError("dataset loading should not run for invalid forced_mode"),
            ):
                with self.assertRaises(ValueError):
                    build_synthetic_inpaint_metadata(
                        dataset_roots={"PANDA": root},
                        output_dir=root / "synthetic_output",
                        forced_mode="unsupported_mode",
                        val_ratio=0.0,
                        seed=13,
                    )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_synthesize_change_region_rejects_invalid_forced_bucket(self):
        tissue_mask = self._make_component_mask()

        with self.assertRaises(ValueError):
            synthesize_change_region(tissue_mask, forced_bucket="unsupported_bucket", seed=11)

    def test_synthetic_metadata_loads_through_inpaint_dataset_and_near_identity_has_change(self):
        tmpdir = _TMP_ROOT / f"case_{uuid.uuid4().hex}"
        try:
            root = Path(tmpdir) / "PANDA"
            (root / "images").mkdir(parents=True)
            (root / "tissue_masks").mkdir()
            (root / "nuclei_masks").mkdir()

            sample_name = "case2_py0_px0.png"
            _write_rgb(root / "images" / sample_name, 88)
            _write_mask(root / "tissue_masks" / sample_name, np.full((8, 8), 1, dtype=np.uint8))
            _write_mask(root / "nuclei_masks" / sample_name, np.full((8, 8), 102, dtype=np.uint8))

            metadata_path = root / "metadata.jsonl"
            metadata_path.write_text(
                json.dumps({"image": f"images\\{sample_name}", "text": "prostate prompt"}) + "\n",
                encoding="utf8",
            )

            train_path, _ = build_synthetic_inpaint_metadata(
                dataset_roots={"PANDA": root},
                output_dir=root / "synthetic_output",
                forced_mode="near_identity",
                val_ratio=0.0,
                seed=17,
            )

            dataset = InpaintDataset(train_path)
            sample = dataset[0]
            row = json.loads(train_path.read_text(encoding="utf8").splitlines()[0])

            self.assertEqual(len(dataset), 1)
            self.assertGreater(float(sample["change_ratio"]), 0.0)
            self.assertEqual(row["mask_mode"], "near_identity")
            self.assertEqual(row["size_bucket"], "small")
            self.assertGreater(float(row["change_ratio"]), 0.0)
            self.assertTrue(Path(sample["erased_source_image_path"]).exists())
            self.assertFalse(torch.equal(sample["source_image"], sample["erased_source_image"]))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_near_identity_mask_can_satisfy_requested_change_pixels(self):
        tissue_mask = np.zeros((4, 4), dtype=np.uint8)
        tissue_mask[3, 3] = 1

        mask = _build_near_identity_mask(tissue_mask, change_pixels=2)

        self.assertEqual(int(np.count_nonzero(mask)), 2)

    def test_expand_band_stays_near_exterior_boundary_and_keeps_center_clear(self):
        tissue_mask = np.zeros((8, 8), dtype=np.uint8)
        tissue_mask[0:4, 0:4] = 1

        mask = expand_band(tissue_mask, seed=11)

        self.assertGreater(int(np.count_nonzero(mask)), 0)
        self.assertEqual(int(mask[1, 1]), 0)
        self.assertTrue(np.any((mask > 0) & (tissue_mask == 0)))
        self.assertFalse(np.any((mask > 0) & (tissue_mask > 0)))

    def test_expand_band_returns_empty_for_full_frame_component(self):
        tissue_mask = np.ones((8, 8), dtype=np.uint8)

        mask = expand_band(tissue_mask, seed=11)

        self.assertEqual(int(np.count_nonzero(mask)), 0)

    def test_expand_band_prefers_the_largest_component_when_multiple_are_present(self):
        tissue_mask = np.zeros((8, 8), dtype=np.uint8)
        tissue_mask[1:4, 1:4] = 1
        tissue_mask[5:7, 5:7] = 1

        mask = expand_band(tissue_mask, seed=11)

        self.assertGreater(int(np.count_nonzero(mask)), 0)
        self.assertGreater(int(np.count_nonzero(mask[:5, :5])), 0)
        self.assertEqual(int(np.count_nonzero(mask[5:, 5:])), 0)

    def test_expand_band_tie_break_is_seeded_for_equal_components(self):
        tissue_mask = np.zeros((8, 8), dtype=np.uint8)
        tissue_mask[1:3, 1:3] = 1
        tissue_mask[5:7, 5:7] = 1

        first = expand_band(tissue_mask, seed=11)
        second = expand_band(tissue_mask, seed=11)

        self.assertTrue(np.array_equal(first, second))
        self.assertNotEqual(
            int(np.count_nonzero(first[:4, :4])) > 0,
            int(np.count_nonzero(first[5:, 5:])) > 0,
        )

    def test_shrink_band_preserves_component_core(self):
        tissue_mask = self._make_component_mask()

        mask = shrink_band(tissue_mask, seed=11)

        self.assertGreater(int(np.count_nonzero(mask)), 0)
        self.assertEqual(int(mask[4, 4]), 0)
        self.assertTrue(np.any((mask > 0) & (tissue_mask > 0)))
        self.assertFalse(np.any((mask > 0) & (tissue_mask == 0)))

    def test_shrink_band_never_absorbs_a_thin_component(self):
        tissue_mask = np.zeros((8, 8), dtype=np.uint8)
        tissue_mask[4, 2:7] = 1

        mask = shrink_band(tissue_mask, seed=11)

        self.assertGreater(int(np.count_nonzero(mask)), 0)
        self.assertLess(int(np.count_nonzero(mask)), int(np.count_nonzero(tissue_mask)))
        self.assertTrue(np.any((mask > 0) & (tissue_mask > 0)))
        self.assertFalse(np.any((mask > 0) & (tissue_mask == 0)))

    def test_replace_like_blob_attaches_to_boundary_without_center_hole(self):
        tissue_mask = self._make_component_mask()

        mask = replace_like_blob(tissue_mask, seed=11)

        self.assertGreater(int(np.count_nonzero(mask)), 0)
        self.assertEqual(int(mask[4, 4]), 0)
        self.assertFalse(np.any((mask > 0) & (tissue_mask == 0)))
        self.assertTrue(np.any((mask > 0) & (tissue_mask > 0)))

    def test_replace_like_blob_never_seeds_from_protected_core_on_thin_component(self):
        tissue_mask = np.zeros((8, 8), dtype=np.uint8)
        tissue_mask[4, 2:7] = 1

        mask = replace_like_blob(tissue_mask, seed=1)

        self.assertGreater(int(np.count_nonzero(mask)), 0)
        self.assertLess(int(np.count_nonzero(mask)), int(np.count_nonzero(tissue_mask)))
        self.assertFalse(np.any((mask > 0) & (tissue_mask == 0)))
        self.assertEqual(int(mask[4, 4]), 0)

    def test_synthesize_change_region_respects_forced_bucket(self):
        tissue_mask = self._make_component_mask()

        for forced_bucket in ("expand_band", "shrink_band", "replace_like_blob"):
            mask, bucket = synthesize_change_region(tissue_mask, forced_bucket=forced_bucket, seed=11)

            self.assertEqual(bucket, forced_bucket)
            self.assertGreater(int(np.count_nonzero(mask)), 0)


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

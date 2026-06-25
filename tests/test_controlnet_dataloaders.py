import contextlib
import json
import shutil
import unittest
import uuid
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import controlnet_train
from controlnet_train.cli import build_inpaint_dataset, generate_training_pairs
from controlnet_train.data import build_synthetic_inpaint_metadata as exported_build_synthetic_inpaint_metadata
from controlnet_train.data.common import load_layered_dataset_samples, normalize_metadata_path_value
from controlnet_train.data.cross import CrossReconstructionDataset, build_cross_metadata
from controlnet_train.data.inpaint import InpaintDataset, build_inpaint_metadata
from controlnet_train.data.inpaint_synthesis import (
    expand_band,
    _build_near_identity_mask,
    _size_bucket_for_change_ratio,
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
    def test_normalize_metadata_path_value_converts_windows_separators(self):
        self.assertEqual(
            normalize_metadata_path_value("images\\caseA_py0_px0.png"),
            "images/caseA_py0_px0.png",
        )

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


class InpaintCliTests(unittest.TestCase):
    def test_parse_args_rejects_both_input_jsonl_and_dataset_root(self):
        with contextlib.redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                build_inpaint_dataset.parse_args(
                    [
                        "--input-jsonl",
                        "D:/tmp/input.jsonl",
                        "--dataset-root",
                        "PANDA=D:/datasets/PANDA",
                        "--output-dir",
                        "D:/tmp/out",
                    ]
                )

    def test_main_rejects_synthetic_only_knobs_in_input_jsonl_mode(self):
        stderr = StringIO()
        with contextlib.redirect_stderr(stderr):
            with patch(
                "sys.argv",
                [
                    "build_inpaint_dataset.py",
                    "--input-jsonl",
                    "D:/tmp/input.jsonl",
                    "--output-dir",
                    "D:/tmp/out",
                    "--forced-mode",
                    "replace_like_blob",
                    "--samples-per-dataset",
                    "3",
                ],
            ):
                with self.assertRaises(SystemExit):
                    build_inpaint_dataset.main()

        self.assertIn("only supported with --dataset-root", stderr.getvalue())

    def test_parse_args_accepts_dataset_root_mode(self):
        args = build_inpaint_dataset.parse_args(
            [
                "--dataset-root",
                "PANDA=D:/datasets/PANDA",
                "--output-dir",
                "D:/tmp/out",
            ]
        )

        self.assertEqual(args.dataset_root, ["PANDA=D:/datasets/PANDA"])
        self.assertEqual(args.output_dir, Path("D:/tmp/out"))
        self.assertEqual(args.forced_mode, "mixed")
        self.assertIsNone(args.samples_per_dataset)
        self.assertIsNone(args.max_attempts_per_sample)
        self.assertIsNone(args.input_jsonl)

    def test_main_rejects_invalid_dataset_root_format_with_argparse_error(self):
        stderr = StringIO()
        with contextlib.redirect_stderr(stderr):
            with patch(
                "sys.argv",
                [
                    "build_inpaint_dataset.py",
                    "--dataset-root",
                    "PANDA",
                    "--output-dir",
                    "D:/tmp/out",
                ],
            ):
                with self.assertRaises(SystemExit):
                    build_inpaint_dataset.main()

        self.assertIn("Expected DATASET=PATH", stderr.getvalue())

    def test_parse_args_rejects_invalid_forced_mode_with_argparse_error(self):
        stderr = StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                build_inpaint_dataset.parse_args(
                    [
                        "--dataset-root",
                        "PANDA=D:/datasets/PANDA",
                        "--output-dir",
                        "D:/tmp/out",
                        "--forced-mode",
                        "unsupported_mode",
                    ]
                )

        self.assertIn("invalid choice", stderr.getvalue())

    def test_parse_args_rejects_invalid_forced_size_bucket_with_argparse_error(self):
        stderr = StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                build_inpaint_dataset.parse_args(
                    [
                        "--dataset-root",
                        "PANDA=D:/datasets/PANDA",
                        "--output-dir",
                        "D:/tmp/out",
                        "--forced-size-bucket",
                        "extra_large",
                    ]
                )

        self.assertIn("invalid choice", stderr.getvalue())

    def test_main_rejects_non_positive_samples_per_dataset_with_argparse_error(self):
        stderr = StringIO()
        with contextlib.redirect_stderr(stderr):
            with patch(
                "sys.argv",
                [
                    "build_inpaint_dataset.py",
                    "--dataset-root",
                    "PANDA=D:/datasets/PANDA",
                    "--output-dir",
                    "D:/tmp/out",
                    "--samples-per-dataset",
                    "0",
                ],
            ):
                with self.assertRaises(SystemExit):
                    build_inpaint_dataset.main()

        self.assertIn("--samples-per-dataset must be positive, got 0", stderr.getvalue())

    def test_main_rejects_non_positive_max_attempts_per_sample_with_argparse_error(self):
        stderr = StringIO()
        with contextlib.redirect_stderr(stderr):
            with patch(
                "sys.argv",
                [
                    "build_inpaint_dataset.py",
                    "--dataset-root",
                    "PANDA=D:/datasets/PANDA",
                    "--output-dir",
                    "D:/tmp/out",
                    "--max-attempts-per-sample",
                    "0",
                ],
            ):
                with self.assertRaises(SystemExit):
                    build_inpaint_dataset.main()

        self.assertIn("--max-attempts-per-sample must be positive, got 0", stderr.getvalue())

    def test_main_routes_dataset_root_mode_to_synthetic_builder(self):
        with patch(
            "controlnet_train.cli.build_inpaint_dataset.build_synthetic_inpaint_metadata",
            return_value=(Path("train.jsonl"), Path("val.jsonl")),
        ) as mock_build:
            with patch(
                "sys.argv",
                [
                    "build_inpaint_dataset.py",
                    "--dataset-root",
                    "PANDA=D:/datasets/PANDA",
                    "--dataset-root",
                    "BCSS=D:/datasets/BCSS",
                    "--output-dir",
                    "D:/tmp/out",
                    "--forced-mode",
                    "replace_like_blob",
                    "--forced-size-bucket",
                    "medium",
                    "--val-ratio",
                    "0.25",
                    "--seed",
                    "99",
                    "--samples-per-dataset",
                    "3",
                    "--max-attempts-per-sample",
                    "5",
                ],
            ):
                build_inpaint_dataset.main()

        mock_build.assert_called_once_with(
            dataset_roots={
                "PANDA": Path("D:/datasets/PANDA"),
                "BCSS": Path("D:/datasets/BCSS"),
            },
            output_dir=Path("D:/tmp/out"),
            forced_mode="replace_like_blob",
            forced_bucket="medium",
            val_ratio=0.25,
            seed=99,
            samples_per_dataset=3,
            max_attempts_per_sample=5,
        )

    def test_main_routes_input_jsonl_mode_to_legacy_builder(self):
        with patch(
            "controlnet_train.cli.build_inpaint_dataset.build_inpaint_metadata",
            return_value=(Path("train.jsonl"), Path("val.jsonl")),
        ) as mock_build:
            with patch(
                "sys.argv",
                [
                    "build_inpaint_dataset.py",
                    "--input-jsonl",
                    "D:/tmp/input.jsonl",
                    "--output-dir",
                    "D:/tmp/out",
                    "--val-ratio",
                    "0.25",
                    "--seed",
                    "99",
                ],
            ):
                build_inpaint_dataset.main()

        mock_build.assert_called_once_with(
            input_jsonl_paths=[Path("D:/tmp/input.jsonl")],
            output_dir=Path("D:/tmp/out"),
            val_ratio=0.25,
            seed=99,
        )

    def test_main_keeps_dataset_root_knobs_optional_when_omitted(self):
        with patch(
            "controlnet_train.cli.build_inpaint_dataset.build_synthetic_inpaint_metadata",
            return_value=(Path("train.jsonl"), Path("val.jsonl")),
        ) as mock_build:
            with patch(
                "sys.argv",
                [
                    "build_inpaint_dataset.py",
                    "--dataset-root",
                    "PANDA=D:/datasets/PANDA",
                    "--output-dir",
                    "D:/tmp/out",
                ],
            ):
                build_inpaint_dataset.main()

        mock_build.assert_called_once_with(
            dataset_roots={"PANDA": Path("D:/datasets/PANDA")},
            output_dir=Path("D:/tmp/out"),
            forced_mode="mixed",
            forced_bucket=None,
            val_ratio=0.1,
            seed=42,
            samples_per_dataset=None,
            max_attempts_per_sample=None,
        )

    def test_build_synthetic_inpaint_metadata_is_exported(self):
        self.assertIs(
            exported_build_synthetic_inpaint_metadata,
            controlnet_train.build_synthetic_inpaint_metadata,
        )

    def test_package_wrapper_forwards_sizing_knobs_to_synthesis_helper(self):
        with patch(
            "controlnet_train.data._build_synthetic_inpaint_metadata",
            return_value=(Path("train.jsonl"), Path("val.jsonl")),
        ) as mock_build:
            train_path, val_path = exported_build_synthetic_inpaint_metadata(
                dataset_roots={"PANDA": Path("D:/datasets/PANDA")},
                output_dir=Path("D:/tmp/out"),
                val_ratio=0.25,
                seed=9,
                samples_per_dataset=4,
                max_attempts_per_sample=6,
            )

        self.assertEqual(train_path, Path("train.jsonl"))
        self.assertEqual(val_path, Path("val.jsonl"))
        mock_build.assert_called_once_with(
            dataset_roots={"PANDA": Path("D:/datasets/PANDA")},
            output_dir=Path("D:/tmp/out"),
            forced_mode="mixed",
            forced_bucket=None,
            val_ratio=0.25,
            seed=9,
            samples_per_dataset=4,
            max_attempts_per_sample=6,
        )

    def test_build_synthetic_metadata_defaults_to_mixed_modes(self):
        tmpdir = _TMP_ROOT / f"case_{uuid.uuid4().hex}"
        try:
            root = Path(tmpdir) / "PANDA"
            (root / "images").mkdir(parents=True)
            (root / "tissue_masks").mkdir()
            (root / "nuclei_masks").mkdir()

            sample_name = "case_mixed_py0_px0.png"
            _write_rgb(root / "images" / sample_name, 72)
            tissue = np.full((8, 8), 8, dtype=np.uint8)
            tissue[:, :2] = 2
            _write_mask(root / "tissue_masks" / sample_name, tissue)
            _write_mask(root / "nuclei_masks" / sample_name, np.full((8, 8), 101, dtype=np.uint8))
            (root / "metadata.jsonl").write_text(
                json.dumps({"image": f"images\\{sample_name}", "text": "prostate prompt"}) + "\n",
                encoding="utf8",
            )

            train_path, _ = build_synthetic_inpaint_metadata(
                dataset_roots={"PANDA": root},
                output_dir=root / "synthetic_output",
                val_ratio=0.0,
                seed=7,
                samples_per_dataset=1,
                max_attempts_per_sample=3,
            )

            row = json.loads(train_path.read_text(encoding="utf8").splitlines()[0])
            self.assertIn(
                row["mask_mode"],
                {"identity", "near_identity", "expand_band", "shrink_band", "replace_like_blob"},
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_build_synthetic_metadata_retries_with_varying_seeds_for_structured_modes(self):
        tmpdir = _TMP_ROOT / f"case_{uuid.uuid4().hex}"
        try:
            root = Path(tmpdir) / "PANDA"
            (root / "images").mkdir(parents=True)
            (root / "tissue_masks").mkdir()
            (root / "nuclei_masks").mkdir()

            sample_name = "case_retry_py0_px0.png"
            _write_rgb(root / "images" / sample_name, 88)
            _write_mask(root / "tissue_masks" / sample_name, np.full((8, 8), 1, dtype=np.uint8))
            _write_mask(root / "nuclei_masks" / sample_name, np.full((8, 8), 102, dtype=np.uint8))

            metadata_path = root / "metadata.jsonl"
            metadata_path.write_text(
                json.dumps({"image": f"images\\{sample_name}", "text": "prostate prompt"}) + "\n",
                encoding="utf8",
            )

            seeds_seen: list[int] = []

            def synthesize_side_effect(
                tissue_mask,
                forced_bucket=None,
                size_bucket=None,
                seed=None,
                preferred_labels=None,
            ):
                seeds_seen.append(seed)
                if seed == 17:
                    return np.zeros_like(tissue_mask, dtype=np.uint8), "replace_like_blob"
                mask = np.zeros_like(tissue_mask, dtype=np.uint8)
                mask[0, 0] = 255
                return mask, "replace_like_blob"

            with patch(
                "controlnet_train.data.inpaint_synthesis.synthesize_change_region",
                side_effect=synthesize_side_effect,
            ):
                train_path, _ = build_synthetic_inpaint_metadata(
                    dataset_roots={"PANDA": root},
                    output_dir=root / "synthetic_output",
                    forced_mode="replace_like_blob",
                    forced_bucket="small",
                    val_ratio=0.0,
                    seed=17,
                    samples_per_dataset=1,
                    max_attempts_per_sample=2,
                )

            self.assertEqual(seeds_seen, [17, 18])
            row = json.loads(train_path.read_text(encoding="utf8").splitlines()[0])
            self.assertEqual(row["mask_mode"], "replace_like_blob")
            self.assertEqual(row["size_bucket"], "small")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_build_synthetic_metadata_retries_when_change_ratio_exceeds_maximum(self):
        tmpdir = _TMP_ROOT / f"case_{uuid.uuid4().hex}"
        try:
            root = Path(tmpdir) / "PANDA"
            (root / "images").mkdir(parents=True)
            (root / "tissue_masks").mkdir()
            (root / "nuclei_masks").mkdir()

            sample_name = "case_retry_ratio_py0_px0.png"
            _write_rgb(root / "images" / sample_name, 88)
            _write_mask(root / "tissue_masks" / sample_name, np.full((8, 8), 8, dtype=np.uint8))
            _write_mask(root / "nuclei_masks" / sample_name, np.full((8, 8), 102, dtype=np.uint8))
            (root / "metadata.jsonl").write_text(
                json.dumps({"image": f"images\\{sample_name}", "text": "prostate prompt"}) + "\n",
                encoding="utf8",
            )

            seeds_seen: list[int] = []

            def synthesize_side_effect(
                tissue_mask,
                forced_bucket=None,
                size_bucket=None,
                seed=None,
                preferred_labels=None,
            ):
                seeds_seen.append(seed)
                if seed == 17:
                    return np.full_like(tissue_mask, 255, dtype=np.uint8), "expand_band"
                mask = np.zeros_like(tissue_mask, dtype=np.uint8)
                mask[:4, :4] = 255
                return mask, "expand_band"

            with patch(
                "controlnet_train.data.inpaint_synthesis.synthesize_change_region",
                side_effect=synthesize_side_effect,
            ):
                train_path, _ = build_synthetic_inpaint_metadata(
                    dataset_roots={"PANDA": root},
                    output_dir=root / "synthetic_output",
                    forced_mode="expand_band",
                    val_ratio=0.0,
                    seed=17,
                    samples_per_dataset=1,
                    max_attempts_per_sample=2,
                )

            self.assertEqual(seeds_seen, [17, 18])
            row = json.loads(train_path.read_text(encoding="utf8").splitlines()[0])
            self.assertLessEqual(float(row["change_ratio"]), 0.7)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_build_synthetic_metadata_emits_extra_variant_only_for_tumor_plus_other_patches(self):
        tmpdir = _TMP_ROOT / f"case_{uuid.uuid4().hex}"
        try:
            root = Path(tmpdir) / "PANDA"
            (root / "images").mkdir(parents=True)
            (root / "tissue_masks").mkdir()
            (root / "nuclei_masks").mkdir()

            mixed_name = "case_tumor_other_py0_px0.png"
            pure_name = "case_pure_tumor_py0_px0.png"

            _write_rgb(root / "images" / mixed_name, 72)
            _write_rgb(root / "images" / pure_name, 96)

            mixed_tissue = np.full((8, 8), 8, dtype=np.uint8)
            mixed_tissue[:, :3] = 2
            pure_tissue = np.full((8, 8), 8, dtype=np.uint8)

            _write_mask(root / "tissue_masks" / mixed_name, mixed_tissue)
            _write_mask(root / "tissue_masks" / pure_name, pure_tissue)
            _write_mask(root / "nuclei_masks" / mixed_name, np.full((8, 8), 101, dtype=np.uint8))
            _write_mask(root / "nuclei_masks" / pure_name, np.full((8, 8), 101, dtype=np.uint8))

            (root / "metadata.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"image": f"images\\{mixed_name}", "text": "prostate prompt"}),
                        json.dumps({"image": f"images\\{pure_name}", "text": "prostate prompt"}),
                    ]
                )
                + "\n",
                encoding="utf8",
            )

            train_path, _ = build_synthetic_inpaint_metadata(
                dataset_roots={"PANDA": root},
                output_dir=root / "synthetic_output",
                forced_mode="mixed",
                val_ratio=0.0,
                seed=7,
            )

            rows = [json.loads(line) for line in train_path.read_text(encoding="utf8").splitlines() if line]
            self.assertEqual(len(rows), 3)

            rows_by_sample: dict[str, list[dict]] = {}
            for row in rows:
                rows_by_sample.setdefault(row["sample_id"], []).append(row)

            self.assertEqual(len(rows_by_sample["case_tumor_other_py0_px0"]), 2)
            self.assertEqual(
                sorted(row["variant_index"] for row in rows_by_sample["case_tumor_other_py0_px0"]),
                [0, 1],
            )
            self.assertEqual(len({row["mask_mode"] for row in rows_by_sample["case_tumor_other_py0_px0"]}), 2)
            self.assertEqual(len(rows_by_sample["case_pure_tumor_py0_px0"]), 1)
            self.assertEqual(rows_by_sample["case_pure_tumor_py0_px0"][0]["variant_index"], 0)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_package_wrapper_forwards_forced_bucket_to_synthesis_helper(self):
        with patch(
            "controlnet_train.data._build_synthetic_inpaint_metadata",
            return_value=(Path("train.jsonl"), Path("val.jsonl")),
        ) as mock_build:
            train_path, val_path = exported_build_synthetic_inpaint_metadata(
                dataset_roots={"PANDA": Path("D:/datasets/PANDA")},
                output_dir=Path("D:/tmp/out"),
                forced_mode="replace_like_blob",
                forced_bucket="medium",
                val_ratio=0.25,
                seed=9,
                samples_per_dataset=4,
                max_attempts_per_sample=6,
            )

        self.assertEqual(train_path, Path("train.jsonl"))
        self.assertEqual(val_path, Path("val.jsonl"))
        mock_build.assert_called_once_with(
            dataset_roots={"PANDA": Path("D:/datasets/PANDA")},
            output_dir=Path("D:/tmp/out"),
            forced_mode="replace_like_blob",
            forced_bucket="medium",
            val_ratio=0.25,
            seed=9,
            samples_per_dataset=4,
            max_attempts_per_sample=6,
        )


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

    def test_build_synthetic_metadata_rejects_empty_non_identity_mask(self):
        tmpdir = _TMP_ROOT / f"case_{uuid.uuid4().hex}"
        try:
            root = Path(tmpdir) / "PANDA"
            (root / "images").mkdir(parents=True)
            (root / "tissue_masks").mkdir()
            (root / "nuclei_masks").mkdir()

            sample_name = "case_empty_py0_px0.png"
            Image.fromarray(np.full((4, 4, 3), 72, dtype=np.uint8)).save(root / "images" / sample_name)
            _write_mask(root / "tissue_masks" / sample_name, np.full((4, 4), 1, dtype=np.uint8))
            _write_mask(root / "nuclei_masks" / sample_name, np.full((4, 4), 101, dtype=np.uint8))

            metadata_path = root / "metadata.jsonl"
            metadata_path.write_text(
                json.dumps({"image": f"images\\{sample_name}", "text": "prostate prompt"}) + "\n",
                encoding="utf8",
            )

            with patch(
                "controlnet_train.data.inpaint_synthesis.synthesize_change_region",
                return_value=(np.zeros((4, 4), dtype=np.uint8), "replace_like_blob"),
            ):
                with self.assertRaises(ValueError) as ctx:
                    build_synthetic_inpaint_metadata(
                        dataset_roots={"PANDA": root},
                        output_dir=root / "synthetic_output",
                        forced_mode="replace_like_blob",
                        forced_bucket="medium",
                        val_ratio=0.0,
                        seed=13,
                    )

            self.assertIn("non-empty", str(ctx.exception))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_build_synthetic_metadata_limits_samples_per_dataset(self):
        tmpdir = _TMP_ROOT / f"case_{uuid.uuid4().hex}"
        try:
            samples = [
                SimpleNamespace(sample_id="sample_a", dataset_name="PANDA", image_path=Path("a.png")),
                SimpleNamespace(sample_id="sample_b", dataset_name="PANDA", image_path=Path("b.png")),
                SimpleNamespace(sample_id="sample_c", dataset_name="PANDA", image_path=Path("c.png")),
            ]

            with patch(
                "controlnet_train.data.inpaint_synthesis.load_layered_dataset_samples",
                return_value=samples,
            ):
                with patch(
                    "controlnet_train.data.inpaint_synthesis._build_synthetic_record",
                    side_effect=lambda *, sample, output_dir, config, attempt_seed=None, **_: {
                        "dataset": sample.dataset_name,
                        "sample_id": sample.sample_id,
                        "case_id": sample.sample_id,
                        "mask_mode": "identity",
                    },
                ) as mock_build:
                    train_path, _ = build_synthetic_inpaint_metadata(
                        dataset_roots={"PANDA": Path("D:/datasets/PANDA")},
                        output_dir=tmpdir / "synthetic_output",
                        forced_mode="identity",
                        val_ratio=0.0,
                        seed=13,
                        samples_per_dataset=2,
                        max_attempts_per_sample=1,
                    )

            self.assertEqual(mock_build.call_count, 2)
            self.assertEqual(len(train_path.read_text(encoding="utf8").splitlines()), 2)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_build_synthetic_metadata_retries_failed_samples(self):
        tmpdir = _TMP_ROOT / f"case_{uuid.uuid4().hex}"
        try:
            sample = SimpleNamespace(sample_id="sample_retry", dataset_name="PANDA", image_path=Path("retry.png"))

            with patch(
                "controlnet_train.data.inpaint_synthesis.load_layered_dataset_samples",
                return_value=[sample],
            ):
                with patch(
                    "controlnet_train.data.inpaint_synthesis._build_synthetic_record",
                    side_effect=[
                        RuntimeError("temporary failure"),
                        {
                            "dataset": "PANDA",
                            "sample_id": "sample_retry",
                            "case_id": "sample_retry",
                            "mask_mode": "identity",
                        },
                    ],
                ) as mock_build:
                    train_path, _ = build_synthetic_inpaint_metadata(
                        dataset_roots={"PANDA": Path("D:/datasets/PANDA")},
                        output_dir=tmpdir / "synthetic_output",
                        forced_mode="identity",
                        val_ratio=0.0,
                        seed=13,
                        samples_per_dataset=1,
                        max_attempts_per_sample=2,
                    )

            self.assertEqual(mock_build.call_count, 2)
            self.assertEqual(len(train_path.read_text(encoding="utf8").splitlines()), 1)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_build_synthetic_metadata_skips_unreadable_source_images(self):
        tmpdir = _TMP_ROOT / f"case_{uuid.uuid4().hex}"
        try:
            root = Path(tmpdir) / "PANDA"
            (root / "images").mkdir(parents=True)
            (root / "tissue_masks").mkdir()
            (root / "nuclei_masks").mkdir()

            good_name = "case_good_py0_px0.png"
            bad_name = "case_bad_py0_px0.png"

            _write_rgb(root / "images" / good_name, 72)
            (root / "images" / bad_name).write_bytes(b"not-a-real-png")

            for sample_name in (good_name, bad_name):
                _write_mask(root / "tissue_masks" / sample_name, np.full((8, 8), 1, dtype=np.uint8))
                _write_mask(root / "nuclei_masks" / sample_name, np.full((8, 8), 101, dtype=np.uint8))

            (root / "metadata.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"image": f"images\\{good_name}", "text": "prostate prompt"}),
                        json.dumps({"image": f"images\\{bad_name}", "text": "prostate prompt"}),
                    ]
                )
                + "\n",
                encoding="utf8",
            )

            train_path, _ = build_synthetic_inpaint_metadata(
                dataset_roots={"PANDA": root},
                output_dir=root / "synthetic_output",
                forced_mode="near_identity",
                val_ratio=0.0,
                seed=13,
                max_attempts_per_sample=2,
            )

            rows = [json.loads(line) for line in train_path.read_text(encoding="utf8").splitlines() if line]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sample_id"], "case_good_py0_px0")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_build_synthetic_metadata_preserves_final_retry_exception_type_and_message(self):
        tmpdir = _TMP_ROOT / f"case_{uuid.uuid4().hex}"
        try:
            sample = SimpleNamespace(sample_id="sample_retry", dataset_name="PANDA", image_path=Path("retry.png"))

            class SentinelRetryError(RuntimeError):
                pass

            with patch(
                "controlnet_train.data.inpaint_synthesis.load_layered_dataset_samples",
                return_value=[sample],
            ):
                with patch(
                    "controlnet_train.data.inpaint_synthesis._build_synthetic_record",
                    side_effect=SentinelRetryError("permanent failure"),
                ):
                    with self.assertRaises(SentinelRetryError) as ctx:
                        build_synthetic_inpaint_metadata(
                            dataset_roots={"PANDA": Path("D:/datasets/PANDA")},
                            output_dir=tmpdir / "synthetic_output",
                            forced_mode="identity",
                            val_ratio=0.0,
                            seed=13,
                            samples_per_dataset=1,
                            max_attempts_per_sample=2,
                        )

            self.assertEqual(str(ctx.exception), "permanent failure")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_build_synthetic_metadata_skips_samples_that_only_exceed_max_change_ratio(self):
        tmpdir = _TMP_ROOT / f"case_{uuid.uuid4().hex}"
        try:
            root = Path(tmpdir) / "PANDA"
            (root / "images").mkdir(parents=True)
            (root / "tissue_masks").mkdir()
            (root / "nuclei_masks").mkdir()

            sample_name = "case_skip_ratio_py0_px0.png"
            _write_rgb(root / "images" / sample_name, 88)
            _write_mask(root / "tissue_masks" / sample_name, np.full((8, 8), 8, dtype=np.uint8))
            _write_mask(root / "nuclei_masks" / sample_name, np.full((8, 8), 102, dtype=np.uint8))
            (root / "metadata.jsonl").write_text(
                json.dumps({"image": f"images\\{sample_name}", "text": "prostate prompt"}) + "\n",
                encoding="utf8",
            )

            with patch(
                "controlnet_train.data.inpaint_synthesis.synthesize_change_region",
                return_value=(np.full((8, 8), 255, dtype=np.uint8), "replace_like_blob"),
            ):
                train_path, val_path = build_synthetic_inpaint_metadata(
                    dataset_roots={"PANDA": root},
                    output_dir=root / "synthetic_output",
                    forced_mode="replace_like_blob",
                    val_ratio=0.0,
                    seed=13,
                    max_attempts_per_sample=2,
                )

            self.assertEqual(train_path.read_text(encoding="utf8").strip(), "")
            self.assertEqual(val_path.read_text(encoding="utf8").strip(), "")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_build_synthetic_metadata_skips_samples_when_forced_bucket_is_unreachable(self):
        tmpdir = _TMP_ROOT / f"case_{uuid.uuid4().hex}"
        try:
            root = Path(tmpdir) / "ORCA"
            (root / "images").mkdir(parents=True)
            (root / "tissue_masks").mkdir()
            (root / "nuclei_masks").mkdir()

            sample_name = "case_bucket_py0_px0.png"
            _write_rgb(root / "images" / sample_name, 88)
            _write_mask(root / "tissue_masks" / sample_name, np.full((20, 20), 1, dtype=np.uint8))
            _write_mask(root / "nuclei_masks" / sample_name, np.full((20, 20), 101, dtype=np.uint8))
            (root / "metadata.jsonl").write_text(
                json.dumps({"image": f"images\\{sample_name}", "text": "oral prompt"}) + "\n",
                encoding="utf8",
            )

            small_mask = np.zeros((20, 20), dtype=np.uint8)
            small_mask[0:2, 0:2] = 255
            with patch(
                "controlnet_train.data.inpaint_synthesis.synthesize_change_region",
                return_value=(small_mask, "shrink_band"),
            ):
                train_path, val_path = build_synthetic_inpaint_metadata(
                    dataset_roots={"ORCA": root},
                    output_dir=root / "synthetic_output",
                    forced_mode="mixed",
                    forced_bucket="medium",
                    val_ratio=0.0,
                    seed=13,
                    max_attempts_per_sample=2,
                )

            self.assertEqual(train_path.read_text(encoding="utf8").strip(), "")
            self.assertEqual(val_path.read_text(encoding="utf8").strip(), "")
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

    def test_synthetic_metadata_replace_like_blob_erases_the_selected_component(self):
        tmpdir = _TMP_ROOT / f"case_{uuid.uuid4().hex}"
        try:
            root = Path(tmpdir) / "PANDA"
            (root / "images").mkdir(parents=True)
            (root / "tissue_masks").mkdir()
            (root / "nuclei_masks").mkdir()

            sample_name = "case3_py0_px0.png"
            Image.fromarray(np.full((32, 32, 3), 88, dtype=np.uint8)).save(root / "images" / sample_name)
            tissue_mask = np.zeros((32, 32), dtype=np.uint8)
            tissue_mask[8:24, 8:24] = 1
            _write_mask(root / "tissue_masks" / sample_name, tissue_mask)
            _write_mask(root / "nuclei_masks" / sample_name, np.full((32, 32), 102, dtype=np.uint8))

            metadata_path = root / "metadata.jsonl"
            metadata_path.write_text(
                json.dumps({"image": f"images\\{sample_name}", "text": "prostate prompt"}) + "\n",
                encoding="utf8",
            )

            train_path, _ = build_synthetic_inpaint_metadata(
                dataset_roots={"PANDA": root},
                output_dir=root / "synthetic_output",
                forced_mode="replace_like_blob",
                val_ratio=0.0,
                seed=17,
            )

            dataset = InpaintDataset(train_path)
            sample = dataset[0]
            row = json.loads(train_path.read_text(encoding="utf8").splitlines()[0])
            change_mask = np.asarray(Image.open(row["change_region_mask"]))

            self.assertEqual(len(dataset), 1)
            self.assertEqual(row["mask_mode"], "replace_like_blob")
            self.assertTrue(np.array_equal(change_mask > 0, tissue_mask > 0))
            self.assertEqual(float(row["change_ratio"]), float(sample["change_ratio"]))
            self.assertTrue(Path(sample["erased_source_image_path"]).exists())
            self.assertEqual(row["size_bucket"], "large")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_near_identity_mask_can_satisfy_requested_change_pixels(self):
        tissue_mask = np.zeros((4, 4), dtype=np.uint8)
        tissue_mask[3, 3] = 1

        mask = _build_near_identity_mask(tissue_mask, change_pixels=2)

        self.assertEqual(int(np.count_nonzero(mask)), 2)

    def test_expand_band_intersects_component_boundary_without_crossing_background(self):
        tissue_mask = np.zeros((64, 64), dtype=np.uint8)
        tissue_mask[12:52, 12:52] = 1

        mask = expand_band(tissue_mask, seed=11)
        boundary = np.zeros_like(tissue_mask, dtype=bool)
        boundary[12, 12:52] = True
        boundary[51, 12:52] = True
        boundary[12:52, 12] = True
        boundary[12:52, 51] = True

        self.assertGreater(int(np.count_nonzero(mask)), 0)
        self.assertTrue(np.any((mask > 0) & (tissue_mask > 0)))
        self.assertFalse(np.any((mask > 0) & (tissue_mask == 0)))
        self.assertTrue(np.any((mask > 0) & boundary))

    def test_expand_band_prefers_a_dominant_tissue_label(self):
        tissue_mask = np.zeros((8, 8), dtype=np.uint8)
        tissue_mask[0:5, 0:4] = 1
        tissue_mask[0:4, 4:8] = 2

        mask = expand_band(tissue_mask, seed=11)

        self.assertGreater(int(np.count_nonzero(mask)), 0)
        self.assertTrue(np.any((mask > 0) & (tissue_mask == 1)))
        self.assertFalse(np.any((mask > 0) & (tissue_mask == 2)))
        self.assertFalse(np.any((mask > 0) & (tissue_mask == 0)))

    def test_expand_band_falls_back_to_interior_when_no_non_frame_boundary_exists(self):
        tissue_mask = np.ones((8, 8), dtype=np.uint8)

        mask = expand_band(tissue_mask, seed=11)

        self.assertGreater(int(np.count_nonzero(mask)), 0)
        self.assertGreater(int(mask[4, 4]), 0)
        self.assertEqual(int(mask[0, 0]), 0)

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

    def test_shrink_band_targets_component_interior(self):
        tissue_mask = np.zeros((64, 64), dtype=np.uint8)
        tissue_mask[8:56, 8:56] = 1

        mask = shrink_band(tissue_mask, seed=11)

        self.assertGreater(int(np.count_nonzero(mask)), 0)
        self.assertTrue(np.any((mask > 0) & (tissue_mask > 0)))
        self.assertFalse(np.any((mask > 0) & (tissue_mask == 0)))
        self.assertGreater(int(np.count_nonzero(mask[24:40, 24:40])), 0)

    def test_shrink_band_never_absorbs_a_thin_component(self):
        tissue_mask = np.zeros((8, 8), dtype=np.uint8)
        tissue_mask[4, 2:7] = 1

        mask = shrink_band(tissue_mask, seed=11)

        self.assertGreater(int(np.count_nonzero(mask)), 0)
        self.assertLess(int(np.count_nonzero(mask)), int(np.count_nonzero(tissue_mask)))
        self.assertTrue(np.any((mask > 0) & (tissue_mask > 0)))
        self.assertFalse(np.any((mask > 0) & (tissue_mask == 0)))

    def test_replace_like_blob_erases_the_entire_selected_component(self):
        tissue_mask = self._make_component_mask()

        mask = replace_like_blob(tissue_mask, seed=11)

        self.assertGreater(int(np.count_nonzero(mask)), 0)
        self.assertFalse(np.any((mask > 0) & (tissue_mask == 0)))
        self.assertTrue(np.array_equal(mask > 0, tissue_mask > 0))

    def test_replace_like_blob_small_bucket_returns_the_full_component(self):
        tissue_mask = np.zeros((128, 128), dtype=np.uint8)
        tissue_mask[32:80, 32:80] = 1

        mask = replace_like_blob(tissue_mask, seed=11, size_bucket="small")

        component_pixels = int(np.count_nonzero(tissue_mask))
        changed_pixels = int(np.count_nonzero(mask))
        changed_fraction = changed_pixels / component_pixels

        self.assertEqual(changed_fraction, 1.0)
        self.assertFalse(np.any((mask > 0) & (tissue_mask == 0)))

    def test_replace_like_blob_remains_a_single_component_on_thin_component(self):
        tissue_mask = np.zeros((8, 8), dtype=np.uint8)
        tissue_mask[4, 2:7] = 1

        mask = replace_like_blob(tissue_mask, seed=1)

        self.assertGreater(int(np.count_nonzero(mask)), 0)
        self.assertFalse(np.any((mask > 0) & (tissue_mask == 0)))
        self.assertTrue(np.array_equal(mask > 0, tissue_mask > 0))

    def test_expand_and_shrink_scale_change_area_with_size_bucket(self):
        tissue_mask = np.zeros((32, 32), dtype=np.uint8)
        tissue_mask[4:28, 4:28] = 1

        for fn in (expand_band, shrink_band):
            small = fn(tissue_mask, seed=11, size_bucket="small")
            medium = fn(tissue_mask, seed=11, size_bucket="medium")
            large = fn(tissue_mask, seed=11, size_bucket="large")

            small_ratio = float(np.count_nonzero(small) / small.size)
            medium_ratio = float(np.count_nonzero(medium) / medium.size)
            large_ratio = float(np.count_nonzero(large) / large.size)

            self.assertLess(small_ratio, medium_ratio)
            self.assertLess(medium_ratio, large_ratio)
            self.assertGreater(small_ratio, 0.05)
            self.assertGreater(medium_ratio, 0.10)
            self.assertGreater(large_ratio, 0.18)

    def test_build_synthetic_metadata_prefers_tumor_component_over_larger_non_tumor_region(self):
        tmpdir = _TMP_ROOT / f"case_{uuid.uuid4().hex}"
        try:
            root = Path(tmpdir) / "PANDA"
            (root / "images").mkdir(parents=True)
            (root / "tissue_masks").mkdir()
            (root / "nuclei_masks").mkdir()

            sample_name = "case_tumor_priority_py0_px0.png"
            Image.fromarray(np.full((64, 64, 3), 88, dtype=np.uint8)).save(root / "images" / sample_name)
            tissue_mask = np.zeros((64, 64), dtype=np.uint8)
            tissue_mask[6:58, 6:30] = 2
            tissue_mask[18:46, 34:58] = 8
            _write_mask(root / "tissue_masks" / sample_name, tissue_mask)
            _write_mask(root / "nuclei_masks" / sample_name, np.full((64, 64), 102, dtype=np.uint8))
            (root / "metadata.jsonl").write_text(
                json.dumps({"image": f"images\\{sample_name}", "text": "prostate prompt"}) + "\n",
                encoding="utf8",
            )

            train_path, _ = build_synthetic_inpaint_metadata(
                dataset_roots={"PANDA": root},
                output_dir=root / "synthetic_output",
                forced_mode="replace_like_blob",
                val_ratio=0.0,
                seed=17,
            )

            row = json.loads(train_path.read_text(encoding="utf8").splitlines()[0])
            change_mask = np.asarray(Image.open(row["change_region_mask"])) > 0

            self.assertTrue(np.any(change_mask & (tissue_mask == 8)))
            self.assertFalse(np.any(change_mask & (tissue_mask == 2)))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_expand_and_shrink_medium_large_masks_are_not_one_pixel_traces(self):
        tissue_mask = np.zeros((64, 64), dtype=np.uint8)
        tissue_mask[8:56, 8:56] = 1

        expand_medium = expand_band(tissue_mask, seed=11, size_bucket="medium")
        shrink_large = shrink_band(tissue_mask, seed=11, size_bucket="large")

        self.assertGreater(int(np.count_nonzero(expand_medium)), 200)
        self.assertGreater(int(np.count_nonzero(shrink_large)), 600)
        self.assertGreaterEqual(int(expand_medium[32, 32]), 0)
        self.assertGreater(int(shrink_large[32, 32]), 0)

    def test_synthesize_change_region_respects_forced_bucket(self):
        tissue_mask = self._make_component_mask()

        for forced_bucket in ("expand_band", "shrink_band", "replace_like_blob"):
            mask, bucket = synthesize_change_region(tissue_mask, forced_bucket=forced_bucket, seed=11)

            self.assertEqual(bucket, forced_bucket)
            self.assertGreater(int(np.count_nonzero(mask)), 0)
            self.assertFalse(np.any((mask > 0) & (tissue_mask == 0)))


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

    def test_noising_degradation_uses_degraded_input_but_original_reference_and_target(self):
        tmpdir = _TMP_ROOT / f"case_{uuid.uuid4().hex}"
        try:
            root = Path(tmpdir) / "BCSS"
            (root / "images").mkdir(parents=True)
            (root / "tissue_masks").mkdir()
            (root / "nuclei_masks").mkdir()

            names = ["case_deg_py0_px0.png", "case_deg_py0_px256.png"]
            for index, name in enumerate(names):
                arr = np.zeros((8, 8, 3), dtype=np.uint8)
                arr[..., 0] = np.arange(8, dtype=np.uint8)[None, :] * 20 + index * 10
                arr[..., 1] = np.arange(8, dtype=np.uint8)[:, None] * 20 + 40
                arr[..., 2] = 128
                Image.fromarray(arr).save(root / "images" / name)
                _write_mask(root / "tissue_masks" / name, np.full((8, 8), 1, dtype=np.uint8))
                _write_mask(root / "nuclei_masks" / name, np.full((8, 8), 101, dtype=np.uint8))

            with (root / "metadata.jsonl").open("w", encoding="utf8") as f:
                for name in names:
                    f.write(json.dumps({"image": f"images\\{name}", "text": "breast prompt"}) + "\n")

            train_path, _ = build_cross_metadata(
                dataset_roots={"BCSS": root},
                output_dir=root / "cross_output",
                num_ref_per_target=1,
                val_ratio=0.0,
                seed=11,
                top_k=2,
            )

            torch.manual_seed(7)
            dataset = CrossReconstructionDataset(
                train_path,
                noising_degradation="texture",
                texture_blur_prob=0.0,
                texture_downsample_prob=1.0,
                texture_downsample_scale_min=0.25,
                texture_downsample_scale_max=0.25,
                texture_noise_prob=0.0,
            )
            for index in range(len(dataset)):
                sample = dataset[index]

                self.assertEqual(sample["sample_mode"], "appearance_degraded")
                self.assertTrue(sample["uses_degraded_noising"])
                self.assertNotEqual(sample["reference_sample_id"], sample["sample_id"])
                self.assertFalse(torch.equal(sample["reference_image"], sample["target_image"]))
                self.assertFalse(torch.equal(sample["clean_image_for_noising"], sample["target_image"]))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_build_cross_metadata_can_mix_reference_coverage_difficulties(self):
        tmpdir = _TMP_ROOT / f"case_{uuid.uuid4().hex}"
        try:
            root = Path(tmpdir) / "BCSS"
            (root / "images").mkdir(parents=True)
            (root / "tissue_masks").mkdir()
            (root / "nuclei_masks").mkdir()

            target = np.zeros((8, 8), dtype=np.uint8)
            target[:, :4] = 1
            target[:, 4:] = 2
            full = target.copy()
            partial = np.full((8, 8), 1, dtype=np.uint8)
            low = np.full((8, 8), 3, dtype=np.uint8)
            sample_defs = [
                ("case_mix_py0_px0.png", target, 101, 32),
                ("case_mix_py0_px256.png", full, 101, 48),
                ("case_mix_py256_px0.png", partial, 102, 64),
                ("case_mix_py256_px256.png", low, 103, 80),
            ]
            for name, tissue, nuclei_id, image_value in sample_defs:
                _write_rgb(root / "images" / name, image_value)
                _write_mask(root / "tissue_masks" / name, tissue)
                _write_mask(root / "nuclei_masks" / name, np.full((8, 8), nuclei_id, dtype=np.uint8))

            with (root / "metadata.jsonl").open("w", encoding="utf8") as f:
                for name, _, _, _ in sample_defs:
                    f.write(json.dumps({"image": f"images\\{name}", "text": "breast prompt"}) + "\n")

            train_path, _ = build_cross_metadata(
                dataset_roots={"BCSS": root},
                output_dir=root / "cross_output",
                num_ref_per_target=3,
                val_ratio=0.0,
                seed=11,
                top_k=3,
                full_coverage_weight=1.0,
                partial_coverage_weight=1.0,
                low_coverage_weight=1.0,
            )

            rows = json.loads(train_path.read_text(encoding="utf8"))["pairs"]
            target_rows = [row for row in rows if row["sample_id"] == "case_mix_py0_px0"]
            difficulties = {row["pair_difficulty"] for row in target_rows}

            self.assertEqual(difficulties, {"full", "partial", "low"})
            partial_row = next(row for row in target_rows if row["pair_difficulty"] == "partial")
            low_row = next(row for row in target_rows if row["pair_difficulty"] == "low")
            self.assertEqual(partial_row["missing_target_tissue_ids"], [2])
            self.assertAlmostEqual(partial_row["tissue_coverage_ratio"], 0.5)
            self.assertAlmostEqual(partial_row["area_coverage_ratio"], 0.5)
            self.assertEqual(low_row["missing_target_tissue_ids"], [1, 2])
            self.assertEqual(low_row["tissue_coverage_ratio"], 0.0)
            self.assertEqual(low_row["area_coverage_ratio"], 0.0)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_build_cross_metadata_skips_invalid_samples_and_reports_paths(self):
        tmpdir = _TMP_ROOT / f"case_{uuid.uuid4().hex}"
        try:
            root = Path(tmpdir) / "BCSS"
            (root / "images").mkdir(parents=True)
            (root / "tissue_masks").mkdir()
            (root / "nuclei_masks").mkdir()

            sample_defs = [
                ("case_bad_py0_px0.png", 1, 101, 32),
                ("case_bad_py0_px256.png", 1, 101, 48),
                ("case_bad_py256_px0.png", 1, 101, None),
            ]
            for name, tissue_id, nuclei_id, image_value in sample_defs:
                if image_value is None:
                    (root / "images" / name).write_bytes(b"\x89PNG\r\n\x1a\n")
                else:
                    _write_rgb(root / "images" / name, image_value)
                _write_mask(root / "tissue_masks" / name, np.full((8, 8), tissue_id, dtype=np.uint8))
                _write_mask(root / "nuclei_masks" / name, np.full((8, 8), nuclei_id, dtype=np.uint8))

            with (root / "metadata.jsonl").open("w", encoding="utf8") as f:
                for name, _, _, _ in sample_defs:
                    f.write(json.dumps({"image": f"images\\{name}", "text": "breast prompt"}) + "\n")

            train_path, _ = build_cross_metadata(
                dataset_roots={"BCSS": root},
                output_dir=root / "cross_output",
                num_ref_per_target=1,
                val_ratio=0.0,
                seed=11,
            )

            rows = json.loads(train_path.read_text(encoding="utf8"))["pairs"]
            skipped = json.loads((root / "cross_output" / "skipped_cross_samples.json").read_text(encoding="utf8"))

            self.assertGreater(len(rows), 0)
            self.assertTrue(all("case_bad_py256_px0" not in row["sample_id"] for row in rows))
            self.assertEqual(skipped["skipped_count"], 1)
            self.assertEqual(skipped["samples"][0]["sample_id"], "case_bad_py256_px0")
            self.assertIn("image file", skipped["samples"][0]["error"].lower())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_build_cross_metadata_can_fail_fast_on_invalid_samples(self):
        tmpdir = _TMP_ROOT / f"case_{uuid.uuid4().hex}"
        try:
            root = Path(tmpdir) / "BCSS"
            (root / "images").mkdir(parents=True)
            (root / "tissue_masks").mkdir()
            (root / "nuclei_masks").mkdir()

            sample_name = "case_strict_py0_px0.png"
            (root / "images" / sample_name).write_bytes(b"\x89PNG\r\n\x1a\n")
            _write_mask(root / "tissue_masks" / sample_name, np.full((8, 8), 1, dtype=np.uint8))
            _write_mask(root / "nuclei_masks" / sample_name, np.full((8, 8), 101, dtype=np.uint8))
            (root / "metadata.jsonl").write_text(
                json.dumps({"image": f"images\\{sample_name}", "text": "breast prompt"}) + "\n",
                encoding="utf8",
            )

            with self.assertRaises(OSError):
                build_cross_metadata(
                    dataset_roots={"BCSS": root},
                    output_dir=root / "cross_output",
                    val_ratio=0.0,
                    skip_invalid_samples=False,
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_generate_training_pairs_cli_forwards_reference_coverage_weights(self):
        with patch(
            "controlnet_train.cli.generate_training_pairs.build_cross_metadata",
            return_value=(Path("train.json"), Path("val.json")),
        ) as mock_build:
            with patch(
                "sys.argv",
                [
                    "generate_training_pairs.py",
                    "--dataset-root",
                    "BCSS=D:/datasets/BCSS",
                    "--output-dir",
                    "D:/tmp/cross",
                    "--full-coverage-weight",
                    "0.5",
                    "--partial-coverage-weight",
                    "0.4",
                    "--low-coverage-weight",
                    "0.1",
                    "--progress-every",
                    "250",
                    "--strict",
                ],
            ):
                generate_training_pairs.main()

        mock_build.assert_called_once_with(
            dataset_roots={"BCSS": Path("D:/datasets/BCSS")},
            output_dir=Path("D:/tmp/cross"),
            num_ref_per_target=2,
            top_k=8,
            val_ratio=0.1,
            seed=42,
            full_coverage_weight=0.5,
            partial_coverage_weight=0.4,
            low_coverage_weight=0.1,
            skip_invalid_samples=False,
            progress_every=250,
        )


class InpaintReadmeTests(unittest.TestCase):
    def test_readme_covers_dataset_root_synthesis_and_trace_fields(self):
        readme = Path(__file__).resolve().parents[1] / "controlnet_train" / "README.txt"
        readme_text = readme.read_text(encoding="utf8")

        self.assertIn("--dataset-root", readme_text)
        self.assertIn("replace_like_blob", readme_text)
        self.assertIn("mask_mode", readme_text)
        self.assertIn("size_bucket", readme_text)
        self.assertIn("change_ratio", readme_text)


if __name__ == "__main__":
    unittest.main()

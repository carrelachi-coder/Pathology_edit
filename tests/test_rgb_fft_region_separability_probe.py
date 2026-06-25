import importlib.util
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from PIL import Image
except ModuleNotFoundError:
    torch = None
    Image = None


_PROBE = None
_IMPORT_ERROR = None
if torch is not None:
    try:
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "probe_rgb_fft_region_separability.py"
        spec = importlib.util.spec_from_file_location("probe_rgb_fft_region_separability", script_path)
        _PROBE = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = _PROBE
        spec.loader.exec_module(_PROBE)
    except ModuleNotFoundError as exc:
        _IMPORT_ERROR = exc


@unittest.skipIf(torch is None, "torch is required for RGB+FFT separability probe tests")
class RgbFftRegionSeparabilityProbeTests(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"probe dependencies unavailable: {_IMPORT_ERROR}")

    def test_pair_summary_reports_same_label_vs_cross_label_descriptor_gap(self):
        items_a = [
            _descriptor_item(
                label_name="tumor",
                label_id=1,
                sample_id="tumor_ref_0",
                mean=[0.10, 0.20, 0.30],
                std=[0.04, 0.05, 0.06],
                fft=[1.00, 0.90, 0.80, 0.70],
            ),
            _descriptor_item(
                label_name="tumor",
                label_id=1,
                sample_id="tumor_ref_1",
                mean=[0.12, 0.19, 0.31],
                std=[0.05, 0.05, 0.07],
                fft=[1.02, 0.91, 0.82, 0.69],
            ),
        ]
        items_b = [
            _descriptor_item(
                label_name="stroma",
                label_id=2,
                sample_id="stroma_ref_0",
                mean=[0.78, 0.70, 0.60],
                std=[0.18, 0.19, 0.20],
                fft=[2.00, 1.80, 1.60, 1.40],
            ),
            _descriptor_item(
                label_name="stroma",
                label_id=2,
                sample_id="stroma_ref_1",
                mean=[0.80, 0.69, 0.58],
                std=[0.19, 0.18, 0.21],
                fft=[2.03, 1.82, 1.59, 1.42],
            ),
        ]

        pair_rows, summary = _PROBE.build_pair_outputs(
            items_a=items_a,
            items_b=items_b,
            allow_same_image_cross_pairs=False,
            mean_weight=1.0,
            std_weight=0.5,
            fft_weight=0.25,
        )

        self.assertEqual(summary["counts"]["within_a_pairs"], 1)
        self.assertEqual(summary["counts"]["within_b_pairs"], 1)
        self.assertEqual(summary["counts"]["cross_pairs"], 4)
        self.assertIn("total_distance", summary["distance_stats"])
        self.assertIn("concat_cosine_distance", summary["distance_stats"])

        total_stats = summary["distance_stats"]["total_distance"]
        self.assertGreater(total_stats["cross"]["mean"], total_stats["within_all"]["mean"])
        self.assertGreater(total_stats["cross_minus_within_mean"], 0.0)
        self.assertEqual(total_stats["cross_greater_than_within_probability"], 1.0)
        self.assertTrue(all("pixels_i" in row and "pixels_j" in row for row in pair_rows))

    def test_total_distance_uses_region_loss_weight_normalization(self):
        item_i = _descriptor_item(
            label_name="tumor",
            label_id=1,
            sample_id="i",
            mean=[0.0, 0.2, 0.4],
            std=[0.1, 0.2, 0.3],
            fft=[1.0, 1.5],
        )
        item_j = _descriptor_item(
            label_name="tumor",
            label_id=1,
            sample_id="j",
            mean=[0.3, 0.2, 0.1],
            std=[0.2, 0.4, 0.6],
            fft=[2.0, 2.5],
        )

        distances = _PROBE.descriptor_distances(
            item_i,
            item_j,
            mean_weight=1.0,
            std_weight=0.5,
            fft_weight=0.25,
        )
        expected = (
            distances["mean_distance"] * 1.0
            + distances["std_distance"] * 0.5
            + distances["fft_distance"] * 0.25
        ) / 1.75

        self.assertAlmostEqual(distances["total_distance"], expected)

    def test_main_extracts_descriptors_from_rgb_images_and_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "tissue_masks").mkdir()
            metadata_path = root / "metadata.jsonl"
            for index in range(2):
                sample_id = f"sample_{index}"
                _write_two_region_fixture(
                    image_path=root / "images" / f"{sample_id}.png",
                    mask_path=root / "tissue_masks" / f"{sample_id}.png",
                    color_shift=index * 5,
                )
                with metadata_path.open("a", encoding="utf8") as handle:
                    handle.write(json.dumps({"image": f"images/{sample_id}.png"}) + "\n")

            output_dir = root / "probe_out"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = _PROBE.main(
                    [
                        "--metadata",
                        str(metadata_path),
                        "--output-dir",
                        str(output_dir),
                        "--label-mode",
                        "coarse_tissue",
                        "--label-a",
                        "tumor",
                        "--label-b",
                        "stroma",
                        "--samples-per-label",
                        "2",
                        "--candidate-pool-size",
                        "2",
                        "--min-region-pixels",
                        "8",
                        "--batch-size",
                        "1",
                        "--device",
                        "cpu",
                        "--reference-region-fft-bins",
                        "4",
                        "--reference-region-fft-size",
                        "16",
                    ]
                )

            self.assertEqual(rc, 0)
            summary = json.loads(
                (output_dir / "rgb_fft_region_descriptor_separability_summary.json").read_text(
                    encoding="utf8"
                )
            )
            total_stats = summary["distance_stats"]["total_distance"]
            self.assertEqual(summary["counts"]["label_a"], 2)
            self.assertEqual(summary["counts"]["label_b"], 2)
            self.assertGreater(total_stats["cross"]["mean"], total_stats["within_all"]["mean"])
            self.assertGreater(total_stats["cross_minus_within_mean"], 0.0)

    def test_main_accepts_cross_pair_metadata_schema_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "tissue_masks").mkdir()
            pairs = []
            for index, role in enumerate(("target", "reference")):
                sample_id = f"{role}_{index}"
                image_path = root / "images" / f"{sample_id}.png"
                mask_path = root / "tissue_masks" / f"{sample_id}.png"
                _write_two_region_fixture(
                    image_path=image_path,
                    mask_path=mask_path,
                    color_shift=index * 5,
                )
                pairs.append((sample_id, image_path, mask_path))

            metadata_path = root / "metadata_cross_train.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "pairs": [
                            {
                                "dataset": "BCSS",
                                "sample_id": pairs[0][0],
                                "reference_sample_id": pairs[1][0],
                                "target_image": str(pairs[0][1]),
                                "target_tissue_mask": str(pairs[0][2]),
                                "reference_image": str(pairs[1][1]),
                                "reference_tissue_mask": str(pairs[1][2]),
                            }
                        ]
                    }
                ),
                encoding="utf8",
            )

            output_dir = root / "probe_out"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = _PROBE.main(
                    [
                        "--metadata",
                        str(metadata_path),
                        "--output-dir",
                        str(output_dir),
                        "--label-mode",
                        "coarse_tissue",
                        "--label-a",
                        "tumor",
                        "--label-b",
                        "stroma",
                        "--samples-per-label",
                        "2",
                        "--candidate-pool-size",
                        "2",
                        "--min-region-pixels",
                        "8",
                        "--batch-size",
                        "1",
                        "--device",
                        "cpu",
                        "--reference-region-fft-bins",
                        "4",
                        "--reference-region-fft-size",
                        "16",
                    ]
                )

            self.assertEqual(rc, 0)
            summary = json.loads(
                (output_dir / "rgb_fft_region_descriptor_separability_summary.json").read_text(
                    encoding="utf8"
                )
            )
            self.assertEqual(summary["counts"]["label_a"], 2)
            self.assertEqual(summary["counts"]["label_b"], 2)
            self.assertEqual(summary["candidate_entries"], 2)


def _descriptor_item(
    *,
    label_name: str,
    label_id: int,
    sample_id: str,
    mean: list[float],
    std: list[float],
    fft: list[float],
):
    return _PROBE.DescriptorItem(
        label_name=label_name,
        label_id=label_id,
        sample=_PROBE.SampleEntry(
            index=0,
            dataset="synthetic",
            sample_id=sample_id,
            image_path=Path(f"/tmp/{sample_id}.png"),
            tissue_mask_path=Path(f"/tmp/{sample_id}_mask.png"),
        ),
        region_pixels=64,
        region_fraction=0.25,
        mean=torch.tensor(mean, dtype=torch.float32),
        std=torch.tensor(std, dtype=torch.float32),
        fft=torch.tensor(fft, dtype=torch.float32),
    )


def _write_two_region_fixture(*, image_path: Path, mask_path: Path, color_shift: int) -> None:
    image = torch.zeros(32, 32, 3, dtype=torch.uint8)
    image[:, :16] = torch.tensor([200 + color_shift, 60 + color_shift, 70 + color_shift], dtype=torch.uint8)
    image[:, 16:] = torch.tensor([70 + color_shift, 170 + color_shift, 80 + color_shift], dtype=torch.uint8)
    mask = torch.zeros(32, 32, dtype=torch.uint8)
    mask[:, :16] = 1
    mask[:, 16:] = 2
    Image.fromarray(image.numpy()).save(image_path)
    Image.fromarray(mask.numpy()).save(mask_path)

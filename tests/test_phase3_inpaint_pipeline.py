import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from phase3_mask_edit.core.mask_io import save_id_mask
from phase3_mask_edit.parser.semantic_diff import DEFAULT_SEMANTIC_DIFF
from scripts.run_phase3_inpaint_pipeline import (
    _select_generation_mode,
    main as run_phase3_inpaint_pipeline,
)


def _write_rgb(path: Path, value: tuple[int, int, int] = (180, 120, 160)) -> None:
    array = np.full((96, 96, 3), value, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _write_nuclei(path: Path) -> None:
    nuclei = np.zeros((96, 96), dtype=np.uint8)
    nuclei[20:24, 20:24] = 101
    nuclei[50:54, 50:54] = 102
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(nuclei).save(path)


def _source_mask() -> np.ndarray:
    mask = np.zeros((96, 96), dtype=np.int64)
    mask[16:72, 16:72] = 1
    mask[72:88, 16:72] = 2
    return mask


class Phase3InpaintPipelineTests(unittest.TestCase):
    def test_gen_dry_run_writes_generation_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            source = root / "source_mask.png"
            target = root / "target_mask.png"
            nuclei = root / "nuclei.png"
            output = root / "out"

            source_mask = _source_mask()
            target_mask = source_mask.copy()
            target_mask[34:46, 34:46] = 3
            _write_rgb(image)
            save_id_mask(source_mask, source)
            save_id_mask(target_mask, target)
            _write_nuclei(nuclei)

            exit_code = run_phase3_inpaint_pipeline(
                [
                    "--mode",
                    "gen",
                    "--profile",
                    "BCSS",
                    "--reference-image",
                    str(image),
                    "--reference-tissue-mask",
                    str(source),
                    "--reference-nuclei-mask",
                    str(nuclei),
                    "--target-tissue-mask",
                    str(target),
                    "--output",
                    str(output),
                    "--generation-mode",
                    "dry-run",
                    "--cell-fill-mode",
                    "blank",
                ]
            )

            self.assertEqual(exit_code, 0)
            for name in [
                "erased_image.png",
                "target_mask_rgb.png",
                "change_region.png",
                "target_nuclei_mask.png",
                "generated_image.png",
                "compare_panel.png",
                "generation_info.json",
                "cell_fill_log.json",
                "target_combined_mask.png",
                "pipeline_summary.json",
            ]:
                self.assertTrue((output / name).exists(), name)

            summary = json.loads((output / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["mode"], "gen")
            self.assertEqual(summary["generation_mode"], "dry-run")
            self.assertEqual(summary["cell_fill_mode"], "blank")
            self.assertGreater(summary["changed_pixels"], 0)

    def test_gen_blank_deletes_whole_source_cells_crossing_change_region(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            source = root / "source_mask.png"
            target = root / "target_mask.png"
            nuclei = root / "nuclei.png"
            output = root / "out"

            source_mask = _source_mask()
            target_mask = source_mask.copy()
            target_mask[30:40, 30:40] = 3
            source_nuclei = np.zeros((96, 96), dtype=np.uint8)
            source_nuclei[28:34, 32:36] = 101  # Crosses into the changed region.
            source_nuclei[60:64, 60:64] = 102  # Fully outside, should remain.

            _write_rgb(image)
            save_id_mask(source_mask, source)
            save_id_mask(target_mask, target)
            Image.fromarray(source_nuclei).save(nuclei)

            exit_code = run_phase3_inpaint_pipeline(
                [
                    "--mode",
                    "gen",
                    "--profile",
                    "BCSS",
                    "--reference-image",
                    str(image),
                    "--reference-tissue-mask",
                    str(source),
                    "--reference-nuclei-mask",
                    str(nuclei),
                    "--target-tissue-mask",
                    str(target),
                    "--output",
                    str(output),
                    "--generation-mode",
                    "dry-run",
                    "--cell-fill-mode",
                    "blank",
                ]
            )

            self.assertEqual(exit_code, 0)
            retained = np.asarray(Image.open(output / "retained_nuclei_mask.png").convert("L"))
            self.assertFalse(np.any(retained[28:34, 32:36]))
            self.assertTrue(np.any(retained[60:64, 60:64] == 102))

            log = json.loads((output / "cell_fill_log.json").read_text(encoding="utf-8"))
            self.assertEqual(log["source_cell_integrity"]["crossing_components"], 1)
            self.assertGreaterEqual(log["source_cell_integrity"]["deleted_components"], 1)

    def test_auto_generation_route_uses_large_change_for_inpaint(self):
        self.assertEqual(_select_generation_mode("auto", 0.36, 0.35), "inpaint")
        self.assertEqual(_select_generation_mode("auto", 0.35, 0.35), "cross-v1")
        self.assertEqual(_select_generation_mode("auto", 0.10, 0.35), "cross-v1")
        self.assertEqual(
            _select_generation_mode("auto", 0.10, 0.35, cross_backend="cross-v0"),
            "cross-v0",
        )
        self.assertEqual(_select_generation_mode("cross-v0", 0.90, 0.35), "cross-v0")

    def test_diff_dry_run_executes_phase3_mask_edit_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            source = root / "source_mask.png"
            nuclei = root / "nuclei.png"
            diff_path = root / "semantic_diff.json"
            output = root / "out"

            source_mask = _source_mask()
            _write_rgb(image)
            save_id_mask(source_mask, source)
            _write_nuclei(nuclei)

            diff = json.loads(json.dumps(DEFAULT_SEMANTIC_DIFF))
            diff["necrosis_change"] = {"action": "add", "extent": "focal"}
            diff_path.write_text(json.dumps(diff), encoding="utf-8")

            exit_code = run_phase3_inpaint_pipeline(
                [
                    "--mode",
                    "diff",
                    "--profile",
                    "BCSS",
                    "--reference-image",
                    str(image),
                    "--reference-tissue-mask",
                    str(source),
                    "--reference-nuclei-mask",
                    str(nuclei),
                    "--semantic-diff",
                    str(diff_path),
                    "--output",
                    str(output),
                    "--generation-mode",
                    "dry-run",
                    "--cell-fill-mode",
                    "preserve",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output / "phase3_mask_edit" / "edit_intents.json").exists())
            self.assertTrue((output / "phase3_mask_edit" / "mask_edit" / "target_mask.png").exists())
            self.assertTrue((output / "generated_image.png").exists())

            summary = json.loads((output / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["mode"], "diff")
            self.assertIsNotNone(summary["phase3"])
            self.assertGreater(summary["phase3"]["execution"]["executed_steps"], 0)
            self.assertGreater(summary["changed_pixels"], 0)

    def test_prompt_fixture_dry_run_parses_then_executes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            source = root / "source_mask.png"
            nuclei = root / "nuclei.png"
            diff_path = root / "semantic_diff.json"
            output = root / "out"

            _write_rgb(image)
            save_id_mask(_source_mask(), source)
            _write_nuclei(nuclei)

            diff = json.loads(json.dumps(DEFAULT_SEMANTIC_DIFF))
            diff["necrosis_change"] = {"action": "add", "extent": "focal"}
            diff_path.write_text(json.dumps(diff), encoding="utf-8")

            exit_code = run_phase3_inpaint_pipeline(
                [
                    "--mode",
                    "prompt",
                    "--profile",
                    "BCSS",
                    "--reference-image",
                    str(image),
                    "--reference-tissue-mask",
                    str(source),
                    "--reference-nuclei-mask",
                    str(nuclei),
                    "--old-prompt",
                    "High-grade carcinoma without necrosis.",
                    "--new-prompt",
                    "High-grade carcinoma with focal necrosis.",
                    "--parser",
                    "fixture",
                    "--semantic-diff",
                    str(diff_path),
                    "--output",
                    str(output),
                    "--generation-mode",
                    "dry-run",
                ]
            )

            self.assertEqual(exit_code, 0)
            summary = json.loads((output / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["mode"], "prompt")
            self.assertEqual(summary["phase3"]["parser"]["mode"], "fixture")
            self.assertGreater(summary["phase3"]["execution"]["executed_steps"], 0)

    def test_prompt_api_dry_run_uses_parser_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            source = root / "source_mask.png"
            nuclei = root / "nuclei.png"
            output = root / "out"

            _write_rgb(image)
            save_id_mask(_source_mask(), source)
            _write_nuclei(nuclei)

            diff = json.loads(json.dumps(DEFAULT_SEMANTIC_DIFF))
            diff["necrosis_change"] = {"action": "add", "extent": "focal"}

            with patch("scripts.run_phase3_inpaint_pipeline.parse_prompts_with_api", return_value=diff) as mocked:
                exit_code = run_phase3_inpaint_pipeline(
                    [
                        "--mode",
                        "prompt",
                        "--profile",
                        "BCSS",
                        "--reference-image",
                        str(image),
                        "--reference-tissue-mask",
                        str(source),
                        "--reference-nuclei-mask",
                        str(nuclei),
                        "--old-prompt",
                        "High-grade carcinoma without necrosis.",
                        "--new-prompt",
                        "High-grade carcinoma with focal necrosis.",
                        "--parser",
                        "api",
                        "--api-model",
                        "mock-model",
                        "--output",
                        str(output),
                        "--generation-mode",
                        "dry-run",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(mocked.called)
            summary = json.loads((output / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["phase3"]["parser"]["mode"], "api")
            self.assertEqual(summary["phase3"]["parser"]["api_model"], "mock-model")


if __name__ == "__main__":
    unittest.main()

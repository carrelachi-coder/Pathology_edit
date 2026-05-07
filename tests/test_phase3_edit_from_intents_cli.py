import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from phase3_mask_edit.cli.edit_from_intents import execute_intents_on_mask
from phase3_mask_edit.cli.parse_prompts import main as parse_prompts_main
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.mask_io import load_id_mask, save_id_mask
from phase3_mask_edit.parser.semantic_diff import DEFAULT_SEMANTIC_DIFF


class Phase3EditFromIntentsCliTests(unittest.TestCase):
    def test_execute_intents_on_mask_returns_target_and_union_change_region(self):
        mask = np.zeros((96, 96), dtype=np.int64)
        mask[16:72, 16:72] = 1
        mask[72:88, 16:72] = 2
        intent = EditIntent.from_mapping(
            {
                "primitive": "necrosis_appearance",
                "strength": "mild",
                "reference_profile": "BCSS",
                "parameters": {
                    "min_necrosis_component_area_px": 1,
                    "max_necrosis_components": 1,
                },
            }
        )

        result = execute_intents_on_mask(
            mask,
            [intent],
            reference_profile="BCSS",
        )

        self.assertIn(result.status, {"executed", "degraded_executed"})
        self.assertGreater(np.count_nonzero(result.change_region), 0)
        self.assertFalse(np.array_equal(result.source_mask, result.target_mask))

    def test_parse_prompts_fixture_execute_writes_new_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask = np.zeros((96, 96), dtype=np.int64)
            mask[16:72, 16:72] = 1
            mask[72:88, 16:72] = 2
            mask_path = root / "mask.png"
            save_id_mask(mask, mask_path)

            diff = json.loads(json.dumps(DEFAULT_SEMANTIC_DIFF))
            diff["necrosis_change"] = {"action": "add", "extent": "focal"}
            diff_path = root / "semantic_diff.json"
            diff_path.write_text(json.dumps(diff), encoding="utf-8")

            output = root / "out"
            exit_code = parse_prompts_main(
                [
                    "--profile",
                    "BCSS",
                    "--semantic-diff",
                    str(diff_path),
                    "--mask",
                    str(mask_path),
                    "--output",
                    str(output),
                    "--execute",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output / "semantic_diff.json").exists())
            self.assertTrue((output / "edit_intents.json").exists())
            self.assertTrue((output / "planning_summary.json").exists())
            self.assertTrue((output / "mask_edit" / "target_mask.png").exists())
            self.assertTrue((output / "mask_edit" / "change_region.png").exists())
            target = load_id_mask(output / "mask_edit" / "target_mask.png")
            self.assertFalse(np.array_equal(mask, target))


if __name__ == "__main__":
    unittest.main()

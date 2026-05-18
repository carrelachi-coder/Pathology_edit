import json
import types
import unittest
from pathlib import Path

from scripts.run_phase3_manifest_pipeline import (
    REPO_DEFAULT_MANIFEST,
    _case_path_candidates,
    _dataset_roots,
    _resolve_case_paths,
    _resolve_model_path,
    _selected_variants,
)


class Phase3ManifestPipelineTests(unittest.TestCase):
    def test_debug_manifest_has_prompt_and_instruction_inputs(self):
        manifest = json.loads(REPO_DEFAULT_MANIFEST.read_text(encoding="utf-8"))

        variant_ids = [
            item["variant_id"]
            for item in manifest["runtime"]["edit_variants"]
        ]

        self.assertEqual(variant_ids, ["prompt", "instruction"])
        for case in manifest["cases"]:
            self.assertTrue(case.get("old_prompt"))
            self.assertTrue(case.get("new_prompt"))
            self.assertTrue(case.get("instruction"))
            self.assertTrue(case.get("source_image_relative"))
            self.assertTrue(case.get("source_tissue_mask_relative"))
            self.assertTrue(case.get("source_nuclei_mask_relative"))

    def test_dataset_root_override_keeps_explicit_dataset_override(self):
        manifest = {
            "datasets": ["BCSS", "GlaS"],
            "runtime": {
                "data_roots": {
                    "BCSS": "/manifest/BCSS",
                    "GlaS": "/manifest/GlaS",
                }
            },
        }
        args = types.SimpleNamespace(
            data_root=Path("/override"),
            dataset_root=["GlaS=/special/GlaS"],
        )

        roots = _dataset_roots(manifest, args)

        self.assertEqual(roots["BCSS"].as_posix(), "/override")
        self.assertEqual(roots["GlaS"].as_posix(), "/special/GlaS")

    def test_resolve_case_paths_prefers_relative_manifest_paths(self):
        case = {
            "dataset": "GlaS",
            "source_image_relative": "GlaS_PATCHES/images/a.png",
            "source_tissue_mask_relative": "GlaS_PATCHES/tissue_masks/a.png",
            "source_nuclei_mask_relative": "GlaS_PATCHES/nuclei_masks/a.png",
            "source_image": "D:\\WQX\\datasets\\GLAS\\GlaS_PATCHES\\images\\old.png",
            "source_tissue_mask": "D:\\WQX\\datasets\\GLAS\\GlaS_PATCHES\\tissue_masks\\old.png",
            "source_nuclei_mask": "D:\\WQX\\datasets\\GLAS\\GlaS_PATCHES\\nuclei_masks\\old.png",
        }

        paths = _resolve_case_paths(
            case,
            {"GlaS": Path("/data/wqx/flowedit/data/GlaS")},
        )

        self.assertEqual(
            paths["source_image"].as_posix(),
            "/data/wqx/flowedit/data/GlaS/GlaS_PATCHES/images/a.png",
        )
        self.assertEqual(
            paths["source_tissue_mask"].as_posix(),
            "/data/wqx/flowedit/data/GlaS/GlaS_PATCHES/tissue_masks/a.png",
        )
        self.assertEqual(
            paths["source_nuclei_mask"].as_posix(),
            "/data/wqx/flowedit/data/GlaS/GlaS_PATCHES/nuclei_masks/a.png",
        )

    def test_resolve_case_paths_can_rewrite_legacy_windows_paths(self):
        case = {
            "dataset": "IGNITE",
            "source_image": "D:\\WQX\\datasets\\IGNITE_PATCHES\\images\\a.png",
            "source_tissue_mask": "D:\\WQX\\datasets\\IGNITE_PATCHES\\tissue_masks\\a.png",
            "source_nuclei_mask": "D:\\WQX\\datasets\\IGNITE_PATCHES\\nuclei_masks\\a.png",
        }

        paths = _resolve_case_paths(
            case,
            {"IGNITE": Path("/data/wqx/flowedit/data/IGNITE")},
        )

        self.assertEqual(
            paths["source_image"].as_posix(),
            "/data/wqx/flowedit/data/IGNITE/IGNITE_PATCHES/images/a.png",
        )

    def test_case_path_candidates_include_flat_dataset_layout(self):
        case = {
            "dataset": "BCSS",
            "source_image_relative": "BCSS_PATCHES/images/a.png",
            "source_image": "/data/wqx/flowedit/data/BCSS/BCSS_PATCHES/images/a.png",
        }

        candidates = _case_path_candidates(
            case,
            "source_image",
            {"BCSS": Path("/data/wqx/flowedit/data/BCSS")},
        )
        candidate_text = [path.as_posix() for path in candidates]

        self.assertIn("/data/wqx/flowedit/data/BCSS/images/a.png", candidate_text)
        self.assertIn(
            "/data/wqx/flowedit/data/BCSS/BCSS_PATCHES/images/a.png",
            candidate_text,
        )

    def test_selected_variants_support_filtering(self):
        runtime = {
            "edit_variants": [
                {"variant_id": "prompt", "edit_mode": "prompt"},
                {"variant_id": "instruction", "edit_mode": "instruction"},
            ]
        }

        variants = _selected_variants(runtime, "instruction")

        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0]["edit_mode"], "instruction")

    def test_model_path_prefers_dataset_mapping_over_template(self):
        model_paths = {
            "density_scale_json_by_dataset": {
                "BCSS": "/configs/density_scale_bcss.json",
            },
            "density_scale_json_template": "/configs/density_scale_{profile_lower}.json",
        }

        path = _resolve_model_path(
            model_paths,
            "density_scale_json_template",
            None,
            "BCSS",
            "BCSS",
        )

        self.assertEqual(path, "/configs/density_scale_bcss.json")

    def test_density_scale_template_directory_expands_to_profile_json(self):
        model_paths = {
            "density_scale_json_template": "/configs",
        }

        path = _resolve_model_path(
            model_paths,
            "density_scale_json_template",
            None,
            "PANDA",
            "PANDA",
        )

        self.assertEqual(path, "/configs/density_scale_panda.json")

    def test_nuclei_library_template_directory_expands_to_profile_dir(self):
        model_paths = {
            "nuclei_library_template": "/libraries",
        }

        path = _resolve_model_path(
            model_paths,
            "nuclei_library_template",
            None,
            "BCSS",
            "BCSS",
        )

        self.assertEqual(path, "/libraries/BCSS")


if __name__ == "__main__":
    unittest.main()

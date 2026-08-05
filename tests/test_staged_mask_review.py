import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from phase3_mask_edit.audit.staged_review import (
    _phase3_execution_info,
    audit_nuclei_condition,
    audit_target_mask,
    canonicalize_target_mask_changed_islands,
    normalize_stop_after,
    record_mask_stage_decisions,
    record_nuclei_stage_decisions,
    sha256_file,
    sha256_text,
)
from scripts import prepare_online_mask_review_manifest as prepare_manifest
from scripts import run_phase3_manifest_pipeline as manifest_runner


class StagedMaskReviewTests(unittest.TestCase):
    def test_stop_after_public_names_and_legacy_aliases(self):
        expected = {
            "mask": "mask",
            "tissue": "mask",
            "nuclei": "nuclei",
            "cell": "nuclei",
            "image": "image",
            "generation": "image",
        }
        for value, normalized in expected.items():
            with self.subTest(value=value):
                self.assertEqual(normalize_stop_after(value), normalized)

    def test_mask_only_runner_does_not_call_downstream_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "runtime": {
                            "edit_variants": [
                                {
                                    "variant_id": "instruction",
                                    "edit_mode": "instruction",
                                }
                            ]
                        },
                        "cases": [
                            {
                                "case_id": "01_condition",
                                "dataset": "BCSS",
                                "profile": "BCSS",
                                "source_image": "/unused/source.png",
                                "source_tissue_mask": "/unused/source_mask.png",
                                "source_nuclei_mask": "/unused/source_nuclei.png",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            state = {
                "output_dir": str(root / "runs" / "review" / "01_condition"),
                "reference_image": "/unused/source.png",
                "reference_tissue_mask": "/unused/source_mask.png",
                "target_tissue_mask": "/unused/target_mask.png",
                "change_region": "/unused/change.png",
            }
            tissue_info = {"projection_mode": "organic_v2"}
            mask_stage = {
                "stage": "mask",
                "audit_passed": True,
                "lock_path": "/unused/lock.json",
                "panel_path": "/unused/panel.png",
                "target_tissue_sha256": "abc",
            }
            argv = [
                "--manifest",
                str(manifest_path),
                "--output-root",
                str(root / "runs"),
                "--run-id",
                "review",
                "--stop-after",
                "mask",
            ]
            with (
                mock.patch.object(manifest_runner, "_load_ui_backend"),
                mock.patch.object(
                    manifest_runner,
                    "_resolve_case_paths",
                    return_value={},
                ),
                mock.patch.object(
                    manifest_runner,
                    "_prepare_state",
                    return_value=state,
                ),
                mock.patch.object(
                    manifest_runner,
                    "_run_tissue_stage",
                    return_value=(state, tissue_info),
                ),
                mock.patch.object(
                    manifest_runner,
                    "build_mask_stage_review",
                    return_value=mask_stage,
                ),
                mock.patch.object(
                    manifest_runner,
                    "_run_cell_stage",
                    side_effect=AssertionError("cell stage must not run"),
                ) as cell_stage,
                mock.patch.object(
                    manifest_runner,
                    "_run_generation_stage",
                    side_effect=AssertionError("generation stage must not run"),
                ) as generation_stage,
            ):
                exit_code = manifest_runner.main(argv)

            self.assertEqual(exit_code, 0)
            cell_stage.assert_not_called()
            generation_stage.assert_not_called()
            stage_manifest = json.loads(
                (root / "runs" / "review" / "mask_stage_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(stage_manifest["stage"], "mask")
            self.assertEqual(stage_manifest["approval"]["status"], "pending")
            self.assertFalse(stage_manifest["frozen_target_mask_consumed"])

    def test_approved_mask_resume_preserves_exact_hash_without_tissue_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            source_image = root / "source.png"
            source_tissue = root / "source_tissue.png"
            approved_target = root / "approved_target.png"
            approved_change = root / "approved_change.png"
            Image.fromarray(
                np.full((16, 16, 3), 127, dtype=np.uint8), "RGB"
            ).save(source_image)
            source = np.full((16, 16), 2, dtype=np.uint8)
            target = source.copy()
            target[4:12, 4:12] = 1
            change = (source != target).astype(np.uint8) * 255
            Image.fromarray(source, "L").save(source_tissue)
            Image.fromarray(target, "L").save(approved_target)
            Image.fromarray(change, "L").save(approved_change)
            instruction = "Increase tumor in the central region."
            lock_path = root / "mask_stage_lock.json"
            lock = {
                "case_id": "01_case",
                "dataset": "BCSS",
                "profile": "BCSS",
                "variant_id": "instruction",
                "instruction_sha256": sha256_text(instruction),
                "approval": {
                    "status": "approved",
                    "approved_target_sha256": sha256_file(approved_target),
                },
                "target_tissue_mask_path": str(approved_target),
                "change_region_path": str(approved_change),
                "asset_sha256": {
                    "source_image": sha256_file(source_image),
                    "source_tissue": sha256_file(source_tissue),
                    "target_tissue": sha256_file(approved_target),
                    "change_region": sha256_file(approved_change),
                },
            }
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            state = {
                "reference_image": str(source_image),
                "reference_tissue_mask": str(source_tissue),
            }
            ui = types.SimpleNamespace(
                np=np,
                load_id_mask=lambda path: np.asarray(
                    Image.open(path).convert("L"), dtype=np.uint8
                ),
                load_change_region=lambda path: np.asarray(
                    Image.open(path).convert("L"), dtype=np.uint8
                )
                > 0,
            )

            resumed, tissue_info, stage = (
                manifest_runner._resume_approved_mask_stage(
                    ui=ui,
                    state=state,
                    case={
                        "case_id": "01_case",
                        "dataset": "BCSS",
                        "profile": "BCSS",
                        "instruction": instruction,
                    },
                    variant={
                        "variant_id": "instruction",
                        "edit_mode": "instruction",
                    },
                    approved_entry={
                        "case_id": "01_case",
                        "approval": "approved",
                        "approved_target_sha256": sha256_file(approved_target),
                        "target_tissue_mask_path": str(approved_target),
                        "lock_path": str(lock_path),
                    },
                    approved_manifest_path=root / "approved.json",
                    run_dir=run_dir,
                )
            )

            self.assertEqual(
                sha256_file(resumed["target_tissue_mask"]),
                sha256_file(approved_target),
            )
            self.assertFalse(tissue_info["tissue_stage_rerun"])
            self.assertEqual(stage["approval"], "approved")

    def test_approved_mask_resume_rejects_target_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = np.zeros((8, 8), dtype=np.uint8)
            paths = {}
            for name in ("source_image", "source_tissue", "target", "change"):
                path = root / f"{name}.png"
                Image.fromarray(source, "L").save(path)
                paths[name] = path
            lock_path = root / "lock.json"
            lock_path.write_text(
                json.dumps(
                    {
                        "case_id": "case",
                        "instruction_sha256": sha256_text("instruction"),
                        "approval": {
                            "status": "approved",
                            "approved_target_sha256": "wrong",
                        },
                        "asset_sha256": {"target_tissue": "wrong"},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "target hash"):
                manifest_runner._resume_approved_mask_stage(
                    ui=types.SimpleNamespace(np=np),
                    state={
                        "reference_image": str(paths["source_image"]),
                        "reference_tissue_mask": str(paths["source_tissue"]),
                    },
                    case={"case_id": "case", "instruction": "instruction"},
                    variant={},
                    approved_entry={
                        "approval": "approved",
                        "approved_target_sha256": "different",
                        "lock_path": str(lock_path),
                    },
                    approved_manifest_path=root / "approved.json",
                    run_dir=root / "run",
                )

    def test_nuclei_approval_locks_every_image_stage_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "source_run"
            review_dir = run_dir / "stage_review"
            probnet_dir = run_dir / "probnet_cell_fill"
            review_dir.mkdir(parents=True)
            probnet_dir.mkdir()

            image_assets = {
                "target_tissue": run_dir / "target_tissue.png",
                "target_nuclei": run_dir / "target_nuclei.png",
                "new_nuclei": run_dir / "new_nuclei.png",
                "semantic_change_region": run_dir / "semantic_change.png",
                "generation_change_region": run_dir / "change_region.png",
                "erased_image": run_dir / "erased_image.png",
            }
            for index, path in enumerate(image_assets.values(), start=1):
                if path.name == "erased_image.png":
                    Image.fromarray(
                        np.full((8, 8, 3), index, dtype=np.uint8), "RGB"
                    ).save(path)
                else:
                    Image.fromarray(
                        np.full((8, 8), index, dtype=np.uint8), "L"
                    ).save(path)
            diagnostics_path = (
                probnet_dir / "target_nuclei.diagnostics.json"
            )
            diagnostics_path.write_text("{}", encoding="utf-8")
            cell_log_path = run_dir / "cell_fill_log.json"
            cell_log_path.write_text(
                json.dumps({"status": "cell_done"}),
                encoding="utf-8",
            )

            lock_path = review_dir / "nuclei_stage_lock.json"
            lock = {
                "stage": "nuclei",
                "case_id": "01_case",
                "dataset": "BCSS",
                "profile": "BCSS",
                "approval": {"status": "pending"},
                "parent_target_tissue_sha256": sha256_file(
                    image_assets["target_tissue"]
                ),
                "target_tissue_mask_path": str(
                    image_assets["target_tissue"]
                ),
                "target_nuclei_mask_path": str(
                    image_assets["target_nuclei"]
                ),
                "new_nuclei_mask_path": str(
                    image_assets["new_nuclei"]
                ),
                "change_region_path": str(
                    image_assets["semantic_change_region"]
                ),
                "diagnostics_path": str(diagnostics_path),
                "asset_sha256": {
                    "target_tissue": sha256_file(
                        image_assets["target_tissue"]
                    ),
                    "target_nuclei": sha256_file(
                        image_assets["target_nuclei"]
                    ),
                    "new_nuclei": sha256_file(
                        image_assets["new_nuclei"]
                    ),
                    "change_region": sha256_file(
                        image_assets["semantic_change_region"]
                    ),
                    "probnet_diagnostics": sha256_file(diagnostics_path),
                },
            }
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            manifest_path = root / "nuclei_stage_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "stage": "nuclei",
                        "entries": [
                            {
                                "case_id": "01_case",
                                "dataset": "BCSS",
                                "run_dir": str(run_dir),
                                "lock_path": str(lock_path),
                                "target_nuclei_mask_path": str(
                                    image_assets["target_nuclei"]
                                ),
                                "target_nuclei_sha256": sha256_file(
                                    image_assets["target_nuclei"]
                                ),
                                "approval": "pending",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            approved = record_nuclei_stage_decisions(
                manifest_path,
                approved_case_ids=("01_case",),
            )

            self.assertEqual(approved["approval"]["status"], "approved")
            approved_lock = json.loads(lock_path.read_text(encoding="utf-8"))
            self.assertEqual(
                approved_lock["approval"]["status"], "approved"
            )
            for name in (
                "target_tissue",
                "target_nuclei",
                "new_nuclei",
                "semantic_change_region",
                "generation_change_region",
                "probnet_diagnostics",
                "cell_fill_log",
                "erased_image",
            ):
                self.assertTrue(approved_lock["asset_sha256"].get(name), name)

    def test_image_resume_with_approved_nuclei_skips_cell_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "runtime": {
                            "edit_variants": [
                                {
                                    "variant_id": "instruction",
                                    "edit_mode": "instruction",
                                }
                            ]
                        },
                        "cases": [
                            {
                                "case_id": "01_case",
                                "dataset": "BCSS",
                                "profile": "BCSS",
                                "source_image": "/unused/source.png",
                                "source_tissue_mask": "/unused/tissue.png",
                                "source_nuclei_mask": "/unused/nuclei.png",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            approved_mask_path = root / "approved_mask.json"
            approved_mask_path.write_text(
                json.dumps(
                    {
                        "stage": "mask",
                        "entries": [
                            {
                                "case_id": "01_case",
                                "approval": "approved",
                                "approved_target_sha256": "mask-hash",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            approved_nuclei_path = root / "approved_nuclei.json"
            approved_nuclei_path.write_text(
                json.dumps(
                    {
                        "stage": "nuclei",
                        "entries": [
                            {
                                "case_id": "01_case",
                                "approval": "approved",
                                "approved_target_nuclei_sha256": "cell-hash",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            state = {"output_dir": str(root / "runs" / "01_case")}
            mask_stage = {
                "stage": "mask",
                "approval": "approved",
            }
            nuclei_stage = {
                "stage": "nuclei",
                "approval": "approved",
            }
            argv = [
                "--manifest",
                str(manifest_path),
                "--approved-mask-manifest",
                str(approved_mask_path),
                "--approved-nuclei-manifest",
                str(approved_nuclei_path),
                "--output-root",
                str(root / "runs"),
                "--run-id",
                "review",
                "--stop-after",
                "image",
            ]
            with (
                mock.patch.object(manifest_runner, "_load_ui_backend"),
                mock.patch.object(
                    manifest_runner, "_resolve_case_paths", return_value={}
                ),
                mock.patch.object(
                    manifest_runner, "_prepare_state", return_value=state
                ),
                mock.patch.object(
                    manifest_runner,
                    "_resume_approved_mask_stage",
                    return_value=(
                        state,
                        {"status": "approved_mask_reused"},
                        mask_stage,
                    ),
                ),
                mock.patch.object(
                    manifest_runner,
                    "_resume_approved_nuclei_stage",
                    return_value=(
                        state,
                        {"status": "approved_nuclei_reused"},
                        nuclei_stage,
                    ),
                ),
                mock.patch.object(
                    manifest_runner,
                    "_run_cell_stage",
                    side_effect=AssertionError("cell stage must not run"),
                ) as cell_stage,
                mock.patch.object(
                    manifest_runner,
                    "_run_generation_stage",
                    return_value=(state, {"status": "completed"}),
                ) as generation_stage,
            ):
                exit_code = manifest_runner.main(argv)

            self.assertEqual(exit_code, 0)
            cell_stage.assert_not_called()
            generation_stage.assert_called_once()

    def test_nuclei_audit_requires_exact_counts_types_and_parent_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tissue = np.full((32, 32), 2, dtype=np.uint8)
            nuclei = np.zeros_like(tissue)
            nuclei[8:11, 8:11] = 101
            nuclei[20:23, 20:23] = 103
            tissue_path = root / "target.png"
            Image.fromarray(tissue, "L").save(tissue_path)
            diagnostics = {
                "placed": 2,
                "placed_by_shape_source": {"reference": 1, "library": 1},
                "reference_pool": {"count": 1},
                "tissues": {
                    "2": {
                        "target_count": 2,
                        "placed": 2,
                        "target_by_type": {"101": 1, "103": 1},
                        "placed_by_type": {"101": 1, "103": 1},
                        "candidate_queue_policy": "probnet_queue",
                        "retry_tail_policy": "stable_descending",
                    }
                },
                "patch_adaptive_priors": {
                    "count_policy": "source_density",
                    "type_policy": "density_head",
                },
            }

            audit = audit_nuclei_condition(
                target_tissue=tissue,
                target_nuclei=nuclei,
                new_nuclei=nuclei,
                change_region=np.ones_like(tissue, dtype=bool),
                diagnostics=diagnostics,
                expected_target_sha256=sha256_file(tissue_path),
                target_tissue_path=tissue_path,
            )

            self.assertTrue(audit["passed"])
            self.assertEqual(audit["new_nucleus_component_count"], 2)
            self.assertTrue(
                audit["checks"]["exact_type_quota_per_tissue"]
            )

    def test_nuclei_audit_accepts_posterior_type_policy_without_hard_quota(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tissue = np.full((32, 32), 2, dtype=np.uint8)
            nuclei = np.zeros_like(tissue)
            nuclei[8:11, 8:11] = 101
            nuclei[20:23, 20:23] = 103
            tissue_path = root / "target.png"
            Image.fromarray(tissue, "L").save(tissue_path)
            diagnostics = {
                "placed": 2,
                "tissues": {
                    "2": {
                        "target_count": 2,
                        "placed": 2,
                        "target_by_type": {},
                        "placed_by_type": {"101": 1, "103": 1},
                    }
                },
                "sampling_audit": {
                    "passed": True,
                    "score": 0.88,
                    "tissues": {
                        "2": {
                            "type_applicable": False,
                            "type_passed": True,
                            "spatial_applicable": False,
                            "spatial_passed": True,
                        }
                    },
                },
            }

            audit = audit_nuclei_condition(
                target_tissue=tissue,
                target_nuclei=nuclei,
                new_nuclei=nuclei,
                change_region=np.ones_like(tissue, dtype=bool),
                diagnostics=diagnostics,
                expected_target_sha256=sha256_file(tissue_path),
                target_tissue_path=tissue_path,
            )

            self.assertTrue(audit["passed"])
            self.assertFalse(audit["type_quota_applicable"])
            self.assertTrue(
                audit["checks"]["exact_type_quota_per_tissue"]
            )
            self.assertTrue(
                audit["checks"]["probnet_sampling_alignment_passed"]
            )

    def test_mask_audit_rejects_partial_grade_component(self):
        source = np.zeros((32, 32), dtype=np.uint8)
        source[4:28, 4:28] = 11
        target = source.copy()
        target[4:16, 4:28] = 12

        audit = audit_target_mask(
            source_mask=source,
            target_mask=target,
            profile="GlaS",
            case={
                "primitive": "grade_upgrade",
                "source_labels": ["Tumor"],
                "target_label": "Tumor",
                "expected_area_bucket": [0.08, 0.8],
            },
            phase3_info={"projection_mode": "organic_v2"},
        )

        self.assertFalse(
            audit["checks"]["grade_transition_uses_whole_source_components"]
        )
        self.assertFalse(audit["passed"])

    def test_mask_audit_accepts_whole_grade_component(self):
        source = np.zeros((32, 32), dtype=np.uint8)
        source[4:14, 4:14] = 11
        source[18:28, 18:28] = 11
        target = source.copy()
        target[4:14, 4:14] = 12

        audit = audit_target_mask(
            source_mask=source,
            target_mask=target,
            profile="GlaS",
            case={
                "primitive": "grade_upgrade",
                "source_labels": ["Tumor"],
                "target_label": "Tumor",
                "expected_area_bucket": [0.4, 0.6],
            },
            phase3_info={"projection_mode": "organic_v2"},
        )

        self.assertTrue(
            audit["checks"]["grade_transition_uses_whole_source_components"]
        )

    def test_mask_audit_rejects_exact_rectangle(self):
        source = np.full((40, 40), 2, dtype=np.uint8)
        target = source.copy()
        target[10:30, 10:30] = 1

        audit = audit_target_mask(
            source_mask=source,
            target_mask=target,
            profile="BCSS",
            case={
                "primitive": "tumor_burden_increase",
                "source_labels": ["Stroma"],
                "target_label": "Tumor",
                "expected_area_bucket": [0.1, 0.4],
            },
            phase3_info={"projection_mode": "organic_v2"},
        )

        self.assertFalse(
            audit["checks"]["no_exact_rectangle_or_diamond_components"]
        )

    def test_mask_canonicalizer_prunes_tiny_satellite_but_keeps_main_edit(self):
        source = np.full((256, 256), 1, dtype=np.uint8)
        target = source.copy()
        yy, xx = np.indices(source.shape)
        main = (xx - 128) ** 2 + (yy - 128) ** 2 <= 65**2
        target[main] = 7
        target[8:10, 8:10] = 7

        cleaned, metadata = canonicalize_target_mask_changed_islands(
            source_mask=source,
            target_mask=target,
        )
        audit = audit_target_mask(
            source_mask=source,
            target_mask=cleaned,
            profile="ORCA",
            case={
                "primitive": "tumor_burden_decrease",
                "source_labels": ["Tumor"],
                "target_label": "Other tissue",
                "target_labels": ["Other tissue", "Stroma"],
                "expected_area_bucket": [0.14, 0.24],
            },
            phase3_info={"projection_mode": "organic_v2"},
        )

        self.assertTrue(metadata["applied"])
        self.assertEqual(metadata["removed_component_sizes_px"], [4])
        self.assertTrue(np.array_equal(cleaned[main], target[main]))
        self.assertTrue(np.array_equal(cleaned[8:10, 8:10], source[8:10, 8:10]))
        self.assertTrue(audit["checks"]["no_abnormal_isolated_components"])
        self.assertTrue(audit["passed"])

    def test_mask_audit_uses_product_strength_denominator(self):
        source = np.full((100, 100), 1, dtype=np.uint8)
        source[:15] = 2
        target = source.copy()
        target[:15] = 1

        audit = audit_target_mask(
            source_mask=source,
            target_mask=target,
            profile="BCSS",
            case={
                "primitive": "tumor_burden_increase",
                "source_labels": ["Stroma"],
                "target_label": "Tumor",
                "expected_area_bucket": [0.14, 0.24],
            },
            phase3_info={"projection_mode": "organic_v2"},
        )

        self.assertEqual(audit["changed_area_fraction_legal_source"], 1.0)
        self.assertEqual(
            audit["changed_area_fraction_strength_denominator"], 0.15
        )
        self.assertEqual(audit["strength_denominator_pixels"], 10000)
        self.assertTrue(
            audit["checks"]["changed_area_matches_strength_bucket"]
        )

    def test_mask_audit_enforces_optional_full_image_area_floor(self):
        source = np.full((100, 100), 2, dtype=np.uint8)
        target = source.copy()
        target[:4, :] = 1

        audit = audit_target_mask(
            source_mask=source,
            target_mask=target,
            profile="BCSS",
            case={
                "primitive": "tumor_burden_increase",
                "source_labels": ["Stroma"],
                "target_label": "Tumor",
                "expected_area_bucket": [0.03, 0.05],
                "minimum_changed_area_fraction_image": 0.05,
            },
            phase3_info={"projection_mode": "organic_v2"},
        )

        self.assertAlmostEqual(audit["changed_area_fraction_image"], 0.04)
        self.assertFalse(
            audit["checks"]["changed_area_meets_image_fraction_floor"]
        )
        self.assertFalse(audit["passed"])

    def test_mask_audit_accepts_all_declared_target_labels(self):
        source = np.full((40, 40), 3, dtype=np.uint8)
        target = source.copy()
        target[8:24, 8:16] = 1
        target[8:24, 16:24] = 2

        audit = audit_target_mask(
            source_mask=source,
            target_mask=target,
            profile="IGNITE",
            case={
                "primitive": "necrosis_resolution",
                "source_labels": ["Necrosis"],
                "target_label": "Stroma",
                "target_labels": ["Stroma", "Tumor"],
                "expected_area_bucket": [0.1, 0.3],
            },
            phase3_info={"projection_mode": "organic_v2"},
        )

        self.assertTrue(
            audit["checks"]["changed_target_labels_match_instruction"]
        )
        self.assertEqual(audit["expected_target_ids"], [1, 2])

    def test_mask_audit_rejects_nearly_solid_diamond(self):
        source = np.full((80, 80), 2, dtype=np.uint8)
        target = source.copy()
        yy, xx = np.indices(source.shape)
        diamond = np.abs(xx - 40) + np.abs(yy - 40) <= 20
        target[diamond] = 1

        audit = audit_target_mask(
            source_mask=source,
            target_mask=target,
            profile="BCSS",
            case={
                "primitive": "tumor_burden_increase",
                "source_labels": ["Stroma"],
                "target_label": "Tumor",
                "expected_area_bucket": [0.05, 0.5],
            },
            phase3_info={"projection_mode": "organic_v2"},
        )

        self.assertFalse(
            audit["checks"]["no_exact_rectangle_or_diamond_components"]
        )

    def test_mask_audit_allows_whole_grade_component_area_overshoot(self):
        source = np.zeros((40, 40), dtype=np.uint8)
        source[4:36, 4:36] = 11
        target = source.copy()
        target[4:36, 4:36] = 12

        audit = audit_target_mask(
            source_mask=source,
            target_mask=target,
            profile="GlaS",
            case={
                "primitive": "grade_upgrade",
                "source_labels": ["Tumor"],
                "target_label": "Tumor",
                "expected_area_bucket": [0.08, 0.20],
            },
            phase3_info={"projection_mode": "organic_v2"},
        )

        self.assertTrue(audit["checks"]["changed_area_matches_strength_bucket"])
        self.assertEqual(
            audit["changed_area_bucket_exception"],
            "template_selected_whole_component_overshoot",
        )

    def test_manifest_builder_never_reads_or_exports_frozen_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "sample" / "gt"
            sample.mkdir(parents=True)
            frozen_target = sample / "target_mask.png"
            intent = {
                "sample_id": "sample",
                "primitive": "necrosis_resolution",
                "strength": "moderate",
                "source_labels": ["Necrosis"],
                "target_label": "Stroma",
                "expected_area_bucket": [0.14, 0.24],
                "region_hint": {"description": "lower-left region"},
                "seed": 42,
            }
            (sample / "gt_intent.json").write_text(
                json.dumps(intent), encoding="utf-8"
            )
            row = {
                "condition_id": "condition",
                "sample_id": "sample",
                "profile": "BCSS",
                "organ": "breast",
                "reference_image_path": "/source/image.png",
                "reference_tissue_mask_path": "/source/tissue.png",
                "reference_nuclei_mask_path": "/source/nuclei.png",
                "target_tissue_mask_path": str(frozen_target),
            }

            case = prepare_manifest._build_case(
                row,
                review_index=1,
                api_model="gpt-4.1-mini",
            )

            self.assertFalse(case["frozen_target_mask_consumed"])
            self.assertNotIn("target_tissue_mask", case)
            self.assertNotIn(str(frozen_target), json.dumps(case))
            self.assertIn("necrosis resolution", case["instruction"])
            self.assertEqual(case["expected_area_bucket"], [0.20, 0.30])
            self.assertEqual(
                case["expected_area_bucket_source"],
                "current_product_recipe",
            )

    def test_human_approval_is_bound_to_current_target_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.png"
            target.write_bytes(b"target")
            target_hash = __import__("hashlib").sha256(b"target").hexdigest()
            lock = root / "lock.json"
            lock.write_text(
                json.dumps(
                    {
                        "asset_sha256": {"target_tissue": target_hash},
                        "approval": {"status": "pending"},
                    }
                ),
                encoding="utf-8",
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "case_id": "case",
                                "lock_path": str(lock),
                                "target_tissue_mask_path": str(target),
                                "target_tissue_sha256": target_hash,
                                "approval": "pending",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            manifest = record_mask_stage_decisions(
                manifest_path,
                approved_case_ids=("case",),
            )

            self.assertEqual(manifest["approval"]["status"], "approved")
            self.assertEqual(
                manifest["entries"][0]["approved_target_sha256"],
                target_hash,
            )
            saved_lock = json.loads(lock.read_text(encoding="utf-8"))
            self.assertEqual(saved_lock["approval"]["status"], "approved")

    def test_gleason_instruction_keeps_explicit_transition_evidence(self):
        instruction = prepare_manifest._canonical_instruction(
            {
                "primitive": "gleason_upgrade_4to5",
                "strength": "mild",
                "source_labels": ["Tumor"],
                "target_label": "Tumor",
            }
        )

        self.assertIn(
            "Gleason pattern 4 to Gleason pattern 5 upgrade",
            instruction,
        )

    def test_nested_execution_summary_confirms_organic_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "instruction" / "primitive" / "execution_summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                json.dumps({"projection_mode": "organic_v2"}),
                encoding="utf-8",
            )

            info = _phase3_execution_info(root, tissue_info={})

            self.assertEqual(info["projection_mode"], "organic_v2")
            self.assertEqual(info["execution_summary_count"], 1)


if __name__ == "__main__":
    unittest.main()

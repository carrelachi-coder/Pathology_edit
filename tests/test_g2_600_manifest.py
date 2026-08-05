from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from scripts.build_g2_600_manifest import (
    MIN_CHANGED_AREA_FRACTION_IMAGE,
    MIN_TUMOR_NON_TUMOR_BOUNDARY_PIXELS,
    _balanced_cell_targets,
    _quota_feasibility_issues,
    _validate_annotation_paths,
    build_product_manifest,
    eligible_primitives,
    load_candidate_rows,
    validate_reserves,
)
from scripts.run_g2_600 import (
    _partition_case_ids,
    _allocate_reserve_rows,
    _allocate_resume_cohort,
    _collect_nuclei_results,
    _next_nuclei_repair_run_id,
    _nuclei_result_passed,
    _restore_mask_round_history,
    _reconcile_mask_cohort,
    run_image_stage,
    summarize_runs,
)


class G2600ManifestTests(unittest.TestCase):
    def test_reconciliation_does_not_allocate_slots_to_historical_extras(self):
        accepted_cases = [
            {
                "case_id": "keep",
                "organ": "breast",
                "g2_primitive": "tumor_increase",
            },
            {
                "case_id": "historical-extra",
                "organ": "breast",
                "g2_primitive": "tumor_increase",
            },
        ]
        accepted_entries = [
            {"case_id": case["case_id"]} for case in accepted_cases
        ]
        unresolved = [
            {
                "case_id": "current-failure",
                "organ": "breast",
                "g2_primitive": "tumor_increase",
            }
        ]
        reserve = {
            "dataset": "BCSS",
            "stem": "replacement",
            "wsi": "wsi-r",
            "organ": "breast",
            "g2_primitive": "tumor_increase",
        }
        replacement_case = {
            "case_id": "replacement-case",
            "organ": "breast",
            "g2_primitive": "tumor_increase",
        }
        allocation = {
            "retained_case_ids": ["keep"],
            "selected_reserves": {
                ("breast", "tumor_increase"): [reserve]
            },
        }
        source_manifest = {
            "runtime": {"verification": {"product_release": "release.json"}}
        }

        with mock.patch(
            "scripts.run_g2_600._allocate_resume_cohort",
            return_value=allocation,
        ), mock.patch(
            "scripts.run_g2_600.build_product_manifest",
            return_value={"cases": [replacement_case]},
        ):
            result = _reconcile_mask_cohort(
                accepted_entries=accepted_entries,
                accepted_cases=accepted_cases,
                unresolved_cases=unresolved,
                target_cases=[accepted_cases[0], unresolved[0]],
                reserve_index={},
                used_stems=set(),
                seed=42,
                source_manifest_path=Path("manifest.json"),
                source_manifest=source_manifest,
            )

        self.assertEqual(
            result["pending_cases"][0]["replacement_for_case_id"],
            "current-failure",
        )
        self.assertEqual(
            result["provenance"][
                "unassigned_historical_displaced_case_ids"
            ],
            ["historical-extra"],
        )
    def test_product_manifest_freezes_human_visible_change_floor(self):
        selected = [
            {
                "dataset": "BCSS",
                "stem": "sample-a",
                "organ": "breast",
                "g2_primitive": "tumor_increase",
                "image_path": "/tmp/image.png",
                "id_mask_path": "/tmp/tissue.png",
                "cellvit_id_mask_path": "/tmp/nuclei.png",
                "annotation_mask_sha256": "a" * 64,
                "wsi": "wsi-a",
                "valid_pixels": 100,
                "pix_tumor": 25,
                "pix_stroma": 75,
                "pix_necrosis": 0,
                "pix_immune_infiltrate": 0,
                "pix_normal_epithelium": 0,
                "pix_blood_vessel": 0,
                "pix_other_tissue": 0,
                "annotated_coarse_ids": [1, 2],
            }
        ]

        manifest = build_product_manifest(
            selected,
            seed=42,
            source_manifest="eval_meta.json",
            release_path="release.json",
        )

        self.assertEqual(
            manifest["cases"][0]["minimum_changed_area_fraction_image"],
            MIN_CHANGED_AREA_FRACTION_IMAGE,
        )
        self.assertEqual(
            manifest["mask_review_policy"][
                "minimum_changed_area_fraction_image"
            ],
            MIN_CHANGED_AREA_FRACTION_IMAGE,
        )
        context_policy = manifest["g2_constraints"][
            "tumor_edit_context_policy"
        ]
        self.assertEqual(
            context_policy["tumor_increase_non_tumor_fraction_min"],
            0.14,
        )
        self.assertEqual(
            context_policy["tumor_decrease_named_backfill_fraction_min"],
            0.05,
        )
        self.assertEqual(
            context_policy["other_only_non_tumor_fraction_min"],
            0.20,
        )
    def test_parallel_image_stage_uses_product_runner_once_per_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_ids = [f"case-{index:03d}" for index in range(7)]
            manifest = root / "manifest.json"
            approved = root / "approved_nuclei.json"
            manifest.write_text(
                json.dumps(
                    {"cases": [{"case_id": case_id} for case_id in case_ids]}
                ),
                encoding="utf-8",
            )
            approved.write_text(
                json.dumps(
                    {
                        "all_automatic_checks_passed": True,
                        "entries": [
                            {"case_id": case_id, "approval": "approved"}
                            for case_id in case_ids
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                manifest=manifest,
                approved_mask_manifest=root / "approved_masks.json",
                approved_nuclei_manifest=approved,
                output=root / "run",
                expected_count=len(case_ids),
                gpu_ids="1,3,5",
                max_repair_rounds=2,
            )
            processes = []

            def fake_popen(command, **kwargs):
                process = mock.Mock()
                process.wait.return_value = 0
                process.command = command
                process.environment = kwargs["env"]
                processes.append(process)
                return process

            with mock.patch(
                "scripts.run_g2_600.subprocess.Popen",
                side_effect=fake_popen,
            ), mock.patch(
                "scripts.run_g2_600._image_result_passed",
                return_value=True,
            ), mock.patch(
                "scripts.run_g2_600.summarize_runs",
                return_value=0,
            ) as summarize:
                self.assertEqual(run_image_stage(args), 0)

            self.assertEqual(len(processes), 3)
            self.assertEqual(
                [
                    process.environment["CUDA_VISIBLE_DEVICES"]
                    for process in processes
                ],
                ["1", "3", "5"],
            )
            scheduled = []
            for process in processes:
                command = process.command
                scheduled.extend(
                    command[index + 1]
                    for index, value in enumerate(command)
                    if value == "--case-id"
                )
            self.assertEqual(set(scheduled), set(case_ids))
            self.assertEqual(len(scheduled), len(case_ids))
            summarize.assert_called_once_with(
                args.output / "image",
                args.output / "summary",
                expected_count=len(case_ids),
            )

    def test_image_shards_are_deterministic_disjoint_and_complete(self):
        case_ids = [f"case-{index:03d}" for index in range(17)]

        first = _partition_case_ids(case_ids, 5)
        second = _partition_case_ids(case_ids, 5)

        self.assertEqual(first, second)
        self.assertEqual(
            [case_id for shard in first for case_id in shard],
            [
                *case_ids[0::5],
                *case_ids[1::5],
                *case_ids[2::5],
                *case_ids[3::5],
                *case_ids[4::5],
            ],
        )
        self.assertEqual(
            set().union(*(set(shard) for shard in first)),
            set(case_ids),
        )
        self.assertEqual(sum(len(shard) for shard in first), len(case_ids))

    def test_nuclei_resume_uses_latest_repair_result_per_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nuclei").mkdir()
            (root / "nuclei_repair_01").mkdir()
            failed = {"case_id": "case-a", "status": "failed"}
            passed = {
                "case_id": "case-a",
                "status": "completed",
                "cell": {
                    "gland_structure_policy": {
                        "applied": False,
                        "cell_deletion_region_policy": (
                            "semantic_change_region"
                        ),
                        "nuclei_generation_region_policy": (
                            "semantic_change_region_plus_complete_intersecting_instances"
                        ),
                    },
                },
                "nuclei_stage": {"audit_passed": True},
            }
            (root / "nuclei" / "batch_summary.json").write_text(
                json.dumps({"results": [failed]}), encoding="utf-8"
            )
            (
                root / "nuclei_repair_01" / "batch_summary.json"
            ).write_text(
                json.dumps({"results": [passed]}), encoding="utf-8"
            )

            results = _collect_nuclei_results(
                root,
                expected_case_ids=["case-a"],
            )

            self.assertTrue(_nuclei_result_passed(results["case-a"]))
            self.assertEqual(_next_nuclei_repair_run_id(root), "nuclei_repair_02")

    def test_nuclei_gate_requires_whole_gland_cell_rewrite_for_glas(self):
        result = {
            "status": "completed",
            "dataset": "GLAS",
            "cell": {
                "gland_structure_policy": {
                    "applied": True,
                    "cell_deletion_region_policy": (
                        "whole_glas_connected_component"
                    ),
                    "nuclei_generation_region_policy": (
                        "whole_glas_connected_component"
                    ),
                    "image_and_nuclei_region_equal": True,
                },
            },
            "nuclei_stage": {"audit_passed": True},
        }

        self.assertTrue(_nuclei_result_passed(result))
        result["cell"]["gland_structure_policy"].update(
            {
                "cell_deletion_region_policy": "semantic_change_region",
                "nuclei_generation_region_policy": (
                    "semantic_change_region_plus_complete_intersecting_instances"
                ),
            }
        )
        self.assertFalse(_nuclei_result_passed(result))

    def test_mask_round_history_restores_only_unresolved_replacement_chains(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            round_one_cases = [
                {
                    "case_id": "case-a",
                    "condition_id": "tumor_increase",
                    "dataset": "BCSS",
                    "sample_id": "sample-a",
                },
                {
                    "case_id": "case-b",
                    "condition_id": "tumor_increase",
                    "dataset": "BCSS",
                    "sample_id": "sample-b",
                },
            ]
            round_two_cases = [
                {
                    "case_id": "case-c",
                    "condition_id": "tumor_increase",
                    "dataset": "BCSS",
                    "sample_id": "sample-c",
                    "replacement_for_case_id": "case-b",
                }
            ]
            (root / "mask_round_01").mkdir()
            (root / "mask_round_02").mkdir()
            (root / "mask_round_01.json").write_text(
                json.dumps({"cases": round_one_cases}), encoding="utf-8"
            )
            (root / "mask_round_02.json").write_text(
                json.dumps({"cases": round_two_cases}), encoding="utf-8"
            )
            (root / "mask_round_01" / "batch_summary.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "case_id": "case-a",
                                "dataset": "BCSS",
                                "variant_id": "instruction",
                                "output_dir": "/tmp/case-a",
                                "status": "completed",
                                "mask_stage": {
                                    "audit_passed": True,
                                    "lock_path": "/tmp/case-a-lock.json",
                                },
                            },
                            {
                                "case_id": "case-b",
                                "status": "completed",
                                "mask_stage": {"audit_passed": False},
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (root / "mask_round_02" / "batch_summary.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "case_id": "case-c",
                                "status": "completed",
                                "mask_stage": {"audit_passed": False},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            restored = _restore_mask_round_history(root)

        self.assertEqual(
            [entry["case_id"] for entry in restored["accepted_entries"]],
            ["case-a"],
        )
        self.assertEqual(
            [case["case_id"] for case in restored["rejected_cases"]],
            ["case-c"],
        )
        self.assertEqual(restored["next_round"], 3)
        self.assertEqual(
            restored["used_stems"],
            {
                ("BCSS", "sample-a"),
                ("BCSS", "sample-b"),
                ("BCSS", "sample-c"),
            },
        )

    def test_mask_round_history_rejects_partial_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "mask_round_01").mkdir()
            (root / "mask_round_01.json").write_text(
                json.dumps(
                    {
                        "cases": [
                            {"case_id": "case-a"},
                            {"case_id": "case-b"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (root / "mask_round_01" / "batch_summary.json").write_text(
                json.dumps({"results": [{"case_id": "case-a"}]}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                _restore_mask_round_history(root)

    def test_contract_rejection_from_earlier_round_remains_unresolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = []
            for index, case_id in enumerate(("case-a", "case-b"), start=1):
                case = {
                    "case_id": case_id,
                    "dataset": "BCSS",
                    "sample_id": f"sample-{index}",
                }
                cases.append(case)
                run = root / f"mask_round_{index:02d}"
                run.mkdir()
                (root / f"mask_round_{index:02d}.json").write_text(
                    json.dumps({"cases": [case]}), encoding="utf-8"
                )
                (run / "batch_summary.json").write_text(
                    json.dumps(
                        {
                            "results": [
                                {
                                    "case_id": case_id,
                                    "status": "completed",
                                    "mask_stage": {"audit_passed": True},
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )

            with mock.patch(
                "scripts.run_g2_600._revalidate_mask_stage",
                side_effect=[
                    {"audit_passed": False},
                    {"audit_passed": True},
                ],
            ):
                restored = _restore_mask_round_history(
                    root, current_cases=cases
                )

        self.assertEqual(
            [case["case_id"] for case in restored["rejected_cases"]],
            ["case-a"],
        )

    def test_mask_round_history_revalidates_against_current_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "mask_round_01" / "case-a" / "instruction"
            review_dir = run_dir / "stage_review"
            review_dir.mkdir(parents=True)
            source_path = root / "source.png"
            target_path = run_dir / "target_mask.png"
            lock_path = review_dir / "mask_stage_lock.json"
            source = np.full((40, 40), 3, dtype=np.uint8)
            target = source.copy()
            target[8:24, 8:16] = 1
            target[8:24, 16:24] = 2
            Image.fromarray(source).save(source_path)
            Image.fromarray(target).save(target_path)
            lock_path.write_text(json.dumps({"stage": "mask"}), encoding="utf-8")
            historical_case = {
                "case_id": "case-a",
                "condition_id": "necrosis_decrease",
                "dataset": "IGNITE",
                "profile": "IGNITE",
                "sample_id": "sample-a",
                "source_tissue_mask": str(source_path),
                "source_labels": ["Necrosis"],
                "target_label": "Stroma",
                "expected_area_bucket": [0.1, 0.3],
            }
            (root / "mask_round_01.json").write_text(
                json.dumps({"cases": [historical_case]}), encoding="utf-8"
            )
            (root / "mask_round_01" / "batch_summary.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "case_id": "case-a",
                                "dataset": "IGNITE",
                                "variant_id": "instruction",
                                "output_dir": str(run_dir),
                                "status": "completed",
                                "tissue": {"projection_mode": "organic_v2"},
                                "mask_stage": {
                                    "audit_passed": False,
                                    "audit_path": str(
                                        review_dir / "mask_audit.json"
                                    ),
                                    "lock_path": str(lock_path),
                                    "target_tissue_mask_path": str(target_path),
                                    "target_tissue_sha256": "target-sha",
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            current_case = {
                **historical_case,
                "target_labels": ["Stroma", "Tumor"],
            }

            with mock.patch(
                "scripts.run_g2_600.audit_target_mask",
                return_value={
                    "passed": True,
                    "expected_target_ids": [1, 2],
                },
            ) as audit_target:
                restored = _restore_mask_round_history(
                    root,
                    current_cases=[current_case],
                )

            self.assertEqual(len(restored["accepted_entries"]), 1)
            self.assertEqual(restored["rejected_cases"], [])
            self.assertEqual(restored["revalidation_count"], 1)
            self.assertEqual(restored["revalidation_passed"], 1)
            self.assertEqual(
                restored["accepted_cases"][0]["target_labels"],
                ["Stroma", "Tumor"],
            )
            audit_path = Path(restored["accepted_entries"][0]["audit_path"])
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertTrue(audit["passed"])
            self.assertEqual(audit["expected_target_ids"], [1, 2])
            self.assertEqual(
                audit["resume_revalidation"]["source"],
                "current_product_contract",
            )
            self.assertEqual(
                audit_target.call_args.kwargs["case"]["target_labels"],
                ["Stroma", "Tumor"],
            )

    def test_product_manifest_uses_frozen_online_route_threshold(self):
        manifest = build_product_manifest(
            [],
            seed=42,
            source_manifest="eval_meta.json",
            release_path="benchmark_configs/releases/online_agent_product_v1.json",
        )

        self.assertEqual(
            manifest["runtime"]["generation"]["route_threshold"],
            0.30,
        )

    def test_necrosis_decrease_manifest_records_both_legal_backfills(self):
        manifest = build_product_manifest(
            [
                {
                    "g2_primitive": "necrosis_decrease",
                    "dataset": "IGNITE",
                    "stem": "sample-a",
                    "organ": "lung",
                    "image_path": "/data/images/sample-a.png",
                    "id_mask_path": "/data/tissue_masks/sample-a.png",
                    "cellvit_id_mask_path": "/data/nuclei_masks/sample-a.png",
                    "annotation_mask_sha256": "abc",
                    "wsi": "wsi-a",
                    "valid_pixels": 100,
                    "annotated_coarse_ids": [1, 2, 3],
                }
            ],
            seed=42,
            source_manifest="eval_meta.json",
            release_path="benchmark_configs/releases/online_agent_product_v1.json",
        )

        case = manifest["cases"][0]
        self.assertEqual(case["target_label"], "Stroma")
        self.assertEqual(case["target_labels"], ["Stroma", "Tumor"])

    def test_decrease_targets_are_derived_from_product_recipe(self):
        expected = {
            "tumor_decrease": [
                "Stroma",
                "Other tissue",
                "Normal epithelium",
                "Immune infiltrate",
            ],
            "stroma_decrease": [
                "Tumor",
                "Other tissue",
                "Normal epithelium",
            ],
            "immune_decrease": ["Stroma", "Other tissue", "Tumor"],
            "necrosis_decrease": ["Stroma", "Tumor"],
        }
        rows = []
        for index, primitive in enumerate(expected):
            rows.append(
                {
                    "g2_primitive": primitive,
                    "dataset": "BCSS",
                    "stem": f"sample-{index}",
                    "organ": "breast",
                    "image_path": f"/data/images/sample-{index}.png",
                    "id_mask_path": f"/data/tissue_masks/sample-{index}.png",
                    "cellvit_id_mask_path": (
                        f"/data/nuclei_masks/sample-{index}.png"
                    ),
                    "annotation_mask_sha256": f"sha-{index}",
                    "wsi": f"wsi-{index}",
                    "valid_pixels": 100,
                    "annotated_coarse_ids": [1, 2, 3, 4, 5, 7],
                }
            )

        manifest = build_product_manifest(
            rows,
            seed=42,
            source_manifest="eval_meta.json",
            release_path="benchmark_configs/releases/online_agent_product_v1.json",
        )

        actual = {
            case["g2_primitive"]: case["target_labels"]
            for case in manifest["cases"]
        }
        self.assertEqual(actual, expected)

    def test_reserve_allocation_solves_greedy_wsi_cap_trap(self):
        selected = _allocate_reserve_rows(
            [
                {"organ": "lung", "g2_primitive": "primitive-a"},
                {"organ": "lung", "g2_primitive": "primitive-b"},
            ],
            reserve_index={
                ("lung", "primitive-a"): [
                    {
                        "dataset": "IGNITE",
                        "stem": "a-full",
                        "wsi": "wsi-full",
                        "reserve_rank": "1",
                    },
                    {
                        "dataset": "IGNITE",
                        "stem": "a-open",
                        "wsi": "wsi-open",
                        "reserve_rank": "2",
                    },
                ],
                ("lung", "primitive-b"): [
                    {
                        "dataset": "IGNITE",
                        "stem": "b-full",
                        "wsi": "wsi-full",
                        "reserve_rank": "1",
                    }
                ],
            },
            used_stems=set(),
            used_wsi_counts={
                ("lung", "wsi-full"): 1,
                ("lung", "wsi-open"): 0,
            },
            source_manifest={
                "selection_policy": {"actual_wsi_caps": {"lung": 2}}
            },
        )

        self.assertEqual(
            selected[("lung", "primitive-a")][0]["stem"],
            "a-open",
        )
        self.assertEqual(
            selected[("lung", "primitive-b")][0]["stem"],
            "b-full",
        )

    def test_resume_cohort_minimally_swaps_accepted_case_for_wsi_capacity(self):
        accepted = [
            {
                "case_id": "accepted-a-1",
                "dataset": "IGNITE",
                "sample_id": "accepted-a-1",
                "organ": "lung",
                "g2_primitive": "primitive-a",
                "wsi": "wsi-full",
            },
            {
                "case_id": "accepted-a-2",
                "dataset": "IGNITE",
                "sample_id": "accepted-a-2",
                "organ": "lung",
                "g2_primitive": "primitive-a",
                "wsi": "wsi-full",
            },
        ]
        target = [
            *accepted,
            {
                "case_id": "failed-b",
                "dataset": "IGNITE",
                "sample_id": "failed-b",
                "organ": "lung",
                "g2_primitive": "primitive-b",
                "wsi": "wsi-full",
            },
        ]

        allocation = _allocate_resume_cohort(
            accepted_cases=accepted,
            target_cases=target,
            reserve_index={
                ("lung", "primitive-a"): [
                    {
                        "dataset": "IGNITE",
                        "stem": "reserve-a-open",
                        "wsi": "wsi-open",
                        "reserve_rank": "1",
                    }
                ],
                ("lung", "primitive-b"): [
                    {
                        "dataset": "IGNITE",
                        "stem": "reserve-b-full",
                        "wsi": "wsi-full",
                        "reserve_rank": "1",
                    }
                ],
            },
            used_stems={
                ("IGNITE", "accepted-a-1"),
                ("IGNITE", "accepted-a-2"),
                ("IGNITE", "failed-b"),
            },
            source_manifest={
                "selection_policy": {"actual_wsi_caps": {"lung": 2}}
            },
        )

        self.assertEqual(len(allocation["retained_case_ids"]), 1)
        self.assertEqual(
            allocation["selected_reserves"][("lung", "primitive-a")][0][
                "stem"
            ],
            "reserve-a-open",
        )
        self.assertEqual(
            allocation["selected_reserves"][("lung", "primitive-b")][0][
                "stem"
            ],
            "reserve-b-full",
        )

    def test_machine_generated_mask_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "forbids machine-generated"):
            _validate_annotation_paths(
                "BCSS",
                image_path=Path("/data/BCSS_PATCHES/images/a.png"),
                tissue_path=Path(
                    "/data/BCSS_PATCHES/model_masks/a.png"
                ),
                nuclei_path=Path(
                    "/data/BCSS_PATCHES/nuclei_masks/a.png"
                ),
            )

    def test_eval_metadata_deduplicates_targets_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "BCSS_PATCHES"
            images = root / "images"
            tissue = root / "tissue_masks"
            nuclei = root / "nuclei_masks"
            images.mkdir(parents=True)
            tissue.mkdir()
            nuclei.mkdir()
            Image.new("RGB", (32, 32), "white").save(images / "a.png")
            mask = np.full((32, 32), 2, dtype=np.uint8)
            mask[:8] = 1
            Image.fromarray(mask).save(tissue / "a.png")
            Image.fromarray(np.zeros_like(mask)).save(nuclei / "a.png")
            pair = {
                "dataset": "BCSS",
                "sample_id": "a",
                "case_id": "wsi-a",
                "target_image": str(images / "a.png"),
                "target_tissue_mask": str(tissue / "a.png"),
                "target_nuclei_mask": str(nuclei / "a.png"),
            }
            metadata = Path(tmp) / "metadata_cross_val.json"
            metadata.write_text(
                json.dumps({"pairs": [pair, pair]}),
                encoding="utf-8",
            )

            rows = load_candidate_rows(metadata)

        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["annotation_provenance"],
            "human_dataset_annotation",
        )
        self.assertIn("tumor_increase", eligible_primitives(rows[0]))
        self.assertNotIn("immune_increase", eligible_primitives(rows[0]))

    def test_annotation_support_conflict_is_reported_before_selection(self):
        policy = {
            "primitive_supported_organs": {
                "immune_increase": ["breast", "lung"],
                "immune_decrease": ["breast", "lung"],
                "necrosis_increase": ["breast", "lung"],
                "necrosis_decrease": ["breast", "lung"],
                "tumor_increase": [
                    "breast",
                    "colorectal",
                    "lung",
                    "oral",
                    "prostate",
                    "skin",
                ],
            },
            "organ_supported_primitives": {
                "breast": [
                    "immune_increase",
                    "immune_decrease",
                    "necrosis_increase",
                    "necrosis_decrease",
                    "tumor_increase",
                ],
                "colorectal": ["tumor_increase"],
                "lung": [
                    "immune_increase",
                    "immune_decrease",
                    "necrosis_increase",
                    "necrosis_decrease",
                    "tumor_increase",
                ],
                "oral": ["tumor_increase"],
                "prostate": ["tumor_increase"],
                "skin": ["tumor_increase"],
            },
        }

        issues = _quota_feasibility_issues(policy)

        bottleneck = next(
            issue
            for issue in issues
            if issue["constraint"]
            == "annotation_supported_organ_capacity"
        )
        self.assertEqual(bottleneck["required_assignments"], 300)
        self.assertEqual(bottleneck["available_organ_slots"], 200)
        self.assertEqual(bottleneck["deficit"], 100)

    def test_generic_stroma_increase_does_not_require_tumor(self):
        row = {
            "valid_pixels": 100,
            "pix_tumor": 0,
            "pix_stroma": 50,
            "pix_necrosis": 0,
            "pix_immune_infiltrate": 0,
            "pix_normal_epithelium": 50,
            "pix_blood_vessel": 0,
            "pix_other_tissue": 0,
            "annotated_coarse_ids": [1, 2, 5],
        }

        self.assertIn("stroma_increase", eligible_primitives(row))

    def test_tumor_edits_reject_sparse_other_only_context(self):
        row = {
            "valid_pixels": 100_000,
            "pix_tumor": 80_000,
            "pix_stroma": 0,
            "pix_necrosis": 0,
            "pix_immune_infiltrate": 0,
            "pix_normal_epithelium": 0,
            "pix_blood_vessel": 0,
            "pix_other_tissue": 15_000,
            "pix_tumor_non_tumor_boundary_support": (
                MIN_TUMOR_NON_TUMOR_BOUNDARY_PIXELS
            ),
            "annotated_coarse_ids": [1, 7],
        }

        eligible = eligible_primitives(row)

        self.assertNotIn("tumor_increase", eligible)
        self.assertNotIn("tumor_decrease", eligible)

    def test_tumor_decrease_accepts_visible_adjacent_known_backfill(self):
        row = {
            "valid_pixels": 100_000,
            "pix_tumor": 80_000,
            "pix_stroma": 6_000,
            "pix_necrosis": 0,
            "pix_immune_infiltrate": 0,
            "pix_normal_epithelium": 0,
            "pix_blood_vessel": 0,
            "pix_other_tissue": 14_000,
            "pix_tumor_non_tumor_boundary_support": (
                MIN_TUMOR_NON_TUMOR_BOUNDARY_PIXELS
            ),
            "annotated_coarse_ids": [1, 2, 7],
        }

        self.assertIn("tumor_decrease", eligible_primitives(row))

    def test_tumor_decrease_accepts_named_backfill_below_moderate_floor(self):
        row = {
            "valid_pixels": 100_000,
            "pix_tumor": 80_000,
            "pix_stroma": 6_000,
            "pix_necrosis": 0,
            "pix_immune_infiltrate": 0,
            "pix_normal_epithelium": 0,
            "pix_blood_vessel": 0,
            "pix_other_tissue": 0,
            "pix_tumor_non_tumor_boundary_support": (
                MIN_TUMOR_NON_TUMOR_BOUNDARY_PIXELS
            ),
            "annotated_coarse_ids": [1, 2, 7],
        }

        self.assertIn("tumor_decrease", eligible_primitives(row))

    def test_balanced_targets_cap_sparse_cells_and_redistribute(self):
        primitives = (
            "tumor_increase",
            "tumor_decrease",
            "stroma_increase",
            "stroma_decrease",
        )
        policy = {
            "organ_supported_primitives": {
                organ: list(primitives)
                for organ in (
                    "breast",
                    "colorectal",
                    "lung",
                    "oral",
                    "prostate",
                    "skin",
                )
            },
            "capacities": {
                organ: {
                    primitive: (
                        12
                        if organ == "skin"
                        and primitive == "stroma_increase"
                        else 200
                    )
                    for primitive in (
                        "tumor_increase",
                        "tumor_decrease",
                        "stroma_increase",
                        "stroma_decrease",
                        "immune_increase",
                        "immune_decrease",
                        "necrosis_increase",
                        "necrosis_decrease",
                    )
                }
                for organ in (
                    "breast",
                    "colorectal",
                    "lung",
                    "oral",
                    "prostate",
                    "skin",
                )
            },
        }

        targets = _balanced_cell_targets(policy)

        self.assertEqual(targets["skin"]["stroma_increase"], 7)
        self.assertEqual(sum(targets["skin"].values()), 100)
        self.assertEqual(
            {
                targets["skin"]["tumor_increase"],
                targets["skin"]["tumor_decrease"],
                targets["skin"]["stroma_decrease"],
            },
            {31},
        )

    def test_active_cell_requires_same_cell_reserves(self):
        selected = [
            {
                "g2_organ": "breast",
                "g2_primitive": "immune_increase",
            }
        ]
        reserves = [
            {
                "g2_organ": "breast",
                "g2_primitive": "immune_increase",
            }
            for _ in range(4)
        ]

        with self.assertRaisesRegex(ValueError, "lack frozen same-cell"):
            validate_reserves(selected, reserves)

    def test_smoke_summary_uses_same_product_schema_with_smaller_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = root / "run" / "case-a"
            generation = case / "agentic_generation"
            generation.mkdir(parents=True)
            (case / "run_config.json").write_text(
                json.dumps(
                    {
                        "case": {
                            "case_id": "case-a",
                            "organ": "breast",
                            "dataset": "BCSS",
                            "g2_primitive": "tumor_increase",
                        }
                    }
                ),
                encoding="utf-8",
            )
            image = generation / "generated_image.png"
            Image.new("RGB", (8, 8), "white").save(image)
            verification = {
                "passed": True,
                "quality_score": 0.8,
                "evidence_coverage": 1.0,
                "scientific_status": "validated",
                "failed_checks": [],
                "reason_codes": [],
                "component_scores": {"semantic": 0.8},
                "metrics": {},
            }
            attempt = {
                "attempt_index": 1,
                "requested_mode": "inpaint",
                "artifact": {"image_path": str(image)},
                "verification": verification,
            }
            (generation / "pipeline_summary.json").write_text(
                json.dumps(
                    {
                        "status": "validated_first_pass",
                        "attempts": [attempt],
                        "selected_attempt": attempt,
                    }
                ),
                encoding="utf-8",
            )

            result = summarize_runs(
                root / "run",
                root / "summary",
                expected_count=1,
            )

            self.assertEqual(result, 0)
            payload = json.loads(
                (root / "summary" / "g2_600_summary.json").read_text()
            )
            self.assertEqual(payload["final_count"], 1)
            self.assertEqual(
                payload["evaluator_policy_id"],
                "online-quality-evaluator-v2.4",
            )
            self.assertEqual(
                payload["preservation_exclusion_region_policy"],
                "full_generation_change_region",
            )


if __name__ == "__main__":
    unittest.main()

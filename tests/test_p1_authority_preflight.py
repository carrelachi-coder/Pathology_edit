from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from phase3_joint_edit_refine.auxiliary import materialize_profile_auxiliaries
from phase3_joint_edit_refine.models import JointCaseContext
from phase3_joint_edit_refine.p1_authority_preflight import (
    OUTPUT_FILENAMES,
    _build_live_case,
    _compile_live_preflight,
    _effective_source_row,
    _external_auxiliary_authority,
    _profile_required_provenance,
    _runtime_authority,
    _source_authority,
    build_artifacts,
    validate_artifacts,
)
from phase3_joint_edit_refine.p1_authority_preflight import (
    _canonical_json_bytes as canonical_json_bytes,
)
from phase3_joint_edit_refine.p1_authority_preflight import (
    _sealed_record as sealed_record,
)
from phase3_joint_edit_refine.portfolio_authority import (
    array_sha256,
    canonical_metadata_sha256,
)
from phase3_joint_edit_refine.skills.repository import JointSkillRepository
from phase3_mask_edit_refine.evidence import load_id_mask, sha256_file
from phase3_mask_edit_refine.skills import SkillRepository as MaskSkillRepository

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "phase3_joint_edit_refine" / "resources"
SELECTION = RESOURCES / "p1_glas_panda_meta_eval_selection_v1.json"
SOURCE = RESOURCES / "p1_glas_panda_source_case_pool_v1.json"
CODE_COMMIT = "a" * 40


class P1AuthorityPreflightTests(unittest.TestCase):
    def _build(self):
        return build_artifacts(
            root=ROOT,
            selection_path=SELECTION,
            source_manifest_path=SOURCE,
            code_commit=CODE_COMMIT,
        )

    @staticmethod
    def _reseal_manifest(payload):
        unsigned = dict(payload)
        unsigned.pop("manifest_content_sha256", None)
        return {
            **unsigned,
            "manifest_content_sha256": canonical_metadata_sha256(unsigned),
        }

    @staticmethod
    def _write_live_assets(root: Path, *, profile_id: str):
        size = 256
        rows, cols = np.ogrid[:size, :size]
        tissue = np.full((size, size), 2, dtype=np.uint8)
        tumor = (rows - 128) ** 2 + (cols - 128) ** 2 <= 70**2
        tissue[tumor] = 12 if profile_id == "glas-gland-v1" else 9
        nuclei = np.zeros_like(tissue)
        for y in range(12, size - 12, 10):
            for x in range(12, size - 12, 10):
                nuclei[y - 1 : y + 2, x - 1 : x + 2] = (
                    1 if tumor[y, x] else 3
                )
        image = np.full((size, size, 3), (220, 185, 202), dtype=np.uint8)
        image[tumor] = (170, 90, 132)
        image_path = root / "image.png"
        tissue_path = root / "tissue.png"
        nuclei_path = root / "nuclei.png"
        Image.fromarray(image).save(image_path)
        Image.fromarray(tissue).save(tissue_path)
        Image.fromarray(nuclei).save(nuclei_path)
        row = {
            "case_id": "synthetic-live-" + profile_id,
            "case_record_sha256": "c" * 64,
            "organic_seed": 17,
            "source_image": str(image_path),
            "source_image_sha256": sha256_file(image_path),
            "source_tissue_mask": str(tissue_path),
            "source_tissue_mask_sha256": sha256_file(tissue_path),
            "source_nuclei_mask": str(nuclei_path),
            "source_nuclei_mask_sha256": sha256_file(nuclei_path),
        }
        return row, tissue

    @staticmethod
    def _live_runtime(root: Path, *, budget_target: float = 0.08):
        assets = []
        for asset_id in (
            "mature_probnet_checkpoint",
            "frozen_probnet_spatial_ranker_checkpoint",
            "glas_nucleus_instance_library",
            "panda_nucleus_instance_library",
            "later_he_generator_checkpoint",
        ):
            path = root / f"{asset_id}.bin"
            path.write_bytes((asset_id + "\n").encode())
            record = {
                "asset_id": asset_id,
                "asset_kind": "file",
                "path": str(path),
                "sha256": sha256_file(path),
                "required_for_preflight": True,
            }
            if asset_id == "later_he_generator_checkpoint":
                record.update(
                    {
                        "reader_side_only": True,
                        "used_during_this_stage": False,
                    }
                )
            assets.append(record)
        by_id = {item["asset_id"]: item for item in assets}
        library_set = canonical_metadata_sha256(
            [
                by_id["glas_nucleus_instance_library"]["sha256"],
                by_id["panda_nucleus_instance_library"]["sha256"],
            ]
        )
        runtime_input = {
            "preflight_configuration": {
                "cell_budget_policy_id": (
                    "scene-calibrated-local-population-budget-v1"
                ),
                "joint_area_budget": {
                    "target_fraction": budget_target,
                    "min_fraction": 0.04,
                    "max_fraction": 0.12,
                    "tissue_min_fraction": 0.04,
                    "basis": "whole_patch",
                    "relative_tolerance": 0.02,
                    "fallback_policy": "max_feasible_below_target",
                    "capacity_floor_policy": "lower_to_proven_max_safe",
                    "minimum_effective_fraction": 0.04,
                },
                "maximum_tissue_candidates": 4,
            },
            "assets": assets,
        }
        selection_runtime = {
            "mature_probnet_checkpoint_sha256": by_id[
                "mature_probnet_checkpoint"
            ]["sha256"],
            "frozen_spatial_ranker_sha256": by_id[
                "frozen_probnet_spatial_ranker_checkpoint"
            ]["sha256"],
            "instance_library_sha256": library_set,
            "generator_checkpoint_sha256": by_id[
                "later_he_generator_checkpoint"
            ]["sha256"],
        }
        runtime = _runtime_authority(
            root=ROOT,
            selection_runtime=selection_runtime,
            runtime_input=runtime_input,
            runtime_input_sha256="d" * 64,
            code_commit=CODE_COMMIT,
        )
        if not runtime["all_required_runtime_assets_verified"]:
            raise AssertionError(runtime)
        return runtime

    def test_builds_complete_fail_closed_24_by_5_ledgers(self):
        artifacts = self._build()
        validate_artifacts(artifacts)
        summary = json.loads(artifacts[OUTPUT_FILENAMES["summary"]])
        authority = [
            json.loads(line)
            for line in artifacts[OUTPUT_FILENAMES["authority"]].splitlines()
        ]
        preflight = [
            json.loads(line)
            for line in artifacts[OUTPUT_FILENAMES["preflight"]].splitlines()
        ]
        auxiliary = json.loads(artifacts[OUTPUT_FILENAMES["auxiliary"]])
        self.assertEqual(summary["frozen_binding_count"], 120)
        self.assertEqual(summary["evaluation_count"], 24)
        self.assertEqual(summary["status_counts"], {"eligible": 0, "reject": 120, "abstain": 0})
        self.assertEqual(len(authority), 120)
        self.assertEqual(len(preflight), 120)
        self.assertEqual(len(auxiliary["entries"]), 20)
        self.assertTrue(all(item["fixed_case_no_replacement"] for item in authority))
        self.assertTrue(
            all(
                item["terminal_reason_code"] == "frozen_source_authority_failed"
                for item in authority
            )
        )
        self.assertTrue(
            all(
                item["candidate_portfolio"]["status"] == "not_compiled"
                and item["candidate_portfolio"]["survivor_count"] == 0
                and not item["planner_called"]
                for item in preflight
            )
        )
        counts = summary["before_after_counts"]
        self.assertEqual(
            counts["bindings_with_missing_source_digest"],
            {"before": 40, "after": 40},
        )
        self.assertEqual(
            counts["source_digest_fields_missing"],
            {"before": 80, "after": 80},
        )
        self.assertEqual(
            counts["binding_external_auxiliary_missing"],
            {"before": 15, "after": 15},
        )
        self.assertEqual(
            counts["binding_local_clearance_roi_missing"],
            {"before": 5, "after": 5},
        )
        self.assertEqual(
            counts["selection_runtime_digest_fields_missing"],
            {"before": 3, "after": 3},
        )
        self.assertEqual(
            counts["bindings_missing_profile_provenance"],
            {"before": 120, "after": 24},
        )
        self.assertEqual(
            counts["profile_owned_auxiliary_outputs_materialized"],
            {"before": 0, "after": 0},
        )
        self.assertEqual(
            len(artifacts[OUTPUT_FILENAMES["status_table"]].splitlines()),
            121,
        )
        for field in (
            "planner_called",
            "executor_called",
            "visualization_run",
            "api_used",
            "generated_he_run",
            "frozen_cases_changed",
        ):
            self.assertFalse(summary[field])

    def test_generated_authority_is_workspace_location_independent(self):
        artifacts = self._build()
        root_bytes = str(ROOT).encode("utf-8")
        self.assertTrue(all(root_bytes not in payload for payload in artifacts.values()))
        auxiliary = json.loads(artifacts[OUTPUT_FILENAMES["auxiliary"]])
        self.assertTrue(
            all(
                not Path(item["output_path"]).is_absolute()
                for item in auxiliary["entries"]
            )
        )

    def test_live_glas_reaches_compiler_issued_cell_portfolio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_row, _ = self._write_live_assets(
                root, profile_id="glas-gland-v1"
            )
            source_authority = _source_authority(source_row)
            self.assertTrue(
                source_authority["required_source_authority_verified"]
            )
            runtime = self._live_runtime(root)
            evaluation = {
                "annotation_profile_id": "glas-gland-v1",
                "pathology_domain_id": "colorectal-adenocarcinoma-v1",
                "primitive_id": "neoplastic-cell-abundance-increase-v1",
                "mechanism_id": "colorectal-local-population-modulation",
                "instruction": "Increase neoplastic cells in the selected region.",
            }
            case = _build_live_case(
                evaluation=evaluation,
                selected_case={"case_id": source_row["case_id"], "seed": 17},
                source_row=source_row,
                source_authority=source_authority,
                profile_provenance={
                    "bound_fields": {
                        "preprocessing_revision": "synthetic-glas-v1",
                        "original_instance_mask_digest": source_row[
                            "source_nuclei_mask_sha256"
                        ],
                        "patch_grade": "unknown_not_recorded",
                    },
                    "missing_fields": [],
                },
                profile_auxiliary_records=[],
                profile_auxiliary_paths={},
                external_auxiliary_records=[],
                external_auxiliary_paths={},
                runtime_configuration=runtime["preflight_configuration"],
                joint_repository=JointSkillRepository(),
            )
            _, bundle, portfolio = _compile_live_preflight(
                case=case,
                mechanism_id=evaluation["mechanism_id"],
                source_authority=source_authority,
                runtime=runtime,
                mask_repository=MaskSkillRepository(),
                joint_repository=JointSkillRepository(),
            )
            self.assertEqual(bundle["status"], "composed_and_live_bound")
            self.assertEqual(portfolio["portfolio_kind"], "cell")
            self.assertGreater(portfolio["survivor_count"], 0)
            self.assertFalse(portfolio["pixels_persisted"])
            self.assertFalse(portfolio["external_planner_called"])
            runtime_asset = Path(
                runtime["external_assets"][0]["canonical_path"]
            )
            runtime_asset.write_bytes(b"mutated-after-authority-seal\n")
            with self.assertRaisesRegex(
                ValueError, "runtime asset replay mismatch"
            ):
                _compile_live_preflight(
                    case=case,
                    mechanism_id=evaluation["mechanism_id"],
                    source_authority=source_authority,
                    runtime=runtime,
                    mask_repository=MaskSkillRepository(),
                    joint_repository=JointSkillRepository(),
                )

    def test_live_panda_materializes_aux_and_reaches_tissue_portfolio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_row, tissue = self._write_live_assets(
                root, profile_id="panda-gleason-v1"
            )
            source_authority = _source_authority(source_row)
            profile_fields = {
                "preprocessing_revision": "synthetic-panda-v1",
                "original_label_map_digest": source_row[
                    "source_tissue_mask_sha256"
                ],
                "provider": "synthetic-fixture",
            }
            auxiliary_case = JointCaseContext(
                case_id=source_row["case_id"],
                instruction="Increase tumor burden.",
                source_image_uri=source_row["source_image"],
                source_tissue_mask_uri=source_row["source_tissue_mask"],
                source_nuclei_mask_uri=source_row["source_nuclei_mask"],
                pathology_domain_id="prostate-adenocarcinoma-v1",
                annotation_profile_id="panda-gleason-v1",
                cell_observation_profile_id="cellvit-five-class-v1",
                cell_population_profile_id="prostate-cellvit-source-first-v1",
                primitive_id="tumor-burden-increase-v1",
                joint_area_budget=None,
                seed=17,
                provenance={
                    "source_image_sha256": source_row["source_image_sha256"],
                    "source_tissue_mask_sha256": source_row[
                        "source_tissue_mask_sha256"
                    ],
                    "source_nuclei_mask_sha256": source_row[
                        "source_nuclei_mask_sha256"
                    ],
                    **profile_fields,
                },
            )
            _, produced = materialize_profile_auxiliaries(
                auxiliary_case,
                source_tissue=tissue,
                output_dir=root / "profile-auxiliary",
            )
            produced_by_id = {item.structure_id: item for item in produced}
            native = produced_by_id["native_pattern_and_lumen_map"]
            native_array = load_id_mask(native.path)
            profile_record = {
                "structure_id": native.structure_id,
                "status": "materialized",
                "output_file_sha256": native.sha256,
                "output_array_sha256": array_sha256(native_array),
                "producer_provenance": native.provenance,
            }
            runtime = self._live_runtime(root)
            evaluation = {
                "annotation_profile_id": "panda-gleason-v1",
                "pathology_domain_id": "prostate-adenocarcinoma-v1",
                "primitive_id": "tumor-burden-increase-v1",
                "mechanism_id": "prostate-pattern-4-growth",
                "instruction": "Increase tumor burden.",
            }
            case = _build_live_case(
                evaluation=evaluation,
                selected_case={"case_id": source_row["case_id"], "seed": 17},
                source_row=source_row,
                source_authority=source_authority,
                profile_provenance={
                    "bound_fields": profile_fields,
                    "missing_fields": [],
                },
                profile_auxiliary_records=[profile_record],
                profile_auxiliary_paths={native.structure_id: native.path},
                external_auxiliary_records=[],
                external_auxiliary_paths={},
                runtime_configuration=runtime["preflight_configuration"],
                joint_repository=JointSkillRepository(),
            )
            _, bundle, portfolio = _compile_live_preflight(
                case=case,
                mechanism_id=evaluation["mechanism_id"],
                source_authority=source_authority,
                runtime=runtime,
                mask_repository=MaskSkillRepository(),
                joint_repository=JointSkillRepository(),
            )
            self.assertEqual(bundle["status"], "composed_and_live_bound")
            self.assertEqual(portfolio["portfolio_kind"], "tissue")
            self.assertGreater(portfolio["survivor_count"], 0)

    def test_external_roi_requires_typed_live_digest_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_row, tissue = self._write_live_assets(
                root, profile_id="panda-gleason-v1"
            )
            source_authority = _source_authority(source_row)
            binding_id = "evaluation::" + source_row["case_id"]
            records, paths, missing = _external_auxiliary_authority(
                binding_id=binding_id,
                required_structure_ids=("local_clearance_roi",),
                erratum_entries=(),
                source_tissue_authority=source_authority[
                    "source_tissue_mask"
                ],
            )
            self.assertEqual(missing, ["local_clearance_roi"])
            self.assertFalse(paths)
            self.assertIn(
                "typed_external_authority_missing",
                records[0]["failure_codes"],
            )
            roi = np.zeros_like(tissue, dtype=np.uint8)
            roi[40:160, 90:210] = 255
            roi_path = root / "roi.png"
            Image.fromarray(roi).save(roi_path)
            roi_array = load_id_mask(roi_path)
            array_digest = array_sha256(roi_array)
            file_digest = sha256_file(roi_path)
            entry = {
                "binding_id": binding_id,
                "structure_id": "local_clearance_roi",
                "path": str(roi_path),
                "file_sha256": file_digest,
                "decoded_array_sha256": array_digest,
                "provenance": {
                    "producer_id": "synthetic-user-roi",
                    "producer_version": "v1",
                    "authority_type": "digest_bound_user_local_roi",
                    "source_tissue_mask_sha256": source_row[
                        "source_tissue_mask_sha256"
                    ],
                    "output_sha256": file_digest,
                    "decoded_array_sha256": array_digest,
                },
            }
            records, paths, missing = _external_auxiliary_authority(
                binding_id=binding_id,
                required_structure_ids=("local_clearance_roi",),
                erratum_entries=(entry,),
                source_tissue_authority=source_authority[
                    "source_tissue_mask"
                ],
            )
            self.assertFalse(missing)
            self.assertEqual(
                paths["local_clearance_roi"], str(roi_path.resolve())
            )
            self.assertTrue(records[0]["authority_verified"])

    def test_source_erratum_cannot_contradict_frozen_case_record(self):
        row = json.loads(SOURCE.read_text(encoding="utf-8"))["cases"][0]
        with self.assertRaisesRegex(ValueError, "contradicts frozen digest"):
            _effective_source_row(
                row,
                erratum_entry={
                    "case_id": row["case_id"],
                    "source_case_record_sha256": row["case_record_sha256"],
                    "profile_provenance": {},
                    "source_asset_authority": {
                        "source_tissue_mask": {
                            "path": row["source_tissue_mask"],
                            "sha256": "f" * 64,
                        }
                    },
                },
            )

    def test_profile_provenance_is_bound_to_source_mask_digest(self):
        row = json.loads(SOURCE.read_text(encoding="utf-8"))["cases"][5]
        binding = {
            "entry_sha256": "e" * 64,
            "profile_provenance": {
                "provider": "PANDA",
                "preprocessing_revision": "g2-v2-frozen-source-assets-v1",
                "original_label_map_digest": "f" * 64,
            },
        }
        authority = _profile_required_provenance(
            repository=MaskSkillRepository(),
            profile_id="panda-gleason-v1",
            erratum_binding=binding,
            source_row=row,
        )
        self.assertFalse(authority["authority_verified"])
        self.assertEqual(
            authority["invalid_fields"]["original_label_map_digest"],
            "must_equal_frozen_source_tissue_mask_digest",
        )

    def test_runtime_authority_rejects_incomplete_asset_catalog(self):
        with self.assertRaisesRegex(ValueError, "exact asset catalog"):
            _runtime_authority(
                root=ROOT,
                selection_runtime={},
                runtime_input={
                    "preflight_configuration": {
                        "cell_budget_policy_id": (
                            "scene-calibrated-local-population-budget-v1"
                        ),
                        "joint_area_budget": {
                            "target_fraction": 0.08,
                            "min_fraction": 0.04,
                            "max_fraction": 0.12,
                            "tissue_min_fraction": 0.04,
                            "basis": "whole_patch",
                            "relative_tolerance": 0.02,
                            "fallback_policy": "max_feasible_below_target",
                            "capacity_floor_policy": "lower_to_proven_max_safe",
                            "minimum_effective_fraction": 0.04,
                        },
                        "maximum_tissue_candidates": 4,
                    },
                    "assets": [
                        {
                            "asset_id": "mature_probnet_checkpoint",
                            "path": None,
                            "sha256": None,
                            "required_for_preflight": True,
                        }
                    ],
                },
                runtime_input_sha256="d" * 64,
                code_commit=CODE_COMMIT,
            )

    def test_changed_or_unlocked_frozen_binding_is_rejected(self):
        selection = json.loads(SELECTION.read_text(encoding="utf-8"))
        mutations = (
            ("fixed_case_no_replacement", False),
            ("execution_allowed", True),
        )
        for field, value in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                mutated = json.loads(json.dumps(selection))
                mutated["evaluations"][0]["selected_cases"][0][field] = value
                path = Path(directory) / "selection.json"
                path.write_text(json.dumps(mutated), encoding="utf-8")
                with self.assertRaises(ValueError):
                    build_artifacts(
                        root=ROOT,
                        selection_path=path,
                        source_manifest_path=SOURCE,
                        code_commit=CODE_COMMIT,
                    )

    def test_resealed_eligible_record_without_portfolio_is_rejected(self):
        artifacts = dict(self._build())
        records = [
            json.loads(line)
            for line in artifacts[OUTPUT_FILENAMES["preflight"]].splitlines()
        ]
        records[0]["eligible_for_later_visualization"] = True
        records[0] = sealed_record(records[0])
        payload = b"".join(canonical_json_bytes(item) + b"\n" for item in records)
        artifacts[OUTPUT_FILENAMES["preflight"]] = payload
        summary = json.loads(artifacts[OUTPUT_FILENAMES["summary"]])
        summary["candidate_preflight_records_sha256"] = hashlib.sha256(payload).hexdigest()
        summary = self._reseal_manifest(summary)
        artifacts[OUTPUT_FILENAMES["summary"]] = canonical_json_bytes(summary, indent=2)
        with self.assertRaisesRegex(
            ValueError,
            "authority and preflight eligibility differ|illegally enables execution",
        ):
            validate_artifacts(artifacts)

    def test_external_auxiliary_cannot_be_self_materialized(self):
        artifacts = dict(self._build())
        auxiliary = json.loads(artifacts[OUTPUT_FILENAMES["auxiliary"]])
        auxiliary["entries"][0]["structure_id"] = "native_gland_instance_map"
        auxiliary = self._reseal_manifest(auxiliary)
        payload = canonical_json_bytes(auxiliary, indent=2)
        artifacts[OUTPUT_FILENAMES["auxiliary"]] = payload
        summary = json.loads(artifacts[OUTPUT_FILENAMES["summary"]])
        summary["auxiliary_materialization_manifest_sha256"] = hashlib.sha256(
            payload
        ).hexdigest()
        summary = self._reseal_manifest(summary)
        artifacts[OUTPUT_FILENAMES["summary"]] = canonical_json_bytes(summary, indent=2)
        with self.assertRaisesRegex(ValueError, "external-only auxiliary"):
            validate_artifacts(artifacts)


if __name__ == "__main__":
    unittest.main()

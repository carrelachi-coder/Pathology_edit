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
    _materialize_glas_gland_instance_authorities,
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
        gland_instance_path = root / "gland_instances.png"
        profile_metadata_path = root / "profile_metadata.json"
        Image.fromarray(image).save(image_path)
        Image.fromarray(tissue).save(tissue_path)
        Image.fromarray(nuclei).save(nuclei_path)
        gland_instances = np.zeros_like(tissue, dtype=np.uint16)
        gland_instances[tumor] = 1
        Image.fromarray(gland_instances).save(gland_instance_path)
        tissue_sha = sha256_file(tissue_path)
        gland_instance_sha = sha256_file(gland_instance_path)
        profile_metadata = {
            "preprocessing_revision": "synthetic-profile-authority-v1",
            **(
                {
                    "original_instance_mask_digest": gland_instance_sha,
                    "patch_grade": "moderately_differentiated",
                }
                if profile_id == "glas-gland-v1"
                else {
                    "original_label_map_digest": tissue_sha,
                    "provider": "PANDA",
                }
            ),
        }
        profile_metadata_path.write_text(
            json.dumps(profile_metadata, sort_keys=True), encoding="utf-8"
        )
        row = {
            "case_id": "synthetic-live-" + profile_id,
            "case_record_sha256": "c" * 64,
            "organic_seed": 17,
            "source_image": str(image_path),
            "source_image_sha256": sha256_file(image_path),
            "source_tissue_mask": str(tissue_path),
            "source_tissue_mask_sha256": tissue_sha,
            "source_nuclei_mask": str(nuclei_path),
            "source_nuclei_mask_sha256": sha256_file(nuclei_path),
            "source_gland_instance_mask": str(gland_instance_path),
            "source_gland_instance_mask_sha256": gland_instance_sha,
            "source_profile_metadata": str(profile_metadata_path),
            "source_profile_metadata_sha256": sha256_file(
                profile_metadata_path
            ),
        }
        return row, tissue

    @staticmethod
    def _bind_glas_metadata_to_instance_asset(
        row: dict[str, object],
        *,
        instance_digest: str,
    ) -> None:
        metadata_path = Path(str(row["source_profile_metadata"]))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["original_instance_mask_digest"] = instance_digest
        metadata_path.write_text(
            json.dumps(metadata, sort_keys=True), encoding="utf-8"
        )
        row["source_profile_metadata_sha256"] = sha256_file(metadata_path)

    @staticmethod
    def _write_runtime_inputs(root: Path, *, budget_target: float = 0.08):
        root.mkdir(parents=True, exist_ok=True)
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
        return selection_runtime, runtime_input

    @classmethod
    def _live_runtime(cls, root: Path, *, budget_target: float = 0.08):
        selection_runtime, runtime_input = cls._write_runtime_inputs(
            root, budget_target=budget_target
        )
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
        self.assertEqual(len(auxiliary["entries"]), 25)
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
            {"before": 5, "after": 5},
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
            {"before": 120, "after": 120},
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
            profile_fields = {
                "preprocessing_revision": "synthetic-profile-authority-v1",
                "original_instance_mask_digest": source_row[
                    "source_gland_instance_mask_sha256"
                ],
                "patch_grade": "moderately_differentiated",
            }
            profile_authority = _profile_required_provenance(
                repository=MaskSkillRepository(),
                profile_id="glas-gland-v1",
                erratum_binding={
                    "entry_sha256": "e" * 64,
                    "profile_provenance": profile_fields,
                },
                source_row=source_row,
                source_authority=source_authority,
            )
            self.assertTrue(profile_authority["authority_verified"])
            case = _build_live_case(
                evaluation=evaluation,
                selected_case={"case_id": source_row["case_id"], "seed": 17},
                source_row=source_row,
                source_authority=source_authority,
                profile_provenance=profile_authority,
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

            metadata_path = Path(str(source_row["source_profile_metadata"]))
            original_metadata = metadata_path.read_bytes()
            metadata_path.write_text(
                json.dumps(
                    {
                        "preprocessing_revision": "tampered-after-stage-01",
                        "original_instance_mask_digest": source_row[
                            "source_gland_instance_mask_sha256"
                        ],
                        "patch_grade": "moderately_differentiated",
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "source authority replay mismatch"
            ):
                _compile_live_preflight(
                    case=case,
                    mechanism_id=evaluation["mechanism_id"],
                    source_authority=source_authority,
                    runtime=runtime,
                    mask_repository=MaskSkillRepository(),
                    joint_repository=JointSkillRepository(),
                )
            metadata_path.write_bytes(original_metadata)

            gland_path = Path(str(source_row["source_gland_instance_mask"]))
            original_gland = gland_path.read_bytes()
            Image.fromarray(np.zeros_like(load_id_mask(gland_path))).save(
                gland_path
            )
            with self.assertRaisesRegex(
                ValueError, "source authority replay mismatch"
            ):
                _compile_live_preflight(
                    case=case,
                    mechanism_id=evaluation["mechanism_id"],
                    source_authority=source_authority,
                    runtime=runtime,
                    mask_repository=MaskSkillRepository(),
                    joint_repository=JointSkillRepository(),
                )
            gland_path.write_bytes(original_gland)

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

    def test_glas_profile_owned_instance_map_closes_derived_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_row, tissue = self._write_live_assets(
                root, profile_id="glas-gland-v1"
            )
            source_row.pop("source_gland_instance_mask")
            source_row.pop("source_gland_instance_mask_sha256")
            metadata_path = Path(str(source_row["source_profile_metadata"]))
            metadata_path.write_text(
                json.dumps(
                    {
                        "preprocessing_revision": (
                            "synthetic-profile-authority-v1"
                        )
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            source_row["source_profile_metadata_sha256"] = sha256_file(
                metadata_path
            )
            initial = _source_authority(source_row)
            rows, authorities = _materialize_glas_gland_instance_authorities(
                glas_case_ids=(source_row["case_id"],),
                source_rows={source_row["case_id"]: source_row},
                source_authority_by_case={source_row["case_id"]: initial},
                output_dir=root / "profile-auxiliary",
            )
            effective_row = rows[source_row["case_id"]]
            authority = authorities[source_row["case_id"]]
            validation = authority[
                "glas_gland_instance_annotation_validation"
            ]
            self.assertTrue(validation["authority_verified"])
            self.assertTrue(
                validation["deterministic_connectivity_replay_verified"]
            )
            self.assertEqual(
                validation["replayed_patch_grade"],
                "moderately_differentiated",
            )
            with Image.open(
                effective_row["source_gland_instance_mask"]
            ) as instance_image:
                self.assertIn(instance_image.mode, {"I", "I;16"})
            instance_map = load_id_mask(
                effective_row["source_gland_instance_mask"]
            )
            self.assertEqual(set(np.unique(instance_map)), {0, 1})
            np.testing.assert_array_equal(instance_map > 0, tissue == 12)

            profile_authority = _profile_required_provenance(
                repository=MaskSkillRepository(),
                profile_id="glas-gland-v1",
                erratum_binding={
                    "entry_sha256": "e" * 64,
                    "profile_provenance": {
                        "preprocessing_revision": (
                            "synthetic-profile-authority-v1"
                        )
                    },
                },
                source_row=effective_row,
                source_authority=authority,
            )
            self.assertTrue(profile_authority["authority_verified"])
            self.assertEqual(
                set(profile_authority["deterministically_derived_fields"]),
                {"original_instance_mask_digest", "patch_grade"},
            )

    def test_live_panda_materializes_aux_and_reaches_tissue_portfolio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_row, tissue = self._write_live_assets(
                root, profile_id="panda-gleason-v1"
            )
            source_authority = _source_authority(source_row)
            profile_fields = {
                "preprocessing_revision": "synthetic-profile-authority-v1",
                "original_label_map_digest": source_row[
                    "source_tissue_mask_sha256"
                ],
                "provider": "PANDA",
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
            profile_authority = _profile_required_provenance(
                repository=MaskSkillRepository(),
                profile_id="panda-gleason-v1",
                erratum_binding={
                    "entry_sha256": "e" * 64,
                    "profile_provenance": profile_fields,
                },
                source_row=source_row,
                source_authority=source_authority,
            )
            self.assertTrue(profile_authority["authority_verified"])
            case = _build_live_case(
                evaluation=evaluation,
                selected_case={"case_id": source_row["case_id"], "seed": 17},
                source_row=source_row,
                source_authority=source_authority,
                profile_provenance=profile_authority,
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

    def test_full_build_reaches_real_glas_and_panda_compilers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = {}
            for profile_id in ("glas-gland-v1", "panda-gleason-v1"):
                profile_root = root / profile_id
                profile_root.mkdir()
                assets[profile_id] = self._write_live_assets(
                    profile_root, profile_id=profile_id
                )[0]

            source_rows = []
            rows_by_profile = {}
            for profile_id, organ, dataset, prefix in (
                ("glas-gland-v1", "colorectal", "GLAS", "glas"),
                ("panda-gleason-v1", "prostate", "PANDA", "panda"),
            ):
                profile_rows = []
                for index in range(5):
                    base = assets[profile_id]
                    row = {
                        "auxiliary_structure_uris": {},
                        "case_id": f"synthetic-{prefix}-{index + 1}",
                        "dataset": dataset,
                        "organ": organ,
                        "organic_seed": 17 + index,
                        "sample_id": f"synthetic-{prefix}-{index + 1}",
                        **{
                            field: base[field]
                            for field in (
                                "source_image",
                                "source_image_sha256",
                                "source_tissue_mask",
                                "source_tissue_mask_sha256",
                                "source_nuclei_mask",
                                "source_nuclei_mask_sha256",
                                "source_gland_instance_mask",
                                "source_gland_instance_mask_sha256",
                                "source_profile_metadata",
                                "source_profile_metadata_sha256",
                            )
                        },
                    }
                    row["case_record_sha256"] = canonical_metadata_sha256(row)
                    profile_rows.append(row)
                    source_rows.append(row)
                rows_by_profile[profile_id] = profile_rows

            source_payload = {
                "schema_version": "p1-glas-panda-source-case-pool-v1",
                "cases": source_rows,
            }
            source_path = root / "source.json"
            source_path.write_text(
                json.dumps(source_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            selection = json.loads(SELECTION.read_text(encoding="utf-8"))
            selection_runtime, runtime_input = self._write_runtime_inputs(
                root / "runtime"
            )
            selection["runtime_authority"].update(selection_runtime)
            selection["runtime_authority"]["all_required_digests_bound"] = True
            selection["source_manifest"] = "authority-input://source.json"
            selection["source_manifest_sha256"] = sha256_file(source_path)
            for evaluation in selection["evaluations"]:
                profile_id = evaluation["annotation_profile_id"]
                rewritten = []
                for template, source_row in zip(
                    evaluation["selected_cases"], rows_by_profile[profile_id]
                ):
                    item = dict(template)
                    item.update(
                        {
                            "case_id": source_row["case_id"],
                            "source_image": source_row["source_image"],
                            "source_image_sha256": source_row[
                                "source_image_sha256"
                            ],
                            "source_tissue_mask": source_row[
                                "source_tissue_mask"
                            ],
                            "source_tissue_mask_sha256": source_row[
                                "source_tissue_mask_sha256"
                            ],
                            "source_nuclei_mask": source_row[
                                "source_nuclei_mask"
                            ],
                            "source_nuclei_mask_sha256": source_row[
                                "source_nuclei_mask_sha256"
                            ],
                            "source_case_record_sha256": source_row[
                                "case_record_sha256"
                            ],
                            "seed": source_row["organic_seed"],
                            "available_auxiliary_structures": [],
                            "missing_source_asset_digests": [],
                            "execution_allowed": False,
                            "fixed_case_no_replacement": True,
                        }
                    )
                    rewritten.append(item)
                evaluation["selected_cases"] = rewritten
            selection_path = root / "selection.json"
            selection_path.write_text(
                json.dumps(selection, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            selection_sha = sha256_file(selection_path)

            runtime_input.update(
                {
                    "schema_version": "p1-glas-panda-runtime-authority-v1",
                    "production_status": "shadow_only",
                    "selection_manifest_sha256": selection_sha,
                }
            )
            runtime_path = root / "runtime.json"
            runtime_path.write_text(
                json.dumps(runtime_input, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            erratum_entries = []
            for profile_id, profile_rows in rows_by_profile.items():
                for index, source_row in enumerate(profile_rows):
                    metadata = json.loads(
                        Path(source_row["source_profile_metadata"]).read_text(
                            encoding="utf-8"
                        )
                    )
                    erratum_entries.append(
                        {
                            "case_id": source_row["case_id"],
                            "source_case_record_sha256": source_row[
                                "case_record_sha256"
                            ],
                            "profile_provenance": metadata if index == 0 else {},
                            "source_asset_authority": {},
                        }
                    )

            roi = np.zeros((256, 256), dtype=np.uint8)
            roi[35:220, 35:220] = 255
            roi_path = root / "local_clearance_roi.png"
            Image.fromarray(roi).save(roi_path)
            roi_file_sha = sha256_file(roi_path)
            roi_array_sha = array_sha256(load_id_mask(roi_path))
            external_entries = []
            first_by_profile = {
                profile_id: rows[0]
                for profile_id, rows in rows_by_profile.items()
            }
            for evaluation in selection["evaluations"]:
                external_ids = set(
                    evaluation.get("required_auxiliary_structures") or ()
                ) & {"native_gland_instance_map", "local_clearance_roi"}
                if not external_ids:
                    continue
                source_row = first_by_profile[
                    evaluation["annotation_profile_id"]
                ]
                for structure_id in sorted(external_ids):
                    if structure_id == "native_gland_instance_map":
                        path = Path(source_row["source_gland_instance_mask"])
                        file_sha = sha256_file(path)
                        decoded_sha = array_sha256(load_id_mask(path))
                        authority_type = "native_gland_instance_annotation"
                    else:
                        path = roi_path
                        file_sha = roi_file_sha
                        decoded_sha = roi_array_sha
                        authority_type = "digest_bound_user_local_roi"
                    external_entries.append(
                        {
                            "binding_id": (
                                evaluation["evaluation_id"]
                                + "::"
                                + source_row["case_id"]
                            ),
                            "structure_id": structure_id,
                            "path": str(path),
                            "file_sha256": file_sha,
                            "decoded_array_sha256": decoded_sha,
                            "provenance": {
                                "producer_id": "synthetic-typed-authority",
                                "producer_version": "v1",
                                "authority_type": authority_type,
                                "source_tissue_mask_sha256": source_row[
                                    "source_tissue_mask_sha256"
                                ],
                                "output_sha256": file_sha,
                                "decoded_array_sha256": decoded_sha,
                            },
                        }
                    )
            erratum = {
                "schema_version": "p1-glas-panda-authority-erratum-v1",
                "production_status": "shadow_only",
                "selection_manifest_sha256": selection_sha,
                "source_manifest_sha256": sha256_file(source_path),
                "source_case_authority": erratum_entries,
                "external_auxiliary_authority": external_entries,
            }
            erratum_path = root / "erratum.json"
            erratum_path.write_text(
                json.dumps(erratum, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            artifacts = build_artifacts(
                root=ROOT,
                selection_path=selection_path,
                source_manifest_path=source_path,
                code_commit=CODE_COMMIT,
                authority_erratum_path=erratum_path,
                runtime_authority_path=runtime_path,
                auxiliary_output_dir=root / "profile-auxiliary",
            )
            authority = [
                json.loads(line)
                for line in artifacts[OUTPUT_FILENAMES["authority"]].splitlines()
            ]
            preflight = [
                json.loads(line)
                for line in artifacts[OUTPUT_FILENAMES["preflight"]].splitlines()
            ]
            by_id = {item["binding_id"]: item for item in preflight}
            glas_live = [
                item
                for item in authority
                if item["annotation_profile_id"] == "glas-gland-v1"
                and item["case_id"] == "synthetic-glas-1"
                and by_id[item["binding_id"]]["candidate_portfolio"]["status"]
                == "compiled"
            ]
            panda_live = [
                item
                for item in authority
                if item["annotation_profile_id"] == "panda-gleason-v1"
                and item["case_id"] == "synthetic-panda-1"
                and by_id[item["binding_id"]]["candidate_portfolio"]["status"]
                == "compiled"
            ]
            self.assertTrue(glas_live)
            self.assertTrue(panda_live)
            self.assertTrue(
                all(
                    item["required_profile_provenance"]["authority_verified"]
                    for item in (*glas_live, *panda_live)
                )
            )

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
        with tempfile.TemporaryDirectory() as directory:
            row, _ = self._write_live_assets(
                Path(directory), profile_id="panda-gleason-v1"
            )
            binding = {
                "entry_sha256": "e" * 64,
                "profile_provenance": {
                    "provider": "PANDA",
                    "preprocessing_revision": (
                        "synthetic-profile-authority-v1"
                    ),
                    "original_label_map_digest": "f" * 64,
                },
            }
            authority = _profile_required_provenance(
                repository=MaskSkillRepository(),
                profile_id="panda-gleason-v1",
                erratum_binding=binding,
                source_row=row,
                source_authority=_source_authority(row),
            )
            self.assertFalse(authority["authority_verified"])
            self.assertEqual(
                authority["invalid_fields"]["original_label_map_digest"],
                "must_bind_live_source_tissue_label_map",
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

    def test_runtime_authority_allows_explicit_optional_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            selection_runtime, runtime_input = self._write_runtime_inputs(
                Path(directory)
            )
            optional = {
                "frozen_probnet_spatial_ranker_checkpoint",
                "later_he_generator_checkpoint",
            }
            for asset in runtime_input["assets"]:
                if asset["asset_id"] in optional:
                    asset["path"] = None
                    asset["sha256"] = None
                    asset["required_for_preflight"] = False
            selection_runtime["frozen_spatial_ranker_sha256"] = None
            selection_runtime["generator_checkpoint_sha256"] = None
            runtime = _runtime_authority(
                root=ROOT,
                selection_runtime=selection_runtime,
                runtime_input=runtime_input,
                runtime_input_sha256="d" * 64,
                code_commit=CODE_COMMIT,
            )
            self.assertTrue(runtime["all_required_runtime_assets_verified"])
            self.assertEqual(
                runtime["required_runtime_digest_fields"],
                [
                    "instance_library_sha256",
                    "mature_probnet_checkpoint_sha256",
                ],
            )
            self.assertEqual(runtime["unverified_external_asset_ids"], [])

    def test_glas_nuclei_digest_and_unknown_grade_cannot_fill_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            row, _ = self._write_live_assets(
                Path(directory), profile_id="glas-gland-v1"
            )
            authority = _profile_required_provenance(
                repository=MaskSkillRepository(),
                profile_id="glas-gland-v1",
                erratum_binding={
                    "entry_sha256": "e" * 64,
                    "profile_provenance": {
                        "preprocessing_revision": (
                            "synthetic-profile-authority-v1"
                        ),
                        "original_instance_mask_digest": row[
                            "source_nuclei_mask_sha256"
                        ],
                        "patch_grade": "unknown_not_recorded",
                    },
                },
                source_row=row,
                source_authority=_source_authority(row),
            )
            self.assertFalse(authority["authority_verified"])
            self.assertEqual(
                set(authority["invalid_fields"]),
                {"original_instance_mask_digest", "patch_grade"},
            )

    def test_glas_gland_instance_asset_cannot_alias_nuclei_or_tissue(self):
        for source_role in ("source_nuclei_mask", "source_tissue_mask"):
            with self.subTest(source_role=source_role), tempfile.TemporaryDirectory() as directory:
                row, _ = self._write_live_assets(
                    Path(directory), profile_id="glas-gland-v1"
                )
                digest_field = source_role + "_sha256"
                row["source_gland_instance_mask"] = row[source_role]
                row["source_gland_instance_mask_sha256"] = row[digest_field]
                self._bind_glas_metadata_to_instance_asset(
                    row,
                    instance_digest=str(row[digest_field]),
                )
                source_authority = _source_authority(row)
                validation = source_authority[
                    "glas_gland_instance_annotation_validation"
                ]
                self.assertFalse(validation["authority_verified"])
                self.assertIn(source_role, validation["role_aliases"])
                authority = _profile_required_provenance(
                    repository=MaskSkillRepository(),
                    profile_id="glas-gland-v1",
                    erratum_binding={
                        "entry_sha256": "e" * 64,
                        "profile_provenance": {
                            "preprocessing_revision": (
                                "synthetic-profile-authority-v1"
                            ),
                            "original_instance_mask_digest": row[
                                digest_field
                            ],
                            "patch_grade": "moderately_differentiated",
                        },
                    },
                    source_row=row,
                    source_authority=source_authority,
                )
                self.assertFalse(authority["authority_verified"])
                self.assertEqual(
                    authority["invalid_fields"][
                        "original_instance_mask_digest"
                    ],
                    "must_bind_live_native_gland_instance_annotation",
                )

    def test_glas_gland_instance_requires_shape_ids_topology_and_support(self):
        mutations = {
            "shape": (
                lambda tissue: np.ones((64, 64), dtype=np.uint16),
                "gland_instance_shape_mismatch",
            ),
            "positive_ids": (
                lambda tissue: np.zeros_like(tissue, dtype=np.uint16),
                "gland_instance_positive_ids_missing",
            ),
            "topology": (
                lambda tissue: np.pad(
                    np.ones((8, 8), dtype=np.uint16),
                    ((70, tissue.shape[0] - 78), (70, tissue.shape[1] - 78)),
                )
                + np.pad(
                    np.ones((8, 8), dtype=np.uint16),
                    ((170, tissue.shape[0] - 178), (170, tissue.shape[1] - 178)),
                ),
                "gland_instance_id_is_disconnected",
            ),
            "support": (
                lambda tissue: np.ones_like(tissue, dtype=np.uint16),
                "gland_instance_outside_glas_gland_support",
            ),
        }
        for name, (build_mask, expected_failure) in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                row, tissue = self._write_live_assets(
                    Path(directory), profile_id="glas-gland-v1"
                )
                gland_path = Path(str(row["source_gland_instance_mask"]))
                Image.fromarray(build_mask(tissue)).save(gland_path)
                gland_digest = sha256_file(gland_path)
                row["source_gland_instance_mask_sha256"] = gland_digest
                self._bind_glas_metadata_to_instance_asset(
                    row,
                    instance_digest=gland_digest,
                )
                validation = _source_authority(row)[
                    "glas_gland_instance_annotation_validation"
                ]
                self.assertFalse(validation["authority_verified"])
                self.assertIn(expected_failure, validation["failure_codes"])

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
        auxiliary["entries"][0]["structure_id"] = "local_clearance_roi"
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

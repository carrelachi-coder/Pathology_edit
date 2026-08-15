from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from phase3_joint_edit_refine.p1_authority_preflight import (
    OUTPUT_FILENAMES,
    build_artifacts,
    validate_artifacts,
)
from phase3_joint_edit_refine.p1_authority_preflight import (
    _canonical_json_bytes as canonical_json_bytes,
)
from phase3_joint_edit_refine.p1_authority_preflight import (
    _sealed_record as sealed_record,
)
from phase3_joint_edit_refine.portfolio_authority import canonical_metadata_sha256

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
        with self.assertRaisesRegex(ValueError, "illegally enables execution"):
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

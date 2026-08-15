"""Fail-closed tests for explicit candidate approval promotion."""

from __future__ import annotations

import unittest

from phase3_joint_edit_refine.approval_handoff import (
    APPROVAL_SCHEMA_VERSION,
    _canonical_digest,
    _validate_approval,
    _validate_contract_identity,
)
from phase3_joint_edit_refine.models import JointContractError


class ApprovalHandoffTests(unittest.TestCase):
    def test_explicit_approval_is_bound_to_exact_case_and_candidate(self):
        approval = {
            "schema_version": APPROVAL_SCHEMA_VERSION,
            "decision": "approved",
            "case_id": "case-1",
            "candidate_id": "joint-cand:001",
            "approval_scope": "mask_condition_for_online_generation",
            "approved_by": "user",
            "evidence_sha256": "a" * 64,
        }
        self.assertEqual(
            _validate_approval(
                approval,
                case_id="case-1",
                candidate_id="joint-cand:001",
            ),
            approval,
        )
        with self.assertRaisesRegex(JointContractError, "candidate_id"):
            _validate_approval(
                approval,
                case_id="case-1",
                candidate_id="joint-cand:002",
            )

    def test_non_user_or_unbound_evidence_fails_closed(self):
        base = {
            "schema_version": APPROVAL_SCHEMA_VERSION,
            "decision": "approved",
            "case_id": "case-1",
            "candidate_id": "joint-cand:001",
            "approval_scope": "mask_condition_for_online_generation",
            "approved_by": "critic",
            "evidence_sha256": "not-a-digest",
        }
        with self.assertRaisesRegex(JointContractError, "explicit user"):
            _validate_approval(
                base,
                case_id="case-1",
                candidate_id="joint-cand:001",
            )
        base["approved_by"] = "user"
        with self.assertRaisesRegex(JointContractError, "evidence_sha256"):
            _validate_approval(
                base,
                case_id="case-1",
                candidate_id="joint-cand:001",
            )

    def test_contract_metadata_drift_is_rejected(self):
        contract = {"contract_id": "", "case_id": "case-1", "value": 7}
        contract_id = _canonical_digest(contract)
        contract["contract_id"] = contract_id
        _validate_contract_identity(contract, expected_contract_id=contract_id)
        contract["value"] = 8
        with self.assertRaisesRegex(JointContractError, "digest drift"):
            _validate_contract_identity(
                contract,
                expected_contract_id=contract_id,
            )


if __name__ == "__main__":
    unittest.main()

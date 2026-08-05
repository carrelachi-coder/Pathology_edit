"""Blind-evaluation release threshold tests."""

from __future__ import annotations

import unittest

from phase3_mask_edit_refine.evaluation import EvaluationRecord, score_evaluation


def _record(case_id: str, *, accepted: bool = True) -> EvaluationRecord:
    return EvaluationRecord(
        case_id=case_id,
        model_config="terra-medium",
        pathology_domain_id="colorectal-adenocarcinoma-v1",
        annotation_profile_id="glas-gland-v1",
        primitive_id="tumor-burden-increase-v1",
        schema_valid=True,
        predicted_interface_ids=("legal", "other"),
        legal_interface_ids=("legal",),
        hard_violation_count=0,
        changed_area_passed=True,
        unrequested_label_violation_pixels=0,
        expert_morphology_accepted=accepted,
        abstained=False,
        cost_usd=0.10,
        latency_sec=2.0,
    )


class EvaluationReleaseGateTests(unittest.TestCase):
    def test_perfect_records_pass_release_gate(self):
        report = score_evaluation([_record(str(index)) for index in range(20)])
        self.assertTrue(report["models"]["terra-medium"]["release_passed"])

    def test_expert_morphology_failure_blocks_release(self):
        records = [_record(str(index), accepted=index not in {0, 1}) for index in range(20)]
        report = score_evaluation(records)
        checks = report["models"]["terra-medium"]["release_checks"]
        self.assertFalse(checks["expert_accept_rate_ge_95pct"])
        self.assertFalse(report["models"]["terra-medium"]["release_passed"])


if __name__ == "__main__":
    unittest.main()

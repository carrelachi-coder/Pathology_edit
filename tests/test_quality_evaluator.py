from __future__ import annotations

import unittest

from phase3_mask_edit.audit.quality import (
    QualityPolicy,
    build_generation_report,
    evaluate_product_quality,
)
from scripts.replay_product_quality_evaluator import select_replayed_attempt


def _coarse_metrics() -> dict:
    return {
        "changed_region": {
            "accuracy": 0.90,
            "macro_miou": 0.80,
            "no_edit_accuracy": 0.20,
            "soft_target_source_margin": 0.50,
            "soft_no_edit_target_source_margin": -0.20,
        },
        "source_evaluator_calibration": {
            "accuracy": 0.85,
            "macro_miou": 0.70,
        },
        "transition_evaluator_calibration": {
            "transition_pixels": 1200,
            "source_class_recall_min": 0.90,
            "target_reference_available": True,
            "target_reference_recall_min": 0.80,
            "source_to_target_confusion_rate": 0.10,
        },
        "preservation": {
            "prediction_relative_drift_U_far": 0.02,
            "appearance_calibrated_prediction_drift_U_far": 0.02,
            "appearance_calibration_applicable": True,
            "appearance_calibration_coverage_U_far": 0.80,
            "global_appearance_probability_shift_l1": 0.03,
            "inner_ring_target_error": 0.10,
            "outer_ring_spillover": 0.02,
            "appearance_calibrated_outer_ring_spillover": 0.02,
        },
        "boundary": {"class_aware_f1_4": 0.80},
        "region_pixels": {"U_far": 1000, "B": 500},
    }


def _source_nuclei() -> dict:
    return {
        "changed_region": {
            "reference": {101: 10, 102: 10},
            "predicted": {101: 9, 102: 10},
        },
        "full_image": {
            "reference": {101: 20, 102: 20},
            "predicted": {101: 19, 102: 20},
        },
    }


class QualityEvaluatorTests(unittest.TestCase):
    def test_complete_reliable_evidence_passes_frozen_policy(self):
        result = evaluate_product_quality(
            coarse_metrics=_coarse_metrics(),
            source_quality={"metrics": {"source_boundary_f1_4": 0.90}},
            base_metrics={
                "semantic_gate_accuracy": 0.90,
                "semantic_gate_no_edit_accuracy": 0.20,
                "semantic_scale_evaluator_applicable": 1.0,
            },
            source_nuclei_calibration=_source_nuclei(),
            target_nuclei_counts={101: 20, 102: 10},
            generated_nuclei_counts={101: 20, 102: 10},
        )

        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.evidence_coverage, 1.0)
        self.assertGreaterEqual(result.quality_score, 0.75)
        self.assertEqual(result.scientific_status, "validated")
        self.assertEqual(
            result.to_metadata()["validated_interpretation"],
            "frozen_engineering_evaluator_pass_not_clinical_correctness",
        )

    def test_unreliable_segmentator_abstains_and_cannot_validate(self):
        metrics = _coarse_metrics()
        metrics["source_evaluator_calibration"]["accuracy"] = 0.69

        result = evaluate_product_quality(
            coarse_metrics=metrics,
            source_quality={"metrics": {"source_boundary_f1_4": 0.90}},
            base_metrics={"semantic_scale_evaluator_applicable": 1.0},
            source_nuclei_calibration=_source_nuclei(),
            target_nuclei_counts={101: 20, 102: 10},
            generated_nuclei_counts={101: 20, 102: 10},
        )

        self.assertFalse(result.applicability["semantic"])
        self.assertFalse(result.passed)
        self.assertEqual(result.scientific_status, "evaluator_uncertain")
        self.assertIn(
            "semantic_evaluator_unavailable",
            result.failed_checks,
        )

    def test_relative_semantic_can_validate_direction_when_absolute_is_unreliable(self):
        metrics = _coarse_metrics()
        metrics["source_evaluator_calibration"] = {
            "accuracy": 0.30,
            "macro_miou": 0.20,
        }
        metrics["transition_evaluator_calibration"].update(
            {
                "target_reference_available": False,
                "target_reference_recall_min": None,
                "source_to_target_confusion_rate": 0.80,
            }
        )
        metrics["changed_region"].update(
            {
                "target_probability_gain_fraction": 0.85,
                "source_probability_suppression_fraction": 0.80,
                "margin_gain_fraction": 0.90,
                "target_probability_gain": 0.12,
                "source_probability_suppression": 0.10,
                "soft_margin_gain": 0.22,
                "semantic_direction_epsilon": 1e-4,
            }
        )

        result = evaluate_product_quality(
            coarse_metrics=metrics,
            source_quality={"metrics": {"source_boundary_f1_4": 0.20}},
            base_metrics={"semantic_scale_evaluator_applicable": 1.0},
            source_nuclei_calibration=_source_nuclei(),
            target_nuclei_counts={101: 20, 102: 10},
            generated_nuclei_counts={101: 20, 102: 10},
        )

        self.assertTrue(result.applicability["semantic"])
        self.assertFalse(result.applicability["semantic_absolute"])
        self.assertTrue(result.applicability["semantic_relative"])
        self.assertAlmostEqual(result.evidence_coverage, 0.75)
        self.assertAlmostEqual(
            result.metrics["evidence_coverage_required"], 0.70
        )
        self.assertAlmostEqual(result.component_scores["semantic"], 1.0)
        self.assertTrue(result.passed)

    def test_relative_semantic_rejects_the_wrong_direction_without_abstaining(self):
        metrics = _coarse_metrics()
        metrics["source_evaluator_calibration"]["accuracy"] = 0.20
        metrics["changed_region"].update(
            {
                "target_probability_gain_fraction": 0.35,
                "source_probability_suppression_fraction": 0.30,
                "margin_gain_fraction": 0.40,
                "target_probability_gain": -0.05,
                "source_probability_suppression": -0.03,
                "soft_margin_gain": -0.08,
                "semantic_direction_epsilon": 1e-4,
            }
        )

        result = evaluate_product_quality(
            coarse_metrics=metrics,
            source_quality={"metrics": {"source_boundary_f1_4": 0.90}},
            base_metrics={"semantic_scale_evaluator_applicable": 1.0},
            source_nuclei_calibration=_source_nuclei(),
            target_nuclei_counts={101: 20, 102: 10},
            generated_nuclei_counts={101: 20, 102: 10},
        )

        self.assertTrue(result.applicability["semantic_relative"])
        self.assertEqual(result.scientific_status, "needs_review")
        self.assertIn("relative_semantic_direction", result.failed_checks)
        self.assertIn(
            "changed_region_semantic_direction_mismatch",
            result.reason_codes,
        )

    def test_target_only_relative_semantic_uses_supported_target_direction(self):
        metrics = _coarse_metrics()
        metrics["source_evaluator_calibration"]["accuracy"] = 0.20
        metrics["transition_evaluator_calibration"]["transition_pixels"] = 0
        metrics["changed_region"].update(
            {
                "target_probability_gain_fraction": 0.80,
                "source_probability_suppression_fraction": None,
                "margin_gain_fraction": None,
                "target_probability_gain": 0.12,
                "target_direction_support_pixels": 1200,
                "source_direction_support_pixels": 0,
                "margin_direction_support_pixels": 0,
                "semantic_direction_epsilon": 1e-4,
            }
        )

        result = evaluate_product_quality(
            coarse_metrics=metrics,
            source_quality={"metrics": {"source_boundary_f1_4": 0.20}},
            base_metrics={"semantic_scale_evaluator_applicable": 1.0},
            source_nuclei_calibration=_source_nuclei(),
            target_nuclei_counts={101: 20, 102: 10},
            generated_nuclei_counts={101: 20, 102: 10},
        )

        self.assertTrue(result.applicability["semantic_relative"])
        self.assertTrue(result.applicability["semantic_relative_target_only"])
        self.assertFalse(result.applicability["semantic_relative_source_only"])
        self.assertAlmostEqual(result.component_scores["semantic"], 1.0)
        self.assertTrue(result.passed)

    def test_source_only_relative_semantic_uses_whole_region_direction(self):
        metrics = _coarse_metrics()
        metrics["source_evaluator_calibration"]["accuracy"] = 0.20
        metrics["transition_evaluator_calibration"]["transition_pixels"] = 0
        metrics["changed_region"].update(
            {
                "target_probability_gain_fraction": None,
                "source_probability_suppression_fraction": 0.55,
                "margin_gain_fraction": None,
                "source_probability_suppression": 0.03,
                "target_direction_support_pixels": 0,
                "source_direction_support_pixels": 1200,
                "margin_direction_support_pixels": 0,
                "semantic_direction_epsilon": 1e-4,
            }
        )

        result = evaluate_product_quality(
            coarse_metrics=metrics,
            source_quality={"metrics": {"source_boundary_f1_4": 0.20}},
            base_metrics={"semantic_scale_evaluator_applicable": 1.0},
            source_nuclei_calibration=_source_nuclei(),
            target_nuclei_counts={101: 20, 102: 10},
            generated_nuclei_counts={101: 20, 102: 10},
        )

        self.assertTrue(result.applicability["semantic_relative"])
        self.assertTrue(result.applicability["semantic_relative_source_only"])
        self.assertAlmostEqual(result.component_scores["semantic"], 1.0)
        self.assertNotIn("relative_semantic_direction", result.failed_checks)

    def test_drift_beyond_generation_context_keeps_the_frozen_gate(self):
        metrics = _coarse_metrics()
        metrics["preservation"]["prediction_relative_drift_U_far"] = 0.081
        metrics["preservation"][
            "appearance_calibrated_prediction_drift_U_far"
        ] = 0.081

        result = evaluate_product_quality(
            coarse_metrics=metrics,
            source_quality={"metrics": {"source_boundary_f1_4": 0.90}},
            base_metrics={
                "semantic_gate_accuracy": 0.90,
                "semantic_gate_no_edit_accuracy": 0.20,
                "semantic_scale_evaluator_applicable": 1.0,
            },
            source_nuclei_calibration=_source_nuclei(),
            target_nuclei_counts={101: 20, 102: 10},
            generated_nuclei_counts={101: 20, 102: 10},
        )

        self.assertFalse(result.passed)
        self.assertIn("off_target_drift", result.failed_checks)
        self.assertIn("unedited_region_semantic_drift", result.reason_codes)

    def test_cellvit_count_can_apply_while_type_abstains(self):
        source_nuclei = {
            "changed_region": {
                "reference": {101: 10, 102: 10},
                "predicted": {103: 20},
            },
            "full_image": {
                "reference": {101: 20, 102: 20},
                "predicted": {103: 40},
            },
        }
        result = evaluate_product_quality(
            coarse_metrics=_coarse_metrics(),
            source_quality={"metrics": {"source_boundary_f1_4": 0.90}},
            base_metrics={
                "semantic_gate_accuracy": 0.90,
                "semantic_gate_no_edit_accuracy": 0.20,
                "semantic_scale_evaluator_applicable": 1.0,
            },
            source_nuclei_calibration=source_nuclei,
            target_nuclei_counts={101: 20, 102: 20},
            generated_nuclei_counts={101: 20, 102: 20},
            policy=QualityPolicy(),
        )

        self.assertTrue(result.applicability["nuclei_count"])
        self.assertFalse(result.applicability["nuclei_type"])
        self.assertIn(
            "nuclei_type_evaluator_unreliable",
            result.reason_codes,
        )

    def test_unreliable_local_cellvit_falls_back_to_reliable_full_image(self):
        source_nuclei = {
            "changed_region": {
                "reference": {101: 20},
                "predicted": {101: 5},
            },
            "full_image": {
                "reference": {101: 100},
                "predicted": {101: 100},
            },
        }
        result = evaluate_product_quality(
            coarse_metrics=_coarse_metrics(),
            source_quality={"metrics": {"source_boundary_f1_4": 0.90}},
            base_metrics={
                "semantic_gate_accuracy": 0.90,
                "semantic_gate_no_edit_accuracy": 0.20,
                "semantic_scale_evaluator_applicable": 1.0,
            },
            source_nuclei_calibration=source_nuclei,
            target_nuclei_counts={101: 20},
            generated_nuclei_counts={101: 20},
        )

        self.assertTrue(result.applicability["nuclei_count"])
        self.assertTrue(result.applicability["nuclei_type"])
        self.assertEqual(
            result.metrics["nuclei_detection_calibration_local"], 0.0
        )

    def test_allowed_global_appearance_redraw_uses_calibrated_drift(self):
        metrics = _coarse_metrics()
        metrics["preservation"]["prediction_relative_drift_U_far"] = 0.30
        metrics["preservation"][
            "appearance_calibrated_prediction_drift_U_far"
        ] = 0.03

        result = evaluate_product_quality(
            coarse_metrics=metrics,
            source_quality={"metrics": {"source_boundary_f1_4": 0.90}},
            base_metrics={"semantic_scale_evaluator_applicable": 1.0},
            source_nuclei_calibration=_source_nuclei(),
            target_nuclei_counts={101: 20, 102: 10},
            generated_nuclei_counts={101: 20, 102: 10},
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.metrics["preservation_raw_drift_u_far"], 0.30)
        self.assertEqual(result.metrics["preservation_drift_u_far"], 0.03)

    def test_missing_appearance_anchors_abstains_preservation(self):
        metrics = _coarse_metrics()
        metrics["preservation"]["appearance_calibration_applicable"] = False
        metrics["preservation"][
            "appearance_calibrated_prediction_drift_U_far"
        ] = None

        result = evaluate_product_quality(
            coarse_metrics=metrics,
            source_quality={"metrics": {"source_boundary_f1_4": 0.90}},
            base_metrics={"semantic_scale_evaluator_applicable": 1.0},
            source_nuclei_calibration=None,
            target_nuclei_counts=None,
            generated_nuclei_counts=None,
        )

        self.assertFalse(result.applicability["preservation"])
        self.assertIn(
            "preservation_evaluator_unavailable", result.reason_codes
        )

    def test_relative_boundary_does_not_require_absolute_source_bf1(self):
        metrics = _coarse_metrics()
        metrics["source_evaluator_calibration"]["accuracy"] = 0.20
        metrics["boundary"].update(
            {
                "relative_inner_target_probability_gain": 0.08,
                "relative_inner_source_probability_suppression": 0.06,
                "relative_inner_margin_gain": 0.14,
                "relative_inner_target_support_pixels": 180,
                "relative_inner_source_support_pixels": 180,
                "relative_outer_drift": 0.03,
                "relative_outer_applicable": True,
            }
        )

        result = evaluate_product_quality(
            coarse_metrics=metrics,
            source_quality={"metrics": {"source_boundary_f1_4": 0.20}},
            base_metrics={"semantic_scale_evaluator_applicable": 1.0},
            source_nuclei_calibration=_source_nuclei(),
            target_nuclei_counts={101: 20, 102: 10},
            generated_nuclei_counts={101: 20, 102: 10},
        )

        self.assertTrue(result.applicability["boundary"])
        self.assertFalse(result.applicability["boundary_absolute"])
        self.assertTrue(result.applicability["boundary_relative"])
        self.assertAlmostEqual(result.component_scores["boundary"], 0.985)

    def test_report_explains_semantic_and_preservation_failures(self):
        report = build_generation_report(
            {
                "status": "needs_review",
                "route": {
                    "primary_mode": "inpaint",
                    "reason": "small changed region",
                },
                "attempts": [
                    {
                        "attempt_index": 1,
                        "requested_mode": "inpaint",
                        "verification": {
                            "passed": False,
                            "quality_score": 0.60,
                            "evidence_coverage": 0.90,
                            "reason_codes": [
                                "changed_region_semantic_mismatch",
                                "unedited_region_semantic_drift",
                            ],
                        },
                    }
                ],
                "selected_attempt": {
                    "attempt_index": 1,
                    "requested_mode": "inpaint",
                    "verification": {
                        "passed": False,
                        "quality_score": 0.60,
                        "reason_codes": [
                            "changed_region_semantic_mismatch",
                            "unedited_region_semantic_drift",
                        ],
                    },
                },
            }
        )

        text = " ".join(report["final_assessment"])
        self.assertIn("目标语义", text)
        self.assertIn("未编辑区域", text)
        self.assertIn("未通过自动工程验证", report["validated_interpretation"])

    def test_replay_selection_uses_product_tie_break_order(self):
        attempts = [
            {
                "attempt_index": 1,
                "requested_mode": "inpaint",
                "verification": {
                    "passed": False,
                    "quality_score": 0.70,
                    "scientific_status": "needs_review",
                    "component_scores": {
                        "semantic": 0.65,
                        "preservation": 0.95,
                    },
                },
            },
            {
                "attempt_index": 2,
                "requested_mode": "cross-v1-no-ip-pix2pix-v2",
                "verification": {
                    "passed": False,
                    "quality_score": 0.70,
                    "scientific_status": "evaluator_uncertain",
                    "component_scores": {
                        "semantic": 0.68,
                        "preservation": 0.80,
                    },
                },
            },
        ]

        selected, status = select_replayed_attempt(attempts)

        self.assertEqual(selected["attempt_index"], 2)
        self.assertEqual(status, "evaluator_uncertain")


if __name__ == "__main__":
    unittest.main()

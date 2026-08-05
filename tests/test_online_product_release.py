import json
import unittest
from pathlib import Path

from controlnet_train.inference.model_paths import (
    DEFAULT_CROSS_V1_CHECKPOINT,
    DEFAULT_INPAINT_CHECKPOINT,
    FROZEN_CELLVIT_SHA256,
    FROZEN_PROBNET_SHA256,
    PRODUCTION_CONTROLNET_RELEASES,
    PRODUCTION_PIX2PIX_EPOCH,
    PRODUCTION_PIX2PIX_GLOBAL_STEP,
    PRODUCTION_PIX2PIX_SHA256,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCT_RELEASE = (
    REPO_ROOT
    / "benchmark_configs"
    / "releases"
    / "online_agent_product_v1.json"
)
SEGMENTATOR_RELEASE = (
    REPO_ROOT
    / "benchmark_configs"
    / "releases"
    / "segmentator_fine_legacy_anchor.json"
)


class OnlineProductReleaseTests(unittest.TestCase):
    def test_manifest_matches_runtime_model_pins(self):
        release = json.loads(PRODUCT_RELEASE.read_text(encoding="utf-8"))
        segmentator = json.loads(SEGMENTATOR_RELEASE.read_text(encoding="utf-8"))

        nuclei = release["nuclei_generation"]
        generation = release["image_generation"]
        verification = release["verification"]
        self.assertRegex(release["code_commit"], r"^[0-9a-f]{40}$")
        self.assertRegex(segmentator["code_commit"], r"^[0-9a-f]{40}$")
        self.assertEqual(nuclei["checkpoint_sha256"], FROZEN_PROBNET_SHA256)
        self.assertEqual(nuclei["repo_id"], "Qinxin11/pathology-probnet")
        self.assertEqual(
            nuclei["repo_checkpoint_path"],
            "best_epoch29_c29607f1b609accb.pt",
        )
        self.assertRegex(nuclei["repo_checkpoint_revision"], r"^[0-9a-f]{40}$")
        self.assertEqual(
            nuclei["candidate_queue_policy"],
            "probnet_odds_mass_without_replacement",
        )
        self.assertEqual(
            nuclei["candidate_quality_score"],
            "gamma_times_logit_probnet_probability_plus_seeded_gumbel",
        )
        self.assertEqual(
            nuclei["candidate_probability_mass_exponent"],
            3.0,
        )
        self.assertEqual(nuclei["candidate_diversity_weight"], 0.0)
        self.assertEqual(nuclei["quota_coverage_spacing_scale"], 0.0)
        self.assertEqual(nuclei["quota_coverage_max_radius"], 0.0)
        self.assertEqual(
            nuclei["retry_tail_policy"],
            "same_probnet_mass_permutation_then_component_pixel_backfill_"
            "then_same_tissue_quota_reassignment",
        )
        self.assertEqual(
            nuclei["component_quota_reassignment_policy"],
            "unplaceable_component_quota_to_same_tissue_probnet_mass_tail",
        )
        self.assertEqual(
            nuclei["failed_reference_shape_policy"],
            "quarantine_within_current_candidate_try_alternative_reference_"
            "or_library_then_restore",
        )
        self.assertEqual(
            nuclei["exact_backfill_candidate_policy"],
            "quota_scaled_budget_exhaustion_advances_to_next_deterministic_"
            "seed_without_relaxing_count_type_or_shape",
        )
        self.assertEqual(nuclei["exact_backfill_candidates_per_missing"], 128)
        self.assertEqual(nuclei["exact_backfill_candidate_floor"], 512)
        self.assertEqual(nuclei["exact_backfill_candidate_ceiling"], 4096)
        self.assertEqual(
            nuclei["count_policy"],
            "pre_edit_source_tissue_density_or_target_prior_calibrated_by_"
            "pre_edit_source_times_post_edit_target_area",
        )
        self.assertEqual(
            nuclei["type_quota_routing_policy"],
            "prior_total_count_then_probnet_local_type_log_pool_with_"
            "cumulative_posterior_balancing",
        )
        self.assertEqual(nuclei["gamma"], 3.0)
        self.assertFalse(nuclei["organ_specific_constraints"])
        self.assertEqual(
            nuclei["sampling_audit_policy"],
            "probnet_patch_relative_count_type_spatial_v3",
        )
        frozen_v2 = nuclei["compatible_frozen_audit_policies"][
            "probnet_patch_relative_count_type_spatial_v2"
        ]
        self.assertEqual(
            frozen_v2["scope"],
            "hash_locked_approved_nuclei_replay_only",
        )
        self.assertEqual(frozen_v2["sampling_audit_max_attempts"], 6)
        self.assertFalse(frozen_v2["sampling_feedback_required"])
        self.assertEqual(
            nuclei["sampling_audit_probability_bin_tv_role"],
            "diagnostic_score_term_not_hard_gate_under_constrained_without_"
            "replacement_sampling",
        )
        self.assertEqual(nuclei["sampling_audit_attempts"], 3)
        self.assertTrue(nuclei["sampling_audit_required"])
        self.assertEqual(
            nuclei["sampling_feedback_policy"],
            "reason_directed_gamma_then_seed_v1",
        )
        self.assertEqual(nuclei["sampling_feedback_max_attempts"], 3)
        self.assertEqual(nuclei["sampling_feedback_gamma_down_factor"], 0.75)
        self.assertAlmostEqual(
            nuclei["sampling_feedback_gamma_up_factor"],
            4.0 / 3.0,
        )
        self.assertEqual(nuclei["sampling_feedback_gamma_min"], 1.5)
        self.assertEqual(nuclei["sampling_feedback_gamma_max"], 5.0)
        self.assertEqual(
            nuclei["sampling_feedback_g2_600_validation"]["cases"],
            600,
        )
        self.assertEqual(
            nuclei["sampling_feedback_g2_600_validation"][
                "required_more_than_three_attempts"
            ],
            0,
        )
        self.assertEqual(
            nuclei["sampling_feedback_v3_validation"]["status"],
            "targeted_probe_passed",
        )
        self.assertFalse(
            nuclei["sampling_feedback_v3_validation"]["threshold_calibration"]
        )
        self.assertEqual(nuclei["nucleus_spacing_margin_px"], 1)
        self.assertEqual(
            nuclei["instance_connectivity_policy"],
            "largest_8_connected_component_after_transform",
        )
        self.assertEqual(
            nuclei["nuclei_generation_region_policy"],
            "semantic_region_for_generic_edits_whole_connected_gland_"
            "for_glas_structure_edits",
        )
        self.assertEqual(
            generation["inference"],
            {
                "num_inference_steps": 28,
                "guidance_scale": 3.5,
                "controlnet_conditioning_scale": 1.0,
                "torch_dtype": "bf16",
                "seed": 42,
            },
        )
        self.assertEqual(
            generation["pix2pix"]["checkpoint_sha256"],
            PRODUCTION_PIX2PIX_SHA256,
        )
        self.assertEqual(
            generation["pix2pix"]["epoch"],
            PRODUCTION_PIX2PIX_EPOCH,
        )
        self.assertEqual(
            generation["pix2pix"]["global_step"],
            PRODUCTION_PIX2PIX_GLOBAL_STEP,
        )
        low_stain = generation["pix2pix"]["low_stain_protection"]
        self.assertEqual(low_stain["policy"], "cross_rgb_od_low_stain_v1")
        self.assertEqual(low_stain["scope"], "generation_change_region")
        self.assertFalse(low_stain["organ_specific_constraints"])
        self.assertFalse(generation["pix2pix"]["color_matching_postprocess"])
        self.assertEqual(
            verification["cellvit_checkpoint_sha256"],
            FROZEN_CELLVIT_SHA256,
        )
        self.assertEqual(
            verification["segmentator_release_id"],
            segmentator["release_id"],
        )
        self.assertEqual(
            verification["segmentator_checkpoint_sha256"],
            segmentator["checkpoint_sha256"],
        )
        self.assertEqual(
            verification["segmentator_distribution"],
            segmentator["distribution"]["mode"],
        )
        self.assertEqual(
            segmentator["checkpoint_environment_selector"],
            "PATHOLOGY_SEGMENTATOR_CHECKPOINT",
        )
        evaluator = verification["evaluator"]
        self.assertEqual(
            evaluator["policy_id"], "online-quality-evaluator-v2.4"
        )
        self.assertEqual(evaluator["relative_semantic_evidence_weight"], 0.35)
        self.assertEqual(evaluator["relative_evidence_coverage_min"], 0.70)
        self.assertEqual(evaluator["relative_semantic_score_min"], 0.60)
        self.assertEqual(
            evaluator["preservation_exclusion_region"],
            "full_generation_change_region",
        )
        self.assertIn(
            "appearance_calibrated",
            evaluator["preservation_formula"],
        )
        self.assertIn("full image", evaluator["global_appearance_redraw_policy"])

    def test_manifest_matches_packaged_controlnet_defaults(self):
        release = json.loads(PRODUCT_RELEASE.read_text(encoding="utf-8"))
        generation = release["image_generation"]

        self.assertEqual(
            generation["inpaint"]["checkpoint"],
            DEFAULT_INPAINT_CHECKPOINT,
        )
        self.assertEqual(
            generation["cross_v1"]["checkpoint"],
            DEFAULT_CROSS_V1_CHECKPOINT,
        )
        for mode, key in (("inpaint", "inpaint"), ("cross-v1", "cross_v1")):
            expected = PRODUCTION_CONTROLNET_RELEASES[mode]
            actual = generation[key]
            self.assertEqual(
                actual["weight_size_bytes"],
                expected["weight_size_bytes"],
            )
            self.assertEqual(
                actual["weight_sha256"],
                expected["weight_sha256"],
            )

    def test_release_entrypoints_exist(self):
        release = json.loads(PRODUCT_RELEASE.read_text(encoding="utf-8"))

        for relative_path in release["entrypoints"].values():
            self.assertTrue((REPO_ROOT / relative_path).is_file(), relative_path)


if __name__ == "__main__":
    unittest.main()

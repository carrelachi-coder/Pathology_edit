import csv
import json
import re
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from phase3_mask_edit.benchmark.intents import (
    BuildConfig,
    ProfileSource,
    build_benchmark_intents,
    estimate_capacity,
    inject_region_hint,
    infer_patient_id,
    infer_wsi_id,
    primitive_config_by_name,
    recommend_region_hint,
    source_target_labels_for_primitive,
    ordinal_groups_from_intents,
)
from phase3_mask_edit.benchmark.metrics import (
    evaluate_mask_edit,
    mode_aware_score_fields,
)
from phase3_mask_edit.benchmark.models import (
    BenchmarkIntent,
    BenchmarkPrompt,
    read_intents_jsonl,
    read_prompts_csv,
    write_eval_csv,
    write_intents_jsonl,
    write_prompts_csv,
)
from phase3_mask_edit.benchmark.prompts import (
    LLMConfig,
    _checker_gt,
    _report_pair_few_shots,
    accepted_prompts,
    check_prompt_with_parser,
    semantic_diff_for_intent,
    template_prompt_for_intent,
    validate_report_pair_language,
)
from phase3_mask_edit.benchmark.reporting import summarize_semantic_rows
from phase3_mask_edit.benchmark.runner import (
    GT_MODE,
    PROMPT_MODE,
    _benchmark_primitive_config,
    _clamp_near_boundary_point,
    run_benchmark_sample,
)
from phase3_mask_edit.backends.llm_agent import (
    LLMContourAgentResult,
    LLMContourAttempt,
    STATUS_VALIDATED,
)
from phase3_mask_edit.core.config import load_recipe
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.core.mask_io import save_id_mask
from phase3_mask_edit.cli.build_mask_edit_benchmark import main as build_benchmark_main
from phase3_mask_edit.cli.generate_mask_edit_benchmark_prompts import (
    main as generate_prompts_main,
)
from phase3_mask_edit.cli.merge_mask_edit_benchmark_results import (
    main as merge_results_main,
)
from phase3_mask_edit.cli.run_mask_edit_benchmark import (
    _enrich_agentic_row_from_artifacts,
    main as run_benchmark_main,
    summarize_rows,
)
from phase3_mask_edit.cli.select_mask_edit_benchmark_preflight import (
    select_preflight_intents,
    summarize_selection,
)
from phase3_mask_edit.cli.select_mask_edit_benchmark_rerun import (
    main as select_rerun_main,
)
from phase3_mask_edit.cli.validate_mask_edit_benchmark_manifest import (
    main as validate_manifest_main,
)
from phase3_mask_edit.generic.tumor_burden import PrimitiveEditResult
from phase3_mask_edit.parser.semantic_diff import DEFAULT_SEMANTIC_DIFF


class MaskEditBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.schema = MaskProfileSchema.from_reference_profile("BCSS")
        self.recipe = load_recipe("phase3_mask_edit/recipes/generic.yaml")

    def test_benchmark_necrosis_config_disables_off_target_intrusion_engulfment(self):
        source = {
            "name": "necrosis_appearance",
            "parameter_ranges": {"necrosis_intrusion_closing_radius_px": 6},
        }

        strict = _benchmark_primitive_config(source)

        self.assertEqual(
            strict["parameter_ranges"]["necrosis_intrusion_closing_radius_px"],
            0,
        )
        self.assertEqual(
            source["parameter_ranges"]["necrosis_intrusion_closing_radius_px"],
            6,
        )

    def test_preflight_selection_is_deterministic_and_prefers_accepted_qc(self):
        intents = []
        for organ in ("breast", "lung"):
            for strength in ("mild", "moderate"):
                for index, qc_status in enumerate(("pending", "accepted")):
                    intents.append(
                        BenchmarkIntent(
                            sample_id=f"{organ}-{strength}-{index}",
                            organ=organ,
                            profile="BCSS",
                            image_path=None,
                            mask_path="mask.png",
                            primitive="necrosis_appearance",
                            strength=strength,
                            region_hint={},
                            source_labels=("Tumor",),
                            target_label="Necrosis",
                            expected_direction="increase",
                            expected_area_bucket=(0.01, 0.10),
                            seed=index,
                            qc_status=qc_status,
                        )
                    )

        first = select_preflight_intents(intents, seed=13)
        second = select_preflight_intents(reversed(intents), seed=13)
        summary = summarize_selection(intents, first)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertTrue(all(item.qc_status == "accepted" for item in first))
        self.assertEqual(summary["num_source_cells"], 4)
        self.assertEqual(summary["num_selected_cells"], 4)
        self.assertEqual(summary["missing_cells"], [])

    def test_capacity_and_region_hint_for_necrosis(self):
        mask = np.zeros((64, 64), dtype=np.int64)
        mask[8:48, 8:48] = 1
        mask[48:60, 8:48] = 2
        primitive = primitive_config_by_name(self.recipe, "necrosis_appearance")
        intent = EditIntent(
            primitive="necrosis_appearance", strength="mild", reference_profile="BCSS"
        )

        capacity = estimate_capacity(mask, intent, primitive, self.schema)
        region_hint = recommend_region_hint(mask, self.schema, intent, primitive)

        self.assertEqual(capacity["status"], "executable")
        self.assertGreater(region_hint["area_pixels"], 0)
        self.assertIn("centroid_xy", region_hint)

    def test_jsonl_roundtrip_and_region_injection(self):
        gt = BenchmarkIntent(
            sample_id="s1",
            organ="breast",
            profile="BCSS",
            image_path=None,
            mask_path="mask.png",
            primitive="necrosis_appearance",
            strength="mild",
            region_hint={"location": "center", "centroid_xy": [3, 4]},
            source_labels=("Tumor",),
            target_label="Necrosis",
            expected_direction="increase",
            expected_area_bucket=(0.08, 0.14),
            seed=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "intents.jsonl"
            write_intents_jsonl([gt], path)
            loaded = read_intents_jsonl(path)[0]

        intent = inject_region_hint(
            EditIntent(primitive="necrosis_appearance", reference_profile="BCSS"),
            loaded.region_hint,
        )
        self.assertEqual(loaded.source_labels, ("Tumor",))
        self.assertEqual(intent.region_hint["location"], "center")

    def test_template_prompt_and_semantic_diff(self):
        gt = BenchmarkIntent(
            sample_id="s1",
            organ="breast",
            profile="BCSS",
            image_path=None,
            mask_path="mask.png",
            primitive="tumor_burden_decrease",
            strength="significant",
            region_hint={"location": "upper_left", "relation": "peripheral"},
            source_labels=("Tumor",),
            target_label="Stroma",
            expected_direction="decrease",
            expected_area_bucket=(0.24, 0.40),
            seed=1,
        )
        prompt = template_prompt_for_intent(gt)
        diff = semantic_diff_for_intent(gt)

        self.assertIn("decrease the tumor burden", prompt.instruction)
        self.assertEqual(diff["tumor_change"]["growth"], "decrease")
        self.assertEqual(diff["benchmark_gt"]["region_hint"]["location"], "upper_left")

    def test_transition_checker_gt_includes_exact_fine_states(self):
        intent = BenchmarkIntent(
            sample_id="transition",
            organ="colorectal",
            profile="GlaS",
            image_path=None,
            mask_path="mask.png",
            primitive="treatment_dedifferentiation",
            strength="mild",
            region_hint={"location": "center"},
            source_labels=("Tumor",),
            target_label="Tumor",
            expected_direction="transition",
            expected_area_bucket=(0.08, 0.14),
            seed=1,
        )

        gt = _checker_gt(intent)

        self.assertEqual(
            gt["source_state"], "poorly differentiated colorectal carcinoma"
        )
        self.assertEqual(
            gt["target_state"], "moderately differentiated colorectal carcinoma"
        )

        diff = semantic_diff_for_intent(intent)
        self.assertEqual(
            diff["transition_change"],
            {
                "source_state": "poorly_differentiated_carcinoma",
                "target_state": "moderately_differentiated_carcinoma",
                "degree": "mild",
            },
        )
        self.assertEqual(diff["tumor_change"]["grade_change"], "downgrade")

    def test_parser_checker_rejects_secondary_planned_edit(self):
        intent = BenchmarkIntent(
            sample_id="parser-check",
            organ="colorectal",
            profile="GlaS",
            image_path=None,
            mask_path="mask.png",
            primitive="adenoma_to_carcinoma",
            strength="mild",
            region_hint={"location": "center"},
            source_labels=("Tumor",),
            target_label="Tumor",
            expected_direction="transition",
            expected_area_bucket=(0.08, 0.14),
            seed=1,
        )
        prompt = BenchmarkPrompt(
            sample_id=intent.sample_id,
            old_prompt="Central adenomatous colorectal glands.",
            new_prompt="Central carcinoma with much more tumor and dense stroma.",
            instruction="Convert adenoma to carcinoma.",
        )
        parsed = semantic_diff_for_intent(intent)
        parsed["tumor_change"]["growth"] = "increase"
        with mock.patch(
            "phase3_mask_edit.benchmark.prompts.parse_prompts_with_api",
            return_value=parsed,
        ):
            checked = check_prompt_with_parser(
                intent,
                prompt,
                LLMConfig(model="parser"),
            )

        self.assertEqual(checked.checker_status, "rejected")
        self.assertIn("parser_checker_mismatch", checked.checker_reason)
        self.assertIn("tumor_burden_increase", checked.checker_reason)

    def test_parser_checker_treats_strength_as_non_exact_prompt_metadata(self):
        intent = BenchmarkIntent(
            sample_id="strength-calibration",
            organ="breast",
            profile="BCSS",
            image_path=None,
            mask_path="mask.png",
            primitive="tumor_burden_increase",
            strength="significant",
            region_hint={"location": "center"},
            source_labels=("Stroma",),
            target_label="Tumor",
            expected_direction="increase",
            expected_area_bucket=(0.24, 0.40),
            seed=1,
        )
        prompt = BenchmarkPrompt(
            sample_id=intent.sample_id,
            old_prompt="The breast specimen contains sparse central tumor nests.",
            new_prompt="The breast specimen contains conspicuous central tumor nests.",
            instruction="Increase the central tumor burden.",
        )
        parsed = semantic_diff_for_intent(intent)
        parsed["tumor_change"]["degree"] = "moderate"
        with mock.patch(
            "phase3_mask_edit.benchmark.prompts.parse_prompts_with_api",
            return_value=parsed,
        ):
            checked = check_prompt_with_parser(
                intent,
                prompt,
                LLMConfig(model="parser"),
            )

        self.assertEqual(checked.checker_status, "accepted")
        self.assertIn("strength_label_agreement=false", checked.checker_reason)

    def test_report_language_validator_rejects_process_and_comparison_leaks(self):
        violations = (
            "The stroma is prominent relative to the tumor component.",
            "A focal reduction of stromal tissue is present.",
            "Tumor nests remain visible in the center.",
            "The central necrosis has resolved.",
            "A denser collagenous compartment is present.",
            "The epithelium is replaced by necrotic material.",
        )
        for text in violations:
            with self.subTest(text=text):
                prompt = BenchmarkPrompt(
                    sample_id="language",
                    old_prompt="The breast specimen contains central tumor nests.",
                    new_prompt=text,
                    instruction="Edit the mask.",
                )
                self.assertIsNotNone(validate_report_pair_language(prompt))

    def test_report_language_validator_allows_static_pathology_terms(self):
        prompt = BenchmarkPrompt(
            sample_id="state-only",
            old_prompt=(
                "The colorectal specimen shows adenomatous change with preserved "
                "glandular architecture and residual necrotic debris."
            ),
            new_prompt=(
                "The colorectal specimen shows adenomatous change with preserved "
                "glandular architecture and residual necrotic debris."
            ),
            instruction="Edit the mask.",
        )

        self.assertIsNone(validate_report_pair_language(prompt))

    def test_accepted_prompts_excludes_deterministic_language_violations(self):
        valid = BenchmarkPrompt(
            sample_id="valid",
            old_prompt="The breast specimen contains sparse central tumor nests.",
            new_prompt="The breast specimen contains conspicuous central tumor nests.",
            instruction="Increase the central tumor burden.",
            checker_status="accepted",
        )
        invalid = BenchmarkPrompt(
            sample_id="invalid",
            old_prompt="The breast specimen contains sparse central tumor nests.",
            new_prompt="The tumor is prominent relative to the stromal compartment.",
            instruction="Increase the central tumor burden.",
            checker_status="accepted",
        )

        self.assertEqual(
            [item.sample_id for item in accepted_prompts([valid, invalid])],
            ["valid"],
        )

    def test_report_pair_few_shots_share_scaffold_and_are_state_only(self):
        for payload in _report_pair_few_shots():
            with self.subTest(summary=payload["gt_summary"]):
                prompt = BenchmarkPrompt(
                    sample_id="few-shot",
                    old_prompt=payload["old_prompt"],
                    new_prompt=payload["new_prompt"],
                    instruction=payload["instruction"],
                )
                self.assertIsNone(validate_report_pair_language(prompt))
                old_sentences = {
                    item.strip()
                    for item in re.split(r"(?<=[.!?])\s+", prompt.old_prompt)
                    if item.strip()
                }
                new_sentences = {
                    item.strip()
                    for item in re.split(r"(?<=[.!?])\s+", prompt.new_prompt)
                    if item.strip()
                }
                self.assertGreaterEqual(len(old_sentences & new_sentences), 2)

    def test_tumor_increase_gt_uses_editable_source_and_tumor_anchor(self):
        primitive = primitive_config_by_name(self.recipe, "tumor_burden_increase")
        source_labels, target_label = source_target_labels_for_primitive(
            primitive, self.schema
        )

        self.assertIn("Stroma", source_labels)
        self.assertNotIn("Tumor", source_labels)
        self.assertEqual(target_label, "Tumor")

    def test_metrics_detect_expected_change(self):
        source = np.ones((32, 32), dtype=np.int64)
        target = source.copy()
        target[8:16, 8:16] = 3
        gt = BenchmarkIntent(
            sample_id="s1",
            organ="breast",
            profile="BCSS",
            image_path=None,
            mask_path="mask.png",
            primitive="necrosis_appearance",
            strength="mild",
            region_hint={"bbox_xyxy": [6, 6, 18, 18], "centroid_xy": [12, 12]},
            source_labels=("Tumor",),
            target_label="Necrosis",
            expected_direction="increase",
            expected_area_bucket=(0.04, 0.20),
            seed=1,
        )

        metrics = evaluate_mask_edit(source, target, gt)

        self.assertTrue(metrics["class_ok"])
        self.assertTrue(metrics["direction_ok"])
        self.assertTrue(metrics["strength_ok"])
        self.assertTrue(metrics["location_ok"])
        self.assertEqual(metrics["on_target_transition_ratio"], 1.0)
        self.assertEqual(metrics["off_target_change_ratio"], 0.0)
        self.assertEqual(metrics["spatial_containment_ratio"], 1.0)

    def test_prompt_primary_score_excludes_hidden_exact_strength_bucket(self):
        raw = {
            "class_ok": True,
            "direction_ok": True,
            "location_ok": True,
            "magnitude_bucket_pass": False,
        }

        prompt_scores = mode_aware_score_fields(raw, mode="prompt")
        gt_scores = mode_aware_score_fields(raw, mode="gt")

        self.assertTrue(prompt_scores["semantic_core_ok"])
        self.assertTrue(prompt_scores["primary_ok"])
        self.assertFalse(prompt_scores["strict_all_ok"])
        self.assertFalse(gt_scores["primary_ok"])
        self.assertEqual(
            prompt_scores["strength_evaluation_policy"],
            "ordinal_secondary_hidden_bucket_diagnostic",
        )

    def test_transition_metrics_use_fine_ids_for_treatment_dedifferentiation(self):
        source = np.full((20, 20), 13, dtype=np.int64)
        target = source.copy()
        target[5:15, 5:15] = 12
        gt = BenchmarkIntent(
            sample_id="g1",
            organ="colorectal",
            profile="GlaS",
            image_path=None,
            mask_path="mask.png",
            primitive="treatment_dedifferentiation",
            strength="moderate",
            region_hint={"bbox_xyxy": [4, 4, 16, 16], "centroid_xy": [10, 10]},
            source_labels=("Tumor",),
            target_label="Tumor",
            expected_direction="transition",
            expected_area_bucket=(0.20, 0.40),
            seed=1,
        )

        metrics = evaluate_mask_edit(source, target, gt)

        self.assertTrue(metrics["direction_hit"])
        self.assertEqual(metrics["on_target_transition_ratio"], 1.0)
        self.assertTrue(metrics["magnitude_bucket_pass"])

    def test_ottr_accepts_every_recipe_legal_backfill_label(self):
        source = np.full((20, 20), 2, dtype=np.int64)
        target = source.copy()
        target[5:15, 5:15] = 1
        gt = BenchmarkIntent(
            sample_id="s1",
            organ="breast",
            profile="BCSS",
            image_path=None,
            mask_path="mask.png",
            primitive="stroma_decrease",
            strength="moderate",
            region_hint={"bbox_xyxy": [4, 4, 16, 16], "centroid_xy": [10, 10]},
            source_labels=("Stroma",),
            target_label="Other tissue",
            expected_direction="decrease",
            expected_area_bucket=(0.14, 0.24),
            seed=1,
        )

        metrics = evaluate_mask_edit(source, target, gt)

        self.assertEqual(metrics["on_target_transition_ratio"], 1.0)
        self.assertEqual(metrics["off_target_change_ratio"], 0.0)

    def test_boundary_tolerance_only_clamps_small_overshoot(self):
        self.assertEqual(
            _clamp_near_boundary_point([512, 520], width=512, height=512, tolerance=16),
            [511.0, 511.0],
        )
        self.assertEqual(
            _clamp_near_boundary_point([540, 10], width=512, height=512, tolerance=16),
            [540, 10.0],
        )

    def test_build_intents_from_temp_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask = np.zeros((96, 96), dtype=np.int64)
            mask[16:72, 16:72] = 1
            mask[72:88, 16:72] = 2
            mask_path = root / "BCSS" / "sample_mask.png"
            save_id_mask(mask, mask_path)
            config = BuildConfig(
                data_root=root,
                output_dir=root / "out",
                profiles=(ProfileSource("breast", "BCSS", ("BCSS/*mask.png",)),),
                patches_per_combo=1,
                strengths=("mild",),
                allowed_primitives=("necrosis_appearance",),
                seed=7,
            )

            intents, summary = build_benchmark_intents(config)

        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].primitive, "necrosis_appearance")
        self.assertFalse(summary["shortfalls"])

    def test_complete_ordinal_groups_share_reference_across_strengths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask = np.ones((128, 128), dtype=np.int64)
            mask[:8, :] = 2
            mask_path = root / "BCSS" / "sample_mask.png"
            save_id_mask(mask, mask_path)
            config = BuildConfig(
                data_root=root,
                output_dir=root / "out",
                profiles=(ProfileSource("breast", "BCSS", ("BCSS/*mask.png",)),),
                patches_per_combo=1,
                strengths=("mild", "moderate", "significant"),
                allowed_primitives=("necrosis_appearance",),
                seed=7,
                early_stop_when_full=False,
                require_complete_ordinal_groups=True,
            )

            intents, summary = build_benchmark_intents(config)
            groups = ordinal_groups_from_intents(intents)

        self.assertEqual(len(intents), 3)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["strengths"], ["mild", "moderate", "significant"])
        self.assertEqual(summary["num_ordinal_groups"], 1)

    def test_zero_candidate_expected_cell_is_reported_as_shortfall(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask = np.ones((64, 64), dtype=np.int64)
            mask[:8, :] = 2
            save_id_mask(mask, root / "BCSS" / "sample_mask.png")
            config = BuildConfig(
                data_root=root,
                output_dir=root / "out",
                profiles=(ProfileSource("breast", "BCSS", ("BCSS/*mask.png",)),),
                patches_per_combo=1,
                strengths=("mild",),
                allowed_primitives=("immune_infiltration_decrease",),
                seed=7,
            )

            intents, summary = build_benchmark_intents(config)

        self.assertFalse(intents)
        self.assertEqual(len(summary["shortfalls"]), 1)
        self.assertEqual(summary["shortfalls"][0]["available"], 0)

    def test_build_cli_writes_formal_manifests_and_image_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "BCSS_PATCHES" / "images"
            masks = root / "BCSS_PATCHES" / "tissue_masks"
            stem = "TCGA-A1-A0SK_x1_y2_py0_px0"
            mask = np.ones((128, 128), dtype=np.int64)
            mask[:8, :] = 2
            save_id_mask(mask, masks / f"{stem}.png")
            rng = np.random.default_rng(3)
            image = rng.integers(25, 230, size=(128, 128, 3), dtype=np.uint8)
            images.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(images / f"{stem}.png")
            output = root / "out"
            config_path = root / "benchmark.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "data_root": str(root),
                        "output_dir": str(output),
                        "patches_per_combo": 1,
                        "strengths": ["mild"],
                        "allowed_primitives": ["necrosis_appearance"],
                        "require_image": True,
                        "require_complete_ordinal_groups": True,
                        "profiles": [
                            {
                                "organ": "breast",
                                "profile": "BCSS",
                                "mask_globs": ["BCSS_PATCHES/tissue_masks/*.png"],
                                "image_globs": ["BCSS_PATCHES/images/*.png"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            exit_code = build_benchmark_main(["--config", str(config_path)])
            intents = read_intents_jsonl(output / "mask_semantic_intents.jsonl")
            validation_path = output / "validation.json"
            validation_exit = validate_manifest_main(
                [
                    "--intents",
                    str(output / "mask_semantic_intents.jsonl"),
                    "--shortfalls",
                    str(output / "shortfalls.csv"),
                    "--output",
                    str(validation_path),
                    "--expected-per-cell",
                    "1",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(validation_exit, 0)
            self.assertTrue(json.loads(validation_path.read_text())["valid"])
            self.assertEqual(len(intents), 1)
            self.assertEqual(intents[0].wsi_id, "TCGA-A1-A0SK")
            self.assertEqual(intents[0].patient_id, "TCGA-A1-A0SK")
            self.assertEqual(intents[0].qc_status, "accepted")
            self.assertTrue(intents[0].image_path)
            for name in (
                "shortfalls.csv",
                "ordinal_groups.jsonl",
                "intent_qc.manual_review.csv",
                "effective_build_config.json",
            ):
                self.assertTrue((output / name).is_file())

    def test_semantic_report_computes_ordinal_monotonicity(self):
        rows = []
        for strength, value in (("mild", 0.1), ("moderate", 0.2), ("significant", 0.3)):
            rows.append(
                {
                    "sample_id": strength,
                    "status": "completed",
                    "mode": "gt",
                    "organ": "breast",
                    "primitive": "necrosis_appearance",
                    "strength": strength,
                    "wsi_id": "wsi-1",
                    "ordinal_group_id": "group-1",
                    "measured_area_fraction": value,
                    "direction_hit": True,
                    "on_target_transition_ratio": 1.0,
                    "off_target_change_ratio": 0.0,
                    "spatial_containment_ratio": 1.0,
                    "magnitude_bucket_pass": True,
                    "all_ok": True,
                }
            )

        report = summarize_semantic_rows(rows, bootstrap_iterations=20, seed=3)

        self.assertEqual(report["groups"]["overall"]["cluster_unit"], "wsi_id")
        self.assertEqual(report["ordinal"]["n_groups"], 1)
        self.assertEqual(report["ordinal"]["mean_spearman_rho"], 1.0)
        self.assertEqual(report["ordinal"]["nondecreasing_monotonicity_rate"], 1.0)
        self.assertEqual(report["ordinal"]["pairwise_concordance_rate"], 1.0)
        self.assertEqual(report["ordinal"]["pairwise_reversal_rate"], 0.0)
        self.assertEqual(report["ordinal"]["by_mode"]["gt"]["n_groups"], 1)
        self.assertEqual(
            report["ordinal"]["by_mode_and_n_strengths"]["gt|n_strengths:3"][
                "n_groups"
            ],
            1,
        )

    def test_semantic_binary_metrics_include_failed_rows(self):
        rows = [
            {
                "sample_id": "ok",
                "status": "completed",
                "mode": "gt",
                "organ": "breast",
                "primitive": "necrosis_appearance",
                "strength": "mild",
                "wsi_id": "wsi-1",
                "direction_hit": True,
                "on_target_transition_ratio": 1.0,
                "off_target_change_ratio": 0.0,
                "spatial_containment_ratio": 1.0,
                "magnitude_bucket_pass": True,
                "all_ok": True,
            },
            {
                "sample_id": "failed",
                "status": "failed",
                "mode": "gt",
                "organ": "breast",
                "primitive": "necrosis_appearance",
                "strength": "mild",
                "wsi_id": "wsi-2",
            },
        ]

        overall = summarize_semantic_rows(rows, bootstrap_iterations=20, seed=3)[
            "groups"
        ]["overall"]

        self.assertEqual(overall["completion_rate"], 0.5)
        self.assertEqual(overall["direction_hit_rate"]["value"], 0.5)
        self.assertEqual(overall["direction_hit_rate"]["metric_n"], 2)
        self.assertEqual(overall["on_target_transition_ratio"]["metric_n"], 1)

    def test_infers_wsi_and_patient_from_patch_paths(self):
        panda = "0018ae58b01bdadc8e347995b69f99aa_y6912_x768_py0_px1024.png"
        orca = "TCGA-4P-AA8J-01Z-00-DX1.slide_0_py0_px1536.png"
        self.assertEqual(infer_wsi_id(panda), "0018ae58b01bdadc8e347995b69f99aa")
        self.assertEqual(infer_wsi_id(orca), "TCGA-4P-AA8J-01Z-00-DX1.slide")
        self.assertEqual(infer_patient_id(orca), "TCGA-4P-AA8J")

    def test_direct_gt_runner_bypasses_prompt_and_planner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = np.ones((32, 32), dtype=np.int64)
            mask_path = root / "mask.png"
            save_id_mask(source, mask_path)
            gt = BenchmarkIntent(
                sample_id="s1",
                organ="breast",
                profile="BCSS",
                image_path=None,
                mask_path=str(mask_path),
                primitive="necrosis_appearance",
                strength="mild",
                region_hint={"bbox_xyxy": [6, 6, 18, 18], "centroid_xy": [12, 12]},
                source_labels=("Tumor",),
                target_label="Necrosis",
                expected_direction="increase",
                expected_area_bucket=(0.04, 0.20),
                seed=1,
            )

            def fake_execute(**kwargs):
                target = kwargs["old_mask"].copy()
                target[8:16, 8:16] = 3
                edit_result = PrimitiveEditResult(
                    target_mask=target,
                    change_region=target != kwargs["old_mask"],
                    changed_area_fraction=float(np.mean(target != kwargs["old_mask"])),
                    selected_pixels=int(np.count_nonzero(target != kwargs["old_mask"])),
                    warnings=(),
                    ops_log={},
                )
                attempt = LLMContourAttempt(
                    attempt_index=1,
                    status=STATUS_VALIDATED,
                    edit_result=edit_result,
                )
                return LLMContourAgentResult(
                    status=STATUS_VALIDATED,
                    source_mask=kwargs["old_mask"],
                    attempts=(attempt,),
                    final_attempt=attempt,
                    context={"intent": kwargs["intent"].to_metadata()},
                    artifact_paths={},
                )

            with mock.patch(
                "phase3_mask_edit.benchmark.runner.execute_llm_contour_agent",
                side_effect=fake_execute,
            ):
                row = run_benchmark_sample(
                    gt,
                    mode=GT_MODE,
                    output_dir=root / "out",
                    contour_provider="api-text",
                    contour_model="dummy",
                )

        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["mode"], GT_MODE)
        self.assertEqual(row["planned_primitive"], "necrosis_appearance")
        self.assertTrue(row["all_ok"])
        self.assertEqual(row["attempt_count"], 1)
        self.assertEqual(row["first_attempt_status"], STATUS_VALIDATED)
        self.assertEqual(row["final_attempt_status"], STATUS_VALIDATED)
        self.assertFalse(row["replanned"])
        self.assertFalse(row["repair_success"])
        self.assertTrue(row["cumulative_success_at_k"]["1"])

    def test_prompt_mode_repairs_no_executable_semantic_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = np.ones((32, 32), dtype=np.int64)
            mask_path = root / "mask.png"
            save_id_mask(source, mask_path)
            intent = BenchmarkIntent(
                sample_id="semantic-repair",
                organ="breast",
                profile="BCSS",
                image_path=None,
                mask_path=str(mask_path),
                primitive="necrosis_appearance",
                strength="mild",
                region_hint={"bbox_xyxy": [6, 6, 18, 18], "centroid_xy": [12, 12]},
                source_labels=("Tumor",),
                target_label="Necrosis",
                expected_direction="increase",
                expected_area_bucket=(0.04, 0.20),
                seed=1,
            )
            prompt = BenchmarkPrompt(
                sample_id=intent.sample_id,
                old_prompt="No necrosis is identified.",
                new_prompt="A focal necrotic area is present.",
                instruction="add focal necrosis",
            )

            def fake_execute(**kwargs):
                target = kwargs["old_mask"].copy()
                target[8:16, 8:16] = 3
                edit_result = PrimitiveEditResult(
                    target_mask=target,
                    change_region=target != kwargs["old_mask"],
                    changed_area_fraction=float(np.mean(target != kwargs["old_mask"])),
                    selected_pixels=int(np.count_nonzero(target != kwargs["old_mask"])),
                    warnings=(),
                    ops_log={},
                )
                attempt = LLMContourAttempt(
                    attempt_index=1,
                    status=STATUS_VALIDATED,
                    edit_result=edit_result,
                )
                return LLMContourAgentResult(
                    status=STATUS_VALIDATED,
                    source_mask=kwargs["old_mask"],
                    attempts=(attempt,),
                    final_attempt=attempt,
                    context={},
                    artifact_paths={},
                )

            repaired_diff = semantic_diff_for_intent(intent)
            with mock.patch(
                "phase3_mask_edit.benchmark.runner._resolve_semantic_diff",
                side_effect=[DEFAULT_SEMANTIC_DIFF, repaired_diff],
            ) as parser_mock, mock.patch(
                "phase3_mask_edit.benchmark.runner.execute_llm_contour_agent",
                side_effect=fake_execute,
            ):
                row = run_benchmark_sample(
                    intent,
                    prompt,
                    mode=PROMPT_MODE,
                    output_dir=root / "out",
                    prompt_parser="api",
                    parser_model="dummy",
                    contour_provider="api-text",
                    contour_model="dummy",
                    semantic_repair_attempts=2,
                )

        self.assertEqual(row["status"], "completed")
        self.assertEqual(parser_mock.call_count, 2)
        second_call = parser_mock.call_args_list[1].kwargs
        self.assertEqual(
            second_call["repair_feedback"]["status"],
            "planner_no_executable_intents",
        )
        self.assertIn(
            "standalone, non-comparative absolute states",
            second_call["repair_feedback"]["instruction"],
        )
        self.assertEqual(second_call["previous_semantic_diff"], DEFAULT_SEMANTIC_DIFF)
        self.assertEqual(row["semantic_attempt_count"], 2)
        self.assertTrue(row["semantic_replanned"])
        self.assertTrue(row["semantic_repair_success"])
        self.assertEqual(row["semantic_final_attempt_status"], "planned")

    def test_agentic_summary_reports_repair_success_and_cumulative_rates(self):
        summary = summarize_rows(
            [
                {
                    "sample_id": "first-pass",
                    "mode": "gt",
                    "status": "completed",
                    "attempt_count": 1,
                    "first_attempt_status": "validated",
                    "final_attempt_status": "validated",
                    "replanned": False,
                    "repair_success": False,
                    "cumulative_success_at_k": {"1": True, "2": True},
                },
                {
                    "sample_id": "repaired",
                    "mode": "gt",
                    "status": "completed",
                    "attempt_count": 2,
                    "first_attempt_status": "validation_failed",
                    "final_attempt_status": "validated",
                    "replanned": True,
                    "repair_success": True,
                    "cumulative_success_at_k": {"1": False, "2": True},
                },
            ]
        )["overall"]

        self.assertEqual(summary["contour_attempted"], 2)
        self.assertEqual(summary["mean_attempt_count"], 1.5)
        self.assertEqual(summary["first_attempt_success_rate"], 0.5)
        self.assertEqual(summary["replan_rate"], 0.5)
        self.assertEqual(summary["repair_attempted"], 1)
        self.assertEqual(summary["repair_success_rate"], 1.0)
        self.assertEqual(summary["final_contour_success_rate"], 1.0)
        self.assertEqual(summary["cumulative_success_at_k"], {"1": 0.5, "2": 1.0})

    def test_semantic_plan_fails_only_after_repair_attempts_are_exhausted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask_path = root / "mask.png"
            save_id_mask(np.ones((16, 16), dtype=np.int64), mask_path)
            intent = BenchmarkIntent(
                sample_id="semantic-terminal-failure",
                organ="breast",
                profile="BCSS",
                image_path=None,
                mask_path=str(mask_path),
                primitive="necrosis_appearance",
                strength="mild",
                region_hint={},
                source_labels=("Tumor",),
                target_label="Necrosis",
                expected_direction="increase",
                expected_area_bucket=(0.04, 0.20),
                seed=1,
            )
            prompt = BenchmarkPrompt(
                sample_id=intent.sample_id,
                old_prompt="No necrosis.",
                new_prompt="Focal necrosis.",
                instruction="add focal necrosis",
            )

            with mock.patch(
                "phase3_mask_edit.benchmark.runner._resolve_semantic_diff",
                return_value=DEFAULT_SEMANTIC_DIFF,
            ) as parser_mock, mock.patch(
                "phase3_mask_edit.benchmark.runner.execute_llm_contour_agent"
            ) as contour_mock:
                row = run_benchmark_sample(
                    intent,
                    prompt,
                    mode=PROMPT_MODE,
                    output_dir=root / "out",
                    prompt_parser="api",
                    parser_model="dummy",
                    semantic_repair_attempts=2,
                )

        self.assertEqual(row["status"], "failed")
        self.assertEqual(parser_mock.call_count, 3)
        contour_mock.assert_not_called()
        self.assertEqual(row["semantic_attempt_count"], 3)
        self.assertTrue(row["semantic_replanned"])
        self.assertFalse(row["semantic_repair_success"])
        self.assertEqual(
            row["semantic_terminal_failure_reason"],
            "planner_no_executable_intents",
        )
        self.assertEqual(row["failure_stage"], "semantic_planning")

    def test_existing_artifacts_backfill_agentic_metrics_without_api_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample_dir = Path(tmp) / "sample" / "prompt"
            agent_dir = sample_dir / "phase3_mask_edit"
            agent_dir.mkdir(parents=True)
            (agent_dir / "execution_summary.json").write_text(
                json.dumps(
                    {
                        "status": "validated",
                        "attempts": [
                            {"status": "validation_failed"},
                            {"status": "validated"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (sample_dir / "semantic_planning_trace.json").write_text(
                json.dumps(
                    {
                        "semantic_attempt_count": 2,
                        "semantic_first_attempt_status": (
                            "planner_no_executable_intents"
                        ),
                        "semantic_final_attempt_status": "planned",
                        "semantic_replanned": True,
                        "semantic_repair_success": True,
                        "semantic_terminal_failure_reason": "",
                    }
                ),
                encoding="utf-8",
            )

            row = _enrich_agentic_row_from_artifacts(
                {
                    "sample_id": "existing",
                    "mode": "prompt",
                    "status": "completed",
                    "output_dir": str(sample_dir),
                },
                max_attempts=3,
            )

        self.assertEqual(row["attempt_count"], 2)
        self.assertTrue(row["replanned"])
        self.assertTrue(row["repair_success"])
        self.assertEqual(
            row["cumulative_success_at_k"],
            {"1": False, "2": True, "3": True},
        )
        self.assertEqual(row["semantic_attempt_count"], 2)
        self.assertTrue(row["semantic_repair_success"])

    def test_prompt_mode_counts_missing_prompt_as_failed_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intent = BenchmarkIntent(
                sample_id="missing-prompt",
                organ="breast",
                profile="BCSS",
                image_path=None,
                mask_path="unused.png",
                primitive="necrosis_appearance",
                strength="mild",
                region_hint={},
                source_labels=("Tumor",),
                target_label="Necrosis",
                expected_direction="increase",
                expected_area_bucket=(0.08, 0.14),
                seed=1,
            )
            intents_path = write_intents_jsonl([intent], root / "intents.jsonl")
            prompts_path = write_prompts_csv([], root / "prompts.csv")
            output = root / "out"

            exit_code = run_benchmark_main(
                [
                    "--intents",
                    str(intents_path),
                    "--prompts",
                    str(prompts_path),
                    "--output",
                    str(output),
                    "--modes",
                    "prompt",
                    "--parser-model",
                    "dummy",
                    "--bootstrap-iterations",
                    "10",
                ]
            )
            rows = list(csv.DictReader((output / "benchmark_eval_results.csv").open()))

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "failed")
        self.assertEqual(rows[0]["error"], "prompt_missing")

    def test_prompt_generation_retries_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intent = BenchmarkIntent(
                sample_id="retry-prompt",
                organ="breast",
                profile="BCSS",
                image_path=None,
                mask_path="unused.png",
                primitive="necrosis_appearance",
                strength="mild",
                region_hint={"location": "center"},
                source_labels=("Tumor",),
                target_label="Necrosis",
                expected_direction="increase",
                expected_area_bucket=(0.08, 0.14),
                seed=1,
            )
            intents_path = write_intents_jsonl([intent], root / "intents.jsonl")
            output = root / "prompts"
            accepted = BenchmarkPrompt(
                sample_id=intent.sample_id,
                old_prompt="old",
                new_prompt="new",
                instruction="instruction",
            )

            with mock.patch(
                "phase3_mask_edit.cli.generate_mask_edit_benchmark_prompts.generate_prompts",
                side_effect=[RuntimeError("temporary"), [accepted]],
            ) as generate_mock, mock.patch(
                "phase3_mask_edit.cli.generate_mask_edit_benchmark_prompts.time.sleep"
            ):
                exit_code = generate_prompts_main(
                    [
                        "--intents",
                        str(intents_path),
                        "--output",
                        str(output),
                        "--checkpoint-every",
                        "1",
                        "--max-retries",
                        "2",
                    ]
                )

            prompts = read_prompts_csv(output / "benchmark_prompts.csv")

        self.assertEqual(exit_code, 0)
        self.assertEqual(generate_mock.call_count, 2)
        self.assertEqual(prompts[intent.sample_id].checker_status, "accepted")

    def test_prompt_generation_retries_only_existing_language_violations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intent = BenchmarkIntent(
                sample_id="language-retry",
                organ="breast",
                profile="BCSS",
                image_path=None,
                mask_path="unused.png",
                primitive="tumor_burden_increase",
                strength="mild",
                region_hint={"location": "center"},
                source_labels=("Stroma",),
                target_label="Tumor",
                expected_direction="increase",
                expected_area_bucket=(0.08, 0.14),
                seed=1,
            )
            intents_path = write_intents_jsonl([intent], root / "intents.jsonl")
            output = root / "prompts"
            output.mkdir()
            write_prompts_csv(
                [
                    BenchmarkPrompt(
                        sample_id=intent.sample_id,
                        old_prompt="The breast specimen contains sparse central tumor nests.",
                        new_prompt=(
                            "The tumor is prominent relative to the central stroma."
                        ),
                        instruction="Increase the central tumor burden.",
                        checker_status="accepted",
                    )
                ],
                output / "benchmark_prompts.csv",
            )
            repaired = BenchmarkPrompt(
                sample_id=intent.sample_id,
                old_prompt="The breast specimen contains sparse central tumor nests.",
                new_prompt="The breast specimen contains conspicuous central tumor nests.",
                instruction="Increase the central tumor burden.",
                checker_status="accepted",
            )

            with mock.patch(
                "phase3_mask_edit.cli.generate_mask_edit_benchmark_prompts.generate_prompts",
                return_value=[repaired],
            ) as generate_mock:
                exit_code = generate_prompts_main(
                    [
                        "--intents",
                        str(intents_path),
                        "--output",
                        str(output),
                        "--resume",
                        "--retry-language-violations",
                    ]
                )

            accepted = read_prompts_csv(output / "benchmark_prompts.accepted.csv")

        self.assertEqual(exit_code, 0)
        self.assertEqual(generate_mock.call_count, 1)
        self.assertIn(intent.sample_id, accepted)

    def test_prompt_generation_repairs_checker_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intent = BenchmarkIntent(
                sample_id="repair-prompt",
                organ="breast",
                profile="BCSS",
                image_path=None,
                mask_path="unused.png",
                primitive="stroma_decrease",
                strength="mild",
                region_hint={"location": "center"},
                source_labels=("Stroma",),
                target_label="Tumor",
                expected_direction="decrease",
                expected_area_bucket=(0.08, 0.14),
                seed=1,
            )
            intents_path = write_intents_jsonl([intent], root / "intents.jsonl")
            output = root / "prompts"
            rejected = BenchmarkPrompt(
                sample_id=intent.sample_id,
                old_prompt="dense stroma",
                new_prompt="less stroma",
                instruction="decrease stroma",
                checker_status="rejected",
                checker_reason="report_language_violation:'less'",
            )
            accepted = BenchmarkPrompt(
                sample_id=intent.sample_id,
                old_prompt="dense stroma",
                new_prompt="scant stroma",
                instruction="decrease stroma",
            )

            with mock.patch(
                "phase3_mask_edit.cli.generate_mask_edit_benchmark_prompts.generate_prompts",
                side_effect=[[rejected], [accepted]],
            ) as generate_mock:
                exit_code = generate_prompts_main(
                    [
                        "--intents",
                        str(intents_path),
                        "--output",
                        str(output),
                        "--use-llm-checker",
                        "--checker-repair-attempts",
                        "1",
                    ]
                )

            prompts = read_prompts_csv(output / "benchmark_prompts.csv")

        self.assertEqual(exit_code, 0)
        self.assertEqual(generate_mock.call_count, 2)
        self.assertEqual(
            generate_mock.call_args_list[1].kwargs["repair_feedback"],
            rejected.checker_reason,
        )
        self.assertEqual(
            generate_mock.call_args_list[1].kwargs["repair_prompt"],
            rejected,
        )
        self.assertEqual(prompts[intent.sample_id].checker_status, "accepted")

    def test_selects_exact_failed_rerun_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_path = write_eval_csv(
                [
                    {
                        "sample_id": "ok",
                        "status": "completed",
                        "primitive": "necrosis_appearance",
                        "all_ok": True,
                    },
                    {
                        "sample_id": "failed",
                        "status": "failed",
                        "primitive": "necrosis_appearance",
                        "all_ok": False,
                        "error": "proposal_failed",
                    },
                ],
                root / "eval.csv",
            )
            output = root / "failed.jsonl"

            exit_code = select_rerun_main(
                [
                    "--eval-results",
                    str(eval_path),
                    "--output",
                    str(output),
                    "--failed-only",
                ]
            )
            selected = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual(exit_code, 0)
        self.assertEqual([item["sample_id"] for item in selected], ["failed"])

    def test_run_resume_skips_completed_sample_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intent = BenchmarkIntent(
                sample_id="already-complete",
                organ="breast",
                profile="BCSS",
                image_path=None,
                mask_path="unused.png",
                primitive="necrosis_appearance",
                strength="mild",
                region_hint={},
                source_labels=("Tumor",),
                target_label="Necrosis",
                expected_direction="increase",
                expected_area_bucket=(0.08, 0.14),
                seed=1,
            )
            intents_path = write_intents_jsonl([intent], root / "intents.jsonl")
            output = root / "out"
            output.mkdir()
            write_eval_csv(
                [
                    {
                        "sample_id": intent.sample_id,
                        "mode": "gt",
                        "status": "completed",
                        "all_ok": True,
                    }
                ],
                output / "benchmark_eval_results.csv",
            )

            with mock.patch(
                "phase3_mask_edit.cli.run_mask_edit_benchmark.run_benchmark_sample"
            ) as runner_mock:
                exit_code = run_benchmark_main(
                    [
                        "--intents",
                        str(intents_path),
                        "--output",
                        str(output),
                        "--modes",
                        "gt",
                        "--resume",
                        "--bootstrap-iterations",
                        "10",
                    ]
                )

            rows = list(csv.DictReader((output / "benchmark_eval_results.csv").open()))

        self.assertEqual(exit_code, 0)
        self.assertFalse(runner_mock.called)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["primary_ok"], "true")
        self.assertEqual(
            rows[0]["strength_evaluation_policy"], "strict_intended_bucket"
        )

    def test_merge_replaces_only_matching_sample_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = write_eval_csv(
                [
                    {"sample_id": "a", "mode": "gt", "status": "failed"},
                    {"sample_id": "b", "mode": "gt", "status": "completed"},
                ],
                root / "base.csv",
            )
            rerun = write_eval_csv(
                [{"sample_id": "a", "mode": "gt", "status": "completed"}],
                root / "rerun.csv",
            )
            output = root / "merged"

            exit_code = merge_results_main(
                [
                    "--base",
                    str(base),
                    "--rerun",
                    str(rerun),
                    "--output",
                    str(output),
                    "--bootstrap-iterations",
                    "10",
                ]
            )
            rows = list(csv.DictReader((output / "benchmark_eval_results.csv").open()))
            manifest = json.loads((output / "merge_manifest.json").read_text())

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            {row["sample_id"]: row["status"] for row in rows},
            {"a": "completed", "b": "completed"},
        )
        self.assertEqual(manifest["replacements"], 1)
        self.assertEqual(manifest["additions"], 0)


if __name__ == "__main__":
    unittest.main()

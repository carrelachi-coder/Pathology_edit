import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np

from phase3_mask_edit.backends.fixture_contour import (
    STATUS_PROPOSAL_REJECTED,
    STATUS_VALIDATED,
    STATUS_VALIDATION_FAILED,
)
from phase3_mask_edit.backends.llm_agent import (
    STATUS_PROVIDER_ERROR,
    STATUS_PROPOSAL_FAILED,
    FakeSequenceContourProvider,
    FixtureContourProvider,
    OpenAICompatibleMultimodalContourProvider,
    OpenAICompatibleTextContourProvider,
    execute_llm_contour_agent,
)
from phase3_mask_edit.backends.llm_prompt import build_contour_prompt, build_mask_context
from phase3_mask_edit.cli.run_llm_contour_api import main as run_api_main
from phase3_mask_edit.core.config import load_recipe
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.core.mask_io import load_id_mask, load_metadata


WORKSPACE_TMP = Path(".tmp_phase3_llm_contour_agent_tests")


class LLMContourAgentTests(unittest.TestCase):
    def setUp(self):
        self.schema = MaskProfileSchema.from_reference_profile("BCSS")
        self.recipe = load_recipe("phase3_mask_edit/recipes/generic.yaml")
        self.mask = _synthetic_bcss_mask()
        self.intent = EditIntent(
            primitive="stromal_immune_infiltration",
            strength="mild",
            reference_profile="BCSS",
            source_labels=("Stroma",),
            target_label="Immune infiltrate",
        )
        self.primitive_config = _primitive(self.recipe, "stromal_immune_infiltration")

    def test_prompt_context_contains_coordinate_contract(self):
        context = build_mask_context(
            self.mask,
            schema=self.schema,
            intent=self.intent,
            primitive_config=self.primitive_config,
            allowed_source_labels=("Stroma",),
            target_label="Immune infiltrate",
        )

        prompt = build_contour_prompt(context=context)

        self.assertEqual(context["mask_shape"], [64, 64])
        self.assertEqual(context["preview"]["point_format"], "[x, y]")
        self.assertEqual(context["visual_label_legend"]["Stroma"], "green")
        self.assertEqual(
            context["target_area_hint"]["area_semantics"],
            "projection_after_legal_source_label_filtering",
        )
        self.assertIn("Stroma", context["source_spatial_hints"])
        self.assertGreater(
            len(context["source_spatial_hints"]["Stroma"]["high_purity_grid_tiles"]),
            0,
        )
        self.assertGreater(
            len(context["source_spatial_hints"]["Stroma"]["components"]),
            0,
        )
        self.assertIn("Stroma", context["source_contour_context"])
        stroma_contours = context["source_contour_context"]["Stroma"]["components"]
        self.assertGreater(len(stroma_contours), 0)
        self.assertGreater(len(stroma_contours[0]["contour_simplified"]), 0)
        self.assertIn("adjacent_tissue", stroma_contours[0])
        self.assertEqual(context["target_area_hint"]["target_changed_pixels_min"], 122)
        self.assertEqual(context["target_area_hint"]["target_changed_pixels_max"], 212)
        self.assertEqual(
            context["llm_task_requirements"]["pathology_goal"],
            "Increase stromal tumor-infiltrating lymphocytes around tumor.",
        )
        self.assertIn(
            "Prefer peritumoral Stroma near the red Tumor boundary.",
            context["llm_task_requirements"]["where_to_draw"],
        )
        self.assertIn("area_requirement", context["llm_task_requirements"])
        self.assertEqual(
            context["contour_style_hint"]["recommended_region_count_range"],
            [1, 2],
        )
        self.assertEqual(
            context["contour_style_hint"]["points_per_region_range"],
            [24, 48],
        )
        self.assertIn(
            "Do not duplicate",
            context["contour_style_hint"]["region_variation_requirement"],
        )
        self.assertIn("x is the horizontal column increasing right", prompt)
        self.assertIn("y is the vertical row increasing down", prompt)
        self.assertIn("after deterministic projection", prompt)
        self.assertIn("coarse organic template", prompt)
        self.assertIn('"template_role": "coarse_template"', prompt)
        self.assertIn('"source_component_ids"', prompt)
        self.assertIn('"placement_relation": "tumor_adjacent_stroma"', prompt)
        self.assertIn("source_spatial_hints", prompt)
        self.assertIn("only as location anchors", prompt)
        self.assertIn("natural, pathology-like irregular stromal immune coarse template", prompt)
        self.assertIn("Avoid rectangles, diamonds", prompt)
        self.assertIn("one large wedge-shaped band", prompt)
        self.assertIn("prefer multiple patchy organic contours", prompt)
        self.assertIn("do not add an identical copy", prompt)
        self.assertIn("Follow llm_task_requirements exactly", prompt)
        self.assertIn("Follow contour_style_hint", prompt)
        self.assertIn("never output identical diamonds", prompt)
        self.assertIn("Use source_contour_context as the primary placement reference", prompt)
        self.assertIn("adjacent tissue on the other side of the boundary", prompt)
        self.assertIn("Stroma (green)", prompt)
        self.assertIn("coarse template mainly over allowed source tissue", prompt)
        self.assertIn("does not need pixel-perfect vertices", prompt)
        self.assertNotIn("Every polygon vertex must lie on a pixel", prompt)
        self.assertIn("Consecutive points should be close together", prompt)
        self.assertIn("intended placement should be on the Stroma side", prompt)
        self.assertIn('"target_label": "Immune infiltrate"', prompt)

    def test_backfill_priority_supplies_target_label_when_intent_omits_it(self):
        primitive_config = _primitive(self.recipe, "necrosis_resolution")
        mask = np.array(self.mask, copy=True)
        mask[24:40, 24:40] = self.schema.resolve_fine_ids("Necrosis")[0]
        provider = FakeSequenceContourProvider(
            [
                {
                    "schema_version": "0.1",
                    "backend": "llm_contour_proposal",
                    "primitive": "necrosis_resolution",
                    "reference_profile": "BCSS",
                    "target_label": "Stroma",
                    "coordinate_system": {
                        "origin": "top_left",
                        "point_format": "[x, y]",
                        "x_axis": "horizontal_column_right",
                        "y_axis": "vertical_row_down",
                        "width": 64,
                        "height": 64,
                    },
                    "regions": [
                        {
                            "region_id": "necrosis-core",
                            "source_labels": ["Necrosis"],
                            "points": [[24, 24], [39, 24], [39, 39], [24, 39]],
                            "confidence": 0.9,
                        }
                    ],
                }
            ]
        )
        intent = EditIntent(
            primitive="necrosis_resolution",
            strength="mild",
            reference_profile="BCSS",
        )

        result = execute_llm_contour_agent(
            old_mask=mask,
            schema=self.schema,
            intent=intent,
            primitive_config=primitive_config,
            provider=provider,
            max_attempts=1,
        )

        self.assertNotIn("requires a target label", result.error or "")
        self.assertEqual(result.context["target_label"], "Stroma")

    def test_necrosis_resolution_falls_back_to_tumor_on_tumor_necrosis_only_mask(self):
        primitive_config = _primitive(self.recipe, "necrosis_resolution")
        mask = np.full((64, 64), self.schema.resolve_fine_ids("Tumor")[0], dtype=np.int64)
        mask[20:44, 20:44] = self.schema.resolve_fine_ids("Necrosis")[0]
        provider = FakeSequenceContourProvider(
            [
                {
                    "schema_version": "0.1",
                    "backend": "llm_contour_proposal",
                    "primitive": "necrosis_resolution",
                    "reference_profile": "BCSS",
                    "target_label": "Tumor",
                    "coordinate_system": {
                        "origin": "top_left",
                        "point_format": "[x, y]",
                        "x_axis": "horizontal_column_right",
                        "y_axis": "vertical_row_down",
                        "width": 64,
                        "height": 64,
                    },
                    "regions": [
                        {
                            "region_id": "necrosis-core",
                            "type": "polygon",
                            "source_labels": ["Necrosis"],
                            "points": [[20, 20], [31, 20], [31, 31], [20, 31]],
                            "confidence": 0.9,
                        }
                    ],
                }
            ]
        )
        intent = EditIntent(
            primitive="necrosis_resolution",
            strength="mild",
            reference_profile="BCSS",
        )

        result = execute_llm_contour_agent(
            old_mask=mask,
            schema=self.schema,
            intent=intent,
            primitive_config=primitive_config,
            provider=provider,
            max_attempts=1,
        )

        self.assertEqual(result.status, STATUS_VALIDATED)
        self.assertIsNotNone(result.edit_result)
        self.assertGreater(result.edit_result.selected_pixels, 0)
        self.assertTrue(np.all(mask[result.edit_result.change_region] == 3))
        self.assertTrue(np.all(result.edit_result.target_mask[result.edit_result.change_region] == 1))
        self.assertIn(
            "necrosis_resolution_fallback_backfill_to_tumor",
            result.edit_result.warnings,
        )
        self.assertEqual(result.edit_result.ops_log["backfill_labels"], ["Tumor"])

    def test_fixture_provider_runs_single_valid_attempt(self):
        provider = FixtureContourProvider(
            "tests/fixtures/llm_contour_stromal_immune_bcss.json"
        )

        result = execute_llm_contour_agent(
            old_mask=self.mask,
            schema=self.schema,
            intent=self.intent,
            primitive_config=self.primitive_config,
            provider=provider,
        )

        self.assertEqual(result.status, STATUS_VALIDATED)
        self.assertEqual(len(result.attempts), 1)
        self.assertEqual(result.attempts[0].status, STATUS_VALIDATED)
        self.assertIsNotNone(result.edit_result)
        _assert_final_diff_labels(
            self,
            old_mask=self.mask,
            target_mask=result.edit_result.target_mask,
            allowed_source_ids={2},
            target_id=4,
        )

    def test_stromal_desmoplasia_defaults_source_policy_from_recipe(self):
        primitive_config = _primitive(self.recipe, "stromal_desmoplasia")
        intent = EditIntent(
            primitive="stromal_desmoplasia",
            strength="mild",
            reference_profile="BCSS",
            target_label="Stroma",
        )
        mask = _synthetic_desmoplasia_mask()
        provider = FakeSequenceContourProvider(
            (
                {
                    "schema_version": "0.1",
                    "backend": "llm_contour_proposal",
                    "primitive": "stromal_desmoplasia",
                    "reference_profile": "BCSS",
                    "target_label": "Stroma",
                    "coordinate_system": {
                        "origin": "top_left",
                        "point_format": "[x, y]",
                        "x_axis": "horizontal_column_right",
                        "y_axis": "vertical_row_down",
                        "width": 260,
                        "height": 260,
                    },
                    "regions": [
                        {
                            "region_id": "r1",
                            "type": "polygon",
                            "source_labels": ["Other tissue", "Normal epithelium"],
                            "points": [[45, 45], [214, 45], [214, 214], [45, 214]],
                            "confidence": 0.8,
                        }
                    ],
                },
            )
        )

        result = execute_llm_contour_agent(
            old_mask=mask,
            schema=self.schema,
            intent=intent,
            primitive_config=primitive_config,
            provider=provider,
        )

        self.assertEqual(result.status, STATUS_VALIDATED)
        self.assertEqual(
            result.context["allowed_source_labels"],
            ["Other tissue", "Normal epithelium", "Immune infiltrate"],
        )
        self.assertEqual(
            result.context["llm_task_requirements"]["pathology_goal"],
            "Increase peritumoral desmoplastic stromal reaction.",
        )
        prompt = build_contour_prompt(context=result.context)
        self.assertIn('"placement_relation": "peritumoral_desmoplastic_stroma_expansion"', prompt)
        self.assertIn('"stromal_reinforcement"', prompt)
        self.assertEqual(
            result.context["primitive_policy"]["mask_operation"]["primary_sources"],
            ["Other tissue", "Normal epithelium"],
        )
        self.assertIsNotNone(result.edit_result)
        self.assertEqual(
            result.edit_result.ops_log["projection_mode"],
            "organic_v2",
        )
        self.assertEqual(
            result.edit_result.ops_log["component_policy"]["policy_name"],
            "stromal_desmoplasia_peritumoral_stroma_expansion",
        )
        _assert_final_diff_labels(
            self,
            old_mask=mask,
            target_mask=result.edit_result.target_mask,
            allowed_source_ids={4, 6, 7},
            target_id=2,
        )

    def test_fake_sequence_repairs_rejected_proposal_on_second_attempt(self):
        bad = _proposal(points=[[1, 1], [99, 1], [1, 8]])
        good = _proposal(points=[[2, 2], [17, 2], [17, 25], [2, 25]])
        provider = FakeSequenceContourProvider((bad, good))

        result = execute_llm_contour_agent(
            old_mask=self.mask,
            schema=self.schema,
            intent=self.intent,
            primitive_config=self.primitive_config,
            provider=provider,
            max_attempts=3,
        )

        self.assertEqual(result.status, STATUS_VALIDATED)
        self.assertEqual([a.status for a in result.attempts], [STATUS_PROPOSAL_REJECTED, STATUS_VALIDATED])
        feedback = result.attempts[0].repair_feedback
        self.assertIsNotNone(feedback)
        self.assertIn("outside mask bounds", feedback["error"])
        self.assertNotIn("repair_instruction", feedback)

    def test_fake_sequence_repairs_validation_failure_on_second_attempt(self):
        tiny = _proposal(points=[[1, 1], [6, 1], [6, 6], [1, 6]])
        good = _proposal(points=[[2, 2], [17, 2], [17, 25], [2, 25]])
        provider = FakeSequenceContourProvider((tiny, good))

        result = execute_llm_contour_agent(
            old_mask=self.mask,
            schema=self.schema,
            intent=self.intent,
            primitive_config=self.primitive_config,
            provider=provider,
            max_attempts=3,
        )

        self.assertEqual(result.status, STATUS_VALIDATED)
        self.assertEqual([a.status for a in result.attempts], [STATUS_VALIDATION_FAILED, STATUS_VALIDATED])
        feedback = result.attempts[0].repair_feedback
        self.assertIsNotNone(feedback)
        self.assertIn("failed_checks", feedback)
        failed_names = {check["name"] for check in feedback["failed_checks"]}
        self.assertIn("change_area_within_range", failed_names)
        self.assertIn("projection", feedback)
        self.assertNotIn("repair_instruction", feedback)

    def test_max_attempts_returns_proposal_failed_and_saves_artifacts(self):
        bad = _proposal(points=[[1, 1], [99, 1], [1, 8]])
        provider = FakeSequenceContourProvider((bad,))
        WORKSPACE_TMP.mkdir(exist_ok=True)
        tmp = WORKSPACE_TMP / f"agent_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True)
        try:
            result = execute_llm_contour_agent(
                old_mask=self.mask,
                schema=self.schema,
                intent=self.intent,
                primitive_config=self.primitive_config,
                provider=provider,
                output_dir=tmp,
                max_attempts=2,
            )

            self.assertEqual(result.status, STATUS_PROPOSAL_FAILED)
            self.assertEqual(len(result.attempts), 2)
            self.assertTrue((tmp / "mask_context.json").exists())
            self.assertTrue((tmp / "source_mask_llm_rgb_grid.png").exists())
            self.assertTrue((tmp / "attempt_001" / "repair_feedback.json").exists())
            self.assertTrue((tmp / "attempt_002" / "prompt.txt").exists())
            summary = load_metadata(tmp / "execution_summary.json")
            self.assertEqual(summary["status"], STATUS_PROPOSAL_FAILED)
            self.assertEqual(len(summary["attempts"]), 2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_message_length_error_retries_with_compact_context(self):
        class TooLongThenGoodProvider:
            name = "too_long_then_good"

            def __init__(self):
                self.context_modes = []

            def propose(self, request):
                self.context_modes.append(
                    request.provider_metadata.get("context_mode")
                )
                if request.attempt_index == 1:
                    raise RuntimeError(
                        "API request failed with HTTP 429: "
                        "message_length_exceeds_limit"
                    )
                return _proposal(points=[[2, 2], [17, 2], [17, 25], [2, 25]])

        provider = TooLongThenGoodProvider()
        WORKSPACE_TMP.mkdir(exist_ok=True)
        tmp = WORKSPACE_TMP / f"agent_compact_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True)
        try:
            result = execute_llm_contour_agent(
                old_mask=self.mask,
                schema=self.schema,
                intent=self.intent,
                primitive_config=self.primitive_config,
                provider=provider,
                output_dir=tmp,
                max_attempts=3,
            )

            self.assertEqual(result.status, STATUS_VALIDATED)
            self.assertEqual(
                [attempt.status for attempt in result.attempts],
                [STATUS_PROVIDER_ERROR, STATUS_VALIDATED],
            )
            self.assertEqual(provider.context_modes, ["full", "compact"])
            self.assertTrue((tmp / "mask_context_compact.json").exists())
            first_request = load_metadata(tmp / "attempt_001" / "llm_request.json")
            second_request = load_metadata(tmp / "attempt_002" / "llm_request.json")
            self.assertEqual(
                first_request["provider_metadata"]["context_mode"], "full"
            )
            self.assertEqual(
                second_request["provider_metadata"]["context_mode"], "compact"
            )
            self.assertTrue(
                second_request["provider_metadata"]["compact_context_enabled"]
            )
            compact = load_metadata(tmp / "mask_context_compact.json")
            self.assertTrue(compact["context_compression"]["enabled"])
            self.assertEqual(
                result.attempts[0].repair_feedback["context_mode_next_attempt"],
                "compact",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_validated_run_saves_final_outputs(self):
        provider = FixtureContourProvider(
            "tests/fixtures/llm_contour_stromal_immune_bcss.json"
        )
        WORKSPACE_TMP.mkdir(exist_ok=True)
        tmp = WORKSPACE_TMP / f"agent_valid_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True)
        try:
            result = execute_llm_contour_agent(
                old_mask=self.mask,
                schema=self.schema,
                intent=self.intent,
                primitive_config=self.primitive_config,
                provider=provider,
                output_dir=tmp,
            )

            self.assertEqual(result.status, STATUS_VALIDATED)
            self.assertTrue((tmp / "final_target_mask.png").exists())
            self.assertTrue((tmp / "final_change_region.png").exists())
            self.assertTrue((tmp / "attempt_001" / "llm_request.json").exists())
            request = load_metadata(tmp / "attempt_001" / "llm_request.json")
            self.assertGreaterEqual(len(request["image_paths"]), 1)
            self.assertEqual(request["provider_metadata"]["request_mode"], "text")
            self.assertEqual(request["provider_metadata"]["image_parts_expected"], 0)
            target_mask = load_id_mask(tmp / "final_target_mask.png")
            _assert_final_diff_labels(
                self,
                old_mask=self.mask,
                target_mask=target_mask,
                allowed_source_ids={2},
                target_id=4,
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_openai_compatible_text_provider_parses_chat_completion(self):
        provider = OpenAICompatibleTextContourProvider(
            model="gpt-4o",
            api_base_url="https://relay.example/v1",
            api_key_env="TEST_CONTOUR_API_KEY",
        )
        captured = {}

        def fake_post(payload, *, api_base_url, api_key, timeout_sec):
            captured["payload"] = payload
            captured["api_base_url"] = api_base_url
            captured["api_key"] = api_key
            return {
                "choices": [
                    {
                        "message": {
                            "content": json_dumps(_proposal(points=[[2, 2], [17, 2], [17, 25], [2, 25]]))
                        }
                    }
                ]
            }

        import phase3_mask_edit.backends.llm_agent as llm_agent
        old_post = llm_agent._post_chat_completion
        import os
        os.environ["TEST_CONTOUR_API_KEY"] = "secret"
        try:
            llm_agent._post_chat_completion = fake_post
            context = build_mask_context(
                self.mask,
                schema=self.schema,
                intent=self.intent,
                primitive_config=self.primitive_config,
                allowed_source_labels=("Stroma",),
                target_label="Immune infiltrate",
            )
            request = llm_agent.ContourProposalRequest(
                prompt=build_contour_prompt(context=context),
                context=context,
                attempt_index=1,
                image_paths=("unused-grid.png",),
                provider_metadata={"request_mode": "text"},
            )

            payload = provider.propose(request)

            self.assertEqual(payload["backend"], "llm_contour_proposal")
            self.assertEqual(captured["payload"]["model"], "gpt-4o")
            self.assertEqual(captured["api_base_url"], "https://relay.example/v1")
            self.assertEqual(captured["api_key"], "secret")
            self.assertEqual(captured["payload"]["response_format"], {"type": "json_object"})
        finally:
            llm_agent._post_chat_completion = old_post
            os.environ.pop("TEST_CONTOUR_API_KEY", None)

    def test_openai_compatible_text_provider_accepts_fenced_json_content(self):
        provider = OpenAICompatibleTextContourProvider(
            model="gpt-4o",
            api_base_url="https://relay.example/v1",
            api_key_env="TEST_CONTOUR_API_KEY",
        )
        proposal = json_dumps(_proposal(points=[[2, 2], [17, 2], [17, 25], [2, 25]]))

        def fake_post(payload, *, api_base_url, api_key, timeout_sec):
            return {
                "choices": [
                    {
                        "message": {
                            "content": f"```json\n{proposal}\n```"
                        }
                    }
                ]
            }

        import phase3_mask_edit.backends.llm_agent as llm_agent
        old_post = llm_agent._post_chat_completion
        import os
        os.environ["TEST_CONTOUR_API_KEY"] = "secret"
        try:
            llm_agent._post_chat_completion = fake_post
            context = build_mask_context(
                self.mask,
                schema=self.schema,
                intent=self.intent,
                primitive_config=self.primitive_config,
                allowed_source_labels=("Stroma",),
                target_label="Immune infiltrate",
            )
            request = llm_agent.ContourProposalRequest(
                prompt=build_contour_prompt(context=context),
                context=context,
                attempt_index=1,
            )

            payload = provider.propose(request)

            self.assertEqual(payload["backend"], "llm_contour_proposal")
            self.assertEqual(payload["regions"][0]["region_id"], "r1")
        finally:
            llm_agent._post_chat_completion = old_post
            os.environ.pop("TEST_CONTOUR_API_KEY", None)

    def test_openai_compatible_multimodal_provider_sends_one_grid_image(self):
        provider = OpenAICompatibleMultimodalContourProvider(
            model="gpt-4o",
            api_base_url="https://relay.example/v1",
            api_key_env="TEST_CONTOUR_API_KEY",
            image_detail="high",
        )
        captured = {}
        WORKSPACE_TMP.mkdir(exist_ok=True)
        tmp = WORKSPACE_TMP / f"mm_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True)
        grid_path = tmp / "grid.png"
        plain_path = tmp / "plain.png"
        grid_path.write_bytes(b"grid-bytes")
        plain_path.write_bytes(b"plain-bytes")

        def fake_post(payload, *, api_base_url, api_key, timeout_sec):
            captured["payload"] = payload
            captured["api_base_url"] = api_base_url
            captured["api_key"] = api_key
            return {
                "choices": [
                    {
                        "message": {
                            "content": json_dumps(_proposal(points=[[2, 2], [17, 2], [17, 25], [2, 25]]))
                        }
                    }
                ]
            }

        import phase3_mask_edit.backends.llm_agent as llm_agent
        old_post = llm_agent._post_chat_completion
        import os
        os.environ["TEST_CONTOUR_API_KEY"] = "secret"
        try:
            llm_agent._post_chat_completion = fake_post
            context = build_mask_context(
                self.mask,
                schema=self.schema,
                intent=self.intent,
                primitive_config=self.primitive_config,
                allowed_source_labels=("Stroma",),
                target_label="Immune infiltrate",
            )
            request = llm_agent.ContourProposalRequest(
                prompt=build_contour_prompt(context=context),
                context=context,
                attempt_index=1,
                image_paths=(str(grid_path), str(plain_path)),
                provider_metadata={"request_mode": "multimodal"},
            )

            payload = provider.propose(request)

            self.assertEqual(payload["backend"], "llm_contour_proposal")
            user_content = captured["payload"]["messages"][1]["content"]
            image_parts = [part for part in user_content if part["type"] == "image_url"]
            self.assertEqual(len(image_parts), 1)
            self.assertTrue(
                image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
            )
            self.assertEqual(image_parts[0]["image_url"]["detail"], "high")
            self.assertNotIn("plain-bytes", captured["payload"]["messages"][1]["content"][0]["text"])
        finally:
            llm_agent._post_chat_completion = old_post
            os.environ.pop("TEST_CONTOUR_API_KEY", None)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_api_cli_fixture_mode_runs_agent_entrypoint(self):
        WORKSPACE_TMP.mkdir(exist_ok=True)
        tmp = WORKSPACE_TMP / f"api_cli_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True)
        try:
            from phase3_mask_edit.core.mask_io import save_id_mask

            mask_path = tmp / "source_mask.png"
            output_dir = tmp / "run"
            save_id_mask(self.mask, mask_path)

            code = run_api_main(
                [
                    "--profile",
                    "BCSS",
                    "--primitive",
                    "stromal_immune_infiltration",
                    "--strength",
                    "mild",
                    "--mask",
                    str(mask_path),
                    "--output",
                    str(output_dir),
                    "--provider",
                    "fixture",
                    "--fixture",
                    "tests/fixtures/llm_contour_stromal_immune_bcss.json",
                ]
            )

            self.assertEqual(code, 0)
            self.assertTrue((output_dir / "execution_summary.json").exists())
            self.assertTrue((output_dir / "final_target_mask.png").exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_source_contour_context_groups_segments_by_adjacent_tissue(self):
        mask = np.zeros((48, 48), dtype=np.int64)
        mask[:, :16] = 1
        mask[:, 16:32] = 2
        mask[:, 32:] = 3
        context = build_mask_context(
            mask,
            schema=self.schema,
            intent=self.intent,
            primitive_config=self.primitive_config,
            allowed_source_labels=("Stroma",),
            target_label="Immune infiltrate",
            grid_spacing_px=16,
        )

        component = context["source_contour_context"]["Stroma"]["components"][0]
        segments = component["contour_adjacency_segments"]

        self.assertIn("Tumor", segments)
        self.assertIn("Necrosis", segments)
        self.assertGreater(len(segments["Tumor"]), 0)
        self.assertGreater(len(segments["Necrosis"]), 0)
        self.assertIn("points", segments["Tumor"][0])
        prompt = build_contour_prompt(context=context)
        self.assertIn("contour_adjacency_segments groups contour coordinates", prompt)
        self.assertIn("prefer Stroma contour segments adjacent to Tumor", prompt)

    def test_patch_edge_is_not_reported_as_background_adjacency_segment(self):
        mask = np.zeros((48, 48), dtype=np.int64)
        mask[:, :24] = 2
        mask[:, 24:] = 1
        context = build_mask_context(
            mask,
            schema=self.schema,
            intent=self.intent,
            primitive_config=self.primitive_config,
            allowed_source_labels=("Stroma",),
            target_label="Immune infiltrate",
            grid_spacing_px=16,
        )

        component = context["source_contour_context"]["Stroma"]["components"][0]
        segments = component["contour_adjacency_segments"]

        self.assertIn("Tumor", segments)
        self.assertNotIn("Background", segments)

    def test_fine_transition_organic_projection_relabels_whole_source_components(self):
        schema = MaskProfileSchema.from_reference_profile("GlaS")
        recipe = load_recipe("phase3_mask_edit/recipes/glas.yaml")
        primitive_config = _primitive(recipe, "adenoma_to_carcinoma")
        mask = np.zeros((40, 40), dtype=np.int64)
        mask[4:14, 4:14] = 11
        mask[20:36, 20:36] = 11
        provider = FakeSequenceContourProvider(
            [
                {
                    "schema_version": "0.1",
                    "backend": "llm_contour_proposal",
                    "primitive": "adenoma_to_carcinoma",
                    "reference_profile": "GlaS",
                    "target_label": "Tumor",
                    "coordinate_system": {
                        "origin": "top_left",
                        "point_format": "[x, y]",
                        "x_axis": "horizontal_column_right",
                        "y_axis": "vertical_row_down",
                        "width": 40,
                        "height": 40,
                    },
                    "regions": [
                        {
                            "region_id": "adenoma-template",
                            "type": "polygon",
                            "source_labels": ["Tumor"],
                            "points": [[20, 20], [35, 20], [35, 35], [20, 35]],
                            "confidence": 0.9,
                        }
                    ],
                }
            ]
        )
        intent = EditIntent(
            primitive="adenoma_to_carcinoma",
            strength="moderate",
            reference_profile="GlaS",
            source_labels=("Tumor",),
            target_label="Tumor",
        )

        result = execute_llm_contour_agent(
            old_mask=mask,
            schema=schema,
            intent=intent,
            primitive_config=primitive_config,
            provider=provider,
            max_attempts=1,
        )

        self.assertEqual(result.status, STATUS_VALIDATED)
        self.assertIsNotNone(result.edit_result)
        selected_pixels = int(result.edit_result.selected_pixels)
        self.assertEqual(selected_pixels, 100)
        self.assertEqual(np.count_nonzero(result.edit_result.target_mask == 12), 100)
        self.assertEqual(np.count_nonzero(result.edit_result.target_mask == 11), 256)
        self.assertEqual(
            result.edit_result.ops_log["selection_policy"],
            "whole_source_components_template_prioritized",
        )
        self.assertEqual(result.edit_result.ops_log["selection_unit"], "connected_component")
        self.assertEqual(result.edit_result.ops_log["selected_component_areas"], [100])


def _synthetic_bcss_mask() -> np.ndarray:
    mask = np.zeros((64, 64), dtype=np.int64)
    mask[8:56, 8:56] = 2
    mask[18:46, 18:46] = 1
    return mask


def _synthetic_desmoplasia_mask() -> np.ndarray:
    mask = np.zeros((260, 260), dtype=np.int64)
    mask[20:240, 20:240] = 2
    mask[100:160, 100:160] = 1
    mask[50:100, 50:210] = 7
    mask[160:210, 50:210] = 7
    mask[100:160, 50:100] = 6
    mask[100:160, 160:210] = 6
    mask[86:102, 118:142] = 4
    return mask


def _primitive(recipe, name):
    for primitive in recipe["primitives"]:
        if primitive["name"] == name:
            return primitive
    raise AssertionError(f"missing primitive {name}")


def _proposal(*, points):
    return {
        "schema_version": "0.1",
        "backend": "llm_contour_proposal",
        "primitive": "stromal_immune_infiltration",
        "reference_profile": "BCSS",
        "target_label": "Immune infiltrate",
        "coordinate_system": {
            "origin": "top_left",
            "point_format": "[x, y]",
            "x_axis": "horizontal_column_right",
            "y_axis": "vertical_row_down",
            "width": 64,
            "height": 64,
        },
        "regions": [
            {
                "region_id": "r1",
                "type": "polygon",
                "source_labels": ["Stroma"],
                "points": points,
                "confidence": 0.8,
            }
        ],
    }


def json_dumps(value):
    import json

    return json.dumps(value, ensure_ascii=False)


def _assert_final_diff_labels(
    case,
    *,
    old_mask: np.ndarray,
    target_mask: np.ndarray,
    allowed_source_ids: set[int],
    target_id: int,
) -> None:
    diff = old_mask != target_mask
    case.assertGreater(int(np.count_nonzero(diff)), 0)
    changed_old_labels = set(np.unique(old_mask[diff]).astype(int).tolist())
    changed_new_labels = set(np.unique(target_mask[diff]).astype(int).tolist())
    case.assertLessEqual(changed_old_labels, allowed_source_ids)
    case.assertEqual(changed_new_labels, {target_id})


if __name__ == "__main__":
    unittest.main()

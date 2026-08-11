"""Contracts for direct-edit and clinical-scenario instruction parsing."""

from __future__ import annotations

import json
import unittest

from phase3_joint_edit_refine.models import JointContractError
from phase3_joint_edit_refine.semantic_parser import (
    OpenAIClinicalScenarioParser,
    PreboundSemanticParser,
    RuleBasedSemanticParser,
)


class _FixtureJSONClient:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[dict] = []

    def call(self, **kwargs):
        self.calls.append(kwargs)
        return dict(self.response), {
            "model": "fixture-text-model",
            "prompt_hash": "fixture-prompt-hash",
        }


def _response(**updates) -> dict:
    value = {
        "abstain": False,
        "abstain_reason": None,
        "instruction_mode": "clinical_scenario",
        "scenario": "disease_progression",
        "clinical_direction": "worsen",
        "treatment_context": "unspecified",
        "target": "tumor",
        "explicit_edit_scope": "unspecified",
        "primitive_id": None,
        "subject": None,
        "edit_direction": None,
        "explicit_cell_class": None,
        "explicit_location": None,
        "user_constraints": [],
        "uncertainties": [],
    }
    value.update(updates)
    return value


class ClinicalScenarioParserTests(unittest.TestCase):
    def test_llm_scenario_does_not_own_primitive_selection(self):
        client = _FixtureJSONClient(_response())
        intent = OpenAIClinicalScenarioParser(client).parse(
            "模拟这个肿瘤继续进展"
        )

        self.assertEqual(intent.scenario, "disease_progression")
        self.assertEqual(intent.direction, "worsen")
        self.assertEqual(
            [item.primitive_id for item in intent.primitive_hypotheses],
            [
                "tumor-burden-increase-v1",
                "invasive-front-expansion-v1",
                "neoplastic-microinfiltration-increase-v1",
                "cellularity-increase-v1",
            ],
        )
        payload = json.loads(client.calls[0]["user_prompt"])
        self.assertIn("clinical_scenario", payload["closed_ontology"]["instruction_modes"])
        self.assertGreaterEqual(len(payload["few_shot_examples"]), 5)
        self.assertEqual(client.calls[0]["image_paths"], ())

    def test_direct_engineering_instruction_keeps_explicit_scope(self):
        client = _FixtureJSONClient(
            _response(
                instruction_mode="direct_edit",
                scenario="direct_edit",
                clinical_direction="unspecified",
                treatment_context="none",
                target="tumor",
                explicit_edit_scope="tissue_burden",
                primitive_id="tumor-burden-increase-v1",
                subject="tumor-burden",
                edit_direction="increase",
            )
        )
        intent = OpenAIClinicalScenarioParser(client).parse(
            "Increase tumor area"
        )

        self.assertEqual(intent.instruction_mode, "direct_edit")
        self.assertEqual(intent.explicit_edit_scope, "tissue_burden")
        self.assertEqual(len(intent.primitive_hypotheses), 1)

    def test_post_treatment_progression_preserves_worsening_direction(self):
        client = _FixtureJSONClient(
            _response(
                scenario="post_treatment_progression",
                treatment_context="post_treatment",
            )
        )
        intent = OpenAIClinicalScenarioParser(client).parse(
            "模拟治疗后肿瘤仍然进展"
        )

        self.assertEqual(intent.clinical_direction, "worsen")
        self.assertEqual(intent.treatment_context, "post_treatment")
        self.assertEqual(
            intent.primitive_hypotheses[0].primitive_id,
            "tumor-burden-increase-v1",
        )

    def test_clinical_scenario_cannot_smuggle_a_primitive(self):
        client = _FixtureJSONClient(
            _response(
                primitive_id="tumor-burden-increase-v1",
                subject="tumor-burden",
                edit_direction="increase",
            )
        )
        with self.assertRaisesRegex(JointContractError, "must not select"):
            OpenAIClinicalScenarioParser(client).parse("模拟肿瘤继续进展")

    def test_directionless_post_treatment_change_requires_rewrite(self):
        client = _FixtureJSONClient(
            _response(
                scenario="post_treatment_change",
                clinical_direction="unspecified",
                treatment_context="post_treatment",
            )
        )
        with self.assertRaisesRegex(JointContractError, "rewrite is required"):
            OpenAIClinicalScenarioParser(client).parse("模拟治疗后的变化")

    def test_offline_parser_exercises_same_scenario_lattice(self):
        intent = RuleBasedSemanticParser().parse("模拟治疗后肿瘤缩小")

        self.assertEqual(intent.scenario, "treatment_response")
        self.assertEqual(intent.explicit_edit_scope, "tissue_burden")
        self.assertEqual(
            [item.primitive_id for item in intent.primitive_hypotheses],
            ["tumor-burden-decrease-v1"],
        )

    def test_offline_parser_understands_plain_tumor_infiltration(self):
        intent = RuleBasedSemanticParser().parse(
            "increase tumor infiltration"
        )

        self.assertEqual(
            intent.primitive_id,
            "neoplastic-microinfiltration-increase-v1",
        )
        self.assertEqual(intent.explicit_edit_scope, "unspecified")
        self.assertEqual(
            [item.primitive_id for item in intent.primitive_hypotheses],
            [
                "neoplastic-microinfiltration-increase-v1",
                "invasive-front-expansion-v1",
            ],
        )

    def test_parser_keeps_front_void_and_architecture_scales_separate(self):
        parser = RuleBasedSemanticParser()
        self.assertEqual(
            parser.parse("expand the invasive front").primitive_id,
            "invasive-front-expansion-v1",
        )
        self.assertEqual(
            parser.parse("increase STAS").primitive_id,
            "structural-void-spread-v1",
        )
        self.assertEqual(
            parser.parse("progress the Gleason architectural pattern").primitive_id,
            "architecture-progression-v1",
        )

    def test_prebound_codex_intent_cannot_be_reparsed_or_detached(self):
        payload = RuleBasedSemanticParser().parse(
            "increase tumor infiltration"
        ).to_metadata()
        payload["parser"] = "current_codex_session_semantic_parser_v1"
        payload["parser_metadata"] = {
            "reviewer": "current_codex_session",
            "llm_api_used": False,
        }
        parser = PreboundSemanticParser(payload)

        self.assertEqual(
            parser.parse("increase tumor infiltration").parser,
            "current_codex_session_semantic_parser_v1",
        )
        with self.assertRaisesRegex(JointContractError, "detached"):
            parser.parse("increase tumor burden")


if __name__ == "__main__":
    unittest.main()

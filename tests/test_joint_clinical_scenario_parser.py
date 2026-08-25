"""Contracts for direct-edit and clinical-scenario instruction parsing."""

from __future__ import annotations

import json
import unittest

from phase3_joint_edit_refine.models import JointContractError
from phase3_joint_edit_refine.semantic_parser import (
    OpenAIClinicalScenarioParser,
    PreboundSemanticParser,
    RuleBasedSemanticParser,
    bind_semantic_intent,
    semantic_intent_from_metadata,
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
                "cohesive-boundary-expansion-v1",
                "peritumoral-neoplastic-scatter-increase-v1",
                "neoplastic-cell-abundance-increase-v1",
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

    def test_directionless_post_treatment_change_is_deferred_to_preflight_choice(self):
        client = _FixtureJSONClient(
            _response(
                scenario="post_treatment_change",
                clinical_direction="unspecified",
                treatment_context="post_treatment",
            )
        )
        intent = OpenAIClinicalScenarioParser(client).parse("模拟治疗后的变化")
        self.assertEqual(intent.scenario, "post_treatment_change")
        self.assertIn(
            "later deterministic representability preflight",
            client.calls[0]["system_prompt"],
        )
        directionless = next(
            item
            for item in json.loads(client.calls[0]["user_prompt"])[
                "few_shot_examples"
            ]
            if item["instruction"] == "Simulate a post-treatment change."
        )
        self.assertFalse(directionless["output"]["abstain"])
        self.assertIn(
            "residual-tumor-fragmentation-v1",
            [item.primitive_id for item in intent.primitive_hypotheses],
        )
        priorities = [item.priority for item in intent.primitive_hypotheses]
        self.assertEqual(priorities, list(range(len(priorities))))
        frozen = semantic_intent_from_metadata(intent.to_metadata())
        self.assertEqual(
            [item.scenario for item in frozen.primitive_hypotheses],
            [item.scenario for item in intent.primitive_hypotheses],
        )

    def test_offline_parser_exercises_same_scenario_lattice(self):
        intent = RuleBasedSemanticParser().parse("模拟治疗后肿瘤缩小")

        self.assertEqual(intent.scenario, "treatment_response")
        self.assertEqual(intent.explicit_edit_scope, "tissue_burden")
        self.assertEqual(
            [item.primitive_id for item in intent.primitive_hypotheses],
            ["invasive-tumor-footprint-decrease-v1"],
        )

    def test_directionless_post_treatment_language_binds_for_preflight_choice(self):
        raw = {
            "case_id": "scenario-clarify",
            "instruction": "模拟治疗后的变化",
            "source_image_uri": "/tmp/image.png",
            "source_tissue_mask_uri": "/tmp/tissue.png",
            "source_nuclei_mask_uri": "/tmp/nuclei.png",
            "pathology_domain_id": "prostate-adenocarcinoma-v1",
            "annotation_profile_id": "panda-gleason-v1",
            "cell_observation_profile_id": "cellvit-five-class-v1",
            "cell_population_profile_id": "prostate-cell-population-v1",
            "seed": 42,
            "provenance": {
                "source_image_sha256": "image-a",
                "source_tissue_mask_sha256": "tissue-a",
                "source_nuclei_mask_sha256": "nuclei-a",
            },
        }
        case, intent = bind_semantic_intent(raw, RuleBasedSemanticParser())
        self.assertEqual(intent.scenario, "post_treatment_change")
        self.assertEqual(intent.treatment_context, "post_treatment")
        self.assertFalse(case.clarification_decision)

    def test_offline_parser_understands_plain_tumor_infiltration(self):
        intent = RuleBasedSemanticParser().parse(
            "increase tumor infiltration"
        )
        self.assertEqual(
            intent.primitive_id,
            "peritumoral-neoplastic-scatter-increase-v1",
        )
        self.assertEqual(intent.explicit_edit_scope, "unspecified")
        self.assertEqual(
            [item.primitive_id for item in intent.primitive_hypotheses],
            [
                "peritumoral-neoplastic-scatter-increase-v1",
                "invasive-cord-formation-v1",
                "peritumoral-tumor-nest-formation-v1",
            ],
        )

    def test_panda_retires_generic_burden_alias_but_keeps_language_compatibility(self):
        raw = {
            "case_id": "panda-retired-burden",
            "instruction": "increase tumor burden",
            "primitive_id": "tumor-burden-increase-v1",
            "source_image_uri": "/tmp/image.png",
            "source_tissue_mask_uri": "/tmp/tissue.png",
            "source_nuclei_mask_uri": "/tmp/nuclei.png",
            "pathology_domain_id": "prostate-adenocarcinoma-v1",
            "annotation_profile_id": "panda-gleason-v1",
            "cell_observation_profile_id": "cellvit-five-class-v1",
            "cell_population_profile_id": "prostate-cell-population-v1",
            "seed": 42,
            "provenance": {
                "source_image_sha256": "image-digest",
                "source_tissue_mask_sha256": "tissue-digest",
                "source_nuclei_mask_sha256": "nuclei-digest",
            },
        }

        case, intent = bind_semantic_intent(raw, RuleBasedSemanticParser())

        self.assertEqual(case.primitive_id, "cohesive-boundary-expansion-v1")
        self.assertEqual(intent.primitive_id, "cohesive-boundary-expansion-v1")
        self.assertNotIn(
            "tumor-burden-increase-v1",
            [item.primitive_id for item in intent.primitive_hypotheses],
        )
        self.assertEqual(
            case.provenance["retired_primitive_alias"],
            "tumor-burden-increase-v1",
        )

    def test_breast_retires_generic_burden_alias_but_keeps_language_compatibility(self):
        raw = {
            "case_id": "breast-retired-burden",
            "instruction": "increase tumor burden",
            "primitive_id": "tumor-burden-increase-v1",
            "source_image_uri": "/tmp/image.png",
            "source_tissue_mask_uri": "/tmp/tissue.png",
            "source_nuclei_mask_uri": "/tmp/nuclei.png",
            "pathology_domain_id": "breast-invasive-carcinoma-v1",
            "annotation_profile_id": "bcss-semantic-v1",
            "cell_observation_profile_id": "cellvit-five-class-v1",
            "cell_population_profile_id": "breast-cellvit-source-first-v1",
            "seed": 42,
            "provenance": {
                "source_image_sha256": "image-digest",
                "source_tissue_mask_sha256": "tissue-digest",
                "source_nuclei_mask_sha256": "nuclei-digest",
            },
        }

        case, intent = bind_semantic_intent(raw, RuleBasedSemanticParser())

        self.assertEqual(case.primitive_id, "cohesive-boundary-expansion-v1")
        self.assertEqual(intent.primitive_id, "cohesive-boundary-expansion-v1")
        self.assertEqual(
            case.provenance["retired_primitive_alias"],
            "tumor-burden-increase-v1",
        )

    def test_infiltration_rules_do_not_collide_with_abundance(self):
        parser = RuleBasedSemanticParser()
        for instruction in (
            "increase tumor cell infiltration",
            "增加肿瘤细胞浸润",
        ):
            intent = parser.parse(instruction)
            self.assertEqual(
                intent.primitive_id,
                "peritumoral-neoplastic-scatter-increase-v1",
            )

    def test_residual_neoplastic_cell_reduction_keeps_cell_scope(self):
        parser = RuleBasedSemanticParser()
        for instruction in (
            "Reduce residual tumor cells after treatment",
            "减少治疗后残余肿瘤细胞",
        ):
            intent = parser.parse(instruction)
            self.assertEqual(intent.scenario, "residual_disease")
            self.assertEqual(intent.explicit_edit_scope, "cell_population")
            self.assertEqual(
                intent.primitive_id,
                "neoplastic-cell-abundance-decrease-v1",
            )

    def test_connective_cell_abundance_is_an_explicit_cell_type_edit(self):
        parser = RuleBasedSemanticParser()
        for instruction, primitive in (
            (
                "Increase connective tissue cells in the selected region.",
                "cell-type-abundance-increase-v1",
            ),
            (
                "Decrease fibroblasts in the selected region.",
                "cell-type-abundance-decrease-v1",
            ),
        ):
            intent = parser.parse(instruction)
            self.assertEqual(intent.primitive_id, primitive)
            self.assertEqual(intent.explicit_cell_class, "connective")

    def test_parser_keeps_front_void_and_architecture_scales_separate(self):
        parser = RuleBasedSemanticParser()
        self.assertEqual(
            parser.parse("expand the invasive front").primitive_id,
            "cohesive-boundary-expansion-v1",
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

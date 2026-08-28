"""Regression tests for natural-language benchmark semantic guards."""

from __future__ import annotations

import pytest

from scripts.generate_natural_semantic_parser_benchmark import _validate_generated


def _record(*, category="underspecified_intent", strength="unspecified", relations=()):
    return {
        "case_id": "spp-v1-9001",
        "language": "zh",
        "category": category,
        "gold_semantic_request": {
            "intents": [
                {
                    "polarity": "affirmed",
                    "strength": strength,
                    "clinical_context": "none",
                    "morphology": "unspecified",
                }
            ],
            "relations": list(relations),
        },
    }


def _validate(record, instruction):
    return _validate_generated(
        [(0, record)],
        {"cases": [{"case_id": record["case_id"], "instruction": instruction}]},
    )


def test_generator_guard_rejects_affirmed_edit_inverted_to_negation():
    record = _record()

    with pytest.raises(ValueError, match="inverted an affirmed edit"):
        _validate(record, "请把局部浸润保持原样，不要往增加的方向改。")


def test_generator_guard_rejects_sequence_for_unordered_intents():
    record = _record(
        category="unordered_conflict",
        relations=(
            {
                "before_intent_id": "intent-001",
                "after_intent_id": "intent-002",
                "relation_type": "unordered",
            },
        ),
    )

    with pytest.raises(ValueError, match="invented intent order"):
        _validate(record, "先增加肿瘤面积，再减少肿瘤面积。")


def test_generator_guard_does_not_treat_polite_question_as_negation():
    record = _record(category="catalog_single_intent", strength="mild")

    validated = _validate(record, "能不能轻微地增加局部浸润？")

    assert validated[record["case_id"]]["instruction"].startswith("能不能")

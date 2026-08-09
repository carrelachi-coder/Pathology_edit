"""Instruction-only semantic parsing for the joint pathology editor.

This stage deliberately knows nothing about pixels, interfaces, pathology
mechanisms or numeric execution parameters.  Its only authority is the edit
intent explicitly stated by the user.  Visual mechanism selection and skill
composition happen later and remain fail-closed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Protocol

from phase3_mask_edit_refine.agents import OpenAIResponsesJSONClient

from .models import JointCaseContext, JointContractError

SEMANTIC_INTENT_SCHEMA_VERSION = "joint-semantic-intent-v2"


@dataclass(frozen=True)
class PrimitiveHypothesis:
    primitive_id: str
    semantic_fit: str
    priority: int
    rationale: str

    def __post_init__(self) -> None:
        if self.semantic_fit not in {"explicit", "direct", "contextual"}:
            raise JointContractError("unsupported primitive semantic-fit level")
        if self.priority < 0:
            raise JointContractError("primitive hypothesis priority cannot be negative")


@dataclass(frozen=True)
class SemanticEditIntent:
    instruction: str
    primitive_id: str
    subject: str
    direction: str
    explicit_cell_class: str | None
    explicit_location: str | None
    user_constraints: tuple[str, ...]
    uncertainties: tuple[str, ...]
    parser: str
    primitive_hypotheses: tuple[PrimitiveHypothesis, ...]
    parser_metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SEMANTIC_INTENT_SCHEMA_VERSION

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


class SemanticParser(Protocol):
    name: str

    def parse(self, instruction: str) -> SemanticEditIntent:
        """Extract only user-stated semantic intent."""


class RuleBasedSemanticParser:
    """Deterministic offline parser for concise supported instructions."""

    name = "rule_based_semantic_parser_v1"

    _RULES = (
        (
            "necrosis-resolution-v1",
            "necrosis",
            "decrease",
            (r"\b(resolve|reduce|decrease|remove)\b.*\bnecrosis\b", r"(减轻|减少|消退|去除).*(坏死)"),
        ),
        (
            "necrosis-appearance-v1",
            "necrosis",
            "increase",
            (r"\b(increase|add|expand|create)\b.*\bnecrosis\b", r"(增加|扩大|形成|添加).*(坏死)"),
        ),
        (
            "neoplastic-cell-infiltration-increase-v1",
            "neoplastic-cell-infiltration",
            "increase",
            (r"\b(increase|add)\b.*\b(tumou?r budding|tumou?r buds?|neoplastic cell infiltration)\b", r"(增加|添加).*(肿瘤出芽|癌细胞浸润|肿瘤细胞浸润)"),
        ),
        (
            "stroma-increase-v1",
            "stroma",
            "increase",
            (r"\b(increase|expand|add)\b.*\bstroma(?:l)?\b", r"(增加|扩大|添加).*(间质)"),
        ),
        (
            "tumor-burden-decrease-v1",
            "tumor-burden",
            "decrease",
            (r"\b(decrease|reduce|shrink|lower)\b.*\btumou?r\b(?!\s+(?:buds?|budding|cells?))(?:\s+(?:burden|area))?", r"(减少|降低|缩小).*(肿瘤负荷|肿瘤面积|肿瘤)(?!细胞|出芽)"),
        ),
        (
            "tumor-burden-increase-v1",
            "tumor-burden",
            "increase",
            (r"\b(increase|expand|enlarge|raise)\b.*\btumou?r\b(?!\s+(?:buds?|budding|cells?))(?:\s+(?:burden|area))?", r"(增加|提高|扩大).*(肿瘤负荷|肿瘤面积|肿瘤)(?!细胞|出芽)"),
        ),
        (
            "cellularity-decrease-v1",
            "cellularity",
            "decrease",
            (r"\b(decrease|reduce|lower)\b.*\b(cellularity|cell density|nuclear density)\b", r"(降低|减少).*(细胞密度|细胞丰富度|细胞量)"),
        ),
        (
            "cellularity-increase-v1",
            "cellularity",
            "increase",
            (r"\b(increase|raise)\b.*\b(cellularity|cell density|nuclear density)\b", r"(提高|增加).*(细胞密度|细胞丰富度|细胞量)"),
        ),
        (
            "cell-type-abundance-decrease-v1",
            "cell-type-abundance",
            "decrease",
            (
                r"\b(decrease|reduce)\b.*\b(immune cells?|lymphocytes?|plasma cells?|macrophages?|neoplastic cells?|tumou?r cells?)\b",
                r"(减少|降低).*(免疫细胞|淋巴细胞|浆细胞|巨噬细胞|肿瘤细胞)",
            ),
        ),
        (
            "cell-type-abundance-increase-v1",
            "cell-type-abundance",
            "increase",
            (
                r"\b(increase|add)\b.*\b(immune cells?|lymphocytes?|plasma cells?|macrophages?|neoplastic cells?|tumou?r cells?)\b",
                r"(增加|添加).*(免疫细胞|淋巴细胞|浆细胞|巨噬细胞|肿瘤细胞)",
            ),
        ),
    )

    _CELL_CLASSES = (
        (
            "immune",
            (r"\b(?:immune cells?|lymphocytes?)\b", r"免疫细胞|淋巴细胞"),
        ),
        ("plasma_cell", (r"\bplasma cells?\b", r"浆细胞")),
        ("macrophage", (r"\bmacrophages?\b", r"巨噬细胞")),
        (
            "neoplastic",
            (r"\b(?:neoplastic|tumou?r) cells?\b", r"肿瘤细胞|癌细胞"),
        ),
    )

    def parse(self, instruction: str) -> SemanticEditIntent:
        text = _instruction(instruction)
        matches = []
        for primitive_id, subject, direction, patterns in self._RULES:
            if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
                matches.append((primitive_id, subject, direction))
        if len(matches) != 1:
            raise JointContractError(
                "instruction must express exactly one supported edit intent; "
                f"semantic parser found {len(matches)}"
            )
        primitive_id, subject, direction = matches[0]
        hypotheses = _compile_primitive_hypotheses(
            instruction=text,
            primary_primitive_id=primitive_id,
        )
        primitive_id = hypotheses[0].primitive_id
        if len(hypotheses) > 1:
            subject = "tumor"
        cell_class = next(
            (
                name
                for name, patterns in self._CELL_CLASSES
                if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)
            ),
            None,
        )
        return SemanticEditIntent(
            instruction=text,
            primitive_id=primitive_id,
            subject=subject,
            direction=direction,
            explicit_cell_class=cell_class,
            explicit_location=None,
            user_constraints=(),
            uncertainties=(),
            parser=self.name,
            primitive_hypotheses=hypotheses,
            parser_metadata={"mode": "deterministic_offline"},
        )


class OpenAISemanticParser:
    """Strict text-only parser; it may not invent a pathology mechanism."""

    name = "openai_semantic_parser_v1"

    def __init__(self, client: OpenAIResponsesJSONClient) -> None:
        self.client = client

    def parse(self, instruction: str) -> SemanticEditIntent:
        text = _instruction(instruction)
        raw, _usage = self.client.call(
            system_prompt=(
                "You are the instruction-only Semantic Parser for a pathology "
                "counterfactual editor. Extract exactly what the user requested. "
                "Do not inspect or infer organ morphology, dataset labels, edit "
                "mechanism, interface, cell coordinates, counts, area, density "
                "multipliers, or tool parameters. Natural language may omit edit "
                "scope: for a generic request such as 'increase tumor', return "
                "tumor-burden-increase-v1 as the primary semantic reading; a "
                "deterministic compiler will add safe contextual hypotheses. Do not "
                "abstain solely because burden versus infiltration is unspecified. "
                "Abstain only when action, direction, or requested object is unclear."
            ),
            user_prompt=json.dumps(
                {
                    "instruction": text,
                    "supported_primitives": [
                        {
                            "primitive_id": primitive_id,
                            "subject": subject,
                            "direction": direction,
                        }
                        for primitive_id, subject, direction, _patterns in (
                            RuleBasedSemanticParser._RULES
                        )
                    ],
                    "deterministic_ambiguity_policy": {
                        "generic_tumor_increase": [
                            "tumor-burden-increase-v1",
                            "neoplastic-cell-infiltration-increase-v1",
                        ],
                        "parser_returns_primary_only": True,
                    },
                    "null_means_not_explicitly_requested": True,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            image_paths=(),
            schema_name="joint_semantic_intent",
            json_schema=SEMANTIC_INTENT_JSON_SCHEMA,
        )
        if raw.get("abstain") is True:
            raise JointContractError(
                "semantic parser abstained: "
                + str(raw.get("abstain_reason") or "ambiguous instruction")
            )
        primitive_id = str(raw.get("primitive_id") or "")
        supported = {
            item[0]: (item[1], item[2])
            for item in RuleBasedSemanticParser._RULES
        }
        if primitive_id not in supported:
            raise JointContractError("semantic parser returned an unsupported primitive")
        subject = _required_text(raw, "subject")
        direction = _required_text(raw, "direction")
        if (subject, direction) != supported[primitive_id]:
            raise JointContractError(
                "semantic parser subject/direction conflicts with its primitive"
            )
        cell_class = _optional_text(raw, "explicit_cell_class")
        allowed_cell_classes = {item[0] for item in RuleBasedSemanticParser._CELL_CLASSES}
        if cell_class is not None and cell_class not in allowed_cell_classes:
            raise JointContractError("semantic parser returned an unsupported cell class")
        if primitive_id.startswith("cell-type-abundance-") and cell_class is None:
            raise JointContractError(
                "cell abundance instruction must identify the requested cell class"
            )
        hypotheses = _compile_primitive_hypotheses(
            instruction=text,
            primary_primitive_id=primitive_id,
        )
        normalized_subject = "tumor" if len(hypotheses) > 1 else subject
        return SemanticEditIntent(
            instruction=text,
            primitive_id=hypotheses[0].primitive_id,
            subject=normalized_subject,
            direction=direction,
            explicit_cell_class=cell_class,
            explicit_location=_optional_text(raw, "explicit_location"),
            user_constraints=tuple(_text_list(raw, "user_constraints")),
            uncertainties=tuple(_text_list(raw, "uncertainties")),
            parser=self.name,
            primitive_hypotheses=hypotheses,
            parser_metadata=dict(_usage),
        )


def bind_semantic_intent(
    raw_case: Mapping[str, Any], parser: SemanticParser
) -> tuple[JointCaseContext, SemanticEditIntent]:
    """Parse, validate any manifest hint, and bind immutable semantic intent."""

    payload = dict(raw_case)
    intent = parser.parse(_required_text(payload, "instruction"))
    manifest_primitive = payload.get("primitive_id")
    candidate_ids = {
        item.primitive_id for item in intent.primitive_hypotheses
    }
    if manifest_primitive is not None and str(manifest_primitive) not in candidate_ids:
        raise JointContractError(
            "manifest primitive_id contradicts every interpretation of the user "
            f"instruction: {manifest_primitive} not in {sorted(candidate_ids)}"
        )
    initial_primitive = (
        str(manifest_primitive)
        if manifest_primitive is not None
        else intent.primitive_id
    )
    payload["primitive_id"] = initial_primitive
    payload["instruction"] = intent.instruction
    case = JointCaseContext.from_mapping(payload)
    metadata = intent.to_metadata()
    metadata["manifest_primitive_hint"] = (
        str(manifest_primitive) if manifest_primitive is not None else None
    )
    return replace(case, semantic_intent=metadata), intent


def _compile_primitive_hypotheses(
    *, instruction: str, primary_primitive_id: str
) -> tuple[PrimitiveHypothesis, ...]:
    """Expand only a genuinely underspecified tumor-increase expression.

    The lattice is deterministic.  An LLM cannot introduce an arbitrary
    primitive, reverse direction, or reinterpret another tissue/cell object.
    """

    lowered = instruction.casefold()
    explicit_burden = bool(
        re.search(r"\b(tumou?r|cancer)\s+(burden|area|volume)\b", lowered)
        or re.search(r"肿瘤(负荷|面积|体积)", instruction)
    )
    explicit_budding = bool(
        re.search(r"\b(tumou?r buds?|tumou?r budding|neoplastic cell infiltration)\b", lowered)
        or re.search(r"肿瘤出芽|癌细胞浸润|肿瘤细胞浸润", instruction)
    )
    generic_tumor_increase = bool(
        primary_primitive_id
        in {
            "tumor-burden-increase-v1",
            "neoplastic-cell-infiltration-increase-v1",
        }
        and not explicit_burden
        and not explicit_budding
        and (
            re.search(r"\b(increase|expand|enlarge|raise|add)\b.*\b(tumou?r|cancer)\b", lowered)
            or re.search(r"(增加|提高|扩大|增多).*(肿瘤|癌)", instruction)
        )
    )
    if generic_tumor_increase:
        return (
            PrimitiveHypothesis(
                primitive_id="tumor-burden-increase-v1",
                semantic_fit="direct",
                priority=0,
                rationale=(
                    "generic tumor increase most directly denotes greater tissue-level tumor burden"
                ),
            ),
            PrimitiveHypothesis(
                primitive_id="neoplastic-cell-infiltration-increase-v1",
                semantic_fit="contextual",
                priority=1,
                rationale=(
                    "a verified invasive front with budding can realize tumor increase at the cellular level"
                ),
            ),
        )
    return (
        PrimitiveHypothesis(
            primitive_id=primary_primitive_id,
            semantic_fit="explicit",
            priority=0,
            rationale="the instruction explicitly identifies this edit scope",
        ),
    )


SEMANTIC_INTENT_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "abstain",
        "abstain_reason",
        "primitive_id",
        "subject",
        "direction",
        "explicit_cell_class",
        "explicit_location",
        "user_constraints",
        "uncertainties",
    ],
    "properties": {
        "abstain": {"type": "boolean"},
        "abstain_reason": {"type": ["string", "null"]},
        "primitive_id": {"type": ["string", "null"]},
        "subject": {"type": ["string", "null"]},
        "direction": {"type": ["string", "null"]},
        "explicit_cell_class": {"type": ["string", "null"]},
        "explicit_location": {"type": ["string", "null"]},
        "user_constraints": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
}


def _instruction(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise JointContractError("instruction must be a non-empty string")
    return " ".join(value.strip().split())


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise JointContractError(f"semantic intent {key} must be non-empty")
    return value.strip()


def _optional_text(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise JointContractError(f"semantic intent {key} must be null or non-empty")
    return value.strip()


def _text_list(payload: Mapping[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise JointContractError(f"semantic intent {key} must be a string array")
    return [item.strip() for item in value]

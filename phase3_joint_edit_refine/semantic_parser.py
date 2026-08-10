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

SEMANTIC_INTENT_SCHEMA_VERSION = "joint-semantic-intent-v3"

INSTRUCTION_MODES = frozenset({"direct_edit", "clinical_scenario"})
CLINICAL_SCENARIOS = frozenset(
    {
        "direct_edit",
        "disease_progression",
        "disease_regression",
        "treatment_response",
        "post_treatment_progression",
        "residual_disease",
        "local_recurrence",
        "invasion_progression",
        "post_treatment_change",
    }
)
CLINICAL_DIRECTIONS = frozenset(
    {"worsen", "improve", "persist", "unspecified"}
)
TREATMENT_CONTEXTS = frozenset(
    {"none", "post_treatment", "unspecified"}
)
SCENARIO_TARGETS = frozenset(
    {"tumor", "stroma", "necrosis", "cellularity", "immune", "cell_population"}
)
EXPLICIT_EDIT_SCOPES = frozenset(
    {
        "tissue_burden",
        "tissue_compartment",
        "cell_population",
        "joint",
        "unspecified",
    }
)

# The Semantic Parser owns clinical words, while the observation profile owns
# what those words are actually distinguishable as.  CellViT-5 can resolve a
# broad inflammatory class, but it cannot honestly separate plasma cells from
# macrophages.  Unsupported fine classes therefore fail closed instead of
# being silently relabelled as generic inflammation.
OBSERVATION_CELL_CLASS_IDS = {
    "cellvit-five-class-v1": {
        "immune": (2,),
        "neoplastic": (1,),
    }
}


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
    instruction_mode: str
    scenario: str
    clinical_direction: str
    treatment_context: str
    scenario_target: str
    explicit_edit_scope: str
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
    """Deterministic offline parser for tests and concise research instructions.

    Production natural-language input is owned by
    :class:`OpenAIClinicalScenarioParser`.  This parser intentionally supports
    only a small, explicit subset so offline regression tests can exercise the
    same scenario compiler without pretending to be a clinical language model.
    """

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
        scenario = _rule_based_clinical_scenario(text)
        if scenario is not None:
            return _build_scenario_intent(
                instruction=text,
                parser=self.name,
                parser_metadata={"mode": "deterministic_offline"},
                **scenario,
            )
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
            instruction_mode="direct_edit",
            scenario="direct_edit",
            clinical_direction="unspecified",
            treatment_context="none",
            scenario_target=_scenario_target_for_subject(subject),
            explicit_edit_scope=(
                "unspecified"
                if len(hypotheses) > 1
                else _direct_scope_for_primitive(primitive_id)
            ),
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


class OpenAIClinicalScenarioParser:
    """LLM parser for both direct edits and clinical trajectory language.

    The model extracts a closed scenario ontology.  A deterministic compiler,
    not the model, expands clinical scenarios into primitive hypotheses.
    Pathology mechanism selection remains a later multimodal stage.
    """

    name = "openai_clinical_scenario_parser_v1"

    def __init__(self, client: OpenAIResponsesJSONClient) -> None:
        self.client = client

    def parse(self, instruction: str) -> SemanticEditIntent:
        text = _instruction(instruction)
        raw, _usage = self.client.call(
            system_prompt=(
                "You are the instruction-only Clinical Scenario Parser for a joint "
                "pathology counterfactual editor. Understand concise engineering edit "
                "commands and natural clinical trajectory descriptions in Chinese or "
                "English. Extract only what the user stated into the closed ontology "
                "provided by the caller. For a direct edit, select exactly one listed "
                "primitive. For a clinical scenario, set primitive_id, subject and "
                "edit_direction to null: a deterministic compiler will enumerate safe "
                "primitive hypotheses later. Never infer organ morphology, dataset "
                "labels, pathology mechanism, interface, coordinates, counts, area, "
                "density multiplier or tool parameters. Preserve negation and explicit "
                "scope. Do not treat post-treatment context as improvement: 'progresses "
                "after treatment' is worsening. Abstain when the requested future "
                "direction is genuinely unresolved, such as 'simulate changes after "
                "treatment', because an image cannot recover the user's desired future."
            ),
            user_prompt=json.dumps(
                {
                    "instruction": text,
                    "closed_ontology": {
                        "instruction_modes": sorted(INSTRUCTION_MODES),
                        "scenarios": sorted(CLINICAL_SCENARIOS),
                        "clinical_directions": sorted(CLINICAL_DIRECTIONS),
                        "treatment_contexts": sorted(TREATMENT_CONTEXTS),
                        "targets": sorted(SCENARIO_TARGETS),
                        "explicit_edit_scopes": sorted(EXPLICIT_EDIT_SCOPES),
                    },
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
                    "few_shot_examples": _CLINICAL_SCENARIO_FEW_SHOTS,
                    "null_means_not_explicitly_requested": True,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            image_paths=(),
            schema_name="joint_clinical_scenario_intent",
            json_schema=SEMANTIC_INTENT_JSON_SCHEMA,
        )
        if raw.get("abstain") is True:
            raise JointContractError(
                "semantic parser abstained: "
                + str(raw.get("abstain_reason") or "ambiguous instruction")
            )
        instruction_mode = _enum_text(
            raw, "instruction_mode", INSTRUCTION_MODES
        )
        scenario = _enum_text(raw, "scenario", CLINICAL_SCENARIOS)
        clinical_direction = _enum_text(
            raw, "clinical_direction", CLINICAL_DIRECTIONS
        )
        treatment_context = _enum_text(
            raw, "treatment_context", TREATMENT_CONTEXTS
        )
        scenario_target = _enum_text(raw, "target", SCENARIO_TARGETS)
        explicit_edit_scope = _enum_text(
            raw, "explicit_edit_scope", EXPLICIT_EDIT_SCOPES
        )
        supported = {
            item[0]: (item[1], item[2])
            for item in RuleBasedSemanticParser._RULES
        }
        cell_class = _optional_text(raw, "explicit_cell_class")
        allowed_cell_classes = {item[0] for item in RuleBasedSemanticParser._CELL_CLASSES}
        if cell_class is not None and cell_class not in allowed_cell_classes:
            raise JointContractError("semantic parser returned an unsupported cell class")
        shared = {
            "instruction": text,
            "instruction_mode": instruction_mode,
            "scenario": scenario,
            "clinical_direction": clinical_direction,
            "treatment_context": treatment_context,
            "scenario_target": scenario_target,
            "explicit_edit_scope": explicit_edit_scope,
            "explicit_cell_class": cell_class,
            "explicit_location": _optional_text(raw, "explicit_location"),
            "user_constraints": tuple(_text_list(raw, "user_constraints")),
            "uncertainties": tuple(_text_list(raw, "uncertainties")),
            "parser": self.name,
            "parser_metadata": dict(_usage),
        }
        if instruction_mode == "direct_edit":
            if scenario != "direct_edit":
                raise JointContractError(
                    "direct edit must use the direct_edit scenario"
                )
            primitive_id = str(raw.get("primitive_id") or "")
            if primitive_id not in supported:
                raise JointContractError(
                    "semantic parser returned an unsupported direct primitive"
                )
            subject = _required_text(raw, "subject")
            direction = _required_text(raw, "edit_direction")
            if (subject, direction) != supported[primitive_id]:
                raise JointContractError(
                    "semantic parser subject/direction conflicts with its primitive"
                )
            if primitive_id.startswith("cell-type-abundance-") and cell_class is None:
                raise JointContractError(
                    "cell abundance instruction must identify the requested cell class"
                )
            hypotheses = _compile_primitive_hypotheses(
                instruction=text,
                primary_primitive_id=primitive_id,
            )
            return SemanticEditIntent(
                **shared,
                primitive_id=hypotheses[0].primitive_id,
                subject=("tumor" if len(hypotheses) > 1 else subject),
                direction=direction,
                primitive_hypotheses=hypotheses,
            )
        if scenario == "direct_edit":
            raise JointContractError(
                "clinical scenario cannot use the direct_edit scenario"
            )
        if any(raw.get(key) is not None for key in ("primitive_id", "subject", "edit_direction")):
            raise JointContractError(
                "clinical scenario parser must not select an executable primitive"
            )
        return _build_scenario_intent(**shared)


# Backward-compatible import name.  Production behavior is the clinical parser.
OpenAISemanticParser = OpenAIClinicalScenarioParser


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
    metadata = intent.to_metadata()
    metadata["manifest_primitive_hint"] = (
        str(manifest_primitive) if manifest_primitive is not None else None
    )
    if intent.explicit_cell_class is not None:
        profile_id = _required_text(payload, "cell_observation_profile_id")
        resolved = OBSERVATION_CELL_CLASS_IDS.get(profile_id, {}).get(
            intent.explicit_cell_class
        )
        if resolved is None:
            raise JointContractError(
                f"{profile_id} cannot distinguish requested cell class "
                f"{intent.explicit_cell_class!r}"
            )
        provenance = dict(payload.get("provenance") or {})
        existing = provenance.get("target_cell_class_ids")
        if existing is not None:
            existing_ids = (
                (int(existing),)
                if isinstance(existing, int)
                else tuple(sorted(int(value) for value in existing))
            )
            if existing_ids != tuple(sorted(resolved)):
                raise JointContractError(
                    "manifest target_cell_class_ids contradict the parsed cell class"
                )
        provenance["target_cell_class_ids"] = list(resolved)
        provenance["target_cell_class_resolution"] = {
            "semantic_cell_class": intent.explicit_cell_class,
            "observation_profile_id": profile_id,
            "resolved_class_ids": list(resolved),
            "authority": "versioned_observation_profile",
        }
        payload["provenance"] = provenance
        metadata["resolved_cell_class_ids"] = list(resolved)
        metadata["cell_class_resolution"] = dict(
            provenance["target_cell_class_resolution"]
        )
    case = JointCaseContext.from_mapping(payload)
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


def _build_scenario_intent(
    *,
    instruction: str,
    scenario: str,
    clinical_direction: str,
    treatment_context: str,
    scenario_target: str,
    explicit_edit_scope: str,
    parser: str,
    parser_metadata: dict[str, Any],
    instruction_mode: str = "clinical_scenario",
    explicit_cell_class: str | None = None,
    explicit_location: str | None = None,
    user_constraints: tuple[str, ...] = (),
    uncertainties: tuple[str, ...] = (),
) -> SemanticEditIntent:
    """Compile one clinical trajectory into a bounded primitive lattice."""

    _validate_scenario_contract(
        instruction_mode=instruction_mode,
        scenario=scenario,
        clinical_direction=clinical_direction,
        treatment_context=treatment_context,
        scenario_target=scenario_target,
        explicit_edit_scope=explicit_edit_scope,
    )
    specs = _scenario_primitive_specs(
        scenario=scenario,
        clinical_direction=clinical_direction,
        scenario_target=scenario_target,
        explicit_edit_scope=explicit_edit_scope,
    )
    hypotheses = tuple(
        PrimitiveHypothesis(
            primitive_id=primitive_id,
            semantic_fit=fit,
            priority=priority,
            rationale=rationale,
        )
        for priority, (primitive_id, fit, rationale) in enumerate(specs)
    )
    return SemanticEditIntent(
        instruction=instruction,
        instruction_mode=instruction_mode,
        scenario=scenario,
        clinical_direction=clinical_direction,
        treatment_context=treatment_context,
        scenario_target=scenario_target,
        explicit_edit_scope=explicit_edit_scope,
        primitive_id=hypotheses[0].primitive_id,
        subject=scenario_target,
        direction=clinical_direction,
        explicit_cell_class=explicit_cell_class,
        explicit_location=explicit_location,
        user_constraints=user_constraints,
        uncertainties=uncertainties,
        parser=parser,
        primitive_hypotheses=hypotheses,
        parser_metadata=parser_metadata,
    )


def _validate_scenario_contract(
    *,
    instruction_mode: str,
    scenario: str,
    clinical_direction: str,
    treatment_context: str,
    scenario_target: str,
    explicit_edit_scope: str,
) -> None:
    if instruction_mode != "clinical_scenario":
        raise JointContractError("scenario compiler requires clinical_scenario mode")
    if scenario not in CLINICAL_SCENARIOS or scenario == "direct_edit":
        raise JointContractError("unsupported clinical scenario")
    if clinical_direction not in CLINICAL_DIRECTIONS:
        raise JointContractError("unsupported clinical direction")
    if treatment_context not in TREATMENT_CONTEXTS:
        raise JointContractError("unsupported treatment context")
    if scenario_target not in SCENARIO_TARGETS:
        raise JointContractError("unsupported clinical scenario target")
    if explicit_edit_scope not in EXPLICIT_EDIT_SCOPES:
        raise JointContractError("unsupported explicit edit scope")
    required = {
        "disease_progression": "worsen",
        "disease_regression": "improve",
        "treatment_response": "improve",
        "post_treatment_progression": "worsen",
        "residual_disease": "persist",
        "local_recurrence": "worsen",
        "invasion_progression": "worsen",
    }
    if scenario == "post_treatment_change" or clinical_direction == "unspecified":
        raise JointContractError(
            "clinical scenario leaves the requested future direction unresolved; "
            "instruction rewrite is required"
        )
    if scenario in required and clinical_direction != required[scenario]:
        raise JointContractError(
            "clinical scenario and requested trajectory direction conflict"
        )
    if scenario in {
        "treatment_response",
        "post_treatment_progression",
        "residual_disease",
    } and treatment_context != "post_treatment":
        raise JointContractError(
            "treatment-related scenario must preserve post_treatment context"
        )
    if scenario_target != "tumor":
        raise JointContractError(
            "v1 clinical trajectory compiler supports tumor-target scenarios only; "
            "other targets require a direct edit instruction"
        )


def _scenario_primitive_specs(
    *,
    scenario: str,
    clinical_direction: str,
    scenario_target: str,
    explicit_edit_scope: str,
) -> tuple[tuple[str, str, str], ...]:
    del clinical_direction, scenario_target
    if scenario in {
        "disease_progression",
        "post_treatment_progression",
        "local_recurrence",
    }:
        if explicit_edit_scope == "tissue_burden":
            return (
                (
                    "tumor-burden-increase-v1",
                    "explicit",
                    "the clinical instruction explicitly requires greater tumor tissue burden",
                ),
            )
        if explicit_edit_scope == "cell_population":
            return (
                (
                    "cellularity-increase-v1",
                    "direct",
                    "the requested progression is explicitly limited to tumor cellularity",
                ),
                (
                    "neoplastic-cell-infiltration-increase-v1",
                    "contextual",
                    "a visually supported invasive population can realize cellular progression",
                ),
            )
        return (
            (
                "tumor-burden-increase-v1",
                "direct",
                "greater tissue-level tumor burden is the primary generic progression reading",
            ),
            (
                "neoplastic-cell-infiltration-increase-v1",
                "contextual",
                "a verified invasive front can express progression through neoplastic infiltration",
            ),
            (
                "cellularity-increase-v1",
                "contextual",
                "a viable tumor compartment may express progression through higher cellularity",
            ),
        )
    if scenario == "invasion_progression":
        return (
            (
                "neoplastic-cell-infiltration-increase-v1",
                "direct",
                "the user explicitly requested a more invasive tumor phenotype",
            ),
            (
                "tumor-burden-increase-v1",
                "contextual",
                "interface-bound tissue expansion may realize invasion when supported by the patch",
            ),
        )
    if scenario in {"disease_regression", "treatment_response"}:
        if explicit_edit_scope == "tissue_burden":
            return (
                (
                    "tumor-burden-decrease-v1",
                    "explicit",
                    "the instruction explicitly requests tumor shrinkage",
                ),
            )
        if explicit_edit_scope == "cell_population":
            return (
                (
                    "cellularity-decrease-v1",
                    "explicit",
                    "the instruction explicitly requests reduced viable tumor cellularity",
                ),
            )
        return (
            (
                "tumor-burden-decrease-v1",
                "direct",
                "lower tissue-level tumor burden is the primary response reading",
            ),
            (
                "cellularity-decrease-v1",
                "contextual",
                "reduced viable tumor cellularity can express treatment response without changing the compartment",
            ),
            (
                "necrosis-appearance-v1",
                "contextual",
                "new necrosis can express response only when the domain and visible patch support it",
            ),
        )
    if scenario == "residual_disease":
        if explicit_edit_scope == "cell_population":
            return (
                (
                    "cellularity-decrease-v1",
                    "direct",
                    "residual disease is explicitly framed as reduced viable tumor cellularity",
                ),
            )
        return (
            (
                "tumor-burden-decrease-v1",
                "direct",
                "residual disease may be represented by a reduced but retained tumor compartment",
            ),
            (
                "cellularity-decrease-v1",
                "contextual",
                "residual disease may instead preserve the tumor bed while reducing viable cellularity",
            ),
        )
    raise JointContractError("clinical scenario has no executable primitive lattice")


def _rule_based_clinical_scenario(instruction: str) -> dict[str, str] | None:
    lowered = instruction.casefold()
    is_post_treatment = bool(
        re.search(r"\b(?:after|post)[ -]?treatment\b", lowered)
        or re.search(r"治疗后|治疗之后", instruction)
    )
    if is_post_treatment and (
        re.search(r"\b(?:continue|continued|still)\w*\s+(?:progress|worsen)", lowered)
        or re.search(r"治疗后.*(?:继续进展|仍.*进展|恶化|变严重|复发)", instruction)
    ):
        return _scenario_fields(
            "post_treatment_progression", "worsen", "post_treatment"
        )
    if re.search(r"局部复发", instruction) or re.search(r"\blocal recurrence\b", lowered):
        return _scenario_fields("local_recurrence", "worsen", "unspecified")
    if is_post_treatment and (
        re.search(r"残余|残留", instruction)
        or re.search(r"\bresidual (?:tumou?r|disease)\b", lowered)
    ):
        return _scenario_fields("residual_disease", "persist", "post_treatment")
    if (
        re.search(r"侵袭性.*(?:增强|增加)|更具侵袭性", instruction)
        or re.search(r"\b(?:more invasive|increase invasiveness)\b", lowered)
    ):
        return _scenario_fields("invasion_progression", "worsen", "unspecified")
    if is_post_treatment and (
        re.search(r"肿瘤.*(?:缩小|减少)|缓解|改善|治疗反应", instruction)
        or re.search(r"\b(?:respond|response|regress|shrink)\w*\b", lowered)
    ):
        scope = (
            "tissue_burden"
            if re.search(r"肿瘤.*缩小|tumou?r.*shrink", instruction, re.IGNORECASE)
            else "unspecified"
        )
        return _scenario_fields(
            "treatment_response", "improve", "post_treatment", scope
        )
    if is_post_treatment and (
        re.search(r"变化|改变", instruction)
        or re.search(r"\bchanges?\b", lowered)
    ):
        return _scenario_fields(
            "post_treatment_change", "unspecified", "post_treatment"
        )
    if (
        re.search(r"肿瘤.*(?:继续进展|变严重|恶化)|疾病进展", instruction)
        or re.search(r"\b(?:tumou?r|disease).*(?:progress|worsen|more severe)\b", lowered)
    ):
        return _scenario_fields("disease_progression", "worsen", "unspecified")
    if (
        re.search(r"肿瘤.*(?:消退|缓解)|疾病缓解", instruction)
        or re.search(r"\b(?:tumou?r|disease).*(?:regress|improve)\b", lowered)
    ):
        return _scenario_fields("disease_regression", "improve", "unspecified")
    return None


def _scenario_fields(
    scenario: str,
    direction: str,
    treatment_context: str,
    scope: str = "unspecified",
) -> dict[str, str]:
    return {
        "scenario": scenario,
        "clinical_direction": direction,
        "treatment_context": treatment_context,
        "scenario_target": "tumor",
        "explicit_edit_scope": scope,
    }


def _scenario_target_for_subject(subject: str) -> str:
    if subject in {"tumor", "tumor-burden", "neoplastic-cell-infiltration"}:
        return "tumor"
    if subject == "cell-type-abundance":
        return "cell_population"
    if subject in SCENARIO_TARGETS:
        return subject
    raise JointContractError("direct primitive subject has no scenario target")


def _direct_scope_for_primitive(primitive_id: str) -> str:
    if primitive_id.startswith("tumor-burden-"):
        return "tissue_burden"
    if primitive_id in {
        "stroma-increase-v1",
        "necrosis-appearance-v1",
        "necrosis-resolution-v1",
    }:
        return "tissue_compartment"
    if primitive_id.startswith(("cellularity-", "cell-type-abundance-")) or (
        primitive_id == "neoplastic-cell-infiltration-increase-v1"
    ):
        return "cell_population"
    return "unspecified"


_CLINICAL_SCENARIO_FEW_SHOTS = [
    {
        "instruction": "Increase tumor area.",
        "output": {
            "instruction_mode": "direct_edit",
            "scenario": "direct_edit",
            "clinical_direction": "unspecified",
            "treatment_context": "none",
            "target": "tumor",
            "explicit_edit_scope": "tissue_burden",
            "primitive_id": "tumor-burden-increase-v1",
            "subject": "tumor-burden",
            "edit_direction": "increase",
        },
        "why": "Area is an explicit tissue-burden edit, not an invitation to choose budding.",
    },
    {
        "instruction": "模拟这个肿瘤继续进展。",
        "output": {
            "instruction_mode": "clinical_scenario",
            "scenario": "disease_progression",
            "clinical_direction": "worsen",
            "treatment_context": "unspecified",
            "target": "tumor",
            "explicit_edit_scope": "unspecified",
            "primitive_id": None,
            "subject": None,
            "edit_direction": None,
        },
        "why": "The user specified a trajectory but did not prescribe its tissue or cellular realization.",
    },
    {
        "instruction": "模拟治疗后肿瘤缩小。",
        "output": {
            "instruction_mode": "clinical_scenario",
            "scenario": "treatment_response",
            "clinical_direction": "improve",
            "treatment_context": "post_treatment",
            "target": "tumor",
            "explicit_edit_scope": "tissue_burden",
            "primitive_id": None,
            "subject": None,
            "edit_direction": None,
        },
        "why": "Treatment is context and shrinkage explicitly constrains the realization to tissue burden.",
    },
    {
        "instruction": "模拟治疗后肿瘤仍然进展。",
        "output": {
            "instruction_mode": "clinical_scenario",
            "scenario": "post_treatment_progression",
            "clinical_direction": "worsen",
            "treatment_context": "post_treatment",
            "target": "tumor",
            "explicit_edit_scope": "unspecified",
            "primitive_id": None,
            "subject": None,
            "edit_direction": None,
        },
        "why": "Post-treatment does not reverse the explicit worsening direction.",
    },
    {
        "instruction": "模拟治疗后的变化。",
        "output": {
            "abstain": True,
            "abstain_reason": "The requested future direction could be response or progression.",
            "instruction_mode": "clinical_scenario",
            "scenario": "post_treatment_change",
            "clinical_direction": "unspecified",
            "treatment_context": "post_treatment",
            "target": "tumor",
            "explicit_edit_scope": "unspecified",
            "primitive_id": None,
            "subject": None,
            "edit_direction": None,
        },
        "why": "An H&E patch cannot recover which future the user intended.",
    },
]


SEMANTIC_INTENT_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "abstain",
        "abstain_reason",
        "instruction_mode",
        "scenario",
        "clinical_direction",
        "treatment_context",
        "target",
        "explicit_edit_scope",
        "primitive_id",
        "subject",
        "edit_direction",
        "explicit_cell_class",
        "explicit_location",
        "user_constraints",
        "uncertainties",
    ],
    "properties": {
        "abstain": {"type": "boolean"},
        "abstain_reason": {"type": ["string", "null"]},
        "instruction_mode": {
            "type": ["string", "null"],
            "enum": [None, *sorted(INSTRUCTION_MODES)],
        },
        "scenario": {
            "type": ["string", "null"],
            "enum": [None, *sorted(CLINICAL_SCENARIOS)],
        },
        "clinical_direction": {
            "type": ["string", "null"],
            "enum": [None, *sorted(CLINICAL_DIRECTIONS)],
        },
        "treatment_context": {
            "type": ["string", "null"],
            "enum": [None, *sorted(TREATMENT_CONTEXTS)],
        },
        "target": {
            "type": ["string", "null"],
            "enum": [None, *sorted(SCENARIO_TARGETS)],
        },
        "explicit_edit_scope": {
            "type": ["string", "null"],
            "enum": [None, *sorted(EXPLICIT_EDIT_SCOPES)],
        },
        "primitive_id": {"type": ["string", "null"]},
        "subject": {"type": ["string", "null"]},
        "edit_direction": {"type": ["string", "null"]},
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


def _enum_text(
    payload: Mapping[str, Any], key: str, allowed: frozenset[str]
) -> str:
    value = _required_text(payload, key)
    if value not in allowed:
        raise JointContractError(
            f"semantic intent {key} is outside the closed ontology"
        )
    return value


def _text_list(payload: Mapping[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise JointContractError(f"semantic intent {key} must be a string array")
    return [item.strip() for item in value]

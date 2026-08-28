"""Primitive-free semantic requests for single- and multi-intent editing.

The Parser owns language understanding only.  It may split one instruction
into several user intents and preserve an explicit order, but it cannot name a
primitive, inspect a mask, or choose an executable mechanism.  Those decisions
belong to the program Planner.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from phase3_mask_edit_refine.agents import OpenAIResponsesJSONClient

from .models import JointContractError

SEMANTIC_REQUEST_SCHEMA_VERSION = "joint-semantic-request-v4"

INTENT_TYPES = frozenset({"direct_edit", "clinical_trajectory"})
BIOLOGICAL_TARGETS = frozenset(
    {
        "tumor_state",
        "tumor_extent",
        "tumor_topology",
        "invasion_pattern",
        "necrosis",
        "neoplastic_cell_population",
        "selected_cell_population",
        "overall_cellularity",
        "immune_compartment",
        "stroma",
    }
)
OPERATIONS = frozenset(
    {
        "increase",
        "decrease",
        "fragment",
        "clear",
        "appear",
        "repopulate",
        "worsen",
        "improve",
        "unspecified",
    }
)
CLINICAL_CONTEXTS = frozenset(
    {
        "none",
        "disease_progression",
        "disease_regression",
        "post_treatment",
        "residual_disease",
        "local_recurrence",
        "unspecified",
    }
)
SPATIAL_SCOPES = frozenset(
    {
        "whole_lesion",
        "local",
        "boundary",
        "peritumoral",
        "intratumoral",
        "selected_roi",
        "unspecified",
    }
)
MORPHOLOGIES = frozenset(
    {
        "cohesive",
        "invasive_front",
        "cord",
        "nest_cord",
        "nest",
        "single_file",
        "single_cell",
        "small_cluster",
        "fragmented",
        "unspecified",
    }
)
CELL_CLASSES = frozenset({"neoplastic", "inflammatory", "connective"})
SEMANTIC_CELL_CLASS_IDS = {
    "cellvit-five-class-v1": {
        "neoplastic": (1,),
        "inflammatory": (2,),
        "connective": (3,),
    }
}
STRENGTHS = frozenset({"mild", "moderate", "strong", "unspecified"})
POLARITIES = frozenset({"affirmed", "negated"})
RELATION_TYPES = frozenset({"explicit_sequence", "unordered"})


SEMANTIC_REQUEST_SYSTEM_PROMPT = """You are the instruction-only Semantic Request Parser for a pathology mask editor.

Your sole responsibility is to translate the user's Chinese or English language into the caller's closed semantic ontology. A request may contain one intent or several intents. Split only user-stated biological intentions; preserve explicit order words such as first, then, after, and finally. An implementation detail needed to realize one biological change is not a second user intent.

For every intent, extract the biological target, requested operation, polarity, clinical context, spatial scope, morphology, explicitly named cell class, strength, literal source span, and uncertainty. Use `direct_edit` for an explicit requested change and `clinical_trajectory` for progression, regression, treatment-response, residual-disease, or recurrence language. Normalize paraphrases into the supplied enum values.

Assign intent IDs consecutively in textual order, starting with `intent-001`. Use the following target boundaries consistently:
- `selected_cell_population`: the user explicitly changes the number or abundance of one named non-neoplastic cell class, such as inflammatory or connective cells.
- `neoplastic_cell_population`: the user changes the number or abundance of tumour cells without requesting a tissue-area or invasion-pattern change.
- `overall_cellularity`: the user changes total local nucleus density across cell classes.
- `immune_compartment`: the user changes the area or extent of an immune-rich/infiltrated tissue region; do not use it for a request that only adds or removes inflammatory cells.
- `tumor_extent`: the user changes tumour area, footprint, boundary extent, or requests local clearance.
- `tumor_topology`: the user requests fragmentation or multiple separated residual foci.
- `invasion_pattern`: the user requests how tumour crosses or extends beyond a boundary, such as cords, nests, single cells, small clusters, or an otherwise unspecified infiltrative pattern.
- `tumor_state`: reserve this for generic progression or response language whose concrete endpoint is not stated.
- `necrosis` and `stroma`: use only when those biological compartments are explicitly requested.

An explicit morphology or spatial qualifier belongs to the same biological intent, not a separate intent. In particular, “make the boundary cohesive and broader”, “keep the boundary continuous while expanding it”, and “replace/repopulate necrosis with viable tumour” are each one endpoint, not two intents.

Apply these normalization rules before returning the fields:
- Peritumoral or boundary-crossing tumour cords, nests, single cells, or small clusters always describe `invasion_pattern`, even when the wording says to add, form, appear, or increase their number. Normalize their operation to `increase`. This rule applies independently to every intent in a multi-intent request. Do not reinterpret them as generic `neoplastic_cell_population` or as `tumor_topology`.
- Bare “local invasion” / “局部浸润” means `invasion_pattern` with `morphology=unspecified`. It means `immune_compartment` only when immune, inflammatory, lymphocytic, or another immune-rich compartment is stated.
- `tumor_topology` + `fragment` + `fragmented` is reserved for splitting an established local tumour into multiple separated foci. A new peritumoral nest is an invasion morphology, not fragmentation, even if the nests are called separate or discrete.
- A continuous/cohesive outward tumour-boundary expansion is one `tumor_extent` + `increase` intent with `morphology=cohesive`. Continuity is not topology and outward expansion is not clearance.
- For `necrosis`, creation/increase of an intratumoral necrotic region always normalizes to `appear`; replacement of necrosis by viable tumour always normalizes to `repopulate`. In this ontology, `necrosis` never uses the generic `increase` or `decrease` operations. The latter endpoint is one intent and is not clearance or generic decrease.
- A whole-lesion footprint reduction is `tumor_extent` + `decrease`; use `clear` only for explicit local removal/clearance of a tumour focus.
- Negation changes only `polarity`. Thus “do not increase X” remains `operation=increase` with `polarity=negated`; it is not a decrease.
- Set `morphology=fragmented` without exception whenever the user asks to fragment/split tumour into multiple foci. Set `morphology=cohesive` for continuous/cohesive boundary expansion. Do not infer `invasive_front` merely from the adjective “invasive” when no front or boundary is named.
- `cell_class` records a population selector, not every cell word in the sentence. Use `neoplastic` for `neoplastic_cell_population`, and the named non-neoplastic class for `selected_cell_population`. Use null for `invasion_pattern`, `tumor_extent`, `tumor_topology`, necrosis, stroma, and compartment-level intents, even if their morphology is made of tumour cells.
- Map “whole/overall/entire lesion” and “整体/全病灶” to `whole_lesion`. A “local/localized area” is `local`; use `selected_roi` only when the user explicitly refers to a selected, marked, or chosen ROI. Map “slight/slightly/a bit/mild/modest” and “轻微/轻度/稍微” to `mild`; do not infer mild strength from polite or hesitant wording alone.

Relations are mandatory for every pair of separately stated intentions. Use `explicit_sequence` only when the user states an order such as first/then/after/先/然后. When two intentions are joined without an order—including “and”, “while”, “at the same time”, “同时”, or a simultaneous contradiction—return an `unordered` relation and never invent execution order.

A named cell class must not be inferred when the user only refers to a tissue compartment. Preserve deliberately underspecified morphology as `unspecified`.

Use `clinical_context=none` when an explicit edit contains no disease-course, treatment, residual-disease, or recurrence framing. Use `clinical_context=unspecified` only when the user invokes a clinical context but the available enum cannot be determined. Do not treat ordinary uncertainty or polite wording as a clinical context.

Never select, name, rank, or suggest an edit primitive or pathology mechanism. Never inspect or infer image morphology, annotation labels, coordinates, masks, connected components, area, cell count, density, tool parameters, or feasibility. Do not invent an order that the user did not state. Preserve negation and do not convert post-treatment context into improvement unless the user states response or regression. If the biological direction is genuinely missing, use `unspecified` and record the uncertainty instead of guessing.

Return only JSON conforming to the supplied strict schema."""


SEMANTIC_REQUEST_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "intents",
        "relations",
        "global_constraints",
        "uncertainties",
    ],
    "properties": {
        "schema_version": {"type": "string", "enum": [SEMANTIC_REQUEST_SCHEMA_VERSION]},
        "intents": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "intent_id",
                    "intent_type",
                    "target",
                    "operation",
                    "polarity",
                    "clinical_context",
                    "spatial_scope",
                    "morphology",
                    "cell_class",
                    "strength",
                    "source_text",
                    "constraints",
                    "uncertainties",
                ],
                "properties": {
                    "intent_id": {"type": "string", "pattern": "^intent-[0-9]{3}$"},
                    "intent_type": {"type": "string", "enum": sorted(INTENT_TYPES)},
                    "target": {"type": "string", "enum": sorted(BIOLOGICAL_TARGETS)},
                    "operation": {"type": "string", "enum": sorted(OPERATIONS)},
                    "polarity": {"type": "string", "enum": sorted(POLARITIES)},
                    "clinical_context": {
                        "type": "string",
                        "enum": sorted(CLINICAL_CONTEXTS),
                    },
                    "spatial_scope": {"type": "string", "enum": sorted(SPATIAL_SCOPES)},
                    "morphology": {"type": "string", "enum": sorted(MORPHOLOGIES)},
                    "cell_class": {
                        "type": ["string", "null"],
                        "enum": [None, *sorted(CELL_CLASSES)],
                    },
                    "strength": {"type": "string", "enum": sorted(STRENGTHS)},
                    "source_text": {"type": "string", "minLength": 1},
                    "constraints": {"type": "array", "items": {"type": "string"}},
                    "uncertainties": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["before_intent_id", "after_intent_id", "relation_type"],
                "properties": {
                    "before_intent_id": {"type": "string"},
                    "after_intent_id": {"type": "string"},
                    "relation_type": {"type": "string", "enum": sorted(RELATION_TYPES)},
                },
            },
        },
        "global_constraints": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
}


@dataclass(frozen=True)
class SemanticIntentClause:
    intent_id: str
    intent_type: str
    target: str
    operation: str
    polarity: str
    clinical_context: str
    spatial_scope: str
    morphology: str
    cell_class: str | None
    strength: str
    source_text: str
    constraints: tuple[str, ...] = ()
    uncertainties: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not re.fullmatch(r"intent-[0-9]{3}", self.intent_id):
            raise JointContractError("semantic intent_id must use intent-NNN")
        _require_enum(self.intent_type, INTENT_TYPES, "intent_type")
        _require_enum(self.target, BIOLOGICAL_TARGETS, "target")
        _require_enum(self.operation, OPERATIONS, "operation")
        _require_enum(self.polarity, POLARITIES, "polarity")
        _require_enum(self.clinical_context, CLINICAL_CONTEXTS, "clinical_context")
        _require_enum(self.spatial_scope, SPATIAL_SCOPES, "spatial_scope")
        _require_enum(self.morphology, MORPHOLOGIES, "morphology")
        _require_enum(self.strength, STRENGTHS, "strength")
        if not self.source_text.strip():
            raise JointContractError("semantic intent source_text cannot be empty")
        if self.cell_class is not None and not self.cell_class.strip():
            raise JointContractError("semantic intent cell_class cannot be blank")
        if self.cell_class is not None:
            _require_enum(self.cell_class, CELL_CLASSES, "cell_class")
        if self.target == "selected_cell_population" and self.cell_class is None:
            raise JointContractError(
                "selected_cell_population requires an explicit cell_class"
            )

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IntentRelation:
    before_intent_id: str
    after_intent_id: str
    relation_type: str

    def __post_init__(self) -> None:
        _require_enum(self.relation_type, RELATION_TYPES, "relation_type")
        if self.before_intent_id == self.after_intent_id:
            raise JointContractError("an intent cannot precede itself")

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticRequest:
    instruction: str
    intents: tuple[SemanticIntentClause, ...]
    relations: tuple[IntentRelation, ...] = ()
    global_constraints: tuple[str, ...] = ()
    uncertainties: tuple[str, ...] = ()
    parser: str = "unknown"
    parser_metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SEMANTIC_REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEMANTIC_REQUEST_SCHEMA_VERSION:
            raise JointContractError("unsupported semantic-request schema")
        if not self.instruction.strip():
            raise JointContractError("semantic request instruction cannot be empty")
        if not self.intents:
            raise JointContractError("semantic request requires at least one intent")
        intent_ids = tuple(item.intent_id for item in self.intents)
        if len(set(intent_ids)) != len(intent_ids):
            raise JointContractError("semantic request contains duplicate intent IDs")
        known = set(intent_ids)
        for relation in self.relations:
            if {relation.before_intent_id, relation.after_intent_id} - known:
                raise JointContractError("semantic relation names an unknown intent")
        _validate_relation_dag(intent_ids, self.relations)

    @property
    def request_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_metadata(include_digest=False),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def ordered_intents(self) -> tuple[SemanticIntentClause, ...]:
        """Return a stable topological order without inventing new semantics."""

        position = {item.intent_id: index for index, item in enumerate(self.intents)}
        incoming = {item.intent_id: 0 for item in self.intents}
        outgoing: dict[str, list[str]] = {item.intent_id: [] for item in self.intents}
        for relation in self.relations:
            if relation.relation_type != "explicit_sequence":
                continue
            incoming[relation.after_intent_id] += 1
            outgoing[relation.before_intent_id].append(relation.after_intent_id)
        ready = sorted(
            (key for key, value in incoming.items() if value == 0),
            key=position.__getitem__,
        )
        result: list[SemanticIntentClause] = []
        by_id = {item.intent_id: item for item in self.intents}
        while ready:
            current = ready.pop(0)
            result.append(by_id[current])
            for target in sorted(outgoing[current], key=position.__getitem__):
                incoming[target] -= 1
                if incoming[target] == 0:
                    ready.append(target)
                    ready.sort(key=position.__getitem__)
        return tuple(result)

    def to_metadata(self, *, include_digest: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "instruction": self.instruction,
            "intents": [item.to_metadata() for item in self.intents],
            "relations": [item.to_metadata() for item in self.relations],
            "global_constraints": list(self.global_constraints),
            "uncertainties": list(self.uncertainties),
            "parser": self.parser,
            "parser_metadata": dict(self.parser_metadata),
        }
        if include_digest:
            result["request_sha256"] = self.request_sha256
        return result


class SemanticRequestParser(Protocol):
    name: str

    def parse(self, instruction: str) -> SemanticRequest: ...


class OpenAISemanticRequestParser:
    """Strict-schema LLM parser that has no primitive catalog access."""

    name = "openai_semantic_request_parser_v4"

    def __init__(self, client: OpenAIResponsesJSONClient) -> None:
        self.client = client

    def parse(self, instruction: str) -> SemanticRequest:
        text = _instruction(instruction)
        raw, usage = self.client.call(
            system_prompt=SEMANTIC_REQUEST_SYSTEM_PROMPT,
            user_prompt=json.dumps(
                {
                    "instruction": text,
                    "closed_ontology": {
                        "intent_types": sorted(INTENT_TYPES),
                        "biological_targets": sorted(BIOLOGICAL_TARGETS),
                        "operations": sorted(OPERATIONS),
                        "polarities": sorted(POLARITIES),
                        "clinical_contexts": sorted(CLINICAL_CONTEXTS),
                        "spatial_scopes": sorted(SPATIAL_SCOPES),
                        "morphologies": sorted(MORPHOLOGIES),
                        "strengths": sorted(STRENGTHS),
                        "relation_types": sorted(RELATION_TYPES),
                    },
                    "rules": {
                        "one_intent_is_one_user_goal": True,
                        "implementation_substeps_are_not_user_intents": True,
                        "primitive_selection_is_forbidden": True,
                        "image_or_mask_inference_is_forbidden": True,
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            image_paths=(),
            schema_name="joint_semantic_request_v4",
            json_schema=SEMANTIC_REQUEST_JSON_SCHEMA,
        )
        return semantic_request_from_metadata(
            {
                **dict(raw),
                "instruction": text,
                "parser": self.name,
                "parser_metadata": usage,
            }
        )


class PreboundSemanticRequestParser:
    """Consume one already reviewed v4 request without reinterpreting it."""

    name = "prebound_semantic_request_parser_v4"

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.request = semantic_request_from_metadata(payload)

    def parse(self, instruction: str) -> SemanticRequest:
        if _instruction(instruction) != self.request.instruction:
            raise JointContractError(
                "prebound semantic request is detached from the instruction"
            )
        return self.request


class RuleBasedSemanticRequestParser:
    """Small deterministic parser for offline regression tests only."""

    name = "rule_based_semantic_request_parser_v4"

    def parse(self, instruction: str) -> SemanticRequest:
        text = _instruction(instruction)
        clauses, connectors = _split_clauses(text)
        intents = tuple(
            _classify_clause(clause, index=index)
            for index, clause in enumerate(clauses, start=1)
        )
        relations: list[IntentRelation] = []
        for index, connector in enumerate(connectors):
            relation_type = (
                "unordered"
                if connector.casefold() in {"并且", "同时", "以及", "and"}
                else "explicit_sequence"
            )
            relations.append(
                IntentRelation(
                    before_intent_id=intents[index].intent_id,
                    after_intent_id=intents[index + 1].intent_id,
                    relation_type=relation_type,
                )
            )
        return SemanticRequest(
            instruction=text,
            intents=intents,
            relations=tuple(relations),
            parser=self.name,
            parser_metadata={"mode": "deterministic_offline"},
        )


def semantic_request_from_metadata(payload: Mapping[str, Any]) -> SemanticRequest:
    if not isinstance(payload, Mapping):
        raise JointContractError("semantic request must be an object")
    forbidden = {"primitive_id", "primitive_hypotheses", "mechanism_id"}
    if _contains_forbidden_key(payload, forbidden):
        raise JointContractError(
            "semantic request illegally contains a primitive or mechanism decision"
        )
    allowed_top_level = {
        "schema_version",
        "instruction",
        "intents",
        "relations",
        "global_constraints",
        "uncertainties",
        "parser",
        "parser_metadata",
        "request_sha256",
    }
    if set(payload) - allowed_top_level:
        raise JointContractError("semantic request has unknown top-level fields")
    if payload.get("schema_version") != SEMANTIC_REQUEST_SCHEMA_VERSION:
        raise JointContractError("unsupported semantic-request schema")
    raw_intents = payload.get("intents")
    raw_relations = payload.get("relations")
    if not isinstance(raw_intents, list) or not isinstance(raw_relations, list):
        raise JointContractError("semantic request intents/relations must be arrays")
    intents = tuple(_intent_from_mapping(item) for item in raw_intents)
    relations = tuple(_relation_from_mapping(item) for item in raw_relations)
    request = SemanticRequest(
        instruction=_instruction(str(payload.get("instruction") or "")),
        intents=intents,
        relations=relations,
        global_constraints=_string_tuple(payload.get("global_constraints", ())),
        uncertainties=_string_tuple(payload.get("uncertainties", ())),
        parser=str(payload.get("parser") or "prebound_semantic_request_parser_v4"),
        parser_metadata=(
            dict(payload.get("parser_metadata") or {})
            if isinstance(payload.get("parser_metadata", {}), Mapping)
            else {}
        ),
    )
    supplied_digest = payload.get("request_sha256")
    if supplied_digest is not None and supplied_digest != request.request_sha256:
        raise JointContractError("semantic request digest is detached from its content")
    return request


def _intent_from_mapping(raw: Any) -> SemanticIntentClause:
    if not isinstance(raw, Mapping):
        raise JointContractError("semantic intent must be an object")
    expected = {
        "intent_id",
        "intent_type",
        "target",
        "operation",
        "polarity",
        "clinical_context",
        "spatial_scope",
        "morphology",
        "cell_class",
        "strength",
        "source_text",
        "constraints",
        "uncertainties",
    }
    if set(raw) != expected:
        raise JointContractError("semantic intent fields do not match the v4 schema")
    cell_class = raw.get("cell_class")
    if cell_class is not None and not isinstance(cell_class, str):
        raise JointContractError("semantic intent cell_class must be text or null")
    return SemanticIntentClause(
        intent_id=str(raw["intent_id"]),
        intent_type=str(raw["intent_type"]),
        target=str(raw["target"]),
        operation=str(raw["operation"]),
        polarity=str(raw["polarity"]),
        clinical_context=str(raw["clinical_context"]),
        spatial_scope=str(raw["spatial_scope"]),
        morphology=str(raw["morphology"]),
        cell_class=cell_class,
        strength=str(raw["strength"]),
        source_text=str(raw["source_text"]),
        constraints=_string_tuple(raw["constraints"]),
        uncertainties=_string_tuple(raw["uncertainties"]),
    )


def _relation_from_mapping(raw: Any) -> IntentRelation:
    if not isinstance(raw, Mapping) or set(raw) != {
        "before_intent_id",
        "after_intent_id",
        "relation_type",
    }:
        raise JointContractError("semantic relation fields do not match the v4 schema")
    return IntentRelation(
        before_intent_id=str(raw["before_intent_id"]),
        after_intent_id=str(raw["after_intent_id"]),
        relation_type=str(raw["relation_type"]),
    )


def _split_clauses(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    pattern = re.compile(
        r"\s*((?<![A-Za-z])(?:and then|after that|then|finally|and)(?![A-Za-z])|"
        r"然后|接着|随后|最后|再|并且|同时|以及)\s*",
        flags=re.IGNORECASE,
    )
    parts = pattern.split(text)
    clauses = [parts[0].strip(" ，,;；")]
    connectors: list[str] = []
    for index in range(1, len(parts), 2):
        connector = parts[index].strip()
        clause = parts[index + 1].strip(" ，,;；")
        if clause:
            connectors.append(connector)
            clauses.append(clause)
    clauses = [_strip_order_prefix(item) for item in clauses if item]
    if not clauses:
        raise JointContractError("offline semantic parser found no intent clause")
    return tuple(clauses), tuple(connectors[: max(0, len(clauses) - 1)])


def _strip_order_prefix(text: str) -> str:
    return re.sub(
        r"^(?:先|首先|第一步|first(?:ly)?)[：:\s]*",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    ).strip()


def _classify_clause(clause: str, *, index: int) -> SemanticIntentClause:
    lowered = clause.casefold()
    clinical = bool(
        re.search(
            r"治疗|进展|退缩|缓解|残余|复发|病情|疗效|treat|progress|regress|response|residual|recurr",
            lowered,
        )
    )
    context = _clinical_context(lowered)
    scope = _spatial_scope(lowered)
    strength = _strength(lowered)
    target = "tumor_state"
    operation = "unspecified"
    morphology = "unspecified"
    cell_class: str | None = None
    uncertainties: tuple[str, ...] = ()
    polarity = (
        "negated"
        if re.search(r"不要|不能|别|禁止|do not|don't|must not|never", lowered)
        else "affirmed"
    )

    if re.search(
        r"碎片|分裂|分开|割裂|多个(?:残余)?(?:病灶|小灶)|fragment|split|multiple residual foci",
        lowered,
    ):
        target, operation, morphology = "tumor_topology", "fragment", "fragmented"
    elif re.search(r"清除|清空|clear|eradicate", lowered):
        target, operation = "tumor_extent", "clear"
    elif re.search(r"坏死.*(恢复|再生|肿瘤)|repopulat.*necros|viable.*necros", lowered):
        target, operation = "necrosis", "repopulate"
    elif re.search(
        r"增加坏死|出现坏死|坏死形成|necrosis.*appear|increase necrosis", lowered
    ):
        target, operation, scope = "necrosis", "appear", "intratumoral"
    elif re.search(r"免疫.*(区域|区室|浸润区)|immune.*compartment", lowered):
        target = "immune_compartment"
        operation = _increase_or_decrease(lowered)
    elif re.search(r"细胞密度|总体细胞|overall cellularity|cellularity", lowered):
        target = "overall_cellularity"
        operation = _increase_or_decrease(lowered)
    elif re.search(r"淋巴|免疫细胞|炎症细胞|lymph|immune cell|inflammatory", lowered):
        target = "selected_cell_population"
        operation = _increase_or_decrease(lowered)
        cell_class = "inflammatory"
    elif re.search(r"成纤维|纤维母|fibroblast|connective", lowered):
        target = "selected_cell_population"
        operation = _increase_or_decrease(lowered)
        cell_class = "connective"
    elif re.search(
        r"缩小|减少.*面积|退缩|shrink|decrease.*(area|footprint)|regress", lowered
    ):
        target, operation = "tumor_extent", "decrease"
    elif re.search(
        r"肿瘤细胞|癌细胞|neoplastic cell|tumou?r cell", lowered
    ) and not re.search(
        r"浸润|散落|单细胞|小簇|细胞簇|小团|出芽|infiltrat|cord|nest|scatter|cluster",
        lowered,
    ):
        target = "neoplastic_cell_population"
        operation = _increase_or_decrease(lowered)
        cell_class = "neoplastic"
    elif re.search(r"间质|stroma", lowered) and re.search(
        r"增加|增多|扩张|increase|expand", lowered
    ):
        target, operation = "stroma", "increase"
    elif re.search(r"单列|单行|single[- ]file", lowered):
        target, operation, morphology = "invasion_pattern", "increase", "single_file"
        scope = "peritumoral" if scope == "unspecified" else scope
    elif re.search(r"巢索|巢状条索|nest[- /]cord", lowered):
        target, operation, morphology = "invasion_pattern", "increase", "nest_cord"
        scope = "boundary" if scope == "unspecified" else scope
    elif re.search(r"浸润前沿|invasive front|infiltrative front", lowered):
        target, operation, morphology = (
            "invasion_pattern",
            "increase",
            "invasive_front",
        )
        scope = "boundary"
    elif re.search(r"条索|肿瘤索|cord", lowered):
        target, operation, morphology = "invasion_pattern", "increase", "cord"
        scope = "peritumoral" if scope == "unspecified" else scope
    elif re.search(r"肿瘤巢|小巢|nest", lowered):
        target, operation, morphology = "invasion_pattern", "increase", "nest"
        scope = "peritumoral" if scope == "unspecified" else scope
    elif re.search(
        r"单细胞|散落(?:的)?(?:单个)?(?:肿瘤)?细胞|single[- ]cell|scatter",
        lowered,
    ):
        target, operation, morphology = "invasion_pattern", "increase", "single_cell"
        scope = "peritumoral" if scope == "unspecified" else scope
    elif re.search(
        r"小(?:细胞)?簇|小团|出芽|small (?:tumou?r[- ]?)?(?:cell[- ]?)?cluster|budding",
        lowered,
    ):
        target, operation, morphology = "invasion_pattern", "increase", "small_cluster"
        scope = "peritumoral" if scope == "unspecified" else scope
    elif re.search(r"浸润|infiltrat|invasive|invasion", lowered):
        target, operation = "invasion_pattern", "increase"
        uncertainties = ("invasion morphology is unspecified",)
    elif re.search(r"边界.*(扩|长)|外沿.*(扩|长)|expand.*boundar|cohesive", lowered):
        target, operation, morphology, scope = (
            "tumor_extent",
            "increase",
            "cohesive",
            "boundary",
        )
    elif re.search(r"进展|恶化|progress|worsen", lowered):
        target, operation = "tumor_state", "worsen"
        context = (
            "post_treatment" if context == "post_treatment" else "disease_progression"
        )
        uncertainties = ("specific morphological endpoint is unspecified",)
    elif re.search(r"缓解|改善|response|improve", lowered):
        target, operation = "tumor_state", "improve"
        uncertainties = ("specific response endpoint is unspecified",)
    elif re.search(r"增加|扩大|增长|increase|expand|grow", lowered):
        target, operation = "tumor_extent", "increase"
        uncertainties = ("tumor growth morphology is unspecified",)
    elif re.search(r"减少|降低|decrease|reduce", lowered):
        target, operation = "tumor_extent", "decrease"
    else:
        uncertainties = ("offline parser cannot resolve the biological direction",)

    return SemanticIntentClause(
        intent_id=f"intent-{index:03d}",
        intent_type="clinical_trajectory" if clinical else "direct_edit",
        target=target,
        operation=operation,
        polarity=polarity,
        clinical_context=context,
        spatial_scope=scope,
        morphology=morphology,
        cell_class=cell_class,
        strength=strength,
        source_text=clause.strip(),
        constraints=(),
        uncertainties=uncertainties,
    )


def _clinical_context(text: str) -> str:
    if re.search(r"残余|residual", text):
        return "residual_disease"
    if re.search(r"复发|recurr", text):
        return "local_recurrence"
    if re.search(r"治疗|treat|therapy", text):
        return "post_treatment"
    if re.search(r"退缩|缓解|regress|response|improve", text):
        return "disease_regression"
    if re.search(r"进展|恶化|progress|worsen", text):
        return "disease_progression"
    return "none"


def _spatial_scope(text: str) -> str:
    if re.search(r"roi|选定区域|指定区域", text):
        return "selected_roi"
    if re.search(r"肿瘤周围|瘤周|周边|peritumoral", text):
        return "peritumoral"
    if re.search(r"肿瘤内部|瘤内|intratumoral", text):
        return "intratumoral"
    if re.search(r"边界|外沿|前沿|boundary|front", text):
        return "boundary"
    if re.search(r"局部|local", text):
        return "local"
    if re.search(r"整个|整体|whole", text):
        return "whole_lesion"
    return "unspecified"


def _strength(text: str) -> str:
    if re.search(r"轻微|稍微|mild|slight", text):
        return "mild"
    if re.search(r"明显|强烈|大量|strong|marked|substantial", text):
        return "strong"
    if re.search(r"适中|中等|moderate", text):
        return "moderate"
    return "unspecified"


def _increase_or_decrease(text: str) -> str:
    if re.search(r"减少|降低|去掉|删除|decrease|reduce|remove", text):
        return "decrease"
    if re.search(r"增加|增多|提高|increase|add|more", text):
        return "increase"
    return "unspecified"


def _validate_relation_dag(
    intent_ids: tuple[str, ...], relations: tuple[IntentRelation, ...]
) -> None:
    graph = {key: [] for key in intent_ids}
    indegree = {key: 0 for key in intent_ids}
    for relation in relations:
        if relation.relation_type != "explicit_sequence":
            continue
        graph[relation.before_intent_id].append(relation.after_intent_id)
        indegree[relation.after_intent_id] += 1
    ready = [key for key, value in indegree.items() if value == 0]
    visited = 0
    while ready:
        current = ready.pop()
        visited += 1
        for target in graph[current]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if visited != len(intent_ids):
        raise JointContractError("semantic intent relations contain a cycle")


def _contains_forbidden_key(value: Any, forbidden: set[str]) -> bool:
    if isinstance(value, Mapping):
        return bool(set(value).intersection(forbidden)) or any(
            _contains_forbidden_key(item, forbidden) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_key(item, forbidden) for item in value)
    return False


def _require_enum(value: str, allowed: frozenset[str], label: str) -> None:
    if value not in allowed:
        raise JointContractError(f"semantic {label} is outside the closed ontology")


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) for item in value
    ):
        raise JointContractError("semantic string-list field is malformed")
    return tuple(item for item in value if item.strip())


def _instruction(value: str) -> str:
    text = " ".join(value.strip().split())
    if not text:
        raise JointContractError("instruction cannot be empty")
    return text

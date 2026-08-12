"""Digest-bound user clarification for executable primitive ambiguity.

Clarification is deliberately compiled *after* skill composition and
deterministic feasibility.  A user can therefore choose only between
pathology meanings that the current image/profile/tool stack can execute.
The decision locks an edit primitive, never pixels, numeric parameters or a
specific interface.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .models import JointCaseContext, JointContractError

CLARIFICATION_REQUEST_SCHEMA = "joint-primitive-clarification-request-v1"
CLARIFICATION_DECISION_SCHEMA = "joint-primitive-clarification-decision-v1"
SCENARIO_CLARIFICATION_REQUEST_SCHEMA = "joint-scenario-clarification-request-v1"
SCENARIO_CLARIFICATION_DECISION_SCHEMA = "joint-scenario-clarification-decision-v1"


@dataclass(frozen=True)
class PrimitiveClarificationOption:
    option_id: str
    clinician_label: str
    clinician_description: str
    primitive_id: str
    compatible_mechanism_ids: tuple[str, ...]
    expected_tissue_effect: str
    expected_cell_effect: str

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrimitiveClarificationRequest:
    clarification_id: str
    case_id: str
    instruction: str
    question: str
    why_required: str
    input_digests: dict[str, str]
    knowledge_context: dict[str, str]
    semantic_intent_sha256: str
    options: tuple[PrimitiveClarificationOption, ...]
    schema_version: str = CLARIFICATION_REQUEST_SCHEMA

    def to_metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["request_sha256"] = clarification_request_sha256(payload)
        return payload


@dataclass(frozen=True)
class ScenarioClarificationOption:
    option_id: str
    clinician_label: str
    clinician_description: str
    scenario: str
    clinical_direction: str
    treatment_context: str
    scenario_target: str
    explicit_edit_scope: str

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScenarioClarificationRequest:
    clarification_id: str
    case_id: str
    instruction: str
    question: str
    why_required: str
    input_digests: dict[str, str]
    knowledge_context: dict[str, str]
    options: tuple[ScenarioClarificationOption, ...]
    schema_version: str = SCENARIO_CLARIFICATION_REQUEST_SCHEMA

    def to_metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["request_sha256"] = clarification_request_sha256(payload)
        return payload


class PlannerClarificationRequired(JointContractError):
    """Signal that H&E cannot resolve two executable primitive meanings."""

    def __init__(
        self,
        reason: str,
        *,
        primitive_ids: Sequence[str],
    ) -> None:
        values = tuple(dict.fromkeys(str(value) for value in primitive_ids if value))
        if len(values) < 2 or len(values) > 3:
            raise JointContractError(
                "primitive clarification must name two or three alternatives"
            )
        self.reason = str(reason).strip()
        self.primitive_ids = values
        super().__init__(self.reason)


_PRIMITIVE_COPY: dict[str, tuple[str, str, str, str]] = {
    "tumor-burden-increase-v1": (
        "扩大肿瘤组织负荷",
        "改变肿瘤组织边界，使肿瘤占据更多相邻的合法组织区域。",
        "肿瘤组织区域扩大",
        "在新肿瘤区域内按匹配组织重新生成细胞",
    ),
    "tumor-burden-decrease-v1": (
        "缩小肿瘤组织负荷",
        "回退肿瘤组织边界，并在合法界面上恢复相邻组织。",
        "肿瘤组织区域缩小",
        "删除被替换区域内的完整肿瘤细胞并重建目标组织细胞",
    ),
    "invasive-front-expansion-v1": (
        "推进肿瘤浸润前缘",
        "沿当前可见且合法的浸润界面推进肿瘤，而不是制造远端肿瘤岛。",
        "浸润前缘发生组织级推进",
        "前缘内肿瘤细胞与目标组织共同重建",
    ),
    "neoplastic-microinfiltration-increase-v1": (
        "增加微小肿瘤浸润灶",
        "保持组织标签边界，在合法界面邻近区域增加散在或小簇肿瘤细胞。",
        "组织标签保持不变",
        "增加受约束的散在、小簇或短链肿瘤细胞",
    ),
    "cellularity-increase-v1": (
        "提高局部细胞密度",
        "保持组织结构，仅提高一个有明确空间锚点区域的细胞密度。",
        "组织标签保持不变",
        "在局部范围增加完整细胞实例",
    ),
    "cellularity-decrease-v1": (
        "降低局部细胞密度",
        "保持组织结构，使有病理锚点的局部区域出现渐变式细胞密度下降。",
        "组织标签保持不变",
        "按空间梯度删除完整细胞实例",
    ),
    "cell-type-abundance-increase-v1": (
        "增加指定细胞群",
        "保持组织结构，在合法区域增加用户指定且可观察的细胞类别。",
        "组织标签保持不变",
        "增加指定类别的完整细胞实例",
    ),
    "cell-type-abundance-decrease-v1": (
        "减少指定细胞群",
        "保持组织结构，在合法区域减少用户指定且可观察的细胞类别。",
        "组织标签保持不变",
        "删除指定类别的完整细胞实例",
    ),
    "necrosis-appearance-v1": (
        "增加坏死区域",
        "从现有合法界面扩展坏死，并保留坏死组织仍可含有核碎屑或残余细胞的条件。",
        "坏死组织区域扩大",
        "完整活性细胞被移除，坏死区域细胞条件按合同重新生成",
    ),
    "necrosis-resolution-v1": (
        "减少坏死区域",
        "从合法界面回退坏死，并以相邻可恢复组织替换。",
        "坏死组织区域缩小",
        "在恢复组织区域重新生成匹配细胞",
    ),
    "stroma-increase-v1": (
        "增加治疗相关纤维化间质",
        "仅在明确的治疗后语境和受支持机制下，以纤维化间质替换相应区域。",
        "受支持区域转为间质",
        "按治疗相关纤维化合同重建细胞组成",
    ),
    "architecture-progression-v1": (
        "改变肿瘤结构模式",
        "改变可被当前标注、辅助结构和生成器共同表达的肿瘤结构身份。",
        "细粒度结构标签和几何共同改变",
        "细胞沿新结构单元重新布局",
    ),
    "structural-void-spread-v1": (
        "增加原生腔隙内传播",
        "仅在原生腔隙有可靠辅助标注且生成器已验证响应时模拟特殊空间传播。",
        "主组织边界通常保持或按机制受限改变",
        "肿瘤细胞只进入被认证的原生腔隙",
    ),
}


def build_primitive_clarification_request(
    *,
    case: JointCaseContext,
    prepared_options: Sequence[Any],
    why_required: str,
    primitive_ids: Sequence[str],
) -> PrimitiveClarificationRequest:
    """Build one stable request from already executable interpretations."""

    requested = tuple(dict.fromkeys(str(value) for value in primitive_ids if value))
    if len(requested) < 2 or len(requested) > 3:
        raise JointContractError(
            "clarification request requires two or three primitive alternatives"
        )
    mechanism_ids: dict[str, set[str]] = defaultdict(set)
    executable_primitives = set()
    for option in prepared_options:
        primitive_id = str(option.primitive_id)
        executable_primitives.add(primitive_id)
        if primitive_id in requested:
            mechanism_ids[primitive_id].add(str(option.mechanism.mechanism_id))
    missing = sorted(set(requested) - executable_primitives)
    if missing:
        raise JointContractError(
            "clarification named non-executable primitives: " + ", ".join(missing)
        )
    options = tuple(
        _build_option(
            primitive_id,
            tuple(sorted(mechanism_ids[primitive_id])),
        )
        for primitive_id in requested
    )
    core = {
        "schema_version": CLARIFICATION_REQUEST_SCHEMA,
        "case_id": case.case_id,
        "instruction": case.instruction,
        "question": "您希望这次编辑主要表现哪一种变化？",
        "why_required": str(why_required).strip(),
        "input_digests": _input_digests(case),
        "knowledge_context": _knowledge_context(case),
        "semantic_intent_sha256": _sha256(case.semantic_intent),
        "options": [item.to_metadata() for item in options],
    }
    clarification_id = "clarify-" + _sha256(core)[:20]
    return PrimitiveClarificationRequest(
        clarification_id=clarification_id,
        case_id=case.case_id,
        instruction=case.instruction,
        question=core["question"],
        why_required=core["why_required"],
        input_digests=core["input_digests"],
        knowledge_context=core["knowledge_context"],
        semantic_intent_sha256=core["semantic_intent_sha256"],
        options=options,
    )


def create_clarification_decision(
    request: Mapping[str, Any],
    *,
    selected_option_id: str,
    responder: str,
    provider: str,
    rationale: str | None = None,
) -> dict[str, Any]:
    """Create an immutable answer that carries its complete source request."""

    normalized_request = validate_request_metadata(request)
    available = {
        str(item["option_id"])
        for item in normalized_request["options"]
    }
    if selected_option_id not in available:
        raise JointContractError("clarification selected an unavailable option")
    if not str(responder).strip() or not str(provider).strip():
        raise JointContractError(
            "clarification decision requires responder and provider provenance"
        )
    core = {
        "schema_version": CLARIFICATION_DECISION_SCHEMA,
        "request": normalized_request,
        "selected_option_id": selected_option_id,
        "responder": str(responder).strip(),
        "provider": str(provider).strip(),
        "rationale": str(rationale).strip() if rationale else None,
    }
    return {**core, "decision_sha256": _sha256(core)}


def build_scenario_clarification_request(
    *,
    case_id: str,
    instruction: str,
    input_digests: Mapping[str, str],
    knowledge_context: Mapping[str, str],
    why_required: str,
) -> ScenarioClarificationRequest:
    """Ask a clinician to resolve directionless post-treatment language.

    The answer supplies clinical semantics only. It never selects pixels,
    mechanisms, tool parameters or numeric edit strength.
    """

    options = (
        ScenarioClarificationOption(
            option_id="scenario:treatment_response",
            clinician_label="治疗后出现缓解",
            clinician_description="模拟肿瘤负荷或活性细胞减少，并允许受支持的治疗相关坏死或纤维化替代。",
            scenario="treatment_response",
            clinical_direction="improve",
            treatment_context="post_treatment",
            scenario_target="tumor",
            explicit_edit_scope="unspecified",
        ),
        ScenarioClarificationOption(
            option_id="scenario:post_treatment_progression",
            clinician_label="治疗后仍然进展",
            clinician_description="模拟治疗后肿瘤继续扩大、浸润或局部细胞负荷升高。",
            scenario="post_treatment_progression",
            clinical_direction="worsen",
            treatment_context="post_treatment",
            scenario_target="tumor",
            explicit_edit_scope="unspecified",
        ),
        ScenarioClarificationOption(
            option_id="scenario:residual_disease",
            clinician_label="治疗后仍有残余病灶",
            clinician_description="模拟治疗后仍保留肿瘤，但可表现为较低组织负荷或较低活性细胞密度。",
            scenario="residual_disease",
            clinical_direction="persist",
            treatment_context="post_treatment",
            scenario_target="tumor",
            explicit_edit_scope="unspecified",
        ),
    )
    core = {
        "schema_version": SCENARIO_CLARIFICATION_REQUEST_SCHEMA,
        "case_id": str(case_id),
        "instruction": str(instruction).strip(),
        "question": "您希望这次治疗后的变化主要表现为哪一种情况？",
        "why_required": str(why_required).strip(),
        "input_digests": dict(sorted(input_digests.items())),
        "knowledge_context": dict(sorted(knowledge_context.items())),
        "options": [item.to_metadata() for item in options],
    }
    return ScenarioClarificationRequest(
        clarification_id="clarify-scenario-" + _sha256(core)[:20],
        case_id=core["case_id"],
        instruction=core["instruction"],
        question=core["question"],
        why_required=core["why_required"],
        input_digests=core["input_digests"],
        knowledge_context=core["knowledge_context"],
        options=options,
    )


def create_scenario_clarification_decision(
    request: Mapping[str, Any],
    *,
    selected_option_id: str,
    responder: str,
    provider: str,
    rationale: str | None = None,
) -> dict[str, Any]:
    normalized = validate_scenario_request_metadata(request)
    if selected_option_id not in {
        str(item["option_id"]) for item in normalized["options"]
    }:
        raise JointContractError("scenario clarification selected an unavailable option")
    if not str(responder).strip() or not str(provider).strip():
        raise JointContractError(
            "scenario clarification requires responder and provider provenance"
        )
    core = {
        "schema_version": SCENARIO_CLARIFICATION_DECISION_SCHEMA,
        "request": normalized,
        "selected_option_id": selected_option_id,
        "responder": str(responder).strip(),
        "provider": str(provider).strip(),
        "rationale": str(rationale).strip() if rationale else None,
    }
    return {**core, "decision_sha256": _sha256(core)}


def resolve_scenario_clarification_decision(
    decision: Mapping[str, Any],
    *,
    case_id: str,
    instruction: str,
    input_digests: Mapping[str, str],
    knowledge_context: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, Any]]:
    if decision.get("schema_version") != SCENARIO_CLARIFICATION_DECISION_SCHEMA:
        raise JointContractError("unsupported scenario clarification decision schema")
    core = {
        key: decision.get(key)
        for key in (
            "schema_version",
            "request",
            "selected_option_id",
            "responder",
            "provider",
            "rationale",
        )
    }
    if decision.get("decision_sha256") != _sha256(core):
        raise JointContractError("scenario clarification decision digest mismatch")
    request = validate_scenario_request_metadata(decision.get("request"))
    if (
        request["case_id"] != str(case_id)
        or request["instruction"] != str(instruction).strip()
        or request["input_digests"] != dict(sorted(input_digests.items()))
        or request["knowledge_context"] != dict(sorted(knowledge_context.items()))
    ):
        raise JointContractError(
            "scenario clarification decision is detached from the current case"
        )
    selected = next(
        (
            item
            for item in request["options"]
            if item["option_id"] == decision.get("selected_option_id")
        ),
        None,
    )
    if selected is None:
        raise JointContractError("scenario clarification selected an unknown option")
    fields = {
        key: str(selected[key])
        for key in (
            "scenario",
            "clinical_direction",
            "treatment_context",
            "scenario_target",
            "explicit_edit_scope",
        )
    }
    usage = {
        "provider": str(decision["provider"]),
        "responder": str(decision["responder"]),
        "request_id": request["clarification_id"],
        "request_sha256": request["request_sha256"],
        "decision_sha256": str(decision["decision_sha256"]),
        "selected_option_id": str(decision["selected_option_id"]),
        "rationale": decision.get("rationale"),
    }
    return fields, usage


def resolve_clarification_decision(
    *,
    case: JointCaseContext,
    prepared_options: Sequence[Any],
) -> tuple[str, dict[str, Any]] | None:
    """Validate and resolve the case-bound primitive selected by the user."""

    raw = case.clarification_decision
    if not raw:
        return None
    if not isinstance(raw, Mapping):
        raise JointContractError("clarification_decision must be an object")
    if raw.get("schema_version") != CLARIFICATION_DECISION_SCHEMA:
        raise JointContractError("unsupported clarification decision schema")
    core = {key: raw.get(key) for key in (
        "schema_version",
        "request",
        "selected_option_id",
        "responder",
        "provider",
        "rationale",
    )}
    if raw.get("decision_sha256") != _sha256(core):
        raise JointContractError("clarification decision digest mismatch")
    request = validate_request_metadata(raw.get("request"))
    if (
        request["case_id"] != case.case_id
        or request["instruction"] != case.instruction
        or request["input_digests"] != _input_digests(case)
        or request["knowledge_context"] != _knowledge_context(case)
        or request["semantic_intent_sha256"]
        != _sha256(case.semantic_intent)
    ):
        raise JointContractError(
            "clarification decision is detached from the current case inputs"
        )
    selected_option_id = str(raw.get("selected_option_id") or "")
    selected = next(
        (
            item
            for item in request["options"]
            if item["option_id"] == selected_option_id
        ),
        None,
    )
    if selected is None:
        raise JointContractError("clarification decision selected an unknown option")
    primitive_id = str(selected["primitive_id"])
    executable = {
        str(option.mechanism.mechanism_id)
        for option in prepared_options
        if str(option.primitive_id) == primitive_id
    }
    if not executable:
        raise JointContractError(
            "clarification-selected primitive is no longer executable"
        )
    current_mechanisms = sorted(executable)
    if current_mechanisms != sorted(selected["compatible_mechanism_ids"]):
        raise JointContractError(
            "clarification options changed after the request was issued"
        )
    return primitive_id, {
        "provider": str(raw["provider"]),
        "responder": str(raw["responder"]),
        "request_id": request["clarification_id"],
        "request_sha256": request["request_sha256"],
        "decision_sha256": str(raw["decision_sha256"]),
        "selected_option_id": selected_option_id,
        "selected_primitive_id": primitive_id,
        "rationale": raw.get("rationale"),
    }


def validate_request_metadata(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise JointContractError("clarification request must be an object")
    normalized = dict(payload)
    if normalized.get("schema_version") != CLARIFICATION_REQUEST_SCHEMA:
        raise JointContractError("unsupported clarification request schema")
    required_request_fields = {
        "clarification_id",
        "case_id",
        "instruction",
        "question",
        "why_required",
        "input_digests",
        "knowledge_context",
        "semantic_intent_sha256",
        "options",
        "request_sha256",
    }
    if required_request_fields - set(normalized):
        raise JointContractError("clarification request is incomplete")
    identity_core = dict(normalized)
    identity_core.pop("clarification_id", None)
    identity_core.pop("request_sha256", None)
    if normalized.get("clarification_id") != (
        "clarify-" + _sha256(identity_core)[:20]
    ):
        raise JointContractError("clarification request identity mismatch")
    if normalized.get("request_sha256") != clarification_request_sha256(normalized):
        raise JointContractError("clarification request digest mismatch")
    options = normalized.get("options")
    if (
        not isinstance(options, Sequence)
        or isinstance(options, (str, bytes))
        or not 2 <= len(options) <= 3
    ):
        raise JointContractError("clarification request requires two or three options")
    normalized["options"] = [dict(option) for option in options]
    options = normalized["options"]
    required = {
        "option_id",
        "clinician_label",
        "clinician_description",
        "primitive_id",
        "compatible_mechanism_ids",
        "expected_tissue_effect",
        "expected_cell_effect",
    }
    for option in options:
        if not isinstance(option, Mapping) or required - set(option):
            raise JointContractError("clarification option is malformed")
        if not option["compatible_mechanism_ids"]:
            raise JointContractError(
                "clarification option has no executable mechanism"
            )
    return normalized


def validate_scenario_request_metadata(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise JointContractError("scenario clarification request must be an object")
    normalized = dict(payload)
    if normalized.get("schema_version") != SCENARIO_CLARIFICATION_REQUEST_SCHEMA:
        raise JointContractError("unsupported scenario clarification request schema")
    required_fields = {
        "clarification_id",
        "case_id",
        "instruction",
        "question",
        "why_required",
        "input_digests",
        "knowledge_context",
        "options",
        "request_sha256",
    }
    if required_fields - set(normalized):
        raise JointContractError("scenario clarification request is incomplete")
    identity_core = dict(normalized)
    identity_core.pop("clarification_id", None)
    identity_core.pop("request_sha256", None)
    if normalized["clarification_id"] != (
        "clarify-scenario-" + _sha256(identity_core)[:20]
    ):
        raise JointContractError("scenario clarification request identity mismatch")
    if normalized["request_sha256"] != clarification_request_sha256(normalized):
        raise JointContractError("scenario clarification request digest mismatch")
    options = normalized.get("options")
    if (
        not isinstance(options, Sequence)
        or isinstance(options, (str, bytes))
        or len(options) != 3
    ):
        raise JointContractError("scenario clarification requires three options")
    normalized["options"] = [dict(item) for item in options]
    required_option = {
        "option_id",
        "clinician_label",
        "clinician_description",
        "scenario",
        "clinical_direction",
        "treatment_context",
        "scenario_target",
        "explicit_edit_scope",
    }
    if any(required_option - set(item) for item in normalized["options"]):
        raise JointContractError("scenario clarification option is malformed")
    return normalized


def clarification_request_sha256(payload: Mapping[str, Any]) -> str:
    core = dict(payload)
    core.pop("request_sha256", None)
    return _sha256(core)


def _build_option(
    primitive_id: str,
    mechanism_ids: tuple[str, ...],
) -> PrimitiveClarificationOption:
    copy = _PRIMITIVE_COPY.get(primitive_id)
    if copy is None:
        raise JointContractError(
            f"primitive {primitive_id!r} lacks clinician-facing clarification copy"
        )
    label, description, tissue_effect, cell_effect = copy
    return PrimitiveClarificationOption(
        option_id=f"primitive:{primitive_id}",
        clinician_label=label,
        clinician_description=description,
        primitive_id=primitive_id,
        compatible_mechanism_ids=mechanism_ids,
        expected_tissue_effect=tissue_effect,
        expected_cell_effect=cell_effect,
    )


def _input_digests(case: JointCaseContext) -> dict[str, str]:
    values = {
        key: str(value)
        for key, value in case.provenance.items()
        if key.startswith("source_") and key.endswith("_sha256") and value
    }
    auxiliary = case.provenance.get("auxiliary_structure_sha256", {})
    if isinstance(auxiliary, Mapping):
        values.update(
            {
                f"auxiliary:{key}": str(value)
                for key, value in sorted(auxiliary.items())
                if value
            }
        )
    return dict(sorted(values.items()))


def _knowledge_context(case: JointCaseContext) -> dict[str, str]:
    return {
        "pathology_domain_id": case.pathology_domain_id,
        "annotation_profile_id": case.annotation_profile_id,
        "cell_observation_profile_id": case.cell_observation_profile_id,
        "cell_population_profile_id": case.cell_population_profile_id,
    }


def _sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

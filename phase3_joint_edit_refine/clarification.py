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
    """Signal that mask/annotation authority cannot resolve user intent."""

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
        "Increase tumor tissue burden",
        "Advance the tumor boundary into adjacent tissue that the active contracts permit.",
        "The tumor tissue footprint increases.",
        "Cells are regenerated in the new tumor region using the matched population profile.",
    ),
    "tumor-burden-decrease-v1": (
        "Decrease tumor tissue burden",
        "Retract the tumor boundary at a legal interface and restore the adjacent tissue class.",
        "The tumor tissue footprint decreases.",
        "Whole tumor-cell instances are removed from the converted region and target-tissue cells are regenerated.",
    ),
    "cohesive-boundary-expansion-v1": (
        "Expand the annotated tumor boundary",
        "Create a shallow connected expansion from a certified external Tumor-to-operational-Stroma boundary without diagnosing a histologic invasive front.",
        "The annotated invasive-tumor footprint expands locally.",
        "Whole incompatible instances are removed and complete neoplastic instances are regenerated under the target-population contract.",
    ),
    "infiltrative-nest-cord-extension-v1": (
        "Create a narrow connected tumor extension",
        "Create one tapered connected projection from a certified external boundary; this is synthetic mask geometry, not a histologic cord diagnosis.",
        "A narrow, single-parent tumor projection extends into operational Stroma.",
        "Complete neoplastic instances are regenerated within the connected projection under the target-population contract.",
    ),
    "peritumoral-neoplastic-scatter-increase-v1": (
        "Add sparse peritumoral neoplastic cells",
        "Keep tissue labels unchanged and add separated complete neoplastic instances in a certified outer annulus.",
        "Tissue labels remain unchanged.",
        "Frozen ProbNet ranks legal centers for separated complete class-1 instances.",
    ),
    "peritumoral-small-cluster-increase-v1": (
        "Add non-diagnostic small peritumoral clusters",
        "Keep tissue labels unchanged and add multiple non-diagnostic synthetic or budding-like foci of one to four complete neoplastic nuclei. This does not create a diagnostic tumor-budding score.",
        "Tissue labels remain unchanged.",
        "Frozen ProbNet ranks legal centers for multiple separated one-to-four-cell foci.",
    ),
    "invasive-tumor-footprint-decrease-v1": (
        "Decrease the invasive-tumor footprint",
        "Retract invasive tumor at a legal interface while preserving measurable residual tumor; this does not represent complete response for the case.",
        "The invasive-tumor footprint decreases while residual disease remains.",
        "Whole neoplastic-cell instances are removed from the converted region and non-neoplastic cells are regenerated.",
    ),
    "residual-tumor-fragmentation-v1": (
        "Create separated residual-tumor foci",
        "Convert contiguous invasive tumor into controlled residual foci only within the source tumor support.",
        "Contiguous tumor becomes separated residual foci that each meet a minimum area.",
        "Whole neoplastic-cell instances are removed from the intervening stromal channels and retained within residual foci.",
    ),
    "neoplastic-cell-abundance-decrease-v1": (
        "Decrease neoplastic-cell abundance",
        "Keep tissue architecture unchanged and reduce only the observable neoplastic-cell population.",
        "Tissue labels remain unchanged.",
        "Whole class-1 neoplastic nuclei are removed under a spatially graded field.",
    ),
    "neoplastic-cell-abundance-increase-v1": (
        "Increase neoplastic-cell abundance",
        "Keep tissue architecture unchanged and increase neoplastic-cell abundance within an existing tumor region.",
        "Tissue labels remain unchanged.",
        "Whole class-1 neoplastic nuclei are added using same-patch size and density priors.",
    ),
    "local-invasive-clearance-v1": (
        "Clear invasive tumor inside the selected region",
        "Clear invasive tumor only within an explicit user ROI; this does not represent pCR or complete response for the case.",
        "Invasive tumor is locally replaced inside the ROI and preserved outside it.",
        "Whole neoplastic-cell instances are removed inside the ROI and target-tissue cells are regenerated.",
    ),
    "invasive-front-expansion-v1": (
        "Advance the invasive front",
        "Advance tumor along a visible legal invasion interface without creating a distant tumor island.",
        "The invasive front advances at tissue level.",
        "Tumor cells and displaced target-tissue cells are jointly regenerated at the new front.",
    ),
    "neoplastic-microinfiltration-increase-v1": (
        "Increase microscopic neoplastic infiltration",
        "Keep tissue labels unchanged and add isolated or small neoplastic clusters near a legal interface.",
        "Tissue labels remain unchanged.",
        "Constrained single cells, small clusters, or short cords are added.",
    ),
    "cellularity-increase-v1": (
        "Increase local cellularity",
        "Keep tissue architecture unchanged and increase cellularity in a spatially anchored region.",
        "Tissue labels remain unchanged.",
        "Whole-cell instances are added within the local support.",
    ),
    "cellularity-decrease-v1": (
        "Decrease local cellularity",
        "Keep tissue architecture unchanged and create a graded cellularity reduction in a pathologically anchored region.",
        "Tissue labels remain unchanged.",
        "Whole-cell instances are removed under a spatial gradient.",
    ),
    "cell-type-abundance-increase-v1": (
        "Increase a specified cell population",
        "Keep tissue architecture unchanged and increase a user-specified observable cell class in a legal region.",
        "Tissue labels remain unchanged.",
        "Whole instances of the specified class are added.",
    ),
    "cell-type-abundance-decrease-v1": (
        "Decrease a specified cell population",
        "Keep tissue architecture unchanged and reduce a user-specified observable cell class in a legal region.",
        "Tissue labels remain unchanged.",
        "Whole instances of the specified class are removed.",
    ),
    "necrosis-appearance-v1": (
        "Increase the necrotic region",
        "Expand necrosis from an existing legal interface while allowing the necrotic condition to retain nuclear debris or residual cells.",
        "The necrotic tissue footprint increases.",
        "Whole viable cells are removed and the necrotic cellular condition is regenerated under contract.",
    ),
    "necrosis-resolution-v1": (
        "Decrease the necrotic region",
        "Retract necrosis at a legal interface and replace it with an adjacent recoverable tissue class.",
        "The necrotic tissue footprint decreases.",
        "Matched cells are regenerated in the restored tissue region.",
    ),
    "stroma-increase-v1": (
        "Increase post-treatment stromal replacement",
        "Replace local tumor with operational Stroma only under explicit post-treatment context and a supported mechanism; the mask does not claim fibrosis or a tumor bed.",
        "The supported region becomes operational Stroma.",
        "Whole neoplastic-cell instances are removed and the cellular condition is rebuilt from stromal priors.",
    ),
    "generic-immune-infiltrate-increase-v1": (
        "Increase the generic immune-infiltrate region",
        "Expand a generic immune region from an existing stroma-immune interface without inferring an immune subtype or treatment response.",
        "Stroma becomes generic immune-infiltrate tissue.",
        "Whole connective-tissue cells are removed from the converted region and class-2 inflammatory cells are regenerated.",
    ),
    "generic-immune-infiltrate-decrease-v1": (
        "Decrease the generic immune-infiltrate region",
        "Retract a generic immune region at an existing immune-stroma interface without claiming immune suppression or a treatment outcome.",
        "Generic immune-infiltrate tissue becomes Stroma.",
        "Whole class-2 inflammatory cells are removed and connective-tissue cells are regenerated.",
    ),
    "architecture-progression-v1": (
        "Change the tumor architecture pattern",
        "Change only an architectural identity jointly representable by the annotation, auxiliary structures, and generator.",
        "Fine architectural labels and geometry change together.",
        "Cells are relaid along the new architectural units.",
    ),
    "structural-void-spread-v1": (
        "Increase spread through an authenticated native void",
        "Simulate special spatial spread only when native voids have authoritative auxiliary labels and the generator has demonstrated a response.",
        "The main tissue boundary remains fixed or changes only as the mechanism permits.",
        "Neoplastic cells enter only authenticated native voids.",
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
    claim_downgrade = (
        requested == ("peritumoral-small-cluster-increase-v1",)
        and requires_budding_claim_downgrade(case.instruction)
    )
    if (len(requested) < 2 or len(requested) > 3) and not claim_downgrade:
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
        "question": (
            "Do you accept a non-diagnostic synthetic one-to-four-cell small-cluster representation?"
            if claim_downgrade
            else "Which change should this edit primarily represent?"
        ),
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
    available_scenarios: Sequence[str] | None = None,
) -> ScenarioClarificationRequest:
    """Ask a clinician to resolve directionless post-treatment language.

    The answer supplies clinical semantics only. It never selects pixels,
    mechanisms, tool parameters or numeric edit strength.
    """

    all_options = (
        ScenarioClarificationOption(
            option_id="scenario:treatment_response",
            clinician_label="Post-treatment response",
            clinician_description="Simulate reduced local tumor footprint or viable neoplastic-cell abundance, with only independently supported contextual changes. Stromal replacement does not imply fibrosis or a tumor bed.",
            scenario="treatment_response",
            clinical_direction="improve",
            treatment_context="post_treatment",
            scenario_target="tumor",
            explicit_edit_scope="unspecified",
        ),
        ScenarioClarificationOption(
            option_id="scenario:post_treatment_progression",
            clinician_label="Post-treatment progression",
            clinician_description="Simulate continued local tumor expansion, invasive-front advancement, microinfiltration, or increased neoplastic-cell abundance after treatment.",
            scenario="post_treatment_progression",
            clinical_direction="worsen",
            treatment_context="post_treatment",
            scenario_target="tumor",
            explicit_edit_scope="unspecified",
        ),
        ScenarioClarificationOption(
            option_id="scenario:residual_disease",
            clinician_label="Residual disease after treatment",
            clinician_description="Simulate persistent residual tumor with a protected residual floor, optionally represented by a smaller footprint, lower neoplastic-cell abundance, or bounded residual fragmentation.",
            scenario="residual_disease",
            clinical_direction="persist",
            treatment_context="post_treatment",
            scenario_target="tumor",
            explicit_edit_scope="unspecified",
        ),
    )
    allowed = (
        {str(value) for value in available_scenarios}
        if available_scenarios is not None
        else {item.scenario for item in all_options}
    )
    options = tuple(item for item in all_options if item.scenario in allowed)
    if not options:
        raise JointContractError(
            "no post-treatment scenario survives deterministic execution preflight"
        )
    core = {
        "schema_version": SCENARIO_CLARIFICATION_REQUEST_SCHEMA,
        "case_id": str(case_id),
        "instruction": str(instruction).strip(),
        "question": "Which post-treatment course should this edit represent?",
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
        or not 1 <= len(options) <= 3
    ):
        raise JointContractError("clarification request requires one to three options")
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
    if len(options) == 1 and (
        options[0].get("primitive_id")
        != "peritumoral-small-cluster-increase-v1"
        or not requires_budding_claim_downgrade(str(normalized["instruction"]))
    ):
        raise JointContractError(
            "a single-option clarification is reserved for diagnostic-claim downgrade acceptance"
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
        or not 1 <= len(options) <= 3
    ):
        raise JointContractError(
            "scenario clarification requires one to three executable options"
        )
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


def requires_budding_claim_downgrade(instruction: str) -> bool:
    import re

    return bool(
        re.search(r"\btumou?r budding\b|\btumou?r buds?\b", instruction, re.IGNORECASE)
        or re.search(r"\u80bf\u7624\u51fa\u82bd", instruction)
    )

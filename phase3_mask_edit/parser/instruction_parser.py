"""Single-instruction parser for Phase 3 natural language edit requests."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from phase3_mask_edit.parser.semantic_diff import (
    DEFAULT_SEMANTIC_DIFF,
    SEMANTIC_DIFF_SCHEMA_VERSION,
    SemanticDiffValidationError,
    extract_json_object,
    normalize_semantic_diff,
    semantic_diff_response_format,
)


def _instruction_example(**sections: Mapping[str, str]) -> str:
    payload = json.loads(json.dumps(DEFAULT_SEMANTIC_DIFF))
    for section, values in sections.items():
        payload[section].update(values)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


INSTRUCTION_SYSTEM_PROMPT = f"""You convert one user pathology mask edit instruction into a Phase3 semantic-diff JSON object.

Choose only changes directly requested by the instruction. Output every field in schema version 0.2. Use "none" or "unspecified" rather than inferring an unstated change.

Interpretation rules:
- Tumor growth is only tumor amount, area, extent, expansion, regression, or shrinkage. A phenotype/grade transition alone is not growth.
- Necrosis refers to necrotic/dead tissue. Larger necrosis means necrosis_change.action = "increase", not tumor growth.
- Replacement/backfill targets are not separate edits. Immune, necrotic, or tumor tissue replaced by stroma keeps stroma_change.density = "none" unless desmoplasia, fibrosis, or stromal reaction is independently requested.
- lymphocyte_change.location is "intratumoral" only inside/within tumor, "peritumoral" around tumor, "stromal" in stroma, otherwise "unspecified".
- Fine transitions must use one exact transition_change pair:
  benign_epithelium -> gleason_pattern_3
  benign_epithelium -> stromal_tissue
  gleason_pattern_3 -> gleason_pattern_4
  gleason_pattern_4 -> gleason_pattern_5
  gleason_pattern_4 -> gleason_pattern_3
  normal_gland -> adenomatous_gland
  adenomatous_gland -> moderately_differentiated_carcinoma
  moderately_differentiated_carcinoma -> poorly_differentiated_carcinoma
  poorly_differentiated_carcinoma -> moderately_differentiated_carcinoma
- For all other instructions, transition_change.source_state = transition_change.target_state = "none".
- Set grade_change to upgrade/downgrade for grade-bearing transitions, but keep it "none" for benign epithelium -> stromal tissue.
- Use mild for small/slight/focal, moderate for unspecified or moderate, and significant for marked/strong/extensive/substantial.
- For necrosis extent, mild -> focal, moderate -> moderate, significant -> extensive.
- If only appearance/style changes without a supported semantic edit, leave all fields at defaults.

Examples:
User: "make the dead-looking region a lot bigger"
JSON: {_instruction_example(necrosis_change={"action": "increase", "extent": "extensive"})}

User: "stronger intratumoral immune presence"
JSON: {_instruction_example(lymphocyte_change={"infiltration": "increase", "degree": "significant", "location": "intratumoral"})}

User: "decrease the lymphocytic immune infiltrate and replace it with stromal tissue"
JSON: {_instruction_example(lymphocyte_change={"infiltration": "decrease", "degree": "moderate", "location": "unspecified"})}

User: "upgrade prostate tumor from Gleason pattern 3 to pattern 4"
JSON: {_instruction_example(tumor_change={"growth": "none", "degree": "moderate", "grade_change": "upgrade"}, transition_change={"source_state": "gleason_pattern_3", "target_state": "gleason_pattern_4", "degree": "moderate"})}

User: "convert normal colonic glands into adenomatous glands"
JSON: {_instruction_example(tumor_change={"growth": "none", "degree": "moderate", "grade_change": "upgrade"}, transition_change={"source_state": "normal_gland", "target_state": "adenomatous_gland", "degree": "moderate"})}

User: "replace benign prostate epithelium with stromal tissue"
JSON: {_instruction_example(transition_change={"source_state": "benign_epithelium", "target_state": "stromal_tissue", "degree": "moderate"})}

Output JSON only. No markdown."""


@dataclass(frozen=True)
class InstructionParserConfig:
    model: str
    api_base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_sec: float = 120.0
    max_retries: int = 2
    retry_backoff_sec: float = 2.0
    temperature: float = 0.0
    debug_dir: str | None = None


class InstructionParserError(RuntimeError):
    """Raised when an instruction cannot be parsed into a semantic diff."""


def parse_instruction_rule_based(instruction: str) -> dict[str, Any]:
    """Parse common English/Chinese edit instructions without model access."""

    text = _normalize_text(instruction)
    if not text:
        raise InstructionParserError("instruction is required.")

    semantic_diff = json.loads(json.dumps(DEFAULT_SEMANTIC_DIFF))
    strength = _strength_from_text(text)
    necrosis_extent = _necrosis_extent_from_strength(strength)

    if _mentions_any(text, _TUMOR_TERMS):
        if _mentions_any(text, _INCREASE_TERMS) and not _mentions_any(
            text, _DECREASE_TERMS
        ):
            semantic_diff["tumor_change"]["growth"] = "increase"
            semantic_diff["tumor_change"]["degree"] = strength
        elif _mentions_any(text, _DECREASE_TERMS) and not _mentions_any(
            text, _INCREASE_TERMS
        ):
            semantic_diff["tumor_change"]["growth"] = "decrease"
            semantic_diff["tumor_change"]["degree"] = strength

    if _mentions_any(text, _NECROSIS_TERMS):
        if _mentions_any(text, _REMOVE_TERMS):
            semantic_diff["necrosis_change"]["action"] = "remove"
            semantic_diff["necrosis_change"]["extent"] = necrosis_extent
        elif _mentions_any(text, _DECREASE_TERMS) and not _mentions_any(
            text, _INCREASE_TERMS
        ):
            semantic_diff["necrosis_change"]["action"] = "decrease"
            semantic_diff["necrosis_change"]["extent"] = necrosis_extent
        elif _mentions_any(text, _ADD_TERMS):
            semantic_diff["necrosis_change"]["action"] = "add"
            semantic_diff["necrosis_change"]["extent"] = necrosis_extent
        elif _mentions_any(text, _INCREASE_TERMS):
            semantic_diff["necrosis_change"]["action"] = "increase"
            semantic_diff["necrosis_change"]["extent"] = necrosis_extent

    if _mentions_any(text, _IMMUNE_TERMS):
        semantic_diff["lymphocyte_change"]["location"] = _immune_location_from_text(
            text
        )
        if _mentions_any(text, _DECREASE_TERMS) and not _mentions_any(
            text, _INCREASE_TERMS
        ):
            semantic_diff["lymphocyte_change"]["infiltration"] = "decrease"
            semantic_diff["lymphocyte_change"]["degree"] = strength
        elif _mentions_any(text, _INCREASE_TERMS) or _mentions_any(text, _ADD_TERMS):
            semantic_diff["lymphocyte_change"]["infiltration"] = "increase"
            semantic_diff["lymphocyte_change"]["degree"] = strength

    if _mentions_any(text, _STROMA_TERMS):
        stroma_is_immune_backfill = (
            semantic_diff["lymphocyte_change"]["infiltration"] == "decrease"
            and _mentions_any(text, _REPLACEMENT_TERMS)
            and _mentions_any(text, _IMMUNE_TERMS)
            and _mentions_any(text, _STROMA_TERMS)
            and not _mentions_any(text, _INDEPENDENT_STROMA_TERMS)
        )
        stroma_is_necrosis_backfill = (
            semantic_diff["necrosis_change"]["action"] in {"decrease", "remove"}
            and _mentions_any(text, _REPLACEMENT_TERMS + _NECROSIS_RESOLUTION_TERMS)
            and _mentions_any(text, _NECROSIS_TERMS)
            and _mentions_any(text, _STROMA_TERMS)
            and not _mentions_any(text, _INDEPENDENT_STROMA_TERMS)
        )
        stroma_is_tumor_backfill = (
            semantic_diff["tumor_change"]["growth"] == "decrease"
            and _mentions_any(text, _REPLACEMENT_TERMS)
            and _mentions_any(text, _TUMOR_TERMS)
            and _mentions_any(text, _STROMA_TERMS)
            and not _mentions_any(text, _INDEPENDENT_STROMA_TERMS)
        )
        if (
            stroma_is_immune_backfill
            or stroma_is_necrosis_backfill
            or stroma_is_tumor_backfill
        ):
            pass
        elif _mentions_any(text, _DECREASE_TERMS) and not _mentions_any(
            text, _INCREASE_TERMS
        ):
            semantic_diff["stroma_change"]["density"] = "decrease"
            semantic_diff["stroma_change"]["degree"] = strength
        elif _mentions_any(text, _INCREASE_TERMS) or _mentions_any(text, _ADD_TERMS):
            semantic_diff["stroma_change"]["density"] = "increase"
            semantic_diff["stroma_change"]["degree"] = strength

    transition = _fine_transition_from_text(text)
    if transition is not None:
        source_state, target_state, grade_change = transition
        semantic_diff["transition_change"] = {
            "source_state": source_state,
            "target_state": target_state,
            "degree": strength,
        }
        semantic_diff["tumor_change"]["grade_change"] = grade_change
        semantic_diff["tumor_change"]["growth"] = "none"
        semantic_diff["tumor_change"]["degree"] = strength

    normalized = normalize_semantic_diff(semantic_diff, fill_missing=True)
    if normalized == DEFAULT_SEMANTIC_DIFF:
        raise InstructionParserError(
            "Could not infer a supported edit from the instruction."
        )
    return normalized


def parse_instruction_with_api(
    instruction: str,
    *,
    config: InstructionParserConfig,
    repair_feedback: Mapping[str, Any] | None = None,
    previous_semantic_diff: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse a single edit instruction using an OpenAI-compatible chat API."""

    if not instruction.strip():
        raise InstructionParserError("instruction is required.")
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise InstructionParserError(
            f"Missing API key environment variable: {config.api_key_env}"
        )

    request_payload = {
        "model": config.model,
        "temperature": config.temperature,
        "messages": [
            {"role": "system", "content": INSTRUCTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _instruction_api_prompt(
                    instruction,
                    repair_feedback=repair_feedback,
                    previous_semantic_diff=previous_semantic_diff,
                ),
            },
        ],
        "response_format": semantic_diff_response_format(),
    }
    debug_dir = Path(config.debug_dir) if config.debug_dir else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "instruction_parser_request.json").write_text(
            json.dumps(request_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    response_payload = _post_chat_completion(
        request_payload,
        api_base_url=config.api_base_url,
        api_key=api_key,
        timeout_sec=config.timeout_sec,
        max_retries=config.max_retries,
        retry_backoff_sec=config.retry_backoff_sec,
        debug_dir=debug_dir,
    )
    content = _response_content(response_payload)
    if debug_dir is not None:
        (debug_dir / "instruction_parser_content.txt").write_text(
            content,
            encoding="utf-8",
        )
    try:
        parsed = extract_json_object(content)
        parsed.setdefault("schema_version", SEMANTIC_DIFF_SCHEMA_VERSION)
        return normalize_semantic_diff(parsed, fill_missing=True)
    except SemanticDiffValidationError as exc:
        raise InstructionParserError(
            f"API response did not match semantic_diff schema: {exc}"
        ) from exc


def parse_instruction(
    instruction: str,
    *,
    mode: str = "rule-based",
    config: InstructionParserConfig | None = None,
    repair_feedback: Mapping[str, Any] | None = None,
    previous_semantic_diff: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse a single natural-language edit instruction."""

    if mode == "rule-based":
        return parse_instruction_rule_based(instruction)
    if mode == "api":
        if config is None:
            raise InstructionParserError("config is required for api mode.")
        return parse_instruction_with_api(
            instruction,
            config=config,
            repair_feedback=repair_feedback,
            previous_semantic_diff=previous_semantic_diff,
        )
    raise InstructionParserError(f"Unsupported instruction parser mode: {mode}")


def _instruction_api_prompt(
    instruction: str,
    *,
    repair_feedback: Mapping[str, Any] | None,
    previous_semantic_diff: Mapping[str, Any] | None,
) -> str:
    if not repair_feedback:
        return instruction
    return (
        instruction
        + "\n\nThe previous semantic-diff output could not produce an executable "
        "mask-edit intent. Re-read the instruction and correct any missed or "
        "misclassified explicit edit. Do not invent a change merely to satisfy "
        "the planner. Return the complete semantic-diff schema."
        + "\n\nPREVIOUS SEMANTIC DIFF:\n"
        + json.dumps(previous_semantic_diff or {}, indent=2, ensure_ascii=False)
        + "\n\nDOWNSTREAM PLANNER FEEDBACK:\n"
        + json.dumps(repair_feedback, indent=2, ensure_ascii=False)
    )


def _post_chat_completion(
    payload: Mapping[str, Any],
    *,
    api_base_url: str,
    api_key: str,
    timeout_sec: float,
    max_retries: int,
    retry_backoff_sec: float,
    debug_dir: Path | None = None,
) -> dict[str, Any]:
    endpoint = api_base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    response_data = ""
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(
            endpoint,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                response_data = response.read().decode("utf-8")
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if debug_dir is not None:
                (
                    debug_dir / f"instruction_parser_http_error_{attempt + 1}.txt"
                ).write_text(
                    body,
                    encoding="utf-8",
                )
            retryable = exc.code in {408, 429, 500, 502, 503, 504}
            if not retryable or attempt >= max_retries:
                raise InstructionParserError(
                    f"API request failed with HTTP {exc.code}: {body}"
                ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt >= max_retries:
                raise InstructionParserError(f"API request failed: {exc}") from exc
        time.sleep(retry_backoff_sec * (attempt + 1))

    if debug_dir is not None:
        (debug_dir / "instruction_parser_response_raw.txt").write_text(
            response_data,
            encoding="utf-8",
        )
    try:
        decoded = json.loads(response_data)
    except json.JSONDecodeError as exc:
        raise InstructionParserError("API response was not valid JSON.") from exc
    if not isinstance(decoded, dict):
        raise InstructionParserError("API response root must be a JSON object.")
    return decoded


def _response_content(response_payload: Mapping[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise InstructionParserError("API response missing choices.")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise InstructionParserError("API response choice must be a mapping.")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise InstructionParserError("API response choice missing message.")
    content = message.get("content")
    if not isinstance(content, str):
        raise InstructionParserError("API response message content must be a string.")
    return content


def _normalize_text(value: str) -> str:
    return value.strip().lower().replace("-", " ").replace("_", " ")


def _mentions_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _strength_from_text(text: str) -> str:
    if _mentions_any(text, _MODERATE_TERMS):
        return "moderate"
    if _mentions_any(text, _SIGNIFICANT_TERMS):
        return "significant"
    if _mentions_any(text, _MILD_TERMS):
        return "mild"
    return "moderate"


def _necrosis_extent_from_strength(strength: str) -> str:
    if strength == "mild":
        return "focal"
    if strength == "significant":
        return "extensive"
    return "moderate"


def _immune_location_from_text(text: str) -> str:
    if _mentions_any(
        text,
        ("intratumoral", "inside tumor", "within tumor", "among tumor"),
    ):
        return "intratumoral"
    if _mentions_any(text, ("peritumoral", "around tumor", "surrounding tumor")):
        return "peritumoral"
    if _mentions_any(text, ("stromal", "in stroma", "within stroma")):
        return "stromal"
    return "unspecified"


def _fine_transition_from_text(
    text: str,
) -> tuple[str, str, str] | None:
    if _ordered_mentions(
        text, ("benign epithelium", "normal epithelium"), ("stroma", "stromal tissue")
    ):
        return "benign_epithelium", "stromal_tissue", "none"
    if _ordered_mentions(
        text,
        ("benign epithelium", "normal epithelium"),
        ("gleason pattern 3", "gleason 3", "pattern 3"),
    ):
        return "benign_epithelium", "gleason_pattern_3", "upgrade"
    if _ordered_mentions(
        text,
        ("gleason pattern 3", "gleason 3", "pattern 3"),
        ("gleason pattern 4", "gleason 4", "pattern 4"),
    ):
        return "gleason_pattern_3", "gleason_pattern_4", "upgrade"
    if _ordered_mentions(
        text,
        ("gleason pattern 4", "gleason 4", "pattern 4"),
        ("gleason pattern 5", "gleason 5", "pattern 5"),
    ):
        return "gleason_pattern_4", "gleason_pattern_5", "upgrade"
    if _ordered_mentions(
        text,
        ("gleason pattern 4", "gleason 4", "pattern 4"),
        ("gleason pattern 3", "gleason 3", "pattern 3"),
    ):
        return "gleason_pattern_4", "gleason_pattern_3", "downgrade"
    if _ordered_mentions(
        text, ("normal gland", "normal colonic gland"), ("adenoma", "adenomatous gland")
    ):
        return "normal_gland", "adenomatous_gland", "upgrade"
    if _ordered_mentions(
        text,
        ("adenoma", "adenomatous gland"),
        ("moderately differentiated", "moderate differentiation"),
    ):
        return (
            "adenomatous_gland",
            "moderately_differentiated_carcinoma",
            "upgrade",
        )
    if _ordered_mentions(
        text,
        ("moderately differentiated", "moderate differentiation"),
        ("poorly differentiated", "poor differentiation"),
    ):
        return (
            "moderately_differentiated_carcinoma",
            "poorly_differentiated_carcinoma",
            "upgrade",
        )
    if _ordered_mentions(
        text,
        ("poorly differentiated", "poor differentiation"),
        ("moderately differentiated", "moderate differentiation"),
    ):
        return (
            "poorly_differentiated_carcinoma",
            "moderately_differentiated_carcinoma",
            "downgrade",
        )
    return None


def _ordered_mentions(
    text: str,
    source_terms: tuple[str, ...],
    target_terms: tuple[str, ...],
) -> bool:
    source_positions = [text.find(term) for term in source_terms if term in text]
    target_positions = [text.find(term) for term in target_terms if term in text]
    return bool(
        source_positions
        and target_positions
        and min(source_positions) < max(target_positions)
    )


_TUMOR_TERMS = (
    "tumor",
    "tumour",
    "neoplasm",
    "cancer",
    "carcinoma",
    "肿瘤",
    "癌",
    "癌巢",
)
_NECROSIS_TERMS = (
    "necrosis",
    "necrotic",
    "debris",
    "dead tissue",
    "dead-looking",
    "dead looking",
    "坏死",
)
_IMMUNE_TERMS = (
    "immune",
    "lymphocyte",
    "lymphocytic",
    "til",
    "tils",
    "infiltrate",
    "炎症",
    "免疫",
    "淋巴",
    "浸润",
)
_STROMA_TERMS = (
    "stroma",
    "stromal",
    "desmoplasia",
    "fibrosis",
    "fibrotic",
    "connective",
    "间质",
    "纤维",
)
_REPLACEMENT_TERMS = (
    "replace",
    "replaced",
    "replacement",
    "backfill",
    "backfilled",
    "fill with",
    "filled with",
    "convert",
    "converted",
    "turn into",
    "turned into",
)
_NECROSIS_RESOLUTION_TERMS = ("resolve", "resolved", "resolution")
_INDEPENDENT_STROMA_TERMS = (
    "desmoplasia",
    "desmoplastic",
    "stromal reaction",
    "stromal response",
    "fibrosis",
    "fibrotic",
    "collagenous",
)
_INCREASE_TERMS = (
    "increase",
    "increased",
    "larger",
    "more",
    "expand",
    "expanded",
    "growth",
    "higher",
    "greater",
    "bigger",
    "enlarge",
    "enlarged",
    "增大",
    "增加",
    "更多",
    "更大",
    "大一点",
    "多一点",
    "变大",
    "扩大",
    "上升",
    "提高",
)
_DECREASE_TERMS = (
    "decrease",
    "decreased",
    "less",
    "smaller",
    "reduce",
    "reduced",
    "regress",
    "regression",
    "shrink",
    "resolution",
    "resolve",
    "减少",
    "更少",
    "缩小",
    "降低",
    "消退",
    "减小",
)
_ADD_TERMS = ("add", "appearance", "appear", "new", "create", "出现", "新增", "生成")
_REMOVE_TERMS = ("remove", "absent", "none", "消除", "去除", "没有", "无")
_MILD_TERMS = ("mild", "slight", "small", "focal", "一点", "轻度", "少量", "稍微")
_MODERATE_TERMS = ("moderate", "medium", "适中", "中等", "中度")
_SIGNIFICANT_TERMS = (
    "significant",
    "marked",
    "extensive",
    "large",
    "dramatic",
    "显著",
    "大量",
    "明显",
    "重度",
    "广泛",
)

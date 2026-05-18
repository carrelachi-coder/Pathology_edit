"""Single-instruction parser for Phase 3 natural language edit requests."""

from __future__ import annotations

import json
import os
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
)


INSTRUCTION_SYSTEM_PROMPT = """You convert one user pathology mask edit instruction into a Phase3 semantic-diff JSON object.

The user may write short, informal, or non-template natural language. They may
use synonyms instead of schema words, for example "a bit more dead tissue",
"make the cancer less prominent", "stronger immune presence", or "fibrous
reaction should be lighter".

Choose only semantic changes that the instruction directly requests. Output ONLY JSON matching this schema:
{
  "schema_version": "0.1",
  "tumor_change": {
    "growth": "none" | "increase" | "decrease",
    "degree": "mild" | "moderate" | "significant",
    "grade_change": "none" | "upgrade" | "downgrade"
  },
  "lymphocyte_change": {
    "infiltration": "none" | "increase" | "decrease",
    "degree": "mild" | "moderate" | "significant"
  },
  "necrosis_change": {
    "action": "none" | "add" | "increase" | "decrease" | "remove",
    "extent": "focal" | "moderate" | "extensive"
  },
  "stroma_change": {
    "density": "none" | "increase" | "decrease",
    "degree": "mild" | "moderate" | "significant"
  }
}

Interpretation rules:
- Tumor burden is about tumor amount, size, prominence, expansion, regression,
  shrinkage, or replacement by non-tumor tissue.
- Necrosis is about necrotic/dead tissue appearing, expanding, becoming larger,
  resolving, disappearing, or being removed. "Larger necrosis" means
  necrosis_change.action = "increase", not tumor growth.
- Lymphocyte change is about immune infiltrate, TILs, lymphocytes, inflammation,
  immune presence, immune cells, or intratumoral immune infiltration.
- Stroma change is about stroma, stromal reaction, desmoplasia, fibrosis,
  fibrous tissue, or connective tissue density.
- Replacement/backfill targets are not separate edits. If the instruction says
  immune infiltrate decreases and is replaced/backfilled/converted into stroma,
  set lymphocyte_change.infiltration = "decrease" and stroma_change.density =
  "none" unless it separately asks for desmoplasia, fibrosis, or stromal
  reaction.
- Dataset-specialized fine-ID edits are represented through
  tumor_change.grade_change. The schema has no direct primitive-name field, so
  do not invent one. Use grade_change = "upgrade" or "downgrade" and keep tumor
  growth = "none" unless the instruction also requests an area/size change.
- PANDA special edits: Gleason pattern 3 to 4, Gleason 4 to 5, grade group
  increase, or benign epithelium becoming Gleason 3 / low-grade malignant gland
  are grade_change = "upgrade". Gleason pattern 4 to 3 or lower grade is
  grade_change = "downgrade".
- GlaS special edits: normal gland becoming adenomatous, adenoma/adenomatous
  gland becoming carcinoma, moderately differentiated becoming poorly
  differentiated, or high-grade transformation are grade_change = "upgrade".
  Poorly differentiated shifting toward moderate differentiation is
  grade_change = "downgrade".
- Use "mild" for small, slight, subtle, a bit, a little, tiny, limited, or focal.
- Use "moderate" for medium, moderate, noticeable but not marked, balanced, or
  unspecified degree.
- Use "significant" for marked, strong, dramatic, extensive, widespread, large,
  a lot, much more, much less, or substantially changed.
- For necrosis extent, map mild -> focal, moderate -> moderate, significant -> extensive.
- If an instruction says tumor becomes more because necrosis decreases, include
  both tumor growth increase and necrosis decrease. If it only says tumor more,
  set only tumor growth increase.
- If an instruction only describes appearance/style without one of the supported
  semantic changes, leave every field at "none" defaults.

Few-shot examples. The interpretation notes are guidance only; never output them.

User: "make the dead-looking region a lot bigger"
Note: "dead-looking region" refers to necrosis; "a lot bigger" means significant increase.
JSON:
{"schema_version":"0.1","tumor_change":{"growth":"none","degree":"mild","grade_change":"none"},"lymphocyte_change":{"infiltration":"none","degree":"mild"},"necrosis_change":{"action":"increase","extent":"extensive"},"stroma_change":{"density":"none","degree":"moderate"}}

User: "let the cancer occupy just a little more space"
Note: "cancer occupy more space" refers to mild tumor growth.
JSON:
{"schema_version":"0.1","tumor_change":{"growth":"increase","degree":"mild","grade_change":"none"},"lymphocyte_change":{"infiltration":"none","degree":"mild"},"necrosis_change":{"action":"none","extent":"focal"},"stroma_change":{"density":"none","degree":"moderate"}}

User: "the tissue should look less dead, with viable tumor replacing it"
Note: This directly requests necrosis decrease and tumor increase.
JSON:
{"schema_version":"0.1","tumor_change":{"growth":"increase","degree":"moderate","grade_change":"none"},"lymphocyte_change":{"infiltration":"none","degree":"mild"},"necrosis_change":{"action":"decrease","extent":"moderate"},"stroma_change":{"density":"none","degree":"moderate"}}

User: "stronger intratumoral immune presence"
Note: Immune presence refers to lymphocyte/TIL infiltration.
JSON:
{"schema_version":"0.1","tumor_change":{"growth":"none","degree":"mild","grade_change":"none"},"lymphocyte_change":{"infiltration":"increase","degree":"significant"},"necrosis_change":{"action":"none","extent":"focal"},"stroma_change":{"density":"none","degree":"moderate"}}

User: "decrease the lymphocytic immune infiltrate and replace it with stromal tissue"
Note: Stroma is the backfill target for the immune decrease, not a separate stromal desmoplasia edit.
JSON:
{"schema_version":"0.1","tumor_change":{"growth":"none","degree":"mild","grade_change":"none"},"lymphocyte_change":{"infiltration":"decrease","degree":"moderate"},"necrosis_change":{"action":"none","extent":"focal"},"stroma_change":{"density":"none","degree":"moderate"}}

User: "make the fibrous background lighter"
Note: Fibrous background refers to stromal/fibrotic density decrease.
JSON:
{"schema_version":"0.1","tumor_change":{"growth":"none","degree":"mild","grade_change":"none"},"lymphocyte_change":{"infiltration":"none","degree":"mild"},"necrosis_change":{"action":"none","extent":"focal"},"stroma_change":{"density":"decrease","degree":"moderate"}}

User: "upgrade prostate tumor from Gleason pattern 3 to pattern 4"
Note: PANDA Gleason 3 to 4 is a fine-ID grade upgrade, not tumor area growth.
JSON:
{"schema_version":"0.1","tumor_change":{"growth":"none","degree":"moderate","grade_change":"upgrade"},"lymphocyte_change":{"infiltration":"none","degree":"mild"},"necrosis_change":{"action":"none","extent":"focal"},"stroma_change":{"density":"none","degree":"moderate"}}

User: "convert normal colonic glands into adenomatous glands"
Note: GlaS normal to adenomatous is a fine-ID grade/progression upgrade.
JSON:
{"schema_version":"0.1","tumor_change":{"growth":"none","degree":"moderate","grade_change":"upgrade"},"lymphocyte_change":{"infiltration":"none","degree":"mild"},"necrosis_change":{"action":"none","extent":"focal"},"stroma_change":{"density":"none","degree":"moderate"}}

User: "make the moderately differentiated colorectal carcinoma poorly differentiated"
Note: GlaS moderate to poor differentiation is a grade upgrade.
JSON:
{"schema_version":"0.1","tumor_change":{"growth":"none","degree":"moderate","grade_change":"upgrade"},"lymphocyte_change":{"infiltration":"none","degree":"mild"},"necrosis_change":{"action":"none","extent":"focal"},"stroma_change":{"density":"none","degree":"moderate"}}

Output JSON only. No markdown."""


@dataclass(frozen=True)
class InstructionParserConfig:
    model: str
    api_base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_sec: float = 60.0
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
        if _mentions_any(text, _INCREASE_TERMS) and not _mentions_any(text, _DECREASE_TERMS):
            semantic_diff["tumor_change"]["growth"] = "increase"
            semantic_diff["tumor_change"]["degree"] = strength
        elif _mentions_any(text, _DECREASE_TERMS) and not _mentions_any(text, _INCREASE_TERMS):
            semantic_diff["tumor_change"]["growth"] = "decrease"
            semantic_diff["tumor_change"]["degree"] = strength

    if _mentions_any(text, _NECROSIS_TERMS):
        if _mentions_any(text, _REMOVE_TERMS):
            semantic_diff["necrosis_change"]["action"] = "remove"
            semantic_diff["necrosis_change"]["extent"] = necrosis_extent
        elif _mentions_any(text, _DECREASE_TERMS) and not _mentions_any(text, _INCREASE_TERMS):
            semantic_diff["necrosis_change"]["action"] = "decrease"
            semantic_diff["necrosis_change"]["extent"] = necrosis_extent
        elif _mentions_any(text, _ADD_TERMS):
            semantic_diff["necrosis_change"]["action"] = "add"
            semantic_diff["necrosis_change"]["extent"] = necrosis_extent
        elif _mentions_any(text, _INCREASE_TERMS):
            semantic_diff["necrosis_change"]["action"] = "increase"
            semantic_diff["necrosis_change"]["extent"] = necrosis_extent

    if _mentions_any(text, _IMMUNE_TERMS):
        if _mentions_any(text, _DECREASE_TERMS) and not _mentions_any(text, _INCREASE_TERMS):
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
        if stroma_is_immune_backfill:
            pass
        elif _mentions_any(text, _DECREASE_TERMS) and not _mentions_any(text, _INCREASE_TERMS):
            semantic_diff["stroma_change"]["density"] = "decrease"
            semantic_diff["stroma_change"]["degree"] = strength
        elif _mentions_any(text, _INCREASE_TERMS) or _mentions_any(text, _ADD_TERMS):
            semantic_diff["stroma_change"]["density"] = "increase"
            semantic_diff["stroma_change"]["degree"] = strength

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
            {"role": "user", "content": instruction},
        ],
        "response_format": {"type": "json_object"},
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
) -> dict[str, Any]:
    """Parse a single natural-language edit instruction."""

    if mode == "rule-based":
        return parse_instruction_rule_based(instruction)
    if mode == "api":
        if config is None:
            raise InstructionParserError("config is required for api mode.")
        return parse_instruction_with_api(instruction, config=config)
    raise InstructionParserError(f"Unsupported instruction parser mode: {mode}")


def _post_chat_completion(
    payload: Mapping[str, Any],
    *,
    api_base_url: str,
    api_key: str,
    timeout_sec: float,
    debug_dir: Path | None = None,
) -> dict[str, Any]:
    endpoint = api_base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(payload).encode("utf-8")
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
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if debug_dir is not None:
            (debug_dir / "instruction_parser_http_error.txt").write_text(
                body,
                encoding="utf-8",
            )
        raise InstructionParserError(
            f"API request failed with HTTP {exc.code}: {body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise InstructionParserError(f"API request failed: {exc}") from exc

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
_NECROSIS_TERMS = ("necrosis", "necrotic", "坏死")
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

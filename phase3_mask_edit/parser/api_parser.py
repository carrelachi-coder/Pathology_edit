"""API-backed prompt parser for Phase 3 semantic diffs."""


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


SYSTEM_PROMPT = """You are a pathology report difference analyzer. You compare an ORIGINAL report and an EDITED report, and output ONLY the changes as a JSON object.

The downstream consumer is the Phase3 pathology mask-edit planner. It will map this JSON into deterministic EditIntent primitives, so conservative explicit-change detection is more important than broad inference.

CRITICAL RULES:
1. ONLY report changes that are EXPLICITLY stated in the text. Do NOT infer or assume changes.
2. If a feature (necrosis, lymphocytes, stroma, etc.) is described THE SAME WAY in both reports, or is NOT MENTIONED in the edited report, set it to "none".
3. If necrosis is mentioned identically in both reports (e.g., both say "extensive necrosis"), necrosis_change.action = "none".
4. If stroma is not explicitly discussed as changing, stroma_change.density = "none". Stroma being consumed by tumor expansion is NOT a stroma density change.
5. Replacement/backfill targets are not separate edits. If the edited report says immune/lymphocytic infiltrate decreases and is replaced/backfilled/converted into stroma, set lymphocyte_change.infiltration = "decrease" and stroma_change.density = "none" unless it separately requests desmoplasia/fibrosis/stromal reaction.
6. When in doubt, output "none". It is much better to miss a subtle change than to hallucinate one.
7. grade_change refers to histological grade / differentiation ONLY, not tumor size. If grade or differentiation changes but tumor extent does not, set growth = "none".
8. If the report describes treatment effect (tumor regression, residual tumor, therapy response), set growth = "decrease".

Output ONLY this JSON schema:
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

Field mapping rules:

TUMOR_CHANGE:
- growth: ONLY if the report explicitly describes tumor size/volume/extent changing. "expansion", "enlarged", "occupying majority" -> increase. "residual", "regression", "treatment effect", "shrinkage" -> decrease.
- degree: mild = minor wording change. moderate = clear change. significant = dramatic change.
- grade_change: ONLY if grade or differentiation explicitly changes. "well-differentiated" -> "poorly-differentiated" = upgrade. "high-grade" -> "intermediate-to-low-grade" = downgrade. If grade stays the same, = "none".

LYMPHOCYTE_CHANGE:
- infiltration: ONLY if TIL/lymphocyte description explicitly changes. "sparse" -> "dense" = increase. "brisk TILs" -> "sparse TILs" = decrease.
- If lymphocytes are not mentioned in either report, set to "none".

NECROSIS_CHANGE:
- action: ONLY if necrosis description explicitly changes between the two reports.
  - "no necrosis" -> "focal necrosis" = add
  - "focal" -> "extensive" = increase
  - "extensive necrosis" -> "limited necrosis with fibrotic repair" = decrease
  - "necrosis" -> "no necrosis" = remove
  - Both say "extensive necrosis" = "none" (NO CHANGE)
- If necrosis is described the same way in both reports, action MUST be "none".

STROMA_CHANGE:
- density: ONLY if stromal density/desmoplasia/fibrosis/stromal reaction is explicitly described as changing as its own edit. "fibrous stroma" -> "dense desmoplastic stroma" = increase. "immune infiltrate is replaced with stroma" = none because stroma is just the backfill target for immune decrease. Almost always "none".
- degree: mild/moderate/significant, only meaningful when density != "none".

Output JSON only. No markdown. No explanation."""


FEW_SHOT_EXAMPLES: tuple[tuple[str, str, Mapping[str, Any]], ...] = (
    (
        "Well-differentiated invasive ductal carcinoma forming tubular structures, with minimal lymphocytic response. No necrosis identified.",
        "Poorly-differentiated invasive ductal carcinoma with solid growth pattern, moderate lymphocytic infiltrate and focal necrosis.",
        {
            "schema_version": "0.1",
            "tumor_change": {
                "growth": "increase",
                "degree": "moderate",
                "grade_change": "upgrade",
            },
            "lymphocyte_change": {
                "infiltration": "increase",
                "degree": "moderate",
            },
            "necrosis_change": {"action": "add", "extent": "focal"},
            "stroma_change": {"density": "none", "degree": "moderate"},
        },
    ),
    (
        "High-grade invasive ductal carcinoma with extensive necrosis. A small viable tumor island is present.",
        "High-grade invasive ductal carcinoma with extensive necrosis. The viable tumor shows moderate expansion into surrounding stroma.",
        {
            "schema_version": "0.1",
            "tumor_change": {
                "growth": "increase",
                "degree": "moderate",
                "grade_change": "none",
            },
            "lymphocyte_change": {"infiltration": "none", "degree": "mild"},
            "necrosis_change": {"action": "none", "extent": "focal"},
            "stroma_change": {"density": "none", "degree": "moderate"},
        },
    ),
    (
        "Invasive carcinoma with sparse peritumoral lymphocytes.",
        "Invasive carcinoma with brisk tumor-infiltrating lymphocytes (TILs >50%).",
        {
            "schema_version": "0.1",
            "tumor_change": {
                "growth": "none",
                "degree": "mild",
                "grade_change": "none",
            },
            "lymphocyte_change": {
                "infiltration": "increase",
                "degree": "significant",
            },
            "necrosis_change": {"action": "none", "extent": "focal"},
            "stroma_change": {"density": "none", "degree": "moderate"},
        },
    ),
    (
        "H&E stained breast cancer histopathology with tumor nests surrounded by dense lymphocytic immune infiltrate, sparse stroma, and visible blood vessel tissue.",
        "Decrease the lymphocytic immune infiltrate substantially and replace it with stromal tissue, keeping tumor burden approximately unchanged.",
        {
            "schema_version": "0.1",
            "tumor_change": {
                "growth": "none",
                "degree": "mild",
                "grade_change": "none",
            },
            "lymphocyte_change": {
                "infiltration": "decrease",
                "degree": "significant",
            },
            "necrosis_change": {"action": "none", "extent": "focal"},
            "stroma_change": {"density": "none", "degree": "moderate"},
        },
    ),
    (
        "Invasive carcinoma with dense lymphocytic infiltrate and sparse connective tissue.",
        "Reduce the lymphocytic infiltrate and backfill the removed immune regions with stroma; leave tumor unchanged.",
        {
            "schema_version": "0.1",
            "tumor_change": {
                "growth": "none",
                "degree": "mild",
                "grade_change": "none",
            },
            "lymphocyte_change": {
                "infiltration": "decrease",
                "degree": "moderate",
            },
            "necrosis_change": {"action": "none", "extent": "focal"},
            "stroma_change": {"density": "none", "degree": "moderate"},
        },
    ),
    (
        "Grade II invasive ductal carcinoma with moderate TILs and focal necrosis.",
        "Grade II invasive ductal carcinoma with moderate TILs and focal necrosis.",
        DEFAULT_SEMANTIC_DIFF,
    ),
    (
        "High-grade carcinoma with small foci of necrosis. Sparse TILs.",
        "High-grade carcinoma with extensive comedo-type necrosis. Sparse TILs.",
        {
            "schema_version": "0.1",
            "tumor_change": {
                "growth": "none",
                "degree": "mild",
                "grade_change": "none",
            },
            "lymphocyte_change": {"infiltration": "none", "degree": "mild"},
            "necrosis_change": {"action": "increase", "extent": "extensive"},
            "stroma_change": {"density": "none", "degree": "moderate"},
        },
    ),
    (
        "High-grade invasive ductal carcinoma occupying most of the field. Moderate stromal component.",
        "Invasive ductal carcinoma with treatment effect. Residual tumor nests are small, scattered within fibrotic stroma. Decreased cellularity with scattered tumor cell necrosis and pyknotic nuclei.",
        {
            "schema_version": "0.1",
            "tumor_change": {
                "growth": "decrease",
                "degree": "moderate",
                "grade_change": "none",
            },
            "lymphocyte_change": {"infiltration": "none", "degree": "mild"},
            "necrosis_change": {"action": "none", "extent": "focal"},
            "stroma_change": {"density": "none", "degree": "moderate"},
        },
    ),
    (
        "Invasive ductal carcinoma, high histological grade. Tumor cells with marked nuclear atypia and frequent mitotic figures.",
        "Invasive ductal carcinoma, intermediate-to-low histological grade. Tumor cells are well-differentiated with mild nuclear atypia and rare mitotic figures.",
        {
            "schema_version": "0.1",
            "tumor_change": {
                "growth": "none",
                "degree": "moderate",
                "grade_change": "downgrade",
            },
            "lymphocyte_change": {"infiltration": "none", "degree": "mild"},
            "necrosis_change": {"action": "none", "extent": "focal"},
            "stroma_change": {"density": "none", "degree": "moderate"},
        },
    ),
    (
        "High-grade carcinoma. Extensive coagulative necrosis occupies a large portion of the field.",
        "High-grade carcinoma. The necrotic area is limited, with a peripheral fibrotic reparative zone containing macrophage infiltration and fibroblast proliferation.",
        {
            "schema_version": "0.1",
            "tumor_change": {
                "growth": "none",
                "degree": "mild",
                "grade_change": "none",
            },
            "lymphocyte_change": {"infiltration": "none", "degree": "mild"},
            "necrosis_change": {"action": "decrease", "extent": "moderate"},
            "stroma_change": {"density": "none", "degree": "moderate"},
        },
    ),
    (
        "Invasive carcinoma with loose myxoid stroma and scattered adipose tissue between tumor nests.",
        "Invasive carcinoma with dense desmoplastic stroma. Collagenous fibrous tissue replaces the previously loose stroma, with markedly reduced cellularity in the stromal compartment.",
        {
            "schema_version": "0.1",
            "tumor_change": {
                "growth": "none",
                "degree": "mild",
                "grade_change": "none",
            },
            "lymphocyte_change": {"infiltration": "none", "degree": "mild"},
            "necrosis_change": {"action": "none", "extent": "focal"},
            "stroma_change": {"density": "increase", "degree": "moderate"},
        },
    ),
)


@dataclass(frozen=True)
class ApiParserConfig:
    """Configuration for an OpenAI-compatible chat-completions parser."""

    model: str
    api_base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_sec: float = 60.0
    temperature: float = 0.0
    use_few_shot: bool = True
    debug_dir: str | None = None


class ApiParserError(RuntimeError):
    """Raised when the API parser cannot produce a validated semantic diff."""


def parse_prompts_with_api(
    old_prompt: str,
    new_prompt: str,
    *,
    config: ApiParserConfig,
) -> dict[str, Any]:
    """Parse prompt pair into a validated semantic diff using a model API."""

    if not old_prompt or not new_prompt:
        raise ApiParserError("old_prompt and new_prompt are required.")

    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise ApiParserError(f"Missing API key environment variable: {config.api_key_env}")

    request_payload = {
        "model": config.model,
        "temperature": config.temperature,
        "messages": _messages_for_prompt_pair(
            old_prompt, new_prompt, use_few_shot=config.use_few_shot
        ),
        "response_format": {"type": "json_object"},
    }
    debug_dir = Path(config.debug_dir) if config.debug_dir else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "api_request.json").write_text(
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
        (debug_dir / "api_message_content.txt").write_text(content, encoding="utf-8")
    try:
        parsed = extract_json_object(content)
        parsed.setdefault("schema_version", SEMANTIC_DIFF_SCHEMA_VERSION)
        normalized = normalize_semantic_diff(parsed, fill_missing=True)
        if debug_dir is not None:
            (debug_dir / "semantic_diff_normalized.json").write_text(
                json.dumps(normalized, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return normalized
    except SemanticDiffValidationError as exc:
        suffix = (
            f" Raw model content saved to: {debug_dir / 'api_message_content.txt'}"
            if debug_dir is not None
            else ""
        )
        raise ApiParserError(f"API response did not match semantic_diff schema: {exc}.{suffix}") from exc


def build_parser_prompt(old_prompt: str, new_prompt: str) -> str:
    """Build the user prompt sent to the semantic parser model."""

    return (
        "Compare the reports and return only the semantic-diff JSON.\n\n"
        f"ORIGINAL REPORT:\n{old_prompt}\n\n"
        f"EDITED REPORT:\n{new_prompt}"
    )


def _messages_for_prompt_pair(
    old_prompt: str, new_prompt: str, *, use_few_shot: bool
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if use_few_shot:
        for original, edited, output in FEW_SHOT_EXAMPLES:
            messages.append(
                {
                    "role": "user",
                    "content": build_parser_prompt(original, edited),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(output, ensure_ascii=False),
                }
            )
    messages.append(
        {"role": "user", "content": build_parser_prompt(old_prompt, new_prompt)}
    )
    return messages


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
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / "api_http_error_body.txt").write_text(body, encoding="utf-8")
        raise ApiParserError(f"API request failed with HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise ApiParserError(f"API request failed: {exc}") from exc

    raw_path = None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        raw_path = debug_dir / "api_response_raw.txt"
        raw_path.write_text(response_data, encoding="utf-8")

    try:
        decoded = json.loads(response_data)
    except json.JSONDecodeError as exc:
        suffix = f" Raw response saved to: {raw_path}" if raw_path is not None else ""
        raise ApiParserError(f"API response was not valid JSON.{suffix}") from exc
    if not isinstance(decoded, dict):
        raise ApiParserError("API response root must be a JSON object.")
    if debug_dir is not None:
        (debug_dir / "api_response.json").write_text(
            json.dumps(decoded, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return decoded


def _response_content(response_payload: Mapping[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ApiParserError("API response missing choices.")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise ApiParserError("API response choice must be a mapping.")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise ApiParserError("API response choice missing message.")
    content = message.get("content")
    if not isinstance(content, str):
        raise ApiParserError("API response message content must be a string.")
    return content

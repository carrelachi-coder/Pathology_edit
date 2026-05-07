"""API-backed prompt parser for Phase 3 semantic diffs."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

from phase3_mask_edit.parser.semantic_diff import (
    DEFAULT_SEMANTIC_DIFF,
    SEMANTIC_DIFF_SCHEMA_VERSION,
    SemanticDiffValidationError,
    extract_json_object,
    normalize_semantic_diff,
)


SYSTEM_PROMPT = """You are a pathology report difference analyzer. Compare an ORIGINAL report and an EDITED report, then output ONLY a JSON object matching the schema.

Rules:
1. Only report changes explicitly stated in the text.
2. If a feature is described the same way in both reports, or is not mentioned as changing, output "none" for that feature.
3. Prefer false negatives over hallucinated edits.
4. grade_change refers to histological grade or differentiation only, not tumor size.
5. Treatment response, residual tumor, regression, or shrinkage means tumor growth is "decrease".
6. Stroma density changes only when desmoplasia/fibrosis/stromal density is explicitly described as changing.

Required JSON schema:
{
  "schema_version": "0.1",
  "tumor_change": {
    "growth": "none|increase|decrease",
    "degree": "mild|moderate|significant",
    "grade_change": "none|upgrade|downgrade"
  },
  "lymphocyte_change": {
    "infiltration": "none|increase|decrease",
    "degree": "mild|moderate|significant"
  },
  "necrosis_change": {
    "action": "none|add|increase|decrease|remove",
    "extent": "focal|moderate|extensive"
  },
  "stroma_change": {
    "density": "none|increase|decrease",
    "degree": "mild|moderate|significant"
  }
}

Output JSON only. No markdown. No explanation."""


FEW_SHOT_EXAMPLES: tuple[tuple[str, str, Mapping[str, Any]], ...] = (
    (
        "Grade II invasive ductal carcinoma with moderate TILs and focal necrosis.",
        "Grade II invasive ductal carcinoma with moderate TILs and focal necrosis.",
        DEFAULT_SEMANTIC_DIFF,
    ),
    (
        "High-grade carcinoma without necrosis.",
        "High-grade carcinoma with focal necrosis.",
        {
            **DEFAULT_SEMANTIC_DIFF,
            "necrosis_change": {"action": "add", "extent": "focal"},
        },
    ),
    (
        "Invasive carcinoma with sparse peritumoral lymphocytes.",
        "Invasive carcinoma with brisk tumor-infiltrating lymphocytes.",
        {
            **DEFAULT_SEMANTIC_DIFF,
            "lymphocyte_change": {
                "infiltration": "increase",
                "degree": "significant",
            },
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
    response_payload = _post_chat_completion(
        request_payload,
        api_base_url=config.api_base_url,
        api_key=api_key,
        timeout_sec=config.timeout_sec,
    )
    content = _response_content(response_payload)
    try:
        parsed = extract_json_object(content)
        parsed.setdefault("schema_version", SEMANTIC_DIFF_SCHEMA_VERSION)
        return normalize_semantic_diff(parsed, fill_missing=True)
    except SemanticDiffValidationError as exc:
        raise ApiParserError(f"API response did not match semantic_diff schema: {exc}") from exc


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
        raise ApiParserError(f"API request failed with HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise ApiParserError(f"API request failed: {exc}") from exc

    try:
        decoded = json.loads(response_data)
    except json.JSONDecodeError as exc:
        raise ApiParserError("API response was not valid JSON.") from exc
    if not isinstance(decoded, dict):
        raise ApiParserError("API response root must be a JSON object.")
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

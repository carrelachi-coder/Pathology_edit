"""API-backed prompt parser for Phase 3 semantic diffs."""


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


SYSTEM_PROMPT = """You are a pathology report difference analyzer. You compare an ORIGINAL report and an EDITED report, and output ONLY the changes as a JSON object.

The downstream consumer is the Phase3 pathology mask-edit planner. It will map this JSON into deterministic EditIntent primitives, so conservative explicit-change detection is more important than broad inference.

CRITICAL RULES:
1. These report pairs encode one primary mask edit. Return the dominant independent edit, not every correlated descriptor difference. Tissue occupying space vacated by the primary edit is a consequence, not a second edit.
2. ONLY report changes that are EXPLICITLY stated in the text. Do NOT infer or assume changes.
3. If a feature (necrosis, lymphocytes, stroma, etc.) is described THE SAME WAY in both reports, or is NOT MENTIONED in the edited report, set it to "none".
4. If necrosis is mentioned identically in both reports (e.g., both say "extensive necrosis"), necrosis_change.action = "none".
5. If stroma is not explicitly discussed as an independent desmoplastic/stromal-reaction edit, stroma_change.density = "none". More visible stroma after tumor or necrosis decreases is reciprocal backfill, not desmoplasia.
6. Replacement/backfill targets are not separate edits. If immune, necrotic, or tumor tissue decreases while stroma becomes more prominent, encode the source-tissue decrease and set stroma_change.density = "none" unless a separate desmoplastic reaction is explicitly the main subject.
7. A background immune adjective is not a second edit when tumor extent is clearly the main changed subject. Encode immune change only when immune infiltration is itself the main changed finding.
8. When in doubt, output "none". It is much better to miss a subtle change than to hallucinate one.
9. grade_change refers to histological grade / differentiation ONLY, not tumor size. If grade or differentiation changes but tumor extent does not, set growth = "none".
10. If the report describes treatment effect (tumor regression, residual tumor, therapy response), set growth = "decrease".
11. Use transition_change for an explicit fine-grained phenotype transition. When an exact transition is present, tumor_change.growth = "none" unless a separate area/extent change is explicitly quantified.
12. For transition_change, source_state and target_state must describe the exact ORIGINAL -> EDITED phenotype pair. Otherwise set both to "none".
13. lymphocyte_change.location describes where the requested immune change occurs. Use "intratumoral" only for immune cells inside/within tumor, "peritumoral" for around tumor, "stromal" for stromal immune change, and "unspecified" when location is not explicit. If infiltration = "none", location MUST be "unspecified".

Output ONLY this JSON schema:
{
  "schema_version": "0.2",
  "tumor_change": {
    "growth": "none" | "increase" | "decrease",
    "degree": "mild" | "moderate" | "significant",
    "grade_change": "none" | "upgrade" | "downgrade"
  },
  "lymphocyte_change": {
    "infiltration": "none" | "increase" | "decrease",
    "degree": "mild" | "moderate" | "significant",
    "location": "unspecified" | "stromal" | "intratumoral" | "peritumoral"
  },
  "necrosis_change": {
    "action": "none" | "add" | "increase" | "decrease" | "remove",
    "extent": "focal" | "moderate" | "extensive"
  },
  "stroma_change": {
    "density": "none" | "increase" | "decrease",
    "degree": "mild" | "moderate" | "significant"
  },
  "transition_change": {
    "source_state": "none" | "benign_epithelium" | "stromal_tissue" | "gleason_pattern_3" | "gleason_pattern_4" | "gleason_pattern_5" | "normal_gland" | "adenomatous_gland" | "moderately_differentiated_carcinoma" | "poorly_differentiated_carcinoma",
    "target_state": "none" | "benign_epithelium" | "stromal_tissue" | "gleason_pattern_3" | "gleason_pattern_4" | "gleason_pattern_5" | "normal_gland" | "adenomatous_gland" | "moderately_differentiated_carcinoma" | "poorly_differentiated_carcinoma",
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
- location: use the location of the changed immune infiltrate in the EDITED report. Do not infer intratumoral from generic TIL/immune wording.

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

TRANSITION_CHANGE:
- Encode only these exact phenotype transitions:
  - benign_epithelium -> gleason_pattern_3
  - benign_epithelium -> stromal_tissue
  - gleason_pattern_3 -> gleason_pattern_4
  - gleason_pattern_4 -> gleason_pattern_5
  - gleason_pattern_4 -> gleason_pattern_3
  - normal_gland -> adenomatous_gland
  - adenomatous_gland -> moderately_differentiated_carcinoma
  - moderately_differentiated_carcinoma -> poorly_differentiated_carcinoma
  - poorly_differentiated_carcinoma -> moderately_differentiated_carcinoma
- For every other report pair, source_state = target_state = "none".
- A transition changes phenotype in place. Keep tumor_change.growth = "none" unless area/extent also explicitly changes.

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
                "location": "intratumoral",
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
    (
        "Sparse tumor nests occupy a central stromal compartment. Scattered immune cells are present nearby.",
        "Conspicuous tumor nests now occupy the central compartment. Immune cells remain adjacent to the tumor.",
        {
            **DEFAULT_SEMANTIC_DIFF,
            "tumor_change": {
                "growth": "increase",
                "degree": "mild",
                "grade_change": "none",
            },
        },
    ),
    (
        "A large central necrotic focus leaves only scant viable stroma.",
        "Only minimal necrotic debris remains and viable collagenous stroma predominates centrally.",
        {
            **DEFAULT_SEMANTIC_DIFF,
            "necrosis_change": {"action": "decrease", "extent": "focal"},
        },
    ),
    (
        "Prominent tumor nests occupy most of the central compartment, with scant fibrous stroma.",
        "Sparse tumor nests remain in abundant fibrous stroma in the central compartment.",
        {
            **DEFAULT_SEMANTIC_DIFF,
            "tumor_change": {
                "growth": "decrease",
                "degree": "significant",
                "grade_change": "none",
            },
        },
    ),
    (
        "Prostate adenocarcinoma composed predominantly of Gleason pattern 4 glands.",
        "Prostate adenocarcinoma composed predominantly of Gleason pattern 5 solid tumor sheets.",
        {
            **DEFAULT_SEMANTIC_DIFF,
            "tumor_change": {
                "growth": "none",
                "degree": "moderate",
                "grade_change": "none",
            },
            "transition_change": {
                "source_state": "gleason_pattern_4",
                "target_state": "gleason_pattern_5",
                "degree": "moderate",
            },
        },
    ),
    (
        "Colorectal biopsy showing adenomatous glands.",
        "Colorectal biopsy showing moderately differentiated carcinoma.",
        {
            **DEFAULT_SEMANTIC_DIFF,
            "tumor_change": {
                "growth": "none",
                "degree": "moderate",
                "grade_change": "none",
            },
            "transition_change": {
                "source_state": "adenomatous_gland",
                "target_state": "moderately_differentiated_carcinoma",
                "degree": "moderate",
            },
        },
    ),
    (
        "Prostate tissue containing benign epithelium.",
        "The corresponding region contains stromal tissue without epithelial glands.",
        {
            **DEFAULT_SEMANTIC_DIFF,
            "transition_change": {
                "source_state": "benign_epithelium",
                "target_state": "stromal_tissue",
                "degree": "moderate",
            },
        },
    ),
)

FEW_SHOT_EXAMPLES = tuple(
    (original, edited, normalize_semantic_diff(output, fill_missing=True))
    for original, edited, output in FEW_SHOT_EXAMPLES
)


@dataclass(frozen=True)
class ApiParserConfig:
    """Configuration for an OpenAI-compatible chat-completions parser."""

    model: str
    api_base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_sec: float = 120.0
    max_retries: int = 2
    retry_backoff_sec: float = 2.0
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
    repair_feedback: Mapping[str, Any] | None = None,
    previous_semantic_diff: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse prompt pair into a validated semantic diff using a model API."""

    if not old_prompt or not new_prompt:
        raise ApiParserError("old_prompt and new_prompt are required.")

    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise ApiParserError(
            f"Missing API key environment variable: {config.api_key_env}"
        )

    request_payload = {
        "model": config.model,
        "temperature": config.temperature,
        "messages": _messages_for_prompt_pair(
            old_prompt,
            new_prompt,
            use_few_shot=config.use_few_shot,
            repair_feedback=repair_feedback,
            previous_semantic_diff=previous_semantic_diff,
        ),
        "response_format": semantic_diff_response_format(),
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
        max_retries=config.max_retries,
        retry_backoff_sec=config.retry_backoff_sec,
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
        raise ApiParserError(
            f"API response did not match semantic_diff schema: {exc}.{suffix}"
        ) from exc


def build_parser_prompt(
    old_prompt: str,
    new_prompt: str,
    *,
    repair_feedback: Mapping[str, Any] | None = None,
    previous_semantic_diff: Mapping[str, Any] | None = None,
) -> str:
    """Build the user prompt sent to the semantic parser model."""

    prompt = (
        "Compare the reports and return only the semantic-diff JSON.\n\n"
        f"ORIGINAL REPORT:\n{old_prompt}\n\n"
        f"EDITED REPORT:\n{new_prompt}"
    )
    if not repair_feedback:
        return prompt
    return (
        prompt
        + "\n\nThe previous semantic-diff output could not produce an executable "
        "mask-edit intent. Re-read both reports and correct any missed or "
        "misclassified explicit primary change. Do not invent a change merely "
        "to satisfy the planner. Return the complete semantic-diff schema."
        + "\n\nPREVIOUS SEMANTIC DIFF:\n"
        + json.dumps(previous_semantic_diff or {}, indent=2, ensure_ascii=False)
        + "\n\nDOWNSTREAM PLANNER FEEDBACK:\n"
        + json.dumps(repair_feedback, indent=2, ensure_ascii=False)
    )


def _messages_for_prompt_pair(
    old_prompt: str,
    new_prompt: str,
    *,
    use_few_shot: bool,
    repair_feedback: Mapping[str, Any] | None = None,
    previous_semantic_diff: Mapping[str, Any] | None = None,
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
        {
            "role": "user",
            "content": build_parser_prompt(
                old_prompt,
                new_prompt,
                repair_feedback=repair_feedback,
                previous_semantic_diff=previous_semantic_diff,
            ),
        }
    )
    return messages


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
                debug_dir.mkdir(parents=True, exist_ok=True)
                (debug_dir / f"api_http_error_attempt_{attempt + 1}.txt").write_text(
                    body,
                    encoding="utf-8",
                )
            retryable = exc.code in {408, 429, 500, 502, 503, 504}
            if not retryable or attempt >= max_retries:
                raise ApiParserError(
                    f"API request failed with HTTP {exc.code}: {body}"
                ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt >= max_retries:
                raise ApiParserError(f"API request failed: {exc}") from exc
        time.sleep(retry_backoff_sec * (attempt + 1))

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

"""Prompt generation, checking, and GT-to-semantic-diff mapping."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from phase3_mask_edit.benchmark.models import BenchmarkIntent, BenchmarkPrompt
from phase3_mask_edit.parser.semantic_diff import DEFAULT_SEMANTIC_DIFF, normalize_semantic_diff


@dataclass(frozen=True)
class LLMConfig:
    model: str
    api_base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_sec: float = 60.0
    temperature: float = 0.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None, *, default_model: str = "template") -> "LLMConfig":
        payload = payload or {}
        return cls(
            model=str(payload.get("model") or default_model),
            api_base_url=str(payload.get("api_base_url") or "https://api.openai.com/v1"),
            api_key_env=str(payload.get("api_key_env") or "OPENAI_API_KEY"),
            timeout_sec=float(payload.get("timeout_sec", 60.0)),
            temperature=float(payload.get("temperature", 0.0)),
        )


def generate_prompts(
    intents: Iterable[BenchmarkIntent],
    *,
    generator: LLMConfig | None = None,
    checker: LLMConfig | None = None,
    use_llm_generator: bool = False,
    use_llm_checker: bool = False,
) -> list[BenchmarkPrompt]:
    prompts: list[BenchmarkPrompt] = []
    generator = generator or LLMConfig(model="template")
    checker = checker or LLMConfig(model="not_checked")
    for intent in intents:
        prompt = _generate_one_with_llm(intent, generator) if use_llm_generator else template_prompt_for_intent(intent)
        if use_llm_checker:
            prompt = check_prompt_with_llm(intent, prompt, checker)
        prompts.append(prompt)
    return prompts


def template_prompt_for_intent(intent: BenchmarkIntent) -> BenchmarkPrompt:
    old_prompt = _baseline_prompt(intent)
    phrase = _edit_phrase(intent)
    location = _location_phrase(intent.region_hint)
    strength = _strength_phrase(intent.strength)
    new_prompt = f"{old_prompt} Edit request: {phrase} with {strength} magnitude in the {location}; preserve unrelated tissue compartments."
    instruction = f"{phrase} with {strength} magnitude in the {location}, while preserving unrelated tissue compartments."
    return BenchmarkPrompt(
        sample_id=intent.sample_id,
        old_prompt=old_prompt,
        new_prompt=new_prompt,
        instruction=instruction,
        generator_model="template",
        checker_model="template",
        checker_status="accepted",
        checker_reason="deterministic_template_matches_gt",
    )


def check_prompt_with_llm(intent: BenchmarkIntent, prompt: BenchmarkPrompt, checker: LLMConfig) -> BenchmarkPrompt:
    payload = {
        "sample_id": intent.sample_id,
        "gt": _checker_gt(intent),
        "old_prompt": prompt.old_prompt,
        "new_prompt": prompt.new_prompt,
        "instruction": prompt.instruction,
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You audit pathology mask-edit benchmark prompts. Return JSON only with "
                "status accepted/rejected, reason, primitive, strength, location, direction. "
                "Accept only if both the old/new prompt and instruction express the GT edit."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        response = _chat_json(messages, checker)
        status = str(response.get("status") or "rejected").lower()
        reason = str(response.get("reason") or "")
        if status == "accepted" and not _checker_response_matches(intent, response):
            status = "rejected"
            reason = reason or "checker_response_fields_do_not_match_gt"
    except Exception as exc:
        status = "rejected"
        reason = f"checker_error:{exc}"
    return BenchmarkPrompt(
        sample_id=prompt.sample_id,
        old_prompt=prompt.old_prompt,
        new_prompt=prompt.new_prompt,
        instruction=prompt.instruction,
        generator_model=prompt.generator_model,
        checker_model=checker.model,
        checker_status=status,
        checker_reason=reason,
    )


def accepted_prompts(prompts: Iterable[BenchmarkPrompt]) -> list[BenchmarkPrompt]:
    return [prompt for prompt in prompts if prompt.checker_status.lower() == "accepted"]


def write_manual_review_csv(prompts: Iterable[BenchmarkPrompt], path: str | Path, *, per_group: int = 3) -> Path:
    from collections import defaultdict
    import csv

    grouped: dict[str, list[BenchmarkPrompt]] = defaultdict(list)
    for prompt in prompts:
        key = prompt.sample_id.split("_")[0]
        grouped[key].append(prompt)
    rows: list[BenchmarkPrompt] = []
    for key in sorted(grouped):
        rows.extend(grouped[key][:per_group])
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "sample_id",
                "old_prompt",
                "new_prompt",
                "instruction",
                "checker_status",
                "checker_reason",
                "human_status",
                "human_note",
            ],
        )
        writer.writeheader()
        for prompt in rows:
            writer.writerow(
                {
                    "sample_id": prompt.sample_id,
                    "old_prompt": prompt.old_prompt,
                    "new_prompt": prompt.new_prompt,
                    "instruction": prompt.instruction,
                    "checker_status": prompt.checker_status,
                    "checker_reason": prompt.checker_reason,
                    "human_status": "",
                    "human_note": "",
                }
            )
    return output


def semantic_diff_for_intent(intent: BenchmarkIntent) -> dict[str, Any]:
    diff = json.loads(json.dumps(DEFAULT_SEMANTIC_DIFF))
    degree = _degree_for_strength(intent.strength)
    extent = _extent_for_strength(intent.strength)
    primitive = intent.primitive
    if primitive == "tumor_burden_increase":
        diff["tumor_change"] = {"growth": "increase", "degree": degree, "grade_change": "none"}
    elif primitive == "tumor_burden_decrease":
        diff["tumor_change"] = {"growth": "decrease", "degree": degree, "grade_change": "none"}
    elif primitive == "necrosis_appearance":
        diff["necrosis_change"] = {"action": "increase", "extent": extent}
    elif primitive == "necrosis_resolution":
        diff["necrosis_change"] = {"action": "remove" if intent.strength == "xlarge_deid" else "decrease", "extent": extent}
    elif primitive in {"stromal_immune_infiltration", "intratumoral_immune_infiltration"}:
        diff["lymphocyte_change"] = {"infiltration": "increase", "degree": degree}
    elif primitive == "immune_infiltration_decrease":
        diff["lymphocyte_change"] = {"infiltration": "decrease", "degree": degree}
    elif primitive == "stromal_desmoplasia":
        diff["stroma_change"] = {"density": "increase", "degree": degree}
    elif primitive in {"stroma_decrease", "stromal_reduction"}:
        diff["stroma_change"] = {"density": "decrease", "degree": degree}
    elif primitive in {"gleason_upgrade_3to4", "gleason_upgrade_4to5", "benign_to_gleason3", "normal_to_adenomatous", "adenoma_to_carcinoma", "grade_upgrade"}:
        diff["tumor_change"] = {"growth": "none", "degree": degree, "grade_change": "upgrade"}
    elif primitive in {"gleason_downgrade_4to3", "treatment_dedifferentiation"}:
        diff["tumor_change"] = {"growth": "none", "degree": degree, "grade_change": "downgrade"}
    else:
        diff["benchmark_unmapped_primitive"] = primitive
    diff["benchmark_gt"] = {
        "sample_id": intent.sample_id,
        "primitive": intent.primitive,
        "strength": intent.strength,
        "region_hint": intent.region_hint,
        "source_labels": list(intent.source_labels),
        "target_label": intent.target_label,
        "expected_direction": intent.expected_direction,
    }
    return normalize_semantic_diff(diff, fill_missing=True)


def _generate_one_with_llm(intent: BenchmarkIntent, generator: LLMConfig) -> BenchmarkPrompt:
    payload = {
        "gt": _checker_gt(intent),
        "instructions": (
            "Generate benchmark prompts for a pathology mask edit. Return JSON with "
            "old_prompt, new_prompt, instruction. The prompts must include category, "
            "direction, strength, and location, but must not mention internal primitive names."
        ),
    }
    messages = [
        {"role": "system", "content": "You write concise pathology mask-edit prompts. Return JSON only."},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    response = _chat_json(messages, generator)
    return BenchmarkPrompt(
        sample_id=intent.sample_id,
        old_prompt=str(response["old_prompt"]),
        new_prompt=str(response["new_prompt"]),
        instruction=str(response["instruction"]),
        generator_model=generator.model,
        checker_model="not_checked",
        checker_status="accepted",
        checker_reason="generated_not_checked",
    )


def _chat_json(messages: list[dict[str, Any]], config: LLMConfig) -> dict[str, Any]:
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {config.api_key_env}")
    payload = {
        "model": config.model,
        "temperature": config.temperature,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        config.api_base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=config.timeout_sec) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    return json.loads(content)


def _baseline_prompt(intent: BenchmarkIntent) -> str:
    organ = intent.organ.replace("_", " ")
    return f"H&E stained {organ} pathology patch with existing tumor and surrounding tissue compartments."


def _edit_phrase(intent: BenchmarkIntent) -> str:
    primitive = intent.primitive
    if primitive == "tumor_burden_increase":
        return "increase the tumor burden"
    if primitive == "tumor_burden_decrease":
        return "decrease the tumor burden"
    if primitive == "necrosis_appearance":
        return "increase intratumoral necrosis"
    if primitive == "necrosis_resolution":
        return "decrease the necrotic tissue"
    if primitive == "stromal_immune_infiltration":
        return "increase stromal immune infiltrate around tumor"
    if primitive == "intratumoral_immune_infiltration":
        return "increase immune infiltrate inside tumor"
    if primitive == "immune_infiltration_decrease":
        return "decrease immune infiltrate"
    if primitive == "stromal_desmoplasia":
        return "increase desmoplastic stromal reaction"
    if primitive in {"stroma_decrease", "stromal_reduction"}:
        return "decrease stromal tissue"
    specialized_phrases = {
        "gleason_upgrade_3to4": "convert part of the Gleason pattern 3 tumor to Gleason pattern 4",
        "gleason_upgrade_4to5": "convert part of the Gleason pattern 4 tumor to Gleason pattern 5",
        "gleason_downgrade_4to3": "convert part of the Gleason pattern 4 tumor toward Gleason pattern 3",
        "benign_to_gleason3": "convert benign epithelium into low-grade Gleason pattern 3 tumor",
        "benign_atrophy": "replace benign epithelium with stromal tissue",
        "normal_to_adenomatous": "convert normal gland epithelium into adenomatous gland tissue",
        "adenoma_to_carcinoma": "convert adenomatous gland tissue into carcinoma",
        "grade_upgrade": "shift moderately differentiated tumor toward poor differentiation",
        "treatment_dedifferentiation": "shift poorly differentiated tumor toward moderate differentiation",
    }
    if primitive in specialized_phrases:
        return specialized_phrases[primitive]
    if intent.expected_direction == "transition":
        return "shift the target tissue phenotype as specified by the benchmark GT"
    return f"apply a {intent.expected_direction} tissue edit"


def _location_phrase(region_hint: Mapping[str, Any]) -> str:
    location = str(region_hint.get("location") or "selected")
    relation = str(region_hint.get("relation") or "")
    text = location.replace("_", " ")
    if relation and relation not in text:
        text = f"{relation} {text}"
    return text.strip()


def _strength_phrase(strength: str) -> str:
    return {
        "mild": "mild",
        "moderate": "moderate",
        "significant": "marked",
        "xlarge_deid": "very large",
    }.get(strength, strength)


def _degree_for_strength(strength: str) -> str:
    return "significant" if strength == "xlarge_deid" else strength if strength in {"mild", "moderate", "significant"} else "moderate"


def _extent_for_strength(strength: str) -> str:
    return {"mild": "focal", "moderate": "moderate", "significant": "extensive", "xlarge_deid": "extensive"}.get(strength, "moderate")


def _checker_gt(intent: BenchmarkIntent) -> dict[str, Any]:
    return {
        "primitive": intent.primitive,
        "strength": intent.strength,
        "direction": intent.expected_direction,
        "location": intent.region_hint.get("location"),
        "relation": intent.region_hint.get("relation"),
        "source_labels": list(intent.source_labels),
        "target_label": intent.target_label,
    }


def _checker_response_matches(intent: BenchmarkIntent, response: Mapping[str, Any]) -> bool:
    primitive = str(response.get("primitive") or "")
    strength = str(response.get("strength") or "")
    direction = str(response.get("direction") or "")
    return primitive in {"", intent.primitive} and strength in {"", intent.strength} and direction in {"", intent.expected_direction}

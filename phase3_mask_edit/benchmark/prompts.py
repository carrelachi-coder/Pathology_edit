"""Prompt generation, checking, and GT-to-semantic-diff mapping."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from phase3_mask_edit.benchmark.models import BenchmarkIntent, BenchmarkPrompt
from phase3_mask_edit.parser.api_parser import ApiParserConfig, parse_prompts_with_api
from phase3_mask_edit.parser.semantic_diff import (
    DEFAULT_SEMANTIC_DIFF,
    normalize_semantic_diff,
)
from phase3_mask_edit.rules.semantic_to_intent import plan_edit_intents


@dataclass(frozen=True)
class LLMConfig:
    model: str
    api_base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_sec: float = 60.0
    temperature: float = 0.0

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any] | None, *, default_model: str = "template"
    ) -> "LLMConfig":
        payload = payload or {}
        return cls(
            model=str(payload.get("model") or default_model),
            api_base_url=str(
                payload.get("api_base_url") or "https://api.openai.com/v1"
            ),
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
    parser_checker: LLMConfig | None = None,
    use_parser_checker: bool = False,
    repair_feedback: str = "",
    repair_prompt: BenchmarkPrompt | None = None,
) -> list[BenchmarkPrompt]:
    prompts: list[BenchmarkPrompt] = []
    generator = generator or LLMConfig(model="template")
    checker = checker or LLMConfig(model="not_checked")
    parser_checker = parser_checker or LLMConfig(model="not_checked")
    for intent in intents:
        prompt = (
            _generate_one_with_llm(
                intent,
                generator,
                repair_feedback=repair_feedback,
                repair_prompt=repair_prompt,
            )
            if use_llm_generator
            else template_prompt_for_intent(intent)
        )
        if use_llm_checker:
            prompt = check_prompt_with_llm(intent, prompt, checker)
        if use_parser_checker and prompt.checker_status.lower() == "accepted":
            prompt = check_prompt_with_parser(intent, prompt, parser_checker)
        prompts.append(prompt)
    return prompts


def template_prompt_for_intent(intent: BenchmarkIntent) -> BenchmarkPrompt:
    old_prompt, new_prompt = _template_report_pair(intent)
    phrase = _edit_phrase(intent)
    location = _location_phrase(intent.region_hint)
    strength = _strength_phrase(intent.strength)
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


def check_prompt_with_llm(
    intent: BenchmarkIntent, prompt: BenchmarkPrompt, checker: LLMConfig
) -> BenchmarkPrompt:
    report_violation = validate_report_pair_language(prompt)
    if report_violation:
        return BenchmarkPrompt(
            sample_id=prompt.sample_id,
            old_prompt=prompt.old_prompt,
            new_prompt=prompt.new_prompt,
            instruction=prompt.instruction,
            generator_model=prompt.generator_model,
            checker_model=checker.model,
            checker_status="rejected",
            checker_reason=report_violation,
        )
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
                "status accepted/rejected, reason, primitive, strength, location, direction, organ. "
                "Accept only if old_prompt and new_prompt are standalone pathology reports, "
                "not edit commands, and their report-level semantic difference matches the GT edit. "
                "Most wording may be identical across the two reports. Prefer an identical organ, location, "
                "and unrelated-finding scaffold with only the target-state observation written differently. "
                "Shared wording is a benchmark control and must not be described as preserved or unchanged. "
                "The organ/site must match the GT exactly or use a direct synonym; reject reports that "
                "switch to another organ. Treat strength as ordinal calibration metadata rather than an exact "
                "language label: do not reject an otherwise valid report pair solely because an adjacent "
                "mild/moderate/significant label is also plausible. "
                "Reject if old_prompt or new_prompt uses imperative/edit wording such as edit, modify, "
                "change, increase, decrease, reduce, add, remove, make, or convert. These restrictions "
                "apply only to old_prompt and new_prompt. The instruction field is expected to be an edit "
                "command, so do not reject it for imperative wording; only check whether its edit semantics "
                "match the GT. Also reject old_prompt/new_prompt if either contains cross-report comparison "
                "or preservation wording such as unchanged, stable, remains, compared, prior, previously, "
                "surrounding tissue unchanged, no additional, no other, transition from, or preserving. "
                "Static single-image pathology terms such as preserved architecture, residual debris, and "
                "adenomatous change are allowed when they do not refer to the other report. "
                "Each report must read as an independent descriptive finding. For a fine-state transition, "
                "the old report must describe gt.source_state and the new report must describe "
                "gt.target_state in that exact direction."
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
    return [
        prompt
        for prompt in prompts
        if prompt.checker_status.lower() == "accepted"
        and validate_report_pair_language(prompt) is None
    ]


def check_prompt_with_parser(
    intent: BenchmarkIntent,
    prompt: BenchmarkPrompt,
    parser: LLMConfig,
) -> BenchmarkPrompt:
    try:
        semantic_diff = parse_prompts_with_api(
            prompt.old_prompt,
            prompt.new_prompt,
            config=ApiParserConfig(
                model=parser.model,
                api_base_url=parser.api_base_url,
                api_key_env=parser.api_key_env,
                timeout_sec=parser.timeout_sec,
                temperature=parser.temperature,
            ),
        )
        plan = plan_edit_intents(
            semantic_diff,
            reference_profile=intent.profile,
            old_prompt=prompt.old_prompt,
            new_prompt=prompt.new_prompt,
        )
        planned = [
            item.intent
            for item in plan.items
            if item.intent is not None and item.role != "fallback"
        ]
        planned_summary = [f"{item.primitive}:{item.strength}" for item in planned]
        matches = len(planned) == 1 and planned[0].primitive == intent.primitive
        if matches:
            status = "accepted"
            strength_agreement = planned[0].strength == intent.strength
            reason = (
                "llm_and_parser_checker_match_gt_primitive;"
                f"intended_strength={intent.strength};"
                f"parsed_strength={planned[0].strength};"
                f"strength_label_agreement={str(strength_agreement).lower()}"
            )
        else:
            status = "rejected"
            reason = (
                "parser_checker_mismatch:"
                f"expected={intent.primitive}:{intent.strength};"
                f"planned={','.join(planned_summary) or 'none'}"
            )
    except Exception as exc:
        status = "rejected"
        reason = f"parser_checker_error:{exc}"
    checker_model = "+".join(
        item for item in (prompt.checker_model, parser.model) if item
    )
    return BenchmarkPrompt(
        sample_id=prompt.sample_id,
        old_prompt=prompt.old_prompt,
        new_prompt=prompt.new_prompt,
        instruction=prompt.instruction,
        generator_model=prompt.generator_model,
        checker_model=checker_model,
        checker_status=status,
        checker_reason=reason,
    )


def write_manual_review_csv(
    prompts: Iterable[BenchmarkPrompt],
    path: str | Path,
    *,
    per_group: int = 3,
    intents_by_id: Mapping[str, BenchmarkIntent] | None = None,
) -> Path:
    from collections import defaultdict
    import csv

    grouped: dict[str, list[BenchmarkPrompt]] = defaultdict(list)
    for prompt in prompts:
        intent = (intents_by_id or {}).get(prompt.sample_id)
        key = (
            f"{intent.organ}|{intent.primitive}|{intent.strength}"
            if intent is not None
            else prompt.sample_id.split("_")[0]
        )
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


_TRANSITION_STATES_BY_PRIMITIVE = {
    "gleason_upgrade_3to4": ("gleason_pattern_3", "gleason_pattern_4"),
    "gleason_upgrade_4to5": ("gleason_pattern_4", "gleason_pattern_5"),
    "gleason_downgrade_4to3": ("gleason_pattern_4", "gleason_pattern_3"),
    "benign_to_gleason3": ("benign_epithelium", "gleason_pattern_3"),
    "benign_atrophy": ("benign_epithelium", "stromal_tissue"),
    "normal_to_adenomatous": ("normal_gland", "adenomatous_gland"),
    "adenoma_to_carcinoma": (
        "adenomatous_gland",
        "moderately_differentiated_carcinoma",
    ),
    "grade_upgrade": (
        "moderately_differentiated_carcinoma",
        "poorly_differentiated_carcinoma",
    ),
    "treatment_dedifferentiation": (
        "poorly_differentiated_carcinoma",
        "moderately_differentiated_carcinoma",
    ),
}

_DOWNGRADE_TRANSITIONS = {
    "gleason_downgrade_4to3",
    "treatment_dedifferentiation",
}


def semantic_diff_for_intent(intent: BenchmarkIntent) -> dict[str, Any]:
    diff = json.loads(json.dumps(DEFAULT_SEMANTIC_DIFF))
    degree = _degree_for_strength(intent.strength)
    extent = _extent_for_strength(intent.strength)
    primitive = intent.primitive
    if primitive == "tumor_burden_increase":
        diff["tumor_change"] = {
            "growth": "increase",
            "degree": degree,
            "grade_change": "none",
        }
    elif primitive == "tumor_burden_decrease":
        diff["tumor_change"] = {
            "growth": "decrease",
            "degree": degree,
            "grade_change": "none",
        }
    elif primitive == "necrosis_appearance":
        diff["necrosis_change"] = {"action": "increase", "extent": extent}
    elif primitive == "necrosis_resolution":
        diff["necrosis_change"] = {
            "action": "remove" if intent.strength == "xlarge_deid" else "decrease",
            "extent": extent,
        }
    elif primitive in {
        "stromal_immune_infiltration",
        "intratumoral_immune_infiltration",
    }:
        diff["lymphocyte_change"] = {
            "infiltration": "increase",
            "degree": degree,
            "location": (
                "intratumoral"
                if primitive == "intratumoral_immune_infiltration"
                else "stromal"
            ),
        }
    elif primitive == "immune_infiltration_decrease":
        diff["lymphocyte_change"] = {
            "infiltration": "decrease",
            "degree": degree,
            "location": "unspecified",
        }
    elif primitive in {"stroma_increase", "stromal_desmoplasia"}:
        diff["stroma_change"] = {"density": "increase", "degree": degree}
    elif primitive in {"stroma_decrease", "stromal_reduction"}:
        diff["stroma_change"] = {"density": "decrease", "degree": degree}
    elif primitive in _TRANSITION_STATES_BY_PRIMITIVE:
        source_state, target_state = _TRANSITION_STATES_BY_PRIMITIVE[primitive]
        diff["transition_change"] = {
            "source_state": source_state,
            "target_state": target_state,
            "degree": degree,
        }
        grade_change = "none"
        if primitive in _DOWNGRADE_TRANSITIONS:
            grade_change = "downgrade"
        elif primitive != "benign_atrophy":
            grade_change = "upgrade"
        diff["tumor_change"] = {
            "growth": "none",
            "degree": degree,
            "grade_change": grade_change,
        }
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


def _generate_one_with_llm(
    intent: BenchmarkIntent,
    generator: LLMConfig,
    *,
    repair_feedback: str = "",
    repair_prompt: BenchmarkPrompt | None = None,
) -> BenchmarkPrompt:
    payload = {
        "gt": _checker_gt(intent),
        "sample_id": intent.sample_id,
        "organ": intent.organ,
        "profile": intent.profile,
        "strength_guidance": _strength_report_guidance(intent.strength),
        "report_generation_contract": {
            "old_prompt": "A standalone descriptive pathology report for the reference image only.",
            "new_prompt": "A standalone descriptive pathology report for the target image only.",
            "shared_scaffold": (
                "Reuse the same wording verbatim for the organ, location, and every unrelated finding. "
                "Prefer two or three identical scaffold sentences and change only the sentence or phrase "
                "that describes the intended target tissue state."
            ),
            "critical_rule": (
                "The two reports must not talk about each other. They must not describe an edit, "
                "a change, a comparison, preservation, or what stayed the same. Each report only "
                "states what is visible in that single image."
            ),
        },
        "few_shot_examples": _report_pair_few_shots(),
        "instructions": (
            "Generate prompts for a pathology mask-edit semantic fidelity benchmark. "
            "Return JSON with exactly old_prompt, new_prompt, instruction. "
            f"Use the exact organ/site '{intent.organ}' from the GT; do not substitute another organ. "
            "old_prompt and new_prompt must be two independent, standalone pathology-style reports. "
            "old_prompt describes only the reference image state. new_prompt describes only the target image state. "
            "Write them as if two different pathologists independently dictated two reports for two separate images. "
            "For benchmark control, deliberately reuse most wording verbatim across the two reports. Copy the organ, "
            "location, sentence order, and unrelated findings exactly. Prefer three sentences with two sentences "
            "identical, changing only one state-only observation about the intended tissue category. Do not paraphrase "
            "shared facts, because synonym changes can create unintended semantic edits. Identical shared text does "
            "not mean either report may say that a finding was preserved, stable, or unchanged. "
            "The reports must not mention ref, target, before, after, editing, or that one report differs from the other. "
            "The parser will infer the edit later from the semantic difference, but the reports themselves must not explain that difference. "
            "instruction must be a direct user edit instruction, not a meta-instruction about modifying prompts. "
            "The old/new reports should be 2-4 sentences each, written like concise pathology findings. "
            "Both reports must mention the exact organ/site, the relevant tissue category, and the GT location. "
            "Only the intended GT semantic dimension may differ between reports. Keep every unrelated tissue finding "
            "identical in both reports or omit it from both. Do not introduce secondary tumor growth, immune, necrosis, "
            "stromal, desmoplastic, or treatment-response differences. For fine-state transitions, do not mention "
            "invasive growth, stromal infiltration, expansion, tumor burden, or a desmoplastic response. "
            "Do not add sentences about other areas being unchanged/stable/preserved. Do not mention internal primitive names, "
            "JSON fields, masks, GT, or benchmark. Do not use imperative/edit words in old_prompt or new_prompt: "
            "avoid edit, modify, change, increase, decrease, reduce, add, remove, make, convert. "
            "Also avoid any cross-report/comparison/preservation wording in old_prompt/new_prompt: unchanged, stable, "
            "remains, compared, relative to, reduction, prior, previously, additional, other areas, surrounding tissue unchanged, "
            "transition from, transitioned, more, less, larger, smaller, higher, lower, no longer, newly, and while. "
            "Each report should only describe its own image state. "
            "A static pathology phrase such as 'preserved glandular architecture', 'residual necrotic debris', or "
            "'adenomatous change' is allowed when it describes the single image itself and does not assert what stayed "
            "the same across reports. If such an unrelated phrase is used, copy it verbatim into both reports. "
            "Use state-only descriptions: for low abundance say sparse/rare/scant/minimal; for high abundance say conspicuous/abundant/dense/prominent. "
            "Do not say 'more sparse', 'less conspicuous', 'more prominent', 'larger', 'smaller', or similar comparative phrases. "
            "For transition edits, old_prompt reports the source phenotype as an observation; new_prompt reports the target phenotype as an observation. "
            "When gt.source_state and gt.target_state are present, use those exact fine-state meanings in that exact direction. "
            "Do not say 'transition from', 'replacing', or 'previously observed' in either report. "
            "Forbidden in old_prompt/new_prompt: increase, increased, decrease, decreased, reduce, reduced, reduction, "
            "enhance, enhanced, more, less, fewer, greater, relative, denser, sparser, higher-grade, lower-grade, larger, "
            "smaller, unchanged, stable, remains, compared, prior, previously, resolved, preserving, transition, replace. "
            "Allowed state-only wording examples: 'sparse lymphocytic infiltrate', 'mild focal lymphocytic infiltrate', 'prominent tumor nests', 'scattered tumor cells', 'central Gleason pattern 4 glands', 'central Gleason pattern 5 solid sheets'. "
            "Bad report wording examples: 'mild increase in immune infiltrate', 'tumor cells are less conspicuous', 'other areas are unchanged', 'transitioning from pattern 4'."
        ),
    }
    if repair_feedback:
        forbidden_match = re.search(
            r"forbidden term ['\"]([^'\"]+)['\"]",
            repair_feedback,
            flags=re.IGNORECASE,
        )
        forbidden_token = forbidden_match.group(1) if forbidden_match else ""
        repair_instruction = (
            "Rewrite the previous rejected output to fix this exact rejection. "
            "The exact forbidden token must not appear anywhere in old_prompt or "
            "new_prompt. Replace comparative wording with a state-only observation, "
            "such as prominent, pronounced, scant, sparse, or minimal as appropriate. "
            "Before returning JSON, scan both reports for the forbidden token. "
            "Preserve the GT semantics and the direct edit instruction."
            if forbidden_token
            else (
                "Rewrite the previous rejected output so a blind pathology difference "
                "parser finds exactly the intended GT edit and no secondary edits. "
                "Remove incidental changes in tumor amount, immune infiltrate, necrosis, "
                "or stroma unless that compartment is the GT target. Keep unrelated "
                "findings identical in both reports or omit them from both."
            )
        )
        payload["repair_request"] = {
            "checker_rejection": repair_feedback,
            "forbidden_token": forbidden_token,
            "previous_rejected_output": (
                {
                    "old_prompt": repair_prompt.old_prompt,
                    "new_prompt": repair_prompt.new_prompt,
                    "instruction": repair_prompt.instruction,
                }
                if repair_prompt is not None
                else None
            ),
            "instruction": repair_instruction,
        }
    messages = [
        {
            "role": "system",
            "content": (
                "You write concise pathology image edit prompts. Return strict JSON only. "
                "Do not write meta-prompts such as 'modify the prompt'."
            ),
        },
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
        raise RuntimeError(
            f"Missing API key environment variable: {config.api_key_env}"
        )
    payload = {
        "model": config.model,
        "temperature": config.temperature,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        config.api_base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=config.timeout_sec) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    return json.loads(content)


def _baseline_prompt(intent: BenchmarkIntent) -> str:
    organ = intent.organ.replace("_", " ")
    return f"H&E stained {organ} pathology patch with existing tumor and surrounding tissue compartments."


def _template_report_pair(intent: BenchmarkIntent) -> tuple[str, str]:
    organ = intent.organ.replace("_", " ")
    location = _location_phrase(intent.region_hint)
    primitive = intent.primitive
    if primitive == "immune_infiltration_decrease":
        return (
            f"H&E stained {organ} pathology report. The {location} region shows conspicuous immune infiltrate within the tissue compartment.",
            f"H&E stained {organ} pathology report. The {location} region shows sparse residual immune infiltrate within a stromal background.",
        )
    if primitive in {"stromal_immune_infiltration", "intratumoral_immune_infiltration"}:
        compartment = (
            "stromal" if primitive == "stromal_immune_infiltration" else "intratumoral"
        )
        return (
            f"H&E stained {organ} pathology report. The {location} region has limited {compartment} lymphocytic infiltrate.",
            f"H&E stained {organ} pathology report. The {location} region contains conspicuous {compartment} lymphocytic infiltrate.",
        )
    if primitive == "necrosis_appearance":
        return (
            f"H&E stained {organ} pathology report. The {location} tumor region is predominantly viable with little necrotic debris.",
            f"H&E stained {organ} pathology report. The {location} tumor region contains evident necrotic debris within the tumor.",
        )
    if primitive == "necrosis_resolution":
        return (
            f"H&E stained {organ} pathology report. The {location} tumor region contains evident necrotic debris.",
            f"H&E stained {organ} pathology report. The {location} tumor region is largely viable or stromal with limited residual necrotic debris.",
        )
    if primitive == "tumor_burden_increase":
        return (
            f"H&E stained {organ} pathology report. The {location} region contains predominantly non-neoplastic supporting tissue with focal tumor at its margin.",
            f"H&E stained {organ} pathology report. The {location} region contains a prominent tumor component occupying the local tissue compartment.",
        )
    if primitive == "tumor_burden_decrease":
        return (
            f"H&E stained {organ} pathology report. The {location} region contains a prominent tumor component.",
            f"H&E stained {organ} pathology report. The {location} region contains small residual tumor nests with non-neoplastic supporting tissue.",
        )
    if primitive == "stromal_desmoplasia":
        return (
            f"H&E stained {organ} pathology report. The {location} region has loose non-desmoplastic stroma around tumor.",
            f"H&E stained {organ} pathology report. The {location} region has dense collagenous desmoplastic stroma around tumor.",
        )
    if primitive == "stroma_increase":
        return (
            f"H&E stained {organ} pathology report. The {location} region contains a limited existing stromal compartment.",
            f"H&E stained {organ} pathology report. The {location} region contains an expanded contiguous stromal compartment.",
        )
    if primitive in {"stroma_decrease", "stromal_reduction"}:
        return (
            f"H&E stained {organ} pathology report. The {location} region contains abundant stromal tissue.",
            f"H&E stained {organ} pathology report. The {location} region contains limited stromal tissue with adjacent non-stromal tissue occupying the area.",
        )
    if intent.expected_direction == "transition":
        source_state, target_state = _transition_report_states(intent.primitive)
        return (
            f"H&E stained {organ} pathology report. The {location} region shows {source_state}.",
            f"H&E stained {organ} pathology report. The {location} region shows {target_state}.",
        )
    return (_baseline_prompt(intent), _baseline_prompt(intent))


def _transition_report_states(primitive: str) -> tuple[str, str]:
    states = {
        "gleason_upgrade_3to4": (
            "Gleason pattern 3 glands",
            "Gleason pattern 4 glands",
        ),
        "gleason_upgrade_4to5": (
            "Gleason pattern 4 glands",
            "Gleason pattern 5 solid tumor sheets",
        ),
        "gleason_downgrade_4to3": (
            "Gleason pattern 4 glands",
            "Gleason pattern 3 glands",
        ),
        "benign_to_gleason3": (
            "benign prostatic epithelium",
            "Gleason pattern 3 malignant glands",
        ),
        "benign_atrophy": (
            "benign prostatic epithelium",
            "stromal tissue without epithelial glands",
        ),
        "normal_to_adenomatous": (
            "normal colorectal glandular epithelium",
            "adenomatous colorectal glands",
        ),
        "adenoma_to_carcinoma": (
            "adenomatous colorectal glands",
            "moderately differentiated colorectal carcinoma",
        ),
        "grade_upgrade": (
            "moderately differentiated colorectal carcinoma",
            "poorly differentiated colorectal carcinoma",
        ),
        "treatment_dedifferentiation": (
            "poorly differentiated colorectal carcinoma",
            "moderately differentiated colorectal carcinoma",
        ),
    }
    return states.get(
        primitive,
        (
            "the source glandular or tumor phenotype",
            "the target glandular or tumor phenotype",
        ),
    )


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
    if primitive == "stroma_increase":
        return "increase the existing stromal compartment"
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
    return (
        "significant"
        if strength == "xlarge_deid"
        else strength
        if strength in {"mild", "moderate", "significant"}
        else "moderate"
    )


def _extent_for_strength(strength: str) -> str:
    return {
        "mild": "focal",
        "moderate": "moderate",
        "significant": "extensive",
        "xlarge_deid": "extensive",
    }.get(strength, "moderate")


def _checker_gt(intent: BenchmarkIntent) -> dict[str, Any]:
    payload = {
        "primitive": intent.primitive,
        "strength": intent.strength,
        "direction": intent.expected_direction,
        "location": intent.region_hint.get("location"),
        "relation": intent.region_hint.get("relation"),
        "source_labels": list(intent.source_labels),
        "target_label": intent.target_label,
    }
    if intent.expected_direction == "transition":
        source_state, target_state = _transition_report_states(intent.primitive)
        payload["source_state"] = source_state
        payload["target_state"] = target_state
    return payload


def _checker_response_matches(
    intent: BenchmarkIntent, response: Mapping[str, Any]
) -> bool:
    direction = str(response.get("direction") or "")
    organ = _normalize_checker_text(response.get("organ"))
    expected_organ = _normalize_checker_text(intent.organ)
    organ_ok = (
        not organ
        or organ == expected_organ
        or expected_organ in organ
        or organ in expected_organ
    )
    return direction in {"", intent.expected_direction} and organ_ok


_REPORT_FORBIDDEN_RULE_SPECS = (
    ("increase", r"\bincreas(?:e|ed|ing)\b"),
    ("decrease", r"\bdecreas(?:e|ed|ing)\b"),
    ("reduction", r"\breduc(?:e|ed|ing|tion)\b"),
    ("enhance", r"\benhanc(?:e|ed|ing|ement)\b"),
    ("more", r"\bmore\b"),
    ("less", r"\bless\b"),
    ("fewer", r"\bfewer\b"),
    ("greater", r"\bgreater\b"),
    ("larger", r"\blarger\b"),
    ("smaller", r"\bsmaller\b"),
    ("denser", r"\bdenser\b"),
    ("sparser", r"\bsparser\b"),
    ("higher/lower grade", r"\b(?:higher|lower)[ -]grade\b"),
    ("relative comparison", r"\brelative(?:ly)?\b"),
    ("than", r"\bthan\b"),
    ("versus", r"\bversus\b|\bvs\.?\b"),
    ("unchanged", r"\bunchanged\b"),
    ("stable", r"\bstable\b"),
    ("remain", r"\bremain(?:s|ed|ing)?\b"),
    ("compared", r"\bcompar(?:e|ed|ing|ison)\b"),
    ("prior", r"\bprior\b"),
    ("previous", r"\bprevious(?:ly)?\b"),
    ("no longer", r"\bno\s+longer\b"),
    ("newly", r"\bnewly\b"),
    ("temporal now", r"\b(?:now|currently)\b"),
    ("before/after", r"\b(?:before|after)\b"),
    ("resolved process", r"\bresolv(?:e|ed|ing|ution)\b"),
    ("edit", r"\bedit(?:s|ed|ing)?\b"),
    ("modify", r"\bmodif(?:y|ies|ied|ying|ication)\b"),
    ("add", r"\badd(?:s|ed|ing)?\b"),
    ("remove", r"\bremov(?:e|es|ed|ing|al)\b"),
    ("make", r"\bmake\b|\bmade\b"),
    ("convert", r"\bconvert(?:s|ed|ing)?\b"),
    ("replace", r"\breplac(?:e|es|ed|ing|ement)\b"),
    ("transition", r"\btransition(?:s|ed|ing)?\b"),
    ("preserving", r"\bpreserving\b"),
    ("while", r"\bwhile\b"),
    ("other areas", r"\bother\s+areas?\b"),
    ("additional", r"\badditional\b"),
)

REPORT_FORBIDDEN_TERMS = tuple(label for label, _ in _REPORT_FORBIDDEN_RULE_SPECS)
REPORT_FORBIDDEN_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for _, pattern in _REPORT_FORBIDDEN_RULE_SPECS
)


def validate_report_pair_language(prompt: BenchmarkPrompt) -> str | None:
    """Reject old/new report wording that leaks edit/comparison semantics."""

    for field_name, text in (
        ("old_prompt", prompt.old_prompt),
        ("new_prompt", prompt.new_prompt),
    ):
        normalized = str(text)
        for term, pattern in zip(REPORT_FORBIDDEN_TERMS, REPORT_FORBIDDEN_PATTERNS):
            if pattern.search(normalized):
                return f"report_language_violation:{field_name} contains forbidden term {term!r}"
    return None


def _strength_report_guidance(strength: str) -> str:
    return {
        "mild": (
            "Use two nearby absolute states in separate standalone reports: one may say "
            "scant and the other mild focal. Never write more/less or compare the reports."
        ),
        "moderate": (
            "Use two clearly distinguishable but non-extreme absolute states in separate "
            "standalone reports. Express each state directly, without comparison wording."
        ),
        "significant": (
            "Use strongly separated absolute states: one report may say sparse, and the other "
            "conspicuous or abundant. Keep each report standalone and non-comparative."
        ),
        "xlarge_deid": (
            "Use maximally separated absolute states: one report may say near-absent or minimal, "
            "and the other diffuse or extensive. Each report must describe only its own image."
        ),
    }.get(strength, "Use a report-level difference matching the GT strength.")


def _report_pair_few_shots() -> list[dict[str, str]]:
    """Few-shot report pairs that avoid edit/comparison language in A-mode reports."""

    return [
        {
            "gt_summary": "breast, tumor burden decrease, mild, center",
            "old_prompt": (
                "The breast specimen contains a central stromal compartment. "
                "Several compact tumor nests occupy the central compartment. "
                "The tumor cells form epithelial clusters with mild nuclear atypia."
            ),
            "new_prompt": (
                "The breast specimen contains a central stromal compartment. "
                "A few scattered tumor nests occupy the central compartment. "
                "The tumor cells form epithelial clusters with mild nuclear atypia."
            ),
            "instruction": "Reduce the tumor burden mildly in the central stromal compartment of the breast specimen.",
        },
        {
            "gt_summary": "oral, tumor burden increase, mild, upper right peripheral",
            "old_prompt": (
                "The oral mucosal specimen contains an upper right peripheral stromal compartment. "
                "Rare atypical epithelial nests occupy small foci in this compartment. "
                "The local connective tissue contains a loose stromal background."
            ),
            "new_prompt": (
                "The oral mucosal specimen contains an upper right peripheral stromal compartment. "
                "Several conspicuous atypical epithelial nests occupy this compartment. "
                "The local connective tissue contains a loose stromal background."
            ),
            "instruction": "Increase the tumor burden mildly in the upper right peripheral region of the oral mucosa.",
        },
        {
            "gt_summary": "breast, necrosis appearance, mild, center",
            "old_prompt": (
                "The breast tumor contains a central tumor compartment. "
                "Rare punctate necrotic debris is present among the tumor nests. "
                "Compact viable tumor nests occupy the surrounding central tissue."
            ),
            "new_prompt": (
                "The breast tumor contains a central tumor compartment. "
                "A small pale necrotic focus is present among the tumor nests. "
                "Compact viable tumor nests occupy the surrounding central tissue."
            ),
            "instruction": "Add mild focal necrosis in the central region of the breast tumor.",
        },
        {
            "gt_summary": "melanoma, necrosis resolution, mild, upper left",
            "old_prompt": (
                "The melanoma specimen contains an upper left tissue compartment. "
                "A focal collection of pale necrotic debris and fragmented nuclei occupies this compartment. "
                "Collagenous stromal tissue is present at its periphery."
            ),
            "new_prompt": (
                "The melanoma specimen contains an upper left tissue compartment. "
                "Scant punctate necrotic debris and fragmented nuclei occupy this compartment. "
                "Collagenous stromal tissue is present at its periphery."
            ),
            "instruction": "Resolve a mild amount of necrosis in the upper left region of the melanoma specimen.",
        },
        {
            "gt_summary": "lung, stromal immune infiltration increase, mild, upper",
            "old_prompt": (
                "The lung specimen contains an upper stromal compartment. "
                "Rare scattered lymphocytes are present between the collagen bundles. "
                "Tumor-adjacent stroma forms the local tissue background."
            ),
            "new_prompt": (
                "The lung specimen contains an upper stromal compartment. "
                "Small focal lymphocytic clusters are present between the collagen bundles. "
                "Tumor-adjacent stroma forms the local tissue background."
            ),
            "instruction": "Add mild focal immune infiltrate in the upper stromal compartment of the lung specimen.",
        },
        {
            "gt_summary": "lung, intratumoral immune infiltration increase, mild, center",
            "old_prompt": (
                "The lung tumor contains a central intratumoral compartment. "
                "Rare lymphocytes are interspersed among the tumor cells. "
                "Tumor nests form the local epithelial component."
            ),
            "new_prompt": (
                "The lung tumor contains a central intratumoral compartment. "
                "Small focal lymphocytic clusters are interspersed among the tumor cells. "
                "Tumor nests form the local epithelial component."
            ),
            "instruction": "Add mild intratumoral immune infiltrate in the center of the lung tumor.",
        },
        {
            "gt_summary": "lung, immune infiltration decrease, mild, lower left",
            "old_prompt": (
                "The lung specimen contains a lower left stromal compartment. "
                "Small loose lymphocytic aggregates are present among the collagen bundles. "
                "Fibrous stroma forms the local tissue background."
            ),
            "new_prompt": (
                "The lung specimen contains a lower left stromal compartment. "
                "Rare scattered lymphocytes are present among the collagen bundles. "
                "Fibrous stroma forms the local tissue background."
            ),
            "instruction": "Reduce the immune infiltrate mildly in the lower left stromal compartment of the lung specimen.",
        },
        {
            "gt_summary": "prostate, Gleason 4 to 5, mild, center",
            "old_prompt": (
                "The prostate specimen contains a central malignant epithelial focus. "
                "Fused and poorly formed glands show Gleason pattern 4 morphology in this focus. "
                "Nuclear atypia is present within the malignant epithelial cells."
            ),
            "new_prompt": (
                "The prostate specimen contains a central malignant epithelial focus. "
                "A small group of solid sheets and single cells shows Gleason pattern 5 morphology in this focus. "
                "Nuclear atypia is present within the malignant epithelial cells."
            ),
            "instruction": "Convert a mild portion of the central prostate tumor from Gleason pattern 4 to Gleason pattern 5 morphology.",
        },
        {
            "gt_summary": "prostate, Gleason 4 to 3, mild, center",
            "old_prompt": (
                "The prostate specimen contains a central malignant epithelial focus. "
                "Fused and poorly formed glands show focal Gleason pattern 4 morphology. "
                "Nuclear atypia is present within the malignant epithelial cells."
            ),
            "new_prompt": (
                "The prostate specimen contains a central malignant epithelial focus. "
                "Well-formed individual glands show focal Gleason pattern 3 morphology. "
                "Nuclear atypia is present within the malignant epithelial cells."
            ),
            "instruction": "Convert a mild portion of the central prostate tumor from Gleason pattern 4 to Gleason pattern 3 morphology.",
        },
        {
            "gt_summary": "colorectal, normal epithelium to adenomatous, mild, center",
            "old_prompt": (
                "The colorectal specimen contains a central glandular mucosal compartment. "
                "Regular glands with uniform epithelial nuclei occupy a small focus. "
                "The surrounding mucosa contains a collagenous stromal background."
            ),
            "new_prompt": (
                "The colorectal specimen contains a central glandular mucosal compartment. "
                "Mildly crowded adenomatous glands with elongated hyperchromatic nuclei occupy a small focus. "
                "The surrounding mucosa contains a collagenous stromal background."
            ),
            "instruction": "Convert central normal colorectal epithelium into mild adenomatous glandular epithelium.",
        },
    ]


def _normalize_checker_text(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

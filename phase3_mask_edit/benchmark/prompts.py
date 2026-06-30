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


def check_prompt_with_llm(intent: BenchmarkIntent, prompt: BenchmarkPrompt, checker: LLMConfig) -> BenchmarkPrompt:
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
                "The organ/site must match the GT exactly or use a direct synonym; reject reports that "
                "switch to another organ. The strength should match the GT magnitude, not a larger or smaller change. "
                "Reject if old_prompt or new_prompt uses imperative/edit wording such as edit, modify, "
                "change, increase, decrease, reduce, add, remove, make, or convert. These restrictions "
                "apply only to old_prompt and new_prompt. The instruction field is expected to be an edit "
                "command, so do not reject it for imperative wording; only check whether its edit semantics "
                "match the GT. Also reject old_prompt/new_prompt if either contains cross-report comparison "
                "or preservation wording such as unchanged, stable, remains, compared, prior, previously, "
                "surrounding tissue unchanged, no additional, no other, transition from, or preserving. "
                "Each report must read as an independent descriptive finding."
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
        "sample_id": intent.sample_id,
        "organ": intent.organ,
        "profile": intent.profile,
        "strength_guidance": _strength_report_guidance(intent.strength),
        "report_generation_contract": {
            "old_prompt": "A standalone descriptive pathology report for the reference image only.",
            "new_prompt": "A standalone descriptive pathology report for the target image only.",
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
            "The reports must not mention ref, target, before, after, editing, or that one report differs from the other. "
            "The parser will infer the edit later from the semantic difference, but the reports themselves must not explain that difference. "
            "instruction must be a direct user edit instruction, not a meta-instruction about modifying prompts. "
            "The old/new reports should be 2-4 sentences each, written like concise pathology findings. "
            "Both reports must mention the exact organ/site, the relevant tissue category, and the GT location. "
            "Do not add sentences about other areas being unchanged/stable/preserved. Do not mention internal primitive names, "
            "JSON fields, masks, GT, or benchmark. Do not use imperative/edit words in old_prompt or new_prompt: "
            "avoid edit, modify, change, increase, decrease, reduce, add, remove, make, convert. "
            "Also avoid any cross-report/comparison/preservation wording in old_prompt/new_prompt: unchanged, stable, "
            "remains, compared, prior, previously, preserved, additional, other areas, surrounding tissue unchanged, "
            "transition from, transitioned, more, less, larger, smaller, higher, lower, no longer, newly, and while. "
            "Each report should only describe its own image state. "
            "Use state-only descriptions: for low abundance say sparse/rare/scant/minimal; for high abundance say conspicuous/abundant/dense/prominent. "
            "Do not say 'more sparse', 'less conspicuous', 'more prominent', 'larger', 'smaller', or similar comparative phrases. "
            "For transition edits, old_prompt reports the source phenotype as an observation; new_prompt reports the target phenotype as an observation. "
            "Do not say 'transition from', 'replacing', or 'previously observed' in either report. "
            "Forbidden in old_prompt/new_prompt: increase, increased, decrease, decreased, reduce, reduced, enhance, enhanced, more, less, fewer, greater, lower, higher, larger, smaller, unchanged, stable, remains, compared, prior, previously, preserving, transition, replace. "
            "Allowed state-only wording examples: 'sparse lymphocytic infiltrate', 'mild focal lymphocytic infiltrate', 'prominent tumor nests', 'scattered tumor cells', 'central Gleason pattern 4 glands', 'central Gleason pattern 5 solid sheets'. "
            "Bad report wording examples: 'mild increase in immune infiltrate', 'tumor cells are less conspicuous', 'other areas are unchanged', 'transitioning from pattern 4'."
        ),
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
        compartment = "stromal" if primitive == "stromal_immune_infiltration" else "intratumoral"
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
            f"H&E stained {organ} pathology report. The {location} region contains non-neoplastic supporting tissue adjacent to tumor.",
            f"H&E stained {organ} pathology report. The {location} region contains a larger tumor component occupying the local tissue compartment.",
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
    if primitive in {"stroma_decrease", "stromal_reduction"}:
        return (
            f"H&E stained {organ} pathology report. The {location} region contains abundant stromal tissue.",
            f"H&E stained {organ} pathology report. The {location} region contains limited stromal tissue with adjacent non-stromal tissue occupying the area.",
        )
    if intent.expected_direction == "transition":
        return (
            f"H&E stained {organ} pathology report. The {location} region shows the source glandular or tumor phenotype specified by the case context.",
            f"H&E stained {organ} pathology report. The {location} region shows the target glandular or tumor phenotype specified by the case context.",
        )
    return (_baseline_prompt(intent), _baseline_prompt(intent))


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
    strength = str(response.get("strength") or "")
    direction = str(response.get("direction") or "")
    organ = _normalize_checker_text(response.get("organ"))
    expected_organ = _normalize_checker_text(intent.organ)
    organ_ok = (
        not organ
        or organ == expected_organ
        or expected_organ in organ
        or organ in expected_organ
    )
    return strength in {"", intent.strength} and direction in {"", intent.expected_direction} and organ_ok


REPORT_FORBIDDEN_TERMS = (
    "increase",
    "increased",
    "decrease",
    "decreased",
    "reduce",
    "reduced",
    "enhance",
    "enhanced",
    "more",
    "less",
    "fewer",
    "greater",
    "larger",
    "smaller",
    "unchanged",
    "stable",
    "remains",
    "remain ",
    "compared",
    "prior",
    "previously",
    "preserving",
    "transition",
    "transitioning",
    "replace",
    "replacing",
    "converted",
    "while",
    "other areas",
    "additional",
)

REPORT_FORBIDDEN_PATTERNS = tuple(
    re.compile(rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])")
    for term in REPORT_FORBIDDEN_TERMS
)


def validate_report_pair_language(prompt: BenchmarkPrompt) -> str | None:
    """Reject old/new report wording that leaks edit/comparison semantics."""

    for field_name, text in (
        ("old_prompt", prompt.old_prompt),
        ("new_prompt", prompt.new_prompt),
    ):
        normalized = str(text).lower()
        for term, pattern in zip(REPORT_FORBIDDEN_TERMS, REPORT_FORBIDDEN_PATTERNS):
            if pattern.search(normalized):
                return f"report_language_violation:{field_name} contains forbidden term {term!r}"
    return None


def _strength_report_guidance(strength: str) -> str:
    return {
        "mild": "Use a subtle/small report-level difference, e.g. conspicuous to slightly less conspicuous; do not describe a dramatic change.",
        "moderate": "Use a clear but not extreme report-level difference.",
        "significant": "Use a marked report-level difference.",
        "xlarge_deid": "Use a very large/extensive report-level difference.",
    }.get(strength, "Use a report-level difference matching the GT strength.")


def _report_pair_few_shots() -> list[dict[str, str]]:
    """Few-shot report pairs that avoid edit/comparison language in A-mode reports."""

    return [
        {
            "gt_summary": "breast, tumor burden decrease, mild, center",
            "old_prompt": (
                "The breast specimen shows several compact tumor nests in the central stromal compartment. "
                "The tumor cells form small epithelial clusters with mild nuclear atypia. "
                "Fibrous stroma is visible between the tumor nests."
            ),
            "new_prompt": (
                "The breast specimen shows a few scattered small tumor nests in the central stromal compartment. "
                "The tumor cells appear as limited epithelial clusters with mild nuclear atypia. "
                "Fibrous stroma is visible around the small tumor nests."
            ),
            "instruction": "Reduce the tumor burden mildly in the central stromal compartment of the breast specimen.",
        },
        {
            "gt_summary": "oral, tumor burden increase, mild, upper right peripheral",
            "old_prompt": (
                "The oral mucosal specimen shows scant tumor cells in the upper right peripheral region. "
                "The tumor component is focal and composed of rare atypical epithelial nests. "
                "The local connective tissue contains a loose stromal background."
            ),
            "new_prompt": (
                "The oral mucosal specimen shows mild focal tumor burden in the upper right peripheral region. "
                "Atypical epithelial nests are conspicuous within the local connective tissue. "
                "The tumor component occupies a small peripheral stromal compartment."
            ),
            "instruction": "Increase the tumor burden mildly in the upper right peripheral region of the oral mucosa.",
        },
        {
            "gt_summary": "breast, necrosis appearance, mild, center",
            "old_prompt": (
                "The breast tumor shows predominantly viable tumor cells in the central region. "
                "Only rare necrotic debris is present among compact tumor nests. "
                "The central tumor compartment has preserved cellular detail."
            ),
            "new_prompt": (
                "The breast tumor shows a mild focal necrotic area in the central region. "
                "Necrotic debris is visible among adjacent viable tumor nests. "
                "The central tumor compartment contains a small pale necrotic focus."
            ),
            "instruction": "Add mild focal necrosis in the central region of the breast tumor.",
        },
        {
            "gt_summary": "melanoma, necrosis resolution, mild, upper left",
            "old_prompt": (
                "The melanoma specimen shows focal necrotic debris in the upper left region. "
                "The necrotic compartment contains pale acellular material and fragmented nuclei. "
                "Adjacent stromal tissue is present at the edge of the necrotic focus."
            ),
            "new_prompt": (
                "The melanoma specimen shows collagenous stromal tissue intermingled with limited necrotic debris in the upper left region. "
                "The local compartment contains small foci of fragmented nuclei. "
                "Viable tissue elements are present around the residual necrotic material."
            ),
            "instruction": "Resolve a mild amount of necrosis in the upper left region of the melanoma specimen.",
        },
        {
            "gt_summary": "lung, stromal immune infiltration increase, mild, upper",
            "old_prompt": (
                "The lung specimen shows scant lymphocytic infiltrate in the upper stromal compartment. "
                "The stroma contains rare scattered immune cells between collagen bundles. "
                "Tumor-adjacent stromal tissue is lightly cellular."
            ),
            "new_prompt": (
                "The lung specimen shows mild focal lymphocytic infiltrate in the upper stromal compartment. "
                "Small clusters of immune cells are present between collagen bundles. "
                "Tumor-adjacent stromal tissue contains a visible inflammatory component."
            ),
            "instruction": "Add mild focal immune infiltrate in the upper stromal compartment of the lung specimen.",
        },
        {
            "gt_summary": "lung, intratumoral immune infiltration increase, mild, center",
            "old_prompt": (
                "The lung tumor shows scant immune cells in the central tumor compartment. "
                "Tumor nests are the dominant component in the center. "
                "Only rare lymphocytes are interspersed among tumor cells."
            ),
            "new_prompt": (
                "The lung tumor shows mild focal immune cells in the central tumor compartment. "
                "Small lymphocytic clusters are interspersed among tumor cells. "
                "The center contains a visible intratumoral inflammatory component."
            ),
            "instruction": "Add mild intratumoral immune infiltrate in the center of the lung tumor.",
        },
        {
            "gt_summary": "lung, immune infiltration decrease, mild, lower left",
            "old_prompt": (
                "The lung specimen shows mild lymphocytic infiltrate in the lower left stromal compartment. "
                "Immune cells form small loose aggregates among collagen bundles. "
                "The lower left stroma has a lightly inflammatory appearance."
            ),
            "new_prompt": (
                "The lung specimen shows scant lymphocytic infiltrate in the lower left stromal compartment. "
                "Scattered immune cells are present among collagen bundles. "
                "The lower left stroma has a mildly cellular appearance."
            ),
            "instruction": "Reduce the immune infiltrate mildly in the lower left stromal compartment of the lung specimen.",
        },
        {
            "gt_summary": "prostate, Gleason 4 to 5, mild, center",
            "old_prompt": (
                "The prostate specimen shows central tumor glands with predominant Gleason pattern 4 morphology. "
                "The glands are fused and poorly formed in the central tumor focus. "
                "Nuclear atypia is present within the malignant epithelial cells."
            ),
            "new_prompt": (
                "The prostate specimen shows central tumor with a small focus of Gleason pattern 5 morphology. "
                "Solid sheets and single malignant cells are present within part of the central tumor focus. "
                "Gleason pattern 4 glands are also visible in the central malignant component."
            ),
            "instruction": "Convert a mild portion of the central prostate tumor from Gleason pattern 4 to Gleason pattern 5 morphology.",
        },
        {
            "gt_summary": "prostate, Gleason 4 to 3, mild, center",
            "old_prompt": (
                "The prostate specimen shows central tumor with a focal Gleason pattern 4 component. "
                "Fused glands and poorly formed glandular structures are present in the central focus. "
                "The malignant glands show moderate architectural complexity."
            ),
            "new_prompt": (
                "The prostate specimen shows central tumor with a small Gleason pattern 3 component. "
                "Well-formed individual glands are present in part of the central focus. "
                "Some fused glands remain visible within the malignant component."
            ),
            "instruction": "Convert a mild portion of the central prostate tumor from Gleason pattern 4 to Gleason pattern 3 morphology.",
        },
        {
            "gt_summary": "colorectal, normal epithelium to adenomatous, mild, center",
            "old_prompt": (
                "The colorectal specimen shows central normal glandular epithelium. "
                "The glands have regular architecture and uniform epithelial nuclei. "
                "The central mucosa has a non-dysplastic appearance."
            ),
            "new_prompt": (
                "The colorectal specimen shows a small central focus of adenomatous glandular epithelium. "
                "The glands have mild crowding and elongated hyperchromatic nuclei. "
                "Adjacent central glands have regular low-grade architecture."
            ),
            "instruction": "Convert central normal colorectal epithelium into mild adenomatous glandular epithelium.",
        },
    ]


def _normalize_checker_text(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

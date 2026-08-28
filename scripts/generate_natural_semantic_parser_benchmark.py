#!/usr/bin/env python3
"""Generate a natural-language holdout from the frozen semantic benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from phase3_joint_edit_refine.compatible_api import (
    OpenAIChatCompletionsJSONClient,
)
from phase3_joint_edit_refine.semantic_request import semantic_request_from_metadata

DEFAULT_SOURCE = Path("benchmarks/semantic_parser_planner_v1/benchmark.jsonl")
DEFAULT_SOURCE_MANIFEST = Path("benchmarks/semantic_parser_planner_v1/manifest.json")
DEFAULT_OUTPUT_DIR = Path("benchmarks/semantic_parser_planner_natural_v1")
BENCHMARK_VERSION = "semantic-parser-planner-natural-benchmark-v1"

STYLE_PROFILES = (
    "concise pathologist instruction",
    "natural research-user request",
    "colloquial but unambiguous request",
    "polite request with ordinary function words",
    "compact annotation-review shorthand",
    "full-sentence morphology description",
    "hesitant wording that still preserves the requested endpoint",
    "request embedded in a brief editing rationale without adding a new intent",
)

GENERATOR_SYSTEM_PROMPT = """You write realistic Chinese and English user requests for a pathology mask-editing product.

Rewrite every supplied case once. Preserve exactly the supplied number of biological intentions, target, direction, polarity, clinical context, spatial scope, morphology, named cell class, strength, and ordered-versus-unordered relation. Do not add a diagnosis, treatment response, location, morphology, strength, or cell class that is absent. Deliberately underspecified cases must remain underspecified. A negated case must remain negated. An unordered conflict must remain unordered rather than becoming a sequence.

Make the language sound like something a pathologist, pathology researcher, annotation reviewer, or ordinary user might actually type. Vary syntax and wording substantially; do not merely prepend an adjective or replace one synonym. Keep Chinese idiomatic and English idiomatic. Do not expose internal enum values, schema fields, primitive names, IDs, or implementation details.

Return one rewritten complete instruction per case. Return only strict JSON."""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _generation_schema(batch_size: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["cases"],
        "properties": {
            "cases": {
                "type": "array",
                "minItems": batch_size,
                "maxItems": batch_size,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["case_id", "instruction"],
                    "properties": {
                        "case_id": {"type": "string"},
                        "instruction": {"type": "string", "minLength": 2},
                    },
                },
            }
        },
    }


def _generator_payload(record: dict[str, Any], *, index: int) -> dict[str, Any]:
    gold = record["gold_semantic_request"]
    return {
        "case_id": record["case_id"],
        "language": record["language"],
        "category": record["category"],
        "style_profile": STYLE_PROFILES[index % len(STYLE_PROFILES)],
        "original_instruction": record["instruction"],
        "intents": [
            {
                key: intent[key]
                for key in (
                    "intent_id",
                    "intent_type",
                    "target",
                    "operation",
                    "polarity",
                    "clinical_context",
                    "spatial_scope",
                    "morphology",
                    "cell_class",
                    "strength",
                )
            }
            for intent in gold["intents"]
        ],
        "relations": gold["relations"],
    }


def _validate_generated(
    batch: list[tuple[int, dict[str, Any]]], raw: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    cases = raw.get("cases")
    if not isinstance(cases, list) or len(cases) != len(batch):
        raise ValueError("generator returned the wrong number of cases")
    by_id = {str(item.get("case_id")): item for item in cases if isinstance(item, dict)}
    expected_ids = {record["case_id"] for _, record in batch}
    if set(by_id) != expected_ids:
        raise ValueError("generator returned detached or duplicate case IDs")
    forbidden = re.compile(
        r"(?:primitive|mechanism|intent-\d{3}|[a-z]+(?:_[a-z]+){1,}|-v1)",
        flags=re.IGNORECASE,
    )
    for _, record in batch:
        item = by_id[record["case_id"]]
        instruction = str(item.get("instruction") or "").strip()
        if not instruction:
            raise ValueError(f"{record['case_id']} has malformed generated text")
        if forbidden.search(instruction):
            raise ValueError(f"{record['case_id']} leaked internal vocabulary")
        if record["language"] == "zh" and not re.search(
            r"[\u4e00-\u9fff]", instruction
        ):
            raise ValueError(f"{record['case_id']} lost Chinese language identity")
        if record["language"] == "en" and re.search(r"[\u4e00-\u9fff]", instruction):
            raise ValueError(f"{record['case_id']} lost English language identity")
        _validate_semantic_surface(record, instruction)
    return by_id


def _validate_semantic_surface(record: dict[str, Any], instruction: str) -> None:
    """Reject common generator drifts without using the evaluated Parser."""

    lowered = instruction.casefold()
    intents = record["gold_semantic_request"]["intents"]
    polarity_text = re.sub(r"能不能|可不可以", "", lowered)
    negation = re.search(
        r"不|别|勿|避免|不要|不能|保持原样|维持不变|"
        r"do not|don't|without|avoid|must not|never|keep .{0,40} from",
        polarity_text,
    )
    polarities = {intent["polarity"] for intent in intents}
    if polarities == {"negated"} and not negation:
        raise ValueError(f"{record['case_id']} lost negation")
    if polarities == {"affirmed"} and negation:
        raise ValueError(f"{record['case_id']} inverted an affirmed edit")

    strengths = {intent["strength"] for intent in intents}
    mild = re.search(
        r"一点|一些|稍微|轻微|轻度|小幅|mild|slight|modest|a little|a bit|somewhat",
        lowered,
    )
    moderate = re.search(r"适中|中等|moderate", lowered)
    strong = re.search(r"明显|显著|强烈|大量|大幅|strong|marked|substantial", lowered)
    if strengths == {"unspecified"} and (mild or moderate or strong):
        raise ValueError(f"{record['case_id']} invented an edit strength")
    if "mild" in strengths and not mild:
        raise ValueError(f"{record['case_id']} lost the mild strength")

    contexts = {intent["clinical_context"] for intent in intents}
    if contexts == {"none"} and re.search(
        r"治疗|疗效|进展|恶化|退缩|缓解|残余|复发|"
        r"treat|therapy|progress|worsen|regress|response|residual|recurr",
        lowered,
    ):
        raise ValueError(f"{record['case_id']} invented a clinical context")

    morphologies = {intent["morphology"] for intent in intents}
    if record["category"] == "underspecified_intent" and morphologies == {
        "unspecified"
    }:
        if re.search(
            r"条索|巢|单列|单细胞|散落|细胞簇|前沿|"
            r"cord|nest|single[- ]file|single[- ]cell|scatter|cluster|front",
            lowered,
        ):
            raise ValueError(
                f"{record['case_id']} resolved deliberately unspecified morphology"
            )
        if not re.search(
            r"增加|增强|加重|更多|扩大|提高|"
            r"increase|more|enhance|worsen|expand|raise",
            lowered,
        ):
            raise ValueError(
                f"{record['case_id']} lost the requested increase direction"
            )

    relations = record["gold_semantic_request"]["relations"]
    relation_types = {item["relation_type"] for item in relations}
    sequential = re.search(
        r"先.{0,80}(?:再|然后|接着|随后)|然后|接着|随后|第一步|第二步|同时再|"
        r"(?:^|[，,、；;\s])再(?:把|将|对|让)|"
        r"\bfirst\b.{0,120}\b(?:then|next|afterwards)\b|"
        r"\bthen\b|\bafter that\b|\bafterwards\b|\bfollowed by\b",
        lowered,
    )
    if relation_types == {"explicit_sequence"} and not sequential:
        raise ValueError(f"{record['case_id']} lost explicit intent order")
    if relation_types == {"unordered"}:
        if sequential:
            raise ValueError(f"{record['case_id']} invented intent order")
        if not re.search(
            r"同时|并且|一边.{0,80}一边|既.{0,80}又|"
            r"\band\b|\bwhile\b|\bbut also\b|\bat the same time\b|\bsimultaneously\b",
            lowered,
        ):
            raise ValueError(f"{record['case_id']} lost the unordered relation")


def _generate_batch(
    batch: list[tuple[int, dict[str, Any]]],
    *,
    client: OpenAIChatCompletionsJSONClient,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    payload = [_generator_payload(record, index=index) for index, record in batch]
    feedback = ""
    last_error: Exception | None = None
    usage_rows: list[dict[str, Any]] = []
    for _ in range(3):
        try:
            raw, usage = client.call(
                system_prompt=GENERATOR_SYSTEM_PROMPT,
                user_prompt=json.dumps(
                    {
                        "cases": payload,
                        "validation_feedback_from_previous_attempt": feedback,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                image_paths=(),
                schema_name=f"natural_semantic_batch_{len(batch)}",
                json_schema=_generation_schema(len(batch)),
            )
            usage_rows.append(usage)
            return _validate_generated(batch, dict(raw)), _merge_usage(usage_rows)
        except Exception as exc:  # noqa: BLE001 - bounded generation repair
            last_error = exc
            feedback = str(exc)[:600]
    if len(batch) > 1:
        merged: dict[str, dict[str, Any]] = {}
        child_usage: list[dict[str, Any]] = [*usage_rows]
        for item in batch:
            generated, usage = _generate_batch([item], client=client)
            merged.update(generated)
            child_usage.append(usage)
        return merged, _merge_usage(child_usage)
    _, record = batch[0]
    return (
        {
            record["case_id"]: {
                "case_id": record["case_id"],
                "instruction": record["instruction"],
                "_generation_fallback": str(last_error)[:600],
            }
        },
        _merge_usage(usage_rows, fallback_count=1),
    )


def _merge_usage(
    rows: list[dict[str, Any]], *, fallback_count: int = 0
) -> dict[str, Any]:
    return {
        "model": next(
            (str(item.get("model")) for item in rows if item.get("model")), "unknown"
        ),
        "request_count": sum(int(item.get("request_count", 1)) for item in rows),
        "prompt_tokens": sum(int(item.get("prompt_tokens", 0)) for item in rows),
        "completion_tokens": sum(
            int(item.get("completion_tokens", 0)) for item in rows
        ),
        "total_tokens": sum(int(item.get("total_tokens", 0)) for item in rows),
        "generation_fallback_count": fallback_count
        + sum(int(item.get("generation_fallback_count", 0)) for item in rows),
    }


def _materialize_record(
    record: dict[str, Any],
    generated: dict[str, Any],
    *,
    serial: int,
    model: str,
) -> dict[str, Any]:
    instruction = str(generated["instruction"]).strip()
    gold = dict(record["gold_semantic_request"])
    gold.pop("request_sha256", None)
    gold["instruction"] = instruction
    gold["parser"] = "natural_benchmark_gold_v1"
    gold["parser_metadata"] = {
        "source_benchmark_case_id": record["case_id"],
        "generator_model": model,
        "gold_labels_changed": False,
    }
    gold["intents"] = [
        {**intent, "source_text": instruction} for intent in gold["intents"]
    ]
    request = semantic_request_from_metadata(gold)
    return {
        **record,
        "benchmark_version": BENCHMARK_VERSION,
        "case_id": f"spp-natural-v1-{serial:04d}",
        "base_case_id": record["case_id"],
        "instruction": instruction,
        "original_template_instruction": record["instruction"],
        "gold_semantic_request": request.to_metadata(),
        "natural_language_generation": {
            "model": model,
            "protocol": "chat_completions",
            "fallback_to_template": bool(generated.get("_generation_fallback")),
            "rewritten": (
                instruction.casefold() != str(record["instruction"]).strip().casefold()
            ),
        },
    }


def _assign_evaluation_splits(records: list[dict[str, Any]]) -> None:
    """Create a stratified holdout not used during prompt development.

    The first 16 source cases were used by the initial API smoke test, so they
    remain in the development split. The remaining cases are stratified by
    language and benchmark category, then deterministically sampled by hash.
    """

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        serial = int(str(record["base_case_id"]).rsplit("-", 1)[-1])
        record["evaluation_split"] = "development"
        if (
            serial <= 16
            or record["natural_language_generation"]["fallback_to_template"]
        ):
            continue
        key = (record["language"], record["category"])
        groups.setdefault(key, []).append(record)
    for group in groups.values():
        ranked = sorted(
            group,
            key=lambda item: hashlib.sha256(
                str(item["base_case_id"]).encode("utf-8")
            ).hexdigest(),
        )
        holdout_count = max(1, round(len(ranked) * 0.25))
        for record in ranked[:holdout_count]:
            record["evaluation_split"] = "final_holdout"


def _write_sample_review(records: list[dict[str, Any]], path: Path) -> None:
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        key = (
            record["language"],
            record["category"],
            record["case_profile"]["dataset"],
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(record)
    lines = [
        "# Natural-language benchmark sample review",
        "",
        "These examples are stratified by language, category and dataset. Gold labels were inherited from the frozen source benchmark and were not generated by the language model.",
        "",
        "| Case | Dataset | Language | Category | Original template | Natural rewrite |",
        "|---|---|---|---|---|---|",
    ]
    for record in selected:
        clean = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| `{record['case_id']}` | {record['case_profile']['dataset']} | "
            f"{record['language']} | `{record['category']}` | "
            f"{clean(record['original_template_instruction'])} | "
            f"{clean(record['instruction'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="gpt-5.4-mini-2026-03-17")
    parser.add_argument("--api-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--repair-existing",
        type=Path,
        help="Reuse valid rewrites from an existing natural benchmark and regenerate only invalid ones",
    )
    args = parser.parse_args()

    source_records = _load_jsonl(args.source)
    if args.limit is not None:
        source_records = source_records[: args.limit]
    indexed = list(enumerate(source_records))
    generated_by_id: dict[str, dict[str, Any]] = {}
    previous_usage: dict[str, Any] = {}
    repair_count = 0
    if args.repair_existing:
        existing_records = {
            item["base_case_id"]: item for item in _load_jsonl(args.repair_existing)
        }
        previous_report = args.repair_existing.parent / "generation_report.json"
        if previous_report.exists():
            previous_usage = json.loads(previous_report.read_text(encoding="utf-8"))
        invalid_ids: set[str] = set()
        for index, record in indexed:
            existing = existing_records.get(record["case_id"])
            if existing is None:
                invalid_ids.add(record["case_id"])
                continue
            candidate = {
                "case_id": record["case_id"],
                "instruction": existing["instruction"],
            }
            try:
                _validate_generated([(index, record)], {"cases": [candidate]})
                if existing.get("natural_language_generation", {}).get(
                    "fallback_to_template"
                ):
                    candidate["_generation_fallback"] = "reused prior fallback"
                generated_by_id[record["case_id"]] = candidate
            except ValueError:
                invalid_ids.add(record["case_id"])
        repair_count = len(invalid_ids)
        indexed_to_generate = [
            item for item in indexed if item[1]["case_id"] in invalid_ids
        ]
    else:
        indexed_to_generate = indexed
    batches = [
        indexed_to_generate[start : start + args.batch_size]
        for start in range(0, len(indexed_to_generate), args.batch_size)
    ]
    client = OpenAIChatCompletionsJSONClient(
        model=args.model,
        api_base_url=args.api_base_url,
        api_key_env=args.api_key_env,
        max_retries=4,
    )
    usage_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_batch = {
            executor.submit(_generate_batch, batch, client=client): batch
            for batch in batches
        }
        for future in as_completed(future_to_batch):
            generated, usage = future.result()
            generated_by_id.update(generated)
            usage_rows.append(usage)

    records = [
        _materialize_record(
            record,
            generated_by_id[record["case_id"]],
            serial=index + 1,
            model=args.model,
        )
        for index, record in indexed
    ]
    _assign_evaluation_splits(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    body = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    benchmark_path = args.output_dir / "benchmark.jsonl"
    benchmark_path.write_text(body, encoding="utf-8")
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    manifest = {
        **source_manifest,
        "benchmark_version": BENCHMARK_VERSION,
        "source_benchmark_version": source_manifest["benchmark_version"],
        "source_benchmark_sha256": source_manifest["benchmark_jsonl_sha256"],
        "benchmark_jsonl_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "record_count": len(records),
        "generator_model": args.model,
        "generator_protocol": "chat_completions",
        "gold_labels_changed": False,
        "substantively_rewritten_count": sum(
            bool(record["natural_language_generation"]["rewritten"])
            for record in records
        ),
        "language_counts": dict(
            sorted(Counter(record["language"] for record in records).items())
        ),
        "category_counts": dict(
            sorted(Counter(record["category"] for record in records).items())
        ),
        "evaluation_split_counts": dict(
            sorted(Counter(record["evaluation_split"] for record in records).items())
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    current_request_count = sum(
        int(item.get("request_count", 1)) for item in usage_rows
    )
    current_input_tokens = sum(int(item.get("prompt_tokens", 0)) for item in usage_rows)
    current_output_tokens = sum(
        int(item.get("completion_tokens", 0)) for item in usage_rows
    )
    current_total_tokens = sum(int(item.get("total_tokens", 0)) for item in usage_rows)
    usage_report = {
        "model": args.model,
        "request_count": int(previous_usage.get("request_count", 0))
        + current_request_count,
        "input_tokens": int(previous_usage.get("input_tokens", 0))
        + current_input_tokens,
        "output_tokens": int(previous_usage.get("output_tokens", 0))
        + current_output_tokens,
        "total_tokens": int(previous_usage.get("total_tokens", 0))
        + current_total_tokens,
        "generation_fallback_count": sum(
            bool(record["natural_language_generation"]["fallback_to_template"])
            for record in records
        ),
        "repair_candidate_count": repair_count,
        "repair_request_count": current_request_count,
        "repair_input_tokens": current_input_tokens,
        "repair_output_tokens": current_output_tokens,
        "repair_total_tokens": current_total_tokens,
    }
    (args.output_dir / "generation_report.json").write_text(
        json.dumps(usage_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_sample_review(records, args.output_dir / "natural_language_samples.md")
    print(
        json.dumps(
            {
                "benchmark": str(benchmark_path),
                "manifest": str(args.output_dir / "manifest.json"),
                "record_count": len(records),
                **usage_report,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

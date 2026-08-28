#!/usr/bin/env python3
"""Evaluate the v4 Semantic Request Parser and Program Planner.

Use ``--parser gold`` to audit the frozen Planner expectations without a
language model, ``--parser rule-based`` for the offline regression baseline,
or ``--parser api`` for the product Parser.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from phase3_joint_edit_refine.models import JointAreaBudget, JointCaseContext
from phase3_joint_edit_refine.program_planner import SemanticProgramPlanner
from phase3_joint_edit_refine.semantic_request import (
    OpenAISemanticRequestParser,
    RuleBasedSemanticRequestParser,
    SemanticRequest,
    semantic_request_from_metadata,
)
from phase3_mask_edit_refine.agents import OpenAIResponsesJSONClient

DEFAULT_BENCHMARK = Path("benchmarks/semantic_parser_planner_v1/benchmark.jsonl")
DEFAULT_MANIFEST = Path("benchmarks/semantic_parser_planner_v1/manifest.json")
INTENT_FIELDS = (
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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number} is not a JSON object")
        records.append(value)
    return records


def _case_stub(record: Mapping[str, Any]) -> JointCaseContext:
    profile = record["case_profile"]
    return JointCaseContext(
        case_id=str(record["case_id"]),
        instruction=str(record["instruction"]),
        source_image_uri="benchmark-image.png",
        source_tissue_mask_uri="benchmark-tissue.npy",
        source_nuclei_mask_uri="benchmark-nuclei.png",
        pathology_domain_id=str(profile["pathology_domain_id"]),
        annotation_profile_id=str(profile["annotation_profile_id"]),
        cell_observation_profile_id=str(profile["cell_observation_profile_id"]),
        cell_population_profile_id=str(profile["cell_population_profile_id"]),
        primitive_id="cohesive-boundary-expansion-v1",
        joint_area_budget=JointAreaBudget(
            target_fraction=0.04,
            min_fraction=0.01,
            max_fraction=0.08,
            tissue_min_fraction=0.01,
        ),
        seed=17,
        provenance={
            "source_image_sha256": "benchmark-image",
            "source_tissue_mask_sha256": "benchmark-tissue",
            "source_nuclei_mask_sha256": "benchmark-nuclei",
        },
    )


def _program_projection(program) -> dict[str, Any]:
    return {
        "status": program.status,
        "conflicts": list(program.conflicts),
        "steps": [
            {
                "intent_id": step.intent_id,
                "order_index": step.order_index,
                "depends_on": list(step.depends_on),
                "status": step.status,
                "selected_primitive_id": step.selected_primitive_id,
                "candidates": [
                    {
                        "primitive_id": item.primitive_id,
                        "semantic_priority": item.semantic_priority,
                        "compatible_mechanism_ids": list(item.compatible_mechanism_ids),
                    }
                    for item in step.candidates
                ],
            }
            for step in program.steps
        ],
    }


def _request_projection(request: SemanticRequest) -> dict[str, Any]:
    return {
        "intents": [
            {field: getattr(intent, field) for field in INTENT_FIELDS}
            for intent in request.intents
        ],
        "relations": sorted(
            (
                relation.before_intent_id,
                relation.after_intent_id,
                relation.relation_type,
            )
            for relation in request.relations
        ),
        "global_constraints": list(request.global_constraints),
    }


def _contains_key(value: Any, keys: set[str]) -> bool:
    if isinstance(value, Mapping):
        return bool(set(value) & keys) or any(
            _contains_key(item, keys) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_key(item, keys) for item in value)
    return False


def _parser_for(args: argparse.Namespace):
    if args.parser == "gold":
        return None
    if args.parser == "rule-based":
        return RuleBasedSemanticRequestParser()
    client = OpenAIResponsesJSONClient(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        api_base_url=args.api_base_url,
        api_key_env=args.api_key_env,
        timeout_sec=args.timeout_sec,
        max_retries=args.max_retries,
    )
    return OpenAISemanticRequestParser(client)


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _aggregate(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = list(rows)
    parser_items = [item for item in items if item["parser_scored"]]
    parsed_items = [item for item in parser_items if item["parse_success"]]
    summary = {
        "count": len(items),
        "parser_scored_count": len(parser_items),
        "parse_success_count": sum(
            bool(item["parse_success"]) for item in parser_items
        ),
        "parse_success_rate": _rate(
            sum(bool(item["parse_success"]) for item in parser_items),
            len(parser_items),
        ),
        "intent_count_exact_rate": _rate(
            sum(bool(item["intent_count_exact"]) for item in parsed_items),
            len(parsed_items),
        ),
        "semantic_request_exact_rate": _rate(
            sum(bool(item["semantic_request_exact"]) for item in parsed_items),
            len(parsed_items),
        ),
        "relation_exact_rate": _rate(
            sum(bool(item["relation_exact"]) for item in parsed_items),
            len(parsed_items),
        ),
        "primitive_leakage_count": sum(
            bool(item["primitive_leakage"]) for item in parsed_items
        ),
        "gold_planner_exact_rate": _rate(
            sum(bool(item["gold_planner_exact"]) for item in items), len(items)
        ),
        "parsed_pipeline_planner_exact_rate": _rate(
            sum(bool(item["parsed_pipeline_planner_exact"]) for item in parsed_items),
            len(parsed_items),
        ),
    }
    field_scores: dict[str, float | None] = {}
    for field in INTENT_FIELDS:
        correct = sum(
            int(item["intent_field_correct"].get(field, 0)) for item in parsed_items
        )
        total = sum(int(item["gold_intent_count"]) for item in parsed_items)
        field_scores[field] = _rate(correct, total)
    summary["intent_field_accuracy"] = field_scores
    return summary


def _fmt_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def _markdown_report(result: Mapping[str, Any]) -> str:
    overall = result["overall"]
    manifest = result["benchmark_manifest"]
    lines = [
        "# Semantic Parser / Planner benchmark v1",
        "",
        "## Scope / 范围",
        "",
        (
            f"This run evaluated **{overall['count']}** catalog-derived Chinese and "
            f"English instructions with parser mode **{result['parser_mode']}**."
        ),
        "",
        (
            f"本次运行使用 **{result['parser_mode']}** 模式，评估了 **{overall['count']}** "
            "条由 ontology 与 primitive catalog 派生的中英文指令。"
        ),
        "",
        "> This is a synthetic interface-conformance and regression benchmark. It is not evidence of unrestricted clinical-language understanding or mask quality.",
        "",
        "> 这是合成的接口一致性与回归测试，不代表系统已经具备不受限制的临床语言理解能力，也不评价最终 mask 的病理质量。",
        "",
        "## Overall results / 总体结果",
        "",
        "| Metric | Result | Interpretation |",
        "|---|---:|---|",
        f"| Parse success | {_fmt_rate(overall['parse_success_rate'])} | Returned a schema-valid semantic request |",
        f"| Intent-count exact | {_fmt_rate(overall['intent_count_exact_rate'])} | Preserved the number of user goals |",
        f"| Closed-ontology semantic exact | {_fmt_rate(overall['semantic_request_exact_rate'])} | All scored intent fields and relations matched |",
        f"| Relation exact | {_fmt_rate(overall['relation_exact_rate'])} | Preserved ordered versus unordered composition |",
        f"| Primitive leakage | {overall['primitive_leakage_count']} | Parser output must contain no primitive or mechanism IDs |",
        f"| Gold-request Planner replay exact | {_fmt_rate(overall['gold_planner_exact_rate'])} | Frozen catalog expectations reproduce in the current tree |",
        f"| Parsed end-to-end Planner exact | {_fmt_rate(overall['parsed_pipeline_planner_exact_rate'])} | Parser output led to the expected organ-compatible program |",
        "",
        "## Intent-field accuracy / 意图字段准确率",
        "",
        "| Field | Accuracy |",
        "|---|---:|",
    ]
    for field, value in overall["intent_field_accuracy"].items():
        lines.append(f"| `{field}` | {_fmt_rate(value)} |")
    lines.extend(
        [
            "",
            "## Breakdown by language / 按语言分层",
            "",
            "| Language | n | Parse | Semantic exact | Planner exact |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for language, item in result["by_language"].items():
        lines.append(
            f"| {language} | {item['count']} | {_fmt_rate(item['parse_success_rate'])} | "
            f"{_fmt_rate(item['semantic_request_exact_rate'])} | "
            f"{_fmt_rate(item['parsed_pipeline_planner_exact_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Breakdown by case type / 按测试类型分层",
            "",
            "| Category | n | Semantic exact | Planner exact |",
            "|---|---:|---:|---:|",
        ]
    )
    for category, item in result["by_category"].items():
        lines.append(
            f"| `{category}` | {item['count']} | "
            f"{_fmt_rate(item['semantic_request_exact_rate'])} | "
            f"{_fmt_rate(item['parsed_pipeline_planner_exact_rate'])} |"
        )
    orphan = manifest["executable_scope_without_open_profile_binding"]
    lines.extend(
        [
            "",
            "## Catalog audit / 目录审计",
            "",
            (
                f"The benchmark covers **{manifest['open_primitive_type_count']}** open "
                f"primitive types and **{manifest['open_organ_profile_binding_count']}** "
                "organ–profile bindings."
            ),
            "",
            (
                f"测试覆盖 **{manifest['open_primitive_type_count']}** 种当前开放的 primitive，"
                f"以及 **{manifest['open_organ_profile_binding_count']}** 个器官—标注体系绑定。"
            ),
            "",
            (
                "Executable-scope identifiers with no open canonical profile binding: "
                + (", ".join(f"`{item}`" for item in orphan) if orphan else "none")
                + "."
            ),
            "",
            "These identifiers are reported as catalog-consistency warnings and are not counted as supported benchmark primitives.",
            "",
            "这些标识仅作为目录一致性警告，不计入当前受支持的 benchmark primitive。",
            "",
            "## Interpretation / 解读",
            "",
        ]
    )
    if result["parser_mode"] == "gold":
        lines.extend(
            [
                "This run validates only the deterministic Planner projection from reviewed structured requests; it does not score natural-language parsing.",
                "",
                "本次运行仅验证已审查结构化请求到确定性 Planner 输出的映射，不评价自然语言解析。",
            ]
        )
    elif result["parser_mode"] == "rule-based":
        if overall["semantic_request_exact_rate"] == 1.0:
            lines.extend(
                [
                    "The offline rule-based parser passed every frozen generated form after catalog-phrase and connector regressions were corrected. This is a known-template conformance ceiling, not a measurement of the product API Parser.",
                    "",
                    "在修复目录短语与连接词回归后，离线规则解析器通过了全部冻结的生成式语句；这是已知模板上的一致性上限，不是产品 API Parser 的性能测量。",
                ]
            )
        else:
            lines.extend(
                [
                    "The rule-based parser is an offline regression baseline, not the product parser. Its errors identify linguistic forms that the API Parser test must cover before release.",
                    "",
                    "规则解析器只是离线回归基线，不是产品 Parser；它的错误用于定位正式 API Parser 在发布前必须覆盖的语言形式。",
                ]
            )
    else:
        lines.extend(
            [
                "The API run measures the frozen Parser prompt and model on synthetic catalog-derived language. An independent clinician-authored test set is still required for an external language-generalization claim.",
                "",
                "API 运行衡量的是冻结 prompt 与模型在目录派生合成语言上的表现；若要提出外部语言泛化结论，仍需独立的临床专家原创指令集。",
            ]
        )
    lines.extend(
        [
            "",
            f"Benchmark SHA-256: `{manifest['benchmark_jsonl_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--parser", choices=("gold", "rule-based", "api"), default="rule-based"
    )
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--api-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--language", choices=("zh", "en"))
    parser.add_argument("--category")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--write-predictions",
        action="store_true",
        help="Also retain per-case parser and Planner outputs as JSONL",
    )
    args = parser.parse_args()

    raw_body = args.benchmark.read_bytes()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    actual_sha = hashlib.sha256(raw_body).hexdigest()
    if actual_sha != manifest.get("benchmark_jsonl_sha256"):
        raise ValueError("benchmark JSONL digest does not match its manifest")
    records = _load_jsonl(args.benchmark)
    if args.language:
        records = [item for item in records if item["language"] == args.language]
    if args.category:
        records = [item for item in records if item["category"] == args.category]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError("benchmark filter selected no records")

    parser_impl = _parser_for(args)
    planner = SemanticProgramPlanner()
    results: list[dict[str, Any]] = []
    started = time.time()
    for record in records:
        gold = semantic_request_from_metadata(record["gold_semantic_request"])
        expected_program = record["expected_planner"]
        gold_program = _program_projection(
            planner.plan(gold, case_template=_case_stub(record))
        )
        row: dict[str, Any] = {
            "case_id": record["case_id"],
            "language": record["language"],
            "category": record["category"],
            "dataset": record["case_profile"]["dataset"],
            "instruction": record["instruction"],
            "gold_intent_count": len(gold.intents),
            "gold_planner_exact": gold_program == expected_program,
            "parser_scored": parser_impl is not None,
            "parse_success": True if parser_impl is not None else None,
            "parse_error": None,
            "intent_count_exact": True,
            "semantic_request_exact": True,
            "relation_exact": True,
            "primitive_leakage": False,
            "parsed_pipeline_planner_exact": True,
            "intent_field_correct": {
                field: len(gold.intents) for field in INTENT_FIELDS
            },
        }
        if parser_impl is not None:
            try:
                parsed = parser_impl.parse(str(record["instruction"]))
                gold_projection = _request_projection(gold)
                parsed_projection = _request_projection(parsed)
                row["intent_count_exact"] = len(parsed.intents) == len(gold.intents)
                row["relation_exact"] = (
                    parsed_projection["relations"] == gold_projection["relations"]
                )
                row["semantic_request_exact"] = parsed_projection == gold_projection
                row["primitive_leakage"] = _contains_key(
                    parsed.to_metadata(),
                    {"primitive_id", "primitive_hypotheses", "mechanism_id"},
                )
                field_correct = Counter()
                for index, gold_intent in enumerate(gold_projection["intents"]):
                    parsed_intent = (
                        parsed_projection["intents"][index]
                        if index < len(parsed_projection["intents"])
                        else {}
                    )
                    for field in INTENT_FIELDS:
                        field_correct[field] += int(
                            parsed_intent.get(field) == gold_intent[field]
                        )
                row["intent_field_correct"] = dict(field_correct)
                parsed_program = _program_projection(
                    planner.plan(parsed, case_template=_case_stub(record))
                )
                row["parsed_pipeline_planner_exact"] = (
                    parsed_program == expected_program
                )
                row["parsed_semantic_request"] = parsed.to_metadata()
                row["parsed_program"] = parsed_program
            except (
                Exception  # noqa: BLE001 - preserve per-case provider failures
            ) as exc:
                row.update(
                    {
                        "parse_success": False,
                        "parse_error": f"{type(exc).__name__}: {exc}",
                        "intent_count_exact": False,
                        "semantic_request_exact": False,
                        "relation_exact": False,
                        "primitive_leakage": False,
                        "parsed_pipeline_planner_exact": False,
                        "intent_field_correct": {},
                    }
                )
        results.append(row)

    by_language = {
        key: _aggregate(value)
        for key, value in sorted(_group(results, "language").items())
    }
    by_category = {
        key: _aggregate(value)
        for key, value in sorted(_group(results, "category").items())
    }
    report = {
        "benchmark_version": manifest["benchmark_version"],
        "parser_mode": args.parser,
        "model": args.model if args.parser == "api" else None,
        "reasoning_effort": args.reasoning_effort if args.parser == "api" else None,
        "elapsed_sec": round(time.time() - started, 3),
        "filters": {
            "language": args.language,
            "category": args.category,
            "limit": args.limit,
        },
        "overall": _aggregate(results),
        "by_language": by_language,
        "by_category": by_category,
        "benchmark_manifest": manifest,
    }
    output_dir = (
        args.output_dir
        or args.benchmark.parent / f"results_{args.parser.replace('-', '_')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.write_predictions:
        (output_dir / "predictions.jsonl").write_text(
            "".join(
                json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                for item in results
            ),
            encoding="utf-8",
        )
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(_markdown_report(report), encoding="utf-8")
    print(
        json.dumps(
            {"output_dir": str(output_dir), **report["overall"]}, ensure_ascii=False
        )
    )
    return 0


def _group(
    rows: Iterable[Mapping[str, Any]], key: str
) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        result[str(row[key])].append(row)
    return result


if __name__ == "__main__":
    raise SystemExit(main())

"""Run benchmark GT intents through Phase3 mask edit and report metrics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from phase3_mask_edit.benchmark.models import read_intents_jsonl, read_prompts_csv, write_eval_csv
from phase3_mask_edit.benchmark.runner import GT_MODE, INSTRUCTION_MODE, PROMPT_MODE, run_benchmark_sample
from phase3_mask_edit.core.mask_io import save_metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intents", required=True, type=Path)
    parser.add_argument(
        "--prompts",
        type=Path,
        help="Optional prompt CSV for legacy prompt/instruction parser benchmarks.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[GT_MODE],
        choices=[GT_MODE, PROMPT_MODE, INSTRUCTION_MODE],
        help="Use gt for the direct structured-intent benchmark; prompt/instruction are legacy parser modes.",
    )
    parser.add_argument(
        "--prompt-parser",
        default="api",
        choices=["api", "gt"],
        help="Use api for formal benchmark runs; gt is only for local smoke/debug.",
    )
    parser.add_argument(
        "--instruction-parser",
        default="api",
        choices=["api", "gt", "rule-based"],
        help="Use api for formal benchmark runs; gt/rule-based are only for local smoke/debug.",
    )
    parser.add_argument("--parser-model", default="")
    parser.add_argument("--parser-api-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--parser-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--contour-provider", default="fixture", choices=["fixture", "api-text", "api-vision"])
    parser.add_argument("--contour-fixture", type=Path)
    parser.add_argument("--contour-model", default="")
    parser.add_argument("--contour-api-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--contour-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-image-detail", default="high")
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--max-regions", type=int, default=8)
    parser.add_argument("--max-points-per-region", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args(argv)

    legacy_modes = {PROMPT_MODE, INSTRUCTION_MODE}.intersection(args.modes)
    if legacy_modes and args.prompts is None:
        parser.error("--prompts is required when running prompt or instruction modes")
    if legacy_modes and (args.prompt_parser == "api" or args.instruction_parser == "api") and not args.parser_model:
        parser.error("--parser-model is required when prompt/instruction parser is api")

    intents = read_intents_jsonl(args.intents)
    prompts = read_prompts_csv(args.prompts) if args.prompts else {}
    rows: list[dict[str, Any]] = []
    for intent in intents[: args.limit]:
        for mode in args.modes:
            prompt = prompts.get(intent.sample_id)
            if mode != GT_MODE and (prompt is None or prompt.checker_status.lower() != "accepted"):
                continue
            rows.append(
                run_benchmark_sample(
                    intent,
                    prompt,
                    mode=mode,
                    output_dir=args.output / "samples",
                    prompt_parser=args.prompt_parser,
                    instruction_parser=args.instruction_parser,
                    parser_api_base_url=args.parser_api_base_url,
                    parser_api_key_env=args.parser_api_key_env,
                    parser_model=args.parser_model,
                    contour_provider=args.contour_provider,
                    contour_api_base_url=args.contour_api_base_url,
                    contour_api_key_env=args.contour_api_key_env,
                    contour_model=args.contour_model,
                    contour_fixture=args.contour_fixture,
                    api_image_detail=args.api_image_detail,
                    max_attempts=args.max_attempts,
                    max_regions=args.max_regions,
                    max_points_per_region=args.max_points_per_region,
                )
            )
    args.output.mkdir(parents=True, exist_ok=True)
    write_eval_csv(rows, args.output / "benchmark_eval_results.csv")
    summary = summarize_rows(rows)
    save_metadata(summary, args.output / "benchmark_report.json")
    write_report_csv(summary, args.output / "benchmark_report.csv")
    if args.print_summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups["overall"] = rows
    for row in rows:
        for key in ("mode", "organ", "primitive", "strength"):
            groups[f"{key}:{row.get(key, '')}"].append(row)
        combo = "|".join(str(row.get(key, "")) for key in ("organ", "primitive", "strength", "mode"))
        groups[f"combo:{combo}"].append(row)
    summary = {name: _summarize_group(items) for name, items in sorted(groups.items())}
    summary["failed_samples"] = [row for row in rows if row.get("status") != "completed" or not _truthy(row.get("all_ok"))]
    return summary


def write_report_csv(summary: dict[str, Any], path: Path) -> Path:
    rows = [
        {"group": group, **metrics}
        for group, metrics in summary.items()
        if isinstance(metrics, dict) and "n" in metrics
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["group", "n", "completed", "all_pass_rate", "class_pass_rate", "direction_pass_rate", "strength_pass_rate", "location_pass_rate"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    completed = [row for row in rows if row.get("status") == "completed"]
    return {
        "n": n,
        "completed": len(completed),
        "all_pass_rate": _rate(rows, "all_ok"),
        "class_pass_rate": _rate(rows, "class_ok"),
        "direction_pass_rate": _rate(rows, "direction_ok"),
        "strength_pass_rate": _rate(rows, "strength_ok"),
        "location_pass_rate": _rate(rows, "location_ok"),
    }


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if _truthy(row.get(key))) / len(rows)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


if __name__ == "__main__":
    raise SystemExit(main())

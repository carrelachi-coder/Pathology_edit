"""Run parser-only mask-edit benchmark evaluation without contour generation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from phase3_mask_edit.benchmark.models import (
    BenchmarkIntent,
    BenchmarkPrompt,
    read_intents_jsonl,
    read_prompts_csv,
)
from phase3_mask_edit.benchmark.prompts import semantic_diff_for_intent
from phase3_mask_edit.parser.api_parser import ApiParserConfig, parse_prompts_with_api
from phase3_mask_edit.parser.instruction_parser import (
    InstructionParserConfig,
    parse_instruction,
)
from phase3_mask_edit.parser.semantic_diff import SEMANTIC_DIFF_SCHEMA_VERSION
from phase3_mask_edit.rules.semantic_to_intent import semantic_diff_to_intents

PROMPT_MODE = "prompt"
INSTRUCTION_MODE = "instruction"
PARSER_MODES = (PROMPT_MODE, INSTRUCTION_MODE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intents", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--parser-model", required=True)
    parser.add_argument("--parser-api-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--parser-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(PARSER_MODES),
        choices=PARSER_MODES,
    )
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--recompute-existing",
        action="store_true",
        help="Re-map saved parser JSON without making API or contour calls.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args(argv)

    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be >= 1")

    intents = read_intents_jsonl(args.intents)
    if args.limit is not None:
        intents = intents[: args.limit]
    prompts = read_prompts_csv(args.prompts)
    args.output.mkdir(parents=True, exist_ok=True)

    results_path = args.output / "parser_eval_results.csv"
    if args.recompute_existing:
        report = _recompute_existing_results(
            intents,
            prompts,
            modes=args.modes,
            model=args.parser_model,
            results_path=results_path,
            output_dir=args.output,
        )
        _write_manifest(args, intent_count=len(intents))
        if args.print_summary:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    rows_by_key = _read_existing_rows(results_path) if args.resume else {}
    _write_manifest(args, intent_count=len(intents))

    processed = 0
    for intent in intents:
        prompt = prompts.get(intent.sample_id)
        for mode in args.modes:
            key = (intent.sample_id, mode)
            existing = rows_by_key.get(key)
            if existing is not None and existing.get("status") == "completed":
                continue
            rows_by_key[key] = evaluate_parser_sample(
                intent,
                prompt,
                mode=mode,
                model=args.parser_model,
                api_base_url=args.parser_api_base_url,
                api_key_env=args.parser_api_key_env,
                output_dir=args.output / "samples",
            )
            processed += 1
            if processed % args.checkpoint_every == 0:
                _write_results(rows_by_key.values(), results_path)
                _write_report(rows_by_key.values(), args.output, args.parser_model)

    _write_results(rows_by_key.values(), results_path)
    report = _write_report(rows_by_key.values(), args.output, args.parser_model)
    if args.print_summary:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def evaluate_parser_sample(
    intent: BenchmarkIntent,
    prompt: BenchmarkPrompt | None,
    *,
    mode: str,
    model: str,
    api_base_url: str,
    api_key_env: str,
    output_dir: Path,
) -> dict[str, Any]:
    sample_dir = output_dir / intent.sample_id / mode
    sample_dir.mkdir(parents=True, exist_ok=True)
    expected = semantic_diff_for_intent(intent)
    parsed: dict[str, Any] | None = None
    parsed_primitives: list[str] = []
    try:
        if prompt is None:
            raise RuntimeError("prompt_missing")
        if prompt.checker_status.lower() != "accepted":
            raise RuntimeError(f"prompt_checker_{prompt.checker_status}")
        if mode == PROMPT_MODE:
            parsed = parse_prompts_with_api(
                prompt.old_prompt,
                prompt.new_prompt,
                config=ApiParserConfig(
                    model=model,
                    api_base_url=api_base_url,
                    api_key_env=api_key_env,
                    debug_dir=str(sample_dir / "api_parser_debug"),
                ),
            )
            planner_prompt = prompt.new_prompt
        elif mode == INSTRUCTION_MODE:
            parsed = parse_instruction(
                prompt.instruction,
                mode="api",
                config=InstructionParserConfig(
                    model=model,
                    api_base_url=api_base_url,
                    api_key_env=api_key_env,
                    debug_dir=str(sample_dir / "instruction_parser_debug"),
                ),
            )
            planner_prompt = prompt.instruction
        else:
            raise RuntimeError(f"unsupported mode: {mode}")

        (sample_dir / "semantic_diff.json").write_text(
            json.dumps(parsed, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        mapped_intents = semantic_diff_to_intents(
            parsed,
            reference_profile=intent.profile,
            old_prompt=prompt.old_prompt,
            new_prompt=planner_prompt,
        )
        parsed_primitives = [item.primitive for item in mapped_intents]
        (sample_dir / "semantic_mapping.json").write_text(
            json.dumps(
                {
                    "applicability_checked": False,
                    "parsed_primitives": parsed_primitives,
                    "intents": [item.to_metadata() for item in mapped_intents],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return _result_row(
            intent,
            mode=mode,
            model=model,
            status="completed",
            expected=expected,
            parsed=parsed,
            parsed_primitives=parsed_primitives,
            error="",
            output_dir=sample_dir,
        )
    except Exception as exc:
        (sample_dir / "failure.json").write_text(
            json.dumps({"status": "failed", "error": str(exc)}, indent=2),
            encoding="utf-8",
        )
        return _result_row(
            intent,
            mode=mode,
            model=model,
            status="failed",
            expected=expected,
            parsed=parsed,
            parsed_primitives=parsed_primitives,
            error=str(exc),
            output_dir=sample_dir,
        )


def _result_row(
    intent: BenchmarkIntent,
    *,
    mode: str,
    model: str,
    status: str,
    expected: Mapping[str, Any],
    parsed: Mapping[str, Any] | None,
    parsed_primitives: list[str],
    error: str,
    output_dir: Path,
) -> dict[str, Any]:
    expected_transition = expected["transition_change"]
    parsed_transition = (parsed or {}).get("transition_change") or {}
    expected_location = expected["lymphocyte_change"]["location"]
    parsed_location = ((parsed or {}).get("lymphocyte_change") or {}).get("location")
    transition_applicable = expected_transition["source_state"] != "none"
    location_applicable = expected_location != "unspecified"
    parsed_primitive = parsed_primitives[0] if len(parsed_primitives) == 1 else ""
    return {
        "sample_id": intent.sample_id,
        "organ": intent.organ,
        "profile": intent.profile,
        "mode": mode,
        "model": model,
        "status": status,
        "expected_primitive": intent.primitive,
        "parsed_primitive": parsed_primitive,
        "parsed_primitives": parsed_primitives,
        "primitive_exact": status == "completed"
        and parsed_primitives == [intent.primitive],
        "expected_transition_source": expected_transition["source_state"],
        "expected_transition_target": expected_transition["target_state"],
        "parsed_transition_source": parsed_transition.get("source_state", ""),
        "parsed_transition_target": parsed_transition.get("target_state", ""),
        "transition_applicable": transition_applicable,
        "transition_exact": (
            transition_applicable
            and parsed_transition.get("source_state")
            == expected_transition["source_state"]
            and parsed_transition.get("target_state")
            == expected_transition["target_state"]
        ),
        "expected_immune_location": expected_location,
        "parsed_immune_location": parsed_location or "",
        "immune_location_applicable": location_applicable,
        "immune_location_exact": location_applicable
        and parsed_location == expected_location,
        "parsed_semantic_diff": parsed,
        "error": error,
        "output_dir": str(output_dir),
    }


def _write_report(
    rows: Iterable[Mapping[str, Any]], output_dir: Path, model: str
) -> dict[str, Any]:
    materialized = [dict(row) for row in rows]
    modes: dict[str, Any] = {}
    for mode in PARSER_MODES:
        selected = [row for row in materialized if row.get("mode") == mode]
        completed = sum(row.get("status") == "completed" for row in selected)
        exact = sum(_as_bool(row.get("primitive_exact")) for row in selected)
        transition_rows = [
            row for row in selected if _as_bool(row.get("transition_applicable"))
        ]
        location_rows = [
            row for row in selected if _as_bool(row.get("immune_location_applicable"))
        ]
        modes[mode] = {
            "total": len(selected),
            "completed": completed,
            "completion_rate": _rate(completed, len(selected)),
            "primitive_exact": exact,
            "primitive_exact_rate": _rate(exact, len(selected)),
            "transition_total": len(transition_rows),
            "transition_exact": sum(
                _as_bool(row.get("transition_exact")) for row in transition_rows
            ),
            "transition_exact_rate": _rate(
                sum(_as_bool(row.get("transition_exact")) for row in transition_rows),
                len(transition_rows),
            ),
            "immune_location_total": len(location_rows),
            "immune_location_exact": sum(
                _as_bool(row.get("immune_location_exact")) for row in location_rows
            ),
            "immune_location_exact_rate": _rate(
                sum(
                    _as_bool(row.get("immune_location_exact")) for row in location_rows
                ),
                len(location_rows),
            ),
            "failures": len(selected) - completed,
        }
    prompt_rate = modes[PROMPT_MODE]["primitive_exact_rate"]
    instruction_rate = modes[INSTRUCTION_MODE]["primitive_exact_rate"]
    gate = {
        "prompt_primitive_exact_at_least_98pct": prompt_rate >= 0.98,
        "instruction_primitive_exact_100pct": instruction_rate == 1.0,
    }
    gate["passed"] = all(gate.values())
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parser_only": True,
        "contour_calls": 0,
        "mask_reads": 0,
        "applicability_checks": 0,
        "model": model,
        "semantic_diff_schema_version": SEMANTIC_DIFF_SCHEMA_VERSION,
        "modes": modes,
        "gate": gate,
    }
    (output_dir / "parser_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def _recompute_existing_results(
    intents: list[BenchmarkIntent],
    prompts: Mapping[str, BenchmarkPrompt],
    *,
    modes: Iterable[str],
    model: str,
    results_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if not results_path.exists():
        raise RuntimeError(f"Existing parser results are missing: {results_path}")
    existing = _read_existing_rows(results_path)
    expected_keys = {
        (intent.sample_id, mode) for intent in intents for mode in tuple(modes)
    }
    actual_keys = set(existing)
    if actual_keys != expected_keys:
        raise RuntimeError(
            "Existing parser result keys do not match the frozen evaluation set: "
            f"expected={len(expected_keys)} actual={len(actual_keys)} "
            f"missing={len(expected_keys - actual_keys)} "
            f"extra={len(actual_keys - expected_keys)}"
        )

    recomputed: list[dict[str, Any]] = []
    for intent in intents:
        prompt = prompts.get(intent.sample_id)
        if prompt is None:
            raise RuntimeError(f"Prompt missing during recompute: {intent.sample_id}")
        for mode in modes:
            existing_row = existing[(intent.sample_id, mode)]
            parsed = _decode_json_field(existing_row.get("parsed_semantic_diff"))
            parsed_primitives: list[str] = []
            if parsed is not None:
                semantic_text = (
                    prompt.new_prompt if mode == PROMPT_MODE else prompt.instruction
                )
                parsed_primitives = [
                    item.primitive
                    for item in semantic_diff_to_intents(
                        parsed,
                        reference_profile=intent.profile,
                        old_prompt=prompt.old_prompt,
                        new_prompt=semantic_text,
                    )
                ]
            recomputed.append(
                _result_row(
                    intent,
                    mode=mode,
                    model=model,
                    status=str(existing_row.get("status") or "failed"),
                    expected=semantic_diff_for_intent(intent),
                    parsed=parsed,
                    parsed_primitives=parsed_primitives,
                    error=str(existing_row.get("error") or ""),
                    output_dir=Path(
                        str(existing_row.get("output_dir") or output_dir / "samples")
                    ),
                )
            )

    _write_results(recomputed, results_path)
    return _write_report(recomputed, output_dir, model)


def _write_manifest(args: argparse.Namespace, *, intent_count: int) -> None:
    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "parser_only": True,
        "contour_calls": 0,
        "mask_reads": 0,
        "applicability_checks": 0,
        "recompute_existing": bool(args.recompute_existing),
        "semantic_diff_schema_version": SEMANTIC_DIFF_SCHEMA_VERSION,
        "model": args.parser_model,
        "modes": list(args.modes),
        "intent_count": intent_count,
        "inputs": {
            "intents": _file_record(args.intents),
            "prompts": _file_record(args.prompts),
        },
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_results(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    materialized = sorted(
        (dict(row) for row in rows),
        key=lambda row: (str(row.get("sample_id")), str(row.get("mode"))),
    )
    fieldnames = [
        "sample_id",
        "organ",
        "profile",
        "mode",
        "model",
        "status",
        "expected_primitive",
        "parsed_primitive",
        "parsed_primitives",
        "primitive_exact",
        "expected_transition_source",
        "expected_transition_target",
        "parsed_transition_source",
        "parsed_transition_target",
        "transition_applicable",
        "transition_exact",
        "expected_immune_location",
        "parsed_immune_location",
        "immune_location_applicable",
        "immune_location_exact",
        "parsed_semantic_diff",
        "error",
        "output_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _read_existing_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {
        (str(row.get("sample_id") or ""), str(row.get("mode") or "")): row
        for row in rows
    }


def _file_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, bool)) or value is None:
        return json.dumps(value, ensure_ascii=False)
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _decode_json_field(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    if not value:
        return None
    parsed = json.loads(str(value))
    if parsed is None:
        return None
    if not isinstance(parsed, dict):
        raise RuntimeError("parsed_semantic_diff must decode to an object")
    return parsed


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())

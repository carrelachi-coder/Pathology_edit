"""Run benchmark GT intents through Phase3 mask edit and report metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase3_mask_edit.benchmark.models import (
    read_intents_jsonl,
    read_prompts_csv,
    write_eval_csv,
)
from phase3_mask_edit.benchmark.metrics import mode_aware_score_fields, row_for_eval
from phase3_mask_edit.benchmark.reporting import (
    summarize_semantic_rows,
    write_semantic_report,
)
from phase3_mask_edit.benchmark.runner import (
    GT_MODE,
    INSTRUCTION_MODE,
    PROMPT_MODE,
    run_benchmark_sample,
)
from phase3_mask_edit.core.mask_io import save_metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intents", required=True, type=Path)
    parser.add_argument(
        "--prompts",
        type=Path,
        help="Prompt CSV required for formal prompt and instruction modes.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[GT_MODE],
        choices=[GT_MODE, PROMPT_MODE, INSTRUCTION_MODE],
        help="Report gt, prompt, and instruction as separate benchmark modes.",
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
    parser.add_argument(
        "--contour-provider",
        default="fixture",
        choices=["fixture", "api-text", "api-vision"],
    )
    parser.add_argument("--contour-fixture", type=Path)
    parser.add_argument("--contour-model", default="")
    parser.add_argument("--contour-api-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--contour-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-image-detail", default="high")
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument(
        "--semantic-repair-attempts",
        type=int,
        default=2,
        help=(
            "Additional parser calls after a valid semantic diff produces no "
            "executable planner intent."
        ),
    )
    parser.add_argument("--max-regions", type=int, default=8)
    parser.add_argument("--max-points-per-region", type=int, default=64)
    parser.add_argument(
        "--coordinate-tolerance-px",
        type=float,
        default=16.0,
        help="Clamp only contour coordinates this many pixels beyond a mask edge.",
    )
    parser.add_argument("--primitives", nargs="+", help="Optional primitive subset.")
    parser.add_argument(
        "--sample-ids", type=Path, help="Optional newline or JSONL sample-id subset."
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run selected sample/mode rows even when a completed row exists.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--no-retry-failed", action="store_true")
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args(argv)

    if args.semantic_repair_attempts < 0:
        parser.error("--semantic-repair-attempts must be >= 0")

    semantic_modes = {PROMPT_MODE, INSTRUCTION_MODE}.intersection(args.modes)
    if semantic_modes and args.prompts is None:
        parser.error("--prompts is required when running prompt or instruction modes")
    if (
        semantic_modes
        and (args.prompt_parser == "api" or args.instruction_parser == "api")
        and not args.parser_model
    ):
        parser.error("--parser-model is required when prompt/instruction parser is api")

    intents = read_intents_jsonl(args.intents)
    if args.primitives:
        selected_primitives = set(args.primitives)
        intents = [item for item in intents if item.primitive in selected_primitives]
    if args.sample_ids:
        selected_ids = _read_sample_ids(args.sample_ids)
        intents = [item for item in intents if item.sample_id in selected_ids]
    prompts = read_prompts_csv(args.prompts) if args.prompts else {}
    args.output.mkdir(parents=True, exist_ok=True)
    eval_path = args.output / "benchmark_eval_results.csv"
    existing_rows = (
        _read_eval_rows(eval_path) if args.resume and eval_path.exists() else []
    )
    rows_by_key = {
        (str(row.get("sample_id") or ""), str(row.get("mode") or "")): row
        for row in existing_rows
    }
    run_manifest = _run_manifest(args, intent_count=len(intents))
    run_manifest["resumed_rows"] = len(existing_rows)
    save_metadata(run_manifest, args.output / "run_manifest.json")
    processed_since_checkpoint = 0
    for intent in intents[: args.limit]:
        for mode in args.modes:
            key = (intent.sample_id, mode)
            existing_row = rows_by_key.get(key)
            if (
                not args.force_rerun
                and existing_row is not None
                and (existing_row.get("status") == "completed" or args.no_retry_failed)
            ):
                continue
            prompt = prompts.get(intent.sample_id)
            if mode != GT_MODE and (
                prompt is None or prompt.checker_status.lower() != "accepted"
            ):
                reason = (
                    "prompt_missing"
                    if prompt is None
                    else f"prompt_checker_{prompt.checker_status}:{prompt.checker_reason}"
                )
                rows_by_key[key] = row_for_eval(
                    sample_id=intent.sample_id,
                    mode=mode,
                    status="failed",
                    parsed_semantic_diff=None,
                    planned_primitive=None,
                    metrics=None,
                    error=reason,
                    output_dir=str(args.output / "samples" / intent.sample_id / mode),
                    organ=intent.organ,
                    profile=intent.profile,
                    primitive=intent.primitive,
                    strength=intent.strength,
                    source_dataset=intent.source_dataset,
                    wsi_id=intent.wsi_id,
                    patient_id=intent.patient_id,
                    ordinal_group_id=intent.ordinal_group_id,
                )
            else:
                rows_by_key[key] = run_benchmark_sample(
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
                    semantic_repair_attempts=args.semantic_repair_attempts,
                    max_regions=args.max_regions,
                    max_points_per_region=args.max_points_per_region,
                    coordinate_tolerance_px=args.coordinate_tolerance_px,
                )
            processed_since_checkpoint += 1
            if processed_since_checkpoint >= max(1, args.checkpoint_every):
                write_eval_csv(_sorted_rows(rows_by_key.values()), eval_path)
                processed_since_checkpoint = 0
    rows = [
        _with_mode_aware_eval_fields(
            _enrich_agentic_row_from_artifacts(row, max_attempts=args.max_attempts)
        )
        for row in _sorted_rows(rows_by_key.values())
    ]
    write_eval_csv(rows, eval_path)
    summary = summarize_rows(rows)
    save_metadata(summary, args.output / "benchmark_report.json")
    write_report_csv(summary, args.output / "benchmark_report.csv")
    semantic_report = summarize_semantic_rows(
        rows,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    write_semantic_report(semantic_report, args.output)
    run_manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    run_manifest["result_counts"] = {
        "rows": len(rows),
        "completed": sum(row.get("status") == "completed" for row in rows),
        "primary_ok": sum(
            mode_aware_score_fields(
                row,
                mode=str(row.get("mode") or ""),
            )["primary_ok"]
            for row in rows
        ),
        "all_ok": sum(_truthy(row.get("all_ok")) for row in rows),
    }
    save_metadata(run_manifest, args.output / "run_manifest.json")
    if args.print_summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored_rows = [_with_mode_aware_eval_fields(row) for row in rows]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups["overall"] = scored_rows
    for row in scored_rows:
        for key in ("mode", "organ", "primitive", "strength"):
            groups[f"{key}:{row.get(key, '')}"].append(row)
        combo = "|".join(
            str(row.get(key, "")) for key in ("organ", "primitive", "strength", "mode")
        )
        groups[f"combo:{combo}"].append(row)
    summary = {name: _summarize_group(items) for name, items in sorted(groups.items())}
    summary["failed_samples"] = [
        row
        for row in scored_rows
        if row.get("status") != "completed" or not _truthy(row.get("primary_ok"))
    ]
    return summary


def write_report_csv(summary: dict[str, Any], path: Path) -> Path:
    rows = [
        {
            "group": group,
            **metrics,
            "cumulative_success_at_k": json.dumps(
                metrics.get("cumulative_success_at_k", {}),
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
        for group, metrics in summary.items()
        if isinstance(metrics, dict) and "n" in metrics
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "group",
                "n",
                "completed",
                "primary_pass_rate",
                "semantic_core_pass_rate",
                "intended_magnitude_bucket_agreement_rate",
                "strict_all_pass_rate",
                "all_pass_rate",
                "class_pass_rate",
                "direction_pass_rate",
                "strength_pass_rate",
                "location_pass_rate",
                "contour_attempted",
                "mean_attempt_count",
                "first_attempt_success_rate",
                "replan_rate",
                "repair_attempted",
                "repair_success_rate",
                "final_contour_success_rate",
                "cumulative_success_at_k",
                "semantic_attempted",
                "semantic_replan_rate",
                "semantic_repair_attempted",
                "semantic_repair_success_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    completed = [row for row in rows if row.get("status") == "completed"]
    contour_rows = [row for row in rows if _integer(row.get("attempt_count")) > 0]
    repair_rows = [row for row in contour_rows if _truthy(row.get("replanned"))]
    semantic_rows = [
        row for row in rows if _integer(row.get("semantic_attempt_count")) > 0
    ]
    semantic_repair_rows = [
        row for row in semantic_rows if _truthy(row.get("semantic_replanned"))
    ]
    return {
        "n": n,
        "completed": len(completed),
        "primary_pass_rate": _rate(rows, "primary_ok"),
        "semantic_core_pass_rate": _rate(rows, "semantic_core_ok"),
        "intended_magnitude_bucket_agreement_rate": _rate(
            rows, "intended_magnitude_bucket_agreement"
        ),
        "strict_all_pass_rate": _rate(rows, "strict_all_ok"),
        "all_pass_rate": _rate(rows, "all_ok"),
        "class_pass_rate": _rate(rows, "class_ok"),
        "direction_pass_rate": _rate(rows, "direction_ok"),
        "strength_pass_rate": _rate(rows, "strength_ok"),
        "location_pass_rate": _rate(rows, "location_ok"),
        "contour_attempted": len(contour_rows),
        "mean_attempt_count": (
            sum(_integer(row.get("attempt_count")) for row in contour_rows)
            / len(contour_rows)
            if contour_rows
            else 0.0
        ),
        "first_attempt_success_rate": _status_rate(
            contour_rows,
            key="first_attempt_status",
            expected="validated",
        ),
        "replan_rate": len(repair_rows) / len(contour_rows) if contour_rows else 0.0,
        "repair_attempted": len(repair_rows),
        "repair_success_rate": _rate(repair_rows, "repair_success"),
        "final_contour_success_rate": _status_rate(
            contour_rows,
            key="final_attempt_status",
            expected="validated",
        ),
        "cumulative_success_at_k": _cumulative_success_rates(contour_rows),
        "semantic_attempted": len(semantic_rows),
        "semantic_replan_rate": (
            len(semantic_repair_rows) / len(semantic_rows) if semantic_rows else 0.0
        ),
        "semantic_repair_attempted": len(semantic_repair_rows),
        "semantic_repair_success_rate": _rate(
            semantic_repair_rows,
            "semantic_repair_success",
        ),
    }


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if _truthy(row.get(key))) / len(rows)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _integer(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _status_rate(
    rows: list[dict[str, Any]],
    *,
    key: str,
    expected: str,
) -> float:
    if not rows:
        return 0.0
    return sum(str(row.get(key) or "") == expected for row in rows) / len(rows)


def _cumulative_success_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    payloads = [_mapping(row.get("cumulative_success_at_k")) for row in rows]
    indices = sorted(
        {_integer(key) for payload in payloads for key in payload if _integer(key) > 0}
    )
    return {
        str(index): sum(_truthy(payload.get(str(index))) for payload in payloads)
        / len(rows)
        for index in indices
    }


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _read_sample_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            text = line.strip()
            if not text:
                continue
            if text.startswith("{"):
                payload = json.loads(text)
                text = str(payload.get("sample_id") or "")
            if text:
                ids.add(text)
    return ids


def _read_eval_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _enrich_agentic_row_from_artifacts(
    row: dict[str, Any],
    *,
    max_attempts: int,
) -> dict[str, Any]:
    enriched = dict(row)
    output_dir = Path(str(row.get("output_dir") or ""))
    execution_path = output_dir / "phase3_mask_edit" / "execution_summary.json"
    if execution_path.is_file():
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        attempts = execution.get("attempts") or []
        statuses = [
            str(attempt.get("status") or "")
            for attempt in attempts
            if isinstance(attempt, dict)
        ]
        successful_attempt = next(
            (
                index
                for index, status in enumerate(statuses, start=1)
                if status == "validated"
            ),
            None,
        )
        final_status = statuses[-1] if statuses else ""
        validated = str(execution.get("status") or "") == "validated"
        enriched.update(
            {
                "attempt_count": len(statuses),
                "first_attempt_status": statuses[0] if statuses else "",
                "final_attempt_status": final_status,
                "replanned": len(statuses) > 1,
                "repair_success": len(statuses) > 1 and validated,
                "terminal_failure_reason": (
                    ""
                    if validated
                    else str(
                        execution.get("error") or final_status or "proposal_failed"
                    )
                ),
                "cumulative_success_at_k": {
                    str(index): (
                        successful_attempt is not None and successful_attempt <= index
                    )
                    for index in range(1, max(0, max_attempts) + 1)
                },
            }
        )

    semantic_path = output_dir / "semantic_planning_trace.json"
    if semantic_path.is_file():
        semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
        for key in (
            "semantic_attempt_count",
            "semantic_first_attempt_status",
            "semantic_final_attempt_status",
            "semantic_replanned",
            "semantic_repair_success",
            "semantic_terminal_failure_reason",
        ):
            enriched[key] = semantic.get(key, enriched.get(key, ""))
    elif str(row.get("mode") or "") in {PROMPT_MODE, INSTRUCTION_MODE}:
        planning_path = output_dir / "planning_summary.json"
        if planning_path.is_file():
            planning = json.loads(planning_path.read_text(encoding="utf-8"))
            planned = bool(planning.get("intents"))
            status = "planned" if planned else "planner_no_executable_intents"
            enriched.update(
                {
                    "semantic_attempt_count": 1,
                    "semantic_first_attempt_status": status,
                    "semantic_final_attempt_status": status,
                    "semantic_replanned": False,
                    "semantic_repair_success": False,
                    "semantic_terminal_failure_reason": "" if planned else status,
                }
            )

    if str(enriched.get("status") or "") == "completed":
        enriched["failure_stage"] = ""
    elif not enriched.get("failure_stage"):
        semantic_final = str(enriched.get("semantic_final_attempt_status") or "")
        if semantic_final and semantic_final != "planned":
            enriched["failure_stage"] = "semantic_planning"
        elif _integer(enriched.get("attempt_count")) > 0:
            enriched["failure_stage"] = "contour"
        else:
            enriched["failure_stage"] = "pre_contour"
    return enriched


def _sorted_rows(rows: Any) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (str(row.get("sample_id") or ""), str(row.get("mode") or "")),
    )


def _with_mode_aware_eval_fields(row: dict[str, Any]) -> dict[str, Any]:
    scored = dict(row)
    scored.update(
        mode_aware_score_fields(
            scored,
            mode=str(scored.get("mode") or ""),
        )
    )
    return scored


def _run_manifest(args: argparse.Namespace, *, intent_count: int) -> dict[str, Any]:
    return {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [
            sys.executable,
            "-m",
            "phase3_mask_edit.cli.run_mask_edit_benchmark",
            *sys.argv[1:],
        ],
        "arguments": {key: _jsonable(value) for key, value in vars(args).items()},
        "intent_count_after_filters": intent_count,
        "inputs": {
            "intents": {
                "path": str(args.intents.resolve()),
                "sha256": _sha256(args.intents),
            },
            "prompts": (
                {"path": str(args.prompts.resolve()), "sha256": _sha256(args.prompts)}
                if args.prompts
                else None
            ),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "git_commit": _git_output("rev-parse", "HEAD"),
            "git_status": _git_output("status", "--short"),
        },
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception as exc:
        return f"unavailable:{exc}"


if __name__ == "__main__":
    raise SystemExit(main())

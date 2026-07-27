"""Generate and optionally LLM-check prompts for benchmark GT intents."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from phase3_mask_edit.benchmark.models import (
    BenchmarkPrompt,
    read_intents_jsonl,
    read_prompts_csv,
    write_prompts_csv,
)
from phase3_mask_edit.benchmark.prompts import (
    LLMConfig,
    accepted_prompts,
    generate_prompts,
    validate_report_pair_language,
    write_manual_review_csv,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--intents", required=True, type=Path, help="benchmark_intents.jsonl"
    )
    parser.add_argument("--output", required=True, type=Path, help="Output directory.")
    parser.add_argument("--use-llm-generator", action="store_true")
    parser.add_argument("--generator-model", default="template")
    parser.add_argument("--generator-api-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--generator-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--use-llm-checker", action="store_true")
    parser.add_argument("--checker-model", default="not_checked")
    parser.add_argument("--checker-api-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--checker-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--use-parser-checker", action="store_true")
    parser.add_argument("--parser-checker-model", default="not_checked")
    parser.add_argument(
        "--parser-checker-api-base-url", default="https://api.openai.com/v1"
    )
    parser.add_argument("--parser-checker-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--manual-review-per-group", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--checker-repair-attempts",
        type=int,
        default=0,
        help="Regenerate a prompt after checker rejection, carrying forward its reason.",
    )
    parser.add_argument(
        "--retry-rejected",
        action="store_true",
        help="On resume, retry existing prompts whose checker status is not accepted.",
    )
    parser.add_argument(
        "--retry-existing",
        action="store_true",
        help="On resume, regenerate every selected intent while preserving unselected rows.",
    )
    parser.add_argument(
        "--retry-language-violations",
        action="store_true",
        help=(
            "On resume, regenerate only existing prompts rejected by the current "
            "deterministic standalone-report language validator."
        ),
    )
    parser.add_argument("--no-retry-errors", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--primitives", nargs="+")
    parser.add_argument("--sample-ids", type=Path)
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args(argv)

    intents = read_intents_jsonl(args.intents)
    if args.primitives:
        selected = set(args.primitives)
        intents = [item for item in intents if item.primitive in selected]
    if args.sample_ids:
        selected_ids = {
            line.strip()
            for line in args.sample_ids.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        intents = [item for item in intents if item.sample_id in selected_ids]
    if args.limit is not None:
        intents = intents[: args.limit]
    generator = LLMConfig(
        model=args.generator_model,
        api_base_url=args.generator_api_base_url,
        api_key_env=args.generator_api_key_env,
    )
    checker = LLMConfig(
        model=args.checker_model,
        api_base_url=args.checker_api_base_url,
        api_key_env=args.checker_api_key_env,
    )
    parser_checker = LLMConfig(
        model=args.parser_checker_model,
        api_base_url=args.parser_checker_api_base_url,
        api_key_env=args.parser_checker_api_key_env,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    prompts_path = args.output / "benchmark_prompts.csv"
    existing = (
        read_prompts_csv(prompts_path) if args.resume and prompts_path.exists() else {}
    )
    prompts_by_id = dict(existing)
    completed_ids = (
        set()
        if args.retry_existing
        else {
            sample_id
            for sample_id, prompt in existing.items()
            if args.no_retry_errors
            or (
                not prompt.checker_reason.startswith("generation_error:")
                and not (
                    args.retry_language_violations
                    and validate_report_pair_language(prompt) is not None
                )
                and not (
                    args.retry_rejected and prompt.checker_status.lower() != "accepted"
                )
            )
        }
    )
    for index, intent in enumerate(
        (item for item in intents if item.sample_id not in completed_ids), start=1
    ):
        prompt = None
        last_error: Exception | None = None
        for attempt in range(1, max(1, args.max_retries) + 1):
            try:
                repair_feedback = ""
                repair_prompt: BenchmarkPrompt | None = None
                for _ in range(max(0, args.checker_repair_attempts) + 1):
                    prompt = generate_prompts(
                        [intent],
                        generator=generator,
                        checker=checker,
                        use_llm_generator=args.use_llm_generator,
                        use_llm_checker=args.use_llm_checker,
                        parser_checker=parser_checker,
                        use_parser_checker=args.use_parser_checker,
                        repair_feedback=repair_feedback,
                        repair_prompt=repair_prompt,
                    )[0]
                    if (
                        not args.use_llm_checker
                        or prompt.checker_status.lower() == "accepted"
                    ):
                        break
                    repair_feedback = prompt.checker_reason
                    repair_prompt = prompt
                break
            except Exception as exc:
                last_error = exc
                if attempt < max(1, args.max_retries):
                    time.sleep(min(30.0, float(2**attempt)))
        if prompt is None:
            prompt = BenchmarkPrompt(
                sample_id=intent.sample_id,
                old_prompt="",
                new_prompt="",
                instruction="",
                generator_model=generator.model,
                checker_model=checker.model,
                checker_status="rejected",
                checker_reason=f"generation_error:{last_error}",
            )
        prompts_by_id[intent.sample_id] = prompt
        if index % max(1, args.checkpoint_every) == 0:
            write_prompts_csv(
                sorted(prompts_by_id.values(), key=lambda item: item.sample_id),
                prompts_path,
            )
    prompts = sorted(prompts_by_id.values(), key=lambda item: item.sample_id)
    write_prompts_csv(prompts, prompts_path)
    accepted = accepted_prompts(prompts)
    write_prompts_csv(accepted, args.output / "benchmark_prompts.accepted.csv")
    write_manual_review_csv(
        accepted,
        args.output / "benchmark_prompts.manual_review.csv",
        per_group=args.manual_review_per_group,
        intents_by_id={item.sample_id: item for item in intents},
    )
    if args.print_summary:
        print(
            json.dumps(
                {"num_prompts": len(prompts), "num_accepted": len(accepted)},
                indent=2,
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

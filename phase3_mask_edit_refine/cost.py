"""Versioned GPT-5.6 token cost accounting for audit artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

PRICE_VERSION = "openai-public-2026-08-04"
USD_PER_MILLION_TOKENS = {
    "gpt-5.6-luna": {"input": 1.00, "cached_input": 0.10, "output": 6.00},
    "gpt-5.6-terra": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
    "gpt-5.6-sol": {"input": 5.00, "cached_input": 0.50, "output": 30.00},
    "gpt-5.6": {"input": 5.00, "cached_input": 0.50, "output": 30.00},
}


def call_cost_usd(usage: Mapping[str, Any]) -> float | None:
    model = usage.get("model")
    if not isinstance(model, str):
        return None
    rate = _model_rate(model)
    if rate is None:
        return None
    input_tokens = _number(usage.get("input_tokens"))
    output_tokens = _number(usage.get("output_tokens"))
    details = usage.get("input_tokens_details")
    cached_tokens = 0.0
    if isinstance(details, Mapping):
        cached_tokens = _number(details.get("cached_tokens"))
    uncached_tokens = max(0.0, input_tokens - cached_tokens)
    return (
        uncached_tokens * rate["input"]
        + cached_tokens * rate["cached_input"]
        + output_tokens * rate["output"]
    ) / 1_000_000.0


def summarize_cost(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    known_total = 0.0
    unknown = 0
    per_call: list[dict[str, Any]] = []
    for usage in calls:
        cost = call_cost_usd(usage)
        if cost is None:
            unknown += 1
        else:
            known_total += cost
        per_call.append(
            {
                "model": usage.get("model"),
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cost_usd": cost,
            }
        )
    return {
        "price_version": PRICE_VERSION,
        "known_total_usd": known_total,
        "unknown_price_call_count": unknown,
        "calls": per_call,
    }


def _model_rate(model: str) -> dict[str, float] | None:
    if model in USD_PER_MILLION_TOKENS:
        return USD_PER_MILLION_TOKENS[model]
    for prefix, rate in USD_PER_MILLION_TOKENS.items():
        if model.startswith(prefix + "-"):
            return rate
    return None


def _number(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0

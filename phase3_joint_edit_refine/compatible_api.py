"""Small OpenAI-compatible JSON client used by semantic API evaluations."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phase3_mask_edit_refine.agents import AgentProviderError


@dataclass(frozen=True)
class OpenAIChatCompletionsJSONClient:
    """Call an OpenAI-compatible Chat Completions strict-JSON endpoint.

    The product Parser remains protocol-independent because this class exposes
    the same ``call`` boundary as ``OpenAIResponsesJSONClient``. API keys are
    read only from the named environment variable.
    """

    model: str
    api_base_url: str
    api_key_env: str = "OPENAI_API_KEY"
    timeout_sec: float = 180.0
    max_retries: int = 4

    def call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_paths: Sequence[str | Path],
        schema_name: str,
        json_schema: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], dict[str, Any]]:
        if image_paths:
            raise AgentProviderError(
                "semantic Chat Completions client is intentionally text-only"
            )
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise AgentProviderError(
                f"missing API key environment variable: {self.api_key_env}"
            )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": dict(json_schema),
                },
            },
        }
        response = self._post(payload, api_key=api_key)
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise AgentProviderError(
                "Chat Completions response contains no completion choice"
            )
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str):
            raise AgentProviderError(
                "Chat Completions response contains no text JSON content"
            )
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise AgentProviderError(
                "Chat Completions output was not valid JSON"
            ) from exc
        if not isinstance(parsed, Mapping):
            raise AgentProviderError(
                "Chat Completions JSON output root must be an object"
            )
        usage = response.get("usage")
        metadata = dict(usage) if isinstance(usage, Mapping) else {}
        metadata.update(
            {
                "model": str(response.get("model") or self.model),
                "protocol": "chat_completions",
                "prompt_sha256": hashlib.sha256(
                    (
                        system_prompt
                        + "\n"
                        + user_prompt
                        + "\n"
                        + json.dumps(json_schema, sort_keys=True)
                    ).encode("utf-8")
                ).hexdigest(),
                "image_sha256": [],
            }
        )
        return parsed, metadata

    def _post(self, payload: Mapping[str, Any], *, api_key: str) -> dict[str, Any]:
        endpoint = self.api_base_url.rstrip("/") + "/chat/completions"
        request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                endpoint,
                data=request_body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_sec
                ) as response:
                    decoded = json.loads(response.read().decode("utf-8"))
                    if not isinstance(decoded, dict):
                        raise AgentProviderError(
                            "Chat Completions response root must be an object"
                        )
                    return decoded
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = AgentProviderError(
                    f"Chat Completions HTTP {exc.code}: {body[:800]}"
                )
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
            if attempt < self.max_retries:
                time.sleep(min(8.0, 0.75 * (2**attempt)))
        raise AgentProviderError(
            f"Chat Completions request failed after retries: {last_error}"
        )

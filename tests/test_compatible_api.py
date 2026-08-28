"""Unit tests for the text-only OpenAI-compatible JSON client."""

from __future__ import annotations

import json

import pytest

from phase3_joint_edit_refine.compatible_api import (
    OpenAIChatCompletionsJSONClient,
)
from phase3_mask_edit_refine.agents import AgentProviderError


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_chat_completions_client_requests_strict_json(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _Response(
            {
                "model": "fixture-model",
                "choices": [{"message": {"content": '{"answer":"ok"}'}}],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            }
        )

    monkeypatch.setenv("TEST_COMPATIBLE_API_KEY", "secret-fixture")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAIChatCompletionsJSONClient(
        model="fixture-model",
        api_base_url="https://example.invalid/v1",
        api_key_env="TEST_COMPATIBLE_API_KEY",
    )
    parsed, metadata = client.call(
        system_prompt="system",
        user_prompt="user",
        image_paths=(),
        schema_name="fixture_schema",
        json_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["answer"],
            "properties": {"answer": {"type": "string"}},
        },
    )

    payload = json.loads(captured["request"].data.decode("utf-8"))
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert parsed == {"answer": "ok"}
    assert metadata["protocol"] == "chat_completions"
    assert metadata["total_tokens"] == 5


def test_chat_completions_client_never_accepts_images(monkeypatch):
    monkeypatch.setenv("TEST_COMPATIBLE_API_KEY", "secret-fixture")
    client = OpenAIChatCompletionsJSONClient(
        model="fixture-model",
        api_base_url="https://example.invalid/v1",
        api_key_env="TEST_COMPATIBLE_API_KEY",
    )

    with pytest.raises(AgentProviderError, match="text-only"):
        client.call(
            system_prompt="system",
            user_prompt="user",
            image_paths=("image.png",),
            schema_name="fixture_schema",
            json_schema={"type": "object"},
        )

"""Filesystem handoff for fresh, account-authenticated Codex CLI Planner calls.

The GPU executor may run on a host without Codex. It writes an immutable
candidate-packet request; a separate local CLI worker returns one JSON choice.
No API key, conversation history, H&E image, or executor authority crosses this
boundary. The existing Planner still validates every returned candidate ID.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CodexCLIQueueError(RuntimeError):
    pass


@dataclass(frozen=True)
class CodexCLIQueueJSONClient:
    queue_root: Path
    model: str = "gpt-5.6-terra"
    reasoning_effort: str = "medium"
    timeout_sec: float = 900.0
    poll_sec: float = 0.5

    def call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_paths: Sequence[str | Path],
        schema_name: str,
        json_schema: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], dict[str, Any]]:
        if self.model != "gpt-5.6-terra":
            raise CodexCLIQueueError("D/E CLI Planner is pinned to gpt-5.6-terra")
        root = Path(self.queue_root)
        requests = root / "requests"
        responses = root / "responses"
        requests.mkdir(parents=True, exist_ok=True)
        responses.mkdir(parents=True, exist_ok=True)
        request_id = uuid.uuid4().hex
        staging = requests / ("." + request_id + ".tmp")
        final = requests / request_id
        staging.mkdir()
        images = []
        for index, source in enumerate(image_paths):
            path = Path(source)
            payload = path.read_bytes()
            name = f"mask_{index:02d}{path.suffix.lower()}"
            (staging / name).write_bytes(payload)
            images.append(
                {"name": name, "sha256": hashlib.sha256(payload).hexdigest()}
            )
        prompt_digest = hashlib.sha256(
            (system_prompt + "\n" + user_prompt + "\n" +
             json.dumps(json_schema, sort_keys=True)).encode("utf-8")
        ).hexdigest()
        request = {
            "request_id": request_id,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "schema_name": schema_name,
            "json_schema": dict(json_schema),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "prompt_sha256": prompt_digest,
            "images": images,
        }
        (staging / "request.json").write_text(
            json.dumps(request, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        staging.rename(final)

        response_path = responses / f"{request_id}.json"
        deadline = time.monotonic() + self.timeout_sec
        while not response_path.is_file():
            if time.monotonic() >= deadline:
                raise CodexCLIQueueError(
                    f"Codex CLI response timed out for {request_id}"
                )
            time.sleep(self.poll_sec)
        response = json.loads(response_path.read_text(encoding="utf-8"))
        if response.get("request_id") != request_id:
            raise CodexCLIQueueError("Codex CLI response ID is detached")
        if response.get("prompt_sha256") != prompt_digest:
            raise CodexCLIQueueError("Codex CLI response prompt hash is detached")
        if response.get("error"):
            raise CodexCLIQueueError(str(response["error"]))
        parsed = response.get("output")
        if not isinstance(parsed, Mapping):
            raise CodexCLIQueueError("Codex CLI output root must be an object")
        usage = {
            "transport": "codex_cli_ephemeral",
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "prompt_sha256": prompt_digest,
            "image_sha256": [item["sha256"] for item in images],
            "request_id": request_id,
            "session_id": response.get("session_id"),
            "deterministic": False,
        }
        return parsed, usage

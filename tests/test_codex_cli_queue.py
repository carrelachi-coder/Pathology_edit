"""Fresh-session CLI handoff and candidate-packet binding tests."""

from __future__ import annotations

import json
import hashlib
import subprocess
import threading
import time
from pathlib import Path

import pytest

from phase3_joint_edit_refine.codex_cli_queue import CodexCLIQueueJSONClient
from scripts.codex_cli_planner_worker import (
    _check_stripped_constraints,
    _cli_compatible_schema,
    _process_one,
    _run_packet,
    _remote_transfer,
)


def test_remote_transfer_retries_transient_ssh_failure(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if len(calls) == 1:
            raise subprocess.CalledProcessError(255, command, stderr="temporary SSH failure")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("scripts.codex_cli_planner_worker.time.sleep", lambda _: None)
    result = _remote_transfer(["ssh", "amax2", "true"])
    assert result.stdout == "ok"
    assert len(calls) == 2


def test_cli_schema_projection_preserves_original_array_constraints():
    original = {
        "type": "object",
        "properties": {
            "ids": {"type": "array", "items": {"type": "string"},
                    "minItems": 1, "uniqueItems": True},
        },
    }
    projected = _cli_compatible_schema(original)
    assert "uniqueItems" not in projected["properties"]["ids"]
    assert "minItems" not in projected["properties"]["ids"]
    _check_stripped_constraints({"ids": ["C1"]}, original)
    with pytest.raises(ValueError, match="duplicate items"):
        _check_stripped_constraints({"ids": ["C1", "C1"]}, original)


def test_cli_queue_round_trip_uses_fresh_terra_session(tmp_path):
    fake = tmp_path / "fake-codex"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json,sys\n"
        "a=sys.argv[1:]\n"
        "assert a[0]=='exec' and '--ephemeral' in a\n"
        "assert a[a.index('-m')+1]=='gpt-5.6-terra'\n"
        "assert '--output-schema' in a and 'resume' not in a\n"
        "with open(a[a.index('-o')+1],'w') as f: json.dump({'choice':'C1'},f)\n"
        "print('session id: 01234567-89ab-cdef-0123-456789abcdef',file=sys.stderr)\n"
    )
    fake.chmod(0o755)
    image = tmp_path / "mask.png"
    image.write_bytes(b"categorical-mask-bytes")
    root = tmp_path / "queue"
    client = CodexCLIQueueJSONClient(root, timeout_sec=10, poll_sec=0.01)
    received = {}

    def caller():
        received["result"] = client.call(
            system_prompt="Select a certified candidate only",
            user_prompt='{"candidates":["C1","C2"]}',
            image_paths=(image,),
            schema_name="candidate_choice",
            json_schema={"type": "object", "properties": {"choice": {"type": "string"}}},
        )

    thread = threading.Thread(target=caller)
    thread.start()
    deadline = time.monotonic() + 5
    while not list((root / "requests").glob("*/request.json")):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    request_id = next((root / "requests").iterdir()).name
    response = _process_one(
        request_id, root=str(root), host=None, codex=str(fake), timeout=5,
    )
    thread.join(timeout=5)
    assert not thread.is_alive()
    output, usage = received["result"]
    assert output == {"choice": "C1"}
    assert usage["transport"] == "codex_cli_ephemeral"
    assert usage["model"] == "gpt-5.6-terra"
    assert response["prompt_sha256"] == usage["prompt_sha256"]
    assert response["session_id"] == "01234567-89ab-cdef-0123-456789abcdef"


def test_cli_worker_retries_original_nonempty_array_constraint(tmp_path):
    fake = tmp_path / "fake-codex"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json,pathlib,sys\n"
        "a=sys.argv[1:]; n=pathlib.Path('attempts')\n"
        "i=int(n.read_text())+1 if n.exists() else 1; n.write_text(str(i))\n"
        "value=[] if i==1 else ['capacity_margin']\n"
        "pathlib.Path(a[a.index('-o')+1]).write_text(json.dumps({'ids':value}))\n"
        "print('session id: 01234567-89ab-cdef-0123-456789abcdef',file=sys.stderr)\n"
    )
    fake.chmod(0o755)
    schema = {"type": "object", "properties": {"ids": {
        "type": "array", "items": {"type": "string"}, "minItems": 1,
    }}}
    packet = tmp_path / "packet"
    packet.mkdir()
    system = "Select a supported metric"
    user = '{"metrics":["capacity_margin"]}'
    digest = hashlib.sha256(
        (system + "\n" + user + "\n" + json.dumps(schema, sort_keys=True)).encode()
    ).hexdigest()
    (packet / "request.json").write_text(json.dumps({
        "request_id": "a" * 32, "model": "gpt-5.6-terra",
        "reasoning_effort": "medium", "schema_name": "test",
        "system_prompt": system, "user_prompt": user,
        "prompt_sha256": digest, "json_schema": schema, "images": [],
    }))
    response = _run_packet(packet, codex=str(fake), timeout=5)
    assert response["output"] == {"ids": ["capacity_margin"]}
    assert response["attempt_count"] == 2
    assert "too few items" in response["retry_errors"][0]


def test_cli_worker_retries_transient_terra_transport_error(tmp_path, monkeypatch):
    fake = tmp_path / "fake-codex"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json,pathlib,sys\n"
        "a=sys.argv[1:]; n=pathlib.Path('transport_attempts')\n"
        "i=int(n.read_text())+1 if n.exists() else 1; n.write_text(str(i))\n"
        "if i==1:\n"
        " print('stream disconnected: tls handshake eof',file=sys.stderr); sys.exit(1)\n"
        "pathlib.Path(a[a.index('-o')+1]).write_text(json.dumps({'choice':'C1'}))\n"
        "print('session id: 01234567-89ab-cdef-0123-456789abcdef',file=sys.stderr)\n"
    )
    fake.chmod(0o755)
    schema = {"type": "object", "properties": {"choice": {"type": "string"}}}
    packet = tmp_path / "packet"
    packet.mkdir()
    system = "Select a certified option"
    user = '{"options":["C1"]}'
    digest = hashlib.sha256(
        (system + "\n" + user + "\n" + json.dumps(schema, sort_keys=True)).encode()
    ).hexdigest()
    (packet / "request.json").write_text(json.dumps({
        "request_id": "b" * 32, "model": "gpt-5.6-terra",
        "reasoning_effort": "medium", "schema_name": "test",
        "system_prompt": system, "user_prompt": user,
        "prompt_sha256": digest, "json_schema": schema, "images": [],
    }))
    monkeypatch.setattr("scripts.codex_cli_planner_worker.time.sleep", lambda _: None)
    response = _run_packet(packet, codex=str(fake), timeout=5)
    assert response["output"] == {"choice": "C1"}
    assert response["total_attempt_count"] == 2
    assert len(response["transport_retry_errors"]) == 1

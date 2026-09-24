"""Answer certified Planner packets with a fresh GPT-5.6 Terra CLI session.

Run on the local machine with Codex account authentication. The GPU workflow
may write requests on a remote server; --remote-host copies only the packet's
mask images and schema into an isolated temporary directory. Every call uses
`codex exec --ephemeral` and never resumes the current conversation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
import shlex
import subprocess
import tempfile
import time
from pathlib import Path


_CLI_UNSUPPORTED_SCHEMA_KEYS = frozenset({
    "uniqueItems", "minItems", "maxItems", "minimum", "maximum",
    "minLength", "maxLength", "pattern",
})


def _cli_compatible_schema(value):
    if isinstance(value, dict):
        return {
            key: _cli_compatible_schema(child)
            for key, child in value.items()
            if key not in _CLI_UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(value, list):
        return [_cli_compatible_schema(child) for child in value]
    return value


def _check_stripped_constraints(output, schema, location="root"):
    if not isinstance(schema, dict):
        return
    if "enum" in schema and output not in schema["enum"]:
        raise ValueError(f"{location}: value is outside original schema enum")
    if isinstance(output, (int, float)) and not isinstance(output, bool):
        if "minimum" in schema and output < schema["minimum"]:
            raise ValueError(f"{location}: value is below original minimum")
        if "maximum" in schema and output > schema["maximum"]:
            raise ValueError(f"{location}: value is above original maximum")
    if isinstance(output, list):
        if "minItems" in schema and len(output) < schema["minItems"]:
            raise ValueError(f"{location}: too few items")
        if "maxItems" in schema and len(output) > schema["maxItems"]:
            raise ValueError(f"{location}: too many items")
        if schema.get("uniqueItems") and len({json.dumps(x, sort_keys=True) for x in output}) != len(output):
            raise ValueError(f"{location}: duplicate items")
        for index, item in enumerate(output):
            _check_stripped_constraints(item, schema.get("items"), f"{location}[{index}]")
    if isinstance(output, dict):
        for key, item in output.items():
            _check_stripped_constraints(
                item, schema.get("properties", {}).get(key), f"{location}.{key}"
            )


def _original_constraint_hints(schema, location="root"):
    """Describe constraints removed only for the CLI's schema subset."""
    if not isinstance(schema, dict):
        return []
    hints = []
    for key in ("minItems", "maxItems", "minimum", "maximum"):
        if key in schema:
            hints.append(f"{location}: {key}={schema[key]}")
    if schema.get("uniqueItems"):
        hints.append(f"{location}: array items must be unique")
    for key, child in schema.get("properties", {}).items():
        hints.extend(_original_constraint_hints(child, f"{location}.{key}"))
    if "items" in schema:
        hints.extend(_original_constraint_hints(schema["items"], f"{location}[]"))
    return hints


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _remote_transfer(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Retry transient SSH/SCP failures without repeating a Terra decision."""
    last_error = None
    for attempt in range(5):
        try:
            return subprocess.run(
                command, capture_output=True, text=True, timeout=120, check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            last_error = exc
            if attempt < 4:
                time.sleep(min(2 ** attempt, 8))
    assert last_error is not None
    raise last_error


def _run_packet(packet_dir: Path, *, codex: str, timeout: int) -> dict:
    request = json.loads((packet_dir / "request.json").read_text(encoding="utf-8"))
    if request["model"] != "gpt-5.6-terra":
        raise ValueError("worker accepts only gpt-5.6-terra")
    if not re.fullmatch(r"[0-9a-f]{32}", request["request_id"]):
        raise ValueError("invalid queue request ID")
    images = []
    for item in request["images"]:
        name = item["name"]
        if not re.fullmatch(r"mask_[0-9]{2}\.(png|jpg|jpeg|webp)", name):
            raise ValueError("invalid mask image name")
        path = packet_dir / name
        if _sha(path.read_bytes()) != item["sha256"]:
            raise ValueError("mask image digest mismatch")
        images.append(path)
    expected_prompt = _sha(
        (request["system_prompt"] + "\n" + request["user_prompt"] + "\n" +
         json.dumps(request["json_schema"], sort_keys=True)).encode("utf-8")
    )
    if expected_prompt != request["prompt_sha256"]:
        raise ValueError("planner packet prompt digest mismatch")

    schema = packet_dir / "response_schema.json"
    output = packet_dir / "response_text.json"
    schema.write_text(
        json.dumps(_cli_compatible_schema(request["json_schema"])),
        encoding="utf-8",
    )
    prompt = (
        "Follow these system instructions for this isolated Planner call:\n"
        + request["system_prompt"]
        + "\n\nCertified Planner input packet (JSON):\n"
        + request["user_prompt"]
        + "\n\nReturn only one JSON object satisfying the supplied output schema. "
        "Use no tools and do not inspect local files."
    )
    hints = _original_constraint_hints(request["json_schema"])
    if hints:
        prompt += "\n\nAdditional original schema requirements (the CLI schema cannot encode these):\n"
        prompt += "\n".join("- " + item for item in hints)
    command = [
        codex, "exec", "-m", "gpt-5.6-terra", "-s", "read-only",
        "-C", str(packet_dir), "--skip-git-repo-check", "--ephemeral",
        "-c", f'model_reasoning_effort="{request["reasoning_effort"]}"',
        "--output-schema", str(schema), "-o", str(output),
    ]
    # `--image` accepts one or more values; place the positional stdin marker
    # before images so argparse cannot consume the prompt as another filename.
    command.append("-")
    for path in images:
        command.extend(["-i", str(path)])
    retry_errors = []
    for attempt in range(1, 4):
        completed = subprocess.run(
            command, cwd=packet_dir, input=prompt,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=timeout, check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"Codex CLI exit {completed.returncode}: {completed.stderr[-1000:]}"
            )
        raw = json.loads(output.read_text(encoding="utf-8"))
        try:
            if not isinstance(raw, dict):
                raise ValueError("Codex CLI output is not a JSON object")
            _check_stripped_constraints(raw, request["json_schema"])
        except ValueError as exc:
            retry_errors.append(str(exc))
            if attempt == 3:
                raise
            prompt += ("\n\nYour previous JSON violated the original output schema: "
                       + str(exc) + ". Produce a corrected JSON object. "
                       "Every required nonempty array must contain a supported ID. "
                       "Previous JSON: " + json.dumps(raw, ensure_ascii=False))
            continue
        match = re.search(r"session id: ([0-9a-f-]+)", completed.stderr)
        return {
            "request_id": request["request_id"],
            "prompt_sha256": request["prompt_sha256"],
            "output": raw,
            "session_id": match.group(1) if match else None,
            "attempt_count": attempt,
            "retry_errors": retry_errors,
            "effective_prompt_sha256": _sha(prompt.encode("utf-8")),
        }
    raise AssertionError("unreachable CLI retry state")


def _pending_ids(*, root: str, host: str | None) -> list[str]:
    if host:
        command = ["ssh", "-o", "BatchMode=yes", host,
                   "ls -1 " + shlex.quote(root.rstrip("/") + "/requests")]
        completed = _remote_transfer(command)
        names = completed.stdout.splitlines()
        completed = _remote_transfer(
            ["ssh", "-o", "BatchMode=yes", host,
             "ls -1 " + shlex.quote(root.rstrip("/") + "/responses")],
        )
        done = {name.removesuffix(".json") for name in completed.stdout.splitlines()}
    else:
        base = Path(root)
        names = [item.name for item in (base / "requests").iterdir()]
        done = {item.stem for item in (base / "responses").glob("*.json")}
    return sorted(
        name for name in names
        if re.fullmatch(r"[0-9a-f]{32}", name) and name not in done
    )


def _process_one(request_id: str, *, root: str, host: str | None,
                 codex: str, timeout: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="codex-planner-") as directory:
        local = Path(directory)
        if host:
            remote = root.rstrip("/") + "/requests/" + request_id
            _remote_transfer(["scp", "-q", "-r", f"{host}:{remote}", str(local)])
            packet = local / request_id
        else:
            packet = Path(root) / "requests" / request_id
        request = json.loads((packet / "request.json").read_text(encoding="utf-8"))
        try:
            response = _run_packet(packet, codex=codex, timeout=timeout)
        except Exception as exc:
            response = {
                "request_id": request_id,
                "prompt_sha256": request.get("prompt_sha256"),
                "error": f"{type(exc).__name__}: {exc}",
            }
        payload = json.dumps(response, ensure_ascii=False, sort_keys=True)
        if host:
            local_response = local / f"{request_id}.json"
            local_response.write_text(payload, encoding="utf-8")
            temp_name = "." + request_id + ".tmp"
            responses = root.rstrip("/") + "/responses/"
            _remote_transfer(["scp", "-q", str(local_response),
                              f"{host}:{responses}{temp_name}"])
            _remote_transfer(
                ["ssh", "-o", "BatchMode=yes", host,
                 "mv " + shlex.quote(responses + temp_name) + " "
                 + shlex.quote(responses + request_id + ".json")],
            )
        else:
            target = Path(root) / "responses" / f"{request_id}.json"
            temporary = target.with_suffix(".tmp")
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(target)
        return response


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--remote-host")
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--idle-exit-seconds", type=float, default=0,
        help="Exit after this much idle time following at least one response",
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--parallelism", type=int, default=1)
    args = parser.parse_args()
    if args.parallelism < 1 or (args.once and args.parallelism != 1):
        parser.error("--parallelism must be positive and --once requires one worker")
    processed = 0
    idle_since = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.parallelism) as executor:
        active = {}
        while True:
            for future, request_id in list(active.items()):
                if not future.done():
                    continue
                del active[future]
                try:
                    response = future.result()
                except Exception as exc:
                    response = {"request_id": request_id,
                                "error": f"transport error: {type(exc).__name__}: {exc}"}
                print(json.dumps({k: response.get(k) for k in
                                  ("request_id", "session_id", "error")}), flush=True)
                processed += 1
                idle_since = time.monotonic()
                if args.once:
                    return 1 if response.get("error") else 0
            try:
                pending = _pending_ids(root=args.queue_root, host=args.remote_host)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                # A transient SSH listing failure must not terminate the queue
                # consumer while GPU cases are waiting for Planner responses.
                print(json.dumps({
                    "warning": "queue_listing_retry",
                    "return_code": getattr(exc, "returncode", None),
                    "stderr": str(getattr(exc, "stderr", "") or "")[-500:],
                }), flush=True)
                time.sleep(5.0)
                continue
            active_ids = set(active.values())
            available = args.parallelism - len(active)
            for request_id in (name for name in pending if name not in active_ids):
                if available <= 0:
                    break
                future = executor.submit(
                    _process_one, request_id, root=args.queue_root,
                    host=args.remote_host, codex=args.codex, timeout=args.timeout,
                )
                active[future] = request_id
                available -= 1
            if args.once and not active:
                return 2
            if (not active and processed and args.idle_exit_seconds and
                    time.monotonic() - idle_since >= args.idle_exit_seconds):
                return 0
            time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())

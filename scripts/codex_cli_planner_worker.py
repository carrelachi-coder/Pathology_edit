"""Answer certified Planner packets with a fresh GPT-5.6 Terra CLI session.

Run on the local machine with Codex account authentication. The GPU workflow
may write requests on a remote server; --remote-host copies only the packet's
mask images and schema into an isolated temporary directory. Every call uses
`codex exec --ephemeral` and never resumes the current conversation.
"""

from __future__ import annotations

import argparse
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


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
    if not isinstance(raw, dict):
        raise ValueError("Codex CLI output is not a JSON object")
    _check_stripped_constraints(raw, request["json_schema"])
    match = re.search(r"session id: ([0-9a-f-]+)", completed.stderr)
    return {
        "request_id": request["request_id"],
        "prompt_sha256": request["prompt_sha256"],
        "output": raw,
        "session_id": match.group(1) if match else None,
    }


def _pending_ids(*, root: str, host: str | None) -> list[str]:
    if host:
        command = ["ssh", "-o", "BatchMode=yes", host,
                   "ls -1 " + shlex.quote(root.rstrip("/") + "/requests")]
        completed = subprocess.run(command, capture_output=True, text=True,
                                   check=True)
        names = completed.stdout.splitlines()
        completed = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", host,
             "ls -1 " + shlex.quote(root.rstrip("/") + "/responses")],
            capture_output=True, text=True, check=True,
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
            subprocess.run(["scp", "-q", "-r", f"{host}:{remote}", str(local)],
                           check=True)
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
            subprocess.run(["scp", "-q", str(local_response),
                            f"{host}:{responses}{temp_name}"], check=True)
            subprocess.run(
                ["ssh", "-o", "BatchMode=yes", host,
                 "mv " + shlex.quote(responses + temp_name) + " "
                 + shlex.quote(responses + request_id + ".json")],
                check=True,
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
    args = parser.parse_args()
    processed = 0
    idle_since = time.monotonic()
    while True:
        pending = _pending_ids(root=args.queue_root, host=args.remote_host)
        if pending:
            response = _process_one(
                pending[0], root=args.queue_root, host=args.remote_host,
                codex=args.codex, timeout=args.timeout,
            )
            print(json.dumps({k: response.get(k) for k in
                              ("request_id", "session_id", "error")} ), flush=True)
            processed += 1
            idle_since = time.monotonic()
            if args.once:
                return 1 if response.get("error") else 0
        elif args.once:
            return 2
        elif processed and args.idle_exit_seconds and (
            time.monotonic() - idle_since >= args.idle_exit_seconds
        ):
            return 0
        else:
            time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Read-only finite S07 identity preflight; never invokes a witness."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(manifest: Path) -> dict:
    plan = json.loads(manifest.read_text())
    if plan.get("preparation_only") is not True or plan.get("execution_permitted") is not False:
        raise PermissionError("preflight requires a closed candidate manifest")
    if plan.get("S07_authorized") is not False:
        raise PermissionError("preflight cannot authorize S07")
    controller_path = Path(plan["controller"]["path"])
    if sha(controller_path) != plan["controller"]["sha256"]:
        raise ValueError("controller SHA drift")
    spec = importlib.util.spec_from_file_location("_s07_v2_preflight_only", controller_path)
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    controller.identity_gate(plan)
    controller.scope_gate(plan)
    # Check no-replace hardlinks at the actual output parent, without touching
    # the future output directory or any business input.
    parent = Path(plan["output_root"]).parent
    with tempfile.TemporaryDirectory(prefix="s07_v2_link_probe_", dir=parent) as temporary:
        root = Path(temporary)
        pending = root / "pending"
        final = root / "final"
        pending.write_bytes(b"probe\n")
        os.link(pending, final)
        try:
            os.link(pending, final)
        except FileExistsError:
            pass
        else:
            raise RuntimeError("filesystem overwrote existing final link")
        if pending.stat().st_ino != final.stat().st_ino or final.read_bytes() != b"probe\n":
            raise RuntimeError("hardlink content or inode mismatch")
    return {
        "status": "preflight_pass",
        "manifest_sha256": sha(manifest),
        "controller_sha256": sha(controller_path),
        "snapshot_files_checked": sum(len(s["files_sha256"]) for s in plan["snapshots"]),
        "pinned_files_checked": len(plan["pinned_files"]),
        "witness_calls": 0,
        "business_imports": 0,
        "execution_permitted": False,
        "future_live_checks_required": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest), sort_keys=True))


if __name__ == "__main__":
    main()

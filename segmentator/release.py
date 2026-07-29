from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REQUIRED_ARCHITECTURE_KEYS = {
    "num_classes",
    "mask2former_queries",
    "mask2former_ignore_index",
    "hierarchical_fine",
    "boundary_refinement",
    "refinement_gate_mode",
    "symmetric_padding",
    "cellvit_mode",
}


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_segmentator_release(
    path: str | Path,
    *,
    verify_checkpoint: bool = True,
) -> dict[str, Any]:
    release_path = Path(path).expanduser().resolve()
    if not release_path.is_file():
        raise FileNotFoundError(release_path)
    if release_path.suffix.lower() == ".json":
        payload = json.loads(release_path.read_text(encoding="utf-8"))
    else:
        import yaml

        payload = yaml.safe_load(release_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Segmentator release must be an object: {release_path}")
    if "segmentator" in payload:
        payload = payload["segmentator"]
    validate_segmentator_release(payload, verify_checkpoint=verify_checkpoint)
    payload = dict(payload)
    payload["_release_path"] = str(release_path)
    payload["_release_sha256"] = sha256_file(release_path)
    return payload


def validate_segmentator_release(
    release: Mapping[str, Any],
    *,
    verify_checkpoint: bool,
) -> None:
    missing = [
        key
        for key in (
            "schema_version",
            "release_id",
            "release_status",
            "checkpoint",
            "checkpoint_sha256",
            "decoder",
            "architecture",
            "runtime",
            "input",
            "output",
        )
        if key not in release
    ]
    if missing:
        raise ValueError(f"Segmentator release is missing fields: {missing}")
    architecture = release["architecture"]
    if not isinstance(architecture, Mapping):
        raise ValueError("Segmentator release architecture must be an object")
    missing_architecture = sorted(REQUIRED_ARCHITECTURE_KEYS - set(architecture))
    if missing_architecture:
        raise ValueError(
            "Segmentator release architecture is missing fields: "
            f"{missing_architecture}"
        )
    if release["decoder"] != "mask2former":
        raise ValueError("G2 Segmentator release must use mask2former")
    if release["runtime"].get("strict_checkpoint_load") is not True:
        raise ValueError("G2 Segmentator release must require strict checkpoint loading")

    checkpoint = Path(str(release["checkpoint"])).expanduser()
    if not verify_checkpoint:
        return
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    expected_size = release.get("checkpoint_size_bytes")
    if expected_size is not None and checkpoint.stat().st_size != int(expected_size):
        raise ValueError(
            f"Segmentator checkpoint size mismatch: expected {expected_size}, "
            f"got {checkpoint.stat().st_size}"
        )
    actual_sha256 = sha256_file(checkpoint)
    if actual_sha256 != str(release["checkpoint_sha256"]):
        raise ValueError(
            "Segmentator checkpoint SHA256 mismatch: "
            f"expected {release['checkpoint_sha256']}, got {actual_sha256}"
        )


def release_model_kwargs(release: Mapping[str, Any]) -> dict[str, Any]:
    architecture = release["architecture"]
    runtime = release["runtime"]
    return {
        "num_classes": int(architecture["num_classes"]),
        "freeze_encoder": True,
        "local_repo": str(runtime["local_repo"]),
        "decoder": str(release["decoder"]),
        "mask2former_queries": int(architecture["mask2former_queries"]),
        "mask2former_ignore_index": int(
            architecture["mask2former_ignore_index"]
        ),
        "symmetric_padding": bool(architecture["symmetric_padding"]),
        "boundary_refinement": bool(architecture["boundary_refinement"]),
        "refinement_gate_mode": str(architecture["refinement_gate_mode"]),
        "cellvit_mode": str(architecture["cellvit_mode"]),
        "hierarchical_fine": bool(architecture["hierarchical_fine"]),
        "fine_supported_ids": tuple(
            int(value) for value in architecture.get("fine_supported_ids", ())
        )
        or None,
    }

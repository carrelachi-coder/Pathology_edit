"""Canonical production model locations with environment overrides."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import os
from pathlib import Path
from typing import Any


def _model_path(env_name: str, fallback: str) -> str:
    value = os.environ.get(env_name, "").strip()
    return value or fallback


DEFAULT_INPAINT_CHECKPOINT = _model_path(
    "PATHOLOGY_INPAINT_CHECKPOINT",
    "/home/lyw/wqx-DL/flow-edit/hf_generation_release/"
    "pathology-inpaint-controlnet",
)
DEFAULT_CROSS_V1_CHECKPOINT = _model_path(
    "PATHOLOGY_CROSS_V1_CHECKPOINT",
    "/home/lyw/wqx-DL/flow-edit/hf_generation_release/"
    "pathology-cross-v1-pix2pix/cross_v1",
)
DEFAULT_PIX2PIX_CHECKPOINT = _model_path(
    "PATHOLOGY_PIX2PIX_CHECKPOINT",
    "/home/lyw/wqx-DL/flow-edit/hf_generation_release/"
    "pathology-cross-v1-pix2pix/pix2pix/"
    "pix2pix_epoch26_step214895.pt",
)
DEFAULT_PROBNET_CHECKPOINT = _model_path(
    "PATHOLOGY_PROBNET_CHECKPOINT",
    "/data1/zhao/wqx/probnet_density/frozen/epoch29_C3_shape_group_total_count/"
    "best_epoch29_c29607f1b609accb.pt",
)
DEFAULT_CELLVIT_ROOT = _model_path(
    "PATHOLOGY_CELLVIT_ROOT",
    "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/CellViT-plus-plus-main/"
    "CellViT-plus-plus-main",
)
DEFAULT_CELLVIT_MODEL = _model_path(
    "PATHOLOGY_CELLVIT_MODEL",
    f"{DEFAULT_CELLVIT_ROOT}/checkpoints/CellViT-SAM-H-x40-AMP-001.pth",
)
DEFAULT_CELLVIT_PYTHON = _model_path(
    "PATHOLOGY_CELLVIT_PYTHON",
    "/home/lyw/anaconda3/envs/pathology-phase5-inpaint/bin/python",
)

PRODUCTION_PIX2PIX_SHA256 = (
    "be5fe9376efdb5620a57481082f6d5738b6353796fb00fe6e58f6b212ba7c2ac"
)
PRODUCTION_PIX2PIX_EPOCH = 26
PRODUCTION_PIX2PIX_GLOBAL_STEP = 214895
PRODUCTION_PIX2PIX_ENV = "PATHOLOGY_PIX2PIX_CHECKPOINT"
FROZEN_PROBNET_SHA256 = (
    "c29607f1b609accbb6ee0fceccb9ead02cd266cce67cec1d8df7c0b7da571211"
)
FROZEN_CELLVIT_SHA256 = (
    "356418f19d9d478f164c7a31f85274584fefaa02355815c09f52346c658c8ec4"
)
PRODUCTION_GENERATION_RELEASE_COMMIT = (
    "6129422cc677d0183f3234ae17b049c76fc57024"
)
PRODUCTION_CONTROLNET_RELEASES = {
    "inpaint": {
        "repo_suffix": "/pathology-inpaint-controlnet",
        "weight_path": "diffusion_pytorch_model.safetensors",
        "weight_size_bytes": 8_190_001_728,
        "weight_sha256": (
            "402c836c553410355cf2912518f69339d8eb61f1c9cc588d3020367121a6060c"
        ),
    },
    "cross-v1": {
        "repo_suffix": "/pathology-cross-v1-pix2pix",
        "weight_path": "cross_v1/diffusion_pytorch_model.safetensors",
        "weight_size_bytes": 8_192_950_848,
        "weight_sha256": (
            "b0442d93aa2b2649e3506620c36c4cc54ba55d377f4c7f767f19147ea83d276e"
        ),
    },
}


@lru_cache(maxsize=8)
def _sha256_for_file_state(
    resolved_path: str,
    size_bytes: int,
    mtime_ns: int,
) -> str:
    del size_bytes, mtime_ns
    digest = hashlib.sha256()
    with Path(resolved_path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cached_file_sha256(path: Path) -> str:
    stat = path.stat()
    return _sha256_for_file_state(
        str(path),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


@lru_cache(maxsize=4)
def _load_pix2pix_release_metadata(
    resolved_path: str,
    size_bytes: int,
    mtime_ns: int,
    expected_sha256: str,
) -> dict[str, Any]:
    del size_bytes, mtime_ns
    import torch

    path = Path(resolved_path)
    sha256 = _cached_file_sha256(path)
    if sha256 != expected_sha256:
        raise ValueError(
            "Pix2pix checkpoint is not the frozen production release: "
            f"{sha256} != {expected_sha256} ({path})"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "sha256": sha256,
        "epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", -1)),
        "args": dict(checkpoint.get("args") or {}),
    }


def validate_production_pix2pix_checkpoint(
    checkpoint_path: str | Path,
    *,
    require_environment_selector: bool = True,
) -> dict[str, Any]:
    """Reject historical or incompletely packaged pix2pix checkpoints."""

    path = Path(checkpoint_path).expanduser().resolve()
    if require_environment_selector:
        selected = os.environ.get(PRODUCTION_PIX2PIX_ENV, "").strip()
        if not selected:
            raise RuntimeError(
                f"{PRODUCTION_PIX2PIX_ENV} must explicitly select the packaged "
                f"epoch {PRODUCTION_PIX2PIX_EPOCH} / step "
                f"{PRODUCTION_PIX2PIX_GLOBAL_STEP} checkpoint."
            )
        selected_path = Path(selected).expanduser().resolve()
        if selected_path != path:
            raise RuntimeError(
                f"{PRODUCTION_PIX2PIX_ENV} resolves to {selected_path}, "
                f"but the runtime requested {path}."
            )
    if not path.is_file():
        raise FileNotFoundError(f"Pix2pix checkpoint not found: {path}")
    stat = path.stat()
    metadata = _load_pix2pix_release_metadata(
        str(path),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        PRODUCTION_PIX2PIX_SHA256,
    )
    sha256 = str(metadata["sha256"])
    epoch = int(metadata["epoch"])
    global_step = int(metadata["global_step"])
    args = dict(metadata["args"])
    required = {
        "cross4_texture_steering": True,
        "cross4_steering_reference_mode": "local_histogram",
        "cross4_steering_scales": "1/1,1/2,1/4,1/8,1/16",
        "full_pyramid_texture_steering": True,
        "highres_nuclei_trust_enabled": True,
        "wsi_identity_adapter": True,
    }
    mismatches = {
        key: {"expected": expected, "actual": args.get(key)}
        for key, expected in required.items()
        if args.get(key) != expected
    }
    if (
        epoch != PRODUCTION_PIX2PIX_EPOCH
        or global_step != PRODUCTION_PIX2PIX_GLOBAL_STEP
        or mismatches
    ):
        raise ValueError(
            "Pix2pix release metadata mismatch: "
            f"epoch={epoch}, global_step={global_step}, fields={mismatches}"
        )
    return {
        "checkpoint": str(path),
        "sha256": sha256,
        "epoch": epoch,
        "global_step": global_step,
        "orientation_policy": "full_pyramid_local_histogram",
        "texture_steering_scales": required["cross4_steering_scales"].split(
            ","
        ),
        "nuclei_reference_policy": "nuclei_reference_support_v2",
        "highres_nuclei_unmatched_scale": float(
            args.get("highres_nuclei_unmatched_scale", 0.20)
        ),
        "highres_nuclei_matched_floor": float(
            args.get("highres_nuclei_matched_floor", 0.60)
        ),
    }


def validate_frozen_probnet_checkpoint(
    checkpoint_path: str | Path,
) -> dict[str, str]:
    """Reject non-frozen CellDistNet/ProbNet checkpoints in the online product."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ProbNet checkpoint not found: {path}")
    sha256 = _cached_file_sha256(path)
    if sha256 != FROZEN_PROBNET_SHA256:
        raise ValueError(
            "ProbNet checkpoint is not the frozen epoch-29 release: "
            f"{sha256} != {FROZEN_PROBNET_SHA256} ({path})"
        )
    return {
        "checkpoint": str(path),
        "sha256": sha256,
        "policy": "frozen_epoch29_patch_adaptive_spatial_sampling_v1",
    }


def validate_frozen_cellvit_checkpoint(
    checkpoint_path: str | Path,
) -> dict[str, str]:
    """Reject a CellViT evaluator that is not the frozen production release."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CellViT checkpoint not found: {path}")
    sha256 = _cached_file_sha256(path)
    if sha256 != FROZEN_CELLVIT_SHA256:
        raise ValueError(
            "CellViT checkpoint is not the frozen production evaluator: "
            f"{sha256} != {FROZEN_CELLVIT_SHA256} ({path})"
        )
    return {
        "checkpoint": str(path),
        "sha256": sha256,
        "policy": "cellvit-sam-h-x40-amp-001-512px-0.25mpp-v1",
    }


def validate_production_controlnet_checkpoint(
    checkpoint_path: str | Path,
    *,
    mode: str,
) -> dict[str, Any]:
    """Validate an inference-only Inpaint or Cross V1 package manifest."""

    if mode not in PRODUCTION_CONTROLNET_RELEASES:
        raise ValueError(f"unsupported production ControlNet mode: {mode}")
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"{mode} production checkpoint directory not found: {path}"
        )
    release = PRODUCTION_CONTROLNET_RELEASES[mode]
    manifest_candidates = (path / "manifest.json", path.parent / "manifest.json")
    manifest_path = next(
        (candidate for candidate in manifest_candidates if candidate.is_file()),
        None,
    )
    if manifest_path is None:
        raise FileNotFoundError(
            f"{mode} production package manifest not found beside {path}"
        )
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo_id = str(manifest.get("repo_id") or "")
    git_commit = str(manifest.get("git_commit") or "")
    if not repo_id.endswith(str(release["repo_suffix"])):
        raise ValueError(
            f"{mode} package repo mismatch: {repo_id!r} does not end with "
            f"{release['repo_suffix']!r}"
        )
    if git_commit != PRODUCTION_GENERATION_RELEASE_COMMIT:
        raise ValueError(
            f"{mode} package code commit mismatch: {git_commit!r} != "
            f"{PRODUCTION_GENERATION_RELEASE_COMMIT!r}"
        )
    file_records = {
        str(record.get("path")): record
        for record in manifest.get("files", [])
        if isinstance(record, dict)
    }
    weight_path = str(release["weight_path"])
    record = file_records.get(weight_path)
    if record is None:
        raise ValueError(
            f"{mode} package manifest does not contain {weight_path}"
        )
    expected_size = int(release["weight_size_bytes"])
    expected_sha256 = str(release["weight_sha256"])
    if (
        int(record.get("size_bytes", -1)) != expected_size
        or str(record.get("sha256") or "") != expected_sha256
    ):
        raise ValueError(
            f"{mode} package weight manifest mismatch for {weight_path}"
        )
    package_root = manifest_path.parent
    weight_file = package_root / weight_path
    if not weight_file.is_file():
        raise FileNotFoundError(
            f"{mode} packaged ControlNet weight not found: {weight_file}"
        )
    if weight_file.stat().st_size != expected_size:
        raise ValueError(
            f"{mode} packaged ControlNet size mismatch: "
            f"{weight_file.stat().st_size} != {expected_size}"
        )
    return {
        "mode": mode,
        "checkpoint": str(path),
        "manifest": str(manifest_path),
        "repo_id": repo_id,
        "code_commit": git_commit,
        "weight": str(weight_file),
        "weight_size_bytes": expected_size,
        "weight_sha256": expected_sha256,
    }


__all__ = [
    "DEFAULT_CELLVIT_MODEL",
    "DEFAULT_CELLVIT_PYTHON",
    "DEFAULT_CELLVIT_ROOT",
    "DEFAULT_CROSS_V1_CHECKPOINT",
    "DEFAULT_INPAINT_CHECKPOINT",
    "DEFAULT_PIX2PIX_CHECKPOINT",
    "DEFAULT_PROBNET_CHECKPOINT",
    "FROZEN_CELLVIT_SHA256",
    "FROZEN_PROBNET_SHA256",
    "PRODUCTION_PIX2PIX_ENV",
    "PRODUCTION_PIX2PIX_EPOCH",
    "PRODUCTION_PIX2PIX_GLOBAL_STEP",
    "PRODUCTION_PIX2PIX_SHA256",
    "PRODUCTION_CONTROLNET_RELEASES",
    "PRODUCTION_GENERATION_RELEASE_COMMIT",
    "validate_frozen_cellvit_checkpoint",
    "validate_frozen_probnet_checkpoint",
    "validate_production_controlnet_checkpoint",
    "validate_production_pix2pix_checkpoint",
]

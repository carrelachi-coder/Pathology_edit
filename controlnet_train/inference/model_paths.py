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
    "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/phase5_runs/controlnet_inpaint_all",
)
DEFAULT_CROSS_V1_CHECKPOINT = _model_path(
    "PATHOLOGY_CROSS_V1_CHECKPOINT",
    "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/phase5_runs/controlnet_cross_v1",
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

PRODUCTION_PIX2PIX_SHA256 = (
    "be5fe9376efdb5620a57481082f6d5738b6353796fb00fe6e58f6b212ba7c2ac"
)
PRODUCTION_PIX2PIX_EPOCH = 26
PRODUCTION_PIX2PIX_GLOBAL_STEP = 214895
PRODUCTION_PIX2PIX_ENV = "PATHOLOGY_PIX2PIX_CHECKPOINT"
FROZEN_PROBNET_SHA256 = (
    "c29607f1b609accbb6ee0fceccb9ead02cd266cce67cec1d8df7c0b7da571211"
)


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


__all__ = [
    "DEFAULT_CROSS_V1_CHECKPOINT",
    "DEFAULT_INPAINT_CHECKPOINT",
    "DEFAULT_PIX2PIX_CHECKPOINT",
    "DEFAULT_PROBNET_CHECKPOINT",
    "FROZEN_PROBNET_SHA256",
    "PRODUCTION_PIX2PIX_ENV",
    "PRODUCTION_PIX2PIX_EPOCH",
    "PRODUCTION_PIX2PIX_GLOBAL_STEP",
    "PRODUCTION_PIX2PIX_SHA256",
    "validate_frozen_probnet_checkpoint",
    "validate_production_pix2pix_checkpoint",
]

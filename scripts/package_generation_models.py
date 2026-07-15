#!/usr/bin/env python3
"""Build inference-only Hugging Face release folders for production models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controlnet_train.inference.model_paths import (
    DEFAULT_CROSS_V1_CHECKPOINT,
    DEFAULT_INPAINT_CHECKPOINT,
    DEFAULT_PIX2PIX_CHECKPOINT,
    DEFAULT_PROBNET_CHECKPOINT,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PIX2PIX_INFERENCE_ARG_NAMES = {
    "base_channels",
    "num_heads",
    "cross_attn_scales",
    "upsample_mode",
    "region_label_mode",
    "no_region_mask",
    "no_residual_output",
    "wsi_identity_adapter",
    "identity_gamma_max",
    "identity_gamma_init",
    "identity_min_tissue_pixels",
    "identity_min_nuclei_pixels",
    "cross4_texture_steering",
    "cross4_steering_angles",
    "cross4_steering_smoothing_sigma",
    "cross4_steering_min_coherence",
    "cross4_steering_min_relative_energy",
    "cross4_steering_min_resultant",
    "cross4_steering_minimum_strength",
    "cross4_steering_minimum_support",
    "cross4_steering_temperature",
    "cross4_steering_reference_mode",
    "cross4_steering_local_bins",
    "cross4_steering_local_kappa",
    "cross4_steering_scales",
    "cross4_steering_gain",
    "cross8_steering_gain",
    "cross16_steering_gain",
    "cross2_steering_gain",
    "cross1_steering_gain",
    "full_pyramid_texture_steering",
    "steering_highres_reference_size",
    "highres_nuclei_trust_enabled",
    "highres_nuclei_unmatched_scale",
    "highres_nuclei_matched_floor",
    "highres_nuclei_sufficient_tokens",
    "highres_nuclei_min_reference_pixels",
}


def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_or_copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _prepare_directory(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Release directory already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _file_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "manifest.json":
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records


def _write_manifest(
    root: Path,
    *,
    repo_id: str,
    git_commit: str,
    sources: list[dict[str, Any]],
    model_metadata: dict[str, Any],
    dependencies: dict[str, Any],
    loading: dict[str, Any],
) -> None:
    payload = {
        "schema_version": 1,
        "repo_id": repo_id,
        "private": True,
        "git_commit": git_commit,
        "source_artifacts": sources,
        "model_metadata": model_metadata,
        "dependencies": dependencies,
        "loading": loading,
        "validation": {"packaging": "passed", "hub_roundtrip": "pending"},
        "files": _file_records(root),
    }
    (root / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def package_inpaint(args: argparse.Namespace) -> Path:
    source = args.inpaint_checkpoint.resolve()
    output = args.output_root / "pathology-inpaint-controlnet"
    _prepare_directory(output, overwrite=args.overwrite)
    for name in ("config.json", "diffusion_pytorch_model.safetensors", "phase5_conditioning.pt"):
        _link_or_copy(source / name, output / name)
    (output / "README.md").write_text(
        f"""---
library_name: diffusers
pipeline_tag: image-to-image
---

# Pathology Inpaint ControlNet

Private inference-only release for local preservation-sensitive pathology edits.
It requires an authorized local copy of `black-forest-labs/FLUX.1-dev` and the
production code at the Git commit recorded in `manifest.json`.

The package intentionally excludes optimizer, scheduler, random state, logs and
training data. Pass the downloaded directory as `--inpaint-checkpoint` or set
`PATHOLOGY_INPAINT_CHECKPOINT`.

```bash
hf download {args.hf_namespace}/pathology-inpaint-controlnet --local-dir /models/pathology-inpaint-controlnet
export PATHOLOGY_INPAINT_CHECKPOINT=/models/pathology-inpaint-controlnet
python -m controlnet_train.cli.eval_controlnet_flux_inpaint --help
```
""",
        encoding="utf-8",
    )
    _write_manifest(
        output,
        repo_id=f"{args.hf_namespace}/pathology-inpaint-controlnet",
        git_commit=args.git_commit,
        sources=[{"path": str(source), "excluded": ["checkpoint-40000", "logs"]}],
        model_metadata={"type": "flux-controlnet-inpaint", "base_model": "black-forest-labs/FLUX.1-dev"},
        dependencies={
            "code_repository": "https://github.com/carrelachi-coder/Pathology_edit",
            "base_model": "black-forest-labs/FLUX.1-dev",
            "training_data_included": False,
        },
        loading={
            "environment_variables": {
                "PATHOLOGY_INPAINT_CHECKPOINT": "/models/pathology-inpaint-controlnet"
            },
            "command": "python -m controlnet_train.cli.eval_controlnet_flux_inpaint --help",
        },
    )
    return output


def package_cross_pix2pix(args: argparse.Namespace) -> Path:
    cross_source = args.cross_v1_checkpoint.resolve()
    pix2pix_source = args.pix2pix_checkpoint.resolve()
    output = args.output_root / "pathology-cross-v1-pix2pix"
    _prepare_directory(output, overwrite=args.overwrite)

    cross_output = output / "cross_v1"
    _link_or_copy(cross_source / "config.json", cross_output / "config.json")
    _link_or_copy(
        cross_source / "diffusion_pytorch_model.safetensors",
        cross_output / "diffusion_pytorch_model.safetensors",
    )
    conditioning = torch.load(
        cross_source / "phase5_conditioning.pt", map_location="cpu", weights_only=False
    )
    torch.save(
        {
            "hte": conditioning["hte"],
            "tissue_downsampler": conditioning["tissue_downsampler"],
            "nuclei_encoder": conditioning["nuclei_encoder"],
        },
        cross_output / "phase5_conditioning.pt",
    )

    pix2pix = torch.load(pix2pix_source, map_location="cpu", weights_only=False)
    source_args = dict(pix2pix.get("args") or {})
    inference_args = {
        key: source_args[key]
        for key in sorted(PIX2PIX_INFERENCE_ARG_NAMES)
        if key in source_args
    }
    inference_args.update(
        {
            "highres_nuclei_trust_enabled": True,
            "highres_nuclei_unmatched_scale": 0.20,
            "highres_nuclei_matched_floor": 0.60,
            "highres_nuclei_sufficient_tokens": 4,
            "highres_nuclei_min_reference_pixels": 64,
        }
    )
    pix2pix_output = output / "pix2pix" / "pix2pix_epoch26_step214895.pt"
    pix2pix_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model": pix2pix["model"],
            "args": inference_args,
            "epoch": int(pix2pix["epoch"]),
            "global_step": int(pix2pix["global_step"]),
            "source_checkpoint_sha256": _sha256(pix2pix_source),
            "trust_gate": "nuclei_reference_support_v2",
        },
        pix2pix_output,
    )
    if int(pix2pix["epoch"]) != 26 or int(pix2pix["global_step"]) != 214895:
        raise ValueError("Unexpected production pix2pix epoch/global_step")

    (output / "README.md").write_text(
        f"""---
library_name: diffusers
pipeline_tag: image-to-image
---

# Pathology Cross V1 + Pix2pix

Private production release containing two sequential stages:

1. `cross_v1/`: strict no-IP/no-UNI FLUX ControlNet generation.
2. `pix2pix/`: epoch 26 / step 214895 full-pyramid texture transfer with
   local orientation steering and `nuclei_reference_support_v2`.

The Cross V1 package intentionally excludes `phase5_ip_adapter.pt`, UNI,
optimizer state and training checkpoints. It requires an authorized local copy
of `black-forest-labs/FLUX.1-dev`.

```bash
hf download {args.hf_namespace}/pathology-cross-v1-pix2pix --local-dir /models/pathology-cross-v1-pix2pix
export PATHOLOGY_CROSS_V1_CHECKPOINT=/models/pathology-cross-v1-pix2pix/cross_v1
export PATHOLOGY_PIX2PIX_CHECKPOINT=/models/pathology-cross-v1-pix2pix/pix2pix/pix2pix_epoch26_step214895.pt
python scripts/generate_cross_v1_no_ip_strict.py --help
```
""",
        encoding="utf-8",
    )
    _write_manifest(
        output,
        repo_id=f"{args.hf_namespace}/pathology-cross-v1-pix2pix",
        git_commit=args.git_commit,
        sources=[
            {
                "path": str(cross_source),
                "excluded": ["phase5_ip_adapter.pt", "checkpoint-66000", "train.log"],
            },
            {
                "path": str(pix2pix_source),
                "sha256": _sha256(pix2pix_source),
                "excluded_keys": ["optimizer", "discriminator", "d_optimizer"],
            },
        ],
        model_metadata={
            "type": "cross-v1-no-ip-plus-pix2pix",
            "base_model": "black-forest-labs/FLUX.1-dev",
            "pix2pix_epoch": 26,
            "pix2pix_global_step": 214895,
            "trust_gate": "nuclei_reference_support_v2",
        },
        dependencies={
            "code_repository": "https://github.com/carrelachi-coder/Pathology_edit",
            "base_model": "black-forest-labs/FLUX.1-dev",
            "uni_required_for_inference": False,
            "ip_adapter_required_for_inference": False,
        },
        loading={
            "environment_variables": {
                "PATHOLOGY_CROSS_V1_CHECKPOINT": "/models/pathology-cross-v1-pix2pix/cross_v1",
                "PATHOLOGY_PIX2PIX_CHECKPOINT": "/models/pathology-cross-v1-pix2pix/pix2pix/pix2pix_epoch26_step214895.pt",
            },
            "command": "python scripts/generate_cross_v1_no_ip_strict.py --help",
        },
    )
    return output


def package_probnet(args: argparse.Namespace) -> Path:
    source = args.probnet_checkpoint.resolve()
    output = args.output_root / "pathology-probnet"
    _prepare_directory(output, overwrite=args.overwrite)
    _link_or_copy(source, output / "best.pt")
    config_output = output / "configs"
    config_output.mkdir(parents=True)
    for config in sorted((REPO_ROOT / "inpaint_cells" / "configs").glob("*.json")):
        shutil.copy2(config, config_output / config.name)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    (output / "README.md").write_text(
        f"""---
library_name: pytorch
pipeline_tag: image-to-image
---

# Pathology ProbNet

Private inference-only ProbUNet release used by the Phase 4 nuclei generation
workflow. The checkpoint uses `base_ch=64` and predicts six classes (background
plus five project nuclei classes).

ProbNet placement also requires a dataset-specific nuclei instance library.
Those approximately 1GB of real instance crops are intentionally not included
in this model repository and should be managed as a separate private dataset.

```bash
hf download {args.hf_namespace}/pathology-probnet --local-dir /models/pathology-probnet
export PATHOLOGY_PROBNET_CHECKPOINT=/models/pathology-probnet/best.pt
python scripts/phase4_single_sample_smoke.py --help
```
""",
        encoding="utf-8",
    )
    _write_manifest(
        output,
        repo_id=f"{args.hf_namespace}/pathology-probnet",
        git_commit=args.git_commit,
        sources=[{"path": str(source), "sha256": _sha256(source)}],
        model_metadata={
            "type": "prob-unet",
            "base_channels": 64,
            "epoch": int(checkpoint["epoch"]),
            "global_step": int(checkpoint["global_step"]),
            "val_loss": float(checkpoint["val_loss"]),
            "val_metrics": checkpoint.get("val_metrics", {}),
            "nuclei_library_included": False,
        },
        dependencies={
            "code_repository": "https://github.com/carrelachi-coder/Pathology_edit",
            "external_nuclei_instance_library": True,
            "training_data_included": False,
        },
        loading={
            "environment_variables": {
                "PATHOLOGY_PROBNET_CHECKPOINT": "/models/pathology-probnet/best.pt"
            },
            "command": "python scripts/phase4_single_sample_smoke.py --help",
        },
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hf-namespace", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--inpaint-checkpoint", type=Path, default=Path(DEFAULT_INPAINT_CHECKPOINT))
    parser.add_argument("--cross-v1-checkpoint", type=Path, default=Path(DEFAULT_CROSS_V1_CHECKPOINT))
    parser.add_argument("--pix2pix-checkpoint", type=Path, default=Path(DEFAULT_PIX2PIX_CHECKPOINT))
    parser.add_argument("--probnet-checkpoint", type=Path, default=Path(DEFAULT_PROBNET_CHECKPOINT))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    outputs = [
        package_inpaint(args),
        package_cross_pix2pix(args),
        package_probnet(args),
    ]
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

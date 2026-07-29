#!/usr/bin/env python3
"""Run strict Segmentator inference for one image."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument(
        "--release",
        type=Path,
        help="Versioned Segmentator release JSON/YAML; preferred for G2.",
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        help="Legacy coarse-mask output path. May be combined with --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write the complete coarse/fine prediction bundle here.",
    )
    parser.add_argument("--profile", default="default")
    parser.add_argument(
        "--fine-applicability",
        default=None,
        help="Manifest policy, e.g. supported, not_applicable, unsupported_dcis.",
    )
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--decoder", choices=("upernet", "mask2former"), default="mask2former")
    parser.add_argument("--mask2former-queries", type=int, default=100)
    parser.add_argument("--mask2former-ignore-index", type=int, default=255)
    parser.add_argument("--symmetric-padding", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--boundary-refinement", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--refinement-gate-mode", choices=("hard", "learned_soft"), default="hard")
    parser.add_argument("--cellvit-mode", choices=("none", "teacher", "input"), default="none")
    parser.add_argument("--hierarchical-fine", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--save-entropy", action="store_true")
    parser.add_argument("--save-fine-when-applicable", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output is None and args.output_dir is None:
        raise ValueError("one of --output or --output-dir is required")
    import torch
    import numpy as np
    from PIL import Image
    import torchvision.transforms.functional as TF

    from segmentator.data import fine_supervision_for_dataset, normalize_image_tensor
    from segmentator.inference import (
        load_checkpoint,
        normalized_entropy,
        normalized_hierarchical_entropy,
        save_prediction,
        save_probability_tensor,
    )
    from segmentator.release import (
        load_segmentator_release,
        release_model_kwargs,
        sha256_file,
    )

    started = time.perf_counter()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    release = None
    if args.release is not None:
        release = load_segmentator_release(args.release, verify_checkpoint=True)
        checkpoint = str(release["checkpoint"])
        model_kwargs = release_model_kwargs(release)
    else:
        checkpoint = str(args.checkpoint)
        model_kwargs = {
            "num_classes": args.num_classes,
            "decoder": args.decoder,
            "mask2former_queries": args.mask2former_queries,
            "mask2former_ignore_index": args.mask2former_ignore_index,
            "symmetric_padding": args.symmetric_padding,
            "boundary_refinement": args.boundary_refinement,
            "refinement_gate_mode": args.refinement_gate_mode,
            "cellvit_mode": args.cellvit_mode,
            "hierarchical_fine": args.hierarchical_fine,
        }
    model = load_checkpoint(checkpoint, **model_kwargs).to(device)
    source_image = Image.open(args.input).convert("RGB")
    if release is not None:
        expected_size = tuple(int(value) for value in release["input"]["image_size"])
        if source_image.size != expected_size:
            raise ValueError(
                f"release requires image size {expected_size}, got {source_image.size}"
            )
    image = normalize_image_tensor(TF.to_tensor(source_image)).to(device)
    fine_applicable = (
        args.fine_applicability not in {"not_applicable", "unsupported_dcis"}
        and args.profile.lower() in {"bcss", "glas", "panda"}
    )
    fine_allowed = (
        fine_supervision_for_dataset(args.profile).to(device)
        if fine_applicable and model_kwargs.get("hierarchical_fine")
        else None
    )
    with torch.inference_mode():
        outputs = model(image.unsqueeze(0), fine_allowed=fine_allowed)

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path(args.output).parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    coarse_path = Path(args.output) if args.output else output_dir / "coarse_mask.png"
    save_prediction(outputs["pred"][0], coarse_path)
    written = {"coarse_mask": str(coarse_path)}
    if args.save_probabilities:
        probability_path = output_dir / "coarse_probabilities.npz"
        save_probability_tensor(
            outputs["probs"],
            probability_path,
            class_ids=tuple(range(outputs["probs"].shape[1])),
        )
        written["coarse_probabilities"] = str(probability_path)
    if args.save_entropy:
        entropy_path = output_dir / "entropy.npy"
        entropy = normalized_entropy(outputs["probs"])[0]
        np.save(
            entropy_path,
            entropy.detach().cpu().numpy().astype(np.float16),
        )
        written["entropy"] = str(entropy_path)
    if (
        args.save_fine_when_applicable
        and fine_applicable
        and "hierarchical_pred" in outputs
    ):
        fine_path = output_dir / "fine_mask.png"
        save_prediction(outputs["hierarchical_pred"][0], fine_path)
        written["fine_mask"] = str(fine_path)
        if args.save_probabilities:
            fine_probability_path = output_dir / "fine_probabilities.npz"
            save_probability_tensor(
                outputs["fine_probs"],
                fine_probability_path,
                class_ids=tuple(range(outputs["fine_probs"].shape[1])),
            )
            written["fine_probabilities"] = str(fine_probability_path)
        if args.save_entropy:
            fine_entropy_path = output_dir / "fine_entropy.npy"
            effective_allowed = model._effective_fine_allowed(
                fine_allowed,
                outputs["fine_probs"].shape[0],
                outputs["fine_probs"].device,
            )
            fine_entropy = normalized_hierarchical_entropy(
                outputs["fine_probs"],
                outputs["pred"],
                effective_allowed,
            )[0]
            np.save(
                fine_entropy_path,
                fine_entropy.detach().cpu().numpy().astype(np.float16),
            )
            written["fine_entropy"] = str(fine_entropy_path)

    output_hashes = {
        name: sha256_file(path)
        for name, path in written.items()
    }
    try:
        code_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        code_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        code_commit = None
        code_dirty = None
    provenance = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_image": str(Path(args.input).resolve()),
        "input_image_sha256": sha256_file(args.input),
        "checkpoint": checkpoint,
        "checkpoint_sha256": (
            release["checkpoint_sha256"] if release is not None else sha256_file(checkpoint)
        ),
        "release_id": release.get("release_id") if release is not None else None,
        "release_sha256": release.get("_release_sha256") if release is not None else None,
        "architecture": (
            release.get("architecture") if release is not None else model_kwargs
        ),
        "profile": args.profile,
        "fine_applicability": (
            args.fine_applicability
            or ("supported" if fine_applicable else "not_applicable")
        ),
        "code_commit": code_commit,
        "code_worktree_dirty": code_dirty,
        "device": str(device),
        "runtime_seconds": time.perf_counter() - started,
        "output_files": written,
        "output_sha256": output_hashes,
    }
    provenance_path = output_dir / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    written["provenance"] = str(provenance_path)
    print(json.dumps(written, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

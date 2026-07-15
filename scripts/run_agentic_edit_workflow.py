#!/usr/bin/env python3
"""Run the bounded pathology edit agent: route, generate, verify, and recover."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controlnet_train.inference.agentic import (
    AgenticWorkflowConfig,
    FidelityThresholds,
    GenerationArtifact,
    run_agentic_workflow,
    verify_mask_fidelity,
)
from controlnet_train.inference.router import AgenticRoutingConfig
from controlnet_train.inference.model_paths import (
    DEFAULT_CROSS_V1_CHECKPOINT,
    DEFAULT_INPAINT_CHECKPOINT,
    DEFAULT_PIX2PIX_CHECKPOINT,
)
from phase3_mask_edit.core.mask_io import load_change_region, load_id_mask
from scripts.run_phase3_inpaint_pipeline import (
    _load_rgb_image,
    _load_uint8_mask,
    _run_generation_stage,
    _validate_same_size,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRETRAINED_MODEL = "/data/huggingface/FLUX.1-dev"
DEFAULT_SEGMENTATOR_CHECKPOINT = (
    "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/segmentator_runs/"
    "stage4_mask2former_multidataset_a800_v2/best_mIoU.pt"
)
DEFAULT_CELLVIT_ROOT = (
    "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/CellViT-plus-plus-main/"
    "CellViT-plus-plus-main"
)
DEFAULT_CELLVIT_MODEL = (
    f"{DEFAULT_CELLVIT_ROOT}/checkpoints/CellViT-SAM-H-x40-AMP-001.pth"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded agentic pathology edit workflow over already prepared target "
            "tissue/nuclei masks. The workflow routes to inpaint or production cross-v1 "
            "+ pix2pix-v2, re-segments the output, and tries at most one fallback by default."
        )
    )
    parser.add_argument("--profile", required=True, help="Dataset/profile name, e.g. BCSS.")
    parser.add_argument("--reference-image", required=True, type=Path)
    parser.add_argument("--reference-tissue-mask", required=True, type=Path)
    parser.add_argument("--reference-nuclei-mask", required=True, type=Path)
    parser.add_argument("--target-tissue-mask", required=True, type=Path)
    parser.add_argument("--target-nuclei-mask", required=True, type=Path)
    parser.add_argument(
        "--change-region",
        type=Path,
        help="Optional binary change mask; defaults to reference_tissue != target_tissue.",
    )
    parser.add_argument("--output", required=True, type=Path)

    parser.add_argument("--pretrained-model-name-or-path", default=DEFAULT_PRETRAINED_MODEL)
    parser.add_argument("--inpaint-checkpoint", type=Path, default=Path(DEFAULT_INPAINT_CHECKPOINT))
    parser.add_argument("--cross-v1-checkpoint", type=Path, default=Path(DEFAULT_CROSS_V1_CHECKPOINT))
    parser.add_argument("--pix2pix-checkpoint", type=Path, default=Path(DEFAULT_PIX2PIX_CHECKPOINT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-source", choices=("metadata", "dataset"), default="dataset")
    parser.add_argument("--torch-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--color-match", choices=("none", "lab"), default="lab")

    parser.add_argument("--t-inpaint", type=float, default=0.12)
    parser.add_argument("--t-cross", type=float, default=0.30)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--changed-region-accuracy-min", type=float, default=0.70)
    parser.add_argument("--changed-region-macro-iou-min", type=float, default=0.55)
    parser.add_argument("--off-target-drift-max", type=float, default=0.08)
    parser.add_argument("--nuclei-density-relative-error-max", type=float, default=0.35)

    parser.add_argument("--segmentator-checkpoint", type=Path, default=Path(DEFAULT_SEGMENTATOR_CHECKPOINT))
    parser.add_argument("--segmentator-env", default="pathology-segmentator-mmseg")
    parser.add_argument(
        "--segmentator-python",
        type=Path,
        help="Run segmentator with this Python instead of `conda run -n SEGMENTATOR_ENV`.",
    )
    parser.add_argument("--segmentator-decoder", choices=("upernet", "mask2former"), default="mask2former")
    parser.add_argument("--segmentator-device", default="cuda:1")
    parser.add_argument("--cellvit-model", type=Path, default=Path(DEFAULT_CELLVIT_MODEL))
    parser.add_argument("--cellvit-root", type=Path, default=Path(DEFAULT_CELLVIT_ROOT))
    parser.add_argument(
        "--cellvit-launch-python",
        type=Path,
        default=Path(sys.executable),
        help="Python used to launch scripts/run_cellvit_single_patch.py.",
    )
    parser.add_argument(
        "--cellvit-python",
        type=Path,
        default=Path(sys.executable),
        help="Python used by the CellViT wrapper to run upstream CellViT code.",
    )
    parser.add_argument("--cellvit-gpu", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = _load_and_validate_inputs(args)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    generation_args = _generation_namespace(args)
    thresholds = FidelityThresholds(
        changed_region_accuracy_min=args.changed_region_accuracy_min,
        changed_region_macro_iou_min=args.changed_region_macro_iou_min,
        off_target_drift_max=args.off_target_drift_max,
        nuclei_density_relative_error_max=args.nuclei_density_relative_error_max,
    )

    def generate(mode: str, attempt_dir: Path) -> GenerationArtifact:
        _validate_generation_runtime(args, mode)
        attempt_args = SimpleNamespace(**vars(generation_args))
        attempt_args.generation_mode = "inpaint" if mode == "inpaint" else "cross-v1"
        image_path, metadata = _run_generation_stage(
            args=attempt_args,
            output_dir=attempt_dir,
            reference_image=inputs["reference_image"],
            change_region=inputs["change_region"],
            target_tissue_path=args.target_tissue_mask.resolve(),
            target_nuclei_path=args.target_nuclei_mask.resolve(),
        )
        return GenerationArtifact(mode=mode, image_path=image_path, metadata=metadata)

    def verify(artifact: GenerationArtifact):
        _validate_verification_runtime(args)
        verification_dir = artifact.image_path.parent / "verification"
        verification_dir.mkdir(parents=True, exist_ok=True)
        predicted_tissue_path = _run_segmentator(
            args=args,
            image_path=artifact.image_path,
            output_dir=verification_dir,
        )
        predicted_nuclei_path = _run_cellvit(
            args=args,
            image_path=artifact.image_path,
            output_dir=verification_dir,
        )
        result = verify_mask_fidelity(
            reference_tissue_mask=inputs["reference_tissue"],
            target_tissue_mask=inputs["target_tissue"],
            predicted_tissue_mask=load_id_mask(predicted_tissue_path),
            change_region=inputs["change_region"],
            target_nuclei_mask=inputs["target_nuclei"],
            predicted_nuclei_mask=_load_uint8_mask(predicted_nuclei_path),
            thresholds=thresholds,
        )
        _write_json(
            verification_dir / "verification.json",
            {
                "passed": result.passed,
                "score": result.score,
                "metrics": dict(result.metrics),
                "failed_checks": list(result.failed_checks),
                "predicted_tissue_mask": str(predicted_tissue_path),
                "predicted_nuclei_mask": str(predicted_nuclei_path),
            },
        )
        return result

    workflow = run_agentic_workflow(
        reference_tissue_mask=inputs["reference_tissue"],
        target_tissue_mask=inputs["target_tissue"],
        output_dir=output_dir,
        generate=generate,
        verify=verify,
        config=AgenticWorkflowConfig(
            routing=AgenticRoutingConfig(t_inpaint=args.t_inpaint, t_cross=args.t_cross),
            max_attempts=args.max_attempts,
        ),
    )

    final_path = output_dir / "generated_image.png"
    if workflow.status == "noop":
        shutil.copy2(args.reference_image, final_path)
    elif workflow.selected_attempt and workflow.selected_attempt.artifact:
        shutil.copy2(workflow.selected_attempt.artifact.image_path, final_path)

    summary = workflow.to_metadata()
    summary["generated_image"] = str(final_path) if final_path.exists() else None
    _write_json(output_dir / "pipeline_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if workflow.status in {"validated", "noop"} else 2


def _load_and_validate_inputs(args: argparse.Namespace) -> dict[str, np.ndarray]:
    required_paths = {
        "reference image": args.reference_image,
        "reference tissue mask": args.reference_tissue_mask,
        "reference nuclei mask": args.reference_nuclei_mask,
        "target tissue mask": args.target_tissue_mask,
        "target nuclei mask": args.target_nuclei_mask,
    }
    missing = [f"{label}: {path}" for label, path in required_paths.items() if not Path(path).exists()]
    if args.change_region is not None and not args.change_region.exists():
        missing.append(f"change region: {args.change_region}")
    if missing:
        raise FileNotFoundError("Required runtime paths not found:\n" + "\n".join(missing))

    reference_image = _load_rgb_image(args.reference_image)
    reference_tissue = load_id_mask(args.reference_tissue_mask)
    reference_nuclei = _load_uint8_mask(args.reference_nuclei_mask)
    target_tissue = load_id_mask(args.target_tissue_mask)
    target_nuclei = _load_uint8_mask(args.target_nuclei_mask)
    for label, mask in (
        ("reference tissue mask", reference_tissue),
        ("reference nuclei mask", reference_nuclei),
        ("target tissue mask", target_tissue),
        ("target nuclei mask", target_nuclei),
    ):
        _validate_same_size(reference_image, mask, label)
    change_region = (
        load_change_region(args.change_region)
        if args.change_region is not None
        else reference_tissue != target_tissue
    )
    _validate_same_size(reference_image, change_region, "change region")
    return {
        "reference_image": reference_image,
        "reference_tissue": reference_tissue,
        "reference_nuclei": reference_nuclei,
        "target_tissue": target_tissue,
        "target_nuclei": target_nuclei,
        "change_region": np.asarray(change_region, dtype=bool),
    }


def _validate_generation_runtime(args: argparse.Namespace, mode: str) -> None:
    required = {"pretrained model": Path(args.pretrained_model_name_or_path)}
    if mode == "inpaint":
        required["inpaint checkpoint"] = args.inpaint_checkpoint
    else:
        required["cross-v1 checkpoint"] = args.cross_v1_checkpoint
        required["pix2pix-v2 checkpoint"] = args.pix2pix_checkpoint
    missing = [f"{label}: {path}" for label, path in required.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Generation runtime paths not found:\n" + "\n".join(missing))


def _validate_verification_runtime(args: argparse.Namespace) -> None:
    required = {
        "segmentator checkpoint": args.segmentator_checkpoint,
        "CellViT model": args.cellvit_model,
        "CellViT root": args.cellvit_root,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Verification runtime paths not found:\n" + "\n".join(missing))


def _generation_namespace(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        profile=args.profile,
        reference_image=args.reference_image.resolve(),
        reference_tissue_mask=args.reference_tissue_mask.resolve(),
        reference_nuclei_mask=args.reference_nuclei_mask.resolve(),
        generation_mode="auto",
        cross_backend="cross-v1",
        route_threshold=args.t_cross,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        inpaint_checkpoint=args.inpaint_checkpoint.resolve(),
        cross_v1_checkpoint=args.cross_v1_checkpoint.resolve(),
        pix2pix_checkpoint=args.pix2pix_checkpoint.resolve(),
        device=args.device,
        prompt=args.prompt,
        prompt_source=args.prompt_source,
        torch_dtype=args.torch_dtype,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        seed=args.seed,
        color_match=args.color_match,
    )


def _run_segmentator(*, args: argparse.Namespace, image_path: Path, output_dir: Path) -> Path:
    output_path = output_dir / "predicted_tissue_mask.png"
    if args.segmentator_python:
        command = [str(args.segmentator_python)]
    else:
        command = ["conda", "run", "-n", args.segmentator_env, "python"]
    command.extend(
        [
            str(REPO_ROOT / "scripts" / "predict_segmentator_mask.py"),
            "--checkpoint",
            str(args.segmentator_checkpoint),
            "--input",
            str(image_path),
            "--output",
            str(output_path),
            "--decoder",
            args.segmentator_decoder,
            "--device",
            args.segmentator_device,
        ]
    )
    _run_logged(command, output_dir / "segmentator.log")
    if not output_path.exists():
        raise RuntimeError(f"Segmentator completed without writing {output_path}")
    return output_path


def _run_cellvit(*, args: argparse.Namespace, image_path: Path, output_dir: Path) -> Path:
    output_path = output_dir / "predicted_nuclei_mask.png"
    command = [
        str(args.cellvit_launch_python),
        str(REPO_ROOT / "scripts" / "run_cellvit_single_patch.py"),
        "--image",
        str(image_path),
        "--output-mask",
        str(output_path),
        "--model",
        str(args.cellvit_model),
        "--cellvit-root",
        str(args.cellvit_root),
        "--cellvit-python",
        str(args.cellvit_python),
        "--gpu",
        str(args.cellvit_gpu),
    ]
    _run_logged(command, output_dir / "cellvit.log")
    if not output_path.exists():
        raise RuntimeError(f"CellViT completed without writing {output_path}")
    return output_path


def _run_logged(command: list[str], log_path: Path) -> None:
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Executable not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        log_path.write_text(_format_process(command, exc.stdout, exc.stderr), encoding="utf-8")
        raise RuntimeError(f"Command failed; see {log_path}") from exc
    log_path.write_text(_format_process(command, result.stdout, result.stderr), encoding="utf-8")


def _format_process(command: list[str], stdout: str | None, stderr: str | None) -> str:
    parts = ["command: " + " ".join(command)]
    if stdout and stdout.strip():
        parts.append("stdout:\n" + stdout.strip())
    if stderr and stderr.strip():
        parts.append("stderr:\n" + stderr.strip())
    return "\n\n".join(parts) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
